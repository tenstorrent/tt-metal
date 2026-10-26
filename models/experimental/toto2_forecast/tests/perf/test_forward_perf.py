# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks: TTNN Toto 2.0 ``forward`` (one forecast forward, ``decode_block_size=None``) on one chip.

Shapes (CONTRACT sections 1-2), parametrized ``<shape>-<precision>``:
    paper         2.5B-FT, 1024 univariate series x 2048 -> 64 (66 tokens per row)            target: accuracy
    long_context  2.5B-FT, 32 series x 8192 -> 1024 (288 tokens per row, 32 read-out tokens)   guard
    multivariate  2.5B-FT, 128 series x 8 variates sharing a series id x 2048 -> 64            guard
    paper_313m    313m, 1024 x 2048 -> 64                                                       guard
    paper / performance                                                                         guard

Inputs and CPU reference outputs come from the realistic-mix goldens (``tests/golden/make_goldens.py``, run in the
reference venv; PLAN section 9); ``cpu_s`` is the reference ``forecast`` time recorded there (8 threads).

Per record: ``cold_s`` (first call, compile included), ``steady_s`` (median of 5 end-to-end ``forecast`` calls),
the host / device split (medians of 5: ``prepare_s``, ``upload_s``, ``device_s`` after a 0.3 s idle,
``postprocess_s``), the gate metrics of ``goldens.gate_metrics`` (``pcc_normalized``: head output in asinh space vs
the hook on ``m.output_head``; ``pcc_spread``; ``pcc_row_p01`` / ``pcc_row_min``; ``pcc_final``: quantiles;
``pcc_final_spread``; ``abs_err_max``), ``chip`` ($TTQ_CHIP_SERIAL). ``MODEL_PERF_JSON=<path>`` appends the records; ``BENCH_SAVE_DIR=<dir>`` also saves
each case's head output and quantiles (``<ckpt>-<golden>-<precision>.pt``) for host-side scoring. Precision
``performance_P<n>`` picks a performance-policy variant (``Toto2Precision.VARIANTS``).

Screening (skill section 5.2): ``BENCH_LAYERS=<L>`` keeps time layers 0 .. L-3, the last time layer and the variate
layer (L layers). Device time then compares candidates at the same depth; the gate metrics are meaningless there
and the floors are not asserted. ``BENCH_CASES=paper_313m-accuracy,...`` picks cases.
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from models.experimental.toto2_forecast.tests.golden.goldens import (
    below_floors,
    floors_for,
    gate_metrics,
    have_golden,
    load_golden,
)

# One device per module, so a loaded model (2.5B: about a minute) serves every case that uses it.
pytestmark = pytest.mark.use_module_device

STEADY_ITERS = 5
# Back-to-back forwards slow the device down, so device_s would follow host-side speedups; the split idles
# this long before each timed forward.
DEVICE_SETTLE_S = 0.3


@dataclass(frozen=True)
class Shape:
    name: str
    checkpoint: str
    golden: str


SHAPES = {
    s.name: s
    for s in (
        Shape("paper", "2.5B-FT", "paper_mix_1024x2048"),
        Shape("long_context", "2.5B-FT", "long_context_mix_32x8192"),
        Shape("multivariate", "2.5B-FT", "multivariate_mix_128x8x2048"),
        Shape("paper_313m", "313m", "paper_mix_1024x2048"),
        Shape("single_sparse", "2.5B-FT", "single_sparse_1x2048"),
    )
}
# One model per process at a time, so 2.5B holds one precision's weights.
CASES = [
    ("paper", "accuracy"),
    ("long_context", "accuracy"),
    ("multivariate", "accuracy"),
    ("paper", "performance"),
    ("paper_313m", "accuracy"),
]
BENCH_LAYERS = int(os.environ.get("BENCH_LAYERS", "0"))
if os.environ.get("BENCH_CASES"):
    CASES = [tuple(c.rsplit("-", 1)) for c in os.environ["BENCH_CASES"].split(",")]
FLOORS = {"accuracy": (0.999, 0.99), "performance": (0.995, 0.98)}
_MODEL: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _free_models_before_device_close(_device_module_impl):
    """The cached model's device tensors must go before the module's device closes."""
    yield
    _MODEL.clear()
    gc.collect()


def _sync_time(device, fn):
    import ttnn

    start = time.perf_counter()
    out = fn()
    ttnn.synchronize_device(device)
    return out, time.perf_counter() - start


def tt_model(device, checkpoint: str, precision_name: str):
    from models.experimental.toto2_forecast.tt.common import Toto2Precision
    from models.experimental.toto2_forecast.tt.model import TtToto2

    key = (checkpoint, precision_name)
    if key not in _MODEL:
        _MODEL.clear()
        gc.collect()
        precision = Toto2Precision.named(precision_name)
        if BENCH_LAYERS:
            from models.experimental.toto2_forecast.tt.weights import Toto2Weights

            weights = Toto2Weights.from_checkpoint(checkpoint)
            n = weights.config.num_layers
            keep = [*range(BENCH_LAYERS - 2), n - 2, n - 1]
            assert [weights.layers[i].is_variate for i in keep[-2:]] == [False, True], "layout: ... time, variate"
            _MODEL[key] = TtToto2(device, weights.reduced(keep), precision)
        else:
            _MODEL[key] = TtToto2.from_checkpoint(device, checkpoint, precision)
    return _MODEL[key]


def _record(entry: dict) -> None:
    path = os.environ.get("MODEL_PERF_JSON")
    if not path:
        return
    out = Path(path)
    records = json.loads(out.read_text()) if out.is_file() else []
    records.append(entry)
    out.write_text(json.dumps(records, indent=2) + "\n")


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("shape_name,precision_name", CASES, ids=[f"{s}-{p}" for s, p in CASES])
def test_forward_perf(device, shape_name, precision_name):
    import ttnn

    shape = SHAPES[shape_name]
    if not have_golden(shape.checkpoint, shape.golden):
        pytest.skip(f"golden {shape.checkpoint}/{shape.golden} not generated yet")
    golden = load_golden(shape.checkpoint, shape.golden)
    model = tt_model(device, shape.checkpoint, precision_name)
    i = golden["inputs"]
    args = (i["target"], i["target_mask"], i["series_ids"], i["horizon"])

    def run():
        return model.forecast(*args)

    _, cold_s = _sync_time(device, run)
    steady = [_sync_time(device, run)[1] for _ in range(STEADY_ITERS)]

    splits = []
    for _ in range(STEADY_ITERS):
        start = time.perf_counter()
        host = model.prepare_inputs(*args)
        prepare_s = time.perf_counter() - start
        inputs, upload_s = _sync_time(device, lambda: model.upload_inputs(host))
        time.sleep(DEVICE_SETTLE_S)
        out, device_s = _sync_time(device, lambda: model.forward_device(inputs))
        start = time.perf_counter()
        q, head = model.postprocess(out, host, normalized=True)
        postprocess_s = time.perf_counter() - start
        ttnn.deallocate(out)
        model.deallocate_inputs(inputs)
        splits.append((prepare_s, upload_s, device_s, postprocess_s))
    prepare_s, upload_s, device_s, postprocess_s = (statistics.median(col) for col in zip(*splits))

    cap = golden["captures"]
    metrics = gate_metrics(head, q, cap["head_out"], cap["q"])
    pcc_normalized, pcc_final = metrics["pcc_normalized"], metrics["pcc_final"]
    b, v, t = i["target"].shape
    record = {
        "test": "forward",
        "name": shape.name,
        "precision": precision_name,
        "checkpoint": shape.checkpoint,
        "batch": b,
        "variates": v,
        "context": t,
        "horizon": i["horizon"],
        "rows": host.rows,
        "seq_len": host.seq_len,
        "cold_s": cold_s,
        "steady_s": statistics.median(steady),
        "steady_runs": steady,
        "prepare_s": prepare_s,
        "upload_s": upload_s,
        "device_s": device_s,
        "postprocess_s": postprocess_s,
        "split_runs": [dict(zip(("prepare_s", "upload_s", "device_s", "postprocess_s"), s)) for s in splits],
        **metrics,
        "cpu_s": golden["cpu_s"],
        "cpu_threads": golden.get("threads"),
        "chip": os.environ.get("TTQ_CHIP_SERIAL"),
        "bench_layers": BENCH_LAYERS or None,
    }
    print(
        f"\n[PERF] forward {shape.name} / {precision_name} ({shape.checkpoint})"
        f"\n  rows x context -> horizon: {host.rows} x {t} -> {i['horizon']} ({b} series x {v} variates),"
        f" {host.seq_len} tokens per row"
        f"\n  cold_s {cold_s:.3f}  steady_s {record['steady_s']:.3f}  ({', '.join(f'{s:.3f}' for s in steady)})"
        f"\n  split: prepare {prepare_s:.3f}  upload {upload_s:.3f}  device {device_s:.3f}  postprocess {postprocess_s:.3f}"
        f"\n  cpu_s {golden['cpu_s']:.1f} ({golden.get('threads')} threads)  speedup {golden['cpu_s'] / record['steady_s']:.1f}x"
        f"\n  " + "  ".join(f"{k} {v:.6f}" for k, v in metrics.items())
    )
    record["below_floors"] = below_floors(metrics, floors_for(shape.checkpoint, precision_name))
    _record(record)
    if os.environ.get("BENCH_SAVE_DIR"):
        save = Path(os.environ["BENCH_SAVE_DIR"]) / f"{shape.checkpoint}-{shape.golden}-{precision_name}.pt"
        save.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"head": head.float(), "q": q.float(), "record": record}, save)
    if BENCH_LAYERS:
        return
    # The bench records every metric and asserts only the contract floors; the suite asserts the others.
    if metrics["rows_degenerate"] == metrics["rows"]:
        # D2 item 4: an all-degenerate case is judged as the suite judges it (abs_err_max plus the degenerate
        # spread_ratio_p50 range); its pcc_normalized measures bf16 noise.
        assert not record["below_floors"], record["below_floors"]
        return
    floor_n, floor_f = FLOORS[precision_name.split("_")[0]]
    assert pcc_normalized >= floor_n and pcc_final >= floor_f, (pcc_normalized, pcc_final)
