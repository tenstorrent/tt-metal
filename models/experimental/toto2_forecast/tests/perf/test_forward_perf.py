# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks: TTNN Toto 2.0 ``forward`` (one forecast forward, ``decode_block_size=None``) on one chip.

Shapes (CONTRACT sections 1-2), parametrized ``<shape>-<precision>``:
    paper         2.5B-FT, 1024 univariate series x 2048 -> 64 (66 tokens per row)            target: accuracy
    long_context  2.5B-FT, 32 series x 8192 -> 1024 (288 tokens per row, 32 read-out tokens)   guard
    multivariate  2.5B-FT, 128 series x 8 variates sharing a series id x 2048 -> 64            guard
    paper_313m    313m, 1024 x 2048 -> 64                                                       guard
    paper / performance                                                                         guard

Inputs and CPU reference outputs come from the goldens (``tests/golden/make_goldens.py``, run in the reference
venv); ``cpu_s`` is the reference ``forecast`` time recorded there (8 threads).

Per record: ``cold_s`` (first call, compile included), ``steady_s`` (median of 5 end-to-end ``forecast`` calls),
the host / device split (medians of 5: ``prepare_s``, ``upload_s``, ``device_s`` after a 0.3 s idle,
``postprocess_s``), ``pcc_normalized`` (head output in asinh space vs the hook on ``m.output_head``) and
``pcc_final`` (quantiles), ``chip`` ($TTQ_CHIP_SERIAL). ``MODEL_PERF_JSON=<path>`` appends the records to a list.
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

from models.experimental.toto2_forecast.tests.golden.goldens import have_golden, load_golden, pcc

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
        Shape("paper", "2.5B-FT", "paper_1024x2048"),
        Shape("long_context", "2.5B-FT", "long_context_32x8192"),
        Shape("multivariate", "2.5B-FT", "multivariate_128x8x2048"),
        Shape("paper_313m", "313m", "paper_1024x2048"),
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
        _MODEL[key] = TtToto2.from_checkpoint(device, checkpoint, Toto2Precision.named(precision_name))
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
    pcc_normalized, pcc_final = pcc(head, cap["head_out"]), pcc(q, cap["q"])
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
        "pcc_normalized": pcc_normalized,
        "pcc_final": pcc_final,
        "cpu_s": golden["cpu_s"],
        "cpu_threads": golden.get("threads"),
        "chip": os.environ.get("TTQ_CHIP_SERIAL"),
    }
    print(
        f"\n[PERF] forward {shape.name} / {precision_name} ({shape.checkpoint})"
        f"\n  rows x context -> horizon: {host.rows} x {t} -> {i['horizon']} ({b} series x {v} variates),"
        f" {host.seq_len} tokens per row"
        f"\n  cold_s {cold_s:.3f}  steady_s {record['steady_s']:.3f}  ({', '.join(f'{s:.3f}' for s in steady)})"
        f"\n  split: prepare {prepare_s:.3f}  upload {upload_s:.3f}  device {device_s:.3f}  postprocess {postprocess_s:.3f}"
        f"\n  cpu_s {golden['cpu_s']:.1f} ({golden.get('threads')} threads)  speedup {golden['cpu_s'] / record['steady_s']:.1f}x"
        f"\n  pcc_normalized {pcc_normalized:.6f}  pcc_final {pcc_final:.6f}"
    )
    _record(record)
    floor_n, floor_f = FLOORS[precision_name]
    assert pcc_normalized >= floor_n and pcc_final >= floor_f, (pcc_normalized, pcc_final)
