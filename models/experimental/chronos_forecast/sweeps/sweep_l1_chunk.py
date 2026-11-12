# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Trace-replay time of the paper-shape forward for precision x L1 series-chunk configs."""

import gc
import statistics
import sys
import time

import torch
import ttnn

from models.experimental.chronos_forecast.tests.perf.test_paper_forward import (
    BATCH,
    CONTEXT,
    NUM_OUTPUT_PATCHES,
    _load_reference,
)
from models.experimental.chronos_forecast.tt.model import (
    TtChronos,
    TtChronosWeights,
    tt_chronos_config_from_torch_model,
)
from models.experimental.chronos_forecast.tt.program_configs import TtChronosPrecision
from models.experimental.chronos_forecast.tt.trace_runner import TtChronosTraceRunner


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main(configs):
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    dev.enable_program_cache()
    try:
        reference, _ = _load_reference()
        weights = TtChronosWeights.from_torch_model(reference)
        cfg = tt_chronos_config_from_torch_model(reference)
        torch.manual_seed(0)
        context = torch.randn(BATCH, CONTEXT)
        baseline = {}
        for precision_name, chunk in configs:
            precision = TtChronosPrecision.performance() if precision_name == "performance" else TtChronosPrecision()
            model = TtChronos(dev, weights, cfg, precision, l1_chunk_tokens=None if chunk is None else chunk * 160)
            prepared = model.prepare_inputs(context=context, num_output_patches=NUM_OUTPUT_PATCHES)
            runner = None
            try:
                runner = TtChronosTraceRunner(model, prepared)
                runner.capture()
                for _ in range(2):
                    runner.execute(blocking=True, readback=False)
                times = []
                for _ in range(8):
                    t0 = time.perf_counter()
                    runner.execute(blocking=True, readback=False)
                    times.append(time.perf_counter() - t0)
                out = runner.execute(blocking=True).quantile_preds
                if chunk is None:
                    baseline[precision_name] = out
                ref = baseline.get(precision_name)
                p = pcc(ref, out) if ref is not None else float("nan")
                print(
                    f"{precision_name:>12} chunk={str(chunk):>5} replay={statistics.median(times) * 1e3:8.2f} ms pcc_vs_dram={p:.6f}",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                print(f"{precision_name:>12} chunk={str(chunk):>5} ERR {str(e).splitlines()[0][:200]}", flush=True)
            finally:
                if runner is not None:
                    runner.release()
                del model, runner
                gc.collect()
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    specs = sys.argv[1:] or ["performance:none", "performance:32", "performance:64"]
    parsed = []
    for s in specs:
        name, chunk = s.split(":")
        parsed.append((name, None if chunk == "none" else int(chunk)))
    main(parsed)
