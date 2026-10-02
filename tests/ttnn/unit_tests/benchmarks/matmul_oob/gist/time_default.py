# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Time ttnn.matmul's default config (no program_config) on the shapes of the #57884 gist sweeps
(matmul_sweep_2d.py, matmul_sweep_1d.py), with their Runner: bf16, HiFi4, fp32 dest accumulation, DRAM
interleaved, mean synchronized wall time over --iters calls. One CSV row per shape.

  python time_default.py {2d|1d} out.csv [--v2] [--iters 20] [--no-pcc]
"""

import argparse
import csv
import importlib
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("suite", choices=["2d", "1d"])
    p.add_argument("out")
    p.add_argument("--v2", action="store_true", help="ttnn.CONFIG.matmul_auto_config_v2 on (default: legacy)")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--no-pcc", action="store_true", help="skip the torch reference (fast runs)")
    a = p.parse_args()
    sweep = importlib.import_module(f"matmul_sweep_{a.suite}")
    import ttnn

    if a.v2 or hasattr(ttnn.CONFIG, "matmul_auto_config_v2"):
        ttnn.CONFIG.matmul_auto_config_v2 = a.v2  # absent on main
    mm = ttnn._ttnn.operations.matmul
    last_cfg = getattr(mm, "matmul_last_auto_program_config", None)
    fell_back = getattr(mm, "matmul_last_auto_config_fell_back", None)

    shapes = [
        s
        for s in sweep.llama_shapes(sweep.DEFAULT_MODELS, sweep.TOKENS, sweep.DEFAULT_PASSES)
        if sweep.route(s) in sweep.KERNELS
    ]
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    grid = dev.compute_with_storage_grid_size()
    runner = sweep.Runner(ttnn, dev, (grid.x, grid.y), types.SimpleNamespace(warmup=a.warmup))
    rows = []
    try:
        for i, s in enumerate(shapes):
            runner.load(s)
            try:
                ms = runner.time_ms(None, a.iters)
                cfg = last_cfg(reset=False) if last_cfg else ""
                fb = fell_back() if fell_back else ""
                pcc = float("nan")
                if not a.no_pcc:
                    out = runner.output(None)
                    ta, tb = ttnn.to_torch(runner.a).float(), ttnn.to_torch(runner.b).float()
                    ref = (ta.T if s.transpose_a else ta) @ (tb.T if s.transpose_b else tb)
                    pcc = sweep.pcc(out.reshape(ref.shape), ref)
            finally:
                runner.unload()
            rows.append(
                dict(
                    i=i,
                    operands=s.operands,
                    M=s.M,
                    K=s.K,
                    N=s.N,
                    transpose_a=int(s.transpose_a),
                    transpose_b=int(s.transpose_b),
                    name=s.name,
                    ms=ms,
                    pcc=pcc,
                    config=cfg,
                    fell_back=fb,
                )
            )
            print(
                f"[{i + 1}/{len(shapes)}] {s.operands:36s} {ms if ms is None else round(ms, 4)} ms  pcc {pcc:.5f}",
                flush=True,
            )
    finally:
        ttnn.close_device(dev)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
