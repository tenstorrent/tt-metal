# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""R1-4 sweep: ``minimal_matmul(fuse_swiglu=True)`` block configs for t0's MLP front half at the model's M.

    ttq run --model t0-gamma --part 3 --kind bench -- python models/experimental/t0_forecast/tests/perf/sweep_swiglu.py \
        --rows 98304 --out models/experimental/t0_forecast/doc/optimize/iter_01/swiglu_sweep.json [--precision performance]

Real layer-0 weights (w1 rows [gate | up], the gate first; packed with ``prepare_for_fused_swiglu(gate_is_first=True)``,
bias included) on a normal input of ``--rows`` tokens. Each config: device time per call (``time_op``: batches of 10
back-to-back calls) and the PCC of its output against the unfused path the model runs today (gate linear + SiLU, up
linear, MUL) and against torch fp32. Gate (D2): per-call PCC >= 0.9999 against the unfused path.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

CONFIGS = [  # (M_block, K_block, N_block, subblock_h, subblock_w)
    (8, 8, 2, 2, 2),
    (8, 16, 2, 2, 2),
    (8, 32, 2, 2, 2),
    (4, 8, 2, 2, 2),
    (16, 8, 2, 2, 2),
    (16, 16, 2, 2, 2),
    (8, 8, 4, 1, 4),
    (8, 16, 4, 1, 4),
    (16, 8, 4, 2, 2),
    (8, 8, 2, 4, 1),
    (16, 16, 4, 4, 1),
    (4, 16, 4, 1, 4),
]


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (a.norm() * b.norm()))


def time_op(device, fn, batches=5, calls=10):
    import ttnn

    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(device)
    samples = []
    for _ in range(batches):
        start = time.perf_counter()
        outs = [fn() for _ in range(calls)]
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - start) / calls)
        for o in outs:
            ttnn.deallocate(o)
    return statistics.median(samples)


def main():
    import ttnn

    from models.experimental.t0_forecast.tests.helpers import weights
    from models.experimental.t0_forecast.tt.common import TtT0Precision, linear, upload, upload_bias, upload_linear
    from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=98304)
    ap.add_argument("--precision", default="accuracy")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--configs", type=json.loads, default=CONFIGS, help="JSON list of [M, K, N, sub_h, sub_w]")
    args = ap.parse_args()
    p = TtT0Precision.named(args.precision)
    m = weights().layers[0].mlp
    h = m.hidden
    g = torch.Generator().manual_seed(5)
    x = torch.randn(1, 1, args.rows, 1024, generator=g)
    head = x[0, 0, :4096]
    ref = torch.nn.functional.silu(head @ m.w1[:h].T + m.b1[:h]) * (head @ m.w1[h:].T + m.b1[h:])

    device = ttnn.open_device(device_id=0)
    results = []
    try:
        dtype = p.weight_dtype("ff1")
        x_dev = upload(device, x)
        wg, bg = upload_linear(device, m.w1[:h], dtype), upload_bias(device, m.b1[:h])
        wu, bu = upload_linear(device, m.w1[h:], dtype), upload_bias(device, m.b1[h:])

        def unfused():
            gate = linear(x_dev, wg, p, role="ff1", bias=bg, activation="silu")
            up = linear(x_dev, wu, p, role="ff1", bias=bu)
            out = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
            return out

        base = ttnn.to_torch(unfused()).float()[0, 0]
        base_ms = time_op(device, unfused) * 1e3
        results.append({"config": "unfused (round 0)", "ms": base_ms, "pcc_vs_torch": pcc(ref, base[:4096])})
        print(f"[swiglu] unfused {base_ms:.3f} ms, pcc vs torch {results[0]['pcc_vs_torch']:.6f}", flush=True)
        w_packed = upload(device, prepare_for_fused_swiglu(m.w1.T.contiguous(), 1, gate_is_first=True), dtype=dtype)
        b_packed = upload(device, prepare_for_fused_swiglu(m.b1.reshape(1, -1), 1, gate_is_first=True))
        grid = device.compute_with_storage_grid_size()
        kernel = p.compute_kernel_config(role="ff1")
        for mb, kb, nb, sh, sw in args.configs:
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kb,
                N_block_size=nb,
                subblock_h=sh,
                subblock_w=sw,
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            )
            fused = lambda: ttnn.experimental.minimal_matmul(
                x_dev,
                w_packed,
                bias_tensor=b_packed,
                config=cfg,
                fuse_swiglu=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=kernel,
            )
            record = {"config": [mb, kb, nb, sh, sw]}
            try:
                got = ttnn.to_torch(fused()).float()[0, 0]
                record.update(
                    ms=time_op(device, fused) * 1e3, pcc_vs_unfused=pcc(base, got), pcc_vs_torch=pcc(ref, got[:4096])
                )
            except Exception as e:  # illegal block / L1 overflow
                record["error"] = str(e).splitlines()[0][:200]
            results.append(record)
            print(f"[swiglu] {record}", flush=True)
    finally:
        ttnn.close_device(device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"rows": args.rows, "precision": args.precision, "results": results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
