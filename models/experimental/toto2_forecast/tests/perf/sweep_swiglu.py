# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""R1-3 sweep: ``minimal_matmul(fuse_swiglu=True)`` block configs for Toto 2.0's FFN front half at the model's M.

    ttq run --model toto2 --part 2 --kind bench -- python models/experimental/toto2_forecast/tests/perf/sweep_swiglu.py \
        --ckpt 313m --rows 98304 --out models/experimental/toto2_forecast/doc/optimize/iter_01/swiglu_sweep_313m.json

Copied from t0-gamma's sweep. Real layer-0 weights (w_fc1 rows [g | u], SiLU on u, no bias; packed with
``prepare_for_fused_swiglu(gate_is_first=False)``, d_ff zero-padded to whole tiles per half) on a normal input of
``--rows`` tokens. Each config: device time per call and PCC against the unfused path the model runs today
(g linear, u linear + SiLU, MUL) and against torch fp32.
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


def packed_fc1(w_fc1: torch.Tensor, f: int, fp: int) -> torch.Tensor:
    """``(2f, d)`` [g | u] -> the ``(d, 2fp)`` fused-SwiGLU weight: silu(u) * g per tile pair, zero d_ff padding."""
    from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

    pad = lambda w: torch.nn.functional.pad(w, (0, 0, 0, fp - f))
    stacked = torch.cat([pad(w_fc1[:f]), pad(w_fc1[f:])]).T.contiguous()  # [g (mult) | u (silu)]
    return prepare_for_fused_swiglu(stacked, 1, gate_is_first=False)


def main():
    import ttnn

    from models.experimental.toto2_forecast.tt.common import Toto2Precision, linear, pad_to_tile, upload, upload_linear
    from models.experimental.toto2_forecast.tt.weights import Toto2Weights

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="313m")
    ap.add_argument("--rows", type=int, default=98304)
    ap.add_argument("--precision", default="accuracy")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--configs", type=json.loads, default=CONFIGS, help="JSON list of [M, K, N, sub_h, sub_w]")
    args = ap.parse_args()
    p = Toto2Precision.named(args.precision).for_role("ffn")
    w = Toto2Weights.from_checkpoint(args.ckpt)
    lw, d, f = w.layers[0], w.config.d_model, w.config.d_ff
    fp = pad_to_tile(f)
    g = torch.Generator().manual_seed(5)
    x = torch.randn(1, 1, args.rows, d, generator=g)
    head = x[0, 0, :4096]
    ref = (head @ lw.w_fc1[:f].T) * torch.nn.functional.silu(head @ lw.w_fc1[f:].T)

    device = ttnn.open_device(device_id=0)
    results = []
    try:
        dtype = Toto2Precision.named(args.precision).weight_dtype("ffn")
        x_dev = upload(device, x)
        wg = upload_linear(device, lw.w_fc1[:f], dtype, pad_out=fp)
        wu = upload_linear(device, lw.w_fc1[f:], dtype, pad_out=fp)

        def unfused():
            gg = linear(x_dev, wg, p)
            uu = linear(x_dev, wu, p, activation="silu")
            out = ttnn.mul(gg, uu, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(gg)
            ttnn.deallocate(uu)
            return out

        base = ttnn.to_torch(unfused()).float()[0, 0]
        base_ms = time_op(device, unfused) * 1e3
        results.append({"config": "unfused (round 0)", "ms": base_ms, "pcc_vs_torch": pcc(ref, base[:4096, :f])})
        print(f"[swiglu] unfused {base_ms:.3f} ms, pcc vs torch {results[0]['pcc_vs_torch']:.6f}", flush=True)
        w_packed = upload(device, packed_fc1(lw.w_fc1, f, fp), dtype=dtype)
        grid = device.compute_with_storage_grid_size()
        kernel = p.matmul_config()
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
                config=cfg,
                fuse_swiglu=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=kernel,
            )
            record = {"config": [mb, kb, nb, sh, sw]}
            try:
                got = ttnn.to_torch(fused()).float()[0, 0]
                record.update(
                    ms=time_op(device, fused) * 1e3,
                    pcc_vs_unfused=pcc(base, got),
                    pcc_vs_torch=pcc(ref, got[:4096, :f]),
                )
            except Exception as e:  # illegal block / L1 overflow
                record["error"] = str(e).splitlines()[0][:200]
            results.append(record)
            print(f"[swiglu] {record}", flush=True)
    finally:
        ttnn.close_device(device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"ckpt": args.ckpt, "rows": args.rows, "precision": args.precision, "results": results}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
