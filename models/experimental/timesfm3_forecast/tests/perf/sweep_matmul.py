# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""R1-4: program-config sweep for every matmul role at the rows the model runs, in one job (pattern from t0).

    ttq run --model timesfm3 --part 0 --kind bench -- python models/experimental/timesfm3_forecast/tests/perf/sweep_matmul.py \
        --out models/experimental/timesfm3_forecast/doc/optimize/iter_00/r1_4/matmul_sweep.json

Rows: 65536 (paper / multivariate after R1-1: 1024 x 64) and 16384 (long_context: 32 x 512). Roles use the real
layer-0 weights. Candidates: ``auto`` (the current ``ttnn.linear``), ``2d:GXxGY:cap`` and ``mm:M,K,N,h,w``
(``minimal_matmul`` blocks). Each: device time per call (batches of 10) and PCC against ``auto``. The accuracy
policy throughout; ``qkv_bf8`` is the performance policy's bf8 QKV.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

SQ = "auto;2d:11x10:12;2d:11x10:8;2d:10x10:8;2d:8x10:8;mm:8,8,8,2,2;mm:16,8,4,4,1;mm:8,8,4,2,2;mm:8,16,8,2,2;mm:16,8,8,4,1;mm:4,8,8,1,4"
QKV = "auto;2d:11x10:12;2d:11x10:8;2d:8x10:8;2d:8x10:6;mm:8,8,8,2,2;mm:16,8,8,4,1;mm:8,8,16,2,2;mm:8,16,12,1,4;mm:16,8,12,4,1;mm:4,16,16,1,4"
TOK = "auto;2d:11x10:12;2d:11x10:8;2d:8x10:8;mm:8,6,8,2,2;mm:16,6,4,4,1;mm:16,6,8,4,1;mm:8,6,4,2,2"
CONFIGS = {
    "qkv": QKV,
    "qkv_bf8": QKV,
    "wo": SQ,
    "w_ov": SQ,
    "ff0": SQ,
    "ff1": SQ,
    "tok_hidden": TOK,
    "tok_res": TOK,
    "tok_out": SQ,
}


def role_weights(role: str):
    """``(weight (out, in), activation, bf8)``."""
    from models.experimental.timesfm3_forecast.tt.model import load_checkpoint_weights

    w = load_checkpoint_weights(num_layers=1)
    l0 = w.layers[0]
    return {
        "qkv": (l0.seq.wqkv, None, False),
        "qkv_bf8": (l0.seq.wqkv, None, True),
        "wo": (l0.seq.wo, None, False),
        "w_ov": (l0.var.w_ov, None, False),
        "ff0": (l0.ff0, "relu", False),
        "ff1": (l0.ff1, None, False),
        "tok_hidden": (w.tok_hidden, "relu", False),
        "tok_res": (w.tok_res, None, False),
        "tok_out": (w.tok_out, None, False),
    }[role]


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


def make_fn(ttnn, pc, spec, x_dev, w_dev, act, kernel, grid, m_tiles, k_tiles, n_tiles, record):
    if spec == "auto":
        return lambda: ttnn.linear(
            x_dev, w_dev, activation=act, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=kernel
        )
    if spec.startswith("2d:"):
        _, g, cap = spec.split(":")
        gx, gy = (int(v) for v in g.split("x"))
        cfg = pc.matmul_2d_config(
            (gx, gy), m_tiles, k_tiles, n_tiles, fused_activation=pc.fused_activation(act), out_block_cap=int(cap)
        )
        record["program_config"] = str(cfg)[:300]
        return lambda: ttnn.linear(
            x_dev, w_dev, program_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=kernel
        )
    cfg = pc.minimal_config((grid.x, grid.y), m_tiles, k_tiles, n_tiles, tuple(int(v) for v in spec[3:].split(",")))
    return lambda: ttnn.experimental.minimal_matmul(
        x_dev,
        w_dev,
        fused_activation=pc.fused_activation(act),
        config=cfg,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=kernel,
    )


def main():
    import ttnn

    from models.experimental.timesfm3_forecast.tt import program_configs as pc
    from models.experimental.timesfm3_forecast.tt.common import TtPrecision, upload, upload_linear

    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", default=",".join(CONFIGS))
    ap.add_argument("--rows", default="65536,16384")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--configs", help="';'-separated specs that replace every role's default list")
    args = ap.parse_args()
    kernel = TtPrecision.accuracy().matmul_config()
    device = ttnn.open_device(device_id=0)
    results = []
    try:
        grid = device.compute_with_storage_grid_size()
        for role in args.roles.split(","):
            weight, act, bf8 = role_weights(role)
            n_out, k_in = weight.shape
            w_dev = upload_linear(device, weight, ttnn.bfloat8_b if bf8 else ttnn.bfloat16)
            for rows in (int(r) for r in args.rows.split(",")):
                x = torch.randn(1, 1, rows, k_in, generator=torch.Generator().manual_seed(9))
                x_dev = upload(device, x)
                m_tiles, k_tiles, n_tiles = rows // 32, -(-k_in // 32), n_out // 32
                base = None
                for spec in (args.configs or CONFIGS[role]).split(";"):
                    record = {"role": role, "rows": rows, "k": k_in, "n": n_out, "config": spec}
                    try:
                        fn = make_fn(ttnn, pc, spec, x_dev, w_dev, act, kernel, grid, m_tiles, k_tiles, n_tiles, record)
                        out = ttnn.to_torch(fn()).float()
                        base = out if base is None else base
                        record.update(ms=time_op(device, fn) * 1e3, pcc_vs_auto=pcc(base, out))
                    except Exception as e:
                        record["error"] = str(e).splitlines()[0][:200]
                    results.append(record)
                    print(
                        f"[matmul] {role} M={rows} {spec}: {record.get('ms', float('nan')):.3f} ms, "
                        f"pcc {record.get('pcc_vs_auto', float('nan')):.6f} {record.get('error', '')}",
                        flush=True,
                    )
                ttnn.deallocate(x_dev)
            ttnn.deallocate(w_dev)
    finally:
        ttnn.close_device(device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"precision": "accuracy", "results": results}, indent=1) + "\n")


if __name__ == "__main__":
    main()
