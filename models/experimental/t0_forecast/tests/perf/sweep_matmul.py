# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Program-config sweep for one matmul role at the M the model runs (R1-5, R1-8): a search table per role.

    ttq run --model t0-gamma --part 3 --kind bench -- python models/experimental/t0_forecast/tests/perf/sweep_matmul.py \
        --role qkv --rows 98304 --out models/experimental/t0_forecast/doc/optimize/iter_01/matmul_sweep_qkv.json

Roles (real layer weights and biases): qkv (1024 -> 3072), wo (1024 -> 1024), wvo (the folded G = 1 group matmul),
ff2 (2048 -> 1024), embed_hidden (96 -> 1024, ReLU), embed_out (1024 -> 1024), embed_skip (96 -> 1024).
Candidates: ``auto`` (the round-0 ``ttnn.linear``), ``2d:GXxGY:cap`` (``program_configs.matmul_2d_config``) and
``mm:M,K,N,h,w`` (``ttnn.experimental.minimal_matmul`` blocks). Each: device time per call (batches of 10
back-to-back calls) and PCC against ``auto``. ``--precision performance`` uses bfp8 weights and LoFi.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

DEFAULTS = {
    "qkv": "auto;2d:11x10:12;2d:8x10:12;2d:8x10:8;2d:6x10:8;mm:8,8,8,2,2;mm:16,8,8,4,1;mm:8,16,12,1,4;mm:16,8,4,4,1;"
    "mm:8,8,16,2,2;mm:4,16,8,1,4;mm:8,32,8,2,2;mm:16,16,6,2,2",
    "wo": "auto;2d:11x10:12;2d:8x10:12;2d:8x10:8;mm:16,8,4,4,1;mm:8,8,8,2,2;mm:8,16,4,2,2;mm:16,16,4,4,1;mm:8,8,4,1,4",
    "ff2": "auto;2d:11x10:12;2d:8x10:12;2d:8x10:8;mm:16,8,4,4,1;mm:8,8,8,2,2;mm:8,16,4,2,2;mm:16,16,4,4,1;mm:8,32,4,2,2",
    "embed_hidden": "auto;2d:11x10:12;2d:8x10:12;mm:16,3,4,4,1;mm:8,3,8,2,2;mm:16,3,8,4,1",
    "embed_skip": "auto;2d:11x10:12;2d:8x10:12;mm:16,3,4,4,1;mm:8,3,8,2,2;mm:16,3,8,4,1",
    "embed_out": "auto;2d:11x10:12;2d:8x10:12;mm:16,8,4,4,1;mm:8,8,8,2,2;mm:8,16,4,2,2",
}
DEFAULTS["wvo"] = DEFAULTS["wo"]


def role_weights(role: str):
    """``(weight (out, in), bias, activation, fidelity role)`` for ``role``."""
    from models.experimental.t0_forecast.tests.helpers import weights

    w = weights()
    a0, a2, m0 = w.layers[0].attention, w.layers[2].attention, w.layers[0].mlp
    pe = w.patch_encoder
    table = {
        "qkv": (a0.wqkv, a0.bqkv, None, "qkv"),
        "wo": (a0.wo, a0.bo, None, "wo"),
        "wvo": (a2.wo @ a2.part(2)[0], a2.wo @ a2.part(2)[1] + a2.bo, None, "wo"),
        "ff2": (m0.w2, m0.b2, None, "ff2"),
        "embed_hidden": (pe.hidden_w, pe.hidden_b, "relu", "embed"),
        "embed_out": (pe.output_w, pe.output_b + pe.residual_b, None, "embed"),
        "embed_skip": (pe.residual_w, None, None, "embed"),
    }
    return table[role]


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

    from models.experimental.t0_forecast.tt import program_configs as pc
    from models.experimental.t0_forecast.tt.common import TtT0Precision, upload, upload_bias, upload_linear

    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=sorted(DEFAULTS))
    ap.add_argument("--rows", type=int, default=98304)
    ap.add_argument("--precision", default="accuracy")
    ap.add_argument("--configs", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    p = TtT0Precision.named(args.precision)
    weight, bias, act, fid_role = role_weights(args.role)
    n_out, k_in = weight.shape
    x = torch.randn(1, 1, args.rows, k_in, generator=torch.Generator().manual_seed(9))

    device = ttnn.open_device(device_id=0)
    results = []
    try:
        grid = device.compute_with_storage_grid_size()
        kernel = p.compute_kernel_config(role=fid_role)
        x_dev = upload(device, x)
        w_dev = upload_linear(device, weight, p.weight_dtype(fid_role))
        b_dev = None if bias is None else upload_bias(device, bias)
        m_tiles, k_tiles, n_tiles = args.rows // 32, -(-k_in // 32), n_out // 32
        base = None
        for spec in (args.configs or DEFAULTS[args.role]).split(";"):
            record = {"config": spec}
            try:
                if spec == "auto":
                    fn = lambda: ttnn.linear(
                        x_dev,
                        w_dev,
                        bias=b_dev,
                        activation=act,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        compute_kernel_config=kernel,
                    )
                elif spec.startswith("2d:"):
                    _, g, cap = spec.split(":")
                    gx, gy = (int(v) for v in g.split("x"))
                    cfg = pc.matmul_2d_config(
                        (gx, gy),
                        m_tiles,
                        k_tiles,
                        n_tiles,
                        fused_activation=pc.fused_activation(act),
                        out_block_cap=int(cap),
                    )
                    record["program_config"] = str(cfg)[:300]
                    fn = lambda cfg=cfg: ttnn.linear(
                        x_dev,
                        w_dev,
                        bias=b_dev,
                        program_config=cfg,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        compute_kernel_config=kernel,
                    )
                else:
                    blocks = tuple(int(v) for v in spec[3:].split(","))
                    cfg = pc.minimal_config((grid.x, grid.y), m_tiles, k_tiles, n_tiles, blocks)
                    fn = lambda cfg=cfg: ttnn.experimental.minimal_matmul(
                        x_dev,
                        w_dev,
                        bias_tensor=b_dev,
                        fused_activation=pc.fused_activation(act),
                        config=cfg,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        compute_kernel_config=kernel,
                    )
                out = ttnn.to_torch(fn()).float()
                if base is None:
                    base = out
                record.update(ms=time_op(device, fn) * 1e3, pcc_vs_auto=pcc(base, out))
            except Exception as e:
                record["error"] = str(e).splitlines()[0][:200]
            results.append(record)
            print(
                f"[matmul] {args.role} {record.get('config')}: {record.get('ms', float('nan')):.3f} ms, pcc {record.get('pcc_vs_auto', float('nan')):.6f} {record.get('error', '')}",
                flush=True,
            )
    finally:
        ttnn.close_device(device)
    payload = {
        "role": args.role,
        "rows": args.rows,
        "k": k_in,
        "n": n_out,
        "precision": args.precision,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
