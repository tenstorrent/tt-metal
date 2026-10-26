# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Round 1 program-config sweep: ``minimal_matmul`` blocks for the time layer's QKV, out_proj and FFN down
matmuls against the ``ttnn.linear`` auto config the model runs today, at the model's M.

    ttq run --model toto2 --part 2 --kind bench -- python models/experimental/toto2_forecast/tests/perf/sweep_matmul.py \
        --ckpt 313m --rows 98304 --out models/experimental/toto2_forecast/doc/optimize/iter_01/matmul_sweep_313m.json

Real layer-0 weights (bias where the model has one) on a normal input of ``--rows`` tokens. Each config: device
time per call (host-timed over synchronized batches, as ``sweep_swiglu.py``) and PCC against the linear output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.experimental.toto2_forecast.tests.perf.sweep_swiglu import pcc, time_op

CONFIGS = [  # (M_block, K_block, N_block, subblock_h, subblock_w)
    (8, 8, 8, 2, 4),
    (8, 16, 8, 2, 4),
    (8, 32, 8, 2, 4),
    (8, 8, 4, 2, 2),
    (8, 16, 4, 2, 2),
    (8, 32, 4, 2, 2),
    (16, 8, 8, 4, 2),
    (16, 16, 4, 4, 1),
    (8, 8, 16, 1, 4),
    (4, 16, 8, 1, 4),
    (16, 8, 16, 2, 4),
    (8, 32, 2, 2, 2),
]


def main():
    import ttnn

    from models.experimental.toto2_forecast.tt.common import (
        Toto2Precision,
        linear,
        pad_to_tile,
        upload,
        upload_bias,
        upload_linear,
    )
    from models.experimental.toto2_forecast.tt.weights import Toto2Weights

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="313m")
    ap.add_argument("--rows", type=int, default=98304)
    ap.add_argument("--precision", default="accuracy")
    ap.add_argument("--roles", default="qkv,out,down")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--configs", type=json.loads, default=CONFIGS, help="JSON list of [M, K, N, sub_h, sub_w]")
    args = ap.parse_args()
    prec = Toto2Precision.named(args.precision)
    w = Toto2Weights.from_checkpoint(args.ckpt)
    lw, d, f = w.layers[0], w.config.d_model, w.config.d_ff
    fp = pad_to_tile(f)
    roles = {
        "qkv": (d, lw.w_qkv, lw.b_qkv, prec.weight_dtype(), prec, {}),
        "out": (d, lw.w_o, lw.b_o, prec.weight_dtype("out"), prec, {}),
        "down": (fp, lw.w_fc2, None, prec.weight_dtype("ffn_out"), prec.for_role("ffn"), {"pad_in": fp}),
    }
    g = torch.Generator().manual_seed(5)
    device = ttnn.open_device(device_id=0)
    grid = device.compute_with_storage_grid_size()
    results = {}
    try:
        for role in args.roles.split(","):
            k, weight, bias, dtype, p, kw = roles[role]
            x_dev = upload(device, torch.randn(1, 1, args.rows, k, generator=g))
            w_dev = upload_linear(device, weight, dtype, **kw)
            b_dev = upload_bias(device, bias) if bias is not None else None
            kernel = p.matmul_config()
            base_fn = lambda: linear(x_dev, w_dev, p, bias=b_dev)
            base = ttnn.to_torch(base_fn()).float()
            rows = [{"config": "linear (round 0)", "ms": time_op(device, base_fn) * 1e3}]
            print(f"[{role}] {rows[0]}", flush=True)
            for mb, kb, nb, sh, sw in args.configs:
                kt = k // 32
                if kt % kb:
                    continue
                cfg = ttnn.MinimalMatmulConfig(
                    M_block_size=mb,
                    K_block_size=kb,
                    N_block_size=nb,
                    subblock_h=sh,
                    subblock_w=sw,
                    compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
                )
                fn = lambda: ttnn.experimental.minimal_matmul(
                    x_dev,
                    w_dev,
                    bias_tensor=b_dev,
                    config=cfg,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    compute_kernel_config=kernel,
                )
                rec = {"config": [mb, kb, nb, sh, sw]}
                try:
                    got = ttnn.to_torch(fn()).float()
                    rec.update(ms=time_op(device, fn) * 1e3, pcc_vs_linear=pcc(base, got))
                except Exception as e:  # illegal block / L1 overflow
                    rec["error"] = str(e).splitlines()[0][:200]
                rows.append(rec)
                print(f"[{role}] {rec}", flush=True)
            results[role] = rows
            for t in (x_dev, w_dev, b_dev):
                if t is not None:
                    ttnn.deallocate(t)
    finally:
        ttnn.close_device(device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"ckpt": args.ckpt, "rows": args.rows, "precision": args.precision, "roles": results}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
