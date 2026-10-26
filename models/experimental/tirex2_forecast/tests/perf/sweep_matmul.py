# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""R1-2f: per-role dense matmul configs at the post-R1-1 shapes (M = 1024 rows x 128 tokens), one job.

    TIREX2_SWEEP_OUT=<dir> python -m pytest .../tests/perf/sweep_matmul.py -s        (under ttq --kind bench)

For every role, ``ttnn.linear`` with the default config (the baseline) against ``minimal_matmul`` with a grid of
(M, K, N) blocks and subblocks, at the model's precision (accuracy: HiFi4, fp32 accumulation). Device time is
the median of 10 synchronized calls; every candidate's output is compared with the baseline's (PCC). Writes
``matmul_sweep.json`` / ``.md`` (the per-role table).
"""

from __future__ import annotations

import itertools
import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch

M_ROWS = 1024 * 128
# role: (K, N, input dtype, output dtype, weight dtype)
ROLES = {
    "mlstm_qkv": (512, 1536, "bf16", "bf16"),
    "mlstm_o_out": (512, 512, "bf16", "bf16"),
    "ffn_down": (2048, 512, "bf16", "bf16"),
    "w_in": (512, 2048, "bf16", "fp32"),
    "embed_hidden": (64, 512, "bf16", "bf16"),
    "head_out": (512, 288, "bf16", "bf16"),
}
BLOCKS = [(8, 8, 8), (8, 16, 8), (16, 8, 8), (8, 8, 4), (4, 8, 8), (16, 16, 4), (8, 16, 16)]
SUBBLOCKS = [(2, 2), (1, 4), (4, 1)]
ITERS = 10


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_sweep_matmul(device):
    import ttnn

    from models.experimental.tirex2_forecast.tt.common import kernel_config

    dt = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32}
    cfg = kernel_config(ttnn.MathFidelity.HiFi4)
    grid = device.compute_with_storage_grid_size()
    out_dir = Path(os.environ.get("TIREX2_SWEEP_OUT", "/tmp/tirex2_mm"))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    results = []

    def timed(fn):
        out = fn()
        ttnn.synchronize_device(device)
        ts = []
        for _ in range(ITERS):
            ttnn.deallocate(out)
            t0 = time.perf_counter()
            out = fn()
            ttnn.synchronize_device(device)
            ts.append(time.perf_counter() - t0)
        return out, statistics.median(ts)

    for role, (K, N, din, dout) in ROLES.items():
        x = ttnn.from_torch(torch.randn(1, 1, M_ROWS, K), dtype=dt[din], layout=ttnn.TILE_LAYOUT, device=device)
        w = ttnn.from_torch(torch.randn(K, N) / K**0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        base, base_s = timed(
            lambda: ttnn.linear(x, w, dtype=dt[dout], compute_kernel_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        )
        ref = ttnn.to_torch(base).float()
        ttnn.deallocate(base)
        rows = [{"role": role, "config": "linear default", "ms": base_s * 1e3, "pcc": 1.0}]
        kt, nt = K // 32, N // 32
        for (mb, kb, nb), (sh, sw) in itertools.product(BLOCKS, SUBBLOCKS):
            kb, nb = min(kb, kt), min(nb, nt)
            if kt % kb or mb % sh or nb % sw or sh * sw > 4:
                continue
            name = f"minimal M{mb} K{kb} N{nb} sub{sh}x{sw}"
            if any(r["config"] == name for r in rows):
                continue
            mcfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kb,
                N_block_size=nb,
                subblock_h=sh,
                subblock_w=sw,
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            )
            try:
                out, s = timed(
                    lambda: ttnn.experimental.minimal_matmul(
                        x,
                        w,
                        config=mcfg,
                        compute_kernel_config=cfg,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        dtype=dt[dout],
                    )
                )
            except Exception as e:  # noqa: BLE001 - a config the op rejects is a sweep result too
                rows.append({"role": role, "config": name, "error": str(e).splitlines()[0][:120]})
                continue
            rows.append({"role": role, "config": name, "ms": s * 1e3, "pcc": _pcc(ref, ttnn.to_torch(out).float())})
            ttnn.deallocate(out)
        ttnn.deallocate(x)
        ttnn.deallocate(w)
        best = min((r for r in rows if "ms" in r and r["pcc"] > 0.9999), key=lambda r: r["ms"])
        print(f"[SWEEP] {role}: default {base_s * 1e3:.2f} ms, best {best['config']} {best['ms']:.2f} ms")
        results += rows
        (out_dir / "matmul_sweep.json").write_text(json.dumps(results, indent=1) + "\n")
    lines = [
        "# R1-2f matmul sweep (M = 131072, HiFi4, fp32 acc; median of 10)",
        "",
        "| role | config | ms | pcc |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['role']} | {r['config']} | {r.get('ms', float('nan')):.2f} | {r.get('pcc', r.get('error', ''))} |"
        )
    (out_dir / "matmul_sweep.md").write_text("\n".join(lines) + "\n")
