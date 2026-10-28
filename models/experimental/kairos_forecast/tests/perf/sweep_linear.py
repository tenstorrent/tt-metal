# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""R1-4: sweep the program config of every linear role at paper (512 rows), both presets, in one job.

    KAIROS_SWEEP_JSON=out.jsonl pytest -q -s models/experimental/kairos_forecast/tests/perf/sweep_linear.py

Each role runs with the model's dtypes and fidelity (tt/common.Precision), on random inputs. Per config: the
synchronized wall ms (median of 5 after 2 warm-ups) and the max abs difference to the default config's output
(another in0_block_w changes the accumulation order, so 0 is not expected; check_gates decides). "default" is the
model's current auto config (relu as a separate op where the role has one); the configs fuse relu in the epilogue
(a precision change, D3 R1-4, disclosed)."""

import itertools
import json
import os
import time

import pytest
import torch
import ttnn

from models.experimental.kairos_forecast.tt.common import Precision

GRID = (11, 10)
ENC_M, DEC_M = 512 * 96, 512 * 32
# role: (M, K, N, keep_bf16 weight, relu)
ROLES = {
    "enc_qkv": (ENC_M, 512, 1536, False, False),
    "enc_o": (ENC_M, 512, 512, True, False),
    "enc_wi": (ENC_M, 512, 2048, False, True),
    "enc_wo": (ENC_M, 2048, 512, True, False),
    "cross_kv": (ENC_M, 512, 1536, False, False),
    "dec_qkv": (DEC_M, 512, 1536, False, False),
    "dec_o": (DEC_M, 512, 512, True, False),
    "dec_wi": (DEC_M, 512, 2048, False, True),
    "dec_wo": (DEC_M, 2048, 512, True, False),
}


def _divs(n, cap):
    return [d for d in range(1, min(n, cap) + 1) if n % d == 0]


def configs(M, K, N, relu, dst):
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    act = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if relu else None
    yield "default", None
    # 1D: the whole N per core, M split over 110 cores
    pcm = -(-Mt // (GRID[0] * GRID[1]))
    for kw, sw, bh in itertools.product(_divs(Kt, 8)[-3:], _divs(Nt, dst)[-2:], _divs(pcm, 64)[-3:]):
        sh = max(d for d in _divs(bh, dst) if d * sw <= dst)
        if Nt * bh > 64 * 8 or sw * sh > dst:
            continue
        yield f"1d_kw{kw}_sb{sh}x{sw}_bh{bh}", ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=GRID,
            in0_block_w=kw,
            out_subblock_h=sh,
            out_subblock_w=sw,
            out_block_h=bh,
            out_block_w=Nt,
            per_core_M=pcm,
            per_core_N=Nt,
            fuse_batch=True,
            fused_activation=act,
            mcast_in0=False,
        )
    # 2D: M over the 10 rows, N over the 11 columns
    pm, pn = -(-Mt // GRID[1]), -(-Nt // GRID[0])
    for kw in _divs(Kt, 8)[-3:]:
        for sw in _divs(pn, dst):
            for sh in _divs(pm, dst):
                if sh * sw > dst or sh * sw < min(dst, pn * pm) // 2:
                    continue
                for bh in [d for d in _divs(pm, 64) if d % sh == 0][-3:]:
                    yield f"2d_kw{kw}_sb{sh}x{sw}_bh{bh}", ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=GRID,
                        in0_block_w=kw,
                        out_subblock_h=sh,
                        out_subblock_w=sw,
                        out_block_h=bh,
                        out_block_w=pn,
                        per_core_M=pm,
                        per_core_N=pn,
                        transpose_mcast=False,
                        fused_activation=act,
                    )


def _time(fn, device, reps=5):
    for _ in range(2):
        ttnn.deallocate(fn())
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(device)
        t = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t) * 1e3)
        ttnn.deallocate(out)
    return sorted(ts)[len(ts) // 2]


@pytest.mark.parametrize("precision", ["accuracy", "performance"])
@pytest.mark.parametrize("role", list(ROLES))
def test_sweep(device, role, precision):
    M, K, N, keep, relu = ROLES[role]
    p = getattr(Precision, precision)()
    dst = 4 if p.fp32_dest_acc_en else 8
    g = torch.Generator().manual_seed(0)
    a = ttnn.from_torch(
        torch.randn(1, 1, M, K, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    w = ttnn.from_torch(
        torch.randn(K, N, generator=g) / K**0.5,
        dtype=p.weight_dtype(keep_bf16=keep),
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    ref, rows = None, []
    for name, cfg in configs(M, K, N, relu, dst):

        def fn():
            kw = dict(compute_kernel_config=p.mm(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if cfg is None:
                return ttnn.linear(a, w, activation="relu" if relu else None, **kw)
            return ttnn.linear(a, w, program_config=cfg, **kw)

        try:
            out = ttnn.to_torch(fn()).float()
            ms = _time(fn, device)
        except Exception as e:  # L1 overflow, invalid config
            rows.append({"config": name, "error": str(e).splitlines()[0][:160]})
            continue
        if ref is None:
            ref = out
        diff = float((out - ref).abs().max())
        rows.append({"config": name, "ms": ms, "max_abs_diff": diff})
    ok = [r for r in rows if "ms" in r]
    best = min(ok, key=lambda r: r["ms"])
    print(
        f"BEST {role} {precision} {best['config']} {best['ms']:.3f} ms vs default {ok[0]['ms']:.3f} "
        f"({len(ok)} ok, {len(rows) - len(ok)} failed)"
    )
    if path := os.environ.get("KAIROS_SWEEP_JSON"):
        with open(path, "a") as f:
            f.write(json.dumps({"role": role, "precision": precision, "rows": rows}) + "\n")
