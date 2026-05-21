# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Sweep ttnn.transformer.scaled_dot_product_attention for the Swin-L window-attention
shapes used inside ATSS DyHead. Modeled on bge_m3/sweep_sdpa_resweep.py.

Each variant runs:
    1 warmup call (compile + cache)
    N timed back-to-back calls (default N=16), single ttnn.synchronize_device
    median wall-time printed and appended to CSV.

No PCC check — we'll PCC-gate only the winner end-to-end.

Stages (window=12 → S=144, head_dim=32 across all):
    S0: dim=192,  heads=6,   nW=196
    S1: dim=384,  heads=12,  nW=49
    S2: dim=768,  heads=24,  nW=16
    S3: dim=1536, heads=48,  nW=4
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from itertools import product

import pytest
import torch
from loguru import logger

import ttnn


# ── Workload ─────────────────────────────────────────────────────────────────

STAGES = [
    ("S0", 192, 6, 196),
    ("S1", 384, 12, 49),
    ("S2", 768, 24, 16),
    ("S3", 1536, 48, 4),
]

WINDOW_TOKENS = 144  # 12*12 window
REPEATS = int(os.environ.get("SWIN_SDPA_REPEATS", "8"))


# ── Variants ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SDPAVariant:
    name: str
    q_chunk: int
    k_chunk: int
    exp_approx: bool
    math_fidelity: ttnn.MathFidelity
    fp32_dest_acc_en: bool = False
    packer_l1_acc: bool = True

    def build_program_config(self, device):
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=self.q_chunk,
            k_chunk_size=self.k_chunk,
            exp_approx_mode=self.exp_approx,
        )

    def build_compute_kernel(self, device):
        return ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=self.math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            packer_l1_acc=self.packer_l1_acc,
        )


def _make_variants() -> list[SDPAVariant]:
    """BGE pattern: sweep one axis at a time, not the Cartesian product.
    Production reference is q=128, k=128, ea=False, HiFi2.
    """
    out: list[SDPAVariant] = []
    chunks = [32, 64, 128, 256, 512]

    # Production reference (always first).
    out.append(
        SDPAVariant(
            name="prod",
            q_chunk=128,
            k_chunk=128,
            exp_approx=False,
            math_fidelity=ttnn.MathFidelity.HiFi2,
        )
    )

    # Chunk grid at production fidelity / exp_approx.
    for q, k in product(chunks, repeat=2):
        if q == 128 and k == 128:
            continue  # = prod
        out.append(
            SDPAVariant(
                name=f"q{q}_k{k}",
                q_chunk=q,
                k_chunk=k,
                exp_approx=False,
                math_fidelity=ttnn.MathFidelity.HiFi2,
            )
        )

    # exp_approx_mode toggle at production chunks.
    out.append(
        SDPAVariant(
            name="prod_expapx",
            q_chunk=128,
            k_chunk=128,
            exp_approx=True,
            math_fidelity=ttnn.MathFidelity.HiFi2,
        )
    )

    # Math fidelity sweep at production chunks.
    out.append(
        SDPAVariant(
            name="hifi4",
            q_chunk=128,
            k_chunk=128,
            exp_approx=False,
            math_fidelity=ttnn.MathFidelity.HiFi4,
        )
    )
    out.append(
        SDPAVariant(
            name="lofi",
            q_chunk=128,
            k_chunk=128,
            exp_approx=False,
            math_fidelity=ttnn.MathFidelity.LoFi,
        )
    )

    # A couple combined candidates that are usually the practical winners.
    out.append(
        SDPAVariant(
            name="prod_expapx_lofi",
            q_chunk=128,
            k_chunk=128,
            exp_approx=True,
            math_fidelity=ttnn.MathFidelity.LoFi,
        )
    )
    out.append(
        SDPAVariant(
            name="q256_k256_expapx",
            q_chunk=256,
            k_chunk=256,
            exp_approx=True,
            math_fidelity=ttnn.MathFidelity.HiFi2,
        )
    )
    out.append(
        SDPAVariant(
            name="q32_k32_expapx_lofi",
            q_chunk=32,
            k_chunk=32,
            exp_approx=True,
            math_fidelity=ttnn.MathFidelity.LoFi,
        )
    )
    return out


VARIANTS = _make_variants()


# ── Inputs ───────────────────────────────────────────────────────────────────


def _make_inputs(device, dim, num_heads, nW):
    head_dim = dim // num_heads
    B = nW
    S = WINDOW_TOKENS
    torch.manual_seed(0)
    q_t = (torch.randn(B, num_heads, S, head_dim) * 0.05).to(torch.bfloat16)
    k_t = (torch.randn(B, num_heads, S, head_dim) * 0.05).to(torch.bfloat16)
    v_t = (torch.randn(B, num_heads, S, head_dim) * 0.05).to(torch.bfloat16)
    m_t = (torch.randn(B, num_heads, S, S) * 0.1).to(torch.bfloat16)

    def _tt(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return _tt(q_t), _tt(k_t), _tt(v_t), _tt(m_t)


def _sdpa(q, k, v, m, *, pcfg, ck):
    return ttnn.transformer.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=m,
        is_causal=False,
        scale=1.0,
        program_config=pcfg,
        compute_kernel_config=ck,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ── Test ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("stage_name,dim,num_heads,nW", STAGES, ids=lambda x: str(x))
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_sweep_swin_sdpa(device, stage_name, dim, num_heads, nW):
    q, k, v, m = _make_inputs(device, dim, num_heads, nW)

    out_dir = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated", "swin_sdpa_sweep")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"sweep_{stage_name}.csv")

    f_csv = open(csv_path, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(
        ["stage", "variant", "q_chunk", "k_chunk", "exp_approx", "fidelity", "median_us_per_call", "status"]
    )

    def emit(variant, median_us, status):
        writer.writerow(
            [
                stage_name,
                variant.name,
                variant.q_chunk,
                variant.k_chunk,
                int(variant.exp_approx),
                variant.math_fidelity.name,
                median_us if isinstance(median_us, str) else f"{median_us:.1f}",
                status,
            ]
        )
        f_csv.flush()

    n_ok = 0
    n_err = 0
    for variant in VARIANTS:
        try:
            pcfg = variant.build_program_config(device)
            ck = variant.build_compute_kernel(device)

            # 1 warmup
            out = _sdpa(q, k, v, m, pcfg=pcfg, ck=ck)
            ttnn.deallocate(out)
            ttnn.synchronize_device(device)

            # REPEATS timed back-to-back
            t0 = time.perf_counter()
            for _ in range(REPEATS):
                out = _sdpa(q, k, v, m, pcfg=pcfg, ck=ck)
                ttnn.deallocate(out)
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            us_per_call = (t1 - t0) * 1e6 / REPEATS
            emit(variant, us_per_call, "ok")
            n_ok += 1
        except Exception as e:
            emit(variant, "", f"ERR:{str(e)[:60]}")
            n_err += 1

    logger.info(f"[{stage_name}] {n_ok} ok / {n_err} err. CSV={csv_path}")

    f_csv.close()
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)
    ttnn.deallocate(m)
