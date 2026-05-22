# SPDX-FileCopyrightText: (C) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
SDPA config sweep for the dots_ocr vision-tower attention.

Production shape (TP=1 reference, Wormhole B0):

    B=1, H=12, S=12288 (tile-padded), D=128
    Q dtype = BFLOAT8_B
    K dtype = BFLOAT8_B
    V dtype = BFLOAT4_B   (production path; halves V DRAM stream)
    is_causal = False
    attn_mask = optional (key-padding mask; lives in DRAM)
    math_fidelity = LoFi  (production)
    exp_approx_mode = True

Baseline (production config in modules/dots_ocr_vision.py):
    q_chunk = 256, k_chunk = 512, grid = 8x8, LoFi, exp_approx_mode=True
    Tracy shows ~12,246 us / SDPA call.

This test sweeps the levers that can move that number, organised in
named groups so the .txt perf report can be diffed config-by-config:

    Group 1 (Q/K/V memory layout)
        DRAM interleaved (production)
        L1 interleaved (negative — Q/K/V too big at S=12288)
        Sharded inputs (negative — SDPA validates !is_sharded())

    Group 2 (chunk-size sweep at production grid 8x8, LoFi, BFP4 V)
        q256_k512  (production baseline)
        q512_k256  (Tracy ablation: +10%)
        q128_k128
        q256_k256
        q128_k256
        q64_k512
        q256_k128
        q128_k512
        q512_k512
        q64_k256

    Group 3 (compute grid sweep)
        grid_8x8 (production)
        grid_8x4
        grid_8x2
        grid_4x4

    Group 4 (math fidelity sweep)
        lofi  (production)
        hifi2
        hifi4

    Group 5 (exp_approx_mode)
        approx_on  (production)
        approx_off

    Group 6 (V-dtype sweep)
        v_bfp4  (production — halves V DRAM stream)
        v_bfp8
        v_bf16

    Group 7 (combined candidate configs that may beat baseline)
        cand_q256_k512_bfp4_grid8x8_lofi  (baseline reference)
        cand_q128_k256_bfp4_grid8x8_lofi
        cand_q128_k128_bfp4_grid8x8_lofi
        cand_q256_k1024_bfp4_grid8x8_lofi
        cand_q512_k1024_bfp4_grid8x8_lofi

Notes on legality (sdpa_device_operation.cpp):
    L40  : Q/K/V must be BF16, BFP8_B, or BFP4_B
    L38  : Layout::TILE
    L44  : !is_sharded()   — sharded Q/K/V FAILS validation
    L78  : attn_mask must live in DRAM
    L73  : attn_mask dtype must be BF16/BFP8/BFP4

Run:
    pytest sdpa_tests/test_dots_ocr_vision_sdpa_configs.py -s -v

Or filter to a single group:
    pytest sdpa_tests/test_dots_ocr_vision_sdpa_configs.py -s -v \
        -k "chunk_sweep"
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


# ---------------------------------------------------------------------------
# Shape (dots_ocr vision tower, TP=1 reference)
# ---------------------------------------------------------------------------

B = 1
H = 12  # num_heads
NKV = 12  # MHA (num_kv_heads == num_heads)
S = 12288  # tile-aligned vision-prefill seq len
D = 128  # head_dim
TILE = 32

NUM_WARMUP = 0
NUM_ITERS = 1
# TARGET_US = 12300.0   # production baseline observed via Tracy
# ROOFLINE_US = 8000.0  # rough soft floor for reporting only


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


@dataclass
class SDPACfg:
    """One concrete SDPA invocation knob set."""

    name: str
    q_chunk: int
    k_chunk: int
    grid_x: int = 8
    grid_y: int = 8
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi
    exp_approx_mode: bool = True
    q_dtype: ttnn.DataType = ttnn.bfloat8_b
    k_dtype: ttnn.DataType = ttnn.bfloat8_b
    v_dtype: ttnn.DataType = ttnn.bfloat4_b
    qkv_mem: ttnn.MemoryConfig = field(default_factory=lambda: ttnn.DRAM_MEMORY_CONFIG)
    out_mem: ttnn.MemoryConfig = field(default_factory=lambda: ttnn.L1_MEMORY_CONFIG)
    sharded: bool = False  # for negative-test variants
    use_attn_mask: bool = False
    must_fail: bool = False  # for negative variants we expect to raise
    expected_error_substr: str = ""


def _compute_cfg(device, math_fidelity, math_approx_mode=True):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=math_approx_mode,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _sdpa_program_config(device, cfg: SDPACfg):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cfg.grid_x, cfg.grid_y),
        q_chunk_size=cfg.q_chunk,
        k_chunk_size=cfg.k_chunk,
        exp_approx_mode=cfg.exp_approx_mode,
    )


def _sharded_qkv_mem(device, dtype: ttnn.DataType) -> ttnn.MemoryConfig:
    """Build a HEIGHT_SHARDED memory config for test coverage.

    SDPA validation forbids sharded Q/K/V (sdpa_device_operation.cpp:44),
    so this is only here so the customer report shows the explicit
    failure mode rather than an unexplained gap.
    """
    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    num_cores = grid_x * grid_y
    shard_h = (S + num_cores - 1) // num_cores
    # round up to a tile multiple
    shard_h = ((shard_h + TILE - 1) // TILE) * TILE
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
    return ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=ttnn.ShardSpec(shard_grid, [shard_h, D], ttnn.ShardOrientation.ROW_MAJOR),
    )


# ---------------------------------------------------------------------------
# Config registry — grouped by ablation axis
# ---------------------------------------------------------------------------
#
# Each tuple: (group, SDPACfg).
#
# Sequence-divisibility rule: q_chunk and k_chunk must each divide S=12288
# without remainder (otherwise SDPA refuses the program config). 12288 =
# 2^12 * 3, so legal chunk sizes are {32, 64, 96, 128, 192, 256, 384, 512,
# 768, 1024, 1536, ...}. We stick to power-of-two chunks below.
#

# -- Group 2: chunk-size sweep (baseline knob set) --------------------------
CHUNK_SWEEP: list[SDPACfg] = [
    SDPACfg("chunk_q256_k512_baseline", 256, 512),
    SDPACfg("chunk_q512_k256", 512, 256),
    SDPACfg("chunk_q128_k128", 128, 128),
    SDPACfg("chunk_q256_k256", 256, 256),
    SDPACfg("chunk_q128_k256", 128, 256),
    SDPACfg("chunk_q64_k512", 64, 512),
    SDPACfg("chunk_q256_k128", 256, 128),
    SDPACfg("chunk_q128_k512", 128, 512),
    SDPACfg("chunk_q512_k512", 512, 512),
    SDPACfg("chunk_q64_k256", 64, 256),
]

# -- Group 3: compute grid sweep -------------------------------------------
GRID_SWEEP: list[SDPACfg] = [
    SDPACfg("grid_8x8", 256, 512, grid_x=8, grid_y=8),
    SDPACfg("grid_8x4", 256, 512, grid_x=8, grid_y=4),
    SDPACfg("grid_8x2", 256, 512, grid_x=8, grid_y=2),
    SDPACfg("grid_4x4", 256, 512, grid_x=4, grid_y=4),
]

# -- Group 4: math fidelity ------------------------------------------------
FIDELITY_SWEEP: list[SDPACfg] = [
    SDPACfg("fidelity_lofi_baseline", 256, 512, math_fidelity=ttnn.MathFidelity.LoFi),
    SDPACfg("fidelity_hifi2", 256, 512, math_fidelity=ttnn.MathFidelity.HiFi2),
    SDPACfg("fidelity_hifi4", 256, 512, math_fidelity=ttnn.MathFidelity.HiFi4),
]


def _mem_variant_name(cfg: SDPACfg) -> str:
    if cfg.qkv_mem == ttnn.DRAM_MEMORY_CONFIG and cfg.out_mem == ttnn.L1_MEMORY_CONFIG:
        return "mem_qkv_dram_out_l1"
    if cfg.qkv_mem == ttnn.DRAM_MEMORY_CONFIG and cfg.out_mem == ttnn.DRAM_MEMORY_CONFIG:
        return "mem_qkv_dram_out_dram"
    if cfg.qkv_mem == ttnn.L1_MEMORY_CONFIG and cfg.out_mem == ttnn.L1_MEMORY_CONFIG:
        return "mem_qkv_l1_out_l1"
    if cfg.sharded:
        return "mem_qkv_sharded"
    return "mem_custom"


def _fidelity_name(fidelity: ttnn.MathFidelity) -> str:
    return str(fidelity).split(".")[-1].lower()


# -- Group 5: exp_approx_mode ----------------------------------------------
APPROX_SWEEP: list[SDPACfg] = [
    SDPACfg("approx_on_baseline", 256, 512, exp_approx_mode=True),
    SDPACfg("approx_off", 256, 512, exp_approx_mode=False),
]

# -- Group 6: V dtype ------------------------------------------------------
VDTYPE_SWEEP: list[SDPACfg] = [
    SDPACfg("vdtype_bfp4_baseline", 256, 512, v_dtype=ttnn.bfloat4_b),
    SDPACfg("vdtype_bfp8", 256, 512, v_dtype=ttnn.bfloat8_b),
    SDPACfg("vdtype_bf16", 256, 512, v_dtype=ttnn.bfloat16),
]

# -- Group 1: memory-layout sweep ------------------------------------------
# DRAM_MEMORY_CONFIG vs L1_MEMORY_CONFIG vs a sharded (illegal) variant.
# L1 interleaved Q+K+V at S=12288 (~36 MB BFP8 + ~10 MB BFP4) does not fit
# inside the 64-core L1 budget on a single chip — we expect this to OOM at
# allocation time, which is itself the customer-relevant signal.
MEM_SWEEP: list[SDPACfg] = [
    SDPACfg(
        "mem_qkv_dram_out_l1_baseline",
        256,
        512,
        qkv_mem=ttnn.DRAM_MEMORY_CONFIG,
        out_mem=ttnn.L1_MEMORY_CONFIG,
    ),
    SDPACfg(
        "mem_qkv_dram_out_dram",
        256,
        512,
        qkv_mem=ttnn.DRAM_MEMORY_CONFIG,
        out_mem=ttnn.DRAM_MEMORY_CONFIG,
    ),
    SDPACfg(
        "mem_qkv_l1_out_l1",
        256,
        512,
        qkv_mem=ttnn.L1_MEMORY_CONFIG,
        out_mem=ttnn.L1_MEMORY_CONFIG,
        # must_fail=True,
        # expected_error_substr="L1",
    ),
    SDPACfg(
        "mem_qkv_sharded_negative",
        256,
        512,
        sharded=True,
        # must_fail=True,
        # expected_error_substr="sharded",
    ),
]

# -- Cross product: memory layout x chunks x math fidelity -------------------
# Keep sharded Q/K/V above as a single expected-fail row; SDPA validates that
# prefill Q/K/V are interleaved, independent of chunk/fidelity knobs.
CROSS_PRODUCT_MEMORIES: list[SDPACfg] = [
    SDPACfg(
        "mem_qkv_dram_out_l1",
        256,
        512,
        qkv_mem=ttnn.DRAM_MEMORY_CONFIG,
        out_mem=ttnn.L1_MEMORY_CONFIG,
    ),
    SDPACfg(
        "mem_qkv_dram_out_dram",
        256,
        512,
        qkv_mem=ttnn.DRAM_MEMORY_CONFIG,
        out_mem=ttnn.DRAM_MEMORY_CONFIG,
    ),
    SDPACfg(
        "mem_qkv_l1_out_l1",
        256,
        512,
        qkv_mem=ttnn.L1_MEMORY_CONFIG,
        out_mem=ttnn.L1_MEMORY_CONFIG,
    ),
]

CROSS_PRODUCT_SWEEP: list[SDPACfg] = [
    SDPACfg(
        f"xprod_{_mem_variant_name(mem)}__q{chunk.q_chunk}_k{chunk.k_chunk}__{_fidelity_name(fid.math_fidelity)}",
        chunk.q_chunk,
        chunk.k_chunk,
        qkv_mem=mem.qkv_mem,
        out_mem=mem.out_mem,
        math_fidelity=fid.math_fidelity,
        exp_approx_mode=chunk.exp_approx_mode,
        q_dtype=chunk.q_dtype,
        k_dtype=chunk.k_dtype,
        v_dtype=chunk.v_dtype,
    )
    for mem in CROSS_PRODUCT_MEMORIES
    for chunk in CHUNK_SWEEP
    for fid in FIDELITY_SWEEP
]

# -- Group 7: combined candidate configs ----------------------------------
# Candidates worth comparing head-to-head against the baseline for a
# "best combo" search; results are reported sorted so the winner is
# obvious in the .txt report.
#
# Rationale for the shortlist:
#   * Production SDPA at S=12288 is V-DRAM-bandwidth-bound: per-core SDPA
#     re-reads V tiles ``q_per_core`` times during prefill (10 outer-Q
#     iterations at q_chunk=256). Larger k_chunk cuts the number of outer
#     K/V passes, which is the most likely single-knob win.
#   * BFP4 V (production) is the cheaper V stream; BFP8 V doubles the V
#     bytes per pass, so it only helps if a much larger k_chunk amortises
#     the extra bandwidth. Try a few BFP8-V + large-k_chunk combos.
#   * Output in DRAM removes L1 pressure on the partial-output CB, which
#     can let larger chunks fit; included as a separate row.
#   * Key-padding mask matches what _sdpa_padded_with_key_mask actually
#     feeds when attention_logical_seq_len < S_pad.
CANDIDATES: list[SDPACfg] = [
    # EXACT current production knob set — Q/K BFP8, V BFP4, 8x8, LoFi,
    # exp_approx_mode=True, q=256, k=512. Anchor row.
    SDPACfg("cand_q256_k512_bfp4_g8x8_lofi_baseline", 256, 512),
    # Chunk variations on the production data path (BFP4 V, 8x8, LoFi).
    SDPACfg("cand_q128_k256_bfp4_g8x8_lofi", 128, 256),
    SDPACfg("cand_q128_k128_bfp4_g8x8_lofi", 128, 128),
    SDPACfg("cand_q256_k1024_bfp4_g8x8_lofi", 256, 1024),
    SDPACfg("cand_q512_k1024_bfp4_g8x8_lofi", 512, 1024),
    SDPACfg("cand_q256_k384_bfp4_g8x8_lofi", 256, 384),
    SDPACfg("cand_q384_k256_bfp4_g8x8_lofi", 384, 256),
    SDPACfg("cand_q192_k384_bfp4_g8x8_lofi", 192, 384),
    # Larger k_chunk hunt — fewer outer K/V passes. 12288/1536 = 8 passes,
    # 12288/2048 would be illegal (12288 % 2048 != 0) so capped at 1536.
    SDPACfg("cand_q256_k1536_bfp4_g8x8_lofi", 256, 1536),
    SDPACfg("cand_q128_k1536_bfp4_g8x8_lofi", 128, 1536),
    SDPACfg("cand_q384_k1536_bfp4_g8x8_lofi", 384, 1536),
    # Output to DRAM (frees L1 partial-output budget for bigger chunks).
    SDPACfg("cand_q256_k1024_bfp4_g8x8_lofi_outDram", 256, 1024, out_mem=ttnn.DRAM_MEMORY_CONFIG),
    SDPACfg("cand_q256_k1536_bfp4_g8x8_lofi_outDram", 256, 1536, out_mem=ttnn.DRAM_MEMORY_CONFIG),
    # BFP8 V at larger k_chunk (only competitive if k_chunk amortises the
    # extra V bandwidth vs BFP4).
    SDPACfg("cand_q256_k1024_bfp8V_g8x8_lofi", 256, 1024, v_dtype=ttnn.bfloat8_b),
    SDPACfg("cand_q256_k1536_bfp8V_g8x8_lofi", 256, 1536, v_dtype=ttnn.bfloat8_b),
    # HiFi2 + BFP4 V at the baseline chunk shape — does the small fidelity
    # bump cost less than the V bandwidth win expected from BFP4? Only
    # meaningful if Tracy says the per-tile math is dwarfed by V DRAM time.
    SDPACfg("cand_q256_k512_bfp4_g8x8_hifi2", 256, 512, math_fidelity=ttnn.MathFidelity.HiFi2),
    # Key-padding mask path (matches _sdpa_padded_with_key_mask). Mask
    # lives in DRAM (sdpa_device_operation.cpp:78).
    SDPACfg("cand_q256_k512_with_keymask", 256, 512, use_attn_mask=True),
    SDPACfg("cand_q256_k1024_with_keymask", 256, 1024, use_attn_mask=True),
]


GROUPS: dict[str, list[SDPACfg]] = {
    "mem_sweep": MEM_SWEEP,
    "chunk_sweep": CHUNK_SWEEP,
    "grid_sweep": GRID_SWEEP,
    "fidelity_sweep": FIDELITY_SWEEP,
    "approx_sweep": APPROX_SWEEP,
    "vdtype_sweep": VDTYPE_SWEEP,
    "cross_product": CROSS_PRODUCT_SWEEP,
    "candidates": CANDIDATES,
}

# Flat list of (group, cfg) tuples for the @pytest.mark.parametrize call.
_ALL: list[tuple[str, SDPACfg]] = [(g, c) for g, lst in GROUPS.items() for c in lst]
_ALL_IDS = [f"{g}::{c.name}" for g, c in _ALL]


# ---------------------------------------------------------------------------
# Reference (slow torch SDPA)
# ---------------------------------------------------------------------------


def _torch_reference(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attn_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(
        Q.to(torch.float32),
        K.to(torch.float32),
        V.to(torch.float32),
        attn_mask=attn_mask.to(torch.float32) if attn_mask is not None else None,
        is_causal=False,
    ).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Single-config runner
# ---------------------------------------------------------------------------


def _run_one(device, cfg: SDPACfg, *, check_pcc: bool):
    torch.manual_seed(0)

    # ---- Sequence-divisibility guard -------------------------------------
    if S % cfg.q_chunk != 0 or S % cfg.k_chunk != 0:
        pytest.skip(f"S={S} not divisible by q_chunk={cfg.q_chunk} or k_chunk={cfg.k_chunk}")

    # ---- Torch tensors (BF16) --------------------------------------------
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16) * 0.1
    K = torch.randn(B, NKV, S, D, dtype=torch.bfloat16) * 0.1
    V = torch.randn(B, NKV, S, D, dtype=torch.bfloat16) * 0.1

    attn_mask_torch = None
    tt_mask = None
    if cfg.use_attn_mask:
        # Vision tower style: -inf the trailing 8 query/key tiles to emulate
        # a logical-seq-len shorter than the tile-padded extent.
        logical = S - 8 * TILE
        m = torch.zeros(1, 1, S, S, dtype=torch.float32)
        m[..., logical:] = -1e9
        attn_mask_torch = m
        tt_mask = ttnn.from_torch(
            m,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=device,
        )

    # ---- TT tensors ------------------------------------------------------
    if cfg.sharded:
        qkv_mem = _sharded_qkv_mem(device, cfg.q_dtype)
    else:
        qkv_mem = cfg.qkv_mem

    def _err_msg(stage):
        return f"[{cfg.name}] {stage} failed"

    if cfg.must_fail:
        # Negative path: assert ttnn raises during input upload OR sdpa call.
        with pytest.raises(Exception) as excinfo:
            tt_Q = ttnn.from_torch(
                Q, dtype=cfg.q_dtype, layout=ttnn.TILE_LAYOUT, memory_config=qkv_mem, device=device, pad_value=0.0
            )
            tt_K = ttnn.from_torch(
                K, dtype=cfg.k_dtype, layout=ttnn.TILE_LAYOUT, memory_config=qkv_mem, device=device, pad_value=0.0
            )
            tt_V = ttnn.from_torch(
                V, dtype=cfg.v_dtype, layout=ttnn.TILE_LAYOUT, memory_config=qkv_mem, device=device, pad_value=0.0
            )
            prog = _sdpa_program_config(device, cfg)
            compute = _compute_cfg(device, cfg.math_fidelity)
            _ = ttnn.transformer.scaled_dot_product_attention(
                tt_Q,
                tt_K,
                tt_V,
                is_causal=False,
                attn_mask=tt_mask,
                program_config=prog,
                compute_kernel_config=compute,
                memory_config=cfg.out_mem,
            )
            ttnn.synchronize_device(device)
        print(
            f"\n  [EXPECTED-FAIL] {cfg.name:<48} " f"raised: {type(excinfo.value).__name__}: {str(excinfo.value)[:120]}"
        )
        return

    tt_Q = ttnn.from_torch(
        Q, dtype=cfg.q_dtype, layout=ttnn.TILE_LAYOUT, memory_config=qkv_mem, device=device, pad_value=0.0
    )
    tt_K = ttnn.from_torch(
        K, dtype=cfg.k_dtype, layout=ttnn.TILE_LAYOUT, memory_config=qkv_mem, device=device, pad_value=0.0
    )
    tt_V = ttnn.from_torch(
        V, dtype=cfg.v_dtype, layout=ttnn.TILE_LAYOUT, memory_config=qkv_mem, device=device, pad_value=0.0
    )

    prog = _sdpa_program_config(device, cfg)
    compute = _compute_cfg(device, cfg.math_fidelity)

    def _run():
        return ttnn.transformer.scaled_dot_product_attention(
            tt_Q,
            tt_K,
            tt_V,
            is_causal=False,
            attn_mask=tt_mask,
            program_config=prog,
            compute_kernel_config=compute,
            memory_config=cfg.out_mem,
        )

    # warmup
    for _ in range(NUM_WARMUP):
        out = _run()
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)

    # timed loop
    last_out = None
    start = time.perf_counter()
    for _ in range(NUM_ITERS):
        if last_out is not None:
            ttnn.deallocate(last_out)
        last_out = _run()
    ttnn.synchronize_device(device)
    elapsed_us = (time.perf_counter() - start) * 1e6 / NUM_ITERS

    # status = "PASS" if elapsed_us < TARGET_US else "SLOW"
    # print(
    #     f"\n  [{status}] {cfg.name:<48} "
    #     f"avg = {elapsed_us:7.1f} us "
    #     f"({elapsed_us / ROOFLINE_US:.2f}x soft-floor ~{ROOFLINE_US:.0f} us)"
    # )

    if check_pcc:
        result = ttnn.to_torch(last_out)
        result = result[:, :, :S, :]
        gt = _torch_reference(Q, K, V, attn_mask_torch)
        ok, pcc_msg = comp_pcc(gt, result, 0.97)
        print(f"        pcc: {pcc_msg}")
        assert ok, f"[{cfg.name}] PCC check failed: {pcc_msg}"

    ttnn.deallocate(last_out)
    ttnn.deallocate(tt_Q)
    ttnn.deallocate(tt_K)
    ttnn.deallocate(tt_V)
    if tt_mask is not None:
        ttnn.deallocate(tt_mask)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("group,cfg", _ALL, ids=_ALL_IDS)
def test_dots_ocr_vision_sdpa_configs(device, group: str, cfg: SDPACfg):
    """Time one SDPA config and (where viable) verify against torch SDPA.

    PCC is only checked for the baseline-shaped positive configs in the
    chunk_sweep and candidates groups — the rest are timing-only because
    the BFP4 V path and the various low-fidelity combinations expand the
    PCC tolerance enough that a strict 0.97 floor is misleading without
    per-config thresholds, and the customer ask here is the timing
    comparison.
    """
    pcc_groups = {"chunk_sweep", "candidates"}
    _run_one(device, cfg, check_pcc=(group in pcc_groups and not cfg.use_attn_mask))


# # ---------------------------------------------------------------------------
# # Convenience: print a recap so the .txt perf report ends with one table.
# # ---------------------------------------------------------------------------

# def test_zzz_print_recap():
#     """Always-pass sentinel that prints the full config registry.

#     Pytest collects in source order, so naming this with a 'zzz_' prefix
#     keeps it as the final line of the perf report and makes the table
#     easy to copy/paste into the .txt report file.
#     """
#     print("\n\n=== dots_ocr vision SDPA config registry ===")
#     for g, cfgs in GROUPS.items():
#         print(f"\n[{g}]")
#         for c in cfgs:
#             print(
#                 f"  {c.name:<48} "
#                 f"q={c.q_chunk:>4} k={c.k_chunk:>4} "
#                 f"grid={c.grid_x}x{c.grid_y} "
#                 f"fid={str(c.math_fidelity).split('.')[-1]:<5} "
#                 f"approx={int(c.exp_approx_mode)} "
#                 f"V={str(c.v_dtype).split('.')[-1]:<10} "
#                 f"mem={'sharded' if c.sharded else c.qkv_mem.buffer_type} "
#                 f"must_fail={int(c.must_fail)}"
#             )
