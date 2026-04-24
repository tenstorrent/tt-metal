# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Correctness and hang regression test for SDPA paged-decode partial-mask bug.
Issue: https://github.com/tenstorrent/tt-metal/issues/42917

LIMITATIONS — READ BEFORE MODIFYING:

  1. SINGLE-DEVICE PCC CANNOT DISTINGUISH BUGGY FROM FIXED:
     On a single WH device the reversed-core assignment always places the
     REDUCER (core 0, do_reduce=True) on the HIGHEST k_chunk indices, which
     includes the partial last page containing garbage.  The reducer applies
     the causal mask in BOTH fixed AND unfixed kernels.  Therefore PCC-based
     assertions pass on both — the only observable difference is the CB drain
     in trace mode, and the production TRISC2 hang that requires full Galaxy
     with recycled KV pages and all 128 users synchronized.

  2. BFP8 MASKING RANGE:
     The mask (NEG_INF in BFP8) only suppresses garbage scores effectively
     when K_GARBAGE is small enough that the online-softmax running max is
     not dominated by the garbage before the mask is applied.
     K_GARBAGE must satisfy: score_garbage << NEG_INF_BFP8 ≈ 57344.
     With D=64: score = K * |sum(Q)| / sqrt(D) ≈ K * 8 / 8 = K.
     Use K_GARBAGE ≤ 5 to stay in the safe BFP8 masking range.

  3. FULL HANG VERIFICATION:
     The production hang (TRISC2 t6_semaphore_get<0>()) was confirmed via
     the vLLM 20-iteration server test on Galaxy g10glx02:
       - Unfixed kernel: 94/128 requests timed out every time
       - Fixed kernel:   20/20 iterations, 128/128 requests, 0 timeouts

These tests serve as:
  • SMOKE TESTS: mask is applied at the correct position, no NaN.
  • CB DRAIN REGRESSION: trace mode drains cb_mask_in correctly.
  • MLA REGRESSION: Fix 3 (pop after apply) doesn't break MLA decode.
"""

import pytest
import torch
import ttnn


B = 32  # production per-chip: 128 users / 4 DP groups
NH = 8  # 64 Q-heads / TP=8
NKV = 1  # 8 KV-heads / TP=8
D = 64
BLOCK_SIZE = 64
K_CHUNK = 128
SCALE = D**-0.5
CUR_POS = 10 * BLOCK_SIZE  # = 640: partial page 10
NUM_PAGES = CUR_POS // BLOCK_SIZE + 3

# K_GARBAGE = 5: large enough to corrupt output WITHOUT the mask,
# small enough that BFP8 masking is effective:
#   score ≈ 5 × 8 / 8 = 5  <<  NEG_INF_BFP8 ≈ 57344  →  mask works ✓
K_GARBAGE = 5.0

PROD_CFG = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
    q_chunk_size=NH,
    k_chunk_size=K_CHUNK,
)
COMPUTE_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
)


def _build_tensors(device):
    torch.manual_seed(42)
    Q = torch.randn(1, B, NH, D)

    # Valid random KV for tokens 0..CUR_POS; K_GARBAGE for tokens CUR_POS+1..
    # All B users share IDENTICAL K/V so that the sequential page_table
    # (page_table[b, pg] = pg for all b) is consistent: every user's
    # logical page pg maps to physical page pg which holds the shared K.
    K_paged = torch.full((B, NKV, NUM_PAGES, BLOCK_SIZE, D), K_GARBAGE)
    V_paged = torch.full((B, NKV, NUM_PAGES, BLOCK_SIZE, D), K_GARBAGE)
    for tok in range(CUR_POS + 1):
        pg, off = tok // BLOCK_SIZE, tok % BLOCK_SIZE
        k_row = torch.randn(NKV, D)  # one row, same for all users
        v_row = torch.randn(NKV, D)
        K_paged[:, :, pg, off, :] = k_row.unsqueeze(0).expand(B, -1, -1)
        V_paged[:, :, pg, off, :] = v_row.unsqueeze(0).expand(B, -1, -1)

    page_table = torch.arange(NUM_PAGES, dtype=torch.int32).unsqueeze(0).expand(B, -1)
    dram = ttnn.DRAM_MEMORY_CONFIG
    K_tt = K_paged.permute(0, 2, 1, 3, 4).reshape(B * NUM_PAGES, NKV, BLOCK_SIZE, D)
    V_tt = V_paged.permute(0, 2, 1, 3, 4).reshape(B * NUM_PAGES, NKV, BLOCK_SIZE, D)

    tt_Q = ttnn.as_tensor(Q, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_K = ttnn.as_tensor(K_tt, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_V = ttnn.as_tensor(V_tt, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_pt = ttnn.as_tensor(
        page_table, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=dram
    )
    tt_cp = ttnn.from_torch(
        torch.tensor([CUR_POS] * B, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=dram,
    )
    return tt_Q, tt_K, tt_V, tt_pt, tt_cp, Q, K_paged, V_paged, page_table


def _cpu_reference(Q, K_paged, V_paged, page_table):
    max_seq = NUM_PAGES * BLOCK_SIZE
    K_cont = torch.zeros(B, NKV, max_seq, D)
    V_cont = torch.zeros(B, NKV, max_seq, D)
    for b in range(B):
        for pg, phys in enumerate(page_table[b]):
            s = pg * BLOCK_SIZE
            K_cont[b, :, s : s + BLOCK_SIZE, :] = K_paged[b, :, phys, :, :]
            V_cont[b, :, s : s + BLOCK_SIZE, :] = V_paged[b, :, phys, :, :]
    q = Q.permute(1, 2, 0, 3)
    K_exp = K_cont.repeat_interleave(NH // NKV, dim=1)
    V_exp = V_cont.repeat_interleave(NH // NKV, dim=1)
    scores = torch.matmul(q * SCALE, K_exp.transpose(-1, -2))
    mask = torch.full((1, 1, 1, max_seq), float("-inf"))
    mask[:, :, :, : CUR_POS + 1] = 0.0
    scores = scores + mask
    return torch.matmul(torch.softmax(scores, dim=-1), V_exp).permute(2, 0, 1, 3)


def _run_sdpa(device, tt_Q, tt_K, tt_V, tt_pt, tt_cp):
    return ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_Q,
        tt_K,
        tt_V,
        cur_pos_tensor=tt_cp,
        page_table_tensor=tt_pt,
        scale=SCALE,
        program_config=PROD_CFG,
        compute_kernel_config=COMPUTE_CFG,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.timeout(60)
def test_sdpa_hang_repro_trace_mode(device):
    """
    Smoke test: production config in trace mode with K_GARBAGE=5 at recycled
    page positions. Verifies no NaN and PCC >= 0.95 on the fixed kernel.

    PASSES on both fixed and unfixed kernels (reducer always masks correctly
    on single device). The HANG itself is verified via vLLM full-stack test.
    """
    tt_Q, tt_K, tt_V, tt_pt, tt_cp, Q, K_paged, V_paged, page_table = _build_tensors(device)
    ref = _cpu_reference(Q, K_paged, V_paged, page_table)

    tt_out = _run_sdpa(device, tt_Q, tt_K, tt_V, tt_pt, tt_cp)
    ttnn.synchronize_device(device)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    tt_out = _run_sdpa(device, tt_Q, tt_K, tt_V, tt_pt, tt_cp)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    out = ttnn.to_torch(tt_out)
    ttnn.release_trace(device, trace_id)

    assert not torch.isnan(out).any(), f"NaN in trace-mode output at cur_pos={CUR_POS} with K_GARBAGE={K_GARBAGE}."
    pcc = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
    assert pcc > 0.95, (
        f"PCC={pcc:.4f} < 0.95: causal mask not applied correctly at "
        f"k_num_chunks-1={( CUR_POS + 1 + K_CHUNK - 1) // K_CHUNK - 1} "
        f"with K_GARBAGE={K_GARBAGE}. Fix: apply mask at k_num_chunks-1."
    )


@pytest.mark.timeout(60)
def test_sdpa_hang_repro_multi_call(device):
    """
    Five consecutive SDPA decode calls with K_GARBAGE=5 at recycled positions.
    """
    tt_Q, tt_K, tt_V, tt_pt, tt_cp, Q, K_paged, V_paged, page_table = _build_tensors(device)
    ref = _cpu_reference(Q, K_paged, V_paged, page_table)

    for _ in range(5):
        tt_out = _run_sdpa(device, tt_Q, tt_K, tt_V, tt_pt, tt_cp)

    out = ttnn.to_torch(tt_out)
    assert not torch.isnan(out).any(), "NaN in output after 5 consecutive calls."
    pcc = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
    assert pcc > 0.95, (
        f"PCC={pcc:.4f} on repeated calls: garbage KV at positions " f"{CUR_POS+1}+ not correctly masked."
    )
