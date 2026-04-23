# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Deterministic trace-staleness detector for the SDPA paged-decode partial-mask bug.

Bug (fixed in commit d0b7927279e + d2aaf1faeb5):
    apply_mask_at_last_chunk = do_reduce && is_causal
    Worker cores (do_reduce=False) skip the causal mask on their last K chunk.

Detection mechanism (zero timing dependency):
    In Metal trace mode, the trace captures the command dispatch stream.
    A stale trace (no-op due to a CB stall) returns the DRAM content from
    the previous run rather than computing fresh output.

    We detect a stale trace by:
      1. Capture the trace with Q1.
      2. Update the Q device tensor IN-PLACE to Q2 (a very different value).
      3. Execute the trace.
      4. If trace is REAL:  output = SDPA(Q2) ≠ SDPA(Q1)  → PCC(out_trace, ref_Q1) < 0.9
      5. If trace is STALE: output = stale = SDPA(Q1)     → PCC(out_trace, ref_Q1) ≈ 1.0

    We assert PCC(out_trace_after_q2_update, warmup_out_q1) < 0.5 to confirm
    the trace actually recomputed with the new Q2 value.

Issue: https://github.com/tenstorrent/tt-metal/issues/42917
"""

import pytest
import torch
import ttnn


B, NKV, NH, D = 1, 1, 8, 128
BLOCK_SIZE = 64
CUR_POS = 2 * BLOCK_SIZE + 3  # = 131 (4 valid tokens in last page, 60 zeros)
NUM_PAGES = 4
SCALE = D**-0.5

PROG_CFG = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
    q_chunk_size=NH,
    k_chunk_size=BLOCK_SIZE,
    max_cores_per_head_batch=3,
)


def _to_tt(tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.as_tensor(
        tensor,
        device=device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _build_kv_page_table(Q_val: float = 0.0):
    """Build K/V paged tensors with zero-padding beyond CUR_POS."""
    torch.manual_seed(0)
    K_paged = torch.zeros(B, NKV, NUM_PAGES, BLOCK_SIZE, D)
    V_paged = torch.zeros(B, NKV, NUM_PAGES, BLOCK_SIZE, D)
    for tok in range(CUR_POS + 1):
        pg, off = tok // BLOCK_SIZE, tok % BLOCK_SIZE
        K_paged[0, 0, pg, off, :] = torch.randn(D)
        V_paged[0, 0, pg, off, :] = torch.randn(D)
    page_table = torch.arange(NUM_PAGES, dtype=torch.int32).unsqueeze(0)
    return K_paged, V_paged, page_table


@pytest.mark.timeout(120)
def test_trace_stale_detection(device):
    """
    Deterministic test: the trace must recompute when Q is updated in-place.

    Procedure:
        1. Warmup with Q1 (random), record out_warmup_q1.
        2. Capture trace with Q1.
        3. Execute trace with Q1 → out_trace_q1 (should ≈ out_warmup_q1).
        4. Update Q tensor in-place to Q2 = Q1 + 100 (very large shift).
        5. Execute trace again → out_trace_q2.
        6. Run non-trace with Q2 → out_ref_q2.
        7. Assert PCC(out_trace_q2, out_warmup_q1) < 0.5  [Q2 is very different from Q1]
        8. Assert PCC(out_trace_q2, out_ref_q2) > 0.95   [trace matches non-trace with Q2]

    If trace is REAL:
        Step 7: out_trace_q2 = SDPA(Q2) ≠ SDPA(Q1) → PCC < 0.5 ✓
        Step 8: out_trace_q2 ≈ out_ref_q2 → PCC > 0.95 ✓

    If trace is a NO-OP (stale):
        Step 7: out_trace_q2 = stale = out_warmup_q1 → PCC ≈ 1.0 → FAIL ✗
        Step 8: out_trace_q2 ≠ out_ref_q2 → PCC < 0.95 → FAIL ✗
    """
    K_paged, V_paged, page_table = _build_kv_page_table()
    K_tt = K_paged.permute(0, 2, 1, 3, 4).reshape(B * NUM_PAGES, NKV, BLOCK_SIZE, D)
    V_tt = V_paged.permute(0, 2, 1, 3, 4).reshape(B * NUM_PAGES, NKV, BLOCK_SIZE, D)

    torch.manual_seed(42)
    Q1 = torch.randn(1, B, NH, D)
    Q2 = Q1 + 100.0  # large shift → very different attention scores

    # Allocate all device tensors (allocated BEFORE trace capture)
    tt_Q = _to_tt(Q1, device)
    tt_K = _to_tt(K_tt, device)
    tt_V = _to_tt(V_tt, device)
    tt_pt = _to_tt(page_table, device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_cp = ttnn.from_torch(
        torch.tensor([CUR_POS] * B, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def _run():
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tt_Q,
            tt_K,
            tt_V,
            cur_pos_tensor=tt_cp,
            page_table_tensor=tt_pt,
            scale=SCALE,
            program_config=PROG_CFG,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Warmup: compile kernels, first run with Q1
    tt_out = _run()
    ttnn.synchronize_device(device)
    out_warmup_q1 = ttnn.to_torch(tt_out)

    # Capture trace (with Q1 in tt_Q)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    tt_out = _run()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    # Sanity: first replay with Q1 unchanged — should match warmup
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    out_trace_q1 = ttnn.to_torch(tt_out)
    pcc_sanity = torch.nn.functional.cosine_similarity(
        out_trace_q1.float().flatten(), out_warmup_q1.float().flatten(), dim=0
    ).item()
    assert pcc_sanity > 0.99, (
        f"Sanity check failed: first trace replay (Q1 unchanged) diverged from "
        f"warmup (PCC={pcc_sanity:.4f}). Trace is not reproducing the captured run."
    )

    # --- KEY STEP: update Q in-place to Q2 ---
    tt_Q2 = _to_tt(Q2, device)
    ttnn.copy(tt_Q2, tt_Q)  # write Q2 values into tt_Q's DRAM buffer
    ttnn.synchronize_device(device)

    # Replay with updated Q2
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    out_trace_q2 = ttnn.to_torch(tt_out)

    # Reference: non-trace run with current tt_Q (which now holds Q2)
    out_ref_q2 = ttnn.to_torch(_run())
    ttnn.synchronize_device(device)

    ttnn.release_trace(device, trace_id)

    # --- Assertions ---
    # 1. Trace output with Q2 must DIFFER from warmup output with Q1
    #    (Q2 = Q1 + 100 → completely different attention weights → different output)
    pcc_q1_vs_q2 = torch.nn.functional.cosine_similarity(
        out_trace_q2.float().flatten(), out_warmup_q1.float().flatten(), dim=0
    ).item()
    assert pcc_q1_vs_q2 < 0.5, (
        f"Stale trace detected: after updating Q from Q1 to Q2 (Q2=Q1+100), "
        f"trace output still matches Q1 warmup output (PCC={pcc_q1_vs_q2:.4f} >= 0.5). "
        f"The trace is a NO-OP — it returned stale DRAM data from the trace-capture run "
        f"instead of recomputing with the new Q2. "
        f"Root cause: cb_mask_in is not drained for non-reducer workers, so the CB is "
        f"full at trace-capture end; on replay the writer's cb_reserve_back stalls. "
        f"Fix: apply_mask_at_last_chunk = is_causal (ensures all cores drain cb_mask_in)."
    )

    # 2. Trace output with Q2 must MATCH non-trace run with Q2 (correctness check)
    pcc_q2_trace_vs_ref = torch.nn.functional.cosine_similarity(
        out_trace_q2.float().flatten(), out_ref_q2.float().flatten(), dim=0
    ).item()
    assert pcc_q2_trace_vs_ref > 0.95, (
        f"Trace output with Q2 does not match non-trace reference with Q2 "
        f"(PCC={pcc_q2_trace_vs_ref:.4f}). "
        f"Trace may be computing but producing incorrect output."
    )
