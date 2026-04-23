# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests documenting the SDPA paged-decode partial-mask bug and its fix.

Bug:  apply_mask_at_last_chunk = do_reduce && is_causal
      Non-reducer workers (do_reduce=False) never apply the causal mask on
      any K chunk; the causal mask may be applied at the wrong chunk position.

Fix:  apply_mask_at_last_chunk = is_causal, with inline check k_chunk == k_num_chunks - 1
      Only the core that owns the actual last global K chunk (k_num_chunks-1) applies
      the causal mask.  Workers' fully-valid intermediate pages are never masked.

Issue: https://github.com/tenstorrent/tt-metal/issues/42917

--------------------------------------------------------------------------------
Test-design notes
-----------------
The production hang (GPT-OSS-120B on Galaxy) required multi-device, production-scale
conditions.  On a single Wormhole device the core-to-chunk assignment always places
the reducer (do_reduce=True, Core 0) on the partial last page, so the causal mask IS
applied correctly even with the buggy code.  As a result, neither output values nor
Metal-trace staleness differ between fixed and unfixed on a single device.

The tests below therefore serve as REGRESSION SMOKE TESTS that document correct
behaviour rather than as bug-reproduction tests.  The golden rule they enforce:
  • SDPA output must match the CPU reference (no PCC regression).
  • Metal trace must recompute correctly when input tensors are updated in-place
    between replays (the standard vLLM-TT trace pattern).
  • Consecutive calls must produce consistent results.

THEORY A — trace correctness via in-place Q update
  In Metal trace mode, the trace command stream references tensors by DRAM address.
  If Q is updated in-place (ttnn.copy) between trace replays, real computation
  produces a different output; a stale trace returns the capture-time output.
  The test asserts:
    PCC(replay_with_Q2, warmup_with_Q1)  < 0.5  (trace used new Q2, not stale Q1)
    PCC(replay_with_Q2, nontrace_with_Q2) > 0.95  (trace output matches non-trace)

THEORY B — garbage KV at uninitialized page positions
  Fill positions > cur_pos with large K/V values (1e3) to stress-test the causal
  mask.  The output must still match the CPU reference that masks those positions
  to -inf.  Verifies that the reducer's mask application is sufficient.

THEORY C — two consecutive non-trace calls produce consistent results
  Running the same inputs twice must produce identical output.  Detects any tile or
  semaphore state leaked between calls.
"""

import pytest
import torch
import ttnn


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

B, NKV, NH, D = 1, 1, 8, 128
BLOCK_SIZE = 64
# cur_pos = 2*BLOCK_SIZE + 3  =>  4 valid tokens in page 2, 60 garbage tokens
CUR_POS = 2 * BLOCK_SIZE + 3  # = 131
NUM_PAGES = 4
SCALE = D**-0.5

# Program config: 3 cores per head-batch (one per KV page)
#   Core 0 (do_reduce=True)  → reversed=2 → page 2 (PARTIAL LAST PAGE)
#   Core 1 (do_reduce=False) → reversed=1 → page 1 (fully valid)
#   Core 2 (do_reduce=False) → reversed=0 → page 0 (fully valid)
PROG_CFG = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
    q_chunk_size=NH,
    k_chunk_size=BLOCK_SIZE,
    max_cores_per_head_batch=3,
)


def _make_tensors(garbage_value: float = 0.0):
    """
    Build Q, K_paged, V_paged, page_table.
    Positions 0..CUR_POS hold valid random data.
    Positions CUR_POS+1..NUM_PAGES*BLOCK_SIZE-1 hold `garbage_value`.
    """
    torch.manual_seed(42)
    Q = torch.randn(1, B, NH, D)
    K_paged = torch.full((B, NKV, NUM_PAGES, BLOCK_SIZE, D), garbage_value)
    V_paged = torch.full((B, NKV, NUM_PAGES, BLOCK_SIZE, D), garbage_value)
    for tok in range(CUR_POS + 1):
        pg, off = tok // BLOCK_SIZE, tok % BLOCK_SIZE
        K_paged[0, 0, pg, off, :] = torch.randn(D)
        V_paged[0, 0, pg, off, :] = torch.randn(D)
    page_table = torch.arange(NUM_PAGES, dtype=torch.int32).unsqueeze(0)
    return Q, K_paged, V_paged, page_table


def _cpu_reference(Q, K_paged, V_paged, page_table):
    """CPU attention with explicit -inf mask for positions > CUR_POS."""
    max_seq = NUM_PAGES * BLOCK_SIZE
    K_cont = torch.zeros(B, NKV, max_seq, D)
    V_cont = torch.zeros(B, NKV, max_seq, D)
    for b in range(B):
        for pg, phys in enumerate(page_table[b]):
            s = pg * BLOCK_SIZE
            K_cont[b, :, s : s + BLOCK_SIZE, :] = K_paged[b, :, phys, :, :]
            V_cont[b, :, s : s + BLOCK_SIZE, :] = V_paged[b, :, phys, :, :]
    q = Q.permute(1, 2, 0, 3)  # [B, NH, 1, D]
    K_exp = K_cont.repeat_interleave(NH // NKV, dim=1)  # [B, NH, S, D]
    V_exp = V_cont.repeat_interleave(NH // NKV, dim=1)
    scores = torch.matmul(q * SCALE, K_exp.transpose(-1, -2))  # [B, NH, 1, S]
    mask = torch.full((1, 1, 1, max_seq), float("-inf"))
    mask[:, :, :, : CUR_POS + 1] = 0.0
    scores = scores + mask
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, V_exp)  # [B, NH, 1, D]
    return out.permute(2, 0, 1, 3)  # [1, B, NH, D]


def _to_tt(tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.as_tensor(tensor, device=device, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _run_paged_sdpa(device, Q, K_paged, V_paged, page_table):
    tt_Q = _to_tt(Q, device)
    K_tt = K_paged.permute(0, 2, 1, 3, 4).reshape(B * NUM_PAGES, NKV, BLOCK_SIZE, D)
    V_tt = V_paged.permute(0, 2, 1, 3, 4).reshape(B * NUM_PAGES, NKV, BLOCK_SIZE, D)
    tt_K = _to_tt(K_tt, device)
    tt_V = _to_tt(V_tt, device)
    tt_pt = _to_tt(page_table, device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_cp = ttnn.Tensor(torch.tensor([CUR_POS] * B, dtype=torch.int32), ttnn.int32).to(device)
    out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_Q,
        tt_K,
        tt_V,
        cur_pos_tensor=tt_cp,
        page_table_tensor=tt_pt,
        scale=SCALE,
        program_config=PROG_CFG,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.to_torch(out)


# ---------------------------------------------------------------------------
# THEORY A — Trace correctness: in-place Q update between replays
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
def test_theory_a_trace_replay_correctness(device):
    """
    Verify that Metal trace mode correctly uses updated input tensors.

    The standard vLLM-TT decode pattern is:
        capture_trace(inputs) → for each step: update_inputs(); execute_trace()

    This test mimics that pattern:
      1. Capture trace with Q1.
      2. Update Q in-place to Q2 = Q1 + 100 (large shift → very different output).
      3. Execute trace.
      4. Assert trace output ≈ non-trace output with Q2 (PCC > 0.95).
      5. Assert trace output ≠ warmup output with Q1 (PCC < 0.5).

    A stale or broken trace would return Q1's output regardless of the Q update,
    causing assertions 4 and 5 to fail.

    Note: on a single device with this config (3 cores per head, tiny sequence),
    fixed and unfixed kernels produce identical output.  This test therefore
    verifies trace machinery, not the mask bug itself.
    """
    Q, K_paged, V_paged, page_table = _make_tensors(garbage_value=0.0)
    Q1 = Q.clone()
    Q2 = Q1 + 100.0  # large shift → very different attention scores

    K_tt = K_paged.permute(0, 2, 1, 3, 4).reshape(B * NUM_PAGES, NKV, BLOCK_SIZE, D)
    V_tt = V_paged.permute(0, 2, 1, 3, 4).reshape(B * NUM_PAGES, NKV, BLOCK_SIZE, D)

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

    # Warmup + capture
    tt_out = _run()
    ttnn.synchronize_device(device)
    out_warmup_q1 = ttnn.to_torch(tt_out)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    tt_out = _run()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    # First replay with Q1 unchanged — should match warmup
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    out_trace_q1 = ttnn.to_torch(tt_out)
    pcc_sanity = torch.nn.functional.cosine_similarity(
        out_trace_q1.float().flatten(), out_warmup_q1.float().flatten(), dim=0
    ).item()
    assert pcc_sanity > 0.99, f"Sanity: first replay (Q1 unchanged) diverged from warmup (PCC={pcc_sanity:.4f})"

    # Update Q in-place to Q2 (standard vLLM-TT update pattern)
    ttnn.copy(_to_tt(Q2, device), tt_Q)
    ttnn.synchronize_device(device)

    # Replay with Q2
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    out_trace_q2 = ttnn.to_torch(tt_out)

    # Non-trace reference with Q2 (tt_Q now holds Q2)
    out_ref_q2 = ttnn.to_torch(_run())
    ttnn.synchronize_device(device)

    ttnn.release_trace(device, trace_id)

    # Trace output with Q2 must DIFFER from Q1 warmup
    pcc_q1_vs_q2 = torch.nn.functional.cosine_similarity(
        out_trace_q2.float().flatten(), out_warmup_q1.float().flatten(), dim=0
    ).item()
    assert pcc_q1_vs_q2 < 0.5, (
        f"Theory A: trace output did not change after Q update "
        f"(PCC_vs_Q1={pcc_q1_vs_q2:.4f} >= 0.5). "
        f"The trace may be returning stale output from capture."
    )

    # Trace output with Q2 must MATCH non-trace reference with Q2
    pcc_q2_match = torch.nn.functional.cosine_similarity(
        out_trace_q2.float().flatten(), out_ref_q2.float().flatten(), dim=0
    ).item()
    assert pcc_q2_match > 0.95, (
        f"Theory A: trace output with Q2 does not match non-trace reference "
        f"(PCC={pcc_q2_match:.4f}). Trace is producing incorrect output."
    )


# ---------------------------------------------------------------------------
# THEORY B — Garbage KV data at uninitialized page positions
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_theory_b_garbage_kv_overflow(device):
    """
    Fill positions > CUR_POS with large K/V values (1000.0) to simulate
    uninitialized garbage from a recycled KV-cache page.

    The CPU reference masks those positions to -inf; the TT kernel should produce
    output matching that reference.  This verifies the causal mask is applied
    correctly on the reducer core that owns the partial last page.

    On a single device with this config, the reducer (Core 0) always owns the
    partial page and applies the mask correctly regardless of the bug fix status.
    The test therefore documents that the REDUCER'S masking is correct.
    """
    GARBAGE = 1000.0
    Q, K_paged, V_paged, page_table = _make_tensors(garbage_value=GARBAGE)
    ref = _cpu_reference(Q, K_paged, V_paged, page_table)
    out = _run_paged_sdpa(device, Q, K_paged, V_paged, page_table)

    assert not torch.isnan(out).any(), (
        f"Theory B: NaN in output with garbage KV (GARBAGE={GARBAGE}). "
        f"Large Q·K values overflow BFP8 exp() and produce NaN."
    )
    pcc = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
    assert pcc > 0.95, (
        f"Theory B: PCC={pcc:.4f} — output does not match CPU reference with "
        f"garbage KV at positions {CUR_POS+1}+. "
        f"The causal mask is not masking those positions correctly."
    )


# ---------------------------------------------------------------------------
# THEORY C — Two consecutive calls produce consistent results
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_theory_c_consecutive_call_consistency(device):
    """
    Run the same SDPA decode twice in succession and verify both outputs are
    identical and match the CPU reference.

    Detects any tile, semaphore, or TRISC state that leaks between calls,
    which would produce diverging outputs on the second call.
    """
    Q, K_paged, V_paged, page_table = _make_tensors(garbage_value=0.0)
    ref = _cpu_reference(Q, K_paged, V_paged, page_table)

    out1 = _run_paged_sdpa(device, Q, K_paged, V_paged, page_table)
    out2 = _run_paged_sdpa(device, Q, K_paged, V_paged, page_table)

    assert not torch.isnan(out1).any(), "Theory C: NaN in first call"
    assert not torch.isnan(out2).any(), "Theory C: NaN in second call"

    pcc_self = torch.nn.functional.cosine_similarity(out1.float().flatten(), out2.float().flatten(), dim=0).item()
    assert pcc_self > 0.999, (
        f"Theory C: two consecutive calls diverged (PCC={pcc_self:.6f}). "
        f"Tile or semaphore state leaked between calls."
    )

    pcc_ref = torch.nn.functional.cosine_similarity(out1.float().flatten(), ref.float().flatten(), dim=0).item()
    assert pcc_ref > 0.95, f"Theory C: output differs from CPU reference (PCC={pcc_ref:.4f})."
