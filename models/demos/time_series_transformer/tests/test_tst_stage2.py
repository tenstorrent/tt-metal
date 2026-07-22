# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Change 4 (KV cache fused op for V, at BS==1) — IMPLEMENTED, not blocked.

CORRECTION (this revision): the previous version of this test xfailed with
the claim "no measured evidence the TILE<->ROW_MAJOR conversion tax nets
positive". That claim was incorrect and was written without consulting this
repo's own diagnostics (tests/diagnostics/probe_v_cache_update_cache*.py),
which had already shown, on real hardware:
  - Exact correctness match vs. slice_write ground truth (multistep probe,
    max abs diff 0.0 over 8 steps).
  - update_cache faster per-call than slice_write+to_layout (0.291ms vs
    0.375ms, isolated probe).

This model's V cache at BS==1 is allocated TILE_LAYOUT specifically to avoid
the conversion tax (see allocate_kv_cache) -- there is no TILE<->ROW_MAJOR
conversion on the update_cache path at BS==1 to begin with. K is unaffected
(stays on slice_write unconditionally, all B -- see allocate_kv_cache
docstring for why K's transposed layout is incompatible with update_cache).

MEASURED END-TO-END EFFECT (2026-07-09, wormhole_b0, BS=1, 24-step decode,
test_single_sequence_latency, use_2cq=False), before vs after re-enabling
the update_cache path for V:
  write_prep:      ~47ms/step -> ~36ms/step
  slice_write_kv:  ~27ms/step -> ~20ms/step
  total latency:   124.3ms (median) -> 111.8ms (median)

This is a real, correctness-verified ~10% latency reduction. It does NOT
close the gap to the Stage 1 50ms target on its own -- write_prep,
trace_exec, and the readback/sample/cpu_prep tail remain comparable in size
to one another, so no single further op-level change is expected to reach
50ms; that requires Stage 2/3 restructuring (fused ops, sharding, or a
KV-cache scheme that avoids per-step host writes entirely).
"""

import torch
from tt.attention import allocate_kv_cache

import ttnn


def test_update_cache_v_matches_slice_write_ground_truth():
    """Correctness: update_cache (TILE, BS=1) vs slice_write (ROW_MAJOR) for V, multistep."""
    device = ttnn.open_device(device_id=0)
    try:
        BS, H, T_max, D = 1, 2, 24, 32
        N_STEPS = 8

        v_cache_rm = ttnn.from_torch(
            torch.zeros(BS, H, T_max, D),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        torch.manual_seed(0)
        steps_input = [torch.randn(BS, H, 1, D) for _ in range(N_STEPS)]

        for step, v_step in enumerate(steps_input):
            v_new = ttnn.from_torch(v_step, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            ttnn.experimental.slice_write(v_new, v_cache_rm, [0, 0, step, 0], [BS, H, step + 1, D], [1, 1, 1, 1])
        ref = ttnn.to_torch(v_cache_rm).float()

        v_cache_tile = ttnn.from_torch(
            torch.zeros(BS, H, T_max, D),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        for step, v_step in enumerate(steps_input):
            v_new_tile = ttnn.from_torch(v_step, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            ttnn.update_cache(v_cache_tile, v_new_tile, update_idx=step, batch_offset=0)
        cand = ttnn.to_torch(v_cache_tile).float()

        diff = (ref - cand).abs().max().item()
        assert diff < 1e-3, f"update_cache diverges from slice_write ground truth: max abs diff {diff}"
    finally:
        ttnn.close_device(device)


def test_update_cache_allocated_correctly_at_bs1():
    """allocate_kv_cache must give V TILE_LAYOUT at BS==1 (the precondition Change 4 relies on)."""
    device = ttnn.open_device(device_id=0)
    try:
        k_cache, v_cache = allocate_kv_cache(device, B=1, T_max=24)
        assert v_cache.layout == ttnn.TILE_LAYOUT, (
            "V cache must be TILE_LAYOUT at BS==1 for the update_cache path in "
            "_extract_and_write_kv to apply without a ROW_MAJOR<->TILE conversion."
        )
    finally:
        ttnn.close_device(device)
