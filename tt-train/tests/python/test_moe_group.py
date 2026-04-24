# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MoE group op.

The op takes the outputs of ttnn.all_to_all_dispatch:
  dispatched: [D, B, S, H] bf16 row-major (dense per-device token view)
  metadata:   [D, B, S, K] int32        (top-K expert ids per token)
and produces:
  grouped:  [1, 1, T_cap, H] tiled bf16 — active rows packed by expert
  counts:   [E_local]                   — real active count per expert
  offsets:  [E_local + 1]               — row-granular prefix-sum
                                          (each expert rounded up to 32)
  plan:     [T_cap]                     — flat source row index per output row,
                                          SENTINEL (0xFFFFFFFF) for pad slots

T_cap = min(E_local, K) * D * B * S  +  E_local * 32.

This file defines:
  * `moe_group_torch_reference(...)`  — the contract, in torch
  * `test_moe_group_reference_*`      — self-tests for the reference
  * `test_moe_group_*`                — will exercise the real device op once Stage 2+ lands
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest
import torch


SENTINEL = np.uint32(0xFFFFFFFF)


def moe_group_t_cap(e_local: int, k: int, d: int, b: int, s: int, num_total_cores: int = 64) -> int:
    # Mirrors moe_group_device_operation.cpp: per-core padding adds up to 3
    # SENTINEL slots per (core, expert), plus the original 32-row tail per expert.
    return min(e_local, k) * d * b * s + e_local * (32 + 3 * num_total_cores)


def _round_up_32(x: int) -> int:
    return ((x + 31) // 32) * 32


def moe_group_torch_reference(
    dispatched: torch.Tensor,  # [D, B, S, H] bf16 / float
    metadata: torch.Tensor,  # [D, B, S, K] int32/int64
    local_expert_ids: torch.Tensor,  # [E_local] int
    *,
    k: int,
    num_total_cores: int = 64,  # parallel scan splits rows across this many cores
    t_cap: int = 0,  # if nonzero, use this as allocated T_cap (match device)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation of the group op.

    Semantics:
      - For each local expert e with global id g = local_expert_ids[e], gather
        every source row (d, b, s) of `dispatched` such that g appears in
        metadata[d, b, s, :K]. Scan order is row-major over (d, b, s).
      - Pack these active rows into `grouped[0, 0, offsets[e]:offsets[e]+counts[e], :]`.
      - Pad each expert's slice up to a multiple of 32 rows with zeros.
      - `plan[i]` = flat source row index `d*B*S + b*S + s` for row i of
        grouped. Padding rows (and all rows past offsets[E_local]) are
        SENTINEL (0xFFFFFFFF).

    Returns numpy-friendly torch tensors. `grouped` is returned as
    [1, 1, T_cap, H] with padding zeros beyond offsets[E_local].
    """
    assert dispatched.dim() == 4, f"dispatched must be [D,B,S,H], got {dispatched.shape}"
    assert metadata.dim() == 4, f"metadata must be [D,B,S,K], got {metadata.shape}"
    D, B, S, H = dispatched.shape
    assert metadata.shape[:3] == (D, B, S)
    assert metadata.shape[3] == k, f"metadata last dim {metadata.shape[3]} != k {k}"
    E_local = int(local_expert_ids.numel())
    T_cap = t_cap if t_cap > 0 else moe_group_t_cap(E_local, k, D, B, S, num_total_cores=num_total_cores)

    # Flatten (d, b, s) to a single axis for easier indexing.
    total_rows = D * B * S
    flat = dispatched.reshape(total_rows, H)
    md = metadata.reshape(total_rows, k)

    # Which rows are active for which local expert?
    # active_mask[e, i] = True if local_expert_ids[e] in md[i]
    md_np = md.numpy()  # [total_rows, K]
    leids = local_expert_ids.numpy().astype(md_np.dtype)  # [E_local]

    # Compute per-row, per-expert hit mask.
    hits = np.zeros((E_local, total_rows), dtype=np.bool_)
    for e_idx in range(E_local):
        hits[e_idx] = (md_np == leids[e_idx]).any(axis=1)
    counts = hits.sum(axis=1).astype(np.uint32)

    # Parallel scan layout: split rows into num_total_cores slices. Per expert e,
    # core c writes its round_up_4(local_counts[c][e]) entries starting at
    # per_core_start[c][e]. The round_up_4 padding (0..3 SENTINELs per core per
    # expert) keeps each core's cursor 16B-aligned (required for DRAM writes).
    # offsets[e+1] is rounded up to 32 for tile alignment in grouped.
    slice_size = (total_rows + num_total_cores - 1) // num_total_cores

    def _round_up_4(x):
        return (x + 3) & ~3

    # Per-core local counts per expert: [num_total_cores, E_local]
    local_counts = np.zeros((num_total_cores, E_local), dtype=np.uint32)
    for c in range(num_total_cores):
        s_start = c * slice_size
        s_end = min(s_start + slice_size, total_rows)
        if s_start >= total_rows:
            continue
        for e_idx in range(E_local):
            local_counts[c, e_idx] = hits[e_idx, s_start:s_end].sum()

    # Offsets with per-core padding accounted for.
    offsets = np.zeros(E_local + 1, dtype=np.uint32)
    for e_idx in range(E_local):
        running = int(offsets[e_idx])
        for c in range(num_total_cores):
            running += int(_round_up_4(int(local_counts[c, e_idx])))
        offsets[e_idx + 1] = _round_up_32(running)
    T_used = int(offsets[-1])
    assert T_used <= T_cap, f"T_used {T_used} exceeds T_cap {T_cap}; " f"E_local={E_local} K={k} D={D} B={B} S={S}"

    # Build plan: each core contributes round_up_4(local_counts[c][e]) entries
    # (real + up to 3 SENTINEL pad) starting at per_core_start[c][e].
    plan = np.full(T_cap, SENTINEL, dtype=np.uint32)
    for e_idx in range(E_local):
        running = int(offsets[e_idx])
        for c in range(num_total_cores):
            s_start = c * slice_size
            s_end = min(s_start + slice_size, total_rows)
            if s_start >= total_rows:
                continue
            core_hits = np.nonzero(hits[e_idx, s_start:s_end])[0].astype(np.uint32) + s_start
            n = len(core_hits)
            if n > 0:
                plan[running : running + n] = core_hits
            running += int(_round_up_4(n))

    # Materialize grouped [1, 1, T_cap, H] using plan.
    grouped = torch.zeros(1, 1, T_cap, H, dtype=dispatched.dtype)
    # Convert valid plan entries into a gather.
    valid_mask = plan != SENTINEL
    valid_idx = np.nonzero(valid_mask)[0]
    src_idx = plan[valid_idx].astype(np.int64)
    grouped[0, 0, valid_idx, :] = flat[src_idx]

    return (
        grouped,
        torch.from_numpy(counts.astype(np.int64)),
        torch.from_numpy(offsets.astype(np.int64)),
        torch.from_numpy(plan.astype(np.int64)),
    )


# ---------------------------------------------------------------------------
# Reference self-tests (no device required)
# ---------------------------------------------------------------------------


class TestMoeGroupReference:
    """Self-tests for the torch reference — sanity checks the contract."""

    @staticmethod
    def _make_dispatched(D: int, B: int, S: int, H: int, seed: int = 0) -> torch.Tensor:
        """Fill each (d, b, s) row with a distinct recognizable value per column.

        row (d, b, s) has value (d*B*S + b*S + s) broadcast across all H columns.
        Lets us read back where each row came from just by inspecting a grouped row.
        """
        torch.manual_seed(seed)
        idx = torch.arange(D * B * S, dtype=torch.float32).reshape(D, B, S, 1)
        return idx.expand(D, B, S, H).clone()

    @staticmethod
    def _make_metadata(D: int, B: int, S: int, K: int, E: int, seed: int = 0) -> torch.Tensor:
        torch.manual_seed(seed + 1)
        # For each token pick K distinct experts in [0, E).
        out = torch.zeros(D, B, S, K, dtype=torch.int32)
        for d in range(D):
            for b in range(B):
                for s in range(S):
                    out[d, b, s] = torch.randperm(E)[:K].to(torch.int32)
        return out

    def test_basic_shape_and_pack(self):
        D, B, S, H = 2, 1, 32, 16
        E, K = 4, 2
        local_expert_ids = torch.tensor([0, 1], dtype=torch.int32)
        dispatched = self._make_dispatched(D, B, S, H, seed=0)
        metadata = self._make_metadata(D, B, S, K, E, seed=0)

        grouped, counts, offsets, plan = moe_group_torch_reference(dispatched, metadata, local_expert_ids, k=K)

        E_local = local_expert_ids.numel()
        T_cap = moe_group_t_cap(E_local, K, D, B, S)
        assert grouped.shape == (1, 1, T_cap, H)
        assert counts.shape == (E_local,)
        assert offsets.shape == (E_local + 1,)
        assert plan.shape == (T_cap,)

        # offsets must be row-granular prefix-sum of round_up(counts, 32)
        for e in range(E_local):
            expected = int(offsets[e]) + _round_up_32(int(counts[e]))
            assert int(offsets[e + 1]) == expected, f"offsets mismatch at e={e}"

        # Each filled row of grouped must match a source row in dispatched.
        flat = dispatched.reshape(D * B * S, H)
        for e in range(E_local):
            start = int(offsets[e])
            n = int(counts[e])
            for i in range(n):
                row_i = grouped[0, 0, start + i]
                src = int(plan[start + i])
                assert src != SENTINEL.item()
                assert torch.allclose(row_i, flat[src])

    def test_pad_rows_are_zero_and_sentinel(self):
        D, B, S, H = 2, 1, 32, 16
        E, K = 4, 2
        local_expert_ids = torch.tensor([0, 1], dtype=torch.int32)
        dispatched = self._make_dispatched(D, B, S, H, seed=0)
        metadata = self._make_metadata(D, B, S, K, E, seed=0)

        grouped, counts, offsets, plan = moe_group_torch_reference(dispatched, metadata, local_expert_ids, k=K)

        E_local = local_expert_ids.numel()
        # Per-expert pad slots + trailing T_cap slots must be SENTINEL.
        for e in range(E_local):
            start = int(offsets[e]) + int(counts[e])
            end = int(offsets[e + 1])
            for i in range(start, end):
                assert int(plan[i]) == SENTINEL.item(), f"pad slot {i} not SENTINEL"
                assert torch.all(grouped[0, 0, i] == 0), f"pad row {i} not zero"
        # Tail past offsets[E_local]
        for i in range(int(offsets[-1]), plan.numel()):
            assert int(plan[i]) == SENTINEL.item()

    def test_expert_with_zero_active(self):
        # Construct metadata where one local expert gets no tokens.
        D, B, S, H = 1, 1, 32, 8
        E, K = 8, 2
        local_expert_ids = torch.tensor([0, 5], dtype=torch.int32)
        dispatched = self._make_dispatched(D, B, S, H, seed=2)
        # Build metadata that never picks expert 5.
        md = torch.zeros(D, B, S, K, dtype=torch.int32)
        choices = torch.tensor([0, 1, 2, 3, 4, 6, 7])  # no 5
        for s in range(S):
            perm = choices[torch.randperm(choices.numel())][:K]
            md[0, 0, s] = perm.to(torch.int32)

        grouped, counts, offsets, plan = moe_group_torch_reference(dispatched, md, local_expert_ids, k=K)
        assert int(counts[1]) == 0
        # expert 1's slice is zero-width
        assert int(offsets[2]) == int(offsets[1])

    def test_counts_match_metadata_occurrences(self):
        D, B, S, H = 3, 2, 32, 4
        E, K = 6, 3
        local_expert_ids = torch.tensor([1, 4], dtype=torch.int32)
        dispatched = self._make_dispatched(D, B, S, H, seed=7)
        metadata = self._make_metadata(D, B, S, K, E, seed=7)

        _, counts, _, _ = moe_group_torch_reference(dispatched, metadata, local_expert_ids, k=K)

        md = metadata.reshape(-1, K)
        for e_idx, g in enumerate(local_expert_ids.tolist()):
            expected = ((md == g).any(dim=1)).sum().item()
            assert int(counts[e_idx]) == expected, f"counts[{e_idx}] expected {expected} got {int(counts[e_idx])}"

    def test_t_cap_is_upper_bound(self):
        # Stress: force every token to pick every local expert (K >= E_local).
        D, B, S = 2, 1, 32
        E, K = 4, 4
        H = 8
        local_expert_ids = torch.tensor([0, 1], dtype=torch.int32)
        dispatched = self._make_dispatched(D, B, S, H, seed=9)
        # Every token's K slots contain both 0 and 1.
        md = torch.zeros(D, B, S, K, dtype=torch.int32)
        for d in range(D):
            for b in range(B):
                for s in range(S):
                    md[d, b, s, 0] = 0
                    md[d, b, s, 1] = 1
                    md[d, b, s, 2] = 2
                    md[d, b, s, 3] = 3

        grouped, counts, offsets, plan = moe_group_torch_reference(dispatched, md, local_expert_ids, k=K)
        # Every token active for both local experts.
        assert int(counts[0]) == D * B * S
        assert int(counts[1]) == D * B * S
        T_cap = moe_group_t_cap(2, K, D, B, S)
        assert int(offsets[-1]) <= T_cap


# ---------------------------------------------------------------------------
# Device tests — require ttml device. Will exercise the real op (Stage 2+).
# ---------------------------------------------------------------------------


# Gated import: device-less environments (reference self-tests) still run.
try:
    import ttnn  # noqa: F401
    import ttml  # noqa: F401

    _TTML_AVAILABLE = True
except Exception:
    _TTML_AVAILABLE = False


def _to_device_tensor(t: torch.Tensor, layout, dtype) -> "ttnn.Tensor":
    """Push a torch tensor to device as a raw ttnn Tensor (not autograd)."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    return ttnn.from_torch(
        t.float() if t.dtype == torch.bfloat16 else t,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _from_device_tensor(t: "ttnn.Tensor") -> np.ndarray:
    return ttnn.to_torch(t).float().numpy()


@pytest.mark.skipif(not _TTML_AVAILABLE, reason="ttml / ttnn not importable")
@pytest.mark.requires_device
class TestMoeGroupDevice:
    """Device tests: call the real op and compare to torch reference.

    Stage 2: op runs (empty kernel), outputs are zero-filled.
    We only check shapes and that it doesn't crash. Correctness
    tests will pass in Stage 3+ once kernels are implemented.
    """

    @staticmethod
    def _run_op(dispatched: torch.Tensor, metadata: torch.Tensor, local_expert_ids: torch.Tensor, k: int):
        E_local = local_expert_ids.numel()

        d_tt = _to_device_tensor(dispatched.float(), ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16)
        md_tt = _to_device_tensor(metadata.to(torch.int32), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)
        le_tt = _to_device_tensor(local_expert_ids.to(torch.int32), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)

        device = ttml.autograd.AutoContext.get_instance().get_device()
        ttnn.synchronize_device(device)
        grouped, counts, offsets, plan = ttml.ops.metal_ops.moe_group(d_tt, md_tt, le_tt, int(E_local), int(k))
        ttnn.synchronize_device(device)
        return grouped, counts, offsets, plan

    def test_output_shapes(self):
        D, B, S, H = 2, 1, 32, 64
        E, K = 4, 2
        E_local = 2
        local_expert_ids = torch.tensor([0, 1], dtype=torch.int32)
        dispatched = TestMoeGroupReference._make_dispatched(D, B, S, H)
        metadata = TestMoeGroupReference._make_metadata(D, B, S, K, E)

        T_cap = moe_group_t_cap(E_local, K, D, B, S, num_total_cores=self._device_num_total_cores())
        grouped, counts, offsets, plan = self._run_op(dispatched, metadata, local_expert_ids, K)

        assert list(grouped.shape) == [1, 1, T_cap, H], f"grouped shape {grouped.shape}"
        assert list(counts.shape) == [1, 1, 1, E_local], f"counts shape {counts.shape}"
        assert list(offsets.shape) == [1, 1, 1, E_local + 1], f"offsets shape {offsets.shape}"
        assert list(plan.shape) == [1, 1, 1, T_cap], f"plan shape {plan.shape}"

    @staticmethod
    def _device_grid_size() -> int:
        device = ttml.autograd.AutoContext.get_instance().get_device()
        grid = device.compute_with_storage_grid_size()
        return int(grid.x) * int(grid.y)

    @staticmethod
    def _device_num_total_cores(e_local: int = 0, k: int = 0, d: int = 0, b: int = 0, s: int = 0) -> int:
        # Mirror the kernel's split_work_to_cores(grid, total_tiles) logic:
        # num_workers = min(total_tiles, grid_size). t_cap is sized with the
        # full grid, so we compute total_tiles from the same t_cap formula.
        grid_size = TestMoeGroupDevice._device_grid_size()
        if e_local == 0:
            return grid_size
        t_cap = moe_group_t_cap(e_local, k, d, b, s, num_total_cores=grid_size)
        total_tiles = t_cap // 32
        return min(total_tiles, grid_size)

    @staticmethod
    def _check_correctness(
        dispatched: torch.Tensor,
        metadata: torch.Tensor,
        local_expert_ids: torch.Tensor,
        k: int,
        label: str = "",
    ):
        """Run op + reference, compare every active row content, counts, offsets, plan."""
        D, B, S, _H = dispatched.shape
        E_local = int(local_expert_ids.numel())
        num_total_cores = TestMoeGroupDevice._device_num_total_cores(E_local, k, D, B, S)
        # T_cap is sized with the full grid (matches host moe_group_device_operation.cpp).
        grid_size = TestMoeGroupDevice._device_grid_size()
        device_t_cap = moe_group_t_cap(E_local, k, D, B, S, num_total_cores=grid_size)
        ref_grouped, ref_counts, ref_offsets, ref_plan = moe_group_torch_reference(
            dispatched,
            metadata,
            local_expert_ids,
            k=k,
            num_total_cores=num_total_cores,
            t_cap=device_t_cap,
        )
        print(
            f"\n[{label}] dispatched={dispatched.shape} metadata={metadata.shape} k={k} E_local={local_expert_ids.numel()}"
        )
        print(f"[{label}] ref_counts={ref_counts.tolist()} ref_offsets={ref_offsets.tolist()}")
        print(f"[{label}] calling device op...")
        grouped_tt, counts_tt, offsets_tt, plan_tt = TestMoeGroupDevice._run_op(
            dispatched, metadata, local_expert_ids, k
        )
        print(f"[{label}] device op returned", flush=True)

        # -- counts --
        print(f"[{label}] reading counts...", flush=True)
        counts_np = ttnn.to_torch(counts_tt).flatten().numpy().astype(np.int64)
        print(f"[{label}] counts={counts_np.tolist()}", flush=True)
        ref_counts_np = ref_counts.numpy()
        np.testing.assert_array_equal(
            counts_np, ref_counts_np, err_msg=f"{label}: counts mismatch: got {counts_np} expected {ref_counts_np}"
        )

        # -- offsets --
        print(f"[{label}] reading offsets...", flush=True)
        offsets_np = ttnn.to_torch(offsets_tt).flatten().numpy().astype(np.int64)
        print(f"[{label}] offsets={offsets_np.tolist()}", flush=True)
        ref_offsets_np = ref_offsets.numpy()
        np.testing.assert_array_equal(
            offsets_np, ref_offsets_np, err_msg=f"{label}: offsets mismatch: got {offsets_np} expected {ref_offsets_np}"
        )

        # -- plan --
        print(f"[{label}] reading plan...", flush=True)
        plan_np = ttnn.to_torch(plan_tt).flatten().numpy().astype(np.int64)
        print(f"[{label}] plan (first 16): {plan_np[:16].tolist()}", flush=True)
        ref_plan_np = ref_plan.numpy()
        np.testing.assert_array_equal(plan_np, ref_plan_np, err_msg=f"{label}: plan mismatch")

        # -- grouped: check every active row for every expert --
        print(f"[{label}] untiling grouped...", flush=True)
        grouped_rm = ttnn.to_layout(grouped_tt, ttnn.ROW_MAJOR_LAYOUT)
        print(f"[{label}] reading grouped to torch...", flush=True)
        grouped_np = ttnn.to_torch(grouped_rm).float().numpy()  # [1, 1, T_cap, H]
        print(f"[{label}] grouped ready", flush=True)
        ref_np = ref_grouped.float().numpy()
        D, B, S, H = dispatched.shape
        # Round-trip dispatched through bf16 so expected values match what the device sees.
        flat = dispatched.to(torch.bfloat16).float().reshape(D * B * S, H).numpy()

        sentinel = int(SENTINEL)
        for e in range(E_local):
            n = int(counts_np[e])
            start = int(offsets_np[e])
            end = int(offsets_np[e + 1])
            seen_active = 0
            for row_idx in range(start, end):
                src = int(plan_np[row_idx])
                if src == sentinel:
                    # pad slot (per-core or tail) — grouped row must be zero
                    if not np.allclose(grouped_np[0, 0, row_idx], 0.0, atol=1e-6):
                        raise AssertionError(
                            f"{label}: expert {e} pad row {row_idx} is not zero: " f"{grouped_np[0, 0, row_idx, :8]}"
                        )
                    continue
                expected_row = flat[src]
                got_row = grouped_np[0, 0, row_idx]
                if not np.allclose(got_row, expected_row, atol=1e-2):
                    raise AssertionError(
                        f"{label}: expert {e}, grouped row {row_idx} "
                        f"(src {src}) mismatch.\n"
                        f"  expected: {expected_row[:8]}...\n"
                        f"  got:      {got_row[:8]}..."
                    )
                seen_active += 1
            assert seen_active == n, f"{label}: expert {e} active row count {seen_active} != counts {n}"

    def test_grouped_matches_reference(self):
        D, B, S, H = 2, 1, 32, 64
        E, K = 4, 2
        local_expert_ids = torch.tensor([0, 1], dtype=torch.int32)
        dispatched = TestMoeGroupReference._make_dispatched(D, B, S, H)
        metadata = TestMoeGroupReference._make_metadata(D, B, S, K, E)
        self._check_correctness(dispatched, metadata, local_expert_ids, K, "small")

    def test_expert_zero_active_rows(self):
        """One expert gets no tokens — its slice must be all zeros, counts[e]=0."""
        D, B, S, H = 1, 1, 32, 64
        E, K = 8, 2
        local_expert_ids = torch.tensor([0, 5], dtype=torch.int32)
        dispatched = TestMoeGroupReference._make_dispatched(D, B, S, H, seed=3)
        # Build metadata that never assigns expert 5.
        choices = torch.tensor([0, 1, 2, 3, 4, 6, 7])
        md = torch.zeros(D, B, S, K, dtype=torch.int32)
        for s in range(S):
            perm = choices[torch.randperm(len(choices))][:K]
            md[0, 0, s] = perm.to(torch.int32)
        self._check_correctness(dispatched, md, local_expert_ids, K, "zero_active")

    def test_all_tokens_active_for_all_experts(self):
        """Every token routed to both local experts — maximum packing."""
        D, B, S, H = 2, 1, 32, 64
        E, K = 4, 2
        local_expert_ids = torch.tensor([0, 1], dtype=torch.int32)
        dispatched = TestMoeGroupReference._make_dispatched(D, B, S, H, seed=5)
        # Force all tokens to pick experts 0 and 1.
        md = torch.zeros(D, B, S, K, dtype=torch.int32)
        md[..., 0] = 0
        md[..., 1] = 1
        self._check_correctness(dispatched, md, local_expert_ids, K, "all_active")

    def test_non_tile_aligned_counts(self):
        """Active count not a multiple of 32 — tail pad rows must be zero."""
        D, B, S, H = 1, 1, 35, 64  # S=35 → non-tile-multiple token count
        E, K = 4, 2
        local_expert_ids = torch.tensor([0, 1], dtype=torch.int32)
        dispatched = TestMoeGroupReference._make_dispatched(D, B, S, H, seed=7)
        metadata = TestMoeGroupReference._make_metadata(D, B, S, K, E, seed=7)
        self._check_correctness(dispatched, metadata, local_expert_ids, K, "non_tile_aligned")

    def test_larger_h(self):
        """H=128 to exercise H-chunking path."""
        D, B, S, H = 2, 1, 32, 128
        E, K = 4, 2
        local_expert_ids = torch.tensor([0, 1], dtype=torch.int32)
        dispatched = TestMoeGroupReference._make_dispatched(D, B, S, H, seed=11)
        metadata = TestMoeGroupReference._make_metadata(D, B, S, K, E, seed=11)
        self._check_correctness(dispatched, metadata, local_expert_ids, K, "h128")

    def test_roofline_shape(self):
        """MoE roofline Config B from moe_summary.md:
        D=8, B=1, S=4096 (T_total=32768), H=4096, E=96, K=8, E_local=12.
        """
        D, B, S, H = 8, 1, 4096, 4096
        E, K = 96, 8
        E_local = 12
        local_expert_ids = torch.arange(E_local, dtype=torch.int32)
        dispatched = TestMoeGroupReference._make_dispatched(D, B, S, H, seed=42)
        metadata = TestMoeGroupReference._make_metadata(D, B, S, K, E, seed=42)
        self._check_correctness(dispatched, metadata, local_expert_ids, K, "roofline")

    def test_all_tokens_local_routing(self):
        """Worst-case T_active: every token's top-K picks only local experts.
        Exercises the `min(E_local, K) · T_total` upper bound used in T_cap —
        T_active equals that bound exactly, so `grouped` fills to capacity
        (modulo padding) and no tile is tail-skipped.
        """
        D, B, S, H = 4, 1, 512, 1024
        E, K = 16, 4
        E_local = 4
        local_expert_ids = torch.arange(E_local, dtype=torch.int32)
        dispatched = TestMoeGroupReference._make_dispatched(D, B, S, H, seed=23)
        # Force every token's top-K to be exactly the local experts.
        md = torch.zeros(D, B, S, K, dtype=torch.int32)
        for ki in range(K):
            md[..., ki] = int(local_expert_ids[ki % E_local])
        self._check_correctness(dispatched, md, local_expert_ids, K, "all_local")


# ---------------------------------------------------------------------------
# Device-time profiling (Tracy).
#
# Launch via:
#   tt-train/run_profiler.sh tt-train/tests/python/test_moe_group.py::TestMoeGroupProfile
#
# The run_profiler.sh wrapper invokes `python -m tracy -r -v -p` which writes
# a host-side ops log CSV (`ops_perf_results_<timestamp>.csv`) with one row per
# op launch, including `DEVICE KERNEL DURATION [ns]`.
#
# Each profile test brackets the measured iterations with `signpost("start")`
# and `signpost("stop")`, and we filter the ops log by those markers so warmup
# and other ops are excluded from the stats.
# ---------------------------------------------------------------------------


_PROFILE_LOG_DEVICE_CSV = "/home/training-team/danik/tt-metal/generated/profiler/.logs/profile_log_device.csv"


def _last_n_device_kernel_durations_ns(n):  # noqa: unused — kept for local debugging
    """Parse profile_log_device.csv and return the last `n` per-op device
    kernel durations in ns.

    profile_log_device.csv has raw device-side zones: one row per
    (core, RISC, zone, ZONE_START|ZONE_END) with a cycle timestamp.
    Each op launch has a unique `run host ID`. For each run_host_id with
    any *-KERNEL zones, duration = max(kernel ZONE_END cycles) -
    min(kernel ZONE_START cycles), converted to ns via CHIP_FREQ from
    the file header.

    Returns None if the CSV is unavailable or empty.
    """
    import os

    if not os.path.exists(_PROFILE_LOG_DEVICE_CSV):
        return None
    try:
        import pandas as pd
    except ImportError:
        return None

    with open(_PROFILE_LOG_DEVICE_CSV) as f:
        header_line = f.readline()
    # header line: "ARCH: wormhole_b0, CHIP_FREQ[MHz]: 1000, Max Compute Cores: 80"
    chip_freq_mhz = 1000
    for tok in header_line.split(","):
        if "CHIP_FREQ" in tok:
            try:
                chip_freq_mhz = int(tok.strip().split(":")[1].strip())
            except Exception:
                pass
    ns_per_cycle = 1000.0 / chip_freq_mhz  # 1e9 ns/s / (freq_mhz * 1e6 cycles/s)

    df = pd.read_csv(_PROFILE_LOG_DEVICE_CSV, skiprows=1, skipinitialspace=True)
    required = {"run host ID", "time[cycles since reset]", "zone name", "type"}
    if not required.issubset(df.columns):
        return None

    # Keep only KERNEL zones (BRISC-KERNEL, NCRISC-KERNEL, TRISCn-KERNEL).
    df = df[df["zone name"].astype(str).str.endswith("-KERNEL", na=False)]
    df["cycles"] = pd.to_numeric(df["time[cycles since reset]"], errors="coerce")
    df = df.dropna(subset=["cycles"])

    # Per run host ID: kernel duration = max(end cycles) - min(start cycles).
    durations_ns = []
    for run_id, group in df.groupby("run host ID"):
        starts = group[group["type"] == "ZONE_START"]["cycles"]
        ends = group[group["type"] == "ZONE_END"]["cycles"]
        if len(starts) == 0 or len(ends) == 0:
            continue
        cycles = ends.max() - starts.min()
        durations_ns.append((int(run_id), cycles * ns_per_cycle))

    if not durations_ns:
        return None
    # Sort by run_id (ascending), take last n.
    durations_ns.sort(key=lambda x: x[0])
    return [d for _, d in durations_ns[-n:]]


@pytest.mark.skipif(not _TTML_AVAILABLE, reason="ttml / ttnn not importable")
@pytest.mark.requires_device
class TestMoeGroupProfile:
    """Run the op under Tracy, filter ops log by signposts, print device time."""

    @staticmethod
    def _run_and_report(label, D, B, S, H, E, K, E_local, seed=0, num_iters=10, warmup=2, all_local=False):
        local_expert_ids = torch.arange(E_local, dtype=torch.int32)
        dispatched = TestMoeGroupReference._make_dispatched(D, B, S, H, seed=seed)
        if all_local:
            # Worst-case routing: every token's top-K is entirely local.
            # T_active = min(E_local, K) · T_total = the hard upper bound
            # the op's T_cap is sized for.
            metadata = torch.zeros(D, B, S, K, dtype=torch.int32)
            for ki in range(K):
                metadata[..., ki] = int(local_expert_ids[ki % E_local])
        else:
            metadata = TestMoeGroupReference._make_metadata(D, B, S, K, E, seed=seed)

        d_tt = _to_device_tensor(dispatched.float(), ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16)
        md_tt = _to_device_tensor(metadata.to(torch.int32), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)
        le_tt = _to_device_tensor(local_expert_ids.to(torch.int32), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)

        # Tracy signposts tag op rows in the CSV with a routing label so the
        # summary table can split "balanced" vs "fully_skewed" runs even at
        # identical shapes (op ATTRIBUTES don't carry routing info). Emit the
        # start signpost BEFORE the correctness check and warmup too so those
        # launches aren't tagged with an empty "-" routing in the table.
        try:
            from tracy import signpost as _signpost
        except Exception:
            _signpost = lambda _name: None
        routing = "fully_skewed" if all_local else "balanced"
        _signpost(f"moe_group_start_{routing}")

        # Correctness sanity — run one op invocation and compare against the
        # torch reference before starting timed iters. Catches silent shape /
        # routing regressions that the correctness suite might not cover.
        TestMoeGroupDevice._check_correctness(
            dispatched, metadata, local_expert_ids, int(K), label=f"{label}[correctness]"
        )

        device = ttml.autograd.AutoContext.get_instance().get_device()
        for _ in range(warmup):
            ttml.ops.metal_ops.moe_group(d_tt, md_tt, le_tt, int(E_local), int(K))
        ttnn.synchronize_device(device)

        # Run N iters with per-iter profiler flush so the Tracy device CSV gets
        # one row per launch. Host-wall timing is intentionally NOT reported —
        # it conflates python overhead + synchronize + device kernel and was
        # misleading next to the DRAM roofline. The summary table produced by
        # parse_profile_results.py uses device-kernel-only times from the CSV.
        for _ in range(num_iters):
            ttml.ops.metal_ops.moe_group(d_tt, md_tt, le_tt, int(E_local), int(K))
            ttnn.synchronize_device(device)
            ttnn.ReadDeviceProfiler(device)  # flush device zones for this op
        _signpost(f"moe_group_end_{routing}")

        T_total = D * B * S
        T_cap = moe_group_t_cap(E_local, K, D, B, S, num_total_cores=TestMoeGroupDevice._device_num_total_cores())
        print(
            f"\n[{label}] D={D} B={B} S={S} T_total={T_total} H={H} "
            f"E={E} K={K} E_local={E_local} T_cap={T_cap}  iters={num_iters}  "
            f"(device-kernel times: see summary table from parse_profile_results.py)"
        )

    # -------- parametrized shape sweeps --------

    # Sweep over (D, B, S, H, E, K, E_local).
    # Organized by category so the output groups cleanly by what's being tested.
    _SWEEP_TINY = [
        # D  B   S     H     E   K  E_local   — smoke / scan fast-path
        (2, 1, 128, 512, 4, 2, 2),
        (2, 1, 128, 1024, 8, 2, 2),
        (4, 1, 256, 2048, 16, 4, 4),
    ]
    _SWEEP_H_SWEEP = [
        # Vary H alone (D=8, S=2048, E=32, K=4, E_local=4)
        (8, 1, 2048, 1024, 32, 4, 4),
        (8, 1, 2048, 2048, 32, 4, 4),
        (8, 1, 2048, 4096, 32, 4, 4),
        (8, 1, 2048, 7168, 32, 4, 4),
        (8, 1, 2048, 8192, 32, 4, 4),
    ]
    _SWEEP_S_SWEEP = [
        # Vary S alone (D=8, H=4096, E=96, K=8, E_local=12)
        (8, 1, 512, 4096, 96, 8, 12),
        (8, 1, 1024, 4096, 96, 8, 12),
        (8, 1, 2048, 4096, 96, 8, 12),
        (8, 1, 4096, 4096, 96, 8, 12),  # roofline Config B
        (8, 1, 8192, 4096, 96, 8, 12),
    ]
    _SWEEP_ROUTING = [
        # Vary E/K/E_local at fixed D=8, S=4096, H=4096: sparsity & local-experts count
        (8, 1, 4096, 4096, 16, 2, 2),  # sparse: K=2, few local experts
        (8, 1, 4096, 4096, 32, 8, 2),  # dense-k, few local
        (8, 1, 4096, 4096, 64, 8, 4),  # mid
        (8, 1, 4096, 4096, 96, 8, 12),  # Config B
        (8, 1, 4096, 4096, 128, 8, 16),  # many experts
    ]
    _SWEEP_BIG_H = [
        # DeepSeek / SwiGLU-like very large hidden size
        (8, 1, 1024, 7168, 64, 8, 2),
        (8, 1, 2048, 7168, 64, 8, 2),
        (8, 1, 4096, 7168, 64, 8, 2),
        (8, 1, 4096, 8192, 64, 8, 4),
    ]

    @pytest.mark.parametrize(
        "D,B,S,H,E,K,E_local",
        _SWEEP_TINY + _SWEEP_H_SWEEP + _SWEEP_S_SWEEP + _SWEEP_ROUTING + _SWEEP_BIG_H,
    )
    def test_profile_sweep(self, D, B, S, H, E, K, E_local):
        """Profile a grid of shape configs covering: tiny, H-sweep, S-sweep,
        routing sparsity, and large-H (DeepSeek-like) cases."""
        self._run_and_report(
            label=f"D={D}_S={S}_H={H}_E={E}_K={K}_Eloc={E_local}",
            D=D,
            B=B,
            S=S,
            H=H,
            E=E,
            K=K,
            E_local=E_local,
            seed=42,
            num_iters=10,
            warmup=2,
        )

    def test_profile_roofline(self):
        """Roofline Config B shortcut (same numbers as sweep row 3)."""
        self._run_and_report(
            label="roofline",
            D=8,
            B=1,
            S=4096,
            H=4096,
            E=96,
            K=8,
            E_local=12,
            seed=42,
            num_iters=10,
            warmup=2,
        )

    def test_profile_all_local_routing(self):
        """Worst-case routing: every token's top-K is entirely local experts.
        Max T_active = min(E_local, K) · T_total. Stress-tests the full
        allocation path and the scatter at peak load.
        Uses the same shape as the balanced roofline test — the Tracy
        signpost label "fully_skewed" vs "balanced" lets the parser
        separate the two routing patterns in the summary table.
        """
        self._run_and_report(
            label="all_local",
            D=8,
            B=1,
            S=4096,
            H=4096,
            E=96,
            K=8,
            E_local=12,
            seed=42,
            num_iters=10,
            warmup=2,
            all_local=True,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
