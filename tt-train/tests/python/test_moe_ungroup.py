# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MoE ungroup op (the inverse of moe_group, fused with top-K weighting).

The op consumes:
  expert_out      : [1, 1, T_cap, H]      TILE bf16
  plan            : [1, 1, 1, T_cap]      uint32  — moe_group's plan
  offsets         : [1, 1, 1, E_local+1]  uint32  — moe_group's offsets
  grouped_scores  : [1, 1, 1, T_cap]      bf16    — moe_group's grouped_scores
                                                    (= scores[plan[i], k_slot[i]])

and produces:
  ungrouped: [D, B, S, H] ROW_MAJOR bf16

The torch reference and tests additionally use counts / metadata / scores /
local_expert_ids to build inputs and check correctness; these are NOT op
inputs.

Reference implementation in `moe_ungroup_torch_reference()` below.
"""

from __future__ import annotations

import math
import os
from typing import Tuple

import numpy as np
import pytest
import torch


SENTINEL = np.uint32(0xFFFFFFFF)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def moe_ungroup_torch_reference(
    expert_out: torch.Tensor,  # [1, 1, T_cap, H]
    plan: torch.Tensor,  # [T_cap]
    offsets: torch.Tensor,  # [E_local + 1]
    counts: torch.Tensor,  # [E_local]
    metadata: torch.Tensor,  # [D, B, S, K]
    scores: torch.Tensor,  # [D, B, S, K]
    local_expert_ids: torch.Tensor,  # [E_local]
    *,
    d: int,
    b: int,
    s: int,
    h: int,
    k: int,
) -> torch.Tensor:
    """Torch reference for moe_ungroup.

    For each local expert e, walk the active rows [offsets[e], offsets[e]+counts[e]):
      - flat = plan[i]
      - decode (d_, b_, s_) from flat = d_*B*S + b_*S + s_
      - find k_slot in metadata[d_, b_, s_, :K] equal to local_expert_ids[e]
      - w = scores[d_, b_, s_, k_slot]
      - ungrouped[d_, b_, s_, :] += w * expert_out[0, 0, i, :]
    """
    E_local = int(local_expert_ids.numel())
    out = torch.zeros(d, b, s, h, dtype=expert_out.dtype)

    plan_np = plan.numpy().astype(np.int64)
    offsets_np = offsets.numpy().astype(np.int64)
    counts_np = counts.numpy().astype(np.int64)
    md_np = metadata.numpy()
    leids_np = local_expert_ids.numpy()
    eo = expert_out.reshape(-1, h)  # [T_cap, H]
    sc_flat = scores.reshape(d * b * s, k)

    sentinel = int(SENTINEL)
    for e in range(E_local):
        eid = int(leids_np[e])
        for i in range(int(offsets_np[e]), int(offsets_np[e + 1])):
            flat = int(plan_np[i])
            if flat == sentinel:
                # Pad slot (per-core round_up_4 padding or tile-alignment tail).
                # Real entries can be interleaved with SENTINELs across the
                # whole [offsets[e], offsets[e+1]) range — do NOT cut off at
                # offsets[e]+counts[e]; the SENTINEL check is what filters pads.
                continue
            # Decode (d_, b_, s_) from flat = d_*B*S + b_*S + s_
            d_ = flat // (b * s)
            rem = flat - d_ * (b * s)
            b_ = rem // s
            s_ = rem - b_ * s
            md_row = md_np[d_, b_, s_]
            k_slot = -1
            for ki in range(k):
                if int(md_row[ki]) == eid:
                    k_slot = ki
                    break
            if k_slot < 0:
                # Shouldn't happen for real plan entries.
                continue
            w = sc_flat[flat, k_slot].to(torch.float32)
            out[d_, b_, s_] += (w * eo[i].to(torch.float32)).to(out.dtype)

    return out


# ---------------------------------------------------------------------------
# Device tests — require ttml device.
# ---------------------------------------------------------------------------


try:
    import ttnn  # noqa: F401
    import ttml  # noqa: F401

    _TTML_AVAILABLE = True
except Exception as _e:
    _TTML_AVAILABLE = False
    _IMPORT_ERROR = _e
else:
    _IMPORT_ERROR = None

# ---------------------------------------------------------------------------
# Standalone helpers (no dependency on test_moe_group.py — this PR can land
# before the moe_group PR). These mirror the moe_group reference; we only
# need them to fabricate (grouped, counts, offsets, plan) inputs to the
# ungroup op without invoking the device moe_group kernel.
# ---------------------------------------------------------------------------


def _round_up_32(x: int) -> int:
    return ((x + 31) // 32) * 32


def _hal_l1_alignment_bytes() -> int:
    """L1 NOC alignment in bytes from the same HAL-backed API the op uses."""
    try:
        import ttnn

        return int(ttnn.get_l1_alignment())
    except Exception:
        # Reference-only tests can run without a visible device.
        return 16


def _cursor_align() -> int:
    return _hal_l1_alignment_bytes() // 2


def moe_group_t_cap(e_local: int, k: int, d: int, b: int, s: int, num_total_cores: int = 64) -> int:
    cursor_align = _cursor_align()
    return _round_up_32(min(e_local, k) * d * b * s + e_local * (32 + (cursor_align - 1) * num_total_cores))


def _make_dispatched(D: int, B: int, S: int, H: int, seed: int = 0) -> torch.Tensor:
    """Each (d, b, s) row carries its flat row index broadcast across H."""
    torch.manual_seed(seed)
    idx = torch.arange(D * B * S, dtype=torch.float32).reshape(D, B, S, 1)
    return idx.expand(D, B, S, H).clone()


def _make_metadata(D: int, B: int, S: int, K: int, E: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed + 1)
    out = torch.zeros(D, B, S, K, dtype=torch.int32)
    for d in range(D):
        for b in range(B):
            for s in range(S):
                out[d, b, s] = torch.randperm(E)[:K].to(torch.int32)
    return out


def moe_group_torch_reference(
    dispatched: torch.Tensor,
    metadata: torch.Tensor,
    scores: torch.Tensor,
    local_expert_ids: torch.Tensor,
    *,
    k: int,
    num_total_cores: int = 64,
    t_cap: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch reference for moe_group — fabricates (grouped, grouped_scores,
    counts, offsets, plan) inputs to ungroup without calling the device
    moe_group op so this test file stays independent of moe_group's
    implementation/PR.

    `grouped_scores[i] = scores[plan[i], k_slot]` for each active row i,
    where `k_slot` is the position in metadata[plan[i], :K] that matches
    the local expert owning row i. 0.0 in pad/sentinel slots."""
    D, B, S, H = dispatched.shape
    E_local = int(local_expert_ids.numel())
    T_cap = t_cap if t_cap > 0 else moe_group_t_cap(E_local, k, D, B, S, num_total_cores=num_total_cores)

    total_rows = D * B * S
    flat = dispatched.reshape(total_rows, H)
    md_np = metadata.reshape(total_rows, k).numpy()
    leids = local_expert_ids.numpy().astype(md_np.dtype)
    sc_np = scores.reshape(total_rows, k).to(torch.bfloat16).float().numpy()

    # Per-expert hit mask + the matching k_slot per active row.
    hits = np.zeros((E_local, total_rows), dtype=np.bool_)
    k_slots = np.zeros((E_local, total_rows), dtype=np.uint32)
    for e_idx in range(E_local):
        eq = md_np == leids[e_idx]
        hits[e_idx] = eq.any(axis=1)
        k_slots[e_idx] = np.argmax(eq, axis=1).astype(np.uint32)
    counts = hits.sum(axis=1).astype(np.uint32)

    slice_size = (total_rows + num_total_cores - 1) // num_total_cores

    def _round_up_4(x):
        return (x + 3) & ~3

    local_counts = np.zeros((num_total_cores, E_local), dtype=np.uint32)
    for c in range(num_total_cores):
        s_start = c * slice_size
        s_end = min(s_start + slice_size, total_rows)
        if s_start >= total_rows:
            continue
        for e_idx in range(E_local):
            local_counts[c, e_idx] = hits[e_idx, s_start:s_end].sum()

    offsets = np.zeros(E_local + 1, dtype=np.uint32)
    for e_idx in range(E_local):
        running = int(offsets[e_idx])
        for c in range(num_total_cores):
            running += int(_round_up_4(int(local_counts[c, e_idx])))
        offsets[e_idx + 1] = _round_up_32(running)

    plan = np.full(T_cap, SENTINEL, dtype=np.uint32)
    grouped_scores_np = np.zeros(T_cap, dtype=np.float32)
    for e_idx in range(E_local):
        running = int(offsets[e_idx])
        for c in range(num_total_cores):
            s_start = c * slice_size
            s_end = min(s_start + slice_size, total_rows)
            if s_start >= total_rows:
                continue
            core_hit_idx = np.nonzero(hits[e_idx, s_start:s_end])[0].astype(np.uint32) + s_start
            n = len(core_hit_idx)
            if n > 0:
                plan[running : running + n] = core_hit_idx
                # grouped_scores[i] = scores[t, k_slot] for each active row.
                ks = k_slots[e_idx, core_hit_idx]
                grouped_scores_np[running : running + n] = sc_np[core_hit_idx, ks]
            running += int(_round_up_4(n))

    grouped = torch.zeros(1, 1, T_cap, H, dtype=dispatched.dtype)
    valid_mask = plan != SENTINEL
    valid_idx = np.nonzero(valid_mask)[0]
    src_idx = plan[valid_idx].astype(np.int64)
    grouped[0, 0, valid_idx, :] = flat[src_idx]

    return (
        grouped,
        torch.from_numpy(grouped_scores_np),
        torch.from_numpy(counts.astype(np.int64)),
        torch.from_numpy(offsets.astype(np.int64)),
        torch.from_numpy(plan.astype(np.int64)),
    )


def _to_device_tensor(t: torch.Tensor, layout, dtype):
    device = ttml.autograd.AutoContext.get_instance().get_device()
    return ttnn.from_torch(
        t.float() if t.dtype == torch.bfloat16 else t,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _device_grid_size() -> int:
    device = ttml.autograd.AutoContext.get_instance().get_device()
    grid = device.compute_with_storage_grid_size()
    return int(grid.x) * int(grid.y)


def _device_num_total_cores(e_local: int = 0, k: int = 0, d: int = 0, b: int = 0, s: int = 0) -> int:
    grid_size = _device_grid_size()
    if e_local == 0:
        return grid_size
    t_cap = moe_group_t_cap(e_local, k, d, b, s, num_total_cores=grid_size)
    total_tiles = t_cap // 32
    return min(total_tiles, grid_size)


def _bf16_round(t: torch.Tensor) -> torch.Tensor:
    """Round-trip through bf16 to match what the device would produce per element."""
    return t.to(torch.bfloat16).float()


@pytest.mark.skipif(not _TTML_AVAILABLE, reason="ttml / ttnn not importable")
@pytest.mark.requires_device
class TestMoeUngroupDevice:
    """Device tests: run group → ungroup, compare against torch reference."""

    @staticmethod
    def _build_group_inputs(
        dispatched: torch.Tensor,
        metadata: torch.Tensor,
        scores: torch.Tensor,
        local_expert_ids: torch.Tensor,
        k: int,
    ):
        """Use the torch reference for moe_group to fabricate
        (grouped, grouped_scores, counts, offsets, plan), then push to
        device. Keeps this test file independent of the moe_group device op
        — it only exercises moe_ungroup."""
        D, B, S, H = dispatched.shape
        E_local = int(local_expert_ids.numel())
        num_total_cores = _device_num_total_cores(E_local, k, D, B, S)
        grouped_t, grouped_scores_t, counts_t, offsets_t, plan_t = moe_group_torch_reference(
            dispatched, metadata, scores, local_expert_ids, k=k, num_total_cores=num_total_cores
        )
        # Push to device. grouped goes TILE bf16; for H not divisible by 32 the
        # row-major → TILE conversion needs explicit padding, so use from_torch
        # with TILE layout directly which pads with zeros internally.
        grouped_tt = _to_device_tensor(grouped_t.float(), ttnn.TILE_LAYOUT, ttnn.bfloat16)
        grouped_scores_tt = _to_device_tensor(
            grouped_scores_t.reshape(1, 1, 1, -1).float(), ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16
        )
        counts_tt = _to_device_tensor(counts_t.to(torch.int32).reshape(1, 1, 1, -1), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
        offsets_tt = _to_device_tensor(
            offsets_t.to(torch.int32).reshape(1, 1, 1, -1), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32
        )
        plan_tt = _to_device_tensor(plan_t.to(torch.int32).reshape(1, 1, 1, -1), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
        return (
            grouped_tt,
            grouped_scores_tt,
            counts_tt,
            offsets_tt,
            plan_tt,
            grouped_t,
            counts_t,
            offsets_t,
            plan_t,
        )

    @staticmethod
    def _run_group_then_ungroup(
        dispatched: torch.Tensor,
        metadata: torch.Tensor,
        scores: torch.Tensor,
        local_expert_ids: torch.Tensor,
        k: int,
    ):
        """Use torch reference for moe_group, then run device moe_ungroup."""
        D, B, S, H = dispatched.shape
        E_local = int(local_expert_ids.numel())

        (
            grouped,
            grouped_scores,
            counts,
            offsets,
            plan,
            grouped_t,
            counts_t,
            offsets_t,
            plan_t,
        ) = TestMoeUngroupDevice._build_group_inputs(dispatched, metadata, scores, local_expert_ids, k)

        device = ttml.autograd.AutoContext.get_instance().get_device()
        ttnn.synchronize_device(device)
        ungrouped = ttml.ops.metal.moe_ungroup(
            grouped,
            plan,
            offsets,
            grouped_scores,
            int(E_local),
            int(D),
            int(B),
            int(S),
        )
        ttnn.synchronize_device(device)
        return ungrouped, grouped_t, counts_t, offsets_t, plan_t

    @staticmethod
    def _check_correctness(
        dispatched: torch.Tensor,
        metadata: torch.Tensor,
        scores: torch.Tensor,
        local_expert_ids: torch.Tensor,
        k: int,
        label: str,
        atol: float = 5e-2,
        rtol: float = 3e-2,
    ):
        D, B, S, H = dispatched.shape

        ungrouped_tt, grouped_t, counts_t, offsets_t, plan_t = TestMoeUngroupDevice._run_group_then_ungroup(
            dispatched, metadata, scores, local_expert_ids, k
        )

        # The torch reference for moe_group already gave us plan/offsets/counts
        # and the same `grouped` we pushed to device, so we build the ungroup
        # reference against those directly (no device readback needed).
        ref = moe_ungroup_torch_reference(
            grouped_t.to(torch.bfloat16),  # [1,1,T_cap,H]
            plan_t,
            offsets_t,
            counts_t,
            metadata,
            scores.to(torch.bfloat16),
            local_expert_ids,
            d=D,
            b=B,
            s=S,
            h=H,
            k=k,
        )

        ungrouped_np = ttnn.to_torch(ungrouped_tt).float()  # [D, B, S, H]
        ref_np = ref.float()

        # Per-token comparison.
        T_total = D * B * S
        flat_dev = ungrouped_np.reshape(T_total, H).numpy()
        flat_ref = ref_np.reshape(T_total, H).numpy()

        max_diff = np.abs(flat_dev - flat_ref).max()
        print(f"\n[{label}] max abs diff = {max_diff:.4f}")

        if not np.allclose(flat_dev, flat_ref, atol=atol, rtol=rtol):
            mismatch_rows = []
            for t in range(T_total):
                if not np.allclose(flat_dev[t], flat_ref[t], atol=atol, rtol=rtol):
                    mismatch_rows.append(t)
                    if len(mismatch_rows) >= 5:
                        break
            for t in mismatch_rows:
                print(f"  token {t}: ref[:8]={flat_ref[t, :8]} got[:8]={flat_dev[t, :8]}")
            raise AssertionError(
                f"{label}: ungrouped mismatch ({len(mismatch_rows)}+ rows different, " f"max abs diff={max_diff})"
            )

    def test_basic(self):
        D, B, S, H = 2, 1, 32, 64
        E, K = 4, 2
        local_expert_ids = torch.tensor([0, 1], dtype=torch.int32)
        dispatched = _make_dispatched(D, B, S, H, seed=0)
        metadata = _make_metadata(D, B, S, K, E, seed=0)
        scores = torch.rand(D, B, S, K) * 0.5
        self._check_correctness(dispatched, metadata, scores, local_expert_ids, K, "basic")

    def test_zero_active(self):
        """One local expert gets no tokens — those rows accumulate nothing for that expert."""
        D, B, S, H = 1, 1, 32, 64
        E, K = 8, 2
        local_expert_ids = torch.tensor([0, 5], dtype=torch.int32)
        dispatched = _make_dispatched(D, B, S, H, seed=3)
        choices = torch.tensor([0, 1, 2, 3, 4, 6, 7])
        md = torch.zeros(D, B, S, K, dtype=torch.int32)
        for s in range(S):
            perm = choices[torch.randperm(len(choices))][:K]
            md[0, 0, s] = perm.to(torch.int32)
        scores = torch.rand(D, B, S, K) * 0.5
        self._check_correctness(dispatched, md, scores, local_expert_ids, K, "zero_active")

    def test_rejects_zero_local_experts(self):
        """e_local=0 would leave BRISC waiting for a reader release after pre-zero."""
        D, B, S, H = 1, 1, 32, 64
        T_cap = 32
        grouped = _to_device_tensor(torch.zeros(1, 1, T_cap, H), ttnn.TILE_LAYOUT, ttnn.bfloat16)
        plan = _to_device_tensor(torch.zeros(1, 1, 1, T_cap, dtype=torch.int32), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
        offsets = _to_device_tensor(torch.zeros(1, 1, 1, 1, dtype=torch.int32), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
        grouped_scores = _to_device_tensor(torch.zeros(1, 1, 1, T_cap), ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16)

        with pytest.raises(RuntimeError, match="e_local must be > 0"):
            ttml.ops.metal.moe_ungroup(grouped, plan, offsets, grouped_scores, 0, D, B, S)

    def test_rejects_non_tile_aligned_t_cap(self):
        """Reader/writer consume plan/grouped_scores/expert_out in 32-row chunks."""
        D, B, S, H = 1, 1, 32, 64
        T_cap = 33
        grouped = _to_device_tensor(torch.zeros(1, 1, T_cap, H), ttnn.TILE_LAYOUT, ttnn.bfloat16)
        plan = _to_device_tensor(torch.zeros(1, 1, 1, T_cap, dtype=torch.int32), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
        offsets = _to_device_tensor(torch.zeros(1, 1, 1, 2, dtype=torch.int32), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
        grouped_scores = _to_device_tensor(torch.zeros(1, 1, 1, T_cap), ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16)

        with pytest.raises(RuntimeError, match="T_cap must be a multiple of 32"):
            ttml.ops.metal.moe_ungroup(grouped, plan, offsets, grouped_scores, 1, D, B, S)

    def test_all_tokens_active(self):
        """Every token routed to all local experts — multi-expert accumulation per row."""
        D, B, S, H = 2, 1, 32, 64
        E, K = 4, 2
        local_expert_ids = torch.tensor([0, 1], dtype=torch.int32)
        dispatched = _make_dispatched(D, B, S, H, seed=5)
        md = torch.zeros(D, B, S, K, dtype=torch.int32)
        md[..., 0] = 0
        md[..., 1] = 1
        scores = torch.rand(D, B, S, K) * 0.5
        self._check_correctness(dispatched, md, scores, local_expert_ids, K, "all_active")

    def test_non_tile_aligned(self):
        D, B, S, H = 1, 1, 35, 64
        E, K = 4, 2
        local_expert_ids = torch.tensor([0, 1], dtype=torch.int32)
        dispatched = _make_dispatched(D, B, S, H, seed=7)
        metadata = _make_metadata(D, B, S, K, E, seed=7)
        scores = torch.rand(D, B, S, K) * 0.5
        self._check_correctness(dispatched, metadata, scores, local_expert_ids, K, "non_tile_aligned")

    def test_larger_h(self):
        D, B, S, H = 2, 1, 32, 128
        E, K = 4, 2
        local_expert_ids = torch.tensor([0, 1], dtype=torch.int32)
        dispatched = _make_dispatched(D, B, S, H, seed=11)
        metadata = _make_metadata(D, B, S, K, E, seed=11)
        scores = torch.rand(D, B, S, K) * 0.5
        self._check_correctness(dispatched, metadata, scores, local_expert_ids, K, "h128")

    @pytest.mark.parametrize("H", [48, 80, 96, 144])
    def test_non_tile_aligned_h(self, H):
        """H not divisible by TILE_WIDTH=32 — last tile column is partial.
        Reader must read only h*2 valid bytes per row and zero-pad the L1 tail
        so tilize sees zeros in the partial last tile, otherwise the read
        crosses into the next row in DRAM and ungrouped output is wrong."""
        D, B, S = 2, 1, 32
        E, K = 4, 2
        local_expert_ids = torch.tensor([0, 1], dtype=torch.int32)
        dispatched = _make_dispatched(D, B, S, H, seed=H)
        metadata = _make_metadata(D, B, S, K, E, seed=H)
        scores = torch.rand(D, B, S, K, generator=torch.Generator().manual_seed(H)) * 0.5
        self._check_correctness(dispatched, metadata, scores, local_expert_ids, K, f"h{H}")

    @pytest.mark.parametrize("E_local", [32, 64, 128, 300])
    def test_large_e_local(self, E_local):
        """E_local up to 300 — exercises dynamic sizing of cb_scratch (stage,
        offsets_buf, leids_buf, etc.). Pre-fix layouts were sized for small
        E_local and would overflow into adjacent fields for E_local > 16."""
        D, B, S, H = 2, 1, 64, 64
        E = max(E_local * 2, 64)
        K = 4
        local_expert_ids = torch.arange(E_local, dtype=torch.int32)
        dispatched = _make_dispatched(D, B, S, H, seed=E_local)
        metadata = _make_metadata(D, B, S, K, E, seed=E_local)
        scores = torch.rand(D, B, S, K, generator=torch.Generator().manual_seed(E_local)) * 0.5
        self._check_correctness(dispatched, metadata, scores, local_expert_ids, K, f"e_local{E_local}")


# ---------------------------------------------------------------------------
# Device-time profiling (Tracy). Mirrors test_moe_group.py.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _TTML_AVAILABLE, reason="ttml / ttnn not importable")
@pytest.mark.skipif(
    os.environ.get("TTML_RUN_PROFILE_TESTS", "0") not in ("1", "true", "True"),
    reason="Profile sweep is opt-in: set TTML_RUN_PROFILE_TESTS=1 to enable",
)
@pytest.mark.requires_device
class TestMoeUngroupProfile:
    """Run moe_group → moe_ungroup back-to-back under Tracy.

    Reuses the same shape sweep as TestMoeGroupProfile so the two summary
    tables are directly comparable.

    Opt-in to keep this out of the default CI run. Enable with
    `TTML_RUN_PROFILE_TESTS=1`.
    """

    @staticmethod
    def _run_and_report(label, D, B, S, H, E, K, E_local, seed=0, num_iters=10, warmup=2, all_local=False):
        local_expert_ids = torch.arange(E_local, dtype=torch.int32)
        dispatched = _make_dispatched(D, B, S, H, seed=seed)
        if all_local:
            metadata = torch.zeros(D, B, S, K, dtype=torch.int32)
            for ki in range(K):
                metadata[..., ki] = int(local_expert_ids[ki % E_local])
        else:
            metadata = _make_metadata(D, B, S, K, E, seed=seed)
        scores = torch.rand(D, B, S, K, generator=torch.Generator().manual_seed(seed)) * 0.5

        try:
            from tracy import signpost as _signpost
        except Exception:
            _signpost = lambda _name: None

        routing = "fully_skewed" if all_local else "balanced"
        _signpost(f"moe_ungroup_start_{routing}")

        # Correctness sanity (1 invocation, included in signpost range so the
        # parser sees it tagged with the right routing label).
        TestMoeUngroupDevice._check_correctness(
            dispatched, metadata, scores, local_expert_ids, int(K), label=f"{label}[correctness]"
        )

        device = ttml.autograd.AutoContext.get_instance().get_device()
        # Build (grouped, grouped_scores, counts, offsets, plan) once via the
        # torch reference and reuse across iters — we're timing moe_ungroup
        # only (and the torch ref keeps this independent of moe_group).
        grouped, grouped_scores, _counts, offsets, plan, _, _, _, _ = TestMoeUngroupDevice._build_group_inputs(
            dispatched, metadata, scores, local_expert_ids, int(K)
        )
        # Warmup.
        for _ in range(warmup):
            ttml.ops.metal.moe_ungroup(
                grouped,
                plan,
                offsets,
                grouped_scores,
                int(E_local),
                int(D),
                int(B),
                int(S),
            )
        ttnn.synchronize_device(device)

        # Timed iters with per-iter profiler flush.
        for _ in range(num_iters):
            ttml.ops.metal.moe_ungroup(
                grouped,
                plan,
                offsets,
                grouped_scores,
                int(E_local),
                int(D),
                int(B),
                int(S),
            )
            ttnn.synchronize_device(device)
            ttnn.ReadDeviceProfiler(device)
        _signpost(f"moe_ungroup_end_{routing}")

        T_total = D * B * S
        T_cap = moe_group_t_cap(E_local, K, D, B, S, num_total_cores=_device_num_total_cores())
        print(
            f"\n[{label}] D={D} B={B} S={S} T_total={T_total} H={H} "
            f"E={E} K={K} E_local={E_local} T_cap={T_cap}  iters={num_iters}  "
            f"(device-kernel times: see summary table from parse_profile_results.py)"
        )

    # Same shape sweeps as moe_group, so the summary tables are aligned.
    _SWEEP_TINY = [
        (2, 1, 128, 512, 4, 2, 2),
        (2, 1, 128, 1024, 8, 2, 2),
        (4, 1, 256, 2048, 16, 4, 4),
    ]
    _SWEEP_H_SWEEP = [
        (8, 1, 2048, 1024, 32, 4, 4),
        (8, 1, 2048, 2048, 32, 4, 4),
        (8, 1, 2048, 4096, 32, 4, 4),
        (8, 1, 2048, 7168, 32, 4, 4),
        (8, 1, 2048, 8192, 32, 4, 4),
    ]
    _SWEEP_S_SWEEP = [
        (8, 1, 512, 4096, 96, 8, 12),
        (8, 1, 1024, 4096, 96, 8, 12),
        (8, 1, 2048, 4096, 96, 8, 12),
        (8, 1, 4096, 4096, 96, 8, 12),  # roofline Config B
        (8, 1, 8192, 4096, 96, 8, 12),
    ]
    _SWEEP_ROUTING = [
        (8, 1, 4096, 4096, 16, 2, 2),
        (8, 1, 4096, 4096, 32, 8, 2),
        (8, 1, 4096, 4096, 64, 8, 4),
        (8, 1, 4096, 4096, 96, 8, 12),
        (8, 1, 4096, 4096, 128, 8, 16),
    ]
    _SWEEP_BIG_H = [
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
