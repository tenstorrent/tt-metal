# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MoE ungroup autograd wrapper (ttml.ops.moe.moe_ungroup_op).

The wrapper around metal::moe_ungroup adds:
  - autograd: expert_out and grouped_scores get gradients
  - forward: identical output to the raw kernel (sanity-checked)
  - backward Step A: gather d(ungrouped) into grouped layout via metal::moe_group
  - backward Step B: d(expert_out)     = grouped_scores ⊙ grad_grouped
  - backward Step C: d(grouped_scores) = Σ_h expert_out[i,h] · grad_grouped[i,h]

Reuses moe_group_torch_reference to build inputs (so this file is
independent of the device moe_group op).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_moe_group import (  # noqa: E402
    SENTINEL,
    moe_group_t_cap,
    moe_group_torch_reference,
)

from tests.ttnn.utils_for_testing import assert_with_pcc  # noqa: E402

try:
    import ttnn
    import ttml

    _TTML_AVAILABLE = True
except Exception:
    _TTML_AVAILABLE = False


def _device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _to_device_tensor(t: torch.Tensor, layout, dtype) -> "ttnn.Tensor":
    return ttnn.from_torch(
        t.float() if t.dtype == torch.bfloat16 else t,
        dtype=dtype,
        layout=layout,
        device=_device(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_autograd_tensor(
    t: torch.Tensor, *, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
) -> "ttml.autograd.Tensor":
    arr = t.float().numpy() if t.dtype == torch.bfloat16 else t.numpy()
    return ttml.autograd.Tensor.from_numpy(arr.astype(np.float32), layout=layout, new_type=dtype)


def _device_grid_size() -> int:
    grid = _device().compute_with_storage_grid_size()
    return int(grid.x) * int(grid.y)


def _device_num_total_cores(e_local: int, k: int, d: int, b: int, s: int) -> int:
    # Mirror the kernel's split_work_to_cores(grid, total_tiles) split.
    grid_size = _device_grid_size()
    t_cap = moe_group_t_cap(e_local, k, d, b, s, num_total_cores=grid_size)
    return min(grid_size, t_cap // 32)


def _make_inputs(D: int, B: int, S: int, H: int, K: int, E: int, E_local: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    dispatched = torch.randn(D, B, S, H, generator=g)
    metadata = torch.zeros(D, B, S, K, dtype=torch.int32)
    for d in range(D):
        for b in range(B):
            for s in range(S):
                metadata[d, b, s] = torch.randperm(E, generator=g)[:K].to(torch.int32)
    scores = torch.softmax(torch.randn(D, B, S, K, generator=g), dim=-1)
    leids = torch.arange(E_local, dtype=torch.int32)
    return dispatched, metadata, scores, leids


def _build_group_outputs(dispatched, metadata, scores, leids, K, _num_total_cores_unused):
    """Run the device moe_group kernel to produce its routing tensors.

    Using the kernel's plan/offsets (rather than the torch reference's)
    is required because torch reference's per-core padding may not match
    the kernel's split_work_to_cores split — they produce different
    `offsets[-1]` (T_used). The wrapper's backward Step A also calls the
    kernel, so they must agree."""
    D, B, S, H = dispatched.shape
    E_local = leids.numel()
    d_tt = _to_device_tensor(dispatched, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16)
    md_tt = _to_device_tensor(metadata, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)
    sc_tt = _to_device_tensor(scores, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16)
    le_tt = _to_device_tensor(leids, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)
    grouped, grouped_scores, k_slot, counts, offsets, plan = ttml.ops.metal_ops.moe_group(
        d_tt, md_tt, sc_tt, le_tt, int(E_local), int(K)
    )
    # plan_t is host-side numpy of the kernel's plan, used for active-row mask.
    plan_t = ttnn.to_torch(plan).numpy().flatten().astype(np.uint32)
    grouped_scores_t = ttnn.to_torch(grouped_scores).float().flatten()
    return grouped, grouped_scores, plan, offsets, plan_t, grouped_scores_t


# (D, B, S, H, E, K, E_local). "small" is the original tiny shape;
# the others are pulled from the moe_group / moe_ungroup perf tables.
# Tests assume E_local == E (single-device sparse MoE).
SHAPES = [
    pytest.param((1, 1, 32, 32, 4, 2, 4), id="small"),
    pytest.param((2, 1, 128, 512, 2, 2, 2), id="perf-d2-s128-h512"),
    pytest.param((4, 1, 256, 2048, 4, 4, 4), id="perf-d4-s256-h2048"),
]


@pytest.mark.skipif(not _TTML_AVAILABLE, reason="ttml / ttnn not importable")
@pytest.mark.requires_device
class TestMoeUngroupOpDevice:
    """Device tests for ttml.ops.moe.moe_ungroup_op."""

    def setup_method(self, method):
        # Per-test autograd isolation: any leftover graph from a previous
        # test would silently chain into this one's backward.
        ttml.autograd.AutoContext.get_instance().reset_graph()

    @pytest.mark.parametrize("shape", SHAPES)
    def test_forward_shape(self, shape):
        D, B, S, H, E, K, E_local = shape
        dispatched, metadata, scores, leids = _make_inputs(D, B, S, H, K, E, E_local, seed=0)
        num_cores = _device_grid_size()

        grouped_v, grouped_scores_v, plan, offsets, _plan_t, _gs_t = _build_group_outputs(
            dispatched, metadata, scores, leids, K, num_cores
        )

        expert_out = ttml.autograd.Tensor(grouped_v, False)
        grouped_scores = ttml.autograd.Tensor(grouped_scores_v, False)
        md = _to_device_tensor(metadata, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)
        le = _to_device_tensor(leids, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)

        ttnn.synchronize_device(_device())
        out = ttml.ops.moe.moe_ungroup_op(
            expert_out,
            grouped_scores,
            md,
            le,
            plan,
            offsets,
            int(E_local),
            int(K),
            int(D),
            int(B),
            int(S),
        )
        ttnn.synchronize_device(_device())

        assert list(out.shape()) == [D, B, S, H]

        ttml.autograd.AutoContext.get_instance().reset_graph()

    @pytest.mark.parametrize("shape", SHAPES)
    def test_forward_parity_with_kernel(self, shape):
        """Wrapper output must equal raw metal_ops.moe_ungroup output."""
        D, B, S, H, E, K, E_local = shape
        dispatched, metadata, scores, leids = _make_inputs(D, B, S, H, K, E, E_local, seed=1)
        num_cores = _device_grid_size()

        grouped_v, grouped_scores_v, plan, offsets, _plan_t, _gs_t = _build_group_outputs(
            dispatched, metadata, scores, leids, K, num_cores
        )

        # Wrapper
        expert_out = ttml.autograd.Tensor(grouped_v, False)
        grouped_scores = ttml.autograd.Tensor(grouped_scores_v, False)
        md = _to_device_tensor(metadata, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)
        le = _to_device_tensor(leids, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)

        ttnn.synchronize_device(_device())
        wrapped_out = ttml.ops.moe.moe_ungroup_op(
            expert_out,
            grouped_scores,
            md,
            le,
            plan,
            offsets,
            int(E_local),
            int(K),
            int(D),
            int(B),
            int(S),
        )
        ttnn.synchronize_device(_device())

        # Raw kernel
        raw_out = ttml.ops.metal_ops.moe_ungroup(
            grouped_v,
            plan,
            offsets,
            grouped_scores_v,
            int(E_local),
            int(D),
            int(B),
            int(S),
        )
        ttnn.synchronize_device(_device())

        wrapped_torch = ttnn.to_torch(wrapped_out.get_value()).float()
        raw_torch = ttnn.to_torch(raw_out).float()
        assert_with_pcc(wrapped_torch, raw_torch, pcc=0.9999)

        ttml.autograd.AutoContext.get_instance().reset_graph()

    @pytest.mark.parametrize("shape", SHAPES)
    def test_backward_d_expert_out(self, shape):
        """d(expert_out)[i, h] = grouped_scores[i] · grad_grouped[i, h].

        With grouped_scores = ones and d(ungrouped) = ones, every active
        row gets d(expert_out)[i, h] = 1.  Tail rows past offsets[-1]
        are by-design garbage; only the meaningful prefix is checked.
        """
        D, B, S, H, E, K, E_local = shape
        dispatched, metadata, scores, leids = _make_inputs(D, B, S, H, K, E, E_local, seed=2)
        num_cores = _device_grid_size()

        grouped_v, _gs_kernel, plan, offsets, plan_t, _gs_t = _build_group_outputs(
            dispatched, metadata, scores, leids, K, num_cores
        )
        T_cap = grouped_v.shape[2]

        gs_ones = ttnn.from_torch(
            torch.ones(1, 1, 1, T_cap, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=_device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        expert_out = ttml.autograd.Tensor(grouped_v, True)
        grouped_scores = ttml.autograd.Tensor(gs_ones, False)
        md = _to_device_tensor(metadata, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)
        le = _to_device_tensor(leids, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)

        ttnn.synchronize_device(_device())
        out = ttml.ops.moe.moe_ungroup_op(
            expert_out,
            grouped_scores,
            md,
            le,
            plan,
            offsets,
            int(E_local),
            int(K),
            int(D),
            int(B),
            int(S),
        )

        d_ungrouped = ttnn.from_torch(
            torch.ones(D, B, S, H, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=_device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out.set_grad(d_ungrouped)
        out.backward(False)
        ttnn.synchronize_device(_device())

        # Compare only the meaningful range [0, offsets[-1]).
        # Active rows must be 1, intra-stripe pads must be 0; tail rows
        # past offsets[-1] are by-design garbage (don't compare).
        plan_np = plan_t  # already numpy from kernel
        offsets_np = ttnn.to_torch(offsets).numpy().flatten()
        t_used = int(offsets_np[-1])
        active_mask = (plan_np[:t_used] != int(SENTINEL)).astype(np.float32)  # [t_used]
        expected = np.broadcast_to(active_mask[None, None, :, None], (1, 1, t_used, H)).copy()

        d_expert_out = ttnn.to_torch(expert_out.get_grad()).float()
        actual = d_expert_out[:, :, :t_used, :]
        assert_with_pcc(torch.from_numpy(expected), actual, pcc=0.99)

        ttml.autograd.AutoContext.get_instance().reset_graph()

    @pytest.mark.parametrize("shape", SHAPES)
    def test_backward_d_grouped_scores(self, shape):
        """d(grouped_scores)[i] = Σ_h expert_out[i,h] · grad_grouped[i,h].

        With expert_out = ones and d(ungrouped) = ones:
          grad_grouped[i, h] = 1   for active rows, 0 for pad rows in
                                   [0, offsets[-1])
          d(grouped_scores)[i] = H for active rows, 0 for pad rows.
        """
        D, B, S, H, E, K, E_local = shape
        dispatched, metadata, scores, leids = _make_inputs(D, B, S, H, K, E, E_local, seed=3)
        num_cores = _device_grid_size()

        _grouped_v, _gs_kernel, plan, offsets, plan_t, _gs_t = _build_group_outputs(
            dispatched, metadata, scores, leids, K, num_cores
        )
        T_cap = int(plan_t.size)

        # expert_out = ones; grouped_scores = arbitrary (we don't read it for
        # d(grouped_scores) — we only need expert_out values for Step C).
        ones_eo = ttnn.from_torch(
            torch.ones(1, 1, T_cap, H, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=_device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gs_arbitrary = ttnn.from_torch(
            torch.full((1, 1, 1, T_cap), 0.5, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=_device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        expert_out = ttml.autograd.Tensor(ones_eo, False)
        grouped_scores = ttml.autograd.Tensor(gs_arbitrary, True)
        md = _to_device_tensor(metadata, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)
        le = _to_device_tensor(leids, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)

        ttnn.synchronize_device(_device())
        out = ttml.ops.moe.moe_ungroup_op(
            expert_out,
            grouped_scores,
            md,
            le,
            plan,
            offsets,
            int(E_local),
            int(K),
            int(D),
            int(B),
            int(S),
        )

        # Set d(ungrouped) = ones manually (out is ROW_MAJOR, mean() needs TILE).
        d_ungrouped = ttnn.from_torch(
            torch.ones(D, B, S, H, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=_device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out.set_grad(d_ungrouped)
        out.backward(False)
        ttnn.synchronize_device(_device())

        # With d(ungrouped) = ones, expert_out = ones:
        #   grad_grouped[i, h] = 1   for active rows
        #                       = 0   for pad rows (within offsets[-1])
        #   d(grouped_scores)[i] = Σ_h 1·1 = H  for active rows, 0 for pad.
        # Tail rows past offsets[-1] are by-design garbage; don't compare.
        plan_np = plan_t  # already numpy from kernel
        offsets_np = ttnn.to_torch(offsets).numpy().flatten()
        t_used = int(offsets_np[-1])
        active_mask = (plan_np[:t_used] != int(SENTINEL)).astype(np.float32)
        expected = (active_mask * H).reshape(1, 1, 1, t_used)

        d_gs = ttnn.to_torch(grouped_scores.get_grad()).float()[:, :, :, :t_used]
        assert_with_pcc(torch.from_numpy(expected), d_gs, pcc=0.99)

        ttml.autograd.AutoContext.get_instance().reset_graph()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
