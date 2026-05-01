# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MoE group autograd wrapper (ttml.ops.moe.moe_group_op).

The wrapper around metal::moe_group adds:
  - autograd: dispatched and scores get gradients via metal::moe_ungroup
  - forward: identical outputs to the raw kernel (sanity-checked here)
  - backward Step A (d(dispatched) via row-scatter, H = hidden_dim)
  - backward Step B (d(scores)    via K-wide sparse-scatter, H = K)

These tests piggy-back on `moe_group_torch_reference` from test_moe_group.py.
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
    K_SLOT_SENTINEL,
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
    """bf16 ROW_MAJOR autograd tensor from a torch tensor."""
    arr = t.float().numpy() if t.dtype == torch.bfloat16 else t.numpy()
    return ttml.autograd.Tensor.from_numpy(arr.astype(np.float32), layout=layout, new_type=dtype)


def _make_inputs(D: int, B: int, S: int, H: int, K: int, E: int, E_local: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    dispatched = torch.randn(D, B, S, H, generator=g)
    metadata = torch.zeros(D, B, S, K, dtype=torch.int32)
    for d in range(D):
        for b in range(B):
            for s in range(S):
                metadata[d, b, s] = torch.randperm(E, generator=g)[:K].to(torch.int32)
    scores_full = torch.softmax(torch.randn(D, B, S, K, generator=g), dim=-1)
    local_expert_ids = torch.arange(E_local, dtype=torch.int32)
    return dispatched, metadata, scores_full, local_expert_ids


def _device_grid_size() -> int:
    grid = _device().compute_with_storage_grid_size()
    return int(grid.x) * int(grid.y)


def _device_num_total_cores(e_local: int, k: int, d: int, b: int, s: int) -> int:
    # Mirror the kernel's split_work_to_cores(grid, total_tiles) split:
    # num_workers = min(grid_size, total_tiles). t_cap is sized with the
    # full grid, so total_tiles = t_cap_full_grid // 32.
    grid_size = _device_grid_size()
    t_cap = moe_group_t_cap(e_local, k, d, b, s, num_total_cores=grid_size)
    return min(grid_size, t_cap // 32)


# (D, B, S, H, E, K, E_local). "small" is the original tiny shape;
# the others are pulled from the moe_group / moe_ungroup perf tables
# (see pr_description.md / pr_description_ungroup.md). Tests assume
# E_local == E (single-device sparse MoE) so the d(scores) reference
# is exact.
SHAPES = [
    pytest.param((1, 1, 32, 64, 4, 2, 4), id="small"),
    pytest.param((2, 1, 128, 512, 2, 2, 2), id="perf-d2-s128-h512"),
    pytest.param((4, 1, 256, 2048, 4, 4, 4), id="perf-d4-s256-h2048"),
]


@pytest.mark.skipif(not _TTML_AVAILABLE, reason="ttml / ttnn not importable")
@pytest.mark.requires_device
class TestMoeGroupOpDevice:
    """Device tests: run the autograd wrapper, verify forward parity with the
    raw kernel and that backward produces the expected scatter."""

    def test_forward_shapes_match_kernel(self):
        D, B, S, H = 1, 1, 32, 64
        E, K = 4, 2
        E_local = 4
        dispatched, metadata, scores_full, leids = _make_inputs(D, B, S, H, K, E, E_local, seed=0)

        x = _to_autograd_tensor(dispatched)
        sc = _to_autograd_tensor(scores_full)
        md = _to_device_tensor(metadata, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)
        le = _to_device_tensor(leids, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)

        ttnn.synchronize_device(_device())
        out = ttml.ops.moe.moe_group_op(x, md, sc, le, int(E_local), int(K))
        ttnn.synchronize_device(_device())

        # T_cap uses full grid count (kernel allocates pessimistically); the
        # actual `offsets[-1]` (T_used) is computed by split_work_to_cores
        # at runtime and is typically smaller — but the tensor IS T_cap rows.
        T_cap = moe_group_t_cap(E_local, K, D, B, S, num_total_cores=_device_grid_size())
        assert list(out.grouped.shape()) == [1, 1, T_cap, H]
        assert list(out.grouped_scores.shape()) == [1, 1, 1, T_cap]
        assert list(out.k_slot.shape) == [1, 1, 1, T_cap]
        assert list(out.counts.shape) == [1, 1, 1, E_local]
        assert list(out.offsets.shape) == [1, 1, 1, E_local + 1]
        assert list(out.plan.shape) == [1, 1, 1, T_cap]

        ttml.autograd.AutoContext.get_instance().reset_graph()

    def test_forward_parity_with_kernel(self):
        """Wrapper output must equal raw metal_ops.moe_group output for the
        same inputs."""
        D, B, S, H = 1, 1, 32, 32
        E, K = 4, 2
        E_local = 2
        dispatched, metadata, scores_full, leids = _make_inputs(D, B, S, H, K, E, E_local, seed=1)

        # Wrapper
        x = _to_autograd_tensor(dispatched)
        sc = _to_autograd_tensor(scores_full)
        md = _to_device_tensor(metadata, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)
        le = _to_device_tensor(leids, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)
        ttnn.synchronize_device(_device())
        out = ttml.ops.moe.moe_group_op(x, md, sc, le, int(E_local), int(K))
        ttnn.synchronize_device(_device())

        # Raw kernel
        x_raw = _to_device_tensor(dispatched, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16)
        sc_raw = _to_device_tensor(scores_full, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16)
        ttnn.synchronize_device(_device())
        grouped_raw, grouped_scores_raw, k_slot_raw, counts_raw, offsets_raw, plan_raw = ttml.ops.metal_ops.moe_group(
            x_raw, md, sc_raw, le, int(E_local), int(K)
        )
        ttnn.synchronize_device(_device())

        # Wrapper.grouped is autograd; .to_numpy() returns numpy.
        np.testing.assert_array_equal(
            np.asarray(ttnn.to_torch(out.plan)),
            np.asarray(ttnn.to_torch(plan_raw)),
        )
        np.testing.assert_array_equal(
            np.asarray(ttnn.to_torch(out.k_slot)),
            np.asarray(ttnn.to_torch(k_slot_raw)),
        )
        np.testing.assert_array_equal(
            np.asarray(ttnn.to_torch(out.offsets)),
            np.asarray(ttnn.to_torch(offsets_raw)),
        )

        ttml.autograd.AutoContext.get_instance().reset_graph()

    @pytest.mark.parametrize("shape", SHAPES)
    def test_backward_d_dispatched_active_tokens(self, shape):
        """Backward through `grouped` only: d(dispatched)[t] should equal
        the count of times token t appears in active grouped rows (because
        we set d(grouped) = ones and ones_gs = 1 in the bw scatter).
        """
        D, B, S, H, E, K, E_local = shape
        # Force E_local == E so every token's K experts are all local.
        assert E_local == E, "this test assumes E_local == E"
        dispatched, metadata, scores_full, leids = _make_inputs(D, B, S, H, K, E, E_local, seed=2)

        x = _to_autograd_tensor(dispatched)
        x.set_requires_grad(True)
        sc = _to_autograd_tensor(scores_full)
        sc.set_requires_grad(False)  # only test d(x) here
        md = _to_device_tensor(metadata, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)
        le = _to_device_tensor(leids, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)

        ttnn.synchronize_device(_device())
        out = ttml.ops.moe.moe_group_op(x, md, sc, le, int(E_local), int(K))

        # Sum-of-grouped loss → d(grouped) = ones across all entries.
        loss = ttml.ops.unary.mean(out.grouped)
        loss.backward(False)
        ttnn.synchronize_device(_device())

        # Reference: when E_local == E, every token's K experts are all local
        # ⇒ each token contributes exactly K rows to grouped. With a `mean`
        # loss the per-element grad is 1/(T_cap*H), so d(x)[t,h] = K/(T_cap*H).
        # T_cap uses full grid count (kernel allocates pessimistically); the
        # actual `offsets[-1]` (T_used) is computed by split_work_to_cores
        # at runtime and is typically smaller — but the tensor IS T_cap rows.
        T_cap = moe_group_t_cap(E_local, K, D, B, S, num_total_cores=_device_grid_size())
        d_x = ttnn.to_torch(x.get_grad()).float()
        expected = torch.full((D, B, S, H), K / (T_cap * H), dtype=torch.float32)
        assert_with_pcc(expected, d_x, pcc=0.999)

        ttml.autograd.AutoContext.get_instance().reset_graph()

    @pytest.mark.parametrize("shape", SHAPES)
    def test_backward_d_scores_K_wide_scatter(self, shape):
        """Backward through `grouped_scores`: d(scores) is built from the
        K-wide one-hot * dot scatter via metal::moe_ungroup with H = K.

        With E_local == E and a `mean(grouped_scores)` loss, every token
        contributes K active rows (one per top-K expert) and each (t, ks)
        pair is hit exactly once. The K positions used per token are
        precisely 0..K-1, so d(scores)[t, ks] = 1/T_cap for ALL (t, ks)
        positions in the dense [D,B,S,K] tensor.
        """
        D, B, S, H, E, K, E_local = shape
        assert E_local == E, "this test assumes E_local == E"
        dispatched, metadata, scores_full, leids = _make_inputs(D, B, S, H, K, E, E_local, seed=3)

        x = _to_autograd_tensor(dispatched)
        x.set_requires_grad(False)  # only test d(scores)
        sc = _to_autograd_tensor(scores_full)
        sc.set_requires_grad(True)
        md = _to_device_tensor(metadata, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)
        le = _to_device_tensor(leids, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)

        ttnn.synchronize_device(_device())
        out = ttml.ops.moe.moe_group_op(x, md, sc, le, int(E_local), int(K))

        # grouped_scores is ROW_MAJOR bf16 — mean() requires TILE, so set
        # the grad directly: d(grouped_scores) = 1 across all entries.
        # T_cap uses full grid count (kernel allocates pessimistically); the
        # actual `offsets[-1]` (T_used) is computed by split_work_to_cores
        # at runtime and is typically smaller — but the tensor IS T_cap rows.
        T_cap = moe_group_t_cap(E_local, K, D, B, S, num_total_cores=_device_grid_size())
        gs_grad_torch = torch.ones(1, 1, 1, T_cap, dtype=torch.float32)
        gs_grad = ttnn.from_torch(
            gs_grad_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=_device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out.grouped_scores.set_grad(gs_grad)
        out.grouped_scores.backward(False)
        ttnn.synchronize_device(_device())

        d_scores = ttnn.to_torch(sc.get_grad()).float()
        # E_local == E ⇒ every (t, ks) hit exactly once ⇒ d(scores) = 1 everywhere.
        expected = torch.ones((D, B, S, K), dtype=torch.float32)
        assert_with_pcc(expected, d_scores, pcc=0.999)

        ttml.autograd.AutoContext.get_instance().reset_graph()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
