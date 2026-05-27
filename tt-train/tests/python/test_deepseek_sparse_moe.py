# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Forward + backward parity: SparseMoE vs the dense MoE composite.

Builds two MoE modules from the same config, copies weights from the
dense one into the sparse one, and checks that forward outputs and all
parameter gradients match within PCC bounds.

The dense MoE runs every expert over the full input + masks; the sparse
MoE routes tokens to experts via metal::moe_group / metal::moe_ungroup
and runs the FFN only on the per-expert slice. They must agree
numerically when E_local == num_experts (single-device sparse).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from tests.ttnn.utils_for_testing import assert_with_pcc  # noqa: E402

try:
    import ttnn
    import ttml
    from ttml.models.deepseek.moe import MoE
    from ttml.models.deepseek.moe_sparse import SparseMoE

    _AVAILABLE = True
except Exception:
    _AVAILABLE = False


SEED = 2026


class _Cfg:
    """Minimal MoE config covering only the fields MoE.__init__ reads."""

    def __init__(self, **kw):
        self.dim = kw.get("dim", 64)
        self.moe_inter_dim = kw.get("moe_inter_dim", 64)
        self.n_routed_experts = kw.get("n_routed_experts", 4)
        self.n_activated_experts = kw.get("n_activated_experts", 2)
        self.n_shared_experts = kw.get("n_shared_experts", 0)
        self.n_expert_groups = kw.get("n_expert_groups", 1)
        self.n_limited_groups = kw.get("n_limited_groups", 1)
        self.score_func = kw.get("score_func", "sigmoid")
        self.route_scale = kw.get("route_scale", 1.0)


def _device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _copy_param(src_param, dst_param):
    """Copy underlying ttnn tensor from src Parameter into dst Parameter."""
    dst_param.tensor.set_value(src_param.tensor.get_value())


def _copy_moe_weights(src: MoE, dst: MoE) -> None:
    """Mirror every Parameter from src into dst (gate, experts, shared)."""
    _copy_param(src.gate.weight, dst.gate.weight)
    for e in range(src.num_experts):
        _copy_param(src.experts[e].w1.weight, dst.experts[e].w1.weight)
        _copy_param(src.experts[e].w2.weight, dst.experts[e].w2.weight)
        _copy_param(src.experts[e].w3.weight, dst.experts[e].w3.weight)
    if src.shared_experts is not None:
        _copy_param(src.shared_experts.w1.weight, dst.shared_experts.w1.weight)
        _copy_param(src.shared_experts.w2.weight, dst.shared_experts.w2.weight)
        _copy_param(src.shared_experts.w3.weight, dst.shared_experts.w3.weight)


def _make_input(B: int, S: int, dim: int, *, seed: int = SEED) -> "ttml.autograd.Tensor":
    g = torch.Generator().manual_seed(seed)
    arr = torch.randn(B, 1, S, dim, generator=g).numpy().astype(np.float32)
    return ttml.autograd.Tensor.from_numpy(arr, layout=ttnn.Layout.TILE, new_type=ttnn.bfloat16)


def _grad_pcc(label, dense_param, sparse_param, *, pcc=0.95):
    """Compare gradients of a Parameter pair via PCC."""
    g_dense = ttnn.to_torch(dense_param.tensor.get_grad()).float()
    g_sparse = ttnn.to_torch(sparse_param.tensor.get_grad()).float()
    try:
        assert_with_pcc(g_dense, g_sparse, pcc=pcc)
    except AssertionError as exc:
        raise AssertionError(f"{label}: {exc}") from exc


# (B, S, dim, moe_inter_dim, E, K). "tiny" is fastest; the others come
# from the moe_group / moe_ungroup perf tables (see pr_description*.md).
SHAPES = [
    pytest.param((2, 32, 64, 64, 4, 2), id="tiny"),
    pytest.param((2, 128, 512, 128, 2, 2), id="perf-b2-s128-h512"),
    pytest.param((4, 256, 2048, 256, 4, 4), id="perf-b4-s256-h2048"),
]


@pytest.mark.skipif(not _AVAILABLE, reason="ttml / ttnn not importable")
@pytest.mark.requires_device
class TestSparseVsDenseMoE:
    def setup_method(self, method):
        ttml.autograd.AutoContext.get_instance().reset_graph()

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("score_func", ["sigmoid", "softmax"])
    def test_forward_parity(self, score_func, shape):
        B, S, dim, moe_inter_dim, E, K = shape
        cfg = _Cfg(
            dim=dim,
            moe_inter_dim=moe_inter_dim,
            n_routed_experts=E,
            n_activated_experts=K,
            n_shared_experts=0,
            score_func=score_func,
            route_scale=1.0,
        )
        dense = MoE(cfg)
        sparse = SparseMoE(cfg)
        _copy_moe_weights(dense, sparse)

        x_dense = _make_input(B=B, S=S, dim=cfg.dim, seed=SEED)
        x_sparse = _make_input(B=B, S=S, dim=cfg.dim, seed=SEED)

        ttnn.synchronize_device(_device())
        out_dense = dense(x_dense)
        out_sparse = sparse(x_sparse)
        ttnn.synchronize_device(_device())

        out_d = ttnn.to_torch(out_dense.get_value()).float()
        out_s = ttnn.to_torch(out_sparse.get_value()).float()
        assert_with_pcc(out_d, out_s, pcc=0.99)

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("score_func", ["sigmoid", "softmax"])
    def test_backward_parity(self, score_func, shape):
        B, S, dim, moe_inter_dim, E, K = shape
        cfg = _Cfg(
            dim=dim,
            moe_inter_dim=moe_inter_dim,
            n_routed_experts=E,
            n_activated_experts=K,
            n_shared_experts=0,
            score_func=score_func,
            route_scale=1.0,
        )
        dense = MoE(cfg)
        sparse = SparseMoE(cfg)
        _copy_moe_weights(dense, sparse)

        x_dense = _make_input(B=B, S=S, dim=cfg.dim, seed=SEED)
        x_dense.set_requires_grad(True)
        x_sparse = _make_input(B=B, S=S, dim=cfg.dim, seed=SEED)
        x_sparse.set_requires_grad(True)

        ttnn.synchronize_device(_device())
        out_dense = dense(x_dense)
        out_sparse = sparse(x_sparse)

        out_B, _, out_S, out_dim = list(out_dense.get_value().shape)
        # Random upstream gradient: forces every output element's
        # contribution to grads to be distinct (a constant grad masks bugs
        # that perfectly average out, e.g. wrong per-token routing).
        g = torch.Generator().manual_seed(SEED + 1)
        upstream_torch = torch.randn(out_B, 1, out_S, out_dim, generator=g, dtype=torch.float32)
        upstream = ttnn.from_torch(
            upstream_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=_device(),
        )
        # set_grad consumes the tensor; re-build it for the second set_grad
        # so dense and sparse get bit-identical bf16 upstream values.
        upstream2 = ttnn.from_torch(
            upstream_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=_device(),
        )
        out_dense.set_grad(upstream)
        out_sparse.set_grad(upstream2)
        out_dense.backward(False)
        out_sparse.backward(False)
        ttnn.synchronize_device(_device())

        _grad_pcc("gate.weight", dense.gate.weight, sparse.gate.weight, pcc=0.95)

        for e in range(cfg.n_routed_experts):
            _grad_pcc(f"experts[{e}].w1", dense.experts[e].w1.weight, sparse.experts[e].w1.weight, pcc=0.95)
            _grad_pcc(f"experts[{e}].w2", dense.experts[e].w2.weight, sparse.experts[e].w2.weight, pcc=0.95)
            _grad_pcc(f"experts[{e}].w3", dense.experts[e].w3.weight, sparse.experts[e].w3.weight, pcc=0.95)

        # Input gradient — end-to-end through everything (routing weights,
        # gather, group, FFN, ungroup, shared experts). gate.weight grad
        # parity (above) implicitly covers the scores.grad bottleneck:
        # below scores both paths share the same sigmoid/softmax + gate
        # ops, so matching scores.grad ⟹ matching gate.weight.grad.
        g_dense_x = ttnn.to_torch(x_dense.get_grad()).float()
        g_sparse_x = ttnn.to_torch(x_sparse.get_grad()).float()
        try:
            assert_with_pcc(g_dense_x, g_sparse_x, pcc=0.95)
        except AssertionError as exc:
            raise AssertionError(f"input gradient (x.grad): {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
