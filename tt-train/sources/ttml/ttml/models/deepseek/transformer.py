# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek transformer block, MLP, and normalization layers."""

from __future__ import annotations

import ttml
from ttml.modules import AbstractModuleBase, ColumnParallelLinear, LinearLayer, Parameter, RowParallelLinear


class RMSNormLayer(AbstractModuleBase):
    """Root Mean Square Layer Normalization."""

    def __init__(self, features: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.gamma = Parameter(ttml.init.ones()((1, 1, 1, features)))

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        return ttml.ops.rmsnorm.rmsnorm(x, self.gamma.tensor, self.epsilon)


class DeepSeekMLP(AbstractModuleBase):
    """SwiGLU feed-forward network: w2(silu(w1(x)) * w3(x))."""

    def __init__(self, dim: int, inter_dim: int, *, tp_axis_name: str | None = None) -> None:
        super().__init__()
        if tp_axis_name is not None:
            self.w1 = ColumnParallelLinear(
                dim,
                inter_dim,
                has_bias=False,
                gather_output=False,
                axis_name=tp_axis_name,
            )
            self.w3 = ColumnParallelLinear(
                dim,
                inter_dim,
                has_bias=False,
                gather_output=False,
                axis_name=tp_axis_name,
            )
            self.w2 = RowParallelLinear(
                inter_dim,
                dim,
                has_bias=False,
                input_is_parallel=True,
                axis_name=tp_axis_name,
            )
        else:
            self.w1 = LinearLayer(dim, inter_dim, has_bias=False)
            self.w3 = LinearLayer(dim, inter_dim, has_bias=False)
            self.w2 = LinearLayer(inter_dim, dim, has_bias=False)

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        return ttml.ops.swiglu.swiglu(
            x,
            self.w1.weight.tensor,
            self.w2.weight.tensor,
            self.w3.weight.tensor,
        )


def resolve_moe_ep_axis(config) -> str | None:
    """Resolve the mesh axis that ``sparse_ep`` partitions experts across.

    Returns ``"tp"`` under full-model TP, else ``config.moe_axis_name`` when it
    names a real mesh axis of size > 1, else ``None`` (no usable EP axis).
    """
    import ttml as _ttml

    if bool(getattr(config, "use_tp", False)):
        return "tp"
    axis_name = getattr(config, "moe_axis_name", None)
    mesh = _ttml.maybe_mesh()
    if axis_name is not None and mesh is not None and mesh.has_axis(axis_name) and mesh.axis_size(axis_name) > 1:
        return axis_name
    return None


def build_moe_ffn(config):
    """Build the MoE FFN implementation selected by ``config.moe_type``.

    ``dense`` is the reference / cross-check path. ``sparse_ep`` degenerates to
    ``SparseMoEEP`` at EP size 1 when there is no usable EP axis, so
    there is no separate ``sparse`` mode. Exposed as a module-level helper so
    the dispatch — in particular the EP=1 fallback — is testable without
    building a whole block.
    """
    # Lazy imports to avoid circular dependency (moe imports RMSNormLayer from here)
    from .moe import MoE
    from .moe_sparse_ep import SparseMoEEP

    moe_type = str(getattr(config, "moe_type", "sparse_ep")).lower()
    if moe_type == "dense":
        return MoE(config)
    if moe_type != "sparse_ep":
        raise ValueError(
            f"DeepSeekBlock: unknown moe_type={moe_type!r}; expected one of "
            f"'dense', 'sparse_ep' (from DeepSeekConfig.moe_type)"
        )

    # SparseMoEEP covers both cases: with no usable EP axis it runs at EP size 1,
    # owning every expert and skipping the EP collectives.
    return SparseMoEEP(config, axis_name=resolve_moe_ep_axis(config))


class DeepSeekBlock(AbstractModuleBase):
    """Pre-norm residual transformer block.

    First n_dense_layers use dense MLP; remaining layers use MoE.
    """

    def __init__(self, layer_id: int, config, rope_params) -> None:
        # Lazy imports to avoid circular dependency (mla/moe import RMSNormLayer from here)
        from .mla import MultiHeadLatentAttention

        super().__init__()
        self.attn = MultiHeadLatentAttention(config, rope_params)
        use_tp = bool(getattr(config, "use_tp", False))
        if layer_id < config.n_dense_layers:
            self.ffn = DeepSeekMLP(config.dim, config.inter_dim, tp_axis_name="tp" if use_tp else None)
        else:
            self.ffn = build_moe_ffn(config)
            self.ffn._debug_layer_id = layer_id
        self.attn_norm = RMSNormLayer(config.dim)
        self.ffn_norm = RMSNormLayer(config.dim)

    def forward(self, x: ttml.autograd.Tensor, mask: ttml.autograd.Tensor = None) -> ttml.autograd.Tensor:
        # `mask` is accepted (and unused) only to satisfy the shared block(input, mask)
        # contract used by memory_efficient_runner. MLA is causal-only and generates its
        # causal mask on chip, so nothing is forwarded to attention.
        x = ttml.ops.binary.add(x, self.attn(self.attn_norm(x)))
        x = ttml.ops.binary.add(x, self.ffn(self.ffn_norm(x)))
        return x
