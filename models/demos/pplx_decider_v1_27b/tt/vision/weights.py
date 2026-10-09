# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Snapshot ``visual.*`` tensors -> TT-ready ``LazyWeight`` bundles. Host-only, setup-time.

The builders take the vision-local dict (prefix ``visual.`` stripped, keys exactly as in
``Qwen3_5VisionModel.state_dict()``) and check it strictly: every expected key present, nothing
extra (333 tensors). All weights stay BF16 (person decision for the vision tower).

HF -> TT transforms, all here:
- ``Linear`` weights [out, in] -> [in, out]; biases -> [1, out];
- Conv3d patch embed (kernel == stride == (2, 16, 16)) -> one [1536, 1152] matmul weight
  (pixel rows are flattened (C, T, H, W) exactly as HF ``view(-1, 3, 2, 16, 16)`` reads them);
- attention head dim 72 -> 96 (three tiles): q/k/v weight columns and biases zero-padded per head;
  q/k head dims permuted with ``rope_head_permutation(72, 72)`` (neox pair (j, j+36) -> (2j, 2j+1))
  so one ``rotary_embedding_llama`` rotates them (same scheme as the text model, see ../rope.py);
  proj input rows zero-padded per head (v is not permuted, so proj rows keep HF order);
- MLP intermediate 4304 -> 4320 (whole tiles): zero fc1 columns and bias, zero fc2 rows
  (``gelu_tanh(0) = 0``, so the padding contributes exactly nothing);
- LayerNorm weight / bias -> ROW_MAJOR [1, 1, D/32, 32] (the ``ttnn.layer_norm`` gamma/beta layout).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.common.modules.lazy_weight import LazyWeight
from models.demos.pplx_decider_v1_27b.tt.optimizations import VisionPrecisionPolicy
from models.demos.pplx_decider_v1_27b.tt.rope import rope_head_permutation
from models.demos.pplx_decider_v1_27b.tt.vision.config import TILE, PplxVisionArgs

VISION_PREFIX = "visual."


@dataclass(frozen=True)
class LinearWeights:
    weight: LazyWeight  # [in, out]
    bias: LazyWeight  # [1, out]


@dataclass(frozen=True)
class LayerNormWeights:
    weight: LazyWeight  # ROW_MAJOR [1, 1, D/32, 32]
    bias: LazyWeight


@dataclass(frozen=True)
class VisionAttentionWeights:
    qkv: LinearWeights  # [1152, q(16x96) | k(16x96) | v(16x96)]; q/k head dims rope-permuted
    proj: LinearWeights  # [16x96, 1152], zero rows on the padded head dims


@dataclass(frozen=True)
class VisionMLPWeights:
    fc1: LinearWeights  # [1152, 4320]
    fc2: LinearWeights  # [4320, 1152]


@dataclass(frozen=True)
class VisionBlockWeights:
    norm1: LayerNormWeights
    norm2: LayerNormWeights
    attention: VisionAttentionWeights
    mlp: VisionMLPWeights


@dataclass(frozen=True)
class PatchMergerWeights:
    norm: LayerNormWeights  # per patch, before the 2x2 grouping (use_postshuffle_norm=False)
    fc1: LinearWeights  # [4608, 4608]
    fc2: LinearWeights  # [4608, 5120]


@dataclass(frozen=True)
class VisionTowerWeights:
    patch_embed: LinearWeights  # [1536, 1152]
    pos_embed: torch.Tensor  # host BF16 [2304, 1152]: the per-image bilinear interpolation runs on host
    blocks: tuple[VisionBlockWeights, ...]
    merger: PatchMergerWeights


def expected_vision_keys(args: PplxVisionArgs) -> set[str]:
    keys = {
        "patch_embed.proj.weight",
        "patch_embed.proj.bias",
        "pos_embed.weight",
        "merger.norm.weight",
        "merger.norm.bias",
        "merger.linear_fc1.weight",
        "merger.linear_fc1.bias",
        "merger.linear_fc2.weight",
        "merger.linear_fc2.bias",
    }
    for i in range(args.depth):
        for name in ("norm1", "norm2", "attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"):
            keys |= {f"blocks.{i}.{name}.weight", f"blocks.{i}.{name}.bias"}
    return keys


def check_vision_state(state: dict[str, torch.Tensor], args: PplxVisionArgs) -> None:
    """Strict load: exactly the HF ``Qwen3_5VisionModel`` keys, no more, no less."""
    expected = expected_vision_keys(args)
    missing, unexpected = sorted(expected - set(state)), sorted(set(state) - expected)
    if missing or unexpected:
        raise KeyError(f"vision state dict mismatch: missing {missing[:6]}, unexpected {unexpected[:6]}")


def _lazy(tensor: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT) -> LazyWeight:
    return LazyWeight(source=tensor.contiguous(), dtype=dtype, layout=layout)


def _linear(weight: torch.Tensor, bias: torch.Tensor, role: str, policy: VisionPrecisionPolicy) -> LinearWeights:
    """HF Linear [out, in] (+ bias [out]) -> TT [in, out] and [1, out]."""
    return LinearWeights(
        weight=_lazy(weight.T, policy.weight_dtype(role)),
        bias=_lazy(bias.reshape(1, -1), getattr(ttnn, policy.bias_dtype)),
    )


def _layer_norm(weight: torch.Tensor, bias: torch.Tensor, policy: VisionPrecisionPolicy) -> LayerNormWeights:
    dtype = getattr(ttnn, policy.norm_weight_dtype)
    shape = (1, 1, weight.numel() // TILE, TILE)
    return LayerNormWeights(
        weight=_lazy(weight.reshape(shape), dtype, ttnn.ROW_MAJOR_LAYOUT),
        bias=_lazy(bias.reshape(shape), dtype, ttnn.ROW_MAJOR_LAYOUT),
    )


def _pad_rows(t: torch.Tensor, rows: int) -> torch.Tensor:
    return torch.nn.functional.pad(t, (0, 0) * (t.dim() - 1) + (0, rows - t.shape[0]))


def pad_heads(t: torch.Tensor, args: PplxVisionArgs, *, permute: bool) -> torch.Tensor:
    """[heads*72, ...] -> [heads*96, ...]: per-head optional rope permutation, then zero pad 72 -> 96."""
    heads = t.reshape(args.num_heads, args.head_dim, *t.shape[1:])
    if permute:
        heads = heads[:, rope_head_permutation(args.head_dim, args.rotary_dim)]
    pad = (0, 0) * (t.dim() - 1) + (0, args.padded_head_dim - args.head_dim)
    return torch.nn.functional.pad(heads, pad).reshape(args.num_heads * args.padded_head_dim, *t.shape[1:])


def build_attention_weights(
    state: dict[str, torch.Tensor], prefix: str, args: PplxVisionArgs, policy: VisionPrecisionPolicy
) -> VisionAttentionWeights:
    qkv_w = state[f"{prefix}qkv.weight"].reshape(3, args.hidden_size, args.hidden_size)  # rows q | k | v
    qkv_b = state[f"{prefix}qkv.bias"].reshape(3, args.hidden_size)
    w = torch.cat([pad_heads(qkv_w[i], args, permute=i < 2) for i in range(3)], dim=0)
    b = torch.cat([pad_heads(qkv_b[i], args, permute=i < 2) for i in range(3)], dim=0)
    proj_w = state[f"{prefix}proj.weight"]  # [out, in = heads*72]
    proj_w = pad_heads(proj_w.T, args, permute=False).T  # zero input rows for the padded head dims
    return VisionAttentionWeights(
        qkv=_linear(w, b, "vision_qkv", policy),
        proj=_linear(proj_w, state[f"{prefix}proj.bias"], "vision_proj", policy),
    )


def build_mlp_weights(
    state: dict[str, torch.Tensor], prefix: str, args: PplxVisionArgs, policy: VisionPrecisionPolicy
) -> VisionMLPWeights:
    inter = args.padded_intermediate_size
    fc1_w = _pad_rows(state[f"{prefix}linear_fc1.weight"], inter)
    fc1_b = _pad_rows(state[f"{prefix}linear_fc1.bias"], inter)
    fc2_w = _pad_rows(state[f"{prefix}linear_fc2.weight"].T, inter).T
    return VisionMLPWeights(
        fc1=_linear(fc1_w, fc1_b, "vision_fc1", policy),
        fc2=_linear(fc2_w, state[f"{prefix}linear_fc2.bias"], "vision_fc2", policy),
    )


def build_block_weights(
    state: dict[str, torch.Tensor], idx: int, args: PplxVisionArgs, policy: VisionPrecisionPolicy
) -> VisionBlockWeights:
    p = f"blocks.{idx}."
    return VisionBlockWeights(
        norm1=_layer_norm(state[f"{p}norm1.weight"], state[f"{p}norm1.bias"], policy),
        norm2=_layer_norm(state[f"{p}norm2.weight"], state[f"{p}norm2.bias"], policy),
        attention=build_attention_weights(state, f"{p}attn.", args, policy),
        mlp=build_mlp_weights(state, f"{p}mlp.", args, policy),
    )


def build_patch_embed_weights(
    state: dict[str, torch.Tensor], args: PplxVisionArgs, policy: VisionPrecisionPolicy
) -> LinearWeights:
    w = state["patch_embed.proj.weight"]  # [1152, 3, 2, 16, 16]
    return _linear(w.reshape(args.hidden_size, args.patch_dim), state["patch_embed.proj.bias"], "vision_patch", policy)


def build_merger_weights(state: dict[str, torch.Tensor], policy: VisionPrecisionPolicy) -> PatchMergerWeights:
    return PatchMergerWeights(
        norm=_layer_norm(state["merger.norm.weight"], state["merger.norm.bias"], policy),
        fc1=_linear(state["merger.linear_fc1.weight"], state["merger.linear_fc1.bias"], "merger_fc1", policy),
        fc2=_linear(state["merger.linear_fc2.weight"], state["merger.linear_fc2.bias"], "merger_fc2", policy),
    )


def build_vision_weights(
    state: dict[str, torch.Tensor], args: PplxVisionArgs, policy: VisionPrecisionPolicy
) -> VisionTowerWeights:
    check_vision_state(state, args)
    bad = {k: v.dtype for k, v in state.items() if v.dtype != torch.bfloat16}
    if bad:
        raise ValueError(f"vision snapshot tensors are expected in BF16, got {bad}")
    return VisionTowerWeights(
        patch_embed=build_patch_embed_weights(state, args, policy),
        pos_embed=state["pos_embed.weight"].contiguous(),
        blocks=tuple(build_block_weights(state, i, args, policy) for i in range(args.depth)),
        merger=build_merger_weights(state, policy),
    )
