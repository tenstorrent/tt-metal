# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF snapshot tensors -> TT-ready ``LazyWeight`` bundles. Host-only, setup-time.

Snapshot keys have no ``model.`` prefix: decoder layers live under ``language_model.layers.N.``,
the embedding and final norm under ``language_model.``, and the 5120x255 readout in
``readout.safetensors`` (key ``weight``). The builders here take the *layer-local* dict
(prefix stripped, keys exactly as in ``Qwen3_5DecoderLayer.state_dict()``).

All HF->TT transforms live here: transposes to [in, out], zero-centred norm folding
(``1 + w``), q/gate de-interleaving and QKVG packing, GDN in-projection packing, conv
taps and the ``-exp(A_log)`` decay constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.common.modules.lazy_weight import LazyWeight
from models.demos.pplx_decider_v1_27b.tt.model_config import PplxDeciderArgs
from models.demos.pplx_decider_v1_27b.tt.optimizations import PrecisionPolicy

TILE = 32


@dataclass(frozen=True)
class NormWeights:
    weight: LazyWeight  # already (1 + w) for zero-centred norms, shape [1, 1, D]


@dataclass(frozen=True)
class MLPWeights:
    gate: LazyWeight  # [5120, 17408]
    up: LazyWeight  # [5120, 17408]
    down: LazyWeight  # [17408, 5120]


@dataclass(frozen=True)
class AttentionWeights:
    qkvg: LazyWeight  # [5120, q(6144) | k(1024) | v(1024) | gate(6144)]
    o_proj: LazyWeight  # [6144, 5120]
    q_norm: LazyWeight  # (1 + w) [1, 1, 256]
    k_norm: LazyWeight  # (1 + w) [1, 1, 256]


@dataclass(frozen=True)
class GatedDeltaNetWeights:
    in_proj: LazyWeight  # [5120, qkv(10240) | z(6144) | b(48->64) | a(48->64)]
    out_proj: LazyWeight  # [6144, 5120]
    conv_taps: tuple[LazyWeight, ...]  # 4 x [1, 1, 10240], tap i multiplies x[t - 3 + i]
    a_neg: LazyWeight  # -exp(A_log) fp32 [1, 1, 48]
    dt_bias: LazyWeight  # fp32 [1, 1, 48]
    norm: LazyWeight  # gated RMSNorm weight (plain, not 1 + w) [128]


@dataclass(frozen=True)
class DecoderLayerWeights:
    kind: str
    input_norm: NormWeights
    post_norm: NormWeights
    mlp: MLPWeights
    attention: AttentionWeights | None = None
    delta: GatedDeltaNetWeights | None = None


def _lazy(tensor: torch.Tensor, dtype) -> LazyWeight:
    return LazyWeight(source=tensor.contiguous(), dtype=dtype)


def _linear(weight: torch.Tensor, dtype) -> LazyWeight:
    """HF Linear [out, in] -> TT [in, out]."""
    return _lazy(weight.T, dtype)


def _require(state: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in state:
        raise KeyError(f"Missing '{key}' in layer state dict (have {sorted(state)[:6]}...)")
    return state[key]


def zero_centred_norm(weight: torch.Tensor, policy: PrecisionPolicy) -> NormWeights:
    """Qwen3.5 RMSNorm multiplies by (1 + w); fold the +1 in fp32 before the device cast."""
    return NormWeights(weight=_lazy((weight.float() + 1.0).reshape(1, 1, -1), getattr(ttnn, policy.norm_weight_dtype)))


def build_mlp_weights(state: dict[str, torch.Tensor], policy: PrecisionPolicy) -> MLPWeights:
    return MLPWeights(
        gate=_linear(_require(state, "mlp.gate_proj.weight"), policy.weight_dtype("mlp_gate")),
        up=_linear(_require(state, "mlp.up_proj.weight"), policy.weight_dtype("mlp_up")),
        down=_linear(_require(state, "mlp.down_proj.weight"), policy.weight_dtype("mlp_down")),
    )


def build_attention_weights(
    state: dict[str, torch.Tensor], args: PplxDeciderArgs, policy: PrecisionPolicy
) -> AttentionWeights:
    # HF q_proj rows are [head][query(256) | gate(256)] (view(..., heads, 512).chunk(2)).
    qg = _require(state, "self_attn.q_proj.weight").reshape(
        args.num_attention_heads, 2, args.head_dim, args.hidden_size
    )
    packed = torch.cat(
        [
            qg[:, 0].reshape(-1, args.hidden_size),
            _require(state, "self_attn.k_proj.weight"),
            _require(state, "self_attn.v_proj.weight"),
            qg[:, 1].reshape(-1, args.hidden_size),
        ],
        dim=0,
    )
    norm_dtype = getattr(ttnn, policy.norm_weight_dtype)
    return AttentionWeights(
        qkvg=_linear(packed, policy.weight_dtype("attention_qkvg")),
        o_proj=_linear(_require(state, "self_attn.o_proj.weight"), policy.weight_dtype("attention_out")),
        q_norm=_lazy((_require(state, "self_attn.q_norm.weight").float() + 1.0).reshape(1, 1, -1), norm_dtype),
        k_norm=_lazy((_require(state, "self_attn.k_norm.weight").float() + 1.0).reshape(1, 1, -1), norm_dtype),
    )


def build_delta_weights(state: dict[str, torch.Tensor], policy: PrecisionPolicy) -> GatedDeltaNetWeights:
    pieces = []
    for name in ("qkv", "z", "b", "a"):
        w = _require(state, f"linear_attn.in_proj_{name}.weight")
        # Pad each piece to whole tiles so every slice of the packed output starts on a tile column.
        pieces.append(torch.nn.functional.pad(w, (0, 0, 0, (-w.shape[0]) % TILE)))
    conv = _require(state, "linear_attn.conv1d.weight")  # [10240, 1, 4]
    if conv.shape[-1] != 4:
        raise ValueError("The TT causal conv kernel expects kernel size 4")
    return GatedDeltaNetWeights(
        in_proj=_linear(torch.cat(pieces, dim=0), policy.weight_dtype("delta_in")),
        out_proj=_linear(_require(state, "linear_attn.out_proj.weight"), policy.weight_dtype("delta_out")),
        conv_taps=tuple(_lazy(conv[:, 0, i].reshape(1, 1, -1), ttnn.bfloat16) for i in range(4)),
        a_neg=_lazy(-_require(state, "linear_attn.A_log").float().exp().reshape(1, 1, -1), ttnn.float32),
        dt_bias=_lazy(_require(state, "linear_attn.dt_bias").float().reshape(1, 1, -1), ttnn.float32),
        norm=_lazy(_require(state, "linear_attn.norm.weight").reshape(-1), ttnn.bfloat16),
    )


def build_decoder_layer_weights(
    state: dict[str, torch.Tensor], args: PplxDeciderArgs, layer_idx: int, policy: PrecisionPolicy
) -> DecoderLayerWeights:
    kind = args.layer_kind(layer_idx)
    return DecoderLayerWeights(
        kind=kind,
        input_norm=zero_centred_norm(_require(state, "input_layernorm.weight"), policy),
        post_norm=zero_centred_norm(_require(state, "post_attention_layernorm.weight"), policy),
        mlp=build_mlp_weights(state, policy),
        attention=build_attention_weights(state, args, policy) if kind == "full_attention" else None,
        delta=build_delta_weights(state, policy) if kind == "linear_attention" else None,
    )


def build_embedding_weight(weight: torch.Tensor, policy: PrecisionPolicy) -> LazyWeight:
    """[vocab, 5120] ROW_MAJOR table for ``ttnn.embedding``."""
    return LazyWeight(source=weight, dtype=getattr(ttnn, policy.embedding_dtype), layout=ttnn.ROW_MAJOR_LAYOUT)


def build_readout_weight(weight: torch.Tensor, policy: PrecisionPolicy) -> LazyWeight:
    """readout.safetensors ``weight`` [255, 5120] -> [5120, 255]."""
    return _linear(weight, policy.weight_dtype("readout"))
