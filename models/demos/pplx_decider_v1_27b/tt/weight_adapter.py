# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF snapshot tensors -> TT-ready ``LazyWeight`` bundles. Host-only, setup-time.

Snapshot keys have no ``model.`` prefix: decoder layers live under ``language_model.layers.N.``,
the embedding and final norm under ``language_model.``, and the 5120x255 readout in
``readout.safetensors`` (key ``weight``). The builders here take the *layer-local* dict
(prefix stripped, keys exactly as in ``Qwen3_5DecoderLayer.state_dict()``).

All HF->TT transforms live here: transposes to [in, out], zero-centred norm folding
(``1 + w``), q/gate de-interleaving, QKV packing and the RoPE head-dim permutation of q/k,
the GDN [z | b | a] packing, the tile-pair interleaved MLP gate/up weight for the fused SwiGLU
matmul, conv taps and the ``-exp(A_log)`` decay constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.common.modules.lazy_weight import LazyWeight
from models.demos.pplx_decider_v1_27b.tt.model_config import PplxDeciderArgs
from models.demos.pplx_decider_v1_27b.tt.optimizations import PrecisionPolicy
from models.demos.pplx_decider_v1_27b.tt.rope import rope_head_permutation

TILE = 32


@dataclass(frozen=True)
class NormWeights:
    weight: LazyWeight  # already (1 + w) for zero-centred norms, shape [1, 1, D]


@dataclass(frozen=True)
class MLPWeights:
    gate_up: LazyWeight  # [5120, 2 x 17408], tile-pair interleaved [gate_t0 | up_t0 | gate_t1 | up_t1 | ...]
    down: LazyWeight  # [17408, 5120]


@dataclass(frozen=True)
class AttentionWeights:
    qkv: LazyWeight  # [5120, q(6144) | k(1024) | v(1024)]
    gate: LazyWeight  # [5120, 6144], the output gate half of HF q_proj
    o_proj: LazyWeight  # [6144, 5120]
    q_norm: LazyWeight  # (1 + w) [1, 1, 256]
    k_norm: LazyWeight  # (1 + w) [1, 1, 256]


@dataclass(frozen=True)
class GatedDeltaNetWeights:
    in_qkv: LazyWeight  # [5120, 10240], the conv input
    in_zba: LazyWeight  # [5120, z(6144) | b(48->64) | a(48->64)]
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


def interleave_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """HF gate/up [out, in] -> TT [in, 2*out] with 32-column tiles alternating gate, up.

    This is the layout ``minimal_matmul(..., fuse_swiglu=True)`` reads: it emits
    ``silu(tile 2p) * tile 2p+1`` per pair p, so the output is ``silu(x @ gate.T) * (x @ up.T)``.
    """
    if gate.shape != up.shape or gate.shape[0] % TILE:
        raise ValueError(f"gate {tuple(gate.shape)} / up {tuple(up.shape)} must match and be tile aligned")
    hidden = gate.shape[1]
    pairs = torch.stack([gate.T.reshape(hidden, -1, TILE), up.T.reshape(hidden, -1, TILE)], dim=2)
    return pairs.reshape(hidden, -1)


def build_mlp_weights(state: dict[str, torch.Tensor], policy: PrecisionPolicy) -> MLPWeights:
    gate_up = interleave_gate_up(_require(state, "mlp.gate_proj.weight"), _require(state, "mlp.up_proj.weight"))
    return MLPWeights(
        gate_up=_lazy(gate_up, policy.weight_dtype("mlp_gate_up")),
        down=_linear(_require(state, "mlp.down_proj.weight"), policy.weight_dtype("mlp_down")),
    )


def build_attention_weights(
    state: dict[str, torch.Tensor], args: PplxDeciderArgs, policy: PrecisionPolicy
) -> AttentionWeights:
    # HF q_proj rows are [head][query(256) | gate(256)] (view(..., heads, 512).chunk(2)).
    qg = _require(state, "self_attn.q_proj.weight").reshape(
        args.num_attention_heads, 2, args.head_dim, args.hidden_size
    )
    # Same head-dim permutation on q, k and their norm weights so the fused full-width rotary op
    # sees the rotary pairs adjacent (see tt/rope.py). q.k is unchanged; v and the gate are not touched.
    perm = torch.tensor(rope_head_permutation(args.head_dim, args.rotary_dim))
    q = qg[:, 0][:, perm]
    k = _require(state, "self_attn.k_proj.weight").reshape(args.num_key_value_heads, args.head_dim, -1)[:, perm]
    qkv = torch.cat(
        [
            q.reshape(-1, args.hidden_size),
            k.reshape(-1, args.hidden_size),
            _require(state, "self_attn.v_proj.weight"),
        ],
        dim=0,
    )
    norm_dtype = getattr(ttnn, policy.norm_weight_dtype)

    def head_norm(name):
        return _lazy((_require(state, name).float() + 1.0)[perm].reshape(1, 1, -1), norm_dtype)

    return AttentionWeights(
        qkv=_linear(qkv, policy.weight_dtype("attention_qkvg")),
        gate=_linear(qg[:, 1].reshape(-1, args.hidden_size), policy.weight_dtype("attention_qkvg")),
        o_proj=_linear(_require(state, "self_attn.o_proj.weight"), policy.weight_dtype("attention_out")),
        q_norm=head_norm("self_attn.q_norm.weight"),
        k_norm=head_norm("self_attn.k_norm.weight"),
    )


def build_delta_weights(state: dict[str, torch.Tensor], policy: PrecisionPolicy) -> GatedDeltaNetWeights:
    def tile_padded(name):
        # Pad to whole tiles so every slice of a packed output starts on a tile column.
        w = _require(state, f"linear_attn.in_proj_{name}.weight")
        return torch.nn.functional.pad(w, (0, 0, 0, (-w.shape[0]) % TILE))

    conv = _require(state, "linear_attn.conv1d.weight")  # [10240, 1, 4]
    if conv.shape[-1] != 4:
        raise ValueError("The TT causal conv kernel expects kernel size 4")
    return GatedDeltaNetWeights(
        in_qkv=_linear(tile_padded("qkv"), policy.weight_dtype("delta_in")),
        in_zba=_linear(torch.cat([tile_padded(n) for n in ("z", "b", "a")], dim=0), policy.weight_dtype("delta_in")),
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


def build_readout_weight(weight: torch.Tensor, policy: PrecisionPolicy, *, pad_to: int | None = None) -> LazyWeight:
    """readout.safetensors ``weight`` [255, 5120] -> [5120, 255], or zero-padded to [5120, pad_to] columns."""
    if pad_to is not None:
        weight = torch.nn.functional.pad(weight, (0, 0, 0, pad_to - weight.shape[0]))
    return _linear(weight, policy.weight_dtype("readout"))
