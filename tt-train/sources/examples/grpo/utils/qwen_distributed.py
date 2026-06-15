# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Self-contained tensor-parallel (TP) Qwen3 model + HuggingFace weight loading.

This module is vendored into the GRPO example so that
:mod:`utils.qwen_completer` can drive a tensor-parallel Qwen3 without importing
from ``examples/qwen3`` (whose ``utils``/``model_qwen3`` bare imports collide
with the GRPO example's own ``utils`` namespace package). It is adapted from
``examples/qwen3/model_qwen3_distributed.py`` and
``examples/qwen3/utils/{param_utils,tensor_utils,distributed_ops}.py`` and
depends only on ``ttml`` core (plus ``ttml.models.qwen3`` for the shared
config / RMSNorm / ConcatLastDim).

TP strategy (Megatron-LM):
  - Attention Q/K/V: ColumnParallel (shard output heads)
  - Attention O:     RowParallel (shard input, all-reduce)
  - MLP gate/up:     ColumnParallel; MLP down: RowParallel
  - Embedding (tied): VocabParallelEmbedding; (untied): sharded + all-gather
  - LM head:         ColumnParallel, vocab-sharded output (pair with
                     ``vocab_parallel_cross_entropy_loss``)
  - Norms:           Replicated

The 2D device mesh is ``[dp_size, tp_size]`` (mesh dim 0 = DP, mesh dim 1 = TP);
all TP collectives use ``shard_dim`` (== 1).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import ttnn
import ttml
from ttml.modules import AbstractModuleBase, ModuleList, Parameter
from ttml.models.qwen3 import Qwen3Config, Qwen3RMSNorm, ConcatLastDim


TILE_SIZE = 32


# ---------------------------------------------------------------------------
# Tensor / sharding helpers (adapted from examples/qwen3/utils/tensor_utils.py)
# ---------------------------------------------------------------------------


def _device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def get_tp_size() -> int:
    """Number of tensor-parallel devices from the active parallelism context."""
    pctx = ttml.autograd.AutoContext.get_instance().get_parallelism_context()
    return pctx.get_tp_size()


def _tile_pad(dim: int) -> int:
    return ((dim + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


def _empty_on_device(shape):
    """Allocate an uninitialized bf16 TILE tensor on device (last 2 dims tile-padded).

    Used during construction; the real values are loaded from HF immediately
    afterwards via :func:`load_weights_from_hf_distributed`.
    """
    padded = list(shape)
    padded[-2] = _tile_pad(padded[-2])
    padded[-1] = _tile_pad(padded[-1])
    ttnn_tensor = ttnn.empty(padded, ttnn.bfloat16, ttnn.TILE_LAYOUT, _device())
    return ttml.autograd.create_tensor(ttnn_tensor)


def _make_sharded_empty(shape, shard_dim_tensor: int):
    """Per-device empty shard of a full ``shape`` sharded along ``shard_dim_tensor``."""
    per_device = list(shape)
    per_device[shard_dim_tensor] //= get_tp_size()
    return _empty_on_device(per_device)


def _make_replicated_empty(shape):
    return _empty_on_device(shape)


# ---------------------------------------------------------------------------
# Linear wrapper + parallel linears
# ---------------------------------------------------------------------------


def _linear(x, weight, bias=None):
    return ttml.ops.linear.linear(x, weight, bias)


class ColumnParallelLinear(AbstractModuleBase):
    """Shards output features across TP devices. Output stays sharded on dim 3."""

    def __init__(self, in_features, out_features, has_bias=False, shard_dim=None):
        super().__init__()
        self.shard_dim = shard_dim
        self.weight = Parameter(_make_sharded_empty((1, 1, out_features, in_features), 2))
        if has_bias:
            self.col_bias = Parameter(_make_sharded_empty((1, 1, 1, out_features), 3))
        else:
            self.col_bias = None

    def forward(self, x):
        x = ttml.ops.distributed.broadcast(x, self.shard_dim)
        bias_t = self.col_bias.tensor if self.col_bias is not None else None
        return _linear(x, self.weight.tensor, bias_t)


class RowParallelLinear(AbstractModuleBase):
    """Shards input features across TP devices; all-reduces the output."""

    def __init__(self, in_features, out_features, has_bias=False, input_is_parallel=False, shard_dim=None):
        super().__init__()
        self.input_is_parallel = input_is_parallel
        self.shard_dim = shard_dim
        self.weight = Parameter(_make_sharded_empty((1, 1, out_features, in_features), 3))
        if has_bias:
            self.row_bias = Parameter(_make_replicated_empty((1, 1, 1, out_features)))
        else:
            self.row_bias = None

    def forward(self, x):
        if not self.input_is_parallel:
            x = ttml.ops.distributed.scatter(x, 3, self.shard_dim)
        x = _linear(x, self.weight.tensor, None)
        x = ttml.ops.distributed.all_reduce(x, self.input_is_parallel, self.shard_dim)
        if self.row_bias is not None:
            x = ttml.ops.binary.add(x, self.row_bias.tensor)
        return x


# ---------------------------------------------------------------------------
# Vocab-parallel embedding (Megatron-LM style, for tied weights)
# ---------------------------------------------------------------------------


class _BroadcastMul(ttml.autograd.Function):
    """``[B,1,T,H] * [B,1,T,1]`` treating the mask as a non-differentiable constant."""

    @staticmethod
    def forward(ctx, h, mask_val):
        ctx.mask_val = mask_val
        return ttnn.multiply(h.get_value(), mask_val)

    @staticmethod
    def backward(ctx, grad_output):
        return ttnn.multiply(grad_output, ctx.mask_val)


def _vocab_parallel_embedding(input_ids_np, sharded_weight, vocab_size, shard_dim):
    """Megatron-LM VocabParallelEmbedding for tied weights (local lookup -> mask -> all-reduce)."""
    tp_size = get_tp_size()
    local_V = int(sharded_weight.shape()[2])

    ids = input_ids_np.reshape(input_ids_np.shape[0], -1)
    B, T = ids.shape

    all_local_ids = np.zeros((tp_size * B, 1, 1, T), dtype=np.uint32)
    all_valid_mask = np.zeros((tp_size * B, 1, T, 1), dtype=np.float32)

    for k in range(tp_size):
        offset = k * local_V
        local = ids.astype(np.int64) - offset
        valid = (local >= 0) & (local < local_V)
        local = np.clip(local, 0, local_V - 1).astype(np.uint32)
        all_local_ids[k * B : (k + 1) * B, 0, 0, :] = local
        all_valid_mask[k * B : (k + 1) * B, 0, :, 0] = valid.astype(np.float32)

    shard_mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(_device(), 0, shard_dim)
    local_ids_t = ttml.autograd.Tensor.from_numpy(
        all_local_ids, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, shard_mapper
    )
    valid_mask_t = ttml.autograd.Tensor.from_numpy(all_valid_mask, ttnn.Layout.TILE, ttnn.bfloat16, shard_mapper)

    h = ttml.ops.embedding.embedding(local_ids_t, sharded_weight)
    h = _BroadcastMul.apply(h, valid_mask_t.get_value())
    h = ttml.ops.distributed.all_reduce(h, True, shard_dim)
    return h


# ---------------------------------------------------------------------------
# Distributed Qwen3 modules
# ---------------------------------------------------------------------------


class DistributedQwen3Attention(AbstractModuleBase):
    def __init__(self, config: Qwen3Config, layer_idx: int, shard_dim: Optional[int] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.shard_dim = shard_dim

        tp = get_tp_size()
        assert self.num_heads % tp == 0, f"num_heads {self.num_heads} not divisible by tp {tp}"
        assert self.num_kv_heads % tp == 0, f"num_kv_heads {self.num_kv_heads} not divisible by tp {tp}"
        self.num_local_heads = self.num_heads // tp
        self.num_local_kv_heads = self.num_kv_heads // tp

        q_out = self.num_heads * self.head_dim
        kv_out = self.num_kv_heads * self.head_dim

        self.q_proj = ColumnParallelLinear(self.hidden_size, q_out, has_bias=config.attention_bias, shard_dim=shard_dim)
        self.k_proj = ColumnParallelLinear(
            self.hidden_size, kv_out, has_bias=config.attention_bias, shard_dim=shard_dim
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size, kv_out, has_bias=config.attention_bias, shard_dim=shard_dim
        )
        self.o_proj = RowParallelLinear(
            q_out, self.hidden_size, has_bias=config.attention_bias, input_is_parallel=True, shard_dim=shard_dim
        )

        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        rope_scaling = ttml.ops.rope.RopeScalingParams()
        rs = config.rope_scaling
        if rs.scaling_factor != 0.0 and rs.original_context_length != 0:
            rope_scaling.original_context_length = rs.original_context_length
            rope_scaling.scaling_factor = rs.scaling_factor
            rope_scaling.high_freq_factor = rs.high_freq_factor
            rope_scaling.low_freq_factor = rs.low_freq_factor

        self.rope_params = ttml.ops.rope.build_rope_params(
            sequence_length=config.max_position_embeddings,
            head_dim=self.head_dim,
            theta=config.rope_theta,
            rope_scaling_params=rope_scaling,
        )

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, position_offset=0):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q_shape, k_shape = q.shape(), k.shape()
        B, S = q_shape[0], q_shape[2]

        q = ttml.ops.reshape.reshape(q, [B, 1, S * self.num_local_heads, self.head_dim])
        k = ttml.ops.reshape.reshape(k, [B, 1, S * self.num_local_kv_heads, self.head_dim])
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = ttml.ops.reshape.reshape(q, q_shape)
        k = ttml.ops.reshape.reshape(k, k_shape)

        kvs = ConcatLastDim.apply(k, v)
        query_heads, key_heads, value_heads = ttml.ops.multi_head_utils.grouped_heads_creation(
            q, kvs, self.num_local_heads, self.num_local_kv_heads
        )

        query_heads = ttml.ops.rope.rope(query_heads, self.rope_params, position_offset)
        key_heads = ttml.ops.rope.rope(key_heads, self.rope_params, position_offset)

        if past_key_values is not None:
            key_heads, value_heads = past_key_values.update(self.layer_idx, key_heads, value_heads)

        q_seq = query_heads.shape()[2]
        k_seq = key_heads.shape()[2]
        sdpa_fn = (
            ttml.ops.attention.scaled_dot_product_attention
            if q_seq == k_seq
            else ttml.ops.attention.scaled_dot_product_attention_composite
        )
        attn = sdpa_fn(query_heads, key_heads, value_heads, attention_mask)
        return self.o_proj(ttml.ops.multi_head_utils.heads_fusion(attn))


class DistributedQwen3MLP(AbstractModuleBase):
    def __init__(self, config: Qwen3Config, shard_dim=None):
        super().__init__()
        h, inter = config.hidden_size, config.intermediate_size
        self.gate_proj = ColumnParallelLinear(h, inter, shard_dim=shard_dim)
        self.up_proj = ColumnParallelLinear(h, inter, shard_dim=shard_dim)
        self.down_proj = RowParallelLinear(inter, h, input_is_parallel=True, shard_dim=shard_dim)

    def forward(self, x):
        gate = ttml.ops.unary.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(ttml.ops.binary.mul(gate, up))


class DistributedQwen3DecoderLayer(AbstractModuleBase):
    def __init__(self, config: Qwen3Config, layer_idx: int, shard_dim=None):
        super().__init__()
        self.self_attn = DistributedQwen3Attention(config, layer_idx, shard_dim)
        self.mlp = DistributedQwen3MLP(config, shard_dim)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, position_offset=0):
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states), attention_mask, past_key_values, position_offset=position_offset
        )
        hidden_states = ttml.ops.binary.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = ttml.ops.binary.add(residual, hidden_states)
        return hidden_states


class DistributedQwen3Model(AbstractModuleBase):
    def __init__(self, config: Qwen3Config, shard_dim=None, tied_embed_weight=None, use_checkpoint=False):
        super().__init__()
        self.config = config
        self.shard_dim = shard_dim
        self.tied_embed_weight = tied_embed_weight
        self.use_checkpoint = use_checkpoint
        vocab_tiled = ((config.vocab_size + 31) // 32) * 32
        if tied_embed_weight is None:
            self.embed_tokens = Parameter(_make_sharded_empty((1, 1, vocab_tiled, config.hidden_size), 3))
        self.layers = ModuleList(
            [DistributedQwen3DecoderLayer(config, i, shard_dim) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, input_ids_np=None):
        if self.tied_embed_weight is not None:
            h = _vocab_parallel_embedding(input_ids_np, self.tied_embed_weight, self.config.vocab_size, self.shard_dim)
        else:
            h = ttml.ops.embedding.embedding(input_ids, self.embed_tokens.tensor)
            h = ttml.ops.distributed.all_gather(h, 3, self.shard_dim, ttml.ops.distributed.GradOutputType.REPLICATED)
        position_offset = 0
        if past_key_values is not None:
            position_offset = past_key_values.get_seq_length()
        for layer in self.layers:
            if self.use_checkpoint:
                # Gradient checkpointing: disable grad on the forward and recompute
                # activations during backward. Cuts activation memory from
                # O(num_layers) to O(1), which is essential for 32B forward+backward.
                h = ttml.models.memory_efficient_runner(layer, h, attention_mask, past_key_values, position_offset)
            else:
                h = layer(h, attention_mask, past_key_values, position_offset=position_offset)
        return self.norm(h)


class DistributedQwen3ForCausalLM(AbstractModuleBase):
    """TP Qwen3 with a ColumnParallel (vocab-sharded) LM head.

    Pair the logits with ``ttml.ops.distributed.vocab_parallel_cross_entropy_loss``;
    callers wanting full-vocab logits (for sampling) must all-gather along dim 3.
    """

    def __init__(self, config: Qwen3Config, tie_word_embeddings=False, shard_dim=None, use_checkpoint=False):
        super().__init__()
        self.create_name("DistributedQwen3ForCausalLM")
        self.config = config
        self.tie_word_embeddings = tie_word_embeddings
        self.shard_dim = shard_dim

        vocab_tiled = ((config.vocab_size + 31) // 32) * 32
        lm_vocab = vocab_tiled if tie_word_embeddings else config.vocab_size
        self.lm_head = ColumnParallelLinear(config.hidden_size, lm_vocab, has_bias=False, shard_dim=shard_dim)

        self.model = DistributedQwen3Model(
            config,
            shard_dim,
            tied_embed_weight=(self.lm_head.weight.tensor if tie_word_embeddings else None),
            use_checkpoint=use_checkpoint,
        )

    def forward(self, input_ids, attention_mask=None, past_key_values=None, input_ids_np=None):
        h = self.model(input_ids, attention_mask, past_key_values, input_ids_np=input_ids_np)
        return self.lm_head(h)


# ---------------------------------------------------------------------------
# HuggingFace -> ttml weight permutations + name mapping
# (adapted from examples/qwen3/utils/param_utils.py)
# ---------------------------------------------------------------------------


def unpermute_proj_rows(weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    """HF ``[real_half, imag_half]`` row order -> ttml interleaved ``[r0,i0,r1,i1,...]``."""
    if weight.dim() == 1:
        total = weight.shape[0]
        D = total // num_heads
        half = D // 2
        w = weight.view(num_heads, D)
        interleaved = torch.stack([w[:, :half], w[:, half:]], dim=2)
        return interleaved.reshape(total)
    elif weight.dim() == 2:
        rows, cols = weight.shape
        D = rows // num_heads
        half = D // 2
        w = weight.view(num_heads, D, cols)
        interleaved = torch.stack([w[:, :half, :], w[:, half:, :]], dim=2)
        return interleaved.reshape(rows, cols)
    raise ValueError(f"Expected 1D or 2D tensor, got {weight.dim()}D")


def unpermute_norm_weights(weight: torch.Tensor) -> torch.Tensor:
    """HF QK-Norm ``[x1,x2,...,y1,y2,...]`` -> ttml ``[x1,y1,x2,y2,...]``."""
    head_dim = weight.shape[0]
    assert head_dim % 2 == 0
    half = head_dim // 2
    w = weight.view(2, half)
    return w.t().contiguous().flatten()


def build_weight_mapping_distributed(config: Qwen3Config, root_prefix: str, tie_word_embeddings: bool):
    """HF->ttml name mapping, shard types, and transforms for the distributed model.

    shard_types: ``"col_w"`` | ``"row_w"`` | ``"col_b"`` | ``None`` (replicated).
    transforms: ``("unpermute_proj", num_heads)`` | ``("unpermute_norm",)``.
    """
    mapping: dict = {}
    shard_types: dict = {}
    transforms: dict = {}

    if tie_word_embeddings:
        mapping["model.embed_tokens.weight"] = f"{root_prefix}/lm_head/weight"
        shard_types["model.embed_tokens.weight"] = "col_w"
    else:
        mapping["model.embed_tokens.weight"] = f"{root_prefix}/model/embed_tokens"
        shard_types["model.embed_tokens.weight"] = "row_w"
        mapping["lm_head.weight"] = f"{root_prefix}/lm_head/weight"
        shard_types["lm_head.weight"] = "col_w"

    for i in range(config.num_hidden_layers):
        hp = f"model.layers.{i}"
        tp = f"{root_prefix}/model/layers/{i}"

        mapping[f"{hp}.self_attn.q_proj.weight"] = f"{tp}/self_attn/q_proj/weight"
        shard_types[f"{hp}.self_attn.q_proj.weight"] = "col_w"
        transforms[f"{hp}.self_attn.q_proj.weight"] = ("unpermute_proj", config.num_attention_heads)

        mapping[f"{hp}.self_attn.k_proj.weight"] = f"{tp}/self_attn/k_proj/weight"
        shard_types[f"{hp}.self_attn.k_proj.weight"] = "col_w"
        transforms[f"{hp}.self_attn.k_proj.weight"] = ("unpermute_proj", config.num_key_value_heads)

        mapping[f"{hp}.self_attn.v_proj.weight"] = f"{tp}/self_attn/v_proj/weight"
        shard_types[f"{hp}.self_attn.v_proj.weight"] = "col_w"

        mapping[f"{hp}.self_attn.o_proj.weight"] = f"{tp}/self_attn/o_proj/weight"
        shard_types[f"{hp}.self_attn.o_proj.weight"] = "row_w"

        if config.attention_bias:
            mapping[f"{hp}.self_attn.q_proj.bias"] = f"{tp}/self_attn/q_proj/col_bias"
            shard_types[f"{hp}.self_attn.q_proj.bias"] = "col_b"
            transforms[f"{hp}.self_attn.q_proj.bias"] = ("unpermute_proj", config.num_attention_heads)

            mapping[f"{hp}.self_attn.k_proj.bias"] = f"{tp}/self_attn/k_proj/col_bias"
            shard_types[f"{hp}.self_attn.k_proj.bias"] = "col_b"
            transforms[f"{hp}.self_attn.k_proj.bias"] = ("unpermute_proj", config.num_key_value_heads)

            mapping[f"{hp}.self_attn.v_proj.bias"] = f"{tp}/self_attn/v_proj/col_bias"
            shard_types[f"{hp}.self_attn.v_proj.bias"] = "col_b"

            mapping[f"{hp}.self_attn.o_proj.bias"] = f"{tp}/self_attn/o_proj/row_bias"
            shard_types[f"{hp}.self_attn.o_proj.bias"] = None

        mapping[f"{hp}.self_attn.q_norm.weight"] = f"{tp}/self_attn/q_norm/weight"
        shard_types[f"{hp}.self_attn.q_norm.weight"] = None
        transforms[f"{hp}.self_attn.q_norm.weight"] = ("unpermute_norm",)

        mapping[f"{hp}.self_attn.k_norm.weight"] = f"{tp}/self_attn/k_norm/weight"
        shard_types[f"{hp}.self_attn.k_norm.weight"] = None
        transforms[f"{hp}.self_attn.k_norm.weight"] = ("unpermute_norm",)

        mapping[f"{hp}.input_layernorm.weight"] = f"{tp}/input_layernorm/weight"
        shard_types[f"{hp}.input_layernorm.weight"] = None
        mapping[f"{hp}.post_attention_layernorm.weight"] = f"{tp}/post_attention_layernorm/weight"
        shard_types[f"{hp}.post_attention_layernorm.weight"] = None

        mapping[f"{hp}.mlp.gate_proj.weight"] = f"{tp}/mlp/gate_proj/weight"
        shard_types[f"{hp}.mlp.gate_proj.weight"] = "col_w"
        mapping[f"{hp}.mlp.up_proj.weight"] = f"{tp}/mlp/up_proj/weight"
        shard_types[f"{hp}.mlp.up_proj.weight"] = "col_w"
        mapping[f"{hp}.mlp.down_proj.weight"] = f"{tp}/mlp/down_proj/weight"
        shard_types[f"{hp}.mlp.down_proj.weight"] = "row_w"

    mapping["model.norm.weight"] = f"{root_prefix}/model/norm/weight"
    shard_types["model.norm.weight"] = None
    return mapping, shard_types, transforms


def _load_tensor_distributed(weight_np, shard_type, shard_dim):
    """Create a bf16 ttml tensor with the appropriate mesh sharding."""
    device = _device()
    if shard_type == "col_w":
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 2, shard_dim)
        return ttml.autograd.Tensor.from_numpy(weight_np, ttnn.Layout.TILE, ttnn.bfloat16, mapper)
    elif shard_type in ("col_b", "row_w"):
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 3, shard_dim)
        return ttml.autograd.Tensor.from_numpy(weight_np, ttnn.Layout.TILE, ttnn.bfloat16, mapper)
    return ttml.autograd.Tensor.from_numpy(weight_np, ttnn.Layout.TILE, ttnn.bfloat16)


def load_weights_from_hf_distributed(
    ttml_model: DistributedQwen3ForCausalLM,
    hf_state_dict: dict,
    config: Qwen3Config,
    tie_word_embeddings: bool = False,
    shard_dim: Optional[int] = None,
    verbose: bool = False,
) -> None:
    """Load a HuggingFace Qwen3 state-dict into the distributed ttml model (bf16)."""
    from concurrent.futures import ThreadPoolExecutor
    from tqdm.auto import tqdm

    ttml_params = ttml_model.parameters()
    root = next(iter(ttml_params)).split("/")[0]
    mapping, shard_types, transforms = build_weight_mapping_distributed(config, root, tie_word_embeddings)

    tp_size = get_tp_size()
    ttml_shapes = {name: list(ttml_params[name].shape()) for name in ttml_params}

    def _prepare_and_transfer(hf_name, ttml_name):
        if hf_name not in hf_state_dict or ttml_name not in ttml_shapes:
            return None
        weight = hf_state_dict[hf_name].float()
        st = shard_types[hf_name]

        if hf_name in transforms:
            tr = transforms[hf_name]
            if tr[0] == "unpermute_proj":
                weight = unpermute_proj_rows(weight, num_heads=tr[1])
            elif tr[0] == "unpermute_norm":
                weight = unpermute_norm_weights(weight)

        ttml_shape = ttml_shapes[ttml_name]
        if weight.dim() == 2:
            rows, cols = weight.shape
            tgt_rows, tgt_cols = ttml_shape[2], ttml_shape[3]
            if st == "col_w":
                tgt_rows *= tp_size
            elif st == "row_w":
                tgt_cols *= tp_size
            if rows != tgt_rows or cols != tgt_cols:
                padded = torch.zeros(tgt_rows, tgt_cols, dtype=weight.dtype)
                padded[: min(rows, tgt_rows), : min(cols, tgt_cols)] = weight[
                    : min(rows, tgt_rows), : min(cols, tgt_cols)
                ]
                weight = padded
            weight = weight.unsqueeze(0).unsqueeze(0)
        elif weight.dim() == 1:
            dim = weight.shape[0]
            tgt_dim = ttml_shape[-1]
            if st == "col_b":
                tgt_dim *= tp_size
            if dim != tgt_dim:
                padded = torch.zeros(tgt_dim, dtype=weight.dtype)
                padded[: min(dim, tgt_dim)] = weight[: min(dim, tgt_dim)]
                weight = padded
            weight = weight.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected weight dim {weight.dim()} for {hf_name}")

        weight_np = weight.contiguous().float().numpy()
        return _load_tensor_distributed(weight_np, st, shard_dim)

    items = list(mapping.items())
    loaded = 0
    skipped = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(h, t, pool.submit(_prepare_and_transfer, h, t)) for h, t in items]
        for hf_name, ttml_name, future in tqdm(futures, total=len(items), desc="  Loading Qwen3 weights", unit="w"):
            new_tensor = future.result()
            if new_tensor is None:
                if ttml_name not in ttml_shapes:
                    print(f"  WARNING: ttml param '{ttml_name}' not found for HF '{hf_name}'")
                skipped.append(hf_name)
                continue
            ttml_params[ttml_name].assign(new_tensor)
            loaded += 1

    print(f"\n  Qwen3 weight loading: {loaded} loaded, {len(skipped)} skipped")
    if skipped:
        print(f"  Skipped: {skipped}")


# ---------------------------------------------------------------------------
# Single-device HF weight loading (for the core ttml.models.qwen3.Qwen3 model,
# used for small-model validation when tp_size == 1)
# ---------------------------------------------------------------------------


def build_weight_mapping_single(config: Qwen3Config, root_prefix: str, tie_word_embeddings: bool):
    """HF->ttml name mapping + transforms for the single-device core ``Qwen3`` model."""
    mapping: dict = {}
    transforms: dict = {}

    if tie_word_embeddings:
        mapping["model.embed_tokens.weight"] = f"{root_prefix}/fc/weight"
    else:
        mapping["model.embed_tokens.weight"] = f"{root_prefix}/tok_emb/weight"
        mapping["lm_head.weight"] = f"{root_prefix}/fc/weight"

    for i in range(config.num_hidden_layers):
        hp = f"model.layers.{i}"
        tp = f"{root_prefix}/blocks/{i}"

        mapping[f"{hp}.self_attn.q_proj.weight"] = f"{tp}/self_attn/q_proj/weight"
        transforms[f"{hp}.self_attn.q_proj.weight"] = ("unpermute_proj", config.num_attention_heads)
        mapping[f"{hp}.self_attn.k_proj.weight"] = f"{tp}/self_attn/k_proj/weight"
        transforms[f"{hp}.self_attn.k_proj.weight"] = ("unpermute_proj", config.num_key_value_heads)
        mapping[f"{hp}.self_attn.v_proj.weight"] = f"{tp}/self_attn/v_proj/weight"
        mapping[f"{hp}.self_attn.o_proj.weight"] = f"{tp}/self_attn/o_proj/weight"

        if config.attention_bias:
            mapping[f"{hp}.self_attn.q_proj.bias"] = f"{tp}/self_attn/q_proj/bias"
            transforms[f"{hp}.self_attn.q_proj.bias"] = ("unpermute_proj", config.num_attention_heads)
            mapping[f"{hp}.self_attn.k_proj.bias"] = f"{tp}/self_attn/k_proj/bias"
            transforms[f"{hp}.self_attn.k_proj.bias"] = ("unpermute_proj", config.num_key_value_heads)
            mapping[f"{hp}.self_attn.v_proj.bias"] = f"{tp}/self_attn/v_proj/bias"
            mapping[f"{hp}.self_attn.o_proj.bias"] = f"{tp}/self_attn/o_proj/bias"

        mapping[f"{hp}.self_attn.q_norm.weight"] = f"{tp}/self_attn/q_norm/weight"
        transforms[f"{hp}.self_attn.q_norm.weight"] = ("unpermute_norm",)
        mapping[f"{hp}.self_attn.k_norm.weight"] = f"{tp}/self_attn/k_norm/weight"
        transforms[f"{hp}.self_attn.k_norm.weight"] = ("unpermute_norm",)

        mapping[f"{hp}.input_layernorm.weight"] = f"{tp}/input_layernorm/weight"
        mapping[f"{hp}.post_attention_layernorm.weight"] = f"{tp}/post_attention_layernorm/weight"
        mapping[f"{hp}.mlp.gate_proj.weight"] = f"{tp}/mlp/gate_proj/weight"
        mapping[f"{hp}.mlp.up_proj.weight"] = f"{tp}/mlp/up_proj/weight"
        mapping[f"{hp}.mlp.down_proj.weight"] = f"{tp}/mlp/down_proj/weight"

    mapping["model.norm.weight"] = f"{root_prefix}/ln_fc/weight"
    return mapping, transforms


def load_weights_from_hf_single(
    ttml_model, hf_state_dict: dict, config: Qwen3Config, tie_word_embeddings: bool = False
):
    """Load a HuggingFace Qwen3 state-dict into the single-device core ``Qwen3`` model."""
    from concurrent.futures import ThreadPoolExecutor
    from tqdm.auto import tqdm

    ttml_params = ttml_model.parameters()
    root = next(iter(ttml_params)).split("/")[0]
    mapping, transforms = build_weight_mapping_single(config, root, tie_word_embeddings)
    ttml_shapes = {name: list(ttml_params[name].shape()) for name in ttml_params}

    def _prepare(hf_name, ttml_name):
        if hf_name not in hf_state_dict or ttml_name not in ttml_shapes:
            return None
        weight = hf_state_dict[hf_name].float()
        if hf_name in transforms:
            tr = transforms[hf_name]
            if tr[0] == "unpermute_proj":
                weight = unpermute_proj_rows(weight, num_heads=tr[1])
            elif tr[0] == "unpermute_norm":
                weight = unpermute_norm_weights(weight)

        ttml_shape = ttml_shapes[ttml_name]
        if weight.dim() == 2:
            rows, cols = weight.shape
            tgt_rows, tgt_cols = ttml_shape[2], ttml_shape[3]
            if rows != tgt_rows or cols != tgt_cols:
                padded = torch.zeros(tgt_rows, tgt_cols, dtype=weight.dtype)
                padded[: min(rows, tgt_rows), : min(cols, tgt_cols)] = weight[
                    : min(rows, tgt_rows), : min(cols, tgt_cols)
                ]
                weight = padded
            weight = weight.unsqueeze(0).unsqueeze(0)
        elif weight.dim() == 1:
            dim = weight.shape[0]
            tgt_dim = ttml_shape[-1]
            if dim != tgt_dim:
                padded = torch.zeros(tgt_dim, dtype=weight.dtype)
                padded[: min(dim, tgt_dim)] = weight[: min(dim, tgt_dim)]
                weight = padded
            weight = weight.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        weight_np = weight.contiguous().float().numpy()
        return ttml.autograd.Tensor.from_numpy(weight_np, ttnn.Layout.TILE, ttnn.bfloat16)

    items = list(mapping.items())
    loaded = 0
    skipped = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(h, t, pool.submit(_prepare, h, t)) for h, t in items]
        for hf_name, ttml_name, future in tqdm(futures, total=len(items), desc="  Loading Qwen3 weights", unit="w"):
            new_tensor = future.result()
            if new_tensor is None:
                skipped.append(hf_name)
                continue
            ttml_params[ttml_name].assign(new_tensor)
            loaded += 1

    print(f"\n  Qwen3 weight loading: {loaded} loaded, {len(skipped)} skipped")
    if skipped:
        print(f"  Skipped: {skipped}")
