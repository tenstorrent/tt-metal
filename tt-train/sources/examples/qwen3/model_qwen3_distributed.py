# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Distributed (Tensor Parallel + Data Parallel) Qwen3 model implementation.

TP strategy (Megatron-LM / C++ DistributedLlama):
  - Attention Q, K, V:  ColumnParallel (shard output heads)
  - Attention O:         RowParallel    (shard input, all-reduce)
  - MLP gate, up:        ColumnParallel (shard intermediate)
  - MLP down:            RowParallel    (shard input, all-reduce)
  - Embedding (untied):  Sharded along hidden dim, all-gather after lookup
  - Embedding (tied):    VocabParallelEmbedding (Megatron-LM style) — each
                          TP device looks up its local vocab shard, masks
                          out-of-range tokens, all-reduces hidden vectors.
                          No weight all-gather; only hidden-dim communication.
  - Norms (QK, layer, final): Replicated
  - LM head:             Always ColumnParallel; supports sharded_loss in
                          both tied and untied modes

Communication per layer: 2 all-reduces (attention + MLP).

DP+TP support:
  The 2D device mesh is organised as [dp_size, tp_size]:
    - Mesh dim 0 = data-parallel groups  (DP)
    - Mesh dim 1 = tensor-parallel devices (TP)
  All TP ops (broadcast, scatter, all_reduce, all_gather) operate on
  shard_dim=1 (TP axis only).  Weights sharded via
  shard_tensor_to_mesh_mapper(..., shard_dim=1) are automatically
  replicated along mesh dim 0 (DP).  Replicated weights (norms,
  embeddings) use from_numpy without a mapper and are replicated to
  every device in the mesh.

  After backward, call:
    ttml.core.distributed.synchronize_gradients(model.parameters(), 0)
  to average gradients across DP groups.

Usage (TP only):
    ttml.core.distributed.enable_fabric(tp_size)
    ctx.open_device([1, tp_size])
    model = DistributedQwen3ForCausalLM(config, shard_dim=1)

Usage (DP + TP):
    ttml.core.distributed.enable_fabric(dp_size * tp_size)
    ctx.open_device([dp_size, tp_size])
    model = DistributedQwen3ForCausalLM(config, shard_dim=1)
    # Data: shard batch along DP (mesh dim 0), replicate along TP (mesh dim 1)
"""

from typing import Optional
import numpy as np
import torch
from tqdm.auto import tqdm
import ttnn
import ttml
from ttml.modules import AbstractModuleBase, ModuleList, Parameter

from model_qwen3 import Qwen3Config, Qwen3RMSNorm, ConcatLastDim, linear
from utils.memory import memory_snapshot
from utils.checkpoint import (  # noqa: F401 — re-exported for callers
    CheckpointFunction,
    checkpoint,
    checkpoint_scattered,
)
from utils.param_utils import (
    unpermute_proj_rows,
    unpermute_norm_weights,
    build_weight_mapping_distributed,
)
from utils.tensor_utils import (
    get_device,
    get_tp_size,
    tile_pad as _tile_pad,
    make_empty_on_device as _make_empty_on_device,
    make_weight as _make_weight,
    make_ones as _make_ones,
    make_zeros as _make_zeros,
    make_sharded_weight,
    make_sharded_zeros,
    make_dist_replicated,
    make_replicated_ones,
    make_replicated_zeros,
    make_replicated_weight,
)
from utils.distributed_ops import (
    AllGatherFwdScatterBwd,
    all_gather_fwd_scatter_bwd,
    _vocab_parallel_embedding,
)


# ---------------------------------------------------------------------------
# ColumnParallelLinear
# ---------------------------------------------------------------------------


class ColumnParallelLinear(AbstractModuleBase):
    """Column-parallel linear — shards output features across TP devices.

    Weight full shape:  (1, 1, out_features, in_features)
    Per-device shape:   (1, 1, out_features/tp, in_features)
    """

    def __init__(
        self,
        in_features,
        out_features,
        has_bias=False,
        gather_output=False,
        shard_dim=None,
    ):
        super().__init__()
        self.gather_output = gather_output
        self.shard_dim = shard_dim

        self.weight = Parameter(
            make_sharded_weight((1, 1, out_features, in_features), 2, shard_dim)
        )
        if has_bias:
            self.col_bias = Parameter(
                make_sharded_zeros((1, 1, 1, out_features), 3, shard_dim)
            )
        else:
            self.col_bias = None

    def forward(self, x):
        x = ttml.ops.distributed.broadcast(x, self.shard_dim)
        bias_t = self.col_bias.tensor if self.col_bias is not None else None
        x = linear(x, self.weight.tensor, bias_t)
        if self.gather_output:
            x = all_gather_fwd_scatter_bwd(x, 3, self.shard_dim)
        return x


# ---------------------------------------------------------------------------
# RowParallelLinear
# ---------------------------------------------------------------------------


class RowParallelLinear(AbstractModuleBase):
    """Row-parallel linear — shards input features across TP devices.

    Weight full shape:  (1, 1, out_features, in_features)
    Per-device shape:   (1, 1, out_features, in_features/tp)
    Bias: replicated.
    """

    def __init__(
        self,
        in_features,
        out_features,
        has_bias=False,
        input_is_parallel=False,
        shard_dim=None,
    ):
        super().__init__()
        self.input_is_parallel = input_is_parallel
        self.shard_dim = shard_dim

        self.weight = Parameter(
            make_sharded_weight((1, 1, out_features, in_features), 3, shard_dim)
        )
        if has_bias:
            self.row_bias = Parameter(make_replicated_zeros((1, 1, 1, out_features)))
        else:
            self.row_bias = None

    def forward(self, x):
        if not self.input_is_parallel:
            x = ttml.ops.distributed.scatter(x, 3, self.shard_dim)
        x = linear(x, self.weight.tensor, None)
        x = ttml.ops.distributed.all_reduce(x, self.input_is_parallel, self.shard_dim)
        if self.row_bias is not None:
            x = ttml.ops.binary.add(x, self.row_bias.tensor)
        return x


# ---------------------------------------------------------------------------
# DistributedQwen3Attention
# ---------------------------------------------------------------------------


class DistributedQwen3Attention(AbstractModuleBase):
    """TP Qwen3 attention: Q/K/V=ColumnParallel, O=RowParallel."""

    def __init__(
        self, config: Qwen3Config, layer_idx: int, shard_dim: Optional[int] = None
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.shard_dim = shard_dim

        tp = get_tp_size(shard_dim)
        assert self.num_heads % tp == 0
        assert self.num_kv_heads % tp == 0
        self.num_local_heads = self.num_heads // tp
        self.num_local_kv_heads = self.num_kv_heads // tp

        q_out = self.num_heads * self.head_dim
        kv_out = self.num_kv_heads * self.head_dim

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            q_out,
            has_bias=config.attention_bias,
            gather_output=False,
            shard_dim=shard_dim,
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            kv_out,
            has_bias=config.attention_bias,
            gather_output=False,
            shard_dim=shard_dim,
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            kv_out,
            has_bias=config.attention_bias,
            gather_output=False,
            shard_dim=shard_dim,
        )
        self.o_proj = RowParallelLinear(
            q_out,
            self.hidden_size,
            has_bias=config.attention_bias,
            input_is_parallel=True,
            shard_dim=shard_dim,
        )

        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        rope_scaling = ttml.ops.rope.RopeScalingParams()
        if (
            config.rope_scaling_factor != 0.0
            and config.rope_original_context_length != 0
        ):
            rope_scaling.original_context_length = config.rope_original_context_length
            rope_scaling.scaling_factor = config.rope_scaling_factor
            rope_scaling.high_freq_factor = config.rope_high_freq_factor
            rope_scaling.low_freq_factor = config.rope_low_freq_factor

        self.rope_params = ttml.ops.rope.build_rope_params(
            sequence_length=config.max_position_embeddings,
            head_dim=self.head_dim,
            theta=config.rope_theta,
            rope_scaling_params=rope_scaling,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_values=None,
        position_offset=0,
    ):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q_shape, k_shape = q.shape(), k.shape()
        B, S = q_shape[0], q_shape[2]

        q = ttml.ops.reshape.reshape(q, [B, 1, S * self.num_local_heads, self.head_dim])
        k = ttml.ops.reshape.reshape(
            k, [B, 1, S * self.num_local_kv_heads, self.head_dim]
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = ttml.ops.reshape.reshape(q, q_shape)
        k = ttml.ops.reshape.reshape(k, k_shape)

        kvs = ConcatLastDim.apply(k, v)
        (
            query_heads,
            key_heads,
            value_heads,
        ) = ttml.ops.multi_head_utils.grouped_heads_creation(
            q, kvs, self.num_local_heads, self.num_local_kv_heads
        )

        query_heads = ttml.ops.rope.rope(query_heads, self.rope_params, position_offset)
        key_heads = ttml.ops.rope.rope(key_heads, self.rope_params, position_offset)

        if past_key_values is not None:
            key_heads, value_heads = past_key_values.update(
                self.layer_idx, key_heads, value_heads
            )

        attn = ttml.ops.attention.scaled_dot_product_attention(
            query_heads, key_heads, value_heads, attention_mask
        )

        return self.o_proj(ttml.ops.multi_head_utils.heads_fusion(attn))


# ---------------------------------------------------------------------------
# DistributedQwen3MLP
# ---------------------------------------------------------------------------


class DistributedQwen3MLP(AbstractModuleBase):
    """TP SwiGLU MLP: gate/up=ColumnParallel, down=RowParallel."""

    def __init__(self, config: Qwen3Config, shard_dim=None):
        super().__init__()
        h, inter = config.hidden_size, config.intermediate_size
        self.gate_proj = ColumnParallelLinear(h, inter, shard_dim=shard_dim)
        self.up_proj = ColumnParallelLinear(h, inter, shard_dim=shard_dim)
        self.down_proj = RowParallelLinear(
            inter, h, input_is_parallel=True, shard_dim=shard_dim
        )

    def forward(self, x):
        gate = ttml.ops.unary.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(ttml.ops.binary.mul(gate, up))


# ---------------------------------------------------------------------------
# DistributedQwen3DecoderLayer
# ---------------------------------------------------------------------------


class DistributedQwen3DecoderLayer(AbstractModuleBase):
    def __init__(self, config: Qwen3Config, layer_idx: int, shard_dim=None):
        super().__init__()
        self.self_attn = DistributedQwen3Attention(config, layer_idx, shard_dim)
        self.mlp = DistributedQwen3MLP(config, shard_dim)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_values=None,
        position_offset=0,
    ):
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            past_key_values,
            position_offset=position_offset,
        )
        hidden_states = ttml.ops.binary.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = ttml.ops.binary.add(residual, hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# DistributedQwen3Model
# ---------------------------------------------------------------------------


class DistributedQwen3Model(AbstractModuleBase):
    def __init__(
        self,
        config: Qwen3Config,
        shard_dim=None,
        use_checkpoint=False,
        scatter_intermediates=False,
        track_memory=0,
        tied_embed_weight=None,
    ):
        super().__init__()
        self.config = config
        self.shard_dim = shard_dim
        self.use_checkpoint = use_checkpoint
        self.scatter_intermediates = scatter_intermediates
        self.track_memory = track_memory
        self.tied_embed_weight = tied_embed_weight
        vocab_tiled = ((config.vocab_size + 31) // 32) * 32
        if tied_embed_weight is None:
            self.embed_tokens = Parameter(
                make_sharded_weight(
                    (1, 1, vocab_tiled, config.hidden_size), 3, shard_dim
                )
            )
        self.layers = ModuleList(
            [
                DistributedQwen3DecoderLayer(config, i, shard_dim)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, input_ids, attention_mask=None, past_key_values=None, input_ids_np=None
    ):
        if self.tied_embed_weight is not None:
            h = _vocab_parallel_embedding(
                input_ids_np,
                self.tied_embed_weight,
                self.config.vocab_size,
                self.shard_dim,
            )
        else:
            h = ttml.ops.embedding.embedding(input_ids, self.embed_tokens.tensor)
            h = all_gather_fwd_scatter_bwd(h, 3, self.shard_dim)
        if self.track_memory:
            h = memory_snapshot(h, "AFTER_EMBEDDING_FWD", "AFTER_EMBEDDING_BWD")
        position_offset = 0
        if past_key_values is not None:
            position_offset = past_key_values.get_seq_length()
        for i, layer in enumerate(self.layers):
            if self.use_checkpoint and self.scatter_intermediates:
                h = checkpoint_scattered(layer, 0, self.shard_dim, h, attention_mask)
            elif self.use_checkpoint:
                h = checkpoint(
                    layer, h, attention_mask, past_key_values, position_offset
                )
            else:
                h = layer(
                    h, attention_mask, past_key_values, position_offset=position_offset
                )
            if self.track_memory and (i + 1) % self.track_memory == 0:
                h = memory_snapshot(h, f"AFTER_LAYER_{i}_FWD", f"AFTER_LAYER_{i}_BWD")
        return self.norm(h)


# ---------------------------------------------------------------------------
# DistributedQwen3ForCausalLM
# ---------------------------------------------------------------------------


class DistributedQwen3ForCausalLM(AbstractModuleBase):
    """TP Qwen3 with LM head (ColumnParallel).

    The LM head is always a ``ColumnParallelLinear``.  When
    ``tie_word_embeddings`` is True the same vocab-sharded weight is reused
    for input embedding (Megatron-LM VocabParallelEmbedding: local lookup →
    mask → all-reduce) and output projection, enabling ``sharded_loss`` in
    both tied and untied modes.

    When *sharded_loss* is ``True`` the LM head keeps its output sharded
    across TP devices (``gather_output=False``).  The caller is responsible
    for matching the loss target shape and scaling the loss gradient by
    ``1/tp_size`` so that the per-element gradient equals the global mean.

    When ``tie_word_embeddings`` is True, callers must pass ``input_ids_np``
    (numpy uint32 token IDs) to :meth:`forward` for the vocab-parallel
    embedding preprocessing.
    """

    def __init__(
        self,
        config: Qwen3Config,
        tie_word_embeddings=False,
        shard_dim=None,
        use_checkpoint=False,
        scatter_intermediates=False,
        track_memory=0,
        sharded_loss=False,
    ):
        super().__init__()
        self.create_name("DistributedQwen3ForCausalLM")
        self.config = config
        self.tie_word_embeddings = tie_word_embeddings
        self.shard_dim = shard_dim
        self.track_memory = track_memory
        self.sharded_loss = sharded_loss

        vocab_tiled = ((config.vocab_size + 31) // 32) * 32
        lm_vocab = vocab_tiled if tie_word_embeddings else config.vocab_size
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            lm_vocab,
            has_bias=False,
            gather_output=(not sharded_loss),
            shard_dim=shard_dim,
        )

        self.model = DistributedQwen3Model(
            config,
            shard_dim,
            use_checkpoint=use_checkpoint,
            scatter_intermediates=scatter_intermediates,
            track_memory=track_memory,
            tied_embed_weight=(
                self.lm_head.weight.tensor if tie_word_embeddings else None
            ),
        )

    def forward(
        self, input_ids, attention_mask=None, past_key_values=None, input_ids_np=None
    ):
        h = self.model(
            input_ids, attention_mask, past_key_values, input_ids_np=input_ids_np
        )
        if self.track_memory:
            h = memory_snapshot(h, "AFTER_NORM_FWD", "AFTER_NORM_BWD")
        out = self.lm_head(h)
        if self.track_memory:
            out = memory_snapshot(out, "AFTER_LM_HEAD_FWD", "AFTER_LM_HEAD_BWD")
        return out


# ---------------------------------------------------------------------------
# Distributed weight loading from HuggingFace
# ---------------------------------------------------------------------------


def _load_tensor_distributed(weight_np, shard_type, shard_dim, device):
    """Create a bfloat16 ttml tensor with appropriate sharding.

    shard_type: None=replicated, "col_w"=shard dim 2,
                "col_b"=shard dim 3, "row_w"=shard dim 3
    """
    if shard_type in ("col_w",):
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 2, shard_dim)
        return ttml.autograd.Tensor.from_numpy(
            weight_np, ttnn.Layout.TILE, ttnn.bfloat16, mapper
        )
    elif shard_type in ("col_b", "row_w"):
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 3, shard_dim)
        return ttml.autograd.Tensor.from_numpy(
            weight_np, ttnn.Layout.TILE, ttnn.bfloat16, mapper
        )
    else:
        return ttml.autograd.Tensor.from_numpy(
            weight_np, ttnn.Layout.TILE, ttnn.bfloat16
        )


def load_weights_from_hf_distributed(
    ttml_model: DistributedQwen3ForCausalLM,
    hf_state_dict: dict,
    config: Qwen3Config,
    tie_word_embeddings: bool = False,
    shard_dim: Optional[int] = None,
    verbose: bool = False,
) -> None:
    """Load HF weights into distributed ttml model (bfloat16)."""
    device = get_device()
    ttml_params = ttml_model.parameters()

    if verbose:
        print("\n  TTML distributed parameter names:")
        for name in sorted(ttml_params.keys()):
            print(f"    {name}: {list(ttml_params[name].shape())}")

    root = next(iter(ttml_params)).split("/")[0]

    mapping, shard_types, transforms = build_weight_mapping_distributed(
        config, root, tie_word_embeddings
    )

    tp_size = get_tp_size(shard_dim)
    ttml_shapes = {name: list(ttml_params[name].shape()) for name in ttml_params}

    def _prepare_and_transfer(hf_name, ttml_name):
        """CPU prep + host-side tilize + device transfer (pipelined)."""
        if hf_name not in hf_state_dict:
            return None
        if ttml_name not in ttml_shapes:
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
        return _load_tensor_distributed(weight_np, st, shard_dim, device)

    from concurrent.futures import ThreadPoolExecutor

    items = list(mapping.items())
    loaded = 0
    skipped = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            (hf_name, ttml_name, pool.submit(_prepare_and_transfer, hf_name, ttml_name))
            for hf_name, ttml_name in items
        ]

        for hf_name, ttml_name, future in tqdm(
            futures,
            total=len(items),
            desc="  Loading weights",
            unit="w",
        ):
            new_tensor = future.result()
            if new_tensor is None:
                if ttml_name not in ttml_shapes:
                    print(
                        f"  WARNING: ttml param '{ttml_name}' not found for HF '{hf_name}'"
                    )
                skipped.append(hf_name)
                continue
            ttml_params[ttml_name].assign(new_tensor)
            loaded += 1

    print(f"\n  Weight loading: {loaded} loaded, {len(skipped)} skipped")
    if skipped:
        print(f"  Skipped: {skipped}")
