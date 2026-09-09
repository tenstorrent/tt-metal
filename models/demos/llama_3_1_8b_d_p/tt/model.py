# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The whole Llama-3.1-8B model: embedding -> 32 decoder layers -> final norm -> LM head.

Outline borrowed from `minimax_m3/tt/model.py`, with three simplifications this model earns:

* **No per-layer type dispatch.** All 32 layers are identical, so the stack is a plain loop. The
  donor has to consult `moe_layer_freq` and `sparse_attention_freq` per layer.
* **No expert-parallel anything.**
* **No vocab padding.** 128256 is tile-aligned and divides TP=4 exactly.

## The final norm is pinned to the single-pass form

Even under a sharded residual. Its output feeds the column-parallel LM head, which needs full emb
anyway — so a distributed norm here would cost three ops plus a gather where one op plus a gather
does. It also keeps this norm's gain replicated, which is one fewer sharded weight.

## Layer slicing and cache indexing

`state_dict` is sliced per layer with `substate(state_dict, f"model.layers.{i}")`, and layer `i` gets
`layer_idx=i`. Under pipeline parallelism a rank holds layers `[first_layer_idx, +num_layers)` while
its KV cache is packed from 0, so the two indices separate: `layer_idx` selects weights,
`cache_layer_idx` selects the cache slot. Conflating them makes every layer read layer 0's cache —
correct at layer 0 by coincidence, stale everywhere after.

## KV cache sizing

One cache for the whole model, `num_layers` deep, allocated once by :meth:`Model.allocate_kv_cache`
and owned by the caller (the engine treats it as an opaque handle). Capacity is `max_seq_len`
rounded up to a whole chunk — see `tt/attention/kv_cache.py`.
"""

import ttnn
from loguru import logger

from models.demos.llama_3_1_8b_d_p.utils.general_utils import get_cache_file_name
from models.demos.llama_3_1_8b_d_p.utils.substate import substate

from .attention.kv_cache import allocate_kv_caches
from .layer import DecoderLayer
from .lm_head import LMHead
from .parallel_embedding import TtParallelEmbedding, cache_name_for, embed_shard_2d
from .residual import use_sharded_residual
from .rms_norm import RMSNorm
from .rope import create_rope_setup


class Model:
    """Llama-3.1-8B prefill model on the target mesh."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        mesh_config,
        *,
        max_seq_len,
        chunk_size,
        num_layers=None,
        first_layer_idx=0,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        is_first_rank=True,
        is_last_rank=True,
        max_local_batch_size=1,
    ):
        """
        Args:
            num_layers: layers THIS rank builds. None => the config's full 32.
            first_layer_idx: global index of this rank's first layer. Non-zero only under pipeline
                parallelism; the KV cache is still packed from 0, hence `cache_layer_idx` below.
            is_first_rank / is_last_rank: pipeline placement. The embedding is built only on the
                first rank and the final norm + LM head only on the last; a middle rank builds
                neither and just forwards a hidden state.
        """
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hf_config = hf_config
        self.max_seq_len = max_seq_len
        self.chunk_size = chunk_size
        self.num_layers = hf_config.num_hidden_layers if num_layers is None else num_layers
        self.first_layer_idx = first_layer_idx
        self.is_first_rank = is_first_rank
        self.is_last_rank = is_last_rank
        self.sharded_residual = use_sharded_residual() and mesh_config.tp > 1

        logger.info(
            f"Llama-3.1-8B model: {self.num_layers} layers from {first_layer_idx}, "
            f"sp{mesh_config.sp}xtp{mesh_config.tp}, "
            f"{'SHARDED emb/tp' if self.sharded_residual else 'REPLICATED full emb'} residual"
        )

        # rotary_embedding_llama requires bfloat16 tables. Built once for the whole model — the
        # cos/sin are identical across layers, so a per-layer build would be pure waste.
        self.rope_setup = create_rope_setup(mesh_device, hf_config, max_seq_len=max_seq_len, datatype=ttnn.bfloat16)
        self.transformation_mats = self.rope_setup.transformation_mat_prefill

        self.embedding = None
        if is_first_rank:
            shard_vocab = embed_shard_2d()
            embedding_weight = substate(state_dict, "model.embed_tokens").get("weight") if state_dict else None
            self.embedding = TtParallelEmbedding(
                mesh_device=mesh_device,
                vocab_size=hf_config.vocab_size,
                emb_dim=hf_config.hidden_size,
                mesh_config=mesh_config,
                ccl_manager=ccl_manager,
                torch_weight=embedding_weight,
                cache_file_name=get_cache_file_name(tensor_cache_path, cache_name_for(shard_vocab)),
                shard_vocab_on_sp=shard_vocab,
            )

        self.layers = [
            DecoderLayer(
                mesh_device,
                hf_config,
                substate(state_dict, f"model.layers.{first_layer_idx + i}"),
                layer_idx=first_layer_idx + i,
                ccl_manager=ccl_manager,
                mesh_config=mesh_config,
                max_seq_len=max_seq_len,
                chunk_size=chunk_size,
                weight_dtype=weight_dtype,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, f"layers.{first_layer_idx + i}"),
                transformation_mats=self.transformation_mats,
                max_local_batch_size=max_local_batch_size,
                # Weights come from the GLOBAL index; the cache slot is this rank's LOCAL index.
                cache_layer_idx=i,
            )
            for i in range(self.num_layers)
        ]

        self.norm = None
        self.lm_head = None
        if is_last_rank:
            self.norm = RMSNorm(
                mesh_device,
                hf_config,
                substate(state_dict, "model.norm"),
                tensor_cache_path=get_cache_file_name(tensor_cache_path, "norm"),
                mesh_config=mesh_config,
                ccl_manager=ccl_manager,
                is_distributed=False,  # pinned single-pass — see the module docstring
            )
            self.lm_head = LMHead(
                mesh_device,
                hf_config,
                substate(state_dict, "lm_head"),
                mesh_config=mesh_config,
                weight_dtype=weight_dtype,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, "lm_head"),
            )

    def allocate_kv_cache(self, *, num_users=1, cache_dtype=ttnn.bfloat8_b):
        """One KV cache for this rank's layers. Allocated once; the caller owns its lifetime."""
        return allocate_kv_caches(
            self.mesh_device,
            num_layers=self.num_layers,
            max_seq_len=self.max_seq_len,
            chunk_size=self.chunk_size,
            sp_axis=self.mesh_config.sp_axis,
            tp_axis=self.mesh_config.tp_axis,
            num_users=num_users,
            num_kv_heads=self.hf_config.num_key_value_heads,
            head_dim=getattr(self.hf_config, "head_dim", None)
            or self.hf_config.hidden_size // self.hf_config.num_attention_heads,
            cache_dtype=cache_dtype,
        )

    def forward(
        self,
        tokens_or_hidden,
        *,
        rope_mats,
        kv_cache=None,
        slot_idx=0,
        cached_len=0,
        logical_n=None,
        indexed_rope=False,
        return_logits=True,
    ):
        """Run one chunk through this rank's slice of the model.

        Args:
            tokens_or_hidden: token ids `[1, 1, tokens_local]` on the first rank, otherwise an
                already-embedded hidden state.
            rope_mats: `(cos, sin)` in Meta order — per-chunk tables, or the whole-cache
                block-cyclic tables when `indexed_rope`.
            return_logits: False to stop after the final norm. A prefill run that only needs the KV
                cache does not need a 128256-wide projection over every token.
        """
        hidden = tokens_or_hidden
        if self.is_first_rank and self.embedding is not None:
            # The embedding already emits the residual-stream layout: `emb/tp` under a sharded
            # residual (it simply skips its closing TP all-gather) and full emb otherwise. No
            # re-layout here — a reduce-scatter would SUM across TP columns, which is wrong for a
            # value that is replicated across them.
            hidden = self.embedding.forward(hidden)

        for layer in self.layers:
            hidden = layer(
                hidden,
                rope_mats=rope_mats,
                kv_cache=kv_cache,
                slot_idx=slot_idx,
                cached_len=cached_len,
                logical_n=logical_n,
                indexed_rope=indexed_rope,
            )

        if not self.is_last_rank:
            return hidden

        # The final norm needs full emb (the LM head is column-parallel over the vocab and takes emb
        # as its contraction dim), so gather once here under a sharded residual.
        if self.sharded_residual:
            gathered = self.mesh_config.allgather(
                hidden, self.ccl_manager, axis=self.mesh_config.tp_axis, dim=3
            )
            hidden.deallocate(True)
            hidden = gathered
        hidden = self.norm.forward(hidden)
        if not return_logits:
            return hidden
        return self.lm_head(hidden)
