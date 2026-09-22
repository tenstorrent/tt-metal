# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The whole model on the mesh: embedding -> 88 x decoder -> final norm -> lm head.

Structure ported from ``minimax_m3/tt/model.py``; the layer slot is this package's real
:class:`~.layer.DecoderLayer`, so this file is only the stack and the two ends of it. It mirrors
:class:`~...reference.modeling.MistralModel` op for op, which is what makes
``tests/unit/test_model_sp_vs_ref.py`` a meaningful comparison rather than two independent designs
that happen to agree.

Four things are decided here rather than inside a block, because they are properties of the stack:

* **One RoPE table per chunk, not per layer.** All 88 layers see the same absolute positions, so
  ``[cos, sin]`` and the transformation matrix are built once in :meth:`MistralModel.__call__` and
  handed down. Building them per layer would be 88x the cost and 88x the DRAM churn for identical
  tensors. ``rope_mats`` can still be passed in, which is what a traced or pre-staged runtime wants.
* **One KV cache for all layers.** ``allocate_kv_cache(num_layers=88)`` packs the layers into the
  user-major slot axis, and ``DecoderLayer`` picks its slot out with ``layer_idx``. 88 separate
  single-layer caches would be 88 separate DRAM NdShard allocations addressed by 88 different
  base pointers, for no gain.
* **The layer loop is flat.** ``config.json`` has no ``layer_types`` and ``sliding_window`` is null,
  so there is no hybrid dispatch — every layer is the same full-attention block. The assertion is in
  the reference (:class:`~...reference.modeling.MistralModel`); repeating it here would be noise, but
  the flatness is the reason this loop is allowed to be a loop.
* **The lm head is optional.** A 10240-token prefill through ``[12288, 131072]`` is the single most
  expensive matmul in the model and produces nothing the K/V acceptance looks at. ``want_logits`` is
  a run-time switch (skip the matmul) and ``with_lm_head=False`` is a construction-time one (do not
  even load the 1.6 G-parameter weight). The acceptance run uses both.

**Depth is a constructor argument, not a config edit.** ``num_layers`` defaults to
``config.num_hidden_layers`` and exists so a diagnostic can build a 2- or 4-layer stack at *full
width* on the target mesh without perturbing the config that every cache key is derived from. Any
value below the config's is a diagnostic; acceptance runs the full 88.
"""

import torch

import ttnn
from models.demos.mistral_medium_3_5_128b.tt.attention.kv_cache import allocate_kv_cache
from models.demos.mistral_medium_3_5_128b.tt.embedding import ParallelEmbedding
from models.demos.mistral_medium_3_5_128b.tt.layer import DecoderLayer
from models.demos.mistral_medium_3_5_128b.tt.lm_head import LMHead
from models.demos.mistral_medium_3_5_128b.tt.rms_norm import RMSNorm
from models.demos.mistral_medium_3_5_128b.tt.rope import build_rope_mats, build_transformation_mat
from models.demos.mistral_medium_3_5_128b.utils.substate import substate


def shard_tokens(mesh_device, mesh_config, token_ids, *, sequence_parallel: bool = True):
    """Push ``[1, tokens]`` int token ids onto the mesh as the uint32 ``[1, 1, tokens_local]``
    :class:`~.embedding.ParallelEmbedding` expects.

    ROW_MAJOR and uint32 because ``ttnn.embedding`` requires both of its index tensor. The sequence
    is split contiguously over the SP rows and replicated across the TP cols — the same split the
    activations get, so token ``i`` and its embedding live on the same chip.
    """
    ids = token_ids.reshape(1, 1, -1).to(dtype=torch.int32)
    if sequence_parallel:
        sp = mesh_config.sp
        assert (
            ids.shape[-1] % (ttnn.TILE_SIZE * sp) == 0
        ), f"token count ({ids.shape[-1]}) must be a multiple of TILE_SIZE * sp ({ttnn.TILE_SIZE * sp})"
        dims = [None, None]
        dims[mesh_config.sp_axis] = -1
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=tuple(dims))
    else:
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    return ttnn.from_torch(
        ids,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


class MistralModel:
    """Embedding -> ``num_layers`` x :class:`~.layer.DecoderLayer` -> final norm -> lm head."""

    def __init__(
        self,
        mesh_device,
        config,
        state_dict,
        ccl_manager,
        mesh_config,
        *,
        max_seq_len: int,
        chunk_size: int,
        num_layers: int | None = None,
        with_lm_head: bool = True,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        sequence_parallel: bool = True,
        program_config=None,
        shard_vocab_on_sp: bool | None = None,
    ):
        """
        Args:
            mesh_device: the open mesh.
            config: a :class:`~...reference.model_config.MistralMediumConfig`.
            state_dict: the model's weights with the checkpoint's language-model prefix stripped:
                ``embed_tokens.weight``, ``layers.N.*``, ``norm.weight``, ``lm_head.weight``. Empty
                dict => every block loads from ``tensor_cache_path`` instead.
            ccl_manager: :class:`~...tt.ccl.CCLManager`.
            mesh_config: :class:`~...tt.config.MeshConfig`.
            max_seq_len: the sequence capacity to configure attention and size the cache for.
            chunk_size: the block-cyclic cache period, and the length of one prefill call.
            num_layers: stack depth; defaults to ``config.num_hidden_layers``. See the module
                docstring — anything shorter is a diagnostic.
            with_lm_head: False skips loading the ``[12288, 131072]`` head entirely.
            weight_dtype: on-device weight dtype for the projections (the spec's
                ``dataformats.weights``). The embedding table overrides this — see
                :mod:`~.embedding`.
            tensor_cache_path: directory for the tilized-weight cache, or None.
            sequence_parallel: True is the production layout (sequence SP-sharded on the rows).
            program_config: :class:`~.attention.config.ProgramConfig`, shared by every layer.
            shard_vocab_on_sp: embedding layout override; see :func:`~.embedding.embed_shard_vocab_on_sp`.
        """
        self.mesh_device = mesh_device
        self.config = config
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.max_seq_len = max_seq_len
        self.chunk_size = chunk_size
        self.sequence_parallel = sequence_parallel
        self.num_layers = config.num_hidden_layers if num_layers is None else num_layers
        assert (
            1 <= self.num_layers <= config.num_hidden_layers
        ), f"num_layers ({self.num_layers}) must be in [1, {config.num_hidden_layers}]"

        def _sub_cache(name):
            return f"{tensor_cache_path}/{name}" if tensor_cache_path else None

        # Shared across every layer: one table, built once.
        self.transformation_mat = build_transformation_mat(mesh_device)

        self.embed_tokens = ParallelEmbedding(
            mesh_device,
            config,
            substate(state_dict, "embed_tokens"),
            mesh_config,
            ccl_manager,
            shard_vocab_on_sp=shard_vocab_on_sp,
            tensor_cache_path=_sub_cache("embed_tokens"),
        )
        self.layers = [
            DecoderLayer(
                mesh_device,
                config,
                substate(state_dict, f"layers.{i}"),
                i,
                ccl_manager,
                mesh_config,
                max_seq_len=max_seq_len,
                transformation_mat=self.transformation_mat,
                weight_dtype=weight_dtype,
                tensor_cache_path=_sub_cache(f"layers.{i}"),
                sequence_parallel=sequence_parallel,
                program_config=program_config,
            )
            for i in range(self.num_layers)
        ]
        self.norm = RMSNorm(
            mesh_device,
            config,
            substate(state_dict, "norm"),
            mesh_config=mesh_config,
            tensor_cache_path=_sub_cache("norm"),
        )
        self.lm_head = (
            LMHead(
                mesh_device,
                config,
                substate(state_dict, "lm_head"),
                mesh_config,
                ccl_manager,
                weight_dtype=weight_dtype,
                tensor_cache_path=_sub_cache("lm_head"),
            )
            if with_lm_head
            else None
        )

    def allocate_cache(self, num_users: int = 1):
        """One KV cache covering the whole stack, sized from this model's own capacity.

        Sizing from ``max_seq_len`` and not from ``config.max_position_embeddings`` is deliberate:
        ``allocate_kv_cache`` materializes a host ``torch.zeros`` of the full per-chip shape, so a
        262144-token capacity would cost gigabytes of host memory to allocate a cache a 10240-token
        run never reads past the first 4% of.
        """
        return allocate_kv_cache(
            self.mesh_device,
            num_layers=self.num_layers,
            max_seq_len=self.max_seq_len,
            chunk_size=self.chunk_size,
            sp_axis=self.mesh_config.sp_axis,
            tp=self.mesh_config.tp,
            num_users=num_users,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
        )

    def rope_mats(self, start: int, end: int):
        """``[cos, sin]`` for absolute positions ``[start, end)``, in this model's layout."""
        return build_rope_mats(
            self.mesh_device,
            self.config,
            start,
            end,
            mesh_config=self.mesh_config,
            sequence_parallel=self.sequence_parallel,
        )

    def __call__(
        self,
        tokens,
        *,
        kv_cache=None,
        user_id: int = 0,
        cached_len: int = 0,
        want_logits: bool = True,
        rope_mats=None,
    ):
        """One prefill chunk.

        Args:
            tokens: uint32 ``[1, 1, tokens_local]`` from :func:`shard_tokens`, **or** an already
                embedded ``[1, 1, tokens_local, hidden_size]`` activation (4D), which is what the
                whole-model test uses to isolate the stack from the embedding.
            kv_cache: a :class:`~.attention.kv_cache.MistralKVCache` from :meth:`allocate_cache`,
                or None for a cacheless one-shot run.
            user_id: cache user slot.
            cached_len: tokens already written for this user — this chunk's absolute start.
            want_logits: False returns the final-normed hidden state and skips the head matmul.
            rope_mats: pre-built ``[cos, sin]``; built for this chunk's positions when None.

        Returns:
            ``[1, 1, tokens_local, vocab_local]`` logits (vocab column-parallel over TP, *not*
            gathered — call :meth:`~.lm_head.LMHead.gather` for the full width), or the
            ``[1, 1, tokens_local, hidden_size]`` normed hidden state when ``want_logits`` is False.
        """
        needs_embedding = len(tokens.shape) != 4
        hidden = self.embed_tokens(tokens) if needs_embedding else tokens

        if rope_mats is None:
            tokens_local = hidden.shape[-2]
            span = tokens_local * (self.mesh_config.sp if self.sequence_parallel else 1)
            rope_mats = self.rope_mats(cached_len, cached_len + span)

        # `owned` keeps the caller's tensor alive when it handed us a pre-embedded activation: the
        # loop frees each intermediate, and freeing an argument would be a surprise across the seam.
        owned = needs_embedding
        for layer in self.layers:
            out = layer(hidden, rope_mats, kv_cache=kv_cache, user_id=user_id, cached_len=cached_len)
            if owned:
                hidden.deallocate(True)
            hidden, owned = out, True

        normed = self.norm(hidden)
        hidden.deallocate(True)
        if not want_logits:
            return normed
        assert self.lm_head is not None, "want_logits=True on a model built with with_lm_head=False"
        logits = self.lm_head(normed)
        normed.deallocate(True)
        return logits
