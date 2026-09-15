# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B TTNN prefill model (tt-blaze#4148).

    embedding -> [TtLlamaDecoderLayer] * 32 -> final RMSNorm

**No LM head.** Prefill's product is the populated KV cache, not logits; only decode needs the
projection, and at 128256 x 4096 it is 525M parameters this model would carry and never use. The
reference keeps one (``reference/model.py``) because *its* acceptance criterion is a logits
comparison against HF; this one is graded on hidden states and KV.

**The embedding is TP-sharded on the feature dim**, which is what produces the residual stream in
the layout every layer wants. ``embed_tokens.weight`` is ``[vocab, emb]``; sharding ``emb`` across
the TP axis leaves each chip ``[vocab, emb/tp]``, so ``ttnn.embedding`` emits ``[1, 1, seq_local,
emb/tp]`` directly — the TP-sharded, SP-sharded residual, with no collective and no slice. It also
drops the table from 1.05 GB to 131 MB per chip at TP=8. ``gpt_oss_d_p/tt/model.py`` replicates its
embedding and carries a TODO to shard it; that model's residual is replicated full-width, so
sharding there is purely a memory win, whereas here it is also the layout.

Tokens arrive SP-sharded on the sequence dim and replicated across TP, so each chip embeds its own
sequence slab out of its own feature columns.

The chunk loop lives in ``tt/tt_prefill_runtime.py``; this module is one chunk's forward.

Kept free of reference-model and safetensors imports; the import-light contract is asserted by
``tests/unit/test_scaffold.py``.
"""

from pathlib import Path
from typing import Callable, Optional

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.tt.decoder import TtLlamaDecoderLayer
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import Llama31KVCache
from models.demos.llama_3p1_8b_d_p.tt.rms_norm import TtLlamaRMSNorm


class TtLlamaPrefillModel(LightweightModule):
    """The 32-layer prefill stack: embedding, decoder layers, final norm."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        mesh_config,
        state_dict: Optional[dict] = None,
        num_layers: int = Llama31_8BConfig.NUM_LAYERS,
        vocab_size: int = Llama31_8BConfig.VOCAB_SIZE,
        emb_dim: int = Llama31_8BConfig.EMB_SIZE,
        hidden_dim: int = Llama31_8BConfig.INTERMEDIATE_SIZE,
        n_heads: int = Llama31_8BConfig.NUM_ATTENTION_HEADS,
        n_kv_heads: int = Llama31_8BConfig.NUM_KEY_VALUE_HEADS,
        head_dim: int = Llama31_8BConfig.HEAD_DIM,
        rms_norm_eps: float = Llama31_8BConfig.RMS_NORM_EPS,
        first_layer_idx: int = 0,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        activations_dtype: ttnn.DataType = ttnn.bfloat16,
        weights_dtype: ttnn.DataType = ttnn.bfloat16,
        weight_cache_path: Optional[Path] = None,
    ):
        """
        Args:
            state_dict: a HuggingFace Llama-3.1-8B state dict, keyed as the checkpoint ships it
                (``model.embed_tokens.weight``, ``model.layers.N.*``, ``model.norm.weight``).
                ``lm_head.weight`` is ignored if present. Random weights when omitted or empty,
                which is shape and memory bring-up only, never PCC.
            num_layers: layers this rank builds. With ``first_layer_idx``, this is how a pipeline
                rank builds only its own slice.
            first_layer_idx: the GLOBAL index of this rank's first layer. Layers are given their
                global index, so their KV-cache slots and the layer-completion sink agree with
                every other rank — a rank-local index makes each rank's layer k collide.
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.num_layers = num_layers
        self.first_layer_idx = first_layer_idx
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.emb_dim_per_chip = mesh_config.shard_size(emb_dim)
        self.sp_axis = mesh_config.sp_axis

        has_weights = bool(state_dict)
        if not has_weights:
            logger.warning("TtLlamaPrefillModel built with random weights — bring-up only, not valid for PCC")

        # --- token embedding, TP-sharded on the feature dim (see the module docstring) ---
        if has_weights:
            embedding_weight = state_dict["model.embed_tokens.weight"]
            if tuple(embedding_weight.shape) != (vocab_size, emb_dim):
                raise ValueError(
                    f"model.embed_tokens.weight has shape {tuple(embedding_weight.shape)}, expected "
                    f"{(vocab_size, emb_dim)}"
                )
            embedding_weight = embedding_weight.reshape(1, 1, vocab_size, emb_dim)
        else:
            embedding_weight = torch.randn(1, 1, vocab_size, emb_dim) * 0.02

        self.embedding_weight = ttnn.as_tensor(
            embedding_weight,
            dtype=weights_dtype,
            device=mesh_device,
            # ROW_MAJOR: ttnn.embedding gathers rows, it does not matmul.
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_config.column_parallel(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=str(weight_cache_path / "embed_tokens") if weight_cache_path else None,
        )

        # --- decoder layers ---
        self.layers = [
            TtLlamaDecoderLayer(
                mesh_device=mesh_device,
                mesh_config=mesh_config,
                torch_weights=(
                    TtLlamaDecoderLayer.weights_from_layer_state_dict(state_dict, first_layer_idx + offset)
                    if has_weights
                    else None
                ),
                layer_idx=first_layer_idx + offset,
                # Global index for identity, rank-local for the cache slot: this rank's cache holds
                # exactly its own num_layers slots per user, so it must not be indexed globally.
                cache_layer_idx=offset,
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
                num_links=num_links,
                topology=topology,
                activations_dtype=activations_dtype,
                weights_dtype=weights_dtype,
                weight_cache_path=weight_cache_path,
                cache_name_prefix=f"layers.{first_layer_idx + offset}" if weight_cache_path else None,
            )
            for offset in range(num_layers)
        ]

        # --- final norm ---
        self.norm = TtLlamaRMSNorm(
            mesh_device=mesh_device,
            mesh_config=mesh_config,
            torch_weight=state_dict["model.norm.weight"] if has_weights else None,
            emb_dim=emb_dim,
            eps=rms_norm_eps,
            num_links=num_links,
            topology=topology,
            weights_dtype=weights_dtype,
            weight_cache_path=weight_cache_path,
            cache_name_prefix="final" if weight_cache_path else None,
        )

        logger.info(
            f"TtLlamaPrefillModel: layers {first_layer_idx}..{first_layer_idx + num_layers - 1}, "
            f"vocab={vocab_size} ({vocab_size}x{self.emb_dim_per_chip}/chip embedding), "
            f"residual {self.emb_dim_per_chip}/chip"
        )

    def embed(self, tt_tokens: ttnn.Tensor) -> ttnn.Tensor:
        """SP-sharded uint32 token IDs -> the TP-sharded residual ``[1, 1, seq_local, emb/tp]``.

        ``tt_tokens`` is already on device, per-chip shape ``(1, 1, seq_local)``: the sequence dealt
        to the SP rows in the block-cyclic order the KV writer and ``rope.build_indexed_rope``
        assume, and replicated across the TP columns. That is the layout
        ``tt_prefill_runtime.make_chunk_input`` builds *and* the layout the request-mode H2D socket
        delivers, so both feed one code path.

        The order is the plain contiguous split only when the chunk starts on a chunk boundary; a
        continuation resuming mid-chunk arrives rotated (``kv_cache.rotated_chip_positions``). This
        method does not care either way — it embeds row-wise — but the distinction is why it must
        not "helpfully" reorder anything.

        The caller keeps ownership of ``tt_tokens``.
        """
        # bf16, not bf8: the residual stream has to keep its dynamic range. bf8's per-tile shared
        # exponent crushes small channels as soon as Llama's massive activation channels appear,
        # and the residual is added to 32 times.
        embedded = ttnn.embedding(tt_tokens, self.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        if len(embedded.shape) == 3:
            embedded = ttnn.unsqueeze_to_4D(embedded)
        if embedded.shape[-1] != self.emb_dim_per_chip:
            raise RuntimeError(
                f"embedding produced {embedded.shape[-1]}/chip, expected emb_dim/tp = "
                f"{self.emb_dim_per_chip}; the embedding weight must be TP-sharded on the feature dim"
            )
        return embedded

    def forward(
        self,
        x: ttnn.Tensor,
        rope_mats,
        transformation_mat,
        *,
        kv_cache: Optional[Llama31KVCache] = None,
        ccl_manager=None,
        user_id: int = 0,
        cached_len: int = 0,
        indexed_rope: bool = True,
        apply_final_norm: bool = True,
        on_layer_complete: Optional[Callable[[int], None]] = None,
    ) -> ttnn.Tensor:
        """One chunk through this rank's layers. TP-sharded residual in.

        Args:
            x: the TP-sharded residual ``[1, 1, seq_local, emb/tp]``, i.e. :meth:`embed`'s output on
                the first rank, or the activation received over the D2D socket on a later one.
            cached_len: valid prefix already in the cache before this chunk (0 for the first).
            apply_final_norm: run the final norm and return replicated full width. False on a
                non-last pipeline rank, which forwards the raw TP-sharded residual to the next rank.
            on_layer_complete: ``fn(global_layer_idx)``, called after each layer. This is the seam
                the disaggregated pipeline hangs per-layer KV migration off, and it is given the
                GLOBAL index because the sink keys on it across ranks.

        Returns:
            The final-normed replicated ``[1, 1, seq_local, emb_dim]`` when ``apply_final_norm``,
            otherwise the TP-sharded residual.
        """
        for layer in self.layers:
            x = layer(
                x,
                rope_mats,
                transformation_mat,
                kv_cache=kv_cache,
                ccl_manager=ccl_manager,
                user_id=user_id,
                cached_len=cached_len,
                indexed_rope=indexed_rope,
            )
            if on_layer_complete is not None:
                on_layer_complete(layer.layer_idx)

        if not apply_final_norm:
            return x

        out = self.norm(x, ccl_manager)
        ttnn.deallocate(x)
        return out
