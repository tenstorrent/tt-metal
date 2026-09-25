# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Whole Mistral-Medium-3.5 language model for prefill on SP=8 x TP=4 (outline after
gpt_oss_d_p/tt/model.py):

    tokens -> Embedding -> DecoderLayer x N -> final RMSNorm -> LMHead

Layers are built one at a time from a weight source (``tt/weights.py``) so the host never holds more
than a few layers. Layer weights and tensor-cache keys use the GLOBAL layer index; the KV-cache slot
uses the LOCAL index (0-based within this instance). The model is uniform (every layer is the same
dense GQA block), so there is no per-layer type dispatch.
"""

import ttnn

from .common import cache_name
from .embedding import Embedding
from .layer import DecoderLayer
from .lm_head import LMHead
from .precision import precision
from .rms_norm import RMSNorm


class Model:
    def __init__(
        self,
        mesh_device,
        mesh_config,
        ccl_manager,
        cfg,
        weights,
        *,
        dtypes: dict,
        num_layers=None,
        first_layer_idx: int = 0,
        with_embedding: bool = True,
        with_lm_head: bool = True,
        embed_shard_vocab: bool = False,
        tensor_cache_path=None,
        on_layer_built=None,
    ):
        """``weights``: a source from ``tt/weights.py``. ``dtypes``: ttnn dtypes resolved from the spec
        (``attention``, ``mlp_gate``, ``mlp_up``, ``mlp_down``, ``weights_default``)."""
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.cfg = cfg
        self.num_layers = cfg.num_hidden_layers if num_layers is None else num_layers
        self.first_layer_idx = first_layer_idx
        self.residual_dtype = precision().residual_dtype

        self.embedding = (
            Embedding(
                mesh_device,
                mesh_config,
                ccl_manager,
                weights.embedding(),
                shard_vocab_on_sp=embed_shard_vocab,
                tensor_cache_path=cache_name(tensor_cache_path, "embed_tokens"),
            )
            if with_embedding
            else None
        )
        self.layers = []
        for global_idx, sd in weights.iter_layers(range(first_layer_idx, first_layer_idx + self.num_layers)):
            self.layers.append(
                DecoderLayer(
                    mesh_device,
                    mesh_config,
                    ccl_manager,
                    cfg,
                    sd,
                    layer_idx=global_idx - first_layer_idx,
                    dtypes=dtypes,
                    tensor_cache_path=cache_name(tensor_cache_path, f"layers.{global_idx}"),
                )
            )
            if on_layer_built is not None:
                on_layer_built(global_idx)
        self.norm = RMSNorm(
            mesh_device,
            mesh_config,
            ccl_manager,
            weights.final_norm(),
            cfg.rms_norm_eps,
            tensor_cache_path=cache_name(tensor_cache_path, "norm"),
        )
        self.lm_head = (
            LMHead(
                mesh_device,
                mesh_config,
                weights.lm_head(),
                dtype=dtypes["weights_default"],
                tensor_cache_path=cache_name(tensor_cache_path, "lm_head"),
            )
            if with_lm_head
            else None
        )

    def embed(self, tokens):
        """SP-sharded uint32 tokens -> sharded residual ``[1, 1, s_local, hidden/tp]``."""
        x = self.embedding(tokens)
        if self.residual_dtype != x.dtype:
            x_bf16, x = x, ttnn.typecast(x, self.residual_dtype)
            x_bf16.deallocate(True)
        return x

    def forward_layers(self, x, rope, *, kv_cache=None, user_id=0, cached_len=0, on_layer_complete=None):
        """Run every decoder layer on the sharded residual; consumes ``x``. ``on_layer_complete(i)`` is
        called with the LOCAL layer index after each layer (the per-layer KV seam)."""
        for i, layer in enumerate(self.layers):
            y = layer(x, rope, kv_cache=kv_cache, user_id=user_id, cached_len=cached_len)
            x.deallocate(True)
            x = y
            if on_layer_complete is not None:
                on_layer_complete(i)
        return x

    def head(self, x):
        """Final norm + LM head on the sharded residual: ``(normed [1,1,s_local,hidden], logits
        [1,1,s_local,vocab/tp])``; logits are None without an LM head."""
        normed = self.norm(x)
        logits = self.lm_head(normed) if self.lm_head is not None else None
        return normed, logits

    def __call__(self, tokens, rope, *, kv_cache=None, user_id=0, cached_len=0, skip_lm_head=False):
        x = self.forward_layers(self.embed(tokens), rope, kv_cache=kv_cache, user_id=user_id, cached_len=cached_len)
        if skip_lm_head:
            return x
        normed, logits = self.head(x)
        x.deallocate(True)
        if logits is None:
            return normed
        normed.deallocate(True)
        return logits

    @staticmethod
    def token_tensor(token_ids, mesh_device, mesh_config):
        """Host token ids ``[S]`` -> per chip ``[1, 1, S/sp]`` uint32 ROW_MAJOR (row r holds the contiguous
        slice ``[r*S/sp, (r+1)*S/sp)``, replicated across TP) — the layout the embedding consumes."""
        import torch

        ids = torch.as_tensor(token_ids, dtype=torch.int32).reshape(1, 1, -1)
        return ttnn.from_torch(
            ids,
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_config.mapper(mesh_device, sp_dim=2),
        )
