# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The whole Qwen3.8-27B text tower on the mesh: embedding, 64 hybrid layers, final norm, LM head.

The layer stack is where a hybrid model's two model-scale hazards live, and both are handled by
``Qwen35TextConfig`` rather than by index arithmetic here:

* **per-layer type dispatch** — ``layer_types`` decides which mixer each ``DecoderLayer`` builds;
* **the KV cache is sized for the 16 full-attention layers, not for 64** — ``cfg.kv_slot`` maps a
  global layer index to its cache row, and the 48 GDN layers get a ``GdnState`` each instead.

Sequence parallelism: the token ids arrive already sharded across the SP rows, so the residual
stream is SP-sharded from the embedding onward and no layer has to know about it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
from loguru import logger

import ttnn

from ..config import MeshConfig
from ..reference.config import Qwen35TextConfig
from ..utils.general_utils import get_cache_file_name
from ..utils.substate import substate
from .caches import PrefillCaches, allocate_prefill_caches
from .context import ChunkContext
from .layer import DecoderLayer
from .lm_head import LMHead
from .parallel_embedding import ParallelEmbedding
from .rms_norm import RMSNorm
from .rope import RotarySetup


class Qwen35Model:
    def __init__(
        self,
        mesh_device,
        cfg: Qwen35TextConfig,
        state_dict: dict[str, torch.Tensor],
        *,
        mesh_config: MeshConfig,
        ccl_manager,
        weight_dtype=ttnn.bfloat8_b,
        activation_dtype=ttnn.bfloat16,
        cache_dtype=ttnn.bfloat8_b,
        tensor_cache_path: Optional[str] = None,
        layer_indices: Optional[list[int]] = None,
        build_lm_head: bool = True,
    ) -> None:
        """``layer_indices`` are GLOBAL indices; the default is the whole stack. A short list is a
        **reduced** run in the recipe's sense — a diagnostic, never a grade."""
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.cache_dtype = cache_dtype
        self.layer_indices = list(layer_indices if layer_indices is not None else range(cfg.num_hidden_layers))
        self.rope = RotarySetup(mesh_device, cfg, mesh_config)

        # A tilized weight is different BYTES per mesh shape and per dtype, so the cache directory
        # has to carry both. Without this a (2,2) coverage run would happily load the (8,4) shards
        # and produce a plausible, wrong answer instead of a miss.
        if tensor_cache_path is not None:
            rows, cols = mesh_config.mesh_shape
            tensor_cache_path = str(Path(tensor_cache_path) / f"{rows}x{cols}_{weight_dtype}")
            Path(tensor_cache_path).mkdir(parents=True, exist_ok=True)

        self.embedding = ParallelEmbedding(
            mesh_device,
            cfg.vocab_size,
            cfg.hidden_size,
            substate(state_dict, "embed_tokens"),
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            tensor_cache_path=tensor_cache_path,
        )
        self.layers = [
            DecoderLayer(
                mesh_device,
                cfg,
                substate(state_dict, f"layers.{idx}"),
                idx,
                mesh_config=mesh_config,
                ccl_manager=ccl_manager,
                weight_dtype=weight_dtype,
                activation_dtype=activation_dtype,
                cache_dtype=cache_dtype,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, f"layers.{idx}"),
            )
            for idx in self.layer_indices
        ]
        self.norm = RMSNorm(
            mesh_device,
            cfg.hidden_size,
            cfg.rms_norm_eps,
            substate(state_dict, "norm"),
            mesh_config=mesh_config,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "norm"),
        )
        self.lm_head = (
            LMHead(
                mesh_device,
                cfg.vocab_size,
                cfg.hidden_size,
                substate(state_dict, "lm_head"),
                mesh_config=mesh_config,
                weight_dtype=weight_dtype,
                tensor_cache_path=tensor_cache_path,
            )
            if build_lm_head
            else None
        )
        logger.info(
            f"Qwen3.5 model built on {mesh_config}: {len(self.layers)} layers "
            f"({sum(cfg.is_full_attention(i) for i in self.layer_indices)} full-attention, "
            f"{sum(not cfg.is_full_attention(i) for i in self.layer_indices)} gated-deltanet)"
        )

    # --- caches -------------------------------------------------------------------------
    def allocate_caches(self, max_seq_len: int, num_users: int = 1) -> PrefillCaches:
        return allocate_prefill_caches(
            self.mesh_device,
            self.cfg,
            mesh_config=self.mesh_config,
            max_seq_len=max_seq_len,
            num_users=num_users,
            layer_indices=self.layer_indices,
            cache_dtype=self.cache_dtype,
        )

    # --- inputs -------------------------------------------------------------------------
    def shard_tokens(self, token_ids: torch.Tensor) -> ttnn.Tensor:
        """``[chunk]`` or ``[1, chunk]`` host token ids -> SP-sharded uint32 device tensor.

        Row r receives chunk tokens ``[r*s_local, (r+1)*s_local)`` — contiguous within the chunk,
        which is the same block the KV cache's block-cyclic writer and the RoPE table assume.
        """
        ids = token_ids.reshape(1, 1, 1, -1)
        total = ids.shape[-1]
        sp = self.mesh_config.sp
        assert total % sp == 0, f"chunk {total} must split across sp={sp}"
        return ttnn.from_torch(
            ids,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.mesh_config.sequence_parallel(self.mesh_device, seq_dim=3),
        )

    def chunk_context(
        self, *, start_pos: int, chunk_size: int, caches: Optional[PrefillCaches], user_id: int = 0
    ) -> ChunkContext:
        cos, sin = self.rope.chunk_mats(start_pos, chunk_size)
        return ChunkContext(cos=cos, sin=sin, caches=caches, user_id=user_id, cached_len=start_pos)

    # --- forward ------------------------------------------------------------------------
    def forward(
        self,
        hidden: ttnn.Tensor,
        ctx: ChunkContext,
        *,
        skip_lm_head: bool = True,
        on_layer_complete: Optional[Callable[[int], None]] = None,
    ) -> ttnn.Tensor:
        """``hidden`` ``[1, 1, s_local, hidden]`` -> final-norm output, or logits.

        ``on_layer_complete(global_layer_idx)`` is the per-layer seam the KV-PCC harness reads the
        cache through; it is a no-op by default.
        """
        for layer in self.layers:
            hidden = layer(hidden, ctx)
            if on_layer_complete is not None:
                on_layer_complete(layer.layer_idx)

        out = self.norm(hidden)
        hidden.deallocate(True)
        if skip_lm_head or self.lm_head is None:
            return out
        logits = self.lm_head(out)
        out.deallocate(True)
        return logits

    def prefill_chunk(
        self,
        token_ids: torch.Tensor,
        *,
        start_pos: int,
        caches: Optional[PrefillCaches],
        user_id: int = 0,
        skip_lm_head: bool = True,
        on_layer_complete: Optional[Callable[[int], None]] = None,
    ) -> ttnn.Tensor:
        """Embed one chunk of token ids and push it through the stack."""
        tokens = self.shard_tokens(token_ids)
        hidden = self.embedding(tokens)
        tokens.deallocate(True)
        ctx = self.chunk_context(start_pos=start_pos, chunk_size=token_ids.numel(), caches=caches, user_id=user_id)
        out = self.forward(hidden, ctx, skip_lm_head=skip_lm_head, on_layer_complete=on_layer_complete)
        ctx.cos.deallocate(True)
        ctx.sin.deallocate(True)
        return out
