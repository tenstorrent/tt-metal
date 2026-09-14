# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sharded token embedding, in both of the sharding modes the recipe's M2 row asks for.

The table is ``[128256, 4096]`` — 1.05 GiB in bf16, which is why it is not replicated.

**1D** (``shard_vocab_on_sp=False``): shard ``emb_dim`` across TP, replicate ``vocab`` across SP.
Each device stores ``[vocab, emb/tp]`` = 263 MiB. Forward is a local lookup plus one TP all-gather.
CCL-free on the SP axis and the simpler of the two — the default here.

**2D** (``shard_vocab_on_sp=True``): also shard ``vocab`` across SP, so each device stores
``[vocab/sp, emb/tp]`` = 33 MiB. Forward needs cross-SP communication: all-gather the (tiny uint32)
tokens so every row sees all positions, do a SENTINEL lookup against that row's vocab slice, then
SP reduce-scatter on the sequence dim — which both sums across the vocab shards (each token is
resolved by exactly one row) and puts the sequence back to per-row shards.

The sentinel trick: each vocab shard is stored padded as ``[zero row, real slice, zero row]``, and
the token is shifted by ``r*vocab_local - 1`` and clamped to ``[0, vocab_local+1]``. In-range tokens
land on real rows 1..vocab_local; out-of-range tokens clamp onto a zero row. That removes the output
mask entirely. The index arithmetic is fp32 because the shift goes negative for out-of-range tokens
and fp32 is exact for ids <= 2^24 (vocab is 128256).

``vocab_size`` 128256 is divisible by sp=8 (16032 each), so 2D needs no table padding here.

The two cache keys are ``_1d`` / ``_2d`` and neither is a prefix of the other, so a stale layout can
never be loaded as the other one.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

EMBED_CACHE_1D = "embed_tokens_parallel_1d"
EMBED_CACHE_2D = "embed_tokens_parallel_2d"


def embed_shard_2d() -> bool:
    """``LLAMA_EMBED_SHARD_VOCAB=1`` selects the 2D (vocab+hidden) layout; default is 1D."""
    return os.getenv("LLAMA_EMBED_SHARD_VOCAB", "0").strip().lower() in ("1", "true", "yes", "on")


def cache_name_for(shard_vocab_on_sp: bool) -> str:
    return EMBED_CACHE_2D if shard_vocab_on_sp else EMBED_CACHE_1D


class ParallelEmbedding(LightweightModule):
    def __init__(
        self,
        mesh_device,
        vocab_size: int,
        emb_dim: int,
        mesh_config,
        ccl_manager,
        torch_weight: Optional[torch.Tensor] = None,
        cache_file_name: Optional[str] = None,
        dtype=ttnn.bfloat16,
        shard_vocab_on_sp: bool = False,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.dtype = dtype
        self.shard_vocab_on_sp = shard_vocab_on_sp
        tp, sp = mesh_config.tp, mesh_config.sp
        assert emb_dim % tp == 0, f"emb_dim ({emb_dim}) must be divisible by tp ({tp})"

        if torch_weight is not None:
            torch_weight = torch_weight.reshape(vocab_size, emb_dim)

        shard_dims = [None, None]
        shard_dims[mesh_config.tp_axis] = -1
        if shard_vocab_on_sp:
            assert vocab_size % sp == 0, f"2D embedding needs vocab ({vocab_size}) divisible by sp ({sp})"
            shard_dims[mesh_config.sp_axis] = 0
            self.vocab_local = vocab_size // sp
            self.vocab_start = self._build_vocab_start(mesh_device, mesh_config, sp, self.vocab_local)
        else:
            self.vocab_local = vocab_size
            self.vocab_start = None

        self.weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=tuple(shard_dims)),
            cache_file_name=cache_file_name,
        )
        if shard_vocab_on_sp:
            # Sentinel zero rows, added on device at load. The cached .tensorbin stays unpadded.
            pad = [(0, 0)] * len(self.weight.shape)
            pad[-2] = (1, 1)
            self.weight = ttnn.pad(self.weight, pad, value=0.0)

    @staticmethod
    def _build_vocab_start(mesh_device, mesh_config, sp, vocab_local):
        starts = torch.arange(sp, dtype=torch.float32).reshape(sp, 1, 1, 1) * float(vocab_local) - 1.0
        dims = [None, None]
        dims[mesh_config.sp_axis] = 0
        return ttnn.from_torch(
            starts,
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=tuple(dims)),
        )

    def forward(self, tokens):
        """SP-sharded uint32 ids ``[1, 1, s_local]`` -> bf16 ``[1, 1, s_local, emb_dim]``,
        full hidden replicated across the TP columns (the residual-stream contract)."""
        if self.shard_vocab_on_sp:
            return self._forward_2d(tokens)
        emb = ttnn.embedding(tokens, self.weight, layout=ttnn.TILE_LAYOUT, dtype=self.dtype)
        if len(emb.shape) == 3:
            emb = ttnn.unsqueeze_to_4D(emb)
        if self.mesh_config.tp > 1:
            gathered = self.mesh_config.all_gather(emb, self.ccl_manager, axis=self.mesh_config.tp_axis, dim=3)
            emb.deallocate(True)
            return gathered
        return emb

    def _forward_2d(self, tokens):
        mc, sp, tp = self.mesh_config, self.mesh_config.sp, self.mesh_config.tp

        tok4d = ttnn.reshape(tokens, [1, 1, 1, tokens.shape[-1]])
        if sp > 1:
            tok4d = mc.all_gather(tok4d, self.ccl_manager, axis=mc.sp_axis, dim=3)
        s_total = tok4d.shape[-1]

        local = ttnn.subtract(ttnn.typecast(tok4d, ttnn.float32), self.vocab_start)
        local = ttnn.minimum(ttnn.maximum(local, 0.0), float(self.vocab_local + 1))
        local_idx = ttnn.reshape(ttnn.typecast(local, ttnn.uint32), [1, 1, s_total])

        emb = ttnn.embedding(local_idx, self.weight, layout=ttnn.TILE_LAYOUT, dtype=self.dtype)
        if len(emb.shape) == 3:
            emb = ttnn.unsqueeze_to_4D(emb)
        if sp > 1:
            emb = mc.reduce_scatter(emb, self.ccl_manager, dim=2, axis=mc.sp_axis)
        if tp > 1:
            emb = mc.all_gather(emb, self.ccl_manager, axis=mc.tp_axis, dim=3)
        return emb
