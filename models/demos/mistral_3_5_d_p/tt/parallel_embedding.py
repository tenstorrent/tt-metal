# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 parallel (sharded) token embedding.

Fixed reference: ``minimax_m3/tt/parallel_embedding.py`` (recipe §3 "Embedding"), taken as-is apart
from the residual-scheme wiring — this package's residual stream is REPLICATED across TP (the
gpt-oss donor's discipline), so the closing TP all-gather always runs, where M3 skips it under its
``emb/tp``-sharded residual.

Sharding the table is not an optimization here, it is what makes it fit: Mistral's table is
``[131072, 12288]`` = 1.61 B parameters, 3.2 GiB replicated on EVERY chip. Two modes for the
``[vocab, emb_dim]`` table:

1. **1D** (``shard_vocab_on_sp=False``, DEFAULT here) — shard ``emb_dim`` across TP, replicate across
   SP. Each device stores ``[131072, 1536]`` (0.38 GiB), an 8x saving against the replicated table.
   Forward: local emb-dim-slice lookup (no CCL) plus one TP all-gather to rebuild the full width.
   The DeepSeek-style approach.

2. **2D** (``shard_vocab_on_sp=True``) — ALSO shard ``vocab`` across SP, so each device stores
   ``[32768, 1536]`` (0.096 GiB). The Megatron vocab-parallel pattern; it needs cross-SP
   communication in the forward (below), trading two SP-axis CCL ops per chunk for a further
   ~0.28 GiB per device. ``vocab_size % sp == 0`` holds here (131072 / 4).

**Why 1D is the default, unlike the M3 reference this file comes from.** The 2D path returns WRONG
ROWS above 4096 tokens per chunk on this stack. Measured at SP=4, vocab 2048, emb 512, comparing
against ``torch.nn.functional.embedding``:

    tokens per chunk  1024   2048   3072   4096   4608   5120   8192
    2D bad rows          0      0      0      0     32     39     44
    1D bad rows          0      0      0      0      0      0      0

The corruption starts at global position ~1088 and only once the chunk exceeds 4096 tokens; the bad
rows are a mix of zeros (a token no SP row resolved) and wrong content (a token resolved to the wrong
index). The spec's ``chunk_size`` is 5120, so the 2D path is unusable as shipped — and it fails
QUIETLY: the whole-model KV PCC lands at 0.995 on layer 0 with no error raised, which is exactly how
this was found (see bringup_log.jsonl, P1). The 1D path is exact at every length tested and is what
runs. ``MISTRAL_EMBED_SHARD_VOCAB=1`` still selects 2D for anyone chasing the underlying op.

Forward (1D), given SP-seq-sharded tokens ``[1, 1, s_local]``:
  * ``ttnn.embedding`` on the local emb-dim slice -> ``[1, 1, s_local, emb_dim / tp]`` (no CCL);
  * TP all-gather -> ``[1, 1, s_local, emb_dim]``, replicated across the TP cols.

Forward (2D), with per-device weight ``[vocab/sp, emb_dim/tp]`` (SP row ``r`` owning vocab rows
``[r*vocab/sp, (r+1)*vocab/sp)``):
  1. SP all-gather the tokens so every row sees all ``s_total`` positions;
  2. per-row SENTINEL lookup: each vocab shard is stored padded as ``[zero, real slice, zero]``;
     shift the token by the row's start (``r*vocab_local - 1``) and clamp to ``[0, vocab_local+1]``,
     so in-range tokens hit real rows and out-of-range tokens clamp onto a zero sentinel — no output
     mask needed. The index math is fp32 (the subtract goes negative, and fp32 is exact for IDs up to
     2^24, comfortably above this 131072 vocab), then cast to uint32 because ``ttnn.embedding``
     requires it;
  3. SP reduce-scatter on the seq dim: sums across the vocab shards (each token is resolved by
     exactly one row) AND scatters seq back to per-row shards;
  4. TP all-gather on emb_dim -> the same output contract as the 1D path.

Weight caching mirrors every other weight here: a per-tensor tilized ``.tensorbin`` via
``ttnn.as_tensor(cache_file_name=)``; on a hit the torch tensor is ignored and may be ``None``. The
two layouts use cache keys where NEITHER is a prefix of the other (``_1d`` / ``_2d``), because the
cache-completeness check matches by ``startswith`` and a bare ``..._parallel`` would false-positive
against ``..._parallel_2d``.
"""

import os
from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

EMBED_CACHE_NAME_1D = "model.embed_tokens.weight_parallel_1d"
EMBED_CACHE_NAME_2D = "model.embed_tokens.weight_parallel_2d"

# 1D (hidden-only) is the default: the 2D path corrupts rows above 4096 tokens per chunk and the
# spec's chunk_size is 5120 (see the module docstring for the measurements).
# MISTRAL_EMBED_SHARD_VOCAB=1 selects 2D.
DEFAULT_SHARD_VOCAB_ON_SP = False


def embed_shard_2d() -> bool:
    """True -> 2D vocab+hidden sharding; False -> 1D hidden-only (the default; see the docstring)."""
    value = os.getenv("MISTRAL_EMBED_SHARD_VOCAB")
    if value is None:
        return DEFAULT_SHARD_VOCAB_ON_SP
    return value.strip().lower() in ("1", "true", "yes", "on")


def cache_name_for(shard_vocab_on_sp: bool) -> str:
    return EMBED_CACHE_NAME_2D if shard_vocab_on_sp else EMBED_CACHE_NAME_1D


class TtParallelEmbedding(LightweightModule):
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        vocab_size: int,
        emb_dim: int,
        mesh_config,
        ccl_manager,
        torch_weight: Optional[torch.Tensor] = None,
        cache_file_name: Optional[str] = None,
        dtype: ttnn.DataType = ttnn.bfloat16,
        shard_vocab_on_sp: bool = False,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.tp_axis = mesh_config.tp_axis
        self.sp_axis = mesh_config.sp_axis
        self.dtype = dtype
        self.shard_vocab_on_sp = shard_vocab_on_sp
        # This package's residual stream is replicated across TP, so the closing all-gather always
        # runs. (M3 makes this conditional on its sharded-residual scheme.)
        self.gather_emb = True

        tp = mesh_device.shape[self.tp_axis]
        sp = mesh_device.shape[self.sp_axis]
        assert emb_dim % tp == 0, f"emb_dim ({emb_dim}) must be divisible by tp ({tp})"

        if torch_weight is not None:
            # Accept the HF table with any number of leading singleton dims ([1, 1, vocab, emb]).
            torch_weight = torch_weight.reshape(vocab_size, emb_dim)

        shard_dims = [None, None]
        shard_dims[self.tp_axis] = -1  # emb_dim
        if shard_vocab_on_sp:
            assert vocab_size % sp == 0, (
                f"2D embedding: vocab_size ({vocab_size}) must be divisible by sp ({sp}); "
                f"pad the table to a multiple of sp first."
            )
            shard_dims[self.sp_axis] = 0  # vocab
            self.vocab_local = vocab_size // sp
            self.vocab_start = self._build_vocab_start(mesh_device, sp, self.vocab_local)
        else:
            self.vocab_local = vocab_size
            self.vocab_start = None

        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(shard_dims))
        self.weight = ttnn.as_tensor(
            torch_weight,  # ignored on a cache hit; must be present to populate the cache
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
            cache_file_name=cache_file_name,
        )
        if shard_vocab_on_sp:
            # Sentinel zero-rows: pad each per-device vocab shard with one zero row top and bottom ->
            # [vocab_local+2, emb/tp], so out-of-range tokens clamp onto a zero row and no output
            # mask is needed. Done on-device at load; the cached .tensorbin stays un-padded.
            pad = [(0, 0)] * len(self.weight.shape)
            pad[-2] = (1, 1)  # vocab dim
            self.weight = ttnn.pad(self.weight, pad, value=0.0)

    def _build_vocab_start(self, mesh_device, sp, vocab_local):
        """Per-SP-row scalar vocab-start tensor [1,1,1,1] (value ``r*vocab_local - 1`` on row r),
        replicated across the TP cols. fp32 for exact signed index arithmetic; the -1 is the sentinel
        shift that pairs with the zero-row padding."""
        starts = torch.arange(sp, dtype=torch.float32).reshape(sp, 1, 1, 1) * float(vocab_local) - 1.0
        shard_dims = [None, None]
        shard_dims[self.sp_axis] = 0  # give each SP row its own scalar
        return ttnn.from_torch(
            starts,
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(shard_dims)
            ),
        )

    def forward(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """tokens: SP-seq-sharded (TP-replicated) uint32 indices ``[1, 1, s_local]`` -> bf16
        ``[1, 1, s_local, emb_dim]``, full hidden replicated across the TP cols."""
        if not self.shard_vocab_on_sp:
            emb = ttnn.embedding(tokens, self.weight, layout=ttnn.TILE_LAYOUT, dtype=self.dtype)
            if len(emb.shape) == 3:
                emb = ttnn.unsqueeze_to_4D(emb)
            tp = self.mesh_device.shape[self.tp_axis]
            if tp > 1 and self.gather_emb:
                emb = self.mesh_config.allgather(emb, self.ccl_manager, axis=self.tp_axis, dim=3)
            return emb
        return self._forward_2d(tokens)

    def _forward_2d(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        sp = self.mesh_device.shape[self.sp_axis]
        tp = self.mesh_device.shape[self.tp_axis]

        # 1) SP all-gather the tokens (tiny uint32) so every SP row sees all s_total positions.
        tok4d = ttnn.reshape(tokens, [1, 1, 1, tokens.shape[-1]])
        if sp > 1:
            tok4d = self.mesh_config.allgather(tok4d, self.ccl_manager, axis=self.sp_axis, dim=3)
        s_total = tok4d.shape[-1]

        # 2) Per-row SENTINEL lookup (see the module docstring).
        local = ttnn.subtract(ttnn.typecast(tok4d, ttnn.float32), self.vocab_start)
        local = ttnn.minimum(ttnn.maximum(local, 0.0), float(self.vocab_local + 1))
        local_idx = ttnn.reshape(ttnn.typecast(local, ttnn.uint32), [1, 1, s_total])

        emb = ttnn.embedding(local_idx, self.weight, layout=ttnn.TILE_LAYOUT, dtype=self.dtype)
        if len(emb.shape) == 3:
            emb = ttnn.unsqueeze_to_4D(emb)  # [1, 1, s_total, emb_dim/tp]

        # 3) SP reduce-scatter on the seq dim: sum across the vocab shards AND scatter seq back.
        if sp > 1:
            emb = self.mesh_config.reduce_scatter(emb, self.ccl_manager, dim=2, axis=self.sp_axis)

        # 4) TP all-gather on emb_dim -> full hidden, TP-replicated.
        if tp > 1 and self.gather_emb:
            emb = self.mesh_config.allgather(emb, self.ccl_manager, axis=self.tp_axis, dim=3)
        return emb
