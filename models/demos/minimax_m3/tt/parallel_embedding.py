# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M3 parallel (sharded) token embedding.

Two sharding modes for the ``[vocab, emb_dim]`` table, selected by ``shard_vocab_on_sp`` (which the
model derives from ``embed_shard_2d()`` below — 2D by default, ``M3_EMBED_SHARD_VOCAB=0`` for 1D):

1. **1D (``shard_vocab_on_sp=False``)** — shard ``emb_dim`` across the TP axis, replicate across the SP
   axis. Each device stores ``[vocab, emb_dim / tp]`` (0.5724 GiB/device; ~1.72 GiB/device saved vs the
   old fully-replicated table). Forward: local emb-dim-slice lookup (no CCL) + one TP all-gather to
   rebuild the full ``emb_dim`` replicated across the TP cols. This is the DeepSeek-style approach.

2. **2D (``shard_vocab_on_sp=True``, DEFAULT)** — ALSO shard ``vocab`` across the SP axis, so each device
   stores only ``[vocab / sp, emb_dim / tp]`` (0.0715 GiB/device; ~0.50 GiB/device less than 1D,
   allocator-measured). This is the Megatron vocab-parallel pattern and needs cross-SP communication in
   forward (see below): the extra memory win costs two SP-axis CCL ops per chunk (measured perf-neutral,
   KV-PCC bit-identical to 1D). It is the default; flip to 1D only to trade the memory back for a simpler
   (CCL-free) embedding lookup.

Forward (1D):
  * ``ttnn.embedding`` on the local emb-dim slice -> ``[1, 1, s_local, emb_dim / tp]`` (no CCL);
  * TP all-gather -> ``[1, 1, s_local, emb_dim]`` replicated across the TP cols.

Forward (2D vocab-parallel), given SP-seq-sharded tokens ``[1, 1, s_local]`` (row r owns positions
``[r*s_local:(r+1)*s_local]``) and per-device weight ``[vocab/sp, emb_dim/tp]`` (row r owns vocab rows
``[r*vocab/sp:(r+1)*vocab/sp]``):
  1. SP all-gather the tokens so every row sees all ``s_total`` positions;
  2. per-row SENTINEL lookup: each vocab shard is stored padded as [zero, real slice, zero]; shift the
     token by the row's start (r*vocab_local - 1) and clamp to [0, vocab_local+1] — in-range tokens hit
     real rows, out-of-range tokens clamp onto a zero sentinel, so no output mask is needed. Index math in
     fp32 (signed subtract, exact for IDs <= 2^24); int32 would work too, but the index is cast to uint32
     either way — ttnn.embedding requires uint32;
  3. SP reduce-scatter on the seq dim: sums across the SP vocab shards (each token is resolved by exactly
     one row) AND scatters seq back to per-row shards -> ``[1, 1, s_local, emb_dim / tp]``;
  4. TP all-gather on emb_dim -> ``[1, 1, s_local, emb_dim]`` — same output contract as the 1D path.

Weight caching mirrors every other M3 weight: a per-tensor tilized ``.tensorbin`` via
``ttnn.as_tensor(cache_file_name=)``. On a cache hit ``torch_weight`` is ignored (may be ``None``). The
1D and 2D layouts use distinct cache keys (``_1d`` / ``_2d`` — neither a prefix of the other) so a stale
layout is never loaded as the other.
"""

import os
from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.minimax_m3.tt.residual import use_sharded_residual

# Cache keys for the SHARDED embed table. Distinct from the old replicated "model.embed_tokens.weight",
# and — critically — neither key is a prefix of the other: weight_cache_is_complete matches by startswith,
# so a bare "..._parallel" would false-positive on a "..._parallel_2d" .tensorbin (and vice versa). The
# _1d/_2d suffixes keep the prefixes disjoint.
EMBED_CACHE_NAME = "model.embed_tokens.weight_parallel_1d"
EMBED_CACHE_NAME_2D = "model.embed_tokens.weight_parallel_2d"

# The single toggle between the two sharding modes. 2D (vocab+hidden) is the default; the whole model —
# the embedding build (model.py) and the cache-completeness check (weight_cache.py) — reads this one
# helper, so a run is self-consistent. Override for an A/B or to build the other layout's cache:
#   M3_EMBED_SHARD_VOCAB unset / 1 / true  -> 2D (default)
#   M3_EMBED_SHARD_VOCAB = 0 / false        -> 1D
DEFAULT_SHARD_VOCAB_ON_SP = True


def embed_shard_2d() -> bool:
    """True -> 2D vocab+hidden sharding (default); False -> 1D hidden-only. See DEFAULT_SHARD_VOCAB_ON_SP."""
    v = os.getenv("M3_EMBED_SHARD_VOCAB")
    if v is None:
        return DEFAULT_SHARD_VOCAB_ON_SP
    return v.strip().lower() in ("1", "true", "yes", "on")


def cache_name_for(shard_vocab_on_sp: bool) -> str:
    return EMBED_CACHE_NAME_2D if shard_vocab_on_sp else EMBED_CACHE_NAME


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
        # Residual-stream layout (tt/residual.py). Both lookup paths already produce emb/tp naturally —
        # the 1D path from its local emb-dim slice, the 2D path out of its SP reduce-scatter — so the
        # sharded scheme is a matter of NOT running the closing TP all-gather, not of adding anything.
        self.gather_emb = not use_sharded_residual()

        tp = mesh_device.shape[self.tp_axis]
        sp = mesh_device.shape[self.sp_axis]
        assert emb_dim % tp == 0, f"emb_dim ({emb_dim}) must be divisible by tp ({tp})"

        if torch_weight is not None:
            # Accept the HF table with any number of leading singleton dims ([1, 1, vocab, emb]).
            torch_weight = torch_weight.reshape(vocab_size, emb_dim)

        # Shard emb_dim across the TP axis (both modes); in 2D also shard vocab across the SP axis.
        shard_dims = [None, None]
        shard_dims[self.tp_axis] = -1  # emb_dim
        if shard_vocab_on_sp:
            assert vocab_size % sp == 0, (
                f"2D embedding: vocab_size ({vocab_size}) must be divisible by sp ({sp}); "
                f"pad the table to a multiple of sp first."
            )
            shard_dims[self.sp_axis] = 0  # vocab
            self.vocab_local = vocab_size // sp
            # Per-SP-row sentinel-shifted vocab start (r*vocab_local - 1), replicated across the TP cols.
            # The -1 pairs with the zero-row padding below: in-range tokens map to real weight rows
            # 1..vocab_local, out-of-range tokens clamp onto the zero sentinels (rows 0 / vocab_local+1).
            # fp32: signed (the subtract goes negative for out-of-range tokens) and exact for IDs <= 2^24.
            self.vocab_start = self._build_vocab_start(mesh_device, sp, self.vocab_local)
        else:
            self.vocab_local = vocab_size
            self.vocab_start = None

        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=tuple(shard_dims))
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
            # Sentinel zero-rows: pad each per-device vocab shard on the vocab dim with one zero row at top
            # and bottom -> [vocab_local+2, emb/tp], so out-of-range tokens clamp onto a zero row (removes
            # the output mask). Done on-device at load; the cached .tensorbin stays un-padded (reused as-is).
            pad = [(0, 0)] * len(self.weight.shape)
            pad[-2] = (1, 1)  # vocab dim
            self.weight = ttnn.pad(self.weight, pad, value=0.0)

    def _build_vocab_start(self, mesh_device, sp, vocab_local):
        """Per-SP-row scalar vocab-start tensor [1,1,1,1] (value r*vocab_local - 1 on row r), replicated
        across the TP cols. fp32 for exact signed index arithmetic; the -1 is the sentinel shift."""
        starts = torch.arange(sp, dtype=torch.float32).reshape(sp, 1, 1, 1) * float(vocab_local) - 1.0
        shard_dims = [None, None]
        shard_dims[self.sp_axis] = 0  # give each SP row its own scalar
        return ttnn.from_torch(
            starts,
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=tuple(shard_dims)),
        )

    def forward(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """tokens: SP-seq-sharded (replicated across TP) uint32 indices [1, 1, s_local] -> bf16
        [1, 1, s_local, emb_dim] full hidden replicated across the TP cols, or [1, 1, s_local,
        emb_dim / tp] under a sharded residual — the M3 residual-stream contract either way."""
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

        # 1) SP all-gather the tokens (tiny uint32) so every SP row sees all s_total positions. Gather on
        #    a 4D view (dim=3), then reshape back to [1, 1, s_total] for the lookup.
        tok4d = ttnn.reshape(tokens, [1, 1, 1, tokens.shape[-1]])
        if sp > 1:
            tok4d = self.mesh_config.allgather(tok4d, self.ccl_manager, axis=self.sp_axis, dim=3)
        s_total = tok4d.shape[-1]

        # 2) Per-row SENTINEL lookup. The weight shard is [zero, real vocab slice, zero] (padded in
        #    __init__), so shifting the token by the row's start (r*vocab_local - 1) and clamping to
        #    [0, vocab_local+1] lands in-range tokens on real rows 1..vocab_local and clamps out-of-range
        #    tokens onto a zero sentinel row (0 or vocab_local+1) — no output mask needed. Index math in
        #    fp32 (signed subtract, exact for IDs <= 2^24); final index cast to uint32 (embedding needs it).
        local = ttnn.subtract(ttnn.typecast(tok4d, ttnn.float32), self.vocab_start)  # token - (r*vocab_local - 1)
        local = ttnn.minimum(ttnn.maximum(local, 0.0), float(self.vocab_local + 1))  # clamp -> sentinels
        local_idx = ttnn.reshape(ttnn.typecast(local, ttnn.uint32), [1, 1, s_total])

        emb = ttnn.embedding(local_idx, self.weight, layout=ttnn.TILE_LAYOUT, dtype=self.dtype)
        if len(emb.shape) == 3:
            emb = ttnn.unsqueeze_to_4D(emb)  # [1,1,s_total,emb_dim/tp]

        # 3) SP reduce-scatter on the seq dim: sum across vocab shards (each token resolved by exactly one
        #    SP row) AND scatter seq back to per-row shards -> [1,1,s_local,emb_dim/tp].
        if sp > 1:
            emb = self.mesh_config.reduce_scatter(emb, self.ccl_manager, dim=2, axis=self.sp_axis)

        # 4) TP all-gather on emb_dim -> [1,1,s_local,emb_dim] (full hidden, TP-replicated). Skipped
        #    under a sharded residual: step 3 already left the stream at emb_dim/tp.
        if tp > 1 and self.gather_emb:
            emb = self.mesh_config.allgather(emb, self.ccl_manager, axis=self.tp_axis, dim=3)
        return emb
