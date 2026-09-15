# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B chunked-prefill KV cache (tt-blaze#4141).

One packed pair of persistent device caches (K and V), each per-chip shape
``[num_users * num_layers, num_kv_heads_per_chip, seq_local, head_dim]`` in ``bfloat8_b``, DRAM
NdShard, written by ``ttnn.experimental.deepseek_prefill.update_padded_kv_cache``.

The batch dim is **user-major**: ``slot = user_id * num_layers + layer_idx``, so each user's layers
stay contiguous and the packing matches ``update_padded_kv_cache``'s ``slot_idx`` / ``layer_idx``
indexing. Both ``num_layers`` and ``layer_idx`` here are **rank-local** — a pipeline rank holding
layers 24..31 allocates 8 slots per user and indexes them 0..7, so it does not pay for the 24
layers it never fills (see ``tt/decoder.py``'s ``cache_layer_idx``).
At TP=8 the 8 KV heads shard one-per-chip across the TP columns, so a chip's cache holds
exactly one head and a KV chunk for a given layer lives on exactly one chip — which is what
collapses the migration layer's DeviceGroup to a single node. The sequence is block-cyclic over the
4 SP rows.

``num_kv_heads_per_chip`` is 1 in that production layout and exists as a parameter only so SP > 1
is reachable on an 8-chip loudbox: TP=8 with SP=4 needs 32 chips, so validating the block-cyclic
write and the ring cache read on one box means running a lower TP (e.g. 4x2 at TP=2, 4 heads/chip).

Adapted from ``gpt_oss_d_p/tt/attention/kv_cache.py``, which has the same layout for GQA.
**Llama needs only the single packed cache**: every one of the 32 layers is full causal attention,
so there is no ``layer_types``, no bounded sliding-window split, and no circular slab write. That
removal is most of the difference between the two files — gpt-oss alternates sliding (window 128)
and full attention and therefore carries a second, smaller circular cache plus the
``build_layer_map`` / ``sliding_capacity_tokens`` / ``layer_view`` remap machinery. Reproducing it
here would be dead code whose only effect is to make the address table need a per-layer branch.

``NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK`` must stay in sync with **both** the DRAM NdShard below and
the address-table bank walk on the migration side; it is re-exported for that reason.
"""

from dataclasses import dataclass

import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.adapter import KvCaches
from models.demos.common.prefill.runners.migration import get_num_dram_banks
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig

# Must match the DRAM NdShard in allocate_kv_cache AND the address-table bank walk.
NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32


def validate_chunk_layout(max_seq_len: int, chunk_size: int, sp: int) -> None:
    """Enforce the two divisibility rules the block-cyclic layout depends on.

    * ``max_seq_len % chunk_size == 0`` — the cache is an exact number of chunk slabs, so no chunk
      straddles the end of the cache.
    * ``chunk_size % (sp * NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK) == 0`` — each chip receives
      ``chunk_size // sp`` tokens per chunk, and that per-chip slab has to be a whole number of
      32-token DRAM shards.

    Checked rather than assumed: both are satisfied by the served configuration, and violating
    either does not crash. It places tokens in the wrong bank or the wrong chip, which surfaces much
    later as a KV PCC failure at some interior position.
    """
    if max_seq_len % chunk_size:
        raise ValueError(f"max_seq_len ({max_seq_len}) must be a multiple of chunk_size ({chunk_size})")
    granularity = sp * NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    if chunk_size % granularity:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be a multiple of sp * "
            f"{NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK} ({granularity}); each chip's per-chunk slab must "
            f"be a whole number of {NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK}-token DRAM shards"
        )


def slot_index(user_id: int, layer_idx: int, num_layers: int) -> int:
    """User-major packing: ``user_id * num_layers + layer_idx``.

    Spelled out as a function because the same expression appears on the migration side's address
    table, and the two must not drift. Layer-major packing (``layer * num_users + user``) would also
    "work" for a single user, which is exactly how such a mismatch survives bring-up.
    """
    return user_id * num_layers + layer_idx


def rotated_chip_positions(kv_actual: int, sp: int, chunk_local: int) -> list[list[int]]:
    """Which absolute position each chip's local row carries, for a chunk starting at ``kv_actual``.

    ``positions[c][r]`` is the global token position that SP row ``c``'s ``r``-th row holds. This is
    the host-side inverse of the ``update_padded_kv_cache`` writer's ``update_idxt`` staircase: each
    chip writes ``chunk_local`` rows starting at ``update_idxt``, and cache row ``lr`` on chip ``c``
    holds position ``(lr // chunk_local) * chunk_size_global + c * chunk_local + (lr % chunk_local)``.
    Chips before the boundary chip advance a whole slab, the boundary chip advances by the pad
    offset, and chips after it stay at the slab base.

    **Why a chunk is not simply dealt out contiguously.** When ``kv_actual`` is a multiple of
    ``chunk_size`` this returns exactly the contiguous split — row ``c`` gets
    ``[c * chunk_local, (c+1) * chunk_local)`` — which is why prompt order works for every
    single-turn request and why getting this wrong is invisible until the first continuation. At any
    other 32-aligned start the map is a rotation of that by ``kv_actual % chunk_size``, *plus* a
    further rotation within the single boundary chip, which is the part a plain rotation misses.

    The union over all ``(c, r)`` covers ``[kv_actual, kv_actual + sp * chunk_local)`` exactly and is
    increasing in ``r`` on every chip, so each chip's real tokens stay a prefix and its pad a suffix
    — which is what makes the pad tail inert under causality with no extra plumbing.

    Restated from ``deepseek_v3_d_p/tt/mla/utils.py:rotated_chip_positions`` rather than imported:
    that module pulls ``safetensors`` and ``transformers`` onto the import path, and this one is
    reachable from the import-light runtime. ``test_kv_cache_vs_ref.py`` grades the copy against the
    original so the two cannot drift, the same arrangement ``tt/rope.py:block_cyclic_reorder`` uses.
    """
    if chunk_local <= 0 or sp <= 0:
        raise ValueError(f"sp ({sp}) and chunk_local ({chunk_local}) must be positive")
    if kv_actual % NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK:
        raise ValueError(
            f"kv_actual ({kv_actual}) must be a multiple of {NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK}; the "
            f"writer derives its tile offset by dividing by it"
        )

    chunk_size_global = sp * chunk_local
    boundary_slab = kv_actual // chunk_size_global
    boundary_chip = (kv_actual // chunk_local) % sp
    boundary_offset = kv_actual % chunk_local

    positions = [[0] * chunk_local for _ in range(sp)]
    for chip in range(sp):
        if chip < boundary_chip:
            update_idxt = (boundary_slab + 1) * chunk_local
        elif chip == boundary_chip:
            update_idxt = boundary_slab * chunk_local + boundary_offset
        else:
            update_idxt = boundary_slab * chunk_local
        for row in range(chunk_local):
            local_row = update_idxt + row
            positions[chip][row] = (
                (local_row // chunk_local) * chunk_size_global + chip * chunk_local + (local_row % chunk_local)
            )
    return positions


def aligned_resume_length(prev_total_tokens: int) -> int:
    """Where a multi-turn continuation resumes, given the previous turn's total length.

    ``update_padded_kv_cache`` asserts ``kv_actual_global % 32 == 0``, so a continuation cannot
    resume at an arbitrary length. It aligns **down** and replays the dropped tokens (at most 31) in
    its first chunk: those positions get rewritten with identical values, which is free.

    Aligning *up* would satisfy the same assertion while leaving an unwritten hole in the middle of
    the sequence that nothing ever fills — attention would read whatever was in DRAM for up to 31
    positions, permanently, and only for continued conversations.
    """
    return (prev_total_tokens // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK) * NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK


@dataclass
class Llama31KVCache(KvCaches):
    """Externally-owned, user-major packed prefill KV cache pair.

    The engine allocates this once, owns its lifetime, and passes it back into every runtime call.
    """

    k: ttnn.Tensor
    v: ttnn.Tensor
    num_users: int
    num_layers: int
    max_seq_len: int
    sp: int
    # KV heads held per chip. 1 in production (TP=8 over 8 KV heads), but TP < 8 leaves several
    # heads per chip, which is the only way an 8-chip loudbox can exercise SP > 1 at all.
    num_kv_heads_per_chip: int = 1

    def layer_view(self, user_id: int, layer_idx: int):
        """Where a ``(user, layer)`` lives: ``(k, v, batch_idx, capacity_tokens)``.

        Single source of truth for the packing, so call sites never recompute the slot arithmetic.
        Drive the write op with ``slot_idx=batch_idx, layer_idx=0, num_layers=1`` — the kernel only
        linearizes ``slot * num_layers + layer``, so folding the layer into the slot here keeps one
        code path and makes this function the only place the packing is expressed.
        """
        if not 0 <= user_id < self.num_users:
            raise ValueError(f"user_id {user_id} out of range [0, {self.num_users})")
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")
        return self.k, self.v, slot_index(user_id, layer_idx, self.num_layers), self.max_seq_len


def allocate_kv_cache(
    mesh_device,
    *,
    num_layers: int = Llama31_8BConfig.NUM_LAYERS,
    max_seq_len: int,
    sp_axis: int = 0,
    num_users: int = 1,
    head_dim: int = Llama31_8BConfig.HEAD_DIM,
    cache_dtype: ttnn.DataType = ttnn.bfloat8_b,
    chunk_size: int | None = None,
    num_kv_heads_per_chip: int = 1,
) -> Llama31KVCache:
    """Allocate the packed K/V prefill caches. See :class:`Llama31KVCache`.

    Args:
        num_layers: layers per user (32 for the full model). Every layer gets a slot.
        max_seq_len: per-user capacity in tokens. Must be a multiple of ``TILE_SIZE * sp`` so
            ``seq_local`` is tile-aligned, matching both the 32-token DRAM shard and
            ``rope.build_indexed_rope``'s constraint — cache and rope layouts have to agree.
        num_users: independent user slots sharing the cache (1 for bring-up).
        cache_dtype: ``bfloat8_b``, matching the DeepSeek substrate. Goldens must be round-tripped
            through this dtype before PCC; a full-precision golden leaves a spurious ~0.94-0.96 gap
            that reads as a real bug.
        chunk_size: the prefill chunk size this cache will be written with. Optional only because
            allocation does not strictly need it; pass it and the layout rules get checked here,
            at allocation, instead of surfacing as a wrong-position KV read much later.
        num_kv_heads_per_chip: ``NUM_KEY_VALUE_HEADS // tp``, i.e. 1 for the production TP=8 mesh.
            Must equal the head count of the chunks passed to :func:`write_kv_chunk`. Only reason
            it is a parameter: SP > 1 is untestable on an 8-chip loudbox at TP=8 (that needs 4x8 =
            32 chips), so validating the block-cyclic write and the ring cache read on one box means
            running e.g. 4x2 at TP=2, which puts 4 KV heads on each chip.
    """
    if num_kv_heads_per_chip < 1:
        raise ValueError(f"num_kv_heads_per_chip must be >= 1, got {num_kv_heads_per_chip}")
    sp = mesh_device.shape[sp_axis]
    if max_seq_len % (ttnn.TILE_SIZE * sp):
        raise ValueError(
            f"max_seq_len ({max_seq_len}) must be a multiple of TILE_SIZE * sp ({ttnn.TILE_SIZE * sp}); "
            f"seq_local must be tile-aligned"
        )
    if chunk_size is not None:
        validate_chunk_layout(max_seq_len, chunk_size, sp)
    seq_local = max_seq_len // sp

    core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0))
        for bank_id in range(get_num_dram_banks(mesh_device))
    ]
    nd_shard_spec = ttnn.NdShardSpec(
        # One head's 32-token block per shard regardless of how many heads a chip holds, so the
        # bank walk on the migration side is unchanged by num_kv_heads_per_chip.
        shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim],
        grid=ttnn.CoreRangeSet(core_ranges),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd_shard_spec)

    def alloc():
        # Which heads a chip ends up holding is decided at write time by how the incoming chunk is
        # mesh-mapped, not here — so every chip is allocated the same zeroed buffer and the contents
        # diverge on the first write.
        return ttnn.from_torch(
            torch.zeros(num_users * num_layers, num_kv_heads_per_chip, seq_local, head_dim),
            dtype=cache_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    logger.info(
        f"Llama-3.1-8B prefill KV cache: {num_users} user(s) x {num_layers} layers = "
        f"{num_users * num_layers} slots, {max_seq_len} tok ({seq_local}/chip over sp={sp}), "
        f"{num_kv_heads_per_chip} KV head(s)/chip, head_dim={head_dim}, {cache_dtype}"
    )
    return Llama31KVCache(
        k=alloc(),
        v=alloc(),
        num_users=num_users,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        sp=sp,
        num_kv_heads_per_chip=num_kv_heads_per_chip,
    )


def write_kv_chunk(
    kv_cache: Llama31KVCache, tt_k, tt_v, *, slot_idx: int, layer_idx: int, kv_actual: int, sp_axis: int
):
    """Write this chunk's post-RoPE K and raw V into the packed cache.

    ``tt_k`` / ``tt_v`` are the per-device SP shards ``[1, n_kv_local, s_local, head_dim]`` — heads
    TP-sharded on the columns, sequence SP-sharded on the ``sp_axis`` rows. That is already the
    per-chip cache layout, so they write in place with no reshard.

    ``kv_actual`` is the cumulative valid prefix *before* this chunk (0 for the first chunk).
    """
    # One user per call: update_padded_kv_cache writes a single (slot_idx, layer_idx) and ignores the
    # leading batch dim, so a batch>1 input would silently write only slot_idx and drop the rest.
    if tt_k.shape[0] != 1 or tt_v.shape[0] != 1:
        raise ValueError(
            f"write_kv_chunk writes one user per call, got leading (batch) dim k={tt_k.shape[0]}, "
            f"v={tt_v.shape[0]}; loop over users (slot_idx + b) at the call site"
        )
    # The write kernel indexes the cache by (slot, token, head_dim) and takes the head count from
    # the cache buffer, so a chunk carrying a different number of heads is not rejected — it writes
    # the wrong heads, or only some of them, and shows up as a KV PCC failure on one head only.
    for name, tensor in (("k", tt_k), ("v", tt_v)):
        if tensor.shape[1] != kv_cache.num_kv_heads_per_chip:
            raise ValueError(
                f"{name} chunk carries {tensor.shape[1]} KV head(s)/chip but the cache was "
                f"allocated for {kv_cache.num_kv_heads_per_chip}; pass "
                f"num_kv_heads_per_chip=NUM_KEY_VALUE_HEADS // tp to allocate_kv_cache"
            )
    # The block-cyclic per-device write assumes a tile-aligned boundary; the op asserts this too, but
    # failing here names the caller's offset instead of a kernel argument.
    if kv_actual % NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK:
        raise ValueError(
            f"kv_actual ({kv_actual}) must be a multiple of {NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK}; a "
            f"continuation should resume at aligned_resume_length(prev_len) and replay the remainder"
        )

    k_cache, v_cache, batch_idx, _capacity = kv_cache.layer_view(slot_idx, layer_idx)
    for cache, tensor in ((k_cache, tt_k), (v_cache, tt_v)):
        # The op needs TILE layout and input.dtype == cache.dtype. Cast a copy when needed: the
        # original stays live for the attention op that consumes it after this write.
        src = tensor if tensor.dtype == cache.dtype else ttnn.typecast(tensor, cache.dtype)
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache,
            src,
            slot_idx=batch_idx,
            layer_idx=0,
            num_layers=1,
            kv_actual_global=kv_actual,
            cluster_axis=sp_axis,
        )
        if src is not tensor:
            src.deallocate(True)
