# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 Compressed Sparse Attention (TTNN prefill).
Mirrors ``DeepseekV4Attention`` on a ``compressed_sparse_attention`` layer.

CSA compresses every 4 tokens instead of every 128 and runs a Lightning Indexer over the compressed
entries, so each query attends to its sliding window plus at most ``index_topk`` compressed entries. The
stems, sinks and output projection are HCA's, in ``v4_attention_base.py``; what is here is the ratio-4
cache append, the joint key table, and the index list that replaces HCA's dense mask.

One table, one index list
-------------------------
The reference attends over ``[carry | sliding | compressed]`` behind an additive 0/-inf mask.
``sparse_sdpa`` takes a per-query list of key ROWS instead, and it reads a single tensor, so all three
sources share one persistent ROW_MAJOR table:

    [0, capacity)                                  compressed entries, appended at ``entry_count``
    [capacity, capacity + W)                       the previous chunk's raw-key tail (the carry)
    [capacity + W, capacity + W + chunk)           this chunk's raw keys, SP-gathered

with ``W`` the sliding window. A query's row ids into that layout are affine in its position (see
``_build_index_consts``), so the list is built once in ``alloc_state`` and each chunk overwrites only its
top-k columns. Attention costs O(S * (W + index_topk)) rather than the mask route's O(S * S/4), and the
mask, its scratch, and the pad rows that existed only to keep the mask honest are all gone.

Design note: the joint table is REPLICATED, not SP-sharded
----------------------------------------------------------
``sparse_sdpa`` reads arbitrary rows of one tensor, and a query on any chip may name any row, so every
chip holds the whole table. Being 4x compressed does not make that free, because a replicated footprint
does not shrink with SP while a sharded one does. At V4-Pro, 55296 tokens, chunk 1024, an 8x4 mesh:
capacity is 14080 entries, so the table is about 14.5 MB per layer per chip against about 8.0 MB for
GLM's SP-sharded KVPE (576-wide, 6912 rows a chip at sp 8). There is a second-order cost too: the
compressor SP-all-gathers each chunk's compressed rows, per layer per chunk.

Sharding it is blocked on the read, not the write: it needs sparse attention over a sharded or
ring-gathered table, which is the same kernel gap GLM's KVPE hits and works around by gathering the
prefix. The memory is the price of the single-tensor contract; the time is now proportional to the
sparsity, which is what the mask route never was."""

from __future__ import annotations

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.compressor import TtCSACompressor, csa_slab_align, rope_table_tokens
from models.demos.deepseek_v3_d_p.tt.mla.indexer import TtCsaIndexer
from models.demos.deepseek_v3_d_p.tt.mla.v4_attention_base import TtV4AttentionBase
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

_SENTINEL = 0xFFFFFFFF  # sparse_sdpa's "no key here"; the indexer emits it where the reference has -1
_INDEX_ALIGN = 128  # sparse_sdpa reads the index list in k_chunk_size-wide steps


def index_tables(rows: int, sliding: int, topk: int, capacity: int):
    """Host halves of the index list: ``(source, first_chunk_permutation, later_permutation)``.

    Split out of ``_build_index_consts`` so the arithmetic can be checked against a golden without a
    device; see ``test_csa_sparse_attention.py``. ``rows`` is the GLOBAL query count of a chunk -- the
    caller shards the result onto SP.

    The source is ``[sliding ids | top-k | one sentinel]``. Sliding ids: the raw region is
    ``[carry | chunk]`` based at ``capacity``, so chunk-local token offset ``d`` (negative inside the
    carry) sits at row ``capacity + sliding + d``, and query ``i`` wants tokens ``i - sliding + 1 .. i``.
    That makes the id affine in ``i`` and ``j`` and independent of where the chunk sits, which is what
    lets it be built once. The top-k block is zeroed here and overwritten per chunk; its ids need no
    arithmetic, because compressed entries occupy rows ``[0, capacity)`` so an entry id IS its row id.

    The permutations exist because ``sparse_sdpa``'s reader finds the first sentinel and treats it as the
    end of the row, so sentinels must be a contiguous TAIL. Query ``p`` has only ``min(sliding, p + 1)``
    real sliding keys and they are the LAST slots of its block -- the low ``j`` fall off the front of the
    sequence. Left alone those holes sit mid-row and every compressed pick after them is dropped. So the
    first chunk gathers through a permutation that slides the real sliding slots down to 0, abuts the
    pick block to them, and points the rest at the spare sentinel column. Only positions below
    ``sliding`` are short and chunks are whole multiples of ``chunk_align``, so every later chunk's carry
    fills the window and its permutation is the identity, modulo that same tail."""
    dump = sliding + topk  # the source's spare sentinel column, and the gather's dump target
    width = -(-dump // _INDEX_ALIGN) * _INDEX_ALIGN

    i = torch.arange(rows).view(rows, 1)
    j = torch.arange(sliding).view(1, sliding)
    sliding_ids = capacity + sliding + (i - sliding + 1 + j)
    source = torch.cat(
        [sliding_ids, torch.zeros(rows, topk, dtype=torch.int64), torch.full((rows, 1), _SENTINEL)], dim=-1
    )

    out = torch.arange(width).view(1, width).expand(rows, width)
    valid = (i + 1).clamp(max=sliding)  # real sliding keys of the first chunk's row i
    compacted = torch.where(out < valid, out + (sliding - valid), sliding + (out - valid))
    compacted = torch.where(out < valid + topk, compacted, torch.full_like(compacted, dump))
    return source, compacted, out.clamp(max=dump)


def _compute_kernel_config(device, *, fp32: bool):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4 if fp32 else ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=True,
    )


class TtCSAState:
    """Chunked-prefill state, owned by the caller and passed to ``TtCSA.forward``.

    The device tensors keep a FIXED shape for the whole prefill and only their contents advance --
    that is what lets one compiled program serve every chunk. The counters say how much is real; no
    query's index list names a row past that, so the rest is never read.

    One piece of per-prefill state is deliberately NOT here: the indexer runs its own compressor and so
    carries its own overlap state, at ``index_head_dim``, which it keeps privately and advances inside
    ``write_k``. ``alloc_state`` resets it alongside the two below, so both begin at the same position."""

    def __init__(self, joint_kv, kv_state, score_state, index_kv_cache, max_seq_len):
        # The single ROW_MAJOR table sparse_sdpa reads, laid out as the module docstring describes. The
        # carry lives in it too, at a fixed row range, so there is no separate carry tensor: forward
        # writes the next chunk's carry there once this chunk's attention has already read it.
        self.joint_kv = joint_kv  # [1, 1, capacity + sliding_window + chunk, head_dim]
        # The overlap the next chunk's first window needs: its predecessor window's Ca slice, in the
        # decode-compatible Blaze layout the compressor op emits and consumes.
        self.kv_state = kv_state  # [B, 1, CSA_STATE_ROWS, head_dim]
        self.score_state = score_state  # [B, 1, CSA_STATE_ROWS, head_dim]
        self.index_kv_cache = index_kv_cache  # block-cyclic indexer key cache, written in place
        self.max_seq_len = int(max_seq_len)
        self.entry_count = 0
        self.kv_actual = 0


class TtCSA(TtV4AttentionBase):
    """CSA block: query/kv stems + compressor + indexer + attention core + grouped output projection.

    Block I/O is ``[B, 1, S/sp, hidden/tp]``, so layers chain without a reshard."""

    def __init__(self, device, *, indexer_reference, indexer_settings, **kwargs):
        super().__init__(device, **kwargs)
        # The indexer is sized by max_seq_len and the slab width, which only alloc_state knows, so it is
        # built there; these hold what it needs until then.
        self._indexer_reference = indexer_reference
        self._indexer_settings = self._resolve_indexer_settings(indexer_settings)
        self._indexer = None
        # Everything below comes from alloc_state, which every caller has to run before forward.
        self._index_source = None  # persistent [sliding ids | top-k | sentinel] the gather reads
        self._index_perm = None  # (first-chunk compacting permutation, later-chunk one)
        self._carry_row = None  # where the carry starts in the joint table
        self._raw_row = None  # and where this chunk's raw keys do

    @staticmethod
    def _resolve_indexer_settings(settings: dict) -> dict:
        """Fill in the index-cache layout fields, so ``alloc_state`` can size or validate a cache
        whatever the caller supplied.

        ``slot_num``/``layer_num`` describe the shared cache the layer writes into; the default 1x1 is
        the private single-slot cache. ``cache_layer_idx`` is this layer's LOCAL slot in it, which is
        ``layer_idx`` when one TtCSA per layer shares one layer_num-deep cache, and 0 in the single-slot
        default, where ``layer_idx`` only names the weight cache. Under pipeline parallelism the global
        ``layer_idx`` diverges from the local slot, so such a caller passes ``cache_layer_idx``
        explicitly -- the same distinction ttMLA draws for the KVPE cache."""
        resolved = {"slot_num": 1, "layer_num": 1, **settings}
        layer_num = int(resolved["layer_num"])
        resolved.setdefault("cache_layer_idx", resolved["layer_idx"] if layer_num > 1 else 0)
        return resolved

    @property
    def chunk_align(self) -> int:
        """One compression window per row of a tile. At ratio 4 that is 128 tokens, and it buys three
        things at once: the compressed append offset (entry_count = tokens/4) lands tile-aligned so
        fill_cache can take it directly, the carry slice start (real_len - 128) is tile-aligned, and a
        chunk always holds a whole sliding window for the next one to carry.

        This is the granularity of the SLAB WIDTH, not of a chunk's real length: forward requires every
        non-final chunk to be a whole slab, which is stricter. See its own assert for why."""
        return self.compressor.compress_rate * ttnn.TILE_SIZE

    @property
    def indexer(self) -> TtCsaIndexer:
        assert self._indexer is not None, "call alloc_state before touching the indexer"
        return self._indexer

    def alloc_state(
        self,
        max_seq_len: int,
        batch: int = 1,
        chunk_tokens: int | None = None,
        index_kv_cache: ttnn.Tensor | None = None,
    ) -> TtCSAState:
        """Size the state once for the longest context this layer will serve, so its shape is fixed for
        every chunk. ``max_seq_len`` is the longest context to serve, ``chunk_tokens`` the slab width
        forward will be called with (defaults to one chunk).

        Every host tensor this layer will ever need is built here, which is what leaves forward with
        none. So the caller has to own the state: a prefill of one chunk allocates the same way a long
        one does.

        ``index_kv_cache`` is the block-cyclic indexer key cache. Pass one to own it -- that is the
        indexer's real contract, and it is what lets a serving caller allocate a single cache across
        layers and users and describe it with a KV chunk address table. Left None, this allocates a
        private one sized to the same ``slot_num``/``layer_num`` layout, so the shape a migration table
        expects holds either way; it is just not shared."""
        rate = self.compressor.compress_rate
        chunk = chunk_tokens or max_seq_len
        align = csa_slab_align(rate, self.sp_factor, self.tp_factor)
        assert chunk % align == 0, (
            f"the slab is {chunk} wide, which is not a multiple of {align}; see csa_slab_align for what "
            f"asks for that. prepare_input rounds a raw prompt up to it."
        )

        width = chunk // rate  # entries one call emits, a whole number of tiles by chunk_align
        entries = -(-int(max_seq_len) // rate)
        # Writes are exact-width from a tile-aligned offset, so the last one can reach one slab past the
        # entries themselves.
        capacity = -(-entries // ttnn.TILE_SIZE) * ttnn.TILE_SIZE + width
        # The joint table's three regions. The index list is built off the same two numbers, so the rows
        # it names and the rows forward writes cannot drift apart.
        self._carry_row = capacity
        self._raw_row = capacity + self.sliding_window

        self._build_carry_index(chunk)
        self.compressor.alloc_tables(max_seq_len, chunk)

        # One rope table per state, so forward only gathers. This one has a row per TOKEN; the compressor
        # builds its own, with a row per ENTRY.
        self._slab_rope = self.ops.build_rope_table(rope_table_tokens(max_seq_len, chunk), 1)
        self._slab_index = self.ops.rope_index_base(chunk // self.sp_factor)

        index_kv_cache = self._build_indexer(max_seq_len, chunk, index_kv_cache)
        self._build_index_consts(chunk, capacity)

        # Two overlap states advance through a prefill, one per compressor: the block's, which lives in
        # the returned TtCSAState and is advanced by forward, and the indexer's, which the indexer keeps
        # privately and advances inside write_k. They describe the same token positions, so they have to
        # start from the same "no predecessor window" -- reset the indexer's here, next to the block's
        # allocation, rather than leave the pairing to the fact that _build_indexer happens to make a
        # fresh indexer.
        self._indexer.reset_overlap_state()
        kv_state, score_state = self.compressor.alloc_overlap_state(batch)
        # ROW_MAJOR and unpadded because sparse_sdpa reads it by row; replicated because any query may
        # name any row. The carry region starts out zero and the first chunk's index list never points
        # into it, so nothing reads it before forward writes it.
        joint_kv = self.ops.from_torch(
            torch.zeros(batch, 1, self._raw_row + chunk, self.head_dim), layout=ttnn.ROW_MAJOR_LAYOUT
        )
        return TtCSAState(
            joint_kv=joint_kv,
            kv_state=kv_state,
            score_state=score_state,
            index_kv_cache=index_kv_cache,
            max_seq_len=max_seq_len,
        )

    def _build_indexer(self, max_seq_len: int, chunk: int, index_kv_cache):
        """Build the indexer and resolve its block-cyclic key cache.

        The context the indexer is told about is wider than the real one for two reasons: it writes
        whole padded slabs, so a run of ragged chunks lands its last write up to one slab past
        ``max_seq_len``, and each chip's share of the compressed rows has to be a whole number of tiles.
        Both only ever add rows past ``entry_count``, which the score op's own causal mask drops."""
        rate = self.compressor.compress_rate
        settings = self._indexer_settings
        slot_num, layer_num = settings["slot_num"], settings["layer_num"]
        index_align = self.chunk_align * self.sp_factor
        index_tokens = rope_table_tokens(max_seq_len, chunk)
        index_seq_len = -(-index_tokens // index_align) * index_align
        index_head_dim = settings["config"].index_head_dim
        # Entries per chip, since the cache is SP-sharded over the block-cyclic sequence blocks.
        index_entries_local = index_seq_len // rate // self.sp_factor

        self._indexer = TtCsaIndexer.from_reference(
            self._indexer_reference,
            config=settings["config"],
            mesh_device=self.device,
            sp_axis=self.sp_axis,
            tp_axis=self.tp_axis,
            default_compute_kernel_config=settings["default_compute_kernel_config"],
            hifi4_fp32_compute_kernel_config=settings["hifi4_fp32_compute_kernel_config"],
            weight_cache_path=settings["weight_cache_path"],
            layer_idx=settings["layer_idx"],
            tt_ccl=self.tt_ccl,
            ccl_num_links=self.ccl_num_links,
            sp_ccl_topology=self.sp_ccl_topology,
            tp_ccl_topology=self.tp_ccl_topology,
            seq_len=index_seq_len,
            active_seq_len=chunk,
            slot_num=slot_num,
            layer_num=layer_num,
        )
        # The layer stride the write op will index with: layer_num normally, and the compact full-layer
        # count under GLM-style indexer reuse. Both the validation and the fallback go through it, so the
        # two can never disagree about the cache's shape.
        cache_layers = self._indexer.index_cache_layers
        assert settings["cache_layer_idx"] < cache_layers, (
            f"cache_layer_idx {settings['cache_layer_idx']} is outside the {cache_layers}-layer index "
            f"cache; pass the LOCAL cache slot, not a global layer index"
        )

        if index_kv_cache is not None:
            # A mis-sized cache would still be written, just into another layer's or user's rows, so
            # reject it here rather than let a silently-corrupted slot reach decode.
            shape = tuple(index_kv_cache.shape)
            assert shape[0] == slot_num * cache_layers, (
                f"caller-owned index cache has {shape[0]} slots, expected {slot_num * cache_layers} "
                f"(slot_num {slot_num} x {cache_layers} layers, user-major)"
            )
            assert shape[-1] == index_head_dim, (
                f"caller-owned index cache is {shape[-1]} wide, expected index_head_dim " f"{index_head_dim}"
            )
            assert shape[2] >= index_entries_local, (
                f"caller-owned index cache holds {shape[2]} entries per chip, but this context writes "
                f"up to {index_entries_local}"
            )
            return index_kv_cache

        return init_kvpe_cache(
            kvpe_cache_head_dim=index_head_dim,
            mesh_device=self.device,
            seq_len=index_seq_len // rate,
            mesh_shape=list(self.device.shape),
            sp_axis=self.sp_axis,
            num_kvpe_cache_layers=cache_layers,
            num_users=slot_num,
        )

    def _build_index_consts(self, chunk: int, capacity: int):
        """Upload the index source and the two permutations ``index_tables`` lays out.

        Rows are SP-sharded, so each chip gets the ids for its own queries. The sentinel does not fit a
        signed 32-bit value, so it goes up as -1 and is read back through the uint32 dtype -- same bit
        pattern, which is all the reader compares."""
        source, compacted, later = index_tables(chunk, self.sliding_window, self.indexer.index_topk_capacity, capacity)
        sp_mapper = self.ops.mesh_mapper(sp_dim=2)

        def upload(host):
            return self.ops.from_torch(
                host.reshape(1, 1, chunk, -1).to(torch.int32),
                mesh_mapper=sp_mapper,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        self._index_source = upload(source)
        self._index_perm = (upload(compacted), upload(later))

    @classmethod
    def from_reference(
        cls,
        device,
        reference,
        config,
        *,
        layer_idx: int = 0,
        weight_cache_path=None,
        default_compute_kernel_config=None,
        hifi4_fp32_compute_kernel_config=None,
        slot_num: int = 1,
        layer_num: int = 1,
        cache_layer_idx: int | None = None,
        **kwargs,
    ) -> "TtCSA":
        """``slot_num`` / ``layer_num`` / ``cache_layer_idx`` describe this layer's place in the index key
        cache; see ``_resolve_indexer_settings``. The 1x1 default is a private per-layer cache."""
        assert reference.compressor is not None and hasattr(
            reference.compressor, "indexer"
        ), "TtCSA needs a compressed_sparse_attention layer, whose compressor owns the indexer"
        # Forward the mesh/CCL config so the compressor rides the same SP/TP axes as the block.
        compressor_keys = ("sp_axis", "tp_axis", "topology", "dtype", "weights_dtype", "memory_config")
        compressor = TtCSACompressor.from_reference(
            device, reference.compressor, config, **{k: kwargs[k] for k in compressor_keys if k in kwargs}
        )
        return cls(
            device,
            compressor=compressor,
            indexer_reference=reference.compressor.indexer,
            indexer_settings={
                "config": config,
                "layer_idx": layer_idx,
                "slot_num": slot_num,
                "layer_num": layer_num,
                **({} if cache_layer_idx is None else {"cache_layer_idx": cache_layer_idx}),
                "weight_cache_path": weight_cache_path,
                "default_compute_kernel_config": default_compute_kernel_config
                or _compute_kernel_config(device, fp32=False),
                "hifi4_fp32_compute_kernel_config": hifi4_fp32_compute_kernel_config
                or _compute_kernel_config(device, fp32=True),
            },
            q_a_proj_weight=reference.q_a_proj.weight,
            q_a_norm_weight=reference.q_a_norm.weight,
            q_b_proj_weight=reference.q_b_proj.weight,
            kv_proj_weight=reference.kv_proj.weight,
            kv_norm_weight=reference.kv_norm.weight,
            sinks=reference.sinks,
            o_a_proj_weight=reference.o_a_proj.weight,
            o_b_proj_weight=reference.o_b_proj.weight,
            rotary_emb=reference.compressor.rotary_emb,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            rope_head_dim=config.qk_rope_head_dim,
            sliding_window=config.sliding_window,
            o_groups=config.o_groups,
            rms_norm_eps=config.rms_norm_eps,
            **kwargs,
        )

    def _write_rows(self, table, rows, start: int):
        """Write ``rows`` into the joint table at row ``start``. ``rows`` arrives TILE, as every producer
        here does; the table is ROW_MAJOR, so the conversion happens on the small tensor."""
        rm = ttnn.to_layout(rows, ttnn.ROW_MAJOR_LAYOUT)
        end = [rm.shape[0], 1, start + rm.shape[2], rm.shape[3]]
        ttnn.experimental.slice_write(rm, table, start=[0, 0, start, 0], end=end, step=[1, 1, 1, 1])
        ttnn.deallocate(rm)

    def _write_compressed(self, state, new_entries):
        """Append this call's entries to the table at row ``state.entry_count``.

        The whole padded width is written, not just the real entries, so the width is the same every
        chunk; no query's index list names an entry past ``total_entries``, because the score op's causal
        mask already dropped them.

        Unlike every other write here the offset moves, and ``slice_write`` takes it as a host value, so
        chunk ``c`` compiles its own program. That is bounded and shared: the offset is
        ``c * chunk / rate``, so a prefill of N chunks has N of them, and since the hash is over tensor
        SPECS rather than buffers, all layers reuse the same N. A single-shot prefill compiles one.
        Getting to a single program means either a ``slice_write`` whose hash excludes the offset (its
        ROW_MAJOR factory already rebuilds the runtime args from it, but the tiled factories bake it into
        shared state, so the hash cannot simply drop it) or a replicated mode for
        ``update_padded_kv_cache``, whose ``cluster_axis=None`` is block-cyclic over the whole mesh
        rather than replicated. Both are C++ changes; neither is worth N programs."""
        width = new_entries.shape[2]
        assert state.entry_count + width <= self._carry_row, (
            f"compressed cache full: writing rows [{state.entry_count}, {state.entry_count + width}) "
            f"exceeds capacity {self._carry_row}; allocate the state with a larger max_seq_len"
        )
        self._write_rows(state.joint_kv, new_entries, state.entry_count)

    def _build_index(self, state, top_k):
        """This chunk's index list: overwrite the source's top-k block, then gather through the
        permutation the chunk's position calls for."""
        source, rows, k = self._index_source, self._index_source.shape[2], top_k.shape[3]
        ttnn.experimental.slice_write(
            top_k,
            source,
            start=[0, 0, 0, self.sliding_window],
            end=[1, 1, rows, self.sliding_window + k],
            step=[1, 1, 1, 1],
        )
        return self.index_list(first_chunk=state.kv_actual == 0)

    def index_list(self, *, first_chunk: bool):
        """The index list attention reads, gathered from the persistent source buffer.

        The gather is what compacts; see ``index_tables`` for why only the first chunk needs it. Both
        operands are persistent and ``ttnn.gather`` is out-of-place, so neither is written through --
        which is also why this can be called again after a chunk to get that chunk's list back.

        Public because the list is the one intermediate a caller cannot otherwise see: ``forward``
        deallocates its copy, while the source buffer keeps the top-k it gathered from. ``first_chunk``
        has to be passed rather than read off the state, because ``forward`` has already advanced
        ``kv_actual`` by the time a caller gets here."""
        return ttnn.gather(self._index_source, -1, self._index_perm[0 if first_chunk else 1])

    def _sparse_attention(self, q, table, index, cos, sin):
        """One ``sparse_sdpa`` over the joint table, then V's RoPE undone.

        No mask and no causal flag: the index list already says exactly which keys each query sees. The
        reshard is GLM's (``ttMLA._sparse_mla``) -- when the TP head shard is thinner than the op's
        32-head minimum, the sharding moves head -> sequence for the duration of attention, and the index
        list follows it with a local partition rather than a collective, since it is TP-replicated."""
        reshard = self.needs_head_to_seq_reshard
        q_attn = self._reshard(q, in_dim=1, out_dim=2) if reshard else q
        q_rm = ttnn.to_layout(q_attn, ttnn.ROW_MAJOR_LAYOUT)  # the op is ROW_MAJOR-only; q comes in TILE
        if reshard:
            ttnn.deallocate(q_attn)
            index = ttnn.mesh_partition(index, dim=2, cluster_axis=self.tp_axis)

        out = ttnn.transformer.sparse_sdpa(
            q_rm,
            table,
            index,
            self.head_dim,  # V4 has no RoPE-only tail, so v_dim is the full width
            kv_format=ttnn.transformer.SparseKVFormat.BF16,
            scale=self.scaling,
            k_chunk_size=_INDEX_ALIGN,
            attention_sink=self._sparse_sinks(),
        )
        ttnn.deallocate(q_rm)
        attn = ttnn.to_layout(out, ttnn.TILE_LAYOUT)  # back to TILE for the un-rope and the projection
        ttnn.deallocate(out)
        if reshard:
            restored = self._reshard(attn, in_dim=2, out_dim=1)
            ttnn.deallocate(attn)
            attn = restored
        # Only now do the rows match cos/sin again, which cover this chip's own queries.
        return self._unrope(attn, cos, sin)

    def forward(
        self,
        hidden_states,
        seq_len_actual: int | None = None,
        *,
        state: TtCSAState,
        cache_user_id: int = 0,
    ):
        """One chunk: [B, 1, S_pad/sp, hidden/tp] in and out; the caller keeps the first S_real rows.

        ``seq_len_actual`` is the chunk's real pre-pad length. Where the chunk sits in the sequence comes
        from ``state``, which ``alloc_state`` builds and this advances in place -- a prefill of one chunk
        passes a state too, so there is no second path through here.

        ``cache_user_id`` picks this prompt's user slot in the index key cache. It is a per-call property
        (one layer serves many users), unlike the layer slot, which is fixed at construction."""
        batch = hidden_states.shape[0]
        seq_local = hidden_states.shape[2]
        seq_pad_global = seq_local * self.sp_factor
        rate = self.compressor.compress_rate
        real_len = seq_pad_global if seq_len_actual is None else seq_len_actual

        assert batch == 1, f"CSA prefill expects batch 1, got {batch}"
        assert real_len >= rate, (
            f"CSA prefill needs at least one full compression window: got seq_len {real_len} < " f"compress_rate {rate}"
        )
        assert state.kv_actual + real_len <= state.max_seq_len, (
            f"context longer than the state was allocated for: {state.kv_actual + real_len} tokens > "
            f"max_seq_len {state.max_seq_len}"
        )
        # Every chunk before this one must have been WHOLE, not merely chunk_align-aligned. A short
        # non-final chunk leaves kv_actual mid-slab, and the indexer's key cache is block-cyclic:
        # update_padded_kv_cache ROTATES which chip owns which slab as soon as the write offset is not a
        # whole number of slabs (it only checks tile alignment, so it takes this silently), and indexer_score's
        # causal geometry rotates with it -- while the hidden states arrive SP-sharded linearly. Queries
        # then score keys at other tokens' positions, and under a saturating top-k that shows up as a
        # wrong causal threshold rather than a wrong ranking: the picks name entries the query must not
        # see. A slab here IS the padded chunk width, so "slab-aligned" and "every earlier chunk was
        # whole" are the same condition, and it is the one the prefill runtime already meets by only
        # ever ragging the last chunk. Checked on tokens, where a stranded partial window is visible too.
        assert state.kv_actual % seq_pad_global == 0, (
            f"cannot append after a ragged chunk: {state.kv_actual} tokens is not a multiple of the "
            f"{seq_pad_global}-token slab, so the indexer's block-cyclic cache and its causal geometry "
            f"would rotate away from the queries; only the final chunk may be ragged"
        )

        n_new = real_len // rate
        total_entries = state.entry_count + n_new

        # One rotation for the whole padded slab, shared by both stems and the output un-rope.
        slab_index = self.ops.rope_index(self._slab_index, state.kv_actual)
        cos, sin = self.ops.rope_gather(self._slab_rope, slab_index)
        q_lora = self._q_lora(hidden_states)  # the reference's q_residual; the indexer scores from it
        q = self._q_heads(q_lora, cos, sin)
        sliding_kv = self._kv_stem(hidden_states, cos, sin)

        new_entries, kv_state, score_state = self.compressor(
            hidden_states,
            state.kv_state,
            state.score_state,
            seq_len_actual=real_len,
            first_window_position=state.entry_count * rate,
        )
        self._write_compressed(state, new_entries)
        ttnn.deallocate(new_entries)

        # The raw keys go in whole: every chip's queries may reach into any of them, and the gather is
        # what makes that possible. The carry is written after attention, below.
        sliding_kv = self._gather_sliding(sliding_kv)
        self._write_rows(state.joint_kv, sliding_kv, self._raw_row)

        # The indexer runs its own ratio-4 compressor at index_head_dim over the same windows and writes
        # its own block-cyclic key cache, so it only shares the hidden states and q_lora with us.
        top_k = self.indexer.forward(
            hidden_states,
            q_lora,
            seq_len=seq_local,
            start_pos=state.kv_actual,
            cache_user_id=cache_user_id,
            cache_layer_idx=self._indexer_settings["cache_layer_idx"],
            index_kv_cache=state.index_kv_cache,
            seq_len_actual=real_len,
        )
        index = self._build_index(state, top_k)
        ttnn.deallocate(top_k)
        attn = self._sparse_attention(q, state.joint_kv, index, cos, sin)
        ttnn.deallocate(index)

        # Overwrites the carry this chunk just attended over, so it has to follow the attention. Taken
        # from the gathered slab, whose last REAL rows are the next chunk's window.
        next_carry = self._take_carry(sliding_kv, real_len)
        ttnn.deallocate(sliding_kv)
        self._write_rows(state.joint_kv, next_carry, self._carry_row)
        ttnn.deallocate(next_carry)

        state.kv_state = self.compressor.terminal_state(kv_state)
        state.score_state = self.compressor.terminal_state(score_state)
        state.entry_count = total_entries
        state.kv_actual += real_len
        return self._o_proj(attn)
