# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo attention (SWA and global layers), tensor-parallel over one Galaxy row.

SWA layers (8 KV heads): each chip has 8 Q heads and one KV head, in prefill and decode. V has
128 dims; prefill zero-pads it to 192 (and so are the matching ``o_proj`` rows) so the prefill SDPA sees one
head dim. Every V cache is 128 wide (as on the global layers): vLLM sizes each KV buffer to its whole block pool
and shares it between one layer of each KV cache group, so a 192-wide SWA V cache would cost 1632 of 4692 bytes
per token per chip, and with 128 the SWA V caches share the global layers' V buffers. Decode reads it with
``paged_flash_multi_latent_attention_decode`` (``head_dim_v=128``, sinks, sliding window), whose output has no
padding columns, and uses an unpadded copy of ``o_proj``; a later prefill chunk pads the cached window back to
192.

Global layers (4 KV heads over 8 chips): chips 2g and 2g+1 serve KV head g and split its users
(see ``kv.py``). Decode gives each chip all 16 Q heads of its group for the slots it owns, with an
unpadded 128-wide V cache read by ``paged_flash_multi_latent_attention_decode`` (``head_dim_v=128``).
Each KV head is stored once and V is not padded: 320 values per token instead of 2 x 384.

Prefill runs one user's chunk of S tokens at ``start_pos`` (a multiple of the block size) and writes its
K/V into the cache. The first chunk attends to its own fresh K/V; global layers keep 8 Q heads per chip
there. A later chunk needs the keys before it:

* SWA layers read only the window before ``start_pos`` (two blocks) from the cache, put it in front of
  the chunk's fresh K/V and run the same windowed SDPA over 128 + S rows (zero queries for the window
  rows, whose outputs are dropped). The cache never has to hold a whole chunk, so a SWA user keeps
  ``kv.swa_blocks_per_user`` blocks however long the chunks are.
* Global layers need the whole context, which only the owner chip of the pair holds: both chips run
  the group's 16 Q heads (the decode weights) through ``chunked_flash_mla_prefill`` over the K and
  128-wide V caches, and the non-owner's result is zeroed before the row all-reduce.

Global KV pair split (vLLM mode, MIMO_KV_PAIR_SPLIT, the default; reference/kv_pair_split.py, tt/pair_split.py):
under vLLM a block can be shared by requests in slots of either parity (prefix caching), so the pair cannot split
the users; it splits every 128-position block instead, each chip keeping its 64-position halves (``self.pair``, set
by the model). Each chip runs the group's 16 Q heads over its local keys for every user, the attention ops return the
partial output with its LSE (``return_lse``), and ``PairSplit.merge`` scales it by the chip's merge weight from the
pair's two LSEs, so the row all-reduce after ``wo_decode`` sums full attention. Decode writes K / V and attends at the
chip's own positions (``_forward_decode_pair``); a first prefill chunk is unchanged but for its fill table (each
chip's halves, kv.PrefillTables), and a later chunk runs pair-split chunked MLA prefill and the merge
(``_prefill_kv_split_chunk``) in place of the head split (``_prefill_pair_chunk``, both copies on the pair). The two
attention ops' new arguments are used only in ``_paged_mla_decode_lse`` and ``_chunked_mla_prefill_lse``.

Every prefill program keys on the chunk's padded length (and, for a later chunk, on how far ``start_pos`` is
aligned and how many window blocks it reads), never on ``start_pos``, the user or a block id: the model hands each
chunk its block tables as small device tensors (``kv.PrefillTables``) instead of the layers slicing the page tables,
the window blocks are read by tensor-args slices and chunked MLA takes ``start_pos`` from a device tensor. So a
warmup over the padded lengths compiles every program a served prefill runs, and no prefill run between decode
trace replays compiles one (a program compiled after a trace capture keeps buffers the replays can overwrite).
``attention_value_scale`` is folded into the V weights. Sinks (SWA only) are pre-divided by the
softmax scale because the SDPA kernels apply ``scale`` to the sink logit too.
"""

import os

import torch

import ttnn

from ..reference.kv_pair_split import NO_KEY_LSE
from ..reference.spec_decode import ADAPTIVE, GROUP_SLOTS, SPEC_SLOTS, WIDTHS
from .proj_stream import BankStreamMatmul
from .common import TILE, DramShardedMatmul, as_weight, cache_name, compute_config, decode_linear_config, pad_to
from .kv import BatchFillTables, kv_replication, per_chip, slot_owner_masks
from .moe_stream import KERNEL_DIR, user_rows, zero_output
from .profiling import steps
from .rope import permute_heads_for_rope

# chunked_flash_mla_prefill reads its start position from a device tensor (chunk_start_idx_tensor); builds without
# that keyword key the program on the scalar start, which compiles a program for every new start.
MLA_RUNTIME_START = "chunk_start_idx_tensor" in (ttnn.transformer.chunked_flash_mla_prefill.__doc__ or "")

# The KV pair split's op arguments (runs/kvsplit/SPEC.md): both MLA attention ops return (output, lse) with
# ``return_lse=True``; chunked MLA prefill also takes each chip's pair parity (PairSplit.prefill_parity, a per-chip
# one-element int32 tensor) through this keyword.
PAIR_PARITY_KWARG = "kv_pair_parity_tensor"
PAIR_SPLIT_OPS = all(
    word in (op.__doc__ or "")
    for op, words in (
        (ttnn.transformer.paged_flash_multi_latent_attention_decode, ("return_lse",)),
        (ttnn.transformer.chunked_flash_mla_prefill, ("return_lse", PAIR_PARITY_KWARG)),
    )
    for word in words
)

# A speculative verify's global attention under the KV pair split (_verify_grouped): one KV scan per request for all
# its candidates (paged MLA decode's spec_multi_pos_tiles) instead of one per candidate slot. Each candidate's 16 heads
# fill half a 32-row Q tile, so one tile per candidate is compute-bound; spec_multi_pos_packed (GROUPED_PACKED_OPS)
# computes two candidates per tile. One chip, ms per global layer (tests/perf/verify_sdpa_bench.py; per slot /
# grouped one tile per candidate / packed / a plain decode step of the requests):
# - 4 requests of 2 candidates (MIMO_SPEC_GROUPS=4, c=16): 1M 4.33 / 3.04 / 2.19 / 2.19, 256K 1.10 / 0.82 / 0.58 /
#   0.58, 32K 0.17 / 0.16 / 0.12 / 0.11, 4K 0.086 / 0.083 / 0.090 / 0.041;
# - 1 request of 4: 1M 2.18 / 1.57 / 0.87 / 0.62, 128K 0.31 / 0.30 / 0.18 / 0.12, 8K 0.063 / 0.126 / 0.089 / 0.064.
# On by default for groups of 2 slots, off for 4 (MIMO_VERIFY_GROUPED=1 / 0 overrides; packed, groups of 4 win from
# ~32K and lose 0.2 ms a step at 8K); the adapter then packs a verify's requests one per group of GROUP_SLOTS slots
# (verify_grouped()).
VERIFY_GROUPED = os.environ.get("MIMO_VERIFY_GROUPED", "1" if GROUP_SLOTS == 2 or ADAPTIVE else "0") == "1"
GROUPED_VERIFY_OPS = "spec_multi_pos_tiles" in (
    ttnn.transformer.paged_flash_multi_latent_attention_decode.__doc__ or ""
)
GROUPED_PACKED_OPS = "spec_multi_pos_packed" in (
    ttnn.transformer.paged_flash_multi_latent_attention_decode.__doc__ or ""
)


def verify_grouped():
    """Whether a pair-split verify runs its global attention one KV scan per request (VERIFY_GROUPED, groups of 2 or 4
    candidates, and a ttnn build with spec_multi_pos_tiles on paged MLA decode). 8 candidates (DFlash's one group)
    need 128-token K chunks and at most 32 cores per request to fit L1 beside decode's buffers: not wired."""
    return VERIFY_GROUPED and GROUPED_VERIFY_OPS and GROUP_SLOTS in (2, 4)


# The grouped SDPA splits its cores evenly over the groups it is given, used or not: with 4 groups of 2 slots a lone
# request scans its keys on 16 of the 64 cores, a quarter of the bandwidth (at 256K, 0.58 ms per global layer for one
# request as for four). So the verify is captured once per count of leading groups it attends (verify_variants), and
# the adapter packs a step's requests into the lowest groups and replays the narrowest trace that holds them: one
# request scans on 64 cores, two on 32 each. MIMO_VERIFY_VARIANTS=0: one trace over every group.
VERIFY_VARIANTS = os.environ.get("MIMO_VERIFY_VARIANTS", "1") == "1"
# The cores still split evenly over the attended groups, so the longest context sets the time when the requests'
# contexts differ (AgentX: tens of K beside hundreds of K). MIMO_VERIFY_SEQUENTIAL=1: one SDPA per group, each on the
# whole grid, back to back (the time is then the sum of the contexts' scans, plus a launch per group).
VERIFY_SEQUENTIAL = os.environ.get("MIMO_VERIFY_SEQUENTIAL", "0") == "1"


def verify_variants():
    """The verifies captured, (group width, leading groups attended) each (the widest last: every group of
    GROUP_SLOTS slots); [None] without grouping. A pass's group width is GROUP_SLOTS, or with adaptive draft groups
    (reference/spec_decode.py ADAPTIVE) 2 or 4."""
    if not verify_grouped():
        return [None]
    out = []
    for width in WIDTHS:
        groups = SPEC_SLOTS // width
        counts = [n for n in (1, 2, 4, 8) if n < groups] if VERIFY_VARIANTS else []
        out += [(width, n) for n in counts + [groups]]
    return out


def verify_variant(used_groups, width=GROUP_SLOTS):
    """The narrowest captured verify (verify_variants) in groups of ``width`` slots attending groups 0 ..
    ``used_groups`` - 1."""
    variants = verify_variants()
    if variants == [None]:
        return None
    return next(v for v in variants if v[0] == width and v[1] >= max(1, used_groups))


# A global layer's later prefill chunks stream the whole cached context once per (head group, Q chunk) unit. With
# grouped heads (SDPAProgramConfig.q_heads_per_unit: G heads share each K/V chunk read) and a K split (k_splits: each
# unit's keys over S work items, merged after the program) on the full grid, the K / V reads drop by G and every core
# gets the same share; at 1M one layer's chunk goes 203 -> 139 ms per chip. The split's merge costs a few ms, so it
# only pays from about 64K of context (MIMO_GLOBAL_SDPA_MIN_START). MIMO_GLOBAL_SDPA="gGsS" picks G and S; "legacy"
# keeps the 8x8, one head per unit program everywhere.
GLOBAL_SDPA = os.environ.get("MIMO_GLOBAL_SDPA", "g2s15")
GLOBAL_SDPA_MIN_START = int(os.environ.get("MIMO_GLOBAL_SDPA_MIN_START", "65536"))
GROUPED_SDPA = hasattr(ttnn.SDPAProgramConfig, "q_heads_per_unit")


def global_sdpa_split():
    """(G, S) for a global layer's later chunks, or None for the legacy program."""
    if not GROUPED_SDPA or GLOBAL_SDPA == "legacy":
        return None
    g, s = GLOBAL_SDPA.lstrip("g").split("s")
    return int(g), int(s)


def later_chunk_split(chunk, start_pos):
    """(G, S) of a global layer's later chunk at ``start_pos`` whose SDPA chunk size is ``chunk``, or None when it runs
    the legacy program (TtMiMoAttention._prefill_sdpa_program)."""
    split = global_sdpa_split() if start_pos >= GLOBAL_SDPA_MIN_START else None
    fp32_acc = os.environ.get("MIMO_SDPA_FP32_ACC", "0") == "1"  # the grouped program needs bf16 dest
    return split if chunk == 256 and not fp32_acc else None


def page_table_width(blocks):
    """Width in blocks of a page table that covers ``blocks``: padded to whole SwaDecodeQkv.TABLE_READ pieces (int32
    entries). The entries past ``blocks`` are never read."""
    return pad_to(blocks, SwaDecodeQkv.TABLE_READ // 4)


class SwaDecodeQkv:
    """SWA decode, users in rows 0-3 (or 0-7: 8 users): the fused qkv row [1, 1, 32, (nh + 2) * D] (L1 tiles) -> the RoPE'd query in
    the height-sharded layout paged SDPA decode reads ([1, 32, nh, D], one [32, D] shard per user) and the RoPE'd key
    and the value written into the users' positions of the BFP8 paged caches, in one launch in place of
    nlp_create_qkv_heads_decode, two rotary_embedding_llama and two paged_update_cache.

    One core per query head (it reads rows 0-3 of its head's D / 32 tiles: every user at once, so RoPE is
    element-wise against rows 0-3 of cos / sin) and one key and one value core per user (read-modify-write of the
    user's cache tile row: BFP8 -> bf16, the new row patched in, -> BFP8)."""

    CB_X, CB_COS, CB_SIN, CB_TRANS, CB_TMP, CB_ROT, CB_CACHE, CB_C16, CB_SIG, CB_OUT8, CB_META = range(11)
    ROLE_Q, ROLE_K, ROLE_V = 0, 1, 2
    TABLE_READ = 64  # bytes of a page table row the reader fetches around a user's entry (page_table_width)

    def __init__(self, plan, num_heads, head_dim, users, block_size, trans_mat, sample_qkv):
        """``trans_mat``: the decode RoPE transformation tile (torch [32, 32]); ``sample_qkv``: a fused qkv tensor of
        the decode shape, used once to take the query layout nlp_create_qkv_heads_decode produces."""
        device = plan.mesh_device
        self.device = device
        self.nh, self.d, self.users, self.block_size = num_heads, head_dim, users, block_size
        self.rows = user_rows(users)
        assert head_dim % TILE == 0 and block_size % TILE == 0
        self.d_tiles = head_dim // TILE
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            sample_qkv, num_heads=num_heads, num_kv_heads=1, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        self.q_memory_config = q.memory_config()
        self.q_shape = list(q.shape)
        for t in (q, k, v):
            t.deallocate(True)
        shard_cores = ttnn.corerange_to_cores(self.q_memory_config.shard_spec.grid, row_wise=True)
        self.q_out = ttnn.from_torch(
            torch.zeros(self.q_shape),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=plan.replicate(),
            memory_config=self.q_memory_config,
        )
        self.user_noc = [device.worker_core_from_logical_core(shard_cores[u]) for u in range(users)]
        self.trans = ttnn.from_torch(
            trans_mat.reshape(1, 1, TILE, TILE).to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=plan.replicate(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        grid = device.compute_with_storage_grid_size()
        row_q, row_kv = 3, 4
        assert num_heads <= grid.x and users <= grid.x
        self.q_cores = [ttnn.CoreCoord(h, row_q) for h in range(num_heads)]
        self.k_cores = [ttnn.CoreCoord(u, row_kv) for u in range(users)]
        if 2 * users <= grid.x:  # K and V cores on one row
            self.v_cores = [ttnn.CoreCoord(users + u, row_kv) for u in range(users)]
        else:
            self.v_cores = [ttnn.CoreCoord(u, row_kv + 1) for u in range(users)]
        self.all_ranges = ttnn.CoreRangeSet(
            [ttnn.CoreRange(self.q_cores[0], self.q_cores[-1])]
            + [ttnn.CoreRange(c[0], c[-1]) for c in (self.k_cores, self.v_cores)]
        )
        if self.v_cores[0].y == row_kv:
            self.all_ranges = ttnn.CoreRangeSet(
                [ttnn.CoreRange(self.q_cores[0], self.q_cores[-1]), ttnn.CoreRange(self.k_cores[0], self.v_cores[-1])]
            )
        self.compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=False,
        )

    def _cbs(self, cache_dtype):
        tile = ttnn.Tile([TILE, TILE])

        def cb(index, dtype, pages):
            page = tile.get_tile_size(dtype)
            fmt = ttnn.CBFormatDescriptor(
                buffer_index=index, data_format=dtype, page_size=page, tile=ttnn.TileDescriptor(tile)
            )
            return ttnn.CBDescriptor(total_size=pages * page, core_ranges=self.all_ranges, format_descriptors=[fmt])

        n = self.d_tiles
        bf = ttnn.bfloat16
        return [
            cb(self.CB_X, bf, n),
            cb(self.CB_COS, bf, n),
            cb(self.CB_SIN, bf, n),
            cb(self.CB_TRANS, bf, 1),
            cb(self.CB_TMP, bf, 2),
            cb(self.CB_ROT, bf, n),
            cb(self.CB_CACHE, cache_dtype, n),
            cb(self.CB_C16, bf, n),
            cb(self.CB_SIG, bf, 1),
            cb(self.CB_OUT8, cache_dtype, n),
            cb(self.CB_META, bf, 1),
        ]

    def __call__(self, xqkv, cos, sin, pos, page_table, k_cache, v_cache):
        """xqkv rows 0-3 are the users; cos / sin [1, 1, 32, D] (DRAM tiles, rows = users' positions); pos int32 [32]
        and page_table int32 [32, blocks] (DRAM, row-major; pos < 0 leaves that user's cache alone). Returns the
        persistent query tensor (its users' head rows are rewritten every call; shared by every SWA layer)."""
        for t in (xqkv, cos, sin):
            assert t.layout == ttnn.TILE_LAYOUT and t.dtype == ttnn.bfloat16
        assert xqkv.memory_config().buffer_type == ttnn.BufferType.L1
        assert (
            cos.memory_config().buffer_type == ttnn.BufferType.DRAM
            and sin.memory_config().buffer_type == ttnn.BufferType.DRAM
        )
        assert (
            pos.memory_config().buffer_type == ttnn.BufferType.DRAM
            and page_table.memory_config().buffer_type == ttnn.BufferType.DRAM
        )
        cache_dtype = k_cache.dtype
        cache_tile = ttnn.Tile([TILE, TILE]).get_tile_size(cache_dtype)
        pos_page = pos.shape[-1] * 4
        pt_page = page_table.shape[-1] * 4
        assert pt_page % self.TABLE_READ == 0, "the reader fetches the 64 B of the page table row that holds the entry"
        n = self.d_tiles
        assert k_cache.shape[-1] == self.d and v_cache.shape[-1] % TILE == 0 and v_cache.shape[-1] <= self.d
        kernels = []
        # (role, cores, first qkv tile column, cache, tile columns): V cores move only the V cache's real columns.
        roles = (
            (self.ROLE_Q, self.q_cores, lambda i: i * n, None, n),
            (self.ROLE_K, self.k_cores, lambda i: self.nh * n, k_cache, n),
            (self.ROLE_V, self.v_cores, lambda i: (self.nh + 1) * n, v_cache, v_cache.shape[-1] // TILE),
        )
        user_noc = []
        for c in self.user_noc:
            user_noc += [c.x, c.y]
        for role, cores, first, cache, role_tiles in roles:
            ranges = ttnn.CoreRangeSet([ttnn.CoreRange(cores[0], cores[-1])])
            reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for i, c in enumerate(cores):
                reader_rt[c.x][c.y] = [
                    xqkv.buffer_address(),
                    first(i),
                    cos.buffer_address(),
                    sin.buffer_address(),
                    self.trans.buffer_address(),
                    i,
                    pos.buffer_address(),
                    page_table.buffer_address(),
                    cache.buffer_address() if cache is not None else 0,
                ]
                writer_rt[c.x][c.y] = [
                    i,
                    self.q_out.buffer_address(),
                    cache.buffer_address() if cache is not None else 0,
                ] + user_noc
                compute_rt[c.x][c.y] = []
            reader_ct = [
                role,
                self.CB_X,
                self.CB_COS,
                self.CB_SIN,
                self.CB_TRANS,
                self.CB_CACHE,
                self.CB_META,
                role_tiles,
                self.block_size,
                pos_page,
                pt_page,
                cache_tile,
                self.rows,
                self.users,
            ]
            writer_ct = [
                role,
                self.CB_X,
                self.CB_ROT,
                self.CB_META,
                self.CB_C16,
                self.CB_SIG,
                self.CB_OUT8,
                role_tiles,
                self.users,
                cache_tile,
            ]
            compute_ct = [
                role,
                self.CB_X,
                self.CB_COS,
                self.CB_SIN,
                self.CB_TRANS,
                self.CB_TMP,
                self.CB_ROT,
                self.CB_CACHE,
                self.CB_C16,
                self.CB_SIG,
                self.CB_OUT8,
                role_tiles,
            ]
            for name, ct, rt, config in (
                ("swa_qkv_reader.cpp", reader_ct, reader_rt, ttnn.ReaderConfigDescriptor()),
                ("swa_qkv_writer.cpp", writer_ct, writer_rt, ttnn.WriterConfigDescriptor()),
                ("swa_qkv_compute.cpp", compute_ct, compute_rt, self.compute_config),
            ):
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=f"{KERNEL_DIR}/attn/{name}",
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=ranges,
                        compile_time_args=ct,
                        runtime_args=rt,
                        config=config,
                    )
                )
        program = ttnn.ProgramDescriptor(kernels=kernels, cbs=self._cbs(cache_dtype), semaphores=[])
        ttnn.generic_op([xqkv, cos, sin, self.trans, pos, page_table, k_cache, v_cache, self.q_out], program)
        return self.q_out


class ConcatHeads:
    """Decode SDPA output [1, B, heads, D] (bf16 tiles, interleaved) -> the concatenated heads [1, 1, B, heads * vd]
    in L1 (bf16 tiles, interleaved), without the V padding columns D > vd. With ``mask`` ([1, 1, B, 1] bf16 0 / 1
    tiles, interleaved) the rows of masked users are zeros. One data-movement generic_op in place of slice, reshard,
    nlp_concat_heads_decode and the copy to L1 (and the owner-mask mul); a unit is one output tile's 16-row half."""

    MAX_CORES = 64
    GRID_X = 8

    def __init__(self, plan, num_heads, v_head_dim):
        assert v_head_dim % TILE == 0 and num_heads <= TILE
        self.device = plan.mesh_device
        self.num_heads = num_heads
        self.v_tiles = v_head_dim // TILE
        self.width = num_heads * v_head_dim
        units = 2 * num_heads * self.v_tiles
        cores = self.MAX_CORES
        while cores > self.GRID_X and units % cores:
            cores -= self.GRID_X
        assert units % cores == 0, (units, cores)
        self.units_per_core = units // cores
        rows = cores // self.GRID_X
        self.cores = [ttnn.CoreCoord(x, y) for y in range(rows) for x in range(self.GRID_X)]
        self.core_ranges = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.GRID_X - 1, rows - 1))]
        )
        full = ttnn.Tile([TILE, TILE])
        page = full.get_tile_size(ttnn.bfloat16)
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=0, data_format=ttnn.bfloat16, page_size=page, tile=ttnn.TileDescriptor(full)
        )
        # units_per_core 1 KB halves and the mask tile
        self.cbs = [
            ttnn.CBDescriptor(
                total_size=(self.units_per_core + 1) * page, core_ranges=self.core_ranges, format_descriptors=[fmt]
            )
        ]

    def __call__(self, attn, mask=None):
        shape = list(attn.shape)
        mc = attn.memory_config()
        assert attn.layout == ttnn.TILE_LAYOUT and attn.dtype == ttnn.bfloat16
        assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        # shape[1]: the SDPA's batch, 32 or a compact batch of the first users (the other output rows are zeros)
        assert len(shape) == 4 and shape[0] == 1 and shape[1] <= TILE and shape[2] >= self.num_heads
        assert shape[3] % TILE == 0
        d_tiles = shape[3] // TILE
        assert self.v_tiles <= d_tiles
        user_pages = pad_to(shape[2], TILE) // TILE * d_tiles
        in_dram = 1 if mc.buffer_type == ttnn.BufferType.DRAM else 0
        has_mask = mask is not None
        mask_dram = 1
        if has_mask:
            mmc = mask.memory_config()
            assert mask.layout == ttnn.TILE_LAYOUT and mask.dtype == ttnn.bfloat16
            assert mmc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED and mask.shape[-2] == TILE
            mask_dram = 1 if mmc.buffer_type == ttnn.BufferType.DRAM else 0

        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, TILE, self.width]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.device,
            ttnn.L1_MEMORY_CONFIG,
        )
        mask_addr = mask.buffer_address() if has_mask else 0
        rt = ttnn.RuntimeArgs()
        for i, core in enumerate(self.cores):
            rt[core.x][core.y] = [attn.buffer_address(), out.buffer_address(), i * self.units_per_core, mask_addr]
        kernel = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/attn/attn_concat_heads.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core_ranges,
            compile_time_args=[
                0,
                self.units_per_core,
                user_pages,
                d_tiles,
                self.v_tiles,
                in_dram,
                1 if has_mask else 0,
                mask_dram,
                shape[1],
            ],
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_0_default
            ),
        )
        program = ttnn.ProgramDescriptor(kernels=[kernel], cbs=self.cbs, semaphores=[])
        io = [attn] + ([mask] if has_mask else []) + [out]
        ttnn.generic_op(io, program)
        return out


def kv_heads_for_chip(chip, tp, num_kv_heads):
    if tp >= num_kv_heads:
        return [chip // (tp // num_kv_heads)]
    per = num_kv_heads // tp
    return list(range(chip * per, (chip + 1) * per))


def prepare_qkv_o(w, dims, v_scale, tp):
    """Host-side per-chip fused QKV ([hidden, tp * local_qkv]) and padded o_proj ([heads * head_dim, hidden])."""
    hd, vd = dims.head_dim, dims.v_head_dim
    nh_local = dims.num_heads // tp
    q = permute_heads_for_rope(w["q_proj"], dims.num_heads, hd, dims.rope_dim)
    k = permute_heads_for_rope(w["k_proj"], dims.num_kv_heads, hd, dims.rope_dim)
    v = w["v_proj"] * (v_scale if v_scale is not None else 1.0)
    per_chip = []
    for c in range(tp):
        rows = [q[c * nh_local * hd : (c + 1) * nh_local * hd]]
        kv_ids = kv_heads_for_chip(c, tp, dims.num_kv_heads)
        rows += [k[h * hd : (h + 1) * hd] for h in kv_ids]
        rows += [torch.nn.functional.pad(v[h * vd : (h + 1) * vd], (0, 0, 0, hd - vd)) for h in kv_ids]
        per_chip.append(torch.cat(rows))
    wqkv = torch.cat(per_chip).T.contiguous()

    o = w["o_proj"].T.reshape(dims.num_heads, vd, -1)
    o = torch.nn.functional.pad(o, (0, 0, 0, hd - vd)).reshape(dims.num_heads * hd, -1)
    return wqkv, o.contiguous()


def prepare_grouped_decode_qkv_o(w, dims, v_scale, tp):
    """Decode weights for chips that share a KV head: every chip gets all Q heads of its group.

    Fused QKV per chip: [16 Q heads x 192 | K 192 | V padded to 192] (the decode head-split op wants one
    head dim; only the first 128 V columns are cached). o_proj per chip: the group's 16 x 128 rows, unpadded."""
    hd, vd = dims.head_dim, dims.v_head_dim
    group = dims.num_heads // dims.num_kv_heads
    rep = kv_replication(tp, dims.num_kv_heads)
    q = permute_heads_for_rope(w["q_proj"], dims.num_heads, hd, dims.rope_dim)
    k = permute_heads_for_rope(w["k_proj"], dims.num_kv_heads, hd, dims.rope_dim)
    v = w["v_proj"] * (v_scale if v_scale is not None else 1.0)
    o = w["o_proj"].T.reshape(dims.num_heads, vd, -1)
    qkv_chips, o_chips = [], []
    for c in range(tp):
        g = c // rep
        qkv_chips.append(
            torch.cat(
                [
                    q[g * group * hd : (g + 1) * group * hd],
                    k[g * hd : (g + 1) * hd],
                    torch.nn.functional.pad(v[g * vd : (g + 1) * vd], (0, 0, 0, hd - vd)),
                ]
            )
        )
        o_chips.append(o[g * group : (g + 1) * group].reshape(group * vd, -1))
    return torch.cat(qkv_chips).T.contiguous(), torch.cat(o_chips).contiguous()


class TtMiMoAttention:
    def __init__(
        self,
        plan,
        cfg,
        layer_idx,
        weights,
        rope_setup,
        cache_dir=None,
        weight_dtype=ttnn.bfloat8_b,
    ):
        """``weights``: dict from MiMoCheckpoint.attention (or None to load from a warm cache)."""
        self.plan = plan
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.dims = dims = cfg.attention_dims(layer_idx)
        tp = plan.tp
        assert dims.num_heads % tp == 0
        assert tp % dims.num_kv_heads == 0 or dims.num_kv_heads % tp == 0
        self.nh_local = dims.num_heads // tp
        self.nkv_local = len(kv_heads_for_chip(0, tp, dims.num_kv_heads))
        self.head_dim = dims.head_dim
        self.scale = dims.scaling
        self.sliding_window = dims.sliding_window
        self.is_swa = dims.is_swa
        self.rep = kv_replication(tp, dims.num_kv_heads)
        self.split_users = self.rep > 1  # global layers on TP=8: the chip pair splits the users
        self.kv_both_chips = False  # keep_kv_on_both_chips
        self.wo_v = None
        self.v_head_dim = dims.v_head_dim
        self.rope = rope_setup
        prefix = f"layer{layer_idx}_attn_tp{tp}"

        wqkv = wo = sinks = decode_sinks = None
        if weights is not None:
            wqkv, wo = prepare_qkv_o(weights, dims, cfg.attention_value_scale, tp)
            if dims.has_sink:
                s = weights["attention_sink_bias"].float() / self.scale
                sinks = s.reshape(1, dims.num_heads, 1, 1)
                decode_sinks = torch.nn.functional.pad(s.reshape(dims.num_heads, 1), (0, TILE - 1))

        hidden = cfg.hidden_size
        self.qkv_stream = self.o_stream = None
        self.swa_qkv = None  # set by the model: the shared SwaDecodeQkv of the SWA layers' compact decode
        self.fabric_gather = False  # set by the model: forward_decode(compact=True) returns the un-gathered rows
        self.mm_config = compute_config(ttnn.MathFidelity.HiFi2)
        self.qkv_width = (self.nh_local + 2 * self.nkv_local) * self.head_dim
        self.wo = as_weight(wo, plan, plan.shard(-2), weight_dtype, cache_name(cache_dir, prefix, "wo"))
        if self.split_users:
            # Prefill keeps 8 heads per chip on an interleaved weight; decode has its own grouped weight.
            self.wqkv = as_weight(wqkv, plan, plan.shard(-1), weight_dtype, cache_name(cache_dir, prefix, "wqkv"))
            self.nh_decode = dims.num_heads // dims.num_kv_heads
            self.decode_qkv_width = (self.nh_decode + 2) * self.head_dim
            self.decode_qkv = DramShardedMatmul(plan, hidden, self.decode_qkv_width)
            wqkv_d = wo_d = None
            if weights is not None:
                wqkv_d, wo_d = prepare_grouped_decode_qkv_o(weights, dims, cfg.attention_value_scale, tp)
                wqkv_d = self.decode_qkv.pad_chip_blocks(wqkv_d, tp)
            self.wqkv_decode = self._interleaved_qkv(
                as_weight(
                    wqkv_d,
                    plan,
                    plan.shard(-1),
                    weight_dtype,
                    cache_name(cache_dir, prefix, "wqkv_grouped_ds"),
                    memory_config=self.decode_qkv.weight_memory_config,
                ),
                self.decode_qkv_width,
            )
            self.wo_decode = as_weight(
                wo_d, plan, plan.shard(-2), weight_dtype, cache_name(cache_dir, prefix, "wo_grouped")
            )
            self.decode_qkv_program = decode_linear_config(plan.mesh_device, hidden, self.decode_qkv_width)
            self.decode_o_program = decode_linear_config(plan.mesh_device, self.nh_decode * self.v_head_dim, hidden)
            # Bank-local streaming o projection (2048 x 4096 per chip: 23.9 us instead of 35.2 us).
            self.o_stream = self._bank_stream(self.nh_decode * self.v_head_dim, hidden)
        else:
            # The cached weight is DRAM-sharded and padded per chip; decode and prefill both read an unpadded
            # interleaved copy made once here.
            self.decode_qkv = DramShardedMatmul(plan, hidden, self.qkv_width)
            if wqkv is not None:
                wqkv = self.decode_qkv.pad_chip_blocks(wqkv, tp)
            self.wqkv = self._interleaved_qkv(
                as_weight(
                    wqkv,
                    plan,
                    plan.shard(-1),
                    weight_dtype,
                    cache_name(cache_dir, prefix, "wqkv_ds"),
                    memory_config=self.decode_qkv.weight_memory_config,
                ),
                self.qkv_width,
            )
            # Decode o_proj without the V padding rows (they only ever multiplied zeros), sliced on device; streamed
            # bank-locally (1024 x 4096 per chip: 13.2 us instead of the linear's 18.9) straight into compact rows.
            self.wo_decode = self._unpadded_o()
            self.o_stream = self._bank_stream(self.nh_local * self.v_head_dim, hidden)
            self.decode_qkv_program = decode_linear_config(plan.mesh_device, hidden, self.qkv_width)
            self.decode_o_program = decode_linear_config(plan.mesh_device, self.nh_local * self.v_head_dim, hidden)
            # Bank-local streaming qkv (4096 x 1920 per chip, read padded to 2048 columns once: 23.9 us instead of
            # 32.4 us). It reads its own padded copy; prefill keeps the interleaved weight. The SWA o projection
            # (1024 x 4096) streams too (o_stream above). It runs on the cores after each bank core (as measured).
            self.qkv_stream = self._bank_stream(hidden, self.qkv_width, skip_bank_core=True)
            if self.qkv_stream is not None:
                self.wqkv_stream = self.qkv_stream.prepare(self.wqkv)
        self.sinks = self.decode_sinks = None
        if dims.has_sink:
            self.sinks = as_weight(sinks, plan, plan.shard(1), ttnn.bfloat16, cache_name(cache_dir, prefix, "sinks"))
            self.decode_sinks = as_weight(
                decode_sinks, plan, plan.shard(0), ttnn.bfloat16, cache_name(cache_dir, prefix, "decode_sinks")
            )

        # Decode with the users in rows 0-3 (forward_decode(compact=True)): the o stream writes rows 0-3 of its output
        # into a persistent compact [1, 1, 32, hidden / 8] tensor, so the row reduction moves 1/8 of the tile (rows
        # 0-7 into [1, 1, 32, hidden / 4] with 8 users: set_decode_rows).
        self.rows = 4
        self.compact_buf = zero_output(plan, hidden // (TILE // self.rows))
        # Decode: SDPA output -> concatenated heads (V's real dims) in one launch.
        self.concat_heads = ConcatHeads(plan, self.nh_decode if self.split_users else self.nh_local, self.v_head_dim)

        # Chunked prefill of a global layer: 1 on the chips that own a slot of each parity (one user at a time).
        self.prefill_owner_masks = None
        if self.split_users:
            self.prefill_owner_masks = {
                p: per_chip(plan, m, ttnn.bfloat16, ttnn.TILE_LAYOUT) for p, m in slot_owner_masks(tp, self.rep).items()
            }

        # MIMO_SDPA_FP32_ACC=1: prefill SDPA accumulates in fp32 dest (positions >= 2 KL -9% with stock-packed experts).
        # Off by default: its larger circular buffers clash with decode's persistent L1 buffers once the vLLM adapter
        # warms the 2048-token prefill chunks. Decode SDPA stays in bf16 dest (fp32 dest gave non-finite values).
        # Prefill SDPA at HiFi2 (MIMO_PREFILL_SDPA_FIDELITY overrides): a global layer's chunk at 128K is compute bound,
        # 78.3 -> 46.7 ms; top-1, the chunked-prefill PCCs and the 32K pass keys are unchanged (tp8_dp4_plan.md).
        self.sdpa_config = compute_config(
            getattr(ttnn.MathFidelity, os.environ.get("MIMO_PREFILL_SDPA_FIDELITY", "HiFi2")),
            fp32_acc=os.environ.get("MIMO_SDPA_FP32_ACC", "0") == "1",
            packer_l1_acc=False,
        )
        self.decode_sdpa_config = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=False, packer_l1_acc=False)
        self.decode_sdpa_program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=0,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        self._split_sdpa_programs = {}
        # The KV pair split (vLLM mode, MIMO_KV_PAIR_SPLIT): the model's PairSplit, on the global layers.
        self.pair = None
        # The zero query rows in front of a later SWA chunk (its cached window's rows), {lead rows: tensor}, shared by
        # the SWA layers and made before any trace capture (TtMiMoModel._setup_lead_zeros): ttnn.zeros writes from the
        # host, which a captured later chunk cannot do. Missing: made per chunk.
        self.lead_zeros = {}

    def _split_sdpa_program(self, batch, table_tokens):
        """Global decode SDPA program for a batch of ``batch`` page-table rows: every core of the 8 x 8 grid works
        (64 / batch per user; the tree reduction allows up to 64) and 256-token K chunks where the table allows."""
        if batch >= TILE:
            return self.decode_sdpa_program
        key = (batch, table_tokens)
        if key not in self._split_sdpa_programs:
            grid = ttnn.CoreCoord(8, 8)
            if os.environ.get("MIMO_SDPA_FULL_GRID", "0") == "1":
                grid = self.plan.mesh_device.compute_with_storage_grid_size()
            self._split_sdpa_programs[key] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid,
                q_chunk_size=0,
                k_chunk_size=256 if table_tokens % 256 == 0 else 128,
                exp_approx_mode=False,
                max_cores_per_head_batch=max(16, min(64, grid.x * grid.y // batch)),
            )
        return self._split_sdpa_programs[key]

    def _interleaved_qkv(self, w, width):
        """DRAM-sharded fused qkv [K, n_pad] per chip -> DRAM-interleaved [K, width], padding columns dropped.

        The decode 1D matmul reads it with no input reshard, output copy or trimming slice; prefill reads it
        directly. The slice is tile-aligned, so the BFP8 tiles are copied unchanged."""
        k, n_pad = w.shape[-2], w.shape[-1]
        flat = ttnn.to_memory_config(w, ttnn.DRAM_MEMORY_CONFIG)
        w.deallocate(True)
        if n_pad == width:
            return flat
        cols = ttnn.reshape(flat, (1, 1, k, n_pad))
        real = ttnn.slice(cols, (0, 0, 0, 0), (1, 1, k, width), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        flat.deallocate(True)
        return ttnn.reshape(real, (k, width))

    def keep_kv_on_both_chips(self):
        """Both chips of the pair keep every slot's K / V (vLLM's caches, or TtMiMoModel(replicate_global_kv=True)), so a
        later prefill chunk runs 8 of the group's 16 heads on each chip (_prefill_pair_chunk) instead of all 16 on both
        and discarding one. Its o projection reads the prefill o_proj without the V padding rows, made here: before
        any trace is captured."""
        assert self.split_users
        self.kv_both_chips = True
        if self.wo_v is None:
            self.wo_v = self._unpadded_o()

    def _unpadded_o(self):
        """Per-chip padded o_proj [nh_local * head_dim, hidden] -> [nh_local * v_head_dim, hidden], on device.

        The slice is tile-aligned (192 -> 128 rows per head), so the BFP8 tiles are copied unchanged."""
        hidden = self.wo.shape[-1]
        heads = ttnn.reshape(self.wo, (1, self.nh_local, self.head_dim, hidden))
        real = ttnn.slice(
            heads,
            (0, 0, 0, 0),
            (1, self.nh_local, self.v_head_dim, hidden),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.reshape(real, (self.nh_local * self.v_head_dim, hidden))

    # ---------------------------------------------------------------- KV cache
    def kv_tokens_per_user(self, max_seq_len):
        """Positions a user's page table row addresses: the whole context (SWA layers too: their table is indexed
        by absolute position, and kv.SwaBlockTable keeps only the in-window blocks real)."""
        return max_seq_len

    def allocate_kv_cache(self, num_blocks, block_size, dtype=ttnn.bfloat8_b):
        """[K, V] caches ``[num_blocks, kv_heads, block_size, dim]`` per chip; V keeps its 128 real dims."""
        caches = []
        for dim in (self.head_dim, self.v_head_dim):
            # Allocate and zero on device: at 1M context a host torch.zeros + conversion would move ~GBs per layer.
            cache = ttnn.allocate_tensor_on_device(
                ttnn.Shape((num_blocks, self.nkv_local, block_size, dim)),
                dtype,
                ttnn.TILE_LAYOUT,
                self.plan.mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.fill(cache, 0.0, output_tensor=cache)
            caches.append(cache)
        return caches

    # ---------------------------------------------------------------- prefill
    def _prefill_qkv(self, x):
        assert self.wqkv.shape[-1] == self.qkv_width, (self.wqkv.shape, self.qkv_width)
        return ttnn.linear(
            x,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_config,
        )

    def _bank_stream(self, k, n, skip_bank_core=False):
        """A BankStreamMatmul for a [k, n] decode projection, or None when its layout constraints do not hold."""
        try:
            return BankStreamMatmul(self.plan, k, n, self.mm_config, skip_bank_core=skip_bank_core)
        except AssertionError:
            return None

    def make_swa_qkv(self, users, block_size):
        """A SwaDecodeQkv for this (SWA) layer's shapes, to share between the SWA layers."""
        assert not self.split_users and self.nkv_local == 1 and self.is_swa
        sample = ttnn.from_torch(
            torch.zeros(1, 1, TILE, self.qkv_width),
            device=self.plan.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self.plan.replicate(),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        trans = ttnn.to_torch(ttnn.get_device_tensors(self.rope.transformation_mat)[0]).float().reshape(-1, TILE, TILE)
        op = SwaDecodeQkv(self.plan, self.nh_local, self.head_dim, users, block_size, trans[0], sample)
        sample.deallocate(True)
        return op

    def set_decode_rows(self, rows):
        """The compact decode's user rows: 4 (up to 4 decode users) or 8. Before any trace capture (the compact rows
        buffer is persistent)."""
        if rows == self.rows:
            return
        self.compact_buf.deallocate(True)
        self.rows = rows
        self.compact_buf = zero_output(self.plan, self.cfg.hidden_size // (TILE // rows))

    def _decode_qkv(self, x, weight, program_config, compact=False):
        # Keep the fused qkv row in L1: the decode head split reads it with many small per-user row reads. The bank
        # stream computes the user rows only, so it serves the compact (users in rows 0-3 or 0-7) decode.
        if compact and weight is self.wqkv and self.qkv_stream is not None:
            return self.qkv_stream(x, self.wqkv_stream, rows=self.rows)
        return ttnn.linear(
            x,
            weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=program_config,
            compute_kernel_config=self.mm_config,
        )

    def _prefill_sdpa_program(self, seq_len, start_pos=0, later_chunk=False):
        """Q / K chunk sizes that divide ``seq_len`` (and ``start_pos``: chunked SDPA needs that). ``later_chunk``: a
        global layer's chunk after the first, which takes the grouped / K split program (global_sdpa_split)."""
        chunk = 256 if seq_len >= 2048 else (128 if seq_len >= 128 else 32)
        while seq_len % chunk or start_pos % chunk:
            chunk //= 2
        split = later_chunk_split(chunk, start_pos) if later_chunk else None
        if split is not None:
            heads, splits = split
            return ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.plan.mesh_device.compute_with_storage_grid_size(),
                q_chunk_size=chunk,
                k_chunk_size=chunk,
                exp_approx_mode=False,
                q_heads_per_unit=heads,
                k_splits=splits,
            )
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=chunk,
            k_chunk_size=chunk,
            exp_approx_mode=False,
        )

    def _rope_prefill(self, q, k, start_pos, seq_len):
        cos, sin = self.rope.get_prefill_rot_mats(start_pos, seq_len)
        trans = self.rope.transformation_mat_prefill
        q_rot = ttnn.experimental.rotary_embedding_llama(q, cos, sin, trans, is_decode_mode=False)
        k_rot = ttnn.experimental.rotary_embedding_llama(k, cos, sin, trans, is_decode_mode=False)
        for t in (q, k, *((cos, sin) if self.rope.owns_prefill_rot_mats else ())):
            t.deallocate(True)
        return q_rot, k_rot

    def _fill(self, kv_cache, k_rot, v, page_table, user_id, tables):
        """Write the chunk's K and V (only its first ``v_head_dim`` columns when the V cache is that narrow) into
        the user's cache blocks for positions start_pos .. start_pos + S - 1: the blocks of ``tables.fill``, or
        (no ``tables``, a first chunk) row ``user_id`` of ``page_table``."""
        k_cache, v_cache = kv_cache
        seq_len = k_rot.shape[-2]
        table, batch_idx = (page_table, user_id) if tables is None else (tables.fill, 0)
        batch = {"batch_idx": batch_idx}
        if isinstance(tables, BatchFillTables):  # input batch entry r writes through row r of tables.fill
            batch = {"batch_idx_tensor": tables.rows}
        k_fill = ttnn.typecast(k_rot, k_cache.dtype)
        if v_cache.shape[-1] < v.shape[-1]:
            v_real = ttnn.slice(v, (0, 0, 0, 0), (v.shape[0], v.shape[1], seq_len, v_cache.shape[-1]))
            v_fill = ttnn.typecast(v_real, v_cache.dtype)
            v_real.deallocate(True)
        else:
            v_fill = ttnn.typecast(v, v_cache.dtype)
        ttnn.experimental.paged_fill_cache(k_cache, k_fill, table, **batch)
        ttnn.experimental.paged_fill_cache(v_cache, v_fill, table, **batch)
        k_fill.deallocate(True)
        v_fill.deallocate(True)

    def fill_kv_rows(self, x, cos, sin, page_table, kv_cache):
        """The K / V of S rows (x [1, 1, S, hidden], input-normed; S a multiple of the block size) at the positions
        whose RoPE cos / sin [1, 1, S, head_dim] are given, written by paged_fill_cache into the blocks of row 0 of
        ``page_table`` from its first position: the entries a decode step would write for each row, computed in the
        prefill layout and without the attention (the MTP ring backfill, tt/mtp_backfill.py)."""
        assert not self.split_users and self.sliding_window is not None
        xqkv = self._prefill_qkv(x)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.nh_local,
            num_kv_heads=self.nkv_local,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        xqkv.deallocate(True)
        q.deallocate(True)
        trans = self.rope.transformation_mat_prefill
        k_rot = ttnn.experimental.rotary_embedding_llama(k, cos, sin, trans, is_decode_mode=False)
        k.deallocate(True)
        self._fill(kv_cache, k_rot, v, page_table, 0, None)
        k_rot.deallocate(True)
        v.deallocate(True)

    def _cached_window(self, cache, blocks):
        """The cache blocks ``blocks`` (int32 [4] slice starts (block, 0, 0, 0) on device, in position order) as one
        bf16 [1, kv_heads, n * block, dim]. A tensor-args slice keys on the cache's shape, not on the block id."""
        parts = []
        for start in blocks:
            part = ttnn.slice(cache, start, start, slice_dim=0, num_devices=cache.shape[0])
            parts.append(ttnn.typecast(part, ttnn.bfloat16))
            part.deallocate(True)
        out = ttnn.concat(parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG) if len(parts) > 1 else parts[0]
        if len(parts) > 1:
            for t in parts:
                t.deallocate(True)
        return out

    def make_lead_zeros(self, lead):
        """Zero query rows [1, nh_local, lead, head_dim] for the cached window in front of a later SWA chunk."""
        return ttnn.zeros(
            (1, self.nh_local, lead, self.head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.plan.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward_prefill(self, x, user_id, page_table, kv_cache, start_pos=0, real_len=None, tables=None, users=1):
        """x [1, 1, S, hidden] (S a multiple of 32): positions start_pos .. start_pos + S - 1 of the user in slot
        ``user_id`` (``real_len`` real tokens, then padding whose K/V land past them, where decode overwrites them).
        ``start_pos`` is a multiple of the block size; the cache must hold the user's earlier positions.
        ``tables`` (kv.PrefillTables of this layer's page table; required when start_pos > 0): the blocks the chunk
        writes, and for a later chunk the window blocks it reads (SWA) or its whole context and start (global).
        Without it a first chunk writes row ``user_id`` of ``page_table``.

        ``users`` R > 1: first chunks of R users at once, x [1, 1, R * S, hidden] user after user (each padded to S),
        ``tables`` their kv.BatchFillTables. The projections see R * S rows; attention runs as a batch of R sequences.
        """
        assert start_pos == 0 or tables is not None, "a later chunk needs its kv.PrefillTables"
        assert users == 1 or (start_pos == 0 and isinstance(tables, BatchFillTables)), "a batch is first chunks"
        assert self.pair is None or tables is not None, "a pair-split chunk writes through its fill tables"
        with steps() as step:  # profiling zones (profiling.py)
            if start_pos > 0 and self.split_users:
                if self.pair is not None:
                    return self._prefill_kv_split_chunk(x, kv_cache, start_pos, tables, step)
                if self.kv_both_chips:
                    return self._prefill_pair_chunk(x, user_id, kv_cache, start_pos, tables, step)
                return self._prefill_owner_chunk(x, user_id, kv_cache, start_pos, tables, step)
            return self._prefill_chunk(x, user_id, page_table, kv_cache, start_pos, tables, users, step)

    def _prefill_chunk(self, x, user_id, page_table, kv_cache, start_pos, tables, users, step):
        """forward_prefill of a first chunk, or of an SWA layer's later chunk (window from the cache)."""
        seq_len = x.shape[-2] // users
        step("qkv")
        xqkv = self._prefill_qkv(x)
        if users > 1:
            xqkv = ttnn.reshape(xqkv, (users, 1, seq_len, xqkv.shape[-1]))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.nh_local,
            num_kv_heads=self.nkv_local,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        xqkv.deallocate(True)
        step("rope")
        q_rot, k_rot = self._rope_prefill(q, k, start_pos, seq_len)

        step("kv_cache")
        lead = 0
        if start_pos > 0:
            # SWA: the window before the chunk from the cache, read before the fill can reuse any of its blocks.
            assert self.sliding_window is not None and tables.window, "a later SWA chunk needs its window blocks"
            k_cache, v_cache = kv_cache
            lead = len(tables.window) * k_cache.shape[-2]
            k_win = self._cached_window(k_cache, tables.window)
            v_win = self._cached_window(v_cache, tables.window)
            if v_win.shape[-1] < v.shape[-1]:  # the cache keeps V's real dims; the SDPA sees the padded head dim
                padded = ttnn.pad(v_win, [(0, 0), (0, 0), (0, 0), (0, v.shape[-1] - v_win.shape[-1])], 0.0)
                v_win.deallocate(True)
                v_win = padded
        self._fill(kv_cache, k_rot, v, page_table, user_id, tables)
        step("sdpa")
        if lead:
            q_lead = self.lead_zeros.get(lead)
            own_lead = q_lead is None
            if own_lead:
                q_lead = self.make_lead_zeros(lead)
            joined = []
            for head, tail in ((q_lead, q_rot), (k_win, k_rot), (v_win, v)):
                joined.append(ttnn.concat([head, tail], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                if head is not q_lead or own_lead:
                    head.deallocate(True)
                tail.deallocate(True)
            q_rot, k_rot, v = joined

        attn = ttnn.transformer.scaled_dot_product_attention(
            q_rot,
            k_rot,
            v,
            is_causal=True,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            attention_sink=self.sinks,
            program_config=self._prefill_sdpa_program(lead + seq_len),
            compute_kernel_config=self.sdpa_config,
        )
        q_rot.deallocate(True)
        k_rot.deallocate(True)
        v.deallocate(True)
        if lead:
            chunk = ttnn.slice(attn, (0, 0, lead, 0), (1, self.nh_local, lead + seq_len, self.head_dim))
            attn.deallocate(True)
            attn = chunk
        step("o_proj")
        attn_cat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn.deallocate(True)
        if users > 1:  # [R, 1, S, heads * dim] -> the users' rows back to back (after the head concat: see above)
            attn_cat = ttnn.reshape(attn_cat, (1, 1, users * seq_len, attn_cat.shape[-1]))
        out = ttnn.linear(
            attn_cat,
            self.wo,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_config,
        )
        attn_cat.deallocate(True)
        step("all_reduce")
        return self.plan.all_reduce(out)

    def _prefill_pair_chunk(self, x, user_id, kv_cache, start_pos, tables, step):
        """A global layer's chunk after the first when both chips of the pair hold the user's K / V: each chip runs its
        8 prefill heads (``wqkv``, as a first chunk does) over the whole cached context, and the row all-reduce adds
        the chips' o projections."""
        assert self.sinks is None and self.sliding_window is None
        seq_len = x.shape[-2]
        step("qkv")
        xqkv = self._prefill_qkv(x)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.nh_local,
            num_kv_heads=self.nkv_local,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        xqkv.deallocate(True)
        step("rope")
        q_rot, k_rot = self._rope_prefill(q, k, start_pos, seq_len)
        step("kv_cache")
        self._fill(kv_cache, k_rot, v, None, user_id, tables)
        k_rot.deallocate(True)
        v.deallocate(True)

        step("sdpa")
        k_cache, v_cache = kv_cache
        assert tables.context.shape[-1] * k_cache.shape[-2] >= start_pos + seq_len, "chunk runs past its context table"
        start = dict(chunk_start_idx_tensor=tables.start) if MLA_RUNTIME_START else dict(chunk_start_idx=start_pos)
        attn = ttnn.transformer.chunked_flash_mla_prefill(
            q_rot,
            k_cache,
            self.v_head_dim,
            tables.context,
            **start,
            scale=self.scale,
            program_config=self._prefill_sdpa_program(seq_len, start_pos, later_chunk=True),
            compute_kernel_config=self.sdpa_config,
            input_tensor_v=v_cache,
        )  # [1, 8, S, 128]
        q_rot.deallocate(True)
        step("o_proj")
        attn_cat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn.deallocate(True)
        out = ttnn.linear(
            attn_cat,
            self.wo_v,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_config,
        )
        attn_cat.deallocate(True)
        step("all_reduce")
        return self.plan.all_reduce(out)

    def _prefill_owner_chunk(self, x, user_id, kv_cache, start_pos, tables, step):
        """A global layer's chunk after the first: the group's 16 Q heads over the whole cached context on the
        chip that owns the user (the other chip of the pair runs the same program on its scratch block and its
        result is zeroed)."""
        assert self.sinks is None and self.sliding_window is None
        seq_len = x.shape[-2]
        step("qkv")
        xqkv = ttnn.linear(
            x,
            self.wqkv_decode,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_config,
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.nh_decode,
            num_kv_heads=1,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        xqkv.deallocate(True)
        step("rope")
        q_rot, k_rot = self._rope_prefill(q, k, start_pos, seq_len)
        step("kv_cache")
        self._fill(kv_cache, k_rot, v, None, user_id, tables)
        k_rot.deallocate(True)
        v.deallocate(True)

        step("sdpa")
        k_cache, v_cache = kv_cache
        assert tables.context.shape[-1] * k_cache.shape[-2] >= start_pos + seq_len, "chunk runs past its context table"
        # The context table is one fixed width for every chunk; the kernel reads only its blocks before the chunk's end.
        start = dict(chunk_start_idx_tensor=tables.start) if MLA_RUNTIME_START else dict(chunk_start_idx=start_pos)
        attn = ttnn.transformer.chunked_flash_mla_prefill(
            q_rot,
            k_cache,
            self.v_head_dim,
            tables.context,
            **start,
            scale=self.scale,
            program_config=self._prefill_sdpa_program(seq_len, start_pos, later_chunk=True),
            compute_kernel_config=self.sdpa_config,
            input_tensor_v=v_cache,
        )  # [1, 16, S, 128]
        q_rot.deallocate(True)
        step("o_proj")
        attn_cat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn.deallocate(True)
        owned = ttnn.mul(attn_cat, self.prefill_owner_masks[user_id % self.rep], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_cat.deallocate(True)
        out = ttnn.linear(
            owned,
            self.wo_decode,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_config,
        )
        owned.deallocate(True)
        step("all_reduce")
        return self.plan.all_reduce(out)

    def _prefill_kv_split_chunk(self, x, kv_cache, start_pos, tables, step):
        """A global layer's chunk after the first under the KV pair split: the group's 16 Q heads over the chip's local
        keys of the whole context (pair-split chunked MLA prefill, ``tables.context`` the global table from position
        0), the chunk's own K / V written first into the chip's halves (``tables.fill``), then the merge (both chips
        carry every query)."""
        assert self.sinks is None and self.sliding_window is None
        seq_len = x.shape[-2]
        assert start_pos % 128 == 0 and seq_len % 128 == 0, "pair-split chunks start and end on 128-token blocks"
        step("qkv")
        xqkv = ttnn.linear(
            x,
            self.wqkv_decode,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_config,
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.nh_decode,
            num_kv_heads=1,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        xqkv.deallocate(True)
        step("rope")
        q_rot, k_rot = self._rope_prefill(q, k, start_pos, seq_len)
        step("kv_cache")
        self._fill(kv_cache, k_rot, v, None, 0, tables)
        k_rot.deallocate(True)
        v.deallocate(True)

        step("sdpa")
        k_cache = kv_cache[0]
        assert tables.context.shape[-1] * 2 * k_cache.shape[-2] >= start_pos + seq_len, "chunk runs past its context"
        attn, lse = self._chunked_mla_prefill_lse(q_rot, kv_cache, tables, start_pos)  # [1, 16, S, 128], [1, 16, S, 1]
        q_rot.deallocate(True)
        step("pair_merge")
        merged = self.pair.merge(attn, lse, ttnn.DRAM_MEMORY_CONFIG)
        attn.deallocate(True)
        lse.deallocate(True)
        attn_cat = ttnn.experimental.nlp_concat_heads(merged, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        merged.deallocate(True)
        step("o_proj")
        out = ttnn.linear(
            attn_cat,
            self.wo_decode,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_config,
        )
        attn_cat.deallocate(True)
        step("all_reduce")
        return self.plan.all_reduce(out)

    def _chunked_mla_prefill_lse(self, q, kv_cache, tables, start_pos):
        """Pair-split chunked MLA prefill (the ops' interface in runs/kvsplit/SPEC.md): queries q [1, nh, S, D] at
        global positions start .. start + S - 1 (``tables.start``) over the chip's local keys [0, (start + S) / 2)
        through the global table ``tables.context``, the chip's parity from PairSplit.prefill_parity. The program is an
        unsplit later chunk's (``_prefill_sdpa_program``: the Q / K chunk sizes divide S and start, the op walks the
        local keys in tiles; from GLOBAL_SDPA_MIN_START the grouped / K split program, whose 16 heads over half the
        keys are the head split's work: 8 heads over all of them). Returns (output [1, nh, S, 128] bf16, lse [1, nh, S, 1] fp32: NO_KEY_LSE and output 0 for a query
        with no visible local key)."""
        k_cache, v_cache = kv_cache
        attn, lse = ttnn.transformer.chunked_flash_mla_prefill(
            q,
            k_cache,
            self.v_head_dim,
            tables.context,
            chunk_start_idx_tensor=tables.start,
            scale=self.scale,
            program_config=self._prefill_sdpa_program(q.shape[-2], start_pos, later_chunk=True),
            compute_kernel_config=self.sdpa_config,
            input_tensor_v=v_cache,
            return_lse=True,
            **{PAIR_PARITY_KWARG: self.pair.prefill_parity},
        )
        return attn, lse

    def _verify_grouped(self, q, kv_cache, table, cur_pos, used=None):
        """A verify's global SDPA with its LSE, one KV scan per request (paged MLA decode's spec_multi_pos_tiles): the
        pass's SPEC_SLOTS slots are SPEC_SLOTS / GROUP_SLOTS groups, each one request's candidates (the adapter packs
        them so: reference.spec_decode.plan_verify_passes with ``aligned``), candidate j of group g in slot
        g * GROUP_SLOTS + j. q [1, B, 16, 192] (each slot's 16 heads in its own 32-row tile) goes in as
        [1, groups, GROUP_SLOTS * 32, 192], the table ``table`` (a row per slot) as each group's first row, and every
        candidate keeps its SDPA position ``cur_pos`` [SPEC_SLOTS] (-1: no local key or a gap, output 0 and lse
        NO_KEY_LSE). Returns what _paged_mla_decode_lse returns for the slots: output [1, SPEC_SLOTS, 16, 128] in L1,
        lse [1, SPEC_SLOTS, 16, 1] fp32, views of the grouped outputs (the padding rows of each tile are not heads).
        With GROUPED_PACKED_OPS the op computes two candidates per tile (spec_multi_pos_packed, half the compute) on
        the same views. ``used`` (a verify variant, verify_variants): attend only groups 0 .. used - 1, on all the
        cores; the other slots' output is 0 and their lse NO_KEY_LSE, as for an idle group."""
        k_cache, v_cache = kv_cache
        cands = GROUP_SLOTS if used is None else used[0]
        all_slots = SPEC_SLOTS
        groups = all_slots // cands if used is None else used[1]
        slots = groups * cands
        nh, dk, width = q.shape[-2], q.shape[-1], table.shape[-1]
        packed = GROUPED_PACKED_OPS and cands % 2 == 0 and 2 * nh == TILE
        q_dram = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        q_slots = ttnn.slice(q_dram, (0, 0, 0, 0), (1, slots, nh, dk))
        q_dram.deallocate(True)
        tiles = ttnn.Shape([1, slots, TILE, dk])
        q_grouped = ttnn.reshape(ttnn.reshape(q_slots, tiles, tiles), (1, groups, cands * TILE, dk))  # views
        rows = [ttnn.slice(table, (g * cands, 0), (g * cands + 1, width)) for g in range(groups)]
        group_table = ttnn.concat(rows, dim=0) if groups > 1 else rows[0]
        if groups > 1:
            for t in rows:
                t.deallocate(True)
        if slots < all_slots:
            cur_pos = ttnn.slice(cur_pos, (0,), (slots,))
        sdpa = dict(
            head_dim_v=self.v_head_dim,
            scale=self.scale,
            compute_kernel_config=self.decode_sdpa_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            return_lse=True,
            spec_multi_pos_tiles=cands,
            **({"spec_multi_pos_packed": True} if packed else {}),
        )
        mla = ttnn.transformer.paged_flash_multi_latent_attention_decode
        if VERIFY_SEQUENTIAL and groups > 1:
            parts = []
            for g in range(groups):
                q_g = ttnn.slice(q_grouped, (0, g, 0, 0), (1, g + 1, cands * TILE, dk))
                t_g = ttnn.slice(group_table, (g, 0), (g + 1, width))
                p_g = ttnn.slice(cur_pos, (g * cands,), ((g + 1) * cands,))
                program = self._split_sdpa_program(1, width * k_cache.shape[-2])
                parts.append(
                    mla(
                        q_g, k_cache, v_cache, page_table_tensor=t_g, cur_pos_tensor=p_g, program_config=program, **sdpa
                    )
                )
                for t in (q_g, t_g, p_g):
                    t.deallocate(True)
            attn = ttnn.concat([o for o, _ in parts], dim=1, memory_config=parts[0][0].memory_config())
            lse = ttnn.concat([l for _, l in parts], dim=1, memory_config=parts[0][1].memory_config())
            for o, l in parts:
                o.deallocate(True)
                l.deallocate(True)
        else:
            attn, lse = mla(
                q_grouped,
                k_cache,
                v_cache,
                page_table_tensor=group_table,
                cur_pos_tensor=cur_pos,
                program_config=self._split_sdpa_program(groups, width * k_cache.shape[-2]),
                **sdpa,
            )  # [1, groups, cands * 32, 128], [.., 1]
        q_slots.deallocate(True)
        group_table.deallocate(True)
        if slots < all_slots:
            cur_pos.deallocate(True)
        dv = attn.shape[-1]
        out = ttnn.reshape(attn, ttnn.Shape([1, slots, nh, dv]), ttnn.Shape([1, slots, TILE, dv]))
        lse = ttnn.reshape(lse, ttnn.Shape([1, slots, nh, 1]), ttnn.Shape([1, slots, TILE, TILE]))
        if slots < all_slots:
            out, lse = self._pad_idle_slots(out, lse, all_slots - slots)
        return out, lse

    def _pad_idle_slots(self, out, lse, idle):
        """``out`` [1, s, 16, 128] and ``lse`` [1, s, 16, 1] followed by ``idle`` idle slots (output 0, lse
        NO_KEY_LSE), in ``out``'s memory config: what the grouped SDPA leaves for a group with no candidate."""
        zero = ttnn.slice(out, (0, 0, 0, 0), (1, 1, out.shape[2], out.shape[3]))
        zero = ttnn.multiply(zero, 0.0)
        pad = ttnn.repeat(zero, ttnn.Shape([1, idle, 1, 1])) if idle > 1 else zero
        out_all = ttnn.concat([out, pad], dim=1, memory_config=out.memory_config())
        if pad is not zero:
            pad.deallocate(True)
        zero.deallocate(True)
        no_key = ttnn.slice(lse, (0, 0, 0, 0), (1, 1, lse.shape[2], 1))
        no_key = ttnn.add(ttnn.multiply(no_key, 0.0), NO_KEY_LSE)
        pad = ttnn.repeat(no_key, ttnn.Shape([1, idle, 1, 1])) if idle > 1 else no_key
        lse_all = ttnn.concat([lse, pad], dim=1, memory_config=lse.memory_config())
        if pad is not no_key:
            pad.deallocate(True)
        no_key.deallocate(True)
        out.deallocate(True)
        lse.deallocate(True)
        return out_all, lse_all

    def _paged_mla_decode_lse(self, q, kv_cache, page_table, cur_pos):
        """Global decode SDPA with its LSE (the ops' interface in runs/kvsplit/SPEC.md): q (height-sharded, 16 heads)
        over the chip's local keys up to each user's SDPA position ``cur_pos`` (-1: none) through ``page_table``.
        Returns (output [1, B, 16, 128] bf16 in L1, lse [1, B, 16, 1] fp32: NO_KEY_LSE and output 0 for a user with
        no visible local key)."""
        k_cache, v_cache = kv_cache
        attn, lse = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q,
            k_cache,
            v_cache,
            head_dim_v=self.v_head_dim,
            page_table_tensor=page_table,
            cur_pos_tensor=cur_pos,
            scale=self.scale,
            program_config=self._split_sdpa_program(page_table.shape[0], page_table.shape[-1] * k_cache.shape[-2]),
            compute_kernel_config=self.decode_sdpa_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            return_lse=True,
        )
        return attn, lse

    # ---------------------------------------------------------------- decode
    def _cache_width_v(self, v, batch, width):
        """Decode V heads [1, B, 1, D] (height-sharded) -> the cache's ``width`` real dims, height-sharded for
        paged_update_cache (the input is deallocated)."""
        if v.shape[-1] == width:
            return v
        v_inter = ttnn.to_memory_config(v, ttnn.L1_MEMORY_CONFIG)
        v.deallocate(True)
        v_real = ttnn.slice(v_inter, (0, 0, 0, 0), (1, batch, 1, width))
        v_inter.deallocate(True)
        v_shard = ttnn.to_memory_config(v_real, self._height_sharded(batch, TILE, width))
        v_real.deallocate(True)
        return v_shard

    def _height_sharded(self, batch, rows, width):
        grid = ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(8, 8), row_wise=True)
        return ttnn.create_sharded_memory_config(
            shape=(rows, width),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _decode_o(self, attn_cat, compact):
        """o projection and the row reduction (consumes ``attn_cat``). ``compact``: the users are in rows 0-3; the o
        stream writes them straight into the compact rows (``compact_buf``), and this returns them un-gathered when the
        reduction all-gathers over the fabric, else their row all-gather, instead of the all-reduced [1, 1, 32, hidden].
        """
        if compact:
            assert self.o_stream is not None, "compact decode needs the bank-local o stream"
            rows = self.o_stream(attn_cat, self.wo_decode, compact_out=self.compact_buf, rows=self.rows)
            attn_cat.deallocate(True)
            return rows if self.fabric_gather else self.plan.all_gather(rows, dim=3)
        out = ttnn.linear(
            attn_cat,
            self.wo_decode,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.decode_o_program,
            compute_kernel_config=self.mm_config,
        )
        attn_cat.deallocate(True)
        return self.plan.all_reduce(out)

    def forward_decode(self, x, rot_mats, cur_pos, page_table, kv_cache, compact=False):
        """x [1, 1, B, hidden] (B users padded to a tile).

        cur_pos: int32 [B] on device (-1 skips a slot), or a dict {"pos", "pos_owned", "owner_mask"} and optionally
        "pos_update" and "serial"; global layers with split users read the per-chip owned view (see kv.py). Under
        the KV pair split (``self.pair``) the dict has "pair_write" and "pair_sdpa", the global layers' per-chip
        positions (PairSplit.positions), and with "serial" also "pair_serial" (``_forward_decode_pair``).
        "serial": per-slot position tensors, each keeping one slot's position, to write K / V slot by slot: the
        slots of a speculative verify are one request at consecutive positions, whose rows share cache tiles that
        the per-slot cache writers read, patch and write back concurrently (reference/spec_decode.py).
        ``compact``: users only in rows 0-3; returns the gathered compact rows (see ``_decode_o``)."""
        serial = None
        if isinstance(cur_pos, dict):
            serial = cur_pos.get("serial")
            if self.split_users and self.pair is not None:
                return self._forward_decode_pair(x, rot_mats, cur_pos, page_table, kv_cache, compact)
            if self.split_users:
                return self._forward_decode_split(
                    x,
                    rot_mats,
                    cur_pos["pos_owned"],
                    cur_pos["owner_mask"],
                    page_table,
                    kv_cache,
                    compact,
                    pos_update=cur_pos.get("pos_update"),
                    pos_sdpa=cur_pos.get("pos_sdpa"),
                    page_table_sdpa=cur_pos.get("page_table_sdpa"),
                    serial=serial,
                )
            cur_pos = cur_pos["pos"]
        assert not self.split_users, "global layers on TP=8 need the per-chip owned positions"
        writes = serial or [cur_pos]
        xqkv = self._decode_qkv(x, self.wqkv, self.decode_qkv_program, compact)
        k_cache, v_cache = kv_cache
        fused_qkv = compact and self.swa_qkv is not None and len(rot_mats) >= 4
        if fused_qkv:
            # Head split, RoPE and both cache updates in one launch (rot_mats[2:4]: the raw cos / sin rows). The users
            # whose positions share a cache tile row (a verify's slots) are written by one core, so a verify needs no
            # serial writes here.
            q_rot = self.swa_qkv(xqkv, rot_mats[2], rot_mats[3], cur_pos, page_table, k_cache, v_cache)
            xqkv.deallocate(True)
        else:
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                xqkv,
                num_heads=self.nh_local,
                num_kv_heads=self.nkv_local,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            )
            xqkv.deallocate(True)
            trans = self.rope.transformation_mat
            q_rot = ttnn.experimental.rotary_embedding_llama(q, rot_mats[0], rot_mats[1], trans, is_decode_mode=True)
            k_rot = ttnn.experimental.rotary_embedding_llama(k, rot_mats[0], rot_mats[1], trans, is_decode_mode=True)
            q.deallocate(True)
            k.deallocate(True)

            v_cached = self._cache_width_v(v, x.shape[-2], v_cache.shape[-1])
            for pos in writes:
                update = dict(update_idxs_tensor=pos, page_table=page_table)
                ttnn.experimental.paged_update_cache(k_cache, k_rot, **update)
                ttnn.experimental.paged_update_cache(v_cache, v_cached, **update)
            k_rot.deallocate(True)
            v_cached.deallocate(True)

        attn = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_rot,
            k_cache,
            v_cache,
            head_dim_v=v_cache.shape[-1],
            page_table_tensor=page_table,
            cur_pos_tensor=cur_pos,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            attention_sink=self.decode_sinks,
            program_config=self.decode_sdpa_program,
            compute_kernel_config=self.decode_sdpa_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if not fused_qkv:
            q_rot.deallocate(True)
        # Heads concatenated (V's 128 real dims), in L1 for the o matmul.
        attn_cat = self.concat_heads(attn)
        attn.deallocate(True)
        return self._decode_o(attn_cat, compact)

    def _forward_decode_split(
        self,
        x,
        rot_mats,
        pos_owned,
        owner_mask,
        page_table,
        kv_cache,
        compact=False,
        pos_update=None,
        pos_sdpa=None,
        page_table_sdpa=None,
        serial=None,
    ):
        """Global-layer decode: all 16 Q heads of this chip's KV group, for the slots this chip owns.

        ``pos_update``: the positions whose K / V this chip writes (default ``pos_owned``). With vLLM's page tables
        both chips of the pair keep every slot's K / V, so the caller passes every slot's position.
        ``pos_sdpa`` / ``page_table_sdpa``: ``pos_owned`` and the page table cut to the first n decode slots (n < 32).
        The SDPA takes its batch from the page table's rows and splits its cores evenly over that batch, so with the
        32-row table each user gets 2 of 64 cores (21 ms per layer at 1M context); with n = 2 rows it gets 32 (1.15
        ms). ``serial``: the K / V write positions one slot per tensor, written one after another (forward_decode);
        vLLM's tables only."""
        assert serial is None or pos_update is not None, "serial K / V writes need vLLM's tables (pos_update)"
        batch = x.shape[-2]
        xqkv = self._decode_qkv(x, self.wqkv_decode, self.decode_qkv_program)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv,
            num_heads=self.nh_decode,
            num_kv_heads=1,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        xqkv.deallocate(True)
        trans = self.rope.transformation_mat
        q_rot = ttnn.experimental.rotary_embedding_llama(q, rot_mats[0], rot_mats[1], trans, is_decode_mode=True)
        k_rot = ttnn.experimental.rotary_embedding_llama(k, rot_mats[0], rot_mats[1], trans, is_decode_mode=True)
        q.deallocate(True)
        k.deallocate(True)

        v_shard = self._cache_width_v(v, batch, self.v_head_dim)

        k_cache, v_cache = kv_cache
        for pos in serial or [pos_owned if pos_update is None else pos_update]:
            update = dict(update_idxs_tensor=pos, page_table=page_table)
            ttnn.experimental.paged_update_cache(k_cache, k_rot, **update)
            ttnn.experimental.paged_update_cache(v_cache, v_shard, **update)
        k_rot.deallocate(True)
        v_shard.deallocate(True)

        if page_table_sdpa is not None:
            page_table, pos_owned = page_table_sdpa, pos_sdpa
        attn = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_rot,
            k_cache,
            v_cache,
            head_dim_v=self.v_head_dim,
            page_table_tensor=page_table,
            cur_pos_tensor=pos_owned,
            scale=self.scale,
            program_config=self._split_sdpa_program(page_table.shape[0], page_table.shape[-1] * k_cache.shape[-2]),
            compute_kernel_config=self.decode_sdpa_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )  # [1, B or n, 16, 128]
        q_rot.deallocate(True)
        # Heads concatenated in L1; slots owned by the other chip of the pair were skipped here, so the owner mask
        # zeroes their rows before the row all-reduce.
        owned = self.concat_heads(attn, owner_mask)
        attn.deallocate(True)
        return self._decode_o(owned, compact)

    def _forward_decode_pair(self, x, rot_mats, positions, page_table, kv_cache, compact=False):
        """Global-layer decode under the KV pair split: all 16 Q heads of this chip's KV group for every slot, over the
        chip's local keys. ``positions``: forward_decode's dict; "pair_write" the chip's K / V write positions (or
        "pair_serial", one slot per tensor, written one after another as in a verify), "pair_sdpa" its SDPA positions,
        and with "page_table_sdpa" the first n slots' table rows and "pos_sdpa" their SDPA positions (see
        ``_forward_decode_split``). The SDPA's partial output is scaled by the chip's merge weight
        (PairSplit.merge); the row reduction after the o projection adds the pair's two."""
        batch = x.shape[-2]
        xqkv = self._decode_qkv(x, self.wqkv_decode, self.decode_qkv_program)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv,
            num_heads=self.nh_decode,
            num_kv_heads=1,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        xqkv.deallocate(True)
        trans = self.rope.transformation_mat
        q_rot = ttnn.experimental.rotary_embedding_llama(q, rot_mats[0], rot_mats[1], trans, is_decode_mode=True)
        k_rot = ttnn.experimental.rotary_embedding_llama(k, rot_mats[0], rot_mats[1], trans, is_decode_mode=True)
        q.deallocate(True)
        k.deallocate(True)
        v_shard = self._cache_width_v(v, batch, self.v_head_dim)

        k_cache, v_cache = kv_cache
        for pos in positions.get("pair_serial") or [positions["pair_write"]]:
            update = dict(update_idxs_tensor=pos, page_table=page_table)
            ttnn.experimental.paged_update_cache(k_cache, k_rot, **update)
            ttnn.experimental.paged_update_cache(v_cache, v_shard, **update)
        k_rot.deallocate(True)
        v_shard.deallocate(True)

        table, pos = page_table, positions["pair_sdpa"]
        if positions.get("page_table_sdpa") is not None:
            table, pos = positions["page_table_sdpa"], positions["pos_sdpa"]
        # A verify over exactly its pass's slots (the compact rows): one KV scan per request.
        grouped = positions.get("pair_serial") and verify_grouped() and pos is not positions["pair_sdpa"]
        if grouped and pos.shape[0] == SPEC_SLOTS:
            # [1, SPEC_SLOTS, 16, 128], [.., 16, 1]
            attn, lse = self._verify_grouped(q_rot, kv_cache, table, pos, positions.get("verify_groups"))
        else:
            attn, lse = self._paged_mla_decode_lse(q_rot, kv_cache, table, pos)  # [1, B or n, 16, 128], [.., 16, 1]
        q_rot.deallocate(True)
        attn_cat = self.pair.merge_heads(attn, lse, self.concat_heads)
        attn.deallocate(True)
        lse.deallocate(True)
        return self._decode_o(attn_cat, compact)
