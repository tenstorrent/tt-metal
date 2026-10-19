# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo attention (SWA and global layers), tensor-parallel over one Galaxy row.

SWA layers (8 KV heads): each chip has 8 Q heads and one KV head, in prefill and decode. V has
128 dims; it is zero-padded to 192 (and so are the matching ``o_proj`` rows) so the SDPA kernels see
one head dim. The SWA cache keeps only the in-window blocks per user (kv.SwaBlockTable), so the padding costs
~nothing. Decode drops the
padding columns after SDPA and uses an unpadded copy of ``o_proj`` (the dropped rows only ever multiplied
zeros).

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
``attention_value_scale`` is folded into the V weights. Sinks (SWA only) are pre-divided by the
softmax scale because the SDPA kernels apply ``scale`` to the sink logit too.
"""

import torch

import ttnn

from .proj_stream import BankStreamMatmul
from .common import TILE, DramShardedMatmul, as_weight, cache_name, compute_config, decode_linear_config, pad_to
from .kv import kv_replication, per_chip, slot_owner_masks
from .moe_stream import KERNEL_DIR, zero_output
from .rope import permute_heads_for_rope


class SwaDecodeQkv:
    """SWA decode, users in rows 0-3: the fused qkv row [1, 1, 32, (nh + 2) * D] (L1 tiles) -> the RoPE'd query in
    the height-sharded layout paged SDPA decode reads ([1, 32, nh, D], one [32, D] shard per user) and the RoPE'd key
    and the value written into the users' positions of the BFP8 paged caches, in one launch in place of
    nlp_create_qkv_heads_decode, two rotary_embedding_llama and two paged_update_cache.

    One core per query head (it reads rows 0-3 of its head's D / 32 tiles: every user at once, so RoPE is
    element-wise against rows 0-3 of cos / sin) and one key and one value core per user (read-modify-write of the
    user's cache tile row: BFP8 -> bf16, the new row patched in, -> BFP8)."""

    CB_X, CB_COS, CB_SIN, CB_TRANS, CB_TMP, CB_ROT, CB_CACHE, CB_C16, CB_SIG, CB_OUT8, CB_META = range(11)
    ROLE_Q, ROLE_K, ROLE_V = 0, 1, 2

    def __init__(self, plan, num_heads, head_dim, users, block_size, trans_mat, sample_qkv):
        """``trans_mat``: the decode RoPE transformation tile (torch [32, 32]); ``sample_qkv``: a fused qkv tensor of
        the decode shape, used once to take the query layout nlp_create_qkv_heads_decode produces."""
        device = plan.mesh_device
        self.device = device
        self.nh, self.d, self.users, self.block_size = num_heads, head_dim, users, block_size
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
        assert num_heads <= grid.x and 2 * users <= grid.x
        self.q_cores = [ttnn.CoreCoord(h, row_q) for h in range(num_heads)]
        self.k_cores = [ttnn.CoreCoord(u, row_kv) for u in range(users)]
        self.v_cores = [ttnn.CoreCoord(users + u, row_kv) for u in range(users)]
        self.all_ranges = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(self.q_cores[0], self.q_cores[-1]),
                ttnn.CoreRange(self.k_cores[0], self.v_cores[-1]),
            ]
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
        assert pt_page % 64 == 0, "the reader fetches the 64 B of the page table row that holds the entry"
        n = self.d_tiles
        kernels = []
        roles = (
            (self.ROLE_Q, self.q_cores, lambda i: i * n, None),
            (self.ROLE_K, self.k_cores, lambda i: self.nh * n, k_cache),
            (self.ROLE_V, self.v_cores, lambda i: (self.nh + 1) * n, v_cache),
        )
        user_noc = []
        for c in self.user_noc:
            user_noc += [c.x, c.y]
        for role, cores, first, cache in roles:
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
                n,
                self.block_size,
                pos_page,
                pt_page,
                cache_tile,
            ]
            writer_ct = [
                role,
                self.CB_X,
                self.CB_ROT,
                self.CB_META,
                self.CB_C16,
                self.CB_SIG,
                self.CB_OUT8,
                n,
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
                n,
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
        assert len(shape) == 4 and shape[0] == 1 and shape[1] == TILE and shape[2] >= self.num_heads
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
        # into a persistent compact [1, 1, 32, hidden / 8] tensor, so the row reduction moves 1/8 of the tile.
        self.compact_buf = zero_output(plan, hidden // (TILE // 4))
        # Decode: SDPA output -> concatenated heads (V's real dims) in one launch.
        self.concat_heads = ConcatHeads(plan, self.nh_decode if self.split_users else self.nh_local, self.v_head_dim)

        # Chunked prefill of a global layer: 1 on the chips that own a slot of each parity (one user at a time).
        self.prefill_owner_masks = None
        if self.split_users:
            self.prefill_owner_masks = {
                p: per_chip(plan, m, ttnn.bfloat16, ttnn.TILE_LAYOUT) for p, m in slot_owner_masks(tp, self.rep).items()
            }

        self.sdpa_config = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=False, packer_l1_acc=False)
        self.decode_sdpa_program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=0,
            k_chunk_size=128,
            exp_approx_mode=False,
        )

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
        """[K, V] caches ``[num_blocks, kv_heads, block_size, dim]`` per chip; V is 128 wide when users are split."""
        v_dim = self.v_head_dim if self.split_users else self.head_dim
        caches = []
        for dim in (self.head_dim, v_dim):
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

    def _decode_qkv(self, x, weight, program_config, compact=False):
        # Keep the fused qkv row in L1: the decode head split reads it with many small per-user row reads. The bank
        # stream computes rows 0-3 only, so it serves the compact (users in rows 0-3) decode.
        if compact and weight is self.wqkv and self.qkv_stream is not None:
            return self.qkv_stream(x, self.wqkv_stream)
        return ttnn.linear(
            x,
            weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=program_config,
            compute_kernel_config=self.mm_config,
        )

    def _prefill_sdpa_program(self, seq_len, start_pos=0):
        """Q / K chunk sizes that divide ``seq_len`` (and ``start_pos``: chunked SDPA needs that)."""
        chunk = 256 if seq_len >= 2048 else (128 if seq_len >= 128 else 32)
        while seq_len % chunk or start_pos % chunk:
            chunk //= 2
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
        for t in (q, k, cos, sin):
            t.deallocate(True)
        return q_rot, k_rot

    def _fill(self, kv_cache, k_rot, v, page_table, user_id, start_pos):
        """Write the chunk's K and V (only its first ``v_head_dim`` columns when the V cache is that narrow) into
        the user's cache blocks for positions start_pos .. start_pos + S - 1."""
        k_cache, v_cache = kv_cache
        seq_len = k_rot.shape[-2]
        block_size = k_cache.shape[-2]
        if start_pos == 0:
            table, batch_idx = page_table, user_id
        else:
            first, end = start_pos // block_size, -(-(start_pos + seq_len) // block_size)
            assert end <= page_table.shape[-1], "chunk runs past the page table"
            table, batch_idx = ttnn.slice(page_table, (user_id, first), (user_id + 1, end)), 0
        k_fill = ttnn.typecast(k_rot, k_cache.dtype)
        if v_cache.shape[-1] < v.shape[-1]:
            v_real = ttnn.slice(v, (0, 0, 0, 0), (1, v.shape[1], seq_len, v_cache.shape[-1]))
            v_fill = ttnn.typecast(v_real, v_cache.dtype)
            v_real.deallocate(True)
        else:
            v_fill = ttnn.typecast(v, v_cache.dtype)
        ttnn.experimental.paged_fill_cache(k_cache, k_fill, table, batch_idx=batch_idx)
        ttnn.experimental.paged_fill_cache(v_cache, v_fill, table, batch_idx=batch_idx)
        k_fill.deallocate(True)
        v_fill.deallocate(True)
        if table is not page_table:
            table.deallocate(True)

    def _cached_window(self, cache, blocks):
        """The cache blocks ``blocks`` (physical ids, in position order) as one bf16 [1, kv_heads, n * block, dim]."""
        parts = []
        for b in blocks:
            part = ttnn.slice(cache, (b, 0, 0, 0), (b + 1, *tuple(cache.shape)[1:]))
            parts.append(ttnn.typecast(part, ttnn.bfloat16))
            part.deallocate(True)
        out = ttnn.concat(parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG) if len(parts) > 1 else parts[0]
        if len(parts) > 1:
            for t in parts:
                t.deallocate(True)
        return out

    def forward_prefill(self, x, user_id, page_table, kv_cache, start_pos=0, real_len=None, window_blocks=None):
        """x [1, 1, S, hidden] (S a multiple of 32): positions start_pos .. start_pos + S - 1 of the user in slot
        ``user_id`` (``real_len`` real tokens, then padding whose K/V land past them, where decode overwrites them).
        ``start_pos`` is a multiple of the block size; the cache must hold the user's earlier positions.
        ``window_blocks`` (SWA layers, start_pos > 0): physical cache blocks of the positions from
        kv.SwaBlockTable.window_start(start_pos) up to start_pos, which the chunk attends to."""
        if start_pos > 0 and self.split_users:
            return self._prefill_owner_chunk(x, user_id, page_table, kv_cache, start_pos)
        seq_len = x.shape[-2]
        xqkv = self._prefill_qkv(x)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.nh_local,
            num_kv_heads=self.nkv_local,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        xqkv.deallocate(True)
        q_rot, k_rot = self._rope_prefill(q, k, start_pos, seq_len)

        lead = 0
        if start_pos > 0:
            # SWA: the window before the chunk from the cache, read before the fill can reuse any of its blocks.
            assert self.sliding_window is not None and window_blocks, "a later SWA chunk needs its window blocks"
            k_cache, v_cache = kv_cache
            lead = len(window_blocks) * k_cache.shape[-2]
            k_win = self._cached_window(k_cache, window_blocks)
            v_win = self._cached_window(v_cache, window_blocks)
        self._fill(kv_cache, k_rot, v, page_table, user_id, start_pos)
        if lead:
            q_lead = ttnn.zeros(
                (1, self.nh_local, lead, self.head_dim),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.plan.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            joined = []
            for head, tail in ((q_lead, q_rot), (k_win, k_rot), (v_win, v)):
                joined.append(ttnn.concat([head, tail], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG))
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
        attn_cat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn.deallocate(True)
        out = ttnn.linear(
            attn_cat,
            self.wo,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_config,
        )
        attn_cat.deallocate(True)
        return self.plan.all_reduce(out)

    def _prefill_owner_chunk(self, x, user_id, page_table, kv_cache, start_pos):
        """A global layer's chunk after the first: the group's 16 Q heads over the whole cached context on the
        chip that owns the user (the other chip of the pair runs the same program on its scratch block and its
        result is zeroed)."""
        assert self.sinks is None and self.sliding_window is None
        seq_len = x.shape[-2]
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
        q_rot, k_rot = self._rope_prefill(q, k, start_pos, seq_len)
        self._fill(kv_cache, k_rot, v, page_table, user_id, start_pos)
        k_rot.deallocate(True)
        v.deallocate(True)

        k_cache, v_cache = kv_cache
        end = -(-(start_pos + seq_len) // k_cache.shape[-2])
        user_table = ttnn.slice(page_table, (user_id, 0), (user_id + 1, end))
        attn = ttnn.transformer.chunked_flash_mla_prefill(
            q_rot,
            k_cache,
            self.v_head_dim,
            user_table,
            start_pos,
            scale=self.scale,
            program_config=self._prefill_sdpa_program(seq_len, start_pos),
            compute_kernel_config=self.sdpa_config,
            input_tensor_v=v_cache,
        )  # [1, 16, S, 128]
        q_rot.deallocate(True)
        user_table.deallocate(True)
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
        return self.plan.all_reduce(out)

    # ---------------------------------------------------------------- decode
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
            rows = self.o_stream(attn_cat, self.wo_decode, compact_out=self.compact_buf)
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

        cur_pos: int32 [B] on device (-1 skips a slot), or a dict {"pos", "pos_owned", "owner_mask"}; global
        layers with split users read the per-chip owned view (see kv.py).
        ``compact``: users only in rows 0-3; returns the gathered compact rows (see ``_decode_o``)."""
        if isinstance(cur_pos, dict):
            if self.split_users:
                return self._forward_decode_split(
                    x, rot_mats, cur_pos["pos_owned"], cur_pos["owner_mask"], page_table, kv_cache, compact
                )
            cur_pos = cur_pos["pos"]
        assert not self.split_users, "global layers on TP=8 need the per-chip owned positions"
        xqkv = self._decode_qkv(x, self.wqkv, self.decode_qkv_program, compact)
        k_cache, v_cache = kv_cache
        fused_qkv = compact and self.swa_qkv is not None and len(rot_mats) >= 4
        if fused_qkv:
            # Head split, RoPE and both cache updates in one launch (rot_mats[2:4]: the raw cos / sin rows).
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

            update = dict(update_idxs_tensor=cur_pos, page_table=page_table)
            ttnn.experimental.paged_update_cache(k_cache, k_rot, **update)
            ttnn.experimental.paged_update_cache(v_cache, v, **update)
            k_rot.deallocate(True)
            v.deallocate(True)

        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_rot,
            k_cache,
            v_cache,
            cur_pos_tensor=cur_pos,
            page_table_tensor=page_table,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            attention_sink=self.decode_sinks,
            program_config=self.decode_sdpa_program,
            compute_kernel_config=self.sdpa_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if not fused_qkv:
            q_rot.deallocate(True)
        # Heads concatenated with only V's 128 real dims (the padding columns are zeros), in L1 for the o matmul.
        attn_cat = self.concat_heads(attn)
        attn.deallocate(True)
        return self._decode_o(attn_cat, compact)

    def _forward_decode_split(self, x, rot_mats, pos_owned, owner_mask, page_table, kv_cache, compact=False):
        """Global-layer decode: all 16 Q heads of this chip's KV group, for the slots this chip owns."""
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

        # The cache keeps only V's 128 real dims.
        v_inter = ttnn.to_memory_config(v, ttnn.L1_MEMORY_CONFIG)
        v.deallocate(True)
        v_real = ttnn.slice(v_inter, (0, 0, 0, 0), (1, batch, 1, self.v_head_dim))
        v_inter.deallocate(True)
        v_shard = ttnn.to_memory_config(v_real, self._height_sharded(batch, TILE, self.v_head_dim))
        v_real.deallocate(True)

        k_cache, v_cache = kv_cache
        update = dict(update_idxs_tensor=pos_owned, page_table=page_table)
        ttnn.experimental.paged_update_cache(k_cache, k_rot, **update)
        ttnn.experimental.paged_update_cache(v_cache, v_shard, **update)
        k_rot.deallocate(True)
        v_shard.deallocate(True)

        attn = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_rot,
            k_cache,
            v_cache,
            head_dim_v=self.v_head_dim,
            page_table_tensor=page_table,
            cur_pos_tensor=pos_owned,
            scale=self.scale,
            program_config=self.decode_sdpa_program,
            compute_kernel_config=self.sdpa_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )  # [1, B, 16, 128]
        q_rot.deallocate(True)
        # Heads concatenated in L1; slots owned by the other chip of the pair were skipped here, so the owner mask
        # zeroes their rows before the row all-reduce.
        owned = self.concat_heads(attn, owner_mask)
        attn.deallocate(True)
        return self._decode_o(owned, compact)
