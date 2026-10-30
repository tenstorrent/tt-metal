# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``SdpaDecodeSplitfree``: paged decode attention whose output for a row is bitwise identical whatever the padded
batch or the cores the row gets (tt-metal#59300), while a lone row still spreads over the whole grid.

Two ``ttnn.generic_op`` programs per call (the trace captures both), sharing a persistent fp32 scratch the model
allocates once (``allocate_scratch``) and passes in:

1. partials (``kernels/*_partials.cpp``): every (row, KV group, unit) of fixed ``unit`` tokens is assigned to a
   core by the planner's static mapping (``planner.py``: unit-major round robin over the grid, independent of
   ``cur_pos``). A core walks its units in a fixed order; per live unit it runs the stock flash-decode recurrence
   (``sdpa_flash_decode.cpp``'s chunk body on ``compute_common.hpp``) over the unit's fixed ``chunk``-token
   sub-chunks in ascending order, fp32 statistics and accumulator, masking the chunk that holds ``cur_pos`` (and
   the window start on a sliding layer) exactly as the stock kernels mask it, and writes the unit's (m, l, O) as
   fp32 tiles to its scratch slot.
2. combine (``kernels/*_combine.cpp``): one core per (row, group) merges the live units' partials in ascending
   unit order with the stock correction step, normalises once and writes the group's head rows in bf16.

Because every unit partial is computed by identical code on identical data in one fixed order, and the merge
order is fixed by the unit indices, the result cannot depend on which core computed what or on the padded batch.
The chunk is the largest of ``planner.CHUNK_LADDER`` that fits the arch's L1 (head dim and cache format decide
it, never the batch or the grid); the unit is derived from the grid and the context (``planner.derive_unit``: the
smallest chunk multiple whose count at the page table's reach is at most the grid's cores, at most 4096 -- 2048
on a 13x10 Blackhole grid, 4096 on the Wormhole 8x8; one chunk on a sliding layer), so a lone row at the full
context puts one unit on (nearly) every core. Grid, chunk and context are build constants of a chip: every
decode graph of one build shares them, and a row's bits are the same in every graph; a different grid (another
harvesting, another arch) is a different unit and different bits unless ``GEMMA4_SDPA_SPLITFREE_UNIT`` pins it.

Contract (the stock ``paged_scaled_dot_product_attention_decode`` call of ``tt/attention/decode.py``):
``q [1, B <= 32, Hq <= 32, D]`` bf16 TILE interleaved (DRAM or L1), ``k / v [blocks, nkv, block, D]`` bf16 or
bfloat8_b TILE (the cache view may be overridden by ``num_kv_heads`` / ``block_size`` as the stock geometry
override does), ``page_table [B, max_blocks]`` int32 ROW_MAJOR, ``cur_pos [B]`` int32 ROW_MAJOR (-1 = padding row,
its output rows untouched), ``scale``, ``sliding_window`` (None: attend everything), ``cache_position_modulo``
(the bounded ring of a sliding layer). Output ``[1, B, Hq, D]`` bf16 TILE in ``memory_config`` (default DRAM).
Compute: fp32 dest accumulation, exact exp, the scores and the reduce scaler in fp32 CBs and HiFi4 (default;
``GEMMA4_SDPA_SPLITFREE_FIDELITY`` overrides). The stock decode op keeps the scores in bf16 and runs HiFi2: at
head_dim 512 and scale 1 a row's scores sit at |s| ~ 100 where a bf16 ulp is 0.5, and the row max / sum reduces
read the scores through the source registers at the fidelity's mantissa width, so the 0.5-level truncation of
the max (then exp's clamp of s - m > 0) costs the stock op pcc 0.9984 (4K) / 0.9991 (256K) against fp32 torch;
fp32 scores and scaler at HiFi2 still truncate the max at the bf16 ulp (0.9985 at 256K), HiFi4 keeps the tf32
width (0.99999). HiFi4 costs the lone-row 256K op 8 % (1.20 -> 1.30 ms on P150, stock cap 16: 1.39 ms).
"""

import os
import struct
from pathlib import Path

import torch

import ttnn
from models.demos.gemma4.custom_ops.grid import compute_grid, core_range_set, row_major_cores
from models.demos.gemma4.custom_ops.sdpa_decode_splitfree import planner as P
from models.demos.gemma4.tt.compute_config import sdpa_fp32_dest_acc_en, sdpa_math_fidelity

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32
BF16_TILE = 2048
F32_TILE = 4096
DST_TILES = 4  # fp32 dest accumulation: four tiles per dest window (the stock dst_size)
KV_DTYPES = (ttnn.bfloat16, ttnn.bfloat8_b)
FIDELITY_ENV = "GEMMA4_SDPA_SPLITFREE_FIDELITY"  # HiFi4 (default) / HiFi3 / HiFi2 / LoFi: matmul + reduce fidelity
DEFAULT_FIDELITY = "HiFi4"

# CB indices: kernels/splitfree_common.hpp
CB_Q, CB_K, CB_V, CB_MASK, CB_WMASK, CB_SCALE, CB_ZERO, CB_POS_C, CB_POS_D, CB_PT = range(10)
CB_OUT_O, CB_OUT_M, CB_OUT_L = 16, 17, 18
CB_QK, CB_OUT_IM, CB_OUT_ACC, CB_CUR_MAX, CB_PREV_MAX, CB_CUR_SUM, CB_PREV_SUM, CB_EMD = 24, 25, 26, 27, 28, 29, 30, 31
CB_IN_M, CB_IN_L, CB_IN_O = 0, 1, 2
CB_OUT_FINAL, CB_ACC, CB_ACC_2, CB_PREV_SUM_2, CB_EMD_2 = 16, 20, 21, 22, 23


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _pow2(n):
    return n > 0 and (n & (n - 1)) == 0


class SdpaDecodeSplitfree:
    @staticmethod
    def golden(
        q,
        k_cache,
        v_cache,
        page_table,
        cur_pos,
        *,
        scale=1.0,
        num_kv_heads=None,
        block_size=None,
        sliding_window=None,
        cache_position_modulo=None,
    ):
        """torch reference (fp32): ``q [1, B, Hq, D]``, the paged caches ``[blocks, nkv, block, D]`` (the view of
        ``num_kv_heads`` / ``block_size`` when given), ``page_table [B, max_blocks]``, ``cur_pos [B]`` ints (-1:
        the row's output is zero). Returns ``[1, B, Hq, D]`` fp32."""
        q = q.float()
        _, B, Hq, D = q.shape
        nkv = int(num_kv_heads or k_cache.shape[1])
        block = int(block_size or k_cache.shape[2])
        hpg = Hq // nkv
        k_flat = k_cache.float().reshape(-1, nkv, block, D) if num_kv_heads else k_cache.float()
        v_flat = v_cache.float().reshape(-1, nkv, block, D) if num_kv_heads else v_cache.float()
        out = torch.zeros(1, B, Hq, D, dtype=torch.float32)
        for b in range(B):
            pos = int(cur_pos[b])
            if pos < 0:
                continue
            start = max(0, pos + 1 - int(sliding_window)) if sliding_window else 0
            positions = torch.arange(start, pos + 1)
            virt = positions % int(cache_position_modulo) if cache_position_modulo else positions
            blocks = page_table[b, virt // block].long()
            offs = (virt % block).long()
            for g in range(nkv):
                kk = k_flat[blocks, g, offs]  # [n, D]
                vv = v_flat[blocks, g, offs]
                heads = q[0, b, g * hpg : (g + 1) * hpg]  # [hpg, D]
                s = torch.softmax(heads @ kk.T * float(scale), dim=-1)
                out[0, b, g * hpg : (g + 1) * hpg] = s @ vv
        return out

    # ── planning and scratch ──────────────────────────────────────────────────────────────────────────────────
    @classmethod
    def plan(
        cls,
        device,
        q,
        k_cache,
        cur_pos,
        page_table,
        *,
        num_kv_heads=None,
        block_size=None,
        sliding_window=None,
        env=None,
    ):
        _, B, Hq, D = (int(v) for v in q.shape)
        G = int(num_kv_heads or k_cache.shape[1])
        block = int(block_size or k_cache.shape[2])
        if Hq % G:
            raise ValueError(f"{Hq} local Q heads do not split over {G} KV groups")
        max_context = int(page_table.shape[-1]) * block
        return P.plan_splitfree(
            compute_grid(device),
            B,
            G,
            Hq // G,
            D,
            max_context,
            window=sliding_window,
            kv_dtype=k_cache.dtype,
            arch=getattr(device, "arch", lambda: None)(),
            env=env,
            pos_stick=int(cur_pos.buffer_aligned_page_size()),
            pt_stick=int(page_table.buffer_aligned_page_size()),
        )

    @classmethod
    def scratch_tiles(cls, device, batch, layer_specs, max_context, *, kv_dtype=None, env=None):
        """The scratch a model needs (fp32 tiles): ``layer_specs`` of ``(groups, heads_per_group, head_dim,
        window)`` over its layer types, ``batch`` padded rows, ``max_context`` tokens (``planner.scratch_tiles_for``).
        """
        return P.scratch_tiles_for(
            compute_grid(device),
            batch,
            layer_specs,
            max_context,
            kv_dtype=kv_dtype,
            arch=getattr(device, "arch", lambda: None)(),
            env=env,
        )

    @staticmethod
    def allocate_scratch(device, tiles):
        """The persistent fp32 TILE scratch of ``tiles`` tiles (``[tiles * 32, 32]``, DRAM): allocate ONCE, before
        any trace capture, and pass it to every call."""
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([int(tiles) * TILE, TILE]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )

    @staticmethod
    def scratch_capacity(scratch):
        return (int(scratch.padded_shape[0]) // TILE) * (int(scratch.padded_shape[1]) // TILE)

    @classmethod
    def unsupported_reason(
        cls,
        q,
        k_cache,
        v_cache,
        cur_pos,
        page_table,
        scratch,
        *,
        num_kv_heads=None,
        block_size=None,
        sliding_window=None,
        env=None,
    ):
        """None when the call is in contract (shapes, formats, the L1 fit of the plan, the scratch capacity),
        else why (the caller keeps the stock op)."""
        if q.layout != ttnn.TILE_LAYOUT or q.dtype != ttnn.bfloat16 or q.is_sharded():
            return "q must be a bf16 TILE interleaved tensor"
        if len(q.shape) != 4 or int(q.shape[0]) != 1 or int(q.shape[1]) > P.MAX_ROWS or int(q.shape[2]) > P.MAX_ROWS:
            return f"q must be [1, B <= 32, Hq <= 32, D], got {list(q.shape)}"
        if int(q.shape[3]) % TILE:
            return f"head_dim {int(q.shape[3])} is not a multiple of {TILE}"
        for name, t in (("k", k_cache), ("v", v_cache)):
            if t.layout != ttnn.TILE_LAYOUT or t.dtype not in KV_DTYPES or t.is_sharded():
                return f"{name} cache must be a bf16 or bfloat8_b TILE interleaved tensor"
        if k_cache.dtype != v_cache.dtype:
            return "k and v caches must share a dtype"
        block = int(block_size or k_cache.shape[2])
        if block % TILE:
            return f"block_size {block} is not a multiple of {TILE} (no sub-tile block padding here)"
        G = int(num_kv_heads or k_cache.shape[1])
        if int(q.shape[2]) % G:
            return f"{int(q.shape[2])} Q heads do not split over {G} KV groups"
        if page_table.layout != ttnn.ROW_MAJOR_LAYOUT or page_table.dtype != ttnn.int32 or len(page_table.shape) != 2:
            return "page_table must be a 2-D int32 ROW_MAJOR tensor"
        if int(page_table.shape[0]) != int(q.shape[1]):
            return f"page_table rows {int(page_table.shape[0])} != batch {int(q.shape[1])}"
        if (
            cur_pos.layout != ttnn.ROW_MAJOR_LAYOUT
            or cur_pos.dtype not in (ttnn.int32, ttnn.uint32)
            or int(cur_pos.shape[-1]) < int(q.shape[1])
        ):
            return "cur_pos must be an int32 ROW_MAJOR [B] tensor"
        if scratch.dtype != ttnn.float32 or scratch.layout != ttnn.TILE_LAYOUT or scratch.is_sharded():
            return "scratch must be an fp32 TILE interleaved tensor (allocate_scratch)"
        try:
            plan = cls.plan(
                q.device(),
                q,
                k_cache,
                cur_pos,
                page_table,
                num_kv_heads=num_kv_heads,
                block_size=block_size,
                sliding_window=sliding_window,
                env=env,
            )
        except ValueError as e:
            return str(e)
        if not plan.fits:
            return f"no chunk fits the per-core L1: {plan.describe()}"
        cap = cls.scratch_capacity(scratch)
        if plan.scratch_tiles > cap:
            return f"the scratch holds {cap} fp32 tiles, the call needs {plan.scratch_tiles}: {plan.describe()}"
        return None

    # ── the op ────────────────────────────────────────────────────────────────────────────────────────────────
    @classmethod
    def op(
        cls,
        q,
        k_cache,
        v_cache,
        cur_pos,
        page_table,
        scratch,
        *,
        scale=1.0,
        num_kv_heads=None,
        block_size=None,
        sliding_window=None,
        cache_position_modulo=None,
        memory_config=None,
        env=None,
    ):
        reason = cls.unsupported_reason(
            q,
            k_cache,
            v_cache,
            cur_pos,
            page_table,
            scratch,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            sliding_window=sliding_window,
            env=env,
        )
        assert reason is None, f"sdpa_decode_splitfree: {reason}"
        device = q.device()
        plan = cls.plan(
            device,
            q,
            k_cache,
            cur_pos,
            page_table,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            sliding_window=sliding_window,
            env=env,
        )
        _, B, Hq, D = (int(v) for v in q.shape)
        G, HPG, DHt, SK = plan.groups, plan.heads_per_group, plan.dht, plan.sk
        block = int(block_size or k_cache.shape[2])
        block_t = block // TILE
        cap_t = int(cache_position_modulo) // TILE if cache_position_modulo else 0
        assert (
            cap_t % block_t == 0
        ), f"cache_position_modulo {cache_position_modulo} is not a multiple of the block {block}"
        qk_sub_w = min(SK, DST_TILES)
        out_sub_w = min(DHt, DST_TILES)
        assert SK % qk_sub_w == 0 and DHt % out_sub_w == 0, (SK, DHt)
        dht_gran = min(DHt, DST_TILES) if _pow2(min(DHt, DST_TILES)) else 1
        gran = min(SK, DST_TILES)
        assert _pow2(gran), f"the chunk in tiles {SK} must make a power-of-2 granularity"
        pos_stick = int(cur_pos.buffer_aligned_page_size())
        pt_stick = int(page_table.buffer_aligned_page_size())
        prefix = [
            B,
            G,
            HPG,
            DHt,
            SK,
            plan.nsub,
            plan.max_units,
            plan.n_slots,
            plan.modulus,
            plan.n_cores,
            plan.window,
            cap_t,
            block_t,
            G,
            pos_stick,
            pt_stick,
            plan.reads_in_flight,
            _f32_bits(scale),
            SK // qk_sub_w,
            qk_sub_w,
            DHt // out_sub_w,
            out_sub_w,
            plan.pair_space,
            0,
        ]
        assert len(prefix) == 24
        defines = [
            ("EXP_APPROX_MODE", "0"),
            ("SUB_EXP_GRANULARITY", str(gran)),
            ("STATS_GRANULARITY", str(gran)),
            ("DHT_GRANULARITY", str(dht_gran)),
        ]
        cores = row_major_cores(plan.grid, plan.n_cores)
        core_set = core_range_set(cores)
        kv_fmt = k_cache.dtype
        kv_tile = plan.kv_tile_bytes
        qk_fmt = ttnn.float32 if plan.qk_tile_bytes == F32_TILE else ttnn.bfloat16
        env_fidelity = str((os.environ if env is None else env).get(FIDELITY_ENV, "") or "").strip()
        fidelity = getattr(ttnn.MathFidelity, env_fidelity or DEFAULT_FIDELITY)

        def cb(idx, pages, page, fmt):
            return ttnn.CBDescriptor(
                total_size=pages * page,
                core_ranges=core_set,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)],
            )

        def rt_args(values):
            args = ttnn.RuntimeArgs()
            for i, core in enumerate(cores):
                args[core.x][core.y] = [i] + [int(v) for v in values]
            return args

        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=sdpa_math_fidelity(fidelity),
            fp32_dest_acc_en=sdpa_fp32_dest_acc_en(True),
            math_approx_mode=False,
            dst_full_sync_en=False,
        )

        def kernel(name, ct, rt, config, with_defines=False):
            return ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / name),
                core_ranges=core_set,
                compile_time_args=ct,
                runtime_args=rt,
                config=config,
                **({"defines": defines} if with_defines else {}),
            )

        # ── phase 1: the unit partials into the scratch ──
        cbs1 = [
            cb(CB_Q, DHt, BF16_TILE, ttnn.bfloat16),
            cb(CB_K, 2 * SK * DHt, kv_tile, kv_fmt),
            cb(CB_V, 2 * SK * DHt, kv_tile, kv_fmt),
            cb(CB_MASK, SK, BF16_TILE, ttnn.bfloat16),
            cb(CB_WMASK, SK, BF16_TILE, ttnn.bfloat16),
            cb(CB_SCALE, 1, plan.qk_tile_bytes, qk_fmt),  # the reduces' second operand: the scores' format
            cb(CB_ZERO, 1, BF16_TILE, ttnn.bfloat16),
            cb(CB_POS_C, 1, pos_stick, ttnn.int32),
            cb(CB_POS_D, 1, pos_stick, ttnn.int32),
            cb(CB_PT, 1, pt_stick, ttnn.int32),
            cb(CB_OUT_O, DHt, F32_TILE, ttnn.float32),
            cb(CB_OUT_M, 1, F32_TILE, ttnn.float32),
            cb(CB_OUT_L, 1, F32_TILE, ttnn.float32),
            cb(CB_QK, SK, plan.qk_tile_bytes, qk_fmt),
            cb(CB_OUT_IM, DHt, F32_TILE, ttnn.float32),
            cb(CB_OUT_ACC, DHt, F32_TILE, ttnn.float32),
        ] + [cb(i, 1, F32_TILE, ttnn.float32) for i in (CB_CUR_MAX, CB_PREV_MAX, CB_CUR_SUM, CB_PREV_SUM, CB_EMD)]
        reader1_ct = list(prefix)
        for t in (q, k_cache, v_cache, cur_pos, page_table):
            reader1_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer1_ct = list(prefix)
        for t in (cur_pos, scratch):
            writer1_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        kernels1 = [
            kernel(
                "reader_partials.cpp",
                reader1_ct,
                rt_args(
                    [
                        q.buffer_address(),
                        k_cache.buffer_address(),
                        v_cache.buffer_address(),
                        cur_pos.buffer_address(),
                        page_table.buffer_address(),
                    ]
                ),
                ttnn.ReaderConfigDescriptor(),
            ),
            kernel(
                "writer_partials.cpp",
                writer1_ct,
                rt_args([cur_pos.buffer_address(), scratch.buffer_address()]),
                ttnn.WriterConfigDescriptor(),
            ),
            kernel("compute_partials.cpp", list(prefix), rt_args([]), compute_cfg, with_defines=True),
        ]
        ttnn.generic_op(
            [q, k_cache, v_cache, cur_pos, page_table, scratch],
            ttnn.ProgramDescriptor(kernels=kernels1, semaphores=[], cbs=cbs1),
        )

        # ── phase 2: the combine into the bf16 output ──
        out_mem = ttnn.DRAM_MEMORY_CONFIG if memory_config is None else memory_config
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, B, Hq, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, out_mem
        )
        cbs2 = [
            cb(CB_IN_M, 2, F32_TILE, ttnn.float32),
            cb(CB_IN_L, 2, F32_TILE, ttnn.float32),
            cb(CB_IN_O, 2 * DHt, F32_TILE, ttnn.float32),
            cb(CB_POS_C, 1, pos_stick, ttnn.int32),
            cb(CB_POS_D, 1, pos_stick, ttnn.int32),
            cb(CB_OUT_FINAL, DHt, BF16_TILE, ttnn.bfloat16),
            cb(CB_ACC, DHt, F32_TILE, ttnn.float32),
            cb(CB_ACC_2, DHt, F32_TILE, ttnn.float32),
        ] + [
            cb(i, 1, F32_TILE, ttnn.float32)
            for i in (CB_PREV_SUM_2, CB_EMD_2, CB_CUR_MAX, CB_PREV_MAX, CB_CUR_SUM, CB_PREV_SUM, CB_EMD)
        ]
        reader2_ct = list(prefix)
        for t in (cur_pos, scratch):
            reader2_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer2_ct = list(prefix)
        for t in (cur_pos, out):
            writer2_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        kernels2 = [
            kernel(
                "reader_combine.cpp",
                reader2_ct,
                rt_args([cur_pos.buffer_address(), scratch.buffer_address()]),
                ttnn.ReaderConfigDescriptor(),
            ),
            kernel(
                "writer_combine.cpp",
                writer2_ct,
                rt_args([cur_pos.buffer_address(), out.buffer_address()]),
                ttnn.WriterConfigDescriptor(),
            ),
            kernel("compute_combine.cpp", list(prefix), rt_args([]), compute_cfg, with_defines=True),
        ]
        return ttnn.generic_op(
            [scratch, cur_pos, out], ttnn.ProgramDescriptor(kernels=kernels2, semaphores=[], cbs=cbs2)
        )
