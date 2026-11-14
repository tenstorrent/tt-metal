# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``SdpaDecodeLse``: the stock paged decode attention (``ttnn.transformer.paged_scaled_dot_product_attention_decode``)
for the calls Gemma 4's global layers make, with each (row, q-head)'s softmax statistics as a second output -- the
building block of the "pair split" of a global layer's KV sequence over the two chips that hold one KV head.

One ``ttnn.generic_op`` program: copies of the stock flash-decode kernels (``kernels/``, each marked where it
differs) and the stock factory's core assignment re-expressed in python (``planner.py``). Per batch row the
``cores_per_head`` cores of its group split the keys ``0..cur_pos`` in K chunks, run the online-softmax recurrence
and merge their partial (m, l, O) in a binary tree on the group's root, which normalises O (identical to the stock
op) and now also writes its final m and l.

Statistics output (``stats``): ``[1, B, PNH, 64]`` bf16 TILE (``PNH`` = Hq padded to 32), interleaved:

* ``stats[0, b, h, 0]``  = ``m``: the row max of the RAW scores ``q_h . k_j`` (the kernel folds the scale into
  its exponentials, so m is unscaled);
* ``stats[0, b, h, 32]`` = ``l``: ``sum_j exp((q_h . k_j - m) * scale)`` over ``j = 0..cur_pos[b]``;
* the log-sum-exp of the scaled scores is ``lse = scale * m + ln(l)`` (``lse_from_stats``);
* every other element is unspecified (the other columns of the column tiles; with <= 16 heads the kernels run
  16x32 half tiles and rows 16..31 of each tile are never written), and rows with ``cur_pos = -1`` are untouched
  in both outputs (whatever the allocation held).

Why two bf16 numbers instead of one fp32 LSE: the stock kernels keep m and l in bf16 CBs and, at the stock
compute config the model runs (HiFi2, fp32 dest accumulation OFF), the dest itself is 16-bit, so an LSE formed on
device would be rounded to bf16 -- an absolute error up to 0.125 at |lse| ~ 30, i.e. merge weights off by up to
13 %. ``m`` is exact (a max of bf16 scores) and ``l`` carries bf16's 2^-9 relative error (0.002 in ln l), and the
O the op writes was normalised by that same bf16 l, so a merge with these statistics reproduces the unnormalised
accumulators consistently. Form the LSE (or the merge weights) in fp32 from the two: ``lse_from_stats``
(host) / ``merge_torch``; on device, typecast the two columns to fp32 first.

Pair merge of two calls over the halves of a row's keys: ``M = max(lse_a, lse_b)``,
``O = (e^(lse_a - M) O_a + e^(lse_b - M) O_b) / (e^(lse_a - M) + e^(lse_b - M))``.

Pair split (``pair_split=True``; Gemma 4 31B's global layers on a (1, 8) Galaxy row hold each of their 4 KV heads on
two adjacent chips, column c holding KV head c // 2). On a mesh of shape ``(rows, cols)``, ``cols`` even, the
device at ``(r, c)`` runs its own program with pair half ``h = c % 2`` (a compile-time arg, so part of the program
hash) and its partner is ``(r, c ^ 1)``; both get the same q, caches, page table and cur_pos. A row at ``cur_pos``
has ``n = ceil((cur_pos + 1) / k_chunk_size)`` K chunks; half 0 attends chunks ``[0, ceil(n / 2))``, half 1 chunks
``[ceil(n / 2), n)``, each device spreading its half over its ``cores_per_head`` cores like the stock spreads the
whole row (``planner.pair_workload_for_core``), so each chip reads half the row's K / V bytes. Per device:

* ``out`` = the softmax-normalised attention over this half's keys only, ``stats`` = this half's m / l (layout
  above), so ``lse_self = scale * m + ln(l)`` from the device's own stats;
* an empty half (half 1 of a row with one chunk, ``cur_pos < k_chunk_size``) writes the finite sentinel
  ``O = 0``, ``m = -30000`` (bf16 stores -29952) and ``l = 1`` over the row's whole output / stats tiles, so
  ``lse = -30000 * scale`` is hugely negative (for any scale >= 0.01), its merge weight is 0 and a downstream
  ``0 * O`` stays finite;
* rows with ``cur_pos = -1`` are skipped as without the split (both outputs untouched);
* the merge with the partner's ``(O, lse)``: ``w_self = sigmoid(lse_self - lse_partner)``,
  ``w_partner = 1 - w_self = sigmoid(lse_partner - lse_self)``, ``O = w_self * O_self + w_partner * O_partner``
  (``merge_torch`` is the same in fp32), ``lse = max + ln(e^(lse_self - max) + e^(lse_partner - max))``.

``pair_split=False`` is the single-program op above, unchanged (no extra define or argument reaches the kernels).
``unsupported_reason(..., pair_split=True)`` refuses a device that is not a mesh, a single-device mesh and an odd
number of mesh columns.

Contract (``unsupported_reason`` refuses the rest with the reason; the caller keeps the stock op):
``q [1, B, Hq <= 32, D]`` bf16 TILE interleaved (DRAM or L1; a height-sharded q must be moved to DRAM first, as
the splitfree call site does), ``D`` in {256, 512}; ``k_cache / v_cache [blocks, nkv, block, Dk]`` bf16 or
bfloat8_b TILE interleaved with the stock ``PagedCacheGeometryOverride`` view (``num_kv_heads`` must be 1,
``block_size`` a multiple of 32, elements per block preserved); ``page_table [B, max_blocks]`` int32 ROW_MAJOR;
``cur_pos [B]`` int32 ROW_MAJOR (-1 skips the row); causal; ``scale`` (default ``1/sqrt(D)`` like the stock; the
model passes 1.0); no attention mask, sink, sliding window or MLA; ``cache_position_modulo`` passes through to
the stock kernels (a multiple of the block). ``program_config`` (``ttnn.SDPAProgramConfig``): grid, ``k_chunk_size``
(a power-of-2 multiple of 32; 0 = the stock's dynamic chunk is not ported), ``max_cores_per_head_batch``,
``exp_approx_mode``; ``sub_core_grids`` refused. ``program_config=None`` takes the stock paged default (device
grid, k_chunk 32, 1 core per head, approximate exp). ``compute_kernel_config=None`` takes the stock default
(HiFi2, math_approx_mode, fp32 dest OFF); pass one to match a caller that sets it.
"""

from pathlib import Path

import torch

import ttnn
from models.demos.gemma4.custom_ops.grid import compute_grid, sdpa_decode_cb_budget_bytes
from models.demos.gemma4.custom_ops.sdpa_decode_lse import planner as P

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
HEAD_DIMS = (256, 512)
KV_DTYPES = {ttnn.bfloat16: "bfloat16", ttnn.bfloat8_b: "bfloat8_b"}
_CB_DTYPES = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b, "int32": ttnn.int32}
STOCK_PAGED_DEFAULT_K_CHUNK = 32  # sdpa_decode.cpp kDefaultDecodeChunkSize
STOCK_PAGED_DEFAULT_MAX_CORES_PER_HEAD = 1  # sdpa_decode.cpp kDefaultMaxCoresPerHeadBatch


def _shape(t):
    return [int(v) for v in t.shape]


def _mesh_shape(device):
    """``(rows, cols)`` of a mesh device, None for a device without a mesh shape."""
    shape = getattr(device, "shape", None)
    if shape is None or callable(shape):
        return None
    return tuple(int(d) for d in shape)


def _row_ranges(gx, n):
    """The first ``n`` cores of a ``gx``-wide grid in row-major order as at most two ``ttnn.CoreRange``s."""
    full, rem = divmod(int(n), int(gx))
    ranges = []
    if full:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    return ttnn.CoreRangeSet(ranges)


class SdpaDecodeLse:
    # ── torch references and the statistics helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def golden(q, k_cache, v_cache, page_table, cur_pos, *, scale=1.0, num_kv_heads=None, block_size=None):
        """fp32 torch reference: ``q [1, B, Hq, D]``, the paged caches ``[blocks, nkv, block, D]`` (read through the
        ``num_kv_heads`` / ``block_size`` view when given), ``page_table [B, max_blocks]``, ``cur_pos [B]`` (-1: the
        row is skipped). Returns ``(out [1, B, Hq, D], lse [1, B, Hq])`` fp32 (skipped rows: zeros / -inf)."""
        q = q.float()
        _, B, Hq, D = q.shape
        nkv = int(num_kv_heads or k_cache.shape[1])
        block = int(block_size or k_cache.shape[2])
        hpg = Hq // nkv
        k_flat = k_cache.float().reshape(-1, nkv, block, D)
        v_flat = v_cache.float().reshape(-1, nkv, block, D)
        out = torch.zeros(1, B, Hq, D, dtype=torch.float32)
        lse = torch.full((1, B, Hq), float("-inf"), dtype=torch.float32)
        for b in range(B):
            pos = int(cur_pos[b])
            if pos < 0:
                continue
            positions = torch.arange(0, pos + 1)
            blocks = page_table[b, positions // block].long()
            offs = (positions % block).long()
            for g in range(nkv):
                kk = k_flat[blocks, g, offs]  # [n, D]
                vv = v_flat[blocks, g, offs]
                heads = q[0, b, g * hpg : (g + 1) * hpg]  # [hpg, D]
                s = heads @ kk.T * float(scale)
                out[0, b, g * hpg : (g + 1) * hpg] = torch.softmax(s, dim=-1) @ vv
                lse[0, b, g * hpg : (g + 1) * hpg] = torch.logsumexp(s, dim=-1)
        return out, lse

    @staticmethod
    def lse_from_stats(stats, num_heads, scale=1.0):
        """fp32 ``[1, B, num_heads]`` log-sum-exp from a torch copy of the ``stats`` output
        (``ttnn.to_torch(stats)``): ``scale * m + ln(l)``. Rows the op skipped hold whatever the buffer held."""
        stats = stats.float()
        m = stats[..., :num_heads, 0]
        l = stats[..., :num_heads, P.TILE]
        return float(scale) * m + torch.log(l)

    @staticmethod
    def merge_torch(out_a, lse_a, out_b, lse_b):
        """The pair merge in fp32: ``out_* [..., H, D]``, ``lse_* [..., H]`` -> ``(out, lse)`` of the union of the
        two key sets. A half with ``lse = -inf`` (no keys: a skipped row, whose output buffer holds garbage)
        contributes nothing."""
        lse_a, lse_b = lse_a.float(), lse_b.float()
        live_a, live_b = torch.isfinite(lse_a), torch.isfinite(lse_b)
        m = torch.maximum(lse_a, lse_b)
        zero = torch.zeros_like(m)
        wa = torch.where(live_a, torch.exp(lse_a - m), zero)
        wb = torch.where(live_b, torch.exp(lse_b - m), zero)
        oa = torch.where(live_a[..., None], out_a.float(), torch.zeros_like(out_a, dtype=torch.float32))
        ob = torch.where(live_b[..., None], out_b.float(), torch.zeros_like(out_b, dtype=torch.float32))
        den = wa + wb
        out = (wa[..., None] * oa + wb[..., None] * ob) / den[..., None]
        return out, m + torch.log(den)

    # ── configuration and planning ─────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _program_knobs(device, program_config):
        """``(grid, k_chunk_size, max_cores_per_head_batch, exp_approx_mode)`` as the stock paged op reads them."""
        if program_config is None:
            return compute_grid(device), STOCK_PAGED_DEFAULT_K_CHUNK, STOCK_PAGED_DEFAULT_MAX_CORES_PER_HEAD, True
        g = program_config.compute_with_storage_grid_size
        approx = program_config.exp_approx_mode
        return (
            (int(g.x), int(g.y)),
            int(program_config.k_chunk_size),
            int(program_config.max_cores_per_head_batch),
            True if approx is None else bool(approx),
        )

    @staticmethod
    def _compute_knobs(compute_kernel_config):
        """``(math_fidelity, math_approx_mode, fp32_dest_acc_en, dst_full_sync_en)``; None = the stock default
        ``init_device_compute_kernel_config(arch, None, HiFi2, true, false, false)``."""
        if compute_kernel_config is None:
            return ttnn.MathFidelity.HiFi2, True, False, False
        c = compute_kernel_config
        return (
            c.math_fidelity,
            bool(getattr(c, "math_approx_mode", True)),
            bool(getattr(c, "fp32_dest_acc_en", False)),
            bool(getattr(c, "dst_full_sync_en", False)),
        )

    @classmethod
    def plan(
        cls,
        q,
        k_cache,
        v_cache,
        cur_pos,
        page_table,
        *,
        scale=None,
        num_kv_heads=None,
        block_size=None,
        cache_position_modulo=None,
        program_config=None,
        compute_kernel_config=None,
        pair_split=False,
    ):
        """The host plan (``planner.LsePlan``, per device under ``pair_split``) of a call; ValueError when the shapes
        leave the contract."""
        device = q.device()
        _, B, Hq, D = _shape(q)
        grid, k_chunk, max_cph, exp_approx = cls._program_knobs(device, program_config)
        _, _, fp32_dest, _ = cls._compute_knobs(compute_kernel_config)
        block = int(block_size or k_cache.shape[2])
        g = P.Geometry(
            batch=int(page_table.shape[0]),
            num_q_heads=Hq,
            head_dim=D,
            max_blocks=int(page_table.shape[-1]),
            block_size=block,
            num_cache_blocks=int(k_cache.shape[0]),
            kv_dtype=KV_DTYPES.get(k_cache.dtype, str(k_cache.dtype)),
            k_chunk_size=k_chunk,
            scale=float(scale) if scale is not None else 1.0 / float(int(q.padded_shape[-1])) ** 0.5,
            cur_pos_stick=int(cur_pos.buffer_aligned_page_size()),
            page_table_stick=int(page_table.buffer_aligned_page_size()),
            cache_position_modulo=int(cache_position_modulo or 0),
            fp32_dest_acc_en=fp32_dest,
            exp_approx_mode=exp_approx,
        )
        return P.plan(g, grid, max_cph, pair_split=bool(pair_split))

    @classmethod
    def unsupported_reason(
        cls,
        q,
        k_cache,
        v_cache,
        cur_pos,
        page_table,
        *,
        scale=None,
        num_kv_heads=None,
        block_size=None,
        sliding_window=None,
        cache_position_modulo=None,
        program_config=None,
        compute_kernel_config=None,
        memory_config=None,
        pair_split=False,
        half_block=False,
    ):
        """None when the call is in contract (shapes, formats, layouts, the plan and its L1 fit; under ``pair_split``
        or ``half_block`` a mesh with an even number of columns), else why."""
        if pair_split or half_block:
            reason = P.pair_split_reason(_mesh_shape(q.device()))
            if reason is not None:
                return reason
        if q.layout != ttnn.TILE_LAYOUT or q.dtype != ttnn.bfloat16:
            return "q must be a bf16 TILE tensor"
        if q.is_sharded():
            return "q must be interleaved (DRAM or L1): move a height-sharded q to DRAM first"
        qs = _shape(q)
        if len(qs) != 4 or qs[0] != 1 or not 1 <= qs[2] <= P.MAX_Q_HEADS:
            return f"q must be [1, B, Hq <= {P.MAX_Q_HEADS}, D], got {qs}"
        B, Hq, D = qs[1], qs[2], qs[3]
        if D not in HEAD_DIMS:
            return f"head_dim {D} not in {HEAD_DIMS}"
        for name, t in (("k", k_cache), ("v", v_cache)):
            if t.layout != ttnn.TILE_LAYOUT or t.dtype not in KV_DTYPES or t.is_sharded():
                return f"{name} cache must be a bf16 or bfloat8_b TILE interleaved tensor"
            if len(_shape(t)) != 4:
                return f"{name} cache must be [blocks, nkv, block, D], got {_shape(t)}"
        if k_cache.dtype != v_cache.dtype or _shape(k_cache) != _shape(v_cache):
            return "k and v caches must share dtype and shape"
        kb, kh, ks, kd = _shape(k_cache)
        nkv = int(num_kv_heads or kh)
        block = int(block_size or ks)
        if nkv != 1:
            return f"one KV head per chip only (MQA), the view has {nkv}"
        if block % P.TILE:
            return f"block_size {block} is not a multiple of {P.TILE} (no sub-tile block padding here)"
        if num_kv_heads is not None or block_size is not None:
            if kh * ks * kd != nkv * block * D:
                return f"cache view mismatch: cache {kh}x{ks}x{kd} elements per block, view {nkv}x{block}x{D}"
        elif kd != D:
            return f"q head_dim {D} != cache head_dim {kd} (pass num_kv_heads / block_size for a shared cache)"
        if sliding_window:
            return "no sliding window (global layers only)"
        if page_table.layout != ttnn.ROW_MAJOR_LAYOUT or page_table.dtype != ttnn.int32 or len(_shape(page_table)) != 2:
            return "page_table must be a 2-D int32 ROW_MAJOR tensor"
        if page_table.is_sharded():
            return "page_table must be interleaved"
        if int(page_table.shape[0]) != B:
            return f"page_table rows {int(page_table.shape[0])} != batch {B}"
        ps = [int(v) for v in cur_pos.padded_shape]
        if cur_pos.layout != ttnn.ROW_MAJOR_LAYOUT or cur_pos.dtype != ttnn.int32 or cur_pos.is_sharded():
            return "cur_pos must be an int32 ROW_MAJOR interleaved tensor"
        vol = 1
        for v in ps:
            vol *= v
        if vol != ps[-1] or ps[-1] < B:
            return f"cur_pos must hold the B={B} positions in its last dim, got padded shape {ps}"
        if program_config is not None and program_config.sub_core_grids is not None:
            return "sub_core_grids are not supported"
        if memory_config is not None and memory_config.is_sharded():
            return "the output must be interleaved"
        try:
            plan = cls.plan(
                q,
                k_cache,
                v_cache,
                cur_pos,
                page_table,
                scale=scale,
                num_kv_heads=num_kv_heads,
                block_size=block_size,
                cache_position_modulo=cache_position_modulo,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                pair_split=pair_split,
            )
        except ValueError as e:
            return str(e)
        device = q.device()
        dev_grid = device.compute_with_storage_grid_size()
        gx, gy = plan.cores.grid
        if gx > int(dev_grid.x) or gy > int(dev_grid.y):
            return f"program grid {gx}x{gy} exceeds the device grid {int(dev_grid.x)}x{int(dev_grid.y)}"
        budget = sdpa_decode_cb_budget_bytes(getattr(device, "arch", lambda: None)())
        if plan.cb_total_bytes > budget:
            return f"circular buffers need {plan.cb_total_bytes} B per core, budget {budget} B: {plan.describe()}"
        return None

    # ── the op ─────────────────────────────────────────────────────────────────────────────────────────────────
    @classmethod
    def op(
        cls,
        q,
        k_cache,
        v_cache,
        cur_pos,
        page_table,
        *,
        scale=None,
        num_kv_heads=None,
        block_size=None,
        cache_position_modulo=None,
        program_config=None,
        compute_kernel_config=None,
        memory_config=None,
        stats_memory_config=None,
        pair_split=False,
        owned=False,
        half_block=False,
        half=None,
    ):
        """Paged causal decode attention + row statistics. Returns ``(out [1, B, Hq, D] bf16 TILE, stats
        [1, B, PNH, 64] bf16 TILE)``; see the module docstring for the stats layout and the LSE convention.
        ``pair_split``: each device attends only its pair half of every row's keys (module docstring), one program
        per mesh coordinate. ``owned`` (pair-sharded global KV, ``tt/kv_pair_shard.py``): the stock single
        program over each device's own compacted ``page_table`` and ``cur_pos``; a ``cur_pos = -1`` row writes
        the empty-half sentinel (O = 0, m = -30000, l = 1) instead of being skipped. Exclusive with pair_split."""
        assert not (pair_split and owned), "sdpa_decode_lse: owned and pair_split are exclusive"
        # half_block (pair-sharded global KV, the half-block layout): each device of a pair holds the block_size-token
        # half ``h`` (its column's parity, or ``half`` on every device -- a single-chip test) of every logical
        # block in the slot the page table names; the kernels derive the effective cur_pos from the global one and
        # the writer writes the empty-half sentinel when a device owns none of a row's keys.
        assert not (half_block and (pair_split or owned)), "sdpa_decode_lse: half_block is its own mode"
        kw = dict(
            scale=scale,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            cache_position_modulo=cache_position_modulo,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            pair_split=bool(pair_split),
        )
        reason = cls.unsupported_reason(q, k_cache, v_cache, cur_pos, page_table, memory_config=memory_config, **kw)
        assert reason is None, f"sdpa_decode_lse: {reason}"
        plan = cls.plan(q, k_cache, v_cache, cur_pos, page_table, **kw)
        g, cores = plan.geometry, plan.cores
        device = q.device()
        fidelity, approx, fp32_dest, full_sync = cls._compute_knobs(compute_kernel_config)

        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, g.batch, g.num_q_heads, g.head_dim]),
            q.dtype,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG if memory_config is None else memory_config,
        )
        stats = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, g.batch, g.pnh, P.STATS_COLS]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG if stats_memory_config is None else stats_memory_config,
        )

        core_set = _row_ranges(cores.grid[0], cores.active_cores)

        def phys(xy):
            c = device.worker_core_from_logical_core(ttnn.CoreCoord(xy[0], xy[1]))
            return int(c.x), int(c.y)

        addrs = dict(
            q=q.buffer_address(),
            k=k_cache.buffer_address(),
            v=v_cache.buffer_address(),
            cur_pos=cur_pos.buffer_address(),
            page_table=page_table.buffer_address(),
            out=out.buffer_address(),
            stats=stats.buffer_address(),
        )
        per_core = P.runtime_args(g, cores, phys, addrs)

        absent = [0, 0]  # TensorAccessorArgs of an absent tensor (ArgConfig::None, page size 0): mask, sink
        reader_ct = list(plan.reader_ct)
        for t in (q, k_cache, v_cache, None, cur_pos, page_table, None):
            reader_ct += absent if t is None else list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        writer_ct = list(plan.writer_ct)
        for t in (out, stats):
            writer_ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())

        def program(half=None):
            """One device's program. ``half``: its pair half under pair_split (appended to every kernel's
            compile-time args, so part of the program hash), None for the stock single program."""
            if half_block:
                pair_ct = [int(half)]
            else:
                pair_ct = list(plan.pair_ct(half)) if plan.pair_split else []
            # [] without pair_split: the reader / writer descriptors stay exactly the stock ones (no defines)
            dm_defines = {"defines": list(plan.dataflow_defines)} if plan.pair_split else {}
            # owned mode: only the writer differs (the -1 row sentinel); the reader stays the stock descriptor
            writer_defines = {"defines": [("SDPA_LSE_OWNED", "1")]} if owned else dm_defines
            compute_defines = list(plan.defines)
            if half_block:
                hb = ("SDPA_LSE_HALF_BLOCK", str(int(g.block_size)))
                dm_defines = {"defines": [hb]}
                writer_defines = {"defines": [hb, ("SDPA_LSE_OWNED", "1")]}
                compute_defines = compute_defines + [hb]
            reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for (x, y), (r, w, c) in per_core.items():
                reader_rt[x][y] = [int(v) for v in r]
                writer_rt[x][y] = [int(v) for v in w]
                compute_rt[x][y] = [int(v) for v in c]
            cbs = []
            for s in plan.cbs:
                tile = ttnn.TileDescriptor(height=s.tile[0], width=s.tile[1]) if s.tile is not None else None
                cbs.append(
                    ttnn.CBDescriptor(
                        total_size=s.total_size,
                        core_ranges=core_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=s.index, data_format=_CB_DTYPES[s.dtype], page_size=s.page_size, tile=tile
                            )
                        ],
                    )
                )
            sems = [
                ttnn.SemaphoreDescriptor(id=i, core_ranges=core_set, initial_value=0)
                for i in (P.SEM_REDUCER, P.SEM_OUTPUT, P.SEM_K_MCAST)
            ]
            kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "reader_decode_lse.cpp"),
                    core_ranges=core_set,
                    compile_time_args=reader_ct + pair_ct,
                    runtime_args=reader_rt,
                    config=ttnn.ReaderConfigDescriptor(),
                    **dm_defines,
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "writer_decode_lse.cpp"),
                    core_ranges=core_set,
                    compile_time_args=writer_ct + pair_ct,
                    runtime_args=writer_rt,
                    config=ttnn.WriterConfigDescriptor(),
                    **writer_defines,
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "sdpa_flash_decode_lse.cpp"),
                    core_ranges=core_set,
                    compile_time_args=list(plan.compute_ct) + pair_ct,
                    defines=compute_defines,
                    runtime_args=compute_rt,
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=fidelity,
                        fp32_dest_acc_en=fp32_dest,
                        math_approx_mode=approx,
                        dst_full_sync_en=full_sync,
                    ),
                ),
            ]
            return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)

        if half_block and half is not None:
            programs = program(int(half))
        elif plan.pair_split or half_block:
            # one program per mesh coordinate, the half = the column's parity (the pair is columns (c, c ^ 1));
            # generic_op hashes each coordinate range with its program, compile-time args included. The two
            # half programs are built once: MeshProgramDescriptor stores a copy per coordinate.
            rows, cols = _mesh_shape(device)
            by_half = {h: program(h) for h in (0, 1)}
            programs = ttnn.MeshProgramDescriptor()
            for r in range(rows):
                for c in range(cols):
                    coord = ttnn.MeshCoordinate(r, c)
                    programs[ttnn.MeshCoordinateRange(coord, coord)] = by_half[P.pair_half_of(c)]
        else:
            programs = program()
        out = ttnn.generic_op([q, k_cache, v_cache, cur_pos, page_table, stats, out], programs)
        return out, stats
