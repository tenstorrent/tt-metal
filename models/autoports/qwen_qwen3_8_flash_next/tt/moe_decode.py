# SPDX-License-Identifier: Apache-2.0
"""Single-token routed experts: every selected local expert at once.

The prefill expert op walks one expert at a time over the whole grid. For one
decode token each rank instead computes all of its selected experts in two
generic programs, reading only those experts' weights:

1. gate/up: one (local slot, 32-column tile) item per core;
   H = score * SiLU(x @ W_gate) * (x @ W_up), one 640-wide row per slot.
2. down: one 32-column output tile per core; out = sum_slots H @ W_down.

Expert ids and scores stay on device. Each expert tensor keeps its own DRAM
buffer; a replicated table holds their (mesh-identical) base addresses. Only
row 0 of a tile is meaningful; phase 2 keeps its input padding rows zero so its
output is exactly zero below row 0. FP32 destination accumulation throughout.
Measured against an FP32 host evaluation of the same BF4 experts: 0.992x
(packer truncation); the prefill expert op path measures 1.04x.

Diagnostics: MOE_EXP=noread,nomath compiles out weight reads / matmuls.
"""

import os

import numpy as np
import torch

import ttnn

_TILE_BF16 = 2048
_TILE_BFP4 = 576
_SLOTS = 10
_HEADER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// Local selected experts in top-k order. meta: ids[32] u32 @0, scores[32] bf16 @128,
// rank u32 @192, table[384] u32 @256.
template <typename I, typename S, typename R, typename T>
uint32_t local_slots(const Noc& noc, const I& ids, const S& scores, const R& rank_tensor, const T& table,
                     uint32_t meta, uint32_t* expert, uint16_t* score) {
    noc.async_read(ids, CoreLocalMem<uint32_t>(meta), 128, {.page_id=0}, {});
    noc.async_read(scores, CoreLocalMem<uint32_t>(meta + 128), 64, {.page_id=0}, {});
    noc.async_read(rank_tensor, CoreLocalMem<uint32_t>(meta + 192), 4, {.page_id=0}, {});
    noc.async_read(table, CoreLocalMem<uint32_t>(meta + 256), 1536, {.page_id=0}, {});
    noc.async_read_barrier();
    auto id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta);
    auto sc = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(meta + 128);
    const uint32_t rank = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta + 192);
    uint32_t count = 0;
    for (uint32_t k = 0; k < 10; ++k) {
        const uint32_t e = id[k];
        if (e / 128 != rank || sc[k] == 0) continue;
        expert[count] = e % 128;
        score[count] = sc[k];
        ++count;
    }
    return count;
}
// Workers form a 10-wide rectangle at the grid origin; the index is device-derived
// so every core shares one common runtime-argument list.
inline uint32_t worker_index() { return get_absolute_logical_y() * 10 + get_absolute_logical_x(); }
inline uint32_t table_entry(uint32_t meta, uint32_t index) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta + 256)[index];
}
"""

# ---------------------------------------------------------------- phase 1
_GU_READER = _HEADER + r"""
void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0), KT = get_compile_time_arg_val(1);
    constexpr uint32_t NT = get_compile_time_arg_val(2);
    constexpr auto xa = TensorAccessorArgs<4>();
    constexpr auto ia = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ra = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ta = TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto wa = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(xa, get_common_arg_val<uint32_t>(0));
    const auto ids = TensorAccessor(ia, get_common_arg_val<uint32_t>(1));
    const auto scores = TensorAccessor(sa, get_common_arg_val<uint32_t>(2));
    const auto rank_tensor = TensorAccessor(ra, get_common_arg_val<uint32_t>(3));
    const auto table = TensorAccessor(ta, get_common_arg_val<uint32_t>(4));
    const uint32_t worker = worker_index();
    Noc noc;
    CircularBuffer cx(0), cw(1), cs(2), cc(3), cm(6);
    cm.reserve_back(1);
    const uint32_t meta = cm.get_write_ptr();
    uint32_t expert[10];
    uint16_t score[10];
    const uint32_t count = local_slots(noc, ids, scores, rank_tensor, table, meta, expert, score);
    uint32_t items = 0;
    for (uint32_t i = worker; i < count * NT; i += W) ++items;
    cc.reserve_back(1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cc.get_write_ptr()) = items;
    cc.push_back(1);
    if (!items) return;
    // Row 0 of every activation tile (faces 0 and 1); padding rows are never consumed.
    cx.reserve_back(KT);
    const uint32_t xb = cx.get_write_ptr();
    for (uint32_t t = 0; t < KT; ++t) {
        noc.async_read(x, CoreLocalMem<uint32_t>(xb + t * 2048), 32, {.page_id=t, .offset_bytes=0}, {});
        noc.async_read(x, CoreLocalMem<uint32_t>(xb + t * 2048 + 512), 32, {.page_id=t, .offset_bytes=512}, {});
    }
    noc.async_read_barrier();
    cx.push_back(KT);
    for (uint32_t i = worker; i < count * NT; i += W) {
        const uint32_t slot = i / NT, column = i % NT;
        cs.reserve_back(1);
        auto st = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cs.get_write_ptr());
        for (uint32_t c = 0; c < 16; ++c) { st[c] = score[slot]; st[256 + c] = score[slot]; }
        cs.push_back(1);
        // Gate pages on this RISC, up pages on the writer RISC: two independent
        // request streams; one barrier per item.
        const auto gate = TensorAccessor(wa, table_entry(meta, expert[slot]));
        cw.reserve_back(KT);
        const uint32_t wb = cw.get_write_ptr();
#ifndef EXP_NOREAD
        for (uint32_t k = 0; k < KT; ++k) {
            noc.async_read(gate, CoreLocalMem<uint32_t>(wb + k * 576), 576, {.page_id=k * NT + column}, {});
        }
#endif
        noc.async_read_barrier();
        cw.push_back(KT);
    }
}
"""

_GU_WRITER = _HEADER + r"""
void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0), KT = get_compile_time_arg_val(1);
    constexpr uint32_t NT = get_compile_time_arg_val(2);
    constexpr auto ia = TensorAccessorArgs<4>();
    constexpr auto sa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ra = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ta = TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto ha = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    const auto ids = TensorAccessor(ia, get_common_arg_val<uint32_t>(0));
    const auto scores = TensorAccessor(sa, get_common_arg_val<uint32_t>(1));
    const auto rank_tensor = TensorAccessor(ra, get_common_arg_val<uint32_t>(2));
    const auto table = TensorAccessor(ta, get_common_arg_val<uint32_t>(3));
    const auto hidden = TensorAccessor(ha, get_common_arg_val<uint32_t>(4));
    constexpr auto wa = TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    const uint32_t worker = worker_index();
    Noc noc;
    CircularBuffer cu(5), co(4), cm(7);
    cm.reserve_back(1);
    const uint32_t meta = cm.get_write_ptr();
    uint32_t expert[10];
    uint16_t score[10];
    const uint32_t count = local_slots(noc, ids, scores, rank_tensor, table, meta, expert, score);
    uint32_t pending = NT * 10;  // previous item's output page, written after the next item's up reads
    for (uint32_t i = worker; i < count * NT; i += W) {
        const uint32_t slot = i / NT, column = i % NT;
        const auto up = TensorAccessor(wa, table_entry(meta, 128 + expert[slot]));
        cu.reserve_back(KT);
        const uint32_t ub = cu.get_write_ptr();
#ifndef EXP_NOREAD
        for (uint32_t k = 0; k < KT; ++k) {
            noc.async_read(up, CoreLocalMem<uint32_t>(ub + k * 576), 576, {.page_id=k * NT + column}, {});
        }
#endif
        noc.async_read_barrier();
        cu.push_back(KT);
        if (pending < NT * 10) {
            co.wait_front(1);
            noc.async_write(co, hidden, 2048, {}, {.page_id=pending});
            noc.async_write_barrier();
            co.pop_front(1);
        }
        pending = i;
    }
    if (pending < NT * 10) {
        co.wait_front(1);
        noc.async_write(co, hidden, 2048, {}, {.page_id=pending});
        noc.async_write_barrier();
        co.pop_front(1);
    }
}
"""

_COMPUTE_HEADER = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/circular_buffer.h"

// The trip count is device data: UNPACK reads the reader's control page and
// forwards it so MATH and PACK run the identical loop.
inline uint32_t trip_count(uint32_t cb) {
    uint32_t n = 0;
    UNPACK(({
        cb_wait_front(cb, 1);
        n = *reinterpret_cast<volatile uint32_t*>(get_local_cb_interface(cb).fifo_rd_ptr << 4);
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, n);
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, n);
        cb_pop_front(cb, 1);
    }));
    MATH(({ n = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId); }));
    PACK(({ n = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId); }));
    return n;
}
"""

_GU_COMPUTE = _COMPUTE_HEADER + r"""
void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t cx = 0, cw = 1, cs = 2, cc = 3, co = 4, cu = 5;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cx, cw, co);
    const uint32_t items = trip_count(cc);
    if (!items) return;
    cb_wait_front(cx, KT);
    for (uint32_t item = 0; item < items; ++item) {
        reconfig_data_format_srca(cw);
        matmul_init(cx, cw);
        tile_regs_acquire();
        cb_wait_front(cw, KT);
        cb_wait_front(cu, KT);
#ifndef EXP_NOMATH
        for (uint32_t k = 0; k < KT; ++k) {
            matmul_tiles(cx, cw, k, k, 0);
            matmul_tiles(cx, cu, k, k, 1);
        }
#endif
        cb_pop_front(cw, KT);
        cb_pop_front(cu, KT);
        cb_wait_front(cs, 1);
        reconfig_data_format_srca(cs);
        copy_init(cs);
        copy_tile(cs, 0, 2);
        cb_pop_front(cs, 1);
        silu_tile_init();
        silu_tile(0);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        mul_binary_tile(0, 2, 0);
        tile_regs_commit();
        cb_reserve_back(co, 1);
        tile_regs_wait();
        pack_tile(0, co);
        tile_regs_release();
        cb_push_back(co, 1);
    }
}
"""

# ---------------------------------------------------------------- phase 2
_DOWN_READER = _HEADER + r"""
void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0), NT = get_compile_time_arg_val(1);
    constexpr auto ha = TensorAccessorArgs<2>();
    constexpr auto ia = TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ra = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ta = TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto wa = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    const auto hidden = TensorAccessor(ha, get_common_arg_val<uint32_t>(0));
    const auto ids = TensorAccessor(ia, get_common_arg_val<uint32_t>(1));
    const auto scores = TensorAccessor(sa, get_common_arg_val<uint32_t>(2));
    const auto rank_tensor = TensorAccessor(ra, get_common_arg_val<uint32_t>(3));
    const auto table = TensorAccessor(ta, get_common_arg_val<uint32_t>(4));
    const uint32_t column = worker_index();
    Noc noc;
    CircularBuffer ch(0), cw(1), cc(3), cm(6);
    cm.reserve_back(1);
    const uint32_t meta = cm.get_write_ptr();
    uint32_t expert[10];
    uint16_t score[10];
    const uint32_t count = local_slots(noc, ids, scores, rank_tensor, table, meta, expert, score);
    cc.reserve_back(1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cc.get_write_ptr()) = count;
    cc.push_back(1);
    if (!count) return;
    // This RISC streams K tiles [0, KT/2) of every slot's down weights, the writer
    // RISC the rest (into its own CB); all DRAM reads are issued up front.
    constexpr uint32_t HALF = KT / 2;
    cw.reserve_back(count * HALF);
    const uint32_t wb = cw.get_write_ptr();
    for (uint32_t slot = 0; slot < count; ++slot) {
        const auto down = TensorAccessor(wa, table_entry(meta, 256 + expert[slot]));
#ifndef EXP_NOREAD
        for (uint32_t k = 0; k < HALF; ++k) {
            noc.async_read(down, CoreLocalMem<uint32_t>(wb + (slot * HALF + k) * 576), 576, {.page_id=k * NT + column}, {});
        }
#endif
    }
    // The hidden CB holds exactly 2*KT pages (double buffered), so reservations
    // alternate between two fixed L1 regions: zero both once; only row 0 is rewritten.
    noc.async_write_zeros(ch, 2 * KT * 2048);
    noc.write_zeros_l1_barrier();
    for (uint32_t slot = 0; slot < count; ++slot) {
        ch.reserve_back(KT);
        const uint32_t hb = ch.get_write_ptr();
        for (uint32_t k = 0; k < KT; ++k) {
            const uint32_t page = slot * KT + k;
            noc.async_read(hidden, CoreLocalMem<uint32_t>(hb + k * 2048), 32, {.page_id=page, .offset_bytes=0}, {});
            noc.async_read(hidden, CoreLocalMem<uint32_t>(hb + k * 2048 + 512), 32, {.page_id=page, .offset_bytes=512}, {});
        }
        noc.async_read_barrier();
        if (slot == 0) cw.push_back(count * HALF);
        ch.push_back(KT);
    }
}
"""

_DOWN_WRITER = _HEADER + r"""
void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0), NT = get_compile_time_arg_val(1);
    constexpr auto oa = TensorAccessorArgs<2>();
    constexpr auto ia = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ra = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ta = TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto wa = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    const auto output = TensorAccessor(oa, get_common_arg_val<uint32_t>(0));
    const auto ids = TensorAccessor(ia, get_common_arg_val<uint32_t>(1));
    const auto scores = TensorAccessor(sa, get_common_arg_val<uint32_t>(2));
    const auto rank_tensor = TensorAccessor(ra, get_common_arg_val<uint32_t>(3));
    const auto table = TensorAccessor(ta, get_common_arg_val<uint32_t>(4));
    const uint32_t column = worker_index();
    Noc noc;
    CircularBuffer cv(5), co(4), cm(7);
    cm.reserve_back(1);
    const uint32_t meta = cm.get_write_ptr();
    uint32_t expert[10];
    uint16_t score[10];
    const uint32_t count = local_slots(noc, ids, scores, rank_tensor, table, meta, expert, score);
    constexpr uint32_t HALF = KT / 2;
    if (count) {
        cv.reserve_back(count * (KT - HALF));
        const uint32_t vb = cv.get_write_ptr();
        for (uint32_t slot = 0; slot < count; ++slot) {
            const auto down = TensorAccessor(wa, table_entry(meta, 256 + expert[slot]));
#ifndef EXP_NOREAD
            for (uint32_t k = HALF; k < KT; ++k) {
                noc.async_read(down, CoreLocalMem<uint32_t>(vb + (slot * (KT - HALF) + k - HALF) * 576), 576, {.page_id=k * NT + column}, {});
            }
#endif
        }
        noc.async_read_barrier();
        cv.push_back(count * (KT - HALF));
    }
    co.wait_front(1);
    noc.async_write(co, output, 2048, {}, {.page_id=column});
    noc.async_write_barrier();
    co.pop_front(1);
}
"""

_DOWN_COMPUTE = _COMPUTE_HEADER + r"""
void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t ch = 0, cw = 1, cc = 3, co = 4, cv = 5;
    compute_kernel_hw_startup<SrcOrder::Reverse>(ch, cw, co);
    const uint32_t slots = trip_count(cc);
    matmul_init(ch, cw);
    tile_regs_acquire();
    for (uint32_t slot = 0; slot < slots; ++slot) {
        cb_wait_front(ch, KT);
        cb_wait_front(cw, KT / 2);
        cb_wait_front(cv, KT - KT / 2);
#ifndef EXP_NOMATH
        for (uint32_t k = 0; k < KT / 2; ++k) matmul_tiles(ch, cw, k, k, 0);
        for (uint32_t k = KT / 2; k < KT; ++k) matmul_tiles(ch, cv, k, k - KT / 2, 0);
#endif
        cb_pop_front(ch, KT);
        cb_pop_front(cw, KT / 2);
        cb_pop_front(cv, KT - KT / 2);
    }
    tile_regs_commit();
    cb_reserve_back(co, 1);
    tile_regs_wait();
    pack_tile(0, co);
    tile_regs_release();
    cb_push_back(co, 1);
}
"""


_EXP = "".join(f"#define EXP_{flag.upper()}\n" for flag in os.environ.get("MOE_EXP", "").split(",") if flag)


def _cb(index, pages, page_size, data_format, cores):
    return ttnn.CBDescriptor(
        total_size=pages * page_size,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def _accessor_args(tensors):
    args = []
    for tensor in tensors:
        args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    return args


class SingleTokenExperts:
    """Replaces dispatch + prefill expert op + compact + weighting for one token."""

    def __init__(self, moe, *, gate_up_workers=100, fidelity=ttnn.MathFidelity.LoFi):
        self.moe, self.mesh = moe, moe.ops.mesh
        grid = self.mesh.compute_with_storage_grid_size()
        if grid.x < 10 or grid.y < gate_up_workers // 10 or gate_up_workers % 10:
            raise ValueError("Single-token experts use 10-wide worker rectangles")
        self.gu_workers = gate_up_workers
        self.gu_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, gate_up_workers // 10 - 1))]
        )
        self.down_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 7))])
        addresses = [t.buffer_address() for t in (*moe.gates, *moe.ups, *moe.downs)]
        if len(addresses) != 384:
            raise ValueError("Expected 128 local experts per rank")
        table = torch.from_numpy(np.array(addresses, dtype=np.uint32).view(np.int32))
        self.table = moe.ops.upload(table, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.compute = ttnn.ComputeConfigDescriptor(
            math_fidelity=fidelity, fp32_dest_acc_en=True, math_approx_mode=False
        )

    def __call__(self, x, ids, scores):
        """x [1,1,1,2560] BF16 TILE, ids [1,1,1,32] UINT32 RM, scores [1,1,1,32] BF16 RM.

        Returns this rank's routed partial sum [1,1,1,2560] BF16 TILE.
        """
        moe = self.moe
        # Every core reads row 0 of every x/hidden tile: keep both L1-interleaved
        # so those small reads spread over many cores instead of a few DRAM pages.
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        hidden = ttnn.empty(
            (1, _SLOTS, 32, 640),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        output = ttnn.empty(
            (1, 1, 1, 2560),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        meta = [ids, scores, moe.rank, self.table]
        # Phase 1: gate/up items.
        cores, workers = self.gu_cores, self.gu_workers
        reader = ttnn.KernelDescriptor(
            kernel_source=_EXP + _GU_READER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=[workers, 80, 20, 0] + _accessor_args([x, *meta, moe.gates[0]]),
            common_runtime_args=[t.buffer_address() for t in (x, *meta)],
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=_GU_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=[workers, 80, 20, 0] + _accessor_args([*meta, hidden, moe.ups[0]]),
            common_runtime_args=[t.buffer_address() for t in (*meta, hidden)],
            config=ttnn.WriterConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=_EXP + _GU_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=[80],
            config=self.compute,
        )
        cbs = [
            _cb(0, 80, _TILE_BF16, ttnn.bfloat16, cores),
            _cb(1, 2 * 80, _TILE_BFP4, ttnn.bfloat4_b, cores),
            _cb(5, 2 * 80, _TILE_BFP4, ttnn.bfloat4_b, cores),
            _cb(2, 2, _TILE_BF16, ttnn.bfloat16, cores),
            _cb(3, 1, 64, ttnn.uint32, cores),
            _cb(4, 2, _TILE_BF16, ttnn.bfloat16, cores),
            _cb(6, 1, 2048, ttnn.uint32, cores),
            _cb(7, 1, 2048, ttnn.uint32, cores),
        ]
        ttnn.generic_op([x, *meta, hidden], ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs))
        # Phase 2: down projection, one output column tile per core.
        cores = self.down_cores
        reader = ttnn.KernelDescriptor(
            kernel_source=_EXP + _DOWN_READER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=[20, 80] + _accessor_args([hidden, *meta, moe.downs[0]]),
            common_runtime_args=[t.buffer_address() for t in (hidden, *meta)],
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=_DOWN_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=[20, 80] + _accessor_args([output, *meta, moe.downs[0]]),
            common_runtime_args=[t.buffer_address() for t in (output, *meta)],
            config=ttnn.WriterConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=_EXP + _DOWN_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=[20],
            config=self.compute,
        )
        cbs = [
            _cb(0, 2 * 20, _TILE_BF16, ttnn.bfloat16, cores),
            _cb(1, _SLOTS * 10, _TILE_BFP4, ttnn.bfloat4_b, cores),
            _cb(5, _SLOTS * 10, _TILE_BFP4, ttnn.bfloat4_b, cores),
            _cb(3, 1, 64, ttnn.uint32, cores),
            _cb(4, 1, _TILE_BF16, ttnn.bfloat16, cores),
            _cb(6, 1, 2048, ttnn.uint32, cores),
            _cb(7, 1, 2048, ttnn.uint32, cores),
        ]
        ttnn.generic_op([hidden, *meta, output], ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs))
        return output


def group_single(moe, x):
    """ParallelMoE.group_native for one token, with SingleTokenExperts for routed experts."""
    router = moe.ops.linear(x, moe.router, compute_kernel_config=moe.router_compute)
    values, indices = ttnn.topk(
        ttnn.pad(router[..., :512], [(0, 0), (0, 0), (0, 0), (0, 512)], float("-inf")), k=32, dim=-1, sorted=True
    )
    selected = ttnn.softmax(
        ttnn.add(values, moe.mask), dim=-1, numeric_stable=True, compute_kernel_config=moe.ops.compute
    )
    ids = ttnn.to_layout(ttnn.typecast(indices, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
    scores = ttnn.to_layout(selected, ttnn.ROW_MAJOR_LAYOUT)
    result = moe.single_token(x, ids, scores)
    shared = moe.ops.linear(x, moe.shared_gu, compute_kernel_config=moe.shared_compute)
    shared = moe.ops.linear(
        ttnn.multiply(shared[..., :160], shared[..., 160:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
        moe.shared_down,
        compute_kernel_config=moe.shared_compute,
    )
    return ttnn.add(
        result, ttnn.multiply(shared, router[..., 512:513], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    )
