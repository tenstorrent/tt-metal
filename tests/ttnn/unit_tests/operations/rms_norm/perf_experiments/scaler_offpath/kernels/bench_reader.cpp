// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BENCH READER (perf idea I4: get the one-time reduce-scaler preparation
// off the reader's pre-read critical path, and/or make it cheaper).
//
// This is NOT a production kernel. It reconstructs exactly the opening sequence of
// rms_norm_reader.cpp `kernel_main` — prepare the reduce scaler tile, then read
// this core's C input tiles of a DRAM-interleaved TILE tensor behind ONE barrier —
// with everything else (gamma, masks, chunking, sharding, multi-block) removed, so
// the measured delta is attributable to WHERE / HOW the scaler tile is built.
//
// VARIANTS (compile-time arg 1):
//   0 prep_first        the op's CURRENT order: prepare_reduce_scaler(), then the
//                       block read (reserve -> issue -> barrier -> push).
//   1 after_issue       reserve -> issue -> prepare_reduce_scaler() -> barrier -> push.
//   2 after_push        reserve -> issue -> barrier -> push -> prepare_reduce_scaler().
//   3 cheap_first       the scaler built WITHOUT the whole-tile NoC zero-fill
//                       (row 0 of each face only), at the current position.
//   4 cheap_after_push  cheap build, after the block push.
//   5 writer_prep       the reader does NOT build the scaler at all; the WRITER
//                       (BRISC, idle until the stat tile exists) builds it.
//   6 cheap_poisoned    cheap build, after the push, with the scaler tile
//                       PRE-FILLED with +Inf bf16 first. A CORRECTNESS PROBE, not a
//                       perf variant: it proves what the REDUCE_ROW LLK actually
//                       reads out of the scaler tile (its ns includes the poison
//                       fill, which no shipping variant would pay).
//
// RAW-NoC / HELPER NOTE (variants 3/4/6). `dataflow_kernel_lib::prepare_reduce_scaler`
// always zero-fills the whole tile through the NoC
// (reduce_helpers_dataflow.inl:198-200: async_write_zeros + write_zeros_l1_barrier)
// before writing the scaler into row 0 of each face. The cheap variants call the
// helper's OWN inner fill (`fill_each_face_row0`) and skip only that zero-fill —
// i.e. they are the helper minus one step, expressible as a helper option, not a
// re-implementation. What the zero-fill buys is documented by variant 6.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_input_tiles = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_wmask = 3;

// bf16 tile geometry (the scaler CB's format is a helper static_assert).
constexpr uint32_t FACE_U32 = 128;  // 16x16 bf16 = 512 B
constexpr uint32_t NUM_FACES = 4;

// The helper's row-0-per-face fill WITHOUT the preceding whole-tile NoC zero-fill.
FORCE_INLINE void prepare_reduce_scaler_no_zero(uint32_t addr, float scaler_f) {
    static_assert(get_dataformat(cb_scaler) == DataFormat::Float16_b, "scaler CB must be bf16");
    const uint32_t bits = dataflow_kernel_lib::float_to_scaler_bits<DataFormat::Float16_b>(scaler_f);
    dataflow_kernel_lib::fill_each_face_row0<DataFormat::Float16_b, NUM_FACES>(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr), bits);
}

// Worst-case garbage under a skipped zero-fill: +Inf in every bf16 lane.
FORCE_INLINE void poison_tile(uint32_t addr) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t i = 0; i < FACE_U32 * NUM_FACES; ++i) {
        p[i] = 0x7F807F80u;  // two bf16 +Inf
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t C = get_compile_time_arg_val(0);
    constexpr uint32_t VARIANT = get_compile_time_arg_val(1);
    // The HAS_ANY_TAIL leg: also build the 0/1 column mask (`prepare_reduce_mask`),
    // consumed by the compute's masked tail chain INSIDE sumsq — i.e. a constant
    // with an EARLIER deadline than the scaler's. Exercised with valid_elems = 32
    // (an all-ones mask) so the reference stays the plain row sum while the CB, its
    // preparation and its consumer sit exactly where the real op's do.
    constexpr uint32_t WITH_MASK = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_start = get_arg_val<uint32_t>(1);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(2);

    float inv_w;
    __builtin_memcpy(&inv_w, &inv_w_bits, sizeof(inv_w));

    // --- the op's current constant preparation, verbatim ---
    auto prep_helper = [&]() {
        MaybeDeviceZoneScope("rd_prep_scaler");
        dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            inv_w);
    };
    auto prep_cheap = [&](bool poison) {
        MaybeDeviceZoneScope("rd_prep_scaler_cheap");
        cb_reserve_back(cb_scaler, 1);
        const uint32_t addr = get_write_ptr(cb_scaler);
        if (poison) {
            poison_tile(addr);
        }
        prepare_reduce_scaler_no_zero(addr, inv_w);
        cb_push_back(cb_scaler, 1);
    };

    // The mask ALWAYS uses the unmodified helper: `prepare_reduce_mask`'s zeros are
    // LOAD-BEARING (the partial fill writes 1.0 only in the valid columns and relies
    // on the whole-tile zero-fill for the rest), so the cheap no-zero-fill build is
    // valid for a FULL scaler fill only, never for a mask or a PARTIAL scaler.
    auto prep_mask = [&]() {
        if constexpr (WITH_MASK) {
            MaybeDeviceZoneScope("rd_prep_mask");
            dataflow_kernel_lib::prepare_reduce_mask<cb_wmask, ckernel::ReduceDim::REDUCE_ROW>(32);
        }
    };

    if constexpr (VARIANT == 0) {
        prep_helper();
        prep_mask();
    } else if constexpr (VARIANT == 3) {
        prep_cheap(false);
        prep_mask();
    }

    const uint32_t tile_bytes = get_tile_size(cb_input_tiles);
    const auto acc_tiles = TensorAccessor(src_args, src_addr, tile_bytes);
    {
        MaybeDeviceZoneScope("rd_in_reserve");
        cb_reserve_back(cb_input_tiles, C);
    }
    const uint32_t dst = get_write_ptr(cb_input_tiles);
    {
        MaybeDeviceZoneScope("rd_in_issue");
        for (uint32_t c = 0; c < C; ++c) {
            noc_async_read_tile(tile_start + c, acc_tiles, dst + c * tile_bytes);
        }
    }
    if constexpr (VARIANT == 1) {
        prep_helper();
        prep_mask();
    }
    {
        MaybeDeviceZoneScope("rd_in_barrier");
        noc_async_read_barrier();
    }
    cb_push_back(cb_input_tiles, C);

    if constexpr (VARIANT == 2) {
        prep_helper();
        prep_mask();
    } else if constexpr (VARIANT == 4) {
        prep_cheap(false);
        prep_mask();
    } else if constexpr (VARIANT == 6) {
        prep_cheap(true);
        prep_mask();
    }
    // VARIANT == 5: the writer builds the scaler.
}
