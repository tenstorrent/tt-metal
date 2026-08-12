// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BENCH (rms_norm perf idea I8) — the cross-core combine's RENDEZVOUS
// only: gather leg -> root's return multicast -> each member's consume of the
// landing tile. No reader, no compute, no reduce, no apply; the payload bytes and
// the transaction counts match the real op's combine so the ONLY thing that varies
// between variants is whether the return multicast is gated on a receiver->sender
// readiness ack (mcast_pipe PRE_HANDSHAKE).
//
// VARIANTS (compile-time VARIANT):
//   0 baseline    — SenderPipe/ReceiverPipe with PRE_HANDSHAKE=true. The op today.
//   1 elide       — PRE_HANDSHAKE=false, stock kernel_lib ReceiverPipe (its ctor
//                   still inits data_ready; safe here ONLY because every receiver
//                   constructs the pipe before its own gather write, which the root
//                   waits on -> a happens-before edge that replaces the ack's).
//   2 elide_noinit— PRE_HANDSHAKE=false + bench::NoInitReceiverPipe (no ctor init;
//                   the INVALID comes from host CreateSemaphore). Safe for ANY
//                   receiver, including one that never gathers (a mcast-box filler).
//
// WHY ELIDING THE ACK IS SOUND HERE (the same argument the op's writer would carry).
// The ack means "my rstd landing slot is free". Without it, back-pressure comes from
// the GATHER, which is strictly stronger: the root cannot send block b+1 until every
// member has written its block-(b+1) partial, and a member cannot reach that write
// until it has finished consuming its block-b landing tile (store_block's barrier).
// So landing-slot reuse is already ordered behind every receiver's consume, for ANY
// number of blocks and a single-slot landing buffer. The flag itself still carries
// data-before-signal (the linked mcast pair), and the receiver's own set(INVALID)
// after receive(b) is ordered before the sender's next VALID by the same chain.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "bench_pipe.hpp"

using namespace dataflow_kernel_lib;

namespace {
constexpr uint32_t cb_stat_partial = 7;
constexpr uint32_t cb_stat_gather = 8;
constexpr uint32_t cb_rstd_send = 10;
constexpr uint32_t cb_rstd = 11;
constexpr uint32_t SEM_GATHER = 0;

// The op's column-valid stat payload: only faces 0 and 2 carry information, so a
// gather leg sends half a tile and the multicast drops the last tile's face 3.
FORCE_INLINE void write_stat_payload(uint32_t src, uint64_t dst, uint32_t tile_bytes) {
    const uint32_t half = tile_bytes >> 1;
    noc_async_write(src, dst, tile_bytes >> 2);
    noc_async_write(src + half, dst + half, tile_bytes >> 2);
}

FORCE_INLINE void fill16(uint32_t addr, uint32_t bytes, uint16_t pattern) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    const uint32_t w = (static_cast<uint32_t>(pattern) << 16) | pattern;
    for (uint32_t i = 0, n = bytes / 4; i < n; ++i) {
        p[i] = w;
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(0);
    constexpr uint32_t ROWS_T = get_compile_time_arg_val(1);
    constexpr uint32_t G = get_compile_time_arg_val(2);  // group size (members)
    constexpr uint32_t VARIANT = get_compile_time_arg_val(3);
    constexpr uint32_t MCAST_CT_BASE = 4;
    constexpr uint32_t MCAST_RT_BASE = 8;
    constexpr auto mc = McastArgs<MCAST_CT_BASE, MCAST_RT_BASE>();
    constexpr auto dst_args = TensorAccessorArgs<mc.next_compile_time_args_offset()>();

    const uint32_t is_root = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t my_slot = get_arg_val<uint32_t>(2);
    const uint32_t root_x = get_arg_val<uint32_t>(3);
    const uint32_t root_y = get_arg_val<uint32_t>(4);
    const uint32_t core_id = get_arg_val<uint32_t>(5);
    const uint32_t n_cores = get_arg_val<uint32_t>(6);

    Noc noc;
    Semaphore<> gather_sem(SEM_GATHER);

    const uint32_t stat_tile_bytes = get_tile_size(cb_stat_gather);
    const uint32_t gather_base = get_write_ptr(cb_stat_gather);
    const uint32_t partial_src = get_write_ptr(cb_stat_partial);
    const uint32_t rstd_src = get_write_ptr(cb_rstd_send);
    const uint32_t rstd_dst = get_write_ptr(cb_rstd);  // group-uniform landing address
    const auto out_acc = TensorAccessor(dst_args, dst_addr, stat_tile_bytes);

    // POISON the landing buffer: bf16 0xFF80 == -inf. Any -inf reaching the output
    // means a receiver consumed its slot before the broadcast landed.
    fill16(rstd_dst, ROWS_T * stat_tile_bytes, 0xFF80);
    // The gather source never changes; its content is irrelevant to the gate.
    fill16(partial_src, ROWS_T * stat_tile_bytes, 0x3F80);

    // store_block: consume the landing tile(s) -> DRAM. THE consume point that the
    // gather-derived back-pressure argument depends on.
    auto store_block = [&](uint32_t b) {
        for (uint32_t r = 0; r < ROWS_T; ++r) {
            const uint32_t tile = (b * n_cores + core_id) * ROWS_T + r;
            noc_async_write_tile(tile, out_acc, rstd_dst + r * stat_tile_bytes);
        }
        noc_async_write_barrier();
    };

    auto gather_partials = [&](uint32_t b) {
        {
            MaybeDeviceZoneScope("wr_gather_issue");
            for (uint32_t r = 0; r < ROWS_T; ++r) {
                const uint32_t dst = gather_base + (r * G + my_slot) * stat_tile_bytes;
                write_stat_payload(
                    partial_src + r * stat_tile_bytes, get_noc_addr(root_x, root_y, dst), stat_tile_bytes);
            }
        }
        {
            MaybeDeviceZoneScope("wr_gather_barrier");
            noc_async_write_barrier();
        }
        if constexpr (G > 1) {
            if (!is_root) {
                gather_sem.up(noc, root_x, root_y, 1);
            }
        }
        if (is_root) {
            MaybeDeviceZoneScope("wr_gather_sem_wait");
            if constexpr (G > 1) {
                gather_sem.wait_min((b + 1) * (G - 1));
            }
        }
    };

    if (is_root) {
        auto sender = mc.sender(noc);
        for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
            gather_partials(b);
            // The block's broadcast value: bf16 1.0, 2.0, 4.0, 8.0 ... (exact).
            fill16(rstd_src, ROWS_T * stat_tile_bytes, static_cast<uint16_t>(0x3F80 + 0x80 * b));
            {
                MaybeDeviceZoneScope("wr_mcast_send");
                sender.send(rstd_src, rstd_dst, ROWS_T * stat_tile_bytes - (stat_tile_bytes >> 2));
            }
            store_block(b);
        }
    } else if constexpr (VARIANT == 2) {
        bench::NoInitReceiverPipe<mc.data_ready> receiver(noc);
        for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
            gather_partials(b);
            {
                MaybeDeviceZoneScope("wr_mcast_recv");
                receiver.receive();
            }
            store_block(b);
        }
    } else {
        auto receiver = mc.receiver(noc);
        for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
            gather_partials(b);
            {
                MaybeDeviceZoneScope("wr_mcast_recv");
                receiver.receive();
            }
            store_block(b);
        }
    }
}
