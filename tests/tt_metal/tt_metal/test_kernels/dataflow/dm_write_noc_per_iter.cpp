// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Interleaved variant of dm_packet_walk: each iteration does memory write + NOC issue
// together, avoiding the instruction-cache warm-up effects of two separate loops.
//
// Write mode (compile-time arg 6):
//   0 = cached write + flush_l2  (write to cache 0-4MB, then flush to TL1/SRAM)
//   1 = uncached direct SRAM write (write to addr + 0x400000, bypasses cache)
//
// Barrier mode (compile-time arg 7):
//   0 = single barrier after the full loop (current default)
//   1 = barrier after every NOC issue (serialised, measures per-issue round-trip)
//
// results_l1_addr (compile-time arg 8):
//   L1 address on this core where timing results are written for host readback.
//   Three uint32_t words: [0 (unused), 0 (unused), avg_cycles]
//   Written via non-cacheable alias so host sees them immediately after kernel exits.
//
// update_noc_addr (compile-time arg 9):
//   0 = no address register writes between NOC issues (current default)
//   1 = call update_noc_cmdbuf_addrs(src, dst) before every issue (same values, measures register write overhead)

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "risc_common.h"
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"
#include <cstdint>

static inline void setup_noc_cmdbuf(
    uint32_t src_addr, uint32_t dst_addr, uint32_t dst_noc_x, uint32_t dst_noc_y, uint32_t size_bytes) {
    const uint64_t src_noc_xy = NOC_XY_COORD(my_x[noc_index], my_y[noc_index]);
    const uint64_t dst_noc_xy = NOC_XY_COORD(dst_noc_x, dst_noc_y);
    reset_reg_cmdbuf();
    setup_as_copy_reg_cmdbuf(/*wr=*/true, /*mcast=*/false, {0}, /*posted=*/true);
    setup_vcs_reg_cmdbuf(/*wr=*/true);
    set_src_reg_cmdbuf(src_addr, src_noc_xy);
    set_dest_reg_cmdbuf(dst_addr, dst_noc_xy);
    set_len_reg_cmdbuf(size_bytes);
}

static inline void update_noc_cmdbuf_addrs(uint32_t src_addr, uint32_t dst_addr) {
    set_src_reg_cmdbuf(src_addr);
    set_dest_reg_cmdbuf(dst_addr);
}

void kernel_main() {
    constexpr uint32_t src_base_l1_addr = get_compile_time_arg_val(0);
    constexpr uint32_t dst_base_l1_addr = get_compile_time_arg_val(1);
    constexpr uint32_t num_iterations = get_compile_time_arg_val(2);
    constexpr uint32_t packet_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t stride_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t packed_dst_core = get_compile_time_arg_val(5);
    constexpr uint32_t write_mode = get_compile_time_arg_val(6);
    constexpr uint32_t barrier_mode = get_compile_time_arg_val(7);
    constexpr uint32_t results_l1_addr = get_compile_time_arg_val(8);
    constexpr uint32_t update_noc_addr = get_compile_time_arg_val(9);

    constexpr uint32_t kPacketSizeBytes = 3 * 64;
    constexpr uint32_t kDmaTypeWrite = 1;
    constexpr uint32_t kEnDataInDescWriteToDst = 0;
    constexpr uint32_t kPacketTarget3b = 3;
    constexpr uint32_t kCompletionSw2b = 1;
    constexpr uint64_t kPacketDummySrcAddrBase = 0x100000000ULL;
    constexpr uint64_t kPacketDummyDstAddrBase = 0x200000000ULL;
    const uint32_t transfer_size_19b = kPacketSizeBytes & 0x7FFFF;

    (void)packet_size_bytes;
    (void)stride_bytes;

    const uint32_t src_addr = src_base_l1_addr;
    const uint32_t dst_addr = dst_base_l1_addr;
    const uint32_t dst_noc_x = packed_dst_core >> 16;
    const uint32_t dst_noc_y = packed_dst_core & 0xFFFF;

    constexpr uint32_t uncached_offset = MEMORY_PORT_NONCACHEABLE_MEM_PORT_MEM_BASE_ADDR;

    uint64_t packet_src_addr = kPacketDummySrcAddrBase;
    uint64_t packet_dst_addr = kPacketDummyDstAddrBase;

    const uint32_t req_word0 =
        ((kDmaTypeWrite & 0x1) << 8) | ((kEnDataInDescWriteToDst & 0x1) << 9) | (transfer_size_19b << 10);
    const uint32_t req_word1 = (kPacketTarget3b & 0x7) | ((kCompletionSw2b & 0x3) << 8);
    const uint64_t hdr = static_cast<uint64_t>(req_word0) | (static_cast<uint64_t>(req_word1) << 32);

    constexpr uint64_t kMockPayload[5] = {3, 4, 5, 6, 7};

    setup_noc_cmdbuf(src_addr, dst_addr, dst_noc_x, dst_noc_y, kPacketSizeBytes);

    uint64_t t0 = get_timestamp();

    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
        if constexpr (write_mode == 0) {
            volatile tt_l1_ptr uint64_t* pkt0 = reinterpret_cast<volatile tt_l1_ptr uint64_t*>(src_addr);
            pkt0[0] = hdr;
            pkt0[1] = packet_src_addr;
            pkt0[2] = packet_dst_addr;
            pkt0[3] = kMockPayload[0];
            pkt0[4] = kMockPayload[1];
            pkt0[5] = kMockPayload[2];
            pkt0[6] = kMockPayload[3];
            pkt0[7] = kMockPayload[4];

            volatile tt_l1_ptr uint64_t* pkt1 = reinterpret_cast<volatile tt_l1_ptr uint64_t*>(src_addr + 64);
            pkt1[0] = hdr;
            pkt1[1] = packet_src_addr;
            pkt1[2] = packet_dst_addr;
            pkt1[3] = kMockPayload[0];
            pkt1[4] = kMockPayload[1];
            pkt1[5] = kMockPayload[2];
            pkt1[6] = kMockPayload[3];
            pkt1[7] = kMockPayload[4];

            volatile tt_l1_ptr uint64_t* pkt2 = reinterpret_cast<volatile tt_l1_ptr uint64_t*>(src_addr + 128);
            pkt2[0] = hdr;
            pkt2[1] = packet_src_addr;
            pkt2[2] = packet_dst_addr;
            pkt2[3] = kMockPayload[0];
            pkt2[4] = kMockPayload[1];
            pkt2[5] = kMockPayload[2];
            pkt2[6] = kMockPayload[3];
            pkt2[7] = kMockPayload[4];

            flush_l2_cache_range(src_addr, kPacketSizeBytes);
        } else {
            volatile tt_l1_ptr uint64_t* pkt0 =
                reinterpret_cast<volatile tt_l1_ptr uint64_t*>(src_addr + uncached_offset);
            pkt0[0] = hdr;
            pkt0[1] = packet_src_addr;
            pkt0[2] = packet_dst_addr;
            pkt0[3] = kMockPayload[0];
            pkt0[4] = kMockPayload[1];
            pkt0[5] = kMockPayload[2];
            pkt0[6] = kMockPayload[3];
            pkt0[7] = kMockPayload[4];

            volatile tt_l1_ptr uint64_t* pkt1 =
                reinterpret_cast<volatile tt_l1_ptr uint64_t*>(src_addr + 64 + uncached_offset);
            pkt1[0] = hdr;
            pkt1[1] = packet_src_addr;
            pkt1[2] = packet_dst_addr;
            pkt1[3] = kMockPayload[0];
            pkt1[4] = kMockPayload[1];
            pkt1[5] = kMockPayload[2];
            pkt1[6] = kMockPayload[3];
            pkt1[7] = kMockPayload[4];

            volatile tt_l1_ptr uint64_t* pkt2 =
                reinterpret_cast<volatile tt_l1_ptr uint64_t*>(src_addr + 128 + uncached_offset);
            pkt2[0] = hdr;
            pkt2[1] = packet_src_addr;
            pkt2[2] = packet_dst_addr;
            pkt2[3] = kMockPayload[0];
            pkt2[4] = kMockPayload[1];
            pkt2[5] = kMockPayload[2];
            pkt2[6] = kMockPayload[3];
            pkt2[7] = kMockPayload[4];
        }

        if constexpr (update_noc_addr == 1) {
            update_noc_cmdbuf_addrs(src_addr, dst_addr);
        }
        issue_write_reg_cmdbuf();
        if constexpr (barrier_mode == 1) {
            while (!noc_writes_sent_reg_cmdbuf()) {
            }
        }

        packet_src_addr += transfer_size_19b;
        packet_dst_addr += transfer_size_19b;
    }

    if constexpr (barrier_mode == 0) {
        while (!noc_writes_sent_reg_cmdbuf()) {
        }
    }

    uint64_t t1 = get_timestamp();

    const uint32_t avg_cycles = static_cast<uint32_t>((t1 - t0) / num_iterations);

    // Write results to L1 via non-cacheable alias so host sees them immediately after kernel exits.
    // Layout matches two-phase kernel: [write_avg, noc_avg, total_avg].
    // Per-iter has no separate write/noc phases, so words 0 and 1 are 0; word 2 is the combined avg.
    volatile tt_l1_ptr uint32_t* results =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(results_l1_addr + uncached_offset);
    results[0] = 0;
    results[1] = 0;
    results[2] = avg_cycles;

    const char* write_str = (write_mode == 0) ? "CACHE_L2_FLUSH" : "DIRECT_SRAM";
    const char* barrier_str = (barrier_mode == 0) ? "BARRIER_END" : "BARRIER_PER_ITER";
    const char* addr_str = (update_noc_addr == 0) ? "" : "_UPDATE_NOC_ADDR";
    DPRINT << write_str << "_PER_ITER_" << barrier_str << addr_str << " iters=" << num_iterations
           << " avg=" << avg_cycles << ENDL();
}
