// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The pair's LSE exchange of PairSplit.merge_heads (tt/pair_split.py), on the chip's exchange core (BRISC): packs
// column 0 of rows 0-15 of this chip's LSE tiles ([1, B, heads, 1] fp32, head h in row h) into 64 bytes per user
// (head h at word h) and sends them to the pair partner (chip ^ 1, one hop) as one fabric packet: a scatter write
// into slot (n & 1) of the partner's exchange buffer fused with an increment of its arrival semaphore, n this chip's
// call count (a persistent global semaphore, never reset). The merge cores of the partner wait for arrival n + 1.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

using namespace tt::tt_fabric::linear::experimental;

void kernel_main() {
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(0);
    constexpr uint32_t users = get_compile_time_arg_val(1);           // B (<= 32)
    constexpr uint32_t lse_user_pages = get_compile_time_arg_val(2);  // LSE tiles per user
    constexpr bool lse_dram = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t fp32_tile = 4096;
    constexpr uint32_t face_bytes = 1024;  // fp32 face: 16 rows of 64 bytes
    constexpr uint32_t packed_row = 64;    // a user's 16 LSEs
    constexpr uint32_t packed_bytes = users * packed_row;
    constexpr uint32_t chunk = packed_bytes / 2;  // the scatter write's two halves
    static_assert(users >= 1 && users <= 32 && packed_bytes <= slot_bytes);

    size_t arg = 0;
    const uint32_t lse_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t slots_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t arrive_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t count_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t my_x = get_arg_val<uint32_t>(arg++);  // the exchange core (the same on every chip)
    const uint32_t my_y = get_arg_val<uint32_t>(arg++);

    auto connection = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg);
    connection.open_start();

    // Scratch (64-byte aligned): the packed rows, then face 0 of each user's LSE tile.
    cb_reserve_back(cb_scratch, 1);
    const uint32_t base = (get_write_ptr(cb_scratch) + 63) & ~63u;
    const uint32_t packed_l1 = base;
    const uint32_t faces_l1 = base + ((packed_bytes + 63) & ~63u);
    const InterleavedAddrGen<lse_dram> lse_gen = {.bank_base_address = lse_addr, .page_size = fp32_tile};
    for (uint32_t b = 0; b < users; b++) {
        noc_async_read(lse_gen.get_noc_addr(b * lse_user_pages, 0), faces_l1 + b * face_bytes, face_bytes);
    }

    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(count_addr);
    const uint32_t calls = count[0];
    const uint32_t slot = slots_addr + (calls & 1) * slot_bytes;

    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header = PacketHeaderPool::allocate_header();
    connection.open_finish();
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* faces = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(faces_l1);
    volatile tt_l1_ptr uint32_t* packed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(packed_l1);
    for (uint32_t b = 0; b < users; b++) {
        for (uint32_t h = 0; h < 16; h++) {
            packed[b * (packed_row / 4) + h] = faces[b * (face_bytes / 4) + h * 16];
        }
    }

    const uint64_t dst0 = get_noc_addr(my_x, my_y, slot);
    const uint64_t dst1 = get_noc_addr(my_x, my_y, slot + chunk);
    const uint64_t sem = get_noc_addr(my_x, my_y, arrive_addr);
    fabric_multicast_noc_fused_scatter_write_atomic_inc(
        &connection,
        header,
        packed_l1,
        packed_bytes,
        tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader(
            {dst0, dst1}, sem, {static_cast<uint16_t>(chunk)}, 1, true),
        1,
        1);
    count[0] = calls + 1;

    connection.close();
    noc_async_full_barrier();
    cb_push_back(cb_scratch, 1);
}
