// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// One direction (BRISC: forward, NCRISC: backward) of one link of the row all-gather that feeds the fused reduce
// (ReduceExpand.call_fabric). This sender handles `cores` reduce cores starting at first_core: their compact tiles
// (core i owns tiles [2i, 2i + 2) of this chip's [1, 1, 32, hidden / 8] compact rows) are read into L1, then each
// core's two tiles go to every chip `range` hops away in this direction as one fabric multicast packet: a scatter
// write into slots (tile, this chip) of that core's persistent gather buffer plus an increment of its global
// semaphore. The forward sender also writes this chip's own slots and increments the local semaphores, so every
// reduce core waits for `chips` increments.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

using namespace tt::tt_fabric::linear::experimental;

void kernel_main() {
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(0);
    constexpr uint32_t range = get_compile_time_arg_val(1);
    constexpr uint32_t cores = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_core = get_compile_time_arg_val(3);  // 2
    constexpr uint32_t chips = get_compile_time_arg_val(4);
    constexpr bool local_copy = get_compile_time_arg_val(5) == 1;
    constexpr bool src_dram = get_compile_time_arg_val(6) == 1;
    // A producer in the same launch: wait until semaphore wait_sem counts wait_count (its cores' "written").
    constexpr uint32_t wait_sem = get_compile_time_arg_val(7);
    constexpr uint32_t wait_count = get_compile_time_arg_val(8);
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t core_bytes = tiles_per_core * tile_bytes;
    static_assert(tiles_per_core == 2, "one scatter packet carries a core's two tiles");

    size_t arg = 0;
    const uint32_t src_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t gather_addr = get_arg_val<uint32_t>(arg++);  // this phase's slots on every reduce core
    const uint32_t sem_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t my_chip = get_arg_val<uint32_t>(arg++);
    const uint32_t first_core = get_arg_val<uint32_t>(arg++);
    uint32_t core_x[cores], core_y[cores];
    for (uint32_t i = 0; i < cores; i++) {
        core_x[i] = get_arg_val<uint32_t>(arg++);
        core_y[i] = get_arg_val<uint32_t>(arg++);
    }

    auto connection = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg);
    connection.open_start();
    if constexpr (wait_count > 0) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(wait_sem)), wait_count);
    }

    // This chip's compact tiles, core-major (core i's tiles at scratch + i * core_bytes).
    cb_reserve_back(cb_scratch, cores * tiles_per_core);
    const uint32_t scratch = get_write_ptr(cb_scratch);
    const InterleavedAddrGen<src_dram> src_gen = {.bank_base_address = src_addr, .page_size = tile_bytes};
    for (uint32_t t = 0; t < cores * tiles_per_core; t++) {
        noc_async_read(src_gen.get_noc_addr(first_core * tiles_per_core + t, 0), scratch + t * tile_bytes, tile_bytes);
    }

    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* headers[cores];
    for (uint32_t i = 0; i < cores; i++) {
        headers[i] = PacketHeaderPool::allocate_header();
    }
    connection.open_finish();
    noc_async_read_barrier();

    // Slot (tile j, chip c) of a core's gather buffer: gather_addr + (j * chips + c) * tile_bytes.
    for (uint32_t i = 0; i < cores; i++) {
        const uint64_t dst0 = get_noc_addr(core_x[i], core_y[i], gather_addr + (0 * chips + my_chip) * tile_bytes);
        const uint64_t dst1 = get_noc_addr(core_x[i], core_y[i], gather_addr + (1 * chips + my_chip) * tile_bytes);
        const uint64_t sem = get_noc_addr(core_x[i], core_y[i], sem_addr);
        fabric_multicast_noc_fused_scatter_write_atomic_inc(
            &connection,
            headers[i],
            scratch + i * core_bytes,
            core_bytes,
            tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader(
                {dst0, dst1}, sem, {static_cast<uint16_t>(tile_bytes)}, 1, true),
            1,
            range);
    }

    if constexpr (local_copy) {
        for (uint32_t i = 0; i < cores; i++) {
            for (uint32_t j = 0; j < tiles_per_core; j++) {
                noc_async_write(
                    scratch + i * core_bytes + j * tile_bytes,
                    get_noc_addr(core_x[i], core_y[i], gather_addr + (j * chips + my_chip) * tile_bytes),
                    tile_bytes);
            }
        }
        noc_async_write_barrier();
        for (uint32_t i = 0; i < cores; i++) {
            noc_semaphore_inc(get_noc_addr(core_x[i], core_y[i], sem_addr), 1);
        }
        noc_async_atomic_barrier();
    }

    connection.close();
    noc_async_full_barrier();
    cb_push_back(cb_scratch, cores * tiles_per_core);
}
