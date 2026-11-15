// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Rows-phase one-shot reduce, communication core (BRISC), one per (direction, link). Each reducer p of this link stages
// the payload of block b (the T-row tiles of the owner chip at position b) in this core's CB_STG and increments the
// local semaphore (id 0); this core sends every staged chunk as ONE fabric packet (fused write + semaphore increment)
// to reducer p of the owner chip, d hops away (forward: owner r+d, backward: owner r-d), slot = r if r < owner else r
// - 1.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

using namespace tt::tt_fabric::linear::experimental;

void kernel_main() {
    constexpr uint32_t DIR = get_compile_time_arg_val(0);  // 0 forward (owner r+d), 1 backward (owner r-d)
    constexpr uint32_t ND = get_compile_time_arg_val(1);   // distances served (forward 3-r, backward r)
    constexpr uint32_t PN = get_compile_time_arg_val(2);   // reducers served by this core
    constexpr uint32_t P0 = get_compile_time_arg_val(3);
    constexpr uint32_t CB_STG = get_compile_time_arg_val(4);
    constexpr uint32_t CHUNK = get_compile_time_arg_val(5);
    constexpr uint32_t ARGBASE = get_compile_time_arg_val(6);
    constexpr uint32_t STOP = get_compile_time_arg_val(7);  // debug: 1 = wait for stagings only, send nothing
    constexpr uint32_t NB = 4;

    const uint32_t recv_base = get_common_arg_val<uint32_t>(0);
    const uint32_t gsem = get_common_arg_val<uint32_t>(1);
    const uint32_t r = get_arg_val<uint32_t>(0);

    size_t arg_idx = ARGBASE;
    auto mgr = tt::tt_fabric::RoutingPlaneConnectionManager::build_from_args<
        tt::tt_fabric::RoutingPlaneConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(arg_idx, 1);
    auto* sender = &mgr.get(0).sender;
    PacketHeaderPool::reset();
    volatile PACKET_HEADER_TYPE* hdr0 = PacketHeaderPool::allocate_header(1);
    volatile PACKET_HEADER_TYPE* hdr1 = PacketHeaderPool::allocate_header(1);
    {
        DeviceZoneScopedN("C_OPEN");
        mgr.open_finish();
    }
    const uint32_t stg = get_write_ptr(CB_STG);
    uint32_t k = 0;
    {
        // one local semaphore (id 0) counts the staged chunks of the PN reducers of this link
        volatile tt_l1_ptr uint32_t* ssem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        DeviceZoneScopedN("C_WAIT");
        noc_semaphore_wait_min(ssem, PN);
        noc_semaphore_set(ssem, 0);
    }
    for (uint32_t pi = 0; pi < PN; ++pi) {
        const uint32_t p = P0 + pi;
        if constexpr (STOP == 0) {
            const uint32_t rx = get_arg_val<uint32_t>(1 + 2 * p);
            const uint32_t ry = get_arg_val<uint32_t>(2 + 2 * p);
            DeviceZoneScopedN("C_SEND");
            for (uint32_t di = 0; di < ND; ++di) {
                const uint32_t d = ND - di;  // farthest first
                const uint32_t owner = (DIR == 0) ? (r + d) : (r - d);
                const uint32_t slot = (r < owner) ? r : (r - 1);
                const uint64_t dst = get_noc_addr(rx, ry, recv_base + slot * CHUNK);
                const uint64_t sem = get_noc_addr(rx, ry, gsem);
                volatile PACKET_HEADER_TYPE* h = (k & 1) ? hdr1 : hdr0;
                fabric_unicast_noc_fused_unicast_with_atomic_inc(
                    sender,
                    h,
                    stg + (p * NB + owner) * CHUNK,
                    CHUNK,
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem, 1, true},
                    static_cast<uint8_t>(d));
                noc_async_writes_flushed();
                ++k;
            }
        }
    }
    {
        DeviceZoneScopedN("C_CLOSE");
        mgr.close();
        noc_async_full_barrier();
    }
    noc_async_full_barrier();
}
