// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B47b XPUSH comm core (BRISC), one per (direction, lane) over the 4-chip group line. Waits for the 8 workers (local
// semaphore 0), then sends this lane's packets of the chip's 12288 T B x region (NPX = 3 T packets of 4096 B, packet p
// goes to page pos * NPX + p of the peers' x landing) with ONE chip-multicast write each (start distance 1, range ND),
// the route words (64 T B, lane 0 only) to bytes pos * 64 T of the route landing, then ONE multicast atomic increment
// (value = packets of this lane) of the per-source flag word flag[pos] on the flag core of every peer. The core with
// OWN = 1 also copies the own slot locally and increments the own flag. The program does not wait for any peer. CT: 0
// DIR, 1 ND (peers in this direction), 2 LANE, 3 LANES, 4 CB_BUF, 5 NPX, 6 ARGBASE, 7 NH, 8 OWN, 9 T, then
// TensorAccessorArgs(xland), (rland), (route). Common: xland addr, rland addr, route addr. RT: pos, fx, fy, flag addr
// 0..3.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

using namespace tt::tt_fabric::linear::experimental;

void kernel_main() {
    constexpr uint32_t DIR = get_compile_time_arg_val(0);
    constexpr uint32_t ND = get_compile_time_arg_val(1);
    constexpr uint32_t LANE = get_compile_time_arg_val(2);
    constexpr uint32_t LANES = get_compile_time_arg_val(3);
    constexpr uint32_t CB_BUF = get_compile_time_arg_val(4);
    constexpr uint32_t NPX = get_compile_time_arg_val(5);
    constexpr uint32_t ARGBASE = get_compile_time_arg_val(6);
    constexpr uint32_t NH = get_compile_time_arg_val(7);
    constexpr uint32_t OWN = get_compile_time_arg_val(8);
    constexpr uint32_t T = get_compile_time_arg_val(9);
    constexpr auto xl_args = TensorAccessorArgs<10, 0>();
    constexpr auto rl_args =
        TensorAccessorArgs<xl_args.next_compile_time_args_offset(), xl_args.next_common_runtime_args_offset()>();
    constexpr auto rs_args =
        TensorAccessorArgs<rl_args.next_compile_time_args_offset(), rl_args.next_common_runtime_args_offset()>();
    constexpr uint32_t PKT = 4096, RTB = 64 * T, SRCB = 12288 * T;
    const auto xl = TensorAccessor(xl_args, get_common_arg_val<uint32_t>(0), PKT);
    const auto rl = TensorAccessor(rl_args, get_common_arg_val<uint32_t>(1), 64 * 4 * T);
    const auto rs = TensorAccessor(rs_args, get_common_arg_val<uint32_t>(2), RTB);
    const uint32_t pos = get_arg_val<uint32_t>(0);
    const uint32_t fx = get_arg_val<uint32_t>(1), fy = get_arg_val<uint32_t>(2);
    const uint32_t flag_c = get_arg_val<uint32_t>(3 + pos);
    const uint64_t flag_noc = get_noc_addr(fx, fy, flag_c);

    size_t arg_idx = ARGBASE;
    auto mgr = tt::tt_fabric::RoutingPlaneConnectionManager::build_from_args<
        tt::tt_fabric::RoutingPlaneConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(arg_idx, 1);
    auto* sender = &mgr.get(0).sender;
    PacketHeaderPool::reset();
    volatile PACKET_HEADER_TYPE* hdr[NH];
    for (uint32_t i = 0; i < NH; ++i) {
        hdr[i] = PacketHeaderPool::allocate_header(1);
    }
    const uint32_t buf = get_write_ptr(CB_BUF);
    const uint32_t rbuf = buf + SRCB;  // route staging
    constexpr bool HAS_ROUTE = (LANE == 0);
    {
        DeviceZoneScopedN("XC_WAIT");
        volatile tt_l1_ptr uint32_t* ssem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        noc_semaphore_wait_min(ssem, 8);
        noc_semaphore_set(ssem, 0);
    }
    if constexpr (HAS_ROUTE) {
        noc_async_read(rs.get_noc_addr(0), rbuf, RTB);
        noc_async_read_barrier();
    }
    mgr.open_finish();
    uint32_t n = 0, cnt = 0;
    {
        DeviceZoneScopedN("XC_SEND");
        for (uint32_t p = LANE; p < NPX; p += LANES) {
            volatile PACKET_HEADER_TYPE* h = hdr[n % NH];
            fabric_multicast_noc_unicast_write(
                sender,
                h,
                buf + p * PKT,
                PKT,
                tt::tt_fabric::NocUnicastCommandHeader{xl.get_noc_addr(pos * NPX + p)},
                static_cast<uint8_t>(1),
                static_cast<uint8_t>(ND));
            ++n;
            ++cnt;
            if ((n % NH) == 0) {
                noc_async_writes_flushed();
            }
        }
        if constexpr (HAS_ROUTE) {
            volatile PACKET_HEADER_TYPE* h = hdr[n % NH];
            fabric_multicast_noc_unicast_write(
                sender,
                h,
                rbuf,
                RTB,
                tt::tt_fabric::NocUnicastCommandHeader{rl.get_noc_addr(0, pos * RTB)},
                static_cast<uint8_t>(1),
                static_cast<uint8_t>(ND));
            ++n;
            ++cnt;
        }
        noc_async_writes_flushed();
        volatile PACKET_HEADER_TYPE* h = hdr[n % NH];
        fabric_multicast_noc_unicast_atomic_inc(
            sender,
            h,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{flag_noc, static_cast<uint16_t>(cnt), true},
            static_cast<uint8_t>(1),
            static_cast<uint8_t>(ND));
        ++n;
    }
    if constexpr (OWN == 1) {
        for (uint32_t p = LANE; p < NPX; p += LANES) {
            noc_async_write(buf + p * PKT, xl.get_noc_addr(pos * NPX + p), PKT);
        }
        if constexpr (HAS_ROUTE) {
            noc_async_write(rbuf, rl.get_noc_addr(0, pos * RTB), RTB);
        }
        noc_async_write_barrier();
        noc_semaphore_inc(flag_noc, cnt);
        noc_async_atomic_barrier();
    }
    mgr.close();
    noc_async_full_barrier();
}
