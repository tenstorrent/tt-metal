// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B29 q push (flag HY4_B29_QPUSH): replaces the all_gather_async of q. Each chip sends its local q rows [8 heads x S
// rows of 576 bf16, 1152 B per row, page r = h*S + s] to the 7 peers as 2-row packets (2304 B) landing at DRAM page
// (c*4S + r/2) of the persistent landing tensor, each packet with a fused atomic increment of the per-source flag word
// flag[c] on core F of the receiving chip. The program does NOT wait for any peer: the split SDPA reader waits for
// flag[j] (j = owner column of its row) before reading q.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

using namespace tt::tt_fabric::linear::experimental;

void kernel_main() {
    constexpr uint32_t DIR = get_compile_time_arg_val(0);
    constexpr uint32_t ND = get_compile_time_arg_val(1);  // peers in this direction (4 fwd, 3 bwd)
    constexpr uint32_t NR = get_compile_time_arg_val(2);  // pages (q rows / tiles) per source
    constexpr uint32_t LANE = get_compile_time_arg_val(3);
    constexpr uint32_t LANES = get_compile_time_arg_val(4);
    constexpr uint32_t CB_PKT = get_compile_time_arg_val(5);
    constexpr uint32_t ARGBASE = get_compile_time_arg_val(6);
    constexpr uint32_t NH = get_compile_time_arg_val(7);
    constexpr uint32_t RP = get_compile_time_arg_val(8);  // q rows per packet (2 or 3; max fabric payload 4352 B)
    constexpr uint32_t ROWB = get_compile_time_arg_val(9);  // page bytes
    constexpr auto q_args = TensorAccessorArgs<10, 0>();
    constexpr auto l_args =
        TensorAccessorArgs<q_args.next_compile_time_args_offset(), q_args.next_common_runtime_args_offset()>();
    constexpr uint32_t PKT = RP * ROWB, NP = (NR + RP - 1) / RP, RING = 8;
    const uint32_t q_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t land_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t c = get_arg_val<uint32_t>(0);   // ring position of this chip
    const uint32_t fx = get_arg_val<uint32_t>(1);  // physical coords of the flag core
    const uint32_t fy = get_arg_val<uint32_t>(2);
    const uint32_t flag_c = get_arg_val<uint32_t>(3 + c);  // this chip's slot flag address (same on every chip)
    const auto qa = TensorAccessor(q_args, q_addr, ROWB);
    const auto la = TensorAccessor(l_args, land_addr, PKT);
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
    const uint32_t pkt = get_write_ptr(CB_PKT);
    uint32_t k = 0;
#ifdef QF_WAIT
    {  // B47b (HY4_B47B_QFOLD): this program is a part of the fused q-prep program: wait until all QF_WAIT writer cores
       // have finished their q rows in DRAM (each increments local semaphore 0 after its write barrier)
        DeviceZoneScopedN("QF_WAIT");
        volatile tt_l1_ptr uint32_t* ssem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        noc_semaphore_wait_min(ssem, QF_WAIT);
        noc_semaphore_set(ssem, 0);
    }
#endif
    {
        DeviceZoneScopedN("QP_READ");
        for (uint32_t p = LANE; p < NP; p += LANES) {
            for (uint32_t i = 0; i < RP && RP * p + i < NR; ++i) {
                noc_async_read(qa.get_noc_addr(RP * p + i), pkt + k * PKT + i * ROWB, ROWB);
            }
            ++k;
        }
        noc_async_read_barrier();
    }
    mgr.open_finish();
    uint32_t n = 0;
    {
        DeviceZoneScopedN("QP_SEND");
#ifdef QP_MC
        // B47b (HY4_B47B_PUSH_MC=1): ONE chip-multicast write packet per q packet (start distance 1, range ND) instead
        // of ND unicast packets, then ONE multicast atomic increment (value = packets of this lane) of the per-source
        // flag after the last write. The fabric receivers flush their pending writes before an atomic inc (the order
        // write -> inc is the one the stock all_gather_async relies on).
        k = 0;
        for (uint32_t p = LANE; p < NP; p += LANES) {
            const uint64_t dst = la.get_noc_addr(c * NP + p);
            volatile PACKET_HEADER_TYPE* h = hdr[n % NH];
            fabric_multicast_noc_unicast_write(
                sender,
                h,
                pkt + k * PKT,
                (RP * p + RP <= NR) ? PKT : (NR - RP * p) * ROWB,
                tt::tt_fabric::NocUnicastCommandHeader{dst},
                static_cast<uint8_t>(1),
                static_cast<uint8_t>(ND));
            ++n;
            if ((n % NH) == 0) {
                noc_async_writes_flushed();
            }
            ++k;
        }
        {
            noc_async_writes_flushed();
            volatile PACKET_HEADER_TYPE* h = hdr[n % NH];
            fabric_multicast_noc_unicast_atomic_inc(
                sender,
                h,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{flag_noc, static_cast<uint16_t>(k), true},
                static_cast<uint8_t>(1),
                static_cast<uint8_t>(ND));
            ++n;
        }
#else
        for (uint32_t di = 0; di < ND; ++di) {
            const uint32_t d = ND - di;  // farthest first
            k = 0;
            for (uint32_t p = LANE; p < NP; p += LANES) {
                const uint64_t dst = la.get_noc_addr(c * NP + p);
                volatile PACKET_HEADER_TYPE* h = hdr[n % NH];
                fabric_unicast_noc_fused_unicast_with_atomic_inc(
                    sender,
                    h,
                    pkt + k * PKT,
                    (RP * p + RP <= NR) ? PKT : (NR - RP * p) * ROWB,
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, flag_noc, 1, true},
                    static_cast<uint8_t>(d));
                ++n;
                if ((n % NH) == 0) {
                    noc_async_writes_flushed();
                }
                ++k;
            }
        }
#endif
        noc_async_writes_flushed();
    }
    if constexpr (DIR == 0) {  // own slot: local copy + flag increments (the readers of this chip treat all 8 slots
                               // alike)
        k = 0;
        for (uint32_t p = LANE; p < NP; p += LANES) {
            noc_async_write(
                pkt + k * PKT, la.get_noc_addr(c * NP + p), (RP * p + RP <= NR) ? PKT : (NR - RP * p) * ROWB);
            ++k;
        }
        noc_async_write_barrier();
        for (uint32_t p = LANE; p < NP; p += LANES) {
            noc_semaphore_inc(flag_noc, 1);
        }
        noc_async_atomic_barrier();
    }
    mgr.close();
    noc_async_full_barrier();
}
