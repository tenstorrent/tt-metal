// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Traffic-only fabric sender for the Blackhole NoC wedge reproducers: streams a fixed number of packets over one
// fabric connection (one link toward one neighbour chip). Nothing on the receiving chip waits for them: they land in
// scratch L1 / DRAM nobody reads, and atomic increments bump scratch words nobody polls. The only wait is the
// connection's own slot credit from the local fabric router (wait_for_empty_write_slot).
//
// CT arg 0, packet kind:
//   0  unicast writes into the destination core's L1 scratch
//   1  unicast writes into the destination chip's DRAM (round-robin over the banks: the receiving router's DRAM writes)
//   2  atomic increments only (header-only packets) into the destination core's L1 word
//   3  writes into L1, every 4th a fused write + atomic increment (as the MoE head's gather / reduce-scatter ports)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

using namespace tt::tt_fabric;

void kernel_main() {
    constexpr uint32_t kind = get_compile_time_arg_val(0);
    constexpr uint32_t NH = 4;  // packet headers in rotation (one is reused only after its send was flushed)
    size_t a = 0;
    const uint32_t n_packets = get_arg_val<uint32_t>(a++);
    const uint32_t pkt = get_arg_val<uint32_t>(a++);
    const uint32_t src = get_arg_val<uint32_t>(a++);
    const uint32_t dst_chip = get_arg_val<uint32_t>(a++);
    const uint32_t dst_mesh = get_arg_val<uint32_t>(a++);
    const uint32_t dst_x = get_arg_val<uint32_t>(a++);
    const uint32_t dst_y = get_arg_val<uint32_t>(a++);
    const uint32_t land = get_arg_val<uint32_t>(a++);  // L1 landing base on the destination core
    const uint32_t land_slots = get_arg_val<uint32_t>(a++);
    const uint32_t word = get_arg_val<uint32_t>(a++);  // L1 atomic word on the destination core
    const uint32_t dram = get_arg_val<uint32_t>(a++);  // bank-local DRAM landing base on the destination chip
    const uint32_t dram_region = get_arg_val<uint32_t>(a++);
    const uint32_t nbanks = get_arg_val<uint32_t>(a++);
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);

    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr[NH];
    for (uint32_t i = 0; i < NH; ++i) {
        hdr[i] = PacketHeaderPool::allocate_header();
        fabric_set_unicast_route(hdr[i], static_cast<uint16_t>(dst_chip), static_cast<uint16_t>(dst_mesh));
    }
    conn.open();
    const uint32_t dram_slots = dram_region / pkt;
    for (uint32_t i = 0; i < n_packets; ++i) {
        auto* h = hdr[i % NH];
        const uint64_t word_addr = get_noc_addr(dst_x, dst_y, word);
        if constexpr (kind == 2) {
            h->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{word_addr, 1, false});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_non_blocking_from_address(
                reinterpret_cast<uint32_t>(h), sizeof(PACKET_HEADER_TYPE));
        } else {
            uint64_t to;
            if constexpr (kind == 1) {
                to = get_noc_addr_from_bank_id<true>(i % nbanks, dram + ((i / nbanks) % dram_slots) * pkt);
            } else {
                to = get_noc_addr(dst_x, dst_y, land + (i % land_slots) * pkt);
            }
            if (kind == 3 && i % 4 == 3) {
                h->to_noc_fused_unicast_write_atomic_inc(
                    NocUnicastAtomicIncFusedCommandHeader{to, word_addr, 1, false}, pkt);
            } else {
                h->to_noc_unicast_write(NocUnicastCommandHeader{to}, pkt);
            }
            conn.wait_for_empty_write_slot();
            conn.send_payload_without_header_non_blocking_from_address(src, pkt);
            conn.send_payload_flush_non_blocking_from_address(
                reinterpret_cast<uint32_t>(h), sizeof(PACKET_HEADER_TYPE));
        }
        if (i % NH == NH - 1) {
            noc_async_writes_flushed();  // the headers in rotation are free again
        }
    }
    noc_async_writes_flushed();
    conn.close();
    noc_async_full_barrier();
}
