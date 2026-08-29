// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Can a fabric unicast write land in HOST MEMORY, sent from an ethernet core?
//
// This is the one unproven link in the tt-coremon eth-aggregator transport.
// Each piece is separately established; nobody has joined them:
//   - the fabric receiver passes header.noc_address straight into
//     noc_async_write_one_packet_with_trid, unmasked and unvalidated
//     (fabric_edm_packet_transmission.hpp:154)
//   - a kernel writing host memory over PCIe is shipped and works
//     (cq_realtime_profiler_push.cpp)
//   - an eth core can be a fabric client via the runtime-arg path
//     (edm_fabric_worker_adapters.hpp: "VC2 (TENSIX or ETH)")
//
// THE TRAP: the EDM issues its local write on NOC1
// (edm_fabric_utils.hpp: edm_to_local_chip_noc = 1) and the PCIe tile has a
// different XY encoding per NOC. A NOC0-encoded address silently lands on the
// wrong tile: no fault, no error, the data simply goes nowhere -- the same
// failure shape as mis-addressed DRAM NIU registers. The host therefore passes
// a fully-formed destination and we do not re-derive it here.
//
// Runtime args (before the appended fabric-connection args):
//   0: dest_noc_addr_lo   destination on the RECEIVING chip: PCIe tile | host offset
//   1: dest_noc_addr_hi
//   2: payload_l1_addr    scratch in this core's L1
//   3: payload_bytes
//   4: num_sends
//   5: send_interval_cyc  idle cycles between sends -- see note below
//   6: unicast_hops       fabric distance to the receiving chip

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using namespace tt::tt_fabric;

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t dest_lo = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dest_hi = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t payload_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t payload_bytes = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sends = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t send_interval_cyc = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t unicast_hops = get_arg_val<uint32_t>(arg_idx++);

    // Fully-formed by the host, including the NOC1-encoded PCIe tile XY. Do NOT
    // recompute it here: NOC_X_PHYS_COORD() resolves against this kernel's own
    // noc_index, which is not the NOC the EDM writes on.
    const uint64_t dest_noc_addr = ((uint64_t)dest_hi << 32) | (uint64_t)dest_lo;

    // IDLE_ETH takes the runtime-arg connection path, not the Tensix L1 table.
    // Progress markers in our own L1, 64 B above the payload. A local write that
    // cannot fail, so the host can tell "kernel never ran" from "connection
    // never opened" from "sent, but the bytes went nowhere".
    volatile tt_l1_ptr uint32_t* dbg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(payload_addr + 64);
    dbg[0] = 0xA11E0000u;  // reached kernel_main
    dbg[1] = 0;            // sends completed
    dbg[2] = (uint32_t)(dest_noc_addr & 0xFFFFFFFFu);
    dbg[3] = (uint32_t)(dest_noc_addr >> 32);

    auto sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::IDLE_ETH>(arg_idx);
    sender.open();
    dbg[0] = 0x09E00000u;  // fabric connection opened

    auto* packet_header = PacketHeaderPool::allocate_header();
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(payload_addr);

    for (uint32_t n = 0; n < num_sends; n++) {
        // Stamp a sentinel + sequence so the host can distinguish a real arrival
        // from leftover memory, and can watch the stream advance rather than
        // just seeing one write land.
        p[0] = 0x77CAFE00u | (n & 0xFFu);
        p[1] = n;

        sender.wait_for_empty_write_slot();
        fabric_set_unicast_route<false>((LowLatencyPacketHeader*)packet_header, (uint8_t)unicast_hops);
        packet_header->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dest_noc_addr}, payload_bytes);

        sender.send_payload_without_header_non_blocking_from_address(payload_addr, payload_bytes);
        sender.send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
        noc_async_writes_flushed();
        dbg[1] = n + 1;

        // IDLE, do not spin. A telemetry stream that keeps the core hot raises
        // AICLK, which changes both the power envelope and the thing being
        // measured. Real duty cycle here is ~1 KB per 100 ms, so the core should
        // be asleep essentially all the time.
        if (send_interval_cyc) {
            riscv_wait(send_interval_cyc);
        }
    }

    noc_async_write_barrier();
    sender.close();
}
