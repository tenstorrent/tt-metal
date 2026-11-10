// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Positive control: an intentionally UNFENCED cross-chip L1 writer. After pre_delay cycles it sprays a constant
// pattern over the fabric into [target, target + bytes_per_core) of every listed core of ONE downstream chip, round
// after round, for `rounds` rounds, without any handshake with that chip's program.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using namespace tt::tt_fabric;

inline void route(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0))
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
    (void)chip_id;
    (void)mesh_id;
    (void)fabric_set_unicast_route<false>(
        reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(hdr), static_cast<uint16_t>(1));
#endif
}

void kernel_main() {
    size_t a = 0;
    const uint16_t mesh_id = get_arg_val<uint32_t>(a++);
    const uint16_t chip_id = get_arg_val<uint32_t>(a++);
    const uint32_t src = get_write_ptr(get_arg_val<uint32_t>(a++));  // pattern buffer: this core's CB
    const uint32_t target = get_arg_val<uint32_t>(a++);
    const uint32_t bytes_per_core = get_arg_val<uint32_t>(a++);
    const uint32_t chunk = get_arg_val<uint32_t>(a++);
    const uint32_t rounds = get_arg_val<uint32_t>(a++);
    const uint32_t pre_delay = get_arg_val<uint32_t>(a++);
    const uint32_t pattern = get_arg_val<uint32_t>(a++);
    const uint32_t num_cores = get_arg_val<uint32_t>(a++);
    const size_t cores_idx = a;
    a += num_cores;
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
    route(hdr, chip_id, mesh_id);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    for (uint32_t i = 0; i < chunk / 4; ++i) {
        s[i] = pattern;
    }
    conn.open();
    if (pre_delay > 0) {
        riscv_wait(pre_delay);
    }
    for (uint32_t r = 0; r < rounds; ++r) {
        for (uint32_t off = 0; off < bytes_per_core; off += chunk) {
            const uint32_t n = (bytes_per_core - off) < chunk ? (bytes_per_core - off) : chunk;
            for (uint32_t c = 0; c < num_cores; ++c) {
                const uint32_t cw = get_arg_val<uint32_t>(cores_idx + c);
                const uint64_t dst = get_noc_addr(cw & 0xFFFF, cw >> 16, target + off);
                hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, n);
                conn.wait_for_empty_write_slot();
                conn.send_payload_without_header_non_blocking_from_address(src, n);
                conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
            }
        }
    }
    conn.close();
}
