// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Fabric-over-NoC "connection" for the fabric test kernels.
//
// On fabric-over-NoC architectures (Horizon) there is no router to connect to: a device reaches every peer
// directly over the NoC crossbar. This type stands in for a router connection to one peer so that the test
// kernels keep a single send path. It exposes the subset of the WorkerToFabricEdmSender interface that the
// test kernels use; "sending" a packet header executes that header's NoC command against the bound peer
// through the a2a fabric APIs. The header is only the test kernel's description of the command -- it never
// leaves the core.

#if defined(ARCH_QUASAR) && defined(IP_VARIANT_HORIZON)
#define FABRIC_TEST_HAS_NOC_TRANSPORT 1
#endif

#ifdef FABRIC_TEST_HAS_NOC_TRANSPORT

#include <cstdint>

#include "api/debug/assert.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/a2a/api.h"

namespace tt::tt_fabric::fabric_tests {

struct FabricOverNocConnection {
    template <ProgrammableCoreType core_type = ProgrammableCoreType::TENSIX>
    static FabricOverNocConnection build_from_args(size_t& arg_idx) {
        FabricOverNocConnection conn;
        conn.dst_dev_id = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
        conn.dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++));
        return conn;
    }

    // No router: nothing to open, close, or wait on.
    FORCE_INLINE void open() {}
    FORCE_INLINE void close() {}
    FORCE_INLINE void wait_for_empty_write_slot() const {}

    // The payload of the next packet. Consumed by the following header send.
    FORCE_INLINE void send_payload_without_header_non_blocking_from_address(uint32_t payload_addr, size_t size) {
        pending_payload_addr = payload_addr;
        pending_payload_size = static_cast<uint32_t>(size);
    }

    FORCE_INLINE void send_payload_flush_non_blocking_from_address(uint32_t header_addr, size_t /*header_size*/) {
        execute(reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header_addr));
    }

    FORCE_INLINE void send_payload_flush_blocking_from_address(uint32_t header_addr, size_t header_size) {
        send_payload_flush_non_blocking_from_address(header_addr, header_size);
        noc_async_writes_flushed();
    }

    uint8_t dst_dev_id = 0;
    uint16_t dst_mesh_id = 0;

private:
    // Runs the header's NoC command on the peer. Atomics always flush: NoC ordering does not hold across the
    // crossbar, so an increment could otherwise overtake the payload it announces.
    void execute(volatile tt_l1_ptr PACKET_HEADER_TYPE* header) {
        namespace a2a = tt::tt_fabric::a2a::experimental;
        const auto& fields = header->command_fields;
        const uint32_t payload_size = header->payload_size_bytes;
        ASSERT(payload_size == 0 || payload_size == pending_payload_size);

        switch (header->noc_send_type) {
            case NOC_UNICAST_WRITE:
                a2a::fabric_unicast_noc_unicast_write(
                    dst_dev_id,
                    dst_mesh_id,
                    pending_payload_addr,
                    payload_size,
                    NocUnicastCommandHeader{fields.unicast_write.noc_address});
                break;
            case NOC_UNICAST_INLINE_WRITE:
                a2a::fabric_unicast_noc_unicast_inline_write(
                    dst_dev_id,
                    dst_mesh_id,
                    NocUnicastInlineWriteCommandHeader{
                        fields.unicast_inline_write.noc_address, fields.unicast_inline_write.value});
                break;
            case NOC_UNICAST_ATOMIC_INC:
                a2a::fabric_unicast_noc_unicast_atomic_inc(
                    dst_dev_id,
                    dst_mesh_id,
                    NocUnicastAtomicIncCommandHeader(
                        fields.unicast_seminc.noc_address, fields.unicast_seminc.val, /*flush=*/true));
                break;
            case NOC_FUSED_UNICAST_ATOMIC_INC:
                a2a::fabric_unicast_noc_fused_unicast_with_atomic_inc(
                    dst_dev_id,
                    dst_mesh_id,
                    pending_payload_addr,
                    payload_size,
                    NocUnicastAtomicIncFusedCommandHeader(
                        fields.unicast_seminc_fused.noc_address,
                        fields.unicast_seminc_fused.semaphore_noc_address,
                        fields.unicast_seminc_fused.val,
                        /*flush=*/true));
                break;
            case NOC_UNICAST_SCATTER_WRITE: execute_scatter(fields.unicast_scatter_write, payload_size); break;
            default: ASSERT(false); break;
        }
    }

    // A scatter header carries up to four chunks; the last one is either a write or, for the fused
    // scatter + atomic-inc command, a semaphore increment whose value rides in that chunk's size slot.
    void execute_scatter(const volatile NocUnicastScatterCommandHeader& scatter, uint32_t payload_size) {
        namespace a2a = tt::tt_fabric::a2a::experimental;
        const uint8_t chunk_count = scatter.chunk_count;
        const uint8_t last = chunk_count - 1;
        const auto last_encoding =
            static_cast<NocScatterWriteChunkEncoding>((scatter.chunk_encoding >> (2 * last)) & 0b11);
        const bool ends_in_atomic_inc = last_encoding == NocScatterWriteChunkEncoding::CHUNK_ENCODING_SEMINC_FLUSH ||
                                        last_encoding == NocScatterWriteChunkEncoding::CHUNK_ENCODING_SEMINC_NO_FLUSH;
        const uint8_t num_write_chunks = ends_in_atomic_inc ? last : chunk_count;

        uint64_t addresses[NOC_SCATTER_WRITE_MAX_CHUNKS];
        uint16_t chunk_sizes[NOC_SCATTER_WRITE_MAX_CHUNKS - 1];
        for (uint8_t i = 0; i < num_write_chunks; i++) {
            addresses[i] = scatter.noc_address[i];
        }
        for (uint8_t i = 0; i + 1 < num_write_chunks; i++) {
            chunk_sizes[i] = scatter.chunk_size[i];
        }
        a2a::fabric_unicast_noc_scatter_write(
            dst_dev_id,
            dst_mesh_id,
            pending_payload_addr,
            payload_size,
            NocUnicastScatterCommandHeader(addresses, chunk_sizes, num_write_chunks));

        if (ends_in_atomic_inc) {
            a2a::fabric_unicast_noc_unicast_atomic_inc(
                dst_dev_id,
                dst_mesh_id,
                NocUnicastAtomicIncCommandHeader(scatter.noc_address[last], scatter.chunk_size[last], /*flush=*/true));
        }
    }

    uint32_t pending_payload_addr = 0;
    uint32_t pending_payload_size = 0;
};

}  // namespace tt::tt_fabric::fabric_tests

#endif  // FABRIC_TEST_HAS_NOC_TRANSPORT
