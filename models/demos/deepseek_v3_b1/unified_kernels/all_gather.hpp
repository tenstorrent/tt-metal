// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// All-gather for a 4-device torused row ring.
//
// Core assignment:
//   gather core:    NCRISC = controller, BRISC/TRISC = idle
//   transport core: BRISC = fwd sender, NCRISC = bwd sender, TRISC = idle
//
// Semaphores (2 total):
//   handoff_sem on transport core — monotonic (0 → 1 → 2 → 0)
//   recv_sem on gather core — packed cumulative per-slot counter
//
// Data flow (mcast-owned receives, sender-only forwarding):
//   R0: cross-core write to scratch[0] + handoff (unblocks R1 ASAP),
//       then local copy to own output slot
//   R1: bidirectional send of local slice; remote writes land on gather core
//   R1→R2: wait for forwarded slice, copy to scratch[1] + handoff
//   R2: pairwise exchange of the forwarded slice

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include <array>
#include <cstdint>
#endif

namespace deepseek_b1_ops {
namespace AllGather {

// ============================================================================
// Packed receive-semaphore wait helper
// ============================================================================
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)

// Poll a packed per-slot field inside a cumulative receive semaphore until
// the field reaches the expected chunk count.  Each slot occupies
// BitsPerSlice bits; the field for slot_idx starts at bit
// (slot_idx * BitsPerSlice).
template <uint32_t NumChunksPerSlice, uint32_t BitsPerSlice = 4>
FORCE_INLINE void wait_for_slot(volatile tt_l1_ptr uint32_t* sem_ptr, uint32_t slot_idx) {
    static_assert(NumChunksPerSlice <= ((1u << BitsPerSlice) - 1u));

    constexpr uint32_t field_mask = (1u << BitsPerSlice) - 1u;
    const uint32_t shift = slot_idx * BitsPerSlice;

    do {
        invalidate_l1_cache();
    } while (((*sem_ptr >> shift) & field_mask) < NumChunksPerSlice);
}

#endif  // COMPILE_FOR_NCRISC || COMPILE_FOR_BRISC

// ============================================================================
// Compile-time arg structs
// ============================================================================

static constexpr uint32_t MAX_NUM_LINKS = 2;

// Transport core BRISC and NCRISC share this schema; r2_active differs.
template <
    uint32_t SliceSizeBytes,
    uint32_t NumChunks,
    uint32_t ChunkSizeBytes,
    uint32_t LastChunkBytes,
    uint32_t NumLinks,
    uint32_t RecvSemBitsPerSlot,
    uint32_t R2Active>
struct TransportCTArgs {
    static constexpr uint32_t slice_size_bytes = SliceSizeBytes;
    static constexpr uint32_t num_chunks = NumChunks;
    static constexpr uint32_t chunk_size_bytes = ChunkSizeBytes;
    static constexpr uint32_t last_chunk_bytes = LastChunkBytes;
    static constexpr uint32_t num_links = NumLinks;
    static constexpr uint32_t recv_sem_bits_per_slot = RecvSemBitsPerSlot;
    static constexpr bool r2_active = R2Active != 0;
    static_assert(num_links >= 1 && num_links <= MAX_NUM_LINKS, "num_links out of range");
};

// Gather core NCRISC.
template <uint32_t SliceSizeBytes, uint32_t NumChunks, uint32_t RingSize, uint32_t RecvSemBitsPerSlot>
struct GatherCTArgs {
    static constexpr uint32_t slice_size_bytes = SliceSizeBytes;
    static constexpr uint32_t num_chunks = NumChunks;
    static constexpr uint32_t ring_size = RingSize;
    static constexpr uint32_t recv_sem_bits_per_slot = RecvSemBitsPerSlot;
};

// ============================================================================
// Runtime arg structs
// ============================================================================

struct TransportArgs {
    uint32_t scratch_base_addr;
    uint32_t handoff_sem_bank_addr;
    uint32_t dest_output_base_addr;
    uint32_t r1_dest_slot_index;
    uint32_t dest_noc_x;
    uint32_t dest_noc_y;
    uint32_t dest_recv_sem_addr;
    uint32_t r2_dest_slot_index;
    uint32_t per_core_rta_start_idx;
};

struct GatherArgs {
    uint32_t local_input_addr;
    uint32_t output_buffer_addr;
    uint32_t self_slot_index;
    uint32_t transport_scratch_base_addr;
    uint32_t transport_noc_x;
    uint32_t transport_noc_y;
    uint32_t handoff_sem_bank_addr;
    uint32_t recv_sem_addr;
    uint32_t r2_src_slot_index;
};

// ============================================================================
// Transport sender (BRISC fwd / NCRISC bwd on transport core)
//
// Multi-link: opens num_links connections and rotates chunks across them,
// following the broadcast.hpp forward_chunks pattern.
// ============================================================================

template <typename CTArgs>
class TransportSender {
public:
    void operator()(const TransportArgs& args) { impl(args); }

private:
    void impl([[maybe_unused]] const TransportArgs& args) {
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
        PacketHeaderPool::reset();

        size_t arg_idx = size_t(args.per_core_rta_start_idx);
        const uint32_t dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t dst_chip_id = get_arg_val<uint32_t>(arg_idx++);

        std::array<tt::tt_fabric::WorkerToFabricEdmSender, CTArgs::num_links> connections;
        std::array<volatile PACKET_HEADER_TYPE*, CTArgs::num_links> headers;
        for (uint32_t link = 0; link < CTArgs::num_links; link++) {
            connections[link] =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
            connections[link].open_start();
            headers[link] = PacketHeaderPool::allocate_header();
            fabric_set_unicast_route(headers[link], dst_chip_id, dst_mesh_id);
            connections[link].open_finish();
        }

        const uint64_t dest_output_noc =
            safe_get_noc_addr(args.dest_noc_x, args.dest_noc_y, args.dest_output_base_addr, 0);
        const uint64_t dest_recv_sem_noc =
            safe_get_noc_addr(args.dest_noc_x, args.dest_noc_y, args.dest_recv_sem_addr, 0);

        volatile tt_l1_ptr uint32_t* handoff_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.handoff_sem_bank_addr);

        // Send a full slice (possibly multi-chunk) rotating across links.
        auto send_slice = [&](uint32_t src_addr, uint32_t dest_slot_index) __attribute__((always_inline)) {
            const uint32_t inc_value = 1u << (dest_slot_index * CTArgs::recv_sem_bits_per_slot);
            const uint64_t dest_base = dest_output_noc + dest_slot_index * CTArgs::slice_size_bytes;

            uint32_t current_link = 0;
            for (uint32_t chunk_idx = 0; chunk_idx < CTArgs::num_chunks; chunk_idx++) {
                const uint32_t chunk_offset = chunk_idx * CTArgs::chunk_size_bytes;
                const uint32_t payload_bytes =
                    (chunk_idx < CTArgs::num_chunks - 1) ? CTArgs::chunk_size_bytes : CTArgs::last_chunk_bytes;

                headers[current_link]->to_noc_fused_unicast_write_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                        dest_base + chunk_offset, dest_recv_sem_noc, inc_value, false},
                    payload_bytes);
                connections[current_link].wait_for_empty_write_slot();
                connections[current_link].send_payload_without_header_non_blocking_from_address(
                    src_addr + chunk_offset, payload_bytes);
                connections[current_link].send_payload_flush_non_blocking_from_address(
                    reinterpret_cast<uint32_t>(headers[current_link]), sizeof(PACKET_HEADER_TYPE));

                if (++current_link == CTArgs::num_links) {
                    current_link = 0;
                    noc_async_writes_flushed();
                }
            }
            // Trailing flush for partial rotation at end of slice
            noc_async_writes_flushed();
        };

        // Round 1: send local slice to neighbor
        noc_semaphore_wait_min(handoff_sem_ptr, 1);
        send_slice(args.scratch_base_addr, args.r1_dest_slot_index);

        if constexpr (CTArgs::r2_active) {
            // Round 2: forward the received slice to pair partner
            noc_semaphore_wait_min(handoff_sem_ptr, 2);
            send_slice(args.scratch_base_addr + CTArgs::slice_size_bytes, args.r2_dest_slot_index);
            noc_semaphore_set(handoff_sem_ptr, 0);
        }

        for (uint32_t link = 0; link < CTArgs::num_links; link++) {
            connections[link].close();
        }
        noc_async_full_barrier();
#endif
    }
};

// ============================================================================
// Gather controller (NCRISC on gather core)
// ============================================================================

template <typename CTArgs>
class GatherController {
public:
    void operator()(const GatherArgs& args) { impl(args); }

private:
    void impl([[maybe_unused]] const GatherArgs& args) {
#if defined(COMPILE_FOR_NCRISC)
        const uint64_t handoff_sem_noc =
            safe_get_noc_addr(args.transport_noc_x, args.transport_noc_y, args.handoff_sem_bank_addr, 0);
        const uint64_t scratch_base_noc =
            safe_get_noc_addr(args.transport_noc_x, args.transport_noc_y, args.transport_scratch_base_addr, 0);

        volatile tt_l1_ptr uint32_t* recv_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.recv_sem_addr);

        const uint32_t self_slot_offset = args.self_slot_index * CTArgs::slice_size_bytes;
        const uint32_t r2_src_slot_offset = args.r2_src_slot_index * CTArgs::slice_size_bytes;

        // Track which slots have been handled so the final wait is generic.
        uint32_t done_mask = 0;
        done_mask |= (1u << args.self_slot_index);

        // Cross-core write to transport scratch[0] FIRST to unblock R1 ASAP.
        noc_async_write(args.local_input_addr, scratch_base_noc, CTArgs::slice_size_bytes);
        noc_async_writes_flushed();
        noc_semaphore_inc(handoff_sem_noc, 1);  // handoff_sem: 0 → 1

        // Local copy to own output slot (can overlap with R1 fabric sends).
        noc_async_write(
            args.local_input_addr, get_noc_addr(args.output_buffer_addr + self_slot_offset), CTArgs::slice_size_bytes);

        // R1→R2 bridge: wait for the slot that must be forwarded.
        wait_for_slot<CTArgs::num_chunks, CTArgs::recv_sem_bits_per_slot>(recv_sem_ptr, args.r2_src_slot_index);
        done_mask |= (1u << args.r2_src_slot_index);

        // Copy received slot to transport scratch[1].
        const uint64_t scratch_r2_noc = scratch_base_noc + CTArgs::slice_size_bytes;
        noc_async_write(args.output_buffer_addr + r2_src_slot_offset, scratch_r2_noc, CTArgs::slice_size_bytes);
        noc_async_writes_flushed();
        noc_semaphore_inc(handoff_sem_noc, 1);  // handoff_sem: 1 → 2

        // Wait for all remaining slots (topology-agnostic).
        for (uint32_t slot = 0; slot < CTArgs::ring_size; slot++) {
            if (!(done_mask & (1u << slot))) {
                wait_for_slot<CTArgs::num_chunks, CTArgs::recv_sem_bits_per_slot>(recv_sem_ptr, slot);
            }
        }
        noc_semaphore_set(recv_sem_ptr, 0);
        noc_async_full_barrier();
#endif
    }
};

}  // namespace AllGather
}  // namespace deepseek_b1_ops
