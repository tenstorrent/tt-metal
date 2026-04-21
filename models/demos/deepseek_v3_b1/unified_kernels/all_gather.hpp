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
//   handoff_sem on transport core — shared bitfield (BRISC+NCRISC):
//     bit 1: gather → scratch[0] ready  (noc_semaphore_inc 1)
//     bit 2: gather → scratch[1] ready  (noc_semaphore_inc 1 << 1)
//     bit 3: non-R2 RISC ack            (noc_semaphore_inc 1 << 2)
//     R2-active resets to 0 after the non-R2 peer acks (bit 3).
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

// Poll until all bits in BitMask are set in the semaphore word.
template <uint32_t BitMask>
FORCE_INLINE void wait_for_bits(volatile tt_l1_ptr uint32_t* sem_ptr) {
    do {
        invalidate_l1_cache();
    } while ((*sem_ptr & BitMask) != BitMask);
}

#endif  // COMPILE_FOR_NCRISC || COMPILE_FOR_BRISC

// ============================================================================
// Compile-time arg structs
// ============================================================================

static constexpr uint32_t MAX_NUM_LINKS = 1;

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
    uint32_t handoff_sem_bank_addr;  // Shared handoff semaphore; both BRISC and NCRISC use the same one
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
// Single-link only: reuses one connection with a small header ring to reduce
// flush frequency on multi-chunk slices.
// ============================================================================

template <typename CTArgs>
class TransportSender {
public:
    void operator()(const TransportArgs& args) { impl(args); }

private:
    static constexpr uint32_t max_header_ring_size = 2;
    static constexpr uint32_t header_ring_size = CTArgs::num_chunks <= 1 ? 1u : max_header_ring_size;

    void impl([[maybe_unused]] const TransportArgs& args) {
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
        static_assert(CTArgs::num_links == 1, "All-gather TransportSender is single-link only");

        size_t arg_idx = size_t(args.per_core_rta_start_idx);
        [[maybe_unused]] const uint32_t dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
        [[maybe_unused]] const uint32_t dst_chip_id = get_arg_val<uint32_t>(arg_idx++);

        constexpr bool use_posted_transport_writes = true;

        auto connection =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);

        const uint64_t dest_output_noc =
            safe_get_noc_addr(args.dest_noc_x, args.dest_noc_y, args.dest_output_base_addr, 0);
        const uint64_t dest_recv_sem_noc =
            safe_get_noc_addr(args.dest_noc_x, args.dest_noc_y, args.dest_recv_sem_addr, 0);

        const auto connection_direction = connection.get_connection_direction();
        std::array<volatile PACKET_HEADER_TYPE*, header_ring_size> headers = {};
        connection.open_start();
        PacketHeaderPool::reset();
        for (uint32_t i = 0; i < header_ring_size; ++i) {
            headers[i] = PacketHeaderPool::allocate_header();
            fabric_set_single_hop_unicast_route_from_direction(headers[i], connection_direction);
            headers[i]->to_noc_fused_unicast_write_atomic_inc(
                {dest_output_noc, dest_recv_sem_noc, /* placeholder inc_value */ 1, false}, CTArgs::chunk_size_bytes);
        }
        connection.open_finish();
        connection.setup_stateful_send_cmd_bufs<use_posted_transport_writes>();

        volatile tt_l1_ptr uint32_t* handoff_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.handoff_sem_bank_addr);
        // Local NOC address for ack increments (must use NOC atomic to avoid
        // racing with the gather core's NOC atomic incs on the same word).
        const uint64_t local_handoff_noc = get_noc_addr(args.handoff_sem_bank_addr);

        uint32_t cached_free_write_slots = 0;
        uint32_t current_header_idx = 0;
        uint32_t pending_header_count = 0;

        auto flush_transport = [&]() __attribute__((always_inline)) {
            if constexpr (use_posted_transport_writes) {
                noc_async_posted_writes_flushed();
            } else {
                noc_async_writes_flushed();
            }
        };

        auto refill_free_write_slots = [&]() __attribute__((always_inline)) {
            do {
                cached_free_write_slots = connection.get_num_free_write_slots();
            } while (cached_free_write_slots == 0);
        };

        auto drain_tail_before_reuse_or_teardown = [&]() __attribute__((always_inline)) {
            if (pending_header_count != 0) {
                flush_transport();
                pending_header_count = 0;
            }
        };

        // Send a full slice (possibly multi-chunk) over the single connection.
        auto send_slice = [&](uint32_t src_addr, uint32_t dest_slot_index) __attribute__((always_inline)) {
            drain_tail_before_reuse_or_teardown();

            const uint32_t inc_value = 1u << (dest_slot_index * CTArgs::recv_sem_bits_per_slot);
            const uint64_t dest_base = dest_output_noc + dest_slot_index * CTArgs::slice_size_bytes;

            for (uint32_t i = 0; i < header_ring_size; ++i) {
                headers[i]->set_fused_unicast_write_atomic_inc_value(inc_value);
            }

            uint32_t current_chunk_offset_bytes = 0;
            for (uint32_t chunk_idx = 0; chunk_idx < CTArgs::num_chunks; chunk_idx++) {
                const bool is_last_chunk = (chunk_idx == CTArgs::num_chunks - 1);
                const uint32_t payload_bytes = is_last_chunk ? CTArgs::last_chunk_bytes : CTArgs::chunk_size_bytes;

                if (cached_free_write_slots == 0) {
                    refill_free_write_slots();
                }

                auto* header = headers[current_header_idx++];
                if constexpr (CTArgs::last_chunk_bytes != CTArgs::chunk_size_bytes) {
                    header->set_payload_size_bytes(payload_bytes);
                }
                header->set_fused_unicast_write_atomic_inc_write_noc_address(dest_base + current_chunk_offset_bytes);

                connection.send_current_slot_stateful_non_blocking<use_posted_transport_writes>(
                    src_addr + current_chunk_offset_bytes, payload_bytes, reinterpret_cast<uint32_t>(header));

                cached_free_write_slots--;
                pending_header_count++;
                current_chunk_offset_bytes += CTArgs::chunk_size_bytes;

                if (current_header_idx == header_ring_size) {
                    current_header_idx = 0;
                    flush_transport();
                    pending_header_count = 0;
                }
            }
        };

        // Round 1: wait for bit 1 (scratch[0] ready), send local slice
        wait_for_bits<1>(handoff_sem_ptr);
        send_slice(args.scratch_base_addr, args.r1_dest_slot_index);

        if constexpr (CTArgs::r2_active) {
            // Round 2: wait for bit 2 (scratch[1] ready), forward received slice
            wait_for_bits<1 << 1>(handoff_sem_ptr);
            send_slice(args.scratch_base_addr + CTArgs::slice_size_bytes, args.r2_dest_slot_index);

            // Wait for non-R2 peer's bit 3 ack. Once observed, all NOC incs to
            // this word (bits 1/2 from gather, bit 3 from peer) have landed,
            // so the direct L1 reset is safe.
            wait_for_bits<1 << 2>(handoff_sem_ptr);
            noc_semaphore_set(handoff_sem_ptr, 0);
        } else {
            // Ack: set bit 3 via NOC atomic inc
            noc_semaphore_inc(local_handoff_noc, 1 << 2);
        }

        drain_tail_before_reuse_or_teardown();
        connection.close();
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
        // Signal bit 1: scratch[0] ready (unblocks R1 on both BRISC and NCRISC)
        noc_semaphore_inc(handoff_sem_noc, 1);

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
        // Signal bit 2: scratch[1] ready (unblocks R2 on the active RISC)
        noc_semaphore_inc(handoff_sem_noc, 1 << 1);

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
