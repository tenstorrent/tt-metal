// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Sampling unified kernel (k=1 argmax fast path)
//
// Multi-core single-device scope:
// - Each active core computes local argmax over its 160 values.
// - Local winner is packed into one 16B page:
//   [bf16 score, uint32 index, garbage, garbage]
// - Non-final active cores posted-write their page to final core slot and
//   increment final-core semaphore.
// - Final core waits for all remote semaphore increments, then reduces all
//   gathered slots to one final index (tie-break: lowest index).
//
// Mesh extension (R x 2, axis-x first):
// - Stage 1: for each column, all rows send to target_row on final cores, then local compare on target_row.
// - Stage 2: on target_row, non-target columns send to target_col, then final compare at target coord.
// - Inter-device transfers use a fabric packet with fused NOC write + semaphore increment.

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "api/numeric/bfloat16.h"
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#endif

struct Core {
    static constexpr bool is_active_core = get_named_compile_time_arg_val("sampling_is_active_core") == 1;
    static constexpr bool is_final_core = get_named_compile_time_arg_val("sampling_is_final_core") == 1;
    static constexpr bool is_mesh_sender_core = get_named_compile_time_arg_val("sampling_mesh_sender_core") == 1;
};

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
template <typename packet_header_t>
FORCE_INLINE void set_unicast_route(
    volatile tt_l1_ptr packet_header_t* header, uint16_t dst_dev_id, uint16_t dst_mesh_id, uint16_t num_hops) {
    if constexpr (std::is_same_v<packet_header_t, tt::tt_fabric::HybridMeshPacketHeader>) {
        fabric_set_unicast_route(header, dst_dev_id, dst_mesh_id);
    } else {
        fabric_set_unicast_route<false>(header, num_hops);
    }
}

FORCE_INLINE void write_winner_slot(uint32_t slot_addr, uint16_t score, uint32_t index) {
    auto slot_u16_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(slot_addr);
    auto slot_u32_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_addr);
    slot_u16_ptr[0] = score;
    slot_u32_ptr[1] = index;
}

FORCE_INLINE void read_winner_slot(uint32_t slot_addr, uint16_t& score, uint32_t& index) {
    auto slot_u16_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(slot_addr);
    auto slot_u32_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_addr);
    score = slot_u16_ptr[0];
    index = slot_u32_ptr[1];
}
#endif

#if defined(COMPILE_FOR_NCRISC)
FORCE_INLINE bool is_better_candidate(
    uint16_t candidate_score, uint32_t candidate_index, uint16_t best_score, uint32_t best_index) {
    return bfloat16_greater(candidate_score, best_score) ||
           ((candidate_score == best_score) && (candidate_index < best_index));
}

FORCE_INLINE void phase1_reduce_local_values(
    volatile tt_l1_ptr uint16_t* scores_ptr,
    volatile tt_l1_ptr uint32_t* indices_ptr,
    uint32_t num_values,
    uint16_t& best_score,
    uint32_t& best_index) {
    best_score = NEG_INF_BFLOAT16;
    best_index = 0xFFFFFFFF;
    for (uint32_t i = 0; i < num_values; ++i) {
        const uint16_t score = scores_ptr[i];
        if (bfloat16_greater(score, best_score)) {
            best_score = score;
            best_index = indices_ptr[i];
        }
    }
}

FORCE_INLINE void phase1_send_local_winner_to_final(
    uint32_t src_slot_addr,
    uint32_t dst_slot_addr,
    uint32_t final_noc_x,
    uint32_t final_noc_y,
    uint32_t semaphore_id,
    uint32_t winner_page_bytes) {
    const uint64_t final_noc_base = get_noc_addr(final_noc_x, final_noc_y, 0);
    const uint64_t dst_data_noc_addr = final_noc_base | static_cast<uint64_t>(dst_slot_addr);
    const uint64_t dst_sem_noc_addr = final_noc_base | static_cast<uint64_t>(get_semaphore(semaphore_id));
    noc_async_write_one_packet<true, true>(src_slot_addr, dst_data_noc_addr, winner_page_bytes);
    noc_semaphore_inc(dst_sem_noc_addr, 1);
    noc_async_posted_writes_flushed();
    noc_async_atomic_barrier();
}

FORCE_INLINE void wait_and_reset_semaphore(volatile tt_l1_ptr uint32_t* sem_ptr, uint32_t expected_count) {
    noc_semaphore_wait(sem_ptr, expected_count);
    noc_semaphore_set(sem_ptr, 0);
}

FORCE_INLINE void phase2_reduce_intra_device_winners(
    uint32_t gather_addr,
    uint32_t num_senders,
    uint32_t winner_page_bytes,
    uint16_t& global_best_score,
    uint32_t& global_best_index) {
    global_best_score = NEG_INF_BFLOAT16;
    global_best_index = 0xFFFFFFFF;
    for (uint32_t slot = 0; slot < num_senders; ++slot) {
        const uint32_t slot_addr = gather_addr + slot * winner_page_bytes;
        auto slot_u16_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(slot_addr);
        auto slot_u32_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_addr);
        const uint16_t score = slot_u16_ptr[0];
        if (bfloat16_greater(score, global_best_score)) {
            global_best_score = score;
            global_best_index = slot_u32_ptr[1];
        }
    }
}

FORCE_INLINE void phase3_reduce_mesh_stage_slots(
    uint32_t scratch_addr,
    uint32_t stage_slot_base_offset,
    uint32_t stage_num_slots,
    uint32_t winner_page_bytes,
    uint16_t& stage_best_score,
    uint32_t& stage_best_index) {
    stage_best_score = NEG_INF_BFLOAT16;
    stage_best_index = 0xFFFFFFFF;
    for (uint32_t slot = 0; slot < stage_num_slots; ++slot) {
        uint16_t candidate_score = NEG_INF_BFLOAT16;
        uint32_t candidate_index = 0xFFFFFFFF;
        read_winner_slot(
            scratch_addr + stage_slot_base_offset + slot * winner_page_bytes, candidate_score, candidate_index);
        if (is_better_candidate(candidate_score, candidate_index, stage_best_score, stage_best_index)) {
            stage_best_score = candidate_score;
            stage_best_index = candidate_index;
        }
    }
}
#endif

#if defined(COMPILE_FOR_BRISC)
struct BriscMeshSendMetadata {
    uint32_t local_slot_offset;
    uint32_t dst_mesh_id;
    uint32_t dst_chip_id;
    uint32_t dst_l1_addr;
    uint32_t dst_sem_addr;
};

FORCE_INLINE BriscMeshSendMetadata load_mesh_send_metadata(size_t& arg_idx) {
    BriscMeshSendMetadata metadata{};
    metadata.local_slot_offset = get_arg_val<uint32_t>(arg_idx++);
    metadata.dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
    metadata.dst_chip_id = get_arg_val<uint32_t>(arg_idx++);
    metadata.dst_l1_addr = get_arg_val<uint32_t>(arg_idx++);
    metadata.dst_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    return metadata;
}

FORCE_INLINE void send_mesh_winner_via_fabric_brisc(
    uint32_t final_noc_x,
    uint32_t final_noc_y,
    uint32_t local_slot_addr,
    const BriscMeshSendMetadata& metadata,
    uint32_t winner_page_bytes,
    size_t arg_idx) {
    constexpr uint32_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);
    auto route_id = PacketHeaderPool::allocate_header_n(1);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = PacketHeaderPool::header_table[route_id].first;
    set_unicast_route(
        packet_header,
        static_cast<uint16_t>(metadata.dst_chip_id),
        static_cast<uint16_t>(metadata.dst_mesh_id),
        1);
    packet_header->to_noc_fused_unicast_write_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
            get_noc_addr(final_noc_x, final_noc_y, metadata.dst_l1_addr),
            get_noc_addr(final_noc_x, final_noc_y, metadata.dst_sem_addr),
            1,
            false},
        winner_page_bytes);

    auto fabric_sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
    fabric_sender.open();
    fabric_sender.wait_for_empty_write_slot();
    fabric_sender.send_payload_without_header_non_blocking_from_address(local_slot_addr, winner_page_bytes);
    fabric_sender.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(packet_header), packet_header_size_bytes);
    fabric_sender.close();
    noc_async_full_barrier();
}
#endif

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t num_values = get_named_compile_time_arg_val("sampling_num_values");
    constexpr uint32_t winner_page_bytes = get_named_compile_time_arg_val("sampling_winner_page_bytes");
    constexpr uint32_t num_senders = get_named_compile_time_arg_val("sampling_num_senders");
    constexpr uint32_t expected_remote_incs = get_named_compile_time_arg_val("sampling_expected_remote_incs");
    constexpr uint32_t gather_cb = get_named_compile_time_arg_val("sampling_gather_cb");
    constexpr uint32_t semaphore_id = get_named_compile_time_arg_val("sampling_receiver_semaphore_id");
    constexpr uint32_t local_ready_semaphore_id = get_named_compile_time_arg_val("sampling_local_ready_semaphore_id");
    constexpr bool mesh_mode = get_named_compile_time_arg_val("sampling_mesh_mode") == 1;
    constexpr bool stage1_sender = get_named_compile_time_arg_val("sampling_stage1_sender") == 1;
    constexpr bool stage1_receiver = get_named_compile_time_arg_val("sampling_stage1_receiver") == 1;
    constexpr bool stage2_sender = get_named_compile_time_arg_val("sampling_stage2_sender") == 1;
    constexpr bool stage2_receiver = get_named_compile_time_arg_val("sampling_stage2_receiver") == 1;
    constexpr uint32_t stage1_slot_base_offset = get_named_compile_time_arg_val("sampling_stage1_slot_base_offset");
    constexpr uint32_t stage1_num_slots = get_named_compile_time_arg_val("sampling_stage1_num_slots");
    constexpr uint32_t stage1_expected_remote_incs =
        get_named_compile_time_arg_val("sampling_stage1_expected_remote_incs");
    constexpr uint32_t stage1_local_slot_offset = get_named_compile_time_arg_val("sampling_stage1_local_slot_offset");
    constexpr uint32_t stage2_slot_base_offset = get_named_compile_time_arg_val("sampling_stage2_slot_base_offset");
    constexpr uint32_t stage2_num_slots = get_named_compile_time_arg_val("sampling_stage2_num_slots");
    constexpr uint32_t stage2_expected_remote_incs =
        get_named_compile_time_arg_val("sampling_stage2_expected_remote_incs");
    constexpr uint32_t stage2_local_slot_offset = get_named_compile_time_arg_val("sampling_stage2_local_slot_offset");
    constexpr uint32_t mesh_local_send_slot_offset =
        get_named_compile_time_arg_val("sampling_mesh_local_send_slot_offset");

    const uint32_t scores_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t indices_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t output_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t final_noc_x = get_common_arg_val<uint32_t>(3);
    const uint32_t final_noc_y = get_common_arg_val<uint32_t>(4);
    const uint32_t scratch_addr = get_common_arg_val<uint32_t>(5);
    const uint32_t global_sem_addr = get_common_arg_val<uint32_t>(6);
    const uint32_t global_stage2_sem_addr = get_common_arg_val<uint32_t>(7);

    const uint32_t sender_idx = get_named_compile_time_arg_val("sampling_sender_idx");
    const uint32_t slot_offset = sender_idx * winner_page_bytes;
    const uint32_t gather_addr = get_write_ptr(gather_cb);

    invalidate_l1_cache();

    // Phase 1: per-core local argmax and delivery to the final core.
    if constexpr (Core::is_active_core) {
        auto scores_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scores_addr);
        auto indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(indices_addr);
        uint16_t best_score = NEG_INF_BFLOAT16;
        uint32_t best_index = 0xFFFFFFFF;
        phase1_reduce_local_values(scores_ptr, indices_ptr, num_values, best_score, best_index);

        if constexpr (Core::is_final_core) {
            write_winner_slot(gather_addr + slot_offset, best_score, best_index);
        } else {
            const uint32_t local_slot_addr = gather_addr + slot_offset;
            write_winner_slot(local_slot_addr, best_score, best_index);
            phase1_send_local_winner_to_final(
                local_slot_addr, gather_addr + slot_offset, final_noc_x, final_noc_y, semaphore_id, winner_page_bytes);
        }
    }

    // Phase 2: final-core intra-device reduction across all active cores.
    if constexpr (Core::is_final_core) {
        auto recv_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(semaphore_id));
        wait_and_reset_semaphore(recv_sem_ptr, expected_remote_incs);

        uint16_t global_best_score = NEG_INF_BFLOAT16;
        uint32_t global_best_index = 0xFFFFFFFF;
        phase2_reduce_intra_device_winners(
            gather_addr, num_senders, winner_page_bytes, global_best_score, global_best_index);

        // Phase 3: mesh-only inter-device reductions (stage-1 then stage-2).
        if constexpr (mesh_mode) {
            // Stage 1 receiver: combine local winner with remote winners from all non-target rows.
            if constexpr (stage1_receiver) {
                write_winner_slot(scratch_addr + stage1_local_slot_offset, global_best_score, global_best_index);
                auto global_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_sem_addr);
                wait_and_reset_semaphore(global_sem_ptr, stage1_expected_remote_incs);
                uint16_t stage1_best_score = NEG_INF_BFLOAT16;
                uint32_t stage1_best_index = 0xFFFFFFFF;
                phase3_reduce_mesh_stage_slots(
                    scratch_addr,
                    stage1_slot_base_offset,
                    stage1_num_slots,
                    winner_page_bytes,
                    stage1_best_score,
                    stage1_best_index);
                global_best_score = stage1_best_score;
                global_best_index = stage1_best_index;
            }
        }

        if constexpr (!mesh_mode) {
            auto output_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(output_addr);
            output_ptr[0] = global_best_index;
        } else {
            if constexpr (Core::is_mesh_sender_core && (stage1_sender || stage2_sender)) {
                // BRISC owns fabric send; publish sender payload then signal local-ready.
                write_winner_slot(scratch_addr + mesh_local_send_slot_offset, global_best_score, global_best_index);
                auto local_ready_sem_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(local_ready_semaphore_id));
                noc_semaphore_set(local_ready_sem_ptr, 1);
            }

            if constexpr (stage2_receiver) {
                write_winner_slot(scratch_addr + stage2_local_slot_offset, global_best_score, global_best_index);
                auto global_stage2_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_stage2_sem_addr);
                wait_and_reset_semaphore(global_stage2_sem_ptr, stage2_expected_remote_incs);
                uint16_t stage2_best_score = NEG_INF_BFLOAT16;
                uint32_t stage2_best_index = 0xFFFFFFFF;
                phase3_reduce_mesh_stage_slots(
                    scratch_addr,
                    stage2_slot_base_offset,
                    stage2_num_slots,
                    winner_page_bytes,
                    stage2_best_score,
                    stage2_best_index);
                global_best_score = stage2_best_score;
                global_best_index = stage2_best_index;
                auto output_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(output_addr);
                output_ptr[0] = global_best_index;
            }
        }
    }

#elif defined(COMPILE_FOR_BRISC)
    constexpr uint32_t winner_page_bytes = get_named_compile_time_arg_val("sampling_winner_page_bytes");
    constexpr uint32_t local_ready_semaphore_id = get_named_compile_time_arg_val("sampling_local_ready_semaphore_id");

    if constexpr (Core::is_final_core && Core::is_mesh_sender_core) {
        const uint32_t final_noc_x = get_common_arg_val<uint32_t>(3);
        const uint32_t final_noc_y = get_common_arg_val<uint32_t>(4);
        const uint32_t scratch_addr = get_common_arg_val<uint32_t>(5);

        // Wait for NCRISC to publish winner payload to scratch.
        auto local_ready_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(local_ready_semaphore_id));
        noc_semaphore_wait(local_ready_sem_ptr, 1);
        noc_semaphore_set(local_ready_sem_ptr, 0);

        // Sender metadata and fabric connection args are appended by host.
        size_t arg_idx = 0;
        const BriscMeshSendMetadata metadata = load_mesh_send_metadata(arg_idx);
        const uint32_t local_slot_addr = scratch_addr + metadata.local_slot_offset;
        send_mesh_winner_via_fabric_brisc(final_noc_x, final_noc_y, local_slot_addr, metadata, winner_page_bytes, arg_idx);
    }

#elif defined(COMPILE_FOR_TRISC)
    // No-op for k=1 argmax fast path.
#endif
}
