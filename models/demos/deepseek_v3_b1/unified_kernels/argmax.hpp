// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"
#include "api/numeric/bfloat16.h"

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#endif

namespace deepseek_b1_ops {

struct Sampling {
    template <
        uint32_t NumValues,
        uint32_t WinnerPageBytes,
        uint32_t NumSenders,
        uint32_t ExpectedRemoteIncs,
        uint32_t ReceiverSemaphoreId,
        uint32_t LocalReadySemaphoreId,
        uint32_t MeshMode,
        uint32_t Stage1Sender,
        uint32_t Stage1Receiver,
        uint32_t Stage2Sender,
        uint32_t Stage2Receiver,
        uint32_t Stage1SlotBaseOffset,
        uint32_t Stage1NumSlots,
        uint32_t Stage1ExpectedRemoteIncs,
        uint32_t Stage1LocalSlotOffset,
        uint32_t Stage2SlotBaseOffset,
        uint32_t Stage2NumSlots,
        uint32_t Stage2ExpectedRemoteIncs,
        uint32_t Stage2LocalSlotOffset,
        uint32_t MeshLocalSendSlotOffset,
        uint32_t SenderIdx>
    struct ReaderCTArgs {
        static constexpr uint32_t num_values = NumValues;
        static constexpr uint32_t winner_page_bytes = WinnerPageBytes;
        static constexpr uint32_t num_senders = NumSenders;
        static constexpr uint32_t expected_remote_incs = ExpectedRemoteIncs;
        static constexpr uint32_t receiver_semaphore_id = ReceiverSemaphoreId;
        static constexpr uint32_t local_ready_semaphore_id = LocalReadySemaphoreId;
        static constexpr bool mesh_mode = MeshMode == 1;
        static constexpr bool stage1_sender = Stage1Sender == 1;
        static constexpr bool stage1_receiver = Stage1Receiver == 1;
        static constexpr bool stage2_sender = Stage2Sender == 1;
        static constexpr bool stage2_receiver = Stage2Receiver == 1;
        static constexpr uint32_t stage1_slot_base_offset = Stage1SlotBaseOffset;
        static constexpr uint32_t stage1_num_slots = Stage1NumSlots;
        static constexpr uint32_t stage1_expected_remote_incs = Stage1ExpectedRemoteIncs;
        static constexpr uint32_t stage1_local_slot_offset = Stage1LocalSlotOffset;
        static constexpr uint32_t stage2_slot_base_offset = Stage2SlotBaseOffset;
        static constexpr uint32_t stage2_num_slots = Stage2NumSlots;
        static constexpr uint32_t stage2_expected_remote_incs = Stage2ExpectedRemoteIncs;
        static constexpr uint32_t stage2_local_slot_offset = Stage2LocalSlotOffset;
        static constexpr uint32_t mesh_local_send_slot_offset = MeshLocalSendSlotOffset;
        static constexpr uint32_t sender_idx = SenderIdx;
    };

    template <uint32_t WinnerPageBytes, uint32_t LocalReadySemaphoreId>
    struct WriterCTArgs {
        static constexpr uint32_t winner_page_bytes = WinnerPageBytes;
        static constexpr uint32_t local_ready_semaphore_id = LocalReadySemaphoreId;
    };

    struct ReaderArgs {
        uint32_t scores_addr;
        uint32_t indices_addr;
        uint32_t output_addr;
        uint32_t final_noc_x;
        uint32_t final_noc_y;
        uint32_t scratch_addr;
        uint32_t global_sem_addr;
        uint32_t global_stage2_sem_addr;
        uint32_t gather_addr;
    };

    struct WriterArgs {
        uint32_t final_noc_x;
        uint32_t final_noc_y;
        uint32_t scratch_addr;
    };

    struct ComputeArgs {};

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

#if defined(COMPILE_FOR_BRISC)
    struct BriscMeshSendMetadata {
        uint32_t local_slot_offset;
        uint32_t dst_mesh_id;
        uint32_t dst_chip_id;
        uint32_t dst_l1_addr;
        uint32_t dst_sem_addr;
    };
#endif

    template <typename CTArgs, bool IsActiveCore, bool IsFinalCore, bool IsMeshSenderCore>
    class Op {
    public:
        void operator()(const RTArgs& args) { impl(args); }

    private:
#if defined(COMPILE_FOR_NCRISC)
        FORCE_INLINE bool is_better_candidate(
            uint16_t candidate_score, uint32_t candidate_index, uint16_t best_score, uint32_t best_index) {
            return bfloat16_greater(candidate_score, best_score) ||
                   ((candidate_score == best_score) && (candidate_index < best_index));
        }

        FORCE_INLINE void phase1_reduce_local_values(
            volatile tt_l1_ptr uint16_t* scores_ptr,
            volatile tt_l1_ptr uint32_t* indices_ptr,
            uint16_t& best_score,
            uint32_t& best_index) {
            best_score = NEG_INF_BFLOAT16;
            best_index = 0xFFFFFFFF;
            for (uint32_t i = 0; i < CTArgs::num_values; ++i) {
                const uint16_t score = scores_ptr[i];
                if (bfloat16_greater(score, best_score)) {
                    best_score = score;
                    best_index = indices_ptr[i];
                }
            }
        }

        FORCE_INLINE void phase1_send_local_winner_to_final(
            uint32_t src_slot_addr, uint32_t dst_slot_addr, uint32_t final_noc_x, uint32_t final_noc_y) {
            const uint64_t final_noc_base = get_noc_addr(final_noc_x, final_noc_y, 0);
            const uint64_t dst_data_noc_addr = final_noc_base | static_cast<uint64_t>(dst_slot_addr);
            const uint64_t dst_sem_noc_addr =
                final_noc_base | static_cast<uint64_t>(get_semaphore(CTArgs::receiver_semaphore_id));
            noc_async_write_one_packet<true, true>(src_slot_addr, dst_data_noc_addr, CTArgs::winner_page_bytes);
            noc_semaphore_inc(dst_sem_noc_addr, 1);
            noc_async_posted_writes_flushed();
            noc_async_atomic_barrier();
        }

        FORCE_INLINE void wait_and_reset_semaphore(volatile tt_l1_ptr uint32_t* sem_ptr, uint32_t expected_count) {
            noc_semaphore_wait(sem_ptr, expected_count);
            noc_semaphore_set(sem_ptr, 0);
        }

        FORCE_INLINE void phase2_reduce_intra_device_winners(
            uint32_t gather_addr, uint16_t& global_best_score, uint32_t& global_best_index) {
            global_best_score = NEG_INF_BFLOAT16;
            global_best_index = 0xFFFFFFFF;
            for (uint32_t slot = 0; slot < CTArgs::num_senders; ++slot) {
                const uint32_t slot_addr = gather_addr + slot * CTArgs::winner_page_bytes;
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
            uint16_t& stage_best_score,
            uint32_t& stage_best_index) {
            stage_best_score = NEG_INF_BFLOAT16;
            stage_best_index = 0xFFFFFFFF;
            for (uint32_t slot = 0; slot < stage_num_slots; ++slot) {
                uint16_t candidate_score = NEG_INF_BFLOAT16;
                uint32_t candidate_index = 0xFFFFFFFF;
                const uint32_t slot_addr = scratch_addr + stage_slot_base_offset + slot * CTArgs::winner_page_bytes;
                auto slot_u16_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(slot_addr);
                auto slot_u32_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_addr);
                candidate_score = slot_u16_ptr[0];
                candidate_index = slot_u32_ptr[1];
                if (is_better_candidate(candidate_score, candidate_index, stage_best_score, stage_best_index)) {
                    stage_best_score = candidate_score;
                    stage_best_index = candidate_index;
                }
            }
        }
#endif

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
#endif

#if defined(COMPILE_FOR_BRISC)
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
                CTArgs::winner_page_bytes);
            auto fabric_sender =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
            fabric_sender.open();
            fabric_sender.wait_for_empty_write_slot();
            fabric_sender.send_payload_without_header_non_blocking_from_address(
                local_slot_addr, CTArgs::winner_page_bytes);
            fabric_sender.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(packet_header), packet_header_size_bytes);
            fabric_sender.close();
            noc_async_full_barrier();
        }
#endif

        void impl(const RTArgs& args) {
#if defined(COMPILE_FOR_NCRISC)
            const uint32_t slot_offset = CTArgs::sender_idx * CTArgs::winner_page_bytes;
            const uint32_t gather_addr = args.gather_addr;

            invalidate_l1_cache();

            // Phase 1: per-core local argmax and delivery to the final core.
            if constexpr (IsActiveCore) {
                auto scores_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(args.scores_addr);
                auto indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.indices_addr);
                uint16_t best_score = NEG_INF_BFLOAT16;
                uint32_t best_index = 0xFFFFFFFF;
                phase1_reduce_local_values(scores_ptr, indices_ptr, best_score, best_index);

                if constexpr (IsFinalCore) {
                    write_winner_slot(gather_addr + slot_offset, best_score, best_index);
                } else {
                    const uint32_t local_slot_addr = gather_addr + slot_offset;
                    write_winner_slot(local_slot_addr, best_score, best_index);
                    phase1_send_local_winner_to_final(
                        local_slot_addr, gather_addr + slot_offset, args.final_noc_x, args.final_noc_y);
                }
            }

            // Phase 2: final-core intra-device reduction across all active cores.
            if constexpr (IsFinalCore) {
                auto recv_sem_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(CTArgs::receiver_semaphore_id));
                wait_and_reset_semaphore(recv_sem_ptr, CTArgs::expected_remote_incs);

                uint16_t global_best_score = NEG_INF_BFLOAT16;
                uint32_t global_best_index = 0xFFFFFFFF;
                phase2_reduce_intra_device_winners(gather_addr, global_best_score, global_best_index);

                // Phase 3: mesh-only inter-device reductions (stage-1 then stage-2).
                if constexpr (CTArgs::mesh_mode) {
                    if constexpr (CTArgs::stage1_receiver) {
                        write_winner_slot(
                            args.scratch_addr + CTArgs::stage1_local_slot_offset, global_best_score, global_best_index);
                        auto global_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.global_sem_addr);
                        wait_and_reset_semaphore(global_sem_ptr, CTArgs::stage1_expected_remote_incs);
                        uint16_t stage1_best_score = NEG_INF_BFLOAT16;
                        uint32_t stage1_best_index = 0xFFFFFFFF;
                        phase3_reduce_mesh_stage_slots(
                            args.scratch_addr,
                            CTArgs::stage1_slot_base_offset,
                            CTArgs::stage1_num_slots,
                            stage1_best_score,
                            stage1_best_index);
                        global_best_score = stage1_best_score;
                        global_best_index = stage1_best_index;
                    }
                }

                if constexpr (!CTArgs::mesh_mode) {
                    auto output_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.output_addr);
                    output_ptr[0] = global_best_index;
                } else {
                    if constexpr (IsMeshSenderCore && (CTArgs::stage1_sender || CTArgs::stage2_sender)) {
                        write_winner_slot(
                            args.scratch_addr + CTArgs::mesh_local_send_slot_offset,
                            global_best_score,
                            global_best_index);
                        auto local_ready_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            get_semaphore(CTArgs::local_ready_semaphore_id));
                        noc_semaphore_set(local_ready_sem_ptr, 1);
                    }

                    if constexpr (CTArgs::stage2_receiver) {
                        write_winner_slot(
                            args.scratch_addr + CTArgs::stage2_local_slot_offset, global_best_score, global_best_index);
                        auto global_stage2_sem_ptr =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.global_stage2_sem_addr);
                        wait_and_reset_semaphore(global_stage2_sem_ptr, CTArgs::stage2_expected_remote_incs);
                        uint16_t stage2_best_score = NEG_INF_BFLOAT16;
                        uint32_t stage2_best_index = 0xFFFFFFFF;
                        phase3_reduce_mesh_stage_slots(
                            args.scratch_addr,
                            CTArgs::stage2_slot_base_offset,
                            CTArgs::stage2_num_slots,
                            stage2_best_score,
                            stage2_best_index);
                        auto output_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.output_addr);
                        output_ptr[0] = stage2_best_index;
                    }
                }
            }
#elif defined(COMPILE_FOR_BRISC)
            invalidate_l1_cache();
            if constexpr (IsFinalCore && IsMeshSenderCore) {
                auto local_ready_sem_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(CTArgs::local_ready_semaphore_id));
                noc_semaphore_wait(local_ready_sem_ptr, 1);
                noc_semaphore_set(local_ready_sem_ptr, 0);

                size_t arg_idx = 0;
                const BriscMeshSendMetadata metadata = load_mesh_send_metadata(arg_idx);
                const uint32_t local_slot_addr = args.scratch_addr + metadata.local_slot_offset;
                send_mesh_winner_via_fabric_brisc(
                    args.final_noc_x, args.final_noc_y, local_slot_addr, metadata, arg_idx);
            }
#elif defined(COMPILE_FOR_TRISC)
            // No-op for k=1 argmax fast path.
            (void)args;
#endif
        }
    };
};

}  // namespace deepseek_b1_ops
