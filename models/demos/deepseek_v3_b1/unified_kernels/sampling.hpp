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
#include "api/socket_api.h"
#include "../micro_ops/host_io/kernels/pcie_noc_utils.h"
#endif

namespace deepseek_b1_ops {

struct TopKSampling {
    template <
        uint32_t NumValues,
        uint32_t TopK,
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
        uint32_t SenderIdx,
        uint32_t SocketMode = 0,
        uint32_t SocketCBId = 0,
        uint32_t SocketPageSizeBytes = 0,
        uint32_t ScoresCBId = 0xFFFFFFFF,
        uint32_t ScoresNumPages = 0,
        uint32_t GatherCBId = 0xFFFFFFFF,
        uint32_t WinnerCBId = 0xFFFFFFFF>
    struct ReaderCTArgs {
        static constexpr uint32_t num_values = NumValues;
        static constexpr uint32_t topk_k = TopK;
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
        static constexpr uint32_t socket_mode = SocketMode;
        static constexpr uint32_t socket_cb_id = SocketCBId;
        static constexpr uint32_t socket_page_size_bytes = SocketPageSizeBytes;
        static constexpr uint32_t scores_cb_id = ScoresCBId;
        static constexpr uint32_t scores_num_pages = ScoresNumPages;
        static constexpr uint32_t gather_cb_id = GatherCBId;
        static constexpr uint32_t winner_cb_id = WinnerCBId;

        // Gather buffer layout (globally split for LLK compatibility):
        //   [core0 scores | core1 scores | ... | coreN scores]
        //   [core0 indices | core1 indices | ... | coreN indices]
        // Each per-core region is padded to 16-byte alignment for NOC transfers.
        static constexpr uint32_t topk_scores_size = TopK * sizeof(uint16_t);
        static constexpr uint32_t topk_indices_size = TopK * sizeof(uint32_t);
        static constexpr uint32_t topk_scores_stride = (topk_scores_size + 15u) & ~15u;
        static constexpr uint32_t topk_indices_stride = (topk_indices_size + 15u) & ~15u;
        static constexpr uint32_t gather_indices_offset = NumSenders * topk_scores_stride;
    };

    template <
        uint32_t WinnerPageBytes,
        uint32_t LocalReadySemaphoreId,
        uint32_t SocketMode = 0,
        uint32_t SocketCBId = 0,
        uint32_t SocketPageSizeBytes = 0>
    struct WriterCTArgs {
        static constexpr uint32_t winner_page_bytes = WinnerPageBytes;
        static constexpr uint32_t local_ready_semaphore_id = LocalReadySemaphoreId;
        static constexpr uint32_t socket_mode = SocketMode;
        static constexpr uint32_t socket_cb_id = SocketCBId;
        static constexpr uint32_t socket_page_size_bytes = SocketPageSizeBytes;
    };

    struct ComputeCTArgs {};

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
        uint32_t socket_config_addr = 0;
        // Optional persistent-mode next-iteration signal routing (BRISC path).
        uint32_t persistent_enable = 0;
        uint32_t persistent_dst_noc_x = 0;
        uint32_t persistent_dst_noc_y = 0;
        uint32_t persistent_dst_mesh_id = 0;
        uint32_t persistent_dst_chip_id = 0;
        uint32_t persistent_dst_sem_addr = 0;
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

        // Sorted insertion: maintain a descending-sorted array of K (score, index) pairs
        // in two separate output arrays.  After iterating over all num_values elements,
        // out_scores[0..K-1] and out_indices[0..K-1] contain the top-K in descending order.
        FORCE_INLINE void phase1_reduce_local_topk(
            volatile tt_l1_ptr uint16_t* scores_ptr,
            volatile tt_l1_ptr uint32_t* indices_ptr,
            volatile tt_l1_ptr uint16_t* out_scores,
            volatile tt_l1_ptr uint32_t* out_indices) {
            constexpr uint32_t K = CTArgs::topk_k;

            for (uint32_t j = 0; j < K; ++j) {
                out_scores[j] = NEG_INF_BFLOAT16;
                out_indices[j] = 0xFFFFFFFF;
            }

            for (uint32_t i = 0; i < CTArgs::num_values; ++i) {
                const uint16_t score = scores_ptr[i];
                const uint32_t index = indices_ptr[i];

                if (!is_better_candidate(score, index, out_scores[K - 1], out_indices[K - 1])) {
                    continue;
                }

                uint32_t pos = K - 1;
                while (pos > 0 && is_better_candidate(score, index, out_scores[pos - 1], out_indices[pos - 1])) {
                    --pos;
                }

                for (uint32_t j = K - 1; j > pos; --j) {
                    out_scores[j] = out_scores[j - 1];
                    out_indices[j] = out_indices[j - 1];
                }

                out_scores[pos] = score;
                out_indices[pos] = index;
            }
        }

        FORCE_INLINE void phase1_send_topk_to_final(
            uint32_t local_scores_addr,
            uint32_t local_indices_addr,
            uint32_t gather_addr,
            uint32_t final_noc_x,
            uint32_t final_noc_y) {
            const uint64_t final_noc_base = get_noc_addr(final_noc_x, final_noc_y, 0);
            const uint64_t dst_sem_noc_addr =
                final_noc_base | static_cast<uint64_t>(get_semaphore(CTArgs::receiver_semaphore_id));

            const uint32_t dst_scores = gather_addr + CTArgs::sender_idx * CTArgs::topk_scores_stride;
            const uint32_t dst_indices =
                gather_addr + CTArgs::gather_indices_offset + CTArgs::sender_idx * CTArgs::topk_indices_stride;

            noc_async_write(local_scores_addr, final_noc_base | dst_scores, CTArgs::topk_scores_size);
            noc_async_write(local_indices_addr, final_noc_base | dst_indices, CTArgs::topk_indices_size);
            noc_semaphore_inc(dst_sem_noc_addr, 1);
            noc_async_posted_writes_flushed();
            noc_async_atomic_barrier();
        }

        FORCE_INLINE void wait_and_reset_semaphore(volatile tt_l1_ptr uint32_t* sem_ptr, uint32_t expected_count) {
            noc_semaphore_wait(sem_ptr, expected_count);
            noc_semaphore_set(sem_ptr, 0);
        }

        // N-way merge of per-core sorted-descending top-K arrays into a single
        // global top-K (also descending).  Each core contributed a sorted array;
        // we maintain one head pointer per core and greedily pick the best head
        // K times.  O(K * N) comparisons -- for K=32, N~100 that is ~3200.
        FORCE_INLINE void phase2_merge_global_topk(
            uint32_t gather_addr,
            volatile tt_l1_ptr uint16_t* global_scores,
            volatile tt_l1_ptr uint32_t* global_indices) {
            constexpr uint32_t K = CTArgs::topk_k;
            constexpr uint32_t N = CTArgs::num_senders;

            uint8_t heads[N];
            for (uint32_t c = 0; c < N; ++c) {
                heads[c] = 0;
            }

            for (uint32_t out = 0; out < K; ++out) {
                uint16_t best_score = NEG_INF_BFLOAT16;
                uint32_t best_index = 0xFFFFFFFF;
                uint32_t best_core = 0;

                for (uint32_t c = 0; c < N; ++c) {
                    if (heads[c] >= K) {
                        continue;
                    }
                    auto s = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                        gather_addr + c * CTArgs::topk_scores_stride);
                    auto idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        gather_addr + CTArgs::gather_indices_offset + c * CTArgs::topk_indices_stride);
                    if (is_better_candidate(s[heads[c]], idx[heads[c]], best_score, best_index)) {
                        best_score = s[heads[c]];
                        best_index = idx[heads[c]];
                        best_core = c;
                    }
                }

                global_scores[out] = best_score;
                global_indices[out] = best_index;
                heads[best_core]++;
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
            size_t& arg_idx) {
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

        FORCE_INLINE void send_persistent_next_iter_inc_via_fabric_brisc(const WriterArgs& args, size_t& arg_idx) {
            if (args.persistent_enable == 0) {
                return;
            }
            constexpr uint32_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);
            auto route_id = PacketHeaderPool::allocate_header_n(1);
            volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = PacketHeaderPool::header_table[route_id].first;
            set_unicast_route(
                packet_header,
                static_cast<uint16_t>(args.persistent_dst_chip_id),
                static_cast<uint16_t>(args.persistent_dst_mesh_id),
                1);
            packet_header->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                get_noc_addr(args.persistent_dst_noc_x, args.persistent_dst_noc_y, args.persistent_dst_sem_addr), 1});

            auto fabric_sender =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
            fabric_sender.open();
            fabric_sender.wait_for_empty_write_slot();
            fabric_sender.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(packet_header), packet_header_size_bytes);
            fabric_sender.close();
            noc_async_full_barrier();
        }

        FORCE_INLINE void send_d2h_token_from_cb_brisc(const WriterArgs& args) {
            const uint32_t socket_config_addr = args.socket_config_addr;
            SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
            set_sender_socket_page_size(sender_socket, CTArgs::socket_page_size_bytes);
            const uint32_t write_addr_hi = sender_socket.d2h.data_addr_hi;
            const uint32_t pcie_xy_enc = sender_socket.d2h.pcie_xy_enc;

            socket_reserve_pages(sender_socket, 1);
            cb_wait_front(CTArgs::socket_cb_id, 1);
            const uint32_t read_addr = get_read_ptr(CTArgs::socket_cb_id);

            noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
            noc_async_wide_write_any_len_with_state(
                NOC_INDEX,
                read_addr,
                pcie_xy_enc,
                ((static_cast<uint64_t>(write_addr_hi) << 32) | sender_socket.downstream_fifo_addr) +
                    sender_socket.write_ptr,
                sizeof(uint32_t));
            noc_async_writes_flushed();

            cb_pop_front(CTArgs::socket_cb_id, 1);
            socket_push_pages(sender_socket, 1);
            socket_notify_receiver(sender_socket);
            update_socket_config(sender_socket);
            noc_async_write_barrier();
        }

        FORCE_INLINE void send_d2d_token_from_cb_brisc(const WriterArgs& args) {
            const uint32_t socket_config_addr = args.socket_config_addr;
            SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
            set_sender_socket_page_size(sender_socket, CTArgs::socket_page_size_bytes);

            socket_reserve_pages(sender_socket, 1);
            cb_wait_front(CTArgs::socket_cb_id, 1);
            const uint32_t read_addr = get_read_ptr(CTArgs::socket_cb_id);
            for (uint32_t i = 0; i < sender_socket.num_downstreams; i++) {
                sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, i);
                noc_async_write(
                    read_addr,
                    get_noc_addr(
                        downstream_enc.d2d.downstream_noc_x,
                        downstream_enc.d2d.downstream_noc_y,
                        sender_socket.write_ptr + sender_socket.downstream_fifo_addr),
                    CTArgs::socket_page_size_bytes);
            }
            noc_async_write_barrier();

            cb_pop_front(CTArgs::socket_cb_id, 1);
            socket_push_pages(sender_socket, 1);
            socket_notify_receiver(sender_socket);
            update_socket_config(sender_socket);
        }
#endif

        void impl(const RTArgs& args) {
#if defined(COMPILE_FOR_NCRISC)
            const uint32_t gather_addr =
                (CTArgs::gather_cb_id != 0xFFFFFFFF) ? get_write_ptr(CTArgs::gather_cb_id) : args.gather_addr;
            uint32_t scores_addr = args.scores_addr;
            if constexpr (IsActiveCore && (CTArgs::scores_cb_id != 0xFFFFFFFF)) {
                cb_wait_front(CTArgs::scores_cb_id, CTArgs::scores_num_pages);
                scores_addr = get_read_ptr(CTArgs::scores_cb_id);
            }
            invalidate_l1_cache();

            // Phase 1: per-core local top-K and delivery to the final core.
            if constexpr (IsActiveCore) {
                auto scores_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scores_addr);
                auto indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.indices_addr);

                if constexpr (IsFinalCore) {
                    // Final core: reduce directly into its slots within the gather buffer.
                    auto out_scores = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                        gather_addr + CTArgs::sender_idx * CTArgs::topk_scores_stride);
                    auto out_indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        gather_addr + CTArgs::gather_indices_offset + CTArgs::sender_idx * CTArgs::topk_indices_stride);
                    phase1_reduce_local_topk(scores_ptr, indices_ptr, out_scores, out_indices);
                } else {
                    // Non-final core: reduce to local scratch, then NOC-send to final core.
                    const uint32_t local_scores_addr = gather_addr;
                    const uint32_t local_indices_addr = gather_addr + CTArgs::topk_scores_stride;
                    auto out_scores = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(local_scores_addr);
                    auto out_indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_indices_addr);
                    phase1_reduce_local_topk(scores_ptr, indices_ptr, out_scores, out_indices);
                    phase1_send_topk_to_final(
                        local_scores_addr, local_indices_addr, gather_addr, args.final_noc_x, args.final_noc_y);
                }
            }
            if constexpr (IsActiveCore && (CTArgs::scores_cb_id != 0xFFFFFFFF)) {
                cb_pop_front(CTArgs::scores_cb_id, CTArgs::scores_num_pages);
            }

            // Phase 2: merge per-core top-K arrays into a single device-wide top-K.
            // Output goes to the winner CB in split layout [K scores | K indices].
            // The argmax is global_scores[0] / global_indices[0] (descending order).
            if constexpr (IsFinalCore) {
                auto recv_sem_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(CTArgs::receiver_semaphore_id));
                wait_and_reset_semaphore(recv_sem_ptr, CTArgs::expected_remote_incs);

                const uint32_t winner_addr = get_write_ptr(CTArgs::winner_cb_id);
                auto global_scores = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(winner_addr);
                auto global_indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    winner_addr + CTArgs::topk_scores_stride);
                phase2_merge_global_topk(gather_addr, global_scores, global_indices);

                if constexpr (!CTArgs::mesh_mode) {
                    auto output_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.output_addr);
                    output_ptr[0] = global_indices[0];
                } else {
                    // TODO: mesh top-K reduction stages
                    if constexpr (IsMeshSenderCore && (CTArgs::stage1_sender || CTArgs::stage2_sender)) {
                        write_winner_slot(
                            args.scratch_addr + CTArgs::mesh_local_send_slot_offset,
                            global_scores[0],
                            global_indices[0]);
                        auto local_ready_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            get_semaphore(CTArgs::local_ready_semaphore_id));
                        noc_semaphore_set(local_ready_sem_ptr, 1);
                    }
                }
            }
#elif defined(COMPILE_FOR_BRISC)
            invalidate_l1_cache();
            size_t arg_idx = 0;
            PacketHeaderPool::reset();
            if constexpr (IsFinalCore && CTArgs::socket_mode == 1) {
                send_d2h_token_from_cb_brisc(args);
            } else if constexpr (IsFinalCore && CTArgs::socket_mode == 2) {
                send_d2d_token_from_cb_brisc(args);
            }
            if constexpr (IsFinalCore && IsMeshSenderCore) {
                auto local_ready_sem_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(CTArgs::local_ready_semaphore_id));
                noc_semaphore_wait(local_ready_sem_ptr, 1);
                noc_semaphore_set(local_ready_sem_ptr, 0);

                const BriscMeshSendMetadata metadata = load_mesh_send_metadata(arg_idx);
                const uint32_t local_slot_addr = args.scratch_addr + metadata.local_slot_offset;
                send_mesh_winner_via_fabric_brisc(
                    args.final_noc_x, args.final_noc_y, local_slot_addr, metadata, arg_idx);
            }
            if constexpr (IsFinalCore) {
                send_persistent_next_iter_inc_via_fabric_brisc(args, arg_idx);
            }
#elif defined(COMPILE_FOR_TRISC)
            // TODO: softmax over top-K scores (Phase 4)
            (void)args;
#endif
        }
    };
};  // struct TopKSampling

}  // namespace deepseek_b1_ops
