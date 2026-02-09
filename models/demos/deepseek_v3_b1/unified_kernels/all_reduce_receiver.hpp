// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include <cstdint>
#include <utility>
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/api_common.h"

using namespace tt::tt_fabric::linear::experimental;
using namespace tt::tt_fabric::common::experimental;

#elif defined(COMPILE_FOR_TRISC)
#include "api/compute/eltwise_binary.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// CCL All-Reduce Receiver Operations
//
// Receiver core functionality:
// - NCRISC (Reader): Waits for remote data, pushes to compute
// - BRISC: No-op (writer runs on sender core)
// - TRISC (Compute): Performs reduction (local + remote → output)
// ============================================================================
struct AllReduceReceiver {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    // Reader CTArgs (NCRISC)
    template <
        uint32_t packetHeaderCbId,
        uint32_t cbIn1,
        uint32_t alignment,
        uint32_t cbIn2,
        uint32_t remoteSenderNocX,
        uint32_t remoteSenderNocY,
        uint32_t numStandardTiles,
        uint32_t cbResidual,
        uint32_t hasResidual,
        uint32_t usingPersistentBuffer,
        uint32_t skipLocalPush = 0>  // Skip cb_reserve/push on cb_in2 when fused with gather
    struct ReaderCTArgs {
        static constexpr uint32_t packet_header_cb_id = packetHeaderCbId;
        static constexpr uint32_t cb_in1 = cbIn1;
        static constexpr uint32_t l1_alignment = alignment;
        static constexpr uint32_t cb_in2 = cbIn2;
        static constexpr uint32_t remote_sender_noc_x = remoteSenderNocX;
        static constexpr uint32_t remote_sender_noc_y = remoteSenderNocY;
        static constexpr uint32_t num_standard_tiles = numStandardTiles;
        static constexpr uint32_t cb_residual = cbResidual;
        static constexpr bool has_residual = hasResidual;
        static constexpr bool using_persistent_buffer = usingPersistentBuffer;
        static constexpr bool skip_local_push = skipLocalPush;
    };

    // Compute CTArgs (TRISC)
    template <
        uint32_t cbIn0,
        uint32_t cbIn1,
        uint32_t cbOut0,
        uint32_t cbResidual,
        uint32_t cbTemp,
        uint32_t hasResidual,
        uint32_t numTiles>
    struct ComputeCTArgs {
        static constexpr uint32_t cb_in0 = cbIn0;
        static constexpr uint32_t cb_in1 = cbIn1;
        static constexpr uint32_t cb_out0 = cbOut0;
        static constexpr uint32_t cb_residual = cbResidual;
        static constexpr uint32_t cb_temp = cbTemp;
        static constexpr bool has_residual = hasResidual;
        static constexpr uint32_t num_tiles = numTiles;
    };

    // ========================================================================
    // Runtime args structs
    // ========================================================================

    // NCRISC reader args
    struct ReaderArgs {
        uint32_t sender_semaphore_addr;
    };

    // BRISC writer args (no-op for receiver)
    struct WriterArgs {};

    // TRISC compute args
    struct ComputeArgs {};

    // Select args type based on current RISC
    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // ========================================================================
    // Op - unified receiver operation
    //
    // ReaderCT: compile-time args for NCRISC reader
    // ComputeCT: compile-time args for TRISC compute
    // ========================================================================
    template <typename ReaderCT, typename ComputeCT>
    class Op {
    public:
        void operator()(const RTArgs& args) {
            size_t unused = 0;
            impl(args, unused);
        }

        void operator()(const RTArgs& args, size_t& fabric_arg_idx) { impl(args, fabric_arg_idx); }

    private:
        void impl([[maybe_unused]] const RTArgs& args, [[maybe_unused]] size_t& fabric_arg_idx) {
#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC (Reader) - waits for remote data, pushes to compute
            // ================================================================
            constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);
            constexpr uint8_t sender_num_hops = 1;

            tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
            open_connections(fabric_connection, 1, fabric_arg_idx);

            cb_reserve_back(ReaderCT::packet_header_cb_id, 1);
            const uint32_t sem_header_addr = get_write_ptr(ReaderCT::packet_header_cb_id);
            cb_push_back(ReaderCT::packet_header_cb_id, 1);

            const uint64_t sender_sem_noc_addr =
                get_noc_addr(ReaderCT::remote_sender_noc_x, ReaderCT::remote_sender_noc_y, args.sender_semaphore_addr);

            // Signal sender that receiver is ready (if not using persistent buffer)
            if constexpr (!ReaderCT::using_persistent_buffer) {
                auto* sem_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(sem_header_addr);
                fabric_set_unicast_route(fabric_connection, sem_header_ptr, 0);
                sem_header_ptr->to_chip_unicast(sender_num_hops);

                sem_header_ptr->to_noc_unicast_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sender_sem_noc_addr, 1});

                auto& connection = fabric_connection.get(0).sender;
                connection.wait_for_empty_write_slot();
                connection.send_payload_flush_blocking_from_address((uint32_t)sem_header_ptr, packet_header_size_bytes);
            }

            // Push local and residual tiles to compute immediately (they're ready)
            // Skip local push if data is already in CB (e.g., from preceding gather operation)
            if constexpr (!ReaderCT::skip_local_push) {
                cb_reserve_back(ReaderCT::cb_in2, ReaderCT::num_standard_tiles);
                cb_push_back(ReaderCT::cb_in2, ReaderCT::num_standard_tiles);
            }

            if constexpr (ReaderCT::has_residual) {
                cb_reserve_back(ReaderCT::cb_residual, ReaderCT::num_standard_tiles);
                cb_push_back(ReaderCT::cb_residual, ReaderCT::num_standard_tiles);
            }

            // Wait for remote sender to signal data has been written to intermediate tensor
            auto local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.sender_semaphore_addr);
            noc_semaphore_wait(local_semaphore_ptr, 1);
            noc_semaphore_set(local_semaphore_ptr, 0);

            close_connections(fabric_connection);

            // Remote data is now ready, push to compute
            cb_reserve_back(ReaderCT::cb_in1, ReaderCT::num_standard_tiles);
            cb_push_back(ReaderCT::cb_in1, ReaderCT::num_standard_tiles);

#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC - No-op (writer runs on sender core)
            // ================================================================

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC (Compute) - performs reduction: local + remote → output
            // ================================================================
            binary_op_init_common(ComputeCT::cb_in0, ComputeCT::cb_in1, ComputeCT::cb_out0);
            add_tiles_init(ComputeCT::cb_in0, ComputeCT::cb_in1);

            constexpr uint32_t max_dst_tiles = 4;
            constexpr uint32_t num_batches = (ComputeCT::num_tiles + max_dst_tiles - 1) / max_dst_tiles;

            if constexpr (ComputeCT::has_residual) {
                // Fused residual add: (local + residual) + remote → output
                cb_wait_front(ComputeCT::cb_in1, ComputeCT::num_tiles);
                cb_wait_front(ComputeCT::cb_residual, ComputeCT::num_tiles);
                cb_reserve_back(ComputeCT::cb_temp, ComputeCT::num_tiles);

                // First add: local + residual → temp
                for (uint32_t batch = 0; batch < num_batches; ++batch) {
                    uint32_t start_tile = batch * max_dst_tiles;
                    uint32_t batch_size = (start_tile + max_dst_tiles <= ComputeCT::num_tiles)
                                              ? max_dst_tiles
                                              : (ComputeCT::num_tiles - start_tile);

                    tile_regs_acquire();
                    for (uint32_t i = 0; i < batch_size; ++i) {
                        add_tiles(ComputeCT::cb_in1, ComputeCT::cb_residual, start_tile + i, start_tile + i, i);
                    }
                    tile_regs_commit();

                    tile_regs_wait();
                    for (uint32_t i = 0; i < batch_size; ++i) {
                        pack_tile(i, ComputeCT::cb_temp, start_tile + i);
                    }
                    tile_regs_release();
                }
                cb_pop_front(ComputeCT::cb_in1, ComputeCT::num_tiles);
                cb_pop_front(ComputeCT::cb_residual, ComputeCT::num_tiles);
                cb_push_back(ComputeCT::cb_temp, ComputeCT::num_tiles);

                // Second add: (local+residual) + remote → output
                cb_wait_front(ComputeCT::cb_in0, ComputeCT::num_tiles);
                cb_wait_front(ComputeCT::cb_temp, ComputeCT::num_tiles);
                cb_reserve_back(ComputeCT::cb_out0, ComputeCT::num_tiles);

                for (uint32_t batch = 0; batch < num_batches; ++batch) {
                    uint32_t start_tile = batch * max_dst_tiles;
                    uint32_t batch_size = (start_tile + max_dst_tiles <= ComputeCT::num_tiles)
                                              ? max_dst_tiles
                                              : (ComputeCT::num_tiles - start_tile);

                    tile_regs_acquire();
                    for (uint32_t i = 0; i < batch_size; ++i) {
                        add_tiles(ComputeCT::cb_temp, ComputeCT::cb_in0, start_tile + i, start_tile + i, i);
                    }
                    tile_regs_commit();

                    tile_regs_wait();
                    for (uint32_t i = 0; i < batch_size; ++i) {
                        pack_tile(i, ComputeCT::cb_out0, start_tile + i);
                    }
                    tile_regs_release();
                }
                cb_pop_front(ComputeCT::cb_in0, ComputeCT::num_tiles);
                cb_pop_front(ComputeCT::cb_temp, ComputeCT::num_tiles);
                cb_push_back(ComputeCT::cb_out0, ComputeCT::num_tiles);
            } else {
                // Simple all-reduce: local + remote → output
                cb_wait_front(ComputeCT::cb_in0, ComputeCT::num_tiles);
                cb_wait_front(ComputeCT::cb_in1, ComputeCT::num_tiles);
                cb_reserve_back(ComputeCT::cb_out0, ComputeCT::num_tiles);

                for (uint32_t batch = 0; batch < num_batches; ++batch) {
                    uint32_t start_tile = batch * max_dst_tiles;
                    uint32_t batch_size = (start_tile + max_dst_tiles <= ComputeCT::num_tiles)
                                              ? max_dst_tiles
                                              : (ComputeCT::num_tiles - start_tile);

                    tile_regs_acquire();
                    for (uint32_t i = 0; i < batch_size; ++i) {
                        add_tiles(ComputeCT::cb_in0, ComputeCT::cb_in1, start_tile + i, start_tile + i, i);
                    }
                    tile_regs_commit();

                    tile_regs_wait();
                    for (uint32_t i = 0; i < batch_size; ++i) {
                        pack_tile(i, ComputeCT::cb_out0, start_tile + i);
                    }
                    tile_regs_release();
                }
                cb_pop_front(ComputeCT::cb_in0, ComputeCT::num_tiles);
                cb_pop_front(ComputeCT::cb_in1, ComputeCT::num_tiles);
                cb_push_back(ComputeCT::cb_out0, ComputeCT::num_tiles);
            }
#endif
        }
    };

};  // struct AllReduceReceiver

}  // namespace deepseek_b1_ops
