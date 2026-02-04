// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include <cstdint>
#include <utility>
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/api_common.h"

using namespace tt::tt_fabric::linear::experimental;
using namespace tt::tt_fabric::common::experimental;
using tt::data_movement::common::round_up;

#elif defined(COMPILE_FOR_NCRISC)
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
#include "compute_kernel_api/eltwise_binary.h"
#endif

namespace deepseek_b1_ops {

// Unified kernel for CCL All-Reduce operation
// All-Reduce has two worker cores:
// - Sender core: reads local data and sends to neighbor
// - Receiver core: receives remote data and performs reduction with local data
struct AllReduce {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    // Sender Reader CTArgs (NCRISC on sender core)
    template <uint32_t cb0Id, uint32_t numTiles, uint32_t tensorPageSize, uint32_t coreNocX, uint32_t coreNocY>
    struct SenderReaderCTArgs {
        static constexpr uint32_t cb0_id = cb0Id;
        static constexpr uint32_t num_tiles = numTiles;
        static constexpr uint32_t tensor_page_size = tensorPageSize;
        static constexpr uint32_t core_noc_x = coreNocX;
        static constexpr uint32_t core_noc_y = coreNocY;
    };

    // Sender Writer CTArgs (BRISC on sender core)
    template <
        uint32_t packetHeaderCbId,
        uint32_t packetCbId,
        uint32_t alignment,
        uint32_t inputNumTiles,
        uint32_t pageSizeBytes,
        uint32_t payloadSizeBytes,
        uint32_t dataNocX,
        uint32_t dataNocY,
        uint32_t remoteReceiverNocX,
        uint32_t remoteReceiverNocY,
        uint32_t dstNumHops,
        uint32_t numConnections,
        uint32_t usingPersistentBuffer>
    struct SenderWriterCTArgs {
        static constexpr uint32_t packet_header_cb_id = packetHeaderCbId;
        static constexpr uint32_t packet_cb_id = packetCbId;
        static constexpr uint32_t l1_alignment = alignment;
        static constexpr uint32_t input_num_tiles = inputNumTiles;
        static constexpr uint32_t page_size_bytes = pageSizeBytes;
        static constexpr uint32_t payload_size_bytes = payloadSizeBytes;
        static constexpr uint32_t data_noc_x = dataNocX;
        static constexpr uint32_t data_noc_y = dataNocY;
        static constexpr uint32_t remote_receiver_noc_x = remoteReceiverNocX;
        static constexpr uint32_t remote_receiver_noc_y = remoteReceiverNocY;
        static constexpr uint32_t dst_num_hops = dstNumHops;
        static constexpr uint32_t num_connections = numConnections;
        static constexpr bool using_persistent_buffer = usingPersistentBuffer;
    };

    // Receiver Reader CTArgs (NCRISC on receiver core)
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
        uint32_t usingPersistentBuffer>
    struct ReceiverReaderCTArgs {
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
    };

    // Reduction Compute CTArgs (TRISC on receiver core)
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

    struct SenderReaderArgs {
        uint32_t tensor_address;
    };

    struct SenderWriterArgs {
        uint32_t receiver_base_address;
        uint32_t receive_semaphore_addr;
    };

    struct ReceiverReaderArgs {
        uint32_t sender_semaphore_addr;
    };

    struct ComputeArgs {};

    // ========================================================================
    // Sender Reader Op (NCRISC)
    // ========================================================================
    template <typename CTArgs>
    class SenderReaderOp {
    public:
        void operator()(const SenderReaderArgs& args) {
#if defined(COMPILE_FOR_NCRISC)
            cb_reserve_back(CTArgs::cb0_id, CTArgs::num_tiles);
            const uint32_t l1_write_addr = get_write_ptr(CTArgs::cb0_id);
            uint64_t base_src_addr = get_noc_addr(CTArgs::core_noc_x, CTArgs::core_noc_y, args.tensor_address);
            noc_async_read(base_src_addr, l1_write_addr, CTArgs::num_tiles * CTArgs::tensor_page_size);
            noc_async_read_barrier();
            cb_push_back(CTArgs::cb0_id, CTArgs::num_tiles);
#endif
        }
    };

    // ========================================================================
    // Sender Writer Op (BRISC)
    // ========================================================================
    template <typename CTArgs>
    class SenderWriterOp {
    public:
        void operator()(const SenderWriterArgs& args, size_t& arg_idx) {
#if defined(COMPILE_FOR_BRISC)
            constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

            tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
            open_connections(fabric_connection, CTArgs::num_connections, arg_idx);

            cb_reserve_back(CTArgs::packet_header_cb_id, 1);
            uint32_t packet_header_addr = get_read_ptr(CTArgs::packet_header_cb_id);
            cb_push_back(CTArgs::packet_header_cb_id, 1);

            auto* packet_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_addr);
            fabric_set_unicast_route(fabric_connection, packet_header_ptr, 0);
            packet_header_ptr->to_chip_unicast(CTArgs::dst_num_hops);

            // Wait for receiver to signal it is ready
            if constexpr (!CTArgs::using_persistent_buffer) {
                noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.receive_semaphore_addr), 1);
                noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.receive_semaphore_addr), 0);
            }

            cb_wait_front(CTArgs::packet_cb_id, CTArgs::input_num_tiles);
            uint32_t packet_base_addr = get_read_ptr(CTArgs::packet_cb_id);

            const uint64_t dst_noc_addr =
                get_noc_addr(CTArgs::data_noc_x, CTArgs::data_noc_y, args.receiver_base_address);
            const uint64_t receive_sem_noc_addr =
                get_noc_addr(CTArgs::remote_receiver_noc_x, CTArgs::remote_receiver_noc_y, args.receive_semaphore_addr);

            // Use fused packet API to send data + semaphore increment in a single packet
            packet_header_ptr->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, receive_sem_noc_addr, 1, true},
                align(CTArgs::payload_size_bytes, CTArgs::l1_alignment));

            auto& connection = fabric_connection.get(0).sender;
            connection.wait_for_empty_write_slot();
            connection.send_payload_without_header_non_blocking_from_address(
                packet_base_addr, CTArgs::payload_size_bytes);
            connection.send_payload_flush_blocking_from_address(
                (uint32_t)packet_header_ptr, sizeof(PACKET_HEADER_TYPE));

            cb_pop_front(CTArgs::packet_cb_id, CTArgs::input_num_tiles);

            close_connections(fabric_connection);
#endif
        }
    };

    // ========================================================================
    // Receiver Reader Op (NCRISC)
    // ========================================================================
    template <typename CTArgs>
    class ReceiverReaderOp {
    public:
        void operator()(const ReceiverReaderArgs& args, size_t& arg_idx) {
#if defined(COMPILE_FOR_NCRISC)
            constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);
            constexpr uint8_t sender_num_hops = 1;

            tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
            open_connections(fabric_connection, 1, arg_idx);

            cb_reserve_back(CTArgs::packet_header_cb_id, 1);
            const uint32_t sem_header_addr = get_write_ptr(CTArgs::packet_header_cb_id);
            cb_push_back(CTArgs::packet_header_cb_id, 1);

            const uint64_t sender_sem_noc_addr =
                get_noc_addr(CTArgs::remote_sender_noc_x, CTArgs::remote_sender_noc_y, args.sender_semaphore_addr);

            if constexpr (!CTArgs::using_persistent_buffer) {
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
            cb_reserve_back(CTArgs::cb_in2, CTArgs::num_standard_tiles);
            cb_push_back(CTArgs::cb_in2, CTArgs::num_standard_tiles);

            if constexpr (CTArgs::has_residual) {
                cb_reserve_back(CTArgs::cb_residual, CTArgs::num_standard_tiles);
                cb_push_back(CTArgs::cb_residual, CTArgs::num_standard_tiles);
            }

            // Wait for remote sender to signal data has been written to intermediate tensor
            auto local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.sender_semaphore_addr);
            noc_semaphore_wait(local_semaphore_ptr, 1);
            noc_semaphore_set(local_semaphore_ptr, 0);

            close_connections(fabric_connection);

            // Remote data is now ready, push to compute
            cb_reserve_back(CTArgs::cb_in1, CTArgs::num_standard_tiles);
            cb_push_back(CTArgs::cb_in1, CTArgs::num_standard_tiles);
#endif
        }
    };

    // ========================================================================
    // Reduction Compute Op (TRISC)
    // ========================================================================
    template <typename CTArgs>
    class ComputeOp {
    public:
        void operator()([[maybe_unused]] const ComputeArgs& args) {
#if defined(COMPILE_FOR_TRISC)
            binary_op_init_common(CTArgs::cb_in0, CTArgs::cb_in1, CTArgs::cb_out0);
            add_tiles_init(CTArgs::cb_in0, CTArgs::cb_in1);

            constexpr uint32_t max_dst_tiles = 4;
            constexpr uint32_t num_batches = (CTArgs::num_tiles + max_dst_tiles - 1) / max_dst_tiles;

            if constexpr (CTArgs::has_residual) {
                // Fused residual add: (local + residual) + remote → output
                cb_wait_front(CTArgs::cb_in1, CTArgs::num_tiles);
                cb_wait_front(CTArgs::cb_residual, CTArgs::num_tiles);
                cb_reserve_back(CTArgs::cb_temp, CTArgs::num_tiles);

                // First add: local + residual → temp
                for (uint32_t batch = 0; batch < num_batches; ++batch) {
                    uint32_t start_tile = batch * max_dst_tiles;
                    uint32_t batch_size = (start_tile + max_dst_tiles <= CTArgs::num_tiles)
                                              ? max_dst_tiles
                                              : (CTArgs::num_tiles - start_tile);

                    tile_regs_acquire();
                    for (uint32_t i = 0; i < batch_size; ++i) {
                        add_tiles(CTArgs::cb_in1, CTArgs::cb_residual, start_tile + i, start_tile + i, i);
                    }
                    tile_regs_commit();

                    tile_regs_wait();
                    for (uint32_t i = 0; i < batch_size; ++i) {
                        pack_tile(i, CTArgs::cb_temp, start_tile + i);
                    }
                    tile_regs_release();
                }
                cb_pop_front(CTArgs::cb_in1, CTArgs::num_tiles);
                cb_pop_front(CTArgs::cb_residual, CTArgs::num_tiles);
                cb_push_back(CTArgs::cb_temp, CTArgs::num_tiles);

                // Second add: (local+residual) + remote → output
                cb_wait_front(CTArgs::cb_in0, CTArgs::num_tiles);
                cb_wait_front(CTArgs::cb_temp, CTArgs::num_tiles);
                cb_reserve_back(CTArgs::cb_out0, CTArgs::num_tiles);

                for (uint32_t batch = 0; batch < num_batches; ++batch) {
                    uint32_t start_tile = batch * max_dst_tiles;
                    uint32_t batch_size = (start_tile + max_dst_tiles <= CTArgs::num_tiles)
                                              ? max_dst_tiles
                                              : (CTArgs::num_tiles - start_tile);

                    tile_regs_acquire();
                    for (uint32_t i = 0; i < batch_size; ++i) {
                        add_tiles(CTArgs::cb_temp, CTArgs::cb_in0, start_tile + i, start_tile + i, i);
                    }
                    tile_regs_commit();

                    tile_regs_wait();
                    for (uint32_t i = 0; i < batch_size; ++i) {
                        pack_tile(i, CTArgs::cb_out0, start_tile + i);
                    }
                    tile_regs_release();
                }
                cb_pop_front(CTArgs::cb_in0, CTArgs::num_tiles);
                cb_pop_front(CTArgs::cb_temp, CTArgs::num_tiles);
                cb_push_back(CTArgs::cb_out0, CTArgs::num_tiles);
            } else {
                // Simple all-reduce: local + remote → output
                cb_wait_front(CTArgs::cb_in0, CTArgs::num_tiles);
                cb_wait_front(CTArgs::cb_in1, CTArgs::num_tiles);
                cb_reserve_back(CTArgs::cb_out0, CTArgs::num_tiles);

                for (uint32_t batch = 0; batch < num_batches; ++batch) {
                    uint32_t start_tile = batch * max_dst_tiles;
                    uint32_t batch_size = (start_tile + max_dst_tiles <= CTArgs::num_tiles)
                                              ? max_dst_tiles
                                              : (CTArgs::num_tiles - start_tile);

                    tile_regs_acquire();
                    for (uint32_t i = 0; i < batch_size; ++i) {
                        add_tiles(CTArgs::cb_in0, CTArgs::cb_in1, start_tile + i, start_tile + i, i);
                    }
                    tile_regs_commit();

                    tile_regs_wait();
                    for (uint32_t i = 0; i < batch_size; ++i) {
                        pack_tile(i, CTArgs::cb_out0, start_tile + i);
                    }
                    tile_regs_release();
                }
                cb_pop_front(CTArgs::cb_in0, CTArgs::num_tiles);
                cb_pop_front(CTArgs::cb_in1, CTArgs::num_tiles);
                cb_push_back(CTArgs::cb_out0, CTArgs::num_tiles);
            }
#endif
        }
    };

};  // struct AllReduce

}  // namespace deepseek_b1_ops
