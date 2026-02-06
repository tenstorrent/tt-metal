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
#endif

namespace deepseek_b1_ops {

// ============================================================================
// CCL All-Reduce Sender Operations
//
// Sender core functionality:
// - NCRISC (Reader): Reads local tensor data into CB
// - BRISC (Writer): Sends data to remote device via fabric
// - TRISC: No-op (compute runs on receiver core)
// ============================================================================
struct AllReduceSender {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    // Reader CTArgs (NCRISC)
    template <uint32_t cb0Id, uint32_t numTiles, uint32_t tensorPageSize, uint32_t coreNocX, uint32_t coreNocY>
    struct ReaderCTArgs {
        static constexpr uint32_t cb0_id = cb0Id;
        static constexpr uint32_t num_tiles = numTiles;
        static constexpr uint32_t tensor_page_size = tensorPageSize;
        static constexpr uint32_t core_noc_x = coreNocX;
        static constexpr uint32_t core_noc_y = coreNocY;
    };

    // Writer CTArgs (BRISC)
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
    struct WriterCTArgs {
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

    // ========================================================================
    // Runtime args structs
    // ========================================================================

    // NCRISC reader args
    struct ReaderArgs {
        uint32_t tensor_address;
    };

    // BRISC writer args
    struct WriterArgs {
        uint32_t receiver_base_address;
        uint32_t receive_semaphore_addr;
    };

    // TRISC compute args (no-op for sender)
    struct ComputeArgs {};

    // Select args type based on current RISC
    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // ========================================================================
    // Op - unified sender operation
    //
    // ReaderCT: compile-time args for NCRISC reader
    // WriterCT: compile-time args for BRISC writer
    // ========================================================================
    template <typename ReaderCT, typename WriterCT>
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
            // NCRISC (Reader) - reads local tensor data into CB
            // ================================================================
            cb_reserve_back(ReaderCT::cb0_id, ReaderCT::num_tiles);
            const uint32_t l1_write_addr = get_write_ptr(ReaderCT::cb0_id);
            uint64_t base_src_addr = get_noc_addr(ReaderCT::core_noc_x, ReaderCT::core_noc_y, args.tensor_address);
            noc_async_read(base_src_addr, l1_write_addr, ReaderCT::num_tiles * ReaderCT::tensor_page_size);
            noc_async_read_barrier();
            cb_push_back(ReaderCT::cb0_id, ReaderCT::num_tiles);

#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC (Writer) - sends data to remote device via fabric
            // ================================================================
            constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

            tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
            open_connections(fabric_connection, WriterCT::num_connections, fabric_arg_idx);

            cb_reserve_back(WriterCT::packet_header_cb_id, 1);
            uint32_t packet_header_addr = get_read_ptr(WriterCT::packet_header_cb_id);
            cb_push_back(WriterCT::packet_header_cb_id, 1);

            auto* packet_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_addr);
            fabric_set_unicast_route(fabric_connection, packet_header_ptr, 0);
            packet_header_ptr->to_chip_unicast(WriterCT::dst_num_hops);

            // Wait for receiver to signal it is ready (if not using persistent buffer)
            if constexpr (!WriterCT::using_persistent_buffer) {
                noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.receive_semaphore_addr), 1);
                noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.receive_semaphore_addr), 0);
            }

            cb_wait_front(WriterCT::packet_cb_id, WriterCT::input_num_tiles);
            uint32_t packet_base_addr = get_read_ptr(WriterCT::packet_cb_id);

            const uint64_t dst_noc_addr =
                get_noc_addr(WriterCT::data_noc_x, WriterCT::data_noc_y, args.receiver_base_address);
            const uint64_t receive_sem_noc_addr = get_noc_addr(
                WriterCT::remote_receiver_noc_x, WriterCT::remote_receiver_noc_y, args.receive_semaphore_addr);

            // Use fused packet API to send data + semaphore increment in a single packet
            packet_header_ptr->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, receive_sem_noc_addr, 1, true},
                align(WriterCT::payload_size_bytes, WriterCT::l1_alignment));

            auto& connection = fabric_connection.get(0).sender;
            connection.wait_for_empty_write_slot();
            connection.send_payload_without_header_non_blocking_from_address(
                packet_base_addr, WriterCT::payload_size_bytes);
            connection.send_payload_flush_blocking_from_address(
                (uint32_t)packet_header_ptr, sizeof(PACKET_HEADER_TYPE));

            cb_pop_front(WriterCT::packet_cb_id, WriterCT::input_num_tiles);

            close_connections(fabric_connection);

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC - No-op (compute runs on receiver core)
            // ================================================================
#endif
        }
    };

};  // struct AllReduceSender

}  // namespace deepseek_b1_ops
