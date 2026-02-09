// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * Unified Reduce-to-Root B1 Operation
 *
 * This kernel performs multi-device reduce-to-one operation where:
 * - LEAFs (rows 0, 3) send data to ROOT3/ROOT2/ROOT1
 * - ROOT3 accumulates and sends to ROOT2/ROOT1
 * - ROOT2 accumulates and sends to ROOT1
 * - ROOT1 accumulates all data and gathers to output core
 *
 * Device Roles:
 *   MESH_LEAF = 0: Send data, no compute
 *   MESH_ROOT3 = 1: Receive from LEAF, accumulate, send to ROOT2/ROOT1
 *   MESH_ROOT2 = 2: Receive from LEAF + ROOT3, accumulate, send to ROOT1
 *   MESH_ROOT1 = 3: Receive from LEAF + ROOT3 + ROOT2, accumulate, gather to output
 */

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_BRISC)
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"

#elif defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"

#elif defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#endif

namespace deepseek_b1_ops {

// Device roles
constexpr uint32_t MESH_LEAF = 0;
constexpr uint32_t MESH_ROOT3 = 1;
constexpr uint32_t MESH_ROOT2 = 2;
constexpr uint32_t MESH_ROOT1 = 3;

// Unified kernel for Reduce-to-Root B1 operation
struct ReduceToOneB1 {
    // ========================================================================
    // Compile-time args structs - different per RISC
    // ========================================================================

    // Reader (NCRISC) compile-time args
    template <
        uint32_t deviceRole,
        uint32_t numTiles,
        uint32_t localCb,
        uint32_t receivedCbR1,
        uint32_t receivedCbR2,
        uint32_t receivedCbR3,
        uint32_t isFabricCore>
    struct ReaderCTArgs {
        static constexpr uint32_t device_role = deviceRole;
        static constexpr uint32_t num_tiles = numTiles;
        static constexpr uint32_t local_cb = localCb;
        static constexpr uint32_t received_cb_r1 = receivedCbR1;
        static constexpr uint32_t received_cb_r2 = receivedCbR2;
        static constexpr uint32_t received_cb_r3 = receivedCbR3;
        static constexpr uint32_t is_fabric_core = isFabricCore;
    };

    // Writer (BRISC) compile-time args
    template <
        uint32_t deviceRole,
        uint32_t numTiles,
        uint32_t payloadSizeBytes,
        uint32_t localCb,
        uint32_t scratchCb,
        uint32_t packetCb,
        uint32_t numHops,
        uint32_t dstFabricNodeChipId,
        uint32_t dstFabricNodeMeshId,
        uint32_t outputCoreNocX,
        uint32_t outputCoreNocY,
        uint32_t numWorkers,
        uint32_t slotSizeBytes,
        uint32_t isFabricCore>
    struct WriterCTArgs {
        static constexpr uint32_t device_role = deviceRole;
        static constexpr uint32_t num_tiles = numTiles;
        static constexpr uint32_t payload_size_bytes = payloadSizeBytes;
        static constexpr uint32_t local_cb = localCb;
        static constexpr uint32_t scratch_cb = scratchCb;
        static constexpr uint32_t packet_cb = packetCb;
        static constexpr uint32_t num_hops = numHops;
        static constexpr uint32_t dst_fabric_node_chip_id = dstFabricNodeChipId;
        static constexpr uint32_t dst_fabric_node_mesh_id = dstFabricNodeMeshId;
        static constexpr uint32_t output_core_noc_x = outputCoreNocX;
        static constexpr uint32_t output_core_noc_y = outputCoreNocY;
        static constexpr uint32_t num_workers = numWorkers;
        static constexpr uint32_t slot_size_bytes = slotSizeBytes;
        static constexpr uint32_t is_fabric_core = isFabricCore;
        // packet_header_size_bytes derived from sizeof(PACKET_HEADER_TYPE) in kernel
    };

    // Compute (TRISC) compile-time args
    template <
        uint32_t deviceRole,
        uint32_t numTiles,
        uint32_t localCb,
        uint32_t receivedCbR1,
        uint32_t receivedCbR2,
        uint32_t receivedCbR3,
        uint32_t outputCb,
        uint32_t scratchCb,
        uint32_t isFabricCore>
    struct ComputeCTArgs {
        static constexpr uint32_t device_role = deviceRole;
        static constexpr uint32_t num_tiles = numTiles;
        static constexpr uint32_t local_cb = localCb;
        static constexpr uint32_t received_cb_r1 = receivedCbR1;
        static constexpr uint32_t received_cb_r2 = receivedCbR2;
        static constexpr uint32_t received_cb_r3 = receivedCbR3;
        static constexpr uint32_t output_cb = outputCb;
        static constexpr uint32_t scratch_cb = scratchCb;
        static constexpr uint32_t is_fabric_core = isFabricCore;
    };

    // ========================================================================
    // Runtime args structs - different per RISC and core type
    // ========================================================================

    // Reader (NCRISC) runtime args - worker cores only
    struct ReaderArgs {
        uint32_t recv_sem_round1;
        uint32_t recv_sem_round2;
        uint32_t recv_sem_round3;
    };

    // Writer (BRISC) runtime args for worker cores
    struct WorkerWriterArgs {
        uint32_t fabric_core_noc_x;
        uint32_t fabric_core_noc_y;
        uint32_t my_slot_idx;
        uint32_t worker_sem_id;
        uint32_t dst_l1_addr;
        uint32_t dst_sem_addr;
        uint32_t output_base_addr;
        uint32_t shard_idx;
    };

    // Writer (BRISC) runtime args for fabric cores - handled dynamically via build_from_args

    // Compute (TRISC) - no runtime args
    struct ComputeArgs {};

    // Type selection based on RISC
    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WorkerWriterArgs, ComputeArgs>;

    // ========================================================================
    // Op implementation
    // ========================================================================
    // SkipLocalCbPush: When true, skip cb_reserve_back/cb_push_back on local_cb.
    //                  Use this when fused with a previous op that already pushed to local_cb.
    template <typename CTArgs, bool IsWorkerCore, bool SkipLocalCbPush = false>
    class Op {
    public:
        void operator()(const RTArgs& args) { impl(args); }

    private:
#if defined(COMPILE_FOR_BRISC)
        // Template helper for routing - allows if constexpr to work for both 1D and 2D fabric
        template <typename packet_header_t>
        static FORCE_INLINE void set_unicast_route(
            volatile tt_l1_ptr packet_header_t* header, uint16_t dst_dev_id, uint16_t dst_mesh_id, uint16_t num_hops) {
            if constexpr (std::is_same_v<packet_header_t, tt::tt_fabric::HybridMeshPacketHeader>) {
                fabric_set_unicast_route(header, dst_dev_id, dst_mesh_id);
            } else {
                fabric_set_unicast_route<false>(header, num_hops);
            }
        }
#endif

        void impl([[maybe_unused]] const RTArgs& args) {
            // Early return if this core is not a reduce core (worker or fabric)
            // Note: IsWorkerCore should be true for both worker cores AND fabric cores
            // The is_fabric_core compile-time arg distinguishes between the two
            if constexpr (!IsWorkerCore) {
                return;
            }

#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC - Reader: receives data from fabric via semaphore waits
            // ================================================================
            if constexpr (CTArgs::is_fabric_core) {
                // Fabric cores have no reader work
                return;
            }

            if constexpr (
                CTArgs::device_role == MESH_ROOT3 || CTArgs::device_role == MESH_ROOT2 ||
                CTArgs::device_role == MESH_ROOT1) {
                // Push local data to compute (local_cb is in-place on input shard)
                // Skip this when fused - previous op already pushed to local_cb
                if constexpr (!SkipLocalCbPush) {
                    cb_reserve_back(CTArgs::local_cb, CTArgs::num_tiles);
                    cb_push_back(CTArgs::local_cb, CTArgs::num_tiles);
                }

                // Round 1: Wait for shard from LEAF
                cb_reserve_back(CTArgs::received_cb_r1, CTArgs::num_tiles);
                volatile tt_l1_ptr uint32_t* recv_sem1_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.recv_sem_round1);
                noc_semaphore_wait(recv_sem1_ptr, 1);
                noc_semaphore_set(recv_sem1_ptr, 0);
                cb_push_back(CTArgs::received_cb_r1, CTArgs::num_tiles);
            }

            if constexpr (CTArgs::device_role == MESH_ROOT2 || CTArgs::device_role == MESH_ROOT1) {
                // Round 2: Wait for result from ROOT3
                cb_reserve_back(CTArgs::received_cb_r2, CTArgs::num_tiles);
                volatile tt_l1_ptr uint32_t* recv_sem2_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.recv_sem_round2);
                noc_semaphore_wait(recv_sem2_ptr, 1);
                noc_semaphore_set(recv_sem2_ptr, 0);
                cb_push_back(CTArgs::received_cb_r2, CTArgs::num_tiles);
            }

            if constexpr (CTArgs::device_role == MESH_ROOT1) {
                // Round 3: Wait for result from ROOT2
                cb_reserve_back(CTArgs::received_cb_r3, CTArgs::num_tiles);
                volatile tt_l1_ptr uint32_t* recv_sem3_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.recv_sem_round3);
                noc_semaphore_wait(recv_sem3_ptr, 1);
                noc_semaphore_set(recv_sem3_ptr, 0);
                cb_push_back(CTArgs::received_cb_r3, CTArgs::num_tiles);
            }

#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC - Writer: sends data via fabric or NOC
            // ================================================================
            constexpr uint32_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

            if constexpr (CTArgs::is_fabric_core) {
                // Fabric core: forward worker packets via fabric
                if constexpr (CTArgs::device_role == MESH_ROOT1) {
                    // ROOT1 fabric cores have nothing to do
                    return;
                }

                // Read worker semaphore IDs from runtime args
                size_t arg_idx = 0;
                uint32_t worker_sem_addr[CTArgs::num_workers];
                for (uint32_t i = 0; i < CTArgs::num_workers; i++) {
                    uint32_t sem_id = get_arg_val<uint32_t>(arg_idx++);
                    worker_sem_addr[i] = get_semaphore(sem_id);
                }

                const uint32_t packet_buffer_addr = get_write_ptr(CTArgs::packet_cb);

                // Build fabric connection from runtime args
                auto fabric_sender =
                    tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
                fabric_sender.open();

                // Forward worker packets
                uint32_t slot_base = packet_buffer_addr;
                for (uint32_t worker = 0; worker < CTArgs::num_workers; worker++) {
                    uint32_t worker_header_addr = slot_base;
                    uint32_t worker_payload_addr = slot_base + packet_header_size_bytes;

                    volatile tt_l1_ptr uint32_t* worker_sem_ptr =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(worker_sem_addr[worker]);
                    noc_semaphore_wait(worker_sem_ptr, 1);
                    noc_semaphore_set(worker_sem_ptr, 0);

                    fabric_sender.wait_for_empty_write_slot();
                    fabric_sender.send_payload_without_header_non_blocking_from_address(
                        worker_payload_addr, CTArgs::payload_size_bytes);
                    fabric_sender.send_payload_flush_blocking_from_address(
                        worker_header_addr, packet_header_size_bytes);

                    slot_base += CTArgs::slot_size_bytes;
                }

                fabric_sender.close();
                noc_async_write_barrier();
                return;
            }

            // Worker core logic
            const uint32_t my_noc_x = my_x[0];
            const uint32_t my_noc_y = my_y[0];

            // ROOT1: gather final results to output core
            if constexpr (CTArgs::device_role == MESH_ROOT1) {
                uint32_t dst_addr = args.output_base_addr + args.shard_idx * CTArgs::payload_size_bytes;
                uint64_t dst_noc_addr = get_noc_addr(CTArgs::output_core_noc_x, CTArgs::output_core_noc_y, dst_addr);

                // Wait for compute to finish
                cb_wait_front(CTArgs::scratch_cb, CTArgs::num_tiles);
                uint32_t src_addr = get_read_ptr(CTArgs::scratch_cb);

                noc_async_write(src_addr, dst_noc_addr, CTArgs::payload_size_bytes);
                noc_async_write_barrier();
                cb_pop_front(CTArgs::scratch_cb, CTArgs::num_tiles);
                return;
            }
            // Non-ROOT1: send via fabric
            const uint32_t packet_buffer_addr = get_write_ptr(CTArgs::packet_cb);
            const uint32_t arrival_sem_addr = get_semaphore(args.worker_sem_id);

            // Allocate packet header
            auto route_id = PacketHeaderPool::allocate_header_n(1);
            volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = PacketHeaderPool::header_table[route_id].first;

            // Set routing - works for both 1D (num_hops) and 2D (dst_dev_id, dst_mesh_id) fabric
            set_unicast_route(
                packet_header,
                static_cast<uint16_t>(CTArgs::dst_fabric_node_chip_id),
                static_cast<uint16_t>(CTArgs::dst_fabric_node_mesh_id),
                static_cast<uint16_t>(CTArgs::num_hops));

            // Set up fused write + atomic inc
            uint64_t dst_noc_addr = get_noc_addr(my_noc_x, my_noc_y, args.dst_l1_addr);
            uint64_t dst_sem_noc_addr = get_noc_addr(my_noc_x, my_noc_y, args.dst_sem_addr);
            packet_header->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, dst_sem_noc_addr, 1, false},
                CTArgs::payload_size_bytes);

            // Calculate slot in fabric core's packet buffer
            uint32_t slot_offset = args.my_slot_idx * CTArgs::slot_size_bytes;
            uint32_t header_dest_addr = packet_buffer_addr + slot_offset;
            uint32_t payload_dest_addr = header_dest_addr + packet_header_size_bytes;

            uint64_t header_noc_addr = get_noc_addr(args.fabric_core_noc_x, args.fabric_core_noc_y, header_dest_addr);
            uint64_t payload_noc_addr = get_noc_addr(args.fabric_core_noc_x, args.fabric_core_noc_y, payload_dest_addr);

            // Source CB: LEAF uses local_cb, others use scratch_cb
            constexpr uint32_t source_cb = (CTArgs::device_role == MESH_LEAF) ? CTArgs::local_cb : CTArgs::scratch_cb;

            // Wait for data
            // - LEAF: when fused (SkipLocalCbPush=true), previous op pushed to local_cb, so we wait
            //         when standalone (SkipLocalCbPush=false), data is already in-place, no wait needed
            // - Others: always wait for compute to finish writing to scratch_cb
            if constexpr (CTArgs::device_role == MESH_LEAF) {
                if constexpr (SkipLocalCbPush) {
                    cb_wait_front(source_cb, CTArgs::num_tiles);
                }
            } else {
                cb_wait_front(source_cb, CTArgs::num_tiles);
            }
            uint32_t data_addr = get_read_ptr(source_cb);

            // Send header and payload to fabric core
            noc_async_write(reinterpret_cast<uint32_t>(packet_header), header_noc_addr, packet_header_size_bytes);
            noc_async_write(data_addr, payload_noc_addr, CTArgs::payload_size_bytes);

            // Signal fabric core
            uint64_t arrival_sem_noc_addr =
                get_noc_addr(args.fabric_core_noc_x, args.fabric_core_noc_y, arrival_sem_addr);
            noc_semaphore_inc(arrival_sem_noc_addr, 1);

            if constexpr (CTArgs::device_role != MESH_LEAF) {
                cb_pop_front(source_cb, CTArgs::num_tiles);
            }

            noc_async_write_barrier();
            noc_async_atomic_barrier();

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC - Compute: performs reduction
            // ================================================================
            if constexpr (CTArgs::is_fabric_core || CTArgs::device_role == MESH_LEAF) {
                // Fabric cores and LEAFs have no compute
                return;
            }

            // Initialize for binary operations
            binary_op_init_common(CTArgs::local_cb, CTArgs::received_cb_r1, CTArgs::scratch_cb);

            // Load local tiles to dest
            copy_tile_to_dst_init_short(CTArgs::local_cb);
            cb_wait_front(CTArgs::local_cb, CTArgs::num_tiles);
            acquire_dst();
            for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                copy_tile(CTArgs::local_cb, i, i);
            }
            cb_pop_front(CTArgs::local_cb, CTArgs::num_tiles);

            // Accumulate from received_cb_r1 (LEAF data)
            binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CTArgs::received_cb_r1);
            cb_wait_front(CTArgs::received_cb_r1, CTArgs::num_tiles);
            for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CTArgs::received_cb_r1, i, i);
            }
            cb_pop_front(CTArgs::received_cb_r1, CTArgs::num_tiles);

            if constexpr (CTArgs::device_role == MESH_ROOT2 || CTArgs::device_role == MESH_ROOT1) {
                // Accumulate from received_cb_r2 (ROOT3 data)
                cb_wait_front(CTArgs::received_cb_r2, CTArgs::num_tiles);
                for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                    binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                        CTArgs::received_cb_r2, i, i);
                }
                cb_pop_front(CTArgs::received_cb_r2, CTArgs::num_tiles);
            }

            if constexpr (CTArgs::device_role == MESH_ROOT1) {
                // Accumulate from received_cb_r3 (ROOT2 data)
                cb_wait_front(CTArgs::received_cb_r3, CTArgs::num_tiles);
                for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                    binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                        CTArgs::received_cb_r3, i, i);
                }
                cb_pop_front(CTArgs::received_cb_r3, CTArgs::num_tiles);
            }

            // Pack result to scratch_cb
            cb_reserve_back(CTArgs::scratch_cb, CTArgs::num_tiles);
            for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                pack_tile(i, CTArgs::scratch_cb, i);
            }
            release_dst();
            cb_push_back(CTArgs::scratch_cb, CTArgs::num_tiles);
#endif
        }
    };  // class Op

};  // struct ReduceToOneB1

}  // namespace deepseek_b1_ops
