// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
#include "api/socket_api.h"
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
#include "api/compute/experimental/pack_block.h"
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
    template <uint32_t deviceRole, uint32_t numTiles, uint32_t localCb, uint32_t receivedCb, uint32_t isFabricCore>
    struct ReaderCTArgs {
        static constexpr uint32_t device_role = deviceRole;
        static constexpr uint32_t num_tiles = numTiles;
        static constexpr uint32_t local_cb = localCb;
        static constexpr uint32_t received_cb = receivedCb;
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
        uint32_t isFabricCore,
        bool enableDownstreamSocket,
        uint32_t fabricRtArgBase = 0,
        uint32_t totalNumWorkers = 0,
        uint32_t aggOutputSizeBytes = 0,
        uint32_t persistentFabricRtArgBase = 0,
        uint32_t persistentFabricSignalEnable = 0,
        uint32_t forwardMetadataSizeBytes = 0>
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
        static constexpr uint32_t is_fabric_core = isFabricCore;
        static constexpr uint32_t fabric_rt_arg_base = fabricRtArgBase;
        static constexpr uint32_t total_num_workers = totalNumWorkers;
        static constexpr uint32_t agg_output_size_bytes = aggOutputSizeBytes;
        static constexpr bool enable_downstream_socket = enableDownstreamSocket;
        static constexpr uint32_t persistent_fabric_rt_arg_base = persistentFabricRtArgBase;
        static constexpr uint32_t persistent_fabric_signal_enable = persistentFabricSignalEnable;
        static constexpr uint32_t forward_metadata_size_bytes = forwardMetadataSizeBytes;

        static constexpr uint32_t compute_all_sent_mask(uint32_t slots) {
            return (slots == 32) ? 0xFFFF'FFFFu : ((1u << slots) - 1u);
        }
        static constexpr uint32_t all_sent_mask = compute_all_sent_mask(numWorkers);
    };

    // Compute (TRISC) compile-time args
    template <
        uint32_t deviceRole,
        uint32_t numTiles,
        uint32_t localCb,
        uint32_t receivedCb,
        uint32_t outputCb,
        uint32_t scratchCb,
        uint32_t isFabricCore>
    struct ComputeCTArgs {
        static constexpr uint32_t device_role = deviceRole;
        static constexpr uint32_t num_tiles = numTiles;
        static constexpr uint32_t local_cb = localCb;
        static constexpr uint32_t received_cb = receivedCb;
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
        uint32_t worker_sem_addr;
        uint32_t dst_l1_addr;
        uint32_t dst_sem_addr;
        uint32_t output_base_addr;
        uint32_t shard_idx;
        uint32_t socket_config_addr;  // Per-worker downstream socket config address
        uint32_t metadata_addr;       // L1 address of metadata (only used by last worker when forward_metadata > 0)
        uint32_t agg_sem_l1_addr;     // Persistent-signal sync semaphore L1 address (global sem)
        uint32_t agg_core_noc_x;      // Persistent-signal core physical NOC x
        uint32_t agg_core_noc_y;      // Persistent-signal core physical NOC y
        uint32_t persistent_enable;   // 1 if this core should send the persistent signal
        uint32_t persistent_dst_noc_x;     // Bcast sender physical NOC x on entry device
        uint32_t persistent_dst_noc_y;     // Bcast sender physical NOC y on entry device
        uint32_t persistent_dst_mesh_id;   // Entry device fabric mesh id
        uint32_t persistent_dst_chip_id;   // Entry device fabric chip id
        uint32_t persistent_dst_sem_addr;  // persistent_next_iter_semaphore address on entry device
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
        static constexpr uint32_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);
        static constexpr uint32_t slot_size_bytes = packet_header_size_bytes + CTArgs::payload_size_bytes;
        static constexpr bool use_posted_transport_writes = true;
        static constexpr uint8_t worker_to_forwarder_noc = 0;
        static constexpr uint8_t forwarder_to_fabric_noc = 0;
        static_assert(
            noc_mode == DM_DYNAMIC_NOC || worker_to_forwarder_noc == noc_index, "Custom noc requires DM_DYNAMIC_NOC");
        static_assert(
            noc_mode == DM_DYNAMIC_NOC || forwarder_to_fabric_noc == noc_index, "Custom noc requires DM_DYNAMIC_NOC");

        template <typename FabricConnection>
        static FORCE_INLINE uint32_t process_ready_slots(
            volatile tt_l1_ptr uint32_t* sem_ptr,
            uint32_t sent_mask,
            uint32_t buffer_base,
            FabricConnection& conn,
            uint32_t& cached_free_write_slots) {
            uint32_t sem_value = *sem_ptr;
            uint32_t pending = sem_value & ~sent_mask;

            while (pending != 0) {
                if (cached_free_write_slots == 0) {
                    do {
                        invalidate_l1_cache();
                        cached_free_write_slots = conn.get_num_free_write_slots();
                    } while (cached_free_write_slots == 0);
                }

                uint32_t slot = __builtin_ctz(pending);
                uint32_t slot_addr = buffer_base + slot * slot_size_bytes;

                conn.template send_current_slot_stateful_non_blocking_from_address<use_posted_transport_writes>(
                    slot_addr, slot_size_bytes, forwarder_to_fabric_noc);

                sent_mask |= (1u << slot);
                cached_free_write_slots--;
                pending &= pending - 1;
            }
            return sent_mask;
        }

        // Generic routing helper for paths that still determine the destination
        // at runtime (for example the persistent ROOT1 control path).
        template <typename packet_header_t>
        static FORCE_INLINE void set_unicast_route(
            volatile tt_l1_ptr packet_header_t* header, uint16_t dst_dev_id, uint16_t dst_mesh_id, uint16_t num_hops) {
            if constexpr (std::is_same_v<packet_header_t, tt::tt_fabric::HybridMeshPacketHeader>) {
                fabric_set_unicast_route(header, dst_dev_id, dst_mesh_id);
            } else {
                fabric_set_unicast_route<false>(header, num_hops);
            }
        }

        // Worker-path routing helper: derive the fixed destination from CTArgs
        // and use the single-hop fast path when the worker send contract allows it.
        template <typename packet_header_t>
        static FORCE_INLINE void set_unicast_route(volatile tt_l1_ptr packet_header_t* header) {
            constexpr uint16_t dst_dev_id = static_cast<uint16_t>(CTArgs::dst_fabric_node_chip_id);
            constexpr uint16_t dst_mesh_id = static_cast<uint16_t>(CTArgs::dst_fabric_node_mesh_id);

            if constexpr (std::is_same_v<packet_header_t, tt::tt_fabric::HybridMeshPacketHeader>) {
                if constexpr (CTArgs::num_hops == 1) {
                    fabric_set_single_hop_unicast_route(header, dst_dev_id, dst_mesh_id);
                } else {
                    set_unicast_route(header, dst_dev_id, dst_mesh_id, static_cast<uint16_t>(CTArgs::num_hops));
                }
            } else {
                set_unicast_route(header, dst_dev_id, dst_mesh_id, static_cast<uint16_t>(CTArgs::num_hops));
            }
        }

#endif

        void impl([[maybe_unused]] const RTArgs& args) {
            // Early return if this core is not a reduce core (worker or fabric)
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
                // Skip this when fused - previous op already pushed to local_cb
                if constexpr (!SkipLocalCbPush) {
                    cb_reserve_back(CTArgs::local_cb, CTArgs::num_tiles);
                    cb_push_back(CTArgs::local_cb, CTArgs::num_tiles);
                }
                // Round 1: Wait for shard from LEAF (page 0 of received_cb)
                cb_reserve_back(CTArgs::received_cb, CTArgs::num_tiles);
                volatile tt_l1_ptr uint32_t* recv_sem1_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.recv_sem_round1);
                noc_semaphore_wait_min(recv_sem1_ptr, 1);
                unified_kernels::semaphore_dec(recv_sem1_ptr);
                cb_push_back(CTArgs::received_cb, CTArgs::num_tiles);
            }

            if constexpr (CTArgs::device_role == MESH_ROOT2 || CTArgs::device_role == MESH_ROOT1) {
                // Round 2: Wait for result from ROOT3 (page 1 of received_cb)
                cb_reserve_back(CTArgs::received_cb, CTArgs::num_tiles);
                volatile tt_l1_ptr uint32_t* recv_sem2_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.recv_sem_round2);
                noc_semaphore_wait_min(recv_sem2_ptr, 1);
                unified_kernels::semaphore_dec(recv_sem2_ptr);
                cb_push_back(CTArgs::received_cb, CTArgs::num_tiles);
            }

            if constexpr (CTArgs::device_role == MESH_ROOT1) {
                // Round 3: Wait for result from ROOT2 (page 2 of received_cb)
                cb_reserve_back(CTArgs::received_cb, CTArgs::num_tiles);
                volatile tt_l1_ptr uint32_t* recv_sem3_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.recv_sem_round3);
                noc_semaphore_wait_min(recv_sem3_ptr, 1);
                unified_kernels::semaphore_dec(recv_sem3_ptr);
                cb_push_back(CTArgs::received_cb, CTArgs::num_tiles);
            }

#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC - Writer: sends data via fabric or NOC
            // ================================================================
            if constexpr (CTArgs::is_fabric_core) {
                if constexpr (CTArgs::device_role == MESH_ROOT1) {
                    if constexpr (CTArgs::persistent_fabric_signal_enable != 0) {
                        // Persistent fabric core: wait for aggregator signal, then send
                        // cross-device atomic inc to bcast sender on entry device.
                        // Persistent args start after the shared ready semaphore address.
                        size_t p_idx = CTArgs::fabric_rt_arg_base + 1;
                        uint32_t wait_sem_addr = get_arg_val<uint32_t>(p_idx++);
                        uint32_t dst_noc_x = get_arg_val<uint32_t>(p_idx++);
                        uint32_t dst_noc_y = get_arg_val<uint32_t>(p_idx++);
                        uint32_t dst_mesh_id = get_arg_val<uint32_t>(p_idx++);
                        uint32_t dst_chip_id = get_arg_val<uint32_t>(p_idx++);
                        uint32_t dst_sem_addr = get_arg_val<uint32_t>(p_idx++);

                        volatile tt_l1_ptr uint32_t* wait_sem_ptr =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wait_sem_addr);
                        PacketHeaderPool::reset();
                        auto route_id = PacketHeaderPool::allocate_header_n(1);
                        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::header_table[route_id].first;
                        set_unicast_route(
                            hdr, static_cast<uint16_t>(dst_chip_id), static_cast<uint16_t>(dst_mesh_id), 1);
                        hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                            get_noc_addr(dst_noc_x, dst_noc_y, dst_sem_addr), 1});

                        auto sender =
                            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(
                                p_idx);
                        sender.open();
                        sender.wait_for_empty_write_slot();
                        noc_semaphore_wait_min(wait_sem_ptr, 1);
                        unified_kernels::semaphore_dec(wait_sem_ptr);
                        sender.send_payload_flush_blocking_from_address(
                            reinterpret_cast<uint32_t>(hdr), packet_header_size_bytes);
                        sender.close();
                        noc_async_full_barrier();
                    }
                    return;
                }

                size_t arg_idx = CTArgs::fabric_rt_arg_base;
                const uint32_t ready_sem_addr = get_arg_val<uint32_t>(arg_idx++);
                auto fabric_sender =
                    tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);

                fabric_sender.open_start();
                const uint32_t packet_buffer_addr = get_write_ptr(CTArgs::packet_cb);
                fabric_sender.open_finish();

                fabric_sender.template setup_stateful_send_cmd_bufs<use_posted_transport_writes>(
                    forwarder_to_fabric_noc);

                volatile tt_l1_ptr uint32_t* ready_sem_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_sem_addr);
                uint32_t sent_mask = 0;
                uint32_t cached_free_write_slots = 0;

                do {
                    invalidate_l1_cache();
                    sent_mask = process_ready_slots(
                        ready_sem_ptr, sent_mask, packet_buffer_addr, fabric_sender, cached_free_write_slots);
                } while (sent_mask != CTArgs::all_sent_mask);

                if constexpr (use_posted_transport_writes) {
                    noc_async_posted_writes_flushed(forwarder_to_fabric_noc);
                } else {
                    noc_async_writes_flushed(forwarder_to_fabric_noc);
                }

                noc_semaphore_set(ready_sem_ptr, 0);

                fabric_sender.close();
                return;
            }

            // Worker core logic
            const uint32_t my_noc_x = my_x[0];
            const uint32_t my_noc_y = my_y[0];

            // ROOT1: gather all shards to output tensor; each worker sends its shard downstream
            if constexpr (CTArgs::device_role == MESH_ROOT1) {
                // Notify the aggregator (or persistent forwarder) that this worker is done.
                // Issued between socket_notify_receiver and socket_barrier in the socket branch
                // so the downstream consumer can wake up while we wait for the socket ack.
                auto signal_aggregator = [&]() __attribute__((always_inline)) {
                    if (args.persistent_enable != 0) {
                        volatile tt_l1_ptr uint32_t* agg_sem_ptr =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.agg_sem_l1_addr);
                        uint64_t fc_sem = get_noc_addr(
                            args.persistent_dst_noc_x, args.persistent_dst_noc_y, args.persistent_dst_sem_addr);
                        noc_semaphore_wait_min(agg_sem_ptr, CTArgs::total_num_workers - 1);
                        noc_semaphore_set(agg_sem_ptr, 0);
                        noc_semaphore_inc(fc_sem, 1);
                        noc_async_atomic_barrier();
                    } else if (args.agg_sem_l1_addr != 0) {
                        uint64_t agg_sem_noc =
                            get_noc_addr(args.agg_core_noc_x, args.agg_core_noc_y, args.agg_sem_l1_addr);
                        noc_semaphore_inc(agg_sem_noc, 1);
                        noc_async_atomic_barrier();
                    }
                };

                if constexpr (CTArgs::enable_downstream_socket) {
                    constexpr uint32_t useful_per_shard = CTArgs::agg_output_size_bytes / CTArgs::total_num_workers;
                    constexpr bool is_last_worker_metadata_forwarder = CTArgs::forward_metadata_size_bytes > 0;
                    if (args.socket_config_addr != 0) {
                        const bool is_last_worker = args.shard_idx == CTArgs::total_num_workers - 1;
                        const uint32_t socket_page_size = (is_last_worker_metadata_forwarder && is_last_worker)
                                                              ? useful_per_shard + CTArgs::forward_metadata_size_bytes
                                                              : useful_per_shard;
                        SocketSenderInterface sender_socket = create_sender_socket_interface(args.socket_config_addr);
                        set_sender_socket_page_size(sender_socket, socket_page_size);
                        socket_reserve_pages(sender_socket, 1);
                        sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, 0);

                        uint64_t fifo_dst = get_noc_addr(
                            downstream_enc.d2d.downstream_noc_x,
                            downstream_enc.d2d.downstream_noc_y,
                            sender_socket.write_ptr + sender_socket.downstream_fifo_addr);
                        cb_wait_front(CTArgs::scratch_cb, CTArgs::num_tiles);
                        uint32_t src_addr = get_read_ptr(CTArgs::scratch_cb);
                        // Socket barrier means receiver has received and acknowledged the write, so we can use posted
                        // writes here
                        noc_async_write<useful_per_shard, true, /*posted=*/true>(src_addr, fifo_dst, useful_per_shard);

                        if constexpr (is_last_worker_metadata_forwarder) {
                            if (is_last_worker) {
                                noc_async_write<CTArgs::forward_metadata_size_bytes, true, /*posted=*/true>(
                                    args.metadata_addr,
                                    fifo_dst + useful_per_shard,
                                    CTArgs::forward_metadata_size_bytes);
                            }
                        }
                        noc_async_posted_writes_flushed();

                        socket_push_pages(sender_socket, 1);
                        socket_notify_receiver(sender_socket);
                        signal_aggregator();
                        socket_barrier(sender_socket);
                        update_socket_config(sender_socket);
                    }
                } else {
                    uint32_t dst_addr_0 = args.output_base_addr + args.shard_idx * CTArgs::payload_size_bytes;
                    uint64_t dst_noc_addr_0 =
                        get_noc_addr(CTArgs::output_core_noc_x, CTArgs::output_core_noc_y, dst_addr_0);
                    cb_wait_front(CTArgs::scratch_cb, CTArgs::num_tiles);
                    uint32_t src_addr = get_read_ptr(CTArgs::scratch_cb);
                    noc_async_write<CTArgs::payload_size_bytes>(src_addr, dst_noc_addr_0, CTArgs::payload_size_bytes);
                    noc_async_write_barrier();
                    signal_aggregator();
                }

                cb_pop_front(CTArgs::scratch_cb, CTArgs::num_tiles);
                return;
            }

            // Non-ROOT1: send via fabric
            const uint32_t packet_buffer_addr = get_write_ptr(CTArgs::packet_cb);
            const uint32_t arrival_sem_addr = args.worker_sem_addr;

            // Get packet header
            PacketHeaderPool::reset();
            auto* packet_header = PacketHeaderPool::allocate_header(1);

            // Set routing - works for both 1D (num_hops) and 2D (dst_dev_id, dst_mesh_id) fabric
            set_unicast_route(packet_header);

            // Set up fused write + atomic inc
            uint64_t dst_noc_addr = get_noc_addr(my_noc_x, my_noc_y, args.dst_l1_addr);
            uint64_t dst_sem_noc_addr = get_noc_addr(my_noc_x, my_noc_y, args.dst_sem_addr);
            packet_header->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, dst_sem_noc_addr, 1, false},
                CTArgs::payload_size_bytes);

            // Calculate slot in fabric core's packet buffer
            uint32_t slot_offset = args.my_slot_idx * slot_size_bytes;
            uint32_t header_dest_addr = packet_buffer_addr + slot_offset;
            uint32_t payload_dest_addr = header_dest_addr + packet_header_size_bytes;

            uint64_t header_noc_addr =
                get_noc_addr(args.fabric_core_noc_x, args.fabric_core_noc_y, header_dest_addr, worker_to_forwarder_noc);
            uint64_t payload_noc_addr = get_noc_addr(
                args.fabric_core_noc_x, args.fabric_core_noc_y, payload_dest_addr, worker_to_forwarder_noc);
            uint64_t arrival_sem_noc_addr =
                get_noc_addr(args.fabric_core_noc_x, args.fabric_core_noc_y, arrival_sem_addr, worker_to_forwarder_noc);

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
            noc_async_write<packet_header_size_bytes, /*enable_noc_tracing=*/false, /*posted=*/true>(
                reinterpret_cast<uint32_t>(packet_header),
                header_noc_addr,
                packet_header_size_bytes,
                worker_to_forwarder_noc);
            noc_async_write<CTArgs::payload_size_bytes, /*enable_noc_tracing=*/false, /*posted=*/true>(
                data_addr, payload_noc_addr, CTArgs::payload_size_bytes, worker_to_forwarder_noc);

            // Ensure the staged packet is visible before advertising the slot as ready.
            noc_async_posted_writes_flushed(worker_to_forwarder_noc);
            noc_semaphore_inc(arrival_sem_noc_addr, 1u << args.my_slot_idx, worker_to_forwarder_noc);
            noc_async_atomic_barrier();

            // Pop source_cb to free it for the next iteration.
            // Must match the wait_front guard: LEAF standalone (SkipLocalCbPush=false)
            // uses sharded setup (no push/pop tracking), so skip the pop.
            if constexpr (CTArgs::device_role == MESH_LEAF) {
                if constexpr (SkipLocalCbPush) {
                    cb_pop_front(source_cb, CTArgs::num_tiles);
                }
            } else {
                cb_pop_front(source_cb, CTArgs::num_tiles);
            }

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC - Compute: performs reduction
            // ================================================================
            if constexpr (CTArgs::is_fabric_core || CTArgs::device_role == MESH_LEAF) {
                // Fabric cores and LEAFs have no compute
                return;
            }

            // Initialize for binary operations
            reconfig_data_format<false, true>(CTArgs::local_cb, CTArgs::received_cb);
            pack_reconfig_data_format<true>(CTArgs::scratch_cb);
            pack_block_contiguous_init(CTArgs::scratch_cb);

            // Load local tiles to dest
            copy_tile_to_dst_init_short(CTArgs::local_cb);
            cb_wait_front(CTArgs::local_cb, CTArgs::num_tiles);
            tile_regs_acquire();
            for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                copy_tile(CTArgs::local_cb, i, i);
            }
            cb_pop_front(CTArgs::local_cb, CTArgs::num_tiles);

            // Accumulate from received_cb page 0 (LEAF data)
            binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CTArgs::received_cb);
            cb_wait_front(CTArgs::received_cb, CTArgs::num_tiles);
            for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CTArgs::received_cb, i, i);
            }
            cb_pop_front(CTArgs::received_cb, CTArgs::num_tiles);

            if constexpr (CTArgs::device_role == MESH_ROOT2 || CTArgs::device_role == MESH_ROOT1) {
                // Accumulate from received_cb page 1 (ROOT3 data)
                cb_wait_front(CTArgs::received_cb, CTArgs::num_tiles);
                for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                    binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                        CTArgs::received_cb, i, i);
                }
                cb_pop_front(CTArgs::received_cb, CTArgs::num_tiles);
            }

            if constexpr (CTArgs::device_role == MESH_ROOT1) {
                // Accumulate from received_cb page 2 (ROOT2 data)
                cb_wait_front(CTArgs::received_cb, CTArgs::num_tiles);
                for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                    binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                        CTArgs::received_cb, i, i);
                }
                cb_pop_front(CTArgs::received_cb, CTArgs::num_tiles);
            }
            tile_regs_commit();

            // Pack result to scratch_cb
            cb_reserve_back(CTArgs::scratch_cb, CTArgs::num_tiles);
            tile_regs_wait();
            pack_block_contiguous(0, CTArgs::scratch_cb, CTArgs::num_tiles);
            tile_regs_release();
            cb_push_back(CTArgs::scratch_cb, CTArgs::num_tiles);
#endif
        }
    };  // class Op

};  // struct ReduceToOneB1

}  // namespace deepseek_b1_ops
