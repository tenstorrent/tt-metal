// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * Unified Reduce-to-All B1 Operation
 *
 * Performs 3-round hypercube all-reduce across a 4x2 mesh so that every
 * device ends up with the sum of all 8 inputs.
 *
 * Each round, every device simultaneously sends to and receives from a
 * partner determined by XOR-ing one dimension of the device index:
 *
 *   Round 1 (adjacent rows):  (row, col) <-> (row^1, col)
 *   Round 2 (2-apart rows):   (row, col) <-> (row^2, col)
 *   Round 3 (cross-column):   (row, col) <-> (row, col^1)
 *
 * TRISC uses add_tiles (CB-to-CB add) each round with a reload_cb to
 * carry the accumulated partial sum across rounds.  tile_regs_commit/
 * wait/release provide explicit MATH↔PACK sync; the release zeros the
 * current dest half, so reload_cb stores the intermediate result for
 * the next round's add_tiles to re-read.
 *
 * R1/R2 are forwarded by BRISC on each fabric core via its same-column
 * EDM connection.  R3 is forwarded by NCRISC on FC1 (link_idx=1) via a
 * separate cross-column EDM connection, avoiding the circular deadlock
 * that occurs when R3 shares the same-column link.
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
#include "api/debug/dprint.h"

#elif defined(COMPILE_FOR_NCRISC)
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "api/debug/dprint.h"

#elif defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/debug/dprint.h"
#endif

namespace deepseek_b1_ops {

struct ReduceToAllB1 {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    template <
        uint32_t numTiles,
        uint32_t localCb,
        uint32_t receivedCb,
        uint32_t isFabricCore,
        uint32_t deferR3Send,
        uint32_t numWorkers,
        uint32_t slotSizeBytes,
        uint32_t packetCb,
        uint32_t payloadSizeBytes,
        uint32_t isR3Forwarder,
        uint32_t numR3Workers>
    struct ReaderCTArgs {
        static constexpr uint32_t num_tiles = numTiles;
        static constexpr uint32_t local_cb = localCb;
        static constexpr uint32_t received_cb = receivedCb;
        static constexpr uint32_t is_fabric_core = isFabricCore;
        static constexpr uint32_t defer_r3_send = deferR3Send;
        static constexpr uint32_t num_workers = numWorkers;
        static constexpr uint32_t slot_size_bytes = slotSizeBytes;
        static constexpr uint32_t packet_cb = packetCb;
        static constexpr uint32_t payload_size_bytes = payloadSizeBytes;
        static constexpr uint32_t is_r3_forwarder = isR3Forwarder;
        static constexpr uint32_t num_r3_workers = numR3Workers;
    };

    template <
        uint32_t numTiles,
        uint32_t payloadSizeBytes,
        uint32_t localCb,
        uint32_t scratchCb,
        uint32_t packetCb,
        uint32_t numWorkers,
        uint32_t slotSizeBytes,
        uint32_t isFabricCore,
        uint32_t r1DstChipId,
        uint32_t r1DstMeshId,
        uint32_t r2DstChipId,
        uint32_t r2DstMeshId,
        uint32_t r3DstChipId,
        uint32_t r3DstMeshId,
        uint32_t deferR3Send>
    struct WriterCTArgs {
        static constexpr uint32_t num_tiles = numTiles;
        static constexpr uint32_t payload_size_bytes = payloadSizeBytes;
        static constexpr uint32_t local_cb = localCb;
        static constexpr uint32_t scratch_cb = scratchCb;
        static constexpr uint32_t packet_cb = packetCb;
        static constexpr uint32_t num_workers = numWorkers;
        static constexpr uint32_t slot_size_bytes = slotSizeBytes;
        static constexpr uint32_t is_fabric_core = isFabricCore;
        static constexpr uint32_t r1_dst_chip_id = r1DstChipId;
        static constexpr uint32_t r1_dst_mesh_id = r1DstMeshId;
        static constexpr uint32_t r2_dst_chip_id = r2DstChipId;
        static constexpr uint32_t r2_dst_mesh_id = r2DstMeshId;
        static constexpr uint32_t r3_dst_chip_id = r3DstChipId;
        static constexpr uint32_t r3_dst_mesh_id = r3DstMeshId;
        static constexpr uint32_t defer_r3_send = deferR3Send;
    };

    template <
        uint32_t numTiles,
        uint32_t localCb,
        uint32_t receivedCb,
        uint32_t scratchCb,
        uint32_t reloadCb,
        uint32_t isFabricCore>
    struct ComputeCTArgs {
        static constexpr uint32_t num_tiles = numTiles;
        static constexpr uint32_t local_cb = localCb;
        static constexpr uint32_t received_cb = receivedCb;
        static constexpr uint32_t scratch_cb = scratchCb;
        static constexpr uint32_t reload_cb = reloadCb;
        static constexpr uint32_t is_fabric_core = isFabricCore;
    };

    // ========================================================================
    // Runtime args structs
    // ========================================================================

    struct ReaderArgs {
        uint32_t recv_sem_round1;
        uint32_t recv_sem_round2;
        uint32_t recv_sem_round3;
        uint32_t r3_gate_addr;
    };

    struct WorkerWriterArgs {
        uint32_t fabric_core_noc_x;
        uint32_t fabric_core_noc_y;
        uint32_t my_slot_idx;
        uint32_t worker_sem_addr;
        uint32_t r1_dst_l1_addr;
        uint32_t r1_dst_sem_addr;
        uint32_t r2_dst_l1_addr;
        uint32_t r2_dst_sem_addr;
        uint32_t r3_dst_l1_addr;
        uint32_t r3_dst_sem_addr;
        uint32_t output_base_addr;
        uint32_t r3_gate_addr;
        uint32_t r3_fabric_core_noc_x;
        uint32_t r3_fabric_core_noc_y;
        uint32_t r3_slot_idx;
        uint32_t r3_ncrisc_sem_addr;
    };

    struct ComputeArgs {};

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WorkerWriterArgs, ComputeArgs>;

    // ========================================================================
    // Op implementation
    // ========================================================================
    template <typename CTArgs, bool IsWorkerCore, bool SkipLocalCbPush = false>
    class Op {
    public:
        void operator()(const RTArgs& args) { impl(args); }

    private:
#if defined(COMPILE_FOR_BRISC)
        template <typename packet_header_t>
        static FORCE_INLINE void set_unicast_route(
            volatile tt_l1_ptr packet_header_t* header, uint16_t dst_dev_id, uint16_t dst_mesh_id, uint16_t num_hops) {
            if constexpr (std::is_same_v<packet_header_t, tt::tt_fabric::HybridMeshPacketHeader>) {
                fabric_set_unicast_route(header, dst_dev_id, dst_mesh_id);
            } else {
                fabric_set_unicast_route<false>(header, num_hops);
            }
        }

        static FORCE_INLINE void send_via_fabric(
            volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
            uint16_t dst_chip_id,
            uint16_t dst_mesh_id,
            uint32_t my_noc_x,
            uint32_t my_noc_y,
            uint32_t dst_l1_addr,
            uint32_t dst_sem_addr,
            uint32_t data_addr,
            uint64_t header_noc_addr,
            uint64_t payload_noc_addr,
            uint64_t arrival_sem_noc_addr) {
            set_unicast_route(packet_header, dst_chip_id, dst_mesh_id, static_cast<uint16_t>(1));

            uint64_t dst_noc_addr = get_noc_addr(my_noc_x, my_noc_y, dst_l1_addr);
            uint64_t dst_sem_noc_addr = get_noc_addr(my_noc_x, my_noc_y, dst_sem_addr);
            packet_header->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, dst_sem_noc_addr, 1, false},
                CTArgs::payload_size_bytes);

            constexpr uint32_t pkt_hdr_bytes = sizeof(PACKET_HEADER_TYPE);
            noc_async_write(reinterpret_cast<uint32_t>(packet_header), header_noc_addr, pkt_hdr_bytes);
            noc_async_write(data_addr, payload_noc_addr, CTArgs::payload_size_bytes);
            noc_semaphore_inc(arrival_sem_noc_addr, 1);
            noc_async_write_barrier();
            noc_async_atomic_barrier();
        }
#endif

        void impl([[maybe_unused]] const RTArgs& args) {
            if constexpr (!IsWorkerCore) {
                return;
            }

#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC — Reader / R3 fabric forwarder
            // ================================================================
            if constexpr (CTArgs::is_fabric_core) {
                if constexpr (CTArgs::is_r3_forwarder) {
                    DPRINT << "NCRISC FABRIC R3: start, num_r3_workers=" << (uint32_t)CTArgs::num_r3_workers << ENDL();
                    size_t arg_idx = 0;
                    uint32_t r3_worker_sems[CTArgs::num_r3_workers];
                    for (uint32_t i = 0; i < CTArgs::num_r3_workers; i++) {
                        r3_worker_sems[i] = get_arg_val<uint32_t>(arg_idx++);
                        DPRINT << "NCRISC FABRIC R3: sem[" << i << "]=" << r3_worker_sems[i] << ENDL();
                    }

                    const uint32_t packet_buffer_addr = get_write_ptr(CTArgs::packet_cb);
                    constexpr uint32_t r3_region_offset = 2 * CTArgs::num_workers * CTArgs::slot_size_bytes;
                    uint32_t r3_slot_base = packet_buffer_addr + r3_region_offset;
                    DPRINT << "NCRISC FABRIC R3: pkt_buf=" << packet_buffer_addr << " r3_base=" << r3_slot_base
                           << ENDL();

                    DPRINT << "NCRISC FABRIC R3: build_from_args, arg_idx=" << (uint32_t)arg_idx << ENDL();
                    auto fabric_sender =
                        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
                    DPRINT << "NCRISC FABRIC R3: opening" << ENDL();
                    fabric_sender.open();
                    DPRINT << "NCRISC FABRIC R3: opened" << ENDL();

                    constexpr uint32_t pkt_hdr_bytes = sizeof(PACKET_HEADER_TYPE);
                    for (uint32_t w = 0; w < CTArgs::num_r3_workers; w++) {
                        DPRINT << "NCRISC FABRIC R3: waiting worker " << w << " sem @" << r3_worker_sems[w] << ENDL();
                        volatile tt_l1_ptr uint32_t* sem =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r3_worker_sems[w]);
                        noc_semaphore_wait_min(sem, 1);
                        unified_kernels::semaphore_dec(sem);
                        DPRINT << "NCRISC FABRIC R3: worker " << w << " sem acquired" << ENDL();

                        uint32_t hdr_addr = r3_slot_base;
                        uint32_t payload_addr = r3_slot_base + pkt_hdr_bytes;

                        fabric_sender.wait_for_empty_write_slot();
                        fabric_sender.send_payload_without_header_non_blocking_from_address(
                            payload_addr, CTArgs::payload_size_bytes);
                        fabric_sender.send_payload_flush_blocking_from_address(hdr_addr, pkt_hdr_bytes);
                        DPRINT << "NCRISC FABRIC R3: worker " << w << " forwarded" << ENDL();

                        r3_slot_base += CTArgs::slot_size_bytes;
                    }

                    DPRINT << "NCRISC FABRIC R3: closing" << ENDL();
                    fabric_sender.close();
                    noc_async_write_barrier();
                    DPRINT << "NCRISC FABRIC R3: done" << ENDL();
                }
                return;
            }

            DPRINT << "NCRISC: start, num_tiles=" << (uint32_t)CTArgs::num_tiles << ENDL();
            {
                volatile tt_l1_ptr uint32_t* r2_sem =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.recv_sem_round2);
                DPRINT << "NCRISC: INIT r2_sem=" << *r2_sem << ENDL();
            }

            if constexpr (!SkipLocalCbPush) {
                DPRINT << "NCRISC: local_cb reserve_back" << ENDL();
                cb_reserve_back(CTArgs::local_cb, CTArgs::num_tiles);
                cb_push_back(CTArgs::local_cb, CTArgs::num_tiles);
                DPRINT << "NCRISC: local_cb pushed" << ENDL();
            }

            // Round 1
            DPRINT << "NCRISC: R1 reserve received_cb" << ENDL();
            cb_reserve_back(CTArgs::received_cb, CTArgs::num_tiles);
            DPRINT << "NCRISC: R1 waiting on sem @" << (uint32_t)args.recv_sem_round1 << ENDL();
            {
                volatile tt_l1_ptr uint32_t* sem_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.recv_sem_round1);
                noc_semaphore_wait_min(sem_ptr, 1);
                unified_kernels::semaphore_dec(sem_ptr);
            }
            {
                uint32_t recv_addr = get_write_ptr(CTArgs::received_cb);
                volatile uint16_t* d = reinterpret_cast<volatile uint16_t*>(recv_addr + 32);
                DPRINT << "NCRISC: R1 recv_val[0]=0x" << HEX() << (uint32_t)d[0] << ENDL();
            }
            cb_push_back(CTArgs::received_cb, CTArgs::num_tiles);
            DPRINT << "NCRISC: R1 received_cb pushed" << ENDL();

            // Round 2
            DPRINT << "NCRISC: R2 reserve received_cb" << ENDL();
            cb_reserve_back(CTArgs::received_cb, CTArgs::num_tiles);
            {
                uint32_t recv_addr = get_write_ptr(CTArgs::received_cb);
                volatile uint16_t* d_pre = reinterpret_cast<volatile uint16_t*>(recv_addr + 32);
                DPRINT << "NCRISC: R2 PRE-SEM addr=" << recv_addr << " val[0]=0x" << HEX() << (uint32_t)d_pre[0]
                       << ENDL();
            }
            DPRINT << "NCRISC: R2 waiting on sem @" << (uint32_t)args.recv_sem_round2 << ENDL();
            {
                volatile tt_l1_ptr uint32_t* sem_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.recv_sem_round2);
                noc_semaphore_wait_min(sem_ptr, 1);
                unified_kernels::semaphore_dec(sem_ptr);
            }
            {
                uint32_t recv_addr = get_write_ptr(CTArgs::received_cb);
                DPRINT << "NCRISC: R2 recv_addr=" << recv_addr << ENDL();
                volatile uint16_t* d = reinterpret_cast<volatile uint16_t*>(recv_addr + 32);
                DPRINT << "NCRISC: R2 recv_val[0]=0x" << HEX() << (uint32_t)d[0] << ENDL();
            }
            cb_push_back(CTArgs::received_cb, CTArgs::num_tiles);
            DPRINT << "NCRISC: R2 received_cb pushed" << ENDL();

            // Round 3
            DPRINT << "NCRISC: R3 reserve received_cb" << ENDL();
            cb_reserve_back(CTArgs::received_cb, CTArgs::num_tiles);
            DPRINT << "NCRISC: R3 waiting on sem @" << (uint32_t)args.recv_sem_round3 << ENDL();
            {
                volatile tt_l1_ptr uint32_t* sem_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.recv_sem_round3);
                noc_semaphore_wait_min(sem_ptr, 1);
                unified_kernels::semaphore_dec(sem_ptr);
            }
            {
                uint32_t recv_addr = get_write_ptr(CTArgs::received_cb);
                volatile uint16_t* d = reinterpret_cast<volatile uint16_t*>(recv_addr + 32);
                DPRINT << "NCRISC: R3 recv_val[0]=0x" << HEX() << (uint32_t)d[0] << ENDL();
            }
            cb_push_back(CTArgs::received_cb, CTArgs::num_tiles);
            DPRINT << "NCRISC: R3 received_cb pushed" << ENDL();

            if constexpr (CTArgs::defer_r3_send) {
                volatile tt_l1_ptr uint32_t* gate = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.r3_gate_addr);
                *gate = 1;
                DPRINT << "NCRISC: R3 gate set" << ENDL();
            }
            DPRINT << "NCRISC: done" << ENDL();

#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC — Writer: fabric core forwards R1+R2; worker sends + output
            // ================================================================
            constexpr uint32_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

            if constexpr (CTArgs::is_fabric_core) {
                // Fabric core BRISC: forward R1 and R2 only (R3 handled by NCRISC on FC1)
                DPRINT << "BRISC FABRIC: start, num_workers=" << (uint32_t)CTArgs::num_workers << ENDL();
                size_t arg_idx = 0;
                uint32_t worker_sem_addr[CTArgs::num_workers];
                for (uint32_t i = 0; i < CTArgs::num_workers; i++) {
                    worker_sem_addr[i] = get_arg_val<uint32_t>(arg_idx++);
                    DPRINT << "BRISC FABRIC: worker_sem_addr[" << i << "]=" << worker_sem_addr[i] << ENDL();
                }
                const uint32_t packet_buffer_addr = get_write_ptr(CTArgs::packet_cb);
                DPRINT << "BRISC FABRIC: packet_buffer_addr=" << packet_buffer_addr << ENDL();

                DPRINT << "BRISC FABRIC: build sender_r1, arg_idx=" << (uint32_t)arg_idx << ENDL();
                auto sender_r1 =
                    tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
                DPRINT << "BRISC FABRIC: build sender_r2, arg_idx=" << (uint32_t)arg_idx << ENDL();
                auto sender_r2 =
                    tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);

                constexpr uint32_t round_stride = CTArgs::num_workers * CTArgs::slot_size_bytes;

                // R1: use sender_r1 (connection to R1 partner)
                DPRINT << "BRISC FABRIC: R1 opening sender_r1" << ENDL();
                sender_r1.open();
                {
                    uint32_t slot_base = packet_buffer_addr;
                    for (uint32_t worker = 0; worker < CTArgs::num_workers; worker++) {
                        DPRINT << "BRISC FABRIC: R1 waiting worker " << worker << " sem @" << worker_sem_addr[worker]
                               << ENDL();
                        volatile tt_l1_ptr uint32_t* worker_sem_ptr =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(worker_sem_addr[worker]);
                        noc_semaphore_wait_min(worker_sem_ptr, 1);
                        unified_kernels::semaphore_dec(worker_sem_ptr);
                        DPRINT << "BRISC FABRIC: R1 worker " << worker << " sem acquired" << ENDL();

                        uint32_t worker_header_addr = slot_base;
                        uint32_t worker_payload_addr = slot_base + packet_header_size_bytes;

                        sender_r1.wait_for_empty_write_slot();
                        sender_r1.send_payload_without_header_non_blocking_from_address(
                            worker_payload_addr, CTArgs::payload_size_bytes);
                        sender_r1.send_payload_flush_blocking_from_address(
                            worker_header_addr, packet_header_size_bytes);
                        DPRINT << "BRISC FABRIC: R1 worker " << worker << " forwarded" << ENDL();

                        slot_base += CTArgs::slot_size_bytes;
                    }
                }
                DPRINT << "BRISC FABRIC: R1 closing sender_r1" << ENDL();
                sender_r1.close();
                noc_async_write_barrier();

                // R2: use sender_r2 (connection to R2 partner)
                DPRINT << "BRISC FABRIC: R2 opening sender_r2" << ENDL();
                sender_r2.open();
                {
                    uint32_t slot_base = packet_buffer_addr + round_stride;
                    for (uint32_t worker = 0; worker < CTArgs::num_workers; worker++) {
                        DPRINT << "BRISC FABRIC: R2 waiting worker " << worker << " sem @" << worker_sem_addr[worker]
                               << ENDL();
                        volatile tt_l1_ptr uint32_t* worker_sem_ptr =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(worker_sem_addr[worker]);
                        noc_semaphore_wait_min(worker_sem_ptr, 1);
                        unified_kernels::semaphore_dec(worker_sem_ptr);
                        DPRINT << "BRISC FABRIC: R2 worker " << worker << " sem acquired" << ENDL();

                        uint32_t worker_header_addr = slot_base;
                        uint32_t worker_payload_addr = slot_base + packet_header_size_bytes;

                        sender_r2.wait_for_empty_write_slot();
                        sender_r2.send_payload_without_header_non_blocking_from_address(
                            worker_payload_addr, CTArgs::payload_size_bytes);
                        sender_r2.send_payload_flush_blocking_from_address(
                            worker_header_addr, packet_header_size_bytes);
                        DPRINT << "BRISC FABRIC: R2 worker " << worker << " forwarded" << ENDL();

                        slot_base += CTArgs::slot_size_bytes;
                    }
                }
                DPRINT << "BRISC FABRIC: R2 closing sender_r2" << ENDL();
                sender_r2.close();
                noc_async_write_barrier();
                DPRINT << "BRISC FABRIC: done" << ENDL();
                return;
            }

            // ----------------------------------------------------------
            // Worker core: 3 rounds of sending + output write
            // ----------------------------------------------------------
            DPRINT << "BRISC WORKER: start" << ENDL();
            const uint32_t my_noc_x = my_x[0];
            const uint32_t my_noc_y = my_y[0];
            const uint32_t packet_buffer_addr = get_write_ptr(CTArgs::packet_cb);
            const uint32_t arrival_sem_addr = args.worker_sem_addr;

            DPRINT << "BRISC WORKER: noc=(" << my_noc_x << "," << my_noc_y << ") slot=" << (uint32_t)args.my_slot_idx
                   << " fab_core=(" << (uint32_t)args.fabric_core_noc_x << "," << (uint32_t)args.fabric_core_noc_y
                   << ")" << ENDL();
            DPRINT << "BRISC WORKER: r3_fab=(" << (uint32_t)args.r3_fabric_core_noc_x << ","
                   << (uint32_t)args.r3_fabric_core_noc_y << ") r3_slot=" << (uint32_t)args.r3_slot_idx << ENDL();
            DPRINT << "BRISC WORKER: r1_dst_chip=" << (uint32_t)CTArgs::r1_dst_chip_id
                   << " r2_dst_chip=" << (uint32_t)CTArgs::r2_dst_chip_id
                   << " r3_dst_chip=" << (uint32_t)CTArgs::r3_dst_chip_id << ENDL();
            DPRINT << "BRISC WORKER: reload_cb_wptr=" << get_write_ptr(4) << " recv_cb_wptr=" << get_write_ptr(1)
                   << ENDL();

            PacketHeaderPool::reset();
            auto* packet_header = PacketHeaderPool::allocate_header(1);

            // R1/R2 slots: on assigned fabric core (BRISC forwarding)
            constexpr uint32_t round_stride = CTArgs::num_workers * CTArgs::slot_size_bytes;
            uint32_t r1_hdr_dest = packet_buffer_addr + args.my_slot_idx * CTArgs::slot_size_bytes;
            uint32_t r2_hdr_dest = r1_hdr_dest + round_stride;

            uint64_t r1_header_noc = get_noc_addr(args.fabric_core_noc_x, args.fabric_core_noc_y, r1_hdr_dest);
            uint64_t r1_payload_noc =
                get_noc_addr(args.fabric_core_noc_x, args.fabric_core_noc_y, r1_hdr_dest + packet_header_size_bytes);
            uint64_t r2_header_noc = get_noc_addr(args.fabric_core_noc_x, args.fabric_core_noc_y, r2_hdr_dest);
            uint64_t r2_payload_noc =
                get_noc_addr(args.fabric_core_noc_x, args.fabric_core_noc_y, r2_hdr_dest + packet_header_size_bytes);
            uint64_t arrival_sem_noc_addr =
                get_noc_addr(args.fabric_core_noc_x, args.fabric_core_noc_y, arrival_sem_addr);

            // R3 slot: on FC1 (NCRISC forwarding via cross-column link)
            constexpr uint32_t r3_region_offset = 2 * CTArgs::num_workers * CTArgs::slot_size_bytes;
            uint32_t r3_hdr_dest = packet_buffer_addr + r3_region_offset + args.r3_slot_idx * CTArgs::slot_size_bytes;
            uint64_t r3_header_noc = get_noc_addr(args.r3_fabric_core_noc_x, args.r3_fabric_core_noc_y, r3_hdr_dest);
            uint64_t r3_payload_noc = get_noc_addr(
                args.r3_fabric_core_noc_x, args.r3_fabric_core_noc_y, r3_hdr_dest + packet_header_size_bytes);
            uint64_t r3_arrival_sem_noc_addr =
                get_noc_addr(args.r3_fabric_core_noc_x, args.r3_fabric_core_noc_y, args.r3_ncrisc_sem_addr);

            // Round 1: send local data
            DPRINT << "BRISC WORKER: R1 sending local data" << ENDL();
            {
                if constexpr (SkipLocalCbPush) {
                    cb_wait_front(CTArgs::local_cb, CTArgs::num_tiles);
                }
                uint32_t data_addr = get_read_ptr(CTArgs::local_cb);
                DPRINT << "BRISC WORKER: R1 data_addr=" << data_addr << ENDL();
                {
                    volatile uint16_t* d = reinterpret_cast<volatile uint16_t*>(data_addr + 32);
                    DPRINT << "BRISC WORKER: R1 local_val[0]=0x" << HEX() << (uint32_t)d[0] << ENDL();
                }
                send_via_fabric(
                    packet_header,
                    static_cast<uint16_t>(CTArgs::r1_dst_chip_id),
                    static_cast<uint16_t>(CTArgs::r1_dst_mesh_id),
                    my_noc_x,
                    my_noc_y,
                    args.r1_dst_l1_addr,
                    args.r1_dst_sem_addr,
                    data_addr,
                    r1_header_noc,
                    r1_payload_noc,
                    arrival_sem_noc_addr);
            }
            DPRINT << "BRISC WORKER: R1 sent" << ENDL();

            // Round 2: send round-1 partial sum
            DPRINT << "BRISC WORKER: R2 waiting scratch_cb" << ENDL();
            {
                cb_wait_front(CTArgs::scratch_cb, CTArgs::num_tiles);
                DPRINT << "BRISC WORKER: R2 scratch_cb ready" << ENDL();
                uint32_t data_addr = get_read_ptr(CTArgs::scratch_cb);
                DPRINT << "BRISC WORKER: R2 data_addr=" << data_addr << ENDL();
                {
                    volatile uint16_t* d = reinterpret_cast<volatile uint16_t*>(data_addr + 32);
                    DPRINT << "BRISC WORKER: R2 scratch_val[0]=0x" << HEX() << (uint32_t)d[0] << ENDL();
                }
                send_via_fabric(
                    packet_header,
                    static_cast<uint16_t>(CTArgs::r2_dst_chip_id),
                    static_cast<uint16_t>(CTArgs::r2_dst_mesh_id),
                    my_noc_x,
                    my_noc_y,
                    args.r2_dst_l1_addr,
                    args.r2_dst_sem_addr,
                    data_addr,
                    r2_header_noc,
                    r2_payload_noc,
                    arrival_sem_noc_addr);
                cb_pop_front(CTArgs::scratch_cb, CTArgs::num_tiles);
            }
            DPRINT << "BRISC WORKER: R2 sent" << ENDL();

            // Round 3: send round-2 partial sum via FC1 NCRISC (cross-column link)
            if constexpr (CTArgs::defer_r3_send) {
                DPRINT << "BRISC WORKER: R3 waiting for gate" << ENDL();
                volatile tt_l1_ptr uint32_t* gate = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.r3_gate_addr);
                while (*gate == 0) {
                }
                DPRINT << "BRISC WORKER: R3 gate passed" << ENDL();
            }
            DPRINT << "BRISC WORKER: R3 waiting scratch_cb" << ENDL();
            {
                cb_wait_front(CTArgs::scratch_cb, CTArgs::num_tiles);
                DPRINT << "BRISC WORKER: R3 scratch_cb ready" << ENDL();
                uint32_t data_addr = get_read_ptr(CTArgs::scratch_cb);
                DPRINT << "BRISC WORKER: R3 data_addr=" << data_addr << ENDL();
                {
                    volatile uint16_t* d = reinterpret_cast<volatile uint16_t*>(data_addr + 32);
                    DPRINT << "BRISC WORKER: R3 scratch_val[0]=0x" << HEX() << (uint32_t)d[0] << ENDL();
                }
                send_via_fabric(
                    packet_header,
                    static_cast<uint16_t>(CTArgs::r3_dst_chip_id),
                    static_cast<uint16_t>(CTArgs::r3_dst_mesh_id),
                    my_noc_x,
                    my_noc_y,
                    args.r3_dst_l1_addr,
                    args.r3_dst_sem_addr,
                    data_addr,
                    r3_header_noc,
                    r3_payload_noc,
                    r3_arrival_sem_noc_addr);
                cb_pop_front(CTArgs::scratch_cb, CTArgs::num_tiles);
            }
            DPRINT << "BRISC WORKER: R3 sent" << ENDL();

            // Output: write final result to this core's output shard
            DPRINT << "BRISC WORKER: OUTPUT waiting scratch_cb" << ENDL();
            {
                cb_wait_front(CTArgs::scratch_cb, CTArgs::num_tiles);
                DPRINT << "BRISC WORKER: OUTPUT scratch_cb ready" << ENDL();
                uint32_t src_addr = get_read_ptr(CTArgs::scratch_cb);
                {
                    volatile uint16_t* d = reinterpret_cast<volatile uint16_t*>(src_addr + 32);
                    DPRINT << "BRISC WORKER: OUTPUT scratch_val[0]=0x" << HEX() << (uint32_t)d[0] << ENDL();
                }
                uint64_t output_noc_addr = get_noc_addr(my_noc_x, my_noc_y, args.output_base_addr);
                noc_async_write(src_addr, output_noc_addr, CTArgs::payload_size_bytes);
                noc_async_write_barrier();
                cb_pop_front(CTArgs::scratch_cb, CTArgs::num_tiles);
            }
            DPRINT << "BRISC WORKER: done" << ENDL();

            if constexpr (SkipLocalCbPush) {
                cb_pop_front(CTArgs::local_cb, CTArgs::num_tiles);
            }

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC — Compute: add_tiles + reload_cb for cross-round accumulation
            //
            // tile_regs_commit/release zero the current dest half (SyncHalf),
            // so we use add_tiles(cbA, cbB) to read both operands from CBs
            // each round.  reload_cb stores the intermediate accumulated value
            // for the next round.
            // ================================================================
            if constexpr (CTArgs::is_fabric_core) {
                return;
            }

            DPRINT << "TRISC: start, num_tiles=" << (uint32_t)CTArgs::num_tiles << ENDL();

            // --- Round 1: local_cb + received_cb ---
            DPRINT << "TRISC: R1 add_tiles_init" << ENDL();
            add_tiles_init(CTArgs::local_cb, CTArgs::received_cb);
            pack_reconfig_data_format<true>(CTArgs::scratch_cb);

            DPRINT << "TRISC: R1 tile_regs_acquire" << ENDL();
            tile_regs_acquire();
            DPRINT << "TRISC: R1 waiting local_cb + received_cb" << ENDL();
            cb_wait_front(CTArgs::local_cb, CTArgs::num_tiles);
            cb_wait_front(CTArgs::received_cb, CTArgs::num_tiles);
            DPRINT << "TRISC: R1 add_tiles" << ENDL();
            for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                add_tiles(CTArgs::local_cb, CTArgs::received_cb, i, i, i);
            }
            cb_pop_front(CTArgs::local_cb, CTArgs::num_tiles);
            cb_pop_front(CTArgs::received_cb, CTArgs::num_tiles);
            DPRINT << "TRISC: R1 commit" << ENDL();
            tile_regs_commit();

            DPRINT << "TRISC: R1 wait+pack scratch" << ENDL();
            tile_regs_wait();
            cb_reserve_back(CTArgs::scratch_cb, CTArgs::num_tiles);
            for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                pack_tile(i, CTArgs::scratch_cb, i);
            }
            cb_push_back(CTArgs::scratch_cb, CTArgs::num_tiles);
            DPRINT << "TRISC: R1 pack reload" << ENDL();
            cb_reserve_back(CTArgs::reload_cb, CTArgs::num_tiles);
            for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                pack_tile(i, CTArgs::reload_cb, i);
            }
            cb_push_back(CTArgs::reload_cb, CTArgs::num_tiles);
            DPRINT << "TRISC: R1 release" << ENDL();
            tile_regs_release();
            DPRINT << "TRISC: R1 done" << ENDL();

            // --- Round 2: reload_cb (R1 result) + received_cb ---
            DPRINT << "TRISC: R2 add_tiles_init" << ENDL();
            add_tiles_init(CTArgs::reload_cb, CTArgs::received_cb);

            DPRINT << "TRISC: R2 tile_regs_acquire" << ENDL();
            tile_regs_acquire();
            DPRINT << "TRISC: R2 waiting reload_cb + received_cb" << ENDL();
            cb_wait_front(CTArgs::reload_cb, CTArgs::num_tiles);
            cb_wait_front(CTArgs::received_cb, CTArgs::num_tiles);
            DPRINT << "TRISC: R2 add_tiles" << ENDL();
            for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                add_tiles(CTArgs::reload_cb, CTArgs::received_cb, i, i, i);
            }
            cb_pop_front(CTArgs::reload_cb, CTArgs::num_tiles);
            cb_pop_front(CTArgs::received_cb, CTArgs::num_tiles);
            DPRINT << "TRISC: R2 commit" << ENDL();
            tile_regs_commit();

            DPRINT << "TRISC: R2 wait+pack scratch" << ENDL();
            tile_regs_wait();
            cb_reserve_back(CTArgs::scratch_cb, CTArgs::num_tiles);
            for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                pack_tile(i, CTArgs::scratch_cb, i);
            }
            cb_push_back(CTArgs::scratch_cb, CTArgs::num_tiles);
            DPRINT << "TRISC: R2 pack reload" << ENDL();
            cb_reserve_back(CTArgs::reload_cb, CTArgs::num_tiles);
            for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                pack_tile(i, CTArgs::reload_cb, i);
            }
            cb_push_back(CTArgs::reload_cb, CTArgs::num_tiles);
            DPRINT << "TRISC: R2 release" << ENDL();
            tile_regs_release();
            DPRINT << "TRISC: R2 done" << ENDL();

            // --- Round 3: reload_cb (R2 result) + received_cb ---
            DPRINT << "TRISC: R3 tile_regs_acquire" << ENDL();
            tile_regs_acquire();
            DPRINT << "TRISC: R3 waiting reload_cb + received_cb" << ENDL();
            cb_wait_front(CTArgs::reload_cb, CTArgs::num_tiles);
            cb_wait_front(CTArgs::received_cb, CTArgs::num_tiles);
            DPRINT << "TRISC: R3 add_tiles" << ENDL();
            for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                add_tiles(CTArgs::reload_cb, CTArgs::received_cb, i, i, i);
            }
            cb_pop_front(CTArgs::reload_cb, CTArgs::num_tiles);
            cb_pop_front(CTArgs::received_cb, CTArgs::num_tiles);
            DPRINT << "TRISC: R3 commit" << ENDL();
            tile_regs_commit();

            DPRINT << "TRISC: R3 wait+pack scratch" << ENDL();
            tile_regs_wait();
            cb_reserve_back(CTArgs::scratch_cb, CTArgs::num_tiles);
            for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                pack_tile(i, CTArgs::scratch_cb, i);
            }
            cb_push_back(CTArgs::scratch_cb, CTArgs::num_tiles);
            DPRINT << "TRISC: R3 release" << ENDL();
            tile_regs_release();
            DPRINT << "TRISC: R3 done" << ENDL();
#endif
        }
    };  // class Op

};  // struct ReduceToAllB1

}  // namespace deepseek_b1_ops
