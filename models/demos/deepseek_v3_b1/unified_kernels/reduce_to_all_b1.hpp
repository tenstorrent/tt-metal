// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * Unified Reduce-to-All B1 — Ring + Cross-Column
 *
 * Phase 1 — Column all-reduce (2-round ring, A/B split, all 1-hop):
 *   Type A workers: R1 → FWD (row+1), R2 → BWD (row-1)
 *   Type B workers: R1 → BWD (row-1), R2 → FWD (row+1)
 *   FC BRISC: single connection to FWD neighbor, forwards R1(A) + R2(B)
 *   FC NCRISC: single connection to BWD neighbor, forwards R1(B) + R2(A)
 *
 * Phase 2 — Cross-column exchange (FC-forwarded):
 *   Workers write R3 to FC's R3 buffer area, signal r3_fwd_sem bitmask.
 *   FC BRISC closes FWD conn, opens cross-column conn, forwards R3.
 *
 * After both phases every device holds sum(all 8 inputs).
 * All Phase 1 traffic is exactly 1 hop (ring/torus topology).
 * Requires FABRIC_2D_TORUS_X for the wrap-around link.
 */

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_BRISC)
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "api/socket_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"

#elif defined(COMPILE_FOR_NCRISC)
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "api/socket_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"

#elif defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/debug/dprint.h"
#endif

namespace deepseek_b1_ops {

struct ReduceToAllB1 {
    // ================================================================
    // Compile-time args
    // ================================================================

    template <
        uint32_t numTiles,
        uint32_t localCb,
        uint32_t receivedCb,
        uint32_t isFabricCore,
        uint32_t slotsPerDirection,
        uint32_t slotSizeBytes,
        uint32_t packetCb,
        uint32_t payloadSizeBytes,
        uint32_t r2BufferOffset,
        uint32_t ncriscBufferOffset>
    struct ReaderCTArgs {
        static constexpr uint32_t num_tiles = numTiles;
        static constexpr uint32_t local_cb = localCb;
        static constexpr uint32_t received_cb = receivedCb;
        static constexpr uint32_t is_fabric_core = isFabricCore;
        static constexpr uint32_t slots_per_direction = slotsPerDirection;
        static constexpr uint32_t slot_size_bytes = slotSizeBytes;
        static constexpr uint32_t packet_cb = packetCb;
        static constexpr uint32_t payload_size_bytes = payloadSizeBytes;
        static constexpr uint32_t r2_buffer_offset = r2BufferOffset;
        static constexpr uint32_t ncrisc_buffer_offset = ncriscBufferOffset;

        static constexpr uint32_t all_sent_mask =
            (slotsPerDirection == 32) ? 0xFFFF'FFFFu : ((1u << slotsPerDirection) - 1u);
    };

    template <
        uint32_t numTiles,
        uint32_t payloadSizeBytes,
        uint32_t localCb,
        uint32_t scratchCb,
        uint32_t packetCb,
        uint32_t slotSizeBytes,
        uint32_t isFabricCore,
        uint32_t fwdDstChipId,
        uint32_t fwdDstMeshId,
        uint32_t bwdDstChipId,
        uint32_t bwdDstMeshId,
        uint32_t r3DstChipId,
        uint32_t r3DstMeshId,
        uint32_t reloadCb,
        uint32_t computeTileSize,
        uint32_t slotsPerDirection,
        uint32_t r2BufferOffset,
        uint32_t ncriscBufferOffset,
        uint32_t r3BufferOffset>
    struct WriterCTArgs {
        static constexpr uint32_t num_tiles = numTiles;
        static constexpr uint32_t payload_size_bytes = payloadSizeBytes;
        static constexpr uint32_t local_cb = localCb;
        static constexpr uint32_t scratch_cb = scratchCb;
        static constexpr uint32_t packet_cb = packetCb;
        static constexpr uint32_t slot_size_bytes = slotSizeBytes;
        static constexpr uint32_t is_fabric_core = isFabricCore;
        static constexpr uint32_t fwd_dst_chip_id = fwdDstChipId;
        static constexpr uint32_t fwd_dst_mesh_id = fwdDstMeshId;
        static constexpr uint32_t bwd_dst_chip_id = bwdDstChipId;
        static constexpr uint32_t bwd_dst_mesh_id = bwdDstMeshId;
        static constexpr uint32_t r3_dst_chip_id = r3DstChipId;
        static constexpr uint32_t r3_dst_mesh_id = r3DstMeshId;
        static constexpr uint32_t reload_cb = reloadCb;
        static constexpr uint32_t compute_tile_size = computeTileSize;
        static constexpr uint32_t slots_per_direction = slotsPerDirection;
        static constexpr uint32_t r2_buffer_offset = r2BufferOffset;
        static constexpr uint32_t ncrisc_buffer_offset = ncriscBufferOffset;
        static constexpr uint32_t r3_buffer_offset = r3BufferOffset;

        static constexpr uint32_t all_sent_mask =
            (slotsPerDirection == 32) ? 0xFFFF'FFFFu : ((1u << slotsPerDirection) - 1u);
        static constexpr uint32_t r3_all_sent_mask =
            ((2 * slotsPerDirection) == 32) ? 0xFFFF'FFFFu : ((1u << (2 * slotsPerDirection)) - 1u);
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

    // ================================================================
    // Runtime args
    // ================================================================

    struct ReaderArgs {
        uint32_t recv_sem_round1;
        uint32_t recv_sem_round2;
        uint32_t recv_sem_round3;
    };

    // FC forwarder runtime args (both BRISC and NCRISC)
    struct ForwarderArgs {
        uint32_t r1_sem_addr;
        uint32_t r2_sem_addr;
    };

    struct WorkerWriterArgs {
        uint32_t fc_noc_x;
        uint32_t fc_noc_y;
        uint32_t is_type_a;
        uint32_t r1_slot_offset;
        uint32_t r1_slot_bit;
        uint32_t r1_sem_addr;
        uint32_t r2_slot_offset;
        uint32_t r2_slot_bit;
        uint32_t r2_sem_addr;
        uint32_t r1_dst_l1_addr;
        uint32_t r1_dst_sem_addr;
        uint32_t r2_dst_l1_addr;
        uint32_t r2_dst_sem_addr;
        uint32_t r3_dst_l1_addr;
        uint32_t r3_dst_sem_addr;
        uint32_t output_base_addr;
        uint32_t r3_slot_offset;
        uint32_t r3_slot_bit;
        uint32_t r3_sem_addr;
    };

    struct ComputeArgs {};

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WorkerWriterArgs, ComputeArgs>;

    // ================================================================
    // Op implementation
    // ================================================================
    template <typename CTArgs, bool IsWorkerCore, bool SkipLocalCbPush = false>
    class Op {
    public:
        void operator()(const RTArgs& args) { impl(args); }

    private:
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
        template <typename FabricConnection>
        static FORCE_INLINE uint32_t process_ready_slots(
            volatile tt_l1_ptr uint32_t* sem_ptr, uint32_t sent_mask, uint32_t buffer_base, FabricConnection& conn) {
            uint32_t sem_value = *sem_ptr;
            uint32_t pending = sem_value & ~sent_mask;
            while (pending != 0) {
                uint32_t slot = __builtin_ctz(pending);
                uint32_t slot_addr = buffer_base + slot * CTArgs::slot_size_bytes;
                auto* hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(slot_addr);
                uint32_t pkt_bytes = hdr->get_payload_size_including_header();
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_non_blocking_from_address(slot_addr, pkt_bytes);
                sent_mask |= (1u << slot);
                pending &= ~(1u << slot);
            }
            return sent_mask;
        }
#endif

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

        static FORCE_INLINE void send_to_forwarder(
            volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
            uint16_t dst_chip_id,
            uint16_t dst_mesh_id,
            uint32_t my_noc_x,
            uint32_t my_noc_y,
            uint32_t dst_l1_addr,
            uint32_t dst_sem_addr,
            uint32_t data_addr,
            uint32_t fc_noc_x,
            uint32_t fc_noc_y,
            uint32_t fc_slot_offset,
            uint32_t fc_sem_addr,
            uint32_t fc_slot_bit) {
            constexpr uint32_t pkt_hdr_bytes = sizeof(PACKET_HEADER_TYPE);

            set_unicast_route(packet_header, dst_chip_id, dst_mesh_id, static_cast<uint16_t>(1));

            uint64_t dst_noc_addr = get_noc_addr(my_noc_x, my_noc_y, dst_l1_addr);
            uint64_t dst_sem_noc_addr = get_noc_addr(my_noc_x, my_noc_y, dst_sem_addr);
            packet_header->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, dst_sem_noc_addr, 1, false},
                CTArgs::payload_size_bytes);

            uint32_t packet_base = get_write_ptr(CTArgs::packet_cb);
            uint32_t slot_l1 = packet_base + fc_slot_offset;
            uint64_t slot_hdr_noc = get_noc_addr(fc_noc_x, fc_noc_y, slot_l1);
            uint64_t slot_pay_noc = get_noc_addr(fc_noc_x, fc_noc_y, slot_l1 + pkt_hdr_bytes);

            noc_async_write(reinterpret_cast<uint32_t>(packet_header), slot_hdr_noc, pkt_hdr_bytes);
            noc_async_write(data_addr, slot_pay_noc, CTArgs::payload_size_bytes);
            noc_async_writes_flushed();

            uint64_t sem_noc = get_noc_addr(fc_noc_x, fc_noc_y, fc_sem_addr);
            noc_semaphore_inc(sem_noc, fc_slot_bit);
            noc_async_full_barrier();
        }
#endif

        void impl([[maybe_unused]] const RTArgs& args) {
            if constexpr (!IsWorkerCore) {
                return;
            }

#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC — FC: BWD forwarding; Worker: receive R1/R2/R3
            // ================================================================
            if constexpr (CTArgs::is_fabric_core) {
                // FC NCRISC: forward R1(B) + R2(A) to BWD neighbor
                // Identical pattern to sdpa_reduce_forwarder.hpp
                DPRINT << "NCRISC FC BWD start" << ENDL();

                const uint32_t buf_base = get_write_ptr(CTArgs::packet_cb) + CTArgs::ncrisc_buffer_offset;
                const uint32_t r1_base = buf_base;
                const uint32_t r2_base = buf_base + CTArgs::r2_buffer_offset;

                size_t arg_idx = 0;
                const uint32_t r1_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
                const uint32_t r2_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

                auto bwd_sender =
                    tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);

                volatile tt_l1_ptr uint32_t* r1_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r1_sem_addr);
                volatile tt_l1_ptr uint32_t* r2_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r2_sem_addr);

                bwd_sender.open();
                DPRINT << "NCRISC FC BWD conn open" << ENDL();
                {
                    uint32_t r1_sent = 0;
                    uint32_t r2_sent = 0;
                    do {
                        invalidate_l1_cache();
                        r1_sent = process_ready_slots(r1_sem, r1_sent, r1_base, bwd_sender);
                        r2_sent = process_ready_slots(r2_sem, r2_sent, r2_base, bwd_sender);
                    } while (r1_sent != CTArgs::all_sent_mask || r2_sent != CTArgs::all_sent_mask);
                    noc_semaphore_set(r1_sem, 0);
                    noc_semaphore_set(r2_sem, 0);
                }
                bwd_sender.close();
                noc_async_full_barrier();
                DPRINT << "NCRISC FC BWD done" << ENDL();
                return;
            }

            // Worker NCRISC — receive data for 3 rounds
            DPRINT << "NCRISC worker start" << ENDL();
            if constexpr (!SkipLocalCbPush) {
                cb_reserve_back(CTArgs::local_cb, CTArgs::num_tiles);
                cb_push_back(CTArgs::local_cb, CTArgs::num_tiles);
            }

            auto wait_round = [](uint32_t sem_addr) {
                volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
                noc_semaphore_wait_min(sem_ptr, 1);
                unified_kernels::semaphore_dec(sem_ptr);
            };

            cb_reserve_back(CTArgs::received_cb, CTArgs::num_tiles);
            wait_round(args.recv_sem_round1);
            cb_push_back(CTArgs::received_cb, CTArgs::num_tiles);
            DPRINT << "NCRISC R1 done" << ENDL();

            cb_reserve_back(CTArgs::received_cb, CTArgs::num_tiles);
            wait_round(args.recv_sem_round2);
            cb_push_back(CTArgs::received_cb, CTArgs::num_tiles);
            DPRINT << "NCRISC R2 done" << ENDL();

            cb_reserve_back(CTArgs::received_cb, CTArgs::num_tiles);
            wait_round(args.recv_sem_round3);
            cb_push_back(CTArgs::received_cb, CTArgs::num_tiles);
            DPRINT << "NCRISC R3 done" << ENDL();

#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC — FC: FWD + R3 forwarding; Worker: R1/R2/R3 via FC
            // ================================================================
            constexpr uint32_t pkt_hdr_bytes = sizeof(PACKET_HEADER_TYPE);

            if constexpr (CTArgs::is_fabric_core) {
                DPRINT << "BRISC FC start" << ENDL();

                const uint32_t buf_base = get_write_ptr(CTArgs::packet_cb);
                const uint32_t r1_base = buf_base;
                const uint32_t r2_base = buf_base + CTArgs::r2_buffer_offset;
                const uint32_t r3_base = buf_base + CTArgs::r3_buffer_offset;

                size_t arg_idx = 0;
                const uint32_t r1_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
                const uint32_t r2_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
                const uint32_t r3_sem_addr_val = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

                auto fwd_sender =
                    tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);

                volatile tt_l1_ptr uint32_t* r1_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r1_sem_addr);
                volatile tt_l1_ptr uint32_t* r2_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r2_sem_addr);
                volatile tt_l1_ptr uint32_t* r3_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r3_sem_addr_val);

                // Phase 1: forward R1(A)+R2(B) to FWD neighbor
                fwd_sender.open();
                DPRINT << "BRISC FC FWD open" << ENDL();
                {
                    uint32_t r1_sent = 0;
                    uint32_t r2_sent = 0;
                    do {
                        invalidate_l1_cache();
                        r1_sent = process_ready_slots(r1_sem, r1_sent, r1_base, fwd_sender);
                        r2_sent = process_ready_slots(r2_sem, r2_sent, r2_base, fwd_sender);
                    } while (r1_sent != CTArgs::all_sent_mask || r2_sent != CTArgs::all_sent_mask);
                    noc_semaphore_set(r1_sem, 0);
                    noc_semaphore_set(r2_sem, 0);
                }
                fwd_sender.close();
                noc_async_full_barrier();
                DPRINT << "BRISC FC FWD done" << ENDL();

                // Phase 2: forward R3 to cross-column partner
                auto r3_sender =
                    tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
                r3_sender.open();
                DPRINT << "BRISC FC R3 open" << ENDL();
                {
                    uint32_t r3_sent = 0;
                    do {
                        invalidate_l1_cache();
                        r3_sent = process_ready_slots(r3_sem, r3_sent, r3_base, r3_sender);
                    } while (r3_sent != CTArgs::r3_all_sent_mask);
                    noc_semaphore_set(r3_sem, 0);
                }
                r3_sender.close();
                noc_async_full_barrier();
                DPRINT << "BRISC FC R3 done" << ENDL();
                return;
            }

            // ----------------------------------------------------------
            // Worker BRISC: R1 + R2 + R3 via FC forwarders
            // ----------------------------------------------------------
            const uint32_t my_noc_x = my_x[0];
            const uint32_t my_noc_y = my_y[0];

            PacketHeaderPool::reset();
            auto* packet_header = PacketHeaderPool::allocate_header(1);

            DPRINT << "BRISC W start typeA=" << args.is_type_a << ENDL();

            // R1: Type A → FWD (fwd_dst), Type B → BWD (bwd_dst)
            {
                if constexpr (SkipLocalCbPush) {
                    cb_wait_front(CTArgs::local_cb, CTArgs::num_tiles);
                }
                uint32_t data_addr = get_read_ptr(CTArgs::local_cb);
                uint16_t r1_chip = args.is_type_a ? CTArgs::fwd_dst_chip_id : CTArgs::bwd_dst_chip_id;
                uint16_t r1_mesh = args.is_type_a ? CTArgs::fwd_dst_mesh_id : CTArgs::bwd_dst_mesh_id;
                send_to_forwarder(
                    packet_header,
                    r1_chip,
                    r1_mesh,
                    my_noc_x,
                    my_noc_y,
                    args.r1_dst_l1_addr,
                    args.r1_dst_sem_addr,
                    data_addr,
                    args.fc_noc_x,
                    args.fc_noc_y,
                    args.r1_slot_offset,
                    args.r1_sem_addr,
                    args.r1_slot_bit);
            }
            DPRINT << "BRISC W R1 done" << ENDL();

            // R2: Type A → BWD (bwd_dst), Type B → FWD (fwd_dst)
            {
                cb_wait_front(CTArgs::scratch_cb, CTArgs::num_tiles);
                uint32_t data_addr = get_read_ptr(CTArgs::scratch_cb);
                uint16_t r2_chip = args.is_type_a ? CTArgs::bwd_dst_chip_id : CTArgs::fwd_dst_chip_id;
                uint16_t r2_mesh = args.is_type_a ? CTArgs::bwd_dst_mesh_id : CTArgs::fwd_dst_mesh_id;
                send_to_forwarder(
                    packet_header,
                    r2_chip,
                    r2_mesh,
                    my_noc_x,
                    my_noc_y,
                    args.r2_dst_l1_addr,
                    args.r2_dst_sem_addr,
                    data_addr,
                    args.fc_noc_x,
                    args.fc_noc_y,
                    args.r2_slot_offset,
                    args.r2_sem_addr,
                    args.r2_slot_bit);

                cb_reserve_back(CTArgs::reload_cb, CTArgs::num_tiles);
                uint32_t reload_wr = get_write_ptr(CTArgs::reload_cb);
                uint64_t reload_noc = get_noc_addr(my_noc_x, my_noc_y, reload_wr);
                noc_async_write(data_addr, reload_noc, CTArgs::num_tiles * CTArgs::compute_tile_size);
                noc_async_write_barrier();
                cb_push_back(CTArgs::reload_cb, CTArgs::num_tiles);
                cb_pop_front(CTArgs::scratch_cb, CTArgs::num_tiles);
            }
            DPRINT << "BRISC W R2 done" << ENDL();

            // R3: send column sum to FC for cross-column forwarding
            {
                cb_wait_front(CTArgs::scratch_cb, CTArgs::num_tiles);
                uint32_t data_addr = get_read_ptr(CTArgs::scratch_cb);
                send_to_forwarder(
                    packet_header,
                    static_cast<uint16_t>(CTArgs::r3_dst_chip_id),
                    static_cast<uint16_t>(CTArgs::r3_dst_mesh_id),
                    my_noc_x,
                    my_noc_y,
                    args.r3_dst_l1_addr,
                    args.r3_dst_sem_addr,
                    data_addr,
                    args.fc_noc_x,
                    args.fc_noc_y,
                    args.r3_slot_offset,
                    args.r3_sem_addr,
                    args.r3_slot_bit);

                cb_reserve_back(CTArgs::reload_cb, CTArgs::num_tiles);
                uint32_t reload_wr = get_write_ptr(CTArgs::reload_cb);
                uint64_t reload_noc = get_noc_addr(my_noc_x, my_noc_y, reload_wr);
                noc_async_write(data_addr, reload_noc, CTArgs::num_tiles * CTArgs::compute_tile_size);
                noc_async_write_barrier();
                cb_push_back(CTArgs::reload_cb, CTArgs::num_tiles);
                cb_pop_front(CTArgs::scratch_cb, CTArgs::num_tiles);
            }
            DPRINT << "BRISC W R3 done" << ENDL();

            // Output: write final result
            {
                cb_wait_front(CTArgs::scratch_cb, CTArgs::num_tiles);
                uint32_t src_addr = get_read_ptr(CTArgs::scratch_cb);
                uint64_t output_noc = get_noc_addr(my_noc_x, my_noc_y, args.output_base_addr);
                noc_async_write(src_addr, output_noc, CTArgs::payload_size_bytes);
                noc_async_write_barrier();
                cb_pop_front(CTArgs::scratch_cb, CTArgs::num_tiles);
            }

            if constexpr (SkipLocalCbPush) {
                cb_pop_front(CTArgs::local_cb, CTArgs::num_tiles);
            }
            DPRINT << "BRISC W done" << ENDL();

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC — 3-round add_tiles (same as reduce_to_one_b1)
            // ================================================================
            if constexpr (CTArgs::is_fabric_core) {
                return;
            }

            reconfig_data_format<false, true>(CTArgs::local_cb, CTArgs::received_cb);
            pack_reconfig_data_format<true>(CTArgs::scratch_cb);

            // R1: local + received_R1
            add_tiles_init(CTArgs::local_cb, CTArgs::received_cb, true);
            cb_wait_front(CTArgs::local_cb, CTArgs::num_tiles);
            cb_wait_front(CTArgs::received_cb, CTArgs::num_tiles);

            tile_regs_acquire();
            for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                add_tiles(CTArgs::local_cb, CTArgs::received_cb, i, i, i);
            }
            cb_pop_front(CTArgs::local_cb, CTArgs::num_tiles);
            cb_pop_front(CTArgs::received_cb, CTArgs::num_tiles);

            cb_reserve_back(CTArgs::scratch_cb, CTArgs::num_tiles);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                pack_tile(i, CTArgs::scratch_cb, i);
            }
            tile_regs_release();
            cb_push_back(CTArgs::scratch_cb, CTArgs::num_tiles);

            // R2: reload(R1 sum) + received_R2
            add_tiles_init(CTArgs::reload_cb, CTArgs::received_cb, true);
            cb_wait_front(CTArgs::reload_cb, CTArgs::num_tiles);
            cb_wait_front(CTArgs::received_cb, CTArgs::num_tiles);

            tile_regs_acquire();
            for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                add_tiles(CTArgs::reload_cb, CTArgs::received_cb, i, i, i);
            }
            cb_pop_front(CTArgs::reload_cb, CTArgs::num_tiles);
            cb_pop_front(CTArgs::received_cb, CTArgs::num_tiles);

            cb_reserve_back(CTArgs::scratch_cb, CTArgs::num_tiles);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                pack_tile(i, CTArgs::scratch_cb, i);
            }
            tile_regs_release();
            cb_push_back(CTArgs::scratch_cb, CTArgs::num_tiles);

            // R3: reload(column sum) + received_R3
            add_tiles_init(CTArgs::reload_cb, CTArgs::received_cb, true);
            cb_wait_front(CTArgs::reload_cb, CTArgs::num_tiles);
            cb_wait_front(CTArgs::received_cb, CTArgs::num_tiles);

            tile_regs_acquire();
            for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                add_tiles(CTArgs::reload_cb, CTArgs::received_cb, i, i, i);
            }
            cb_pop_front(CTArgs::reload_cb, CTArgs::num_tiles);
            cb_pop_front(CTArgs::received_cb, CTArgs::num_tiles);

            cb_reserve_back(CTArgs::scratch_cb, CTArgs::num_tiles);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                pack_tile(i, CTArgs::scratch_cb, i);
            }
            tile_regs_release();
            cb_push_back(CTArgs::scratch_cb, CTArgs::num_tiles);
#endif
        }
    };
};

}  // namespace deepseek_b1_ops
