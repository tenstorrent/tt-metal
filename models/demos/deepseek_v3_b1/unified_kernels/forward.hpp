// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * Forward — per-device socket read into residual tensor.
 *
 * Replaces Broadcast when every device has its own upstream socket (parallel
 * socket pipeline). Each device independently reads its socket data into the
 * residual tensor so that the downstream residual mcast + RMSNorm can proceed
 * identically to the broadcast path.
 *
 * When cross-column distribution is enabled (isEntryColumn CT arg):
 *   Entry column BRISC: socket → bcast_pkt_cb
 *   Entry column NCRISC: bcast_pkt_cb → tensor + fabric send to partner device
 *   Non-entry column BRISC: no-op
 *   Non-entry column NCRISC: wait for fabric data (written directly to tensor)
 *   TRISC: no-op
 */

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include <cstdint>

using address_t = uint32_t;

#elif defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include <cstdint>

using address_t = uint32_t;
#endif

namespace deepseek_b1_ops {

struct Forward {
    // ========================================================================
    // Compile-time args
    // ========================================================================

    template <uint32_t cb0Id, uint32_t NumPagesToRead, uint32_t IsEntryColumn = 1>
    struct ReaderCTArgs {
        static constexpr uint32_t cb0_id = cb0Id;
        static constexpr uint32_t num_pages_to_read = NumPagesToRead;
        static constexpr uint32_t is_entry_column = IsEntryColumn;
    };

    template <
        uint32_t cb0Id,
        uint32_t NumPagesToRead,
        uint32_t TensorPageSize,
        uint32_t IsEntryColumn = 1,
        uint32_t FabricMaxPayload = 0,
        uint32_t NumFabricPackets = 0,
        uint32_t CrossColumnPayload = 0>
    struct WriterCTArgs {
        static constexpr uint32_t cb0_id = cb0Id;
        static constexpr uint32_t num_pages_to_read = NumPagesToRead;
        static constexpr uint32_t tensor_page_size = TensorPageSize;
        static constexpr uint32_t is_entry_column = IsEntryColumn;
        static constexpr uint32_t fabric_max_payload = FabricMaxPayload;
        static constexpr uint32_t num_fabric_packets = NumFabricPackets;
        static constexpr uint32_t cross_column_payload = CrossColumnPayload;
        static constexpr bool enable_cross_column = (NumFabricPackets > 0);
    };

    struct ComputeCTArgs {};

    // ========================================================================
    // Runtime args
    // ========================================================================

    struct ReaderArgs {
        uint32_t socket_config_addr;
        uint32_t socket_page_size;
        uint32_t socket_num_pages;
    };

    struct WriterArgs {
        uint32_t tensor_address;
        uint32_t my_noc_x;
        uint32_t my_noc_y;
        uint32_t cross_col_sem_addr;
        uint32_t partner_tensor_addr;
        uint32_t partner_noc_x;
        uint32_t partner_noc_y;
        uint32_t partner_chip_id;
        uint32_t partner_mesh_id;
    };

    struct ComputeArgs {};

    using RTArgs = unified_kernels::SelectByRISCV<WriterArgs, ReaderArgs, ComputeArgs>;

    // ========================================================================
    // Op
    // ========================================================================

    template <typename CTArgs, bool IsWorkerCore>
    class Op {
    public:
        void operator()(const RTArgs& args) {
            if constexpr (IsWorkerCore) {
                impl(args);
            }
        }

    private:
        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_BRISC)
            DPRINT << "forward op: socket_config_addr=" << (uint32_t)args.socket_config_addr
                   << " socket_page_size=" << (uint32_t)args.socket_page_size
                   << " socket_num_pages=" << (uint32_t)args.socket_num_pages << "\n";
            DPRINT << "FORWARD: entry_column=" << CTArgs::is_entry_column << "\n";
            DPRINT << "forward is worker_core=" << (uint32_t)IsWorkerCore << "\n";
            if constexpr (IsWorkerCore && CTArgs::is_entry_column) {
                static_assert(noc_mode == DM_DYNAMIC_NOC);
                DPRINT << "forward op (entry): BRISC socket read into residual tensor\n";
                SocketReceiverInterface recv = create_receiver_socket_interface(args.socket_config_addr);
                DPRINT << "socket created\n";
                set_receiver_socket_page_size(recv, args.socket_page_size);
                DPRINT << "socket page size set\n";
                socket_wait_for_pages(recv, args.socket_num_pages);
                DPRINT << "socket pages available\n";
                cb_reserve_back(CTArgs::cb0_id, CTArgs::num_pages_to_read);
                DPRINT << "socket read: reserved " << (uint32_t)CTArgs::num_pages_to_read << " pages in cb\n";
                invalidate_l1_cache();
                noc_async_read(
                    get_noc_addr(recv.read_ptr), get_write_ptr(CTArgs::cb0_id), args.socket_page_size, 1 - noc_index);
                noc_async_read_barrier(1 - noc_index);
                DPRINT << "socket read done, pushing to CB\n";
                cb_push_back(CTArgs::cb0_id, CTArgs::num_pages_to_read);

                socket_pop_pages(recv, args.socket_num_pages);
                socket_notify_sender(recv, 1 - noc_index);
                DPRINT << "socket read complete, popped pages and notified sender\n";
                update_socket_config(recv);
                DPRINT << "end of forward op (entry)\n";
            } else if constexpr (IsWorkerCore && !CTArgs::is_entry_column) {
                DPRINT << "forward op (non-entry): BRISC no-op\n";
            }

#elif defined(COMPILE_FOR_NCRISC)
            DPRINT << "forward op is worker core: " << (uint32_t)IsWorkerCore << "\n";
            DPRINT << "forward op is entry column: " << CTArgs::is_entry_column << "\n";
            if constexpr (IsWorkerCore && CTArgs::is_entry_column) {
                // Wait for BRISC's socket-staged payload in cb0. cb0 is bound to a
                // separate persistent staging region (sdpa_out_interm scratch in
                // op.py:_overlap_cbs_with_sdpa_buffer) — *not* the residual tensor.
                // This mirrors main's Broadcast, where bcast_pkt_cb aliases the
                // standalone bcast_input_tensor (≠ residual_mcast_src tensor).
                DPRINT << "forward op (entry): NCRISC waiting for data in cb\n";
                cb_wait_front(CTArgs::cb0_id, CTArgs::num_pages_to_read);
                const uint32_t src = get_read_ptr(CTArgs::cb0_id);
                constexpr uint32_t tensor_size_bytes = CTArgs::tensor_page_size * CTArgs::num_pages_to_read;
                const uint64_t dst = get_noc_addr(args.my_noc_x, args.my_noc_y, args.tensor_address, 0);
                // Real local copy from the staging buffer into the residual tensor
                // so downstream residual_mcast / RMSNorm can read it. Identical to
                // Broadcast root's noc_async_write(src=bcast_input, dst=residual).
                noc_async_write(src, dst, tensor_size_bytes);
                noc_async_write_barrier();

                if constexpr (CTArgs::enable_cross_column) {
                    // BCAST-style fabric send: header lives in PacketHeaderPool (a
                    // separate L1 region managed by the fabric stack), and the data
                    // is shipped *directly* from the persistent source buffer via
                    // send_payload_without_header. This is the key invariant that
                    // lets the receiver-side semaphore double as a data-ready signal
                    // without needing an extra barrier semaphore — see Broadcast.
                    PacketHeaderPool::reset();

                    size_t fabric_arg_idx = 0;
                    auto sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(
                        fabric_arg_idx);
                    sender.open_start();
                    volatile PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
                    fabric_set_unicast_route(
                        hdr, static_cast<uint16_t>(args.partner_chip_id), static_cast<uint16_t>(args.partner_mesh_id));
                    sender.open_finish();

                    const uint64_t sem_noc =
                        get_noc_addr(args.partner_noc_x, args.partner_noc_y, args.cross_col_sem_addr);
                    uint32_t bytes_sent = 0;
                    for (uint32_t pkt = 0; pkt < CTArgs::num_fabric_packets; pkt++) {
                        uint32_t chunk_size = CTArgs::fabric_max_payload;
                        if (bytes_sent + chunk_size > CTArgs::cross_column_payload) {
                            chunk_size = CTArgs::cross_column_payload - bytes_sent;
                        }
                        const uint64_t dest_noc =
                            get_noc_addr(args.partner_noc_x, args.partner_noc_y, args.partner_tensor_addr + bytes_sent);
                        hdr->to_noc_fused_unicast_write_atomic_inc(
                            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dest_noc, sem_noc, 1, false},
                            chunk_size);

                        sender.wait_for_empty_write_slot();
                        // Ship pure data straight from the persistent source buffer.
                        sender.send_payload_without_header_non_blocking_from_address(src + bytes_sent, chunk_size);
                        // Ship the header from the pool — never touches the data buffer.
                        sender.send_payload_flush_non_blocking_from_address(
                            reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));

                        // Single-header reuse: the NOC reads of `hdr` issued by the
                        // flush above are non-blocking and may still be in flight when
                        // we next overwrite `hdr` via to_noc_fused_unicast_write_atomic_inc.
                        // wait_for_empty_write_slot() waits on EDM slot availability,
                        // *not* on our L1 reads draining. Flush here so the next iter's
                        // header write doesn't race with the previous send. Mirrors
                        // broadcast.hpp's noc_async_writes_flushed() between rotations.
                        noc_async_writes_flushed();

                        bytes_sent += chunk_size;
                    }
                    sender.close();
                    noc_async_full_barrier();
                    DPRINT << "forward: cross-column send done\n";
                }

                cb_pop_front(CTArgs::cb0_id, CTArgs::num_pages_to_read);
                DPRINT << "end of forward op NCRISC (entry)\n";
            } else if constexpr (IsWorkerCore && !CTArgs::is_entry_column) {
                volatile tt_l1_ptr uint32_t* sem =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.cross_col_sem_addr);
                noc_semaphore_wait_min(sem, CTArgs::num_fabric_packets);
                // Atomic decrement, NOT noc_semaphore_set(sem, 0). The next iteration's
                // entry device may have already started fabric-sending; an in-flight
                // atomic_inc that arrives between wait_min returning and us zeroing
                // `sem` would be silently dropped, causing the next wait to hang.
                // Same pattern as broadcast.hpp's semaphore_dec(sem_ptrs[link_idx], ...).
                unified_kernels::semaphore_dec(sem, CTArgs::num_fabric_packets);

                DPRINT << "end of forward op NCRISC (non-entry): data received\n";
            }

#elif defined(COMPILE_FOR_TRISC)
            // No-op — forward is dataflow only.
#endif
        }
    };

};  // struct Forward

}  // namespace deepseek_b1_ops
