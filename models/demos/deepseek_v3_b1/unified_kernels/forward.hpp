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
            if constexpr (IsWorkerCore && CTArgs::is_entry_column) {
                DPRINT << "start of forward op (entry)\n";
                static_assert(noc_mode == DM_DYNAMIC_NOC);
                SocketReceiverInterface recv = create_receiver_socket_interface(args.socket_config_addr);
                set_receiver_socket_page_size(recv, args.socket_page_size);
                socket_wait_for_pages(recv, args.socket_num_pages);
                cb_reserve_back(CTArgs::cb0_id, CTArgs::num_pages_to_read);
                invalidate_l1_cache();
                noc_async_read(
                    get_noc_addr(recv.read_ptr), get_write_ptr(CTArgs::cb0_id), args.socket_page_size, 1 - noc_index);
                noc_async_read_barrier(1 - noc_index);
                cb_push_back(CTArgs::cb0_id, CTArgs::num_pages_to_read);

                socket_pop_pages(recv, args.socket_num_pages);
                socket_notify_sender(recv, 1 - noc_index);
                update_socket_config(recv);
                DPRINT << "end of forward op (entry)\n";
            } else if constexpr (IsWorkerCore && !CTArgs::is_entry_column) {
                DPRINT << "forward op (non-entry): BRISC no-op\n";
            }

#elif defined(COMPILE_FOR_NCRISC)
            if constexpr (IsWorkerCore && CTArgs::is_entry_column) {
                DPRINT << "start of forward op NCRISC (entry)\n";
                cb_wait_front(CTArgs::cb0_id, CTArgs::num_pages_to_read);
                const uint32_t src = get_read_ptr(CTArgs::cb0_id);
                constexpr uint32_t tensor_size_bytes = CTArgs::tensor_page_size * CTArgs::num_pages_to_read;
                const uint64_t dst = get_noc_addr(args.my_noc_x, args.my_noc_y, args.tensor_address, 0);
                DPRINT << "FWD_NCRISC_DBG (entry): src_cb=" << CTArgs::cb0_id << " src_l1_addr=" << HEX() << src
                       << " tensor_address=" << args.tensor_address << " bytes=" << DEC() << tensor_size_bytes
                       << " noc=(" << args.my_noc_x << "," << args.my_noc_y << ")\n";
                noc_async_write(src, dst, tensor_size_bytes);
                noc_async_write_barrier();
                // After write+barrier, dump first 8 u16 from the tensor L1 region to confirm
                // the embedding actually landed at tensor_address (vs. e.g. wrong NOC route).
                {
                    invalidate_l1_cache();
                    volatile tt_l1_ptr uint16_t* ptr =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(args.tensor_address);
                    DPRINT << "FWD_NCRISC_DBG: post-write tensor[0..8] (raw u16): " << HEX() << ptr[0] << " " << ptr[1]
                           << " " << ptr[2] << " " << ptr[3] << " " << ptr[4] << " " << ptr[5] << " " << ptr[6] << " "
                           << ptr[7] << DEC() << "\n";
                }

                if constexpr (CTArgs::enable_cross_column) {
                    DPRINT << "forward: sending cross-column via fabric\n";
                    const uint32_t staging_addr = src;

                    size_t fabric_arg_idx = 0;
                    auto sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(
                        fabric_arg_idx);
                    sender.open();

                    constexpr uint32_t pkt_hdr_bytes = sizeof(PACKET_HEADER_TYPE);
                    uint32_t bytes_sent = 0;
                    for (uint32_t pkt = 0; pkt < CTArgs::num_fabric_packets; pkt++) {
                        uint32_t chunk_size = CTArgs::fabric_max_payload;
                        if (bytes_sent + chunk_size > CTArgs::cross_column_payload) {
                            chunk_size = CTArgs::cross_column_payload - bytes_sent;
                        }

                        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr =
                            reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(staging_addr);
                        fabric_set_unicast_route(
                            hdr,
                            static_cast<uint16_t>(args.partner_chip_id),
                            static_cast<uint16_t>(args.partner_mesh_id));
                        uint64_t dest_noc =
                            get_noc_addr(args.partner_noc_x, args.partner_noc_y, args.partner_tensor_addr + bytes_sent);
                        uint64_t sem_noc =
                            get_noc_addr(args.partner_noc_x, args.partner_noc_y, args.cross_col_sem_addr);
                        hdr->to_noc_fused_unicast_write_atomic_inc(
                            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dest_noc, sem_noc, 1, false},
                            chunk_size);
                        DPRINT << "FWD_NCRISC_DBG: cross-col pkt "
                               << " dest_noc=" << (uint32_t)dest_noc << " sem_noc=" << (uint32_t)sem_noc
                               << " chunk_size=" << (uint32_t)chunk_size << "\n";

                        noc_async_write(
                            args.tensor_address + bytes_sent,
                            get_noc_addr(args.my_noc_x, args.my_noc_y, staging_addr + pkt_hdr_bytes),
                            chunk_size);
                        noc_async_write_barrier();

                        sender.wait_for_empty_write_slot();
                        sender.send_payload_flush_non_blocking_from_address(staging_addr, pkt_hdr_bytes + chunk_size);

                        bytes_sent += chunk_size;
                    }
                    sender.close();
                    noc_async_full_barrier();
                    DPRINT << "forward: cross-column send done\n";
                }

                cb_pop_front(CTArgs::cb0_id, CTArgs::num_pages_to_read);
                DPRINT << "end of forward op NCRISC (entry)\n";
            } else if constexpr (IsWorkerCore && !CTArgs::is_entry_column) {
                DPRINT << "start of forward op NCRISC (non-entry): waiting for fabric"
                       << " tensor_address=" << HEX() << args.tensor_address << DEC() << "\n";
                volatile tt_l1_ptr uint32_t* sem =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.cross_col_sem_addr);
                while (*sem < CTArgs::num_fabric_packets) {
                    invalidate_l1_cache();
                }
                noc_semaphore_set(sem, 0);
                {
                    invalidate_l1_cache();
                    volatile tt_l1_ptr uint16_t* ptr =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(args.tensor_address);
                    DPRINT << "FWD_NCRISC_DBG (non-entry): post-fabric tensor[0..8] (raw u16): " << HEX() << ptr[0]
                           << " " << ptr[1] << " " << ptr[2] << " " << ptr[3] << " " << ptr[4] << " " << ptr[5] << " "
                           << ptr[6] << " " << ptr[7] << DEC() << "\n";
                }
                DPRINT << "end of forward op NCRISC (non-entry): data received\n";
            }

#elif defined(COMPILE_FOR_TRISC)
            // No-op — forward is dataflow only.
#endif
        }
    };

};  // struct Forward

}  // namespace deepseek_b1_ops
