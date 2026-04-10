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
 * BRISC (sender core): socket → bcast_pkt_cb  (via CB reserve/push)
 * NCRISC (sender core): bcast_pkt_cb → tensor  (via CB wait/pop + noc write)
 * TRISC: no-op
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
#include <cstdint>

using address_t = uint32_t;
#endif

namespace deepseek_b1_ops {

struct Forward {
    // ========================================================================
    // Compile-time args
    // ========================================================================

    template <uint32_t cb0Id, uint32_t NumPagesToRead>
    struct ReaderCTArgs {
        static constexpr uint32_t cb0_id = cb0Id;
        static constexpr uint32_t num_pages_to_read = NumPagesToRead;
    };

    template <uint32_t cb0Id, uint32_t NumPagesToRead, uint32_t TensorPageSize>
    struct WriterCTArgs {
        static constexpr uint32_t cb0_id = cb0Id;
        static constexpr uint32_t num_pages_to_read = NumPagesToRead;
        static constexpr uint32_t tensor_page_size = TensorPageSize;
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
            // Read from per-device socket into the intermediate CB.
            DPRINT << "start of forwrad op\n";
            if constexpr (IsWorkerCore) {
                DPRINT << "is worker core\n";
                static_assert(noc_mode == DM_DYNAMIC_NOC);
                SocketReceiverInterface recv = create_receiver_socket_interface(args.socket_config_addr);
                DPRINT << "setting socket page size " << (uint32_t)args.socket_page_size << "\n";
                set_receiver_socket_page_size(recv, args.socket_page_size);
                DPRINT << "reserving pages\n";
                socket_wait_for_pages(recv, args.socket_num_pages);
                DPRINT << "received pages and reserving CB\n";
                cb_reserve_back(CTArgs::cb0_id, CTArgs::num_pages_to_read);
                invalidate_l1_cache();
                noc_async_read(
                    get_noc_addr(recv.read_ptr), get_write_ptr(CTArgs::cb0_id), args.socket_page_size, 1 - noc_index);
                DPRINT << "issued async read, waiting for completion\n";
                noc_async_read_barrier(1 - noc_index);
                cb_push_back(CTArgs::cb0_id, CTArgs::num_pages_to_read);

                socket_pop_pages(recv, args.socket_num_pages);
                socket_notify_sender(recv, 1 - noc_index);
                DPRINT << "finished socket operations\n";
                update_socket_config(recv);
                DPRINT << "end of forward op\n";
            }

#elif defined(COMPILE_FOR_NCRISC)
            // Copy from intermediate CB into the local residual tensor.
            DPRINT << "start of forward op\n";
            if constexpr (IsWorkerCore) {
                DPRINT << "is worker core\n";
                cb_wait_front(CTArgs::cb0_id, CTArgs::num_pages_to_read);
                DPRINT << "CB wait complete, issuing NOC write to tensor\n";
                const uint32_t src = get_read_ptr(CTArgs::cb0_id);
                constexpr uint32_t tensor_size_bytes = CTArgs::tensor_page_size * CTArgs::num_pages_to_read;
                const uint64_t dst = get_noc_addr(args.my_noc_x, args.my_noc_y, args.tensor_address, 0);
                noc_async_write(src, dst, tensor_size_bytes);
                DPRINT << "issued async write, waiting for completion\n";
                noc_async_write_barrier();
                cb_pop_front(CTArgs::cb0_id, CTArgs::num_pages_to_read);
                DPRINT << "finished NOC write, end of forward op\n";
            }

#elif defined(COMPILE_FOR_TRISC)
            // No-op — forward is dataflow only.
#endif
        }
    };

};  // struct Forward

}  // namespace deepseek_b1_ops
