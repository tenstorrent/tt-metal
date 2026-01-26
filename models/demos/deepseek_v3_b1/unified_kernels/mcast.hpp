// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// Mcast utility functions (inlined from mcast_utils.hpp)
// ============================================================================

#if defined(COMPILE_FOR_BRISC)

template <uint8_t noc>
FORCE_INLINE uint64_t get_noc_multicast_addr(
    uint32_t noc_x_start, uint32_t noc_y_start, uint32_t noc_x_end, uint32_t noc_y_end, uint32_t addr) {
    if constexpr (noc == 0) {
        return NOC_MULTICAST_ADDR(
            DYNAMIC_NOC_X(noc, noc_x_start),
            DYNAMIC_NOC_Y(noc, noc_y_start),
            DYNAMIC_NOC_X(noc, noc_x_end),
            DYNAMIC_NOC_Y(noc, noc_y_end),
            addr);
    } else {
        return NOC_MULTICAST_ADDR(
            DYNAMIC_NOC_X(noc, noc_x_end),
            DYNAMIC_NOC_Y(noc, noc_y_end),
            DYNAMIC_NOC_X(noc, noc_x_start),
            DYNAMIC_NOC_Y(noc, noc_y_start),
            addr);
    }
}

template <
    uint32_t mcast_num_cores,
    bool loopback,
    bool is_part_of_receiver_grid,
    bool linked,
    bool posted,
    bool set_addresses,
    bool set_size,
    uint8_t cmd_buf>
FORCE_INLINE void mcast_send_set_state(uint32_t src_local_addr, uint64_t dst_noc_addr, uint32_t len_bytes = 0) {
    constexpr uint32_t noc = noc_index;
    constexpr uint32_t vc = NOC_MULTICAST_WRITE_VC;
    constexpr bool multicast_path_reserve = true;

    while (!noc_cmd_buf_ready(noc, cmd_buf));
    uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) |
                             (linked ? NOC_CMD_VC_LINKED : 0x0) | (multicast_path_reserve ? NOC_CMD_PATH_RESERVE : 0) |
                             (loopback ? NOC_CMD_BRCST_SRC_INCLUDE : 0) | NOC_CMD_BRCST_PACKET |
                             (posted ? 0 : NOC_CMD_RESP_MARKED);

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    // Handles writing to PCIe
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dst_noc_addr >> 32) & NOC_PCIE_MASK);
    NOC_CMD_BUF_WRITE_REG(
        noc, cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dst_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_BRCST_EXCLUDE, 0);
    if constexpr (set_size) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    }
    if constexpr (set_addresses) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_local_addr);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)(dst_noc_addr));
    }
}

template <
    uint32_t mcast_num_cores,
    bool loopback,
    bool is_part_of_receiver_grid,
    bool linked,
    bool posted,
    bool set_addresses,
    bool set_size,
    uint8_t cmd_buf>
FORCE_INLINE void mcast_send_with_state(uint32_t src_local_addr, uint32_t dst_local_addr, uint32_t len_bytes = 0) {
    constexpr uint32_t noc = noc_index;
    if constexpr (loopback) {
        static_assert(is_part_of_receiver_grid, "Loopback mode is only supported for receiver grid");
    }
    constexpr uint32_t num_dests =
        loopback ? mcast_num_cores : (is_part_of_receiver_grid ? mcast_num_cores - 1 : mcast_num_cores);

    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        if constexpr (posted) {
            inc_noc_counter_val<proc_type, NocBarrierType::POSTED_WRITES_NUM_ISSUED>(noc, 1);
        } else {
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, 1);
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, num_dests);
        }
    }

    while (!noc_cmd_buf_ready(noc, cmd_buf));

    if constexpr (set_size) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    }
    if constexpr (set_addresses) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_local_addr);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, dst_local_addr);
    }
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        if constexpr (posted) {
            noc_posted_writes_num_issued[noc] += 1;
        } else {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += num_dests;
        }
    }
}

template <uint32_t mcast_num_cores, bool loopback, bool is_part_of_receiver_grid>
FORCE_INLINE void init_persistent_mcast_sender(
    uint64_t mcast_flag_noc_addr,
    uint32_t data_sender_semaphore_addr,
    volatile tt_l1_ptr uint32_t* data_sender_semaphore_addr_ptr) {
    mcast_send_set_state<mcast_num_cores, loopback, is_part_of_receiver_grid, true, true, false, false, write_cmd_buf>(
        0, mcast_flag_noc_addr, 0);
    mcast_send_set_state<
        mcast_num_cores,
        loopback,
        is_part_of_receiver_grid,
        true,
        true,
        true,
        true,
        write_reg_cmd_buf>(data_sender_semaphore_addr, mcast_flag_noc_addr, 4);
    mcast_send_with_state<
        mcast_num_cores,
        loopback,
        is_part_of_receiver_grid,
        true,
        true,
        false,
        false,
        write_reg_cmd_buf>(0, 0, 0);
    noc_async_posted_writes_flushed();
    noc_semaphore_set(data_sender_semaphore_addr_ptr, VALID);
}

template <uint32_t mcast_num_cores, bool loopback, bool is_part_of_receiver_grid>
FORCE_INLINE void teardown_persistent_mcast_sender(uint64_t mcast_flag_noc_addr) {
    mcast_send_set_state<
        mcast_num_cores,
        loopback,
        is_part_of_receiver_grid,
        false,
        false,
        false,
        false,
        write_reg_cmd_buf>(0, mcast_flag_noc_addr, 0);
    mcast_send_with_state<
        mcast_num_cores,
        loopback,
        is_part_of_receiver_grid,
        false,
        false,
        false,
        false,
        write_reg_cmd_buf>(0, 0, 0);
    noc_async_write_barrier();
    riscv_wait(1000);  // This is just to guarantee safety due to posted mcast hw bug
}

#endif  // defined(COMPILE_FOR_BRISC)

// ============================================================================
// Mcast micro-op
//
// Multicasts data from a single sender core to multiple receiver cores.
// Sender runs on BRISC, Receiver runs on NCRISC.
//
// CB States:
//   BRISC (Sender):
//     - Waits: src_cb (src_num_pages)
//     - Pops: src_cb (src_num_pages) if pop_src=true
//   NCRISC (Receiver):
//     - Reserves: dst_cb (dst_num_pages)
//     - Pushes: dst_cb (dst_num_pages)
//   TRISC: No-op
//
// Semaphore States:
//   Sender: Assumes sender_semaphore contains VALID (set during init)
//   Receiver: Waits for receiver_semaphore == VALID, then resets to INVALID
//
// Note: Sender assumes that receiver's dst_cb is ready to receive at the beginning of NCRISC execution.
// ============================================================================
struct Mcast {
    // ========================================================================
    // Compile-time args structs - different per RISC
    // Only what MUST be compile-time (used as template parameters)
    // ========================================================================

    // Sender CTArgs (BRISC): mcast_num_cores, is_part_of_receiver_grid
    // loopback is inferred: if sender is part of receiver grid, it needs loopback to receive its own mcast
    template <uint32_t McastNumCores, bool IsPartOfReceiverGrid>
    struct SenderCTArgs {
        static constexpr uint32_t mcast_num_cores = McastNumCores;
        static constexpr bool is_part_of_receiver_grid = IsPartOfReceiverGrid;
        static constexpr bool loopback = IsPartOfReceiverGrid;  // Inferred from is_part_of_receiver_grid
    };

    // Receiver CTArgs (NCRISC): none needed
    struct ReceiverCTArgs {};

    // Compute CTArgs (TRISC): none needed
    struct ComputeCTArgs {};

    // ========================================================================
    // Runtime args structs - different layout per RISC
    // ========================================================================

    // Sender args (BRISC): all runtime parameters
    struct SenderArgs {
        uint32_t dest_noc_start_x;
        uint32_t dest_noc_start_y;
        uint32_t dest_noc_end_x;
        uint32_t dest_noc_end_y;
        uint32_t data_sender_semaphore_id;
        uint32_t data_receiver_semaphore_id;
        uint32_t data_size_bytes;
        uint32_t src_cb;
        uint32_t src_num_pages;
        uint32_t input_data_addr;
        uint32_t mcast_receiver_data_addr;
    };

    // Receiver args (NCRISC): all runtime parameters
    struct ReceiverArgs {
        uint32_t data_receiver_semaphore_id;
        uint32_t dst_cb;
        uint32_t dst_num_pages;
    };

    // Compute args (TRISC) - not used for mcast (dataflow only)
    struct ComputeArgs {};

    // Note: For mcast, BRISC=Sender, NCRISC=Receiver
    using RTArgs = unified_kernels::SelectByRISCV<ReceiverArgs, SenderArgs, ComputeArgs>;

    // ========================================================================
    // Op - the actual operation
    //
    // CTArgsT: compile-time args (mcast_num_cores, loopback, is_part_of_receiver_grid)
    // IsSenderCore: compile-time flag to distinguish sender vs receiver cores
    // IsReceiverCore: compile-time flag for receiver cores
    // pop_src: whether to pop the source CB after sending
    //
    // Usage:
    //   Op op;
    //   op.init(args);      // Initialize persistent mcast sender (call once)
    //   op(args);           // Send data (can be called multiple times)
    //   op.teardown();      // Teardown persistent mcast sender (call once)
    //
    // Or use the legacy all-in-one call:
    //   op.init_send_teardown(args);  // Does init + send + teardown
    // ========================================================================
    template <typename CTArgsT, bool IsSenderCore, bool IsMcastGridCore, bool IsReceiverCore, bool pop_src>
    class Op {
    public:
        // ====================================================================
        // init - Initialize persistent mcast sender (BRISC only)
        // Must be called before operator() on sender core
        // No-op for NCRISC/TRISC
        // ====================================================================
        void init([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_BRISC)
            if constexpr (IsSenderCore) {
                // Get semaphore addresses from runtime args
                uint32_t data_sender_semaphore_addr = get_semaphore(args.data_sender_semaphore_id);
                uint32_t data_receiver_semaphore_addr = get_semaphore(args.data_receiver_semaphore_id);

                // Compute multicast NOC address and store for later use
                noc_coord_ = get_noc_multicast_addr<noc_index>(
                    args.dest_noc_start_x, args.dest_noc_start_y, args.dest_noc_end_x, args.dest_noc_end_y, 0);
                mcast_flag_noc_addr_ = noc_coord_ | (uint64_t)(data_receiver_semaphore_addr);

                volatile tt_l1_ptr uint32_t* data_sender_semaphore_addr_ptr =
                    (volatile tt_l1_ptr uint32_t*)data_sender_semaphore_addr;

                // Initialize persistent mcast sender
                init_persistent_mcast_sender<
                    CTArgsT::mcast_num_cores,
                    CTArgsT::loopback,
                    CTArgsT::is_part_of_receiver_grid>(
                    mcast_flag_noc_addr_, data_sender_semaphore_addr, data_sender_semaphore_addr_ptr);
            }
#endif
        }

        // ====================================================================
        // operator() - Send data via mcast (BRISC sender) / Full receive logic (NCRISC receiver)
        // For BRISC: Must call init() first, waits for src CB, sends data
        // For NCRISC: Reserve CB, wait for semaphore, push CB (complete receive)
        // ====================================================================
        void operator()(const RTArgs& args) { impl(args); }

        // ====================================================================
        // teardown - Teardown persistent mcast sender (BRISC only)
        // Must be called after all operator() calls on sender core
        // No-op for NCRISC/TRISC
        // ====================================================================
        void teardown() {
#if defined(COMPILE_FOR_BRISC)
            if constexpr (IsSenderCore) {
                // Teardown persistent mcast sender
                teardown_persistent_mcast_sender<
                    CTArgsT::mcast_num_cores,
                    CTArgsT::loopback,
                    CTArgsT::is_part_of_receiver_grid>(mcast_flag_noc_addr_);
            }
#endif
        }

    private:
        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_BRISC)
            if constexpr (IsSenderCore) {
                // Wait for source CB data to be ready
                cb_wait_front(args.src_cb, args.src_num_pages);
                // Compute mcast data NOC address from runtime args
                uint64_t mcast_data_noc_addr = noc_coord_ | (uint64_t)args.mcast_receiver_data_addr;

                // Send data with state
                mcast_send_with_state<
                    CTArgsT::mcast_num_cores,
                    CTArgsT::loopback,
                    CTArgsT::is_part_of_receiver_grid,
                    true,
                    true,
                    true,
                    true,
                    write_cmd_buf>(args.input_data_addr, mcast_data_noc_addr, args.data_size_bytes);
                mcast_send_with_state<
                    CTArgsT::mcast_num_cores,
                    CTArgsT::loopback,
                    CTArgsT::is_part_of_receiver_grid,
                    true,
                    true,
                    false,
                    false,
                    write_reg_cmd_buf>(0, 0, 0);

                // Pop the source CB after sending
                if constexpr (pop_src) {
                    cb_pop_front(args.src_cb, args.src_num_pages);
                }
            }
#elif defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC - Receiver cores: reserve, wait, push (all in operator)
            // ================================================================
            if constexpr (IsReceiverCore) {
                // Reserve space in destination CB before mcast writes to it
                cb_reserve_back(args.dst_cb, args.dst_num_pages);

                uint32_t data_receiver_semaphore_addr = get_semaphore(args.data_receiver_semaphore_id);

                volatile tt_l1_ptr uint32_t* data_receiver_semaphore_addr_ptr =
                    (volatile tt_l1_ptr uint32_t*)(data_receiver_semaphore_addr);
                noc_semaphore_wait(data_receiver_semaphore_addr_ptr, VALID);
                noc_semaphore_set(data_receiver_semaphore_addr_ptr, INVALID);

                // Push to destination CB after data arrived
                cb_push_back(args.dst_cb, args.dst_num_pages);
            } else if constexpr (IsMcastGridCore) {
                uint32_t data_receiver_semaphore_addr = get_semaphore(args.data_receiver_semaphore_id);

                volatile tt_l1_ptr uint32_t* data_receiver_semaphore_addr_ptr =
                    (volatile tt_l1_ptr uint32_t*)(data_receiver_semaphore_addr);
                noc_semaphore_wait(data_receiver_semaphore_addr_ptr, VALID);
                noc_semaphore_set(data_receiver_semaphore_addr_ptr, INVALID);
            }
#endif
        }

#if defined(COMPILE_FOR_BRISC)
        // Cached addresses computed during init, used during send and teardown
        uint64_t noc_coord_ = 0;
        uint64_t mcast_flag_noc_addr_ = 0;
#endif
    };  // class Op

};  // struct Mcast

}  // namespace deepseek_b1_ops
