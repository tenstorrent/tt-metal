// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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

#if defined(COMPILE_FOR_BRISC) or defined(COMPILE_FOR_NCRISC)

constexpr bool mcast_is_shared_write_cmd_buf = write_cmd_buf == write_reg_cmd_buf;

template <uint32_t mcast_num_cores, bool loopback, bool is_part_of_receiver_grid, bool posted>
FORCE_INLINE void mcast_increment_counters() {
    constexpr uint32_t noc = noc_index;
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

    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        if constexpr (posted) {
            noc_posted_writes_num_issued[noc] += 1;
        } else {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += num_dests;
        }
    }
}

template <bool posted>
FORCE_INLINE void mcast_increment_counters_runtime(uint32_t num_dests, uint32_t count = 1, uint8_t noc = noc_index) {
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        if constexpr (posted) {
            inc_noc_counter_val<proc_type, NocBarrierType::POSTED_WRITES_NUM_ISSUED>(noc, count);
        } else {
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, count);
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, num_dests * count);
        }
    }

    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        if constexpr (posted) {
            noc_posted_writes_num_issued[noc] += count;
        } else {
            noc_nonposted_writes_num_issued[noc] += count;
            noc_nonposted_writes_acked[noc] += num_dests * count;
        }
    }
}

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
    bool posted,
    bool linked,
    bool set_noc_coord,
    bool set_addresses,
    bool set_size,
    bool increment_counters,
    uint8_t cmd_buf>
FORCE_INLINE void mcast_send_set_state_runtime(
    uint32_t src_local_addr,
    uint64_t dst_noc_addr,
    uint32_t len_bytes,
    uint32_t num_dests,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_MULTICAST_WRITE_VC) {
    constexpr bool multicast_path_reserve = true;

    while (!noc_cmd_buf_ready(noc, cmd_buf));
    if constexpr (increment_counters) {
        mcast_increment_counters_runtime<posted>(num_dests, 1, noc);
    }

    uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) |
                             (linked ? NOC_CMD_VC_LINKED : 0x0) | (multicast_path_reserve ? NOC_CMD_PATH_RESERVE : 0) |
                             NOC_CMD_BRCST_PACKET | (posted ? 0 : NOC_CMD_RESP_MARKED);

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    if constexpr (set_noc_coord) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dst_noc_addr >> 32) & NOC_PCIE_MASK);
        NOC_CMD_BUF_WRITE_REG(
            noc,
            cmd_buf,
            NOC_RET_ADDR_COORDINATE,
            (uint32_t)(dst_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    }
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_BRCST_EXCLUDE, 0);
    if constexpr (set_size) {
        ASSERT(len_bytes <= NOC_MAX_BURST_SIZE);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    }
    if constexpr (set_addresses) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_local_addr);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dst_noc_addr);
    }
}

template <uint8_t cmd_buf>
FORCE_INLINE void mcast_send_issue_txn(uint8_t noc = noc_index) {
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
}

template <
    uint32_t mcast_num_cores,
    bool loopback,
    bool is_part_of_receiver_grid,
    bool linked,
    bool posted,
    bool set_noc_coord,
    bool set_addresses,
    bool set_size,
    bool increment_counters,
    uint8_t cmd_buf>
FORCE_INLINE void mcast_send_set_state(uint32_t src_local_addr, uint64_t dst_noc_addr, uint32_t len_bytes = 0) {
    constexpr uint32_t noc = noc_index;
    constexpr uint32_t vc = NOC_MULTICAST_WRITE_VC;
    constexpr bool multicast_path_reserve = true;

    while (!noc_cmd_buf_ready(noc, cmd_buf));
    if constexpr (increment_counters) {
        mcast_increment_counters<mcast_num_cores, loopback, is_part_of_receiver_grid, posted>();
    }

    uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) |
                             (linked ? NOC_CMD_VC_LINKED : 0x0) | (multicast_path_reserve ? NOC_CMD_PATH_RESERVE : 0) |
                             (loopback ? NOC_CMD_BRCST_SRC_INCLUDE : 0) | NOC_CMD_BRCST_PACKET |
                             (posted ? 0 : NOC_CMD_RESP_MARKED);

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    if constexpr (set_noc_coord) {
        // Handles writing to PCIe
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dst_noc_addr >> 32) & NOC_PCIE_MASK);
        NOC_CMD_BUF_WRITE_REG(
            noc,
            cmd_buf,
            NOC_RET_ADDR_COORDINATE,
            (uint32_t)(dst_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    }
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_BRCST_EXCLUDE, 0);
    if constexpr (set_size) {
        ASSERT(len_bytes <= NOC_MAX_BURST_SIZE);
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
    bool wait_cmd_buf_ready,
    bool increment_counters,
    uint8_t cmd_buf>
FORCE_INLINE void mcast_send_with_state(uint32_t src_local_addr, uint32_t dst_local_addr, uint32_t len_bytes = 0) {
    constexpr uint32_t noc = noc_index;
    if constexpr (loopback) {
        static_assert(is_part_of_receiver_grid, "Loopback mode is only supported for receiver grid");
    }

    if constexpr (increment_counters) {
        mcast_increment_counters<mcast_num_cores, loopback, is_part_of_receiver_grid, posted>();
    }

    if constexpr (wait_cmd_buf_ready) {
        while (!noc_cmd_buf_ready(noc, cmd_buf));
    }

    if constexpr (set_size) {
        ASSERT(len_bytes <= NOC_MAX_BURST_SIZE);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    }
    if constexpr (set_addresses) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_local_addr);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, dst_local_addr);
    }
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
}

template <uint32_t mcast_num_cores, bool loopback, bool is_part_of_receiver_grid, bool linked, bool posted>
FORCE_INLINE void init_persistent_mcast_sender(uint64_t mcast_flag_noc_addr, uint32_t data_sender_semaphore_addr) {
    mcast_send_set_state<
        mcast_num_cores,
        loopback,
        is_part_of_receiver_grid,
        linked,
        posted,
        true,
        false,
        false,
        false,
        write_cmd_buf>(0, mcast_flag_noc_addr, 0);
    if constexpr (!mcast_is_shared_write_cmd_buf) {
        mcast_send_set_state<
            mcast_num_cores,
            loopback,
            is_part_of_receiver_grid,
            linked,
            posted,
            true,
            true,
            true,
            true,
            write_reg_cmd_buf>(data_sender_semaphore_addr, mcast_flag_noc_addr, 4);
    }
    mcast_send_with_state<
        mcast_num_cores,
        loopback,
        is_part_of_receiver_grid,
        linked,
        posted,
        mcast_is_shared_write_cmd_buf,
        mcast_is_shared_write_cmd_buf,
        false,
        mcast_is_shared_write_cmd_buf,
        write_reg_cmd_buf>(data_sender_semaphore_addr, mcast_flag_noc_addr, 4);
    noc_async_posted_writes_flushed();
}

template <uint32_t mcast_num_cores, bool loopback, bool is_part_of_receiver_grid>
FORCE_INLINE void teardown_persistent_mcast_sender(uint32_t data_sender_semaphore_addr) {
    mcast_send_set_state<
        mcast_num_cores,
        loopback,
        is_part_of_receiver_grid,
        false,
        false,
        false,
        false,
        false,
        true,
        write_reg_cmd_buf>(0, 0, 0);
    mcast_send_with_state<
        mcast_num_cores,
        loopback,
        is_part_of_receiver_grid,
        false,
        false,
        true,
        mcast_is_shared_write_cmd_buf,
        false,
        false,
        write_reg_cmd_buf>(data_sender_semaphore_addr, data_sender_semaphore_addr, 4);
    noc_async_write_barrier();
    riscv_wait(10000);  // This is just to guarantee safety due to posted mcast hw bug
}

#endif  // defined(COMPILE_FOR_BRISC) or defined(COMPILE_FOR_NCRISC)

// ============================================================================
// Mcast micro-op
//
// Multicasts data from a single sender core to multiple receiver cores.
// Default: Sender on BRISC, Receiver on NCRISC.
// ReceiverOnBrisc mode: Both sender and receiver on BRISC, NCRISC is no-op.
//   Use when NCRISC is needed for other work (e.g. down_proj_mcast).
//
// CB States:
//   Sender (BRISC):
//     - Waits: src_cb (src_num_pages)
//     - Pops: src_cb (src_num_pages) if pop_src=true
//   Receiver (NCRISC, or BRISC when ReceiverOnBrisc=true):
//     - Reserves: dst_cb (dst_num_pages)
//     - Pushes: dst_cb (dst_num_pages)
//   TRISC: No-op
//
// Semaphore States:
//   Sender: Assumes sender_semaphore contains VALID (set during init)
//   Receiver: Waits for receiver_semaphore == VALID, then resets to INVALID
//
// Note: Sender assumes that receiver's dst_cb is ready to receive at the beginning of execution.
// ============================================================================
struct Mcast {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    // Sender CTArgs: mcast_num_cores, is_part_of_receiver_grid
    // If sender is part of receiver grid, it needs loopback to receive its own mcast
    template <uint32_t McastNumCores, bool IsPartOfReceiverGrid, bool Loopback>
    struct SenderCTArgs {
        static constexpr uint32_t mcast_num_cores = McastNumCores;
        static constexpr bool is_part_of_receiver_grid = IsPartOfReceiverGrid;
        static constexpr bool loopback = Loopback;
    };

    struct ReceiverCTArgs {};
    struct ComputeCTArgs {};

    // ========================================================================
    // Runtime args structs
    // ========================================================================

    struct SenderArgs {
        uint32_t dest_noc_start_x;
        uint32_t dest_noc_start_y;
        uint32_t dest_noc_end_x;
        uint32_t dest_noc_end_y;
        uint32_t data_sender_semaphore_addr;
        uint32_t data_receiver_semaphore_addr;
        uint32_t data_size_bytes;
        uint32_t src_cb;
        uint32_t src_num_pages;
        uint32_t input_data_addr;
        uint32_t mcast_receiver_data_addr;
    };

    struct ReceiverArgs {
        uint32_t data_receiver_semaphore_addr;
        uint32_t dst_cb;
        uint32_t dst_num_pages;
    };

    // Unified dataflow args: both BRISC and NCRISC get the full set so the
    // sender/receiver impl can be placed on either RISC without restructuring.
    struct DMArgs {
        SenderArgs sender;
        ReceiverArgs receiver;
    };

    struct ComputeArgs {};

    using RTArgs = unified_kernels::SelectByRISCV<DMArgs, DMArgs, ComputeArgs>;

    // ========================================================================
    // Op - the actual operation
    //
    // CTArgsT: compile-time args (mcast_num_cores, loopback, is_part_of_receiver_grid)
    // IsSenderCore: compile-time flag to distinguish sender vs receiver cores
    // IsMcastGridCore: compile-time flag for cores in the mcast destination grid
    // IsReceiverCore: compile-time flag for receiver cores
    // pop_src: whether to pop the source CB after sending
    // ReceiverOnBrisc: when true, receiver logic runs on BRISC instead of NCRISC.
    //   BRISC does send-then-receive; NCRISC is no-op.
    //
    // Usage:
    //   Op op;
    //   op.init(args);      // Initialize persistent mcast sender (call once)
    //   op(args);           // Send data (can be called multiple times)
    //   op.teardown(args);  // Teardown persistent mcast sender (call once)
    // ========================================================================
    template <
        typename CTArgsT,
        bool IsSenderCore,
        bool IsMcastGridCore,
        bool IsReceiverCore,
        bool pop_src,
        bool ReceiverOnBrisc = false>
    class Op {
    public:
        // ====================================================================
        // init - Initialize persistent mcast sender (BRISC only)
        // ====================================================================
        template <bool init_noc = true>
        void init([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_BRISC)
            if constexpr (IsSenderCore) {
                uint64_t mcast_flag_noc_addr = get_noc_multicast_addr<noc_index>(
                    args.sender.dest_noc_start_x,
                    args.sender.dest_noc_start_y,
                    args.sender.dest_noc_end_x,
                    args.sender.dest_noc_end_y,
                    (uint64_t)(args.sender.data_receiver_semaphore_addr));
                volatile tt_l1_ptr uint32_t* data_sender_semaphore_addr_ptr =
                    (volatile tt_l1_ptr uint32_t*)args.sender.data_sender_semaphore_addr;
                if constexpr (init_noc) {
                    noc_semaphore_set(data_sender_semaphore_addr_ptr, INVALID);
                    init_persistent_mcast_sender<
                        CTArgsT::mcast_num_cores,
                        CTArgsT::loopback,
                        CTArgsT::is_part_of_receiver_grid,
                        linked,
                        posted>(mcast_flag_noc_addr, args.sender.data_sender_semaphore_addr);
                    noc_async_posted_writes_flushed();
                }
                noc_semaphore_set(data_sender_semaphore_addr_ptr, VALID);
            }
#endif
        }

        // ====================================================================
        // operator() - Send data (BRISC sender) and/or receive data (receiver)
        // ====================================================================
        void operator()(const RTArgs& args) { impl(args); }

        // ====================================================================
        // teardown - Teardown persistent mcast sender (BRISC only)
        // ====================================================================
        void teardown([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_BRISC)
            if constexpr (IsSenderCore) {
                teardown_persistent_mcast_sender<
                    CTArgsT::mcast_num_cores,
                    CTArgsT::loopback,
                    CTArgsT::is_part_of_receiver_grid>(args.sender.data_sender_semaphore_addr);
            }
#endif
        }

    private:
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
        static void sender_impl(const SenderArgs& s) {
            // Due to a HW bug, only mcast txns can be sent on the same NOC while linked, so it's safe to pre-increment
            mcast_send_set_state<
                CTArgsT::mcast_num_cores,
                CTArgsT::loopback,
                CTArgsT::is_part_of_receiver_grid,
                linked,
                posted,
                false,
                true,
                true,
                true,
                write_cmd_buf>(s.input_data_addr, s.mcast_receiver_data_addr, s.data_size_bytes);
            if constexpr (!mcast_is_shared_write_cmd_buf) {
                mcast_send_set_state<
                    CTArgsT::mcast_num_cores,
                    CTArgsT::loopback,
                    CTArgsT::is_part_of_receiver_grid,
                    linked,
                    posted,
                    false,
                    true,
                    false,
                    true,
                    write_reg_cmd_buf>(s.data_sender_semaphore_addr, s.data_receiver_semaphore_addr, 4);
            } else {
                mcast_increment_counters<
                    CTArgsT::mcast_num_cores,
                    CTArgsT::loopback,
                    CTArgsT::is_part_of_receiver_grid,
                    posted>();
            }

            cb_wait_front(s.src_cb, s.src_num_pages);

            mcast_send_with_state<
                CTArgsT::mcast_num_cores,
                CTArgsT::loopback,
                CTArgsT::is_part_of_receiver_grid,
                linked,
                posted,
                false,
                false,
                false,
                false,
                write_cmd_buf>(0, 0, 0);
            mcast_send_with_state<
                CTArgsT::mcast_num_cores,
                CTArgsT::loopback,
                CTArgsT::is_part_of_receiver_grid,
                linked,
                posted,
                mcast_is_shared_write_cmd_buf,
                mcast_is_shared_write_cmd_buf,
                mcast_is_shared_write_cmd_buf,
                false,
                write_reg_cmd_buf>(s.data_sender_semaphore_addr, s.data_receiver_semaphore_addr, 4);

            noc_async_posted_writes_flushed();

            if constexpr (pop_src) {
                cb_pop_front(s.src_cb, s.src_num_pages);
            }
        }

        static void receiver_impl(const ReceiverArgs& r) {
            volatile tt_l1_ptr uint32_t* data_receiver_semaphore_addr_ptr =
                (volatile tt_l1_ptr uint32_t*)(r.data_receiver_semaphore_addr);
            cb_reserve_back(r.dst_cb, r.dst_num_pages);
            noc_semaphore_wait(data_receiver_semaphore_addr_ptr, VALID);
            noc_semaphore_set(data_receiver_semaphore_addr_ptr, INVALID);
            cb_push_back(r.dst_cb, r.dst_num_pages);
        }

        static void mcast_grid_impl(const ReceiverArgs& r) {
            volatile tt_l1_ptr uint32_t* data_receiver_semaphore_addr_ptr =
                (volatile tt_l1_ptr uint32_t*)(r.data_receiver_semaphore_addr);
            noc_semaphore_wait(data_receiver_semaphore_addr_ptr, VALID);
            noc_semaphore_set(data_receiver_semaphore_addr_ptr, INVALID);
        }
#endif

        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_BRISC)
            // BRISC: Sender always, Receiver when ReceiverOnBrisc=true
            if constexpr (IsSenderCore) {
                sender_impl(args.sender);
            }
            if constexpr (ReceiverOnBrisc && IsReceiverCore) {
                receiver_impl(args.receiver);
            } else if constexpr (ReceiverOnBrisc && IsMcastGridCore) {
                mcast_grid_impl(args.receiver);
            }
#elif defined(COMPILE_FOR_NCRISC)
            // NCRISC: Receiver when ReceiverOnBrisc=false, no-op otherwise
            if constexpr (!ReceiverOnBrisc && IsReceiverCore) {
                receiver_impl(args.receiver);
            } else if constexpr (!ReceiverOnBrisc && IsMcastGridCore) {
                mcast_grid_impl(args.receiver);
            }
#endif
        }

        static constexpr bool linked = true;
        static constexpr bool posted = true;

    };  // class Op

};  // struct Mcast

}  // namespace deepseek_b1_ops
