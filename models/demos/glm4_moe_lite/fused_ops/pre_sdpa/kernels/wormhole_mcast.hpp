// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Wormhole-compatible mcast.hpp for GLM4 Pre-SDPA fused op.
// Replaces deepseek_v3_b1/unified_kernels/mcast.hpp which uses
// Blackhole-only NOC constants (NOC_PCIE_MASK, NOC_BRCST_EXCLUDE).
//
// Changes vs original:
// - mcast_send_set_state: removed NOC_RET_ADDR_MID write (NOC_PCIE_MASK)
//   and NOC_BRCST_EXCLUDE write. On Wormhole, coordinates are encoded
//   via NOC_RET_ADDR_COORDINATE with NOC_ADDR_COORD_SHIFT, and there is
//   no broadcast exclude register.

#pragma once

#include "../../../../deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// Mcast utility functions (Wormhole-compatible)
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
    // Wormhole: coordinates are in NOC_RET_ADDR_COORDINATE using NOC_ADDR_COORD_SHIFT
    // (Blackhole used NOC_RET_ADDR_MID with NOC_PCIE_MASK instead)
    NOC_CMD_BUF_WRITE_REG(
        noc, cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dst_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    // Wormhole: no NOC_BRCST_EXCLUDE register (Blackhole-only)
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
    riscv_wait(1000);  // Safety delay for posted mcast hw bug
}

#endif  // defined(COMPILE_FOR_BRISC)

// ============================================================================
// Mcast micro-op (identical to original — uses the fixed utility functions above)
// ============================================================================
struct Mcast {
    template <uint32_t McastNumCores, bool IsPartOfReceiverGrid, bool Loopback>
    struct SenderCTArgs {
        static constexpr uint32_t mcast_num_cores = McastNumCores;
        static constexpr bool is_part_of_receiver_grid = IsPartOfReceiverGrid;
        static constexpr bool loopback = Loopback;
    };

    struct ReceiverCTArgs {};
    struct ComputeCTArgs {};

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

    struct ReceiverArgs {
        uint32_t data_receiver_semaphore_id;
        uint32_t dst_cb;
        uint32_t dst_num_pages;
    };

    struct ComputeArgs {};

    using RTArgs = unified_kernels::SelectByRISCV<ReceiverArgs, SenderArgs, ComputeArgs>;

    template <typename CTArgsT, bool IsSenderCore, bool IsMcastGridCore, bool IsReceiverCore, bool pop_src>
    class Op {
    public:
        void init([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_BRISC)
            if constexpr (IsSenderCore) {
                uint32_t data_sender_semaphore_addr = get_semaphore(args.data_sender_semaphore_id);
                uint32_t data_receiver_semaphore_addr = get_semaphore(args.data_receiver_semaphore_id);

                noc_coord_ = get_noc_multicast_addr<noc_index>(
                    args.dest_noc_start_x, args.dest_noc_start_y, args.dest_noc_end_x, args.dest_noc_end_y, 0);
                mcast_flag_noc_addr_ = noc_coord_ | (uint64_t)(data_receiver_semaphore_addr);

                volatile tt_l1_ptr uint32_t* data_sender_semaphore_addr_ptr =
                    (volatile tt_l1_ptr uint32_t*)data_sender_semaphore_addr;

                init_persistent_mcast_sender<
                    CTArgsT::mcast_num_cores,
                    CTArgsT::loopback,
                    CTArgsT::is_part_of_receiver_grid>(
                    mcast_flag_noc_addr_, data_sender_semaphore_addr, data_sender_semaphore_addr_ptr);
            }
#endif
        }

        void operator()(const RTArgs& args) { impl(args); }

        void teardown() {
#if defined(COMPILE_FOR_BRISC)
            if constexpr (IsSenderCore) {
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
                cb_wait_front(args.src_cb, args.src_num_pages);
                uint64_t mcast_data_noc_addr = noc_coord_ | (uint64_t)args.mcast_receiver_data_addr;

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

                if constexpr (pop_src) {
                    cb_pop_front(args.src_cb, args.src_num_pages);
                }
            }
#elif defined(COMPILE_FOR_NCRISC)
            if constexpr (IsReceiverCore) {
                cb_reserve_back(args.dst_cb, args.dst_num_pages);

                uint32_t data_receiver_semaphore_addr = get_semaphore(args.data_receiver_semaphore_id);

                volatile tt_l1_ptr uint32_t* data_receiver_semaphore_addr_ptr =
                    (volatile tt_l1_ptr uint32_t*)(data_receiver_semaphore_addr);
                noc_semaphore_wait(data_receiver_semaphore_addr_ptr, VALID);
                noc_semaphore_set(data_receiver_semaphore_addr_ptr, INVALID);

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
        uint64_t noc_coord_ = 0;
        uint64_t mcast_flag_noc_addr_ = 0;
#endif
    };  // class Op

};  // struct Mcast

}  // namespace deepseek_b1_ops
