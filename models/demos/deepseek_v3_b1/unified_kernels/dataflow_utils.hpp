// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#endif

namespace unified_kernels {

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)

template <bool posted>
FORCE_INLINE void unicast_write_increment_counters(uint32_t count = 1, uint8_t noc = noc_index) {
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        if constexpr (posted) {
            inc_noc_counter_val<proc_type, NocBarrierType::POSTED_WRITES_NUM_ISSUED>(noc, count);
        } else {
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, count);
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, count);
        }
    }

    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        if constexpr (posted) {
            noc_posted_writes_num_issued[noc] += count;
        } else {
            noc_nonposted_writes_num_issued[noc] += count;
            noc_nonposted_writes_acked[noc] += count;
        }
    }
}

template <bool posted>
FORCE_INLINE void unicast_atomic_inc_increment_counters(uint32_t count = 1, uint8_t noc = noc_index) {
#ifdef ARCH_BLACKHOLE
    static_assert(!posted, "Blackhole does not support posted atomics");
#endif
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        if constexpr (!posted) {
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_ATOMICS_ACKED>(noc, count);
        }
    }

    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        if constexpr (!posted) {
            noc_nonposted_atomics_acked[noc] += count;
        }
    }
}

FORCE_INLINE void noc_async_read_increment_counters(uint32_t count = 1, uint8_t noc = noc_index) {
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        inc_noc_counter_val<proc_type, NocBarrierType::READS_NUM_ISSUED>(noc, count);
    }

    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        noc_reads_num_issued[noc] += count;
    }
}

FORCE_INLINE void noc_async_read_preprogram_state(uint64_t src_noc_addr, uint8_t noc = noc_index) {
    noc_async_read_set_state(src_noc_addr, noc);
}

template <bool increment_counters = true, uint8_t cmd_buf = read_cmd_buf>
FORCE_INLINE void noc_async_read_preprogram_all_state(
    uint64_t src_noc_addr, uint32_t dst_local_addr, uint32_t len_bytes, uint8_t noc = noc_index, uint8_t vc = 1) {
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    if constexpr (increment_counters) {
        noc_async_read_increment_counters(1, noc);
    }

    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        uint32_t noc_rd_cmd_field =
            NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_rd_cmd_field);
    }
    NOC_CMD_BUF_WRITE_REG(
        noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(src_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)src_noc_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, dst_local_addr);
}

template <bool increment_counters = true>
FORCE_INLINE void noc_async_read_issue_txn(
    uint32_t src_local_addr, uint32_t dst_local_addr, uint32_t len_bytes, uint8_t noc = noc_index) {
    noc_async_read_with_state<increment_counters>(src_local_addr, dst_local_addr, len_bytes, noc);
}

template <bool increment_counters = true>
FORCE_INLINE void noc_async_read_issue_txn(uint32_t src_local_addr, uint32_t dst_local_addr, uint8_t noc = noc_index) {
    noc_async_read_one_packet_with_state<increment_counters>(src_local_addr, dst_local_addr, 0, noc);
}

template <uint8_t cmd_buf = read_cmd_buf>
FORCE_INLINE void noc_async_read_issue_txn(uint8_t noc = noc_index) {
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
}

template <bool posted, bool set_noc_coord, bool set_addresses, bool set_size, bool increment_counters, uint8_t cmd_buf>
FORCE_INLINE void unicast_write_set_state(
    uint32_t src_local_addr,
    uint64_t dst_noc_addr,
    uint32_t len_bytes,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_UNICAST_WRITE_VC) {
    while (!noc_cmd_buf_ready(noc, cmd_buf));

    if constexpr (increment_counters) {
        unicast_write_increment_counters<posted>(1, noc);
    }

    uint32_t noc_cmd_field =
        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) | (posted ? 0 : NOC_CMD_RESP_MARKED);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    if constexpr (set_noc_coord) {
        NOC_CMD_BUF_WRITE_REG(
            noc,
            cmd_buf,
            NOC_RET_ADDR_COORDINATE,
            (uint32_t)(dst_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    }
    if constexpr (set_size) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    }
    if constexpr (set_addresses) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_local_addr);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dst_noc_addr);
    }
}

template <
    bool posted,
    bool set_addresses,
    bool set_size,
    bool wait_cmd_buf_ready,
    bool increment_counters,
    uint8_t cmd_buf>
FORCE_INLINE void unicast_write_with_state(
    uint32_t src_local_addr, uint32_t dst_local_addr, uint32_t len_bytes, uint8_t noc = noc_index) {
    if constexpr (increment_counters) {
        unicast_write_increment_counters<posted>(1, noc);
    }

    if constexpr (wait_cmd_buf_ready) {
        while (!noc_cmd_buf_ready(noc, cmd_buf));
    }
    if constexpr (set_size) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    }
    if constexpr (set_addresses) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_local_addr);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, dst_local_addr);
    }
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
}

template <bool posted, uint8_t cmd_buf = write_cmd_buf>
FORCE_INLINE void noc_async_write_preprogram_all_state(
    uint32_t src_local_addr,
    uint64_t dst_noc_addr,
    uint32_t len_bytes,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_UNICAST_WRITE_VC) {
    unicast_write_set_state<posted, true, true, true, true, cmd_buf>(src_local_addr, dst_noc_addr, len_bytes, noc, vc);
}

template <uint8_t cmd_buf = write_cmd_buf>
FORCE_INLINE void noc_async_write_issue_txn(uint8_t noc = noc_index) {
    unicast_write_with_state<false, false, false, false, false, cmd_buf>(0, 0, 0, noc);
}

template <bool posted, bool set_addr, bool set_incr, bool increment_counters, uint8_t cmd_buf>
FORCE_INLINE void unicast_atomic_inc_set_state(
    uint64_t addr,
    uint32_t incr,
    uint32_t wrap = 31,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_UNICAST_WRITE_VC,
    uint32_t atomic_ret_val = MEM_NOC_ATOMIC_RET_VAL_ADDR) {
#ifdef ARCH_BLACKHOLE
    static_assert(!posted, "Blackhole does not support posted atomics");
#endif
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    if constexpr (increment_counters) {
        unicast_atomic_inc_increment_counters<posted>(1, noc);
    }
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(noc, 0, NOC_NODE_ID);
        uint32_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
        uint32_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
        uint64_t atomic_ret_addr = NOC_XY_ADDR(my_x, my_y, atomic_ret_val);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)(atomic_ret_addr & 0xFFFFFFFF));
    }
    if constexpr (set_addr) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)(addr & 0xFFFFFFFF));
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(addr >> NOC_ADDR_COORD_SHIFT));
        NOC_CMD_BUF_WRITE_REG(
            noc,
            cmd_buf,
            NOC_AT_LEN_BE,
            NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr >> 2) & 0x3) |
                NOC_AT_IND_32_SRC(0));
    }
    if constexpr (set_incr) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_DATA, incr);
    }
    NOC_CMD_BUF_WRITE_REG(
        noc,
        cmd_buf,
        NOC_CTRL,
        NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) | (posted ? 0 : NOC_CMD_RESP_MARKED) | NOC_CMD_AT);
}

template <bool posted, bool set_addr, bool set_incr, bool wait_cmd_buf_ready, bool increment_counters, uint8_t cmd_buf>
FORCE_INLINE void unicast_atomic_inc_with_state(
    uint64_t addr, uint32_t incr, uint32_t wrap = 31, uint8_t noc = noc_index) {
#ifdef ARCH_BLACKHOLE
    static_assert(!posted, "Blackhole does not support posted atomics");
#endif
    if constexpr (increment_counters) {
        unicast_atomic_inc_increment_counters<posted>(1, noc);
    }

    if constexpr (wait_cmd_buf_ready) {
        while (!noc_cmd_buf_ready(noc, cmd_buf));
    }
    if constexpr (set_addr) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)(addr & 0xFFFFFFFF));
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(addr >> NOC_ADDR_COORD_SHIFT));
        NOC_CMD_BUF_WRITE_REG(
            noc,
            cmd_buf,
            NOC_AT_LEN_BE,
            NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr >> 2) & 0x3) |
                NOC_AT_IND_32_SRC(0));
    }
    if constexpr (set_incr) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_DATA, incr);
    }
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
}

template <bool posted, uint8_t cmd_buf = write_at_cmd_buf>
FORCE_INLINE void noc_async_atomic_inc_preprogram_all_state(
    uint64_t addr,
    uint32_t incr,
    uint32_t wrap = 31,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_UNICAST_WRITE_VC,
    uint32_t atomic_ret_val = MEM_NOC_ATOMIC_RET_VAL_ADDR) {
    unicast_atomic_inc_set_state<posted, true, true, true, cmd_buf>(addr, incr, wrap, noc, vc, atomic_ret_val);
}

template <uint8_t cmd_buf = write_at_cmd_buf>
FORCE_INLINE void noc_async_atomic_inc_issue_txn(uint8_t noc = noc_index) {
    unicast_atomic_inc_with_state<false, false, false, false, false, cmd_buf>(0, 0, 31, noc);
}

#endif  // defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)

}  // namespace unified_kernels
