// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <limits>
#include "noc_parameters.h"
#include "hostdev/dev_msgs.h"
#include "noc_overlay_parameters.h"
#include "internal/risc_attribs.h"
#include "api/debug/assert.h"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
constexpr std::underlying_type_t<TensixProcessorTypes> proc_type =
    static_cast<std::underlying_type_t<TensixProcessorTypes>>(TensixProcessorTypes::DM0);
#else
constexpr std::underlying_type_t<TensixProcessorTypes> proc_type =
    static_cast<std::underlying_type_t<TensixProcessorTypes>>(TensixProcessorTypes::DM1);
#endif

// Helper functions to convert NoC coordinates to NoC-0 coordinates, used in metal as "physical" coordinates.
#define NOC_0_X(noc_index, noc_size_x, x) x
#define NOC_0_Y(noc_index, noc_size_y, y) y
#define NOC_0_X_PHYS_COORD(noc_index, noc_size_x, x) (noc_index == 0 ? (x) : (noc_size_x - 1 - (x)))
#define NOC_0_Y_PHYS_COORD(noc_index, noc_size_y, y) (noc_index == 0 ? (y) : (noc_size_y - 1 - (y)))
#define MY_NOC_ENCODING(noc_index) NOC_CMD_BUF_READ_REG(noc_index, 0, NOC_CFG(NOC_ID_LOGICAL));
////

// Use VC 1 for unicast writes, and VC 4 for mcast writes

// used for ops with USE_MULTI_MOC defined
constexpr uint32_t DYNAMIC_NOC_NCRISC_WR_CMD_BUF = 2;  // all writes share cmd buf
constexpr uint32_t DYNAMIC_NOC_NCRISC_WR_REG_CMD_BUF = 2;
constexpr uint32_t DYNAMIC_NOC_NCRISC_AT_CMD_BUF = 3;
constexpr uint32_t DYNAMIC_NOC_NCRISC_RD_CMD_BUF = 3;

constexpr uint32_t DYNAMIC_NOC_BRISC_WR_CMD_BUF = 0;  // all writes share cmd buf
constexpr uint32_t DYNAMIC_NOC_BRISC_WR_REG_CMD_BUF = 0;
constexpr uint32_t DYNAMIC_NOC_BRISC_AT_CMD_BUF = 1;
constexpr uint32_t DYNAMIC_NOC_BRISC_RD_CMD_BUF = 1;

constexpr uint32_t NCRISC_WR_CMD_BUF = 0;      // for large writes
constexpr uint32_t NCRISC_RD_CMD_BUF = 1;      // for all reads
constexpr uint32_t NCRISC_WR_REG_CMD_BUF = 2;  // for small writes (e.g., registers, semaphores)
constexpr uint32_t NCRISC_AT_CMD_BUF = 3;      // for atomics

constexpr uint32_t BRISC_WR_CMD_BUF = 0;      // for large writes
constexpr uint32_t BRISC_RD_CMD_BUF = 1;      // for all reads
constexpr uint32_t BRISC_WR_REG_CMD_BUF = 2;  // for small writes (e.g., registers, semaphores)
constexpr uint32_t BRISC_AT_CMD_BUF = 3;      // for atomics

// 36 bits of address followed by coordinate. First 32 bits of address go into lo register, remaining address bits and
// coordinates are in the mid register
constexpr uint32_t NOC_ADDR_COORD_SHIFT =
    32;  // address is lower 36 bits and upper bits are the coordinates, 32 bits in lo reg and rest goes to mid
const uint32_t NOC_TARG_ADDR_COORDINATE = NOC_TARG_ADDR_MID;
const uint32_t NOC_RET_ADDR_COORDINATE = NOC_RET_ADDR_MID;
const uint32_t NOC_COORDINATE_MASK = 0xFFFFFFFF;

extern uint32_t noc_reads_num_issued[NUM_NOCS];
extern uint32_t noc_nonposted_writes_num_issued[NUM_NOCS];
extern uint32_t noc_nonposted_writes_acked[NUM_NOCS];
extern uint32_t noc_nonposted_atomics_acked[NUM_NOCS];
extern uint32_t noc_posted_writes_num_issued[NUM_NOCS];

enum class NocBarrierType : uint8_t {
    READS_NUM_ISSUED,
    NONPOSTED_WRITES_NUM_ISSUED,
    NONPOSTED_WRITES_ACKED,
    NONPOSTED_ATOMICS_ACKED,
    POSTED_WRITES_NUM_ISSUED,
    COUNT
};

static constexpr uint8_t NUM_BARRIER_TYPES = static_cast<uint32_t>(NocBarrierType::COUNT);

struct BarrierCounter {
    uint32_t barrier[NUM_BARRIER_TYPES];
};

struct RiscBarrierCounter {
    BarrierCounter risc[MaxDMProcessorsPerCoreType];
};

struct NocBarrierCounter {
    RiscBarrierCounter noc[NUM_NOCS];
};

// Must update the allocated size for the counters in dev_mem_map.h if this changes
static_assert(sizeof(NocBarrierCounter) == 80, "NocBarrierCounter size is not 80 bytes");

template <uint8_t proc_t, NocBarrierType barrier_type>
inline __attribute__((always_inline)) uint32_t get_noc_counter_address(uint32_t noc) {
    static_assert(proc_t < MaxDMProcessorsPerCoreType);
    static_assert(static_cast<std::underlying_type_t<NocBarrierType>>(barrier_type) < NUM_BARRIER_TYPES);

    constexpr uint32_t base = MEM_NOC_COUNTER_BASE;
    constexpr uint32_t size = MEM_NOC_COUNTER_SIZE;

    // Calculate most of the offset at compile time. Only the noc is variable at runtime.
    constexpr uint32_t compile_time_offset =
        offsetof(NocBarrierCounter, noc) + proc_t * sizeof(decltype(std::declval<NocBarrierCounter>().noc[0].risc[0])) +
        static_cast<std::underlying_type_t<NocBarrierType>>(barrier_type) *
            sizeof(decltype(std::declval<NocBarrierCounter>().noc[0].risc[0].barrier[0]));

    constexpr uint32_t noc_stride = sizeof(decltype(std::declval<NocBarrierCounter>().noc[0]));

    return base + noc * noc_stride + compile_time_offset;
}

// noc_nonposted_writes_acked
template <uint8_t proc_t, NocBarrierType barrier_type>
inline __attribute__((always_inline)) uint32_t get_noc_counter_val(uint32_t noc) {
    uint32_t counter_addr = get_noc_counter_address<proc_t, barrier_type>(noc);
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counter_addr);
}

template <uint8_t proc_t, NocBarrierType barrier_type>
inline __attribute__((always_inline)) void inc_noc_counter_val(uint32_t noc, uint32_t inc = 1) {
    uint32_t counter_addr = get_noc_counter_address<proc_t, barrier_type>(noc);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counter_addr) += inc;
}

template <uint8_t proc_t, NocBarrierType barrier_type>
inline __attribute__((always_inline)) void set_noc_counter_val(uint32_t noc, uint32_t val) {
    uint32_t counter_addr = get_noc_counter_address<proc_t, barrier_type>(noc);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counter_addr) = val;
}

inline __attribute__((always_inline)) void NOC_CMD_BUF_WRITE_REG(
    uint32_t noc, uint32_t buf, uint32_t addr, uint32_t val) {
    uint32_t offset = (buf << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT) + addr;
    volatile uint32_t* ptr = (volatile uint32_t*)offset;
    *ptr = val;
}

inline __attribute__((always_inline)) uint32_t NOC_CMD_BUF_READ_REG(uint32_t noc, uint32_t buf, uint32_t addr) {
    uint32_t offset = (buf << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT) + addr;
    volatile uint32_t* ptr = (volatile uint32_t*)offset;
    return *ptr;
}

inline __attribute__((always_inline)) uint32_t NOC_STATUS_READ_REG(uint32_t noc, uint32_t reg_id) {
    uint32_t offset = (noc << NOC_INSTANCE_OFFSET_BIT) + NOC_STATUS(reg_id);
    volatile uint32_t* ptr = (volatile uint32_t*)offset;
    return *ptr;
}

inline __attribute__((always_inline)) uint32_t NOC_CFG_READ_REG(uint32_t noc, uint32_t reg_id) {
    uint32_t offset = (noc << NOC_INSTANCE_OFFSET_BIT) + NOC_CFG(reg_id);
    volatile uint32_t* ptr = (volatile uint32_t*)offset;
    return *ptr;
}

inline __attribute__((always_inline)) bool noc_cmd_buf_ready(uint32_t noc, uint32_t cmd_buf) {
    return (NOC_CMD_BUF_READ_REG(noc, cmd_buf, NOC_CMD_CTRL) == NOC_CTRL_STATUS_READY);
}

inline __attribute__((always_inline)) void noc_clear_outstanding_req_cnt(uint32_t noc, uint32_t id_mask) {
    uint32_t offset = (noc << NOC_INSTANCE_OFFSET_BIT) + NOC_CLEAR_OUTSTANDING_REQ_CNT;
    volatile uint32_t* ptr = (volatile uint32_t*)offset;
    *ptr = id_mask;
}

template <uint8_t noc_mode = DM_DEDICATED_NOC>
inline __attribute__((always_inline)) void ncrisc_noc_fast_read(
    uint32_t noc,
    uint32_t cmd_buf,
    uint64_t src_addr,
    uint32_t dest_addr,
    uint32_t len_bytes,
    uint32_t read_req_vc = 1) {
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        inc_noc_counter_val<proc_type, NocBarrierType::READS_NUM_ISSUED>(noc, 1);
    }
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        uint32_t noc_rd_cmd_field =
            NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(read_req_vc);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_rd_cmd_field);
    }
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)src_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(src_addr >> NOC_ADDR_COORD_SHIFT));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        noc_reads_num_issued[noc] += 1;
    }
}

inline __attribute__((always_inline)) bool ncrisc_dynamic_noc_reads_flushed(uint32_t noc) {
    uint32_t status_reg_val = NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED);
    uint32_t self_risc_acked = get_noc_counter_val<proc_type, NocBarrierType::READS_NUM_ISSUED>(noc);
    uint32_t other_risc_acked = get_noc_counter_val<1 - proc_type, NocBarrierType::READS_NUM_ISSUED>(noc);
    return (status_reg_val == (self_risc_acked + other_risc_acked));
}

inline __attribute__((always_inline)) bool ncrisc_noc_reads_flushed(uint32_t noc) {
    return (NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED) == noc_reads_num_issued[noc]);
}

inline __attribute__((always_inline)) bool ncrisc_noc_read_with_transaction_id_flushed(
    uint32_t noc, uint32_t transcation_id) {
    return (NOC_STATUS_READ_REG(noc, NIU_MST_REQS_OUTSTANDING_ID(transcation_id)) == 0);
}

inline __attribute__((always_inline)) uint32_t noc_available_transactions(uint32_t noc, uint32_t trid) {
    return NOC_MAX_TRANSACTION_ID_COUNT - NOC_STATUS_READ_REG(noc, NIU_MST_REQS_OUTSTANDING_ID(trid));
}

template <uint8_t noc_mode = DM_DEDICATED_NOC, bool use_trid = false, bool update_counter = true>
inline __attribute__((always_inline)) void ncrisc_noc_fast_write(
    uint32_t noc,
    uint32_t cmd_buf,
    uint32_t src_addr,
    uint64_t dest_addr,
    uint32_t len_bytes,
    uint32_t vc,
    bool mcast,
    bool linked,
    uint32_t num_dests,
    bool multicast_path_reserve,
    bool posted = false,
    uint32_t trid = 0) {
    if constexpr (update_counter && noc_mode == DM_DYNAMIC_NOC) {
        if (posted) {
            inc_noc_counter_val<proc_type, NocBarrierType::POSTED_WRITES_NUM_ISSUED>(noc, 1);
        } else {
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, 1);
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, num_dests);
        }
    }
    uint32_t noc_cmd_field =
        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) | (linked ? NOC_CMD_VC_LINKED : 0x0) |
        (mcast ? ((multicast_path_reserve ? NOC_CMD_PATH_RESERVE : 0) | NOC_CMD_BRCST_PACKET) : 0x0) |
        (posted ? 0 : NOC_CMD_RESP_MARKED);

    if constexpr (use_trid) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(trid));
    }

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

    if constexpr (update_counter && noc_mode == DM_DEDICATED_NOC) {
        if (posted) {
            noc_posted_writes_num_issued[noc] += 1;
        } else {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += num_dests;
        }
    }
}

template <uint8_t noc_mode = DM_DEDICATED_NOC>
inline __attribute__((always_inline)) void ncrisc_noc_fast_write_loopback_src(
    uint32_t noc,
    uint32_t cmd_buf,
    uint32_t src_addr,
    uint64_t dest_addr,
    uint32_t len_bytes,
    uint32_t vc,
    bool mcast,
    bool linked,
    uint32_t num_dests,
    bool multicast_path_reserve) {
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, 1);
        inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, num_dests);
    }
    uint32_t noc_cmd_field =
        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) | (linked ? NOC_CMD_VC_LINKED : 0x0) |
        (mcast ? ((multicast_path_reserve ? NOC_CMD_PATH_RESERVE : 0) | NOC_CMD_BRCST_PACKET) : 0x0) |
        NOC_CMD_BRCST_SRC_INCLUDE | NOC_CMD_RESP_MARKED;

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        noc_nonposted_writes_num_issued[noc] += 1;
        noc_nonposted_writes_acked[noc] += num_dests;
    }
}

template <uint8_t noc_mode = DM_DEDICATED_NOC>
inline __attribute__((always_inline)) void ncrisc_noc_blitz_write_setup(
    uint32_t noc, uint32_t cmd_buf, uint64_t dest_addr, uint32_t len_bytes, uint32_t vc, uint32_t num_times_to_write) {
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, num_times_to_write);
        inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, num_times_to_write);
    }
    uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) | NOC_CMD_RESP_MARKED;

    while (!noc_cmd_buf_ready(noc, cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT));
    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        noc_nonposted_writes_num_issued[noc] += num_times_to_write;
        noc_nonposted_writes_acked[noc] += num_times_to_write;
    }
}

inline __attribute__((always_inline)) bool ncrisc_dynamic_noc_nonposted_writes_sent(uint32_t noc) {
    uint32_t status_reg_val = NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_WR_REQ_SENT);
    uint32_t self_risc_acked = get_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc);
    uint32_t other_risc_acked = get_noc_counter_val<1 - proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc);
    return (status_reg_val == (self_risc_acked + other_risc_acked));
}

inline __attribute__((always_inline)) bool ncrisc_noc_nonposted_writes_sent(uint32_t noc) {
    return (NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_WR_REQ_SENT) == noc_nonposted_writes_num_issued[noc]);
}

inline __attribute__((always_inline)) bool ncrisc_dynamic_noc_posted_writes_sent(uint32_t noc) {
    uint32_t status_reg_val = NOC_STATUS_READ_REG(noc, NIU_MST_POSTED_WR_REQ_SENT);
    uint32_t self_risc_acked = get_noc_counter_val<proc_type, NocBarrierType::POSTED_WRITES_NUM_ISSUED>(noc);
    uint32_t other_risc_acked = get_noc_counter_val<1 - proc_type, NocBarrierType::POSTED_WRITES_NUM_ISSUED>(noc);
    return (status_reg_val == (self_risc_acked + other_risc_acked));
}

inline __attribute__((always_inline)) bool ncrisc_noc_posted_writes_sent(uint32_t noc) {
    return (NOC_STATUS_READ_REG(noc, NIU_MST_POSTED_WR_REQ_SENT) == noc_posted_writes_num_issued[noc]);
}

inline __attribute__((always_inline)) bool ncrisc_dynamic_noc_nonposted_writes_flushed(uint32_t noc) {
    uint32_t status_reg_val = NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED);
    uint32_t self_risc_acked = get_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc);
    uint32_t other_risc_acked = get_noc_counter_val<1 - proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc);
    return (status_reg_val == (self_risc_acked + other_risc_acked));
}

inline __attribute__((always_inline)) bool ncrisc_noc_nonposted_writes_flushed(uint32_t noc) {
    return (NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED) == noc_nonposted_writes_acked[noc]);
}

inline __attribute__((always_inline)) bool ncrisc_noc_nonposted_write_with_transaction_id_sent(
    uint32_t noc, uint32_t transcation_id) {
    return (NOC_STATUS_READ_REG(noc, NIU_MST_WRITE_REQS_OUTGOING_ID(transcation_id)) == 0);
}

inline __attribute__((always_inline)) bool ncrisc_noc_nonposted_write_with_transaction_id_flushed(
    uint32_t noc, uint32_t transcation_id) {
    return (NOC_STATUS_READ_REG(noc, NIU_MST_REQS_OUTSTANDING_ID(transcation_id)) == 0);
}

inline __attribute__((always_inline)) bool ncrisc_dynamic_noc_nonposted_atomics_flushed(uint32_t noc) {
    uint32_t status_reg_val = NOC_STATUS_READ_REG(noc, NIU_MST_ATOMIC_RESP_RECEIVED);
    uint32_t self_risc_acked = get_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_ATOMICS_ACKED>(noc);
    uint32_t other_risc_acked = get_noc_counter_val<1 - proc_type, NocBarrierType::NONPOSTED_ATOMICS_ACKED>(noc);
    return (status_reg_val == (self_risc_acked + other_risc_acked));
}

inline __attribute__((always_inline)) bool ncrisc_noc_nonposted_atomics_flushed(uint32_t noc) {
    return (NOC_STATUS_READ_REG(noc, NIU_MST_ATOMIC_RESP_RECEIVED) == noc_nonposted_atomics_acked[noc]);
}

template <uint8_t MAX_NOCS_TO_INIT = NUM_NOCS>
inline __attribute__((always_inline)) void noc_init(uint32_t atomic_ret_val) {
#pragma GCC unroll 0
    for (uint8_t noc = 0; noc < MAX_NOCS_TO_INIT; noc++) {
        uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(noc, 0, NOC_NODE_ID);
        uint32_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
        uint32_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
        uint64_t xy_local_addr = NOC_XY_ADDR(my_x, my_y, 0);

        NOC_CMD_BUF_WRITE_REG(
            noc, NCRISC_WR_CMD_BUF, NOC_TARG_ADDR_COORDINATE, (uint32_t)(xy_local_addr >> NOC_ADDR_COORD_SHIFT));
        NOC_CMD_BUF_WRITE_REG(
            noc, NCRISC_WR_REG_CMD_BUF, NOC_TARG_ADDR_COORDINATE, (uint32_t)(xy_local_addr >> NOC_ADDR_COORD_SHIFT));

        uint64_t atomic_ret_addr = NOC_XY_ADDR(my_x, my_y, atomic_ret_val);
        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_AT_CMD_BUF, NOC_RET_ADDR_LO, (uint32_t)(atomic_ret_addr & 0xFFFFFFFF));
        NOC_CMD_BUF_WRITE_REG(
            noc, NCRISC_AT_CMD_BUF, NOC_RET_ADDR_COORDINATE, (uint32_t)(atomic_ret_addr >> NOC_ADDR_COORD_SHIFT));

        uint32_t noc_rd_cmd_field =
            NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(1);
        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_CTRL, noc_rd_cmd_field);
        NOC_CMD_BUF_WRITE_REG(
            noc, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_COORDINATE, (uint32_t)(xy_local_addr >> NOC_ADDR_COORD_SHIFT));
    }
}

inline __attribute__((always_inline)) void dynamic_noc_init() {
#pragma GCC unroll 0
    for (int noc = 0; noc < NUM_NOCS; noc++) {
        uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(noc, 0, NOC_NODE_ID);
        uint32_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
        uint32_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
        uint64_t xy_local_addr = NOC_XY_ADDR(my_x, my_y, 0);

        // program brisc cmd_buf 0
        NOC_CMD_BUF_WRITE_REG(
            noc,
            DYNAMIC_NOC_BRISC_RD_CMD_BUF,
            NOC_RET_ADDR_COORDINATE,
            (uint32_t)(xy_local_addr >> NOC_ADDR_COORD_SHIFT));

        // program brisc cmd_buf 1
        NOC_CMD_BUF_WRITE_REG(
            noc,
            DYNAMIC_NOC_BRISC_WR_CMD_BUF,
            NOC_TARG_ADDR_COORDINATE,
            (uint32_t)(xy_local_addr >> NOC_ADDR_COORD_SHIFT));

        // program ncrisc cmd_buf 2
        NOC_CMD_BUF_WRITE_REG(
            noc,
            DYNAMIC_NOC_NCRISC_RD_CMD_BUF,
            NOC_RET_ADDR_COORDINATE,
            (uint32_t)(xy_local_addr >> NOC_ADDR_COORD_SHIFT));

        // program ncrisc cmd_buf 3
        NOC_CMD_BUF_WRITE_REG(
            noc,
            DYNAMIC_NOC_NCRISC_WR_CMD_BUF,
            NOC_TARG_ADDR_COORDINATE,
            (uint32_t)(xy_local_addr >> NOC_ADDR_COORD_SHIFT));
    }
}

// set noc local memory state for a single kernel from the global state
inline __attribute__((always_inline)) void noc_local_state_init(int noc) {
    // Hide latency of NOC reg reads by reading first, writing second
    uint32_t reads_num_issued = NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED);
    uint32_t nonposted_writes_num_issued = NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_WR_REQ_SENT);
    uint32_t nonposted_writes_acked = NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED);
    uint32_t nonposted_atomics_acked = NOC_STATUS_READ_REG(noc, NIU_MST_ATOMIC_RESP_RECEIVED);
    uint32_t posted_writes_num_issued = NOC_STATUS_READ_REG(noc, NIU_MST_POSTED_WR_REQ_SENT);

    noc_reads_num_issued[noc] = reads_num_issued;
    noc_nonposted_writes_num_issued[noc] = nonposted_writes_num_issued;
    noc_nonposted_writes_acked[noc] = nonposted_writes_acked;
    noc_nonposted_atomics_acked[noc] = nonposted_atomics_acked;
    noc_posted_writes_num_issued[noc] = posted_writes_num_issued;
}

template <NocBarrierType barrier_type, uint32_t status_register>
inline __attribute__((always_inline)) void dynamic_noc_local_barrier_init(
    uint32_t noc0_status_reg, uint32_t noc1_status_reg) {
    using underlying_tensix_processor_types_t = std::underlying_type_t<TensixProcessorTypes>;
    constexpr underlying_tensix_processor_types_t dm0 =
        static_cast<underlying_tensix_processor_types_t>(TensixProcessorTypes::DM0);
    constexpr underlying_tensix_processor_types_t dm1 =
        static_cast<underlying_tensix_processor_types_t>(TensixProcessorTypes::DM1);

    set_noc_counter_val<dm0, barrier_type>(NOC_0, noc0_status_reg);
    set_noc_counter_val<dm0, barrier_type>(NOC_1, 0);
    set_noc_counter_val<dm1, barrier_type>(NOC_0, 0);
    set_noc_counter_val<dm1, barrier_type>(NOC_1, noc1_status_reg);
}

inline __attribute__((always_inline)) void dynamic_noc_local_state_init() {
    // Pipeline all register reads first to hide latency
    uint32_t noc0_reads_num_issued = NOC_STATUS_READ_REG(NOC_0, NIU_MST_RD_RESP_RECEIVED);
    uint32_t noc1_reads_num_issued = NOC_STATUS_READ_REG(NOC_1, NIU_MST_RD_RESP_RECEIVED);
    uint32_t noc0_nonposted_writes_num_issued = NOC_STATUS_READ_REG(NOC_0, NIU_MST_NONPOSTED_WR_REQ_SENT);
    uint32_t noc1_nonposted_writes_num_issued = NOC_STATUS_READ_REG(NOC_1, NIU_MST_NONPOSTED_WR_REQ_SENT);
    uint32_t noc0_nonposted_writes_acked = NOC_STATUS_READ_REG(NOC_0, NIU_MST_WR_ACK_RECEIVED);
    uint32_t noc1_nonposted_writes_acked = NOC_STATUS_READ_REG(NOC_1, NIU_MST_WR_ACK_RECEIVED);
    uint32_t noc0_nonposted_atomics_acked = NOC_STATUS_READ_REG(NOC_0, NIU_MST_ATOMIC_RESP_RECEIVED);
    uint32_t noc1_nonposted_atomics_acked = NOC_STATUS_READ_REG(NOC_1, NIU_MST_ATOMIC_RESP_RECEIVED);
    uint32_t noc0_posted_writes_num_issued = NOC_STATUS_READ_REG(NOC_0, NIU_MST_POSTED_WR_REQ_SENT);
    uint32_t noc1_posted_writes_num_issued = NOC_STATUS_READ_REG(NOC_1, NIU_MST_POSTED_WR_REQ_SENT);
    dynamic_noc_local_barrier_init<NocBarrierType::READS_NUM_ISSUED, NIU_MST_RD_RESP_RECEIVED>(
        noc0_reads_num_issued, noc1_reads_num_issued);
    dynamic_noc_local_barrier_init<NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED, NIU_MST_NONPOSTED_WR_REQ_SENT>(
        noc0_nonposted_writes_num_issued, noc1_nonposted_writes_num_issued);
    dynamic_noc_local_barrier_init<NocBarrierType::NONPOSTED_WRITES_ACKED, NIU_MST_WR_ACK_RECEIVED>(
        noc0_nonposted_writes_acked, noc1_nonposted_writes_acked);
    dynamic_noc_local_barrier_init<NocBarrierType::NONPOSTED_ATOMICS_ACKED, NIU_MST_ATOMIC_RESP_RECEIVED>(
        noc0_nonposted_atomics_acked, noc1_nonposted_atomics_acked);
    dynamic_noc_local_barrier_init<NocBarrierType::POSTED_WRITES_NUM_ISSUED, NIU_MST_POSTED_WR_REQ_SENT>(
        noc0_posted_writes_num_issued, noc1_posted_writes_num_issued);
}

inline __attribute__((always_inline)) void ncrisc_noc_counters_init() {
    for (int noc = 0; noc < NUM_NOCS; noc++) {
        // Hide latency of NOC reg reads by reading first, writing second
        uint32_t reads_num_issued = NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED);
        uint32_t nonposted_writes_num_issued = NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_WR_REQ_SENT);
        uint32_t nonposted_writes_acked = NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED);
        uint32_t nonposted_atomics_acked = NOC_STATUS_READ_REG(noc, NIU_MST_ATOMIC_RESP_RECEIVED);
        uint32_t posted_writes_num_issued = NOC_STATUS_READ_REG(noc, NIU_MST_POSTED_WR_REQ_SENT);

        noc_reads_num_issued[noc] = reads_num_issued;
        noc_nonposted_writes_num_issued[noc] = nonposted_writes_num_issued;
        noc_nonposted_writes_acked[noc] = nonposted_writes_acked;
        noc_nonposted_atomics_acked[noc] = nonposted_atomics_acked;
        noc_posted_writes_num_issued[noc] = posted_writes_num_issued;
    }
}

inline __attribute__((always_inline)) void ncrisc_noc_full_sync() {
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        while (!ncrisc_noc_reads_flushed(n));
        while (!ncrisc_noc_nonposted_writes_sent(n));
        while (!ncrisc_noc_nonposted_writes_flushed(n));
        while (!ncrisc_noc_nonposted_atomics_flushed(n));
        while (!ncrisc_noc_posted_writes_sent(n));
    }
}

template <uint8_t noc_mode = DM_DEDICATED_NOC>
inline __attribute__((always_inline)) void ncrisc_noc_fast_read_any_len(
    uint32_t noc,
    uint32_t cmd_buf,
    uint64_t src_addr,
    uint32_t dest_addr,
    uint32_t len_bytes,
    uint32_t read_req_vc = 1) {
    while (len_bytes > NOC_MAX_BURST_SIZE) {
        while (!noc_cmd_buf_ready(noc, cmd_buf));
        ncrisc_noc_fast_read<noc_mode>(noc, cmd_buf, src_addr, dest_addr, NOC_MAX_BURST_SIZE, read_req_vc);
        src_addr += NOC_MAX_BURST_SIZE;
        dest_addr += NOC_MAX_BURST_SIZE;
        len_bytes -= NOC_MAX_BURST_SIZE;
    }
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    ncrisc_noc_fast_read<noc_mode>(noc, cmd_buf, src_addr, dest_addr, len_bytes, read_req_vc);
}

template <uint8_t noc_mode = DM_DEDICATED_NOC, bool use_trid = false, bool one_packet = false>
inline __attribute__((always_inline)) void ncrisc_noc_fast_write_any_len(
    uint32_t noc,
    uint32_t cmd_buf,
    uint32_t src_addr,
    uint64_t dest_addr,
    uint32_t len_bytes,
    uint32_t vc,
    bool mcast,
    bool linked,
    uint32_t num_dests,
    bool multicast_path_reserve,
    bool posted = false,
    uint32_t trid = 0) {
    if constexpr (!one_packet) {
        while (len_bytes > NOC_MAX_BURST_SIZE) {
            while (!noc_cmd_buf_ready(noc, cmd_buf));
            ncrisc_noc_fast_write<noc_mode, use_trid>(
                noc,
                cmd_buf,
                src_addr,
                dest_addr,
                NOC_MAX_BURST_SIZE,
                vc,
                mcast,
                linked,
                num_dests,
                multicast_path_reserve,
                posted,
                trid);
            src_addr += NOC_MAX_BURST_SIZE;
            dest_addr += NOC_MAX_BURST_SIZE;
            len_bytes -= NOC_MAX_BURST_SIZE;
        }
    }
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    ncrisc_noc_fast_write<noc_mode, use_trid>(
        noc,
        cmd_buf,
        src_addr,
        dest_addr,
        len_bytes,
        vc,
        mcast,
        linked,
        num_dests,
        multicast_path_reserve,
        posted,
        trid);
}

template <uint8_t noc_mode = DM_DEDICATED_NOC>
inline __attribute__((always_inline)) void ncrisc_noc_fast_write_any_len_loopback_src(
    uint32_t noc,
    uint32_t cmd_buf,
    uint32_t src_addr,
    uint64_t dest_addr,
    uint32_t len_bytes,
    uint32_t vc,
    bool mcast,
    bool linked,
    uint32_t num_dests,
    bool multicast_path_reserve) {
    while (len_bytes > NOC_MAX_BURST_SIZE) {
        while (!noc_cmd_buf_ready(noc, cmd_buf));
        ncrisc_noc_fast_write_loopback_src<noc_mode>(
            noc,
            cmd_buf,
            src_addr,
            dest_addr,
            NOC_MAX_BURST_SIZE,
            vc,
            mcast,
            linked,
            num_dests,
            multicast_path_reserve);
        src_addr += NOC_MAX_BURST_SIZE;
        dest_addr += NOC_MAX_BURST_SIZE;
        len_bytes -= NOC_MAX_BURST_SIZE;
    }
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    ncrisc_noc_fast_write_loopback_src<noc_mode>(
        noc, cmd_buf, src_addr, dest_addr, len_bytes, vc, mcast, linked, num_dests, multicast_path_reserve);
}

// `dst_type` is "Don't care" on Wormhole, it's required on Blackhole to workaround bug that manifests with noc
// transactions using all 4 memory ports. Issuing inline writes and atomics requires all 4 memory ports to accept the
// transaction at the same time. If one port on the receipient has no back-pressure then the transaction will hang
// because there is no mechanism to allow one memory port to move ahead of another. To workaround this hang, we emulate
// inline writes on Blackhole by writing the value to be written to local L1 first and then issue a noc async write.
template <uint8_t noc_mode = DM_DEDICATED_NOC, InlineWriteDst dst_type = InlineWriteDst::DEFAULT, bool flush = true>
inline __attribute__((always_inline)) void noc_fast_write_dw_inline(
    uint32_t noc,
    uint32_t cmd_buf,
    uint32_t val,
    uint64_t dest_addr,
    uint32_t be,
    uint32_t static_vc,
    bool mcast,
    bool posted = false,
    uint32_t customized_src_addr = 0) {
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        if (posted) {
            inc_noc_counter_val<proc_type, NocBarrierType::POSTED_WRITES_NUM_ISSUED>(noc, 1);
        } else {
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, 1);
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, 1);
        }
    }
    bool static_vc_alloc = true;
    uint32_t noc_cmd_field = (static_vc_alloc ? NOC_CMD_VC_STATIC : 0x0) | NOC_CMD_STATIC_VC(static_vc) | NOC_CMD_CPY |
                             NOC_CMD_WR | NOC_CMD_WR_INLINE |
                             (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0) |
                             (posted ? 0x0 : NOC_CMD_RESP_MARKED);

    uint32_t be32 = be;
    uint32_t be_shift = (dest_addr & (NOC_WORD_BYTES - 1));
    // If we're given a misaligned address, don't write to the bytes in the word below the address
    be32 = (be32 << be_shift);

    while (!noc_cmd_buf_ready(noc, cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_DATA, val);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, dest_addr & 0xFFFFFFFF);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, be32);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        if (posted) {
            noc_posted_writes_num_issued[noc] += 1;
        } else {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += 1;
        }
    }
}

template <uint8_t noc_mode = DM_DEDICATED_NOC, bool program_ret_addr = false>
inline __attribute__((always_inline)) void noc_fast_atomic_increment(
    uint32_t noc,
    uint32_t cmd_buf,
    uint64_t addr,
    uint32_t vc,
    uint32_t incr,
    uint32_t wrap,
    bool linked,
    bool posted = false,
    uint32_t atomic_ret_val = 0) {
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        if (!posted) {
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_ATOMICS_ACKED>(noc, 1);
        }
    }
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    if constexpr (noc_mode == DM_DYNAMIC_NOC || program_ret_addr == true) {
        uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(noc, 0, NOC_NODE_ID);
        uint32_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
        uint32_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
        uint64_t atomic_ret_addr = NOC_XY_ADDR(my_x, my_y, atomic_ret_val);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)(atomic_ret_addr & 0xFFFFFFFF));
    }
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)(addr & 0xFFFFFFFF));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(addr >> NOC_ADDR_COORD_SHIFT));
    NOC_CMD_BUF_WRITE_REG(
        noc,
        cmd_buf,
        NOC_CTRL,
        NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) | (linked ? NOC_CMD_VC_LINKED : 0x0) |
            (posted ? 0 : NOC_CMD_RESP_MARKED) | NOC_CMD_AT);
    NOC_CMD_BUF_WRITE_REG(
        noc,
        cmd_buf,
        NOC_AT_LEN_BE,
        NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr >> 2) & 0x3) | NOC_AT_IND_32_SRC(0));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_DATA, incr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, 0x1);
    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        if (!posted) {
            noc_nonposted_atomics_acked[noc] += 1;
        }
    }
}

// issue noc reads while wait for outstanding transactions done
template <uint8_t noc_mode = DM_DEDICATED_NOC, bool skip_ptr_update = false, bool skip_cmdbuf_chk = false>
inline __attribute__((always_inline)) void ncrisc_noc_fast_read_with_transaction_id(
    uint32_t noc, uint32_t cmd_buf, uint32_t src_base_addr, uint32_t src_addr, uint32_t dest_addr, uint32_t trid) {
    if constexpr (noc_mode == DM_DYNAMIC_NOC && !skip_ptr_update) {
        inc_noc_counter_val<proc_type, NocBarrierType::READS_NUM_ISSUED>(noc, 1);
    }
    uint32_t src_addr_;
    src_addr_ = src_base_addr + src_addr;

    if constexpr (!skip_cmdbuf_chk) {
        while (!noc_cmd_buf_ready(noc, cmd_buf));
    } else {
        ASSERT(noc_cmd_buf_ready(noc, cmd_buf));
    }
    while (NOC_STATUS_READ_REG(noc, NIU_MST_REQS_OUTSTANDING_ID(trid)) > ((NOC_MAX_TRANSACTION_ID_COUNT + 1) / 2));

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_addr_);  // (uint32_t)src_addr
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    if constexpr (noc_mode == DM_DEDICATED_NOC && !skip_ptr_update) {
        noc_reads_num_issued[noc] += 1;
    }
}

// clang-format off
/**
 * Sets the transaction id for a noc transaction.
 *
 * Return value: None
 *
 * | Argument | Description                                        | Data type | Valid range | Required |
 * |----------|----------------------------------------------------|-----------|-------------|----------|
 * | noc      | Which NOC to use for the transaction               | uint32_t  | 0 or 1      | True     |
 * | cmd_buf  | Which command buffer to use for the transaction    | uint32_t  | 0 - 3       | True     |
 * | trid     | Transaction id for the transaction                 | uint32_t  | 0x0 - 0xF   | True     |
 */
// clang-format on
inline __attribute__((always_inline)) void ncrisc_noc_set_transaction_id(
    uint32_t noc, uint32_t cmd_buf, uint32_t trid) {
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(trid));
}

// clang-format off
/**
 * Sets the stateful registers for an asynchronous read from a specified source node located at NOC
 * coordinates (x,y) at a local address (encoded as a uint64_t using \a
 * get_noc_addr function). This function is used to set up the state for
 * \a ncrisc_noc_read_with_state, which will issue the actual read request.
 *
 * The source node can be either a DRAM bank, a Tensix core or a PCIe controller.
 *
 * Return value: None
 *
 * | Argument                        | Description                                        | Data type | Valid range                                              | required |
 * |---------------------------------|----------------------------------------------------|-----------|----------------------------------------------------------|----------|
 * | noc                             | Which NOC to use for the transaction               | uint32_t  | 0 or 1                                                   | True     |
 * | cmd_buf                         | Which command buffer to use for the transaction    | uint32_t  | 0 - 3                                                    | True     |
 * | src_noc_addr                    | Encoding of the source NOC location (x,y)+address  | uint64_t  | Results of \a get_noc_addr calls                         | True     |
 * | len_bytes                       | Size of the transaction in bytes.                  | uint32_t  | 0..1 MB                                                  | False    |
 * | vc                              | Which VC to use for the transaction                | uint32_t  | 0 - 3                                                    | False    |
 * | noc_mode (template parameter)   | NOC mode for the transaction                       | uint8_t   | DM_DEDICATED_NOC, DM_DYNAMIC_NOC or DM_INVALID_NOC (0-2) | False    |
 * | one_packet (template parameter) | Whether transaction size is <= NOC_MAX_BURST_SIZE  | bool      | true or false                                            | False    |
 * | use_vc (template parameter)     | Use custom VC, enables vc parameter                | bool      | true or false                                            | False    |
 */
// clang-format on
template <uint8_t noc_mode = DM_DEDICATED_NOC, bool one_packet = false, bool use_vc = false>
inline __attribute__((always_inline)) void ncrisc_noc_read_set_state(
    uint32_t noc, uint32_t cmd_buf, uint64_t src_noc_addr, uint32_t len_bytes = 0, const uint32_t vc = 0) {
    while (!noc_cmd_buf_ready(noc, cmd_buf));

    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        uint32_t noc_rd_cmd_field =
            NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(1);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_rd_cmd_field);
    }
    if constexpr (use_vc) {
        uint32_t noc_rd_cmd_field =
            NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_rd_cmd_field);
    }
    NOC_CMD_BUF_WRITE_REG(
        noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(src_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);

    // If one packet, set data size
    if constexpr (one_packet) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    }
}

// clang-format off
/**
 * Initiates an asynchronous read from a specified source node located at NOC
 * coordinates (x,y) at a local address (encoded as a uint64_t using \a
 * get_noc_addr function) for a single packet with size <= NOC_MAX_BURST_SIZE (i.e. maximum packet size).
 * This function must be preceded by a call to \a ncrisc_noc_read_set_state.
 * This function is used to issue the actual read request after the state has been set up.
 *
 * Return value: None
 *
 * | Argument                            | Description                                        | Data type | Valid range                                              | required |
 * |-------------------------------------|----------------------------------------------------|-----------|----------------------------------------------------------|----------|
 * | noc                                 | Which NOC to use for the transaction               | uint32_t  | 0 or 1                                                   | True     |
 * | cmd_buf                             | Which command buffer to use for the transaction    | uint32_t  | 0 - 3                                                    | True     |
 * | src_local_addr                      | Address in local L1 memory on source core          | uint32_t  | 0..1 MB                                                  | True     |
 * | dst_local_addr                      | Address in local L1 memory on destination core     | uint32_t  | 0..1 MB                                                  | True     |
 * | len_bytes                           | Size of transaction in bytes                       | uint32_t  | 0..1 MB                                                  | False    |
 * | noc_mode (template parameter)       | NOC mode for the transaction                       | uint8_t   | DM_DEDICATED_NOC, DM_DYNAMIC_NOC or DM_INVALID_NOC (0-2) | False    |
 * | inc_num_issued (template parameter) | Increment enable for transaction issued counters   | bool      | true or false                                            | False    |
 * | one_packet (template parameter)     | Whether transaction size is <= NOC_MAX_BURST_SIZE  | bool      | true or false                                            | False    |
 */
// clang-format on
template <uint8_t noc_mode = DM_DEDICATED_NOC, bool inc_num_issued = true, bool one_packet = false>
inline __attribute__((always_inline)) void ncrisc_noc_read_with_state(
    uint32_t noc, uint32_t cmd_buf, uint32_t src_local_addr, uint32_t dst_local_addr, uint32_t len_bytes = 0) {
    if constexpr (inc_num_issued && noc_mode == DM_DYNAMIC_NOC) {
        inc_noc_counter_val<proc_type, NocBarrierType::READS_NUM_ISSUED>(noc, 1);
    }

    while (!noc_cmd_buf_ready(noc, cmd_buf));

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, dst_local_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_local_addr);
    if constexpr (!one_packet) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    }
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

    if constexpr (inc_num_issued && noc_mode == DM_DEDICATED_NOC) {
        noc_reads_num_issued[noc] += 1;
    }
}

// clang-format off
/**
 * Initiates an asynchronous read for all transaction sizes.
 * Refer to \a ncrisc_noc_read_with_state for more details.
 *
 * Return value: None
 *
 * | Argument                            | Description                                        | Data type | Valid range                                              | required |
 * |-------------------------------------|----------------------------------------------------|-----------|----------------------------------------------------------|----------|
 * | noc                                 | Which NOC to use for the transaction               | uint32_t  | 0 or 1                                                   | True     |
 * | cmd_buf                             | Which command buffer to use for the transaction    | uint32_t  | 0 - 3                                                    | True     |
 * | src_local_addr                      | Address in local L1 memory on source core          | uint32_t  | 0..1 MB                                                  | True     |
 * | dst_local_addr                      | Address in local L1 memory on destination core     | uint32_t  | 0..1 MB                                                  | True     |
 * | len_bytes                           | Size of transaction in bytes                       | uint32_t  | 0..1 MB                                                  | True     |
 * | noc_mode (template parameter)       | NOC mode for the transaction                       | uint8_t   | DM_DEDICATED_NOC, DM_DYNAMIC_NOC or DM_INVALID_NOC (0-2) | False    |
 * | inc_num_issued (template parameter) | Increment enable for transaction issued counters   | bool      | true or false                                            | False    |
 */
// clang-format on
template <uint8_t noc_mode = DM_DEDICATED_NOC, bool inc_num_issued = true>
inline __attribute__((always_inline)) void ncrisc_noc_read_any_len_with_state(
    uint32_t noc, uint32_t cmd_buf, uint32_t src_local_addr, uint32_t dst_local_addr, uint32_t len_bytes) {
    if (len_bytes > NOC_MAX_BURST_SIZE) {
        // Set data size for while loop
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, NOC_MAX_BURST_SIZE);

        while (len_bytes > NOC_MAX_BURST_SIZE) {
            ncrisc_noc_read_with_state<noc_mode, inc_num_issued, true /* one_packet */>(
                noc, cmd_buf, src_local_addr, dst_local_addr);

            len_bytes -= NOC_MAX_BURST_SIZE;
            src_local_addr += NOC_MAX_BURST_SIZE;
            dst_local_addr += NOC_MAX_BURST_SIZE;
        }
    }

    // left-over packet
    ncrisc_noc_read_with_state<noc_mode, inc_num_issued>(noc, cmd_buf, src_local_addr, dst_local_addr, len_bytes);
}

// clang-format off
/**
 * Sets the stateful registers for an asynchronous write to a specified destination node located at
 * NOC coordinates (x,y) at a local address (encoded as a uint64_t using \a
 * get_noc_addr function). This function is used to set up the state for
 * \a ncrisc_noc_write_with_state, which will issue the actual
 * write request.
 *
 * The destination node can be either a DRAM bank, a Tensix core or a PCIe controller.
 *
 * Return value: None
 *
 * | Argument                        | Description                                              | Data type | Valid range                      | required |
 * |---------------------------------|----------------------------------------------------------|-----------|----------------------------------|----------|
 * | noc                             | NOC to use for the transaction                           | uint32_t  | 0 or 1                           | True     |
 * | cmd_buf                         | Command buffer to use for the transaction                | uint32_t  | 0 - 3                            | True     |
 * | dst_noc_addr                    | Encoding of the destination NOC location (x,y)+address   | uint64_t  | Results of \a get_noc_addr calls | True     |
 * | len_bytes                       | Size of the transaction in bytes.                        | uint32_t  | 0..1 MB                          | False    |
 * | vc                              | Which VC to use for the transaction                      | uint32_t  | 0 - 3                            | False    |
 * | posted (template parameter)     | Whether the transaction is posted (i.e. no ack required) | bool      | true or false                    | False    |
 * | one_packet (template parameter) | Whether transaction size is <= NOC_MAX_BURST_SIZE        | bool      | true or false                    | False    |
 */
// clang-format on
template <bool posted = false, bool one_packet = false>
inline __attribute__((always_inline)) void ncrisc_noc_write_set_state(
    uint32_t noc, uint32_t cmd_buf, uint64_t dst_noc_addr, uint32_t len_bytes = 0, const uint32_t vc = 0) {
    while (!noc_cmd_buf_ready(noc, cmd_buf));

    uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) |
                             0x0 |  // (linked ? NOC_CMD_VC_LINKED : 0x0)
                             0x0 |  // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
                             (!posted ? NOC_CMD_RESP_MARKED : 0x0);

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(
        noc, cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dst_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);

    // If one packet, set data size
    if constexpr (one_packet) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    }
}

// clang-format off
/**
 * Initiates an asynchronous write to a specified destination node located at
 * NOC coordinates (x,y) at a local address (encoded as a uint64_t using \a
 * get_noc_addr function). This function must be preceded by a call to
 * \a ncrisc_noc_write_set_state. This function is used to issue the actual
 * write request after the state has been set up.
 *
 * Return value: None
 *
 * | Argument                            | Description                                              | Data type | Valid range                                              | required |
 * |-------------------------------------|----------------------------------------------------------|-----------|----------------------------------------------------------|----------|
 * | noc                                 | NOC to use for the transaction                           | uint32_t  | 0 or 1                                                   | True     |
 * | cmd_buf                             | Command buffer to use for the transaction                | uint32_t  | 0 - 3                                                    | True     |
 * | src_local_addr                      | Address in local L1 memory on source core                | uint32_t  | 0..1 MB                                                  | True     |
 * | dst_local_addr                      | Address in local L1 memory on destination core           | uint32_t  | 0..1 MB                                                  | True     |
 * | len_bytes                           | Size of transaction in bytes                             | uint32_t  | 0..1 MB                                                  | False    |
 * | noc_mode (template parameter)       | NOC mode for the transaction                             | uint8_t   | DM_DEDICATED_NOC, DM_DYNAMIC_NOC or DM_INVALID_NOC (0-2) | False    |
 * | posted (template parameter)         | Whether the transaction is posted (i.e. no ack required) | bool      | true or false                                            | False    |
 * | update_counter (template parameter) | Whether to increment write counters                      | bool      | true or false                                            | False    |
 * | one_packet (template parameter)     | Whether transaction size is <= NOC_MAX_BURST_SIZE        | bool      | true or false                                            | False    |
 */
// clang-format on
template <uint8_t noc_mode = DM_DEDICATED_NOC, bool posted = false, bool update_counter = true, bool one_packet = false>
inline __attribute__((always_inline)) void ncrisc_noc_write_with_state(
    uint32_t noc, uint32_t cmd_buf, uint32_t src_local_addr, uint32_t dst_local_addr, uint32_t len_bytes = 0) {
    if constexpr (update_counter && noc_mode == DM_DYNAMIC_NOC) {
        if constexpr (posted) {
            inc_noc_counter_val<proc_type, NocBarrierType::POSTED_WRITES_NUM_ISSUED>(noc, 1);
        } else {
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, 1);
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, 1);
        }
    }

    while (!noc_cmd_buf_ready(noc, cmd_buf));

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_local_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, dst_local_addr);
    if constexpr (!one_packet) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    }
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

    if constexpr (update_counter && noc_mode == DM_DEDICATED_NOC) {
        if constexpr (posted) {
            noc_posted_writes_num_issued[noc] += 1;
        } else {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += 1;
        }
    }
}

// clang-format off
/**
 * Initiates an asynchronous write for all transaction sizes.
 * Refer to \a ncrisc_noc_write_with_state for more details.
 *
 * Return value: None
 *
 * | Argument                            | Description                                              | Data type | Valid range                                              | required |
 * |-------------------------------------|----------------------------------------------------------|-----------|----------------------------------------------------------|----------|
 * | noc                                 | NOC to use for the transaction                           | uint32_t  | 0 or 1                                                   | True     |
 * | cmd_buf                             | Command buffer to use for the transaction                | uint32_t  | 0 - 3                                                    | True     |
 * | src_local_addr                      | Address in local L1 memory on source core                | uint32_t  | 0..1 MB                                                  | True     |
 * | dst_local_addr                      | Address in local L1 memory on destination core           | uint32_t  | 0..1 MB                                                  | True     |
 * | len_bytes                           | Size of transaction in bytes                             | uint32_t  | 0..1 MB                                                  | True     |
 * | noc_mode (template parameter)       | NOC mode for the transaction                             | uint8_t   | DM_DEDICATED_NOC, DM_DYNAMIC_NOC or DM_INVALID_NOC (0-2) | False    |
 * | posted (template parameter)         | Whether the transaction is posted (i.e. no ack required) | bool      | true or false                                            | False    |
 * | update_counter (template parameter) | Whether to increment write counters                      | bool      | true or false                                            | False    |
 */
// clang-format on
template <uint8_t noc_mode = DM_DEDICATED_NOC, bool posted = false, bool update_counter = true>
inline __attribute__((always_inline)) void ncrisc_noc_write_any_len_with_state(
    uint32_t noc, uint32_t cmd_buf, uint32_t src_local_addr, uint32_t dst_local_addr, uint32_t len_bytes) {
    if (len_bytes > NOC_MAX_BURST_SIZE) {
        // Set data size for while loop
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, NOC_MAX_BURST_SIZE);

        while (len_bytes > NOC_MAX_BURST_SIZE) {
            ncrisc_noc_write_with_state<noc_mode, posted, update_counter, true /* one_packet */>(
                noc, cmd_buf, src_local_addr, dst_local_addr);

            len_bytes -= NOC_MAX_BURST_SIZE;
            src_local_addr += NOC_MAX_BURST_SIZE;
            dst_local_addr += NOC_MAX_BURST_SIZE;
        }
    }

    // left-over packet
    ncrisc_noc_write_with_state<noc_mode, posted, update_counter>(
        noc, cmd_buf, src_local_addr, dst_local_addr, len_bytes);
}

// clang-format off
/**
 * Sets the stateful registers for an inline write of a 32-bit value to a NOC destination.
 * This function is used to set up the state for \a noc_fast_write_dw_inline_with_state, which will issue the actual
 * write request. The 32-bit value and part of the destination address can be set later in \a noc_fast_write_dw_inline_with_state.
 *
 * The destination node can be either a Tensix core+L1 memory
 * address or a PCIe controller; This API does not support DRAM addresses.
 *
 * Note: On Blackhole, this API can only write to stream registers, writing to L1 will cause hangs!
 *
 * Return value: None
 *
 * | Argument                     | Description                                            | Type     | Valid Range                      | Required |
 * |------------------------------|--------------------------------------------------------|----------|----------------------------------|----------|
 * | noc                          | NOC to use for the transaction                         | uint32_t | 0 or 1                           | True     |
 * | cmd_buf                      | Command buffer to use for the transaction              | uint32_t | 0 - 3                            | True     |
 * | dest_addr                    | Encoding of the destination NOC location (x,y)+address | uint64_t | Results of \a get_noc_addr calls | True     |
 * | be                           | Byte-enable                                            | uint32_t | 0x1-0xF                          | True     |
 * | static_vc                    | VC to use for the transaction                          | uint32_t | 0 - 3 (Unicast VCs)              | True     |
 * | val                          | The value to be written                                | uint32_t | Any uint32_t value               | False    |
 * | posted (template parameter)  | Whether the call is posted (i.e. ack requirement)      | bool     | true or false                    | False    |
 * | set_val (template parameter) | Whether to set the value for the write here            | bool     | true or false                    | False    |
 */
// clang-format on
template <bool posted = false, bool set_val = false>
inline __attribute__((always_inline)) void noc_fast_write_dw_inline_set_state(
    uint32_t noc, uint32_t cmd_buf, uint64_t dest_addr, uint32_t be, uint32_t static_vc, uint32_t val = 0) {
    while (!noc_cmd_buf_ready(noc, cmd_buf));

    if constexpr (set_val) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_DATA, val);
    }

    uint32_t noc_cmd_field = NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(static_vc) | NOC_CMD_CPY | NOC_CMD_WR |
                             NOC_CMD_WR_INLINE | 0x0 | (posted ? 0x0 : NOC_CMD_RESP_MARKED);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)dest_addr);
    NOC_CMD_BUF_WRITE_REG(
        noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);

    // If we're given a misaligned address, don't write to the bytes in the word below the address
    uint32_t be32 = be << (dest_addr & (NOC_WORD_BYTES - 1));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, be32);
}

// clang-format off
/**
 * Initiates an inline write of a 32-bit value to a NOC destination.
 * This function must be preceded by a call to \a noc_fast_write_dw_inline_set_state.
 * This function is used to issue the actual write request after the state has been set up.
 * The 32-bit value and part of the destination address can also be set in this API
 * (Either hi or lo address should be getting updated).
 *
 * The destination node can be either a Tensix core+L1 memory
 * address or a PCIe controller; This API does not support DRAM addresses.
 *
 * Note: On Blackhole, this API can only write to stream registers, writing to L1 will cause hangs!
 *
 * Return value: None
 *
 * | Argument                                   | Description                                            | Type     | Valid Range                      | Required |
 * |--------------------------------------------|--------------------------------------------------------|----------|----------------------------------|----------|
 * | noc                                        | NOC to use for the transaction                         | uint32_t | 0 or 1                           | True     |
 * | cmd_buf                                    | Command buffer to use for the transaction              | uint32_t | 0 - 3                            | True     |
 * | val                                        | The value to be written                                | uint32_t | Any uint32_t value               | False    |
 * | dest_addr                                  | Encoding of the destination NOC location (x,y)+address | uint64_t | Results of \a get_noc_addr calls | False    |
 * | update_addr_lo (template parameter)        | Whether to update the lower 32 bits of the address     | bool     | true or false                    | False    |
 * | update_addr_hi (template parameter)        | Whether to update the upper 32 bits of the address     | bool     | true or false                    | False    |
 * | update_val (template parameter)            | Whether to set the value to be written                 | bool     | true or false                    | False    |
 * | posted (template parameter)                | Whether the call is posted (i.e. ack requirement)      | bool     | true or false                    | False    |
 * | update_counter (template parameter)        | Whether to update the write counters                   | bool     | true or false                    | False    |
 * | dst_type (template parameter)              | Whether the write is targeting L1 or a Stream Register | InlineWriteDst     | DEFAULT, L1, REG       | False    |
 */
// clang-format on
template <
    uint8_t noc_mode = DM_DEDICATED_NOC,
    bool update_addr_lo = false,
    bool update_addr_hi = false,
    bool update_val = false,
    bool posted = false,
    bool update_counter = true,
    InlineWriteDst dst_type = InlineWriteDst::DEFAULT>
inline __attribute__((always_inline)) void noc_fast_write_dw_inline_with_state(
    uint32_t noc, uint32_t cmd_buf, uint32_t val = 0, uint64_t dest_addr = 0) {
    static_assert("Error: Only High or Low address update is supported" && (update_addr_lo && update_addr_hi) == 0);
    if constexpr (update_counter && noc_mode == DM_DYNAMIC_NOC) {
        if constexpr (posted) {
            inc_noc_counter_val<proc_type, NocBarrierType::POSTED_WRITES_NUM_ISSUED>(noc, 1);
        } else {
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, 1);
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, 1);
        }
    }

    while (!noc_cmd_buf_ready(noc, cmd_buf));

    if constexpr (update_addr_lo) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, dest_addr);
        if constexpr (dst_type != InlineWriteDst::REG) {
            uint32_t be32 = 0xF << (dest_addr & (NOC_WORD_BYTES - 1));
            NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, be32);
        }
    } else if constexpr (update_addr_hi) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, dest_addr);
    }
    if constexpr (update_val) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_DATA, val);
    }
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

    if constexpr (update_counter && noc_mode == DM_DEDICATED_NOC) {
        if constexpr (posted) {
            noc_posted_writes_num_issued[noc] += 1;
        } else {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += 1;
        }
    }
}

// clang-format off
/**
 * The stateful NOC commands provide granular control over NOC register programming by writing
 * only a subset of registers for each transaction. This approach leverages the fact that many
 * transactions re-use certain values (e.g. length, coordinates) while varying others.
 *
 * This design provides significant advantages over previous stateful APIs:
 * - Fine-grained control: Users can specify exactly which registers to update per transaction
 * - Better optimization: Avoid unnecessary register writes for unchanged values
 * - Flexible transaction patterns: Support complex sequences with selective updates
 * - Performance benefits: Reduce NOC register write overhead for repetitive operations
 *
 * The flags parameter uses a bitmask approach to specify which registers to program.
 * Making template functions with a long list of booleans makes understanding what registers
 * are being set tedious. This is an attempt to pack that data in a way thats ~easy to visually parse.
 *
 * S/s: write, do not write to src address register (NOC_TARG_ADDR_LO)
 * N/n: write, do not write to noc coordinates register (NOC_RET_ADDR_COORDINATE)
 * D/d: write, do not write to dst address register (NOC_RET_ADDR_LO)
 * L/l: write, do not write to length register (NOC_AT_LEN_BE)
 *
 * M/m: write, do not write to multicast register (NOC_CMD_BRCST_PACKET)
 * K/k: write, do not write to linked register (NOC_CMD_VC_LINKED)
 * P/p: write, do not write to posted register (NOC_CMD_RESP_MARKED)
 *
 * V/v: write, do not write to value register (NOC_AT_DATA)
 * B/b: write, do not write to byte-enable register (NOC_AT_LEN_BE)
 *
 * WAIT/wait: wait, do not wait for command buffer readiness (NOC_CMD_CTRL)
 * SEND/send: send, do not send the transaction immediately (NOC_CTRL_SEND_REQ)
 */
// clang-format on
constexpr uint32_t CQ_NOC_FLAG_SRC = 0x01;
constexpr uint32_t CQ_NOC_FLAG_NOC = 0x02;
constexpr uint32_t CQ_NOC_FLAG_DST = 0x04;
constexpr uint32_t CQ_NOC_FLAG_LEN = 0x08;

constexpr uint32_t CQ_NOC_INLINE_FLAG_VAL = 0x10;
constexpr uint32_t CQ_NOC_INLINE_FLAG_BE = 0x20;

constexpr uint32_t CQ_NOC_CMD_FLAG_MCAST = 0x01;
constexpr uint32_t CQ_NOC_CMD_FLAG_LINKED = 0x02;
constexpr uint32_t CQ_NOC_CMD_FLAG_POSTED = 0x04;

enum CQNocFlags {
    CQ_NOC_sndl = 0,
    CQ_NOC_sndL = CQ_NOC_FLAG_LEN,
    CQ_NOC_snDl = CQ_NOC_FLAG_DST,
    CQ_NOC_snDL = CQ_NOC_FLAG_DST | CQ_NOC_FLAG_LEN,
    CQ_NOC_sNdl = CQ_NOC_FLAG_NOC,
    CQ_NOC_sNdL = CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_LEN,
    CQ_NOC_sNDl = CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST,
    CQ_NOC_sNDL = CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST | CQ_NOC_FLAG_LEN,
    CQ_NOC_Sndl = CQ_NOC_FLAG_SRC,
    CQ_NOC_SndL = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_LEN,
    CQ_NOC_SnDl = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_DST,
    CQ_NOC_SnDL = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_DST | CQ_NOC_FLAG_LEN,
    CQ_NOC_SNdl = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_NOC,
    CQ_NOC_SNdL = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_LEN,
    CQ_NOC_SNDl = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST,
    CQ_NOC_SNDL = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST | CQ_NOC_FLAG_LEN,
};

enum CQNocCmdFlags {
    CQ_NOC_mkp = 0,
    CQ_NOC_mkP = CQ_NOC_CMD_FLAG_POSTED,
    CQ_NOC_mKp = CQ_NOC_CMD_FLAG_LINKED,
    CQ_NOC_mKP = CQ_NOC_CMD_FLAG_LINKED | CQ_NOC_CMD_FLAG_POSTED,
    CQ_NOC_Mkp = CQ_NOC_CMD_FLAG_MCAST,
    CQ_NOC_MkP = CQ_NOC_CMD_FLAG_MCAST | CQ_NOC_CMD_FLAG_POSTED,
    CQ_NOC_MKp = CQ_NOC_CMD_FLAG_MCAST | CQ_NOC_CMD_FLAG_LINKED,
    CQ_NOC_MKP = CQ_NOC_CMD_FLAG_MCAST | CQ_NOC_CMD_FLAG_LINKED | CQ_NOC_CMD_FLAG_POSTED,
};

enum CQNocInlineFlags {
    CQ_NOC_INLINE_ndvb = 0,
    CQ_NOC_INLINE_ndvB = CQ_NOC_INLINE_FLAG_BE,
    CQ_NOC_INLINE_ndVb = CQ_NOC_INLINE_FLAG_VAL,
    CQ_NOC_INLINE_ndVB = CQ_NOC_INLINE_FLAG_VAL | CQ_NOC_INLINE_FLAG_BE,
    CQ_NOC_INLINE_nDvb = CQ_NOC_FLAG_DST,
    CQ_NOC_INLINE_nDvB = CQ_NOC_FLAG_DST | CQ_NOC_INLINE_FLAG_BE,
    CQ_NOC_INLINE_nDVb = CQ_NOC_FLAG_DST | CQ_NOC_INLINE_FLAG_VAL,
    CQ_NOC_INLINE_nDVB = CQ_NOC_FLAG_DST | CQ_NOC_INLINE_FLAG_VAL | CQ_NOC_INLINE_FLAG_BE,
    CQ_NOC_INLINE_Ndvb = CQ_NOC_FLAG_NOC,
    CQ_NOC_INLINE_NdvB = CQ_NOC_FLAG_NOC | CQ_NOC_INLINE_FLAG_BE,
    CQ_NOC_INLINE_NdVb = CQ_NOC_FLAG_NOC | CQ_NOC_INLINE_FLAG_VAL,
    CQ_NOC_INLINE_NdVB = CQ_NOC_FLAG_NOC | CQ_NOC_INLINE_FLAG_VAL | CQ_NOC_INLINE_FLAG_BE,
    CQ_NOC_INLINE_NDvb = CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST,
    CQ_NOC_INLINE_NDvB = CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST | CQ_NOC_INLINE_FLAG_BE,
    CQ_NOC_INLINE_NDVb = CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST | CQ_NOC_INLINE_FLAG_VAL,
    CQ_NOC_INLINE_NDVB = CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST | CQ_NOC_INLINE_FLAG_VAL | CQ_NOC_INLINE_FLAG_BE,
};

enum CQNocWait {
    CQ_NOC_wait = 0,
    CQ_NOC_WAIT = 1,
};
enum CQNocSend {
    CQ_NOC_send = 0,
    CQ_NOC_SEND = 1,
};

// clang-format off
/**
 * Initializes the stateful registers for NOC read operations using a specific command buffer.
 * This function sets up the basic NOC read command configuration that will be reused across
 * multiple read transactions using the same command buffer.
 *
 * Return value: None
 *
 * | Argument                     | Description                                     | Data type | Valid range | Required |
 * |------------------------------|-------------------------------------------------|-----------|-------------|----------|
 * | noc                          | Which NOC to use for the transaction            | uint32_t  | 0 or 1      | True     |
 * | cmd_buf (template parameter) | Which command buffer to initialize              | uint32_t  | 0 - 3       | True     |
 */
// clang-format on
template <uint32_t cmd_buf>
inline __attribute__((always_inline)) void noc_read_init_state(uint32_t noc) {
    uint32_t noc_rd_cmd_field =
        NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(1);

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_rd_cmd_field);
}

// clang-format off
/**
 * Initiates an asynchronous read transaction using previously initialized stateful registers.
 * This function must be preceded by a call to \a noc_read_init_state for the same command buffer.
 * The function leverages stateful NOC registers to minimize register writes for repeated transactions
 * with similar characteristics.
 *
 * This function provides more granular control compared to previous stateful NOC APIs by allowing
 * selective register updates via the flags parameter. Users can specify exactly which NOC registers
 * (source address, destination address, coordinates, length) should be programmed on each call,
 * enabling fine-tuned optimization for specific transaction patterns.
 *
 * Return value: None
 *
 * | Argument                      | Description                                              | Data type        | Valid range                                              | Required |
 * |-------------------------------|----------------------------------------------------------|------------------|----------------------------------------------------------|----------|
 * | noc                           | Which NOC to use for the transaction                     | uint32_t         | 0 or 1                                                   | True     |
 * | src_addr                      | Source NOC address (x,y)+local address                   | uint64_t         | Results of \a get_noc_addr calls                         | True     |
 * | dst_addr                      | Destination address in local L1 memory                   | uint32_t         | 0..1 MB                                                  | True     |
 * | size                          | Size of transaction in bytes                             | uint32_t         | 0..NOC_MAX_BURST_SIZE for single packet                  | True     |
 * | noc_mode (template parameter) | NOC mode for the transaction                             | uint8_t          | DM_DEDICATED_NOC, DM_DYNAMIC_NOC or DM_INVALID_NOC (0-2) | False    |
 * | cmd_buf (template parameter)  | Which command buffer to use for the transaction          | uint32_t         | 0 - 3                                                    | True     |
 * | flags (template parameter)    | Which NOC registers to update in this call               | enum CQNocFlags  | Combination of CQ_NOC_FLAG_* flags                       | True     |
 * | send (template parameter)     | Whether to send the transaction immediately              | enum CQNocSend   | CQ_NOC_SEND or CQ_NOC_send                               | False    |
 * | wait (template parameter)     | Whether to wait for command buffer readiness             | enum CQNocWait   | CQ_NOC_WAIT or CQ_NOC_wait                               | False    |
 */
// clang-format on
template <uint8_t noc_mode, uint32_t cmd_buf, enum CQNocFlags flags, enum CQNocSend send, enum CQNocWait wait>
inline __attribute__((always_inline)) void noc_common_read_with_state(
    uint32_t noc, uint32_t src_addr, uint32_t dst_addr, uint32_t size) {
    if constexpr (send) {
        if constexpr (noc_mode == DM_DYNAMIC_NOC) {
            inc_noc_counter_val<proc_type, NocBarrierType::READS_NUM_ISSUED>(noc, 1);
        } else if constexpr (noc_mode == DM_DEDICATED_NOC) {
            noc_reads_num_issued[noc] += 1;
        }
    }
    if constexpr (wait) {
        while (!noc_cmd_buf_ready(noc, cmd_buf));
    }
    if constexpr (flags & CQ_NOC_FLAG_SRC) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)src_addr);
    }
    if constexpr (flags & CQ_NOC_FLAG_DST) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, dst_addr);
    }
    if constexpr (flags & CQ_NOC_FLAG_LEN) {
        // TODO: Runtime assert for size < MAX_BURST_SIZE
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, size);
    }
}
template <
    uint8_t noc_mode = DM_DEDICATED_NOC,
    uint32_t cmd_buf,
    enum CQNocFlags flags,
    enum CQNocSend send = CQ_NOC_SEND,
    enum CQNocWait wait = CQ_NOC_WAIT>
inline __attribute__((always_inline)) void noc_read_with_state(
    uint32_t noc, uint64_t src_addr, uint32_t dst_addr, uint32_t size) {
    noc_common_read_with_state<noc_mode, cmd_buf, flags, send, wait>(noc, (uint32_t)src_addr, dst_addr, size);
    if constexpr (flags & CQ_NOC_FLAG_NOC) {
        NOC_CMD_BUF_WRITE_REG(
            noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(src_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    }
    if constexpr (send) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    }
}

// Same as above, but with src_noc_addr giving the source NOC address separately. This allows 64 bit addresses within
// the NOC transactions.
// clang-format on
template <
    uint8_t noc_mode = DM_DEDICATED_NOC,
    uint32_t cmd_buf,
    enum CQNocFlags flags,
    enum CQNocSend send = CQ_NOC_SEND,
    enum CQNocWait wait = CQ_NOC_WAIT>
inline __attribute__((always_inline)) void noc_read_with_state(
    uint32_t noc, uint32_t src_noc_addr, uint64_t src_addr, uint32_t dst_addr, uint32_t size) {
    noc_common_read_with_state<noc_mode, cmd_buf, flags, send, wait>(noc, (uint32_t)src_addr, dst_addr, size);
    if constexpr (flags & CQ_NOC_FLAG_NOC) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, src_noc_addr);
    }
    if constexpr (send) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    }
}

// clang-format off
/**
 * Initializes the stateful registers for NOC write operations using a specific command buffer.
 * This function sets up the basic NOC write command configuration including VC, multicast,
 * linked, and posted flags that will be reused across multiple write transactions using
 * the same command buffer.
 *
 * Return value: None
 *
 * | Argument                       | Description                                        | Data type           | Valid range         | Required |
 * |--------------------------------|----------------------------------------------------|---------------------|---------------------|----------|
 * | noc                            | Which NOC to use for the transaction               | uint32_t            | 0 or 1              | True     |
 * | vc                             | Virtual channel to use for the transactions        | uint32_t            | 0 - 3               | True     |
 * | cmd_buf (template parameter)   | Which command buffer to initialize                 | uint32_t            | 0 - 3               | True     |
 * | cmd_flags (template parameter) | Command flags for multicast/linked/posted options  | enum CQNocCmdFlags  | CQ_NOC_mkp variants | False    |
 */
// clang-format on
template <uint32_t cmd_buf, enum CQNocCmdFlags cmd_flags = CQ_NOC_mkp>
inline __attribute__((always_inline)) void noc_write_init_state(uint32_t noc, uint32_t vc) {
    constexpr bool multicast_path_reserve = true;
    uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) |
                             ((cmd_flags & CQ_NOC_CMD_FLAG_LINKED) ? NOC_CMD_VC_LINKED : 0x0) |
                             ((cmd_flags & CQ_NOC_CMD_FLAG_MCAST)
                                  ? ((multicast_path_reserve ? NOC_CMD_PATH_RESERVE : 0) | NOC_CMD_BRCST_PACKET)
                                  : 0x0) |
                             ((cmd_flags & CQ_NOC_CMD_FLAG_POSTED) ? 0 : NOC_CMD_RESP_MARKED);

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
}

// clang-format off
/**
 * Initiates an asynchronous write transaction using previously initialized stateful registers.
 * This function must be preceded by a call to \a noc_write_init_state for the same command buffer.
 * The function leverages stateful NOC registers to minimize register writes for repeated transactions
 * with similar characteristics.
 *
 * This function provides more granular control compared to previous stateful NOC APIs by allowing
 * selective register updates via the flags parameter. Users can specify exactly which NOC registers
 * (source address, destination address, coordinates, length) should be programmed on each call,
 * enabling fine-tuned optimization for specific transaction patterns.
 *
 * Return value: None
 *
 * | Argument                            | Description                                              | Data type       | Valid range                                              | Required |
 * |-------------------------------------|----------------------------------------------------------|-----------------|----------------------------------------------------------|----------|
 * | noc                                 | Which NOC to use for the transaction                     | uint32_t        | 0 or 1                                                   | True     |
 * | src_addr                            | Source address in local L1 memory                        | uint32_t        | 0..1 MB                                                  | True     |
 * | dst_noc_addr                        | Destination NOC address (x,y)                            | uint32_t        |                                                          | True     |
 * | dst_addr                            | Local address in destination NOC tile                    | uint64_t        | 0..4 GB                                                  | True     |
 * | size                                | Size of transaction in bytes                             | uint32_t        | 0..NOC_MAX_BURST_SIZE for single packet                  | False    |
 * | ndests                              | Number of destinations for multicast operations          | uint32_t        | 1 or more                                                | False    |
 * | noc_mode (template parameter)       | NOC mode for the transaction                             | uint8_t         | DM_DEDICATED_NOC, DM_DYNAMIC_NOC or DM_INVALID_NOC (0-2) | False    |
 * | cmd_buf (template parameter)        | Which command buffer to use for the transaction          | uint32_t        | 0 - 3                                                    | True     |
 * | flags (template parameter)          | Which NOC registers to update in this call               | enum CQNocFlags | Combination of CQ_NOC_FLAG_* flags                       | True     |
 * | send (template parameter)           | Whether to send the transaction immediately              | enum CQNocSend  | CQ_NOC_SEND or CQ_NOC_send                               | False    |
 * | wait (template parameter)           | Whether to wait for command buffer readiness             | enum CQNocWait  | CQ_NOC_WAIT or CQ_NOC_wait                               | False    |
 * | update_counter (template parameter) | Whether to increment write counters                      | bool            | true or false                                            | False    |
 * | posted (template parameter)         | Whether the transaction is posted (no ack required)      | bool            | true or false                                            | False    |
 */
// clang-format on
template <
    uint8_t noc_mode = DM_DEDICATED_NOC,
    uint32_t cmd_buf,
    enum CQNocFlags flags,
    enum CQNocSend send = CQ_NOC_SEND,
    enum CQNocWait wait = CQ_NOC_WAIT,
    bool update_counter = true,
    bool posted = false>
inline __attribute__((always_inline)) void noc_wwrite_with_state(
    uint32_t noc, uint32_t src_addr, uint32_t dst_noc_addr, uint64_t dst_addr, uint32_t size = 0, uint32_t ndests = 1) {
    if constexpr (update_counter && noc_mode == DM_DYNAMIC_NOC) {
        if constexpr (posted) {
            inc_noc_counter_val<proc_type, NocBarrierType::POSTED_WRITES_NUM_ISSUED>(noc, 1);
        } else {
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, 1);
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, ndests);
        }
    }

    if constexpr (wait) {
        while (!noc_cmd_buf_ready(noc, cmd_buf));
    }
    if constexpr (flags & CQ_NOC_FLAG_SRC) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_addr);
    }
    if constexpr (flags & CQ_NOC_FLAG_DST) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dst_addr);
    }
    if constexpr (flags & CQ_NOC_FLAG_NOC) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_COORDINATE, dst_noc_addr);
    }
    if constexpr (flags & CQ_NOC_FLAG_LEN) {
        // TODO: Runtime assert for size < MAX_BURST_SIZE
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, size);
    }
    if constexpr (send) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    }

    if constexpr (update_counter && noc_mode == DM_DEDICATED_NOC) {
        if constexpr (posted) {
            noc_posted_writes_num_issued[noc] += 1;
        } else {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += ndests;
        }
    }
}
// clang-format off
/**
 * Wrapper around the above. \a dst_addr is a composite of NOC-xy coordinate and address within  NOC tile, result of \a get_noc_addr calls
 */
// clang-format on
template <
    uint8_t noc_mode = DM_DEDICATED_NOC,
    uint32_t cmd_buf,
    enum CQNocFlags flags,
    enum CQNocSend send = CQ_NOC_SEND,
    enum CQNocWait wait = CQ_NOC_WAIT,
    bool update_counter = true,
    bool posted = false>
inline __attribute__((always_inline)) void noc_write_with_state(
    uint32_t noc, uint32_t src_addr, uint64_t dst_addr, uint32_t size = 0, uint32_t ndests = 1) {
    uint32_t dst_noc_addr = 0;
    if constexpr (flags & CQ_NOC_FLAG_NOC) {
        dst_noc_addr = (uint32_t)(dst_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
    }
    noc_wwrite_with_state<noc_mode, cmd_buf, flags, send, wait, update_counter, posted>(
        noc, src_addr, dst_noc_addr, dst_addr, size, ndests);
}

// clang-format off
/**
 * Initializes the stateful registers for NOC inline direct write operations using a specific
 * command buffer. This function sets up the basic NOC inline write command configuration including
 * VC, multicast, and posted flags that will be reused across multiple inline write transactions.
 * Inline writes allow writing 32-bit values directly without requiring a separate source buffer.
 *
 * Return value: None
 *
 * | Argument                       | Description                                        | Data type           | Valid range         | Required |
 * |--------------------------------|----------------------------------------------------|---------------------|---------------------|----------|
 * | noc                            | Which NOC to use for the transaction               | uint32_t            | 0 or 1              | True     |
 * | vc                             | Virtual channel to use for the transactions        | uint32_t            | 0 - 3               | True     |
 * | cmd_buf (template parameter)   | Which command buffer to initialize                 | uint32_t            | 0 - 3               | True     |
 * | cmd_flags (template parameter) | Command flags for multicast/posted options         | enum CQNocCmdFlags  | CQ_NOC_mkp variants | False    |
 */
// clang-format on
template <uint32_t cmd_buf, enum CQNocCmdFlags cmd_flags = CQ_NOC_mkp>
FORCE_INLINE void noc_inline_dw_write_init_state(uint32_t noc, uint32_t vc) {
    constexpr bool static_vc_alloc = true;

    uint32_t noc_cmd_field =
        (static_vc_alloc ? NOC_CMD_VC_STATIC : 0x0) | NOC_CMD_STATIC_VC(vc) | NOC_CMD_CPY | NOC_CMD_WR |
        NOC_CMD_WR_INLINE |
        ((cmd_flags & CQ_NOC_CMD_FLAG_MCAST) ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0) |
        ((cmd_flags & CQ_NOC_CMD_FLAG_POSTED) ? 0x0 : NOC_CMD_RESP_MARKED);

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
}

// clang-format off
/**
 * Initiates an inline direct write transaction using previously initialized stateful registers.
 * This function must be preceded by a call to \a noc_inline_dw_write_init_state for the same command buffer.
 * Inline writes allow writing a 32-bit value directly to a NOC destination without requiring a separate
 * source buffer, making them efficient for register writes and small data updates.
 *
 * This function provides more granular control compared to previous stateful NOC APIs by allowing
 * selective register updates via the flags parameter. Users can specify exactly which NOC registers
 * (destination address, coordinates, value, byte enable) should be programmed on each call,
 * enabling fine-tuned optimization for specific inline write patterns.
 *
 * Return value: None
 *
 * | Argument                     | Description                                        | Data type             | Valid range                      | Required |
 * |------------------------------|----------------------------------------------------|-----------------------|----------------------------------|----------|
 * | noc                          | Which NOC to use for the transaction               | uint32_t              | 0 or 1                           | True     |
 * | dst_addr                     | Destination NOC address (x,y)+local address        | uint64_t              | Results of \a get_noc_addr calls | True     |
 * | val                          | 32-bit value to write                              | uint32_t              | Any uint32_t value               | False    |
 * | be                           | Byte enable mask for the write                     | uint8_t               | 0x1-0xF                          | False    |
 * | cmd_buf (template parameter) | Which command buffer to use for the transaction    | uint32_t              | 0 - 3                            | True     |
 * | flags (template parameter)   | Which NOC registers to update in this call         | enum CQNocInlineFlags | CQ_NOC_INLINE_* variants         | True     |
 * | wait (template parameter)    | Whether to wait for command buffer readiness       | enum CQNocWait        | CQ_NOC_WAIT or CQ_NOC_wait       | False    |
 * | send (template parameter)    | Whether to send the transaction immediately        | enum CQNocSend        | CQ_NOC_SEND or CQ_NOC_send       | False    |
 */
// clang-format on
template <
    uint32_t cmd_buf,
    enum CQNocInlineFlags flags,
    enum CQNocWait wait = CQ_NOC_WAIT,
    enum CQNocSend send = CQ_NOC_SEND>
FORCE_INLINE void noc_inline_dw_write_with_state(uint32_t noc, uint64_t dst_addr, uint32_t val = 0, uint8_t be = 0xF) {
    if constexpr (wait) {
        while (!noc_cmd_buf_ready(noc, cmd_buf));
    }

    if constexpr (flags & CQ_NOC_INLINE_FLAG_VAL) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_DATA, val);
    }
    if constexpr (flags & CQ_NOC_FLAG_DST) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)(dst_addr));
    }
    if constexpr (flags & CQ_NOC_FLAG_NOC) {
        NOC_CMD_BUF_WRITE_REG(
            noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(dst_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    }
    if constexpr (flags & CQ_NOC_INLINE_FLAG_BE) {
        uint32_t be32 = be;
        uint32_t be_shift = (dst_addr & (NOC_WORD_BYTES - 1));
        be32 = (be32 << be_shift);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, be32);
    }
    if constexpr (send) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    }
}

template <uint8_t MAX_NOCS_TO_INIT = NUM_NOCS>
inline __attribute__((always_inline)) void ncrisc_dynamic_noc_full_sync() {
    for (uint32_t noc = 0; noc < MAX_NOCS_TO_INIT; noc++) {
        while (!ncrisc_dynamic_noc_reads_flushed(noc)) {
            invalidate_l1_cache();
        }
        while (!ncrisc_dynamic_noc_nonposted_writes_sent(noc)) {
            invalidate_l1_cache();
        }
        while (!ncrisc_dynamic_noc_nonposted_writes_flushed(noc)) {
            invalidate_l1_cache();
        }
        while (!ncrisc_dynamic_noc_nonposted_atomics_flushed(noc)) {
            invalidate_l1_cache();
        }
        while (!ncrisc_dynamic_noc_posted_writes_sent(noc)) {
            invalidate_l1_cache();
        }
    }
}

template <bool write, bool posted>
inline __attribute__((always_inline)) uint32_t get_noc_counter_for_debug(uint32_t noc) {
    if constexpr (write) {
        if constexpr (posted) {
            return NOC_STATUS_READ_REG(noc, NIU_MST_POSTED_WR_REQ_SENT);
        } else {
            return NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_WR_REQ_SENT);
        }
    } else {
        // Read
        static_assert(posted == false, "There is no such thing as posted reads");
        return NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED);
    }
}
