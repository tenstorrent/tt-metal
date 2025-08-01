// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "noc_parameters.h"
#include "dev_msgs.h"
#include "noc_overlay_parameters.h"
#include "debug/assert.h"

#if defined(COMPILE_FOR_BRISC)
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
#define MY_NOC_ENCODING(noc_index) NOC_CMD_BUF_READ_REG(noc_index, 0, NOC_CFG(NOC_ID_LOGICAL))

////
/*TODO: RT review this file, currently using wormhole b0 copy, check if any changes needed for BH*/
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

// BH has 64 bit address space but pipegen was not updated to support this so WH scheme of encoding addresses is used
// (36 bits of address followed by coordinates) This means that lo and mid registers need to have the address portion
// while the coordinates go into hi register Metal does not need to use more than 32 bits for addresses but the 60th bit
// needs to be set to enable NoC transactions through PCIe (see get_pcie_base_addr_from_device)
constexpr uint32_t NOC_ADDR_COORD_SHIFT = 36;
const uint32_t NOC_TARG_ADDR_COORDINATE = NOC_TARG_ADDR_HI;
const uint32_t NOC_RET_ADDR_COORDINATE = NOC_RET_ADDR_HI;
const uint32_t NOC_COORDINATE_MASK = 0xFFFFFF;

// Mask for the 60th bit of the address in NOC_TARG/RET_ADDR_MID, which is set to enable PCIe transactions
constexpr uint32_t NOC_PCIE_MASK = 0x1000000F;

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

template <uint8_t proc_t, NocBarrierType barrier_type>
inline __attribute__((always_inline)) uint32_t get_noc_counter_address(uint32_t noc) {
    static_assert(proc_t < MaxDMProcessorsPerCoreType);
    static_assert(static_cast<std::underlying_type_t<NocBarrierType>>(barrier_type) < NUM_BARRIER_TYPES);
    constexpr uint32_t offset =
        MEM_NOC_COUNTER_BASE +
        (proc_t * NUM_BARRIER_TYPES + static_cast<std::underlying_type_t<NocBarrierType>>(barrier_type)) * NUM_NOCS *
            MEM_NOC_COUNTER_SIZE;
    return offset + noc * MEM_NOC_COUNTER_SIZE;
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
#if defined(WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION)
    if (addr == NOC_CTRL) {
        auto* watcher_msg = GET_MAILBOX_ADDRESS_DEV(watcher);
        watcher_msg->noc_linked_status[noc] = (val & NOC_CMD_VC_LINKED) != 0;
    }
#endif
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

inline __attribute__((always_inline)) uint32_t noc_get_interim_inline_value_addr(uint32_t noc, uint64_t dst_noc_addr) {
    // On Blackhole issuing inline writes and atomics requires all 4 memory ports to accept the transaction at the same
    // time. If one port on the receipient has no back-pressure then the transaction will hang because there is no
    // mechanism to allow one memory port to move ahead of another. To workaround this hang, we emulate inline writes on
    // Blackhole by writing the value to be written to local L1 first and then issue a noc async write.

    // If dst_noc_addr is not L1 aligned then we need to offset the src address by 4B since inline write dst address
    // needs to respect 4B alignment.
    ASSERT((dst_noc_addr & 0x3) == 0);
    uint32_t offset = dst_noc_addr & 0xF;
    uint32_t src_addr = MEM_L1_INLINE_BASE + (2 * MEM_L1_INLINE_SIZE_PER_NOC) * proc_type;
#ifdef COMPILE_FOR_TRISC
    ASSERT(0);  // we do not have L1 space for inline values for TRISCs.
#endif
    src_addr += noc * MEM_L1_INLINE_SIZE_PER_NOC + offset;
    return src_addr;
}

template <uint8_t noc_mode = DM_DEDICATED_NOC>
inline __attribute__((always_inline)) void ncrisc_noc_fast_read(
    uint32_t noc, uint32_t cmd_buf, uint64_t src_addr, uint32_t dest_addr, uint32_t len_bytes) {
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        inc_noc_counter_val<proc_type, NocBarrierType::READS_NUM_ISSUED>(noc, 1);
    }
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        uint32_t noc_rd_cmd_field =
            NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(1);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_rd_cmd_field);
    }
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)src_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_MID, (uint32_t)(src_addr >> 32) & NOC_PCIE_MASK);
    NOC_CMD_BUF_WRITE_REG(
        noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(src_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
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
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dest_addr >> 32) & NOC_PCIE_MASK);
    NOC_CMD_BUF_WRITE_REG(
        noc, cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_BRCST_EXCLUDE, 0);
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
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dest_addr >> 32) & NOC_PCIE_MASK);
    NOC_CMD_BUF_WRITE_REG(
        noc, cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_BRCST_EXCLUDE, 0);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        noc_nonposted_writes_num_issued[noc] += 1;
        noc_nonposted_writes_acked[noc] += num_dests;
    }
}

template <uint8_t noc_mode = DM_DEDICATED_NOC>
inline __attribute__((always_inline)) void ncrisc_noc_fast_write_exclude_region(
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
    uint32_t exclude_region) {
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, 1);
        inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, num_dests);
    }
    uint32_t noc_cmd_field =
        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) | (linked ? NOC_CMD_VC_LINKED : 0x0) |
        (mcast ? ((multicast_path_reserve ? NOC_CMD_PATH_RESERVE : 0) | NOC_CMD_BRCST_PACKET) : 0x0) |
        NOC_CMD_RESP_MARKED;

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dest_addr >> 32) & NOC_PCIE_MASK);
    NOC_CMD_BUF_WRITE_REG(
        noc, cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_BRCST_EXCLUDE, exclude_region);
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
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dest_addr >> 32) & NOC_PCIE_MASK);
    NOC_CMD_BUF_WRITE_REG(
        noc, cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
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

inline __attribute__((always_inline)) void noc_init(uint32_t atomic_ret_val) {
#pragma GCC unroll 0
    for (int noc = 0; noc < NUM_NOCS; noc++) {
        uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(noc, 0, NOC_CFG(NOC_ID_LOGICAL));
        uint32_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
        uint32_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
        uint64_t xy_local_addr = NOC_XY_ADDR(my_x, my_y, 0);

        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_WR_CMD_BUF, NOC_TARG_ADDR_MID, 0x0);
        NOC_CMD_BUF_WRITE_REG(
            noc,
            NCRISC_WR_CMD_BUF,
            NOC_TARG_ADDR_COORDINATE,
            (uint32_t)(xy_local_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_WR_REG_CMD_BUF, NOC_TARG_ADDR_MID, 0x0);
        NOC_CMD_BUF_WRITE_REG(
            noc,
            NCRISC_WR_REG_CMD_BUF,
            NOC_TARG_ADDR_COORDINATE,
            (uint32_t)(xy_local_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);

        uint64_t atomic_ret_addr = NOC_XY_ADDR(my_x, my_y, atomic_ret_val);
        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_AT_CMD_BUF, NOC_RET_ADDR_LO, (uint32_t)(atomic_ret_addr & 0xFFFFFFFF));
        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_AT_CMD_BUF, NOC_RET_ADDR_MID, 0x0);
        NOC_CMD_BUF_WRITE_REG(
            noc,
            NCRISC_AT_CMD_BUF,
            NOC_RET_ADDR_COORDINATE,
            (uint32_t)(atomic_ret_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);

        uint32_t noc_rd_cmd_field =
            NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(1);
        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_CTRL, noc_rd_cmd_field);
        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_MID, 0x0);  // get rid of this?
        NOC_CMD_BUF_WRITE_REG(
            noc,
            NCRISC_RD_CMD_BUF,
            NOC_RET_ADDR_COORDINATE,
            (uint32_t)(xy_local_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    }
}

inline __attribute__((always_inline)) void dynamic_noc_init() {
#pragma GCC unroll 0
    for (int noc = 0; noc < NUM_NOCS; noc++) {
        uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(noc, 0, NOC_CFG(NOC_ID_LOGICAL));
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
    uint32_t noc, uint32_t cmd_buf, uint64_t src_addr, uint32_t dest_addr, uint32_t len_bytes) {
    while (len_bytes > NOC_MAX_BURST_SIZE) {
        while (!noc_cmd_buf_ready(noc, cmd_buf));
        ncrisc_noc_fast_read<noc_mode>(noc, cmd_buf, src_addr, dest_addr, NOC_MAX_BURST_SIZE);
        src_addr += NOC_MAX_BURST_SIZE;
        dest_addr += NOC_MAX_BURST_SIZE;
        len_bytes -= NOC_MAX_BURST_SIZE;
    }
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    ncrisc_noc_fast_read<noc_mode>(noc, cmd_buf, src_addr, dest_addr, len_bytes);
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

template <uint8_t noc_mode = DM_DEDICATED_NOC>
inline __attribute__((always_inline)) void noc_fast_write_dw_inline(
    uint32_t noc,
    uint32_t cmd_buf,
    uint32_t val,
    uint64_t dest_addr,
    uint32_t be,
    uint32_t static_vc,
    bool mcast,
    bool posted = false) {
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
    // If we're given a misaligned address, don't write to the bytes in the word below the address
    uint32_t be_shift = (dest_addr & (NOC_WORD_BYTES - 1));
    be32 = (be32 << be_shift);

    while (!noc_cmd_buf_ready(noc, cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_DATA, val);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)(dest_addr));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_MID, (uint32_t)(dest_addr >> 32) & NOC_PCIE_MASK);
    NOC_CMD_BUF_WRITE_REG(
        noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
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
    // On Blackhole issuing inline writes and atomics requires all 4 memory ports to accept the transaction at the same
    // time. If one port on the receipient has no back-pressure then the transaction will hang because there is no
    // mechanism to allow one memory port to move ahead of another. To workaround this hang, we emulate force atomics to
    // be non-posted.
    posted = false;
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
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_MID, (uint32_t)(addr >> 32) & NOC_PCIE_MASK);
    NOC_CMD_BUF_WRITE_REG(
        noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
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
template <uint8_t noc_mode = DM_DEDICATED_NOC, bool skip_ptr_update = false>
inline __attribute__((always_inline)) void ncrisc_noc_fast_read_with_transaction_id(
    uint32_t noc, uint32_t cmd_buf, uint32_t src_base_addr, uint32_t src_addr, uint32_t dest_addr, uint32_t trid) {
    if constexpr (noc_mode == DM_DYNAMIC_NOC && !skip_ptr_update) {
        inc_noc_counter_val<proc_type, NocBarrierType::READS_NUM_ISSUED>(noc, 1);
    }
    uint32_t src_addr_;
    src_addr_ = src_base_addr + src_addr;

    while (!noc_cmd_buf_ready(noc, cmd_buf));
    while (NOC_STATUS_READ_REG(noc, NIU_MST_REQS_OUTSTANDING_ID(trid)) > ((NOC_MAX_TRANSACTION_ID_COUNT + 1) / 2));

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_addr_);  // (uint32_t)src_addr
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    if constexpr (noc_mode == DM_DEDICATED_NOC && !skip_ptr_update) {
        noc_reads_num_issued[noc] += 1;
    }
}

// set transaction id for a noc read
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
    // Handles reading from PCIe
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_MID, (uint32_t)(src_noc_addr >> 32) & NOC_PCIE_MASK);
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
 * | non_posted (template parameter) | Whether the transaction is nonposted (i.e. requires ack) | bool      | true or false                    | False    |
 * | one_packet (template parameter) | Whether transaction size is <= NOC_MAX_BURST_SIZE        | bool      | true or false                    | False    |
 */
// clang-format on
template <bool non_posted = true, bool one_packet = false>
inline __attribute__((always_inline)) void ncrisc_noc_write_set_state(
    uint32_t noc, uint32_t cmd_buf, uint64_t dst_noc_addr, uint32_t len_bytes = 0, const uint32_t vc = 0) {
    while (!noc_cmd_buf_ready(noc, cmd_buf));

    uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) |
                             0x0 |  // (linked ? NOC_CMD_VC_LINKED : 0x0)
                             0x0 |  // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
                             (non_posted ? NOC_CMD_RESP_MARKED : 0x0);

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    // Handles writing to PCIe
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dst_noc_addr >> 32) & NOC_PCIE_MASK);
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
 * | non_posted (template parameter)     | Whether the transaction is nonposted (i.e. requires ack) | bool      | true or false                                            | False    |
 * | update_counter (template parameter) | Whether to increment write counters                      | bool      | true or false                                            | False    |
 * | one_packet (template parameter)     | Whether transaction size is <= NOC_MAX_BURST_SIZE        | bool      | true or false                                            | False    |
 */
// clang-format on
template <
    uint8_t noc_mode = DM_DEDICATED_NOC,
    bool non_posted = true,
    bool update_counter = true,
    bool one_packet = false>
inline __attribute__((always_inline)) void ncrisc_noc_write_with_state(
    uint32_t noc, uint32_t cmd_buf, uint32_t src_local_addr, uint32_t dst_local_addr, uint32_t len_bytes = 0) {
    if constexpr (update_counter && noc_mode == DM_DYNAMIC_NOC) {
        if constexpr (non_posted) {
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, 1);
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, 1);
        } else {
            inc_noc_counter_val<proc_type, NocBarrierType::POSTED_WRITES_NUM_ISSUED>(noc, 1);
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
        if constexpr (non_posted) {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += 1;
        } else {
            noc_posted_writes_num_issued[noc] += 1;
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
 * | non_posted (template parameter)     | Whether the transaction is nonposted (i.e. requires ack) | bool      | true or false                                            | False    |
 * | update_counter (template parameter) | Whether to increment write counters                      | bool      | true or false                                            | False    |
 */
// clang-format on
template <uint8_t noc_mode = DM_DEDICATED_NOC, bool non_posted = true, bool update_counter = true>
inline __attribute__((always_inline)) void ncrisc_noc_write_any_len_with_state(
    uint32_t noc, uint32_t cmd_buf, uint32_t src_local_addr, uint32_t dst_local_addr, uint32_t len_bytes) {
    if (len_bytes > NOC_MAX_BURST_SIZE) {
        // Set data size for while loop
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, NOC_MAX_BURST_SIZE);

        while (len_bytes > NOC_MAX_BURST_SIZE) {
            ncrisc_noc_write_with_state<noc_mode, non_posted, update_counter, true /* one_packet */>(
                noc, cmd_buf, src_local_addr, dst_local_addr);

            len_bytes -= NOC_MAX_BURST_SIZE;
            src_local_addr += NOC_MAX_BURST_SIZE;
            dst_local_addr += NOC_MAX_BURST_SIZE;
        }
    }

    // left-over packet
    ncrisc_noc_write_with_state<noc_mode, non_posted, update_counter>(
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
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_MID, (uint32_t)(dest_addr >> 32) & NOC_PCIE_MASK);
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
 * | Argument                            | Description                                            | Type     | Valid Range                      | Required |
 * |-------------------------------------|--------------------------------------------------------|----------|----------------------------------|----------|
 * | noc                                 | NOC to use for the transaction                         | uint32_t | 0 or 1                           | True     |
 * | cmd_buf                             | Command buffer to use for the transaction              | uint32_t | 0 - 3                            | True     |
 * | val                                 | The value to be written                                | uint32_t | Any uint32_t value               | False    |
 * | dest_addr                           | Encoding of the destination NOC location (x,y)+address | uint64_t | Results of \a get_noc_addr calls | False    |
 * | update_addr_lo (template parameter) | Whether to update the lower 32 bits of the address     | bool     | true or false                    | False    |
 * | update_addr_hi (template parameter) | Whether to update the upper 32 bits of the address     | bool     | true or false                    | False    |
 * | update_val (template parameter)     | Whether to set the value to be written                 | bool     | true or false                    | False    |
 * | posted (template parameter)         | Whether the call is posted (i.e. ack requirement)      | bool     | true or false                    | False    |
 * | update_counter (template parameter) | Whether to update the write counters                   | bool     | true or false                    | False    |
 */
// clang-format on
template <
    uint8_t noc_mode = DM_DEDICATED_NOC,
    bool update_addr_lo = false,
    bool update_addr_hi = false,
    bool update_val = false,
    bool posted = false,
    bool update_counter = true>
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
