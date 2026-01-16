// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "noc_parameters.h"

#define CPU_TO_TL2_ADDR(addr) (addr - MEM_PORT_CACHEABLE_BASE_ADDR)
#define TL2_TO_CPU_CACHEABLE_ADDR(addr) (addr + MEM_PORT_CACHEABLE_BASE_ADDR)
#define TL2_TO_CPU_UNCACHE_ADDR(addr) (addr + MEM_PORT_NONCACHEABLE_BASE_ADDR)

const uint32_t NCRISC_WR_CMD_BUF = 3;
const uint32_t NCRISC_WR_CMD_BUF_0 = 0;
const uint32_t NCRISC_WR_CMD_BUF_1 = 1;
const uint32_t NCRISC_SMALL_TXN_CMD_BUF = 3;

const uint32_t NCRISC_WR_DEF_TRID = 0;
const uint32_t NCRISC_WR_LOCAL_TRID = 1;
const uint32_t NCRISC_RD_DEF_TRID = 2;
const uint32_t NCRISC_HEADER_RD_TRID = 3;
const uint32_t NCRISC_RD_START_TRID = 4;
const uint32_t NCRISC_RD_END_TRID = 13;
const uint32_t NCRISC_ETH_START_TRID = 14;
const uint32_t NCRISC_ETH_END_TRID = 15;

const uint32_t NCRISC_WR_DEF_VC = 0;
const uint32_t NCRISC_RD_DEF_VC = 1;

inline void NOC_CMD_BUF_WRITE_REG(uint32_t noc, uint32_t buf, uint64_t addr, uint32_t val) {
    uint64_t offset = (buf << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT) + addr;
    volatile uint32_t* ptr = (volatile uint32_t*)offset;
    *ptr = val;
}

inline uint32_t NOC_CMD_BUF_READ_REG(uint32_t noc, uint32_t buf, uint64_t addr) {
    uint64_t offset = (buf << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT) + addr;
    volatile uint32_t* ptr = (volatile uint32_t*)offset;
    return *ptr;
}

inline uint32_t NOC_STATUS_READ_REG(uint32_t noc, uint32_t reg_id) {
    uint64_t offset = (noc << NOC_INSTANCE_OFFSET_BIT) + NOC_STATUS(reg_id);
    volatile uint32_t* ptr = (volatile uint32_t*)offset;
    return *ptr;
}

inline uint32_t NOC_CFG_READ_REG(uint32_t noc, uint32_t reg_id) {
    uint64_t offset = (noc << NOC_INSTANCE_OFFSET_BIT) + NOC_CFG(reg_id);
    volatile uint32_t* ptr = (volatile uint32_t*)offset;
    return *ptr;
}

bool cmd_buf_ok(uint32_t noc, uint32_t cmd_buf = NCRISC_SMALL_TXN_CMD_BUF);
void noc_read(
    uint32_t noc,
    uint64_t src_coordinate,
    uint64_t src_addr,
    uint64_t dest_coordinate,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t transaction_id = NCRISC_RD_DEF_TRID,
    uint32_t static_vc = NCRISC_RD_DEF_VC,
    uint32_t cmd_buf = NCRISC_SMALL_TXN_CMD_BUF);
bool noc_reads_flushed(uint32_t noc, uint32_t transaction_id = NCRISC_RD_DEF_TRID);
bool all_noc_reads_flushed(uint32_t noc);
void noc_write(
    uint32_t noc,
    uint64_t src_coordinate,
    uint64_t src_addr,
    uint64_t dest_coordinate,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t transaction_id = NCRISC_WR_DEF_TRID,
    uint32_t vc = NCRISC_WR_DEF_VC,
    bool mcast = false,
    bool linked = false,
    uint32_t cmd_buf = NCRISC_SMALL_TXN_CMD_BUF);
bool noc_writes_sent(uint32_t noc, uint32_t transaction_id = NCRISC_WR_DEF_TRID);
bool all_noc_writes_sent(uint32_t noc);
bool noc_nonposted_writes_flushed(uint32_t noc, uint32_t transaction_id = NCRISC_WR_DEF_TRID);
bool all_noc_nonposted_writes_flushed(uint32_t noc);
void noc_atomic_increment(
    uint32_t noc,
    uint64_t noc_coordinate,
    uint64_t addr,
    uint32_t incr = 1,
    uint32_t wrap = 31,
    bool linked = false,
    uint32_t cmd_buf = NCRISC_SMALL_TXN_CMD_BUF);
bool noc_command_ready(uint32_t ring, uint32_t cmd_select);
