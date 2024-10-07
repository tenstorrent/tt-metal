// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


namespace interleaved_addr_gen {

template <bool DRAM>
FORCE_INLINE
uint32_t get_bank_offset_index(uint32_t id) {
    if constexpr (DRAM) {   // DRAM
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
        return udivsi3_const_divisor<NUM_DRAM_BANKS>(id);
#else
        return id >> LOG_BASE_2_OF_NUM_DRAM_BANKS;
#endif
    } else {                // L1
#ifdef IS_NOT_POW2_NUM_L1_BANKS
        return udivsi3_const_divisor<NUM_L1_BANKS>(id);
#else
        return id >> LOG_BASE_2_OF_NUM_L1_BANKS;
#endif
    }
}

template <bool DRAM>
FORCE_INLINE
uint32_t get_bank_index(uint32_t id, uint32_t bank_offset_index) {
    if constexpr (DRAM) {   // DRAM
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
        return id - bank_offset_index * NUM_DRAM_BANKS;
#else
        return id & (NUM_DRAM_BANKS - 1);
#endif
    } else {                // L1
#ifdef IS_NOT_POW2_NUM_L1_BANKS
        return id - bank_offset_index * NUM_L1_BANKS;
#else
        return id & (NUM_L1_BANKS - 1);
#endif
    }
}

template <bool DRAM>
FORCE_INLINE
uint32_t get_noc_xy(uint32_t bank_index) {
    if constexpr (DRAM) {   // DRAM
        return dram_bank_to_noc_xy[noc_index][bank_index];
    } else {                // L1
        return l1_bank_to_noc_xy[noc_index][bank_index];

    }
}

template <bool DRAM>
FORCE_INLINE
uint32_t get_bank_offset(uint32_t bank_index) {
    if constexpr (DRAM) {   // DRAM
        return bank_to_dram_offset[bank_index];
    } else {                // L1
        return bank_to_l1_offset[bank_index];

    }
}

}  // namespace addrgen

template <bool DRAM, uint32_t tile_hw = 1024>
struct InterleavedAddrGenFast {
    uint32_t bank_base_address;  // Base address for the whole tensor.
    // TODO: Remove page_size from argument list. This can be derived from data_format
    uint32_t page_size;          // Num bytes in bank unit.
    DataFormat data_format;      // Data format

    FORCE_INLINE
    uint32_t get_addr(const uint32_t id, const uint32_t bank_offset_index, const uint32_t bank_index, const uint32_t offset = 0) const {
        return MUL_WITH_TILE_SIZE<tile_hw>((uint)this->data_format, bank_offset_index) + this->bank_base_address + offset + interleaved_addr_gen::get_bank_offset<DRAM>(bank_index);
    }

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index);

        uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
        return noc_addr;
    }

    FORCE_INLINE
    void noc_async_read_tile(const uint32_t id, uint32_t dest_addr, const uint32_t offset = 0) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t src_addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t src_noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index);

        WAYPOINT("NRTW");
        DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc_index, get_noc_addr_helper(src_noc_xy, src_addr), dest_addr, this->page_size);
        while (!noc_cmd_buf_ready(noc_index, NCRISC_RD_CMD_BUF));
        WAYPOINT("NRTD");

        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dest_addr);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, src_addr);      // (uint32_t)src_addr
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_COORDINATE, src_noc_xy);   // src_addr >> 32
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, this->page_size);  // len_bytes
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_reads_num_issued[noc_index] += 1;
    }

    FORCE_INLINE
    void noc_async_write_tile(const uint32_t id, uint32_t src_addr) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t dest_addr = this->get_addr(id, bank_offset_index, bank_index);
        uint32_t dest_noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index);

        WAYPOINT("NWTW");
        DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc_index, get_noc_addr_helper(dest_noc_xy, dest_addr), src_addr, this->page_size);
        while (!noc_cmd_buf_ready(noc_index, NCRISC_WR_CMD_BUF));
        WAYPOINT("NWTD");

        constexpr uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC |
                                 NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | 0x0 |  // (linked ? NOC_CMD_VC_LINKED : 0x0)
                                 0x0 |  // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
                                 NOC_CMD_RESP_MARKED;

        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_CTRL, noc_cmd_field);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_TARG_ADDR_LO, src_addr);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_LO, dest_addr);  // (uint32_t)dest_addr
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_COORDINATE, dest_noc_xy);   // dest_addr >> 32
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_AT_LEN_BE, this->page_size);  // len_bytes
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_nonposted_writes_num_issued[noc_index] += 1;
        noc_nonposted_writes_acked[noc_index] += 1;  // num_dests
    }
};

// TODO: add noc_async_write_page
// TODO: need static assert + host assert that page size <= 8192, hard constraint
template <bool DRAM>
struct InterleavedPow2AddrGenFast {
    uint32_t bank_base_address;             // Base address for the whole tensor.
    const uint32_t log_base_2_of_page_size; // Num bytes in bank unit.
    const uint32_t aligned_log_base_2_of_page_size = this->log_base_2_of_page_size > LOG_BASE_2_OF_ALLOCATOR_ALIGNMENT
                                                   ? this->log_base_2_of_page_size
                                                   : LOG_BASE_2_OF_ALLOCATOR_ALIGNMENT;

    FORCE_INLINE
    uint32_t get_addr(const uint32_t id, const uint32_t bank_offset_index, const uint32_t bank_index, const uint32_t offset = 0) const {
        return (bank_offset_index << this->aligned_log_base_2_of_page_size) + this->bank_base_address + offset + interleaved_addr_gen::get_bank_offset<DRAM>(bank_index);
    }

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0) const {

        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index);

        uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
        return noc_addr;
    }

    FORCE_INLINE
    void noc_async_read_page(const uint32_t id, uint32_t dest_addr, const uint32_t offset = 0) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t src_addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t src_noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index);

        WAYPOINT("NRPW");
        DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc_index, get_noc_addr_helper(src_noc_xy, src_addr), dest_addr, 1 << this->aligned_log_base_2_of_page_size);
        while (!noc_cmd_buf_ready(noc_index, NCRISC_RD_CMD_BUF));
        WAYPOINT("NRPD");

        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dest_addr);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, src_addr);      // (uint32_t)src_addr
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_COORDINATE, src_noc_xy);   // src_addr >> 32
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, 1 << this->aligned_log_base_2_of_page_size);  // len_bytes
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_reads_num_issued[noc_index] += 1;
    }

    FORCE_INLINE
    void noc_async_read_partial_page(const uint32_t id, uint32_t dest_addr, const uint32_t size, const uint32_t offset) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t src_addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t src_noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index);

        WAYPOINT("RPW");
        while (!noc_cmd_buf_ready(noc_index, NCRISC_RD_CMD_BUF));
        WAYPOINT("RPD");
        DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc_index, get_noc_addr_helper(src_noc_xy, src_addr), dest_addr, size);

        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dest_addr);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, src_addr);      // (uint32_t)src_addr
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_COORDINATE, src_noc_xy);   // src_addr >> 32
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, size);  // len_bytes
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_reads_num_issued[noc_index] += 1;
    }

    FORCE_INLINE
    void noc_async_write_page(const uint32_t id, uint32_t src_addr, const uint32_t write_size_bytes, const uint32_t offset = 0) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t dest_addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t dest_noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index);

        WAYPOINT("NWPW");
        DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc_index, get_noc_addr_helper(dest_noc_xy, dest_addr), src_addr, write_size_bytes);
        while (!noc_cmd_buf_ready(noc_index, NCRISC_WR_CMD_BUF));
        WAYPOINT("NWPD");

        constexpr uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC |
                                 NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | 0x0 |  // (linked ? NOC_CMD_VC_LINKED : 0x0)
                                 0x0 |  // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
                                 NOC_CMD_RESP_MARKED;

        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_CTRL, noc_cmd_field);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_TARG_ADDR_LO, src_addr);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_LO, dest_addr);  // (uint32_t)dest_addr
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_COORDINATE, dest_noc_xy);   // dest_addr >> 32
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_AT_LEN_BE,  write_size_bytes);  // len_bytes
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_nonposted_writes_num_issued[noc_index] += 1;
        noc_nonposted_writes_acked[noc_index] += 1;  // num_dests
    }
};
