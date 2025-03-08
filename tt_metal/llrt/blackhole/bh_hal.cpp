// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include "core_config.h"  // ProgrammableCoreType
#include "dev_mem_map.h"
#include <dev_msgs.h>
#include "noc/noc_parameters.h"
#include "noc/noc_overlay_parameters.h"
#include "tensix.h"

#include "hal.hpp"
#include "blackhole/bh_hal.hpp"

// Reserved DRAM addresses
// Host writes (4B value) to and reads from DRAM_BARRIER_BASE across all channels to ensure previous writes have been
// committed
constexpr static std::uint32_t DRAM_BARRIER_BASE = 0;
constexpr static std::uint32_t DRAM_BARRIER_SIZE =
    ((sizeof(uint32_t) + DRAM_ALIGNMENT - 1) / DRAM_ALIGNMENT) * DRAM_ALIGNMENT;

static constexpr float EPS_BH = 1.19209e-7f;
static constexpr float NAN_BH = 7.0040e+19;
static constexpr float INF_BH = 1.7014e+38;

namespace tt {

namespace tt_metal {

void Hal::initialize_bh() {
    static_assert(static_cast<int>(HalProgrammableCoreType::TENSIX) == static_cast<int>(ProgrammableCoreType::TENSIX));
    static_assert(
        static_cast<int>(HalProgrammableCoreType::ACTIVE_ETH) == static_cast<int>(ProgrammableCoreType::ACTIVE_ETH));
    static_assert(
        static_cast<int>(HalProgrammableCoreType::IDLE_ETH) == static_cast<int>(ProgrammableCoreType::IDLE_ETH));

    HalCoreInfoType tensix_mem_map = blackhole::create_tensix_mem_map();
    this->core_info_.push_back(tensix_mem_map);

    HalCoreInfoType active_eth_mem_map = blackhole::create_active_eth_mem_map();
    this->core_info_.push_back(active_eth_mem_map);

    HalCoreInfoType idle_eth_mem_map = blackhole::create_idle_eth_mem_map();
    this->core_info_.push_back(idle_eth_mem_map);

    this->dram_bases_.resize(static_cast<std::size_t>(HalDramMemAddrType::COUNT));
    this->dram_sizes_.resize(static_cast<std::size_t>(HalDramMemAddrType::COUNT));
    this->dram_bases_[static_cast<std::size_t>(HalDramMemAddrType::DRAM_BARRIER)] = DRAM_BARRIER_BASE;
    this->dram_sizes_[static_cast<std::size_t>(HalDramMemAddrType::DRAM_BARRIER)] = DRAM_BARRIER_SIZE;

    this->mem_alignments_.resize(static_cast<std::size_t>(HalMemType::COUNT));
    this->mem_alignments_[static_cast<std::size_t>(HalMemType::L1)] = L1_ALIGNMENT;
    this->mem_alignments_[static_cast<std::size_t>(HalMemType::DRAM)] = DRAM_ALIGNMENT;
    this->mem_alignments_[static_cast<std::size_t>(HalMemType::HOST)] = PCIE_ALIGNMENT;

    this->relocate_func_ = [](uint64_t addr, uint64_t local_init_addr) {
        if ((addr & MEM_LOCAL_BASE) == MEM_LOCAL_BASE) {
            // Move addresses in the local memory range to l1 (copied by kernel)
            return (addr & ~MEM_LOCAL_BASE) + local_init_addr;
        }

        // Note: Blackhole does not have IRAM

        // No relocation needed
        return addr;
    };

    this->erisc_iram_relocate_func_ = [](uint64_t addr) { return addr; };

    this->valid_reg_addr_func_ = [](uint32_t addr) {
        return (
            ((addr >= NOC_OVERLAY_START_ADDR) &&
             (addr < NOC_OVERLAY_START_ADDR + NOC_STREAM_REG_SPACE_SIZE * NOC_NUM_STREAMS)) ||
            ((addr >= NOC0_REGS_START_ADDR) && (addr < NOC0_REGS_START_ADDR + 0x1000)) ||
            ((addr >= NOC1_REGS_START_ADDR) && (addr < NOC1_REGS_START_ADDR + 0x1000)) ||
            (addr == RISCV_DEBUG_REG_SOFT_RESET_0) ||
            (addr == IERISC_RESET_PC || addr == SLAVE_IERISC_RESET_PC));  // used to program start addr for eth FW
    };

    this->noc_xy_encoding_func_ = [](uint32_t x, uint32_t y) { return NOC_XY_ENCODING(x, y); };
    this->noc_multicast_encoding_func_ = [](uint32_t x_start, uint32_t y_start, uint32_t x_end, uint32_t y_end) {
        return NOC_MULTICAST_ENCODING(x_start, y_start, x_end, y_end);
    };
    this->noc_mcast_addr_start_x_func_ = [](uint64_t addr) -> uint64_t { return NOC_MCAST_ADDR_START_X(addr); };
    this->noc_mcast_addr_start_y_func_ = [](uint64_t addr) -> uint64_t { return NOC_MCAST_ADDR_START_Y(addr); };
    this->noc_mcast_addr_end_x_func_ = [](uint64_t addr) -> uint64_t { return NOC_MCAST_ADDR_END_X(addr); };
    this->noc_mcast_addr_end_y_func_ = [](uint64_t addr) -> uint64_t { return NOC_MCAST_ADDR_END_Y(addr); };
    this->noc_ucast_addr_x_func_ = [](uint64_t addr) -> uint64_t { return NOC_UNICAST_ADDR_X(addr); };
    this->noc_ucast_addr_y_func_ = [](uint64_t addr) -> uint64_t { return NOC_UNICAST_ADDR_Y(addr); };
    this->noc_local_addr_func_ = [](uint64_t addr) -> uint64_t { return NOC_LOCAL_ADDR(addr); };

    this->stack_size_func_ = [](uint32_t type) -> uint32_t {
        switch (type) {
            case DebugBrisc: return MEM_BRISC_STACK_SIZE;
            case DebugNCrisc: return MEM_NCRISC_STACK_SIZE;
            case DebugErisc: return 0;  // Not managed/checked by us.
            case DebugIErisc: return MEM_IERISC_STACK_SIZE;
            case DebugSlaveIErisc: return MEM_BRISC_STACK_SIZE;
            case DebugTrisc0: return MEM_TRISC0_STACK_SIZE;
            case DebugTrisc1: return MEM_TRISC1_STACK_SIZE;
            case DebugTrisc2: return MEM_TRISC2_STACK_SIZE;
        }
        return 0xdeadbeef;
    };

    this->num_nocs_ = NUM_NOCS;
    this->noc_addr_node_id_bits_ = NOC_ADDR_NODE_ID_BITS;
    this->noc_coord_reg_offset_ = NOC_COORD_REG_OFFSET;
    this->noc_overlay_start_addr_ = NOC_OVERLAY_START_ADDR;
    this->noc_stream_reg_space_size_ = NOC_STREAM_REG_SPACE_SIZE;
    this->noc_stream_remote_dest_buf_size_reg_index_ = STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX;
    this->noc_stream_remote_dest_buf_start_reg_index_ = STREAM_REMOTE_DEST_BUF_START_REG_INDEX;
    this->noc_stream_remote_dest_buf_space_available_reg_index_ = STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX;
    this->noc_stream_remote_dest_buf_space_available_update_reg_index_ =
        STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX;
    this->coordinate_virtualization_enabled_ = COORDINATE_VIRTUALIZATION_ENABLED;
    this->virtual_worker_start_x_ = VIRTUAL_TENSIX_START_X;
    this->virtual_worker_start_y_ = VIRTUAL_TENSIX_START_Y;
    this->eth_fw_is_cooperative_ = false;

    this->eps_ = EPS_BH;
    this->nan_ = NAN_BH;
    this->inf_ = INF_BH;
}

}  // namespace tt_metal
}  // namespace tt
