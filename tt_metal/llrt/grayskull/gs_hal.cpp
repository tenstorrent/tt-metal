// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core_config.h"
#include "noc/noc_parameters.h"
#include "llrt/hal.hpp"
#include "tt_metal/third_party/umd/device/tt_soc_descriptor.h"

#if defined (ARCH_GRAYSKULL)

#include "hw/inc/grayskull/dev_mem_map.h"
#include "hw/inc/grayskull/eth_l1_address_map.h" // TODO remove when commonruntimeaddressmap is gone
#include "hostdevcommon/common_runtime_address_map.h"
#include "hw/inc/dev_msgs.h"

#endif

#define GET_MAILBOX_ADDRESS_HOST(x) ((uint64_t) & (((mailboxes_t *)MEM_MAILBOX_BASE)->x))

namespace tt {

namespace tt_metal {

void Hal::initialize_gs() {
#if defined (ARCH_GRAYSKULL)

    static_assert(static_cast<int>(HalProgrammableCoreType::TENSIX) == static_cast<int>(ProgrammableCoreType::TENSIX));

    constexpr uint32_t num_proc_per_tensix_core = 5;
    std::vector<DeviceAddr> mem_map_bases;

    mem_map_bases.resize(utils::underlying_type<HalMemAddrType>(HalMemAddrType::COUNT));
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::BARRIER)] = MEM_L1_BARRIER;
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::LAUNCH)] = GET_MAILBOX_ADDRESS_HOST(launch);
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::WATCHER)] = GET_MAILBOX_ADDRESS_HOST(watcher);
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::DPRINT)] = GET_MAILBOX_ADDRESS_HOST(dprint_buf);
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::PROFILER)] = GET_MAILBOX_ADDRESS_HOST(profiler);
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::KERNEL_CONFIG)] = L1_KERNEL_CONFIG_BASE;
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::UNRESERVED)] = L1_UNRESERVED_BASE;
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::CORE_INFO)] = GET_MAILBOX_ADDRESS_HOST(core_info);
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::GO_MSG)] = GET_MAILBOX_ADDRESS_HOST(go_message);
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = GET_MAILBOX_ADDRESS_HOST(launch_msg_rd_ptr);

    std::vector<uint32_t> mem_map_sizes;
    mem_map_sizes.resize(utils::underlying_type<HalMemAddrType>(HalMemAddrType::COUNT));
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::BARRIER)] = sizeof(uint32_t);
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::LAUNCH)] = sizeof(launch_msg_t);
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::WATCHER)] = sizeof(watcher_msg_t);
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::DPRINT)] = sizeof(dprint_buf_msg_t);
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::PROFILER)] = sizeof(profiler_msg_t);
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::KERNEL_CONFIG)] = L1_KERNEL_CONFIG_SIZE;
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::UNRESERVED)] = MEM_L1_SIZE - L1_UNRESERVED_BASE;
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::GO_MSG)] = sizeof(go_msg_t);
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = sizeof(uint32_t);

    this->core_info_.push_back({HalProgrammableCoreType::TENSIX, CoreType::WORKER, num_proc_per_tensix_core, mem_map_bases, mem_map_sizes, true});

    this->mem_alignments_.resize(utils::underlying_type<HalMemType>(HalMemType::COUNT));
    this->mem_alignments_[utils::underlying_type<HalMemType>(HalMemType::L1)] = L1_ALIGNMENT;
    this->mem_alignments_[utils::underlying_type<HalMemType>(HalMemType::DRAM)] = DRAM_ALIGNMENT;
    this->mem_alignments_[utils::underlying_type<HalMemType>(HalMemType::HOST)] = PCIE_ALIGNMENT;
#endif
}

}  // namespace tt_metal
}  // namespace tt
