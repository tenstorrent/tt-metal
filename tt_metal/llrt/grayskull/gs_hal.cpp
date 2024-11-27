// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "core_config.h"  // ProgrammableCoreType
#include "dev_mem_map.h"
#include "dev_msgs.h"
#include "noc/noc_parameters.h"

#include "hal.hpp"

#include "hal_asserts.hpp"

// FIXME: Eventually this file will be gone
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"  // L1_KERNEL_CONFIG_BASE

#include "tt_soc_descriptor.h"  // CoreType

#define GET_MAILBOX_ADDRESS_HOST(x) ((uint64_t)&(((mailboxes_t*)MEM_MAILBOX_BASE)->x))

// Reserved DRAM addresses
// Host writes (4B value) to and reads from DRAM_BARRIER_BASE across all channels to ensure previous writes have been
// committed
constexpr static std::uint32_t DRAM_BARRIER_BASE = 0;
constexpr static std::uint32_t DRAM_BARRIER_SIZE =
    ((sizeof(uint32_t) + DRAM_ALIGNMENT - 1) / DRAM_ALIGNMENT) * DRAM_ALIGNMENT;

namespace tt {

namespace tt_metal {

void Hal::initialize_gs() {
    static_assert(static_cast<int>(HalProgrammableCoreType::TENSIX) == static_cast<int>(ProgrammableCoreType::TENSIX));

    uint32_t max_alignment = std::max(DRAM_ALIGNMENT, L1_ALIGNMENT);
    std::vector<DeviceAddr> mem_map_bases;

    mem_map_bases.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT));
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = MEM_L1_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = MEM_L1_BARRIER;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_MAILBOX_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = GET_MAILBOX_ADDRESS_HOST(launch);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = GET_MAILBOX_ADDRESS_HOST(watcher);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::DPRINT)] = GET_MAILBOX_ADDRESS_HOST(dprint_buf);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = GET_MAILBOX_ADDRESS_HOST(profiler);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] = L1_KERNEL_CONFIG_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)] =
        ((L1_KERNEL_CONFIG_BASE + L1_KERNEL_CONFIG_SIZE - 1) | (max_alignment - 1)) + 1;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::CORE_INFO)] = GET_MAILBOX_ADDRESS_HOST(core_info);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = GET_MAILBOX_ADDRESS_HOST(go_message);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] =
        GET_MAILBOX_ADDRESS_HOST(launch_msg_rd_ptr);

    std::vector<uint32_t> mem_map_sizes;
    mem_map_sizes.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT));
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = MEM_L1_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = sizeof(uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_MAILBOX_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = sizeof(launch_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = sizeof(watcher_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::DPRINT)] = sizeof(dprint_buf_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = sizeof(profiler_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] = L1_KERNEL_CONFIG_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)] =
        MEM_L1_SIZE - mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)];
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = sizeof(go_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = sizeof(uint32_t);

    std::vector<std::vector<HalJitBuildConfig>> processor_classes(NumTensixDispatchClasses);
    std::vector<HalJitBuildConfig> processor_types;
    for (uint8_t processor_class_idx = 0; processor_class_idx < NumTensixDispatchClasses; processor_class_idx++) {
        uint32_t num_processors = processor_class_idx == (NumTensixDispatchClasses - 1) ? 3 : 1;
        processor_types.resize(num_processors);
        for (uint8_t processor_type_idx = 0; processor_type_idx < processor_types.size(); processor_type_idx++) {
            DeviceAddr fw_base, local_init;
            switch (processor_class_idx) {
                case 0: {
                    fw_base = MEM_BRISC_FIRMWARE_BASE;
                    local_init = MEM_BRISC_INIT_LOCAL_L1_BASE_SCRATCH;
                } break;
                case 1: {
                    fw_base = MEM_NCRISC_FIRMWARE_BASE;
                    local_init = MEM_NCRISC_INIT_LOCAL_L1_BASE_SCRATCH;
                } break;
                case 2: {
                    switch (processor_type_idx) {
                        case 0: {
                            fw_base = MEM_TRISC0_FIRMWARE_BASE;
                            local_init = MEM_TRISC0_INIT_LOCAL_L1_BASE_SCRATCH;
                        } break;
                        case 1: {
                            fw_base = MEM_TRISC1_FIRMWARE_BASE;
                            local_init = MEM_TRISC1_INIT_LOCAL_L1_BASE_SCRATCH;
                        } break;
                        case 2: {
                            fw_base = MEM_TRISC2_FIRMWARE_BASE;
                            local_init = MEM_TRISC2_INIT_LOCAL_L1_BASE_SCRATCH;
                        } break;
                    }
                } break;
                default: TT_THROW("Unexpected processor class {} for Blackhole Tensix", processor_class_idx);
            }

            processor_types[processor_type_idx] =
                HalJitBuildConfig{.fw_base_addr = fw_base, .local_init_addr = local_init};
        }
        processor_classes[processor_class_idx] = processor_types;
    }

    this->core_info_.push_back(
        {HalProgrammableCoreType::TENSIX, CoreType::WORKER, processor_classes, mem_map_bases, mem_map_sizes, true});

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
        } else if ((addr & MEM_NCRISC_IRAM_BASE) == MEM_NCRISC_IRAM_BASE) {
            // Move addresses in the NCRISC memory range to l1 (copied by kernel)
            return (addr & ~MEM_NCRISC_IRAM_BASE) + MEM_NCRISC_INIT_IRAM_L1_BASE;
        }

        // No relocation needed
        return addr;
    };
}

}  // namespace tt_metal
}  // namespace tt
