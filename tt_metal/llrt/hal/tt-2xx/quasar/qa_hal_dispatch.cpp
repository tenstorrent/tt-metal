// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define HAL_BUILD tt::tt_metal::quasar::dispatch
#include "hostdev/dev_msgs.h"
#include "hostdev/fabric_telemetry_msgs.h"
#include "hostdev/realtime_profiler_msgs.h"
using namespace tt::tt_metal::quasar::dispatch;

#include <cstdint>

#include "quasar/qa_hal.hpp"
#include "quasar/qa_hal_dispatch_asserts.hpp"
#include "dev_mem_map.h"
#include "hal_types.hpp"
#include "llrt/hal.hpp"
#include "noc/noc_parameters.h"
#include "tensix.h"
#include "tt_align.hpp"
#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal::quasar {

namespace dispatch_dev_msgs {
#include "hal/generated/dev_msgs_impl.hpp"
}

namespace dispatch_fabric_telemetry {
#include "hal/generated/fabric_telemetry_impl.hpp"
}

namespace dispatch_realtime_profiler_msgs {
#include "hal/generated/realtime_profiler_msgs_impl.hpp"
}

HalCoreInfoType create_dispatch_mem_map() {
    auto tensix_mem_map = create_tensix_mem_map();
    uint32_t max_alignment = std::max(DRAM_ALIGNMENT, L1_ALIGNMENT);

    static constexpr DeviceAddr dispatch_dm_kernel_bases[] = {
        MEM_DISPATCH_DM0_KERNEL_BASE,
        MEM_DISPATCH_DM1_KERNEL_BASE,
        MEM_DISPATCH_DM2_KERNEL_BASE,
        MEM_DISPATCH_DM3_KERNEL_BASE,
        MEM_DISPATCH_DM4_KERNEL_BASE,
        MEM_DISPATCH_DM5_KERNEL_BASE,
        MEM_DISPATCH_DM6_KERNEL_BASE,
        MEM_DISPATCH_DM7_KERNEL_BASE,
    };
    static_assert(std::size(dispatch_dm_kernel_bases) == NUM_DM_CORES);

    std::vector<std::vector<HalJitBuildConfig>> processor_classes(1);
    processor_classes[0].reserve(NUM_DM_CORES);
    for (unsigned long dispatch_dm_kernel_base : dispatch_dm_kernel_bases) {
        processor_classes[0].push_back({
            .fw_base_addr = dispatch_dm_kernel_base,
            .local_init_addr = UINT32_MAX,
            .fw_launch_addr = 0x0,
            // DM firmware is linked/loaded at MEM_DM_FIRMWARE_BASE (main.ld); per-DM fw_base_addr is the
            // cq-kernel link/load slot only. Reset still boots via JAL from L1[0] into firmware.
            .fw_launch_addr_value = generate_risc_startup_addr(MEM_DM_FIRMWARE_BASE),
            .memory_load = ll_api::memory::Loading::CONTIGUOUS,
        });
    }

    std::vector<std::vector<std::pair<std::string, std::string>>> processor_classes_names = {
        {{"DM0", "DM0"},
         {"DM1", "DM1"},
         {"DM2", "DM2"},
         {"DM3", "DM3"},
         {"DM4", "DM4"},
         {"DM5", "DM5"},
         {"DM6", "DM6"},
         {"DM7", "DM7"}},
    };

    auto mem_map_bases = tensix_mem_map.mem_map_bases();
    auto mem_map_sizes = tensix_mem_map.mem_map_sizes();
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] = MEM_DISPATCH_KERNEL_CONFIG_SIZE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::DEFAULT_UNRESERVED)] =
        tt::align(DISPATCH_MEM_MAP_END, max_alignment);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::DEFAULT_UNRESERVED)] =
        MEM_L1_SIZE - mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::DEFAULT_UNRESERVED)];

    return HalCoreInfoType(
        HalProgrammableCoreType::DISPATCH,
        CoreType::DISPATCH,
        std::move(processor_classes),
        std::vector<uint8_t>{1},
        std::move(mem_map_bases),
        std::move(mem_map_sizes),
        tensix_mem_map.eth_fw_mailbox_msgs(),
        std::move(processor_classes_names),
        true,
        true,
        false,
        dispatch_dev_msgs::create_factory(),
        dispatch_fabric_telemetry::create_factory(),
        dispatch_realtime_profiler_msgs::create_factory());
}

}  // namespace tt::tt_metal::quasar
