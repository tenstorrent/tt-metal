// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/per_core_allocation/tt_metal.hpp>
#include "context/metal_context.hpp"
#include "device/device_manager.hpp"
#include "tracy/Tracy.hpp"

namespace tt::tt_metal::experimental::per_core_allocation {

std::map<ChipId, IDevice*> CreateDevices(
    const std::vector<ChipId>& device_ids,
    uint8_t num_hw_cqs,
    size_t l1_small_size,
    size_t trace_region_size,
    const DispatchCoreConfig& dispatch_core_config,
    const std::vector<uint32_t>& /*l1_bank_remap*/,
    size_t worker_l1_size,
    bool init_profiler,
    bool initialize_fabric_and_dispatch_fw,
    AllocatorMode allocator_mode) {
    ZoneScoped;
    bool is_galaxy = MetalContext::instance().get_cluster().is_galaxy_cluster();
    MetalContext::instance().initialize_device_manager(
        device_ids,
        num_hw_cqs,
        l1_small_size,
        trace_region_size,
        dispatch_core_config,
        {},
        worker_l1_size,
        init_profiler,
        initialize_fabric_and_dispatch_fw,
        allocator_mode);

    const auto devices = MetalContext::instance().device_manager()->get_all_active_devices();
    std::map<ChipId, IDevice*> ret_devices;
    for (IDevice* dev : devices) {
        if (is_galaxy and dev->is_mmio_capable()) {
            continue;
        }
        ret_devices.insert({dev->id(), dev});
    }

    return ret_devices;
}

}  // namespace tt::tt_metal::experimental::per_core_allocation
