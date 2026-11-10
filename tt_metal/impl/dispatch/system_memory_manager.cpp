// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include "impl/context/metal_context.hpp"
#include <internal/l1_ecc_halt.hpp>
#include "impl/context/metal_env_impl.hpp"
#include "system_memory_manager.hpp"
#include <tt-metalium/math.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <string>

#include <tt_stl/assert.hpp>
#include "core_coord.hpp"
#include "dispatch_settings.hpp"
#include "hal_types.hpp"
#include "memcpy.hpp"
#include "allocator/allocator.hpp"
#include "command_queue_common.hpp"
#include "system_memory_cq_interface.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-umd/types/cluster_descriptor_types.hpp>
#include <tt-umd/types/xy_pair.hpp>
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"
#include <tt-umd/types/core_coordinates.hpp>
#include <impl/dispatch/dispatch_core_manager.hpp>
#include "impl/dispatch/kernels/cq_prefetch.hpp"
#include <llrt/tt_cluster.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>
#include <impl/device/device_manager.hpp>

namespace tt::tt_metal {

SystemMemoryManager::SystemMemoryManager(MetalEnvImpl& env, MetalContext& ctx, ChipId device_id, uint8_t num_hw_cqs) :
    env_(env), ctx_(ctx), device_id_(device_id), backing_(CqBacking::create(env, ctx, device_id, num_hw_cqs)) {
    const std::vector<SystemMemoryCQInterface> cq_interfaces = backing_->make_cq_interfaces(num_hw_cqs);
    cqs_.reserve(num_hw_cqs);
    for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
        cqs_.push_back(std::make_unique<SystemMemoryCq>(*backing_, cq_id, cq_interfaces[cq_id]));
    }
}

uint32_t SystemMemoryManager::get_dram_region_base_addr() const {
    TT_FATAL(this->is_dram_backed(), "CQs are not DRAM backed");
    return env_.get_hal().get_dev_addr(HalDramMemAddrType::DRAM_BACKED_COMMAND_QUEUES);
}

uint32_t SystemMemoryManager::get_dram_region_bank_id() const {
    TT_FATAL(this->is_dram_backed(), "CQs are not DRAM backed");
    return 0;
}

}  // namespace tt::tt_metal
