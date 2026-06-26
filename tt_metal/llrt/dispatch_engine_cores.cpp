// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/dispatch/dispatch_engine_cores.hpp>

#include <functional>
#include <unordered_map>

#include "common/core_coord.hpp"
#include "core_descriptor.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_accessor.hpp"
#include "impl/dispatch/dispatch_core_common.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <umd/device/types/arch.hpp>
#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal::internal {

std::vector<CoreCoord> get_quasar_soc_dispatch_engine_logical_cores(const metal_SocDescriptor& soc_desc) {
    const auto dispatch_noc0_cores = soc_desc.get_cores(tt::CoreType::DISPATCH, tt::CoordSystem::NOC0);
    std::vector<CoreCoord> logical_cores;
    logical_cores.reserve(dispatch_noc0_cores.size());
    for (size_t index = 0; index < dispatch_noc0_cores.size(); ++index) {
        logical_cores.emplace_back(static_cast<uint32_t>(index), 0);
    }
    return logical_cores;
}

namespace {

std::vector<CoreCoord> get_quasar_tensix_fallback_dispatch_cores_from_yaml(
    tt::tt_metal::MetalEnvImpl& env,
    ChipId device_id,
    uint8_t num_hw_cqs,
    const tt_metal::DispatchCoreConfig& dispatch_core_config) {
    const core_descriptor_t& core_desc =
        get_core_descriptor_config(env, device_id, num_hw_cqs, dispatch_core_config);
    if (!core_desc.relative_dispatch_cores.empty()) {
        const CoreCoord grid_size = env.get_cluster().get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
        std::vector<CoreCoord> logical_cores;
        logical_cores.reserve(core_desc.relative_dispatch_cores.size());
        for (const tt_metal::RelativeCoreCoord& rel_coord : core_desc.relative_dispatch_cores) {
            logical_cores.push_back(get_core_coord_from_relative(rel_coord, grid_size));
        }
        return logical_cores;
    }
    return core_desc.logical_dispatch_cores;
}

struct QuasarDispatchCoresCacheKey {
    ChipId device_id;
    uint8_t num_hw_cqs;
    tt_metal::DispatchCoreConfig dispatch_core_config;
    bool use_tensix_fallback;

    bool operator==(const QuasarDispatchCoresCacheKey& other) const {
        return device_id == other.device_id && num_hw_cqs == other.num_hw_cqs &&
               dispatch_core_config == other.dispatch_core_config &&
               use_tensix_fallback == other.use_tensix_fallback;
    }
};

struct QuasarDispatchCoresCacheKeyHash {
    std::size_t operator()(const QuasarDispatchCoresCacheKey& key) const {
        return std::hash<tt_metal::DispatchCoreConfig>{}(key.dispatch_core_config) ^
               (static_cast<std::size_t>(key.device_id) << 1) ^ (static_cast<std::size_t>(key.num_hw_cqs) << 17) ^
               (static_cast<std::size_t>(key.use_tensix_fallback) << 25);
    }
};

const std::vector<CoreCoord>& get_quasar_dispatch_cores_cached(
    tt::tt_metal::MetalEnvImpl& env,
    ChipId device_id,
    uint8_t num_hw_cqs,
    const tt_metal::DispatchCoreConfig& dispatch_core_config) {
    static std::unordered_map<QuasarDispatchCoresCacheKey, std::vector<CoreCoord>, QuasarDispatchCoresCacheKeyHash>
        cache;

    const bool use_tensix_fallback = env.get_rtoptions().get_use_quasar_tensix_dispatch_cores();
    const QuasarDispatchCoresCacheKey key{
        .device_id = device_id,
        .num_hw_cqs = num_hw_cqs,
        .dispatch_core_config = dispatch_core_config,
        .use_tensix_fallback = use_tensix_fallback,
    };

    if (cache.contains(key)) {
        return cache.at(key);
    }

    std::vector<CoreCoord> logical_cores;
    if (use_tensix_fallback) {
        logical_cores =
            get_quasar_tensix_fallback_dispatch_cores_from_yaml(env, device_id, num_hw_cqs, dispatch_core_config);
    } else {
        logical_cores = get_quasar_soc_dispatch_engine_logical_cores(env.get_cluster().get_soc_desc(device_id));
    }

    cache.emplace(key, std::move(logical_cores));
    return cache.at(key);
}

}  // namespace

const std::vector<CoreCoord>& get_quasar_dispatch_cores(
    tt::tt_metal::MetalEnvImpl& env,
    ChipId device_id,
    uint8_t num_hw_cqs,
    const tt_metal::DispatchCoreConfig& dispatch_core_config) {
    TT_FATAL(
        env.get_cluster().arch() == tt::ARCH::QUASAR,
        "get_quasar_dispatch_cores is only valid on Quasar (device {})",
        device_id);
    return get_quasar_dispatch_cores_cached(env, device_id, num_hw_cqs, dispatch_core_config);
}

void validate_quasar_dispatch_cores_for_fd(
    tt::tt_metal::MetalEnvImpl& env,
    ChipId device_id,
    uint8_t num_hw_cqs,
    const tt_metal::DispatchCoreConfig& dispatch_core_config) {
    if (env.get_cluster().arch() != tt::ARCH::QUASAR || !env.get_rtoptions().get_fast_dispatch()) {
        return;
    }

    const auto& dispatch_cores = get_quasar_dispatch_cores(env, device_id, num_hw_cqs, dispatch_core_config);
    if (!dispatch_cores.empty()) {
        return;
    }

    if (env.get_rtoptions().get_use_quasar_tensix_dispatch_cores()) {
        TT_THROW(
            "Quasar fast dispatch requires non-empty dispatch_cores in the core descriptor YAML when "
            "TT_METAL_TENSIX_DISPATCH_CORES=1 (device {})",
            device_id);
    }

    TT_THROW(
        "Quasar fast dispatch requires dispatch-engine cores in the soc descriptor (dispatch: list). "
        "Set TT_METAL_TENSIX_DISPATCH_CORES=1 to use interim Tensix dispatch cores from core descriptor YAML "
        "(device {})",
        device_id);
}

CoreType resolve_dispatch_core_type(
    tt::ARCH arch,
    const tt_metal::DispatchCoreConfig& dispatch_core_config,
    const metal_SocDescriptor& soc_desc,
    bool use_quasar_tensix_dispatch_cores) {
    if (arch != tt::ARCH::QUASAR) {
        return tt::tt_metal::get_core_type_from_config(dispatch_core_config);
    }
    if (use_quasar_tensix_dispatch_cores) {
        return CoreType::WORKER;
    }
    if (soc_desc.get_num_dispatch_engine_cores() > 0) {
        return CoreType::DISPATCH;
    }
    return CoreType::WORKER;
}

CoreType resolve_dispatch_core_type(
    tt::tt_metal::MetalEnvImpl& env,
    ChipId device_id,
    const tt_metal::DispatchCoreConfig& dispatch_core_config) {
    return resolve_dispatch_core_type(
        env.get_cluster().arch(),
        dispatch_core_config,
        env.get_cluster().get_soc_desc(device_id),
        env.get_rtoptions().get_use_quasar_tensix_dispatch_cores());
}

namespace {

const std::vector<CoreCoord>& get_sd_cq_dispatch_cores(const tt::tt_metal::IDevice* device) {
    auto& env = MetalEnvAccessor(MetalContext::instance().get_env()).impl();
    const auto& dispatch_core_config = MetalContext::instance().get_dispatch_core_config();
    return get_quasar_dispatch_cores(env, device->id(), device->num_hw_cqs(), dispatch_core_config);
}

}  // namespace

CoreType resolve_sd_cq_kernel_core_type(const tt::tt_metal::IDevice* device) {
    auto& env = MetalEnvAccessor(MetalContext::instance().get_env()).impl();
    const auto& dispatch_core_config = MetalContext::instance().get_dispatch_core_config();
    return tt::tt_metal::resolve_dispatch_core_type(env, device->id(), dispatch_core_config);
}

CoreCoord dispatch_engine_core(const tt::tt_metal::IDevice* device, uint32_t index) {
    TT_FATAL(device->arch() == tt::ARCH::QUASAR, "dispatch_engine_core is only valid on Quasar");
    const auto& cores = get_sd_cq_dispatch_cores(device);
    TT_FATAL(
        index < cores.size(),
        "dispatch_engine_core index {} out of range ({} dispatch cores on device {})",
        index,
        cores.size(),
        device->id());
    return cores[index];
}

CoreCoord dispatch_engine_virtual_core(const tt::tt_metal::IDevice* device, uint32_t index) {
    const CoreCoord logical_core = dispatch_engine_core(device, index);
    return device->virtual_core_from_logical_core(logical_core, CoreType::DISPATCH);
}

CoreCoord sd_cq_prefetch_core(const tt::tt_metal::IDevice* device) {
    // Only the Quasar dispatch-engine path uses synthetic dispatch cores. WH/BH and the Quasar
    // interim Tensix path (TT_METAL_TENSIX_DISPATCH_CORES=1, resolves to WORKER) keep the legacy
    // hardcoded worker logical core.
    if (resolve_sd_cq_kernel_core_type(device) == CoreType::DISPATCH) {
        return dispatch_engine_core(device, 0);
    }
    return CoreCoord{0, 0};
}

CoreCoord sd_cq_dispatch_core(const tt::tt_metal::IDevice* device) {
    if (resolve_sd_cq_kernel_core_type(device) == CoreType::DISPATCH) {
        return dispatch_engine_core(device, 0);
    }
    // Legacy interim placement: Quasar shares core {0,0} with prefetch; WH/BH uses a separate core.
    return (device->arch() == tt::ARCH::QUASAR) ? CoreCoord{0, 0} : CoreCoord{4, 0};
}

CoreCoord sd_cq_virtual_core(const tt::tt_metal::IDevice* device, const CoreCoord& logical_core) {
    return device->virtual_core_from_logical_core(logical_core, resolve_sd_cq_kernel_core_type(device));
}

bool sd_cq_kernel_tests_should_skip(const tt::tt_metal::IDevice* device) {
    if (device->arch() != tt::ARCH::QUASAR) {
        return false;
    }
    auto& env = MetalEnvAccessor(MetalContext::instance().get_env()).impl();
    if (env.get_rtoptions().get_use_quasar_tensix_dispatch_cores()) {
        return false;
    }
    return get_sd_cq_dispatch_cores(device).empty();
}

uint32_t fd_core_type_define_value(const tt::tt_metal::IDevice* device) {
    if (resolve_sd_cq_kernel_core_type(device) == CoreType::DISPATCH) {
        return static_cast<uint32_t>(HalProgrammableCoreType::DISPATCH);
    }
    return static_cast<uint32_t>(HalProgrammableCoreType::TENSIX);
}

DataMovementProcessor prefetch_dm_processor() { return DataMovementProcessor::RISCV_0; }

DataMovementProcessor dispatch_dm_processor() { return DataMovementProcessor::RISCV_1; }

}  // namespace tt::tt_metal::internal
