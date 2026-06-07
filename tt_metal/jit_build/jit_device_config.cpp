// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_device_config.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>

#include <tt-logger/tt-logger.hpp>

#include "context/metal_env_accessor.hpp"
#include "core_descriptor.hpp"
#include "dispatch_core_common.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "jit_build/build_env_manager.hpp"
#include "jit_build/jit_build_fingerprint.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_target_device.hpp"
#include "impl/profiler/profiler_state_manager.hpp"
#include "llrt/tt_cluster.hpp"

#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal {

std::string JitBuildFingerprint::serialize() const {
    std::ostringstream os;
    os << "num_l1_banks=" << num_l1_banks << ";dispatch_core_type=" << dispatch_core_type
       << ";dispatch_core_axis=" << dispatch_core_axis << ";enable_2_erisc_mode=" << (enable_2_erisc_mode ? 1 : 0);
    return os.str();
}

std::optional<JitBuildFingerprint> JitBuildFingerprint::deserialize(std::string_view text) {
    JitBuildFingerprint fp;
    bool any = false;
    size_t start = 0;
    while (start <= text.size()) {
        size_t semi = text.find(';', start);
        std::string_view tok =
            text.substr(start, semi == std::string_view::npos ? std::string_view::npos : semi - start);
        size_t eq = tok.find('=');
        if (eq != std::string_view::npos) {
            std::string key(tok.substr(0, eq));
            std::string val(tok.substr(eq + 1));
            try {
                if (key == "num_l1_banks") {
                    fp.num_l1_banks = static_cast<uint32_t>(std::stoul(val));
                    any = true;
                } else if (key == "dispatch_core_type") {
                    fp.dispatch_core_type = static_cast<uint32_t>(std::stoul(val));
                    any = true;
                } else if (key == "dispatch_core_axis") {
                    fp.dispatch_core_axis = static_cast<uint32_t>(std::stoul(val));
                    any = true;
                } else if (key == "enable_2_erisc_mode") {
                    fp.enable_2_erisc_mode = (std::stoi(val) != 0);
                    any = true;
                }
            } catch (...) {
                return std::nullopt;
            }
        }
        if (semi == std::string_view::npos) {
            break;
        }
        start = semi + 1;
    }
    if (!any) {
        return std::nullopt;
    }
    return fp;
}

void capture_jit_build_fingerprint(const std::string& path) {
    const auto& cfg = BuildEnvManager::get_instance().get_device_build_env(0).dev_config;
    JitBuildFingerprint fp;
    fp.num_l1_banks = static_cast<uint32_t>(cfg.num_l1_banks);
    fp.dispatch_core_type = static_cast<uint32_t>(cfg.dispatch_core_type);
    fp.dispatch_core_axis = static_cast<uint32_t>(cfg.dispatch_core_axis);
    fp.enable_2_erisc_mode = MetalContext::instance().rtoptions().get_enable_2_erisc_mode();
    std::ofstream(path) << fp.serialize();
    log_info(tt::LogBuildKernels, "Captured JIT build fingerprint to {}: {}", path, fp.serialize());
}

const std::optional<JitBuildFingerprint>& active_jit_build_fingerprint() {
    static const std::optional<JitBuildFingerprint> cached = []() -> std::optional<JitBuildFingerprint> {
        const char* p = std::getenv("TT_METAL_JIT_BUILD_FINGERPRINT");
        if (p == nullptr || *p == '\0') {
            return std::nullopt;
        }
        std::ifstream f(p);
        if (!f) {
            log_warning(tt::LogBuildKernels, "TT_METAL_JIT_BUILD_FINGERPRINT={} not readable -> ignored", p);
            return std::nullopt;
        }
        std::stringstream ss;
        ss << f.rdbuf();
        auto fp = JitBuildFingerprint::deserialize(ss.str());
        if (fp) {
            log_info(tt::LogBuildKernels, "Loaded JIT build fingerprint from {}: {}", p, fp->serialize());
        } else {
            log_warning(tt::LogBuildKernels, "TT_METAL_JIT_BUILD_FINGERPRINT={} unparseable -> ignored", p);
        }
        return fp;
    }();
    return cached;
}

JitDeviceConfig create_jit_device_config(ChipId device_id, uint8_t num_hw_cqs, ContextId context_id) {
    // Need both runtime state and hardware query
    auto& ctx = MetalContext::instance(context_id);
    auto& env = MetalEnvAccessor(ctx.get_env()).impl();
    const auto& hal = env.get_hal();
    const auto& cluster = env.get_cluster();
    const auto& dispatch_core_config = ctx.get_dispatch_core_manager().get_dispatch_core_config();
    const metal_SocDescriptor& soc_d = cluster.get_soc_desc(device_id);

    const size_t num_dram_banks = static_cast<size_t>(soc_d.get_num_dram_views());

    // Up-front precompile escape hatch (mock/sim only): a hardware-free build runs slow dispatch,
    // which resolves some values differently from the real fast-dispatch run. When the precompile
    // runner has captured the real device's fingerprint and pointed us at it (env
    // TT_METAL_JIT_BUILD_FINGERPRINT), replay those values here -- this single chokepoint feeds BOTH
    // the build_key and the kernel defines -- so the warm cache is keyed identically to the real run.
    // Never applied on silicon. See jit_build_fingerprint.hpp.
    const auto& fp = active_jit_build_fingerprint();
    const bool use_fp = fp.has_value() && ctx.rtoptions().get_target_device() != tt::TargetDevice::Silicon;

    // # of L1 banks needs to match allocator. For L1BankingAllocator this is the # of storage cores. TODO: when
    // allocator is pulled out of device, use it to get that info here.
    const size_t num_l1_banks =
        use_fp ? static_cast<size_t>(fp->num_l1_banks)
               : get_logical_compute_cores(env, device_id, num_hw_cqs, dispatch_core_config).size();
    const DispatchCoreType dispatch_core_type =
        use_fp ? static_cast<DispatchCoreType>(fp->dispatch_core_type) : dispatch_core_config.get_dispatch_core_type();
    const DispatchCoreAxis dispatch_core_axis =
        use_fp ? static_cast<DispatchCoreAxis>(fp->dispatch_core_axis) : dispatch_core_config.get_dispatch_core_axis();

    auto pcie_cores = soc_d.get_cores(CoreType::PCIE, CoordSystem::TRANSLATED);
    CoreCoord pcie_core = pcie_cores.empty() ? soc_d.grid_size : pcie_cores[0];

    return {
        .hal = &hal,
        .arch = cluster.arch(),
        .num_dram_banks = num_dram_banks,
        .num_l1_banks = num_l1_banks,
        .pcie_core = pcie_core,
        .harvesting_mask = cluster.get_harvesting_mask(device_id),
        .dispatch_core_type = dispatch_core_type,
        .dispatch_core_axis = dispatch_core_axis,
        .coordinate_virtualization_enabled = hal.is_coordinate_virtualization_enabled(),
        .dispatch_message_addr = ctx.dispatch_mem_map().get_dispatch_message_addr_start(),
        .max_cbs = hal.get_arch_num_circular_buffers(),
        .num_hw_cqs = num_hw_cqs,
        .routing_fw_enabled = cluster.is_base_routing_fw_enabled(),
        .profiler_dram_bank_size_per_risc_bytes = get_profiler_dram_bank_size_per_risc_bytes(ctx.rtoptions())};
}

}  // namespace tt::tt_metal
