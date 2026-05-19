// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/dispatch_telemetry.hpp>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/xy_pair.hpp>

#include "device.hpp"
#include "dispatch/command_queue_common.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal {

namespace {

template <typename T>
std::optional<T> read_telemetry_impl(
    const char* type_name,
    IDevice* device,
    const CoreCoord& logical_core,
    CoreType core_type,
    uint32_t signature,
    uint32_t version) {
    // Telemetry lives at a fixed dispatch-core-local L1 offset assigned by DispatchMemMap.
    // Prefetch and dispatch both use this section depending on which one owns the core.
    const auto& dispatch_mem_map = MetalContext::instance().dispatch_mem_map();
    uint32_t addr = dispatch_mem_map.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_TELEMETRY);

    // read_core needs a virtual (noc-addressable) coord
    CoreCoord virtual_core = device->virtual_core_from_logical_core(logical_core, core_type);

    // Make sure any in-flight kernel writes to L1 are visible before we sample.
    const auto& cluster = MetalContext::instance().get_cluster();
    cluster.l1_barrier(device->id());

    T telemetry{};
    cluster.read_core(&telemetry, sizeof(telemetry), tt_cxy_pair(device->id(), virtual_core), addr);

    // Sanity-check the buffer actually contains a current-version telemetry block.
    if (telemetry.signature != signature) {
        log_warning(
            tt::LogMetal,
            "{} signature mismatch on chip {} core ({},{}) @ 0x{:x}: got 0x{:x}",
            type_name,
            device->id(),
            logical_core.x,
            logical_core.y,
            addr,
            telemetry.signature);
        return std::nullopt;
    }
    if (telemetry.version != version) {
        log_warning(
            tt::LogMetal,
            "{} version mismatch on chip {}: got {}, expected {}",
            type_name,
            device->id(),
            telemetry.version,
            version);
        return std::nullopt;
    }
    return telemetry;
}

}  // namespace

std::optional<DispatchTelemetry> read_dispatch_telemetry(
    IDevice* device, const CoreCoord& dispatch_logical_core, CoreType core_type) {
    return read_telemetry_impl<DispatchTelemetry>(
        "DispatchTelemetry",
        device,
        dispatch_logical_core,
        core_type,
        DISPATCH_TELEMETRY_SIGNATURE,
        DISPATCH_TELEMETRY_VERSION);
}

std::optional<PrefetchTelemetry> read_prefetch_telemetry(
    IDevice* device, const CoreCoord& prefetch_logical_core, CoreType core_type) {
    return read_telemetry_impl<PrefetchTelemetry>(
        "PrefetchTelemetry",
        device,
        prefetch_logical_core,
        core_type,
        PREFETCH_TELEMETRY_SIGNATURE,
        PREFETCH_TELEMETRY_VERSION);
}

}  // namespace tt::tt_metal
