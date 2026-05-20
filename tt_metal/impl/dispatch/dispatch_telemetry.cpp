// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>
#include <optional>

#include <tt-metalium/experimental/dispatch_telemetry_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/xy_pair.hpp>

#include "device.hpp"
#include "device_types.hpp"
#include "dispatch/command_queue_common.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "impl/dispatch/dispatch_telemetry.hpp"
#include "impl/dispatch/dispatch_telemetry_types.hpp"
#include "llrt/tt_cluster.hpp"
#include "umd/device/types/core_coordinates.hpp"

namespace tt::tt_metal {

namespace {

template <typename T>
std::optional<T> read_telemetry_impl(ChipId chip, const CoreCoord& virtual_core, uint32_t signature, uint32_t version) {
    // Telemetry lives at a fixed dispatch-core-local L1 offset assigned by DispatchMemMap.
    // Prefetch and dispatch both use this section depending on which one is running on the core.
    const auto& dispatch_mem_map = MetalContext::instance().dispatch_mem_map();
    uint32_t addr = dispatch_mem_map.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_TELEMETRY);

    // Make sure any in-flight kernel writes to L1 are visible before we sample.
    const auto& cluster = MetalContext::instance().get_cluster();
    cluster.l1_barrier(chip);

    T telemetry{};
    cluster.read_core(&telemetry, sizeof(telemetry), tt_cxy_pair(chip, virtual_core), addr);

    if (telemetry.signature != signature) {
        log_warning(
            tt::LogMetal,
            "Signature mismatch on chip {} core ({},{}): got 0x{:x}, expected 0x{:x}",
            chip,
            virtual_core.x,
            virtual_core.y,
            telemetry.signature,
            signature);
        return std::nullopt;
    }
    if (telemetry.version != version) {
        log_warning(
            tt::LogMetal,
            "Version mismatch on chip {} core ({},{}): got {}, expected {}",
            chip,
            virtual_core.x,
            virtual_core.y,
            telemetry.version,
            version);
        return std::nullopt;
    }
    return telemetry;
}

enum class CoreRole : uint8_t {
    INVALID = 0,
    PREFETCH,
    PREFETCH_D,
    DISPATCH,
    DISPATCH_D,
    DISPATCH_S,
};

struct CoreEntry {
    CoreRole role = CoreRole::INVALID;
    CoreCoord virtual_core;  // read_core needs a virtual (noc-addressable) coord
    uint8_t cq_id;
};

std::vector<CoreEntry> collect_telemetry_cores(const IDevice& device) {
    std::vector<CoreEntry> entries;
    uint8_t num_prefetch_cores = 0;
    uint8_t num_dispatch_cores = 0;
    uint8_t num_dispatch_s_cores = 0;

    auto& dcm = MetalContext::instance().get_dispatch_core_manager();
    const auto& cluster = MetalContext::instance().get_cluster();
    const ChipId chip = device.id();
    const uint16_t channel = cluster.get_assigned_channel_for_device(chip);
    const uint8_t num_cqs = device.num_hw_cqs();
    const CoreType core_type = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type();

    for (uint8_t cq = 0; cq < num_cqs; ++cq) {
        if (dcm.is_prefetcher_core_allocated(chip, channel, cq)) {
            CoreEntry entry{};
            entry.role = CoreRole::PREFETCH;
            tt_cxy_pair logical_cxy = dcm.prefetcher_core(chip, channel, cq);
            entry.virtual_core =
                device.virtual_core_from_logical_core(CoreCoord{logical_cxy.x, logical_cxy.y}, core_type);
            entry.cq_id = cq;
            entries.push_back(entry);
            num_prefetch_cores++;
        }
        if (dcm.is_prefetcher_d_core_allocated(chip, channel, cq)) {
            CoreEntry entry{};
            entry.role = CoreRole::PREFETCH_D;
            tt_cxy_pair logical_cxy = dcm.prefetcher_d_core(chip, channel, cq);
            entry.virtual_core =
                device.virtual_core_from_logical_core(CoreCoord{logical_cxy.x, logical_cxy.y}, core_type);
            entry.cq_id = cq;
            entries.push_back(entry);
            num_prefetch_cores++;
        }
        if (dcm.is_dispatcher_core_allocated(chip, channel, cq)) {
            CoreEntry entry{};
            entry.role = CoreRole::DISPATCH;
            tt_cxy_pair logical_cxy = dcm.dispatcher_core(chip, channel, cq);
            entry.virtual_core =
                device.virtual_core_from_logical_core(CoreCoord{logical_cxy.x, logical_cxy.y}, core_type);
            entry.cq_id = cq;
            entries.push_back(entry);
            num_dispatch_cores++;
        }
        if (dcm.is_dispatcher_d_core_allocated(chip, channel, cq)) {
            CoreEntry entry{};
            entry.role = CoreRole::DISPATCH_D;
            tt_cxy_pair logical_cxy = dcm.dispatcher_d_core(chip, channel, cq);
            entry.virtual_core =
                device.virtual_core_from_logical_core(CoreCoord{logical_cxy.x, logical_cxy.y}, core_type);
            entry.cq_id = cq;
            entries.push_back(entry);
            num_dispatch_cores++;
        }
        if (dcm.is_dispatcher_s_core_allocated(chip, channel, cq)) {
            CoreEntry entry{};
            entry.role = CoreRole::DISPATCH_S;
            tt_cxy_pair logical_cxy = dcm.dispatcher_s_core(chip, channel, cq);
            entry.virtual_core =
                device.virtual_core_from_logical_core(CoreCoord{logical_cxy.x, logical_cxy.y}, core_type);
            entry.cq_id = cq;
            entries.push_back(entry);
            num_dispatch_s_cores++;
        }
    }
    TT_FATAL(num_prefetch_cores <= 1, "Expected no more than 1 prefetch core, found {}\n", num_prefetch_cores);
    TT_FATAL(num_dispatch_cores <= 1, "Expected no more than 1 dispatch core, found {}\n", num_dispatch_cores);
    TT_FATAL(num_dispatch_s_cores <= 1, "Expected no more than 1 dispatch_s core, found {}\n", num_dispatch_s_cores);
    return entries;
}

// Takes into consideration rollover
uint32_t calc_delta(uint32_t current, uint32_t last) {
    if (current < last) {
        return current + (UINT32_MAX - last);
    }
    return current - last;
}

}  // namespace

std::optional<DispatchCoreTelemetry> read_dispatch_core_telemetry(ChipId chip, const CoreCoord& virtual_core) {
    return read_telemetry_impl<DispatchCoreTelemetry>(
        chip, virtual_core, DISPATCH_CORE_TELEMETRY_SIGNATURE, DISPATCH_TELEMETRY_VERSION);
}

std::optional<PrefetchCoreTelemetry> read_prefetch_core_telemetry(ChipId chip, const CoreCoord& virtual_core) {
    return read_telemetry_impl<PrefetchCoreTelemetry>(
        chip, virtual_core, PREFETCH_CORE_TELEMETRY_SIGNATURE, DISPATCH_TELEMETRY_VERSION);
}

class DispatchTelemetry::Impl {
public:
    Impl(const IDevice& device) :
        chip_(device.id()),
        dispatch_core_type_(MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type()),
        telemetry_cores_(collect_telemetry_cores(device)) {
        TT_FATAL(!telemetry_cores_.empty(), "No dispatch telemetry cores found on device\n");
    }

    ~Impl() = default;

    uint32_t version() const { return DISPATCH_TELEMETRY_VERSION; }

    std::optional<DispatchTelemetryInfo> read_info() {
        DispatchTelemetryInfo info{};
        bool found_prefetch_core = false;
        bool found_dispatch_core = false;

        for (const auto& core : telemetry_cores_) {
            if (core.role == CoreRole::PREFETCH || core.role == CoreRole::PREFETCH_D) {
                auto telemetry = read_prefetch_core_telemetry(chip_, core.virtual_core);
                if (telemetry) {
                    info.prefetch_waiting = (telemetry->upstream_blocked_count != telemetry->upstream_unblocked_count);
                    info.prefetch_blocked_count_since_last_read = calc_delta(
                        telemetry->upstream_blocked_count, last_read_info_.prefetch_blocked_count_since_last_read);
                    info.prefetch_command_count_since_last_read =
                        calc_delta(telemetry->command_count, last_read_info_.prefetch_command_count_since_last_read);
                    found_prefetch_core = true;
                }
            }
            if (core.role == CoreRole::DISPATCH || core.role == CoreRole::DISPATCH_D) {
                auto telemetry = read_dispatch_core_telemetry(chip_, core.virtual_core);
                if (telemetry) {
                    info.dispatch_waiting = (telemetry->upstream_blocked_count != telemetry->upstream_unblocked_count);
                    info.dispatch_blocked_count_since_last_read = calc_delta(
                        telemetry->upstream_blocked_count, last_read_info_.dispatch_blocked_count_since_last_read);
                    info.dispatch_program_count_since_last_read =
                        calc_delta(telemetry->program_count, last_read_info_.dispatch_program_count_since_last_read);
                    found_dispatch_core = true;
                }
            }
        }
        if (!found_prefetch_core || !found_dispatch_core) {
            return std::nullopt;
        }
        last_read_info_ = info;
        return info;
    }

private:
    ChipId chip_;
    CoreType dispatch_core_type_;
    std::vector<CoreEntry> telemetry_cores_;
    DispatchTelemetryInfo last_read_info_{};
};

DispatchTelemetry::DispatchTelemetry(const IDevice& device) : impl_(std::make_unique<Impl>(device)) {}

DispatchTelemetry::~DispatchTelemetry() = default;

uint32_t DispatchTelemetry::version() const { return impl_->version(); }

std::optional<DispatchTelemetryInfo> DispatchTelemetry::read_info() { return impl_->read_info(); }

}  // namespace tt::tt_metal
