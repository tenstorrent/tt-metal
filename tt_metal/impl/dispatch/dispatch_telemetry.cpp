// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>
#include <optional>

#include <tt-metalium/experimental/dispatch_telemetry.hpp>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/xy_pair.hpp>

#include "device.hpp"
#include "device_types.hpp"
#include "dispatch/command_queue_common.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "impl/dispatch/dispatch_telemetry.hpp"
#include <hostdevcommon/dispatch_telemetry_types.hpp>
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
        // Copy to avoid taking a reference to a packed struct's member variable, which could be unaligned.
        uint32_t telemetry_signature = telemetry.signature;
        log_warning(
            tt::LogMetal,
            "Signature mismatch on chip {} core ({},{}): got 0x{:x}, expected 0x{:x}",
            chip,
            virtual_core.x,
            virtual_core.y,
            telemetry_signature,
            signature);
        return std::nullopt;
    }
    if (telemetry.version != version) {
        // Copy to avoid taking a reference to a packed struct's member variable, which could be unaligned.
        uint32_t telemetry_version = telemetry.version;
        log_warning(
            tt::LogMetal,
            "Version mismatch on chip {} core ({},{}): got {}, expected {}",
            chip,
            virtual_core.x,
            virtual_core.y,
            telemetry_version,
            version);
        return std::nullopt;
    }
    return telemetry;
}

// Takes into consideration rollover
uint32_t calc_delta(uint32_t current, uint32_t last) {
    if (current < last) {
        return current + (UINT32_MAX - last + 1);
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
private:
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

    std::vector<std::vector<CoreEntry>> collect_telemetry_cores(const IDevice& device) {
        std::vector<std::vector<CoreEntry>> entries;

        auto& dcm = MetalContext::instance().get_dispatch_core_manager();
        const auto& cluster = MetalContext::instance().get_cluster();
        const ChipId chip = device.id();
        const uint16_t channel = cluster.get_assigned_channel_for_device(chip);
        const uint8_t num_cqs = device.num_hw_cqs();
        const CoreType core_type = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type();

        for (uint8_t cq = 0; cq < num_cqs; ++cq) {
            std::vector<CoreEntry> cq_entries;
            if (dcm.is_prefetcher_core_allocated(chip, channel, cq)) {
                CoreEntry entry{};
                entry.role = CoreRole::PREFETCH;
                tt_cxy_pair logical_cxy = dcm.prefetcher_core(chip, channel, cq);
                entry.virtual_core =
                    device.virtual_core_from_logical_core(CoreCoord{logical_cxy.x, logical_cxy.y}, core_type);
                entry.cq_id = cq;
                cq_entries.push_back(entry);
            }
            if (dcm.is_prefetcher_d_core_allocated(chip, channel, cq)) {
                CoreEntry entry{};
                entry.role = CoreRole::PREFETCH_D;
                tt_cxy_pair logical_cxy = dcm.prefetcher_d_core(chip, channel, cq);
                entry.virtual_core =
                    device.virtual_core_from_logical_core(CoreCoord{logical_cxy.x, logical_cxy.y}, core_type);
                entry.cq_id = cq;
                cq_entries.push_back(entry);
            }
            if (dcm.is_dispatcher_core_allocated(chip, channel, cq)) {
                CoreEntry entry{};
                entry.role = CoreRole::DISPATCH;
                tt_cxy_pair logical_cxy = dcm.dispatcher_core(chip, channel, cq);
                entry.virtual_core =
                    device.virtual_core_from_logical_core(CoreCoord{logical_cxy.x, logical_cxy.y}, core_type);
                entry.cq_id = cq;
                cq_entries.push_back(entry);
            }
            if (dcm.is_dispatcher_d_core_allocated(chip, channel, cq)) {
                CoreEntry entry{};
                entry.role = CoreRole::DISPATCH_D;
                tt_cxy_pair logical_cxy = dcm.dispatcher_d_core(chip, channel, cq);
                entry.virtual_core =
                    device.virtual_core_from_logical_core(CoreCoord{logical_cxy.x, logical_cxy.y}, core_type);
                entry.cq_id = cq;
                cq_entries.push_back(entry);
            }
            if (dcm.is_dispatcher_s_core_allocated(chip, channel, cq)) {
                CoreEntry entry{};
                entry.role = CoreRole::DISPATCH_S;
                tt_cxy_pair logical_cxy = dcm.dispatcher_s_core(chip, channel, cq);
                entry.virtual_core =
                    device.virtual_core_from_logical_core(CoreCoord{logical_cxy.x, logical_cxy.y}, core_type);
                entry.cq_id = cq;
                cq_entries.push_back(entry);
            }
            entries.push_back(cq_entries);
        }

        return entries;
    }

public:
    Impl(const IDevice& device) :
        chip_(device.id()),
        dispatch_core_type_(MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type()),
        telemetry_cores_(collect_telemetry_cores(device)),
        last_read_dispatch_core_telemetry_(telemetry_cores_.size()),
        last_read_prefetch_core_telemetry_(telemetry_cores_.size()) {
        TT_FATAL(!telemetry_cores_.empty(), "No dispatch telemetry cores found on device\n");
    }

    ~Impl() = default;

    uint32_t version() const { return DISPATCH_TELEMETRY_VERSION; }

    std::vector<DispatchTelemetryInfo> read_info() {
        std::vector<DispatchTelemetryInfo> infos;

        for (size_t cq = 0; cq < telemetry_cores_.size(); ++cq) {
            const auto& cq_entries = telemetry_cores_[cq];
            if (cq_entries.empty()) {
                log_warning(tt::LogMetal, "No dispatch telemetry cores found for CQ {}", cq);
                continue;
            }

            DispatchTelemetryInfo info{};
            info.cq_id = cq_entries.front().cq_id;
            bool found_prefetch_core = false;
            bool found_dispatch_core = false;

            for (const auto& core : cq_entries) {
                if (core.role == CoreRole::PREFETCH || core.role == CoreRole::PREFETCH_D) {
                    auto telemetry = read_prefetch_core_telemetry(chip_, core.virtual_core);
                    if (telemetry) {
                        info.prefetch_waiting_on_upstream =
                            (telemetry->upstream_blocked_count != telemetry->upstream_unblocked_count);
                        info.prefetch_blocked_count_since_last_read = calc_delta(
                            telemetry->upstream_blocked_count,
                            last_read_prefetch_core_telemetry_[cq].upstream_blocked_count);
                        info.prefetch_command_count_since_last_read =
                            calc_delta(telemetry->command_count, last_read_prefetch_core_telemetry_[cq].command_count);
                        found_prefetch_core = true;
                        last_read_prefetch_core_telemetry_[cq] = *telemetry;
                    }
                }
                if (core.role == CoreRole::DISPATCH || core.role == CoreRole::DISPATCH_D) {
                    auto telemetry = read_dispatch_core_telemetry(chip_, core.virtual_core);
                    if (telemetry) {
                        info.dispatch_waiting_on_upstream =
                            (telemetry->upstream_blocked_count != telemetry->upstream_unblocked_count);
                        info.dispatch_blocked_count_since_last_read = calc_delta(
                            telemetry->upstream_blocked_count,
                            last_read_dispatch_core_telemetry_[cq].upstream_blocked_count);
                        info.dispatch_program_count_since_last_read =
                            calc_delta(telemetry->program_count, last_read_dispatch_core_telemetry_[cq].program_count);
                        found_dispatch_core = true;
                        last_read_dispatch_core_telemetry_[cq] = *telemetry;
                    }
                }
            }
            if (!found_prefetch_core || !found_dispatch_core) {
                log_warning(tt::LogMetal, "Failed to read dispatch telemetry from core(s)");
                log_warning(tt::LogMetal, "Prefetch core: {}", found_prefetch_core ? "found" : "not found");
                log_warning(tt::LogMetal, "Dispatch core: {}", found_dispatch_core ? "found" : "not found");
                continue;
            }
            infos.push_back(info);
        }
        return infos;
    }

private:
    ChipId chip_;
    CoreType dispatch_core_type_;
    std::vector<std::vector<CoreEntry>> telemetry_cores_;
    std::vector<DispatchCoreTelemetry> last_read_dispatch_core_telemetry_;
    std::vector<PrefetchCoreTelemetry> last_read_prefetch_core_telemetry_;
};

DispatchTelemetry::DispatchTelemetry(const IDevice& device) : impl_(std::make_unique<Impl>(device)) {}

DispatchTelemetry::~DispatchTelemetry() = default;

uint32_t DispatchTelemetry::version() const { return impl_->version(); }

std::vector<DispatchTelemetryInfo> DispatchTelemetry::read_info() { return impl_->read_info(); }

}  // namespace tt::tt_metal
