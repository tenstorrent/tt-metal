// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>
#include <optional>
#include <limits>
#include <algorithm>
#include <type_traits>

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
#include "impl/context/metal_env_accessor.hpp"
#include "llrt/core_descriptor.hpp"

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
template <typename T>
T calc_delta(T current, T last) {
    static_assert(std::is_unsigned_v<T>, "calc_delta requires an unsigned counter type");
    if (current < last) {
        return current + (std::numeric_limits<T>::max() - last + 1);
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

    // TODO: replace with querying device without device instance or bake into telemetry
    static uint32_t get_total_worker_and_active_eth_cores(const IDevice& device) {
        auto& env = MetalEnvAccessor(tt::tt_metal::MetalContext::instance().get_env()).impl();
        auto& dcm = MetalContext::instance().get_dispatch_core_manager();

        const auto& compute_cores =
            tt::get_logical_compute_cores(env, device.id(), device.num_hw_cqs(), dcm.get_dispatch_core_config());

        uint32_t total_cores = compute_cores.size();
        total_cores += device.get_active_ethernet_cores(/*skip_reserved_tunnel_cores=*/true).size();
        return total_cores;
    }

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
        total_number_of_cores_(get_total_worker_and_active_eth_cores(device)),
        // TODO: Note: current impl assumes cq count won't change during its object lifetime
        telemetry_cores_per_cq_(collect_telemetry_cores(device)),
        last_read_dispatch_core_telemetry_(telemetry_cores_per_cq_.size()),
        last_read_prefetch_core_telemetry_(telemetry_cores_per_cq_.size()) {
        TT_FATAL(!telemetry_cores_per_cq_.empty(), "No dispatch telemetry cores found on device\n");
        // Init private members to construction time
        (void)read_info();
    }

    ~Impl() = default;

    uint32_t version() const { return DISPATCH_TELEMETRY_VERSION; }

    std::optional<float> get_normalized_utilization(
        const DispatchCoreTelemetry& current_dispatch_core_telemetry,
        const DispatchCoreTelemetry& last_read_dispatch_core_telemetry) {
        auto current_utilization_work_runtime = current_dispatch_core_telemetry.utilization_work_runtime;
        auto last_utilization_work_runtime = last_read_dispatch_core_telemetry.utilization_work_runtime;

        auto get_workers_in_flight_runtime = [](const DispatchCoreTelemetry& dispatch_core_telemetry) -> uint64_t {
            if (dispatch_core_telemetry.work_runtime_start == 0) {
                return 0;
            }

            for (size_t i = 0; i < MAX_SUB_DEVICES; ++i) {
                if (dispatch_core_telemetry.workers_per_sub_device[i] == 0) {
                    continue;
                }
                const bool workers_are_in_flight =
                    dispatch_core_telemetry.completion_count[i] < dispatch_core_telemetry.workers_per_sub_device[i];
                if (workers_are_in_flight) {
                    return dispatch_core_telemetry.current_timestamp - dispatch_core_telemetry.work_runtime_start;
                }
            }
            return 0;
        };

        current_utilization_work_runtime += get_workers_in_flight_runtime(current_dispatch_core_telemetry);
        last_utilization_work_runtime += get_workers_in_flight_runtime(last_read_dispatch_core_telemetry);

        auto utilization_runtime = calc_delta(current_utilization_work_runtime, last_utilization_work_runtime);

        auto runtime = calc_delta(
            current_dispatch_core_telemetry.current_timestamp, last_read_dispatch_core_telemetry.current_timestamp);
        if (runtime == 0) {
            log_warning(tt::LogMetal, "No time has elapsed since last read");
            return std::nullopt;
        }

        float utilization = static_cast<float>(utilization_runtime) / static_cast<float>(runtime);
        TT_ASSERT(utilization <= 1.0f, "If utilization is greater than 100%, there is an issue with the telemetry");
        TT_ASSERT(utilization >= 0.0f, "If utilization is less than 0%, there is an issue with the telemetry");
        return std::clamp(utilization, 0.0f, 1.0f);
    }

    std::optional<float> get_normalized_core_efficiency(
        const std::vector<DispatchCoreTelemetry>& current_dispatch_core_telemetry) {
        const uint32_t total_number_of_cores = total_number_of_cores_;
        TT_ASSERT(total_number_of_cores > 0, "Error in core detection, found 0 cores");

        auto get_work_times = [&](const std::vector<DispatchCoreTelemetry>& dispatch_core_telemetry_per_cq,
                                  // Cumulative of all work performed
                                  double& total_work_runtime,
                                  uint64_t& elapsed_device_time) {
            for (const auto& dispatch_telemetry : dispatch_core_telemetry_per_cq) {
                // Uncompress the averaged work time
                total_work_runtime += dispatch_telemetry.avg_work_runtime_per_worker * total_number_of_cores;

                for (size_t i = 0; i < MAX_SUB_DEVICES; ++i) {
                    if (dispatch_telemetry.workers_per_sub_device[i] == 0) {
                        continue;
                    }

                    // This value is already cumulative
                    uint64_t sub_device_running_work_time = dispatch_telemetry.current_sub_device_work_runtime[i];

                    const bool workers_are_in_flight =
                        dispatch_telemetry.completion_count[i] < dispatch_telemetry.workers_per_sub_device[i];
                    if (workers_are_in_flight) {
                        uint32_t in_flight_worker_count =
                            dispatch_telemetry.workers_per_sub_device[i] - dispatch_telemetry.completion_count[i];
                        sub_device_running_work_time +=
                            in_flight_worker_count *
                            (dispatch_telemetry.current_timestamp - dispatch_telemetry.last_work_launch_timestamp[i]);
                    }
                    total_work_runtime += static_cast<double>(sub_device_running_work_time);
                }
                uint64_t current_timestamp = dispatch_telemetry.current_timestamp;
                elapsed_device_time = std::max(elapsed_device_time, current_timestamp);
            }
        };

        double delta_total_work_runtime = 0;
        uint64_t delta_elapsed_device_time = 0;
        {
            double current_total_work_runtime = 0;
            uint64_t current_elapsed_device_time = 0;
            double last_total_work_runtime = 0;
            uint64_t last_elapsed_device_time = 0;
            get_work_times(current_dispatch_core_telemetry, current_total_work_runtime, current_elapsed_device_time);
            get_work_times(last_read_dispatch_core_telemetry_, last_total_work_runtime, last_elapsed_device_time);

            delta_total_work_runtime = current_total_work_runtime - last_total_work_runtime;
            delta_elapsed_device_time = calc_delta(current_elapsed_device_time, last_elapsed_device_time);
        }
        if (delta_elapsed_device_time == 0) {
            log_warning(tt::LogMetal, "No time has elapsed since last read");
            return std::nullopt;
        }

        const double avg_work_runtime_per_core = delta_total_work_runtime / static_cast<double>(total_number_of_cores);
        const float core_efficiency = static_cast<float>(avg_work_runtime_per_core / delta_elapsed_device_time);

        TT_ASSERT(
            core_efficiency <= 1.0f, "If core_efficiency is greater than 100%, there is an issue with the telemetry");
        TT_ASSERT(core_efficiency >= 0.0f, "If core_efficiency is less than 0%, there is an issue with the telemetry");
        return std::clamp(core_efficiency, 0.0f, 1.0f);
    }

    // Telemetry will be assembled in order according to its pre-calculated cq id.
    // This id correlates to core coordinates, so it is static for the lifetime of the
    // object.
    bool read_core_telemetry(
        std::vector<DispatchCoreTelemetry>& current_dispatch_core_telemetry,
        std::vector<PrefetchCoreTelemetry>& current_prefetch_core_telemetry) {
        TT_ASSERT(
            current_dispatch_core_telemetry.size() == telemetry_cores_per_cq_.size(),
            "Invalid dispatch telemetry size");
        TT_ASSERT(
            current_prefetch_core_telemetry.size() == telemetry_cores_per_cq_.size(),
            "Invalid prefetch telemetry size");

        for (size_t cq = 0; cq < telemetry_cores_per_cq_.size(); ++cq) {
            const auto& cq_entries = telemetry_cores_per_cq_[cq];
            if (cq_entries.empty()) {
                log_warning(tt::LogMetal, "No dispatch telemetry cores found for CQ {}", cq);
                // TODO: does this trigger on slow dispatch?
                return false;
            }

            bool found_prefetch_core = false;
            bool found_dispatch_core = false;

            for (const auto& core : cq_entries) {
                if (core.role == CoreRole::PREFETCH || core.role == CoreRole::PREFETCH_D) {
                    auto telemetry = read_prefetch_core_telemetry(chip_, core.virtual_core);
                    if (telemetry) {
                        TT_ASSERT(!found_prefetch_core, "Ensure only one prefetcher is found");
                        found_prefetch_core = true;
                        current_prefetch_core_telemetry[cq] = *telemetry;
                    }
                }
                if (core.role == CoreRole::DISPATCH || core.role == CoreRole::DISPATCH_D) {
                    auto telemetry = read_dispatch_core_telemetry(chip_, core.virtual_core);
                    if (telemetry) {
                        // TODO: for now
                        TT_ASSERT(!found_dispatch_core, "Ensure only worker dispatch is found");
                        found_dispatch_core = true;
                        current_dispatch_core_telemetry[cq] = *telemetry;
                    }
                }
            }
            if (!found_prefetch_core || !found_dispatch_core) {
                log_warning(tt::LogMetal, "Failed to read dispatch telemetry from core(s)");
                log_warning(tt::LogMetal, "Prefetch core: {}", found_prefetch_core ? "found" : "not found");
                log_warning(tt::LogMetal, "Dispatch core: {}", found_dispatch_core ? "found" : "not found");
                return false;
            }
        }

        return true;
    }

    std::optional<DispatchTelemetryDeviceInfo> compute_telemetry_info(
        const std::vector<DispatchCoreTelemetry>& current_dispatch_core_telemetry,
        const std::vector<PrefetchCoreTelemetry>& current_prefetch_core_telemetry) {
        TT_ASSERT(
            current_dispatch_core_telemetry.size() == current_prefetch_core_telemetry.size(),
            "Ensure there is one of each per cq");
        DispatchTelemetryDeviceInfo device_info;
        device_info.info_cqs.resize(current_prefetch_core_telemetry.size());

        device_info.device_core_efficiency_since_last_read =
            get_normalized_core_efficiency(current_dispatch_core_telemetry);

        for (size_t cq = 0; cq < current_prefetch_core_telemetry.size(); ++cq) {
            auto& cq_info = device_info.info_cqs[cq];
            const auto& prefetch_telemetry = current_prefetch_core_telemetry[cq];

            cq_info.cq_id = cq;
            cq_info.prefetch_waiting_on_upstream =
                (prefetch_telemetry.upstream_blocked_count != prefetch_telemetry.upstream_unblocked_count);
            cq_info.prefetch_blocked_count_since_last_read = calc_delta(
                prefetch_telemetry.upstream_blocked_count,
                last_read_prefetch_core_telemetry_[cq].upstream_blocked_count);
            cq_info.prefetch_command_count_since_last_read =
                calc_delta(prefetch_telemetry.command_count, last_read_prefetch_core_telemetry_[cq].command_count);

            last_read_prefetch_core_telemetry_[cq] = prefetch_telemetry;
        }
        for (size_t cq = 0; cq < current_dispatch_core_telemetry.size(); ++cq) {
            auto& cq_info = device_info.info_cqs[cq];
            const auto& dispatch_telemetry = current_dispatch_core_telemetry[cq];

            TT_ASSERT(cq_info.cq_id == cq, "cq_id mismatch");

            cq_info.dispatch_waiting_on_upstream =
                (dispatch_telemetry.upstream_blocked_count != dispatch_telemetry.upstream_unblocked_count);
            cq_info.dispatch_blocked_count_since_last_read = calc_delta(
                dispatch_telemetry.upstream_blocked_count,
                last_read_dispatch_core_telemetry_[cq].upstream_blocked_count);
            cq_info.program_count_since_last_read =
                calc_delta(dispatch_telemetry.program_count, last_read_dispatch_core_telemetry_[cq].program_count);
            cq_info.utilization_since_last_read =
                get_normalized_utilization(dispatch_telemetry, last_read_dispatch_core_telemetry_[cq]);

            last_read_dispatch_core_telemetry_[cq] = dispatch_telemetry;
        }

        return device_info;
    }

    std::optional<DispatchTelemetryDeviceInfo> read_info() {
        std::vector<DispatchCoreTelemetry> current_dispatch_core_telemetry(telemetry_cores_per_cq_.size());
        std::vector<PrefetchCoreTelemetry> current_prefetch_core_telemetry(telemetry_cores_per_cq_.size());

        if (!read_core_telemetry(current_dispatch_core_telemetry, current_prefetch_core_telemetry)) {
            return std::nullopt;
        }

        return compute_telemetry_info(current_dispatch_core_telemetry, current_prefetch_core_telemetry);
    }

private:
    ChipId chip_;
    CoreType dispatch_core_type_;
    uint32_t total_number_of_cores_;
    std::vector<std::vector<CoreEntry>> telemetry_cores_per_cq_;
    std::vector<DispatchCoreTelemetry> last_read_dispatch_core_telemetry_;
    std::vector<PrefetchCoreTelemetry> last_read_prefetch_core_telemetry_;
};

DispatchTelemetry::DispatchTelemetry(const IDevice& device) : impl_(std::make_unique<Impl>(device)) {}

DispatchTelemetry::~DispatchTelemetry() = default;

uint32_t DispatchTelemetry::version() const { return impl_->version(); }

std::optional<DispatchTelemetryDeviceInfo> DispatchTelemetry::read_info() { return impl_->read_info(); }

}  // namespace tt::tt_metal
