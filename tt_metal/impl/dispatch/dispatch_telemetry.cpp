// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>
#include <optional>
#include <limits>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <tt-metalium/experimental/dispatch_telemetry.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <umd/device/tt_device/tt_device.hpp>
#include <umd/device/types/xy_pair.hpp>

#include "impl/dispatch/dispatch_telemetry.hpp"
#include <hostdevcommon/dispatch_telemetry_types.hpp>
#include "llrt/tt_cluster.hpp"
#include "umd/device/types/core_coordinates.hpp"

namespace tt::tt_metal {

namespace {

struct SMCRuntimeTelemetryBuffer {
    tt_xy_pair arc_core;
    uint32_t addr = 0;
    uint32_t size = 0;
};

std::optional<SMCRuntimeTelemetryBuffer> discover_smc_dispatch_telemetry_control(tt::umd::TTDevice& tt_device) {
    tt::umd::FirmwareInfoProvider* firmware_info_provider = nullptr;
    try {
        // Throws when the device has no firmware info provider (ex: simulators)
        // Dispatch telemetry is best-effort: any failure to reach the SMC/firmware side must degrade to
        // "unavailable" rather than propagate, otherwise it would break device init/close on devices that
        // don't back a firmware info provider.
        firmware_info_provider = tt_device.get_firmware_info_provider();
    } catch (...) {
        log_warning(
            tt::LogMetal, "Dispatch telemetry SMC buffer unavailable (no firmware info provider, e.g. simulator)");
    }
    if (firmware_info_provider == nullptr) {
        return std::nullopt;
    }

    auto size = firmware_info_provider->get_runtime_telemetry_buffer_size();
    if (!size.has_value()) {
        log_warning(tt::LogMetal, "Dispatch telemetry SMC buffer is unavailable");
        return std::nullopt;
    }
    if (size.value() < sizeof(dispatch_telemetry_types::SMCDispatchTelemetryControl)) {
        log_warning(
            tt::LogMetal,
            "Dispatch telemetry SMC buffer is too small: got {} bytes, expected at least {} bytes",
            size.value(),
            sizeof(dispatch_telemetry_types::SMCDispatchTelemetryControl));
        return std::nullopt;
    }

    auto addr = firmware_info_provider->get_runtime_telemetry_buffer_address();
    if (!addr.has_value()) {
        log_warning(tt::LogMetal, "Dispatch telemetry SMC buffer address is unavailable or invalid");
        return std::nullopt;
    }

    return SMCRuntimeTelemetryBuffer{.arc_core = tt_device.get_arc_core(), .addr = addr.value(), .size = size.value()};
}

template <typename T>
std::optional<T> read_telemetry_impl(
    tt::umd::TTDevice& tt_device, CoreCoord virtual_core, uint32_t signature, uint32_t version, uint32_t addr_offset) {
    if (addr_offset == 0) {
        log_warning(tt::LogMetal, "SMC dispatch telemetry control has no dispatch telemetry address");
        return std::nullopt;
    }

    bool is_known_virtual_core = [&]() {
        const auto& soc_desc = tt_device.get_soc_descriptor();
        for (const tt::CoreType core_type : {tt::CoreType::TENSIX, tt::CoreType::ETH, tt::CoreType::DISPATCH}) {
            for (const auto& core : soc_desc.get_cores(core_type, tt::CoordSystem::TRANSLATED)) {
                if (core.x == virtual_core.x && core.y == virtual_core.y) {
                    return true;
                }
            }
        }
        return false;
    }();

    if (!is_known_virtual_core) {
        log_warning(
            tt::LogMetal,
            "Refusing dispatch telemetry read from unknown virtual core ({},{}); not a valid core in the SoC "
            "descriptor",
            virtual_core.x,
            virtual_core.y);
        return std::nullopt;
    }

    T telemetry{};
    tt_device.read_from_device(&telemetry, virtual_core, addr_offset, sizeof(telemetry));

    if (telemetry.signature != signature) {
        // Copy to avoid taking a reference to a packed struct's member variable, which could be unaligned.
        uint32_t telemetry_signature = telemetry.signature;
        log_warning(
            tt::LogMetal,
            "Signature mismatch on virtual core ({},{}): got 0x{:x}, expected 0x{:x}",
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
            "Version mismatch on virtual core ({},{}): got {}, expected {}",
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

std::optional<dispatch_telemetry_types::SMCDispatchTelemetryControl> read_smc_dispatch_telemetry_control(
    tt::umd::TTDevice& tt_device) {
    auto maybeSmcBufferInfo = discover_smc_dispatch_telemetry_control(tt_device);
    if (!maybeSmcBufferInfo.has_value()) {
        return std::nullopt;
    }
    const auto& smcBufferInfo = maybeSmcBufferInfo.value();

    dispatch_telemetry_types::SMCDispatchTelemetryControl control{};
    tt_device.read_from_device(&control, smcBufferInfo.arc_core, smcBufferInfo.addr, sizeof(control));

    if (control.signature != dispatch_telemetry_types::SMC_TELEMETRY_SIGNATURE) {
        uint32_t control_signature = control.signature;
        log_warning(
            tt::LogMetal,
            "SMC dispatch telemetry signature mismatch: got 0x{:x}, expected 0x{:x}",
            control_signature,
            dispatch_telemetry_types::SMC_TELEMETRY_SIGNATURE);
        return std::nullopt;
    }
    if (control.version != dispatch_telemetry_types::DISPATCH_TELEMETRY_VERSION) {
        uint32_t control_version = control.version;
        log_warning(
            tt::LogMetal,
            "SMC dispatch telemetry version mismatch: got {}, expected {}",
            control_version,
            dispatch_telemetry_types::DISPATCH_TELEMETRY_VERSION);
        return std::nullopt;
    }

    return control;
}

bool write_smc_dispatch_telemetry_control(
    tt::umd::TTDevice& tt_device, const dispatch_telemetry_types::SMCDispatchTelemetryControl& control) {
    auto maybeSmcBufferInfo = discover_smc_dispatch_telemetry_control(tt_device);
    if (!maybeSmcBufferInfo.has_value()) {
        return false;
    }
    const auto& smcBufferInfo = maybeSmcBufferInfo.value();
    tt_device.write_to_device(&control, smcBufferInfo.arc_core, smcBufferInfo.addr, sizeof(control));
    return true;
}

bool invalidate_smc_dispatch_telemetry_control(tt::umd::TTDevice& tt_device) {
    auto maybeSmcBufferInfo = discover_smc_dispatch_telemetry_control(tt_device);
    if (!maybeSmcBufferInfo.has_value()) {
        return false;
    }
    const auto& smcBufferInfo = maybeSmcBufferInfo.value();

    uint32_t invalid_signature = dispatch_telemetry_types::INVALID_TELEMETRY_SIGNATURE;
    tt_device.write_to_device(
        &invalid_signature,
        smcBufferInfo.arc_core,
        smcBufferInfo.addr + offsetof(dispatch_telemetry_types::SMCDispatchTelemetryControl, signature),
        sizeof(invalid_signature));
    return true;
}

std::optional<dispatch_telemetry_types::DispatchCoreTelemetry> read_dispatch_core_telemetry(
    tt::umd::TTDevice& tt_device, CoreCoord virtual_core) {
    auto control = read_smc_dispatch_telemetry_control(tt_device);
    if (!control.has_value()) {
        return std::nullopt;
    }
    uint32_t addr_offset = control->dispatch_telemetry_addr;

    return read_telemetry_impl<dispatch_telemetry_types::DispatchCoreTelemetry>(
        tt_device,
        virtual_core,
        dispatch_telemetry_types::DISPATCH_CORE_TELEMETRY_SIGNATURE,
        dispatch_telemetry_types::DISPATCH_TELEMETRY_VERSION,
        addr_offset);
}

std::optional<dispatch_telemetry_types::PrefetchCoreTelemetry> read_prefetch_core_telemetry(
    tt::umd::TTDevice& tt_device, CoreCoord virtual_core) {
    auto control = read_smc_dispatch_telemetry_control(tt_device);
    if (!control.has_value()) {
        return std::nullopt;
    }
    uint32_t addr_offset = control->dispatch_telemetry_addr;

    return read_telemetry_impl<dispatch_telemetry_types::PrefetchCoreTelemetry>(
        tt_device,
        virtual_core,
        dispatch_telemetry_types::PREFETCH_CORE_TELEMETRY_SIGNATURE,
        dispatch_telemetry_types::DISPATCH_TELEMETRY_VERSION,
        addr_offset);
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
        CoreCoord virtual_core;
        uint8_t cq_id = 0;
    };

    static uint32_t get_total_worker_and_active_eth_cores(tt::umd::TTDevice& tt_device) {
        const auto& soc_desc = tt_device.get_soc_descriptor();
        uint32_t total_cores = soc_desc.get_cores(tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED).size();
        total_cores += soc_desc.get_cores(tt::CoreType::ETH, tt::CoordSystem::TRANSLATED).size();
        return total_cores;
    }

    std::vector<std::vector<CoreEntry>> collect_telemetry_cores() {
        auto control = read_smc_dispatch_telemetry_control(tt_device_);
        if (!control.has_value() || control->num_hw_cqs > dispatch_telemetry_types::RESERVED_CQ_SPACE) {
            return {};
        }

        std::vector<std::vector<CoreEntry>> entries_per_active_cq;
        const uint8_t num_cqs = control->num_hw_cqs;

        for (uint8_t cq = 0; cq < num_cqs; ++cq) {
            const auto core_coords = control->cq_dispatch_core_coords[cq];
            if (core_coords.prefetch_xy == dispatch_telemetry_types::INVALID_SMC_DISPATCH_CORE_COORDS ||
                core_coords.dispatch_xy == dispatch_telemetry_types::INVALID_SMC_DISPATCH_CORE_COORDS) {
                // Assumes valid CQ IDs are contiguous
                break;
            }

            std::vector<CoreEntry> cq_entries;

            auto smc_xy_to_virtual_core = [](uint32_t xy) {
                return CoreCoord{
                    dispatch_telemetry_types::smc_dispatch_core_x(xy),
                    dispatch_telemetry_types::smc_dispatch_core_y(xy)};
            };

            cq_entries.push_back(CoreEntry{
                .role = CoreRole::PREFETCH,
                .virtual_core = smc_xy_to_virtual_core(core_coords.prefetch_xy),
                .cq_id = cq});
            cq_entries.push_back(CoreEntry{
                .role = CoreRole::DISPATCH,
                .virtual_core = smc_xy_to_virtual_core(core_coords.dispatch_xy),
                .cq_id = cq});

            if (core_coords.dispatch_s_xy != dispatch_telemetry_types::INVALID_SMC_DISPATCH_CORE_COORDS) {
                // TODO: Handle dispatch_s telemetry once SMC-discovered dispatch_s data is consumed here.
                cq_entries.push_back(CoreEntry{
                    .role = CoreRole::DISPATCH_S,
                    .virtual_core = smc_xy_to_virtual_core(core_coords.dispatch_s_xy),
                    .cq_id = cq});
            }

            entries_per_active_cq.push_back(std::move(cq_entries));
        }

        return entries_per_active_cq;
    }

public:
    Impl(tt::umd::TTDevice& device) :
        tt_device_(device), total_number_of_cores_(get_total_worker_and_active_eth_cores(device)) {
        // Init private members to construction time
        (void)read_info();
    }

    ~Impl() = default;

    uint32_t version() const { return dispatch_telemetry_types::DISPATCH_TELEMETRY_VERSION; }

    std::optional<float> get_normalized_utilization(
        const dispatch_telemetry_types::DispatchCoreTelemetry& current_dispatch_core_telemetry,
        const dispatch_telemetry_types::DispatchCoreTelemetry& last_read_dispatch_core_telemetry) {
        auto current_utilization_work_runtime = current_dispatch_core_telemetry.utilization_work_runtime;
        auto last_utilization_work_runtime = last_read_dispatch_core_telemetry.utilization_work_runtime;

        auto get_workers_in_flight_runtime =
            [](const dispatch_telemetry_types::DispatchCoreTelemetry& dispatch_core_telemetry) -> uint64_t {
            if (dispatch_core_telemetry.work_runtime_start == 0) {
                return 0;
            }

            for (size_t i = 0; i < dispatch_telemetry_types::MAX_SUB_DEVICES; ++i) {
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
        const std::vector<dispatch_telemetry_types::DispatchCoreTelemetry>& current_dispatch_core_telemetry) {
        const uint32_t total_number_of_cores = total_number_of_cores_;
        TT_ASSERT(total_number_of_cores > 0, "Error in core detection, found 0 cores");

        auto get_work_times = [&](const std::vector<dispatch_telemetry_types::DispatchCoreTelemetry>&
                                      dispatch_core_telemetry_per_cq,
                                  double& total_work_runtime,  // Cumulative of all work performed
                                  uint64_t& elapsed_device_time) {
            for (const auto& dispatch_telemetry : dispatch_core_telemetry_per_cq) {
                // Uncompress the averaged work time
                total_work_runtime += dispatch_telemetry.avg_work_runtime_per_worker * total_number_of_cores;

                for (size_t i = 0; i < dispatch_telemetry_types::MAX_SUB_DEVICES; ++i) {
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

    bool read_core_telemetry(
        const std::vector<std::vector<CoreEntry>>& entries_per_active_cq,
        std::vector<dispatch_telemetry_types::DispatchCoreTelemetry>& current_dispatch_core_telemetry,
        std::vector<dispatch_telemetry_types::PrefetchCoreTelemetry>& current_prefetch_core_telemetry) {
        TT_ASSERT(
            current_dispatch_core_telemetry.size() == entries_per_active_cq.size(), "Invalid dispatch telemetry size");
        TT_ASSERT(
            current_prefetch_core_telemetry.size() == entries_per_active_cq.size(), "Invalid prefetch telemetry size");

        for (size_t cq = 0; cq < entries_per_active_cq.size(); ++cq) {
            const auto& cq_entries = entries_per_active_cq[cq];
            if (cq_entries.empty()) {
                log_warning(tt::LogMetal, "No dispatch telemetry cores found for CQ {}", cq);
                return false;
            }

            bool found_prefetch_core = false;
            bool found_dispatch_core = false;

            for (const auto& core : cq_entries) {
                if (core.role == CoreRole::PREFETCH || core.role == CoreRole::PREFETCH_D) {
                    auto telemetry = read_prefetch_core_telemetry(tt_device_, core.virtual_core);
                    if (telemetry) {
                        TT_ASSERT(!found_prefetch_core, "Ensure only one prefetcher is found");
                        found_prefetch_core = true;
                        current_prefetch_core_telemetry[cq] = *telemetry;
                    }
                }
                if (core.role == CoreRole::DISPATCH || core.role == CoreRole::DISPATCH_D) {
                    auto telemetry = read_dispatch_core_telemetry(tt_device_, core.virtual_core);
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
        const std::vector<std::vector<CoreEntry>>& entries_per_active_cq,
        const std::vector<dispatch_telemetry_types::DispatchCoreTelemetry>& current_dispatch_core_telemetry,
        const std::vector<dispatch_telemetry_types::PrefetchCoreTelemetry>& current_prefetch_core_telemetry) {
        TT_ASSERT(
            current_dispatch_core_telemetry.size() == current_prefetch_core_telemetry.size(),
            "Ensure there is one of each per cq");
        TT_ASSERT(
            entries_per_active_cq.size() == current_prefetch_core_telemetry.size(),
            "Ensure telemetry and core entries are aligned");
        DispatchTelemetryDeviceInfo device_info;
        device_info.info_cqs.resize(current_prefetch_core_telemetry.size());

        device_info.device_core_efficiency_since_last_read =
            get_normalized_core_efficiency(current_dispatch_core_telemetry);

        for (size_t cq = 0; cq < current_prefetch_core_telemetry.size(); ++cq) {
            auto& cq_info = device_info.info_cqs[cq];
            const auto& prefetch_telemetry = current_prefetch_core_telemetry[cq];
            TT_ASSERT(!entries_per_active_cq[cq].empty(), "Missing dispatch telemetry core entry");

            cq_info.cq_id = entries_per_active_cq[cq].front().cq_id;
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

            TT_ASSERT(cq_info.cq_id == entries_per_active_cq[cq].front().cq_id, "cq_id mismatch");

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
        std::vector<std::vector<CoreEntry>> entries_per_active_cq = collect_telemetry_cores();
        if (entries_per_active_cq.empty()) {
            return std::nullopt;
        }

        std::vector<dispatch_telemetry_types::DispatchCoreTelemetry> current_dispatch_core_telemetry(
            entries_per_active_cq.size());
        std::vector<dispatch_telemetry_types::PrefetchCoreTelemetry> current_prefetch_core_telemetry(
            entries_per_active_cq.size());

        last_read_dispatch_core_telemetry_.resize(entries_per_active_cq.size());
        last_read_prefetch_core_telemetry_.resize(entries_per_active_cq.size());

        if (!read_core_telemetry(
                entries_per_active_cq, current_dispatch_core_telemetry, current_prefetch_core_telemetry)) {
            return std::nullopt;
        }

        return compute_telemetry_info(
            entries_per_active_cq, current_dispatch_core_telemetry, current_prefetch_core_telemetry);
    }

private:
    tt::umd::TTDevice& tt_device_;
    uint32_t total_number_of_cores_;
    std::vector<dispatch_telemetry_types::DispatchCoreTelemetry> last_read_dispatch_core_telemetry_;
    std::vector<dispatch_telemetry_types::PrefetchCoreTelemetry> last_read_prefetch_core_telemetry_;
};

DispatchTelemetry::DispatchTelemetry(tt::umd::TTDevice& device) : impl_(std::make_unique<Impl>(device)) {}

DispatchTelemetry::~DispatchTelemetry() = default;

uint32_t DispatchTelemetry::version() const { return impl_->version(); }

std::optional<DispatchTelemetryDeviceInfo> DispatchTelemetry::read_info() { return impl_->read_info(); }

}  // namespace tt::tt_metal
