// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include "fabric_firmware_initializer.hpp"

#include <algorithm>
#include <chrono>
#include <optional>
#include <string_view>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <llrt/tt_cluster.hpp>
#include <tt_metal.hpp>
#include "device/device_impl.hpp"
#include "common/executor.hpp"
#include "impl/context/context_descriptor.hpp"

#include <experimental/fabric/control_plane.hpp>
#include <experimental/fabric/fabric_types.hpp>
#include "fabric/fabric_host_utils.hpp"
#include "fabric/fabric_context.hpp"
#include "fabric/fabric_builder_context.hpp"
#include "fabric/fabric_edm_packet_header.hpp"

namespace tt::tt_metal {

namespace {

using tt::tt_fabric::chan_id_t;
using tt::tt_fabric::EDMStatus;

static_assert(static_cast<uint32_t>(EDMStatus::STARTED) != 0);
static_assert(static_cast<uint32_t>(EDMStatus::REMOTE_HANDSHAKE_COMPLETE) != 0);
static_assert(static_cast<uint32_t>(EDMStatus::LOCAL_HANDSHAKE_COMPLETE) != 0);
static_assert(static_cast<uint32_t>(EDMStatus::READY_FOR_TRAFFIC) != 0);
static_assert(static_cast<uint32_t>(EDMStatus::TERMINATED) != 0);

// Progress stages observable at a router's edm_status_address during init.
// Ordering is significant: declaration order of the first five enumerators
// reflects the kernel's *edm_status_ptr write sequence in
// tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp, so
// `a < b` means stage `a` is earlier in init than stage `b`.
//
// Sentinels that do not lie on the linear progress axis
// (TerminatedIndeterminate, Unknown) must be declared AFTER all comparable
// stages so has_comparable_progress() stays correct.
enum class EdmInitProgress : uint8_t {
    NotStarted,
    Started,
    RemoteHandshakeComplete,
    LocalHandshakeComplete,
    ReadyForTraffic,
    TerminatedIndeterminate,
    Unknown,
};

constexpr bool has_comparable_progress(EdmInitProgress p) { return p <= EdmInitProgress::ReadyForTraffic; }

std::string_view progress_name(EdmInitProgress p) {
    switch (p) {
        case EdmInitProgress::NotStarted: return "NOT_STARTED";
        case EdmInitProgress::Started: return "STARTED";
        case EdmInitProgress::RemoteHandshakeComplete: return "REMOTE_HANDSHAKE_COMPLETE";
        case EdmInitProgress::LocalHandshakeComplete: return "LOCAL_HANDSHAKE_COMPLETE";
        case EdmInitProgress::ReadyForTraffic: return "READY_FOR_TRAFFIC";
        case EdmInitProgress::TerminatedIndeterminate: return "TERMINATED";
        case EdmInitProgress::Unknown: return "UNKNOWN";
    }
    return "UNKNOWN";
}

EdmInitProgress classify_edm_status(uint32_t raw_status) {
    switch (raw_status) {
        case 0: return EdmInitProgress::NotStarted;
        case static_cast<uint32_t>(EDMStatus::STARTED): return EdmInitProgress::Started;
        case static_cast<uint32_t>(EDMStatus::REMOTE_HANDSHAKE_COMPLETE):
            return EdmInitProgress::RemoteHandshakeComplete;
        case static_cast<uint32_t>(EDMStatus::LOCAL_HANDSHAKE_COMPLETE): return EdmInitProgress::LocalHandshakeComplete;
        case static_cast<uint32_t>(EDMStatus::READY_FOR_TRAFFIC): return EdmInitProgress::ReadyForTraffic;
        case static_cast<uint32_t>(EDMStatus::TERMINATED): return EdmInitProgress::TerminatedIndeterminate;
        default: return EdmInitProgress::Unknown;
    }
}

struct RouterStatusReport {
    chan_id_t chan;
    tt::umd::CoreCoord logical_core;
    uint32_t raw_status;
    EdmInitProgress stage;
    bool is_master;
};

// Build a human-readable hint explaining the most likely root cause for a
// router stuck at the given init stage. Returns an empty string when no
// stage-specific hint is available. Stages align with kernel writes to
// *edm_status_ptr in fabric_erisc_router.cpp.
std::string diagnostic_hint_for_stuck_stage(EdmInitProgress stuck_stage, uint32_t num_initialized_routers) {
    switch (stuck_stage) {
        case EdmInitProgress::NotStarted:
            return "Hint: router(s) never reached kernel main loop. The ERISC kernel may have failed "
                   "to launch, the kernel image may not have been loaded, or the core was wedged "
                   "before writing the STARTED status.";
        case EdmInitProgress::Started:
            return "Hint: router(s) likely entered the main function but did not complete the remote "
                   "(ethernet) handshake. Ethernet handshake likely failed -- the link may not be "
                   "healthy. Investigate link training state, cable integrity, and whether the "
                   "remote-side partner kernel is running and responsive.";
        case EdmInitProgress::RemoteHandshakeComplete:
            return fmt::format(
                "Hint: remote (ethernet) handshake completed but local (intra-device, master <-> "
                "subordinate) handshake did not. Master expects {} router(s) on this device. "
                "Likely causes: (1) master was configured with the wrong router count, so it is "
                "waiting on notifications from routers that will never signal; (2) one or more "
                "subordinate routers is corrupted or stuck before reaching the local-handshake "
                "notify step.",
                num_initialized_routers);
        case EdmInitProgress::LocalHandshakeComplete:
            return "Hint: local handshake completed but READY_FOR_TRAFFIC was never signaled. The "
                   "router is likely stuck waiting for local forwarded ethernet connections (over "
                   "NoC). Likely causes: (1) router was misprogrammed to expect a producer when it "
                   "has none; (2) handshake addresses were misconfigured; (3) producer core(s) were "
                   "misconfigured and are pointing to the wrong address, or are corrupted.";
        case EdmInitProgress::ReadyForTraffic:
        case EdmInitProgress::TerminatedIndeterminate:
        case EdmInitProgress::Unknown: return {};
    }
    return {};
}

[[noreturn]] void report_router_sync_timeout_and_throw(
    Device* dev,
    chan_id_t master_chan,
    const tt::umd::CoreCoord& master_core,
    uint32_t master_raw_status,
    uint32_t expected_status,
    uint32_t router_sync_address,
    uint32_t timeout_ms,
    tt::tt_fabric::ControlPlane& control_plane,
    Cluster& cluster) {
    const auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(dev->id());
    const auto active_channels = control_plane.get_active_fabric_eth_channels(fabric_node_id);
    const auto& soc_desc = cluster.get_soc_desc(dev->id());
    const uint32_t num_initialized_routers =
        control_plane.get_fabric_context().get_builder_context().get_num_fabric_initialized_routers(dev->id());

    std::vector<RouterStatusReport> reports;
    reports.reserve(active_channels.size());

    // Master's status was already read by the caller; reuse it.
    reports.push_back({
        master_chan,
        master_core,
        master_raw_status,
        classify_edm_status(master_raw_status),
        /*is_master=*/true,
    });

    for (const auto& [chan, _direction] : active_channels) {
        if (chan == master_chan) {
            continue;
        }
        auto core = soc_desc.get_eth_core_for_channel(chan, CoordSystem::LOGICAL);
        std::vector<uint32_t> status_buf{0};
        detail::ReadFromDeviceL1(dev, core, router_sync_address, 4, status_buf, CoreType::ETH);
        reports.push_back({chan, core, status_buf[0], classify_edm_status(status_buf[0]), /*is_master=*/false});
    }

    std::optional<EdmInitProgress> min_stage;
    for (const auto& r : reports) {
        if (has_comparable_progress(r.stage) && (!min_stage || r.stage < *min_stage)) {
            min_stage = r.stage;
        }
    }
    const RouterStatusReport* representative_laggard = nullptr;
    size_t laggard_count = 0;
    if (min_stage) {
        for (const auto& r : reports) {
            if (r.stage == *min_stage) {
                ++laggard_count;
                if (representative_laggard == nullptr) {
                    representative_laggard = &r;
                }
            }
        }
    }

    log_error(
        tt::LogMetal,
        "Fabric Router Sync: Timeout after {} ms on Device {}. Expected status 0x{:08x} ({}) at "
        "edm_status_address=0x{:08x}. Per-router status:",
        timeout_ms,
        dev->id(),
        expected_status,
        progress_name(classify_edm_status(expected_status)),
        router_sync_address);

    for (const auto& r : reports) {
        const bool is_laggard = min_stage && r.stage == *min_stage;
        log_error(
            tt::LogMetal,
            "  {} chan={:2d} logical={} status=0x{:08x} ({}){}",
            r.is_master ? "master" : "sub   ",
            r.chan,
            r.logical_core.str(),
            r.raw_status,
            progress_name(r.stage),
            is_laggard ? "  <-- least progress" : "");
    }

    std::string diagnostic_hint;
    if (min_stage) {
        log_error(
            tt::LogMetal,
            "Earliest init stage reached: {} ({} core(s) stuck at this stage).",
            progress_name(*min_stage),
            laggard_count);
        diagnostic_hint = diagnostic_hint_for_stuck_stage(*min_stage, num_initialized_routers);
        if (!diagnostic_hint.empty()) {
            log_error(tt::LogMetal, "{}", diagnostic_hint);
        }
    }

    const size_t terminated_count = std::count_if(reports.begin(), reports.end(), [](const auto& r) {
        return r.stage == EdmInitProgress::TerminatedIndeterminate;
    });
    const size_t unknown_count = std::count_if(
        reports.begin(), reports.end(), [](const auto& r) { return r.stage == EdmInitProgress::Unknown; });
    if (terminated_count > 0) {
        log_error(
            tt::LogMetal,
            "{} core(s) in TERMINATED state -- progress before termination is indeterminate.",
            terminated_count);
    }
    if (unknown_count > 0) {
        log_error(
            tt::LogMetal,
            "{} core(s) reported an unrecognized status value -- treated as UNKNOWN progress.",
            unknown_count);
    }

    const std::string laggard_summary = representative_laggard != nullptr
                                            ? fmt::format(
                                                  "furthest-behind stage: {} ({} core(s), e.g. chan={} got 0x{:08x})",
                                                  progress_name(*min_stage),
                                                  laggard_count,
                                                  representative_laggard->chan,
                                                  representative_laggard->raw_status)
                                            : std::string("no cores reached a comparable init stage");

    TT_THROW(
        "Fabric Router Sync: Timeout after {} ms on Device {}: expected status 0x{:08x}. "
        "Master chan={} got 0x{:08x}. {}.{}{} See error log above for per-core status breakdown.",
        timeout_ms,
        dev->id(),
        expected_status,
        master_chan,
        master_raw_status,
        laggard_summary,
        diagnostic_hint.empty() ? "" : " ",
        diagnostic_hint);
}

}  // namespace

FabricFirmwareInitializer::FabricFirmwareInitializer(
    std::shared_ptr<const ContextDescriptor> descriptor, tt::tt_fabric::ControlPlane& control_plane) :
    FirmwareInitializer(std::move(descriptor)), control_plane_(control_plane) {}

void FabricFirmwareInitializer::init(
    const std::vector<Device*>& devices, const std::unordered_set<InitializerKey>& /*init_done*/) {
    devices_ = devices;

    tt_fabric::FabricConfig fabric_config = descriptor_->fabric_config();
    if (!tt_fabric::is_tt_fabric_config(fabric_config)) {
        return;
    }

    if (descriptor_->is_mock_device()) {
        log_info(tt::LogMetal, "Skipping fabric initialization for mock devices");
        return;
    }

    if (has_flag(descriptor_->fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
        log_info(tt::LogMetal, "Initializing Fabric");
        control_plane_.write_routing_tables_to_all_chips();
        compile_and_configure_fabric();
        log_info(tt::LogMetal, "Fabric Initialized with config {}", fabric_config);
    } else if (has_flag(descriptor_->fabric_manager(), tt_fabric::FabricManagerMode::TERMINATE_FABRIC)) {
        log_info(tt::LogMetal, "Compiling fabric to setup fabric context for fabric termination");
        for (auto* dev : devices_) {
            dev->compile_fabric();
        }
    } else {
        log_info(tt::LogMetal, "Fabric initialized through Fabric Manager");
    }
}

void FabricFirmwareInitializer::configure() {
    if (has_flag(descriptor_->fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
        wait_for_fabric_router_sync(get_fabric_router_sync_timeout_ms());
    }
    initialized_.test_and_set();
}

void FabricFirmwareInitializer::teardown(std::unordered_set<InitializerKey>& init_done) {
    TT_FATAL(
        !init_done.contains(InitializerKey::Dispatch),
        "FabricFirmwareInitializer must be torn down after DispatchKernelInitializer");
    if (descriptor_->is_mock_device()) {
        log_info(tt::LogMetal, "Skipping fabric teardown for mock devices");
        init_done.erase(key);
        return;
    }
    if (!has_flag(descriptor_->fabric_manager(), tt_fabric::FabricManagerMode::TERMINATE_FABRIC)) {
        devices_.clear();
        initialized_.clear();
        init_done.erase(key);
        return;
    }

    tt_fabric::FabricConfig fabric_config = descriptor_->fabric_config();
    if (!tt_fabric::is_tt_fabric_config(fabric_config)) {
        devices_.clear();
        initialized_.clear();
        init_done.erase(key);
        return;
    }

    const auto& fabric_context = control_plane_.get_fabric_context();
    const auto& builder_ctx = fabric_context.get_builder_context();
    auto [termination_signal_address, signal] = builder_ctx.get_fabric_router_termination_address_and_signal();
    std::vector<uint32_t> termination_signal(1, signal);

    // Terminate fabric tensix mux cores if enabled
    // TODO: issue #26855, move the termination process to device
    if (descriptor_->fabric_tensix_config() != tt::tt_fabric::FabricTensixConfig::DISABLED) {
        const auto& tensix_config = builder_ctx.get_tensix_config();

        for (auto* dev : devices_) {
            if (builder_ctx.get_num_fabric_initialized_routers(dev->id()) == 0) {
                continue;
            }

            const auto fabric_node_id = control_plane_.get_fabric_node_id_from_physical_chip_id(dev->id());
            const auto& active_fabric_eth_channels = control_plane_.get_active_fabric_eth_channels(fabric_node_id);

            for (const auto& [eth_chan_id, direction] : active_fabric_eth_channels) {
                auto core_id = tensix_config.get_core_id_for_channel(dev->id(), eth_chan_id);
                auto [tensix_termination_address, tensix_signal] =
                    tensix_config.get_termination_address_and_signal(core_id);
                std::vector<uint32_t> tensix_termination_signal(1, tensix_signal);
                auto mux_core = tensix_config.get_core_for_channel(dev->id(), eth_chan_id);

                detail::WriteToDeviceL1(
                    dev, mux_core, tensix_termination_address, tensix_termination_signal, CoreType::WORKER);
            }

            cluster_.l1_barrier(dev->id());
        }
    }

    // Terminate fabric routers via master router on each device
    for (auto* dev : devices_) {
        if (builder_ctx.get_num_fabric_initialized_routers(dev->id()) == 0) {
            continue;
        }

        auto master_router_logical_core = cluster_.get_soc_desc(dev->id()).get_eth_core_for_channel(
            builder_ctx.get_fabric_master_router_chan(dev->id()), CoordSystem::LOGICAL);
        detail::WriteToDeviceL1(
            dev, master_router_logical_core, termination_signal_address, termination_signal, CoreType::ETH);
    }

    devices_.clear();
    initialized_.clear();
    init_done.erase(key);
}

void FabricFirmwareInitializer::post_teardown() {
    // Reset fabric config
    descriptor_->metal_context().set_fabric_config(tt::tt_fabric::FabricConfig::DISABLED);
}

bool FabricFirmwareInitializer::is_initialized() const { return initialized_.test(); }

void FabricFirmwareInitializer::compile_and_configure_fabric() {
    std::vector<std::shared_future<Device*>> events;
    events.reserve(devices_.size());
    for (auto* dev : devices_) {
        events.emplace_back(detail::async([dev]() {
            if (dev->compile_fabric()) {
                return dev;
            }
            // Compile failure mostly comes from Nebula (TG)
            log_trace(tt::LogMetal, "Did not build fabric on Device {}", dev->id());
            return static_cast<Device*>(nullptr);
        }));
    }

    if (!has_flag(descriptor_->fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
        return;
    }

    size_t configured_count = 0;
    for (const auto& event : events) {
        auto* dev = event.get();
        if (dev) {
            dev->configure_fabric();
            configured_count++;
        }
    }
    log_info(tt::LogMetal, "Fabric initialized on {} devices", configured_count);
}

void FabricFirmwareInitializer::wait_for_fabric_router_sync(uint32_t timeout_ms) const {
    tt_fabric::FabricConfig fabric_config = descriptor_->fabric_config();
    if (!tt_fabric::is_tt_fabric_config(fabric_config)) {
        return;
    }

    const auto& fabric_context = control_plane_.get_fabric_context();
    const auto& builder_context = fabric_context.get_builder_context();

    auto wait_for_handshake = [&](Device* dev) {
        if (!dev) {
            TT_THROW("Fabric router sync on null device. All devices must be opened for Fabric.");
        }
        if (builder_context.get_num_fabric_initialized_routers(dev->id()) == 0) {
            return;
        }

        const auto master_router_chan = builder_context.get_fabric_master_router_chan(dev->id());
        const auto master_router_logical_core =
            cluster_.get_soc_desc(dev->id()).get_eth_core_for_channel(master_router_chan, CoordSystem::LOGICAL);

        const auto [router_sync_address, expected_status] = builder_context.get_fabric_router_sync_address_and_status();
        std::vector<std::uint32_t> master_router_status{0};
        auto start_time = std::chrono::steady_clock::now();
        while (master_router_status[0] != expected_status) {
            detail::ReadFromDeviceL1(
                dev, master_router_logical_core, router_sync_address, 4, master_router_status, CoreType::ETH);
            if (master_router_status[0] == expected_status) {
                break;
            }
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
            if (elapsed_ms > timeout_ms) {
                report_router_sync_timeout_and_throw(
                    dev,
                    master_router_chan,
                    master_router_logical_core,
                    master_router_status[0],
                    expected_status,
                    router_sync_address,
                    timeout_ms,
                    control_plane_,
                    cluster_);
            }
        }

        auto ready_address_and_signal = builder_context.get_fabric_router_ready_address_and_signal();
        if (ready_address_and_signal) {
            std::vector<uint32_t> ready_signal(1, ready_address_and_signal->second);
            detail::WriteToDeviceL1(
                dev, master_router_logical_core, ready_address_and_signal->first, ready_signal, CoreType::ETH);
        }
    };

    // Poll devices in tunnel order: farthest-to-closest, then MMIO device itself
    for (auto* dev : devices_) {
        if (cluster_.get_associated_mmio_device(dev->id()) != dev->id()) {
            continue;
        }

        auto tunnels_from_mmio = cluster_.get_tunnels_from_mmio_device(dev->id());
        for (const auto& tunnel : tunnels_from_mmio) {
            for (auto j = tunnel.size() - 1; j > 0; j--) {
                // Find the device in our device list by chip ID
                auto it =
                    std::find_if(devices_.begin(), devices_.end(), [&](Device* d) { return d->id() == tunnel[j]; });
                if (it != devices_.end()) {
                    wait_for_handshake(*it);
                }
            }
        }

        wait_for_handshake(dev);
    }
}

uint32_t FabricFirmwareInitializer::get_fabric_router_sync_timeout_ms() const {
    if (rtoptions_.get_simulator_enabled()) {
        return 15000;
    }
    auto timeout = rtoptions_.get_fabric_router_sync_timeout_ms();
    return timeout.value_or(10000);
}

}  // namespace tt::tt_metal
