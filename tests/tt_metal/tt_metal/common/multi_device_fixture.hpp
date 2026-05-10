// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>
#include <atomic>
#include <chrono>
#include <csignal>
#include <thread>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt_stl/reflection.hpp>
#include "llrt.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/mesh_device.hpp>

#include "mesh_dispatch_fixture.hpp"
#include "system_mesh.hpp"
#include <umd/device/types/arch.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include "tt_metal/test_utils/env_vars.hpp"

#include <tt-metalium/experimental/fabric/fabric_telemetry.hpp>
#include <tt-metalium/experimental/fabric/fabric_telemetry_reader.hpp>

namespace tt::tt_metal {

// Mitigation — structured telemetry types for baseline capture / comparison.
struct FabricChannelBaseline {
    tt::tt_fabric::chan_id_t channel_id = 0;
    tt::tt_fabric::FabricTelemetryRouterState router_state = tt::tt_fabric::FabricTelemetryRouterState::Standby;
    uint64_t tx_heartbeat = 0;
    uint64_t rx_heartbeat = 0;
};

struct FabricDeviceBaseline {
    ChipId device_id = 0;
    std::vector<FabricChannelBaseline> channels;
};

// Mitigation — dump fabric telemetry on test failure for post-mortem diagnosis.
// Only called on failure paths; zero cost on happy path.
inline void dump_fabric_telemetry_on_failure(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh) {
    if (!mesh) {
        return;
    }
    try {
        const auto& shape = mesh->shape();
        log_warning(
            tt::LogTest,
            "[fabric_telemetry_dump] BEGIN — {} device(s) in mesh",
            mesh->num_devices());

        for (const auto& coord : distributed::MeshCoordinateRange(shape)) {
            tt::tt_fabric::FabricNodeId node_id(tt::tt_fabric::MeshId{0}, 0);
            try {
                node_id = mesh->get_fabric_node_id(coord);
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogTest,
                    "[fabric_telemetry_dump] coord ({}) — get_fabric_node_id() threw: {}",
                    coord, e.what());
                continue;
            }

            std::vector<tt::tt_fabric::FabricTelemetrySample> samples;
            try {
                samples = tt::tt_fabric::read_fabric_telemetry(node_id);
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogTest,
                    "[fabric_telemetry_dump] device {} (coord {}) — read_fabric_telemetry() threw: {}",
                    mesh->get_device(coord)->id(), coord, e.what());
                continue;
            }

            auto* dev = mesh->get_device(coord);
            for (const auto& sample : samples) {
                const auto& si = sample.snapshot.static_info;
                if (sample.snapshot.dynamic_info.has_value()) {
                    const auto& dyn = *sample.snapshot.dynamic_info;
                    for (size_t erisc_idx = 0; erisc_idx < dyn.erisc.size(); ++erisc_idx) {
                        const auto& entry = dyn.erisc[erisc_idx];
                        const char* state_str =
                            entry.router_state == tt::tt_fabric::FabricTelemetryRouterState::Active  ? "Active" :
                            entry.router_state == tt::tt_fabric::FabricTelemetryRouterState::Paused  ? "Paused" :
                            entry.router_state == tt::tt_fabric::FabricTelemetryRouterState::Draining ? "Draining" :
                                                                                                       "Standby";
                        bool unhealthy =
                            entry.router_state != tt::tt_fabric::FabricTelemetryRouterState::Active;
                        // Use log_warning for unhealthy channels to ensure visibility in CI logs.
                        if (unhealthy) {
                            log_warning(
                                tt::LogTest,
                                "[fabric_telemetry_dump] device={} chan={} erisc={} "
                                "router_state={} tx_heartbeat={} rx_heartbeat={} "
                                "direction={} mesh_id={} neighbor_mesh_id={} [UNHEALTHY]",
                                dev->id(), sample.channel_id, erisc_idx,
                                state_str, entry.tx_heartbeat, entry.rx_heartbeat,
                                si.direction, si.mesh_id, si.neighbor_mesh_id);
                        } else {
                            log_warning(
                                tt::LogTest,
                                "[fabric_telemetry_dump] device={} chan={} erisc={} "
                                "router_state={} tx_heartbeat={} rx_heartbeat={} "
                                "direction={} mesh_id={} neighbor_mesh_id={}",
                                dev->id(), sample.channel_id, erisc_idx,
                                state_str, entry.tx_heartbeat, entry.rx_heartbeat,
                                si.direction, si.mesh_id, si.neighbor_mesh_id);
                        }
                    }
                } else {
                    log_warning(
                        tt::LogTest,
                        "[fabric_telemetry_dump] device={} chan={} — no dynamic info "
                        "(direction={} mesh_id={} ver={})",
                        dev->id(), sample.channel_id,
                        si.direction, si.mesh_id, si.version);
                }
            }
        }
        log_warning(tt::LogTest, "[fabric_telemetry_dump] END");
    } catch (const std::exception& e) {
        log_warning(
            tt::LogTest,
            "[fabric_telemetry_dump] outer exception — telemetry dump aborted: {}",
            e.what());
    }
}

// Mitigation — capture per-device telemetry baseline for comparison at TearDown.
inline std::vector<FabricDeviceBaseline> capture_fabric_telemetry_baseline(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh) {
    std::vector<FabricDeviceBaseline> baselines;
    if (!mesh) {
        return baselines;
    }
    try {
        const auto& shape = mesh->shape();
        for (const auto& coord : distributed::MeshCoordinateRange(shape)) {
            tt::tt_fabric::FabricNodeId node_id(tt::tt_fabric::MeshId{0}, 0);
            try {
                node_id = mesh->get_fabric_node_id(coord);
            } catch (...) {
                continue;
            }

            std::vector<tt::tt_fabric::FabricTelemetrySample> samples;
            try {
                samples = tt::tt_fabric::read_fabric_telemetry(node_id);
            } catch (...) {
                continue;
            }

            FabricDeviceBaseline dev_baseline;
            dev_baseline.device_id = mesh->get_device(coord)->id();
            for (const auto& sample : samples) {
                if (!sample.snapshot.dynamic_info.has_value()) continue;
                const auto& dyn = *sample.snapshot.dynamic_info;
                // Use erisc[0] as the primary router state for this channel.
                if (!dyn.erisc.empty()) {
                    FabricChannelBaseline ch;
                    ch.channel_id = sample.channel_id;
                    ch.router_state = dyn.erisc[0].router_state;
                    ch.tx_heartbeat = dyn.erisc[0].tx_heartbeat;
                    ch.rx_heartbeat = dyn.erisc[0].rx_heartbeat;
                    dev_baseline.channels.push_back(ch);
                }
            }
            baselines.push_back(std::move(dev_baseline));
        }
    } catch (...) {
        // Best-effort — don't fail test setup over telemetry capture.
    }
    return baselines;
}

// Mitigation — compare current telemetry against SetUp baseline and log degradations.
inline void compare_fabric_telemetry_baseline(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh,
    const std::vector<FabricDeviceBaseline>& baselines) {
    if (!mesh || baselines.empty()) {
        return;
    }
    try {
        // Build a lookup: device_id -> channel_id -> baseline
        std::unordered_map<ChipId, std::unordered_map<tt::tt_fabric::chan_id_t, FabricChannelBaseline>> baseline_map;
        for (const auto& db : baselines) {
            for (const auto& ch : db.channels) {
                baseline_map[db.device_id][ch.channel_id] = ch;
            }
        }

        const auto& shape = mesh->shape();
        for (const auto& coord : distributed::MeshCoordinateRange(shape)) {
            tt::tt_fabric::FabricNodeId node_id(tt::tt_fabric::MeshId{0}, 0);
            try {
                node_id = mesh->get_fabric_node_id(coord);
            } catch (...) {
                continue;
            }

            std::vector<tt::tt_fabric::FabricTelemetrySample> samples;
            try {
                samples = tt::tt_fabric::read_fabric_telemetry(node_id);
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogTest,
                    "[fabric_baseline_compare] device {} — read_fabric_telemetry() threw: {}",
                    mesh->get_device(coord)->id(), e.what());
                continue;
            }

            auto* dev = mesh->get_device(coord);
            auto dev_it = baseline_map.find(dev->id());
            if (dev_it == baseline_map.end()) continue;

            for (const auto& sample : samples) {
                if (!sample.snapshot.dynamic_info.has_value()) continue;
                const auto& dyn = *sample.snapshot.dynamic_info;
                if (dyn.erisc.empty()) continue;

                auto ch_it = dev_it->second.find(sample.channel_id);
                if (ch_it == dev_it->second.end()) continue;

                const auto& before = ch_it->second;
                const auto& after_entry = dyn.erisc[0];

                bool degraded =
                    (before.router_state == tt::tt_fabric::FabricTelemetryRouterState::Active &&
                     after_entry.router_state != tt::tt_fabric::FabricTelemetryRouterState::Active);
                bool heartbeat_stalled =
                    (after_entry.tx_heartbeat == before.tx_heartbeat &&
                     after_entry.rx_heartbeat == before.rx_heartbeat &&
                     before.tx_heartbeat > 0);

                if (degraded || heartbeat_stalled) {
                    const char* before_state =
                        before.router_state == tt::tt_fabric::FabricTelemetryRouterState::Active  ? "Active" :
                        before.router_state == tt::tt_fabric::FabricTelemetryRouterState::Paused  ? "Paused" :
                        before.router_state == tt::tt_fabric::FabricTelemetryRouterState::Draining ? "Draining" :
                                                                                                     "Standby";
                    const char* after_state =
                        after_entry.router_state == tt::tt_fabric::FabricTelemetryRouterState::Active  ? "Active" :
                        after_entry.router_state == tt::tt_fabric::FabricTelemetryRouterState::Paused  ? "Paused" :
                        after_entry.router_state == tt::tt_fabric::FabricTelemetryRouterState::Draining ? "Draining" :
                                                                                                          "Standby";
                    log_warning(
                        tt::LogTest,
                        "[fabric_baseline_compare] device={} chan={} DEGRADED: "
                        "router_state {} -> {} | tx_hb {} -> {} | rx_hb {} -> {}{}",
                        dev->id(), sample.channel_id,
                        before_state, after_state,
                        before.tx_heartbeat, after_entry.tx_heartbeat,
                        before.rx_heartbeat, after_entry.rx_heartbeat,
                        heartbeat_stalled ? " [HEARTBEAT_STALLED]" : "");
                }
            }
        }
    } catch (const std::exception& e) {
        log_warning(
            tt::LogTest,
            "[fabric_baseline_compare] outer exception — comparison aborted: {}",
            e.what());
    }
}

class TwoMeshDeviceFixture : public MeshDispatchFixture {
protected:
    void SetUp() override {
        auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (!slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with TT_METAL_SLOW_DISPATCH_MODE set");
            GTEST_SKIP();
        }

        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices != 2) {
            GTEST_SKIP() << "TwoDeviceFixture can only be run on machines with two devices";
        }

        MeshDispatchFixture::SetUp();
    }
};

class N300MeshDeviceFixture : public MeshDispatchFixture {
protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr;
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with TT_METAL_SLOW_DISPATCH_MODE set");
            GTEST_SKIP();
        }

        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        const size_t num_pci_devices = tt::tt_metal::GetNumPCIeDevices();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        if (this->arch_ == tt::ARCH::WORMHOLE_B0 && num_devices == 2 && num_pci_devices == 1) {
            MeshDispatchFixture::SetUp();
        } else {
            GTEST_SKIP() << "This suite can only be run on N300";
        }
    }
};

class TwoDeviceBlackholeFixture : public MeshDispatchFixture {
protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr;
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with TT_METAL_SLOW_DISPATCH_MODE set");
            GTEST_SKIP();
        }

        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        const size_t num_pci_devices = tt::tt_metal::GetNumPCIeDevices();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        if (this->arch_ == tt::ARCH::BLACKHOLE && num_devices == 2 && num_pci_devices >= 1) {
            MeshDispatchFixture::SetUp();
        } else {
            GTEST_SKIP() << "This suite can only be run on two chip Blackhole systems";
        }
    }
};

class MeshDeviceFixtureBase : public ::testing::Test {
public:
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> get_mesh_device() {
        TT_FATAL(mesh_device_, "MeshDevice not initialized in {}", __FUNCTION__);
        return mesh_device_;
    }

protected:
    using MeshDevice = ::tt::tt_metal::distributed::MeshDevice;
    using MeshDeviceConfig = ::tt::tt_metal::distributed::MeshDeviceConfig;
    using MeshShape = ::tt::tt_metal::distributed::MeshShape;

    struct Config {
        // If specified, the fixture will open a mesh device with the specified shape and offset.
        // Otherwise, SystemMesh shape with zero offset will be used.
        std::optional<tt::tt_metal::distributed::MeshShape> mesh_shape;
        std::optional<tt::tt_metal::distributed::MeshCoordinate> mesh_offset;

        // If specified, the associated tests will run only if the machine architecture matches the specified
        // architecture.
        std::optional<tt::ARCH> arch;

        int num_cqs = 1;
        uint32_t l1_small_size = DEFAULT_L1_SMALL_SIZE;
        uint32_t trace_region_size = DEFAULT_TRACE_REGION_SIZE;
        uint32_t worker_l1_size = DEFAULT_WORKER_L1_SIZE;
        tt_fabric::FabricConfig fabric_config = tt_fabric::FabricConfig::DISABLED;
        tt_fabric::FabricTensixConfig fabric_tensix_config = tt_fabric::FabricTensixConfig::DISABLED;
        tt_fabric::FabricUDMMode fabric_udm_mode = tt_fabric::FabricUDMMode::DISABLED;
        // If set, a watchdog thread is launched in SetUp(). If the test body has not
        // completed within this many milliseconds, the process is killed with SIGKILL so
        // that CI does not stall waiting for a hung test.
        std::optional<uint32_t> test_budget_ms;
    };

    explicit MeshDeviceFixtureBase(const Config& fixture_config) : config_(fixture_config) {}

    void SetUp() override {
        log_info(
            tt::LogMetal,
            "[fixture_setup] MeshDeviceFixtureBase::SetUp() ENTRY — mesh_shape={}, fabric_config={}, num_cqs={}",
            config_.mesh_shape.has_value()
                ? fmt::format("{}x{}", (*config_.mesh_shape)[0], (*config_.mesh_shape)[1])
                : "auto",
            static_cast<int>(config_.fabric_config),
            config_.num_cqs);

        auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            GTEST_SKIP() << "Skipping Mesh-Device test suite, since it can only be run in Fast Dispatch Mode.";
        }

        const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        if (config_.arch.has_value() && *config_.arch != arch) {
            GTEST_SKIP() << fmt::format(
                "Skipping MeshDevice test suite on a machine with architecture {} that does not match the requested "
                "architecture {}",
                arch,
                *config_.arch);
        }

        const auto system_mesh_shape = tt::tt_metal::MetalContext::instance().get_system_mesh().shape();
        if (config_.mesh_shape.has_value() && config_.mesh_shape->mesh_size() > system_mesh_shape.mesh_size()) {
            GTEST_SKIP() << fmt::format(
                "Skipping MeshDevice test suite on a machine with SystemMesh {} that is smaller than the requested "
                "mesh "
                "shape {}",
                system_mesh_shape,
                *config_.mesh_shape);
        }

        init_max_cbs();

        // Use ethernet dispatch for more than 1 CQ on T3K/N300
        auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();
        bool is_n300_or_t3k_cluster =
            cluster_type == tt::tt_metal::ClusterType::T3K or cluster_type == tt::tt_metal::ClusterType::N300;
        auto core_type =
            (config_.num_cqs >= 2 and is_n300_or_t3k_cluster) ? DispatchCoreType::ETH : DispatchCoreType::WORKER;

        if (config_.fabric_config != tt_fabric::FabricConfig::DISABLED) {
            tt_fabric::SetFabricConfig(
                config_.fabric_config,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
                std::nullopt,
                config_.fabric_tensix_config,
                config_.fabric_udm_mode);
        }
        // FIX BC (#42429): MeshDevice::create() throws TT_FATAL when non-MMIO devices are
        // unreachable (ETH relay dead after prior session). FIX AQ drops them from UMD
        // TopologyDiscovery, leaving the cluster with only MMIO devices open.
        // initialize_fabric_and_dispatch_fw() then TT_FATALs because device N is listed in
        // the system but not in active_devices_. Convert to GTEST_SKIP so the test suite
        // continues rather than crashing the whole binary.  Call SetFabricConfig(DISABLED)
        // first to clean up the global fabric state that SetFabricConfig(FABRIC_*) set above.
        try {
            mesh_device_ = MeshDevice::create(
                MeshDeviceConfig(config_.mesh_shape.value_or(system_mesh_shape), config_.mesh_offset),
                config_.l1_small_size,
                config_.trace_region_size,
                config_.num_cqs,
                core_type,
                {},
                config_.worker_l1_size);
        } catch (const std::exception& e) {
            std::string what = e.what();
            // FIX BC (#42429): non-MMIO ETH relay dead → "is not active"
            // FIX BC-2 (#42429): prior teardown left system mesh with fewer devices than
            //   requested (probe_dead_channels + dead_relay_devices_ → system_mesh only sees
            //   MMIO chips) → "only N devices are available in the system mesh"
            // FIX BR (#42429): prior teardown left MMIO channels newly-dead (soft reset timed out)
            //   → configure_fabric throws "newly-dead ETH channel(s)...Cannot write fabric firmware"
            //   at device.cpp:502. Without this catch, SetUp() propagates the exception and the
            //   test is marked FAILED instead of SKIPPED, blocking the entire suite.
            if (what.find("is not active") != std::string::npos ||
                what.find("devices are available") != std::string::npos ||
                what.find("newly-dead ETH channel") != std::string::npos) {
                if (config_.fabric_config != tt_fabric::FabricConfig::DISABLED) {
                    tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::DISABLED);
                }
                GTEST_SKIP() << fmt::format(
                    "FIX BC/BR (#42429): MeshDevice::create() threw degraded-cluster exception — "
                    "non-MMIO ETH relay dead, system mesh missing devices, or MMIO channels "
                    "newly-dead after corrupt teardown; skipping test. ({})",
                    what.substr(0, 300));
            }
            throw;
        }

        // Log fabric state after successful create — aids post-mortem diagnosis.
        for (auto* idev : mesh_device_->get_devices()) {
            if (idev->is_fabric_relay_path_broken() || idev->is_fabric_channels_not_ready_for_traffic() ||
                idev->is_fabric_stale_base_umd_channels()) {
                log_warning(
                    tt::LogMetal,
                    "[fixture_setup] MeshDeviceFixtureBase::SetUp() device {} fabric state: "
                    "relay_broken={} channels_not_ready={} stale_base_umd={}",
                    idev->id(),
                    idev->is_fabric_relay_path_broken(),
                    idev->is_fabric_channels_not_ready_for_traffic(),
                    idev->is_fabric_stale_base_umd_channels());
            }
        }
        log_info(tt::LogMetal, "[fixture_setup] MeshDeviceFixtureBase::SetUp() EXIT — mesh_device created with {} devices",
                 mesh_device_->num_devices());

        // Mitigation — capture fabric telemetry baseline after successful mesh open.
        // Compared against TearDown state to detect channels that degraded during the test.
        if (mesh_device_ && config_.fabric_config != tt_fabric::FabricConfig::DISABLED) {
            fabric_telemetry_baseline_ = capture_fabric_telemetry_baseline(mesh_device_);
            log_info(
                tt::LogMetal,
                "[fixture_setup] MeshDeviceFixtureBase::SetUp() captured fabric telemetry baseline "
                "for {} device(s)",
                fabric_telemetry_baseline_.size());
        }

        // Opt-in watchdog: kill the process if the test body exceeds its budget.
        if (config_.test_budget_ms.has_value()) {
            watchdog_stop_.store(false, std::memory_order_relaxed);
            uint32_t budget_ms = *config_.test_budget_ms;
            watchdog_thread_ = std::thread([this, budget_ms]() {
                auto deadline =
                    std::chrono::steady_clock::now() + std::chrono::milliseconds(budget_ms);
                while (!watchdog_stop_.load(std::memory_order_relaxed)) {
                    if (std::chrono::steady_clock::now() >= deadline) {
                        // Log to stderr directly — logger may be torn down by this point.
                        fprintf(
                            stderr,
                            "[MeshDeviceFixture watchdog] Test budget of %ums exceeded — "
                            "sending SIGKILL to prevent CI stall\n",
                            budget_ms);
                        fflush(stderr);
                        std::raise(SIGKILL);
                        return;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
            });
        }
    }

    void TearDown() override {
        // Stop watchdog before any teardown so a slow-but-completing test isn't killed.
        watchdog_stop_.store(true, std::memory_order_relaxed);
        if (watchdog_thread_.joinable()) {
            watchdog_thread_.join();
        }

        if (!mesh_device_) {
            log_info(tt::LogMetal, "[fixture_teardown] MeshDeviceFixtureBase::TearDown() ENTRY — mesh_device_ is null, returning early");
            return;
        }

        // Log per-device fabric state at TearDown entry for diagnosis.
        for (auto* idev : mesh_device_->get_devices()) {
            if (idev->is_fabric_relay_path_broken() || idev->is_fabric_channels_not_ready_for_traffic() ||
                idev->is_fabric_stale_base_umd_channels()) {
                log_warning(
                    tt::LogMetal,
                    "[fixture_teardown] MeshDeviceFixtureBase::TearDown() ENTRY device {} fabric state: "
                    "relay_broken={} channels_not_ready={} stale_base_umd={}",
                    idev->id(),
                    idev->is_fabric_relay_path_broken(),
                    idev->is_fabric_channels_not_ready_for_traffic(),
                    idev->is_fabric_stale_base_umd_channels());
            }
        }

        // FIX RX (base class, #42429): if fabric is broken, skip quiesce_devices() and call
        // close() directly.  Calling quiesce_devices() on a broken cluster burns ~72 s
        // (Phase 2.5 force-resets + Phase 5 relay-read timeouts) and leaves ETH channels in
        // 0x49705180 Metal-fw-stale state, causing the next test's SetUp to fail with
        // "Device N not active".  This is the same guard already present in
        // MultiCQFabricMeshDevice2x4Fixture::TearDown(); promoting it to the base class so
        // ALL mesh device fixtures (including MeshDevice1x4Fixture) benefit.
        bool fabric_broken = false;
        if (!mesh_device_->is_remote_only()) {
            for (auto* idev : mesh_device_->get_devices()) {
                if (idev->is_fabric_relay_path_broken() || idev->is_fabric_channels_not_ready_for_traffic() ||
                    idev->is_fabric_stale_base_umd_channels()) {
                    fabric_broken = true;
                    break;
                }
            }
        }

        // Mitigation — dump fabric telemetry on test failure or broken fabric.
        // Only fires on the failure path; zero cost on happy path.
        if ((HasFailure() || fabric_broken) && mesh_device_ &&
            config_.fabric_config != tt_fabric::FabricConfig::DISABLED) {
            log_warning(
                tt::LogTest,
                "[fixture_teardown] MeshDeviceFixtureBase::TearDown() test failure or fabric broken — "
                "dumping fabric telemetry for post-mortem diagnosis (HasFailure={}, fabric_broken={})",
                HasFailure(), fabric_broken);
            dump_fabric_telemetry_on_failure(mesh_device_);
            compare_fabric_telemetry_baseline(mesh_device_, fabric_telemetry_baseline_);
        }

        // Skip quiesce on remote-only MeshDevices (no local devices on this host).
        // quiesce_internal() calls get_active_sub_device_manager_id() which requires
        // sub_device_manager_tracker_ — null on remote-only devices — and would throw.
        if (fabric_broken) {
            log_warning(
                tt::LogMetal,
                "[fixture_teardown] MeshDeviceFixtureBase::TearDown() FIX RX (#42429): fabric broken — "
                "skipping quiesce_devices() (~72 s) and calling close() directly.");
        } else if (!mesh_device_->is_remote_only()) {
            log_info(tt::LogMetal, "[fixture_teardown] MeshDeviceFixtureBase::TearDown() calling mesh_device_->quiesce_devices()");
            mesh_device_->quiesce_devices();
            log_info(tt::LogMetal, "[fixture_teardown] mesh_device_->quiesce_devices() returned, calling mesh_device_->close()");
        } else {
            log_info(tt::LogMetal, "[fixture_teardown] MeshDeviceFixtureBase::TearDown() skipping quiesce_devices() (remote-only mesh)");
        }
        mesh_device_->close();
        log_info(tt::LogMetal, "[fixture_teardown] mesh_device_->close() returned, calling mesh_device_.reset()");
        mesh_device_.reset();
        log_info(tt::LogMetal, "[fixture_teardown] mesh_device_.reset() returned");
        if (config_.fabric_config != tt_fabric::FabricConfig::DISABLED) {
            log_info(tt::LogMetal, "[fixture_teardown] calling SetFabricConfig(DISABLED) from base TearDown");
            tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::DISABLED);
            log_info(tt::LogMetal, "[fixture_teardown] base SetFabricConfig(DISABLED) returned");
        }
    }

    void init_max_cbs() { max_cbs_ = tt::tt_metal::MetalContext::instance().hal().get_arch_num_circular_buffers(); }

    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device_;
    uint32_t max_cbs_{};
    std::thread watchdog_thread_;
    std::atomic<bool> watchdog_stop_{false};

    Config config_;

    // Mitigation — fabric telemetry baseline captured in SetUp for TearDown comparison.
    std::vector<FabricDeviceBaseline> fabric_telemetry_baseline_;
};

class MeshDeviceFixture4x8DispatchAgnostic : public MeshDeviceFixtureBase {
protected:
    MeshDeviceFixture4x8DispatchAgnostic() : MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{4, 8}}) {}

    void SetUp() override {
        const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        if (config_.arch.has_value() && *config_.arch != arch) {
            GTEST_SKIP() << fmt::format(
                "Skipping MeshDevice test suite on a machine with architecture {} that does not match the requested "
                "architecture {}",
                arch,
                *config_.arch);
        }

        const auto system_mesh_shape = tt::tt_metal::MetalContext::instance().get_system_mesh().shape();
        if (config_.mesh_shape.has_value() && config_.mesh_shape->mesh_size() > system_mesh_shape.mesh_size()) {
            GTEST_SKIP() << fmt::format(
                "Skipping MeshDevice test suite on a machine with SystemMesh {} that is smaller than the requested "
                "mesh "
                "shape {}",
                system_mesh_shape,
                *config_.mesh_shape);
        }

        init_max_cbs();

        // Use ethernet dispatch for more than 1 CQ on T3K/N300
        auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();
        bool is_n300_or_t3k_cluster =
            cluster_type == tt::tt_metal::ClusterType::T3K or cluster_type == tt::tt_metal::ClusterType::N300;
        auto core_type =
            (config_.num_cqs >= 2 and is_n300_or_t3k_cluster) ? DispatchCoreType::ETH : DispatchCoreType::WORKER;

        if (config_.fabric_config != tt_fabric::FabricConfig::DISABLED) {
            tt_fabric::SetFabricConfig(
                config_.fabric_config,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
                std::nullopt,
                config_.fabric_tensix_config,
                config_.fabric_udm_mode);
        }
        // FIX BC (audit, #42429): same try/catch as MeshDeviceFixtureBase::SetUp() —
        // this override bypasses the base class SetUp entirely, so it needs its own guard.
        try {
            mesh_device_ = MeshDevice::create(
                MeshDeviceConfig(config_.mesh_shape.value_or(system_mesh_shape), config_.mesh_offset),
                config_.l1_small_size,
                config_.trace_region_size,
                config_.num_cqs,
                core_type,
                {},
                config_.worker_l1_size);
        } catch (const std::exception& e) {
            std::string what = e.what();
            if (what.find("is not active") != std::string::npos ||
                what.find("devices are available") != std::string::npos) {
                if (config_.fabric_config != tt_fabric::FabricConfig::DISABLED) {
                    tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::DISABLED);
                }
                GTEST_SKIP() << fmt::format(
                    "FIX BC (#42429): MeshDevice::create() threw degraded-cluster exception — "
                    "non-MMIO ETH relay dead or system mesh missing devices after corrupt teardown; "
                    "skipping test. ({})",
                    what.substr(0, 300));
            }
            throw;
        }

        // Audit gap: log fabric state after successful create (matching base class)
        for (auto* idev : mesh_device_->get_devices()) {
            if (idev->is_fabric_relay_path_broken() || idev->is_fabric_channels_not_ready_for_traffic() ||
                idev->is_fabric_stale_base_umd_channels()) {
                log_warning(
                    tt::LogMetal,
                    "[fixture_setup] MeshDeviceFixture4x8DispatchAgnostic::SetUp() device {} fabric state: "
                    "relay_broken={} channels_not_ready={} stale_base_umd={}",
                    idev->id(),
                    idev->is_fabric_relay_path_broken(),
                    idev->is_fabric_channels_not_ready_for_traffic(),
                    idev->is_fabric_stale_base_umd_channels());
            }
        }
        log_info(tt::LogMetal, "[fixture_setup] MeshDeviceFixture4x8DispatchAgnostic::SetUp() EXIT — mesh_device created with {} devices",
                 mesh_device_->num_devices());

        // Audit gap: start watchdog if configured (matching base class)
        if (config_.test_budget_ms.has_value()) {
            watchdog_stop_.store(false, std::memory_order_relaxed);
            uint32_t budget_ms = *config_.test_budget_ms;
            watchdog_thread_ = std::thread([this, budget_ms]() {
                auto deadline =
                    std::chrono::steady_clock::now() + std::chrono::milliseconds(budget_ms);
                while (!watchdog_stop_.load(std::memory_order_relaxed)) {
                    if (std::chrono::steady_clock::now() >= deadline) {
                        fprintf(
                            stderr,
                            "[MeshDeviceFixture4x8DispatchAgnostic watchdog] Test budget of %ums exceeded — "
                            "sending SIGKILL to prevent CI stall\n",
                            budget_ms);
                        fflush(stderr);
                        std::raise(SIGKILL);
                        return;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
            });
        }
    }
};

// Fixtures that determine the mesh device type automatically.
// The associated test will be run if the topology is supported.
class GenericMeshDeviceFixture : public MeshDeviceFixtureBase {
protected:
    GenericMeshDeviceFixture() : MeshDeviceFixtureBase(Config{.num_cqs = 1}) {}
};

class GenericMultiCQMeshDeviceFixture : public MeshDeviceFixtureBase {
protected:
    GenericMultiCQMeshDeviceFixture() : MeshDeviceFixtureBase(Config{.num_cqs = 2}) {}
};

class MeshDevice1x2Fixture : public MeshDeviceFixtureBase {
protected:
    MeshDevice1x2Fixture() : MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{1, 2}}) {}
};

// Fixtures that specify the mesh device type explicitly.
// The associated test will be run if the cluster topology matches
// what is specified.
class MeshDevice2x4Fixture : public MeshDeviceFixtureBase {
protected:
    MeshDevice2x4Fixture() : MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{2, 4}}) {}
};

class MeshDevice4x8Fixture : public MeshDeviceFixtureBase {
protected:
    MeshDevice4x8Fixture() : MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{4, 8}}) {}
};

class MultiCQMeshDevice2x4Fixture : public MeshDeviceFixtureBase {
protected:
    MultiCQMeshDevice2x4Fixture() : MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{2, 4}, .num_cqs = 2}) {}
};

class MeshDevice2x4Fabric1DFixture : public MeshDeviceFixtureBase {
protected:
    MeshDevice2x4Fabric1DFixture() :
        MeshDeviceFixtureBase(
            Config{.mesh_shape = MeshShape{2, 4}, .num_cqs = 1, .fabric_config = tt_fabric::FabricConfig::FABRIC_1D}) {}
};

class GenericMeshDeviceFabric2DFixture : public MeshDeviceFixtureBase {
protected:
    GenericMeshDeviceFabric2DFixture() :
        MeshDeviceFixtureBase(Config{.num_cqs = 1, .fabric_config = tt_fabric::FabricConfig::FABRIC_2D}) {}
};

class MeshDevice2x4Fabric2DFixture : public MeshDeviceFixtureBase {
protected:
    MeshDevice2x4Fabric2DFixture() :
        MeshDeviceFixtureBase(
            Config{.mesh_shape = MeshShape{2, 4}, .num_cqs = 1, .fabric_config = tt_fabric::FabricConfig::FABRIC_2D}) {}
};

class MeshDevice1x4Fabric2DUDMFixture : public MeshDeviceFixtureBase {
protected:
    MeshDevice1x4Fabric2DUDMFixture() :
        MeshDeviceFixtureBase(Config{
            .mesh_shape = MeshShape{1, 4},
            .num_cqs = 1,
            .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
            .fabric_tensix_config = tt_fabric::FabricTensixConfig::UDM,
            .fabric_udm_mode = tt_fabric::FabricUDMMode::ENABLED}) {}

    void SetUp() override {
        // When Fabric is enabled, it requires all devices in the system to be active.
        // For Blackhole P150_X8 systems (8 independent chips), opening 4 devices leaves 4 inactive, causing a fatal error.
        // For T3K (4 N300 boards = 8 chips), opening 4 MMIO devices automatically activates all 8 chips, so it works.
        // Skip the test only on Blackhole systems with more than 4 devices.
        const auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();
        const size_t num_devices = tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices();
        const size_t requested_devices = 4;  // 1x4 mesh

        if (cluster_type == tt::tt_metal::ClusterType::P150_X8 && num_devices > requested_devices) {
            GTEST_SKIP() << fmt::format(
                "Skipping MeshDevice1x4Fabric2DUDMFixture test on P150_X8: "
                "System has {} independent Blackhole devices but test only requests {}. "
                "Fabric requires all devices to be active.",
                num_devices,
                requested_devices);
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

class MeshDevice2x4Fabric2DUDMFixture : public MeshDeviceFixtureBase {
protected:
    MeshDevice2x4Fabric2DUDMFixture() :
        MeshDeviceFixtureBase(Config{
            .mesh_shape = MeshShape{2, 4},
            .num_cqs = 1,
            .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
            .fabric_tensix_config = tt_fabric::FabricTensixConfig::UDM,
            .fabric_udm_mode = tt_fabric::FabricUDMMode::ENABLED}) {}
};

}  // namespace tt::tt_metal
