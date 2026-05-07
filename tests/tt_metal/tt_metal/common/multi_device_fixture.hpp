// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fmt/format.h>

#include <gtest/gtest.h>
#include <chrono>
#include <map>
#include <optional>
#include <string>
#include <thread>

#include <boost/algorithm/string.hpp>

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/fabric_telemetry.hpp>
#include <tt-metalium/experimental/fabric/fabric_telemetry_reader.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/reflection.hpp>
#include "llrt.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/mesh_device.hpp>

#include "mesh_dispatch_fixture.hpp"
#include "system_mesh.hpp"
#include <umd/device/types/arch.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include "tt_metal/test_utils/env_vars.hpp"

namespace tt::tt_metal {

// Configuration for mesh device test fixtures (shared by MeshDeviceFixtureBase and suite-level shared fixtures).
struct MeshDeviceFixtureConfig {
    std::optional<tt::tt_metal::distributed::MeshShape> mesh_shape;
    std::optional<tt::tt_metal::distributed::MeshCoordinate> mesh_offset;
    std::optional<tt::ARCH> arch;

    int num_cqs = 1;
    uint32_t l1_small_size = DEFAULT_L1_SMALL_SIZE;
    uint32_t trace_region_size = DEFAULT_TRACE_REGION_SIZE;
    uint32_t worker_l1_size = DEFAULT_WORKER_L1_SIZE;
    tt_fabric::FabricConfig fabric_config = tt_fabric::FabricConfig::DISABLED;
    tt_fabric::FabricTensixConfig fabric_tensix_config = tt_fabric::FabricTensixConfig::DISABLED;
    tt_fabric::FabricUDMMode fabric_udm_mode = tt_fabric::FabricUDMMode::DISABLED;
};

namespace mesh_device_shared_detail {

inline std::optional<std::string> mesh_fixture_skip_reason(const MeshDeviceFixtureConfig& config) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
        return std::string("Skipping Mesh-Device test suite, since it can only be run in Fast Dispatch Mode.");
    }

    const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    if (config.arch.has_value() && *config.arch != arch) {
        return fmt::format(
            "Skipping MeshDevice test suite on a machine with architecture {} that does not match the requested "
            "architecture {}",
            arch,
            *config.arch);
    }

    const auto system_mesh_shape = tt::tt_metal::MetalContext::instance().get_system_mesh().shape();
    if (config.mesh_shape.has_value() && config.mesh_shape->mesh_size() > system_mesh_shape.mesh_size()) {
        return fmt::format(
            "Skipping MeshDevice test suite on a machine with SystemMesh {} that is smaller than the requested "
            "mesh "
            "shape {}",
            system_mesh_shape,
            *config.mesh_shape);
    }

    // P150_X8 systems have independent Blackhole chips — fabric requires all devices active,
    // so skip when the fixture requests fewer devices than the system has.
    if (config.fabric_config != tt_fabric::FabricConfig::DISABLED && config.mesh_shape.has_value()) {
        const auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();
        const size_t num_devices = tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices();
        if (cluster_type == tt::tt_metal::ClusterType::P150_X8 &&
            num_devices > config.mesh_shape->mesh_size()) {
            return fmt::format(
                "Skipping on P150_X8: system has {} independent Blackhole devices but fixture requests {}. "
                "Fabric requires all devices to be active.",
                num_devices,
                config.mesh_shape->mesh_size());
        }
    }

    return std::nullopt;
}

inline std::shared_ptr<::tt::tt_metal::distributed::MeshDevice> mesh_fixture_open(
    const MeshDeviceFixtureConfig& config) {
    const auto system_mesh_shape = tt::tt_metal::MetalContext::instance().get_system_mesh().shape();
    auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();
    bool is_n300_or_t3k_cluster =
        cluster_type == tt::tt_metal::ClusterType::T3K || cluster_type == tt::tt_metal::ClusterType::N300;
    auto core_type =
        (config.num_cqs >= 2 && is_n300_or_t3k_cluster) ? DispatchCoreType::ETH : DispatchCoreType::WORKER;

    if (config.fabric_config != tt_fabric::FabricConfig::DISABLED) {
        tt_fabric::SetFabricConfig(
            config.fabric_config,
            tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
            std::nullopt,
            config.fabric_tensix_config,
            config.fabric_udm_mode);
    }
    return ::tt::tt_metal::distributed::MeshDevice::create(
        ::tt::tt_metal::distributed::MeshDeviceConfig(
            config.mesh_shape.value_or(system_mesh_shape), config.mesh_offset),
        config.l1_small_size,
        config.trace_region_size,
        config.num_cqs,
        core_type,
        {},
        config.worker_l1_size);
}

inline void mesh_fixture_close(
    std::shared_ptr<::tt::tt_metal::distributed::MeshDevice>& mesh, const MeshDeviceFixtureConfig& config) {
    if (!mesh) {
        return;
    }
    mesh->close();
    mesh.reset();
    if (config.fabric_config != tt_fabric::FabricConfig::DISABLED) {
        tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::DISABLED);
    }
}

}  // namespace mesh_device_shared_detail

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
    using Config = MeshDeviceFixtureConfig;

    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> get_mesh_device() {
        TT_FATAL(mesh_device_, "MeshDevice not initialized in {}", __FUNCTION__);
        return mesh_device_;
    }

protected:
    using MeshDevice = ::tt::tt_metal::distributed::MeshDevice;
    using MeshDeviceConfig = ::tt::tt_metal::distributed::MeshDeviceConfig;
    using MeshShape = ::tt::tt_metal::distributed::MeshShape;

    explicit MeshDeviceFixtureBase(const Config& fixture_config) : config_(fixture_config) {}

    void SetUp() override {
        if (auto reason = mesh_device_shared_detail::mesh_fixture_skip_reason(config_)) {
            GTEST_SKIP() << *reason;
        }

        init_max_cbs();
        mesh_device_ = mesh_device_shared_detail::mesh_fixture_open(config_);
    }

    void TearDown() override { mesh_device_shared_detail::mesh_fixture_close(mesh_device_, config_); }

    void init_max_cbs() { max_cbs_ = tt::tt_metal::MetalContext::instance().hal().get_arch_num_circular_buffers(); }

    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device_;
    uint32_t max_cbs_{};

    Config config_;
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
        mesh_device_ = MeshDevice::create(
            MeshDeviceConfig(config_.mesh_shape.value_or(system_mesh_shape), config_.mesh_offset),
            config_.l1_small_size,
            config_.trace_region_size,
            config_.num_cqs,
            core_type,
            {},
            config_.worker_l1_size);
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

// Suite-level shared MeshDevice (fabric / CQ config): one open per suite, recreate after failure, no per-test close.
// Mirrors UnitMeshCQSingleCardSharedFixture / MultiCommandQueueT3KFixture recovery. Teardown order: close mesh, then
// SetFabricConfig(DISABLED) when fabric was enabled (see mesh_device_shared_detail::mesh_fixture_close).
// ----- Fabric health detection helpers (Mitigations 1-5) -----

namespace fabric_health_detail {

// Type trait to detect Traits::always_recover() static method.
template <typename T, typename = void>
struct has_always_recover : std::false_type {};
template <typename T>
struct has_always_recover<T, std::void_t<decltype(T::always_recover())>> : std::true_type {};

// Type trait to detect Traits::auto_enable_fabric() static method.  When defined on a Traits
// type and returning true, the fixture is told that the runtime may auto-enable fabric (e.g.
// T3K ETH dispatch flips FABRIC_1D on under the hood) even when Traits::config().fabric_config
// is DISABLED.  When undefined or false, the fixture can compile-time short-circuit fabric
// telemetry/drain entirely on DISABLED traits.
template <typename T, typename = void>
struct has_auto_enable_fabric : std::false_type {};
template <typename T>
struct has_auto_enable_fabric<T, std::void_t<decltype(T::auto_enable_fabric())>> : std::true_type {};

// Snapshot of per-device, per-channel tx/rx heartbeats for stall detection (Mitigation 4).
struct FabricHeartbeatSnapshot {
    // Key: device chip ID.  Value: vector of {tx_heartbeat, rx_heartbeat} per erisc channel.
    std::map<tt::ChipId, std::vector<std::pair<uint64_t, uint64_t>>> snapshots;
};

// Mitigation 1+2: Check that all fabric routers report Active state via telemetry.
// Returns true if all routers are healthy; false if any router is in a non-Active state.
// EDMStatus readback (Mitigation 2) requires FabricBuilderContext which is not accessible from
// test code (internal header, not in public API).  Fall back to telemetry-only check.
//
// Performance short-circuits (no coverage loss):
//   * If the runtime fabric config is DISABLED, the entire telemetry walk is skipped — there
//     are no routers to check.
//   * If a `before_snapshot` is provided AND the heartbeat counters did not advance vs the
//     snapshot for any device, the test did not generate any router traffic.  The full erisc
//     state walk is skipped for that device — a router that hasn't moved cannot have gone
//     unhealthy in a way the test would have noticed.  Devices whose heartbeats DID advance
//     still get the full walk so we never miss a real failure.
inline bool check_fabric_routers_healthy(
    const std::shared_ptr<::tt::tt_metal::distributed::MeshDevice>& mesh,
    const FabricHeartbeatSnapshot* before_snapshot = nullptr) {
    if (!mesh) {
        return true;  // No mesh — nothing to check.
    }
    // Runtime short-circuit: when fabric is genuinely disabled there is no router state
    // worth reading.  GetFabricConfig() is cheap (single static read in tt-fabric).
    try {
        if (tt::tt_fabric::GetFabricConfig() == tt::tt_fabric::FabricConfig::DISABLED) {
            return true;
        }
    } catch (const std::exception&) {
        // Fall through to the full check on error — better safe than sorry.
    }
    try {
        bool any_channel_checked = false;
        const auto device_ids = mesh->get_device_ids();
        for (const auto chip_id : device_ids) {
            auto fabric_node_id = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(chip_id);
            auto samples = tt::tt_fabric::read_fabric_telemetry(fabric_node_id);

            // Heartbeat-delta short-circuit: if a SetUp snapshot is available AND this device's
            // aggregate heartbeats did not advance, the test did not generate any router
            // traffic.  Skip the per-erisc state walk for this device (a router that has not
            // been used cannot have gone unhealthy in a way that affected the test).
            bool skip_state_walk = false;
            if (before_snapshot != nullptr) {
                auto before_it = before_snapshot->snapshots.find(chip_id);
                if (before_it != before_snapshot->snapshots.end()) {
                    uint64_t after_tx_total = 0, after_rx_total = 0;
                    for (const auto& sample : samples) {
                        if (!sample.snapshot.dynamic_info.has_value()) {
                            continue;
                        }
                        for (const auto& erisc : sample.snapshot.dynamic_info->erisc) {
                            after_tx_total += erisc.tx_heartbeat;
                            after_rx_total += erisc.rx_heartbeat;
                        }
                    }
                    uint64_t before_tx_total = 0, before_rx_total = 0;
                    for (const auto& [tx, rx] : before_it->second) {
                        before_tx_total += tx;
                        before_rx_total += rx;
                    }
                    if (after_tx_total == before_tx_total && after_rx_total == before_rx_total) {
                        any_channel_checked = true;  // We did look at telemetry, just briefly.
                        skip_state_walk = true;
                    }
                }
            }
            if (skip_state_walk) {
                continue;
            }

            for (const auto& sample : samples) {
                if (!sample.snapshot.dynamic_info.has_value()) {
                    continue;  // No dynamic info — telemetry not enabled on this channel.
                }
                any_channel_checked = true;
                for (const auto& erisc : sample.snapshot.dynamic_info->erisc) {
                    if (erisc.router_state != tt::tt_fabric::FabricTelemetryRouterState::Active) {
                        log_warning(
                            tt::LogTest,
                            "Fabric health check: chip {} channel {} erisc router_state={} (expected Active). "
                            "Requesting recovery.",
                            chip_id,
                            sample.channel_id,
                            static_cast<int>(erisc.router_state));
                        return false;
                    }
                }
            }
        }
        if (!any_channel_checked) {
            log_info(
                tt::LogTest,
                "Fabric health check: no channels with telemetry data found — check is a no-op "
                "(telemetry may be disabled).");
        }
    } catch (const std::exception& e) {
        log_warning(tt::LogTest, "Fabric health check failed with exception: {}. Requesting recovery.", e.what());
        return false;
    }
    return true;
}

// Mitigation 4: Capture per-device, per-channel heartbeat counters.
inline FabricHeartbeatSnapshot capture_fabric_heartbeats(
    const std::shared_ptr<::tt::tt_metal::distributed::MeshDevice>& mesh) {
    FabricHeartbeatSnapshot snap;
    if (!mesh) {
        return snap;
    }
    try {
        const auto device_ids = mesh->get_device_ids();
        for (const auto chip_id : device_ids) {
            auto fabric_node_id = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(chip_id);
            auto samples = tt::tt_fabric::read_fabric_telemetry(fabric_node_id);
            auto& entries = snap.snapshots[chip_id];
            for (const auto& sample : samples) {
                if (!sample.snapshot.dynamic_info.has_value()) {
                    entries.emplace_back(0ULL, 0ULL);
                    continue;
                }
                // Aggregate heartbeats across erisc entries for this channel.
                uint64_t tx_total = 0, rx_total = 0;
                for (const auto& erisc : sample.snapshot.dynamic_info->erisc) {
                    tx_total += erisc.tx_heartbeat;
                    rx_total += erisc.rx_heartbeat;
                }
                entries.emplace_back(tx_total, rx_total);
            }
        }
    } catch (const std::exception& e) {
        log_warning(tt::LogTest, "Fabric heartbeat capture failed: {}", e.what());
    }
    return snap;
}

// Mitigation 4: Check that heartbeats advanced between snapshots.
// Only flags stall when heartbeats were > 0 at start and stopped advancing (device was active).
// Returns true if heartbeats are healthy; false if a stall is detected.
inline bool check_fabric_heartbeats_advanced(
    const std::shared_ptr<::tt::tt_metal::distributed::MeshDevice>& mesh,
    const FabricHeartbeatSnapshot& before) {
    if (!mesh || before.snapshots.empty()) {
        return true;
    }
    try {
        FabricHeartbeatSnapshot after = capture_fabric_heartbeats(mesh);
        for (const auto& [chip_id, before_channels] : before.snapshots) {
            auto it = after.snapshots.find(chip_id);
            if (it == after.snapshots.end()) {
                continue;
            }
            const auto& after_channels = it->second;
            size_t n = std::min(before_channels.size(), after_channels.size());
            for (size_t i = 0; i < n; ++i) {
                const auto& [tx_before, rx_before] = before_channels[i];
                const auto& [tx_after, rx_after] = after_channels[i];

                // Skip if device was idle (heartbeats zero at start).
                if (tx_before == 0 && rx_before == 0) {
                    continue;
                }
                // Flag stall: heartbeats were non-zero but did not advance.
                if (tx_after <= tx_before && rx_after <= rx_before) {
                    log_warning(
                        tt::LogTest,
                        "Fabric heartbeat stall detected: chip {} channel {} "
                        "tx_before={} tx_after={} rx_before={} rx_after={}. Requesting recovery.",
                        chip_id, i, tx_before, tx_after, rx_before, rx_after);
                    return false;
                }
            }
        }
    } catch (const std::exception& e) {
        log_warning(tt::LogTest, "Fabric heartbeat check failed: {}", e.what());
        return false;
    }
    return true;
}

// Mitigation 5: PAUSE/DRAIN/RUN cycle.
// Issues a host-driven drain to every active fabric router in the mesh to flush in-flight
// messages without tearing down and reinitialising the fabric.  This is lighter-weight than
// a full mesh re-open and avoids the repeated erisc-core degradation we observed on T3K after
// many SetFabricConfig(FABRIC_1D) reinit cycles.
//
// Cycle:  1) Write RouterCommand::PAUSE   → router stops forwarding messages/credits
//          2) Poll RouterState::PAUSED     (device confirms pause, up to 500 ms)
//          3) Write RouterCommand::DRAIN   → router drops queued messages
//          4) Poll RouterState::PAUSED     (drain complete, up to 500 ms)
//          5) Write RouterCommand::RUN     → router resumes normal operation
//          6) Poll RouterState::RUNNING    (up to 500 ms)
//
// Returns true if all routers on all devices reached RUNNING after the cycle.
// Returns false on timeout or exception — caller should fall back to full mesh re-open.
inline bool drain_fabric_routers(
    const std::shared_ptr<::tt::tt_metal::distributed::MeshDevice>& mesh) {
    if (!mesh) {
        return true;
    }

    using namespace std::chrono;
    using tt::tt_fabric::RouterCommand;
    using ::RouterState;  // Global namespace — defined in fabric_telemetry_msgs.h (hostdev header)

    constexpr auto kPollInterval = milliseconds(1);
    constexpr auto kTimeout = milliseconds(500);

    // Helper: write a 4-byte RouterCommand to every active router on one device.
    auto write_cmd = [&](
        tt::tt_metal::IDevice* device,
        const std::vector<tt::tt_fabric::ControlPlane::FabricRouterDrainInfo>& routers,
        RouterCommand cmd) {
        std::vector<uint32_t> buf{static_cast<uint32_t>(cmd)};
        for (const auto& r : routers) {
            tt::tt_metal::detail::WriteToDeviceL1(
                device, r.logical_eth_core, r.command_address, buf, CoreType::ETH);
        }
    };

    // Helper: poll until every router on one device reports the expected state.
    // Returns true on success, false on timeout.
    auto poll_state = [&](
        tt::tt_metal::IDevice* device,
        const std::vector<tt::tt_fabric::ControlPlane::FabricRouterDrainInfo>& routers,
        RouterState expected) -> bool {
        const auto deadline = steady_clock::now() + kTimeout;
        std::vector<uint32_t> buf(1, 0);
        while (steady_clock::now() < deadline) {
            bool all_done = true;
            for (const auto& r : routers) {
                buf[0] = 0;
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device, r.logical_eth_core, r.state_address, sizeof(uint32_t), buf,
                    CoreType::ETH);
                if (static_cast<RouterState>(buf[0]) != expected) {
                    all_done = false;
                    break;
                }
            }
            if (all_done) {
                return true;
            }
            std::this_thread::sleep_for(kPollInterval);
        }
        return false;
    };

    try {
        auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
        const auto device_ids = mesh->get_device_ids();

        // Collect router info for every device upfront (avoids repeated lookups in the loop).
        std::vector<std::pair<tt::tt_metal::IDevice*, std::vector<tt::tt_fabric::ControlPlane::FabricRouterDrainInfo>>>
            per_device_routers;
        per_device_routers.reserve(device_ids.size());
        for (const auto chip_id : device_ids) {
            auto* dev = mesh->get_device(chip_id);
            auto routers = control_plane.get_fabric_router_drain_info(chip_id);
            if (!routers.empty()) {
                per_device_routers.emplace_back(dev, std::move(routers));
            }
        }

        if (per_device_routers.empty()) {
            return true;  // No active routers — nothing to drain.
        }

        // Step 1: PAUSE all routers.
        for (auto& [dev, routers] : per_device_routers) {
            write_cmd(dev, routers, RouterCommand::PAUSE);
        }
        // Step 2: Wait for all routers to confirm PAUSED.
        // Note on atomicity: The PAUSE command was broadcast (fire-and-forget) to every device
        // in Step 1 before any polling begins here.  During the window where some devices have
        // confirmed PAUSED while others are still transitioning, the fabric firmware guarantees
        // safety: a paused router queues or drops messages sent to it (it does NOT cause
        // undefined behavior).  This is safe per the firmware contract.
        for (auto& [dev, routers] : per_device_routers) {
            if (!poll_state(dev, routers, RouterState::PAUSED)) {
                log_warning(tt::LogTest,
                    "drain_fabric_routers: device {} timed out waiting for PAUSED after PAUSE command — "
                    "falling back to full mesh re-open.",
                    dev->id());
                // Best-effort: restore RUN before bailing.
                for (auto& [d2, r2] : per_device_routers) {
                    write_cmd(d2, r2, RouterCommand::RUN);
                }
                return false;
            }
        }

        // Step 3: DRAIN all routers (drop queued messages).
        for (auto& [dev, routers] : per_device_routers) {
            write_cmd(dev, routers, RouterCommand::DRAIN);
        }
        // Step 4: Wait for drain complete (routers return to PAUSED once empty).
        for (auto& [dev, routers] : per_device_routers) {
            if (!poll_state(dev, routers, RouterState::PAUSED)) {
                log_warning(tt::LogTest,
                    "drain_fabric_routers: device {} timed out waiting for PAUSED after DRAIN command — "
                    "falling back to full mesh re-open.",
                    dev->id());
                for (auto& [d2, r2] : per_device_routers) {
                    write_cmd(d2, r2, RouterCommand::RUN);
                }
                return false;
            }
        }

        // Step 5: RUN — resume normal operation.
        for (auto& [dev, routers] : per_device_routers) {
            write_cmd(dev, routers, RouterCommand::RUN);
        }
        // Step 6: Confirm RUNNING.
        for (auto& [dev, routers] : per_device_routers) {
            if (!poll_state(dev, routers, RouterState::RUNNING)) {
                log_warning(tt::LogTest,
                    "drain_fabric_routers: device {} timed out waiting for RUNNING after RUN command.",
                    dev->id());
                return false;
            }
        }

        log_info(tt::LogTest, "drain_fabric_routers: all routers drained and resumed successfully.");
        return true;

    } catch (const std::exception& e) {
        log_warning(tt::LogTest, "drain_fabric_routers: exception during drain cycle: {}", e.what());
        return false;
    }
}

}  // namespace fabric_health_detail

template <typename Traits>
class MeshDeviceConfigSharedFixture : public ::testing::Test {
public:
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> get_mesh_device() {
        TT_FATAL(mesh_device_, "MeshDevice not initialized in {}", __FUNCTION__);
        return mesh_device_;
    }

protected:
    using MeshDevice = ::tt::tt_metal::distributed::MeshDevice;

    // Thread-safety note: these `inline static` members are shared across all tests in the
    // suite.  This is safe because GTest runs SetUpTestSuite / SetUp / TearDown / TearDownTestSuite
    // single-threaded for a given test suite — no concurrent access occurs.
    inline static std::shared_ptr<MeshDevice> shared_mesh_;
    inline static bool devices_valid_ = false;
    inline static bool needs_recovery_ = false;
    inline static fabric_health_detail::FabricHeartbeatSnapshot fabric_heartbeat_before_;
    // Cached runtime fabric config (captured once in SetUpTestSuite after open_shared_mesh).
    // Avoids paying the cost of tt_fabric::GetFabricConfig() once per test in fabric_enabled().
    inline static tt_fabric::FabricConfig cached_runtime_fabric_config_ = tt_fabric::FabricConfig::DISABLED;
    inline static bool runtime_fabric_config_cached_ = false;

    // Mitigation 3: Detect Traits::always_recover() if it exists, default to false.
    static constexpr bool traits_always_recover() {
        if constexpr (fabric_health_detail::has_always_recover<Traits>::value) {
            return Traits::always_recover();
        } else {
            return false;
        }
    }

    // Detect Traits::auto_enable_fabric() if it exists, default to false.  When false (and the
    // trait's compile-time fabric_config is DISABLED) the fixture compile-time skips the
    // telemetry/drain block entirely.
    static constexpr bool traits_auto_enable_fabric() {
        if constexpr (fabric_health_detail::has_auto_enable_fabric<Traits>::value) {
            return Traits::auto_enable_fabric();
        } else {
            return false;
        }
    }

    // Returns true if fabric was explicitly requested by the trait config, OR if the trait
    // opts in to runtime auto-enable detection AND the cached runtime config says fabric is
    // up.  This is the cheap per-test path: when the trait declares DISABLED at compile time
    // and does not opt in to auto-enable detection, we skip the runtime config read entirely.
    static bool fabric_enabled() {
        if (Traits::config().fabric_config != tt_fabric::FabricConfig::DISABLED) {
            return true;
        }
        if constexpr (!fabric_health_detail::has_auto_enable_fabric<Traits>::value) {
            // Compile-time short-circuit: trait has not opted in to auto-enable detection.
            return false;
        } else {
            if (!Traits::auto_enable_fabric()) {
                return false;
            }
            if (!runtime_fabric_config_cached_) {
                return false;  // No suite open yet.
            }
            return cached_runtime_fabric_config_ != tt_fabric::FabricConfig::DISABLED;
        }
    }

    // Cache tt_fabric::GetFabricConfig() so per-test fabric_enabled() is a single static read.
    static void refresh_cached_runtime_fabric_config() {
        try {
            cached_runtime_fabric_config_ = tt_fabric::GetFabricConfig();
        } catch (const std::exception&) {
            cached_runtime_fabric_config_ = tt_fabric::FabricConfig::DISABLED;
        }
        runtime_fabric_config_cached_ = true;
    }

    // Close the previous mesh (safely, even if close() throws on dead hardware), then open fresh.
    static void open_shared_mesh() {
        const MeshDeviceFixtureConfig cfg = Traits::config();
        // Mitigation — exception-safe close: if the device is in an unrecoverable state,
        // mesh->close() may throw. Force-release the pointer so the next open() gets a
        // clean start rather than double-closing a dead device.
        try {
            mesh_device_shared_detail::mesh_fixture_close(shared_mesh_, cfg);
        } catch (const std::exception& e) {
            log_warning(
                tt::LogTest,
                "open_shared_mesh: previous mesh close threw: {}. Force-releasing stale pointer.",
                e.what());
            shared_mesh_.reset();
        }
        auto new_mesh = mesh_device_shared_detail::mesh_fixture_open(cfg);
        // Mitigation — post-open fabric health check: if fabric is broken right after open
        // (e.g. chip 4 erisc was degraded by a prior suite's TearDown), skip rather than
        // letting every test timeout for TT_METAL_OPERATION_TIMEOUT_SECONDS.
        if (new_mesh && Traits::config().fabric_config != tt_fabric::FabricConfig::DISABLED) {
            if (!fabric_health_detail::check_fabric_routers_healthy(new_mesh)) {
                log_warning(
                    tt::LogTest,
                    "open_shared_mesh: post-open fabric health check FAILED — mesh appears degraded. "
                    "Releasing mesh so tests skip instead of timing out.");
                try {
                    mesh_device_shared_detail::mesh_fixture_close(new_mesh, cfg);
                } catch (...) {
                    new_mesh.reset();
                }
                devices_valid_ = false;
                needs_recovery_ = false;
                return;
            }
        }
        shared_mesh_ = new_mesh;
        devices_valid_ = static_cast<bool>(shared_mesh_);
        needs_recovery_ = false;
        // Cache the runtime fabric config now that the mesh (and therefore MetalContext +
        // tt-fabric) is initialised.  Subsequent fabric_enabled() calls use this cached value.
        refresh_cached_runtime_fabric_config();
    }

    static void SetUpTestSuite() {
        const MeshDeviceFixtureConfig cfg = Traits::config();
        if (mesh_device_shared_detail::mesh_fixture_skip_reason(cfg)) {
            mesh_device_shared_detail::mesh_fixture_close(shared_mesh_, cfg);
            devices_valid_ = false;
            return;
        }
        // Mitigation — exception-safe SetUpTestSuite: if the device can't be opened (e.g. driver
        // error, missing hardware), set devices_valid_=false so every test SKIPS cleanly rather
        // than crashing with an unhandled exception.
        try {
            open_shared_mesh();
        } catch (const std::exception& e) {
            log_error(
                tt::LogTest,
                "SetUpTestSuite: failed to open shared mesh: {}. All tests in this suite will be skipped.",
                e.what());
            devices_valid_ = false;
            return;
        }
        // Mitigation — pre-flight fabric health check: verify that the remote devices are
        // responsive before running any tests. On T3K, if fabric (auto-enabled for ETH dispatch)
        // already has broken erisc cores, every test will timeout. Detect this early and skip the
        // entire suite rather than wasting wall-clock time on 57-second per-test timeouts.
        if (shared_mesh_ && fabric_enabled()) {
            if (!fabric_health_detail::check_fabric_routers_healthy(shared_mesh_)) {
                log_error(
                    tt::LogTest,
                    "SetUpTestSuite: pre-flight fabric health check FAILED — remote devices appear "
                    "degraded. Closing mesh and skipping all tests in this suite.");
                try {
                    mesh_device_shared_detail::mesh_fixture_close(shared_mesh_, cfg);
                } catch (...) {
                    shared_mesh_.reset();
                }
                devices_valid_ = false;
            }
        }
    }

    static void TearDownTestSuite() {
        const MeshDeviceFixtureConfig cfg = Traits::config();
        try {
            mesh_device_shared_detail::mesh_fixture_close(shared_mesh_, cfg);
        } catch (const std::exception& e) {
            log_warning(tt::LogTest, "TearDownTestSuite: mesh close threw: {}. Force-releasing.", e.what());
            shared_mesh_.reset();
        }
        devices_valid_ = false;
    }

    void SetUp() override {
        const MeshDeviceFixtureConfig cfg = Traits::config();
        if (auto reason = mesh_device_shared_detail::mesh_fixture_skip_reason(cfg)) {
            GTEST_SKIP() << *reason;
        }
        if (needs_recovery_ || !devices_valid_) {
            // Mitigation — exception-to-skip: if recovery fails (e.g. close() hangs on dead
            // hardware), convert the exception to GTEST_SKIP so subsequent tests skip cleanly
            // instead of each timing out for TT_METAL_OPERATION_TIMEOUT_SECONDS.
            try {
                open_shared_mesh();
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogTest, "SetUp: mesh recovery failed: {}. Skipping test.", e.what());
                devices_valid_ = false;
                shared_mesh_.reset();
            }
        }
        if (!shared_mesh_) {
            GTEST_SKIP() << "Shared mesh device not available (initialization or recovery failed).";
        }
        mesh_device_ = shared_mesh_;
        init_max_cbs();

        // Mitigation 4: Snapshot heartbeats before test (for stall detection in TearDown).
        if (fabric_enabled() && shared_mesh_) {
            fabric_heartbeat_before_ = fabric_health_detail::capture_fabric_heartbeats(shared_mesh_);
        }
    }

    void TearDown() override {
        // Mitigation 3: always_recover() trait override.  We will recover unconditionally
        // before the next test, so skip the entire telemetry/drain block — its only purpose
        // is to decide whether to recover, and the answer is already yes.
        if (HasFailure() || traits_always_recover()) {
            needs_recovery_ = true;
            return;
        }

        // Mitigations 1+2: Telemetry-based fabric router health check.
        // fabric_enabled() consults the cached runtime config (refreshed in open_shared_mesh)
        // so per-test cost is a single static read on DISABLED traits without auto-enable.
        // The check itself receives the SetUp heartbeat snapshot so devices that did no
        // fabric work this test skip the per-erisc state walk (still a per-test check, but
        // a much cheaper one for the dominant idle-fabric case).
        if (fabric_enabled() && shared_mesh_ && !needs_recovery_) {
            if (!fabric_health_detail::check_fabric_routers_healthy(shared_mesh_, &fabric_heartbeat_before_)) {
                // Mitigation 5: attempt a lightweight PAUSE/DRAIN/RUN cycle before falling
                // back to a full mesh re-open.  This avoids the erisc-core degradation that
                // repeated SetFabricConfig reinit cycles cause on T3K.
                log_info(tt::LogTest,
                    "TearDown: fabric health check failed — attempting drain cycle before recovery.");
                if (!fabric_health_detail::drain_fabric_routers(shared_mesh_)) {
                    log_warning(tt::LogTest,
                        "TearDown: drain cycle failed — scheduling full mesh re-open.");
                    needs_recovery_ = true;
                }
                // Whether drain succeeded or not, re-check health.  If drain fixed it we
                // carry on; if not, needs_recovery_ is already set.  Pass nullptr for the
                // snapshot here — after a drain the heartbeats are deliberately disturbed
                // and the snapshot-based skip would mask a genuine failure.
                if (!needs_recovery_ && !fabric_health_detail::check_fabric_routers_healthy(shared_mesh_, nullptr)) {
                    needs_recovery_ = true;
                }
                if (needs_recovery_) {
                    return;
                }
            }
        }

        // Mitigation 4: Heartbeat stall detection (diagnostic only — does not trigger recovery
        // because tests that enable fabric but generate no router traffic would false-positive).
        if (fabric_enabled() && shared_mesh_ && !needs_recovery_) {
            fabric_health_detail::check_fabric_heartbeats_advanced(shared_mesh_, fabric_heartbeat_before_);
        }
    }

    void init_max_cbs() { max_cbs_ = tt::tt_metal::MetalContext::instance().hal().get_arch_num_circular_buffers(); }

    std::shared_ptr<MeshDevice> mesh_device_;
    uint32_t max_cbs_{};
};

struct MeshDevice1x4Fabric1DSharedTraits {
    // Add `static constexpr bool always_recover() { return true; }` to force device
    // re-creation before every test (useful when tests may leave persistent device state).
    static MeshDeviceFixtureConfig config() {
        return MeshDeviceFixtureConfig{
            .mesh_shape = ::tt::tt_metal::distributed::MeshShape{1, 4},
            .num_cqs = 1,
            .fabric_config = tt_fabric::FabricConfig::FABRIC_1D,
        };
    }
};
using MeshDevice1x4Fabric1DSharedFixture = MeshDeviceConfigSharedFixture<MeshDevice1x4Fabric1DSharedTraits>;

struct MultiCQMeshDevice2x4Fabric1DSharedTraits {
    static MeshDeviceFixtureConfig config() {
        return MeshDeviceFixtureConfig{
            .mesh_shape = ::tt::tt_metal::distributed::MeshShape{2, 4},
            .num_cqs = 2,
            .fabric_config = tt_fabric::FabricConfig::FABRIC_1D,
        };
    }
};
using MultiCQMeshDevice2x4Fabric1DSharedFixture =
    MeshDeviceConfigSharedFixture<MultiCQMeshDevice2x4Fabric1DSharedTraits>;

struct MultiCQMeshDevice1x4Fabric1DSharedTraits {
    static MeshDeviceFixtureConfig config() {
        return MeshDeviceFixtureConfig{
            .mesh_shape = ::tt::tt_metal::distributed::MeshShape{1, 4},
            .num_cqs = 2,
            .fabric_config = tt_fabric::FabricConfig::FABRIC_1D,
        };
    }
};
using MultiCQMeshDevice1x4Fabric1DSharedFixture =
    MeshDeviceConfigSharedFixture<MultiCQMeshDevice1x4Fabric1DSharedTraits>;

struct MeshDevice2x4Fabric2DSharedTraits {
    static MeshDeviceFixtureConfig config() {
        return MeshDeviceFixtureConfig{
            .mesh_shape = ::tt::tt_metal::distributed::MeshShape{2, 4},
            .num_cqs = 1,
            .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
        };
    }
};
using MeshDevice2x4Fabric2DSharedFixture = MeshDeviceConfigSharedFixture<MeshDevice2x4Fabric2DSharedTraits>;

struct MeshDevice1x4Fabric2DUDMSharedTraits {
    static MeshDeviceFixtureConfig config() {
        return MeshDeviceFixtureConfig{
            .mesh_shape = ::tt::tt_metal::distributed::MeshShape{1, 4},
            .num_cqs = 1,
            .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
            .fabric_tensix_config = tt_fabric::FabricTensixConfig::UDM,
            .fabric_udm_mode = tt_fabric::FabricUDMMode::ENABLED,
        };
    }
};
using MeshDevice1x4Fabric2DUDMSharedFixture =
    MeshDeviceConfigSharedFixture<MeshDevice1x4Fabric2DUDMSharedTraits>;

struct MeshDevice2x4Fabric2DUDMSharedTraits {
    static MeshDeviceFixtureConfig config() {
        return MeshDeviceFixtureConfig{
            .mesh_shape = ::tt::tt_metal::distributed::MeshShape{2, 4},
            .num_cqs = 1,
            .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
            .fabric_tensix_config = tt_fabric::FabricTensixConfig::UDM,
            .fabric_udm_mode = tt_fabric::FabricUDMMode::ENABLED,
        };
    }
};
using MeshDevice2x4Fabric2DUDMSharedFixture =
    MeshDeviceConfigSharedFixture<MeshDevice2x4Fabric2DUDMSharedTraits>;

// Generic shared fixture: auto-detected mesh shape, 1 CQ, no explicit fabric config.
// On T3K the runtime will auto-enable FABRIC_1D for ETH dispatch, but the fixture itself
// does not call SetFabricConfig — this is intentional so that tests that do not need fabric
// can run without paying the full fabric teardown/reinit cost on every test.
// HasFailure() in TearDown still triggers mesh re-open before the next test.
//
// Declares auto_enable_fabric() = true so the per-test fabric health check still fires on
// T3K (where ETH dispatch auto-enables FABRIC_1D under the hood).  On non-T3K machines
// the runtime config will read DISABLED and the fast-path short-circuit kicks in.
struct GenericMeshDeviceSharedTraits {
    static MeshDeviceFixtureConfig config() {
        return MeshDeviceFixtureConfig{
            .num_cqs = 1,
        };
    }
    static constexpr bool auto_enable_fabric() { return true; }
};
using GenericMeshDeviceSharedFixture = MeshDeviceConfigSharedFixture<GenericMeshDeviceSharedTraits>;

// 1x2 shared mesh, 1 CQ, no fabric. Replacement for per-test MeshDevice1x2Fixture.
// Two chips on T3K do not auto-enable fabric (multi-chip ETH dispatch only kicks in for
// the full 8-chip mesh), so we leave auto_enable_fabric() at its default of false.
struct MeshDevice1x2SharedTraits {
    static MeshDeviceFixtureConfig config() {
        return MeshDeviceFixtureConfig{
            .mesh_shape = ::tt::tt_metal::distributed::MeshShape{1, 2},
            .num_cqs = 1,
            .fabric_config = tt_fabric::FabricConfig::DISABLED,
        };
    }
};
using MeshDevice1x2SharedFixture = MeshDeviceConfigSharedFixture<MeshDevice1x2SharedTraits>;

// 2x4 shared mesh, 1 CQ, no fabric. Replacement for per-test MeshDevice2x4Fixture in
// suites that do not require fabric and do not mutate persistent device state across tests.
// On T3K the runtime auto-enables FABRIC_1D for ETH dispatch, so opt in to detection.
struct MeshDevice2x4SharedTraits {
    static MeshDeviceFixtureConfig config() {
        return MeshDeviceFixtureConfig{
            .mesh_shape = ::tt::tt_metal::distributed::MeshShape{2, 4},
            .num_cqs = 1,
            .fabric_config = tt_fabric::FabricConfig::DISABLED,
        };
    }
    static constexpr bool auto_enable_fabric() { return true; }
};
using MeshDevice2x4SharedFixture = MeshDeviceConfigSharedFixture<MeshDevice2x4SharedTraits>;

// 4x8 shared mesh, 1 CQ, no fabric. Replacement for per-test MeshDevice4x8Fixture used by
// Galaxy distributed tests.
//
// Recovery strategy: standard `HasFailure()`-driven recovery (inherited from
// MeshDeviceConfigSharedFixture). If Galaxy stability requires it, override with
// `static constexpr bool always_recover() { return true; }` to force unconditional
// re-open before every test. We start without that override and revisit if the
// recovery rate proves too low to keep the mesh healthy across a full suite.
struct MeshDevice4x8SharedTraits {
    static MeshDeviceFixtureConfig config() {
        return MeshDeviceFixtureConfig{
            .mesh_shape = ::tt::tt_metal::distributed::MeshShape{4, 8},
            .num_cqs = 1,
            .fabric_config = tt_fabric::FabricConfig::DISABLED,
        };
    }
    static constexpr bool auto_enable_fabric() { return true; }
};
using MeshDevice4x8SharedFixture = MeshDeviceConfigSharedFixture<MeshDevice4x8SharedTraits>;

}  // namespace tt::tt_metal
