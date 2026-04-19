// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <unordered_set>

#include "device/device_impl.hpp"
#include "firmware_initializer.hpp"

namespace tt::tt_fabric {
class ControlPlane;
class FabricBuilderContext;
}  // namespace tt::tt_fabric

namespace tt::tt_metal {

class FabricFirmwareInitializer final : public FirmwareInitializer {
public:
    static constexpr InitializerKey key = InitializerKey::Fabric;

    FabricFirmwareInitializer(
        std::shared_ptr<const ContextDescriptor> descriptor, tt::tt_fabric::ControlPlane& control_plane);

    void init(const std::vector<Device*>& devices, [[maybe_unused]] const std::unordered_set<InitializerKey>& init_done)
        override;
    void configure() override;
    void teardown(std::unordered_set<InitializerKey>& init_done) override;
    void post_teardown() override;
    bool is_initialized() const override;

private:
    // Compile fabric on all devices, parallelized via async.
    // Configure fabric sequentially (Galaxy hangs if parallelized).
    void compile_and_configure_fabric();

    // Wait for fabric router handshake on all devices.
    void wait_for_fabric_router_sync(uint32_t timeout_ms) const;

    // Scan all active ERISC channels and return any devices that have at least one channel
    // not at EDMStatus::READY_FOR_TRAFFIC.  Logs a warning for each bad channel found.
    // Returns empty set if fabric config is DISABLED.
    std::unordered_set<Device*> collect_unhealthy_devices() const;

    // Verify ALL active ERISC channels are healthy after fabric init.
    // wait_for_fabric_router_sync only checks the master channel; this checks every channel
    // to detect persistent ERISC corruption that would cause dispatch hangs later.
    // Throws TT_THROW if any channels are not at READY_FOR_TRAFFIC.
    void verify_all_fabric_channels_healthy() const;

    // Compute the fabric router sync timeout from runtime options.
    uint32_t get_fabric_router_sync_timeout_ms() const;

    // Probe all active ERISC router channels for stale firmware and terminate any found.
    // Sends TERMINATE to each active ERISC channel, polls for EDMStatus::TERMINATED (50 ms).
    // On timeout, logs a warning and continues — does NOT assert RISC reset (that would
    // tear down the WH ETH PHY link and break non-MMIO L1 access for the rest of the mesh).
    // Called before configure_fabric_cores() clears L1 (Fix A).
    void terminate_stale_erisc_routers(
        Device* dev, const tt_fabric::FabricBuilderContext& builder_context) const;

    tt::tt_fabric::ControlPlane& control_plane_;
    std::vector<Device*> devices_;
    std::atomic_flag initialized_;
};

}  // namespace tt::tt_metal
