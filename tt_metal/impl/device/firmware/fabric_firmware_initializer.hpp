// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>

#include "device/device_impl.hpp"
#include "firmware_initializer.hpp"

namespace tt::tt_fabric {
class ControlPlane;
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

    // Waits for the device's master router to report the local sync, then signals READY_FOR_TRAFFIC. A device
    // with a host-bridged router cannot get there until the bridge is armed, so configure() leaves such devices
    // to the FabricHostBridge, which calls this once it has armed its links.
    static void wait_for_router_sync_on_device(
        IDevice* dev, const tt::tt_fabric::ControlPlane& control_plane, const Cluster& cluster, uint32_t timeout_ms);
    static uint32_t fabric_router_sync_timeout_ms(const llrt::RunTimeOptions& rtoptions);

private:
    // Compile fabric on all devices, parallelized via async.
    // Configure fabric sequentially (Galaxy hangs if parallelized).
    void compile_and_configure_fabric();

    // Mock only: compile the fabric program on every device, without programming or syncing routers.
    // No-op when debug tooling disables erisc IRAM.
    void compile_fabric_only();

    // Wait for fabric router handshake on all devices.
    void wait_for_fabric_router_sync(uint32_t timeout_ms) const;

    // Compute the fabric router sync timeout from runtime options.
    uint32_t get_fabric_router_sync_timeout_ms() const;

    tt::tt_fabric::ControlPlane& control_plane_;
    std::vector<Device*> devices_;
    std::atomic_flag initialized_;
};

}  // namespace tt::tt_metal
