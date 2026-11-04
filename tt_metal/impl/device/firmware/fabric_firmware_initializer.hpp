// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <memory>
#include <unordered_set>
#include <vector>

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

private:
    // Compile fabric on all devices, parallelized via async.
    // Configure fabric sequentially (Galaxy hangs if parallelized).
    void compile_and_configure_fabric();

    // Mock only: compile the fabric program on every device, without programming or syncing routers.
    // No-op when debug tooling disables erisc IRAM.
    void compile_fabric_only();

    // Build and compile the fabric program for one device.
    // Returns nullptr if the device has no fabric routers.
    std::unique_ptr<Program> compile_fabric(Device* dev) const;

    // Write runtime args and launch messages for an already-compiled fabric program on one device.
    void configure_fabric(Device* dev, Program& fabric_program) const;

    // Wait for fabric router handshake on all devices.
    void wait_for_fabric_router_sync(uint32_t timeout_ms) const;

    // Compute the fabric router sync timeout from runtime options.
    uint32_t get_fabric_router_sync_timeout_ms() const;

    // Drop all per-device state and mark this initializer as torn down.
    void reset_state(std::unordered_set<InitializerKey>& init_done);

    tt::tt_fabric::ControlPlane& control_plane_;
    std::vector<Device*> devices_;
    // Fabric program per device, parallel to devices_. A null entry means that device has no fabric routers.
    // Each compile task writes only its own slot and the vector is never resized while tasks run, so no lock is needed.
    std::vector<std::unique_ptr<Program>> fabric_programs_;
    std::atomic_flag initialized_;
};

}  // namespace tt::tt_metal
