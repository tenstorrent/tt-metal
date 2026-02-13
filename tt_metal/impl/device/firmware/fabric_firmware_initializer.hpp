// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/device_impl.hpp"
#include "firmware_initializer.hpp"

namespace tt::tt_fabric {
class ControlPlane;
}  // namespace tt::tt_fabric

namespace tt::tt_metal {

class FabricFirmwareInitializer final : public FirmwareInitializer {
public:
    static constexpr InitializerKey key() { return InitializerKey::Fabric; }

    FabricFirmwareInitializer(
        std::shared_ptr<const ContextDescriptor> descriptor, tt::tt_fabric::ControlPlane& control_plane);

    void init(const std::vector<Device*>& devices, [[maybe_unused]] const std::unordered_set<InitializerKey>& init_done)
        override;
    void configure() override;
    void teardown() override;
    void post_teardown() override;
    bool is_initialized() const override;

private:
    // Compile fabric on all devices, parallelized via async.
    // Configure fabric sequentially (Galaxy hangs if parallelized).
    void compile_and_configure_fabric();

    // Wait for fabric router handshake on all devices.
    void wait_for_fabric_router_sync(uint32_t timeout_ms) const;

    // Compute the fabric router sync timeout from runtime options.
    uint32_t get_fabric_router_sync_timeout_ms() const;

    tt::tt_fabric::ControlPlane& control_plane_;
    std::vector<Device*> devices_;
    bool initialized_ = false;
};

}  // namespace tt::tt_metal
