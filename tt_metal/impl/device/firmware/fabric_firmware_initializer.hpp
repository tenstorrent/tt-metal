// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>

#include "device/device_impl.hpp"
#include "firmware_initializer.hpp"

namespace tt::tt_fabric {
class ControlPlane;
class FabricBuildSession;
struct FabricDeviceBuild;
}  // namespace tt::tt_fabric

namespace tt::tt_metal {

class FabricFirmwareInitializer final : public FirmwareInitializer {
public:
    static constexpr InitializerKey key = InitializerKey::Fabric;

    FabricFirmwareInitializer(
        std::shared_ptr<const ContextDescriptor> descriptor, tt::tt_fabric::ControlPlane& control_plane);
    // Out of line: build_session_ holds a type that is only forward-declared here.
    ~FabricFirmwareInitializer() override;

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

    // Every fabric launch builds through here: start the launch's build session, then build each device's fabric
    // program into fabric_programs_. on_compiled(i) runs on the calling thread, in device order, once device i is done.
    void build_fabric_programs(bool parallel, const std::function<void(size_t)>& on_compiled = {});

    // task(i) for every device, on worker threads when parallel; on_done(i) as in build_fabric_programs().
    void for_each_device(
        bool parallel, const std::function<void(size_t)>& task, const std::function<void(size_t)>& on_done = {}) const;

    // The two phases of build_fabric_programs(): every device's routers are created and wired (build_fabric), the
    // routers' channel layouts are exchanged across hosts, then each device's program is compiled (compile_fabric).
    // build_fabric returns nullopt, and compile_fabric then nullptr, for a device without routers.
    std::optional<tt::tt_fabric::FabricDeviceBuild> build_fabric(Device* dev) const;
    std::unique_ptr<Program> compile_fabric(Device* dev, std::optional<tt::tt_fabric::FabricDeviceBuild> build) const;

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
    // Builder state for the current fabric launch. Created before the per-device compiles (which only read it)
    // and destroyed in reset_state(), so each launch, including a relaunch after teardown, gets a fresh one.
    std::unique_ptr<tt::tt_fabric::FabricBuildSession> build_session_;
    std::atomic_flag initialized_;
};

}  // namespace tt::tt_metal
