// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal {

class IDevice;
class Program;

namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace experimental {

// This class provides APIs to dynamically enable and teardown Fast Dispatch during runtime.
// Functionality is currently limited to Galaxy clusters.
// Note: The functionality in this class is extremely application specific, and will likely be
// removed once we implement a proper weight loading solution for Low Latency Decode.
// As such its exposed as experimental.

// Note: Slow Dispatch is a "Back-Door" way of running programs on compute cores.
// This is productized for extremely application specific use cases.

class DispatchContext {
public:
    static DispatchContext& get();
    void initialize_fast_dispatch(distributed::MeshDevice* mesh_device);
    void terminate_fast_dispatch(distributed::MeshDevice* mesh_device);
    void enable_asynchronous_slow_dispatch(distributed::MeshDevice* mesh_device);

    // Configure-without-launch mode: programs are written to L1 but never given the go signal.
    void set_configure_only(distributed::MeshDevice* mesh_device, bool enable);
    void disable_asynchronous_slow_dispatch(distributed::MeshDevice* mesh_device);
    bool is_asynchronous_slow_dispatch_enabled(distributed::MeshDevice* mesh_device) const;

    void reset();

private:
    DispatchContext() = default;
    ~DispatchContext();

    // Custom deleter to allow unique_ptr with private destructor
    struct Deleter {
        void operator()(DispatchContext* p) const { delete p; }
    };
    friend struct Deleter;

    bool fast_dispatch_enabled_ = false;
    uint32_t num_fd_inits_ = 0;
    // SD command queues stashed during an FD session, restored on terminate.
    // Defined in the .cpp to avoid exposing MeshCommandQueueBase in this header.
    struct StashedQueues;
    std::unique_ptr<StashedQueues> stashed_sd_queues_;
    static std::unique_ptr<DispatchContext, Deleter> dispatch_context_ptr_;
};

// Dispatches a pre-compiled program to a device. Requires prior LaunchProgram call on another device
// to compile and finalize the program. Uses thread-local launch messages for safe concurrent dispatch.
void DispatchCompiledProgramToDevice(IDevice* device, Program& program);

// Configure a program on a device without running it: binaries, CB configs, runtime args and the
// launch message land in L1, but no go signal is sent. The kernel-config block can then be read
// back (CaptureKernelConfig) without the program having executed. Safe to repeat on one
// device; each call overwrites the previous config.
void ConfigureProgramWithoutLaunch(IDevice* device, Program& program);

// A relocatable capture of the kernel config a core currently runs. The detailed launch-message
// layout remains private to Metal; consumers get only the source block's L1 range and an opaque
// copy of the launch kernel config needed to restore it at another base.
class CapturedKernelConfig {
public:
    uint32_t kernel_config_base() const;
    uint32_t kernel_config_size() const;
    const std::vector<uint8_t>& launch_kernel_config() const;

private:
    struct Impl;
    explicit CapturedKernelConfig(std::shared_ptr<const Impl> impl);

    std::shared_ptr<const Impl> impl_;

    friend CapturedKernelConfig CaptureKernelConfig(IDevice* device, const CoreCoord& logical_core);
};

CapturedKernelConfig CaptureKernelConfig(IDevice* device, const CoreCoord& logical_core);

}  // namespace experimental

}  // namespace tt::tt_metal
