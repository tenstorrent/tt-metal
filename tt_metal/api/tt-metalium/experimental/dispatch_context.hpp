// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tt::tt_metal {

class Device;
class IDevice;
class MetalContext;
class Program;
enum class DispatchCoreAxis;

namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace experimental {

// Options for a manual fast-dispatch session.
struct FastDispatchSetupOptions {
    // Warn and proceed when fast-dispatch firmware will overwrite an existing
    // L1 allocation. The conflicting allocation may be corrupted.
    bool allow_destructive = false;

    // The session will issue only host-to-device writes. This excludes the
    // prefetch ringbuffer from the checked footprint, but still includes the
    // command-data queue and scratch staging used by pinned writes.
    bool write_only = false;
};

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
    ::tt::tt_metal::DispatchCoreAxis get_dispatch_core_axis(distributed::MeshDevice* mesh_device) const;
    void initialize_fast_dispatch(distributed::MeshDevice* mesh_device);
    void initialize_fast_dispatch(distributed::MeshDevice* mesh_device, const FastDispatchSetupOptions& options);
    void terminate_fast_dispatch(distributed::MeshDevice* mesh_device);
    void enable_asynchronous_slow_dispatch(distributed::MeshDevice* mesh_device);
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

    struct FdL1Conflict;
    std::vector<FdL1Conflict> find_fd_l1_conflicts(
        MetalContext& context,
        distributed::MeshDevice* mesh_device,
        const std::vector<::tt::tt_metal::Device*>& devices,
        bool write_only) const;
    std::string format_fd_l1_conflicts(const std::vector<FdL1Conflict>& conflicts) const;
    void unwind_failed_fd_setup(MetalContext& context, const std::vector<::tt::tt_metal::Device*>& devices);

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

}  // namespace experimental

}  // namespace tt::tt_metal
