#pragma once

#include <cstdint>
#include <memory>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>

namespace tt::tt_metal::experimental {

using distributed::MeshCommandQueue;
using distributed::MeshDevice;
using distributed::MeshWorkload;

class CommandList;

// Records MeshWorkloads for later replay. Only one active builder may exist per
// MeshDevice.
class CommandListBuilder {
public:
    // Captures the active sub-device manager. add() and build() require it to
    // be active.
    explicit CommandListBuilder(MeshDevice& device);

    CommandListBuilder(const CommandListBuilder&) = delete;
    CommandListBuilder& operator=(const CommandListBuilder&) = delete;
    CommandListBuilder(CommandListBuilder&&) noexcept;
    CommandListBuilder& operator=(CommandListBuilder&&) noexcept;
    ~CommandListBuilder();

    // Records one workload without launching it. Compiles it and uploads any
    // required kernel binaries through the current thread's command queue.
    void add(MeshWorkload& workload);

    // Builds an independent command list bound to cq. Recorded device addresses
    // must remain valid through replay.
    CommandList build(MeshCommandQueue& cq) const;

    MeshDevice& device() const;

    // Clears all recorded workloads.
    void clear();

    // Releases the builder lock and invalidates the builder. Repeated calls
    // have no effect.
    void deallocate();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Move-only handle to a replayable command list stored in device DRAM.
// Destruction releases its device resources.
class CommandList {
public:
    CommandList(const CommandList&) = delete;
    CommandList& operator=(const CommandList&) = delete;
    CommandList(CommandList&&) noexcept;
    CommandList& operator=(CommandList&&) noexcept;
    ~CommandList();

    // Replays on the command queue used by build(). The recorded sub-device
    // manager must be active. If blocking, waits for completion.
    void replay(bool blocking) const;

    MeshDevice& device() const;

    uint8_t cq_id() const;

    // Releases device resources and invalidates the handle. Repeated calls
    // have no effect.
    void deallocate();

private:
    friend class CommandListBuilder;
    class Impl;
    explicit CommandList(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

// Replays command_list on cq, which must match the device and queue used to
// build it.
void EnqueueCommandList(MeshCommandQueue& cq, CommandList& command_list, bool blocking);

}  // namespace tt::tt_metal::experimental
