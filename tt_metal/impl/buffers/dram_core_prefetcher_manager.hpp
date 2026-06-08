// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <tt-metalium/experimental/dram_core_prefetcher.hpp>
#include <tt-metalium/experimental/global_circular_buffer.hpp>

namespace tt::tt_metal {

class IDevice;
class MeshTensor;
class Program;

namespace distributed {

class MeshDevice;

// Owns the DRAM-core (DRISC) prefetcher subsystem for a single MeshDevice. State is held
// per-device (one Program per IDevice in the mesh) so each DRISC kernel runs on its own
// chip. Single-prefetcher-at-a-time invariant: start() asserts is_active() is false.
//
// Lifecycle:
//   * start(...) builds the Program(s), launches with wait_until_cores_done=false on
//     each device's slow-dispatch path, and returns immediately. The host thread is then
//     free to enqueue consumer kernels (matmuls) while the prefetcher kernel pushes
//     layers into the GCB.
//   * stop() blocks each launched Program's natural exit (kernel finishes its
//     num_layers loop), then releases per-device state. No-op if not active.
//   * Destructor calls stop().
class DramCorePrefetcherManager {
public:
    explicit DramCorePrefetcherManager(MeshDevice* mesh_device);
    ~DramCorePrefetcherManager();

    DramCorePrefetcherManager(const DramCorePrefetcherManager&) = delete;
    DramCorePrefetcherManager& operator=(const DramCorePrefetcherManager&) = delete;
    DramCorePrefetcherManager(DramCorePrefetcherManager&&) = delete;
    DramCorePrefetcherManager& operator=(DramCorePrefetcherManager&&) = delete;

    void start(
        const std::vector<const MeshTensor*>& input_tensors,
        const experimental::GlobalCircularBuffer& gcb,
        const experimental::DramCorePrefetcherConfig& config);

    // Block until the per-device programs finish their num_layers loop, then release
    // state. Safe to call when inactive (no-op).
    void stop();

    bool is_active() const { return !per_device_state_.empty(); }

private:
    struct PerDeviceState {
        IDevice* device = nullptr;
        std::unique_ptr<Program> program;
    };

    MeshDevice* mesh_device_;
    std::vector<PerDeviceState> per_device_state_;

    // Held for the duration of the launch so the buffers backing the input tensors and
    // the GCB stay alive. The caller's invariant in StartDramCorePrefetcher requires
    // them to outlive stop(); we copy/retain here only as a defense-in-depth measure
    // against accidental destruction races.
    std::optional<experimental::GlobalCircularBuffer> gcb_;
};

}  // namespace distributed
}  // namespace tt::tt_metal
