// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include <tt-metalium/experimental/dram_core_kernels.hpp>

namespace tt::tt_metal {
class IDevice;
class Program;
}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental {

// What a LaunchDramCoreKernels call left running: one program per device, and the owner's stop protocol.
struct DramCoreKernelsState {
    std::vector<IDevice*> devices;
    std::vector<std::unique_ptr<Program>> programs;
    std::function<void()> on_close;
    bool done = false;

    void wait_done();
};

namespace detail {
// MeshDevice close: runs the stop protocol of every launch still running and waits for its kernels.
void StopDramCoreKernelsAtClose(std::vector<std::shared_ptr<DramCoreKernelsState>>& launches);
}  // namespace detail

}  // namespace tt::tt_metal::experimental
