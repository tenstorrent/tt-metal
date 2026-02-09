// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace experimental {

// This class provides APIs to dynamically enable and teardown Fast Dispatch during runtime.
// Functionality is currently limited to Galaxy clusters.
// Note: The functionality in this class is extremely application specific, and will likely be
// removed once we implement a proper weight loading solution for Low Latency Decode.
// As such its exposed as experimental.

class DispatchContext {
public:
    static DispatchContext& get();
    void initialize_fast_dispatch(distributed::MeshDevice* mesh_device);
    void terminate_fast_dispatch(distributed::MeshDevice* mesh_device);

private:
    DispatchContext() = default;
    ~DispatchContext() = default;

    // Custom deleter to allow unique_ptr with private destructor
    struct Deleter {
        void operator()(DispatchContext* p) const { delete p; }
    };
    friend struct Deleter;

    bool fast_dispatch_enabled_ = false;
    uint32_t num_fd_inits_ = 0;
    static std::unique_ptr<DispatchContext, Deleter> dispatch_context_ptr_;
};

}  // namespace experimental

}  // namespace tt::tt_metal
