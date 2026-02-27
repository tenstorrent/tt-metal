// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <memory>
#include <tt-metalium/experimental/context/context_descriptor.hpp>

namespace tt::tt_metal {

class MetaliumEnv {
public:
    // Construct and fully initialize a MetaliumEnv. Only one instance representing a physical cluster
    // is allowed due to UMD limitations.
    explicit MetaliumEnv(MetaliumEnvDescriptor descriptor = {});
    ~MetaliumEnv();

    // Destroy the object. This function may only be called when the object is no longer being used.
    // An exception is thrown if the MetaliumEnv is destroyed while it is still being used.
    void destroy();

    bool is_initialized() const;

    const MetaliumEnvDescriptor& get_descriptor() const;

private:
    friend class MetalContext;
    friend class MetaliumEnvAccessor;
    class MetaliumEnvImpl;
    std::unique_ptr<MetaliumEnvImpl> impl_ = nullptr;

    MetaliumEnvImpl& impl() { return *impl_; }

    // Ownership tracking: ensures at most one MetalContext is bound to this env at a time.
    void acquire(int context_id);
    void release(int context_id);
    bool is_acquired() const;

    static constexpr int NO_OWNER = -1;

    bool initialized_ = false;
    std::atomic<int> owning_context_id_{NO_OWNER};
    MetaliumEnvDescriptor descriptor_;
};

}  // namespace tt::tt_metal
