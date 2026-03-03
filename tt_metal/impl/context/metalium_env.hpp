// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <memory>
#include "impl/context/context_descriptor.hpp"
#include "impl/context/context_id.hpp"

namespace tt::tt_metal {

class MetaliumEnv {
public:
    // Construct and fully initialize a MetaliumEnv. Only one instance representing a physical cluster
    // is allowed due to UMD limitations.
    explicit MetaliumEnv(MetaliumEnvDescriptor descriptor = {});
    ~MetaliumEnv();

    // Destroy the object. This function may only be called when the object is no longer needed.
    void destroy();

    llrt::RunTimeOptions& get_rtoptions() const;
    const tt::tt_metal::Hal& get_hal() const;
    tt::Cluster& get_cluster() const;

    bool is_initialized() const;

    const MetaliumEnvDescriptor& get_descriptor() const;

private:
    friend class MetalContext;

    // Ownership tracking: ensures at most one MetalContext is bound to this env at a time.
    void acquire(ContextId context_id);
    void release(ContextId context_id);
    bool is_acquired() const;

    void initialize_base_objects();

    // Verify the firmware version and enable the appropriate features
    void verify_fw_capabilities();

    static constexpr ContextId NO_OWNER = -1;

    bool initialized_ = false;
    std::atomic<ContextId> owning_context_id_{NO_OWNER};
    MetaliumEnvDescriptor descriptor_;

    // Below objects are listed in the order of dependency
    std::unique_ptr<llrt::RunTimeOptions> rtoptions_ = nullptr;
    std::unique_ptr<Cluster> cluster_ = nullptr;
    std::unique_ptr<Hal> hal_ = nullptr;
    // Above objects are initialized in the order of dependency
};

}  // namespace tt::tt_metal
