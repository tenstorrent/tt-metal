// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "impl/context/context_descriptor.hpp"

namespace tt::tt_metal {

class MetaliumEnv {
public:
    MetaliumEnv();
    ~MetaliumEnv();

    // Init a MetaliumEnv according to a descriptor. Note: Only one MetaliumEnv representing a physical cluster
    // is allowed due to UMD limitations.
    void initialize(const std::shared_ptr<MetaliumEnvDescriptor>& descriptor);

    // Destroy the object. This function may only be called when the object is no longer needed.
    void destroy();

    llrt::RunTimeOptions& get_rtoptions() const;
    const tt::tt_metal::Hal& get_hal() const;
    tt::Cluster& get_cluster() const;

    bool is_initialized() const;

private:
    // Create the base objects according to the descriptor
    void initialize_base_objects(const std::shared_ptr<MetaliumEnvDescriptor>& descriptor);

    // Verify the firmware version and enable the appropriate features
    void verify_fw_capabilities();

    bool initialized_ = false;

    // Below objects are listed in the order of dependency
    std::unique_ptr<llrt::RunTimeOptions> rtoptions_ = nullptr;
    std::unique_ptr<Cluster> cluster_ = nullptr;
    std::unique_ptr<Hal> hal_ = nullptr;
    // Above objects are initialized in the order of dependency
};

}  // namespace tt::tt_metal
