// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "impl/context/context_descriptor.hpp"

namespace tt::tt_metal {

class MetalliumObject {
public:
    MetalliumObject();
    ~MetalliumObject();

    // Init a MetalliumObject according to a descriptor. Note: Only one MetalliumObject representing a physical cluster
    // is allowed due to UMD limitations.
    void initialize(const std::shared_ptr<MetalliumObjectDescriptor>& descriptor);

    // Destroy the object. This function may only be called when the object is no longer needed.
    void destroy();

    llrt::RunTimeOptions& get_rtoptions() const;
    const tt::tt_metal::Hal& get_hal() const;
    tt::Cluster& get_cluster() const;

    bool is_initialized() const;

private:
    // Verify the firmware version and create the base objects according to the descriptor
    void verify_fw_initialize_base_objects(const std::shared_ptr<MetalliumObjectDescriptor>& descriptor);

    bool initialized_ = false;

    // Below objects are listed in the order of dependency
    std::unique_ptr<llrt::RunTimeOptions> rtoptions_ = nullptr;
    std::unique_ptr<Cluster> cluster_ = nullptr;
    std::unique_ptr<Hal> hal_ = nullptr;
    // Above objects are initialized in the order of dependency
};

}  // namespace tt::tt_metal
