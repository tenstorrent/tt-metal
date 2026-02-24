// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "context/context_descriptor.hpp"

namespace tt::tt_metal {

class MetalliumObject {
public:
    // Create a MetalliumObject according to a descriptor. Only one MetalliumObject representing a physical cluster is
    // allowed due to UMD limitations.
    explicit MetalliumObject(const std::shared_ptr<ContextDescriptor>& descriptor);
    ~MetalliumObject();

private:
    friend class MetalContext;

    // Call this from the constructor to verify the firmware version and create the base objects
    // according to the descriptor
    void verify_fw_initialize_base_objects(const std::shared_ptr<ContextDescriptor>& descriptor);

    // Below objects are listed in the order of dependency
    std::unique_ptr<llrt::RunTimeOptions> rtoptions_;
    std::unique_ptr<Hal> hal_;

    std::unique_ptr<Cluster> cluster_;
    std::unique_ptr<tt_fabric::ControlPlane> control_plane_;
    // Above objects are initialized in the order of dependency
};

}  // namespace tt::tt_metal
