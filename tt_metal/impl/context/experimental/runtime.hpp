// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "impl/context/experimental/context.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>

namespace tt {
class Cluster;
}  // namespace tt

namespace tt::llrt {
class RunTimeOptions;
}  // namespace tt::llrt

namespace tt::tt_metal {
class Hal;
class MetaliumObjectAccessor;
}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental {

// Interface to query system state
class MetaliumObject {
private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    MetaliumObject();

public:
    ~MetaliumObject();

    MetaliumObject(const MetaliumObject&) = delete;
    MetaliumObject& operator=(const MetaliumObject&) = delete;
    MetaliumObject(MetaliumObject&&) = delete;
    MetaliumObject& operator=(MetaliumObject&&) = delete;

    // Make these public for now
    std::shared_ptr<tt::Cluster> cluster();
    std::shared_ptr<tt::tt_metal::Hal> hal();
    tt::llrt::RunTimeOptions& rtoptions();
    std::shared_ptr<tt::tt_fabric::ControlPlane> get_control_plane();

    //
    // Create an instance of the MetaliumObject for current cluster visible to the system which is defined by
    // TT_VISIBLE_DEVICES. Only one instance may be created per cluster. Creating more than one instance per cluster
    // will result in undefined behavior.
    //
    static std::shared_ptr<MetaliumObject> create();
};

class RuntimeBackend;

// A runtime context for Metalium.
// Users create and own Context objects directly, passing dependencies explicitly.
// For silicon contexts, a MetaliumObject must be provided as a dependency.
// For mock contexts, no MetaliumObject is needed.
class Context {
public:
    // Create a context with the given descriptor.
    // For silicon contexts (is_mock_device is false), metalium_object must be provided.
    // For mock contexts (is_mock_device is true), metalium_object can be nullptr.
    Context(std::shared_ptr<ContextDescriptor> descriptor, std::shared_ptr<MetaliumObject> metalium_object = nullptr);

    ~Context();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;

    // Check if this is a mock device context.
    bool is_mock_device() const;

    // Returns the descriptor linked to this context.
    std::shared_ptr<ContextDescriptor> get_descriptor() const;

    // Get the MetaliumObject for silicon contexts.
    // Returns nullptr for mock contexts.
    std::shared_ptr<MetaliumObject> get_metalium_object();

private:
    std::unique_ptr<RuntimeBackend> impl_;
    std::shared_ptr<ContextDescriptor> descriptor_;
};

}  // namespace tt::tt_metal::experimental
