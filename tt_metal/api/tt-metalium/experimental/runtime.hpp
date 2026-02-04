// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/context_descriptor.hpp>
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

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"

// Manages the runtime state and context management
class Context {
private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    Context();

public:
    ~Context();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;

    // Returns the Context instance.
    static Context& instance();

    // Try to bind a a descriptor. This will initialize the devices according to the descriptor. Returns false if
    // unsuccessful.
    bool set_descriptor(const std::shared_ptr<ContextDescriptor>& descriptor);

    // Remove a descriptor and return the devices to a reset state. Returns false if unsuccessful.
    bool remove_descriptor();

    // Returns true if a descriptor is set.
    bool has_descriptor() const;

    // Returns the current descriptor.
    std::shared_ptr<ContextDescriptor> get_descriptor() const;
};

#pragma clang diagnostic pop

// Interface to query system state
class MetaliumObject {
private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    MetaliumObject();

    tt::Cluster& cluster();
    const tt::Cluster& cluster() const;
    tt::tt_metal::Hal& hal();
    const tt::tt_metal::Hal& hal() const;
    tt::llrt::RunTimeOptions& rtoptions();
    const tt::llrt::RunTimeOptions& rtoptions() const;
    tt::tt_fabric::ControlPlane& get_control_plane();
    const tt::tt_fabric::ControlPlane& get_control_plane() const;

    friend class tt::tt_metal::MetaliumObjectAccessor;
    friend class SiliconRuntime;

public:
    ~MetaliumObject();

    MetaliumObject(const MetaliumObject&) = delete;
    MetaliumObject& operator=(const MetaliumObject&) = delete;
    MetaliumObject(MetaliumObject&&) = delete;
    MetaliumObject& operator=(MetaliumObject&&) = delete;

    // Returns the instance of the MetaliumObject
    static MetaliumObject& instance();

    // Initialize the MetaliumObject. This must be called before any queries or Context::bind().
    bool initialize();

    // Teardown the MetaliumObject. The MetaliumObject cannot be destructed until the runtime is unbound from any
    // context.
    bool teardown();

    // Returns true if the MetaliumObject is initialized.
    bool is_initialized() const;

    // Returns true if a context is active. The MetaliumObject cannot be destructed until the runtime is unbound.
    bool is_runtime_active() const;

    // Returns the number of visible devices which may be specified by the TT_VISIBLE_DEVICES environment variable
    int get_num_visible_devices() const;

    // Returns the number of PCIe devices
    int get_num_pcie_devices() const;

    // Returns true if the system is a Galaxy cluster
    bool is_galaxy_cluster() const;

    // Returns the PCIe device ID for a given device ID
    int get_pcie_device_id(int device_id) const;
};

}  // namespace tt::tt_metal::experimental
