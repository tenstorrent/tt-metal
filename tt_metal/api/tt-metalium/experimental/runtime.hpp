// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/context.hpp>

namespace tt {
class Cluster;
}  // namespace tt

namespace tt::llrt {
class RunTimeOptions;
}  // namespace tt::llrt

namespace tt::tt_metal {
class Hal;
}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"

// Manages the runtime state and context management
class Runtime {
private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    Runtime();

public:
    ~Runtime();

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime(Runtime&&) = delete;
    Runtime& operator=(Runtime&&) = delete;

    // Returns the Runtime instance.
    static Runtime& instance();

    // Try to bind a context. Returns false if unsuccessful.
    bool bind_context(const std::shared_ptr<Context>& context);

    // Try to unbind a context. Returns false if unsuccessful.
    bool unbind_context();

    // Returns true if a context is bound.
    bool has_bound_context() const;

    // Returns the current context.
    std::shared_ptr<Context> get_context() const;
};

#pragma clang diagnostic pop

// Interface to query system state
class ClusterQuery {
private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    ClusterQuery();

    tt::Cluster& cluster();
    const tt::Cluster& cluster() const;
    tt::tt_metal::Hal& hal();
    const tt::tt_metal::Hal& hal() const;
    tt::llrt::RunTimeOptions& rtoptions();
    const tt::llrt::RunTimeOptions& rtoptions() const;

    friend class Runtime;

public:
    ~ClusterQuery();

    ClusterQuery(const ClusterQuery&) = delete;
    ClusterQuery& operator=(const ClusterQuery&) = delete;
    ClusterQuery(ClusterQuery&&) = delete;
    ClusterQuery& operator=(ClusterQuery&&) = delete;

    // Returns the instance of the ClusterQuery
    static ClusterQuery& instance();

    // Initialize the ClusterQuery. This must be called before any queries or Runtime::bind().
    bool initialize();

    // Teardown the ClusterQuery. The ClusterQuery cannot be destructed until the runtime is unbound from any context.
    bool teardown();

    // Returns true if the ClusterQuery is initialized.
    bool is_initialized() const;

    // Returns true if a context is active. The ClusterQuery cannot be destructed until the runtime is unbound.
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
