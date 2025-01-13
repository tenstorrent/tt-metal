// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <map>
#include <vector>

namespace tt::tt_metal {

using chip_id_t = int;
inline namespace v0 {
class IDevice;
}  // namespace v0
class DispatchCoreConfig;

namespace distributed {

class MeshDeviceConfig;

// Resource management class / RAII wrapper for resources of the mesh
// Users can implement this to mock physical devices.
class IMeshHandle {
public:
    virtual ~IMeshHandle() = default;
    virtual const std::vector<IDevice*>& get_devices() const = 0;
};

// Resource management class / RAII wrapper for *physical devices* of the mesh
class MeshHandle : public IMeshHandle {
    std::map<chip_id_t, IDevice*> opened_devices_;
    std::vector<IDevice*> devices_;

public:
    // Constructor acquires physical resources
    MeshHandle(
        size_t l1_small_size,
        size_t trace_region_size,
        size_t num_command_queues,
        const DispatchCoreConfig& dispatch_core_config,
        const MeshDeviceConfig& config);

    // Destructor releases physical resources
    ~MeshHandle() override;
    MeshHandle(const MeshHandle&) = delete;
    MeshHandle& operator=(const MeshHandle&) = delete;

    const std::vector<IDevice*>& get_devices() const override;
};
}  // namespace distributed

}  // namespace tt::tt_metal
