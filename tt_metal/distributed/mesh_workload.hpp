// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_device.hpp"
#include "tt_metal/host_api.hpp"

namespace tt::tt_metal::distributed {
using LogicalDeviceRange = CoreRange;
using RuntimeArgsPerCore = std::vector<std::vector<RuntimeArgsData>>;
class MeshWorkload {
    // A MeshWorkload can be fully described using a set of programs mapped to different Logical Device Regions
    // in a Mesh + configurable runtime Args
private:
    std::unordered_map<LogicalDeviceRange, Program> programs_;
    std::unordered_map<LogicalDeviceRange, std::unordered_map<KernelHandle, RuntimeArgsPerCore>> runtime_args_;
    void compile(std::shared_ptr<MeshDevice>& mesh_device);
    void finalize(std::shared_ptr<MeshDevice>& mesh_device);
    bool compiled_ = false;
    bool finalized_ = false;

public:
    MeshWorkload() {};
    void add_program(const LogicalDeviceRange& device_range, Program& program);
    void enqueue(std::shared_ptr<MeshDevice>& mesh_device, uint8_t cq_id, bool blocking);
    bool is_compiled() const { return this->compiled_; }
    bool is_finalized() const { return this->finalized_; }
};
}  // namespace tt::tt_metal::distributed
