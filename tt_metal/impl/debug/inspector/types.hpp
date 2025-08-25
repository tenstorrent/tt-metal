// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <memory>
#include <unordered_map>

#include "impl/program/program_impl.hpp"

namespace tt::tt_metal {
    class Inspector;
    class MetalContext;

    namespace distributed {
        class MeshDevice;
        class MeshWorkloadImpl;
    }
}

namespace tt::tt_metal::inspector {

using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

struct KernelData {
    std::weak_ptr<Kernel> kernel;
    std::string name;
    std::string path;
    std::string source;
    int watcher_kernel_id{};
};

struct ProgramData {
    std::weak_ptr<const detail::ProgramImpl> program;
    uint64_t program_id{};
    time_point compile_started_timestamp;
    time_point compile_finished_timestamp;
    std::unordered_map<int, KernelData> kernels;
    std::unordered_map<std::size_t, ProgramBinaryStatus> binary_status_per_device;
};

struct MeshDeviceData {
    const distributed::MeshDevice* mesh_device;
    int mesh_id{};
    std::optional<int> parent_mesh_id;
    bool initialized = false;
};

struct MeshWorkloadData {
    const distributed::MeshWorkloadImpl* mesh_workload;
    uint64_t mesh_workload_id{};
    std::unordered_map<int, ProgramBinaryStatus> binary_status_per_device;
};

}  // namespace tt::tt_metal::inspector
