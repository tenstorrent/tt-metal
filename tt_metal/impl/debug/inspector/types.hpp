// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <memory>
#include <unordered_map>

#include "impl/program/program_impl.hpp"
#include "impl/dispatch/dispatch_core_common.hpp"

namespace tt::tt_metal {
    class Inspector;
    class MetalContext;

    namespace distributed {
    class MeshDeviceImpl;
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
    const distributed::MeshDeviceImpl* mesh_device = nullptr;
    int mesh_id{};
    std::optional<int> parent_mesh_id;
    bool initialized = false;
};

struct MeshWorkloadRuntimeIdEntry {
    uint64_t workload_id = 0;
    uint64_t runtime_id = 0;
};

struct MeshWorkloadData {
    const distributed::MeshWorkloadImpl* mesh_workload = nullptr;
    uint64_t mesh_workload_id{};
    std::unordered_map<int, ProgramBinaryStatus> binary_status_per_device;
    std::string name;
    std::string parameters;
};

struct CoreInfo {
    tt::tt_metal::DispatchWorkerType worker_type;
    ChipId device_id;
    ChipId servicing_device_id;
    uint8_t cq_id;
};

}  // namespace tt::tt_metal::inspector
