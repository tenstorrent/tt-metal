// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "impl/program/program_impl.hpp"
#include "impl/dispatch/dispatch_core_common.hpp"
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/mesh_trace_id.hpp>

namespace tt::tt_metal {
    class Inspector;
    class MetalContext;

    namespace distributed {
    class MeshDeviceImpl;
    class MeshWorkloadImpl;
    class MeshBuffer;
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
    // ELF paths indexed by processor index (risc_id), resolved at compile time. Processor indices not used by this
    // kernel are left empty. Served to tt-triage over RPC so it doesn't have to reconstruct paths from per-processor
    // naming conventions.
    std::vector<std::string> processor_elf_paths;
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

struct MeshSocketCore {
    uint32_t fabric_chip_id{};
    uint32_t core_x{};
    uint32_t core_y{};
};

struct MeshSocketLocalCoreData {
    MeshSocketCore core;
    uint32_t chip_id{};
    std::vector<MeshSocketCore> peers;
};

struct MeshSocketData {
    bool is_sender{};
    uint64_t config_buffer_address{};
    uint64_t data_buffer_address{};
    uint64_t fifo_size{};
    uint32_t bytes_acked_offset_bytes{};
    uint32_t bytes_acked_stride_bytes{};
    uint32_t local_mesh_id{};
    uint32_t peer_mesh_id{};
    std::vector<MeshSocketLocalCoreData> local_cores;
};

struct MeshWorkloadRuntimeEntry {
    uint64_t workload_id = 0;
    uint64_t runtime_id = 0;
    std::string_view operation_name;
    std::vector<TensorSpec> tensor_specs;
    std::optional<distributed::MeshTraceId> trace_id;
};

std::string stringify_tensor_specs(const std::vector<TensorSpec>& tensor_specs);

struct MeshWorkloadData {
    const distributed::MeshWorkloadImpl* mesh_workload = nullptr;
    uint64_t mesh_workload_id{};
    std::unordered_map<int, ProgramBinaryStatus> binary_status_per_device;
};

struct CoreInfo {
    tt::tt_metal::DispatchWorkerType worker_type;
    ChipId device_id;
    ChipId servicing_device_id;
    uint8_t cq_id;
};

}  // namespace tt::tt_metal::inspector
