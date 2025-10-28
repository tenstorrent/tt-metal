// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "impl/program/program_impl.hpp"
#include "mesh_coord.hpp"

namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
class MeshWorkloadImpl;
}  // namespace distributed

namespace inspector {
class Data;
class RpcServer;  // NOLINT(cppcoreguidelines-virtual-class-destructor)
}  // namespace inspector

class Inspector {
public:
    static bool is_enabled();

    static std::unique_ptr<inspector::Data> initialize();
    static void serialize_rpc();

    // Operation tracking
    static void track_operation(
        std::optional<std::int64_t> device_op_id, const std::string& operation_name, const std::string& arguments);

    // Get callstack for current location (handles both C++ and Python)
    static std::string get_call_stack();

    static void program_created(const detail::ProgramImpl* program) noexcept;
    static void program_destroyed(const detail::ProgramImpl* program) noexcept;
    static void program_set_binary_status(
        const detail::ProgramImpl* program, std::size_t device_id, ProgramBinaryStatus status) noexcept;
    static void program_compile_started(
        const detail::ProgramImpl* program, const IDevice* device, uint32_t build_key) noexcept;
    static void program_compile_already_exists(
        const detail::ProgramImpl* program, const IDevice* device, uint32_t build_key) noexcept;
    static void program_kernel_compile_finished(
        const detail::ProgramImpl* program,
        const IDevice* device,
        const std::shared_ptr<Kernel>& kernel,
        const tt::tt_metal::JitBuildOptions& build_options) noexcept;
    static void program_compile_finished(
        const detail::ProgramImpl* program, const IDevice* device, uint32_t build_key) noexcept;

    static void mesh_device_created(
        const distributed::MeshDevice* mesh_device, std::optional<int> parent_mesh_id) noexcept;
    static void mesh_device_destroyed(const distributed::MeshDevice* mesh_device) noexcept;
    static void mesh_device_initialized(const distributed::MeshDevice* mesh_device) noexcept;

    static void mesh_workload_created(const distributed::MeshWorkloadImpl* mesh_workload) noexcept;
    static void mesh_workload_destroyed(const distributed::MeshWorkloadImpl* mesh_workload) noexcept;
    static void mesh_workload_add_program(
        const distributed::MeshWorkloadImpl* mesh_workload,
        const distributed::MeshCoordinateRange& device_range,
        std::size_t program_id) noexcept;
    static void mesh_workload_set_program_binary_status(
        const distributed::MeshWorkloadImpl* mesh_workload, std::size_t mesh_id, ProgramBinaryStatus status) noexcept;

    static inspector::RpcServer& get_rpc_server();

    static void set_build_env_fw_compile_hash(uint64_t fw_compile_hash);
};

}  // namespace tt::tt_metal
