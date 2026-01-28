// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "impl/program/program_impl.hpp"
#include "impl/dispatch/dispatch_core_common.hpp"
#include "mesh_coord.hpp"

namespace tt::tt_metal {

namespace distributed {
class MeshDeviceImpl;
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

    static void program_created(const detail::ProgramImpl* program) noexcept;
    static void program_destroyed(const detail::ProgramImpl* program) noexcept;
    static void program_set_binary_status(
        const detail::ProgramImpl* program, std::size_t device_id, ProgramBinaryStatus status) noexcept;
    static void program_compile_started(
        const detail::ProgramImpl* program, const IDevice* device, uint64_t build_key) noexcept;
    static void program_compile_already_exists(
        const detail::ProgramImpl* program, const IDevice* device, uint64_t build_key) noexcept;
    static void program_kernel_compile_finished(
        const detail::ProgramImpl* program,
        const IDevice* device,
        const std::shared_ptr<Kernel>& kernel,
        const tt::tt_metal::JitBuildOptions& build_options) noexcept;
    static void program_compile_finished(
        const detail::ProgramImpl* program, const IDevice* device, uint64_t build_key) noexcept;

    static void mesh_device_created(
        const distributed::MeshDeviceImpl* mesh_device, std::optional<int> parent_mesh_id) noexcept;
    static void mesh_device_destroyed(const distributed::MeshDeviceImpl* mesh_device) noexcept;
    static void mesh_device_initialized(const distributed::MeshDeviceImpl* mesh_device) noexcept;

    static void mesh_workload_created(const distributed::MeshWorkloadImpl* mesh_workload) noexcept;
    static void mesh_workload_destroyed(const distributed::MeshWorkloadImpl* mesh_workload) noexcept;
    static void mesh_workload_add_program(
        const distributed::MeshWorkloadImpl* mesh_workload,
        const distributed::MeshCoordinateRange& device_range,
        std::size_t program_id) noexcept;
    static void mesh_workload_set_program_binary_status(
        const distributed::MeshWorkloadImpl* mesh_workload, std::size_t mesh_id, ProgramBinaryStatus status) noexcept;
    static void mesh_workload_set_operation_name_and_parameters(
        const distributed::MeshWorkloadImpl* mesh_workload,
        std::string_view operation_name,
        std::string_view operation_parameters) noexcept;
    static void mesh_workload_set_runtime_id(
        const distributed::MeshWorkloadImpl* mesh_workload, uint64_t runtime_id) noexcept;

    // static method for logging dispatch core info
    static void set_dispatch_core_info(
        const tt_cxy_pair& virtual_core,
        const tt::tt_metal::DispatchWorkerType& type,
        uint8_t cq_id,
        ChipId device_id,
        ChipId servicing_device_id);

    // static method for logging dispatch_s core info
    static void set_dispatch_s_core_info(
        const tt_cxy_pair& virtual_core,
        const tt::tt_metal::DispatchWorkerType& type,
        uint8_t cq_id,
        ChipId device_id,
        ChipId servicing_device_id);

    // static method for logging prefetcher core info
    static void set_prefetcher_core_info(
        const tt_cxy_pair& virtual_core,
        const tt::tt_metal::DispatchWorkerType& type,
        uint8_t cq_id,
        ChipId device_id,
        ChipId servicing_device_id);

    // static method for clearing all core info to clear stale entries
    static void clear_all_core_info();

    static inspector::RpcServer& get_rpc_server();

    static void set_build_env_fw_compile_hash(uint64_t fw_compile_hash);
};

}  // namespace tt::tt_metal
