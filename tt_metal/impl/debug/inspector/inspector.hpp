// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <string>
#include "impl/context/context_types.hpp"
#include "impl/program/program_impl.hpp"
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include "impl/dispatch/dispatch_core_common.hpp"
#include "mesh_coord.hpp"

namespace tt::tt_metal {

namespace distributed {
class MeshBuffer;
class MeshDeviceImpl;
class MeshWorkloadImpl;
class MeshSocket;
}  // namespace distributed

class MetalContext;

namespace inspector {
class Data;
class RpcServer;  // NOLINT(cppcoreguidelines-virtual-class-destructor)
}  // namespace inspector

class Inspector {
public:
    // True if an inspector session is active, i.e. a MetalContext on which the inspector is enabled was initialized.
    static bool is_enabled();

    // Whether tensor specs should be captured on op dispatch. False if there is no active session.
    static bool should_capture_tensor_specs();

    // Creates the inspector session for `context` and registers it so that the hooks below record to it. The caller
    // (the MetalContext) owns the returned session and must keep `context` alive for as long as the session lives.
    // Returns nullptr if the inspector is disabled or `context` targets a mock device. Only one context can be
    // inspected at a time: if another context already has a session, a warning is logged and nullptr is returned.
    static std::unique_ptr<inspector::Data> initialize(
        MetalContext& context, std::optional<int> rank, uint64_t fw_compile_hash);
    static void serialize_rpc(const MetalContext& context);

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
        const tt::tt_metal::JitBuildOptions& build_options,
        const std::string& binary_root) noexcept;
    static void program_compile_finished(
        const detail::ProgramImpl* program, const IDevice* device, uint64_t build_key) noexcept;

    static void mesh_device_created(
        const distributed::MeshDeviceImpl* mesh_device, std::optional<int> parent_mesh_id) noexcept;
    static void mesh_device_destroyed(const distributed::MeshDeviceImpl* mesh_device) noexcept;
    static void mesh_device_initialized(const distributed::MeshDeviceImpl* mesh_device) noexcept;

    static void mesh_buffer_allocated(const distributed::MeshBuffer* mesh_buffer) noexcept;
    static void mesh_buffer_deallocated(const distributed::MeshBuffer* mesh_buffer) noexcept;

    static void mesh_socket_created(const distributed::MeshSocket* socket) noexcept;

    static void global_semaphore_created(const distributed::MeshBuffer* buffer, const CoreRangeSet& cores) noexcept;
    static void global_semaphore_reset(const distributed::MeshBuffer* buffer, uint32_t value) noexcept;

    static void mesh_workload_created(const distributed::MeshWorkloadImpl* mesh_workload) noexcept;
    static void mesh_workload_destroyed(const distributed::MeshWorkloadImpl* mesh_workload) noexcept;
    static void mesh_workload_add_program(
        const distributed::MeshWorkloadImpl* mesh_workload,
        const distributed::MeshCoordinateRange& device_range,
        std::size_t program_id) noexcept;
    static void mesh_workload_set_program_binary_status(
        const distributed::MeshWorkloadImpl* mesh_workload, std::size_t mesh_id, ProgramBinaryStatus status) noexcept;
    static void emit_debug_entry(
        const distributed::MeshWorkloadImpl* mesh_workload,
        uint64_t runtime_id,
        std::string_view operation_name,
        std::vector<TensorSpec> tensor_specs,
        std::optional<distributed::MeshTraceId> trace_id = std::nullopt) noexcept;
    static void release_trace(distributed::MeshTraceId trace_id) noexcept;

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

    // Helper function to get the ELF path for a given kernel and processor index (risc_id). The mapping
    // is captured at compile time, so it remains valid after the Kernel object has been destroyed and
    // correctly resolves riscs that share a single binary. Returns an empty string if data is not available.
    static std::string get_kernel_elf_path(int watcher_kernel_id, uint32_t processor_index);
    static void enable_kernel_path_collection();

    static inspector::RpcServer& get_rpc_server();

private:
    friend class inspector::Data;
    // Called by the session's destructor.
    static void unregister_session(const inspector::Data* session) noexcept;
};

}  // namespace tt::tt_metal
