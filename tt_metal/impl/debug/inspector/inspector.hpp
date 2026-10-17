// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <string>
#include "impl/program/program_impl.hpp"
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include "impl/dispatch/dispatch_core_common.hpp"
#include "mesh_coord.hpp"

namespace tt::tt_metal {

namespace distributed {
class MeshBuffer;
class MeshDevice;
class MeshDeviceImpl;
class MeshWorkloadImpl;
class MeshSocket;
}  // namespace distributed

class MetalContext;

namespace inspector {
class Data;
}  // namespace inspector

class Inspector {
public:
    // Creates the inspector session for `context`. The caller (the MetalContext) owns the returned session and must
    // keep `context` alive for as long as the session lives. Returns nullptr if the inspector is disabled or `context`
    // targets a mock device.
    static std::unique_ptr<inspector::Data> initialize(
        MetalContext& context, std::optional<int> rank, uint64_t fw_compile_hash);
    // Whether `context` has an inspector session.
    static bool is_enabled(const MetalContext& context);
    // Whether `context` has an inspector session that captures tensor specs on op dispatch.
    static bool capture_tensor_specs(const MetalContext& context);
    static void serialize_rpc(const MetalContext& context);

    static void program_created(const detail::ProgramImpl* program) noexcept;
    static void program_destroyed(const detail::ProgramImpl* program) noexcept;
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

    // `mesh_device` is the buffer's device, or nullptr if it is already gone; the event is then dropped.
    static void mesh_buffer_allocated(
        const distributed::MeshBuffer* mesh_buffer, const distributed::MeshDevice* mesh_device) noexcept;
    static void mesh_buffer_deallocated(
        const distributed::MeshBuffer* mesh_buffer, const distributed::MeshDevice* mesh_device) noexcept;

    static void mesh_socket_created(const distributed::MeshSocket* socket) noexcept;

    static void global_semaphore_created(const distributed::MeshBuffer* buffer, const CoreRangeSet& cores) noexcept;
    static void global_semaphore_reset(const distributed::MeshBuffer* buffer, uint32_t value) noexcept;

    static void emit_debug_entry(
        const MetalContext& context,
        const distributed::MeshWorkloadImpl* mesh_workload,
        uint64_t runtime_id,
        std::string_view operation_name,
        std::vector<TensorSpec> tensor_specs,
        std::optional<distributed::MeshTraceId> trace_id = std::nullopt) noexcept;
    static void release_trace(const MetalContext& context, distributed::MeshTraceId trace_id) noexcept;

    // static method for logging dispatch core info
    static void set_dispatch_core_info(
        const MetalContext& context,
        const tt_cxy_pair& virtual_core,
        const tt::tt_metal::DispatchWorkerType& type,
        uint8_t cq_id,
        ChipId device_id,
        ChipId servicing_device_id);

    // static method for logging dispatch_s core info
    static void set_dispatch_s_core_info(
        const MetalContext& context,
        const tt_cxy_pair& virtual_core,
        const tt::tt_metal::DispatchWorkerType& type,
        uint8_t cq_id,
        ChipId device_id,
        ChipId servicing_device_id);

    // static method for logging prefetcher core info
    static void set_prefetcher_core_info(
        const MetalContext& context,
        const tt_cxy_pair& virtual_core,
        const tt::tt_metal::DispatchWorkerType& type,
        uint8_t cq_id,
        ChipId device_id,
        ChipId servicing_device_id);

    // Helper function to get the ELF path for a given kernel and processor index (risc_id). The mapping
    // is captured at compile time, so it remains valid after the Kernel object has been destroyed and
    // correctly resolves riscs that share a single binary. Returns an empty string if data is not available.
    static std::string get_kernel_elf_path(
        const MetalContext& context, int watcher_kernel_id, uint32_t processor_index);
    static void enable_kernel_path_collection(const MetalContext& context);
};

}  // namespace tt::tt_metal
