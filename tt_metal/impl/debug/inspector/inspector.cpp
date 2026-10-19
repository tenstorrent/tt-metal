// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include "inspector.hpp"
#include "context/context_types.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/debug/inspector/data.hpp"
#include "impl/debug/inspector/rpc_server_generated.hpp"
#include "impl/program/program_impl.hpp"
#include "jit_build/jit_build_options.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "distributed/mesh_workload_impl.hpp"
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/mesh_buffer.hpp>
#include "distributed/mesh_socket_utils.hpp"
#include "program.hpp"
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <tt-metalium/experimental/inspector.hpp>
#include "impl/kernels/kernel.hpp"

namespace tt::tt_metal {

namespace {
// The session of the context an object belongs to, or nullptr if the inspector is not active on that context.
// Objects can outlive their context, e.g. cached programs after the mesh device owning it is closed. So the context is
// looked up by id rather than through a pointer, and the events of such objects are dropped.
inspector::Data* get_inspector_data(ContextId context_id) {
    return MetalContext::instance_exists(context_id) ? MetalContext::instance(context_id).get_inspector_data()
                                                     : nullptr;
}

inspector::Data* get_inspector_data(const distributed::MeshDeviceImpl& mesh_device) {
    return get_inspector_data(mesh_device.get_context_id());
}

inspector::Data* get_inspector_data(const distributed::MeshDevice* mesh_device) {
    return mesh_device != nullptr ? get_inspector_data(mesh_device->impl()) : nullptr;
}

// nullptr once the buffer's mesh device is gone, where MeshBuffer::device() would throw.
inspector::Data* get_inspector_data(const distributed::MeshBuffer& buffer) {
    return buffer.is_allocated() ? get_inspector_data(buffer.device()) : nullptr;
}

// A program is compiled only on devices of its context.
inspector::Data* get_inspector_data(const detail::ProgramImpl& program) {
    return get_inspector_data(program.get_context_id());
}
}  // namespace

std::unique_ptr<inspector::Data> Inspector::initialize(
    MetalContext& context, std::optional<int> rank, uint64_t fw_compile_hash) {
    const auto& rtoptions = context.rtoptions();
    if (!rtoptions.get_inspector_enabled() ||
        context.get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
        // Inspector is not enabled or not supported for this context, skipping initialization.
        return nullptr;
    }
    try {
        return std::unique_ptr<inspector::Data>(new inspector::Data(context, rank, fw_compile_hash));
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(rtoptions, "Failed to initialize Inspector: {}", e.what());
        throw;
    }
}

bool Inspector::is_enabled(const MetalContext& context) { return context.get_inspector_data() != nullptr; }

bool Inspector::capture_tensor_specs(const MetalContext& context) {
    const auto* data = context.get_inspector_data();
    return data != nullptr && data->capture_tensor_specs();
}

void Inspector::serialize_rpc(const MetalContext& context) {
    auto* data = context.get_inspector_data();
    if (!data) {
        // Inspector is not active on this context or failed to initialize, no need to print failure message again.
        return;
    }
    try {
        data->serialize_rpc();
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(data->rtoptions(), "Failed to serialize RPC: {}", e.what());
    }
}

void Inspector::program_created(const detail::ProgramImpl* program) noexcept {
    auto* data = get_inspector_data(*program);
    if (!data) {
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->programs_mutex);
        auto& program_data = data->programs_data[program->get_id()];
        program_data.program = program->weak_from_this();
        program_data.program_id = program->get_id();
        data->logger.log_program_created(program_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(data->rtoptions(), "Failed to log program created: {}", e.what());
    }
}

void Inspector::program_destroyed(const detail::ProgramImpl* program) noexcept {
    auto* data = get_inspector_data(*program);
    if (!data) {
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->programs_mutex);
        auto it = data->programs_data.find(program->get_id());
        if (it == data->programs_data.end()) {
            return;  // Not recorded by this session, e.g. created before the context was re-initialized.
        }
        data->logger.log_program_destroyed(it->second);
        for (const auto& [kernel_id, _] : it->second.kernels) {
            data->kernel_id_to_program_id.erase(kernel_id);
        }
        data->programs_data.erase(it);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(data->rtoptions(), "Failed to log program destroyed: {}", e.what());
    }
}

void Inspector::program_compile_started(
    const detail::ProgramImpl* program, const IDevice* /*device*/, uint64_t /*build_key*/) noexcept {
    auto* data = get_inspector_data(*program);
    if (!data) {
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->programs_mutex);
        auto& program_data = data->programs_data[program->get_id()];
        program_data.compile_started_timestamp = std::chrono::high_resolution_clock::now();
        data->logger.log_program_compile_started(program_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(data->rtoptions(), "Failed to log program compile started: {}", e.what());
    }
}

void Inspector::program_compile_already_exists(
    const detail::ProgramImpl* program, const IDevice* /*device*/, uint64_t /*build_key*/) noexcept {
    auto* data = get_inspector_data(*program);
    if (!data) {
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->programs_mutex);
        auto& program_data = data->programs_data[program->get_id()];
        data->logger.log_program_compile_already_exists(program_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(data->rtoptions(), "Failed to log program compile already exists: {}", e.what());
    }
}

void Inspector::program_kernel_compile_finished(
    const detail::ProgramImpl* program,
    const IDevice* device,
    const std::shared_ptr<Kernel>& kernel,
    const tt::tt_metal::JitBuildOptions& build_options,
    const std::string& binary_root) noexcept {
    auto* data = get_inspector_data(*program);
    if (!data) {
        return;
    }
    try {
        std::vector<std::string> processor_elf_paths;
        if (device != nullptr) {
            processor_elf_paths = kernel->elf_paths_by_processor_index(*device, binary_root);
        }
        std::lock_guard<std::mutex> lock(data->programs_mutex);
        auto& program_data = data->programs_data[program->get_id()];
        auto& kernel_data = program_data.kernels[kernel->get_watcher_kernel_id()];
        kernel_data.kernel = kernel;
        kernel_data.watcher_kernel_id = kernel->get_watcher_kernel_id();
        kernel_data.name = kernel->name();
        kernel_data.path = build_options.path;
        if (!processor_elf_paths.empty()) {
            if (data->kernel_path_collection_enabled) {
                std::lock_guard<std::mutex> path_lock(data->kernel_path_mutex);
                data->kernel_id_to_processor_elf_paths[kernel->get_watcher_kernel_id()] = processor_elf_paths;
            }
            kernel_data.processor_elf_paths = std::move(processor_elf_paths);
        }
        kernel_data.source = kernel->kernel_source().source_;
        data->kernel_id_to_program_id[kernel->get_watcher_kernel_id()] = program->get_id();
        data->logger.log_program_kernel_compile_finished(program_data, kernel_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(data->rtoptions(), "Failed to log program kernel compile finished: {}", e.what());
    }
}

void Inspector::program_compile_finished(
    const detail::ProgramImpl* program, const IDevice* /*device*/, uint64_t /*build_key*/) noexcept {
    auto* data = get_inspector_data(*program);
    if (!data) {
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->programs_mutex);
        auto& program_data = data->programs_data[program->get_id()];
        program_data.compile_finished_timestamp = std::chrono::high_resolution_clock::now();
        program_data.semaphores = program->semaphores();
        data->logger.log_program_compile_finished(program_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(data->rtoptions(), "Failed to log program compile finished: {}", e.what());
    }
}

void Inspector::mesh_device_created(
    const distributed::MeshDeviceImpl* mesh_device, std::optional<int> parent_mesh_id) noexcept {
    auto* data = get_inspector_data(*mesh_device);
    if (!data) {
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->mesh_devices_mutex);
        auto& mesh_device_data = data->mesh_devices_data[mesh_device->id()];
        mesh_device_data.mesh_device = mesh_device;
        mesh_device_data.mesh_id = mesh_device->id();
        mesh_device_data.parent_mesh_id = parent_mesh_id;
        data->logger.log_mesh_device_created(mesh_device_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(data->rtoptions(), "Failed to log mesh device created: {}", e.what());
    }
}

void Inspector::mesh_device_destroyed(const distributed::MeshDeviceImpl* mesh_device) noexcept {
    auto* data = get_inspector_data(*mesh_device);
    if (!data) {
        return;
    }
    try {
        {
            std::lock_guard<std::mutex> lock(data->mesh_devices_mutex);
            // Not found if not recorded by this session, e.g. created before the context was re-initialized.
            if (auto it = data->mesh_devices_data.find(mesh_device->id()); it != data->mesh_devices_data.end()) {
                data->logger.log_mesh_device_destroyed(it->second);
                data->mesh_devices_data.erase(it);
            }
        }
        // Buffers still allocated on the mesh device are deallocated with it: their own deallocation events, if any,
        // come once the mesh device is gone and are dropped.
        std::vector<const distributed::MeshBuffer*> mesh_buffers;
        {
            std::lock_guard<std::mutex> lock(data->mesh_buffers_mutex);
            for (const auto& [mesh_buffer, mesh_id] : data->mesh_buffers_data) {
                if (mesh_id == mesh_device->id()) {
                    mesh_buffers.push_back(mesh_buffer);
                }
            }
        }
        for (const auto* mesh_buffer : mesh_buffers) {
            remove_mesh_buffer(*data, mesh_buffer);
        }
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(data->rtoptions(), "Failed to log mesh device destroyed: {}", e.what());
    }
}

void Inspector::mesh_device_initialized(const distributed::MeshDeviceImpl* mesh_device) noexcept {
    auto* data = get_inspector_data(*mesh_device);
    if (!data) {
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->mesh_devices_mutex);
        auto& mesh_device_data = data->mesh_devices_data[mesh_device->id()];
        mesh_device_data.initialized = true;
        data->logger.log_mesh_device_initialized(mesh_device_data);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(data->rtoptions(), "Failed to log mesh device initialized: {}", e.what());
    }
}

void Inspector::mesh_buffer_allocated(const distributed::MeshBuffer* mesh_buffer) noexcept {
    auto* data = get_inspector_data(*mesh_buffer);
    if (!data) {
        return;
    }
    try {
        if (data->mesh_buffer_logging_enabled) {
            data->logger.log_mesh_buffer_allocated(mesh_buffer);
        }
        std::lock_guard<std::mutex> lock(data->mesh_buffers_mutex);
        data->mesh_buffers_data.emplace(mesh_buffer, mesh_buffer->device()->id());
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(data->rtoptions(), "Failed to log mesh buffer allocated: {}", e.what());
    }
}

void Inspector::mesh_buffer_deallocated(const distributed::MeshBuffer* mesh_buffer) noexcept {
    auto* data = get_inspector_data(*mesh_buffer);
    if (!data) {
        return;
    }
    try {
        remove_mesh_buffer(*data, mesh_buffer);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(data->rtoptions(), "Failed to log mesh buffer deallocated: {}", e.what());
    }
}

void Inspector::remove_mesh_buffer(inspector::Data& data, const distributed::MeshBuffer* mesh_buffer) {
    if (data.mesh_buffer_logging_enabled) {
        data.logger.log_mesh_buffer_deallocated(mesh_buffer);
    }
    std::optional<inspector::MeshSocketData> destroyed_socket;
    {
        std::lock_guard<std::mutex> lock(data.mesh_buffers_mutex);
        data.mesh_buffers_data.erase(mesh_buffer);
        data.global_semaphores_data.erase(mesh_buffer);
        if (auto it = data.mesh_sockets_data.find(mesh_buffer); it != data.mesh_sockets_data.end()) {
            destroyed_socket = std::move(it->second);
            data.mesh_sockets_data.erase(it);
        }
    }
    if (destroyed_socket.has_value() && data.mesh_socket_logging_enabled) {
        data.logger.log_mesh_socket_destroyed(mesh_buffer, *destroyed_socket);
    }
}

void Inspector::mesh_socket_created(const distributed::MeshSocket* socket) noexcept {
    auto config_buffer = socket->get_config_buffer();
    auto* data = get_inspector_data(*config_buffer);
    if (!data) {
        return;
    }
    try {
        auto* mesh_device = socket->get_mesh_device();
        const distributed::SocketSenderSize sender_size(
            mesh_device->impl().metal_env().get_hal().get_alignment(HalMemType::L1));
        const bool is_sender = socket->get_socket_endpoint_type() == distributed::SocketEndpoint::SENDER;

        inspector::MeshSocketData socket_data;
        socket_data.is_sender = is_sender;
        socket_data.fifo_size = socket->get_config().socket_mem_config.fifo_size;
        socket_data.bytes_acked_offset_bytes = sender_size.md_size_bytes;
        socket_data.bytes_acked_stride_bytes = sender_size.ack_size_bytes;

        const auto local_ep = socket->get_socket_endpoint_type();
        const auto peer_ep = is_sender ? distributed::SocketEndpoint::RECEIVER : distributed::SocketEndpoint::SENDER;
        // One entry per local core; a sender core feeding several downstreams collects several peers.
        std::map<std::tuple<uint32_t, uint32_t, uint32_t>, inspector::MeshSocketLocalCoreData> by_core;
        for (const auto& conn : socket->get_config().socket_connection_config) {
            const auto& local_core = is_sender ? conn.sender_core : conn.receiver_core;
            if (!mesh_device->is_local(local_core.device_coord)) {
                continue;  // Another rank owns this core and reports it itself.
            }
            auto* local_device = mesh_device->get_device(local_core.device_coord);
            if (local_device == nullptr) {
                continue;
            }
            const auto& peer_core = is_sender ? conn.receiver_core : conn.sender_core;
            auto local_node = socket->get_fabric_node_id(local_ep, local_core.device_coord);
            auto peer_node = socket->get_fabric_node_id(peer_ep, peer_core.device_coord);

            // One mesh id per side, so every connection agrees.
            socket_data.local_mesh_id = *local_node.mesh_id;
            socket_data.peer_mesh_id = *peer_node.mesh_id;

            const auto core_key = std::tuple(
                static_cast<uint32_t>(local_device->id()),
                static_cast<uint32_t>(local_core.core_coord.x),
                static_cast<uint32_t>(local_core.core_coord.y));
            // Every connection on this core agrees on these.
            auto& core_data = by_core[core_key];
            core_data.chip_id = local_device->id();
            core_data.core.core_x = local_core.core_coord.x;
            core_data.core.core_y = local_core.core_coord.y;
            core_data.core.fabric_chip_id = local_node.chip_id;
            auto& peer = core_data.peers.emplace_back();
            peer.fabric_chip_id = peer_node.chip_id;
            peer.core_x = peer_core.core_coord.x;
            peer.core_y = peer_core.core_coord.y;
        }
        if (by_core.empty()) {
            return;  // this rank owns no core of this socket
        }
        // address() is 0 for per-core buffers, so resolve the per-core addresses here, on the rank that owns the core.
        socket_data.config_buffer_address = socket->get_config_buffer_address();
        if (!is_sender) {
            const auto& data_buffer = *socket->get_data_buffer();
            const auto& receiver_core = socket->get_config().socket_connection_config.front().receiver_core;
            socket_data.data_buffer_address =
                socket->get_config().socket_mem_config.per_core_allocation
                    ? experimental::per_core_allocation::get_per_core_address(
                          data_buffer, receiver_core.device_coord, receiver_core.core_coord)
                    : data_buffer.address();
        }
        socket_data.local_cores.reserve(by_core.size());
        for (auto& [key, core_data] : by_core) {
            socket_data.local_cores.push_back(std::move(core_data));
        }
        if (data->mesh_socket_logging_enabled) {
            data->logger.log_mesh_socket_created(config_buffer.get(), socket_data);
        }
        std::lock_guard<std::mutex> lock(data->mesh_buffers_mutex);
        data->mesh_sockets_data.insert_or_assign(config_buffer.get(), std::move(socket_data));
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(data->rtoptions(), "Failed to log mesh socket created: {}", e.what());
    }
}

void Inspector::global_semaphore_created(const distributed::MeshBuffer* buffer, const CoreRangeSet& cores) noexcept {
    auto* data = get_inspector_data(*buffer);
    if (!data) {
        return;
    }
    try {
        inspector::GlobalSemaphoreData semaphore_data;
        semaphore_data.address = buffer->address();
        semaphore_data.cores = cores;
        for (const auto* device : buffer->device()->impl().get_devices()) {
            semaphore_data.chip_ids.push_back(device->id());
        }
        std::lock_guard<std::mutex> lock(data->mesh_buffers_mutex);
        data->global_semaphores_data.insert_or_assign(buffer, std::move(semaphore_data));
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(data->rtoptions(), "Failed to log global semaphore created: {}", e.what());
    }
}

void Inspector::global_semaphore_reset(const distributed::MeshBuffer* buffer, uint32_t value) noexcept {
    auto* data = get_inspector_data(*buffer);
    if (!data) {
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->mesh_buffers_mutex);
        if (auto it = data->global_semaphores_data.find(buffer); it != data->global_semaphores_data.end()) {
            it->second.reset_value = value;
        }
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(data->rtoptions(), "Failed to log global semaphore reset: {}", e.what());
    }
}

void Inspector::emit_debug_entry(
    const MetalContext& context,
    const distributed::MeshWorkloadImpl* mesh_workload,
    uint64_t runtime_id,
    std::string_view operation_name,
    std::vector<TensorSpec> tensor_specs,
    std::optional<distributed::MeshTraceId> trace_id) noexcept {
    auto* data = context.get_inspector_data();
    if (!data) {
        return;
    }
    try {
        if (trace_id.has_value()) {
            // Trace-capture path: route into the per-trace bucket so the entry survives until release_trace.
            std::lock_guard<std::mutex> lock(data->trace_runtime_entries_mutex);
            auto& bucket = data->trace_runtime_entries[*trace_id];
            auto& slot = bucket.emplace_back();
            slot.workload_id = mesh_workload->get_id();
            slot.runtime_id = runtime_id;
            slot.operation_name = operation_name;
            slot.tensor_specs = std::move(tensor_specs);
            slot.trace_id = trace_id;
            if (data->runtime_entries_logging_enabled) {
                data->logger.log_runtime_entry(slot);
            }
        } else {
            std::lock_guard<std::mutex> lock(data->runtime_entries_mutex);
            auto pos = data->runtime_entries_write_pos;
            auto& slot = data->runtime_entries[pos % inspector::Data::kRuntimeEntriesCapacity];
            slot.workload_id = mesh_workload->get_id();
            slot.runtime_id = runtime_id;
            slot.operation_name = operation_name;
            slot.tensor_specs = std::move(tensor_specs);
            slot.trace_id.reset();
            if (pos == 2 * inspector::Data::kRuntimeEntriesCapacity) {
                data->runtime_entries_write_pos = inspector::Data::kRuntimeEntriesCapacity + 1;
            } else {
                data->runtime_entries_write_pos++;
            }
            if (data->runtime_entries_logging_enabled) {
                data->logger.log_runtime_entry(slot);
            }
        }
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(data->rtoptions(), "Failed to emit debug entry: {}", e.what());
    }
}

void Inspector::release_trace(const MetalContext& context, distributed::MeshTraceId trace_id) noexcept {
    auto* data = context.get_inspector_data();
    if (!data) {
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->trace_runtime_entries_mutex);
        data->trace_runtime_entries.erase(trace_id);
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(data->rtoptions(), "Failed to release trace runtime entries: {}", e.what());
    }
}

// Set dispatch core info
void Inspector::set_dispatch_core_info(
    const MetalContext& context,
    const tt_cxy_pair& virtual_core,
    const tt::tt_metal::DispatchWorkerType& type,
    const uint8_t cq_id,
    const ChipId device_id,
    const ChipId servicing_device_id) {
    auto* data = context.get_inspector_data();
    if (!data) {
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->dispatch_core_info_mutex);
        data->dispatch_core_info[virtual_core] = {type, device_id, servicing_device_id, cq_id};
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(data->rtoptions(), "Failed to log dispatch core info: {}", e.what());
    }
}

// Set dispatch_s core info
void Inspector::set_dispatch_s_core_info(
    const MetalContext& context,
    const tt_cxy_pair& virtual_core,
    const tt::tt_metal::DispatchWorkerType& type,
    const uint8_t cq_id,
    const ChipId device_id,
    const ChipId servicing_device_id) {
    auto* data = context.get_inspector_data();
    if (!data) {
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->dispatch_s_core_info_mutex);
        data->dispatch_s_core_info[virtual_core] = {type, device_id, servicing_device_id, cq_id};
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(data->rtoptions(), "Failed to log dispatch_s core info: {}", e.what());
    }
}

// Set prefetcher core info
void Inspector::set_prefetcher_core_info(
    const MetalContext& context,
    const tt_cxy_pair& virtual_core,
    const tt::tt_metal::DispatchWorkerType& type,
    const uint8_t cq_id,
    const ChipId device_id,
    const ChipId servicing_device_id) {
    auto* data = context.get_inspector_data();
    if (!data) {
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(data->prefetcher_core_info_mutex);
        data->prefetcher_core_info[virtual_core] = {type, device_id, servicing_device_id, cq_id};
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(data->rtoptions(), "Failed to log prefetcher core info: {}", e.what());
    }
}

void Inspector::enable_kernel_path_collection(const MetalContext& context) {
    auto* data = context.get_inspector_data();
    if (!data) {
        return;
    }
    try {
        data->kernel_path_collection_enabled = true;
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(data->rtoptions(), "Failed to enable kernel path collection: {}", e.what());
    }
}

std::string Inspector::get_kernel_elf_path(
    const MetalContext& context, int watcher_kernel_id, uint32_t processor_index) {
    std::string elf_path;

    auto* data = context.get_inspector_data();
    if (!data) {
        return elf_path;
    }
    try {
        std::lock_guard<std::mutex> lock(data->kernel_path_mutex);
        auto kernel_it = data->kernel_id_to_processor_elf_paths.find(watcher_kernel_id);
        if (kernel_it != data->kernel_id_to_processor_elf_paths.end() && processor_index < kernel_it->second.size()) {
            elf_path = kernel_it->second[processor_index];
        }
    } catch (const std::exception& e) {
        TT_INSPECTOR_LOG(
            data->rtoptions(),
            "Failed to get ELF path for watcher kernel ID {} processor index {}: {}",
            watcher_kernel_id,
            processor_index,
            e.what());
    }
    return elf_path;
}

namespace experimental::inspector {

bool IsEnabled(const tt::tt_metal::distributed::MeshDevice& mesh_device) {
    return tt::tt_metal::Inspector::is_enabled(mesh_device.impl().metal_context());
}

bool ShouldCaptureTensorSpecs(const tt::tt_metal::distributed::MeshDevice& mesh_device) {
    return tt::tt_metal::Inspector::capture_tensor_specs(mesh_device.impl().metal_context());
}

std::optional<tt::tt_metal::distributed::MeshTraceId> GetCurrentMeshTraceId(
    const tt::tt_metal::distributed::MeshDevice& mesh_device) {
    // mesh_command_queue().trace_id() is only supported in fast dispatch and would throw otherwise.
    if (!mesh_device.impl().metal_env().get_rtoptions().get_fast_dispatch()) {
        return std::nullopt;
    }
    return mesh_device.mesh_command_queue().trace_id();
}

void EmitMeshWorkloadDebugEntry(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshWorkload& workload,
    uint64_t runtime_id,
    std::string_view operation_name,
    std::vector<TensorSpec> tensor_specs,
    std::optional<tt::tt_metal::distributed::MeshTraceId> trace_id) {
    tt::tt_metal::Inspector::emit_debug_entry(
        mesh_device.impl().metal_context(),
        &workload.impl(),
        runtime_id,
        operation_name,
        std::move(tensor_specs),
        trace_id);
}

}  // namespace experimental::inspector

}  // namespace tt::tt_metal
