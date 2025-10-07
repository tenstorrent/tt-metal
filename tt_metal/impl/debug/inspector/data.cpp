// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "data.hpp"
#include <stdexcept>
#include <fstream>
#include <sstream>
#include "impl/debug/inspector.hpp"
#include "impl/debug/inspector/rpc_server_controller.hpp"
#include "impl/debug/inspector/logger.hpp"
#include "impl/context/metal_context.hpp"
#include "distributed/mesh_workload_impl.hpp"

namespace tt::tt_metal::inspector {


Data::Data()
    : logger(MetalContext::instance().rtoptions().get_inspector_log_path()) {

    // Initialize RPC server if enabled
    const auto& rtoptions = MetalContext::instance().rtoptions();
    if (rtoptions.get_inspector_rpc_server_enabled()) {
        try {
            auto address = rtoptions.get_inspector_rpc_server_address();
            rpc_server_controller.start(address);

            // Connect callbacks that we want to respond to
            get_rpc_server().setGetProgramsCallback([this](auto result) { this->rpc_get_programs(result); });
            get_rpc_server().setGetMeshDevicesCallback([this](auto result) { this->rpc_get_mesh_devices(result); });
            get_rpc_server().setGetMeshWorkloadsCallback([this](auto result) { this->rpc_get_mesh_workloads(result); });
            get_rpc_server().setGetDevicesInUseCallback([this](auto result) { this->rpc_get_devices_in_use(result); });
            get_rpc_server().setGetKernelCallback([this](auto params, auto result) { this->rpc_get_kernel(params, result); });
            get_rpc_server().setGetOperationsCallback([this](auto result) { this->rpc_get_operations(result); });
        } catch (const std::exception& e) {
            TT_INSPECTOR_THROW("Failed to start Inspector RPC server: {}", e.what());
        }
    }
}

Data::~Data() {
    // Serialize operation tracking before shutting down
    serialize_rpc();
    rpc_server_controller.stop();
}

RpcServer& Data::get_rpc_server() {
    return rpc_server_controller.get_rpc_server();
}

void Data::serialize_rpc() {
    rpc_server_controller.get_rpc_server().serialize(logger.get_logging_path());
}

void Data::rpc_get_programs(rpc::Inspector::GetProgramsResults::Builder& results) {
    std::lock_guard<std::mutex> lock(programs_mutex);
    auto programs = results.initPrograms(programs_data.size());
    uint32_t i = 0;

    for (const auto& [program_id, program_data] : programs_data) {
        auto program = programs[i++];

        // Set basic program info
        program.setProgramId(program_id);

        // Check if program is compiled (has finished compilation)
        bool compiled = program_data.compile_finished_timestamp != inspector::time_point{};
        program.setCompiled(compiled);

        // Set binary status per device
        auto binary_status_list = program.initBinaryStatusPerDevice(program_data.binary_status_per_device.size());
        uint32_t j = 0;
        for (const auto& [device_id, status] : program_data.binary_status_per_device) {
            auto device_status = binary_status_list[j++];
            device_status.setDeviceId(static_cast<uint64_t>(device_id));
            device_status.setStatus(convert_binary_status(status));
        }

        // Set kernels
        auto kernels_list = program.initKernels(program_data.kernels.size());
        j = 0;
        for (const auto& [kernel_id, kernel_data] : program_data.kernels) {
            auto kernel = kernels_list[j++];
            kernel.setWatcherKernelId(kernel_data.watcher_kernel_id);
            kernel.setName(kernel_data.name);
            kernel.setPath(kernel_data.path);
            kernel.setSource(kernel_data.source);
            kernel.setProgramId(program_id);
        }
    }
}

void Data::rpc_get_mesh_devices(rpc::Inspector::GetMeshDevicesResults::Builder& results) {
    std::lock_guard<std::mutex> lock(mesh_devices_mutex);
    auto mesh_devices = results.initMeshDevices(mesh_devices_data.size());
    uint32_t i = 0;
    for (const auto& [mesh_id, mesh_device_data] : mesh_devices_data) {
        auto mesh_device = mesh_devices[i++];
        mesh_device.setMeshId(mesh_id);

        uint32_t j = 0;
        auto devices_view = mesh_device_data.mesh_device->get_devices();
        auto devices = mesh_device.initDevices(devices_view.size());
        for (const auto& device : devices_view) {
            devices.set(j++, device->id());
        }

        auto& shape_view = mesh_device_data.mesh_device->get_view().shape();
        auto shape = mesh_device.initShape(shape_view.dims());
        for (size_t k = 0; k < shape_view.dims(); ++k) {
            shape.set(k, shape_view.get_stride(k));
        }

        mesh_device.setParentMeshId(mesh_device_data.parent_mesh_id.value_or(-1));
        mesh_device.setInitialized(mesh_device_data.initialized);
    }
}

void Data::rpc_get_mesh_workloads(rpc::Inspector::GetMeshWorkloadsResults::Builder& results) {
    std::lock_guard<std::mutex> lock(mesh_workloads_mutex);
    auto mesh_workloads = results.initMeshWorkloads(mesh_workloads_data.size());
    uint32_t i = 0;
    for (const auto& [mesh_workload_id, mesh_workload_data] : mesh_workloads_data) {
        auto mesh_workload = mesh_workloads[i++];
        mesh_workload.setMeshWorkloadId(mesh_workload_id);

        auto& programs = mesh_workload_data.mesh_workload->get_programs();
        auto programs_data = mesh_workload.initPrograms(programs.size());
        uint32_t j = 0;
        for (const auto& [device_range, program] : programs) {
            auto program_data = programs_data[j++];
            program_data.setProgramId(program.impl().get_id());
            auto coordinates_list = program_data.initCoordinates(device_range.shape().mesh_size());
            uint32_t k = 0;
            for (auto& device_coordinate : device_range) {
                auto mesh_coordinate = coordinates_list[k++];
                auto coords = device_coordinate.coords();
                auto coordinates = mesh_coordinate.initCoordinates(coords.size());
                for (size_t l = 0; l < coords.size(); ++l) {
                    coordinates.set(l, coords[l]);
                }
            }
        }

        auto binary_status_list = mesh_workload.initBinaryStatusPerMeshDevice(mesh_workload_data.binary_status_per_device.size());
        j = 0;
        for (const auto& [mesh_id, status] : mesh_workload_data.binary_status_per_device) {
            auto binary_status = binary_status_list[j++];
            binary_status.setMeshId(mesh_id);
            binary_status.setStatus(convert_binary_status(status));
        }
    }
}

void Data::rpc_get_devices_in_use(rpc::Inspector::GetDevicesInUseResults::Builder& results) {
    std::scoped_lock locks(programs_mutex, mesh_devices_mutex, mesh_workloads_mutex);
    std::set<uint64_t> device_ids;

    // First add all devices from mesh workloads
    for (const auto& [mesh_workload_id, mesh_workload_data] : mesh_workloads_data) {
        for (const auto& [mesh_device_id, status] : mesh_workload_data.binary_status_per_device) {
            if (status != ProgramBinaryStatus::NotSent) {
                auto mesh_device_it = mesh_devices_data.find(mesh_device_id);
                if (mesh_device_it != mesh_devices_data.end()) {
                    auto* mesh_device = mesh_device_it->second.mesh_device;
                    for (auto& device : mesh_device->get_devices()) {
                        device_ids.insert(static_cast<uint64_t>(device->id()));
                    }
                }
            }
        }
    }

    // Add all devices from programs
    for (const auto& [program_id, program_data] : programs_data) {
        for (const auto& [device_id, status] : program_data.binary_status_per_device) {
            if (status != ProgramBinaryStatus::NotSent) {
                device_ids.insert(static_cast<uint64_t>(device_id));
            }
        }
    }

    // Write result
    auto result_device_ids = results.initDeviceIds(device_ids.size());
    size_t i = 0;
    for (const auto& device_id : device_ids) {
        result_device_ids.set(i++, device_id);
    }
}

void Data::rpc_get_kernel(rpc::Inspector::GetKernelParams::Reader params, rpc::Inspector::GetKernelResults::Builder results) {
    std::lock_guard<std::mutex> lock(programs_mutex);
    auto kernel_id = params.getWatcherKernelId();
    auto program_id_it = kernel_id_to_program_id.find(kernel_id);
    if (program_id_it == kernel_id_to_program_id.end()) {
        throw std::runtime_error("Kernel not found");
    }
    auto program_id = program_id_it->second;
    auto program_data = programs_data.find(program_id);
    if (program_data == programs_data.end()) {
        throw std::runtime_error("Program not found");
    }
    auto kernel_data_it = program_data->second.kernels.find(kernel_id);
    if (kernel_data_it == program_data->second.kernels.end()) {
        throw std::runtime_error("Kernel not found inside the program");
    }
    auto& kernel_data = kernel_data_it->second;
    auto kernel = results.initKernel();
    kernel.setWatcherKernelId(kernel_data.watcher_kernel_id);
    kernel.setName(kernel_data.name);
    kernel.setPath(kernel_data.path);
    kernel.setSource(kernel_data.source);
    kernel.setProgramId(program_id);
}

void Data::rpc_get_operations(rpc::Inspector::GetOperationsResults::Builder& results) {
    std::lock_guard<std::mutex> lock(operations_mutex);

    auto operations_list = results.initOperations(operations_.size());
    for (size_t i = 0; i < operations_.size(); ++i) {
        const auto& op = operations_[i];
        auto capnp_op = operations_list[i];

        // Set device operation ID - use "none" for host-only operations
        if (op.device_operation_id.has_value()) {
            capnp_op.setDeviceOperationId(std::to_string(op.device_operation_id.value()));
        } else {
            capnp_op.setDeviceOperationId("none");
        }

        capnp_op.setOperationName(op.operation_name);
        capnp_op.setCallstack(op.call_stack);
        capnp_op.setArguments(op.arguments);

        // Convert timestamp to nanoseconds since epoch
        auto time_since_epoch = op.timestamp.time_since_epoch();
        auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(time_since_epoch).count();
        capnp_op.setTimestamp(static_cast<uint64_t>(nanos));
    }
}

// Helper function to convert internal enum to Cap'n Proto enum
rpc::BinaryStatus Data::convert_binary_status(ProgramBinaryStatus status) {
    switch (status) {
        case ProgramBinaryStatus::NotSent:
            return rpc::BinaryStatus::NOT_SENT;
        case ProgramBinaryStatus::InFlight:
            return rpc::BinaryStatus::IN_FLIGHT;
        case ProgramBinaryStatus::Committed:
            return rpc::BinaryStatus::COMMITTED;
        default:
            return rpc::BinaryStatus::NOT_SENT;
    }
}

// Helper function to escape YAML strings
static std::string escape_yaml_string(const std::string& str) {
    // For simplicity, we'll use quoted strings if they contain special characters
    bool needs_quotes = false;
    for (char c : str) {
        if (c == '\n' || c == '\r' || c == ':' || c == '#' || c == '"' || c == '\'' || c == '\\') {
            needs_quotes = true;
            break;
        }
    }

    if (!needs_quotes) {
        return str;
    }

    // Use double quotes and escape special characters
    std::stringstream escaped;
    escaped << '"';
    for (char c : str) {
        switch (c) {
            case '"': escaped << "\\\""; break;
            case '\\': escaped << "\\\\"; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default: escaped << c; break;
        }
    }
    escaped << '"';
    return escaped.str();
}

void Data::dbg_serialize_operations() {
    std::lock_guard<std::mutex> lock(operations_mutex);

    if (operations_.empty()) {
        return;
    }

    // Create directory structure
    auto ops_dir = logger.get_logging_path() / "ops";
    std::filesystem::create_directories(ops_dir);

    // Write to YAML format (matching Python output format)
    auto filepath = ops_dir / "ops.yaml";
    std::ofstream file(filepath);

    if (!file.is_open()) {
        // Silent failure - we don't want to crash the program if we can't write debug info
        return;
    }

    // Write operations as a proper YAML list
    for (const auto& op : operations_) {
        // Start a new list item
        file << "- device_operation_id: ";
        if (op.device_operation_id.has_value()) {
            file << *op.device_operation_id;
        } else {
            file << "none";
        }
        file << "\n";

        // Operation name
        file << "  operation_name: " << op.operation_name << "\n";

        // Call stack (if available)
        if (!op.call_stack.empty()) {
            // Use YAML literal block scalar for multiline strings
            file << "  callstack: ";
            if (op.call_stack.find('\n') != std::string::npos) {
                // Multiline - use literal block scalar
                file << "|\n";
                std::istringstream stream(op.call_stack);
                std::string line;
                while (std::getline(stream, line)) {
                    file << "    " << line << "\n";
                }
            } else {
                // Single line
                file << escape_yaml_string(op.call_stack) << "\n";
            }
        } else {
            file << "  callstack: \"\"\n";
        }

        // Arguments
        file << "  arguments: ";
        if (!op.arguments.empty()) {
            if (op.arguments.find('\n') != std::string::npos) {
                // Multiline - use literal block scalar
                file << "|\n";
                std::istringstream stream(op.arguments);
                std::string line;
                while (std::getline(stream, line)) {
                    file << "    " << line << "\n";
                }
            } else {
                // Single line
                file << escape_yaml_string(op.arguments) << "\n";
            }
        } else {
            file << "\"\"\n";
        }
    }

    file.close();
}

}  // namespace tt::tt_metal::inspector
