// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_program.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"

namespace tt::tt_metal {

MeshProgram::MeshProgram(std::size_t num_devices) {
    this->programs.reserve(num_devices);
    for (int i = 0; i < num_devices; ++i) {
        this->programs.push_back(std::make_shared<Program>());
    }
}

Program& MeshProgram::at(std::size_t device_index) {
    TT_ASSERT(device_index < this->program.size());
    return *this->programs.at(device_index);
}

std::vector<uint32_t> MeshProgram::get_sem_base_addr(std::shared_ptr<MeshDevice> mesh_device, CoreCoord logical_core, CoreType core_type) const {
    return this->distributed_impl_(
        std::variant<std::function<uint32_t(Program&)>, std::function<uint32_t(Program&, Device*)>>(
            std::function<uint32_t(Program&, Device*)>(
                [logical_core, core_type](Program& program, Device* device) -> uint32_t {
                    return program.get_sem_base_addr(device, logical_core, core_type);
                }
            )
        ),
        mesh_device
    );
}

uint32_t CreateSemaphore(
    MeshProgram& mesh_program,
    const std::variant<CoreRange, CoreRangeSet> &core_spec,
    uint32_t initial_value,
    CoreType core_type) {
    return mesh_program.distributed_impl_(
        std::function<uint32_t(Program&)>(
            [&core_spec, initial_value, core_type] (Program& program) -> uint32_t {
                return CreateSemaphore(program, core_spec, initial_value, core_type);
            }
        )
    );
}

uint32_t CreateSemaphore(
    MeshProgram& mesh_program,
    const std::variant<CoreRange, CoreRangeSet> &core_spec,
    uint32_t initial_value,
    CoreType core_type,
    chip_id_t device_id) {
    return CreateSemaphore(mesh_program.at(device_id), core_spec, initial_value, core_type);
}

CBHandle CreateCircularBuffer(
    MeshProgram& mesh_program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const CircularBufferConfig &config) {
    return mesh_program.distributed_impl_(
        std::function<CBHandle(Program&)>(
            [&core_spec, &config] (Program& program) -> CBHandle {
                return CreateCircularBuffer(program, core_spec, config);
            }
        )
    );
}

CBHandle CreateCircularBuffer(
    MeshProgram& mesh_program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const CircularBufferConfig &config,
    chip_id_t device_id) {
        return CreateCircularBuffer(mesh_program.at(device_id), core_spec, config);
}

void SetRuntimeArgs(
    MeshProgram& mesh_program,
    KernelHandle kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::vector<uint32_t> &runtime_args) {
    mesh_program.distributed_impl_(
        std::function<void(Program&)>(
            [kernel, &core_spec, &runtime_args] (Program& program) -> void {
                return SetRuntimeArgs(program, kernel, core_spec, runtime_args);
            }
        )
    );
}

void SetRuntimeArgs(
    MeshProgram& mesh_program,
    KernelHandle kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::vector<uint32_t> &runtime_args,
    chip_id_t device_id) {
        SetRuntimeArgs(mesh_program.at(device_id), kernel, core_spec, runtime_args);
}

KernelHandle CreateKernel(
    MeshProgram& mesh_program,
    const std::string &file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> &config) {
    return mesh_program.distributed_impl_(
        std::function<KernelHandle(Program&)>(
            [&file_name, &core_spec, &config] (Program& program) -> KernelHandle {
                return CreateKernel(program, file_name, core_spec, config);
            }
        )
    );
}

KernelHandle CreateKernel(
    MeshProgram& mesh_program,
    const std::string &file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> &config,
    chip_id_t device_id) {
        return CreateKernel(mesh_program.at(device_id), file_name, core_spec, config);
}

void EnqueueMeshProgram(
    uint8_t cq_id, MeshProgram& mesh_program, std::shared_ptr<MeshDevice> mesh_device, bool blocking) {
    mesh_program.distribute_to_mesh_device_impl_(
        std::function<void(Program&, Device*)>(
            [cq_id, blocking] (Program& program, Device* device) -> void {
                EnqueueProgram(device->command_queue(cq_id), program, blocking);
            }
        ),
        mesh_device
    );
}

void Finish(std::shared_ptr<MeshDevice> mesh_device, uint8_t cq_id) {
    for (auto device : mesh_device->get_devices()) {
        Finish(device->command_queue(cq_id));
    }
}

}
