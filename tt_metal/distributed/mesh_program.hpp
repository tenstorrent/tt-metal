// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mesh_device.hpp"
#include "tt_metal/impl/program/program.hpp"

namespace tt::tt_metal {
// Mesh buffer, Mesh Op, Compilation
class MeshProgram {
   public:
        MeshProgram(std::size_t num_devices);
        ~MeshProgram() = default;
        Program& at(std::size_t device_index);
        std::vector<uint32_t> get_sem_base_addr(std::shared_ptr<MeshDevice> mesh_device, CoreCoord logical_core, CoreType core_type) const;

        template<typename T>
        T distributed_impl_(const std::function<T(Program&)>& callable) {
            if constexpr (std::is_same<T, void>::value) {
                for (std::size_t program_idx = 0; program_idx < this->programs.size(); program_idx++) {
                    callable(*this->programs.at(program_idx));
                }
            } else {
                for (std::size_t program_idx = 0; program_idx < this->programs.size() - 1; program_idx++) {
                    callable(*this->programs.at(program_idx));
                }
                return callable(*this->programs.at(this->programs.size() -1));
            }
        }

        template<typename T>
        std::vector<T> distributed_impl_(const std::variant<std::function<T(Program&)>, std::function<T(Program&, Device*)>>& callable, std::shared_ptr<MeshDevice> mesh_device = nullptr) const {
            std::vector<T> rval = {};
            std::vector<Device*> devices = {};
            if (mesh_device != nullptr) {
                devices = mesh_device->get_devices();
                TT_ASSERT(devices.size() == this->programs.size(),
                    "MeshProgram created for {} devices cannot be mapped to a MeshDevice with {} devices",
                    this->programs.size(), devices.size());
                TT_ASSERT(std::holds_alternative<std::function<T(Program&, Device*)>>(callable));
                auto f = std::get<std::function<T(Program&, Device*)>>(callable);
                for (std::size_t program_idx = 0; program_idx < devices.size(); program_idx++) {
                    rval.push_back(f(*this->programs.at(program_idx), devices.at(program_idx)));
                }
            } else {
                TT_ASSERT(std::holds_alternative<std::function<T(Program&)>>(callable));
                auto f = std::get<std::function<T(Program&)>>(callable);
                for (std::size_t program_idx = 0; program_idx < this->programs.size() - 1; program_idx++) {
                    rval.push_back(f(*this->programs.at(program_idx)));
                }
            }
            return rval;
        }

        template<typename T>
        T distribute_to_mesh_device_impl_(const std::function<T(Program&, Device*)>& callable, std::shared_ptr<MeshDevice>& mesh_device) {
            auto devices = mesh_device->get_devices();
            TT_ASSERT(devices.size() == this->programs.size(),
                    "MeshProgram created for {} devices cannot be mapped to a MeshDevice with {} devices",
                    this->programs.size(), devices.size());
            if constexpr (std::is_same<T, void>::value) {
                for (std::size_t program_idx = 0; program_idx < devices.size(); program_idx++) {
                    callable(*this->programs.at(program_idx), devices.at(program_idx));
                }
            } else {
                for (std::size_t program_idx = 0; program_idx < devices.size() - 1; program_idx++) {
                    callable(*this->programs.at(program_idx), devices.at(program_idx));
                }
                return callable(*this->programs.at(devices.size() -1), devices.at(devices.size() -1));
            }
        }
   private:
        std::vector<std::shared_ptr<Program>> programs = {};

};

uint32_t CreateSemaphore(
    MeshProgram& mesh_program,
    const std::variant<CoreRange, CoreRangeSet> &core_spec,
    uint32_t initial_value,
    CoreType core_type = CoreType::WORKER);

uint32_t CreateSemaphore(
    MeshProgram& mesh_program,
    const std::variant<CoreRange, CoreRangeSet> &core_spec,
    uint32_t initial_value,
    CoreType core_type,
    chip_id_t device_id);

CBHandle CreateCircularBuffer(
    MeshProgram& mesh_program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const CircularBufferConfig &config);

CBHandle CreateCircularBuffer(
    MeshProgram& mesh_program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const CircularBufferConfig &config,
    chip_id_t device_id);

void SetRuntimeArgs(
    MeshProgram& mesh_program,
    KernelHandle kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::vector<uint32_t> &runtime_args);

void SetRuntimeArgs(
    MeshProgram& mesh_program,
    KernelHandle kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::vector<uint32_t> &runtime_args,
    chip_id_t device_id);

KernelHandle CreateKernel(
    MeshProgram& mesh_program,
    const std::string &file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> &config);

KernelHandle CreateKernel(
    MeshProgram& mesh_program,
    const std::string &file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> &config,
    chip_id_t device_id);

void EnqueueMeshProgram(
    uint8_t cq_id, MeshProgram& mesh_program, std::shared_ptr<MeshDevice> mesh_device, bool blocking);

void Finish(std::shared_ptr<MeshDevice> mesh_device, uint8_t cq_id);

}
