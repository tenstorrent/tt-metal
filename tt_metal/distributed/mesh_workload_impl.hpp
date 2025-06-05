// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include "program/program_impl.hpp"

namespace tt::tt_metal::distributed {
using RuntimeArgsPerCore = std::vector<std::vector<RuntimeArgsData>>;

class MeshCommandQueue;
class FDMeshCommandQueue;

class MeshWorkloadImpl {
    // A MeshWorkload can be fully described using a set of programs mapped to different Logical Device Regions
    // in a Mesh + configurable runtime Args
    // The current iteration supports the following compute paradigms:
    //  - Single Program Multi Device (Completely Homogenous MeshWorkload)
    //  - Multi Program Multi Device (Completely Heterogeneous MeshWorkload)
    // Support for configurable runtime arguments will be added in future versions.
private:
    bool runs_on_noc_multicast_only_cores();
    bool runs_on_noc_unicast_only_cores();
    void compile(MeshDevice* mesh_device);
    void load_binaries(MeshCommandQueue& mesh_cq);
    void generate_dispatch_commands(MeshCommandQueue& mesh_cq);
    std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& get_kernels(uint32_t programmable_core_type_index);
    std::vector<std::shared_ptr<KernelGroup>>& get_kernel_groups(uint32_t programmable_core_type_index);
    std::vector<Semaphore>& semaphores();
    std::vector<uint32_t> get_program_config_sizes();
    std::unordered_set<SubDeviceId> determine_sub_device_ids(MeshDevice* mesh_device);
    bool kernel_binary_always_stored_in_ringbuffer();
    bool is_finalized() const { return this->finalized_; }
    void set_finalized() { this->finalized_ = true; };
    ProgramBinaryStatus get_program_binary_status(std::size_t mesh_id) const;
    void set_program_binary_status(std::size_t mesh_id, ProgramBinaryStatus status);
    ProgramConfig& get_program_config(uint32_t index);
    ProgramCommandSequence& get_dispatch_cmds_for_program(Program& program, uint64_t command_hash);
    void compile_program(const MeshCoordinateRange& device_range, MeshDevice* mesh_device);
    void finalize_offsets(MeshDevice* mesh_device);

    std::unordered_map<std::size_t, ProgramBinaryStatus> program_binary_status_;
    std::shared_ptr<MeshBuffer> kernel_bin_buf_;
    std::vector<std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>> kernels_;
    std::vector<std::vector<std::shared_ptr<KernelGroup>>> kernel_groups_;
    std::vector<Semaphore> semaphores_;
    std::unordered_map<MeshCoordinateRange, Program> programs_;
    bool finalized_ = false;
    std::unordered_map<MeshCoordinateRange, std::unordered_map<KernelHandle, RuntimeArgsPerCore>> runtime_args_;
    MeshCommandQueue* last_used_command_queue_ = nullptr;

    template <typename WorkloadType, typename DeviceType>
    friend uint32_t program_dispatch::program_base_addr_on_core(WorkloadType&, DeviceType, HalProgrammableCoreType);
    friend void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking);
    friend FDMeshCommandQueue;
    friend class tt::tt_metal::Program;

public:
    // Main User-Facing API building blocks
    MeshWorkloadImpl();

    void add_program(const MeshCoordinateRange& device_range, Program&& program);
    std::unordered_map<MeshCoordinateRange, Program>& get_programs() { return programs_; }
    const std::unordered_map<MeshCoordinateRange, Program>& get_programs() const { return programs_; }

    // For testing purposes only
    void set_last_used_command_queue_for_testing(MeshCommandQueue* mesh_cq);
    MeshCommandQueue* get_last_used_command_queue() const;
    uint32_t get_sem_base_addr(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_sem_size(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_cb_base_addr(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_cb_size(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);
};
}  // namespace tt::tt_metal::distributed
