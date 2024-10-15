// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mesh_device.hpp"
#include "tt_metal/impl/program/program.hpp"

using LogicalDeviceCoord = tt_xy_pair;

struct LogicalDeviceRange {
    LogicalDeviceCoord start_coord = {0, 0}; // needs to be inf
    LogicalDeviceCoord end_coord = {0, 0};
};

namespace std {
template <>
struct hash<LogicalDeviceRange> {
    std::size_t operator()(const CoreRange &device_range) const {
        std::size_t seed = 0;
        seed = std::hash<LogicalDeviceCoord>{}(device_range.start_coord) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed = std::hash<LogicalDeviceCoord>{}(device_range.end_coord) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};
}  // namespace std

namespace tt::tt_metal {

using KernelDescriptor = std::vector<std::unordered_map<KernelHandle, std::shared_ptr<Kernel> >>;
class MeshWorkload {
    // A single Program fully captures the what a singke device will do when a MeshProgram is enqueued
    // We can choose to run different Kernels across different SubCoreMeshes in a device
    // How does this overlap with MeshprogramCache. Currently needs to be heiarachical with ProgramCache
    // Eventtualy, when we have logical coordinates in FD commands, this can be a single Cache
    public:
        MeshWorkload() {}
        ~MeshWorkload() {}

        void add_program(std::shared_ptr<Program>& program, const LogicalDeviceRange& device_range) {
            this->programs_.insert({device_range, program}); // During validation use this to verify that programs do not overlap
            update_global_device_range(device_range);
        }

        void add_semaphore(
            LogicalDeviceRange& logical_device_range,
            const CoreRangeSet& cores,
            uint32_t semaphore_id,
            uint32_t initial_value,
            CoreType core_type) {
            if (this->semaphores.find(logical_device_range) == this->semaphores.end()) {
                this->semaphores.insert({logical_device_range, {}});
            }
            this->semaphores.at(logical_device_range).push_back(Semaphore(cores, semaphore_id, initial_value, core_type));

        }

        void validate_mesh_workload_cfg(std::shared_ptr<MeshDevice> mesh_device) {
            // Can't run more than one program per device (could do it but we want explicit behaviour)
            // A program must fit in the Mesh
            validate_programs_fit_in_mesh(mesh_device);
            validate_programs_do_not_overlap();
        }

        const std::unordered_map<LogicalDeviceRange, std::shared_ptr<Program>> get_programs() const {
            return this->programs_;
        }


    private:
        void validate_programs_fit_in_mesh(std::shared_ptr<MeshDevice> mesh_device) {
            TT_ASSERT(this->global_device_range_.start_coord.x >= 0 && this->global_device_range_.start_coord.y >= 0);
            TT_ASSERT(this->global_device_range_.end_coord.x < mesh_device->num_cols() && this->global_device_range_.end_coord.x < mesh_device->num_rows());
        }

        void validate_programs_do_not_overlap() {}

        void update_global_device_range(const LogicalDeviceRange& device_range) {
            if (device_range.start_coord.x < global_device_range.start_coord.x) {
                global_device_range.start_coord.x = device_range.start_coord.x;
            }
            if (device_range.start_coord.y < global_device_range.start_coord.y) {
                global_device_range.start_coord.y = device_range.start_coord.y;
            }
            if (device_range.end_coord.x > global_device_range.end_coord.x) {
                global_device_range.end_coord.x = device_range.end_coord.x;
            }
            if (device_range.end_coord.y > global_device_range.end_coord.y) {
                global_device_range.end_coord.y = device_range.end_coord.y;
            }
        }

        std::unordered_map<LogicalDeviceRange, std::shared_ptr<Program>> programs_;

        std::unordered_map<LogicalDeviceRange, std::vector<Semaphore>> semaphores_;
        std::unordered_map<LogicalDeviceRange, std::shared_ptr<CircularBuffer>> circular_buffers_;
        std::unordered_map<LogicalDeviceRange, KernelDescriptor> kernels_;
        std::unordered_map<LogicalDeviceRange, std::vector<std::vector<RuntimeArgsData>>> runtime_args_data_;
        LogicalDeviceRange global_device_range_;
};

void EnqueueMeshWorkload(uint8_t cq_id, MeshWorkload& mesh_workload, std::shared_ptr<MeshDevice> mesh_device, bool blocking) {
    mesh_workload.validate_mesh_workload_cfg();
    for (auto& programs_per_grid : mesh_workload->get_programs()) {
        LogicalDeviceRange program_device_range = programs_per_grid.first;
        for (std::size_t x = program_device_range.start_coord.x; x < program_device_range.end_coord.x; x++) {
            for (std::size_t y = program_device_range.start_coord.y; x < program_device_range.end_coord.y; y++) {
                Device* device = mesh_device->get_device(y, x);
                // This requires programs to be enqueued across multiple devices
                // Which requires either:
                // (First one is better, since device state is fully dictated by API state)
                // (Second one requires separating API state from device state and tracking device state separately)
                //  Virtual coordinates embedded in FD commands
                //  Identical kernel binaries across devices (compile once)
                // or:
                //  Recompile per device (this is currently supported)
                //  FD Commands tracked per device (currently not supported)
                EnqueueProgram(device->command_queue(cq_id), *(programs_per_grid.second), blocking);
            }
        }
    }
}

}
