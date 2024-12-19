// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_workload.hpp"

namespace tt::tt_metal::distributed {
void MeshWorkload::add_program(const LogicalDeviceRange& device_range, Program& program) {
    this->programs_[device_range] = std::move(program);
}

void MeshWorkload::compile(std::shared_ptr<MeshDevice>& mesh_device) {
    // Generate binaries for all programs in the MeshWorkload using
    // the build system exposed by the first device
    for (auto& program_on_grid : this->programs_) {
        program_on_grid.second.compile(mesh_device->get_device(0));
        program_on_grid.second.allocate_circular_buffers(mesh_device->get_device(0));
        tt::tt_metal::detail::ValidateCircularBufferRegion(program_on_grid.second, mesh_device->get_device(0));
    }
    this->compiled_ = true;
}

void MeshWorkload::finalize(std::shared_ptr<MeshDevice>& mesh_device) {
    for (auto& program_on_grid : this->programs_) {
        program_on_grid.second.finalize(mesh_device->get_device(0));
    }
    this->finalized_ = true;
}

void MeshWorkload::enqueue(std::shared_ptr<MeshDevice>& mesh_device, uint8_t cq_id, bool blocking) {
    if (not this->is_compiled()) {
        this->compile(mesh_device);
    }
    if (not this->is_finalized()) {
        this->finalize(mesh_device);
    }
    for (auto& program_on_grid : this->programs_) {
        auto& device_range = program_on_grid.first;
        auto& program = program_on_grid.second;
        for (std::size_t logical_x = device_range.start_coord.x; logical_x < device_range.end_coord.x; logical_x++) {
            for (std::size_t logical_y = device_range.start_coord.y; logical_y < device_range.end_coord.y;
                 logical_y++) {
                Device* device = mesh_device->get_device(logical_y, logical_x);
                EnqueueProgram(
                    device->command_queue(cq_id),
                    program,
                    false);  // Non blocking call across devices, block once at the end
            }
        }
        if (blocking) {
            for (Device* device : mesh_device->get_devices()) {
                Finish(device->command_queue(cq_id));
            }
        }
    }
}

// void MeshWorkload::set_runtime_args(const LogicalDeviceRange& device_range, const CoreRangeSet& core_range_set,
// KernelHandle kernel_id, const std::vector<uint32_t> runtime_args) {
//     std::size_t intersection_count = 0;

//     for (auto& program_on_grid : this->programs_) {
//         auto& program_device_range = program_on_grid.first;
//         if (device_range.intersects(program_device_range)) {
//             program_to_set_rt
//         }
//     }
// }

}  // namespace tt::tt_metal::distributed
