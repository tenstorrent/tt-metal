// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_workload.hpp"
#include "tt_metal/distributed/mesh_workload_impl.hpp"

namespace tt::tt_metal::distributed {

class MeshWorkload::Impl : public MeshWorkloadImpl {
    // Inherit all implementation from MeshWorkloadImpl
};

MeshWorkload::MeshWorkload() : pimpl(std::make_unique<Impl>()) {}

MeshWorkload::~MeshWorkload() = default;

MeshWorkload::MeshWorkload(MeshWorkload&&) noexcept = default;
MeshWorkload& MeshWorkload::operator=(MeshWorkload&&) noexcept = default;

void MeshWorkload::add_program(const MeshCoordinateRange& device_range, Program&& program) {
    pimpl->add_program(device_range, std::move(program));
}

std::unordered_map<MeshCoordinateRange, Program>& MeshWorkload::get_programs() { return pimpl->get_programs(); }

void MeshWorkload::set_last_used_command_queue_for_testing(MeshCommandQueue* mesh_cq) {
    pimpl->set_last_used_command_queue_for_testing(mesh_cq);
}

MeshCommandQueue* MeshWorkload::get_last_used_command_queue() const { return pimpl->get_last_used_command_queue(); }

uint32_t MeshWorkload::get_sem_base_addr(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    return pimpl->get_sem_base_addr(mesh_device, logical_core, core_type);
}

uint32_t MeshWorkload::get_sem_size(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    return pimpl->get_sem_size(mesh_device, logical_core, core_type);
}

uint32_t MeshWorkload::get_cb_base_addr(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    return pimpl->get_cb_base_addr(mesh_device, logical_core, core_type);
}

uint32_t MeshWorkload::get_cb_size(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    return pimpl->get_cb_size(mesh_device, logical_core, core_type);
}

}  // namespace tt::tt_metal::distributed
