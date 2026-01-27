// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/dispatch_context.hpp>
#include <tt-metalium/mesh_device.hpp>
#include "mesh_device_impl.hpp"
#include "mesh_command_queue.hpp"
#include "fd_mesh_command_queue.hpp"
#include "sd_mesh_command_queue.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/device/device_manager.hpp"
#include "impl/device/device_impl.hpp"
#include "impl/dispatch/topology.hpp"
#include "impl/dispatch/cq_shared_state.hpp"
#include "llrt/hal/generated/dev_msgs.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/llrt.hpp"

namespace tt::tt_metal::experimental {

// Define the static member with custom deleter
std::unique_ptr<DispatchContext, DispatchContext::Deleter> DispatchContext::dispatch_context_ptr_ = nullptr;

DispatchContext& DispatchContext::get() {
    if (!dispatch_context_ptr_) {
        dispatch_context_ptr_ = std::unique_ptr<DispatchContext, Deleter>(new DispatchContext());
    }
    return *dispatch_context_ptr_;
}

void DispatchContext::initialize_fast_dispatch(distributed::MeshDevice* mesh_device) {
    fast_dispatch_enabled_ = MetalContext::instance().rtoptions().get_fast_dispatch();
    const auto& cluster = MetalContext::instance().get_cluster();
    TT_FATAL(
        !fast_dispatch_enabled_,
        "Fast Dispatch can only be manually enabled when running the workload with Slow Dispatch mode.");
    TT_FATAL(num_fd_inits_ == 0, "Fast Dispatch can only be manually initialized and torn down once.");
    TT_FATAL(
        cluster.is_ubb_galaxy() || cluster.arch() == tt::ARCH::BLACKHOLE,
        "Manually setting up and tearing down Fast Dispatch is only supported on Galaxy and Blackhole clusters.");

    const auto& device_manager = MetalContext::instance().device_manager();
    const auto& active_devices = device_manager->get_all_active_devices();

    uint8_t num_hw_cqs = active_devices[0]->num_hw_cqs();

    // Reinitialize dispatch managers to pick up FD core descriptor before allocating cores
    tt::tt_metal::MetalContext::instance().rtoptions().set_fast_dispatch(true);
    MetalContext::instance().reinitialize_dispatch_managers();

    for (const auto& dev : active_devices) {
        TT_FATAL(dev->num_hw_cqs() == num_hw_cqs, "All devices must have the same number of command queues.");
        dev->init_command_queue_host();
    }
    // Query the number of command queues requested
    populate_fd_kernels(active_devices, num_hw_cqs);
    device_manager->configure_and_load_fast_dispatch_kernels();
    tt::tt_metal::MetalContext::instance().rtoptions().set_fast_dispatch(fast_dispatch_enabled_);

    auto& mesh_device_impl = mesh_device->impl();
    mesh_device_impl.mesh_command_queues_.clear();
    mesh_device_impl.mesh_command_queues_.reserve(num_hw_cqs);

    auto cq_shared_state = std::make_shared<CQSharedState>();
    cq_shared_state->sub_device_cq_owner.resize(1);

    for (std::size_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
        mesh_device_impl.mesh_command_queues_.push_back(std::make_unique<distributed::FDMeshCommandQueue>(
            mesh_device,
            cq_id,
            mesh_device_impl.dispatch_thread_pool_,
            mesh_device_impl.reader_thread_pool_,
            cq_shared_state,
            std::bind(&distributed::MeshDeviceImpl::lock_api, &mesh_device_impl)));
    }
    fast_dispatch_enabled_ = true;
    num_fd_inits_++;
}

void DispatchContext::terminate_fast_dispatch(distributed::MeshDevice* mesh_device) {
    TT_FATAL(fast_dispatch_enabled_, "Can only manually terminate fast dispatch after initializing it.");
    TT_FATAL(num_fd_inits_ == 1, "Fast Dispatch can only be manually terminated and torn down once.");

    const auto& device_manager = MetalContext::instance().device_manager();
    const auto& active_devices = device_manager->get_all_active_devices();

    uint8_t num_hw_cqs = active_devices[0]->num_hw_cqs();
    auto& mesh_device_impl = mesh_device->impl();
    mesh_device_impl.mesh_command_queues_.clear();
    mesh_device_impl.mesh_command_queues_.reserve(num_hw_cqs);
    for (std::size_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
        mesh_device_impl.mesh_command_queues_.push_back(std::make_unique<distributed::SDMeshCommandQueue>(
            mesh_device, cq_id, std::bind(&distributed::MeshDeviceImpl::lock_api, &mesh_device_impl)));
    }
    for (const auto& dev : active_devices) {
        for (int cq_id = 0; cq_id < dev->num_hw_cqs(); cq_id++) {
            dynamic_cast<tt::tt_metal::Device*>(dev)->command_queues_[cq_id].get()->terminate();
        }
    }

    for (const auto& dev : active_devices) {
        auto dispatch_cores = tt::tt_metal::get_virtual_dispatch_cores(dev->id());
        tt::llrt::internal_::wait_until_cores_done(dev->id(), dev_msgs::RUN_MSG_GO, dispatch_cores, 0);
    }

    fast_dispatch_enabled_ = false;
    tt::tt_metal::MetalContext::instance().rtoptions().set_fast_dispatch(fast_dispatch_enabled_);

    // Reinitialize dispatch managers to pick up SD core descriptor after disabling FD
    MetalContext::instance().reinitialize_dispatch_managers();
}

}  // namespace tt::tt_metal::experimental
