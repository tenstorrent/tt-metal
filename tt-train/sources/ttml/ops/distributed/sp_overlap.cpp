// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sp_overlap.hpp"

#include <array>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_command_queue.hpp>

#include "autograd/auto_context.hpp"
#include "ttnn/core.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::ops::distributed {

namespace {

SPOverlapMode g_sp_overlap_mode = SPOverlapMode::Off;

constexpr ttnn::QueueId kComputeQueue{0};
constexpr ttnn::QueueId kCclQueue{1};
// A collective's buffers are freed once this many later collectives have been waited for on the compute
// queue: the wait proves this device done with them, a slower peer's copy may still be finishing.
constexpr size_t kRetireDepth = 2;

}  // namespace

void set_sp_overlap_mode(SPOverlapMode mode) {
    if (mode != SPOverlapMode::Off) {
        auto& ctx = autograd::ctx();
        TT_FATAL(
            ctx.num_command_queues() == 2U,
            "sp_overlap needs the device opened with two command queues (open_device(..., num_command_queues=2))");
        TT_FATAL(
            ctx.has_ccl_sub_device(),
            "sp_overlap needs the CCL sub-device (AutoContext::enable_ccl_sub_device) so collectives on the second "
            "queue have cores of their own");
        if (ttnn_fixed::distributed::get_sp_linear_backward_impl() == ttnn_fixed::distributed::SPLinearImpl::Fused) {
            log_warning(
                tt::LogAlways,
                "sp_overlap is only applied to the Composed backward of the sequence-parallel linears; the current "
                "backward implementation runs unchanged on the compute queue");
        }
        SPOverlap::instance();  // install the backward hooks
    }
    g_sp_overlap_mode = mode;
}

SPOverlapMode get_sp_overlap_mode() {
    return g_sp_overlap_mode;
}

SPOverlap& SPOverlap::instance() {
    static SPOverlap instance;
    return instance;
}

SPOverlap::SPOverlap() {
    auto& ctx = autograd::ctx();
    ctx.add_backward_end_hook([]() { SPOverlap::instance().finish_backward(); });
    ctx.add_teardown_hook([](autograd::AutoContext::TeardownReason reason) {
        SPOverlap::instance().teardown(reason == autograd::AutoContext::TeardownReason::DeviceClose);
    });
}

bool SPOverlap::backward_active() const {
    if (g_sp_overlap_mode != SPOverlapMode::Backward) {
        return false;
    }
    auto& ctx = autograd::ctx();
    // Outside Tensor::backward() nothing would drain the deferred work; without the second queue and the
    // CCL sub-device (e.g. after the device was closed and reopened plainly) there is nothing to overlap on.
    // NoComm (measurement only) takes the same path with the collectives replaced by their empty outputs, so the
    // cost of the scheduling itself -- events, deferral, staging allocations -- can be measured on its own.
    const auto impl = ttnn_fixed::distributed::get_sp_linear_backward_impl();
    return (impl == ttnn_fixed::distributed::SPLinearImpl::Composed ||
            impl == ttnn_fixed::distributed::SPLinearImpl::NoComm) &&
           ctx.is_backward_in_progress() && ctx.num_command_queues() == 2U && ctx.has_ccl_sub_device();
}

void SPOverlap::compute_drain() {
    auto& ctx = autograd::ctx();
    auto& device = ctx.get_device();
    const std::array<tt::tt_metal::SubDeviceId, 1> compute{ctx.compute_sub_device_id()};
    auto event = device.mesh_command_queue(*kComputeQueue).enqueue_record_event(compute);
    device.mesh_command_queue(*kCclQueue).enqueue_wait_for_event(event);
}

SPOverlap::Collective SPOverlap::issue(
    const std::function<ttnn::Tensor(std::vector<ttnn::Tensor>&)>& collective, std::vector<ttnn::Tensor> reads) {
    auto& ctx = autograd::ctx();
    auto& device = ctx.get_device();
    compute_drain();
    std::vector<ttnn::Tensor> buffers = std::move(reads);
    ttnn::Tensor output;
    {
        auto queue_guard = ttnn::core::with_command_queue_id(kCclQueue);
        output = collective(buffers);
    }
    buffers.push_back(output);
    const std::array<tt::tt_metal::SubDeviceId, 1> ccl{*ctx.ccl_sub_device_id()};
    auto done = device.mesh_command_queue(*kCclQueue).enqueue_record_event(ccl);
    return Collective{std::move(output), std::move(done), std::move(buffers)};
}

void SPOverlap::wait(Collective collective) {
    auto& device = autograd::ctx().get_device();
    device.mesh_command_queue(*kComputeQueue).enqueue_wait_for_event(collective.done);
    m_retire.push_back(std::move(collective.buffers));
    while (m_retire.size() > kRetireDepth) {
        m_retire.pop_front();
    }
}

void SPOverlap::defer(std::function<void()> work) {
    m_deferred.push_back(std::move(work));
}

void SPOverlap::drain_one() {
    if (m_deferred.empty()) {
        return;
    }
    auto work = std::move(m_deferred.front());
    m_deferred.pop_front();
    work();
}

void SPOverlap::drain_all() {
    while (!m_deferred.empty()) {
        drain_one();
    }
}

size_t SPOverlap::num_deferred() const {
    return m_deferred.size();
}

size_t SPOverlap::num_retained() const {
    return m_retire.size();
}

void SPOverlap::finish_backward() {
    drain_all();
}

void SPOverlap::teardown(bool device_closing) {
    if (!m_deferred.empty()) {
        // Only an interrupted backward leaves deferred work behind (a completed one drains it). Its
        // collectives may still be in flight: let the device go idle before their buffers are dropped.
        tt::tt_metal::distributed::Synchronize(&autograd::ctx().get_device(), std::nullopt);
        m_deferred.clear();
    }
    if (device_closing) {
        m_retire.clear();
    }
}

}  // namespace ttml::ops::distributed
