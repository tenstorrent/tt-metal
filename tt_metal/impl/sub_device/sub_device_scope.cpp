// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/sub_device/sub_device_scope.hpp"

#include <atomic>
#include <cstdint>
#include <deque>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/sub_device_scope.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "distributed/mesh_workload_impl.hpp"
#include "impl/buffers/circular_buffer.hpp"
#include "impl/buffers/semaphore.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "llrt/hal.hpp"

namespace tt::tt_metal {

namespace {

// The scopes open on this thread, innermost last. Each one is stored as the single-element list that
// get_sub_device_ids() returns while it is active. A deque is used because callers may hold a reference to one list
// while another scope is pushed. If a thread exits with scopes still open, the destructor removes them from the
// global count.
struct ScopesForThread {
    std::deque<std::vector<SubDeviceId>> open;
    ~ScopesForThread() {
        sub_device_scope::num_open.fetch_sub(static_cast<uint32_t>(open.size()), std::memory_order_relaxed);
    }
};

std::deque<std::vector<SubDeviceId>>& scopes_for_thread() {
    thread_local ScopesForThread scopes;
    return scopes.open;
}

// The innermost open scope on this thread, or null if there is none.
constinit thread_local const std::vector<SubDeviceId>* innermost_scope_for_thread = nullptr;

// How many SuspendSubDeviceScope objects are alive on this thread.
constinit thread_local uint32_t suspensions_for_thread = 0;

// Returns the Tensix cores that belong to every sub-device other than `scoped`. The result is cached per thread for
// the last manager and sub-device asked about. That is safe because a manager's id is never reused and its
// sub-devices never change.
const CoreRangeSet& other_sub_device_cores(const distributed::MeshDevice& mesh_device, SubDeviceId scoped) {
    struct Answer {
        SubDeviceManagerId manager_id;
        SubDeviceId scoped;
        CoreRangeSet cores;
    };
    thread_local std::optional<Answer> last;
    const auto manager_id = mesh_device.get_active_sub_device_manager_id();
    if (!last.has_value() || last->manager_id != manager_id || last->scoped != scoped) {
        CoreRangeSet cores;
        for (uint32_t index = 0; index < mesh_device.num_sub_devices(); ++index) {
            const SubDeviceId other{static_cast<SubDeviceId::value_type>(index)};
            if (other != scoped) {
                cores = cores.merge(mesh_device.worker_cores(HalProgrammableCoreType::TENSIX, other));
            }
        }
        last = Answer{manager_id, scoped, std::move(cores)};
    }
    return last->cores;
}

// Describes a program for an error message, for example "12 (kernels: reader, writer)".
std::string describe(detail::ProgramImpl& program, const Hal& hal) {
    std::set<std::string> kernel_names;
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); ++index) {
        for (const auto& [handle, kernel] : program.get_kernels(index)) {
            kernel_names.insert(kernel->name());
        }
    }
    return fmt::format("{} (kernels: {})", program.get_id(), fmt::join(kernel_names, ", "));
}

[[noreturn]] void refuse_kernels_off_the_scoped_sub_device(
    detail::ProgramImpl& program,
    const distributed::MeshDevice& mesh_device,
    SubDeviceId scoped,
    const std::vector<SubDeviceId>& ran_on) {
    const Hal& hal = mesh_device.impl().metal_env().get_hal();
    CoreRangeSet kernel_cores;
    for (const auto& [handle, kernel] :
         program.get_kernels(hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX))) {
        kernel_cores = kernel_cores.merge(kernel->core_range_set());
    }
    std::set<uint32_t> sub_device_ids;
    for (const auto& sub_device_id : ran_on) {
        sub_device_ids.insert(*sub_device_id);
    }
    TT_THROW(
        "Program {} cannot run inside the scope for sub-device {}. Its kernels are on cores {}, which belong to "
        "sub-device(s) {}, but the scope only allows cores {}. This usually means the operation does not choose its "
        "cores from the compute grid or the sub-device list, or that sub-device {} is not a rectangle starting at "
        "(0, 0).",
        describe(program, hal),
        *scoped,
        kernel_cores.str(),
        fmt::format("{}", fmt::join(sub_device_ids, ", ")),
        mesh_device.worker_cores(HalProgrammableCoreType::TENSIX, scoped).str(),
        *scoped);
}

[[noreturn]] void refuse_object_off_the_scoped_sub_device(
    detail::ProgramImpl& program,
    const distributed::MeshDevice& mesh_device,
    SubDeviceId scoped,
    std::string_view object,
    std::string_view written,
    const CoreRangeSet& object_cores) {
    TT_THROW(
        "Program {} cannot run inside the scope for sub-device {}. Its {} is placed on cores {}, and {} of those "
        "belong to another sub-device. Dispatch would write the {} to those cores and corrupt whatever is running "
        "there.",
        describe(program, mesh_device.impl().metal_env().get_hal()),
        *scoped,
        object,
        object_cores.str(),
        object_cores.intersection(other_sub_device_cores(mesh_device, scoped)).str(),
        written);
}

}  // namespace

namespace experimental {

void PushCurrentSubDeviceIdForThread(SubDeviceId sub_device_id) {
    auto& scopes = scopes_for_thread();
    innermost_scope_for_thread = &scopes.emplace_back(1, sub_device_id);
    sub_device_scope::num_open.fetch_add(1, std::memory_order_relaxed);
}

void PopCurrentSubDeviceIdForThread() {
    auto& scopes = scopes_for_thread();
    TT_FATAL(!scopes.empty(), "There is no sub-device scope to close on this thread");
    sub_device_scope::num_open.fetch_sub(1, std::memory_order_relaxed);
    scopes.pop_back();
    innermost_scope_for_thread = scopes.empty() ? nullptr : &scopes.back();
}

std::optional<SubDeviceId> GetCurrentSubDeviceIdForThread() {
    if (!sub_device_scope::any_open() || innermost_scope_for_thread == nullptr) {
        return std::nullopt;
    }
    return innermost_scope_for_thread->front();
}

}  // namespace experimental

namespace sub_device_scope {

std::atomic<uint32_t> num_open{0};

const std::vector<SubDeviceId>* ids_for_thread() {
    return suspensions_for_thread == 0 ? innermost_scope_for_thread : nullptr;
}

void suspend_for_thread() { ++suspensions_for_thread; }

void resume_for_thread() { --suspensions_for_thread; }

}  // namespace sub_device_scope

void validate_scoped_sub_device(const IDevice& device, SubDeviceId scoped) {
    TT_FATAL(
        *scoped < device.num_sub_devices(),
        "The current scope is for sub-device {}, but the loaded sub-device manager only has {} sub-device(s)",
        *scoped,
        device.num_sub_devices());
}

void sub_device_scope::validate(distributed::MeshWorkloadImpl& workload, distributed::MeshDevice* mesh_device) {
    if (innermost_scope_for_thread == nullptr) {
        return;
    }
    const SubDeviceId scoped = innermost_scope_for_thread->front();
    validate_scoped_sub_device(*mesh_device, scoped);
    const CoreRangeSet& others = other_sub_device_cores(*mesh_device, scoped);
    for (auto& [device_range, program] : workload.get_programs()) {
        auto& program_impl = program.impl();
        // The kernels decide which sub-device a program runs on. A program without kernels is never dispatched,
        // so there is nothing to check.
        const auto& ran_on = program_impl.determine_sub_device_ids(mesh_device);
        if (ran_on.empty()) {
            continue;
        }
        if (ran_on.size() != 1 || ran_on.front() != scoped) [[unlikely]] {
            refuse_kernels_off_the_scoped_sub_device(program_impl, *mesh_device, scoped, ran_on);
        }
        if (others.empty()) {
            continue;
        }
        // Semaphores and buffers matter too. Fast dispatch writes their initial values and configs to every core
        // they are placed on, which would overwrite the state of a program running on another sub-device. Cores
        // that belong to no sub-device are fine, because nothing runs on them.
        for (const Semaphore& semaphore : program_impl.semaphores()) {
            if (semaphore.core_type() != CoreType::WORKER) {
                continue;  // fast dispatch writes no Ethernet semaphores
            }
            const CoreRangeSet semaphore_cores = semaphore.core_range_set();
            if (others.intersects(semaphore_cores)) [[unlikely]] {
                refuse_object_off_the_scoped_sub_device(
                    program_impl,
                    *mesh_device,
                    scoped,
                    fmt::format("semaphore {}", semaphore.id()),
                    "semaphore's initial value",
                    semaphore_cores);
            }
        }
        for (const auto& circular_buffer : program_impl.circular_buffers()) {
            if (others.intersects(circular_buffer->core_ranges())) [[unlikely]] {
                const std::set<uint32_t> indices(
                    circular_buffer->buffer_indices().begin(), circular_buffer->buffer_indices().end());
                refuse_object_off_the_scoped_sub_device(
                    program_impl,
                    *mesh_device,
                    scoped,
                    fmt::format("circular buffer (indices {})", fmt::join(indices, ", ")),
                    "buffer's config",
                    circular_buffer->core_ranges());
            }
        }
        for (const auto& dataflow_buffer : program_impl.dataflow_buffers()) {
            if (others.intersects(dataflow_buffer->core_ranges)) [[unlikely]] {
                refuse_object_off_the_scoped_sub_device(
                    program_impl,
                    *mesh_device,
                    scoped,
                    fmt::format("dataflow buffer {}", dataflow_buffer->id),
                    "buffer's config",
                    dataflow_buffer->core_ranges);
            }
        }
    }
}

}  // namespace tt::tt_metal
