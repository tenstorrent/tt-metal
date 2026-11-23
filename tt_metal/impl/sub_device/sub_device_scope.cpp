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

// The innermost open scope on this thread, or null if there is none.
constinit thread_local const std::vector<SubDeviceId>* innermost_scope_for_thread = nullptr;

// How many SuspendSubDeviceScope objects are alive on this thread.
constinit thread_local uint32_t suspensions_for_thread = 0;

// Set once the thread's ScopesForThread is destroyed. A SubDeviceScope held in static storage, or in a thread_local
// created after the thread's first scope, is destroyed later than that, and must then find no scopes left to close.
constinit thread_local bool scopes_destroyed_for_thread = false;

// The scopes open on this thread, innermost last. Each one is stored as the single-element list that
// get_sub_device_ids() returns while it is active. A deque is used because callers may hold a reference to one list
// while another scope is pushed. If a thread exits with scopes still open, the destructor removes them from the
// global count and forgets the innermost one, so nothing reads the freed list afterwards.
struct ScopesForThread {
    std::deque<std::vector<SubDeviceId>> open;
    ~ScopesForThread() {
        scopes_destroyed_for_thread = true;
        innermost_scope_for_thread = nullptr;
        sub_device_scope::num_open.fetch_sub(static_cast<uint32_t>(open.size()), std::memory_order_relaxed);
    }
};

std::deque<std::vector<SubDeviceId>>& scopes_for_thread() {
    thread_local ScopesForThread scopes;
    return scopes.open;
}

// The Tensix cores of the scoped sub-device, and those of all the other sub-devices.
struct ScopedCores {
    CoreRangeSet own;
    CoreRangeSet others;
};

// Returns the cores for `scoped`, which must exist in the loaded manager. The answers are cached per thread for the
// loaded manager, one per sub-device, so a thread that switches between scopes does not recompute them. That is safe
// because a manager's id is never reused and its sub-devices never change.
const ScopedCores& scoped_cores(const distributed::MeshDevice& mesh_device, SubDeviceId scoped) {
    struct Cache {
        SubDeviceManagerId manager_id;
        std::vector<std::optional<ScopedCores>> by_sub_device;
    };
    thread_local std::optional<Cache> cache;
    const auto manager_id = mesh_device.get_active_sub_device_manager_id();
    if (!cache.has_value() || cache->manager_id != manager_id) {
        cache = Cache{manager_id, std::vector<std::optional<ScopedCores>>(mesh_device.num_sub_devices())};
    }
    auto& answer = cache->by_sub_device.at(*scoped);
    if (!answer.has_value()) {
        ScopedCores cores{.own = mesh_device.worker_cores(HalProgrammableCoreType::TENSIX, scoped)};
        for (uint32_t index = 0; index < mesh_device.num_sub_devices(); ++index) {
            const SubDeviceId other{static_cast<SubDeviceId::value_type>(index)};
            if (other != scoped) {
                cores.others = cores.others.merge(mesh_device.worker_cores(HalProgrammableCoreType::TENSIX, other));
            }
        }
        answer = std::move(cores);
    }
    return *answer;
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

// Says which sub-devices `cores` belong to, for example "which belong to sub-device(s) 0, 1" or "which belong to
// sub-device(s) 1 and to no sub-device".
std::string owners_of(
    const CoreRangeSet& cores, HalProgrammableCoreType core_type, const distributed::MeshDevice& mesh_device) {
    std::vector<uint32_t> owners;
    CoreRangeSet owned;
    for (uint32_t index = 0; index < mesh_device.num_sub_devices(); ++index) {
        const CoreRangeSet sub_device_cores =
            mesh_device.worker_cores(core_type, SubDeviceId{static_cast<SubDeviceId::value_type>(index)});
        if (sub_device_cores.intersects(cores)) {
            owners.push_back(index);
            owned = owned.merge(sub_device_cores);
        }
    }
    if (owners.empty()) {
        return "which belong to no sub-device";
    }
    return fmt::format(
        "which belong to sub-device(s) {}{}",
        fmt::join(owners, ", "),
        owned.contains(cores) ? "" : " and to no sub-device");
}

[[noreturn]] void refuse_kernels_off_the_scoped_sub_device(
    detail::ProgramImpl& program,
    const distributed::MeshDevice& mesh_device,
    SubDeviceId scoped,
    HalProgrammableCoreType core_type,
    const CoreRangeSet& allowed) {
    const Hal& hal = mesh_device.impl().metal_env().get_hal();
    CoreRangeSet kernel_cores;
    for (const auto& [handle, kernel] : program.get_kernels(hal.get_programmable_core_type_index(core_type))) {
        kernel_cores = kernel_cores.merge(kernel->core_range_set());
    }
    TT_THROW(
        "Program {} cannot run inside the scope for sub-device {}. Its kernels are on cores {}, {}, but the scope "
        "only allows cores {}. This usually means the operation does not choose its cores from the compute grid or "
        "the sub-device list, or that sub-device {} is not a rectangle starting at (0, 0).",
        describe(program, hal),
        *scoped,
        kernel_cores.str(),
        owners_of(kernel_cores, core_type, mesh_device),
        allowed.str(),
        *scoped);
}

[[noreturn]] void refuse_object_off_the_scoped_sub_device(
    detail::ProgramImpl& program,
    const distributed::MeshDevice& mesh_device,
    SubDeviceId scoped,
    std::string_view object,
    std::string_view written,
    const CoreRangeSet& object_cores,
    const CoreRangeSet& others) {
    TT_THROW(
        "Program {} cannot run inside the scope for sub-device {}. Its {} is placed on cores {}, and {} of those "
        "belong to another sub-device. Dispatch would write the {} to those cores and corrupt whatever is running "
        "there.",
        describe(program, mesh_device.impl().metal_env().get_hal()),
        *scoped,
        object,
        object_cores.str(),
        object_cores.intersection(others).str(),
        written);
}

}  // namespace

namespace experimental {

void PushCurrentSubDeviceIdForThread(SubDeviceId sub_device_id) {
    if (scopes_destroyed_for_thread) [[unlikely]] {
        return;
    }
    auto& scopes = scopes_for_thread();
    innermost_scope_for_thread = &scopes.emplace_back(1, sub_device_id);
    sub_device_scope::num_open.fetch_add(1, std::memory_order_relaxed);
}

void PopCurrentSubDeviceIdForThread() {
    if (scopes_destroyed_for_thread) [[unlikely]] {
        return;
    }
    auto& scopes = scopes_for_thread();
    TT_FATAL(!scopes.empty(), "There is no sub-device scope to close on this thread");
    sub_device_scope::num_open.fetch_sub(1, std::memory_order_relaxed);
    scopes.pop_back();
    innermost_scope_for_thread = scopes.empty() ? nullptr : &scopes.back();
}

std::optional<SubDeviceId> GetCurrentSubDeviceIdForThread() {
    const auto* scoped = scoped_sub_device_ids_for_thread();
    if (scoped == nullptr) {
        return std::nullopt;
    }
    return scoped->front();
}

std::size_t GetSubDeviceScopeDepthForThread() { return scopes_destroyed_for_thread ? 0 : scopes_for_thread().size(); }

SubDeviceScope::SubDeviceScope(SubDeviceId sub_device_id) : depth_before_(GetSubDeviceScopeDepthForThread()) {
    PushCurrentSubDeviceIdForThread(sub_device_id);
}

// Scopes are matched by position on the stack, not by id, so this never closes a scope opened outside this one.
SubDeviceScope::~SubDeviceScope() {
    while (GetSubDeviceScopeDepthForThread() > depth_before_) {
        PopCurrentSubDeviceIdForThread();
    }
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
    const auto* scoped_ids = ids_for_thread();
    if (scoped_ids == nullptr) {
        return;
    }
    const SubDeviceId scoped = scoped_ids->front();
    validate_scoped_sub_device(*mesh_device, scoped);
    const ScopedCores& cores = scoped_cores(*mesh_device, scoped);
    const Hal& hal = mesh_device->impl().metal_env().get_hal();
    const uint32_t tensix = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    const bool has_active_eth = hal.has_programmable_core_type(HalProgrammableCoreType::ACTIVE_ETH);
    for (auto& [device_range, program] : workload.get_programs()) {
        auto& program_impl = program.impl();
        // The kernels decide which sub-device a program runs on. Their cores are read from the kernels themselves.
        // Reading kernel groups would build and cache them before compile, and a rejected program that the caller then
        // gives a circular buffer would keep the stale groups.
        for (const auto& [handle, kernel] : program_impl.get_kernels(tensix)) {
            if (!cores.own.contains(kernel->core_range_set())) [[unlikely]] {
                refuse_kernels_off_the_scoped_sub_device(
                    program_impl, *mesh_device, scoped, HalProgrammableCoreType::TENSIX, cores.own);
            }
        }
        // An Ethernet kernel runs on the sub-device that owns its core, so it must not be on another sub-device's
        // Ethernet cores. Ethernet cores that belong to no sub-device are not checked here: with a sub-device manager
        // loaded, compiling the program rejects them, as it does outside a scope.
        if (has_active_eth) {
            const uint32_t active_eth = hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
            for (const auto& [handle, kernel] : program_impl.get_kernels(active_eth)) {
                for (uint32_t index = 0; index < mesh_device->num_sub_devices(); ++index) {
                    const SubDeviceId other{static_cast<SubDeviceId::value_type>(index)};
                    if (other != scoped && mesh_device->worker_cores(HalProgrammableCoreType::ACTIVE_ETH, other)
                                               .intersects(kernel->core_range_set())) [[unlikely]] {
                        refuse_kernels_off_the_scoped_sub_device(
                            program_impl,
                            *mesh_device,
                            scoped,
                            HalProgrammableCoreType::ACTIVE_ETH,
                            mesh_device->worker_cores(HalProgrammableCoreType::ACTIVE_ETH, scoped));
                    }
                }
            }
        }
        if (cores.others.empty()) {
            continue;
        }
        // Semaphores and buffers matter too. Fast dispatch writes their initial values and configs to every core
        // they are placed on, which would overwrite the state of a program running on another sub-device. Cores
        // that belong to no sub-device are fine, because nothing runs on them. The object is only named on failure.
        const auto check = [&](const CoreRangeSet& object_cores, std::string_view written, const auto& name) {
            if (cores.others.intersects(object_cores)) [[unlikely]] {
                refuse_object_off_the_scoped_sub_device(
                    program_impl, *mesh_device, scoped, name(), written, object_cores, cores.others);
            }
        };
        for (const Semaphore& semaphore : program_impl.semaphores()) {
            if (semaphore.core_type() == CoreType::WORKER) {  // fast dispatch writes no Ethernet semaphores
                check(semaphore.core_range_set(), "semaphore's initial value", [&] {
                    return fmt::format("semaphore {}", semaphore.id());
                });
            }
        }
        for (const auto& circular_buffer : program_impl.circular_buffers()) {
            check(circular_buffer->core_ranges(), "buffer's config", [&] {
                const std::set<uint32_t> indices(
                    circular_buffer->buffer_indices().begin(), circular_buffer->buffer_indices().end());
                return fmt::format("circular buffer (indices {})", fmt::join(indices, ", "));
            });
        }
        for (const auto& dataflow_buffer : program_impl.dataflow_buffers()) {
            check(dataflow_buffer->core_ranges, "buffer's config", [&] {
                return fmt::format("dataflow buffer {}", dataflow_buffer->id);
            });
        }
    }
}

}  // namespace tt::tt_metal
