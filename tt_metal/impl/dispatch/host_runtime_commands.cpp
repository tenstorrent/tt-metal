// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>
#include <buffer.hpp>
#include <event.hpp>
#include <host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_metal.hpp>
#include <functional>
#include <memory>

#include "device.hpp"
#include "dispatch/device_command.hpp"
#include "impl/context/metal_context.hpp"
#include "hal_types.hpp"
#include "lightmetal/host_api_capture_helpers.hpp"
#include "tt-metalium/program.hpp"
#include <tt_stl/span.hpp>
#include <tt_stl/overloaded.hpp>
#include "system_memory_manager.hpp"
#include "tt_metal/impl/dispatch/data_collection.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"
#include "tt_metal/impl/program/dispatch.hpp"
#include <impl/dispatch/dispatch_query_manager.hpp>
#include <impl/debug/dprint_server.hpp>
#include <impl/debug/watcher_server.hpp>

using namespace tt::tt_metal;

namespace tt::tt_metal {

namespace detail {

bool DispatchStateCheck(bool isFastDispatch) {
    static bool fd = isFastDispatch;
    TT_FATAL(fd == isFastDispatch, "Mixing fast and slow dispatch is prohibited!");
    return fd;
}

Buffer& GetBufferObject(const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer) {
    return std::visit(
        ttsl::overloaded{
            [](const std::shared_ptr<Buffer>& b) -> Buffer& { return *b; },
            [](Buffer& b) -> Buffer& { return b; },
        },
        buffer);
}

void ValidateBufferRegion(
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer, const BufferRegion& region) {
    Buffer& buffer_obj = GetBufferObject(buffer);

    TT_FATAL(
        buffer_obj.is_valid_region(region),
        "Buffer region with offset {} and size {} is invalid.",
        region.offset,
        region.size);
}
}  // namespace detail


bool EventQuery(const std::shared_ptr<Event>& event) {
    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch()) {
        // Slow dispatch always returns true to avoid infinite blocking. Unclear if this is safe for all situations.
        return true;
    }
    detail::DispatchStateCheck(true);
    event->wait_until_ready();  // Block until event populated. Parent thread.
    bool event_completed = event->device->sysmem_manager().get_last_completed_event(event->cq_id) >= event->event_id;
    log_trace(
        tt::LogMetal,
        "Returning event_completed: {} for host query on Event(device_id: {} cq_id: {} event_id: {})",
        event_completed,
        event->device->id(),
        event->cq_id,
        event->event_id);
    return event_completed;
}

}  // namespace tt::tt_metal
