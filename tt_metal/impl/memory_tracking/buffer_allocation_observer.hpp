// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <typeinfo>

namespace tt::tt_metal {

class Buffer;

// A process-wide observer of buffer allocation, notified from Buffer::allocate_impl() and
// Buffer::deallocate().
//
// This is deliberately not an IGraphProcessor. SHM memory tracking wants exactly these two
// notifications, both of which are emitted from impl/buffers/buffer.cpp and consumed here in
// impl -- it has no business on the Metalium public API surface, and routing it through
// GraphTracker would only entrench a component that exists to serve TTNN internals.
//
// The other reason not to use GraphTracker: its processor stack is thread_local, so an
// observer pushed there misses every buffer allocated off the thread that registered it.
// Observers here are process-wide by construction.
class BufferAllocationObserver {
public:
    virtual ~BufferAllocationObserver() = default;

    virtual void track_allocate(const Buffer* buffer) = 0;
    virtual void track_deallocate(Buffer* buffer) = 0;
};

// Register an observer, unless one of the same dynamic type is already registered. Returns true
// if this call registered it.
//
// The lookup and the insertion happen under one exclusive lock. Asking first and inserting after
// would be check-then-act: two devices initializing concurrently could both observe absence and
// register a second observer, and every buffer event would then be recorded twice.
//
// `factory` runs only when a registration is actually going to happen, so the repeat call every
// device after the first makes constructs nothing. It runs with the lock held and must not
// re-enter this registry.
bool register_buffer_allocation_observer_once(
    const std::type_info& type, const std::function<std::shared_ptr<BufferAllocationObserver>()>& factory);

// Notify every registered observer. No-ops when nothing is registered, which is the case
// whenever SHM tracking is off.
void notify_buffer_allocated(const Buffer* buffer);
void notify_buffer_deallocated(Buffer* buffer);

}  // namespace tt::tt_metal
