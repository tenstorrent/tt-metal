// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <typeinfo>

#include <tt-metalium/graph_tracking.hpp>

namespace tt::tt_metal::internal {

// INTERNAL API. Not part of the stable public surface; may change or disappear without notice.
//
// Register a permanently-active, process-wide graph processor, unless one of the same dynamic
// type is already registered. Unlike GraphTracker::push_processor these are visible on every
// thread, are never popped, and survive clear() -- which is what an always-on observer such as
// SHM memory tracking needs, since per-thread storage misses every buffer allocated off the
// thread that initialized the device. Registering a capture processor this way would make it
// observe unrelated threads' work, so this is for is_capture_processor() == false only.
//
// Returns true if this call registered the processor, false if one was already present.
//
// The lookup and the insertion happen under one exclusive lock. Asking first and pushing after
// would be check-then-act: two devices initializing concurrently could both observe absence and
// register a second observer, and every buffer event would then be recorded twice.
//
// `factory` is invoked only when a registration is actually going to happen, so the common
// repeat call constructs nothing. It runs while the lock is held; it must not re-enter
// GraphTracker.
bool register_background_processor_once(
    const std::type_info& type, const std::function<std::shared_ptr<IGraphProcessor>()>& factory);

}  // namespace tt::tt_metal::internal
