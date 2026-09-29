// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal::emule {

// Process-wide coordination for the opt-in cross-queue emulation dispatch generation. The generic
// queue owns the enqueue/fence call sites; this module owns all emulation-only state and locking.
bool deferred_mesh_dispatch_enabled();
bool deferred_mesh_dispatch_has_queue(const void* queue);
void register_deferred_mesh_dispatch_queue(const void* queue);
void flush_deferred_mesh_dispatch();

}  // namespace tt::tt_metal::emule
