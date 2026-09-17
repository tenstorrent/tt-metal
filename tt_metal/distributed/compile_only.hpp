// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "impl/context/context_types.hpp"

// INTERNAL (non-public) tt-metal header. This is not part of the installed tt-metalium API surface
// and carries no stability guarantees. Do not include from downstream/user code.

namespace tt::tt_metal::distributed {

// Compile-only mode (RunTimeOptions::compile_only): EnqueueMeshWorkload fires each workload's kernel
// compilation asynchronously and skips dispatch, so many programs' compiles run concurrently across
// host cores instead of one blocking compile per op. Joins the pending async compiles of one context
// -- e.g. at the end of a pre-compilation pass, before switching back to normal execution or
// clearing the program cache. Other contexts' builds are left untouched.
//
// Internal lifecycle plumbing: MeshDevice invokes this automatically on close and on program-cache
// clear/disable, so there is no public entry point -- callers outside tt-metal should rely on those
// MeshDevice lifecycle points rather than calling this directly.
void WaitForPendingCompiles(ContextId context_id);

}  // namespace tt::tt_metal::distributed
