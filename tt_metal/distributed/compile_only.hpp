// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "impl/context/context_types.hpp"

// Internal tt-metal header.

namespace tt::tt_metal::distributed {

class MeshDevice;

// Compile-only mode (RunTimeOptions::compile_only): EnqueueMeshWorkload fires each workload's kernel
// compilation asynchronously and skips dispatch, so many programs' compiles run concurrently across
// host cores instead of one blocking compile per op. Joins the pending async compiles of one context
// -- e.g. at the end of a pre-compilation pass, before switching back to normal execution or
// clearing the program cache. Other contexts' builds are left untouched.

void WaitForPendingCompiles(ContextId context_id);

void WaitForPendingCompiles(const MeshDevice& mesh_device);

}  // namespace tt::tt_metal::distributed
