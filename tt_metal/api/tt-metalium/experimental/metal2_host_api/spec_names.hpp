// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal::experimental {

// ============================================================================
//  Spec entity names
// ============================================================================
//
// Strong-typed identifiers for the entities declared within a ProgramSpec:
// kernels, dataflow buffers, semaphores, and tensor parameters. A binding or
// run-arg refers to a declared entity by its name.
//
// These names live together in one low-level header (rather than each beside
// its own Spec) so that any header can name an entity without pulling in that
// entity's full Spec definition — and so the DFB name in particular has a home
// that doesn't depend on include ordering between advanced_options.hpp and
// dataflow_buffer_spec.hpp.
//
// ============================================================================

using KernelSpecName = ttsl::StrongType<std::string, struct KernelSpecNameTag>;
using DFBSpecName = ttsl::StrongType<std::string, struct DFBSpecNameTag>;
using SemaphoreSpecName = ttsl::StrongType<std::string, struct SemaphoreSpecNameTag>;
using TensorParamName = ttsl::StrongType<std::string, struct TensorParamNameTag>;

}  // namespace tt::tt_metal::experimental
