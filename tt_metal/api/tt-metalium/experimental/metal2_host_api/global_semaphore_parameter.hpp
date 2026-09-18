// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include <tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal::experimental {

// ============================================================================
//  GlobalSemaphoreParameter API
// ============================================================================
//
// TODO...

// A name identifying a GlobalSemaphoreParameter within a ProgramSpec.
using GlobalSemaphoreParamName = ttsl::StrongType<std::string, struct GlobalSemaphoreParamNameTag>;

struct GlobalSemaphoreParameter {
    // GlobalSemaphore identifier: used to reference this GlobalSemaphore within the ProgramSpec
    GlobalSemaphoreParamName unique_id;

    // TODO...
    // Are there any properties a GlobalSemaphore has, that we need to legality check
    // against the supplied GlobalSemaphore object at runtime?
};

}  // namespace tt::tt_metal::experimental
