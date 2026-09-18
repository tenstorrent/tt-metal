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
//  PrefetcherPipeParameter API
// ============================================================================
//
// TODO...

// A name identifying a PrefetcherPipeParameter within a ProgramSpec.
using PrefetcherPipeParamName = ttsl::StrongType<std::string, struct PrefetcherPipeParamNameTag>;

struct PrefetcherPipeParameter {
    // PrefetcherPipe identifier: used to reference this PrefetcherPipe within the ProgramSpec
    PrefetcherPipeParamName unique_id;

    // TODO...
    // What properties does a PrefetcherPipe have, that we need to legality check
    // against the supplied PrefetcherPipe object at runtime?
    //
    // I want to reuse a struct that is used to construct the actual PrefetcherPipe object.
    // (In much the way we do with TensorParameter and TensorSpec.)
    //
};

}  // namespace tt::tt_metal::experimental
