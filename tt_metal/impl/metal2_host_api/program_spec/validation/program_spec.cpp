// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * This file contains validation of ProgramSpec struct that's not covered witin
 * `../resource` or `../placement`.
 *
 * Normally those files contain the most validations, and current program_spec.cpp is just a catch-all for everything
 * else.
 */

#include <tt_stl/assert.hpp>
#include <tt_stl/fmt.hpp>

#include "impl/metal2_host_api/program_spec/validation/validate_spec.hpp"

namespace tt::tt_metal::experimental {

void ValidateProgramMisc(const ValidationContext& ctx) {
    const ProgramSpec& spec = ctx.spec;

    // A Program needs at least one kernel
    TT_FATAL(!spec.kernels.empty(), "A ProgramSpec must have at least one KernelSpec");
}

}  // namespace tt::tt_metal::experimental
