// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/blaze/named_kernel_args.hpp>

namespace tt::tt_metal::experimental::blaze {

// Reject a cache hit whose merged per-core or common counts differ, including empty and omitted cores.
void validate_cached_runtime_arg_counts(
    Program& program, const KernelDescriptor& kernel_descriptor, uint32_t kernel_index);

}  // namespace tt::tt_metal::experimental::blaze
