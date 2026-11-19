// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <span>

#include "tensor_prefetcher.hpp"

#include "impl/buffers/tensor_prefetcher_request.hpp"

namespace tt::tt_metal {

std::span<const TensorPrefetcherKernelInfo> tensor_prefetcher_kernels() {
    // The reader stages weight blocks from DRAM into the dispatch engine's L1, keeping a group selector's mask
    // in its own state; the pusher sends them on to the receivers. Constant-initialized, so the topology's
    // static node tables may read it.
    static constexpr std::array<TensorPrefetcherKernelInfo, 2> kKernels = {{
        {.file = "tt_metal/impl/buffers/kernels/tensor_prefetcher_reader.cpp",
         .state_bytes = kTensorPrefetcherSelectorScratchBytes},
        {.file = "tt_metal/impl/buffers/kernels/tensor_prefetcher_pusher.cpp"},
    }};
    return kKernels;
}

}  // namespace tt::tt_metal
