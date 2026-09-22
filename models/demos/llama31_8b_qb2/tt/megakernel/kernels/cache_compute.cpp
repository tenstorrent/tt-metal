// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#ifndef QB2_ENTRY
#define QB2_ENTRY kernel_main
#else
#define kernel_main QB2_ENTRY
#endif
#include "experimental/kernel_args.h"
namespace args {
constexpr experimental::CtaVal<uint32_t> Wt{4}, num_heads{2}, has_work{1};
}
namespace dfb {
constexpr uint32_t in=1, cache=0, untilized_in=16, untilized_cache=24, untilized_cache2=25, out=8;
}
#include "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/compute/paged_fused_update_cache.cpp"

#undef kernel_main
