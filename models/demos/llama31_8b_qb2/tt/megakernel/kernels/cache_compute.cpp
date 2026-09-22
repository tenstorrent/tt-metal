// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#ifndef QB2_ENTRY
#define QB2_ENTRY kernel_main
#endif
#include "experimental/kernel_args.h"
namespace args {
constexpr experimental::CtaVal<uint32_t> Wt{4}, num_heads{2}, has_work{1};
}
namespace dfb {
constexpr uint32_t in=1, cache=0, untilized_in=16, untilized_cache=24, untilized_cache2=25, out=8;
}
#define kernel_main native_cache_compute_main
#include "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/compute/paged_fused_update_cache.cpp"
#undef kernel_main
#include "api/compute/cb_api.h"

void QB2_ENTRY() {
    cb_wait_front(30, 1);
    const uint32_t position = read_tile_value(30, 0, 0);
    cb_pop_front(30, 1);
    if (position == UINT32_MAX) { return; }
    native_cache_compute_main();
}
