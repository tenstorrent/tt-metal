// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "views/compute_view.h"
#include "views/transform_view.h"
#include "views/tile_views.h"

namespace NAMESPACE {
void MAIN {
    // rt args are:
    // - n_tiles
    // - any scalar params used in expression tree
    const auto n_tiles = get_arg_val<uint32_t>(0);

    using namespace tt;
    using namespace views;

    // ct args are:
    // - num_tiles_per_cycle
    // - cb_indices for input and internal nodes in postfix order
    // - cb_index for output tensor
    constexpr auto num_tiles_per_cycle = get_compile_time_arg_val(0);

    // 5 circular buffers
    using View = MakeComputeView<5>;

    View::init_tiles(init_sfpu<c_0, c_4>());

    // fused addcmul
    // c_0 + ((c_1 * value) * c_3) -> c_4
    View::compute_tiles<num_tiles_per_cycle>(
        n_tiles,
        // c_1 * value -> c_2
        with_cb_ids<c_1, c_2>(mul_unary(get_arg_val<uint32_t>(1))),
        // c_2 * c_3 -> c_2
        with_cb_ids<c_2, c_3, c_2>(mul()),
        // c_0 + c_2 -> c_4
        with_cb_ids<c_0, c_2, c_4>(add()));
}
}  // namespace NAMESPACE
