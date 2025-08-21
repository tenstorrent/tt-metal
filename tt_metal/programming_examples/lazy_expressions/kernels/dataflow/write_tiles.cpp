// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "views/accessor_view.h"

void kernel_main() {
    const auto n_tiles = get_arg_val<uint32_t>(0);
    const auto start_id = get_arg_val<uint32_t>(1);

    constexpr auto num_tiles_per_cycle = get_compile_time_arg_val(0);

    // 1 circular buffer with bank base address at runtime args offset 2
    const auto view = views::MakeAccessorView<1, num_tiles_per_cycle>(2);
    view.write_tiles(n_tiles, start_id);
}
