// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Compute of PagedUpdateRows: the stock paged_update_cache compute (update_cache.cpp) per unit: the source tile
// rows untilized to bf16 row-major (the stock's input untilize), then per job the cache tile row untilized to
// CB_UNT_A, the writer's patched copy (CB_UNT_B, the same L1 region) tilized back to the cache format. The same
// kernel_lib helpers with the same template arguments as the stock, so the bits of every row (patched or not)
// match the stock's. The loop counts are runtime arguments (no data-dependent count reaches the math core).
// runtime args: [n_units]
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "paged_update_rows_common.hpp"

void kernel_main() {
    using namespace pur;
    const uint32_t n_units = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(CB_SRC, CB_UNT_SRC);
    for (uint32_t u = 0; u < n_units; ++u) {
        compute_kernel_lib::untilize<Wt, CB_SRC, CB_UNT_SRC>(N_SRC);
        for (uint32_t j = 0; j < J; ++j) {
            compute_kernel_lib::untilize<Wt, CB_CACHE, CB_UNT_A>(1);
            compute_kernel_lib::tilize<Wt, CB_UNT_B, CB_OUT>(1);
        }
    }
}
