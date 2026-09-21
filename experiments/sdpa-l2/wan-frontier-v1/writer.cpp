// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#define kernel_main unmasked_writer_main
#include "experiments/sdpa-l2/bfp4-lofi-v2/fullchip/writer.cpp"
#undef kernel_main
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"

void kernel_main() {
    static_assert(SDPA_K_PARTIAL_COL > 0 && SDPA_K_PARTIAL_COL < 32);
    Noc noc;
    generate_lightweight_mask_tiles<SDPA_K_PARTIAL_COL, 0, 15, false>(noc);
    unmasked_writer_main();
}
