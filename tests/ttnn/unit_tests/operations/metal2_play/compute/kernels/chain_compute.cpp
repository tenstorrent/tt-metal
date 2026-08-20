// SPDX-License-Identifier: Apache-2.0
// PROBE B: the eltwise_chain convenience API driven from dfb:: tokens. Here the token has to
// convert inside a *constexpr function call* (`input(...)` / `output(...)`) whose result is itself
// a class-type non-type template argument. Two levels of constexpr indirection.
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

void kernel_main() {
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);

    compute_kernel_hw_startup(dfb::in_tiles, dfb::out_tiles);

    using namespace compute_kernel_lib;
    square<input(dfb::in_tiles), output(dfb::out_tiles)>(IterationShape::grid(Ht, Wt).block_size(1));
}
