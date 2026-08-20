// SPDX-License-Identifier: Apache-2.0
// PROBE I: compute_kernel_lib::matmul_block driven from DataflowBuffer OBJECTS (Buf is a deduced
// TEMPLATE TYPE parameter, so buffer_compat.hpp's buf_id(DataflowBuffer) overload is the hook --
// a different mechanism from the reduce helper's uint32_t NTTPs).
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

void kernel_main() {
    DataflowBuffer in0(dfb::in0), in1(dfb::in1), out(dfb::out_tiles);

    matmul_block_init(dfb::in0, dfb::in1, 0, 1, 1, 1);

    using namespace compute_kernel_lib;
    // num_k_blocks == 1 -> pass out as the interm placeholder (per the helper's contract).
    matmul_block<>(in0, in1, out, out, MatmulBlockShape::of(1, 1, 1, 1, 1, 1));
}
