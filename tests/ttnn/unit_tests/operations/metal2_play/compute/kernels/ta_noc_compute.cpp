// SPDX-License-Identifier: Apache-2.0
// PROBE F1 (expected to FAIL): a COMPUTE kernel that tries to build a full NoC TensorAccessor
// from a tensor binding. The host validator accepts the binding; this is a device-build failure.
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const auto acc = TensorAccessor(tensor::scale);  // <-- should not compile on TRISC
    (void)acc;
    (void)num_tiles;
}
