// SPDX-License-Identifier: Apache-2.0
// PROBE F3: a "producer" that produces nothing.
//
// dfb::in_tiles is `borrowed_from` an L1-resident TensorParameter, so the data is ALREADY at the
// DFB's address before the program starts. But the validator demands >=1 PRODUCER for every DFB,
// so somebody has to hand out the credits. This kernel exists purely to say push_back().
//
// This is role-faking: the ProgramSpec claims a producer/consumer dataflow edge that does not
// exist. There is no "pre-filled" / "resident" DFB endpoint kind to declare instead.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer in(dfb::in_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        in.reserve_back(1);
        // No write. The bytes are already there.
        in.push_back(1);
    }
}
