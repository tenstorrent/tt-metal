// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Writer for GegluShardedDecode: the output is a tensor-backed CB (this core's shard of the down-proj activation)
// written by the packer; wait for all `count` tiles so the program ends only once the shard is complete.
// compile-time args: [cb_out]; runtime args: [count]
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    CircularBuffer c(cb_out);
    c.wait_front(count);
    c.pop_front(count);
}
