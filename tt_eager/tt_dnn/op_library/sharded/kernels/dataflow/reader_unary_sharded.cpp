// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/reader_unary_sharded.hpp"

void kernel_main() {
    const uint32_t num_units = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);

    reader_unary_sharded(num_units, cb_id_in0);
}
