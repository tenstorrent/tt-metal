// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_n_common_2_0.hpp"

void kernel_main() { reader_unary_push_n_common_2_0</*ManualTileReads=*/true>(); }
