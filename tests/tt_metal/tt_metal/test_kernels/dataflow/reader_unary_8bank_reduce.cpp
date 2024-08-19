// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#define GENERATE_BCAST_SCALER 1
#define BLOCK_SIZE 1

#include "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_8bank.cpp"
