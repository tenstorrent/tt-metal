// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define GENERATE_BCAST_SCALER 1
#define BLOCK_SIZE 1

#include "tt_metal/kernels/dataflow/reader_unary_8bank.cpp"
