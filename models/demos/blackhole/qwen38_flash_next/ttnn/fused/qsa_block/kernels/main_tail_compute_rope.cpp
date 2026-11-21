// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail, rope core compute (16-bit dest, the op's own): RoPE of the normalized row's first two tiles.

#include "rope_mirror.h"
#include "main_tail_cbs.h"

using namespace main_tail;

void kernel_main() {
    CircularBuffer(cb_scalar).wait_front(1);
    compute_kernel_hw_startup(cb_rot, cb_scalar, cb_rot_neg);
    rope64_rows<cb_in, cb_rot, cb_cos, cb_sin, cb_scalar, cb_rot_neg, cb_xcos, cb_rsin, cb_out>(1);
}
