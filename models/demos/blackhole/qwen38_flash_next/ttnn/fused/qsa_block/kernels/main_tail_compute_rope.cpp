// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail, rope core compute (16-bit dest, the op's own): RoPE of the normalized row's first two tiles.

#include "rope_mirror.h"
#include "main_tail_cbs.h"
#include "../../kernels/zones.h"

using namespace main_tail;

void kernel_main() {
    FUSED_ZONE("fz_qs_mt_cr_main");
    CircularBuffer(CB_SCALAR).wait_front(1);
    compute_kernel_hw_startup(CB_ROT, CB_SCALAR, CB_ROT_NEG);
    rope64_rows<CB_IN, CB_ROT, CB_COS, CB_SIN, CB_SCALAR, CB_ROT_NEG, CB_XCOS, CB_RSIN, CB_OUT>(1);
}
