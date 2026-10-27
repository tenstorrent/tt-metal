// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail, norm core compute (fp32 dest): rms_norm of one 256-wide tile row (a query head, or the key).

#include "rms_norm_mirror.h"
#include "main_tail_cbs.h"

using namespace main_tail;

void kernel_main() {
    compute_kernel_hw_startup(cb_x, cb_x, cb_xmm2);
    rms_norm_rows<head_tiles, 256, cb_x, cb_scaler, cb_eps, cb_gamma, cb_xmm2, cb_ex2, cb_ex2pe, cb_fusion, cb_n>(1);
}
