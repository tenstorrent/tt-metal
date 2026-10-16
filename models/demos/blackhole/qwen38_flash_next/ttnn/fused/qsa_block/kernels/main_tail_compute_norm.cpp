// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail, norm core compute (fp32 dest): rms_norm of one 256-wide tile row (a query head, or the key).

#include "rms_norm_mirror.h"
#include "main_tail_cbs.h"
#include "../../kernels/zones.h"

using namespace main_tail;

void kernel_main() {
    FUSED_ZONE("fz_qs_mt_cn_main");
    compute_kernel_hw_startup(CB_X, CB_X, CB_XMM2);
    rms_norm_rows<HEAD_TILES, 256, CB_X, CB_SCALER, CB_EPS, CB_GAMMA, CB_XMM2, CB_EX2, CB_EX2PE, CB_FUSION, CB_N>(1);
}
