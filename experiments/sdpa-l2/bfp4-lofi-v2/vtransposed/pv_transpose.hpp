// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/compute/experimental/matmul_custom.h"
#if DST_ACCUM_MODE
#error "V-transposed wrapper is currently BF16 destination only"
#endif
namespace ckernel {
// Logical V always occupies CB2 in this isolated driver. Tile-grid strides
// stay N-major; only the physical contents of each full tile are transposed.
ALWI void vtransposed_mm_init(uint32_t a, uint32_t b, bool transpose = false,
                             uint32_t ct = 1, uint32_t rt = 1, uint32_t kt = 1) {
    mm_no_mop_init_short(a, b, transpose || b == 2, ct, rt, kt);
}
ALWI void vtransposed_mm_reinit(uint32_t a, uint32_t b, bool transpose = false,
                               uint32_t ct = 1, uint32_t rt = 1, uint32_t kt = 1) {
    mm_no_mop_reinit_short(a, b, transpose || b == 2, ct, rt, kt);
}
}  // namespace ckernel
