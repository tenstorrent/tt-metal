// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

namespace {

// clear LREG[0] except for row 3
inline void extract_last_row() {
    TT_SFPCONFIG(0x8000, SFPCONFIG_DEST_SFPU_CTRL, 3);  // disable row 3
    TT_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);   // LREG[0] = 0
    TT_SFPCONFIG(0x7FFF, SFPCONFIG_DEST_SFPU_CTRL, 5);  // enable row 3
}

// row 3 of LREG[0] is written with row 0 of LREG[0]
// other rows of LREG[0] are cleared
inline void extract_first_row() {
    TT_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    TT_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    TT_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
    TT_SFPTRANSP(0, 0, 0, 0);
    TT_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
    TT_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TT_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    TT_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    TT_SFPTRANSP(0, 0, 0, 0);
}

// shift LREG[0] by one row upward
inline void shift_row() {
    TT_SFPSHFT2(0, 0, 0, 1);
    TT_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);
}

// read 4 rows each from src_offset and dest_offset then broadcast-add the last row of src_offset to all rows of
// dest_offset the result is written to dest_offset
inline void broadcast_add_last_row(uint src_offset, uint dest_offset) {
    TT_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, src_offset);
    TT_SFPLOAD(p_sfpu::LREG4, 0, ADDR_MOD_7, dest_offset);
    extract_last_row();
    TT_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG4, 0);  // LREG[4] = LREG[0] * 1 + LREG[4]
    shift_row();
    TT_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG4, 0);
    shift_row();
    TT_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG4, 0);
    shift_row();
    TT_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG4, 0);
    TTI_SFPNOP;
    TT_SFPSTORE(p_sfpu::LREG4, 0, ADDR_MOD_7, dest_offset);
}

inline void broadcast_add_last_row_int(uint src_offset, uint dest_offset) {
    TT_SFPLOAD(p_sfpu::LREG0, 4, ADDR_MOD_7, src_offset);
    TT_SFPLOAD(p_sfpu::LREG4, 4, ADDR_MOD_7, dest_offset);
    extract_last_row();
    TT_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG4, 4);  // LREG[4] += LREG[0]
    shift_row();
    TT_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG4, 4);
    shift_row();
    TT_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG4, 4);
    shift_row();
    TT_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG4, 4);
    TT_SFPSTORE(p_sfpu::LREG4, 4, ADDR_MOD_7, dest_offset);
}

inline void fill_zero(uint offset, uint cnt) {
    for (int i = 0; i < 2; i++) {
        for (uint j = 0; j < 2 * cnt; j++) {
            TT_SFPSTORE(p_sfpu::LCONST_0, 0, ADDR_MOD_7, 16 * i + 2 * j + offset);
        }
    }
}

inline void fill_zero_partial(uint offset, uint rem) {
    int config_mask = ((1 << rem) - 1) << 12;
    TT_SFPCONFIG(config_mask, SFPCONFIG_DEST_SFPU_CTRL, 3);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            TT_SFPSTORE(p_sfpu::LCONST_0, 0, ADDR_MOD_7, 16 * i + 2 * j + offset);
        }
    }
    TT_SFPCONFIG((~config_mask) & 0xFFFF, SFPCONFIG_DEST_SFPU_CTRL, 5);
}

inline void clear_paddings(uint mask_h) {
    if (mask_h < 32) {
        int cnt = mask_h >= 16 ? (32 - mask_h) / 4 : 4;
        fill_zero(48 - 4 * cnt, cnt);
        if (mask_h >= 16 && mask_h % 4 != 0) {
            fill_zero_partial((mask_h - 16) / 4 * 4 + 32, mask_h % 4);
        }
    }
    if (mask_h < 16) {
        int cnt = (16 - mask_h) / 4;
        fill_zero(16 - 4 * cnt, cnt);
        if (mask_h % 4 != 0) {
            fill_zero_partial(mask_h / 4 * 4, mask_h % 4);
        }
    }
}

inline void broadcast_add_first_row(uint src_offset, uint dest_offset) {
    TT_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, src_offset);
    TT_SFPLOAD(p_sfpu::LREG4, 0, ADDR_MOD_7, dest_offset);
    extract_first_row();
    TT_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG4, 0);  // LREG[4] = LREG[0] * 1 + LREG[4]
    shift_row();
    TT_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG4, 0);
    shift_row();
    TT_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG4, 0);
    shift_row();
    TT_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG4, 0);
    TTI_SFPNOP;
    TT_SFPSTORE(p_sfpu::LREG4, 0, ADDR_MOD_7, dest_offset);
}

}  // namespace

template <bool APPROXIMATION_MODE>
inline void calculate_cumsum_row(uint first_tile, uint last_tile) {
    // calculate the cumulative sum of rows in a group of 4 rows
    for (int i = 0; i < 8; i++) {
        TT_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 8 * i);
        TT_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 8 * i + 2);
        TT_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, 8 * i + 4);
        TT_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, 8 * i + 6);
        TT_SFPTRANSP(0, 0, 0, 0);
        TT_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG1, 0);  // LREG[1] = LREG[0] * 1 + LREG[1]
        TTI_SFPNOP;
        TT_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG2, 0);  // LREG[2] = LREG[1] * 1 + LREG[2]
        TTI_SFPNOP;
        TT_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG3, 0);  // LREG[3] = LREG[2] * 1 + LREG[3]
        TTI_SFPNOP;
        TT_SFPTRANSP(0, 0, 0, 0);
        TT_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 8 * i);
        TT_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 8 * i + 2);
        TT_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, 8 * i + 4);
        TT_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_7, 8 * i + 6);
    }

    // left faces (i=0) and right faces (i=1)
    for (int i = 0; i < 2; i++) {
        // broadcast-add row 3 to row 4 - 7, row 7 to row 8 - 11, and row 11 to row 12 - 15
        for (int j = 0; j < 6; j++) {
            broadcast_add_last_row(16 * i + 2 * j, 16 * i + 2 * j + 4);
        }
        // broadcast-add row 15 to row 16 - 19
        for (int j = 0; j < 2; j++) {
            broadcast_add_last_row(16 * i + 2 * j + 12, 16 * i + 2 * j + 32);
        }
        // broadcast-add row 19 to row 20 - 23, row 23 to row 24 - 27, and row 27 to 28 - 31
        for (int j = 0; j < 6; j++) {
            broadcast_add_last_row(16 * i + 2 * j + 32, 16 * i + 2 * j + 36);
        }
    }

    // add accumulator
    if (!first_tile) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                TT_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 16 * i + 2 * j + 64);
                for (int k = 0; k < 4; k++) {
                    // order matters
                    TT_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 16 * i + 2 * j + 4 * k);
                    TT_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG1, 0);
                    TT_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, 16 * i + 2 * j + 4 * k + 32);
                    TT_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG2, 0);
                    TT_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 16 * i + 2 * j + 4 * k);
                    TT_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, 16 * i + 2 * j + 4 * k + 32);
                }
            }
        }
    }

    // set accumulator
    if (!last_tile) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                // read the bottom-most 4 rows into LREG[0]
                TT_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 16 * i + 2 * j + 44);
                TT_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
                TT_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
                TT_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
                TT_SFPTRANSP(0, 0, 0, 0);
                // all rows of LREG[3] are now filled with the former row 3 of LREG[0]
                // this is used as an accumulator for the next tile
                TT_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_7, 16 * i + 2 * j + 64);
            }
        }
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_cumsum_row_int(uint first_tile, uint last_tile) {
    for (int i = 0; i < 8; i++) {
        TT_SFPLOAD(p_sfpu::LREG0, 4, ADDR_MOD_7, 8 * i);
        TT_SFPLOAD(p_sfpu::LREG1, 4, ADDR_MOD_7, 8 * i + 2);
        TT_SFPLOAD(p_sfpu::LREG2, 4, ADDR_MOD_7, 8 * i + 4);
        TT_SFPLOAD(p_sfpu::LREG3, 4, ADDR_MOD_7, 8 * i + 6);
        TT_SFPTRANSP(0, 0, 0, 0);
        TT_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, 4);  // LREG[1] += LREG[0]
        TT_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG2, 4);  // LREG[2] += LREG[1]
        TT_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG3, 4);  // LREG[3] += LREG[2]
        TT_SFPTRANSP(0, 0, 0, 0);
        TT_SFPSTORE(p_sfpu::LREG0, 4, ADDR_MOD_7, 8 * i);
        TT_SFPSTORE(p_sfpu::LREG1, 4, ADDR_MOD_7, 8 * i + 2);
        TT_SFPSTORE(p_sfpu::LREG2, 4, ADDR_MOD_7, 8 * i + 4);
        TT_SFPSTORE(p_sfpu::LREG3, 4, ADDR_MOD_7, 8 * i + 6);
    }

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
            broadcast_add_last_row_int(16 * i + 2 * j, 16 * i + 2 * j + 4);
        }
        for (int j = 0; j < 2; j++) {
            broadcast_add_last_row_int(16 * i + 2 * j + 12, 16 * i + 2 * j + 32);
        }
        for (int j = 0; j < 6; j++) {
            broadcast_add_last_row_int(16 * i + 2 * j + 32, 16 * i + 2 * j + 36);
        }
    }

    if (!first_tile) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                TT_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 16 * i + 2 * j + 64);
                for (int k = 0; k < 4; k++) {
                    TT_SFPLOAD(p_sfpu::LREG1, 4, ADDR_MOD_7, 16 * i + 2 * j + 4 * k);
                    TT_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
                    TT_SFPLOAD(p_sfpu::LREG2, 4, ADDR_MOD_7, 16 * i + 2 * j + 4 * k + 32);
                    TT_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
                    TT_SFPSTORE(p_sfpu::LREG1, 4, ADDR_MOD_7, 16 * i + 2 * j + 4 * k);
                    TT_SFPSTORE(p_sfpu::LREG2, 4, ADDR_MOD_7, 16 * i + 2 * j + 4 * k + 32);
                }
            }
        }
    }

    if (!last_tile) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                TT_SFPLOAD(p_sfpu::LREG0, 4, ADDR_MOD_7, 16 * i + 2 * j + 44);
                TT_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
                TT_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
                TT_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
                TT_SFPTRANSP(0, 0, 0, 0);
                TT_SFPSTORE(p_sfpu::LREG3, 4, ADDR_MOD_7, 16 * i + 2 * j + 64);
            }
        }
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_cumsum_row_flip(uint mask_h, uint first_tile, uint last_tile) {
    clear_paddings(mask_h);

    // calculate the cumulative sum of rows in a group of 4 rows
    for (int i = 0; i < 8; i++) {
        TT_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 8 * i);
        TT_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 8 * i + 2);
        TT_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, 8 * i + 4);
        TT_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, 8 * i + 6);
        TT_SFPTRANSP(0, 0, 0, 0);
        TT_SFPADD(p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG2, 0);  // LREG[2] = LREG[3] * 1 + LREG[2]
        TTI_SFPNOP;
        TT_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG1, 0);  // LREG[1] = LREG[2] * 1 + LREG[1]
        TTI_SFPNOP;
        TT_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG0, 0);  // LREG[0] = LREG[1] * 1 + LREG[0]
        TTI_SFPNOP;
        TT_SFPTRANSP(0, 0, 0, 0);
        TT_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 8 * i);
        TT_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 8 * i + 2);
        TT_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, 8 * i + 4);
        TT_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_7, 8 * i + 6);
    }

    // left faces (i=0) and right faces (i=1)
    for (int i = 0; i < 2; i++) {
        // boradcast-add row 28 to row 24 - 27, row 24 to 20 - 23, and row 20 to 16 - 19
        for (int j = 5; j >= 0; j--) {
            broadcast_add_first_row(16 * i + 2 * j + 36, 16 * i + 2 * j + 32);
        }
        // broadcast-add row 16 to row 12 - 15
        for (int j = 1; j >= 0; j--) {
            broadcast_add_first_row(16 * i + 2 * j + 32, 16 * i + 2 * j + 12);
        }
        // broadcast-add row 12 to row 8 - 11, row 8 to row 4 - 7, and row 4 to row 0 - 3
        for (int j = 5; j >= 0; j--) {
            broadcast_add_first_row(16 * i + 2 * j + 4, 16 * i + 2 * j);
        }
    }

    // add accumulator
    if (!first_tile) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                TT_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 16 * i + 2 * j + 64);
                for (int k = 0; k < 4; k++) {
                    // order matters
                    TT_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 16 * i + 2 * j + 4 * k);
                    TT_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG1, 0);
                    TT_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, 16 * i + 2 * j + 4 * k + 32);
                    TT_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG2, 0);
                    TT_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 16 * i + 2 * j + 4 * k);
                    TT_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, 16 * i + 2 * j + 4 * k + 32);
                }
            }
        }
    }

    // set accumulator
    if (!last_tile) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                // read the top-most 4 rows into LREG[0]
                TT_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 16 * i + 2 * j);
                TT_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
                TT_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
                TT_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
                TT_SFPTRANSP(0, 0, 0, 0);
                // all rows of LREG[0] are now filled with the former row 0 of LREG[0]
                // this is used as an accumulator for the next tile
                TT_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 16 * i + 2 * j + 64);
            }
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
