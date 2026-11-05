// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Quasar TRISC scalar ISA, shared by the host JIT flags, the firmware target attributes and CMake.
// sfpi 7.93.0 gives tt-qsr32 zbc instead of zbb, so an explicit -march restores zbb.
#define QUASAR_TRISC_ISA_BASE "rv32im_zaamo_zba_zbb"
#define QUASAR_TRISC_ISA_TT_EXT "_xtttensixqsr_xttzbkb"
#define QUASAR_TRISC_ISA QUASAR_TRISC_ISA_BASE QUASAR_TRISC_ISA_TT_EXT
