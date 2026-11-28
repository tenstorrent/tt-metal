// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "experimental/llk_unpack_srca_forced_exp.h"

// Unpacker A reads mantissa-only BFP2_b tiles with a forced exponent (experimental/llk_unpack_srca_forced_exp.h).

inline void llk_unpack_srca_forced_exp_bfp2(const std::uint32_t exp) { _llk_unpack_srca_forced_exp_bfp2_(exp); }

inline void llk_unpack_srca_forced_exp_off() { _llk_unpack_srca_forced_exp_off_(); }
