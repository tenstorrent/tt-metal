// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_assert.h"

// The forced SrcA exponent is Blackhole-only (tt-llk experimental/llk_unpack_srca_forced_exp.h). The Compute API keeps
// srca_forced_exp_* on every arch, so their entry points reject them here.

inline void llk_unpack_srca_forced_exp_bfp2([[maybe_unused]] const std::uint32_t exp) { LLK_ASSERT(false, "srca_forced_exp is Blackhole-only; not supported on Quasar"); }

inline void llk_unpack_srca_forced_exp_off() { LLK_ASSERT(false, "srca_forced_exp is Blackhole-only; not supported on Quasar"); }
