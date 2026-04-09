// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#ifdef ARCH_QUASAR
#include "sfpu/ckernel_sfpu_lrelu.h"
#endif
namespace ckernel {
namespace sfpu {}
}  // namespace ckernel
