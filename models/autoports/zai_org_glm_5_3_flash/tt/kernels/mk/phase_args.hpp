// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase argument bases (megakernel composition): every phase kernel reads its compile-time args at PH_CT0 + i and its
// runtime args at ph_rt0 + i. A standalone program uses 0 / 0 (defined here); a combined megakernel source defines
// MK_PHASED and, per phase namespace, its own PH_CT0 / ph_rt0 before including the phase's kernel file.
#pragma once

#include <cstdint>

#if !defined(MK_PHASED)
constexpr uint32_t PH_CT0 = 0;
constexpr uint32_t ph_rt0 = 0;
#endif
