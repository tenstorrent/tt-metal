// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define ALWI inline __attribute__((always_inline))

#include "chlkc_list.h"
#include "ckernel.h"
#include "internal/firmware_common.h"
#include "ckernel_include.h"
#include "hostdevcommon/kernel_structs.h"

#ifdef TRISC_MATH
#include "llk_math_common_api.h"
#define MATH(...) __VA_ARGS__
#define MAIN math_main()
#else
#define MATH(...)
#endif

#ifdef TRISC_PACK
#define PACK(...) __VA_ARGS__
#define MAIN pack_main()
#else
#define PACK(...)
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_common_api.h"
#define UNPACK(...) __VA_ARGS__
#define MAIN unpack_main()
#else
#define UNPACK(...)
#endif
