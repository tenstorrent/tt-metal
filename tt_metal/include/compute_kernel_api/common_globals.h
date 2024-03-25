// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define ALWI inline __attribute__((always_inline))

#include "chlkc_list.h"
#include "ckernel.h"
#include "firmware_common.h"
#include "ckernel_include.h"
#include "hostdevcommon/kernel_structs.h"

#ifdef TRISC_MATH
#include "llk_math_common_api.h"
#define MATH(x) x
#define MAIN math_main()
#else
#define MATH(x)
#endif

#ifdef TRISC_PACK
#include "llk_pack_api.h"
#define PACK(x) x
#define MAIN pack_main()
#else
#define PACK(x)
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_common_api.h"
#define UNPACK(x) x
#define MAIN unpack_main()
#else
#define UNPACK(x)
#endif
