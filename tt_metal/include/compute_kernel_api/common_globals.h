#pragma once

#define SYNC SyncHalf

#define ALWI inline __attribute__((always_inline))

#include "chlkc_list.h"
#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "hostdevcommon/kernel_structs.h"

#ifdef TRISC_MATH
#include "llk_math_common.h"
#define MATH(x) x
#define MAIN math_main()
#else
#define MATH(x)
#endif

#ifdef TRISC_PACK
#include "llk_pack_common.h"
#include "llk_pack.h"
#define PACK(x) x
#define MAIN pack_main()
#else
#define PACK(x)
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_common.h"
#define UNPACK(x) x
#define MAIN unpack_main()
#else
#define UNPACK(x)
#endif
