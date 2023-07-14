#pragma once

#define SYNC SyncHalf

#if __DOXYGEN__
    #define ALWI
#else
    #define ALWI inline __attribute__((always_inline))
#endif

#ifdef TRISC_MATH
#include "llk_math_common.h"
#include "llk_math_matmul.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_reduce.h"
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
#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_A.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_reduce.h"
#include "llk_unpack_tilize.h"
#include "llk_unpack_untilize.h"
#define UNPACK(x) x
#define MAIN unpack_main()
#else
#define UNPACK(x)
#endif
