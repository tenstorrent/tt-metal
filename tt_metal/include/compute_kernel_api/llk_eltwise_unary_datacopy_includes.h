#pragma once


#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif
