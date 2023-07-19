#pragma once


#include "compute_kernel_api/common_globals.h"
#include "compute_kernel_api/common.h"
#ifdef TRISC_UNPACK
#include "llk_unpack_common.h"
#include "llk_unpack_AB.h"
#define MAIN unpack_main()
#else
#define UNPACK(x)
#endif
