#pragma once

#define SYNC SyncHalf

#if __DOXYGEN__
    #define ALWI
#else
    #define ALWI inline __attribute__((always_inline))
#endif

#include "chlkc_list.h"
#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "hostdevcommon/kernel_structs.h"
