// Fused MLP v2: compile-time layout shared by the three RISCs.
#pragma once
#include <stdint.h>
constexpr uint32_t CB_X = 0, CB_W = 1, CB_HL = 2, CB_H = 3, CB_WD = 4, CB_OUT = 16, CB_W2 = 5;
constexpr uint32_t SEM_XCNT = 0, SEM_XFLAG = 1, SEM_HCNT = 2, SEM_HFLAG = 3;
