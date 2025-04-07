// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ckernel.h"
#include "compute_kernel_api.h"
#include <sfpi.h>

using namespace sfpi;
namespace NAMESPACE {
void MAIN {
#if COMPILE_FOR_TRISC == 1  // compute
#include "pre.inc"

    {  // Test -1 is all ones
        vInt minusOne = -1;

        vUInt allOnes = vUInt(0xffff) | vUInt(0xffff) << 16;
        vUInt not2sComp = minusOne ^ allOnes;
        // not2sComp should be all bits zero
        FAIL_IF(not2sComp != 0);

        vUInt signOne = vUInt(1) | vUInt(0x8000) << 16;
        vUInt notSignMag = minusOne ^ signOne;
        FAIL_IF(notSignMag == 0);
    }

    {  // test loading 0x80000001 loads as expected
        vUInt value = vUInt(1) | vUInt(0x8000) << 16;

        vUInt signOne = setsgn(vUInt(1), 1);

        vUInt notCorrect = value ^ signOne;
        FAIL_IF(notCorrect != 0);
    }

    {  // test 0 - 1 in registers results in all ones.
        vInt zero = 0;
        vInt one = 1;
        vInt sub = zero - one;

        vUInt allOnes = vUInt(0xffff) | vUInt(0xffff) << 16;
        vUInt not2sComp = sub ^ allOnes;
        // not2sComp should be all bits zero
        FAIL_IF(not2sComp != 0);

        vUInt signOne = vUInt(1) | vUInt(0x8000) << 16;
        vUInt notSignMag = sub ^ signOne;
        // notSignMag will be zero, if sign-mag
        FAIL_IF(notSignMag == 0);
    }

    {  // test 0 + -1 in registers results in all ones.
        vInt zero = 0;
        vInt minusOne = -1;
        vInt sub = zero + minusOne;

        vUInt allOnes = vUInt(0xffff) | vUInt(0xffff) << 16;
        vUInt not2sComp = sub ^ allOnes;
        // not2sComp should be all bits zero
        FAIL_IF(not2sComp != 0);

        vUInt signOne = vUInt(1) | vUInt(0x8000) << 16;
        vUInt notSignMag = sub ^ signOne;
        // notSignMag will be zero, if sign-mag
        FAIL_IF(notSignMag == 0);
    }

    {  // test 0 - 1 as cst results in all ones.
        vInt zero = 0;
        vInt sub = zero - 1;

        vUInt allOnes = vUInt(0xffff) | vUInt(0xffff) << 16;
        vUInt not2sComp = sub ^ allOnes;
        // not2sComp should be all bits zero
        FAIL_IF(not2sComp != 0);

        vUInt signOne = vUInt(1) | vUInt(0x8000) << 16;
        vUInt notSignMag = sub ^ signOne;
        // notSignMag will be zero, if sign-mag
        FAIL_IF(notSignMag == 0);
    }

    {  // test 0 + -1 as cst results in all ones.
        vInt zero = 0;
        vInt sub = zero + -1;

        vUInt allOnes = vUInt(0xffff) | vUInt(0xffff) << 16;
        vUInt not2sComp = sub ^ allOnes;
        // not2sComp should be all bits zero
        FAIL_IF(not2sComp != 0);

        vUInt signOne = vUInt(1) | vUInt(0x8000) << 16;
        vUInt notSignMag = sub ^ signOne;
        // notSignMag will be zero, if sign-mag
        FAIL_IF(notSignMag == 0);
    }

#if __riscv_tt_blackhole
    {  // test -2 >> 1 is -1
        vInt minusTwo = -2;
        vInt shft = minusTwo >> 1;

        vUInt allOnes = vUInt(0xffff) | vUInt(0xffff) << 16;
        vUInt not2sComp = shft ^ allOnes;
        // not2sComp should be all bits zero
        FAIL_IF(not2sComp != 0);

        vUInt signOne = vUInt(1) | vUInt(0x8000) << 16;
        vUInt notSignMag = shft ^ signOne;
        // notSignMag will be zero, if sign-mag
        FAIL_IF(notSignMag == 0);
    }
#endif

    {  // test unsigned(-2) >> 1 is mostPos
        vUInt minusTwo = -2;
        vUInt shft = minusTwo >> 1;

        vUInt mostPos = vUInt(0xffff) | vUInt(0x7fff) << 16;
        vUInt not2sComp = vInt(shft) ^ mostPos;
        // not2sComp should be all bits zero
        FAIL_IF(not2sComp != 0);

        vUInt smExpected = vUInt(1) | vUInt(0x4000) << 16;
        vUInt notSignMag = shft ^ smExpected;
        // notSignMag will be zero, if sign-mag
        FAIL_IF(notSignMag == 0);
    }

#include "post.inc"
#endif
}
}  // namespace NAMESPACE
