// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "lltt.h"

using namespace sfpi;

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE /*unused*/, int ITERATIONS = 8 /*unused*/>
inline void calculate_cumsum(const bool first) {
    if (first) {
        // Clear context for F0
        TTI_SFPMOV(0, 9, 4, 0);
        TTI_SFPMOV(0, 9, 5, 0);
        TTI_SFPMOV(0, 9, 6, 0);
        TTI_SFPMOV(0, 9, 7, 0);
    }

    // F0,1 R0
    TTI_SFPLOAD(0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
    TTI_SFPLOAD(1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 2);
    TTI_SFPLOAD(2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0 + 16);
    TTI_SFPLOAD(3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 2 + 16);

    TTI_SFPTRANSP(0, 0, 0, 0);
    lltt::replay(0, 8);
    TTI_SFPTRANSP(0, 0, 0, 0);

    TTI_SFPSTORE(0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
    TTI_SFPSTORE(1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 2);
    TTI_SFPSTORE(2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0 + 16);
    TTI_SFPSTORE(3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 2 + 16);

    // F0,1 R4
    TTI_SFPLOAD(4, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 4);
    TTI_SFPLOAD(5, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 6);
    TTI_SFPLOAD(6, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 4 + 16);
    TTI_SFPLOAD(7, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 6 + 16);

    TTI_SFPTRANSP(0, 0, 0, 0);
    lltt::replay(8, 8);
    TTI_SFPTRANSP(0, 0, 0, 0);

    TTI_SFPSTORE(4, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 4);
    TTI_SFPSTORE(5, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 6);
    TTI_SFPSTORE(6, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 4 + 16);
    TTI_SFPSTORE(7, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 6 + 16);

    // F0,1 R8
    TTI_SFPLOAD(0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 8);
    TTI_SFPLOAD(1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 10);
    TTI_SFPLOAD(2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 8 + 16);
    TTI_SFPLOAD(3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 10 + 16);

    TTI_SFPTRANSP(0, 0, 0, 0);
    lltt::replay(0, 8);
    TTI_SFPTRANSP(0, 0, 0, 0);

    TTI_SFPSTORE(0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 8);
    TTI_SFPSTORE(1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 10);
    TTI_SFPSTORE(2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 8 + 16);
    TTI_SFPSTORE(3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 10 + 16);

    // F0,1 R12
    TTI_SFPLOAD(4, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 12);
    TTI_SFPLOAD(5, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 14);
    TTI_SFPLOAD(6, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 12 + 16);
    TTI_SFPLOAD(7, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 14 + 16);

    TTI_SFPTRANSP(0, 0, 0, 0);
    lltt::replay(8, 8);
    TTI_SFPTRANSP(0, 0, 0, 0);

    TTI_SFPSTORE(4, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 12);
    TTI_SFPSTORE(5, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 14);
    TTI_SFPSTORE(6, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 12 + 16);
    TTI_SFPSTORE(7, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 14 + 16);

    // F2,3 R0
    TTI_SFPLOAD(0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0 + 32);
    TTI_SFPLOAD(1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 2 + 32);
    TTI_SFPLOAD(2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0 + 16 + 32);
    TTI_SFPLOAD(3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 2 + 16 + 32);

    TTI_SFPTRANSP(0, 0, 0, 0);
    lltt::replay(0, 8);
    TTI_SFPTRANSP(0, 0, 0, 0);

    TTI_SFPSTORE(0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0 + 32);
    TTI_SFPSTORE(1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 2 + 32);
    TTI_SFPSTORE(2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0 + 16 + 32);
    TTI_SFPSTORE(3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 2 + 16 + 32);

    // F2,3 R4
    TTI_SFPLOAD(4, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 4 + 32);
    TTI_SFPLOAD(5, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 6 + 32);
    TTI_SFPLOAD(6, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 4 + 16 + 32);
    TTI_SFPLOAD(7, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 6 + 16 + 32);

    TTI_SFPTRANSP(0, 0, 0, 0);
    lltt::replay(8, 8);
    TTI_SFPTRANSP(0, 0, 0, 0);

    TTI_SFPSTORE(4, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 4 + 32);
    TTI_SFPSTORE(5, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 6 + 32);
    TTI_SFPSTORE(6, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 4 + 16 + 32);
    TTI_SFPSTORE(7, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 6 + 16 + 32);

    // F2,3 R8
    TTI_SFPLOAD(0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 8 + 32);
    TTI_SFPLOAD(1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 10 + 32);
    TTI_SFPLOAD(2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 8 + 16 + 32);
    TTI_SFPLOAD(3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 10 + 16 + 32);

    TTI_SFPTRANSP(0, 0, 0, 0);
    lltt::replay(0, 8);
    TTI_SFPTRANSP(0, 0, 0, 0);

    TTI_SFPSTORE(0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 8 + 32);
    TTI_SFPSTORE(1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 10 + 32);
    TTI_SFPSTORE(2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 8 + 16 + 32);
    TTI_SFPSTORE(3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 10 + 16 + 32);

    // F2,3 R12
    TTI_SFPLOAD(4, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 12 + 32);
    TTI_SFPLOAD(5, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 14 + 32);
    TTI_SFPLOAD(6, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 12 + 16 + 32);
    TTI_SFPLOAD(7, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 14 + 16 + 32);

    TTI_SFPTRANSP(0, 0, 0, 0);
    lltt::replay(8, 8);
    TTI_SFPTRANSP(0, 0, 0, 0);

    TTI_SFPSTORE(4, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 12 + 32);
    TTI_SFPSTORE(5, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 14 + 32);
    TTI_SFPSTORE(6, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 12 + 16 + 32);
    TTI_SFPSTORE(7, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 14 + 16 + 32);
}

template <bool APPROXIMATION_MODE /*unused*/>
inline void cumsum_init() {
    lltt::record(0, 16);
    // FIXME: These should all be TT_SFP...
    TTI_SFPADD(10, 7, 0, 0, 0);
    TTI_SFPNOP;
    TTI_SFPADD(10, 0, 1, 1, 0);
    TTI_SFPNOP;
    TTI_SFPADD(10, 1, 2, 2, 0);
    TTI_SFPNOP;
    TTI_SFPADD(10, 2, 3, 3, 0);
    TTI_SFPNOP;
    TTI_SFPADD(10, 3, 4, 4, 0);
    TTI_SFPNOP;
    TTI_SFPADD(10, 4, 5, 5, 0);
    TTI_SFPNOP;
    TTI_SFPADD(10, 5, 6, 6, 0);
    TTI_SFPNOP;
    TTI_SFPADD(10, 6, 7, 7, 0);
    TTI_SFPNOP;
}

}  // namespace ckernel::sfpu
