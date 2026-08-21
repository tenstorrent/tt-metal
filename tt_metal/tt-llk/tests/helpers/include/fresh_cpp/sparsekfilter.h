// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `sparsekfilter-fresh` coverage row
// (legacy tt-llk experimental ckernel_sfpu_sparse_k_filter.h
// _sparse_k_filter_tile_, corpus manifest class D-ABSENT — zero dispatch
// anywhere).  Mathematical definition (bank-addressed index filter): each
// lane holds an int32 index whose 6-bit global-bank field sits at
// GLOBAL_BANK_SHIFT and whose low bits are the within-bank slot;
//   y = (bank(x) == MY_BANK) ? ((x & WITHIN_BANK_MASK) + 1) << OUT_SHIFT : 0
// The fresh statement extracts the bank by shift+mask (value-identical to
// the production's in-place field compare).  Exact integer contract.
#include <cstdint>

namespace ckernel::sfpu
{

template <
    std::uint32_t BANK_MASK,
    std::uint32_t MY_BANK,
    std::uint32_t GLOBAL_BANK_SHIFT,
    std::uint32_t WITHIN_BANK_MASK,
    std::uint32_t OUT_SHIFT,
    int ITERATIONS>
__attribute__((noinline)) void calculate_sparse_k_filter_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vInt idx  = sfpi::dst_reg[0];
        const sfpi::vInt bank = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(idx) >> static_cast<int>(GLOBAL_BANK_SHIFT)) & static_cast<int>(BANK_MASK);
        sfpi::vInt enc        = 0;
        v_if (bank == static_cast<int>(MY_BANK))
        {
            const sfpi::vInt local = idx & static_cast<int>(WITHIN_BANK_MASK);
            enc                    = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(local + 1) << static_cast<int>(OUT_SHIFT));
        }
        v_endif;
        sfpi::dst_reg[0] = enc;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
