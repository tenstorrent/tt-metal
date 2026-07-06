// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel
{

/**
 * @brief Compiler-only barrier: prevents reordering of memory accesses across this point.
 * @note Does not enforce CPU or system memory ordering by itself.
 */
inline void fence_compiler()
{
    asm volatile("" ::: "memory");
}

} // namespace ckernel
