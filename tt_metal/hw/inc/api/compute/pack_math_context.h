// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Blackhole llk_pack_block picks its masks by whether the last FPU op init was a matmul init, and whether a pack-thread
// SFPU init ran.
#if defined(TRISC_PACK) && defined(ARCH_BLACKHOLE)
namespace ckernel::packer {
inline void llk_pack_math_context(const std::uint8_t context);
inline void llk_pack_sfpu_context();
}  // namespace ckernel::packer
#define PACK_MATH_CONTEXT(matmul) ckernel::packer::llk_pack_math_context(matmul)
#define PACK_SFPU_CONTEXT() ckernel::packer::llk_pack_sfpu_context()
#else
#define PACK_MATH_CONTEXT(matmul)
#define PACK_SFPU_CONTEXT()
#endif
