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
template <int api>
struct llk_pack_matmul_api;
template <int api>
struct llk_pack_sfpu_api;
template <int api>
struct llk_pack_norm_api;
template <int api>
struct llk_pack_lofi_api;
template <int api>
struct llk_pack_sdpa_api;
template <>
struct llk_pack_lofi_api<1> {
    static constexpr bool value = MATH_FIDELITY == MathFidelity::LoFi;
};
}  // namespace ckernel::packer
#define PACK_MATH_CONTEXT(matmul) ckernel::packer::llk_pack_math_context(matmul)
#define PACK_SFPU_CONTEXT() ckernel::packer::llk_pack_sfpu_context()
// A matmul API header marks the kernel: only such a kernel can be in a matmul context.
#define PACK_MATMUL_API(api)                \
    namespace ckernel::packer {             \
    template <>                             \
    struct llk_pack_matmul_api<api> {       \
        static constexpr bool value = true; \
    };                                      \
    }
// layernorm.h marks the norm kernels, which include matmul.h without being bound by matmul.
#define PACK_NORM_API()                     \
    namespace ckernel::packer {             \
    template <>                             \
    struct llk_pack_norm_api<1> {           \
        static constexpr bool value = true; \
    };                                      \
    }
// So does a header of a pack-thread SFPU init: only such a kernel can reach the pack-thread SFPU context.
#define PACK_SFPU_API(api)                  \
    namespace ckernel::packer {             \
    template <>                             \
    struct llk_pack_sfpu_api<api> {         \
        static constexpr bool value = true; \
    };                                      \
    }
// The SDPA SFPU header marks the SDPA kernels.
#define PACK_SDPA_API()                     \
    namespace ckernel::packer {             \
    template <>                             \
    struct llk_pack_sdpa_api<1> {           \
        static constexpr bool value = true; \
    };                                      \
    }
#else
#define PACK_MATH_CONTEXT(matmul)
#define PACK_SFPU_CONTEXT()
#define PACK_MATMUL_API(api)
#define PACK_SFPU_API(api)
#define PACK_NORM_API()
#define PACK_SDPA_API()
#endif
