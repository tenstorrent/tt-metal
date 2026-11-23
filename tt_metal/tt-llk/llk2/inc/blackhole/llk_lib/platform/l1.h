// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "llk_memory_checks.h"

namespace llk2::platform::l1
{

/**
 * @brief Convert an L1 byte address to the address the unpackers and packers take.
 * @note The result counts 16-byte words and is one word below the data: the engines add the tile header slot back.
 */
constexpr std::uint32_t word_address(const std::uint32_t byte_address)
{
    return L1_ADDRESS(byte_address);
}

/** @brief Return whether an unpacker or packer word address lies inside the L1 region kernels may use. */
inline bool is_valid_word_address(const std::uint32_t word_address)
{
    return ckernel::is_valid_L1_address(word_address);
}

/**
 * @brief Load element @p index of an array of @p T at an L1 byte address with a RISC-V load.
 * @note The load is volatile: another core or engine may have written the word.
 */
template <class T>
[[nodiscard]] inline __attribute__((always_inline)) T load(const std::uint32_t byte_address, const std::uint32_t index)
{
    static_assert(std::is_trivially_copyable_v<T>, "an L1 load reads a plain value");
    return reinterpret_cast<const volatile T*>(byte_address)[index];
}

} // namespace llk2::platform::l1
