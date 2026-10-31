// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ckernel
{

/**
 * @brief Selects which instruction sequence an SFPU kernel runs.
 *
 * Two independent axes packed into one value:
 * - bit 0, precision: Precise runs the accurate algorithm; Approximate trades accuracy for throughput
 *   (fewer series terms or refinement steps, a LUT, or a hardware approximation instruction).
 * - bit 1, special values: Strict meets the op's documented contract for NaN, +-inf, +-0, out-of-domain
 *   and overflow/underflow inputs; Relaxed lets the caller vouch that inputs lie in the interior of the
 *   domain and leaves boundary results unspecified.
 *
 * Relaxed is a permission, not an obligation: a kernel may serve it with its Strict sequence when the guards
 * cost nothing. Each kernel documents which modes are distinct sequences and which alias another.
 *
 * @note The values are an ABI shared with tt::tt_metal::SfpuMode on the host (tt-metalium/base_types.hpp) and
 * with the JIT-generated kernel defines. Never renumber them.
 */
enum class SfpuMode : std::uint8_t
{
    PreciseStrict      = 0b00,
    ApproximateStrict  = 0b01,
    PreciseRelaxed     = 0b10,
    ApproximateRelaxed = 0b11,
};

/** @brief True when @p mode selects the approximate algorithm. */
constexpr bool is_approximate(const SfpuMode mode)
{
    return (static_cast<std::uint8_t>(mode) & 0b01) != 0;
}

/** @brief True when @p mode requires the op's special-value contract to hold. */
constexpr bool is_strict(const SfpuMode mode)
{
    return (static_cast<std::uint8_t>(mode) & 0b10) == 0;
}

/** @brief Builds a mode from its two axes; the bridge from the legacy APPROX / FAST_APPROX booleans. */
constexpr SfpuMode make_sfpu_mode(const bool approximate, const bool strict)
{
    return static_cast<SfpuMode>((approximate ? 0b01 : 0) | (strict ? 0 : 0b10));
}

} // namespace ckernel
