// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal::experimental {

/**
 * Kernel define that compiles a compute kernel without the SFPI compiler's replay optimization.
 *
 * A compute kernel whose defines contain this key (any value) builds its three TRISC binaries with
 * -mno-tt-tensix-optimize-replay. The optimization assumes the replay buffer is free on entry to every
 * function it compiles; a kernel that records a replay sequence in one function and replays it from another
 * can have that recording overwritten by the compiler's own (tenstorrent/tt-metal#58433).
 *
 * Set it through ComputeConfig::defines or KernelDescriptor::defines. Defines are already part of the kernel's
 * JIT cache key and of LightMetal capture. Kernels without it build exactly as before.
 *
 * @note Experimental: a workaround for tenstorrent/tt-metal#58433, removed once the compiler respects
 *       recordings made outside the function it compiles.
 */
inline constexpr const char* DISABLE_SFPU_REPLAY_OPTIMIZATION_DEFINE = "TT_METAL_DISABLE_SFPU_REPLAY_OPTIMIZATION";

}  // namespace tt::tt_metal::experimental
