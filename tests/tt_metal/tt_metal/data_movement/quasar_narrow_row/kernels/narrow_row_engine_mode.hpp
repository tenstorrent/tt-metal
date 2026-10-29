// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The `engine_mode` runtime-arg contract for the Quasar narrow-row pack-untilize (test id
// 918), shared by the host test and the stage-2 DM kernel.
//
// It lives in a header both sides include because it is a host/device contract carried in a
// single uint32 runtime arg, and nothing in the toolchain checks that the value the host
// packs means what the kernel reads. Two independent lists of constants agree right up until
// someone adds a third engine or reorders them -- and that divergence compiles cleanly on
// both sides, then shows up on device as the wrong branch taken, or as a hang.
//
// This is compiled by BOTH toolchains -- the host x86 build and the DM core's RoCC build
// (-mcpu=tt-qsr64-rocc) -- so it holds an enum and nothing else. It sits in kernels/ next to
// the kernel, like barrier_sync.hpp elsewhere in this suite, because a kernel's own source
// directory is the one include path the jit and emulator builds both guarantee
// (Kernel::process_include_paths); the host reaches down into kernels/ to get it.

#pragma once

#include <cstdint>

// Deliberately not nested under the test's own `quasar_narrow_row` namespace: the host test
// body is already inside a namespace of that name, where a qualified reference would resolve
// to the enclosing namespace and fail to compile.
namespace narrow_row {

// Which engine stage 2 uses to compact stage 1's padded rows into dense narrow rows. Both
// produce byte-identical output; EngineParity verifies that, which is what makes the timing
// comparison in README.md like-for-like. These values are the wire format of the
// "engine_mode" runtime arg, so they must stay stable.
enum class EngineMode : std::uint32_t {
    // One iDMA transaction per row, addresses from the address generator, fanned out over
    // num_channels backend VCs, one drain at the end. The engine being proposed.
    IdmaPerRow = 0,
    // One stateful NOC read per row. The current workaround, and the bar to beat.
    NocPerRow = 1,
};

// The prose above is not enforcement: pin the wire values so a reorder is a compile
// error on both sides rather than a wrong branch taken on device.
static_assert(static_cast<std::uint32_t>(EngineMode::IdmaPerRow) == 0u);
static_assert(static_cast<std::uint32_t>(EngineMode::NocPerRow) == 1u);

// The num_channels runtime arg's "use every backend VC the hardware has" value.
//
// The real count is overlay::CMDBUF_NUM_IDMA_VCS, and only the device side can see it:
// cmdbuff_api.hpp brings RoCC intrinsics and ~313 macros with it, which is not something to
// pull into the host test's unity-build translation unit just to read one integer. So the
// host asks for all of them and the kernel resolves the sentinel against the real constant.
// The VC count then lives in exactly one place instead of as a literal 8 on each side.
constexpr std::uint32_t CHANNELS_ALL = 0;

// Deliberately over the VC count, whatever that count is. The test uses it to reach the
// kernel's clamp; the kernel resolves it against the real overlay::CMDBUF_NUM_IDMA_VCS, which
// is the only place that number appears. Not the exact boundary (VC count + 1) -- naming that
// would mean mirroring a device-only constant here, which is the duplication this header
// exists to avoid.
constexpr std::uint32_t CHANNELS_OVER_RANGE = 64;

}  // namespace narrow_row
