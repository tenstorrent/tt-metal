// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>

namespace tt::jit_build {

// Relative to the tt-metal root; installed alongside the other JIT inputs.
inline constexpr std::string_view PCH_UMBRELLA = "tt_metal/hw/firmware/src/pch.h";

// Build, once per process and per compiler flag set, the precompiled header shared by every
// JIT target (hw/firmware/src/pch.h) and return the path of the staged copy to force-include
// with -include. The .gch sits next to that copy, which is where GCC looks for it.
//
// |cflags| and |opt_level| must be what the consuming compile uses, since GCC rejects a PCH
// built with a different target (-march, from -mcpu), debug level, or -mbranch-cost. The last
// of those is set from the optimisation level, which is why that is part of the key: -Os and
// -O2 disagree on it, and a kernel picks its own level (KernelBuildOptLevel) independently of
// the build state's default, so both really do occur against one set of cflags.
//
// Returns an empty string if the PCH could not be built or staged. Callers then simply omit
// the -include and the compile parses the headers as it did before; the failure is logged once
// per flag set.
std::string ensure_pch(
    const std::string& gpp,
    const std::string& opt_level,
    const std::string& cflags,
    const std::string& includes,
    const std::string& root,
    const std::string& pch_root);

}  // namespace tt::jit_build
