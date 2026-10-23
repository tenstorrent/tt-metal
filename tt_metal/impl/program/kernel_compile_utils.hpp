// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace tt::tt_metal {
class JitBuildOptions;
class Kernel;
}  // namespace tt::tt_metal

namespace tt::tt_metal::detail {

// Compute the kernel compile hash used by both JIT compile output paths and
// precompiled-kernel lookup. The offline kernel compiler must produce a
// bit-identical value, so both paths call this single implementation.
//
// Inputs:
// - build_key: device/build-key contribution (command queue, watcher, dprint, etc.)
// - stable_hash_hlk_desc(build_options.hlk_desc): CB format/tile contribution
// - kernel->compute_hash(): kernel source + config contribution
std::size_t KernelCompileHash(
    const std::shared_ptr<Kernel>& kernel, JitBuildOptions& build_options, std::uint64_t build_key);

}  // namespace tt::tt_metal::detail
