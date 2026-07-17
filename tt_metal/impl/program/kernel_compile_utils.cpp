// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kernel_compile_utils.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

#include "common/stable_hash.hpp"
#include "jit_build/hlk_desc.hpp"
#include "jit_build/jit_build_options.hpp"
#include "kernels/kernel.hpp"

#ifdef GENERATE_HASH_LOG
#include <fstream>
#include <mutex>
#endif

namespace tt::tt_metal::detail {

std::size_t KernelCompileHash(
    const std::shared_ptr<Kernel>& kernel, JitBuildOptions& build_options, std::uint64_t build_key) {
    // Store the build key into the KernelCompile hash. This will be unique per command queue
    // configuration (necessary for dispatch kernels).
    // watcher/dprint enabled are accounted for in the build key.
    tt::StableHasher hasher;
    hasher.update(build_key);
    hasher.update(stable_hash_hlk_desc(build_options.hlk_desc));
    hasher.update(kernel->compute_hash());
    std::size_t compile_hash = static_cast<std::size_t>(hasher.digest());

#ifdef GENERATE_HASH_LOG
    static std::ofstream f("/tmp/hashlog.txt");
    static std::mutex mutex_;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        f << kernel->name() << " :: " << build_key << "::" << stable_hash_hlk_desc(build_options.hlk_desc)
          << " :: " << kernel->compute_hash() << " :: " << compile_hash << std::endl
          << std::flush;
    }
#endif
    return compile_hash;
}

}  // namespace tt::tt_metal::detail
