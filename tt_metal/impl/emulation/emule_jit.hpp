// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
// JIT compile + on-disk/in-memory cache subsystem (extracted from emulated_program_runner.cpp).
// collect_kernels consults g_jit_cache / disk_cache_lookup and builds DeferredCompile directly,
// so those are exposed; the rest of the compile machinery is file-local to the .cpp.

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "emule_metal2_emit.hpp"
#include "emule_program_descriptor.hpp"  // tt_emule::SourceRef, Named{Ct,Rt}Namespaces

namespace tt::tt_metal::emule {

// FNV hex buffer size (16 hex digits + null) — shared with program_model's cache-key building.
constexpr size_t FNV_HEX_BUF_SIZE = 17;

struct DeferredCompile {
    std::string src_path;
    std::vector<uint32_t> compile_args;
    std::unordered_map<std::string, uint32_t> named_compile_args;
    ////////////////////////////////////////////////////////////
    // Blaze-only experimental named args
    // Removal is tracked by issue #50953
    tt_emule::NamedCtNamespaces named_ct_arg_namespaces;
    tt_emule::NamedRtNamespaces named_runtime_arg_namespaces;
    ////////////////////////////////////////////////////////////
    std::map<std::string, std::string> defines;
    std::string extra_inc;
    Metal2BindingsSnapshot bindings;
};

// In-memory compiled-kernel cache, shared with collect_kernels' cache-hit fast path.
extern std::mutex g_jit_cache_mutex;
extern std::unordered_map<std::string, std::function<void()>> g_jit_cache;

std::uint64_t fnv1a_hash(const std::string& s);
std::function<void()> disk_cache_lookup(const std::string& cache_key, const std::string& src_path);
std::string get_extra_include_flags();
std::string resolve_kernel_source_path(const tt_emule::SourceRef& src, std::vector<std::string>& inline_src_temps);
std::string resolve_emule_kernel_source_shadow(const std::string& src_path, uint32_t context_id);
void jit_compile_pending(
    std::map<std::string, DeferredCompile>& deferred_compiles,
    std::unordered_map<std::string, std::function<void()>>& resolved_fns,
    std::vector<std::string>& inline_src_temps);

}  // namespace tt::tt_metal::emule
