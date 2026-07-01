// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#include "impl/jit_server/types.hpp"

namespace tt::tt_metal::kernel_prewarm {

// Opt-in parallel kernel prewarm. Both flags default OFF, so an unset environment
// reproduces current behavior exactly.
//
//   TT_METAL_KERNEL_MANIFEST_WRITE=<path>
//     Capture mode. On a normal (warm) run, ProgramImpl::compile appends every
//     kernel's portable compile recipe (jit_server::CompileRequest: build_key,
//     kernel_name, gpp, per-target flags/defines/srcs, and generated-file
//     contents) to <path>. Run once to record the pipeline's full kernel set.
//
//   TT_METAL_KERNEL_PREWARM_MANIFEST=<path>
//     Prewarm mode. At device init (after firmware is linked, before the host
//     issues any op), a background batch (re)builds every recipe in <path> into
//     the on-disk JIT cache using the shared 64-wide jit_build pool. This runs
//     concurrently with the host-idle weight_load window, so the later op-by-op
//     MeshWorkload::compile calls become cache HITs.
//
// Why this is safe/idempotent: a kernel binary is a pure function of (generated
// source + defines + compile-time args + flags + build_key). Populating the disk
// cache is order-independent, and the .dephash gate means kernels whose deps are
// unchanged are skipped -- only edit-affected kernels actually recompile. The
// prewarm never touches the device.

// Capture manifest path (TT_METAL_KERNEL_MANIFEST_WRITE), or nullptr if disabled.
const char* manifest_write_path();

// Capture: append one kernel's compile request to the manifest. Thread-safe;
// called from jit_build pool threads during ProgramImpl::compile.
void append_manifest_entry(const jit_server::CompileRequest& request);

// Prewarm: if TT_METAL_KERNEL_PREWARM_MANIFEST is set, spawn (at most once per
// process) a background thread that batch-builds the manifest into the local JIT
// cache rooted at |out_kernel_root| for the current |build_key|. |firmware_root|
// is the build env's firmware_binary_root (the pre-compiled bundle when present):
// captured recipes store weakened firmware as a bare filename, and the prewarm
// link needs to resolve it back to a full path under that root. No-op when the
// flag is unset or a batch was already launched.
void maybe_launch_prewarm(
    const std::string& out_kernel_root, const std::string& firmware_root, std::uint64_t build_key);

// Prewarm: block until an in-flight prewarm batch finishes (no-op if none).
// Called at the top of ProgramImpl::compile so the op-by-op path observes a fully
// warm cache before issuing per-op compiles.
void wait_for_prewarm();

}  // namespace tt::tt_metal::kernel_prewarm
