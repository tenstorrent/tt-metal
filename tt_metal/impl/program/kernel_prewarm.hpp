// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#include "impl/jit_server/types.hpp"

namespace tt::tt_metal::kernel_prewarm {

// Self-bootstrapping parallel kernel prewarm. ON by default so any checkout iterates fast on any box
// (galaxy or LB) with zero setup: the manifest lives at <cache_root>/kernel_prewarm.manifest (moves
// with TT_METAL_CACHE), a cold run records it, later runs reuse it. Set TT_METAL_KERNEL_PREWARM=0 to
// disable entirely (byte-for-byte the pre-prewarm behavior).
//
//   Capture. ProgramImpl::compile appends each newly-seen kernel's portable compile recipe
//     (jit_server::CompileRequest: build_key, kernel_name, gpp, per-target flags/defines/srcs,
//     generated-file contents) to the manifest. Dedup against the on-disk set means a cold cache
//     records the full pipeline and later runs add only new kernels (new op/shape/config).
//     Override the path with TT_METAL_KERNEL_MANIFEST_WRITE=<path>.
//
//   Prewarm. At device init (after firmware is linked, before the host issues any op), a background
//     batch (re)builds every manifest recipe for the current build_key into the on-disk JIT cache
//     using the shared 64-wide jit_build pool, concurrently with the host-idle weight_load window, so
//     the later op-by-op MeshWorkload::compile calls become cache HITs. Override the path with
//     TT_METAL_KERNEL_PREWARM_MANIFEST=<path>.
//
// Why this is safe/idempotent: a kernel binary is a pure function of (generated source + defines +
// compile-time args + flags + build_key). Populating the disk cache is order-independent, and the
// .dephash gate means kernels whose deps are unchanged are skipped -- only edit-affected kernels
// actually recompile. The prewarm never touches the device; output is byte-identical.

// Resolved capture-manifest path for this process, or nullptr if capture is disabled.
const char* manifest_write_path();

// True (and records it) iff (build_key, kernel_name) has not yet been captured this process and is not
// already on disk. Gates the append so warm runs pay no build_kernel_descriptor cost for known
// kernels. |kernel_name| is "<base>/<hash>" (matches CompileRequest::kernel_name).
bool capture_needed(std::uint64_t build_key, const std::string& kernel_name);

// Capture: append one kernel's compile request to the manifest. Thread-safe; called from jit_build
// pool threads during ProgramImpl::compile, gated by capture_needed().
void append_manifest_entry(const jit_server::CompileRequest& request);

// True iff capture-only mode is active. In this mode ProgramImpl::compile generates a kernel's
// genfiles and captures its manifest recipe but SKIPS the gcc compile/link, and EnqueueMeshWorkload
// skips dispatch. A pipeline run under this flag thus records the full manifest with only a brief
// device touch (device-init kernels still compile); the ~500s of model-kernel gcc then happens
// off-device via prewarm_manifest_offline before the real run. Capture is superset-safe (a missed
// kernel just compiles cold on the real run), so partial traversal degrades gracefully. Initial value
// is TT_METAL_KERNEL_CAPTURE_ONLY; set_capture_only() flips it at runtime so a single process can run
// a capture warmup, batch-compile, then a warm real run (the in-process cold-start).
bool capture_only();

// Flip capture_only() at runtime. Used by the in-process cold-start: enable for the capture warmup
// pass, disable before the batch compile and the real run.
void set_capture_only(bool enabled);

// True iff capture_only() AND |kernel_base_name| (Kernel::name()) is NOT a device-init (cq_/fabric)
// kernel. Device-init kernels always compile for real (the device can't come up otherwise); only
// model kernels are captured-not-compiled. Use this to gate the gcc skip in ProgramImpl::compile.
bool capture_only_skip_gcc(const std::string& kernel_base_name);

// Prewarm: unless TT_METAL_KERNEL_PREWARM=0, resolve the manifest path (from |out_kernel_root|'s
// cache root, or the env override), enable capture, and -- if the manifest already holds entries for
// |build_key| -- spawn (at most once per process) a background thread that batch-builds them into the
// local JIT cache rooted at |out_kernel_root|. |firmware_root| is the build env's firmware_binary_root
// (the pre-compiled bundle when present): captured recipes store weakened firmware as a bare filename,
// and the prewarm link needs to resolve it back to a full path under that root. On a cold cache it
// only arms capture (no batch); a batch launches once per process.
void maybe_launch_prewarm(
    const std::string& out_kernel_root, const std::string& firmware_root, std::uint64_t build_key);

// Off-device prewarm: compile every kernel recipe in the manifest into the JIT cache, for all
// build_keys the manifest holds, WITHOUT opening a device. |out_root| is JitBuildEnv::out_root_ (what
// TT_METAL_CACHE normalizes to, or the default cache root); the per-build_key kernel subtree
// <out_root><build_key>/kernels/ and the sibling manifest are derived from it exactly as the runtime
// does (string concat -- out_root may or may not carry a trailing '/'). |root_dir| is the TT-Metal
// root (RunTimeOptions::get_root_dir) used to locate the pre-compiled firmware bundle the link step
// needs; empty falls back to the jit firmware subtree. Unlike the in-run batch this also builds the
// device-init (cq_/fabric) kernels, so a subsequent real run holds the device only for device work,
// not compilation. Returns targets built. Run before a device/broker job.
std::size_t prewarm_manifest_offline(const std::string& out_root, const std::string& root_dir);

// True iff this process should run the in-process cold-start (capture warmup -> off-device batch
// compile -> real warmup) instead of compiling op-by-op inside the reserved window. That is: capture
// is armed (prewarm not globally disabled), no in-run batch was launched for this build_key (a cold
// cache, or a build_key the on-disk manifest does not cover), and we are not already inside an
// external capture-only pass. A warm build_key (batch launched) is served by the in-run prewarm; an
// externally forced capture_only pass captures the whole process itself, so both return false here.
// Query after device init, before the pipeline's warmup.
bool cold_start_needed();

// Prewarm: block until an in-flight prewarm batch finishes (no-op if none).
// Used to gate the op-by-op path so it observes a fully warm cache before issuing
// per-op compiles of kernels the batch is (re)building.
void wait_for_prewarm();

// True iff a prewarm batch has been launched for this process's build_key. The op-by-op barrier is a
// no-op otherwise (e.g. a cold cache that only armed capture).
bool prewarm_enabled();

// Precise barrier support. True iff the launched prewarm batch is (re)building a kernel
// with this base name (|kernel_name| is Kernel::name(), i.e. the out_dir prefix before the
// per-kernel hash). ProgramImpl::compile calls this for each of its kernels: if any is
// warmed by the batch, it must wait_for_prewarm() before compiling (the batch may be writing
// that same out_dir concurrently, and FileRenamer temp names are per-process, so a concurrent
// same-out_dir build would corrupt the ELF). If none are warmed, the program's kernels are a
// set disjoint from the batch (e.g. device-init dispatch/fabric programs), so it may compile
// concurrently with the batch -- no shared out_dir, no collision. Returns false until the
// batch's kernel-name set has been published (which happens before the batch thread starts,
// so any compile observing an in-flight batch also observes the set).
bool prewarm_warms_kernel(const std::string& kernel_name);

}  // namespace tt::tt_metal::kernel_prewarm
