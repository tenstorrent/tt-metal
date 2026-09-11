// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

// Public control surface for the in-process kernel-prewarm cold-start (implementation in
// impl/program/kernel_prewarm.hpp). Kept out of the host_api.hpp umbrella so binding these into ttnn
// does not recompile every host_api.hpp dependent. A pipeline warms a cold build_key without holding
// the device for the op-by-op compile: on cold-start, run one warmup with capture-only enabled to
// record the manifest, batch-compile it off-device in-process, then run the real warmup against the
// now-warm cache.

namespace tt::tt_metal {

// Flip capture-only mode at runtime (genfiles + manifest capture, skip model gcc and dispatch).
void KernelPrewarmSetCaptureOnly(bool enabled);

// True iff the in-process cold-start is worth running for the current build_key (cold cache / new
// build_key with capture armed and no in-run batch launched). Query after device init, before warmup.
bool KernelPrewarmColdStartNeeded();

// Compile every captured manifest recipe for the resolved cache into the JIT cache without opening a
// device; returns the number of targets built. Resolves the cache/root paths from the active runtime
// options. Run after a capture-only warmup, before the real run.
std::size_t KernelPrewarmOfflineCompile();

}  // namespace tt::tt_metal
