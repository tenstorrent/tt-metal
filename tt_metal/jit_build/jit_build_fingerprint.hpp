// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace tt::tt_metal {

// Up-front precompile build fingerprint.
//
// A hardware-free (mock/sim) build runs slow dispatch, which can resolve some build-determining
// device values DIFFERENTLY from the real fast-dispatch run -- most notably num_l1_banks (fast
// dispatch reserves Tensix cores for prefetch/dispatch, slow dispatch reserves none), which feeds
// the NUM_L1_BANKS kernel define and thus the build_key. Up-front precompile captures the real
// device's resolved values once (capture_jit_build_fingerprint) and replays them in the mock via
// the env var TT_METAL_JIT_BUILD_FINGERPRINT (a file path), so the warm cache is keyed identically
// to the real run.
//
// This is a SINGLE artifact that supersedes per-field TT_METAL_FORCE_* escape hatches: any new
// build-determining field that diverges under mock just gets added to this struct, captured and
// replayed through the one create_jit_device_config chokepoint -- no new env var, no new plumbing.
struct JitBuildFingerprint {
    uint32_t num_l1_banks = 0;
    uint32_t dispatch_core_type = 0;  // tt::tt_metal::DispatchCoreType underlying value
    uint32_t dispatch_core_axis = 0;  // tt::tt_metal::DispatchCoreAxis underlying value
    // Captured so the artifact is self-describing; multi-erisc is applied via the existing
    // TT_METAL_FORCE_2_ERISC_MODE (it is an llrt/firmware-capability concern, not a jit_build one).
    bool enable_2_erisc_mode = true;
    // The real device's compute_with_storage_grid_size() (x,y). On Blackhole fast dispatch reserves
    // Tensix cores for prefetch/dispatch, so the real worker grid is SMALLER than what the slow-dispatch
    // mock reports; ops (conv/matmul/...) read this grid at runtime to derive per-kernel compile-time args,
    // so a mismatch keys those kernels differently -> warm-cache miss. Replayed by Device::
    // compute_with_storage_grid_size on non-silicon so the warm run shards exactly like the real run.
    uint32_t compute_grid_x = 0;
    uint32_t compute_grid_y = 0;

    // Compact "k=v;k=v" form, written to / read from the fingerprint file.
    std::string serialize() const;
    static std::optional<JitBuildFingerprint> deserialize(std::string_view text);
};

// Capture device 0's resolved build fingerprint and write it to `path`. Call on a REAL device
// (reads the resolved JitDeviceConfig stashed in the build env by add_build_env).
void capture_jit_build_fingerprint(const std::string& path);

// The fingerprint selected for this process via env TT_METAL_JIT_BUILD_FINGERPRINT (a file path),
// parsed once and cached. nullopt when the env is unset or the file can't be parsed. Applied only by
// create_jit_device_config, and only on non-silicon (mock/sim) targets.
const std::optional<JitBuildFingerprint>& active_jit_build_fingerprint();

// The captured real-device compute grid (x,y) to replay, iff a fingerprint is active, carries a grid,
// and the target is non-silicon (mock/sim). nullopt otherwise (incl. on real silicon -> use the real
// grid). Called by Device::compute_with_storage_grid_size so warm-collect ops shard like the real run.
std::optional<std::pair<uint32_t, uint32_t>> active_fingerprint_compute_grid();

}  // namespace tt::tt_metal
