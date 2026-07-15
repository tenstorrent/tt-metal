// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host-side emule sanitizer facade. See SANITIZER_CHECKS.md.
//
// This header is included UNCONDITIONALLY by the core `tt_metal` host API
// (tt_metal.cpp), so it must compile and link cleanly even in a non-emule build
// (`-DTT_METAL_USE_EMULE=OFF`). To guarantee that, it is split into two layers:
//
//   * This header — the always-safe facade. It only ever exposes:
//       - the master-switch readers (`emule_asan_enabled` / `dirty_cb_check_skipped`),
//         which are pure `getenv` and safe in any build, and
//       - DECLARATIONS of the host checks. In an emule build they resolve to the
//         real definitions in host_sanitizers.cpp; in a non-emule build they are
//         inline no-ops, so the call sites in tt_metal.cpp need no `#ifdef`.
//
//   * host_sanitizers.cpp — the emule-only implementation. It is the ONLY place
//     that touches MetalContext/Cluster and references `__emule_asan_panic`
//     (whose single definition lives in emulated_program_runner.cpp). Because it
//     is compiled only when TT_METAL_USE_EMULE is set, a non-emule libtt_metal
//     never carries an unresolved `__emule_asan_panic` reference.
//
// Net effect: the panic symbol and the per-poke alignment lookup are confined to
// the emule build; the production build sees inline no-ops with zero cost.

#include <cstdint>
#include <cstdlib>

// `Buffer` is used by value-of-reference in a declaration below; include the
// real header (it is core, safe in any build) rather than forward-declaring, so
// includers that were getting it transitively keep compiling. `IDevice` is only
// used as a pointer, so a forward declaration suffices.
#include <buffer.hpp>

namespace tt::tt_metal {
class IDevice;
class Program;
}  // namespace tt::tt_metal

namespace tt::tt_metal::emule {

// Compile-time build flag: true only in an emule build. Call sites in the core
// host API wrap their emule additions in `if constexpr (kEmuleAsanBuild) { … }`
// so the calls — and their argument evaluation — are eliminated entirely in a
// non-emule build (rather than relying on the inline no-ops below at runtime).
#ifdef TT_METAL_USE_EMULE
inline constexpr bool kEmuleAsanBuild = true;
#else
inline constexpr bool kEmuleAsanBuild = false;
#endif

// ---- Master-switch readers (always safe; pure getenv) -------------------
// Re-read every call: caching breaks combined test runs that toggle the var.
inline bool emule_asan_enabled() {
    const char* v = std::getenv("TT_METAL_EMULE_ASAN");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

// Per-check opt-out for the Dirty CB sanitizer (TT_METAL_EMULE_ASAN_SKIP_DIRTY_CB):
// skips only `sweep_per_kernel_dirty_cbs` while every other check stays active.
// Re-read every call (a static cache would stick across combined gtest runs). What
// it's for and why this is the one check with its own switch: SANITIZER_CHECKS.md §11.
inline bool dirty_cb_check_skipped() {
    const char* v = std::getenv("TT_METAL_EMULE_ASAN_SKIP_DIRTY_CB");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

// ---- Host checks --------------------------------------------------------
// In an emule build these are defined in host_sanitizers.cpp; otherwise they are
// inline no-ops. The alignment checks take (device, size) rather than a
// precomputed alignment so the Cluster::get_alignment_requirements() lookup
// happens INSIDE the implementation, after the enabled-check — never on the
// production host path.
#ifdef TT_METAL_USE_EMULE

// Use-After-Free: `op` touched a buffer that is not currently allocated.
void check_buffer_allocated(const Buffer& buffer, const char* op);

// Host->L1 / host->DRAM alignment. `size` is the transfer size; the real
// requirement is Cluster::get_alignment_requirements(device, size) (DMA
// alignment when DMA-backed, else 1 → no-op). See host_sanitizers.cpp for why a
// hardcoded word/NoC alignment would false-positive legitimate byte-granular
// writes (e.g. row-major remainders).
void check_host_l1_alignment(const IDevice* device, uint32_t address, uint32_t size, const char* op);
void check_host_dram_alignment(const IDevice* device, uint32_t address, uint32_t size, const char* op);

// Metadata Overflow check: throws if any programmable core type's program
// metadata statically exceeds its reserved KERNEL_CONFIG L1 window. This is an
// emule-only sanitizer (on a freshly-initialized emulated device the normal
// CB/L1 validators don't fire); the caller wraps it so the throw is surfaced as
// an ASAN panic via report_metadata_overflow. No-op on hardware.
void check_program_metadata_size(Program& program);

// Surfaces a Metadata Overflow as an ASAN panic (with the underlying exception
// text) when emulating with ASAN on; the caller then rethrows. No-op otherwise.
void report_metadata_overflow(bool is_emulated, const char* what);

// Declares [logical_size, buffer.size()) as tensor padding for the kernel-side
// Tensor-Padding sanitizer (registers it with LiveL1PaddingRanges); size()
// clears it. L1/L1_SMALL only. Lives here — not on the Buffer API — because it
// has no meaning on hardware. No-op when ASAN is off. See SANITIZER_CHECKS.md §5.
void register_logical_size(const Buffer& buffer, DeviceAddr logical_size);

#else  // !TT_METAL_USE_EMULE — inline no-ops so callers stay #ifdef-free.

inline void check_buffer_allocated(const Buffer&, const char*) {}
inline void check_host_l1_alignment(const IDevice*, uint32_t, uint32_t, const char*) {}
inline void check_host_dram_alignment(const IDevice*, uint32_t, uint32_t, const char*) {}
inline void check_program_metadata_size(Program&) {}
inline void report_metadata_overflow(bool, const char*) {}
inline void register_logical_size(const Buffer&, DeviceAddr) {}

#endif  // TT_METAL_USE_EMULE

}  // namespace tt::tt_metal::emule
