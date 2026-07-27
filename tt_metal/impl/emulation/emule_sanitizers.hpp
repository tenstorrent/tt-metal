// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Emule sanitizer logic that runs around each kernel launch:
//   - the per-kernel CB-leak (Dirty CB §11) sweep,
//   - Object-Intent (§12) provenance tracking,
//   - the per-kernel sanitizer thread-locals, set/cleared around each run, and
//   - building the per-launch EmuleOobTensorState from the live-range registries.
// Definitions live in emule_sanitizers.cpp (emule-only). Per-check detail and
// the design rationale are in SANITIZER_CHECKS.md.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tt_emule {
struct CBSyncState;
}
namespace tt::tt_metal {
class IDevice;
}

// Wormhole has 32 CBs; JIT header cb_api.h sizes unpack_tile_size[32].
static constexpr uint32_t EMULE_NUM_CBS = 32;

// The per-launch sanitizer range/counter state used to live in `extern
// thread_local`s here; it now lives in the per-fiber context
// (`__emule_self->san`, EmuleSanitizerState in jit_hw/internal/emule_thread_ctx.h)
// so it travels with a yielding fiber instead of being clobbered by a
// co-scheduled one. emule_sanitizers.cpp reads/writes it through __emule_self.

namespace tt::tt_metal::emule {

// Sanitizer state threaded into each kernel launch. Built once per launch by
// build_oob_tensor_state, then copied into the current fiber's sanitizer state
// (__emule_self->san) by set_sanitizer_thread_locals.
struct EmuleOobTensorState {
    bool asan_enabled = false;
    uint32_t l1_unreserved_base = 0;
    const uint64_t* tensor_ranges = nullptr;
    uint32_t tensor_ranges_count = 0;
    uint32_t dram_unreserved_base = 0;
    const uint64_t* dram_tensor_ranges = nullptr;
    uint32_t dram_tensor_ranges_count = 0;
    bool cb_boundary_strict = false;
    const uint64_t* l1_padding_ranges = nullptr;
    uint32_t l1_padding_ranges_count = 0;
    // Raw L1 the host poked outside the Buffer allocator (valid, but not a tensor):
    // an extra valid-extent set for the OOB check, excluded from Object Intent.
    const uint64_t* l1_host_ranges = nullptr;
    uint32_t l1_host_ranges_count = 0;
    bool object_intent_strict = false;
};

// Object Intent Violation (§12): before a single-kernel launch, snapshot every
// live buffer the kernel was NOT handed as I/O; after the launch, memcmp to
// catch writes into a buffer the kernel never resolved a pointer into. A no-op
// for multi-kernel cores (byte changes can't be attributed to one kernel).
class ObjectIntentTracker {
public:
    // `single_kernel_rt_args` are the runtime-arg values of the sole kernel
    // (used only when num_kernels == 1) — buffer base addresses passed as
    // runtime args mark I/O tensors the kernel is allowed to write.
    void pre_launch_snapshot(
        const EmuleOobTensorState& oob,
        std::size_t num_kernels,
        const std::vector<uint32_t>& single_kernel_rt_args,
        const uint8_t* l1_data,
        const std::vector<uint64_t>& persistent_cb_ranges,
        uint32_t lx,
        uint32_t ly);
    // Fold a finished kernel's per-fiber resolved-range log (from
    // __emule_self->san_resolved_log) into this core's resolved set. No-op unless
    // ASAN is on and the core is single-kernel (snapshots_ non-empty).
    void accumulate_resolved(const EmuleOobTensorState& oob, const uint64_t* resolved_log, uint32_t count);
    void verify_post_launch(const uint8_t* l1_data, uint32_t lx, uint32_t ly, const char* kernel_name) const;

private:
    struct Snap {
        uint64_t packed;
        std::vector<uint8_t> bytes;
    };
    std::vector<Snap> snapshots_;
    std::vector<uint64_t> resolved_acc_;
};

// Push / reset the per-kernel sanitizer thread-locals around a kernel run.
void set_sanitizer_thread_locals(const EmuleOobTensorState& oob, uint32_t sem_base, uint32_t sem_size);
void clear_sanitizer_thread_locals();

// Dirty CB (§11): at kernel exit, abort if any CB is left with a trailing
// dangling reserve (no following push) or wait (no following pop).
void sweep_per_kernel_dirty_cbs(
    const EmuleOobTensorState& oob, tt_emule::CBSyncState* cb_array, uint32_t processor_id, uint32_t lx, uint32_t ly);

// Owns the snapshot vectors that EmuleOobTensorState's pointers reference.
struct OobStateOwner {
    EmuleOobTensorState state;
    std::vector<uint64_t> live_ranges;
    std::vector<uint64_t> dram_live_ranges;
    std::vector<uint64_t> padding_ranges;
    std::vector<uint64_t> l1_host_ranges;
};

OobStateOwner build_oob_tensor_state(IDevice* device, int device_id);

}  // namespace tt::tt_metal::emule
