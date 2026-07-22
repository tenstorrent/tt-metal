// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "emulated_program_runner.hpp"
#include "emule_live_ranges.hpp"
#include "host_sanitizers.hpp"
#include "emule_sanitizers.hpp"

#include <dlfcn.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/resource.h>  // getrlimit(RLIMIT_NOFILE) — bound JIT compile fan-out under the fd limit
#include <csignal>
#if defined(__x86_64__) && defined(__linux__)
#include <ucontext.h>
#include <sys/ucontext.h>
#endif

#include <bit>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <limits>
#include <tt_stl/assert.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <semaphore>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <future>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <vector>

#ifndef TT_EMULE_CXX_COMPILER
#error "TT_EMULE_CXX_COMPILER must be defined by CMake"
#endif
#ifndef TT_EMULE_CXX_STANDARD
#error "TT_EMULE_CXX_STANDARD must be defined by CMake"
#endif

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/buffers/circular_buffer.hpp"
#include "impl/buffers/semaphore.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "umd/device/chip/sw_emule_chip.hpp"
#include "umd/device/chip_helpers/simulation_sysmem_manager.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>  // fabric route table (multi-chip dst resolve)
#include <tt-metalium/experimental/fabric/fabric_types.hpp>   // FabricNodeId, MeshId, FabricConfig
#include <tt-metalium/experimental/fabric/fabric.hpp>         // is_2d_fabric_config
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>     // RoutingDirection
#include "tt_emule/device.hpp"
#include "tt_emule/dfb_sync_state.hpp"
#include "tt_emule/kernel_patcher.hpp"  // tt::emule::patch_kernel_source (the extracted JIT patch pass)
#include "tt_emule/tile_counter.hpp"
#include "jit_hw/internal/emule_thread_ctx.h"
#include "emule_fiber_scheduler.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"

#include <tt-logger/tt-logger.hpp>
#include "tt_metal/common/stable_hash.hpp"

#ifndef TT_EMULE_JIT_INCLUDE_DIR
#error "TT_EMULE_JIT_INCLUDE_DIR must be defined by CMake (path to tt-emule's include/jit_hw)"
#endif
#ifndef TT_EMULE_INCLUDE_DIR
#error "TT_EMULE_INCLUDE_DIR must be defined by CMake (path to tt-emule's include/)"
#endif

// ---------------------------------------------------------------------------
// Thread-local context for JIT kernels.
// Exported via -rdynamic so dlopen'd .so files can resolve them at load time.
// ---------------------------------------------------------------------------

// __emule_cb_state is an alias for tt_emule::CBSyncState (see emule_cb_state.h).
// We use the real type directly here.
using __emule_cb_state = tt_emule::CBSyncState;

// The per-RISC identity / handles — rt_args, common_rt_args, core_obj, device,
// bridge_l1/dram, cbs, dfbs, tc_array, processor_id, neo_id, trisc_id,
// num_threads, my_thread_id, core_map — are now fields of the per-thread
// ThreadCommonCtx, reached via __emule_self (defined just below; see
// emule_thread_ctx.h). The runner sets them in the launch lambda; the JIT kernel
// and the extern-C resolvers above read them through __emule_self->X. (The
// mhartid regex now emits `__emule_self->processor_id`; CSR/get_num_threads/
// get_arg shims read the corresponding ctx fields.)

// Per-thread execution context — the single source of truth for an emulated
// RISC's thread-local state, specialized by RISC type (see emule_thread_ctx.h).
// Defined here, exported via -rdynamic so the JIT .so resolves it at dlopen; set
// per kernel thread in the launch lambda below.
thread_local ThreadCommonCtx* __emule_self = nullptr;

// Core execution state (bridge_l1/dram, cbs, dfbs, tc_array, num_threads, my_thread_id, core_map)
// now lives in ThreadCommonCtx, reached via __emule_self and set per-fiber in launch_cores.
//
// These three Quasar identity signals are ALSO kept as -rdynamic globals: the JIT kernel reads them
// from the ctx (__emule_self->{processor_id,neo_id,trisc_id}), but the ASAN sanitizer
// (emule_sanitizers.cpp) reads the globals, so the launch lambda sets both.
//   __processor_id   — RISC-V mhartid analogue (DM index / Neo engine index).
//   __emule_neo_id   — Quasar NEO_ID CSR (0xBC2).
//   __emule_trisc_id — Quasar TRISC_ID CSR (0xBC3); iterated 0..3 across ki.variants for compute.
thread_local uint8_t __processor_id = 0;
thread_local uint8_t __emule_neo_id = 0;
thread_local uint8_t __emule_trisc_id = 0;

// Sanitizer thread-local state (ASAN feature). Exported via -rdynamic so the jit_hw ASAN headers
// resolve them; all null/zero and inert when TT_METAL_EMULE_ASAN is off. Mirrored from
// tt-emule/src/kernel_runner.cpp — the two libs are never linked into the same binary, so the
// duplicate definitions are benign.
thread_local uint32_t __emule_sem_l1_range_start = 0;
thread_local uint32_t __emule_sem_l1_range_end = 0;
thread_local const char* __emule_kernel_name = nullptr;
thread_local uint32_t __emule_pending_noc_reads = 0;
thread_local uint32_t __emule_l1_unreserved_base = 0;
thread_local const uint64_t* __emule_l1_tensor_ranges = nullptr;
thread_local uint32_t __emule_l1_tensor_ranges_count = 0;
thread_local const uint64_t* __emule_l1_padding_ranges = nullptr;
thread_local uint32_t __emule_l1_padding_ranges_count = 0;
thread_local const uint64_t* __emule_l1_host_ranges = nullptr;
thread_local uint32_t __emule_l1_host_ranges_count = 0;
thread_local uint64_t* __emule_l1_resolved_ranges = nullptr;
thread_local uint32_t* __emule_l1_resolved_ranges_count = nullptr;
thread_local uint32_t __emule_l1_resolved_ranges_capacity = 0;
thread_local uint32_t __emule_cb_reserved_pages[32] = {};
thread_local uint32_t __emule_cb_waited_pages[32] = {};
// Dirty-CB leak flags: set by reserve/wait, cleared by push/pop; still-set at
// kernel exit is the leak. Decoupled from the window counters. See SANITIZER_CHECKS.md §11.
thread_local bool __emule_cb_reserve_dangling[32] = {};
thread_local bool __emule_cb_wait_dangling[32] = {};
thread_local const char* __emule_cb_reserve_file[32] = {};
thread_local uint32_t __emule_cb_reserve_line[32] = {};
thread_local const char* __emule_cb_wait_file[32] = {};
thread_local uint32_t __emule_cb_wait_line[32] = {};
thread_local bool __emule_cb_boundary_strict = false;

// DRAM equivalent of __emule_l1_tensor_ranges; consumed only by __emule_dram_ptr below.
thread_local uint32_t __emule_dram_unreserved_base = 0;
thread_local const uint64_t* __emule_dram_tensor_ranges = nullptr;
thread_local uint32_t __emule_dram_tensor_ranges_count = 0;

// ---------------------------------------------------------------------------
// Bank mapping arrays — populated from SoC descriptor before kernel launch.
// Exported via -rdynamic so JIT .so files can resolve them at dlopen time.
// Match firmware declarations: uint16_t[NUM_NOCS][NUM_DRAM_BANKS], etc.
// ---------------------------------------------------------------------------
static constexpr uint32_t NUM_NOCS = 2;
// L1 banks scale with worker grid: 64 on WH-N150, 140 on BH P100/P150.  Must
// match the array size declared by the JIT side in
// `include/jit_hw/internal/dataflow/dataflow_api_addrgen.h`.
static constexpr uint32_t MAX_NUM_BANKS = 256;
// Semaphore alignment in L1 (must match firmware layout).
static constexpr uint32_t EMULE_SEM_ALIGN = 16;

uint16_t dram_bank_to_noc_xy[NUM_NOCS][MAX_NUM_BANKS] = {};
int32_t bank_to_dram_offset[MAX_NUM_BANKS] = {};
uint16_t l1_bank_to_noc_xy[NUM_NOCS][MAX_NUM_BANKS] = {};
int32_t bank_to_l1_offset[MAX_NUM_BANKS] = {};

// Per-core NOC coordinates — set per kernel thread (thread_local).
// On real HW these are read from NOC registers; we set them from physical coords.
thread_local uint8_t my_x[NUM_NOCS] = {};
thread_local uint8_t my_y[NUM_NOCS] = {};
thread_local uint32_t __emule_logical_x = 0;
thread_local uint32_t __emule_logical_y = 0;
// Silicon-named per-core LOGICAL coords (firmware globals `my_logical_x_/y_`,
// declared extern by blaze/kernels/kernel_utils.hpp). Defined here so compute
// (TRISC) kernels that reference them link; restored per fiber swap-in by the
// scheduler's install_fiber. The dataflow (NCRISC/BRISC) senders that must read a
// CORRECT per-fiber value instead resolve `my_logical_x_/y_` through the
// dataflow_utils.hpp shadow's per-fiber accessor (__emule_self->core->logical_*),
// so this definition is only a link/fallback anchor on other RISCs.
thread_local uint8_t my_logical_x_ = 0;
thread_local uint8_t my_logical_y_ = 0;

// NOC encoding constants (matching firmware for Blackhole/Wormhole).
static constexpr uint32_t NOC_LOCAL_BITS = 36;
static constexpr uint32_t NOC_NODE_ID_BITS = 6;
static constexpr uint64_t NOC_LOCAL_MASK = (1ULL << NOC_LOCAL_BITS) - 1;
static constexpr uint32_t NOC_NODE_MASK = (1 << NOC_NODE_ID_BITS) - 1;

// __emule_asan_panic is defined in emule_asan_panic.cpp (same libtt_metal); the
// checks below and the JIT kernel .so files resolve it at link/dlopen.
extern "C" [[noreturn]] void __emule_asan_panic(const char* fmt, ...);

// C-linkage bridge/fabric hooks for JIT kernels. Every one runs inside a kernel fiber, so
// __emule_self is always set; a null means the hook ran outside a fiber — a contract violation,
// not a recoverable state. Fail loudly (uniform across the whole bridge surface).
static inline void emule_require_self(const char* fn) {
    TT_FATAL(__emule_self != nullptr, "{}: emule bridge call outside a kernel fiber context", fn);
}

extern "C" uint8_t* __emule_dram_ptr(uint64_t offset) {
    emule_require_self(__func__);
    // ASAN out-of-bounds DRAM check (inert when TT_METAL_EMULE_ASAN off — ranges stay null). Range-test
    // in 32 bits to match the live-range registry (uint32_t start/end); the 64-bit offset is used only
    // for the backing-store address. Assumes DRAM addresses fit in 32 bits (true for every WH/BH config).
    if (__emule_dram_tensor_ranges != nullptr &&
        static_cast<uint32_t>(offset) >= __emule_dram_unreserved_base) {
        uint32_t addr = static_cast<uint32_t>(offset);
        bool in_tensor = false;
        for (uint32_t i = 0; i < __emule_dram_tensor_ranges_count; ++i) {
            uint64_t packed = __emule_dram_tensor_ranges[i];
            uint32_t r_start = static_cast<uint32_t>(packed >> 32);
            uint32_t r_end = static_cast<uint32_t>(packed);
            if (addr >= r_start && addr < r_end) {
                in_tensor = true;
                break;
            }
        }
        if (!in_tensor) {
            __emule_asan_panic(
                "[ASAN ERROR] Out-of-Bounds Write: Attempted to access DRAM address 0x%x which is not part of any "
                "allocated tensor\n",
                addr);
        }
    }
    return __emule_self->bridge_dram ? __emule_self->bridge_dram + offset : nullptr;
}

extern "C" uint8_t* __emule_local_l1_ptr(uint32_t offset) {
    emule_require_self(__func__);
    // ASAN illegal-semaphore-region check (inert when ASAN off — range end stays 0).
    if (__emule_sem_l1_range_end > 0 &&
        offset >= __emule_sem_l1_range_start && offset < __emule_sem_l1_range_end) {
        __emule_asan_panic(
            "[ASAN ERROR] Illegal Semaphore Access: Offset 0x%x is inside the reserved Semaphore region [0x%x, 0x%x)\n",
            offset,
            __emule_sem_l1_range_start,
            __emule_sem_l1_range_end);
    }
    return __emule_self->bridge_l1 ? __emule_self->bridge_l1 + offset : nullptr;
}

extern "C" uint8_t* __emule_noc_resolve(uint32_t x, uint32_t y, uint64_t addr) {
    emule_require_self(__func__);
    if (__emule_self->core_map) {
        uint64_t key = (uint64_t(x) << 32) | y;
        auto it = __emule_self->core_map->find(key);
        if (it != __emule_self->core_map->end()) {
            return it->second->l1_ptr(static_cast<uint32_t>(addr));
        }
    }
    return nullptr;
}

// Fiber-scheduler bridge — the dlopen'd kernel .so calls these (declared in
// include/jit_hw/internal/emule_fiber_bridge.h) to park/wake/yield on the one
// scheduler instance. Resolved at dlopen via -rdynamic, like the resolvers above.
namespace efib = tt::tt_metal::emule_fiber;
extern "C" void __emule_fiber_lock(void) { efib::FiberScheduler::instance().lock(); }
extern "C" void __emule_fiber_unlock(void) { efib::FiberScheduler::instance().unlock(); }
extern "C" void __emule_fiber_park_locked(const void* key) {
    efib::FiberScheduler::instance().park_locked(key);
}
extern "C" void __emule_fiber_park_locked_socket(const void* key) {
    efib::FiberScheduler::instance().park_locked_socket(key);
}
extern "C" void __emule_fiber_wake(const void* key) { efib::FiberScheduler::instance().wake(key); }
extern "C" void __emule_fiber_yield(void) { efib::FiberScheduler::instance().yield(); }
extern "C" void __emule_fiber_defer_to_quiescence(void) { efib::FiberScheduler::instance().quiescence_park(); }
extern "C" void __emule_fiber_note_publish(unsigned pages) {
    efib::FiberScheduler::instance().note_publish(pages);
}

// Worker L1 slot size + mask: a worker's L1 field is a 0-based in-slot offset (< 2 MB), so masking the low
// bits is an idempotent guard. Applied ONLY for WORKER cores (DRAM banks are GB-scale — see the
// per-resolver comments). Used by every NOC-address resolver.
static constexpr uint32_t L1_SLOT_SIZE = 2u * 1024 * 1024;  // 2 MB per worker L1 slot
static constexpr uint32_t L1_SLOT_MASK = L1_SLOT_SIZE - 1;  // 0x1FFFFF

// Resolve a NOC address (encoded 64-bit) to a host pointer.
// Real firmware encoding: y in bits [47:42], x in bits [41:36], addr in bits [35:0]
//
// The L1_SLOT_MASK is applied ONLY for WORKER cores. Two reasons:
//  1. Worker L1 fields are 0-based in-slot offsets (from `get_write_ptr()` etc.),
//     always < 2 MB, so the mask is an idempotent guard on the local field.
//  2. DRAM banks are GB-scale (2 GB on Wormhole views, 4 GB on Blackhole)
//     and the kernel-side per-bank addrgen helper produces an `addr` field
//     that is the true in-bank offset (already includes
//     `bank_to_dram_offset[bank_index]`). Masking to 2 MB silently aliases
//     any DRAM access >= 2 MB to an offset within the first 2 MB of the bank.
// Helper: get SWEmuleChip* from MetalContext cluster for a given device_id. (Relocated up from
// later in this file — needed here for the PCIe branch below, and by the fabric teleport hooks
// further down.)
static tt::umd::SWEmuleChip* get_sw_emulated_chip(tt::ChipId device_id) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto* umd_cluster = cluster.get_driver().get();
    if (!umd_cluster) {
        return nullptr;
    }
    auto* chip = umd_cluster->get_chip(device_id);
    return dynamic_cast<tt::umd::SWEmuleChip*>(chip);
}

// Per-device cache of pcie_base_ (the host-facing/PCIe address threshold), keyed by device_id —
// avoids a dynamic_cast + cluster lookup on every single NOC-address resolve. Rebuilt lazily; a
// device close+reopen mints a new SWEmuleChip with a stable arch, so the cached value never goes
// stale the way the core_map cache (which holds raw Core* into per-chip L1) can.
static std::mutex g_pcie_base_mutex;
static std::unordered_map<uint32_t, uint64_t> g_pcie_base_cache;

static uint64_t get_pcie_base_cached(uint32_t device_id) {
    std::lock_guard<std::mutex> lock(g_pcie_base_mutex);
    auto it = g_pcie_base_cache.find(device_id);
    if (it != g_pcie_base_cache.end()) {
        return it->second;
    }
    auto* sw_emu = get_sw_emulated_chip(static_cast<tt::ChipId>(device_id));
    uint64_t pcie_base =
        sw_emu ? tt::umd::SysmemManager::get_pcie_base_for_arch(sw_emu->get_soc_descriptor().arch) : UINT64_MAX;
    g_pcie_base_cache[device_id] = pcie_base;
    return pcie_base;
}

extern "C" uint8_t* __emule_resolve_noc_addr(uint64_t noc_addr) {
    emule_require_self(__func__);

    // Host-facing (PCIe) address: SimulationSysmemManager's device_io_addr space starts at
    // pcie_base_ and shares the same 64-bit range as a real on-chip NOC address, so this branch
    // must run FIRST, before any noc_x/noc_y/local_addr decomposition below.
    uint32_t device_id = __emule_self->chip_id;
    if (noc_addr >= get_pcie_base_cached(device_id)) {
        auto* sw_emu = get_sw_emulated_chip(static_cast<tt::ChipId>(device_id));
        auto* sysmem = sw_emu ? static_cast<tt::umd::SimulationSysmemManager*>(sw_emu->get_sysmem_manager()) : nullptr;
        return sysmem ? static_cast<uint8_t*>(sysmem->get_mapped_host_ptr(noc_addr, /*size=*/1)) : nullptr;
    }

    uint32_t noc_x = (noc_addr >> NOC_LOCAL_BITS) & NOC_NODE_MASK;
    uint32_t noc_y = (noc_addr >> (NOC_LOCAL_BITS + NOC_NODE_ID_BITS)) & NOC_NODE_MASK;
    uint64_t local_addr = noc_addr & NOC_LOCAL_MASK;  // 36 bits, raw

    if (__emule_self->core_map) {
        uint64_t key = (uint64_t(noc_x) << 32) | noc_y;
        auto it = __emule_self->core_map->find(key);
        if (it != __emule_self->core_map->end()) {
            uint32_t offset = (it->second->role() == tt_emule::CoreRole::WORKER)
                                  ? (static_cast<uint32_t>(local_addr) & L1_SLOT_MASK)
                                  : static_cast<uint32_t>(local_addr);
            return it->second->l1_ptr(offset);
        }
    }
    return nullptr;
}

extern "C" bool __emule_noc_addr_is_dram(uint64_t noc_addr) {
    emule_require_self(__func__);
    if (!__emule_self->core_map) {
        return false;
    }
    uint32_t noc_x = (noc_addr >> NOC_LOCAL_BITS) & NOC_NODE_MASK;
    uint32_t noc_y = (noc_addr >> (NOC_LOCAL_BITS + NOC_NODE_ID_BITS)) & NOC_NODE_MASK;
    uint64_t key = (uint64_t(noc_x) << 32) | noc_y;
    auto it = __emule_self->core_map->find(key);
    if (it != __emule_self->core_map->end()) {
        return it->second->role() == tt_emule::CoreRole::DRAM;
    }
    return false;
}

// Resolve multicast: iterate over rectangle of cores and memcpy to each.
// Real firmware encoding: x_start [53:48], y_start [59:54], x_end [41:36], y_end [47:42], addr [35:0]
//
// `include_self`: silicon's NOC_CMD_BRCST_SRC_INCLUDE bit. When the API is
// `noc_async_write_multicast_loopback_src` (or _set_multicast_loopback_src),
// silicon sets the bit and the sender NIU receives its own packet ->
// include_self=true. When the API is `noc_async_write_multicast` (non-loopback),
// silicon clears the bit and the sender NIU drops the packet at itself ->
// include_self=false. Sender coords come from the TLS that thread launch
// wires up (my_x[0], my_y[0]).
extern "C" void __emule_multicast_write(uint64_t mcast_addr, const uint8_t* src, uint32_t size, bool include_self, uint8_t noc) {
    uint32_t x_end = (mcast_addr >> NOC_LOCAL_BITS) & NOC_NODE_MASK;
    uint32_t y_end = (mcast_addr >> (NOC_LOCAL_BITS + NOC_NODE_ID_BITS)) & NOC_NODE_MASK;
    uint32_t x_start = (mcast_addr >> (NOC_LOCAL_BITS + 2 * NOC_NODE_ID_BITS)) & NOC_NODE_MASK;
    uint32_t y_start = (mcast_addr >> (NOC_LOCAL_BITS + 3 * NOC_NODE_ID_BITS)) & NOC_NODE_MASK;
    uint64_t l1_offset = mcast_addr & NOC_LOCAL_MASK;

    // The L1 offset is a 0-based in-slot offset (< 2 MB, from get_write_ptr() etc.), so masking
    // with SLOT_MASK is an idempotent guard on the local field.
    // Multicast targets only WORKER cores (DRAM cores are skipped by the role
    // check in the delivery loop below), so the mask is L1-correct here.
    l1_offset &= L1_SLOT_MASK;

    emule_require_self(__func__);
    if (!__emule_self->core_map) {
        return;
    }

    // Sender coordinates (from the TLS that thread launch wires up). Used to
    // skip self when include_self=false (non-loopback multicast).
    uint32_t self_x = my_x[0];
    uint32_t self_y = my_y[0];

    // NOC1 rectangles arrive with start<->end SWAPPED: silicon describes a NOC1
    // multicast in NOC1's reflected coordinate frame (paired with the DYNAMIC_NOC_X/Y
    // reflection). Emule models NOC coordinates as identity (DYNAMIC_NOC_X/Y is
    // identity — see jit_hw api/dataflow/dataflow_api.h), so no reflection happens and
    // the swap alone leaves the rectangle reversed. Undo it so the walk below runs on
    // physical (NOC0-frame) coordinates for both NOCs. Without this, canonical NOC1
    // in0/in1-mcast ops (matmul/linear, multicore argmax) present start>end and the
    // torus walk misreads it as a wraparound → receivers never see their semaphore →
    // quiescent deadlock.
    if (noc != 0) {
        uint32_t t;
        t = x_start; x_start = x_end; x_end = t;
        t = y_start; y_start = y_end; y_end = t;
    }

    // Torus-wraparound walk on physical coords. Silicon's NOC treats the rectangle on
    // a torus, so a rectangle whose cores straddle the worker-grid seam encodes
    // start > end and wraps around the NOC node space rather than covering the min..max
    // bounding box; flash_mla's S4/S8 SDPA blocks (NOC0) rely on this. Walk each axis
    // start->end stepping +1 mod the node space; coords with no core in the map are
    // skipped. For a non-wrapping rectangle (start <= end) this is identical to
    // min..max — the post-un-swap NOC1 case (matmul/argmax in0-mcast).
    auto axis_count = [](uint32_t s, uint32_t e) -> uint32_t {
        return (e >= s ? (e - s) : ((NOC_NODE_MASK + 1 - s) + e)) + 1;
    };
    const uint32_t nx = axis_count(x_start, x_end);
    const uint32_t ny = axis_count(y_start, y_end);
    uint32_t delivered = 0;
    for (uint32_t ix = 0; ix < nx; ix++) {
        const uint32_t x = (x_start + ix) & NOC_NODE_MASK;
        for (uint32_t iy = 0; iy < ny; iy++) {
            const uint32_t y = (y_start + iy) & NOC_NODE_MASK;
            if (!include_self && x == self_x && y == self_y) {
                continue;
            }
            uint64_t key = (uint64_t(x) << 32) | y;
            auto it = __emule_self->core_map->find(key);
            if (it != __emule_self->core_map->end() && it->second->role() == tt_emule::CoreRole::WORKER) {
                uint8_t* dst = it->second->l1_ptr(static_cast<uint32_t>(l1_offset));
                if (size == sizeof(uint32_t)) {
                    TT_FATAL(
                        reinterpret_cast<uintptr_t>(dst) % alignof(std::atomic<uint32_t>) == 0,
                        "multicast_write: L1 offset 0x{:x} is not 4-byte aligned for atomic store",
                        l1_offset);
                    // Atomic store for semaphore-sized writes (4 bytes)
                    uint32_t val;
                    std::memcpy(&val, src, sizeof(uint32_t));
                    reinterpret_cast<std::atomic<uint32_t>*>(dst)->store(val, std::memory_order_release);
                    efib::FiberScheduler::instance().wake(dst);  // wake the target core's sem waiter
                } else {
                    std::memcpy(dst, src, size);
                    std::atomic_thread_fence(std::memory_order_release);
                }
                delivered++;
            }
        }
    }
    static const bool mdbg = std::getenv("EMULE_DEBUG") != nullptr;
    if (delivered == 0 && mdbg) {
        fprintf(
            stderr,
            "EMULE WARN: multicast (%u,%u)->(%u,%u) offset=0x%lx size=%u: "
            "no worker cores found [from phys (%u,%u)]\n",
            x_start,
            y_start,
            x_end,
            y_end,
            (unsigned long)l1_offset,
            size,
            my_x[0],
            my_y[0]);
    }
}

namespace tt::tt_metal::emule {

// ---------------------------------------------------------------------------
// Shared types used across subfunctions
// ---------------------------------------------------------------------------

// Mirrors RTA_CRTA_NO_ARGS_SENTINEL in tt_metal/hw/inc/hostdev/rta_constants.h.
constexpr uint16_t kRtaCrtaNoArgsSentinel = 0xFFFF;

struct KernelInfo {
    // size 1 for normal kernels; size 4 for Quasar compute (one per TRISC).
    // Either 4 distinct compiled variants (compile-time TRISC_* guards) or 4
    // copies of one function (runtime TRISC_ID). When run_all_variants is true,
    // the launcher iterates and sets __emule_trisc_id per variant.
    std::vector<std::function<void()>> variants;
    bool run_all_variants = false;
    uint8_t processor_id = 0;  // RISC-V processor ID (mhartid); used for DFB role resolution
    uint8_t thread_idx = 0;    // Index within this kernel's processor list → __emule_my_thread_id
    bool is_tensix = false;    // true for Tensix/compute kernels (DFB mask uses bits 8-23)
    uint32_t num_threads = 1;  // number of engines (for get_num_threads())
    // L1 address of rt-args = kernel_config_base + rta_offset_in_kc (per-RISC,
    // read from kg->launch_msg). Sentinel = kernel has no args on this RISC.
    uint32_t kernel_config_base = 0;
    uint16_t rta_offset_in_kc = kRtaCrtaNoArgsSentinel;
    uint16_t crta_offset_in_kc = kRtaCrtaNoArgsSentinel;
    // Runtime-arg values handed to this kernel on its core (see PendingKernelInfo).
    std::vector<uint32_t> rt_arg_values;
    // Kernel source path; owns the string __emule_kernel_name points at during
    // this kernel's launch (used by the ASAN trace to name the offending kernel).
    std::string kernel_name;
};

// Captures a Metal 2.0 kernel's named bindings. Drives both the JIT wrapper's
// namespace emission (see emit_metal2_namespaces) and the JIT cache key
// (see cache_key_suffix). Empty for legacy kernels.
struct Metal2BindingsSnapshot {
    // TA bindings are kept in insertion order (matches genfiles.cpp's vector);
    // their CRTA position drives the get_common_vararg offset.
    struct TaEntry {
        std::string name;
        uint32_t cta_offset;
        uint32_t addr_crta_offset;
    };

    bool is_metal2 = false;
    std::vector<std::string> runtime_arg_names;
    std::vector<std::string> common_runtime_arg_names;
    std::map<std::string, uint32_t> dfb_accessors;
    std::map<std::string, uint16_t> sem_accessors;
    std::vector<TaEntry> ta_accessors;

    // Distinguishes kernels that share source/CTAs/defines but bind different
    // IDs — without this they collide on cache key and the second silently
    // reuses the first's .so.
    std::string cache_key_suffix() const {
        std::string s;
        for (const auto& [name, id] : dfb_accessors) {
            s += ":dfb:" + name + "=" + std::to_string(id);
        }
        for (const auto& [name, id] : sem_accessors) {
            s += ":sem:" + name + "=" + std::to_string(id);
        }
        for (const auto& ta : ta_accessors) {
            s += ":ta:" + ta.name + "=" + std::to_string(ta.cta_offset) + "," +
                 std::to_string(ta.addr_crta_offset);
        }
        for (const auto& name : runtime_arg_names) {
            s += ":rta:" + name;
        }
        for (const auto& name : common_runtime_arg_names) {
            s += ":crta:" + name;
        }
        return s;
    }
};

struct DeferredCompile {
    std::string src_path;
    std::vector<uint32_t> compile_args;
    std::unordered_map<std::string, uint32_t> named_compile_args;
    NamedCTArgNamespaces named_ct_arg_namespaces;
    NamedRuntimeArgNamespaces named_runtime_arg_namespaces;
    std::map<std::string, std::string> defines;
    std::string extra_inc;
    Metal2BindingsSnapshot bindings;
};

struct PendingKernelInfo {
    // Parallels KernelInfo::variants but holds cache keys pending compile-resolution.
    std::vector<std::string> variant_cache_keys;
    bool run_all_variants = false;
    uint8_t processor_id = 0;
    uint8_t thread_idx = 0;    // Index within this kernel's processor list
    bool is_tensix = false;
    uint32_t num_threads = 1;
    uint32_t kernel_config_base = 0;
    uint16_t rta_offset_in_kc = kRtaCrtaNoArgsSentinel;
    uint16_t crta_offset_in_kc = kRtaCrtaNoArgsSentinel;
    // Runtime-arg values (unique + common); buffer L1 addresses appear verbatim, so
    // Object-Intent uses them to find this kernel's I/O tensors (§12).
    std::vector<uint32_t> rt_arg_values;
    std::string kernel_name;  // kernel source path, for the ASAN trace
};

// DFB allocation info for a single DFB on a core. Only dfb_id and base_addr
// are genuinely new per-core state; everything else (entry_size, num_entries,
// risc masks, num_producers/consumers, cap) is read from the borrowed config
// pointer, whose backing DataflowBufferImpl is owned by ProgramImpl and
// outlives one program execution.
struct DFBAllocInfo {
    uint32_t dfb_id = 0;
    uint32_t base_addr = 0;
    const tt::tt_metal::experimental::dfb::DataflowBufferConfig* cfg = nullptr;
};

struct CoreSetup {
    CoreCoord logical_core;
    tt_emule::Core* core;
    std::vector<KernelInfo>* ki_list;
    uint8_t phys_x;
    uint8_t phys_y;
    std::vector<DFBAllocInfo> dfb_allocs;
    bool has_dfbs = false;
    uint32_t sem_base;
    uint32_t sem_size;
    // Globally-allocated (persistent) CB extents on this core, packed (start<<32|end);
    // Object-Intent exempts kernel writes to them (§12).
    std::vector<uint64_t> persistent_cb_ranges;
};

// Per-slot initialization data for a DFB tile-counter slot. wr_ptr and rd_ptr
// always start at the same position — producer-STRIDED/ALL at the per-slot
// offset, consumer-ALL at the sub-range base. The 4 DFB role combinations
// differ only in how these fields are computed (see fill_dfb_slots callers).
struct DfbSlotInit {
    uint8_t counter_id;
    uint32_t base_addr;
    uint32_t limit;
    uint32_t ptr;
};

template <typename SlotFn>
static void fill_dfb_slots(tt_emule::EmuleDFBInterface& iface, uint32_t n, SlotFn&& slot_fn) {
    const uint32_t cap = std::min<uint32_t>(n, tt_emule::MAX_TC_SLOTS_PER_DFB);
    for (uint32_t k = 0; k < cap; ++k) {
        auto& slot = iface.tc_slots[k];
        DfbSlotInit s = slot_fn(k);
        slot.neo_id = 0;
        slot.counter_id = s.counter_id;
        slot.base_addr = s.base_addr;
        slot.limit = s.limit;
        slot.wr_ptr = s.ptr;
        slot.rd_ptr = s.ptr;
    }
}

// ---------------------------------------------------------------------------
// JIT Compilation Cache (in-memory + persistent disk cache)
// ---------------------------------------------------------------------------

static std::mutex g_jit_cache_mutex;
static std::unordered_map<std::string, std::function<void()>> g_jit_cache;

// ---------------------------------------------------------------------------
// Disk JIT cache — survives process restarts (critical for --forked mode)
// ---------------------------------------------------------------------------

static constexpr size_t FNV_HEX_BUF_SIZE = 17;  // 16 hex digits + null

static uint64_t fnv1a_hash(const std::string& s) {
    tt::StableHasher hasher;
    hasher.update(s);
    return hasher.digest();
}

static std::string get_jit_cache_dir() {
    if (const char* dir = std::getenv("TT_EMULE_JIT_CACHE_DIR")) {
        return dir;
    }
    // Fixed per-user path. Staleness is handled at lookup time —
    // disk_cache_lookup() invalidates a cached .so when the kernel source or any
    // JIT header is newer than it — so the directory name needs no fingerprint.
    return "/tmp/tt_emule_jit_cache_" + std::to_string(getuid());
}

// dlopen a previously cached .so and return the kernel entry function.
// Returns nullptr on failure (missing file, symbol resolution error, etc.).
static std::function<void()> dlopen_cached_so(const std::string& so_path) {
    void* metal_lib = dlopen("libtt_metal.so", RTLD_NOW | RTLD_NOLOAD | RTLD_GLOBAL);
    if (!metal_lib) {
        log_warning(tt::LogMetal, "dlopen_cached_so: could not promote libtt_metal.so to RTLD_GLOBAL: {}", dlerror());
    }
    void* handle = dlopen(so_path.c_str(), RTLD_NOW);
    if (!handle) {
        return nullptr;
    }

    dlerror();  // clear
    using RawFn = void (*)();
    RawFn fn = reinterpret_cast<RawFn>(dlsym(handle, "__emule_kernel_entry"));
    if (dlerror()) {
        dlclose(handle);
        return nullptr;
    }

    auto shared_handle = std::shared_ptr<void>(handle, [](void* h) { dlclose(h); });
    return [fn, shared_handle]() { fn(); };
}

// Check disk cache for a compiled .so matching cache_key.
// Returns a callable if cache hit (and source mtime is not newer), else nullptr.
static std::function<void()> disk_cache_lookup(const std::string& cache_key, const std::string& src_path) {
    std::string cache_dir = get_jit_cache_dir();
    char hex[FNV_HEX_BUF_SIZE];
    std::snprintf(hex, sizeof(hex), "%016lx", fnv1a_hash(cache_key));
    std::string so_path = cache_dir + "/" + hex + ".so";

    if (!std::filesystem::exists(so_path)) {
        return nullptr;
    }

    auto so_mtime = std::filesystem::last_write_time(so_path);

    // Invalidate if kernel source file is newer than cached .so
    // (skip for inline sources — their content is hashed into the cache key)
    if (!src_path.empty() && std::filesystem::exists(src_path)) {
        if (std::filesystem::last_write_time(src_path) > so_mtime) {
            return nullptr;
        }
    }

    // Invalidate if any JIT header is newer than the cached .so. The cache
    // key only covers kernel source content and compile flags; header edits
    // (e.g. dataflow_api.h) would otherwise silently serve stale binaries.
    for (auto& entry : std::filesystem::recursive_directory_iterator(TT_EMULE_JIT_INCLUDE_DIR)) {
        if (entry.is_regular_file() && entry.last_write_time() > so_mtime) {
            return nullptr;
        }
    }

    auto fn = dlopen_cached_so(so_path);
    if (fn) {
        log_debug(tt::LogMetal, "JIT disk cache hit: {}", so_path);
    }
    return fn;
}

// Return the disk cache .so path for a given cache key.
static std::string disk_cache_so_path(const std::string& cache_key) {
    std::string cache_dir = get_jit_cache_dir();
    std::filesystem::create_directories(cache_dir);
    char hex[FNV_HEX_BUF_SIZE];
    std::snprintf(hex, sizeof(hex), "%016lx", fnv1a_hash(cache_key));
    return cache_dir + "/" + hex + ".so";
}

// ---------------------------------------------------------------------------
// JIT Kernel Compilation
// ---------------------------------------------------------------------------

static Metal2BindingsSnapshot build_metal2_snapshot(const tt::tt_metal::Kernel& kernel) {
    Metal2BindingsSnapshot s;
    s.is_metal2 = kernel.is_metal2_kernel();
    s.runtime_arg_names = kernel.get_runtime_arg_names();
    s.common_runtime_arg_names = kernel.get_common_runtime_arg_names();
    kernel.process_dataflow_buffer_local_accessor_handles(
        [&s](const std::string& name, uint16_t id) { s.dfb_accessors[name] = id; });
    kernel.process_semaphore_local_accessor_handles(
        [&s](const std::string& name, uint16_t id) { s.sem_accessors[name] = id; });
    kernel.process_tensor_binding_handles(
        // Match the genfiles.cpp pattern: drop num_runtime_field_crta_words. Emule's
        // snapshot doesn't yet model per-binding runtime CRTA words, and the
        // downstream `named_crta_words` math in emit_metal2_namespaces still assumes
        // 1 word per binding — so a dynamic-shape kernel would silently get its
        // CRTAs decoded at the wrong offsets. Static-shape kernels pass
        // num_rt_words == 0 and are unaffected. Fail loudly on dynamic-shape until
        // snapshot + cache key + get_common_vararg offset math are wired up to
        // consume the per-binding count.
        [&s](const std::string& name, uint32_t cta_off, uint32_t addr_crta_off, uint32_t num_rt_words) {
            TT_FATAL(
                num_rt_words == 0,
                "Emule does not yet support dynamic-shape Metal 2.0 tensor bindings "
                "(binding '{}' has num_runtime_field_crta_words={}). Wire the per-"
                "binding word count through Metal2BindingsSnapshot::TaEntry, the "
                "cache key, and emit_metal2_namespaces' get_common_vararg base "
                "before enabling this path.",
                name,
                num_rt_words);
            s.ta_accessors.push_back({name, cta_off, addr_crta_off});
        });
    return s;
}

// Emits args::/dfb::/sem::/ta:: namespaces into the JIT wrapper, replacing
// kernel_args_generated.h + kernel_bindings_generated.h that upstream's JIT
// build produces. Must stay text-equivalent to genfiles.cpp's
// write_kernel_{args,bindings}_generated_header.
static void emit_metal2_namespaces(
    std::ostream& f,
    const Metal2BindingsSnapshot& s,
    const std::unordered_map<std::string, uint32_t>& named_compile_args) {
    const bool has_args =
        !s.runtime_arg_names.empty() || !s.common_runtime_arg_names.empty() || !named_compile_args.empty();
    if (has_args) {
        f << "#include \"experimental/kernel_args.h\"\n";
    }
    if (!s.dfb_accessors.empty()) {
        f << "#include \"api/dataflow/dataflow_buffer.h\"\n";
    }
    if (!s.sem_accessors.empty()) {
        f << "#include <cstdint>\n";
    }
    if (!s.ta_accessors.empty()) {
        f << "#include \"api/tensor/tensor_accessor.h\"\n";
    }

    if (has_args) {
        f << "namespace args {\n";
        uint32_t rta_offset = 0;
        for (const auto& name : s.runtime_arg_names) {
            f << "constexpr ::experimental::RtaArg<uint32_t> " << name << "{" << rta_offset << "};\n";
            rta_offset += sizeof(uint32_t);
        }
        uint32_t crta_offset = 0;
        for (const auto& name : s.common_runtime_arg_names) {
            f << "constexpr ::experimental::CrtaArg<uint32_t> " << name << "{" << crta_offset << "};\n";
            crta_offset += sizeof(uint32_t);
        }
        // Sort CTAs for deterministic wrapper output.
        std::vector<std::pair<std::string, uint32_t>> cta_entries(
            named_compile_args.begin(), named_compile_args.end());
        std::sort(cta_entries.begin(), cta_entries.end());
        for (const auto& [name, value] : cta_entries) {
            // Namespaced compile-time args carry a dotted name (e.g. "cp.dst"),
            // which is not a valid flat C++ identifier, so emitting
            // `constexpr CtaVal<uint32_t> cp.dst{...}` here would fail to compile;
            // skip them to keep the flat `args::` form namespaced-safe. This change
            // does NOT emit the matching `ct_args::<ns>` structs — that is a separate
            // emission step (it needs a Kernel::process_named_ct_arg_namespaces API);
            // a kernel that references `ct_args::<ns>` requires that step to be
            // present, so skipping here only prevents invalid flat C++, it does not
            // itself make namespaced args available.
            if (name.find('.') != std::string::npos) {
                continue;
            }
            f << "constexpr ::experimental::CtaVal<uint32_t> " << name << "{" << value << "u};\n";
        }
        f << "}  // namespace args\n";
    }
    if (!s.dfb_accessors.empty()) {
        f << "namespace dfb {\n";
        for (const auto& [name, id] : s.dfb_accessors) {
            f << "constexpr DFBAccessor " << name << "{" << id << "};\n";
        }
        f << "}  // namespace dfb\n";
    }
    if (!s.sem_accessors.empty()) {
        f << "namespace sem {\n";
        for (const auto& [name, id] : s.sem_accessors) {
            f << "constexpr std::uint32_t " << name << " = " << id << "u;\n";
        }
        f << "}  // namespace sem\n";
    }
    if (!s.ta_accessors.empty()) {
        f << "namespace ta {\n";
        for (const auto& ta : s.ta_accessors) {
            f << "using " << ta.name << "_t = ::tensor_accessor::TensorAccessorBindingToken<"
              << ta.cta_offset << "u, " << ta.addr_crta_offset << "u>;\n";
            f << "constexpr " << ta.name << "_t " << ta.name << "{};\n";
        }
        f << "}  // namespace ta\n";
    }

    // Vararg helpers — always emitted for Metal 2.0 kernels (mirrors
    // genfiles.cpp). The CRTA buffer layout is [user-named CRTAs,
    // TensorBinding addresses, varargs], so get_common_vararg's base skips
    // past both the named CRTAs and the binding section.
    if (s.is_metal2) {
        const uint32_t named_rta_words = static_cast<uint32_t>(s.runtime_arg_names.size());
        const uint32_t named_crta_words =
            static_cast<uint32_t>(s.common_runtime_arg_names.size() + s.ta_accessors.size());
        f << "FORCE_INLINE uint32_t get_vararg(uint32_t idx) { "
          << "return get_arg_val<uint32_t>(" << named_rta_words << " + idx); }\n";
        f << "FORCE_INLINE uint32_t get_common_vararg(uint32_t idx) { "
          << "return get_common_arg_val<uint32_t>(" << named_crta_words << " + idx); }\n";
    }
}

static std::function<void()> jit_compile_kernel(
    const std::string& kernel_src_path,
    const std::vector<uint32_t>& compile_args,
    const std::unordered_map<std::string, uint32_t>& named_compile_args,
    const NamedCTArgNamespaces& named_ct_arg_namespaces,
    const NamedRuntimeArgNamespaces& named_runtime_arg_namespaces,
    const std::map<std::string, std::string>& defines,
    const std::string& extra_include_flags,
    const Metal2BindingsSnapshot& bindings = {},
    const std::string& disk_cache_so_path_arg = "") {
    const std::string jit_inc = TT_EMULE_JIT_INCLUDE_DIR;
    const std::string parent_inc = TT_EMULE_INCLUDE_DIR;

    // 1. Verify kernel source exists
    if (!std::filesystem::exists(kernel_src_path)) {
        throw std::runtime_error("jit_compile_kernel: kernel source not found: " + kernel_src_path);
    }
    std::string abs_kernel = std::filesystem::absolute(kernel_src_path).string();

    // 2. Create temp directory
    char tmpdir[] = "/tmp/tt_emule_jit_XXXXXX";
    if (!mkdtemp(tmpdir)) {
        throw std::runtime_error("jit_compile_kernel: mkdtemp failed");
    }
    std::string dir(tmpdir);

    // 2b. Preprocess the kernel source for x86: rewrite RISC-V inline asm
    // (mhartid, fence) and raw L1 arg-val pointer casts. -I kernel_dir (below)
    // keeps relative includes in the patched file resolvable.
    std::string patched_kernel_path = dir + "/patched_kernel.cpp";
    // WORKAROUND: see tt-emule/.claude/skills/workarounds (WA-2).
    // The fabric mux (tt_fabric_mux.cpp) is a transport-layer aggregation kernel: workers write packets
    // into its L1 channels and it forwards them over ethernet. emule has no ethernet — WorkerToFabricMux
    // Sender teleports each packet straight to its final destination (same as the no-mux direct path),
    // so the mux has nothing to do. The real kernel is also persistent (loops until an external
    // termination signal) and pulls in erisc firmware emule doesn't model, which would both fail to
    // compile and hang emule's run-to-completion join. Substitute a no-op kernel: it compiles, exits
    // immediately, and the teleporting mux sender carries the data. (Mirrors how emule collapses the eth
    // router/switch into the teleport — the mux is the worker-side half of that same transport.)
    if (std::filesystem::path(abs_kernel).filename() == "tt_fabric_mux.cpp") {
        std::ofstream f(patched_kernel_path);
        if (!f) {
            throw std::runtime_error("jit_compile_kernel: cannot write mux stub " + patched_kernel_path);
        }
        f << "// emule no-op stub for tt_fabric_mux.cpp (the teleporting mux sender carries the data).\n"
          << "#include \"api/dataflow/dataflow_api.h\"\n"
          << "void kernel_main() {}\n";
    } else {
        // Kernel include roots (ttnn/, tt_metal/) parsed from the JIT -I flags so the patcher
        // can reach + patch shared kernel helpers that live in another directory (the raw-L1-deref
        // idioms in e.g. kernel_lib/*.inl). The emule shadow roots are checked first, so jit_hw
        // headers are never patched.
        std::vector<std::string> kernel_inc_roots;
        {
            static const std::regex inc_flag_re(R"RE(-I"([^"]+)")RE");
            for (std::sregex_iterator it(extra_include_flags.begin(), extra_include_flags.end(), inc_flag_re), end;
                 it != end; ++it) {
                kernel_inc_roots.push_back((*it)[1].str());
            }
        }
        const std::vector<std::string> emule_inc_roots = {jit_inc, parent_inc};
        tt::emule::patch_kernel_source(abs_kernel, patched_kernel_path, kernel_inc_roots, emule_inc_roots);
    }

    // 2c. Emit named_args_generated.h with the kernel's ct_args:: namespaces
    // (mirrors `write_named_args_generated_header` in jit_build/genfiles.cpp).
    // Silicon's per-kernel build runs genfiles.cpp, which writes this header
    // into the kernel out-dir; emule's JIT path bypasses genfiles entirely,
    // so we replicate the emission here. The header is included from
    // wrapper.cpp below when non-empty.
    //
    // Format matches genfiles.cpp byte-for-byte where practical so kernels
    // see the same `ct_args::<prefix>` struct shape under emule as under
    // silicon build.
    bool has_named_args = false;
    {
        std::set<std::string> all_ns;
        for (const auto& [ns, _] : named_ct_arg_namespaces) {
            all_ns.insert(ns);
        }
        for (const auto& [ns, _] : named_runtime_arg_namespaces) {
            if (!ns.empty()) {
                all_ns.insert(ns);
            }
        }
        std::ostringstream header_ct;
        for (const auto& ns : all_ns) {
            if (!ns.empty()) {
                header_ct << "struct " << ns << " {\n";
            }
            if (auto it = named_ct_arg_namespaces.find(ns); it != named_ct_arg_namespaces.end()) {
                for (const auto& [field, value] : it->second) {
                    header_ct << "    static constexpr uint32_t " << field << " = " << value << ";\n";
                }
            }
            if (auto it = named_runtime_arg_namespaces.find(ns); it != named_runtime_arg_namespaces.end()) {
                for (const auto& entry : it->second) {
                    const char* dispatch_str = entry.dispatch == RuntimeArgDispatch::COMMON
                                                   ? "rt_args::Dispatch::COMMON"
                                                   : "rt_args::Dispatch::PER_CORE";
                    if (entry.length > 1) {
                        header_ct << "    static constexpr rt_args::ArrayArg " << entry.field << " = {" << entry.index
                                  << ", " << entry.length << ", " << dispatch_str << "};\n";
                    } else {
                        header_ct << "    static constexpr rt_args::Arg " << entry.field << " = {" << entry.index
                                  << ", " << dispatch_str << "};\n";
                    }
                }
            }
            if (!ns.empty()) {
                header_ct << "};\n";
            }
        }
        auto ct_str = header_ct.str();
        if (!ct_str.empty()) {
            has_named_args = true;
            std::ofstream f(dir + "/named_args_generated.h");
            f << "#pragma once\n#include \"api/rt_arg.h\"\n\n";
            f << "namespace ct_args {\n" << ct_str << "}\n";
        }
    }

    // 3. Write wrapper.cpp
    // Kernel defines are written as #define directives in the wrapper to avoid
    // shell quoting issues (values like SFPU_OP_CHAIN_0 contain parentheses).
    std::string wrapper_path = dir + "/wrapper.cpp";
    {
        std::ofstream f(wrapper_path);
        if (!f) {
            throw std::runtime_error("jit_compile_kernel: cannot write " + wrapper_path);
        }
        // Emit kernel defines before any includes so they're visible to kernel code
        for (const auto& [key, value] : defines) {
            if (value.empty()) {
                f << "#define " << key << "\n";
            } else {
                f << "#define " << key << " " << value << "\n";
            }
        }
        f << "#include \"jit_kernel_stubs.hpp\"\n";
        // #1219 COEXIST: Metal-2.0 `namespace args` (base) + blaze `ct_args::` header (additive layer).
        emit_metal2_namespaces(f, bindings, named_compile_args);
        if (has_named_args) {
            f << "#include \"" << dir << "/named_args_generated.h\"\n";
        }
        f << "#include \"" << patched_kernel_path << "\"\n";
        f << "extern \"C\" { void __emule_kernel_entry() { kernel_main(); } }\n";
    }

    // 4. Build -DKERNEL_COMPILE_TIME_ARGS=... flag
    std::string ct_flag;
    if (!compile_args.empty()) {
        std::ostringstream ss;
        ss << "-DKERNEL_COMPILE_TIME_ARGS=";
        for (size_t i = 0; i < compile_args.size(); ++i) {
            if (i) {
                ss << ',';
            }
            ss << compile_args[i];
        }
        ct_flag = ss.str();
    }

    // 5. Build extra define flags (emulator-specific only; kernel defines are in wrapper.cpp)
    // Note: EMULE_SEM_BASE and EMULE_SEM_ALIGN are passed via kernel defines (in wrapper.cpp)
    // rather than here, so they can be dynamically computed per-program.
    std::string define_flags = " -DTT_EMULE_USE_L1_POOL";

    // 5b. Build -DKERNEL_COMPILE_TIME_ARG_MAP for named compile-time args
    if (!named_compile_args.empty()) {
        std::ostringstream ss;
        ss << " \"-DKERNEL_COMPILE_TIME_ARG_MAP=";
        bool first = true;
        for (const auto& [name, value] : named_compile_args) {
            if (!first) {
                ss << ',';
            }
            ss << "{\\\"" << name << "\\\"," << value << "}";
            first = false;
        }
        ss << "\"";
        define_flags += ss.str();
    }

    // 6. Compute the kernel's source directory for relative includes
    std::string kernel_dir = std::filesystem::path(abs_kernel).parent_path().string();

    // 7. Compile — output to disk cache path if provided, else temp dir
    std::string so_path = disk_cache_so_path_arg.empty() ? (dir + "/kernel.so") : disk_cache_so_path_arg;
    // Under ASAN, keep -O2 but add debug info + frame pointers so the backtrace can
    // resolve kernel file:line. Folded into the JIT cache key (see compute_cache_key)
    // so these .so files don't collide with the non-ASAN cache. See SANITIZER_CHECKS.md.
    std::string opt_flags = " -O2";
    if (tt::tt_metal::emule::emule_asan_enabled()) {
        opt_flags += " -g -fno-omit-frame-pointer -funwind-tables";
    }
    std::ostringstream cmd;
    // -fms-extensions: fabric/CCL kernels collapse 32-bit-device L1 pointers to uint32_t (e.g.
    // `(uint32_t)pkt_hdr`); on the 64-bit host clang treats pointer→smaller-int as a hard error, but
    // -fms-extensions downgrades it to a warning. The JIT patch pass rewrites those header narrowings to
    // bridge_l1-relative offsets (A-rule), so they stay correct when worker L1 is mapped above 4 GB.
    // (opt_flags = -O2, + ASAN debug info when enabled.)
    cmd << TT_EMULE_CXX_COMPILER << " -std=c++" << TT_EMULE_CXX_STANDARD << " -fPIC -shared" << opt_flags
        << " -Wno-c++11-narrowing -fms-extensions"
        // out_dir first: patched copies of shared kernel headers (written under
        // out_dir/<include-name> by the patcher) must shadow the
        // originals for full-path includes at any nesting depth. out_dir never
        // contains emule (jit_hw) headers — those are skipped — so it can't shadow them.
        << " -I\"" << dir << "\""
        << " -I\"" << jit_inc << "\""
        << " -I\"" << parent_inc << "\""
        << " -I\"" << kernel_dir << "\"";
    // Extra include paths (project source, ttnn, etc.)
    if (!extra_include_flags.empty()) {
        cmd << " " << extra_include_flags;
    }
    cmd << " -o \"" << so_path << "\"";
    if (!ct_flag.empty()) {
        cmd << " \"" << ct_flag << "\"";
    }
    cmd << define_flags;
    cmd << " \"" << wrapper_path << "\"";
    cmd << " 2>&1";

    std::string full_cmd = cmd.str();
    log_debug(tt::LogMetal, "JIT compile: {}", full_cmd);

    // Ensure the output dir exists right before linking: on a cold JIT cache the
    // shared cache dir may not be present yet when ld writes its output.
    std::filesystem::create_directories(std::filesystem::path(so_path).parent_path());

    // Safety: all path/flag inputs are derived from tt-metal internals and CMake
    // constants, not from untrusted user input. Kernel defines are written as
    // #define in the wrapper file, not as -D shell flags.
    int rc = std::system(full_cmd.c_str());
    if (rc != 0) {
        throw std::runtime_error(
            "jit_compile_kernel: compiler failed (exit " + std::to_string(rc) + ") for kernel: " + kernel_src_path);
    }

    // 8. dlopen
    // Promote libtt_metal.so to RTLD_GLOBAL so kernel.so can resolve TLS symbols
    // (e.g. __emule_cbs) that are defined in libtt_metal.so. When loaded via
    // Python module import, shared libraries default to RTLD_LOCAL.
    void* metal_lib = dlopen("libtt_metal.so", RTLD_NOW | RTLD_NOLOAD | RTLD_GLOBAL);
    if (!metal_lib) {
        log_warning(tt::LogMetal, "jit_compile_kernel: could not promote libtt_metal.so to RTLD_GLOBAL: {}", dlerror());
    }
    void* handle = dlopen(so_path.c_str(), RTLD_NOW);
    if (!handle) {
        throw std::runtime_error(std::string("jit_compile_kernel: dlopen failed: ") + dlerror());
    }

    // 9. Resolve entry point
    using RawFn = void (*)();
    dlerror();  // clear
    RawFn fn = reinterpret_cast<RawFn>(dlsym(handle, "__emule_kernel_entry"));
    const char* err = dlerror();
    if (err) {
        std::string msg(err);
        dlclose(handle);
        throw std::runtime_error("jit_compile_kernel: dlsym(__emule_kernel_entry) failed: " + msg);
    }

    // 10. Clean up temp directory (wrapper.cpp etc.) — always safe since .so is
    // either in the disk cache dir or mmap'd into memory from the temp dir.
    // TT_EMULE_KEEP_JIT_SRC keeps the patched_kernel.cpp/wrapper.cpp for inspection.
    if (!std::getenv("TT_EMULE_KEEP_JIT_SRC")) {
        std::filesystem::remove_all(dir);
    } else {
        fprintf(stderr, "[EMULE] kept JIT src: %s\n", dir.c_str());
    }

    // 11. Wrap in shared_ptr for lifetime management (dlclose on destruction).
    auto shared_handle = std::shared_ptr<void>(handle, [](void* h) { dlclose(h); });
    return [fn, shared_handle]() { fn(); };
}

// ---------------------------------------------------------------------------
// Program Execution — multi-threaded with CB synchronization
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// populate_bank_mapping: Set up DRAM/L1 bank arrays from SoC descriptor.
// ---------------------------------------------------------------------------
static void populate_bank_mapping(
    tt::umd::SWEmuleChip* sw_emu,
    IDevice* device,
    ChipId device_id,
    tt_emule::Core*& dram_core_out,
    uint32_t& num_dram_channels_out,
    uint32_t& num_l1_banks_out) {
    dram_core_out = nullptr;
    num_dram_channels_out = 0;
    num_l1_banks_out = 0;
    if (!sw_emu) {
        return;
    }

    auto& soc = sw_emu->get_soc_descriptor();
    auto dram_channels = soc.get_dram_cores();
    num_dram_channels_out = static_cast<uint32_t>(dram_channels.size());

    if (num_dram_channels_out > 0) {
        dram_core_out = sw_emu->get_dram_channel_backing(0);
    }

    // Populate bank mapping arrays using metal_SocDescriptor (matches host write path).
    auto& metal_soc = MetalContext::instance().get_cluster().get_soc_desc(device_id);
    num_dram_channels_out = static_cast<uint32_t>(metal_soc.get_num_dram_views());
    TT_FATAL(
        num_dram_channels_out <= MAX_NUM_BANKS,
        "emule: num_dram_channels ({}) exceeds MAX_NUM_BANKS ({}); bump the constant or implement dynamic backing",
        num_dram_channels_out,
        MAX_NUM_BANKS);

    // noc_xy encoding: (y << 6) | x (matching Blackhole firmware encoding).
    // Per-NOC preferred coords: get_preferred_worker_core_for_dram_view's noc arg
    // selects the view's worker_endpoint[noc] subchannel. On Wormhole the two
    // subchannels coincide (worker_endpoint=[n,n]); on Blackhole they are distinct
    // NOC ports of the same physical bank.
    //
    // The kernel-side extern is declared [NUM_NOCS][NUM_DRAM_BANKS] where the JIT
    // define NUM_DRAM_BANKS == num_dram_channels_out, so the kernel's [noc][bank]
    // row stride is the actual bank count — NOT this array's static MAX_NUM_BANKS
    // dimension. Lay the table out flat with that same actual-count stride (matching
    // silicon's [noc*num_banks + bank] vector) so noc=1 rows align. A 2D [noc][bank]
    // write would stride by MAX_NUM_BANKS and the kernel's noc=1 reads would land on
    // uninitialized zeros → coord (0,0) → wrong (DRAM) backing.
    uint16_t* dram_tbl = &dram_bank_to_noc_xy[0][0];
    std::memset(dram_bank_to_noc_xy, 0, sizeof(dram_bank_to_noc_xy));
    std::memset(bank_to_dram_offset, 0, sizeof(bank_to_dram_offset));
    for (uint32_t ch = 0; ch < num_dram_channels_out && ch < MAX_NUM_BANKS; ch++) {
        auto dc0 = metal_soc.get_preferred_worker_core_for_dram_view(ch, 0 /* NOC 0 */);
        auto dc1 = metal_soc.get_preferred_worker_core_for_dram_view(ch, 1 /* NOC 1 */);
        uint16_t noc_xy0 = (static_cast<uint16_t>(dc0.y) << NOC_NODE_ID_BITS) | static_cast<uint16_t>(dc0.x);
        uint16_t noc_xy1 = (static_cast<uint16_t>(dc1.y) << NOC_NODE_ID_BITS) | static_cast<uint16_t>(dc1.x);
        dram_tbl[0 * num_dram_channels_out + ch] = noc_xy0;
        dram_tbl[1 * num_dram_channels_out + ch] = noc_xy1;
        bank_to_dram_offset[ch] = static_cast<int32_t>(metal_soc.get_address_offset(ch));

        log_debug(
            tt::LogMetal,
            "  DRAM bank[{}]: NOC0=({},{}) NOC1=({},{}) noc_xy0=0x{:04x} noc_xy1=0x{:04x} offset=0x{:x}",
            ch,
            dc0.x,
            dc0.y,
            dc1.x,
            dc1.y,
            noc_xy0,
            noc_xy1,
            bank_to_dram_offset[ch]);
    }

    // L1 bank mapping — mirror the host allocator's bank distribution so the
    // kernel-side `interleaved_addr_gen::get_bank_index<L1>(id)` lands on the
    // same worker core that `SWEmuleChip::write_to_device` wrote a given page
    // to.  Without this, every page maps to bank 0 (a single core) while the
    // host scatters across all worker cores — interleaved-L1 → sharded paths
    // read all zeros.
    // Flat actual-count stride, same rationale as dram_bank_to_noc_xy above: the
    // kernel reads l1_bank_to_noc_xy[noc][bank] with stride NUM_L1_BANKS (== the JIT
    // define == num_l1_banks_out), not MAX_NUM_BANKS.
    uint16_t* l1_tbl = &l1_bank_to_noc_xy[0][0];
    std::memset(l1_bank_to_noc_xy, 0, sizeof(l1_bank_to_noc_xy));
    std::memset(bank_to_l1_offset, 0, sizeof(bank_to_l1_offset));
    if (device) {
        const auto& allocator = device->allocator();
        num_l1_banks_out = allocator->get_num_banks(BufferType::L1);
        TT_FATAL(
            num_l1_banks_out <= MAX_NUM_BANKS,
            "emule: num_l1_banks ({}) exceeds MAX_NUM_BANKS ({}); bump the constant or implement dynamic backing",
            num_l1_banks_out,
            MAX_NUM_BANKS);
        for (uint32_t b = 0; b < num_l1_banks_out && b < MAX_NUM_BANKS; ++b) {
            auto logical = allocator->get_logical_core_from_bank_id(b);
            auto virt = device->virtual_core_from_logical_core(logical, CoreType::WORKER);
            uint16_t noc_xy = (static_cast<uint16_t>(virt.y) << NOC_NODE_ID_BITS) |
                              static_cast<uint16_t>(virt.x);
            l1_tbl[0 * num_l1_banks_out + b] = noc_xy;  // NOC 0
            l1_tbl[1 * num_l1_banks_out + b] = noc_xy;  // NOC 1 (same target in emule)
            // Intentionally leave bank_to_l1_offset[b] = 0.  emule's per-core
            // L1 mmap starts at byte 0 with no firmware-reserved prefix, so
            // silicon's `allocator->get_bank_offset(L1, b)` isn't applicable.
        }
    }
}

// ---------------------------------------------------------------------------
// build_worker_coord_maps: Build logical→virtual coordinate mapping strings.
// ---------------------------------------------------------------------------
static void build_worker_coord_maps(IDevice* device, std::string& worker_col_map_str, std::string& worker_row_map_str) {
    static constexpr uint32_t MAX_LOGICAL_GRID_DIM = 64;
    auto grid = device->compute_with_storage_grid_size();
    std::ostringstream col_ss;
    for (uint32_t lx = 0; lx < MAX_LOGICAL_GRID_DIM; lx++) {
        if (lx) {
            col_ss << ',';
        }
        if (lx < grid.x) {
            auto virt = device->virtual_core_from_logical_core(CoreCoord(lx, 0), CoreType::WORKER);
            col_ss << virt.x;
        } else {
            col_ss << 0;
        }
    }
    worker_col_map_str = col_ss.str();

    std::ostringstream row_ss;
    for (uint32_t ly = 0; ly < MAX_LOGICAL_GRID_DIM; ly++) {
        if (ly) {
            row_ss << ',';
        }
        if (ly < grid.y) {
            auto virt = device->virtual_core_from_logical_core(CoreCoord(0, ly), CoreType::WORKER);
            row_ss << virt.y;
        } else {
            row_ss << 0;
        }
    }
    worker_row_map_str = row_ss.str();
}

// ---------------------------------------------------------------------------
// get_extra_include_flags: Build -I flags for JIT compilation.
// ---------------------------------------------------------------------------
static std::string get_extra_include_flags() {
#ifdef TT_EMULE_PROJECT_SOURCE_DIR
    const std::string project_src = TT_EMULE_PROJECT_SOURCE_DIR;
    std::string extra_inc;
    extra_inc += "-I\"" + project_src + "/ttnn/cpp\"";
    // Resolves headers included with the repo-rooted `cpp/ttnn/...` prefix
    // (e.g. the SDPA dataflow helper chain pulled in by the sampling writer).
    extra_inc += " -I\"" + project_src + "/ttnn\"";
    extra_inc += " -I\"" + project_src + "\"";
    extra_inc += " -I\"" + project_src + "/tt_metal/hw/inc\"";
    extra_inc += " -I\"" + project_src + "/tt_metal/hostdevcommon/api\"";
    return extra_inc;
#else
    return {};
#endif
}

// ---------------------------------------------------------------------------
// collect_kernels: Gather per-core kernel info, check caches, defer misses.
// ---------------------------------------------------------------------------
// Resolve a kernel's source to an on-disk path. FILE_PATH sources are used as-is;
// inline sources are spilled to a temp file and tracked for cleanup.
static std::string resolve_kernel_source_path(const KernelSource& ksrc, std::vector<std::string>& inline_src_temps) {
    if (ksrc.source_type_ == KernelSource::FILE_PATH) {
        return ksrc.path_.string();
    }
    static constexpr int kTmpSuffixLen = 4;  // length of ".cpp" suffix
    char tmpf[] = "/tmp/tt_emule_src_XXXXXX.cpp";
    int fd = mkstemps(tmpf, kTmpSuffixLen);
    if (fd < 0) {
        throw std::runtime_error("execute_program_emulated: mkstemps failed");
    }
    const std::string& content = ksrc.source_;
    const char* buf = content.c_str();
    size_t remaining = content.size();
    while (remaining > 0) {
        ssize_t written = ::write(fd, buf, remaining);
        if (written < 0) {
            ::close(fd);
            throw std::runtime_error("execute_program_emulated: write failed");
        }
        buf += written;
        remaining -= written;
    }
    ::close(fd);
    std::string src_path = tmpf;
    inline_src_temps.push_back(src_path);
    return src_path;
}

// Build the full defines map for a kernel: subclass-derived + arch + emulator
// constants (banking, alignments, worker maps, sem base, CB tile sizes).
static std::map<std::string, std::string> build_kernel_defines(
    Kernel& kernel,
    detail::ProgramImpl& impl,
    uint32_t num_dram_channels,
    uint32_t num_l1_banks,
    const std::string& worker_col_map_str,
    const std::string& worker_row_map_str,
    uint32_t emule_sem_base) {
    std::map<std::string, std::string> defines;
    kernel.process_defines([&](const std::string& k, const std::string& v) { defines[k] = v; });

    // Opt-in deadlock-watchdog timeout. Off by default so <chrono> stays out of
    // the kernel include graph (~1s faster cold JIT compile; see
    // tt-emule include/jit_hw/emule_wait.h). Set TT_EMULE_WAIT_TIMEOUT=1 to
    // restore the bounded cv.wait_for + per-op hang diagnostic. Routed through
    // the defines map (not a bare -D) so it lands in both the wrapper and the
    // JIT cache key — toggling it invalidates stale cached .so files.
    if (std::getenv("TT_EMULE_WAIT_TIMEOUT")) {
        defines["EMULE_WAIT_TIMEOUT"] = "1";
    }

    // Opt-in deep-SFPU override. TT_EMULE_DEEP_SFPU=sqrt,sigmoid promotes those
    // shadowed SFPU ops from their layer-1 libm shadow to the deep path (the real
    // silicon ckernel_sfpu_<op>.h run on emule's faithful sfpi backend — see
    // tt-emule docs/sfpu-deep-path.md). Each comma-separated name becomes an
    // EMULE_DEEP_SFPU_<UPPER> define. Routed through the defines map so it lands
    // in the JIT cache key (toggling invalidates stale cached .so). Ops with no
    // layer-1 shadow take the deep path automatically and need no opt-in.
    if (const char* deep = std::getenv("TT_EMULE_DEEP_SFPU")) {
        const std::string list(deep);
        size_t start = 0;
        while (start <= list.size()) {
            const size_t comma = list.find(',', start);
            const size_t end = (comma == std::string::npos) ? list.size() : comma;
            std::string op;
            for (size_t i = start; i < end; ++i) {
                const char c = list[i];
                if (c == ' ' || c == '\t') {
                    continue;  // trim whitespace
                }
                op.push_back((c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A') : c);
            }
            if (!op.empty()) {
                defines["EMULE_DEEP_SFPU_" + op] = "1";
            }
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
    }

    auto arch = MetalContext::instance().get_cluster().arch();
    if (arch == ARCH::QUASAR) {
        defines["ARCH_QUASAR"] = "1";
    } else if (arch == ARCH::WORMHOLE_B0) {
        defines["ARCH_WORMHOLE"] = "1";
    } else if (arch == ARCH::BLACKHOLE) {
        defines["ARCH_BLACKHOLE"] = "1";
    }

    {
        uint32_t num_dram = num_dram_channels ? num_dram_channels : 1;
        uint32_t num_l1 = num_l1_banks ? num_l1_banks : 1;
        defines["NUM_DRAM_BANKS"] = std::to_string(num_dram);
        defines["NUM_L1_BANKS"] = std::to_string(num_l1);
        // Mirror tt_metal/jit_build/build_env_manager.cpp:118-129. Upstream
        // `interleaved_addr_gen::get_bank_offset_index<DRAM>` chooses bit-shift
        // when banks are a power of two and a constant divisor otherwise.
        // Without these defines, non-pow2 bank counts (12 on WH-N150) silently
        // fall through to a 0-bit shift and every page lands in bank 0.
        auto is_pow2 = [](uint32_t n) { return n > 0 && (n & (n - 1)) == 0; };
        auto log2u = [](uint32_t n) {
            uint32_t l = 0;
            while ((1u << l) < n) {
                ++l;
            }
            return l;
        };
        if (is_pow2(num_dram)) {
            defines["LOG_BASE_2_OF_NUM_DRAM_BANKS"] = std::to_string(log2u(num_dram));
        } else {
            defines["IS_NOT_POW2_NUM_DRAM_BANKS"] = "1";
        }
        if (is_pow2(num_l1)) {
            defines["LOG_BASE_2_OF_NUM_L1_BANKS"] = std::to_string(log2u(num_l1));
        } else {
            defines["IS_NOT_POW2_NUM_L1_BANKS"] = "1";
        }
    }
    defines["NUM_NOCS"] = std::to_string(NUM_NOCS);
    // Fabric routing mode for the emule packet-header stamping shims. The real
    // fabric_set_line_unicast_route dispatches 1D-vs-2D on the header TYPE, but emule aliases
    // LowLatencyPacketHeader == HybridMeshPacketHeader (one 64B layout), so the shim cannot tell
    // them apart by type — it disambiguates on this build-mode define instead.
    if (tt::tt_fabric::is_2d_fabric_config(MetalContext::instance().get_fabric_config())) {
        defines["EMULE_FABRIC_2D"] = "1";
    }
    // Upstream tensor/dspec.h gates `get_common_arg_addr` as a forward-decl
    // under KERNEL_BUILD; emule's jit_kernel_stubs.hpp provides the definition.
    // Without KERNEL_BUILD, dspec.h emits a stub that collides with emule's.
    defines["KERNEL_BUILD"] = "1";
    defines["DRAM_ALIGNMENT"] = std::to_string(hal::get_dram_alignment());
    defines["L1_ALIGNMENT"] = std::to_string(hal::get_l1_alignment());
    defines["EMULE_WORKER_COL_MAP"] = worker_col_map_str;
    defines["EMULE_WORKER_ROW_MAP"] = worker_row_map_str;
    {
        char buf[12];  // "0x" + max 8 hex digits + null
        std::snprintf(buf, sizeof(buf), "0x%x", emule_sem_base);
        defines["EMULE_SEM_BASE"] = buf;
    }
    defines["EMULE_SEM_ALIGN"] = std::to_string(EMULE_SEM_ALIGN);

    // Collect CB tile sizes + per-CB tile shape from program for the constexpr
    // get_tile_size() / get_tile_r_dim() / get_tile_c_dim() metadata. The shape
    // (height/width) is the ground truth for thin tiles (e.g. Tile([1,16])) —
    // the emulated reduce/unpack primitives bound their iteration by it instead
    // of assuming a full 32x32 tile. Default 32x32 when a CB has no Tile spec.
    const auto& core_range_set = kernel.core_range_set();
    if (!core_range_set.ranges().empty()) {
        auto first_core = core_range_set.ranges().begin()->start_coord;
        auto cb_impls = impl.circular_buffers_on_core(first_core);
        uint32_t tile_sizes[EMULE_NUM_CBS] = {};
        // Per-CB data format → emule's analog of genfiles.cpp::compute_data_formats()
        // (which bakes unpack_src_format[]/pack_dst_format[] into chlkc_descriptors.h).
        // 255 == tt::DataFormat::Invalid marks unconfigured slots (mirrors the host's
        // std::optional<DataFormat> empty state); consumers fall back to the page_size
        // heuristic for those. tile_r_dim/tile_c_dim carry the per-CB tile shape
        // (height/width) for thin tiles; default 32x32 when a CB has no Tile spec.
        uint8_t cb_formats[EMULE_NUM_CBS];
        uint32_t tile_r_dim[EMULE_NUM_CBS];
        uint32_t tile_c_dim[EMULE_NUM_CBS];
        for (uint32_t i = 0; i < EMULE_NUM_CBS; i++) {
            cb_formats[i] = static_cast<uint8_t>(tt::DataFormat::Invalid);
            tile_r_dim[i] = tt::constants::TILE_HEIGHT;
            tile_c_dim[i] = tt::constants::TILE_WIDTH;
        }
        for (auto& cb_impl : cb_impls) {
            for (uint8_t idx : cb_impl->local_buffer_indices()) {
                if (idx < EMULE_NUM_CBS) {
                    // Calculate tile size from the CB's data format.
                    const auto& tile = cb_impl->tile(idx);
                    tile_sizes[idx] = tile.has_value() ? tile->get_tile_size(cb_impl->data_format(idx))
                                                       : Tile().get_tile_size(cb_impl->data_format(idx));
                    cb_formats[idx] = static_cast<uint8_t>(cb_impl->data_format(idx));
                    if (tile.has_value()) {
                        tile_r_dim[idx] = tile->get_height();
                        tile_c_dim[idx] = tile->get_width();
                    }
                }
            }
        }
        std::ostringstream ts, df, tr, tc;
        for (uint32_t i = 0; i < EMULE_NUM_CBS; i++) {
            if (i) {
                ts << ',';
                df << ',';
                tr << ',';
                tc << ',';
            }
            ts << tile_sizes[i];
            df << static_cast<uint32_t>(cb_formats[i]);
            tr << tile_r_dim[i];
            tc << tile_c_dim[i];
        }
        defines["EMULE_TILE_SIZES"] = ts.str();
        defines["EMULE_CB_DATA_FORMATS"] = df.str();
        defines["EMULE_TILE_R_DIM"] = tr.str();
        defines["EMULE_TILE_C_DIM"] = tc.str();
    }

    // Thread the compute kernel's resolved fp32_dest_acc_en / dst_full_sync_en
    // into its TU, mirroring silicon genfiles.cpp::emit_compute_scalar_descriptors.
    // dest_helpers.hpp::DEST_AUTO_LIMIT must resolve identically in a program's
    // reader and compute kernels (e.g. multi-core H-reduce interleaves input
    // tiles in chunks of DEST_AUTO_LIMIT). The factory already injects
    // ENABLE_FP32_DEST_ACC/DST_SYNC_FULL into the reader's defines; without this
    // the compute TU falls back to the jit_kernel_stubs defaults (bf16/SyncFull
    // → 16) instead of the program's real mode, scrambling the chunked reduce.
    if (kernel.get_kernel_processor_class() == HalProcessorClassType::COMPUTE) {
        const auto kernel_config = kernel.config();
        if (const auto* cc = std::get_if<ComputeConfig>(&kernel_config)) {
            defines["DST_ACCUM_MODE"] = cc->fp32_dest_acc_en ? "1" : "0";
            defines["ENABLE_FP32_DEST_ACC"] = cc->fp32_dest_acc_en ? "1" : "0";
            defines["DST_SYNC_FULL"] = cc->dst_full_sync_en ? "1" : "0";
        }
    }
    return defines;
}

// Determine per-kernel thread count and the processor ids each thread runs as:
// - QuasarDataMovementKernel: one thread per DM processor (0..7).
// - QuasarComputeKernel: one thread per NEO engine (0..3), each running 4 TRISCs.
// - Other kernels: single thread at the kernel's processor type.
struct ProcIdList {
    std::vector<uint8_t> proc_ids;
    uint32_t num_threads;
};
static ProcIdList compute_proc_ids_and_thread_count(
    Kernel& kernel,
    experimental::quasar::QuasarDataMovementKernel* qdm,
    experimental::quasar::QuasarComputeKernel* qck) {
    ProcIdList out{};
    out.num_threads = 1;
    if (qdm && !qdm->get_dm_processors().empty()) {
        for (const auto& proc : qdm->get_dm_processors()) {
            out.proc_ids.push_back(
                static_cast<uint8_t>(static_cast<std::underlying_type_t<std::remove_cvref_t<decltype(proc)>>>(proc)));
        }
        out.num_threads = static_cast<uint32_t>(qdm->get_dm_processors().size());
    } else if (qck) {
        std::set<uint8_t> neo_ids_seen;
        for (const auto& proc : qck->get_compute_processors()) {
            uint8_t neo_id = static_cast<uint8_t>(
                static_cast<std::underlying_type_t<std::remove_cvref_t<decltype(proc)>>>(proc) /
                experimental::quasar::QUASAR_NUM_COMPUTE_PROCESSORS_PER_TENSIX_ENGINE);
            if (neo_ids_seen.insert(neo_id).second) {
                out.proc_ids.push_back(neo_id);
            }
        }
        out.num_threads = static_cast<uint32_t>(neo_ids_seen.size());
    } else {
        out.proc_ids.push_back(static_cast<uint8_t>(kernel.get_kernel_processor_type(0)));
    }
    return out;
}

// For Quasar compute kernels, scan the source for TRISC guards:
// - TRISC_UNPACK/MATH/PACK/ISOLATE_SFPU defines → compile 4 variants with each define set
// - else mentions TRISC_ID → single compile, run 4 times with different TRISC_ID
// - otherwise → single compile, single run
struct TriscMode {
    bool needs_trisc_compile = false;
    bool needs_runtime_trisc = false;
};
static TriscMode detect_quasar_trisc_mode(bool is_quasar_compute, const std::string& src_path) {
    TriscMode mode;
    if (!is_quasar_compute) {
        return mode;
    }
    static const char* trisc_define_names[] = {"TRISC_UNPACK", "TRISC_MATH", "TRISC_PACK", "TRISC_ISOLATE_SFPU"};
    std::ifstream kscan(src_path);
    if (!kscan) {
        throw std::runtime_error("detect_quasar_trisc_mode: cannot read " + src_path);
    }
    std::string kcontent((std::istreambuf_iterator<char>(kscan)), std::istreambuf_iterator<char>());
    for (int t = 0; t < 4 && !mode.needs_trisc_compile; t++) {
        if (kcontent.find(trisc_define_names[t]) != std::string::npos) {
            mode.needs_trisc_compile = true;
        }
    }
    if (!mode.needs_trisc_compile && kcontent.find("TRISC_ID") != std::string::npos) {
        mode.needs_runtime_trisc = true;
    }
    return mode;
}

static void collect_kernels(
    detail::ProgramImpl& impl,
    uint32_t num_dram_channels,
    uint32_t num_l1_banks,
    const std::string& worker_col_map_str,
    const std::string& worker_row_map_str,
    uint32_t emule_sem_base,
    const std::string& extra_inc,
    std::map<CoreCoord, std::vector<PendingKernelInfo>>& pending_core_kernels,
    std::map<std::string, DeferredCompile>& deferred_compiles,
    std::unordered_map<std::string, std::function<void()>>& resolved_fns,
    std::vector<std::string>& inline_src_temps) {
    static const char* trisc_define_names[] = {"TRISC_UNPACK", "TRISC_MATH", "TRISC_PACK", "TRISC_ISOLATE_SFPU"};

    const auto& hal = MetalContext::instance().hal();
    const uint32_t num_pct = hal.get_programmable_core_type_count();
    for (uint32_t pct = 0; pct < num_pct; ++pct) {
        auto& kernels = impl.get_kernels(pct);
        // (kernel_id, logical_core) → kg: a kernel that runs on cores in
        // multiple KGs has distinct per-KG launch_msg layouts, so keying by
        // kernel alone picks the wrong one for half the cores.
        std::map<std::pair<KernelHandle, CoreCoord>, KernelGroup*> kernel_core_to_kg;
        for (const auto& kg : impl.get_kernel_groups(pct)) {
            for (const auto& cr : kg->core_ranges.ranges()) {
                for (auto x = cr.start_coord.x; x <= cr.end_coord.x; ++x) {
                    for (auto y = cr.start_coord.y; y <= cr.end_coord.y; ++y) {
                        CoreCoord lc(x, y);
                        for (auto kid : kg->kernel_ids) {
                            kernel_core_to_kg.emplace(std::make_pair(kid, lc), kg.get());
                        }
                    }
                }
            }
        }
        for (auto& [kernel_id, kernel] : kernels) {
            const auto& ksrc = kernel->kernel_source();
            std::string src_path = resolve_kernel_source_path(ksrc, inline_src_temps);

            // Thread each kernel's configured include roots into its JIT -I flags.
            // Kernels can declare extra include paths (Kernel::process_include_paths)
            // so root-rooted includes resolve at compile time; silicon's build wires
            // these through the compiler include dirs, so mirror that here.
            std::string kernel_extra_inc = extra_inc;
            kernel->process_include_paths([&kernel_extra_inc](const std::string& p) {
                kernel_extra_inc += " -I\"" + p + "\"";
            });

            auto compile_args = kernel->compile_time_args();
            auto named_compile_args = kernel->named_compile_time_args();
            // Capture the namespace-structured CT + RT args so we can emit
            // `named_args_generated.h` into the JIT temp dir. Silicon's build
            // emits this header via jit_build/genfiles.cpp; emule's JIT path
            // bypasses that, so we mirror the emission inside jit_compile_kernel.
            NamedCTArgNamespaces named_ct_arg_namespaces;
            kernel->process_named_ct_arg_namespaces([&named_ct_arg_namespaces](const NamedCTArgNamespaces& namespaces) {
                named_ct_arg_namespaces = namespaces;
            });
            NamedRuntimeArgNamespaces named_runtime_arg_namespaces;
            kernel->process_named_runtime_args(
                [&named_runtime_arg_namespaces](const NamedRuntimeArgNamespaces& namespaces) {
                    named_runtime_arg_namespaces = namespaces;
                });
            auto defines = build_kernel_defines(
                *kernel, impl, num_dram_channels, num_l1_banks, worker_col_map_str, worker_row_map_str, emule_sem_base);

            // Tensix/compute kernels use bits 8+ in the DFB RISC mask (TENSIX_RISC_OFFSET),
            // while DM kernels use bits 0-7 directly.
            bool is_tensix = (kernel->get_kernel_processor_class() == HalProcessorClassType::COMPUTE);
            auto* qdm = dynamic_cast<experimental::quasar::QuasarDataMovementKernel*>(kernel.get());
            auto* qck = dynamic_cast<experimental::quasar::QuasarComputeKernel*>(kernel.get());
            bool is_quasar_compute = is_tensix && (qck != nullptr);

            // Issue tenstorrent/tt-emule#24: emule runs all three TRISC code paths
            // in a single unified compute thread (no separate UNPACK/MATH/PACK
            // RISC cores). tt-mlir-generated D2M kernels guard hardware code
            // paths with `#ifdef TRISC_UNPACK|MATH|PACK`; without these defines
            // helpers like `experimental::write_row_mask_tile` compile to empty
            // bodies and the downstream `where_tile` reads stale data. Define
            // all three so the kernel's `#ifdef TRISC_*` blocks execute exactly
            // once on the unified thread.
            if (is_tensix && !is_quasar_compute) {
                defines["TRISC_UNPACK"] = "1";
                defines["TRISC_MATH"] = "1";
                defines["TRISC_PACK"] = "1";
            }

            // Metal 2.0 bindings — same across this Kernel's TRISC variants, so
            // capture the cache-key suffix once and append it to every variant key.
            Metal2BindingsSnapshot bindings = build_metal2_snapshot(*kernel);
            const std::string metal2_key_suffix = bindings.cache_key_suffix();

            // COMPILE_FOR_{TRISC,BRISC,NCRISC} defines — silicon's per-RISC
            // kernel build sets exactly one of these. Kernel-author API
            // headers use them to pick the right include chain and to define
            // `is_brisc` / `is_ncrisc` / `is_trisc` constexpr bools; without
            // them, `SelectByRISCV<>` aliases fail to resolve. Emule runs all
            // RISCs in one unified thread, so we set the corresponding macro
            // based on the kernel's processor class.
            if (is_tensix) {
                defines["COMPILE_FOR_TRISC"] = "1";
            } else if (auto* dm_kernel = dynamic_cast<DataMovementKernel*>(kernel.get()); dm_kernel != nullptr) {
                auto cfg_variant = dm_kernel->config();
                const auto& cfg = std::get<DataMovementConfig>(cfg_variant);
                switch (cfg.processor) {
                    case DataMovementProcessor::RISCV_0: defines["COMPILE_FOR_BRISC"] = "1"; break;
                    case DataMovementProcessor::RISCV_1: defines["COMPILE_FOR_NCRISC"] = "1"; break;
                    default: break;
                }
            }

            // Helper: compute cache key from a defines map (preserves upstream's sorted
            // iteration of named_compile_args and defines for key stability).
            auto compute_cache_key = [&](const std::map<std::string, std::string>& defs) -> std::string {
                std::string key;
                if (ksrc.source_type_ == KernelSource::FILE_PATH) {
                    key = src_path;
                } else {
                    char hex[FNV_HEX_BUF_SIZE];
                    std::snprintf(hex, sizeof(hex), "%016lx", fnv1a_hash(ksrc.source_));
                    key = std::string("inline:") + hex;
                }
                for (auto v : compile_args) {
                    key += ":" + std::to_string(v);
                }
                std::vector<std::pair<std::string, uint32_t>> sorted_named(
                    named_compile_args.begin(), named_compile_args.end());
                std::sort(sorted_named.begin(), sorted_named.end());
                for (const auto& [k, v] : sorted_named) {
                    key += ":N" + k + "=" + std::to_string(v);
                }
                for (const auto& [k, v] : defs) {
                    key += ":" + k + "=" + v;
                }
                key += metal2_key_suffix;
                // Per-kernel include roots (Kernel::process_include_paths) change
                // which headers resolve, and therefore the compiled artifact, so
                // fold them into the key. Without this, two kernels that share
                // src_path/compile_args/named_compile_args/defines but differ in
                // include configuration would alias in the JIT cache (in-memory
                // g_jit_cache and deferred_compiles) and a disk-cached .so built
                // under a different include config could be reused. Kernels with no
                // extra include roots keep their previous key (backward-compatible).
                if (kernel_extra_inc != extra_inc) {
                    char inc_hex[FNV_HEX_BUF_SIZE];
                    std::snprintf(inc_hex, sizeof(inc_hex), "%016lx", fnv1a_hash(kernel_extra_inc));
                    key += ":inc";
                    key += inc_hex;
                }
                // ASAN builds are -g; keep their cache distinct from the lean build.
                if (emule_asan_enabled()) {
                    key += ":asan_g";
                }
                return key;
            };

            auto register_cache_key = [&](const std::string& key,
                                          const std::map<std::string, std::string>& defs) {
                std::lock_guard<std::mutex> lock(g_jit_cache_mutex);
                auto it = g_jit_cache.find(key);
                if (it != g_jit_cache.end()) {
                    resolved_fns[key] = it->second;
                } else if (
                    resolved_fns.find(key) == resolved_fns.end() &&
                    deferred_compiles.find(key) == deferred_compiles.end()) {
                    std::string mtime_path = (ksrc.source_type_ == KernelSource::FILE_PATH) ? src_path : "";
                    auto disk_fn = disk_cache_lookup(key, mtime_path);
                    if (disk_fn) {
                        resolved_fns[key] = disk_fn;
                        g_jit_cache[key] = disk_fn;
                    } else {
                        deferred_compiles[key] = DeferredCompile{
                            src_path,
                            compile_args,
                            named_compile_args,
                            named_ct_arg_namespaces,
                            named_runtime_arg_namespaces,
                            defs,
                            kernel_extra_inc,
                            bindings};
                    }
                }
            };

            // For Quasar compute kernels with TRISC guards, compile 4 variants
            // (UNPACK, MATH, PACK, ISOLATE_SFPU) — each has a different cache key.
            // Kernels without TRISC guards (e.g. DFB compute bridges) compile once.
            TriscMode trisc = detect_quasar_trisc_mode(is_quasar_compute, src_path);
            std::vector<std::string> variant_cache_keys;
            bool run_all_variants = false;
            if (trisc.needs_trisc_compile) {
                for (int t = 0; t < 4; t++) {
                    auto trisc_defs = defines;
                    trisc_defs[trisc_define_names[t]] = "1";
                    std::string key = compute_cache_key(trisc_defs);
                    register_cache_key(key, trisc_defs);
                    variant_cache_keys.push_back(std::move(key));
                }
                run_all_variants = true;
            } else {
                std::string key = compute_cache_key(defines);
                register_cache_key(key, defines);
                if (trisc.needs_runtime_trisc) {
                    variant_cache_keys.assign(4, key);
                    run_all_variants = true;
                } else {
                    variant_cache_keys.push_back(std::move(key));
                }
            }

            ProcIdList procs = compute_proc_ids_and_thread_count(*kernel, qdm, qck);

            const uint32_t processor_index = hal.get_processor_index(
                kernel->get_kernel_programmable_core_type(),
                kernel->get_kernel_processor_class(),
                kernel->get_kernel_processor_type(0));

            const auto& core_range_set = kernel->core_range_set();
            for (const auto& core_range : core_range_set.ranges()) {
                for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; ++x) {
                    for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; ++y) {
                        CoreCoord logical_core(x, y);
                        // KG lookup is per (kernel, logical_core): same kernel
                        // on different cores may sit in different KGs with
                        // distinct kernel_config layouts.
                        auto kg_it = kernel_core_to_kg.find({kernel_id, logical_core});
                        uint32_t kernel_config_base = 0;
                        uint16_t rta_off = kRtaCrtaNoArgsSentinel;
                        uint16_t crta_off = kRtaCrtaNoArgsSentinel;
                        if (kg_it != kernel_core_to_kg.end()) {
                            auto kc = kg_it->second->launch_msg.view().kernel_config();
                            kernel_config_base = static_cast<uint32_t>(kc.kernel_config_base()[pct]);
                            auto rta = kc.rta_offset()[processor_index];
                            rta_off = rta.rta_offset();
                            crta_off = rta.crta_offset();
                        }
                        // Runtime-arg values (unique + common) for this kernel on this
                        // core; the Object-Intent check uses them to find its I/O tensors
                        // (see ObjectIntentTracker::pre_launch_snapshot). Build once, copy.
                        std::vector<uint32_t> rt_arg_values;
                        if (kernel->cores_with_runtime_args().count(logical_core) != 0) {
                            const auto& ra = kernel->runtime_args(logical_core);
                            rt_arg_values.insert(rt_arg_values.end(), ra.begin(), ra.end());
                        }
                        const auto& cra = kernel->common_runtime_args();
                        rt_arg_values.insert(rt_arg_values.end(), cra.begin(), cra.end());

                        uint8_t tidx = 0;
                        for (uint8_t proc_id : procs.proc_ids) {
                            pending_core_kernels[logical_core].push_back(PendingKernelInfo{
                                variant_cache_keys,
                                run_all_variants,
                                proc_id,
                                tidx++,
                                is_tensix,
                                procs.num_threads,
                                kernel_config_base,
                                rta_off,
                                crta_off,
                                rt_arg_values,
                                src_path});
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// jit_compile_pending: Compile cache misses in parallel, resolve all fns.
// ---------------------------------------------------------------------------
// Global compile-once registry: dedups kernel compilation across jit_compile_pending's parallel compile
// tasks. The first task to need a key publishes a shared_future the rest reuse, so each kernel compiles
// exactly once to a unique tmp with an atomic rename (racing clang on one `.so.tmp` would corrupt it).
// Shared across programs, hence the mutex. See tt-emule docs/metal-integration.md.
static std::mutex g_compile_inflight_mutex;
static std::unordered_map<std::string, std::shared_future<std::function<void()>>> g_compile_inflight;
static std::atomic<uint64_t> g_compile_tmp_seq{0};

// Max jit_compile_kernel calls allowed to run concurrently. Each in-flight compile
// holds a burst of open file descriptors (patched-header mirror writes, header reads,
// the clang subprocess pipes — order tens of fds each), so an UNBOUNDED std::async
// fan-out over every cache-miss makes peak fd use scale with the kernel count.
// A large mesh program (8-chip loudbox on the full 14x10 grid compiles ~hundreds
// of distinct kernels) then blows past a container's RLIMIT_NOFILE soft limit
// (commonly 1024) → open() fails mid-compile → "kernel_patcher: cannot read/write"
// (and the failing file varies run-to-run with the interleaving). A dev box with a
// high fd limit (e.g. 65536) never hits it, so this is CI/container-only. Bounding
// concurrent compiles keeps peak fds (and concurrent clang processes) under the
// limit. Note this bounds concurrent *compiles*, not the thread count: std::async
// still spawns one thread per cache-miss which then blocks on the gate — the fd/
// clang-process footprint is what EMFILE cares about here.
static unsigned jit_compile_concurrency_cap() {
    // First, opportunistically raise the soft fd limit to the hard limit — unprivileged
    // and enough on its own in most containers (large hard limit), so the throttle
    // rarely engages on well-provisioned hosts. Kept as a floor because some containers
    // pin the hard limit at 1024 too; derive the cap from the (possibly raised) soft limit.
    struct rlimit rl {};
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur != rl.rlim_max) {
        rl.rlim_cur = rl.rlim_max;
        setrlimit(RLIMIT_NOFILE, &rl);
        getrlimit(RLIMIT_NOFILE, &rl);  // re-read the effective soft limit
    }

    unsigned cap = std::max(1u, std::thread::hardware_concurrency());
    if (rl.rlim_cur != RLIM_INFINITY) {
        // Budget the fd limit across concurrent compiles. kReserve covers the process's
        // baseline fds (libs, python, fibers, the cached kernel .sos); kFdsPerCompile is
        // an estimate of the fds one compile holds at its peak (patched-header writes +
        // header reads + clang pipes). Both are conservative estimates, not measured; the
        // soft→hard raise above and TT_EMULE_JIT_COMPILE_JOBS relieve any over-throttling.
        constexpr rlim_t kReserve = 512;
        constexpr rlim_t kFdsPerCompile = 48;
        rlim_t budget = rl.rlim_cur > kReserve ? (rl.rlim_cur - kReserve) / kFdsPerCompile : 1;
        cap = std::min<unsigned>(cap, std::max<rlim_t>(1, budget));
    }

    // Explicit override for tuning/debug. Only a fully-parsed, in-range, nonzero value
    // wins; anything malformed (sign, trailing junk, overflow, 0) keeps the fd-derived
    // cap rather than silently disabling the gate (e.g. "-1" must not become UINT_MAX).
    if (const char* s = std::getenv("TT_EMULE_JIT_COMPILE_JOBS")) {
        errno = 0;
        char* end = nullptr;
        unsigned long v = std::strtoul(s, &end, 10);
        if (end != s && *end == '\0' && errno == 0 && v > 0 && v <= std::numeric_limits<unsigned>::max()) {
            cap = static_cast<unsigned>(v);
        }
    }
    return std::max(1u, cap);
}

// Counting gate limiting concurrently-executing jit_compile_kernel calls to the
// fd-safe cap above. Process-global so it also bounds the total across the several
// jit_compile_pending calls a multi-chip mesh setup may run in parallel. Each
// compile acquires a slot for the duration of its (fd-heavy) work and releases it
// via RAII, so peak open fds stay bounded regardless of the kernel count.
static std::counting_semaphore<>& compile_slots() {
    static std::counting_semaphore<> slots(jit_compile_concurrency_cap());
    return slots;
}

static void jit_compile_pending(
    std::map<std::string, DeferredCompile>& deferred_compiles,
    std::unordered_map<std::string, std::function<void()>>& resolved_fns,
    std::vector<std::string>& inline_src_temps) {
    if (!deferred_compiles.empty()) {
        log_info(tt::LogMetal, "JIT parallel compile: {} unique kernels to compile", deferred_compiles.size());

        std::vector<std::pair<std::string, std::shared_future<std::function<void()>>>> futures;
        futures.reserve(deferred_compiles.size());

        for (auto& [key, dc] : deferred_compiles) {
            std::shared_future<std::function<void()>> fut;
            {
                std::lock_guard<std::mutex> lock(g_compile_inflight_mutex);
                // Another thread may have already finished this key (published to g_jit_cache) or be
                // mid-compile (published an inflight future) since we built deferred_compiles.
                {
                    std::lock_guard<std::mutex> clock(g_jit_cache_mutex);
                    auto cit = g_jit_cache.find(key);
                    if (cit != g_jit_cache.end()) {
                        resolved_fns[key] = cit->second;
                        continue;
                    }
                }
                auto iit = g_compile_inflight.find(key);
                if (iit != g_compile_inflight.end()) {
                    fut = iit->second;  // reuse the in-progress compile launched by another thread
                } else {
                    std::string cache_path = disk_cache_so_path(key);
                    DeferredCompile dc_copy = dc;  // own a copy: the future may outlive this call's map
                    fut = std::async(std::launch::async, [dc_copy, cache_path]() {
                              // Bound concurrent compiles to the fd-safe cap: a slot is
                              // held for the whole (fd-heavy) compile and released even if
                              // it throws. std::async spawns the thread eagerly, but the
                              // fd-consuming work waits here until a slot frees.
                              compile_slots().acquire();
                              struct SlotGuard {
                                  ~SlotGuard() { compile_slots().release(); }
                              } slot_guard;
                              std::string tmp_path = cache_path + ".tmp." + std::to_string(::getpid()) + "." +
                                                     std::to_string(g_compile_tmp_seq.fetch_add(1));
                              auto fn = jit_compile_kernel(
                                  dc_copy.src_path,
                                  dc_copy.compile_args,
                                  dc_copy.named_compile_args,
                                  dc_copy.named_ct_arg_namespaces,
                                  dc_copy.named_runtime_arg_namespaces,
                                  dc_copy.defines,
                                  dc_copy.extra_inc,
                                  dc_copy.bindings,
                                  tmp_path);
                              std::filesystem::rename(tmp_path, cache_path);
                              return fn;
                          }).share();
                    g_compile_inflight[key] = fut;
                }
            }
            futures.emplace_back(key, fut);
        }

        for (auto& [key, fut] : futures) {
            auto fn = fut.get();
            resolved_fns[key] = fn;
            {
                std::lock_guard<std::mutex> lock(g_jit_cache_mutex);
                g_jit_cache[key] = fn;
            }
            std::lock_guard<std::mutex> lock(g_compile_inflight_mutex);
            g_compile_inflight.erase(key);
        }
    }

    // Clean up inline source temp files
    if (!std::getenv("TT_EMULE_KEEP_JIT_SRC")) {
        for (auto& tmp : inline_src_temps) {
            std::filesystem::remove(tmp);
        }
    } else {
        for (auto& tmp : inline_src_temps) {
            fprintf(stderr, "[EMULE-DBG] kept JIT source: %s\n", tmp.c_str());
        }
    }
}

// ---------------------------------------------------------------------------
// build_core_map: Build physical {x,y} → tt_emule::Core* for NOC resolution.
// Cached per device_id since chip topology doesn't change between calls.
// ---------------------------------------------------------------------------
// Per-device physical {x,y}->Core* maps, at file scope so the fabric teleport hooks below can resolve a
// remote chip's core (its map is built by that device's own concurrent run). See tt-emule docs/fabric-ccl-emulation.md.
static std::mutex g_core_map_mutex;
static std::unordered_map<uint32_t, std::shared_ptr<std::unordered_map<uint64_t, tt_emule::Core*>>>
    g_core_map_cache;
// The SWEmuleChip each cached core_map was built against. A device close+reopen mints
// a NEW SWEmuleChip with fresh per-core L1 mmaps (single-process-galaxy L1 model), so a
// core_map cached from the prior chip holds Core* into a now-disjoint L1 region. The NOC
// path (this map) would then resolve a worker's semaphore to a different L1 backing than
// that worker's own fiber (built from the CURRENT chip in setup_core_state) reads —
// cross-core sems never observed → deadlock. Rebuild when the chip identity changes.
static std::unordered_map<uint32_t, tt::umd::SWEmuleChip*> g_core_map_sw_emu;

static std::unordered_map<uint64_t, tt_emule::Core*>* build_core_map(
    tt::umd::SWEmuleChip* sw_emu, IDevice* device, ChipId device_id) {
    std::lock_guard<std::mutex> lock(g_core_map_mutex);
    auto& core_map = g_core_map_cache[device_id];
    if (core_map && g_core_map_sw_emu[device_id] != sw_emu) {
        core_map.reset();  // stale: built against a different (now-replaced) SWEmuleChip
    }
    if (!core_map && sw_emu) {
        g_core_map_sw_emu[device_id] = sw_emu;
        core_map = std::make_shared<std::unordered_map<uint64_t, tt_emule::Core*>>();
        // Add ALL worker cores from the device grid
        auto grid = device->compute_with_storage_grid_size();
        for (uint32_t lx = 0; lx < grid.x; lx++) {
            for (uint32_t ly = 0; ly < grid.y; ly++) {
                auto phys = device->virtual_core_from_logical_core(CoreCoord(lx, ly), tt::CoreType::WORKER);
                auto* core = sw_emu->get_core(tt_xy_pair(phys.x, phys.y));
                uint64_t key = (uint64_t(phys.x) << 32) | phys.y;
                (*core_map)[key] = core;
            }
        }
        // Add DRAM cores. Post-uplift get_core() is worker-only (mints a bogus
        // CoreRole::WORKER for a DRAM coord), so back DRAM per physical channel via
        // get_dram_channel_backing(channel): every NOC endpoint of a channel must
        // alias onto that one CoreRole::DRAM core so host writes and kernel NOC reads
        // hit the same memory. get_dram_cores() groups by LOGICAL channel (outer index).
        auto& umd_soc = sw_emu->get_soc_descriptor();
        auto dram_cores = umd_soc.get_dram_cores();
        for (uint32_t ch = 0; ch < dram_cores.size(); ch++) {
            auto* core = sw_emu->get_dram_channel_backing(ch);
            for (auto& dc : dram_cores[ch]) {
                uint64_t key = (uint64_t(dc.x) << 32) | dc.y;
                (*core_map)[key] = core;
            }
        }
        // Add DRAM cores (metal_SocDescriptor preferred worker coords). Register BOTH
        // NOC0 and NOC1 preferred coords (on Wormhole they differ per view) so
        // __emule_resolve_noc_addr can route either. Key the backing by the coord's
        // umd LOGICAL channel (the physical DRAM channel) — not the metal dram-view
        // index: several views alias one physical channel (at different offsets), and
        // the host write path resolves the same LOGICAL channel. Keying by view index
        // would split one channel across multiple backings → host/kernel read mismatch.
        {
            auto& umd = sw_emu->get_soc_descriptor();
            auto& msoc = MetalContext::instance().get_cluster().get_soc_desc(device_id);
            for (uint32_t view = 0; view < msoc.get_num_dram_views() && view < MAX_NUM_BANKS; view++) {
                for (uint32_t noc = 0; noc < NUM_NOCS; noc++) {
                    auto dc = msoc.get_preferred_worker_core_for_dram_view(view, noc);
                    auto lg = umd.translate_coord_to(
                        tt_xy_pair(dc.x, dc.y), CoordSystem::TRANSLATED, CoordSystem::LOGICAL);
                    auto* core = sw_emu->get_dram_channel_backing(static_cast<uint32_t>(lg.x));
                    uint64_t key = (uint64_t(dc.x) << 32) | dc.y;
                    (*core_map)[key] = core;
                }
            }
        }
    } else if (!core_map) {
        core_map = std::make_shared<std::unordered_map<uint64_t, tt_emule::Core*>>();
    }
    return core_map.get();
}

// ---------------------------------------------------------------------------
// Fabric route table: resolve a send's FINAL destination chip by a static walk of the
// control-plane mesh graph (no multi-hop router sim). 1D dst = (src, dir, distance);
// 2D dst = explicit FabricNodeId. See tt-emule docs/fabric-ccl-emulation.md.
// ---------------------------------------------------------------------------
static std::mutex g_fabric_route_mutex;
// (src_chip << 3 | dir) -> ordered chips at distance 1,2,... in that direction (cached; topology is static).
static std::unordered_map<uint32_t, std::vector<uint32_t>> g_fabric_walk_cache;

// Immediate same-mesh neighbor physical chip of `chip` in `dir`, or -1 if none.
static int __emule_fabric_dir_neighbor(
    tt::tt_fabric::ControlPlane& cp, uint32_t chip, tt::tt_fabric::RoutingDirection dir) {
    try {
        auto node = cp.get_fabric_node_id_from_physical_chip_id(static_cast<ChipId>(chip));
        auto neighbors = cp.get_chip_neighbors(node, dir);
        for (auto& [mesh, chips] : neighbors) {
            if (!chips.empty()) {
                return static_cast<int>(cp.get_physical_chip_id_from_fabric_node_id(
                    tt::tt_fabric::FabricNodeId(mesh, static_cast<std::uint32_t>(chips.front()))));
            }
        }
    } catch (...) {
        // control plane unavailable / chip not in graph — caller falls back to the neighbor table
    }
    return -1;
}

// Ordered chips reachable from `src` at distance 1,2,... in `dir` (line or ring), cached.
static const std::vector<uint32_t>& __emule_fabric_walk(uint32_t src, tt::tt_fabric::RoutingDirection dir) {
    std::lock_guard<std::mutex> lock(g_fabric_route_mutex);
    uint32_t key = (src << 3) | static_cast<uint32_t>(dir);
    auto it = g_fabric_walk_cache.find(key);
    if (it != g_fabric_walk_cache.end()) {
        return it->second;
    }
    std::vector<uint32_t> walk;
    auto& cp = MetalContext::instance().get_control_plane();
    uint32_t cur = src;
    for (int hop = 0; hop < 64; ++hop) {  // 64 = chip-count backstop (loudbox = 8)
        int nxt = __emule_fabric_dir_neighbor(cp, cur, dir);
        if (nxt < 0 || static_cast<uint32_t>(nxt) == src) {
            break;  // line end, or ring wrapped back to the source
        }
        walk.push_back(static_cast<uint32_t>(nxt));
        cur = static_cast<uint32_t>(nxt);
    }
    return g_fabric_walk_cache.emplace(key, std::move(walk)).first->second;
}

// ===========================================================================
// Fabric teleport hooks (multi-chip CCL): decode the real-layout packet header,
// resolve the destination chip, and apply the terminal NOC command directly into
// that chip's L1. Delivery is synchronous; the peer-chip consumer observes it via
// its semaphore wait. See tt-emule docs/fabric-ccl-emulation.md.
// ---------------------------------------------------------------------------

// Resolve (noc_addr) -> host pointer on an arbitrary chip, mirroring __emule_resolve_noc_addr but
// against the destination chip's cached core map (already built by that chip's launch).
extern "C" uint8_t* __emule_fabric_resolve_remote(uint32_t dst_chip, uint64_t noc_addr) {
    emule_require_self(__func__);
    std::lock_guard<std::mutex> lock(g_core_map_mutex);
    static const bool rdbg = std::getenv("EMULE_FABRIC_DEBUG") != nullptr;
    auto it = g_core_map_cache.find(dst_chip);
    if (it == g_core_map_cache.end() || !it->second) {
        if (rdbg) {
            fprintf(stderr, "[EMULE_FABRIC]   resolve_remote: NO CORE MAP for dst_chip=%u (cache has %zu chips)\n",
                    dst_chip, g_core_map_cache.size());
        }
        return nullptr;
    }
    auto& m = *it->second;
    uint32_t noc_x = (noc_addr >> NOC_LOCAL_BITS) & NOC_NODE_MASK;
    uint32_t noc_y = (noc_addr >> (NOC_LOCAL_BITS + NOC_NODE_ID_BITS)) & NOC_NODE_MASK;
    uint64_t local_addr = noc_addr & NOC_LOCAL_MASK;

    auto find_core = [&](uint32_t x, uint32_t y) {
        return m.find((uint64_t(x) << 32) | y);
    };

    // (noc_x,noc_y) are the SOURCE chip's coords (get_noc_addr packs the caller core's); resolve against
    // the destination chip's map. Cross-chip src->logical->dst translation applies ONLY to WORKER cores
    // (harvesting can shift them per chip); DRAM/ETH coords are chip-invariant, so translating them would
    // alias distinct banks. Try verbatim first; translate only if verbatim is a WORKER or missed.
    // See tt-emule docs/fabric-ccl-emulation.md.
    auto cit = find_core(noc_x, noc_y);
    const uint32_t src_chip = __emule_self->chip_id;
    const bool verbatim_is_worker =
        (cit != m.end() && cit->second->role() == tt_emule::CoreRole::WORKER);
    if (src_chip != dst_chip && (verbatim_is_worker || cit == m.end())) {
        auto* src_obj = get_sw_emulated_chip(src_chip);
        auto* dst_obj = get_sw_emulated_chip(dst_chip);
        if (src_obj != nullptr && dst_obj != nullptr) {
            try {
                auto logical = src_obj->get_soc_descriptor().translate_coord_to(
                    tt_xy_pair(noc_x, noc_y), CoordSystem::TRANSLATED, CoordSystem::LOGICAL);
                auto dst_xy = dst_obj->get_soc_descriptor().translate_coord_to(
                    tt_xy_pair(logical.x, logical.y), CoordSystem::LOGICAL, CoordSystem::TRANSLATED);
                auto t = find_core(static_cast<uint32_t>(dst_xy.x), static_cast<uint32_t>(dst_xy.y));
                if (t != m.end() && t->second->role() == tt_emule::CoreRole::WORKER) {
                    cit = t;  // harvesting-correct worker on the dest chip
                }
            } catch (...) {
                // translation unavailable — keep the verbatim result
            }
        }
    }
    if (cit == m.end()) {
        if (rdbg) {
            fprintf(stderr, "[EMULE_FABRIC]   resolve_remote: dst_chip=%u has map (%zu cores) but core (%u,%u) NOT FOUND\n",
                    dst_chip, m.size(), noc_x, noc_y);
        }
        return nullptr;
    }
    uint32_t offset = (cit->second->role() == tt_emule::CoreRole::WORKER)
                          ? (static_cast<uint32_t>(local_addr) & L1_SLOT_MASK)
                          : static_cast<uint32_t>(local_addr);
    return cit->second->l1_ptr(offset);
}

// Destination chip for a fabric send from src_chip: the single ethernet-connected neighbor of a
// directly-connected 2-chip system, from the cluster descriptor. See tt-emule docs/fabric-ccl-emulation.md.
extern "C" uint32_t __emule_fabric_neighbor(uint32_t src_chip) {
    auto ids = MetalContext::instance().get_cluster().get_ethernet_connected_device_ids(src_chip);
    if (!ids.empty()) {
        return *ids.begin();
    }
    return src_chip;
}

// emule route metadata, keyed by packet-header L1-alias address: the fabric_set_*_route shims record the
// kernel's semantic dst (2D FabricNodeId, 1D hop distance, or line-multicast extent) here; the teleport
// resolves it to physical chip(s). KIND constants KEEP IN SYNC with the shim. See tt-emule docs/fabric-ccl-emulation.md.
namespace emule_route_kind {
constexpr uint32_t UNSET = 0, UNICAST_1D = 1, UNICAST_2D = 2, MCAST_1D = 3, MCAST_2D = 4;
}
struct EmuleRoute {
    uint32_t kind = 0, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0;
    uint32_t dir_index = 0;  // 1D: which of the worker's connections (fwd=0/bwd=1), set at send time
    // Mux-path direction hint (preferred over the range-match heuristic), set at send time:
    uint32_t mux_x = 0xFFFF, mux_y = 0xFFFF;   // worker's mux NOC (TRANSLATED) coords (fabric MUX path)
};
static std::mutex g_route_meta_mu;
// Keyed by the header's FULL host pointer (bridge_l1 + offset). Post-offset-migration a packet
// header's L1 offset is both chip- AND core-agnostic — it is 0-based within each core's L1, so the
// same offset recurs on every core of every chip. A (chip, offset) key would collide across cores
// of one chip (one core's route overwriting another's → wrong-chip delivery → PCC fail / a fiber
// waiting on an atomic-inc that lands elsewhere → quiescent deadlock). The baseline's key was the
// header's host pointer, which is inherently unique per (chip, core, offset) because every core's
// L1 is a distinct mapping; reconstruct that full (untruncated) pointer here — untruncated so it
// stays unique above 4 GB, and host-side-only (a runner map key, never a kernel/L1 value). Both set
// (shim offset) and read (teleport) run on the same fiber → same bridge_l1 → same key.
static std::unordered_map<uint64_t, EmuleRoute> g_route_meta;
static inline uint64_t emule_route_key(uint32_t hdr_off) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(__emule_self->bridge_l1 + hdr_off));
}

extern "C" void __emule_fabric_set_route(
    uint32_t hdr, uint32_t kind, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e, uint32_t f) {
    emule_require_self(__func__);  // keys through __emule_self->bridge_l1 via emule_route_key
    std::lock_guard<std::mutex> lk(g_route_meta_mu);
    auto& r = g_route_meta[emule_route_key(hdr)];
    r.kind = kind; r.a = a; r.b = b; r.c = c; r.d = d; r.e = e; r.f = f;  // dir_index set separately at send
}

// Record a 1D send's per-connection direction signals: the fwd/bwd conn_index (direct path) and the
// worker's mux NOC coords (MUX path); 0xFFFF means unset. See tt-emule docs/fabric-ccl-emulation.md.
extern "C" void __emule_fabric_set_route_dir(
    uint32_t hdr, uint32_t conn_index, uint32_t mux_x, uint32_t mux_y) {
    emule_require_self(__func__);  // keys through __emule_self->bridge_l1 via emule_route_key
    std::lock_guard<std::mutex> lk(g_route_meta_mu);
    auto& r = g_route_meta[emule_route_key(hdr)];
    r.dir_index = conn_index;
    r.mux_x = mux_x;
    r.mux_y = mux_y;
}

// Carry a stamped route from a source header address to a destination when the header bytes are copied
// (a worker staging a packet header into a forwarder relay slot). On silicon the routing fields ride inside
// the header, so a byte copy carries them for free; emule keeps them in the address-keyed side-table, so the
// copy must replicate the entry. No-op when src carries no route. src_key/dst_key are 0-based L1 offsets
// (the fabric shim passes bridge_l1-relative offsets); widen through emule_route_key to match the set/read
// sides. See tt-emule docs/fabric-ccl-emulation.md.
extern "C" void __emule_fabric_route_follow(uint32_t src_key, uint32_t dst_key) {
    emule_require_self(__func__);  // keys through __emule_self->bridge_l1 via emule_route_key
    if (src_key == dst_key) {
        return;
    }
    std::lock_guard<std::mutex> lk(g_route_meta_mu);
    auto it = g_route_meta.find(emule_route_key(src_key));
    if (it == g_route_meta.end()) {
        return;  // src carries no route — not a packet-header copy; nothing to follow.
    }
    const EmuleRoute r = it->second;  // copy before the insert below can rehash/invalidate `it`.
    g_route_meta[emule_route_key(dst_key)] = r;
}

// Fabric connection routes recorded host-side by append_fabric_connection_rt_args: for 1D the dst chip is
// bound to the connection, not the header, so the host records each connection's direction + immediate
// neighbor here. See tt-emule docs/fabric-ccl-emulation.md.
struct ConnRoute {
    uint32_t dir;       // RoutingDirection (N/E/S/W)
    uint32_t neighbor;  // immediate neighbor physical chip
};
static std::mutex g_conn_route_mu;
// Keyed by SRC CHIP only (line direction is a per-chip property; the connection-owner core can differ from
// the sender, so a per-core key would miss). Deduped by direction; append order makes index 0=fwd, 1=bwd.
// See tt-emule docs/fabric-ccl-emulation.md.
static std::unordered_map<uint32_t, std::vector<ConnRoute>> g_conn_route;
// Per-op reset flag: cleared at each new op's first connection-record so a later op's different line
// orientation can't corrupt the src-keyed, direction-deduped table. See tt-emule docs/fabric-ccl-emulation.md.
static std::atomic<bool> g_conn_route_dirty{true};
// Per-worker resolved line direction, keyed (src<<32 | wx<<16 | wy): on the MUX path the sender carries no
// direction, so infer it once from a multicast's range and reuse for that worker's unicasts. Reset per op.
// See tt-emule docs/fabric-ccl-emulation.md.
static std::unordered_map<uint64_t, uint32_t> g_worker_dir;
// Per-mux-core line direction, keyed (src<<32 | logical_x<<16 | logical_y): the mux→EDM append records the
// mux's forwarding_direction; the teleport recovers it from the worker's carried mux NOC coords. Reset per
// op. See tt-emule docs/fabric-ccl-emulation.md.
static std::unordered_map<uint64_t, uint32_t> g_mux_dir;
static inline uint64_t __emule_worker_key(uint32_t src, uint32_t wx, uint32_t wy) {
    return (static_cast<uint64_t>(src) << 32) | (static_cast<uint64_t>(wx & 0xFFFF) << 16) | (wy & 0xFFFF);
}
extern "C" void __emule_fabric_record_conn(uint32_t src, uint32_t wx, uint32_t wy, uint32_t dir, uint32_t neighbor) {
    std::lock_guard<std::mutex> lk(g_conn_route_mu);
    if (g_conn_route_dirty.exchange(false)) {
        g_conn_route.clear();
        g_worker_dir.clear();
        g_mux_dir.clear();
    }
    // Record the connection-owner core's (the mux core, on the MUX path) direction, keyed by its LOGICAL
    // coords — before the per-direction dedup below, which is for the src-keyed g_conn_route only.
    g_mux_dir[__emule_worker_key(src, wx, wy)] = dir;
    auto& v = g_conn_route[src];
    for (const auto& c : v) {
        if (c.dir == dir) {
            return;  // already recorded this direction for src
        }
    }
    v.push_back(ConnRoute{dir, neighbor});
}

// Ordered ring members at distance 1,2,... from `src` starting in `start_dir`, by chaining the per-chip
// recorded ring neighbors (g_conn_route) — at each hop taking the neighbor that is NOT where we came from,
// so it follows the turning Hamiltonian cycle the compass walk can't. Returns empty if the chain is
// incomplete (caller falls back to the compass walk). See tt-emule docs/fabric-ccl-emulation.md.
static std::vector<uint32_t> __emule_fabric_walk_ring(uint32_t src, uint32_t start_dir) {
    std::vector<uint32_t> walk;
    std::lock_guard<std::mutex> lk(g_conn_route_mu);
    auto sit = g_conn_route.find(src);
    if (sit == g_conn_route.end()) {
        return walk;
    }
    int first = -1;
    for (const auto& c : sit->second) {
        if (c.dir == start_dir) {
            first = static_cast<int>(c.neighbor);
            break;
        }
    }
    if (first < 0) {
        return walk;  // start direction not recorded — caller falls back
    }
    walk.push_back(static_cast<uint32_t>(first));
    uint32_t prev = src, cur = static_cast<uint32_t>(first);
    for (int hop = 0; hop < 64; ++hop) {  // 64 = chip-count backstop
        auto cit = g_conn_route.find(cur);
        if (cit == g_conn_route.end()) {
            break;
        }
        int next = -1;
        for (const auto& c : cit->second) {
            if (c.neighbor != prev) {  // the ring edge that continues forward (not the one back to prev)
                next = static_cast<int>(c.neighbor);
                break;
            }
        }
        if (next < 0 || static_cast<uint32_t>(next) == src) {
            break;  // dead end, or the cycle closed back at the source
        }
        walk.push_back(static_cast<uint32_t>(next));
        prev = cur;
        cur = static_cast<uint32_t>(next);
    }
    return walk;
}

// Resolve the FINAL destination chip(s) for a send: one chip for unicast, the line members for a multicast.
// Gated by EMULE_FABRIC8 (off → legacy single neighbor). See tt-emule docs/fabric-ccl-emulation.md.
static std::vector<uint32_t> __emule_fabric_resolve_targets(const uint8_t* h, uint32_t src_chip) {
    static const bool fabric8 = std::getenv("EMULE_FABRIC8") != nullptr;
    if (!fabric8) {
        return {__emule_fabric_neighbor(src_chip)};
    }
    EmuleRoute r;
    {
        std::lock_guard<std::mutex> lk(g_route_meta_mu);
        // Round-trip h -> offset -> bridge_l1+offset via emule_route_key so the read key is derived
        // by the same helper as the set-side key (one source of truth for the key formula).
        auto it = g_route_meta.find(emule_route_key(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(h) - reinterpret_cast<uintptr_t>(__emule_self->bridge_l1))));
        if (it == g_route_meta.end()) {
            return {__emule_fabric_neighbor(src_chip)};  // unstamped (e.g. 1D direct, not yet wired)
        }
        r = it->second;
    }
    static const bool rdbg = std::getenv("EMULE_FABRIC_DEBUG") != nullptr;
    if (rdbg) {
        std::lock_guard<std::mutex> lk(g_conn_route_mu);
        auto cit = g_conn_route.find(src_chip);
        fprintf(stderr, "[EMULE_FABRIC]   resolve src=%u kind=%u a=%u b=%u ewns=%u/%u/%u/%u dir_idx=%u conns=%zu mux=(%u,%u)\n",
                src_chip, r.kind, r.a, r.b, r.c, r.d, r.e, r.f, r.dir_index,
                cit == g_conn_route.end() ? (size_t)0 : cit->second.size(), r.mux_x, r.mux_y);
    }
    auto& cp = MetalContext::instance().get_control_plane();
    if (r.kind == emule_route_kind::UNICAST_2D) {  // a=dst_dev, b=dst_mesh
        try {
            return {static_cast<uint32_t>(cp.get_physical_chip_id_from_fabric_node_id(
                tt::tt_fabric::FabricNodeId(tt::tt_fabric::MeshId{r.b}, r.a)))};
        } catch (...) {
        }
    } else if (r.kind == emule_route_kind::MCAST_2D) {
        // 2D line multicast: {c,d,e,f}={E,W,N,S} per-direction hop counts; walk each non-zero direction.
        using RD = tt::tt_fabric::RoutingDirection;
        const std::pair<RD, uint32_t> dirs[4] = {{RD::E, r.c}, {RD::W, r.d}, {RD::N, r.e}, {RD::S, r.f}};
        std::vector<uint32_t> tgts;
        for (const auto& [dir, hops] : dirs) {
            if (hops == 0) {
                continue;
            }
            const auto& walk = __emule_fabric_walk(src_chip, dir);
            for (uint32_t k = 0; k < hops && k < walk.size(); ++k) {
                tgts.push_back(walk[k]);
            }
        }
        if (!tgts.empty()) {
            return tgts;
        }
    } else if (r.kind == emule_route_kind::MCAST_1D || r.kind == emule_route_kind::UNICAST_1D) {
        // 1D MUX path carries no direction tag: infer the worker's direction from a multicast's range and
        // cache it per (src, worker_core) for that worker's unicasts. See tt-emule docs/fabric-ccl-emulation.md.
        emule_require_self(__func__);
        TT_FATAL(__emule_self->core != nullptr, "{}: fiber has no core context", __func__);
        const uint32_t wx = __emule_self->core->logical_x;
        const uint32_t wy = __emule_self->core->logical_y;
        const uint64_t wkey = __emule_worker_key(src_chip, wx, wy);
        std::vector<ConnRoute> conns;
        {
            std::lock_guard<std::mutex> lk(g_conn_route_mu);
            auto it = g_conn_route.find(src_chip);
            if (it != g_conn_route.end()) {
                conns = it->second;
            }
        }
        int dir = -1;
        // (1) Mux-core direction: translate the worker's mux NOC coords to the mux's LOGICAL core and look up
        // the direction the mux→EDM append recorded. Resolves ring, where the range-match below cannot.
        if (dir < 0 && r.mux_x != 0xFFFF) {
            auto* src_obj = get_sw_emulated_chip(src_chip);
            if (src_obj != nullptr) {
                try {
                    auto lg = src_obj->get_soc_descriptor().translate_coord_to(
                        tt_xy_pair(r.mux_x, r.mux_y), CoordSystem::TRANSLATED, CoordSystem::LOGICAL);
                    std::lock_guard<std::mutex> lk(g_conn_route_mu);
                    auto mit = g_mux_dir.find(__emule_worker_key(
                        src_chip, static_cast<uint32_t>(lg.x), static_cast<uint32_t>(lg.y)));
                    if (mit != g_mux_dir.end()) {
                        dir = static_cast<int>(mit->second);
                    }
                } catch (...) {
                }
            }
        }
        // (2) Fallback — range-match heuristic (and its cached g_worker_dir / conn-index), used only when the
        // mux signal above is absent. See tt-emule docs/fabric-ccl-emulation.md.
        if (dir < 0 && r.kind == emule_route_kind::MCAST_1D) {
            const uint32_t range = r.b ? r.b : 1;
            for (const auto& cr : conns) {
                if (__emule_fabric_walk(src_chip, static_cast<tt::tt_fabric::RoutingDirection>(cr.dir)).size() == range) {
                    dir = static_cast<int>(cr.dir);
                    break;
                }
            }
            if (dir < 0 && !conns.empty()) {
                dir = static_cast<int>(conns[r.dir_index < conns.size() ? r.dir_index : 0].dir);
            }
            if (dir >= 0) {
                std::lock_guard<std::mutex> lk(g_conn_route_mu);
                g_worker_dir[wkey] = static_cast<uint32_t>(dir);
            }
        } else if (dir < 0) {  // UNICAST_1D — reuse this worker's multicast-inferred direction; else the conn index
            std::lock_guard<std::mutex> lk(g_conn_route_mu);
            auto wit = g_worker_dir.find(wkey);
            if (wit != g_worker_dir.end()) {
                dir = static_cast<int>(wit->second);
            } else if (!conns.empty()) {
                dir = static_cast<int>(conns[r.dir_index < conns.size() ? r.dir_index : 0].dir);
            }
        }
        if (dir >= 0) {
            // Follow the real 1D ring via the recorded per-chip neighbors; fall back to the compass walk if
            // the chain is incomplete (walk[0] equals the compass neighbor). See tt-emule docs/fabric-ccl-emulation.md.
            std::vector<uint32_t> walk = __emule_fabric_walk_ring(src_chip, static_cast<uint32_t>(dir));
            if (walk.empty()) {
                walk = __emule_fabric_walk(src_chip, static_cast<tt::tt_fabric::RoutingDirection>(dir));
            }
            std::vector<uint32_t> tgts;
            if (r.kind == emule_route_kind::MCAST_1D) {
                const uint32_t start = r.a ? r.a : 1, range = r.b ? r.b : 1;
                for (uint32_t hop = start; hop < start + range && hop - 1 < walk.size(); ++hop) {
                    tgts.push_back(walk[hop - 1]);
                }
            } else {
                const uint32_t dist = r.a ? r.a : 1;
                if (dist - 1 < walk.size()) {
                    tgts.push_back(walk[dist - 1]);
                }
            }
            if (!tgts.empty()) {
                return tgts;
            }
        }
    }
    // Fallthrough (unstamped / no recorded connection) → neighbor.
    return {__emule_fabric_neighbor(src_chip)};
}

// Apply the terminal NOC command of a fabric send to ONE destination chip's L1 (the per-target delivery,
// looped over by the teleport for multicast).
static void __emule_fabric_deliver(
    uint32_t dst_chip, const uint8_t* h, const void* payload, uint32_t size, uint8_t noc_send_type, bool dbg) {
    const uint64_t noc_address = *reinterpret_cast<const uint64_t*>(h + 0);
    switch (noc_send_type) {
        case 0: {  // NOC_UNICAST_WRITE
            uint8_t* d = __emule_fabric_resolve_remote(dst_chip, noc_address);
            if (d != nullptr && payload != nullptr && size > 0) {
                std::memcpy(d, payload, size);
                std::atomic_thread_fence(std::memory_order_release);
                __emule_fiber_wake(d);
            }
            break;
        }
        case 1: {  // NOC_UNICAST_INLINE_WRITE: {noc_address; value@8}
            uint32_t value = *reinterpret_cast<const uint32_t*>(h + 8);
            uint8_t* d = __emule_fabric_resolve_remote(dst_chip, noc_address);
            if (d != nullptr) {
                reinterpret_cast<std::atomic<uint32_t>*>(d)->store(value, std::memory_order_release);
                __emule_fiber_wake(d);
            }
            break;
        }
        case 2: {  // NOC_UNICAST_ATOMIC_INC: {noc_address; val@8}
            uint32_t val = *reinterpret_cast<const uint32_t*>(h + 8);
            uint8_t* d = __emule_fabric_resolve_remote(dst_chip, noc_address);
            if (d != nullptr) {
                uint32_t old = reinterpret_cast<std::atomic<uint32_t>*>(d)->fetch_add(val, std::memory_order_release);
                if (dbg) {
                    fprintf(stderr, "[EMULE_FABRIC]   atomic_inc chip=%u dst=%p %u->%u (val=%u)\n",
                            dst_chip, (void*)d, old, old + val, val);
                }
                __emule_fiber_wake(d);
            }
            break;
        }
        case 3: {  // NOC_FUSED_UNICAST_ATOMIC_INC: {noc_address; semaphore_noc_address@8; val@16}
            uint64_t sem_addr = *reinterpret_cast<const uint64_t*>(h + 8);
            uint32_t val = *reinterpret_cast<const uint32_t*>(h + 16);
            uint8_t* d = __emule_fabric_resolve_remote(dst_chip, noc_address);
            if (d != nullptr && payload != nullptr && size > 0) {
                std::memcpy(d, payload, size);
                std::atomic_thread_fence(std::memory_order_release);
                __emule_fiber_wake(d);
            }
            uint8_t* s = __emule_fabric_resolve_remote(dst_chip, sem_addr);
            if (s != nullptr) {
                reinterpret_cast<std::atomic<uint32_t>*>(s)->fetch_add(val, std::memory_order_release);
                __emule_fiber_wake(s);
            }
            break;
        }
        case 4: {  // NOC_UNICAST_SCATTER_WRITE: noc_address[4]@0, chunk_size[3]@32, chunk_count@38, chunk_encoding@39
            // Also carries the fused scatter-write + atomic-inc: chunk_encoding holds a 2-bit code per chunk
            // (silicon NocScatterWriteChunkEncoding: 0 = NOP, 1 = unicast write, 2/3 = semaphore increment).
            // On silicon a scatter write is NOT left at 0 — to_noc_unicast_scatter_write fills every chunk with
            // encoding 1, and a fused packet marks its trailing chunk as a seminc (2/3) with the writes at 1.
            // Encoding 0 is CHUNK_ENCODING_NOP on the wire, so the write branch below handling enc 0 the same
            // as enc 1 is an emulator compatibility fallback only: emule's own NocUnicastScatterCommandHeader
            // defaults chunk_encoding to 0 for a plain scatter write. For a seminc chunk, fetch_add the value
            // stored in that chunk's size slot instead of copying payload (the seminc chunk carries no bytes).
            const uint64_t* na = reinterpret_cast<const uint64_t*>(h + 0);
            const uint16_t* cs = reinterpret_cast<const uint16_t*>(h + 32);
            uint8_t chunk_count = *(h + 38);
            uint8_t chunk_encoding = *(h + 39);
            uint32_t off = 0;
            for (uint8_t i = 0; i < chunk_count; ++i) {
                const uint8_t enc = (chunk_encoding >> (i * 2)) & 0x3;
                uint8_t* d = __emule_fabric_resolve_remote(dst_chip, na[i]);
                if (enc == 2 /*SEMINC_NO_FLUSH*/ || enc == 3 /*SEMINC_FLUSH*/) {
                    uint32_t val = cs[i];  // seminc value packed into this chunk's size slot
                    if (d != nullptr) {
                        reinterpret_cast<std::atomic<uint32_t>*>(d)->fetch_add(val, std::memory_order_release);
                        if (dbg) {
                            fprintf(stderr, "[EMULE_FABRIC]   scatter_seminc chip=%u dst=%p val=%u\n",
                                    dst_chip, (void*)d, val);
                        }
                        __emule_fiber_wake(d);
                    }
                    continue;  // no payload advance for a seminc chunk
                }
                // Write chunk. The last write chunk's size is implicit (remaining payload).
                uint32_t csz = (i + 1 < chunk_count) ? cs[i] : (size - off);
                if (payload != nullptr && d != nullptr && csz > 0) {
                    std::memcpy(d, static_cast<const uint8_t*>(payload) + off, csz);
                    __emule_fiber_wake(d);
                }
                off += csz;
            }
            std::atomic_thread_fence(std::memory_order_release);
            break;
        }
        default:
            break;  // 5/6 native multicast send-types: emule expresses multicast via the target list above
    }
}

// Top-level teleport: decode the real-layout packet header (NocCommandFields @0, payload_size @40,
// noc_send_type @42), resolve the destination chip(s), and apply the terminal NOC command. payload may be
// null for header-only commands (e.g. a bare atomic-inc). See tt-emule docs/fabric-ccl-emulation.md.
extern "C" void __emule_fabric_teleport(const void* packet_header, const void* payload, uint32_t payload_size) {
    emule_require_self(__func__);
    const uint8_t* h = static_cast<const uint8_t*>(packet_header);
    if (h == nullptr) {
        return;
    }
    const uint16_t hdr_payload_size = *reinterpret_cast<const uint16_t*>(h + 40);
    const uint8_t noc_send_type = *(h + 42);
    const uint32_t size = payload_size ? payload_size : hdr_payload_size;
    const uint32_t src_chip = __emule_self->chip_id;
    const std::vector<uint32_t> targets = __emule_fabric_resolve_targets(h, src_chip);
    static const bool dbg = std::getenv("EMULE_FABRIC_DEBUG") != nullptr;
    if (dbg) {
        const uint64_t noc_address = *reinterpret_cast<const uint64_t*>(h + 0);
        std::string ts;
        for (auto t : targets) {
            ts += " " + std::to_string(t);
        }
        fprintf(stderr,
                "[EMULE_FABRIC] teleport src=%u targets=[%s ] (neighbor=%u) send_type=%u noc_addr=0x%llx "
                "payload_size=%u\n",
                src_chip, ts.c_str(), __emule_fabric_neighbor(src_chip), noc_send_type,
                (unsigned long long)noc_address, size);
        // Route-table self-consistency: dump this src chip's per-direction distance walk (once per src).
        static std::mutex dump_mu;
        static std::unordered_map<uint32_t, bool> dumped;
        std::lock_guard<std::mutex> lk(dump_mu);
        if (!dumped[src_chip]) {
            dumped[src_chip] = true;
            const char* dn[4] = {"N", "E", "S", "W"};
            for (int d = 0; d < 4; ++d) {
                const auto& w = __emule_fabric_walk(src_chip, static_cast<tt::tt_fabric::RoutingDirection>(d));
                std::string s;
                for (auto c : w) {
                    s += " " + std::to_string(c);
                }
                fprintf(stderr, "[EMULE_FABRIC]   route src=%u dir=%s walk=[%s ]\n", src_chip, dn[d], s.c_str());
            }
        }
    }
    // One target for unicast; the line members for a multicast. Replay the terminal NOC op to each.
    for (uint32_t dst_chip : targets) {
        __emule_fabric_deliver(dst_chip, h, payload, size, noc_send_type, dbg);
    }
}

// ---------------------------------------------------------------------------
// setup_core_state: Configure CBs and semaphores per core, build CoreSetup list.
// ---------------------------------------------------------------------------
// Initialize CB-sync state on a core from the program's circular buffer list.
static void init_core_cb_sync(
    tt_emule::Core* core,
    detail::ProgramImpl& impl,
    const CoreCoord& logical_core,
    std::vector<uint64_t>& persistent_cb_ranges) {
    core->reset_cb_sync();
    // Record this core's globally-allocated (persistent) CB extents so Object-Intent
    // exempts the kernel's writes anywhere in them (see §12). Its own pass — not folded
    // into the configure lambda below, which also walks remote pass-2 CBs — to keep the
    // exempt set exactly the local ones.
    for (auto& cb_impl : impl.circular_buffers_on_core(logical_core)) {
        if (cb_impl->globally_allocated()) {
            uint32_t start = cb_impl->address();
            persistent_cb_ranges.push_back((static_cast<uint64_t>(start) << 32) | (start + cb_impl->size()));
        }
    }

    bool configured[EMULE_NUM_CBS] = {};
    auto configure = [&](const std::shared_ptr<CircularBufferImpl>& cb_impl, const CoreCoord& lc) {
        for (uint8_t idx : cb_impl->local_buffer_indices()) {
            if (idx >= EMULE_NUM_CBS || configured[idx]) {
                continue;
            }
            uint32_t cb_addr = cb_impl->address();
            uint32_t page_size = cb_impl->page_size(idx);
            uint32_t num_pages = (page_size > 0) ? cb_impl->num_pages(idx) : 0;
            uint8_t* base = (page_size > 0) ? core->l1_ptr(cb_addr) : nullptr;
            core->init_cb_sync(idx, base, page_size, num_pages, cb_impl->globally_allocated());
            configured[idx] = true;
            log_debug(
                tt::LogMetal,
                "  Core({},{}) CB[{}]: addr=0x{:x} page_size={} num_pages={} base={:p}",
                lc.x, lc.y, idx, cb_addr, page_size, num_pages, (void*)base);
        }
    };
    // Pass 1: CBs allocated on this core take precedence (own addresses).
    for (auto& cb_impl : impl.circular_buffers_on_core(logical_core)) {
        configure(cb_impl, logical_core);
    }
    // Pass 2: register the remaining program CBs at their global L1 address so a
    // kernel can get_write_ptr() a CB allocated only on a remote core (silicon CB
    // addresses are program-global). Needed for multi-core topk, where local cores
    // NOC-write into the final core's final_*_cb. Used only as cross-core NOC
    // targets here; the masked L1 offset is what __emule_resolve_noc_addr routes.
    for (auto& cb_impl : impl.circular_buffers()) {
        configure(cb_impl, logical_core);
    }
}

// Write semaphore initial values into L1 at the HAL-derived semaphore base.
static void init_core_semaphores(
    tt_emule::Core* core, detail::ProgramImpl& impl, const CoreCoord& logical_core, uint32_t emule_sem_base) {
    for (auto& sem : impl.semaphores()) {
        if (!sem.initialized_on_logical_core(logical_core)) {
            continue;
        }
        uint32_t sem_id = sem.id();
        uint32_t initial_value = sem.initial_value();
        uint32_t sem_addr = emule_sem_base + sem_id * EMULE_SEM_ALIGN;
        if (sem_addr + sizeof(uint32_t) > core->l1_size()) {
            continue;
        }
        auto* sem_ptr = reinterpret_cast<uint32_t*>(core->l1_ptr(sem_addr));
        *sem_ptr = initial_value;
        log_debug(
            tt::LogMetal,
            "  Core({},{}) Sem[{}]: addr=0x{:x} initial={}",
            logical_core.x,
            logical_core.y,
            sem_id,
            sem_addr,
            initial_value);
    }
}

// Allocate L1 for each DFB on a core, register CB-sync bridges, and initialize
// tile counters. Returns per-DFB allocation info consumed by launch_cores.
static std::vector<DFBAllocInfo> allocate_dfbs_on_core(
    tt_emule::Core* core,
    const CoreCoord& logical_core,
    const std::vector<std::shared_ptr<tt::tt_metal::experimental::dfb::detail::DataflowBufferImpl>>& dfb_impls) {
    core->reset_dfb_sync();
    if (dfb_impls.empty()) {
        // No DFBs to allocate (always the case on WH/BH; DFBs are Quasar-only),
        // so the L1 bump allocator never grows and there's nothing to reset.
        // Skipping reset also leaves the mmap-init zeros at MEM_ZEROS_BASE
        // undisturbed for kernels that NOC-read the region.
        return {};
    }
    // DFB fallback path (Quasar): start the bump allocator at 0.  When Quasar
    // bring-up needs to protect MEM_ZEROS from bump-allocator overlap, dispatch
    // its per-arch MEM_ZEROS_BASE here.
    core->reset_l1_bump();
    if (!core->tile_counters()) {
        core->init_tile_counters(4);
    }

    std::vector<DFBAllocInfo> dfb_allocs;
    // Compute bridge sharing: a compute-consumer input DFB and a compute-producer
    // output DFB with matching dimensions share L1 (real HW routes through the
    // register file). Independent compute-consumer inputs (e.g. matmul in0/in1)
    // must NOT share.
    constexpr uint16_t TENSIX_MASK = 0xFF00u;  // bits 8-15
    std::unordered_map<uint64_t, uint32_t> bridge_consumer_alloc;
    for (auto& dfb_impl : dfb_impls) {
        uint32_t dfb_id = dfb_impl->id;
        auto& cfg = dfb_impl->config;
        uint32_t total = cfg.entry_size * cfg.num_entries;
        uint64_t dim_key = (static_cast<uint64_t>(cfg.entry_size) << 32) | cfg.num_entries;
        bool compute_is_consumer = (cfg.consumer_risc_mask & TENSIX_MASK) != 0;
        bool compute_is_producer = (cfg.producer_risc_mask & TENSIX_MASK) != 0;
        // Prefer the finalize-allocated L1 offset (so host/test verification
        // hits the same offset); fall back to bump-alloc when absent. L1 offset
        // model: base_addr is a 0-based L1 offset (finalize supplies the offset
        // directly; l1_alloc returns one too). Use a found-flag, not addr != 0,
        // as the "has finalize" test — offset 0 is a valid L1 address.
        auto cl = dfb_impl->core_lookup_.find(logical_core);
        bool has_finalize = (cl != dfb_impl->core_lookup_.end());
        uint32_t finalize_addr = has_finalize ? cl->second.second : 0;  // 0-based L1 offset
        uint32_t base_addr;
        if (compute_is_producer && !compute_is_consumer) {
            auto it = bridge_consumer_alloc.find(dim_key);
            base_addr = (it != bridge_consumer_alloc.end()) ? it->second
                                                            : (has_finalize ? finalize_addr : core->l1_alloc(total));
        } else {
            base_addr = has_finalize ? finalize_addr : core->l1_alloc(total);
            if (compute_is_consumer && !compute_is_producer) {
                bridge_consumer_alloc.emplace(dim_key, base_addr);
            }
        }
        // base_addr is a 0-based L1 offset (L1 offset model); rebase onto this
        // core's L1 to get the host pointer the DFB/CB sync state stores.
        uint8_t* base = core->l1_data() + base_addr;
        // STRIDED: M = max(P, C); ALL: M = P.
        bool is_all = (cfg.cap == ::dfb::AccessPattern::ALL);
        uint32_t M = is_all ? cfg.num_producers : std::max<uint32_t>(cfg.num_producers, cfg.num_consumers);
        uint32_t capacity = cfg.num_entries / M;
        core->init_dfb_sync(dfb_id, base, cfg.entry_size, cfg.num_entries, capacity);

        // Also populate CB sync state for this DFB so compute ops (pack_tile,
        // matmul_tiles) can reuse the same L1 buffer via cb_read_ptr/cb_write_ptr.
        if (dfb_id < EMULE_NUM_CBS) {
            core->init_cb_sync(static_cast<uint8_t>(dfb_id), base, cfg.entry_size, cfg.num_entries);
        }

        // Initialize tile counters for this DFB.
        // STRIDED: M TCs. ALL DM-DM: P*C TCs. Counter IDs are spaced by
        // MAX_TC_SLOTS_PER_DFB to prevent cross-DFB collisions.
        if (dfb_id >= (tt_emule::TILE_COUNTERS_PER_NEO / tt_emule::MAX_TC_SLOTS_PER_DFB)) {
            throw std::out_of_range("dfb_id exceeds safe TC range (max 8 DFBs per NEO with neo_id=0)");
        }
        uint8_t counter_base = static_cast<uint8_t>(dfb_id * tt_emule::MAX_TC_SLOTS_PER_DFB);
        uint32_t num_tcs_to_init = is_all ? static_cast<uint32_t>(cfg.num_producers) * cfg.num_consumers : M;
        for (uint32_t tc_idx = 0; tc_idx < num_tcs_to_init; ++tc_idx) {
            auto& tc = core->tile_counters()->get(0, counter_base + static_cast<uint8_t>(tc_idx));
            tc.capacity = capacity;
            tc.posted.store(0, std::memory_order_relaxed);
            tc.acked.store(0, std::memory_order_relaxed);
        }

        dfb_allocs.push_back({dfb_id, base_addr, &cfg});
        log_debug(
            tt::LogMetal,
            "  Core({},{}) DFB[{}]: addr=0x{:x} entry_size={} num_entries={} total={}",
            logical_core.x,
            logical_core.y,
            dfb_id,
            base_addr,
            cfg.entry_size,
            cfg.num_entries,
            total);
    }
    return dfb_allocs;
}

static void setup_core_state(
    detail::ProgramImpl& impl,
    IDevice* device,
    tt::umd::SWEmuleChip* sw_emu,
    std::map<CoreCoord, std::vector<KernelInfo>>& core_kernels,
    uint32_t emule_sem_base,
    std::vector<CoreSetup>& core_setups) {
    for (auto& [logical_core, ki_list] : core_kernels) {
        if (!sw_emu) {
            continue;
        }
        auto phys = device->virtual_core_from_logical_core(logical_core, tt::CoreType::WORKER);
        tt_emule::Core* core = sw_emu->get_core(tt_xy_pair(phys.x, phys.y));
        if (!core) {
            continue;
        }
        uint8_t phys_x = static_cast<uint8_t>(phys.x);
        uint8_t phys_y = static_cast<uint8_t>(phys.y);

        std::vector<uint64_t> persistent_cb_ranges;
        init_core_cb_sync(core, impl, logical_core, persistent_cb_ranges);
        init_core_semaphores(core, impl, logical_core, emule_sem_base);

        auto dfb_impls = impl.dataflow_buffers_on_core(logical_core);
        bool has_dfbs = !dfb_impls.empty();
        std::vector<DFBAllocInfo> dfb_allocs = allocate_dfbs_on_core(core, logical_core, dfb_impls);

        uint32_t sem_region_size = tt::tt_metal::NUM_SEMAPHORES * EMULE_SEM_ALIGN;
        core_setups.push_back(
            {logical_core,
             core,
             &ki_list,
             phys_x,
             phys_y,
             std::move(dfb_allocs),
             has_dfbs,
             emule_sem_base,
             sem_region_size,
             std::move(persistent_cb_ranges)});
    }
}

// ---------------------------------------------------------------------------
// Populate tile-counter slot state on a single EmuleDFBInterface for one
// (producer/consumer) × (STRIDED/ALL) role.  Pulled out of launch_cores
// because the 4-case slot-init block dominated the parent function.
// ---------------------------------------------------------------------------
static void populate_dfb_interface_slots(
    tt_emule::EmuleDFBInterface& iface, const DFBAllocInfo& alloc, uint8_t proc_id, bool is_tensix) {
    const auto& cfg = *alloc.cfg;
    const uint32_t total = cfg.entry_size * cfg.num_entries;

    // Compute proc_bit per-alloc: WH/BH ComputeKernel sets bit 2,
    // Quasar uses bits 8+ (detect by presence of high mask bits).
    uint16_t proc_bit;
    if (is_tensix) {
        bool quasar_masks = ((cfg.producer_risc_mask | cfg.consumer_risc_mask) & 0xFF00u) != 0;
        proc_bit = quasar_masks ? static_cast<uint16_t>(1u << (proc_id + ::dfb::TENSIX_RISC_OFFSET))
                                : static_cast<uint16_t>(1u << 2);
    } else {
        proc_bit = static_cast<uint16_t>(1u << proc_id);
    }
    bool is_all = (cfg.cap == ::dfb::AccessPattern::ALL);
    uint32_t M = is_all ? cfg.num_producers : std::max<uint32_t>(cfg.num_producers, cfg.num_consumers);
    uint32_t stride_size = M * cfg.entry_size;
    if (alloc.dfb_id >= (tt_emule::TILE_COUNTERS_PER_NEO / tt_emule::MAX_TC_SLOTS_PER_DFB)) {
        return;
    }
    uint8_t counter_base = static_cast<uint8_t>(alloc.dfb_id * tt_emule::MAX_TC_SLOTS_PER_DFB);

    bool is_producer = (cfg.producer_risc_mask & proc_bit) != 0;
    bool is_consumer = (cfg.consumer_risc_mask & proc_bit) != 0;
    if (!is_producer && !is_consumer) {
        return;
    }

    iface.active = true;
    iface.entry_size = cfg.entry_size;
    iface.stride_size = stride_size;
    iface.num_entries = cfg.num_entries;
    iface.tc_idx = 0;
    iface.broadcast_tc = false;
    iface.rd_entry_idx = 0;
    iface.wr_entry_idx = 0;

    if (is_producer) {
        uint8_t p = static_cast<uint8_t>(std::popcount(cfg.producer_risc_mask & (proc_bit - 1u)));
        if (is_all) {
            // ALL DM-DM: producer broadcasts to all consumer TCs.
            iface.broadcast_tc = true;
            iface.num_tcs_to_rr = static_cast<uint8_t>(cfg.num_consumers);
            iface.stride_size = cfg.entry_size;
            uint32_t capacity_per_p = cfg.num_entries / cfg.num_producers;
            uint32_t producer_ptr = alloc.base_addr + p * capacity_per_p * cfg.entry_size;
            fill_dfb_slots(iface, cfg.num_consumers, [&](uint32_t c) {
                return DfbSlotInit{
                    static_cast<uint8_t>(counter_base + p * cfg.num_consumers + c),
                    alloc.base_addr,
                    alloc.base_addr + total,
                    producer_ptr};
            });
        } else {
            uint32_t num_tcs = M / cfg.num_producers;
            iface.num_tcs_to_rr = static_cast<uint8_t>(num_tcs);
            fill_dfb_slots(iface, num_tcs, [&](uint32_t k) {
                uint8_t tc_idx = static_cast<uint8_t>(p + k * cfg.num_producers);
                return DfbSlotInit{
                    static_cast<uint8_t>(counter_base + tc_idx),
                    alloc.base_addr,
                    alloc.base_addr + total,
                    alloc.base_addr + tc_idx * cfg.entry_size};
            });
        }
    } else {
        uint8_t c = static_cast<uint8_t>(std::popcount(cfg.consumer_risc_mask & (proc_bit - 1u)));
        if (is_all) {
            // ALL DM-DM consumer: drain each producer's TC block fully.
            iface.num_tcs_to_rr = static_cast<uint8_t>(cfg.num_producers);
            iface.stride_size = cfg.entry_size;
            iface.drain_per_tc = true;
            uint32_t capacity_per_p = cfg.num_entries / cfg.num_producers;
            uint32_t sub_range = capacity_per_p * cfg.entry_size;
            fill_dfb_slots(iface, cfg.num_producers, [&](uint32_t p) {
                uint32_t sub_base = alloc.base_addr + p * sub_range;
                return DfbSlotInit{
                    static_cast<uint8_t>(counter_base + p * cfg.num_consumers + c),
                    sub_base,
                    sub_base + sub_range,
                    sub_base};
            });
        } else {
            uint32_t num_tcs = M / cfg.num_consumers;
            iface.num_tcs_to_rr = static_cast<uint8_t>(num_tcs);
            fill_dfb_slots(iface, num_tcs, [&](uint32_t k) {
                uint8_t tc_idx = static_cast<uint8_t>(c + k * cfg.num_consumers);
                return DfbSlotInit{
                    static_cast<uint8_t>(counter_base + tc_idx),
                    alloc.base_addr,
                    alloc.base_addr + total,
                    alloc.base_addr + tc_idx * cfg.entry_size};
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Build per-thread DFB interface arrays for one core.  Each kernel thread
// gets its own copy with independent wr/rd ptrs (matching real HW where each
// RISC has a separate LocalDFBInterface).
// ---------------------------------------------------------------------------
static std::vector<std::unique_ptr<tt_emule::EmuleDFBInterface[]>> build_per_thread_dfb_interfaces(
    const std::vector<KernelInfo>& ki_list, const std::vector<DFBAllocInfo>& dfb_allocs) {
    std::vector<std::unique_ptr<tt_emule::EmuleDFBInterface[]>> per_thread_dfbs;
    per_thread_dfbs.resize(ki_list.size());
    for (size_t t = 0; t < ki_list.size(); t++) {
        per_thread_dfbs[t] = std::make_unique<tt_emule::EmuleDFBInterface[]>(tt_emule::MAX_DFBS);
        for (const auto& alloc : dfb_allocs) {
            if (alloc.dfb_id >= tt_emule::MAX_DFBS) {
                continue;
            }
            populate_dfb_interface_slots(
                per_thread_dfbs[t][alloc.dfb_id], alloc, ki_list[t].processor_id, ki_list[t].is_tensix);
        }
    }
    return per_thread_dfbs;
}

// EmuleOobTensorState, the Object-Intent tracker, the per-kernel sanitizer
// thread-local set/clear, the Dirty-CB sweep, and build_oob_tensor_state now
// live in emule_sanitizers.{hpp,cpp}. See SANITIZER_CHECKS.md.

// ---------------------------------------------------------------------------
// launch_cores: Spawn concurrent threads per core, each runs its kernels.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// RISC-V-faithful integer divide/modulo fault recovery on x86 hosts. RISC-V div/rem faults are DEFINED
// and non-trapping, but the JIT-compiled x86 `div`/`idiv` raises #DE -> SIGFPE; trap it, write RISC-V's
// defined result into the saved register image, and step RIP past the faulting instruction. The handler
// is process-global for launch_cores' lifetime. See docs/riscv-intdiv-by-zero.md in the tt-emule repo.
#if defined(__x86_64__) && defined(__linux__)
namespace {

// Length in bytes of the div/idiv at `p`, or 0 if it is not a *recoverable* one.
// We recover only the 32-bit and 64-bit `F7 /6,/7` forms — the only integer-divide
// widths a RISC-V-derived kernel compiled to x86 emits (RV32/RV64 have no 8/16-bit
// divide; C integer promotion never yields one either). 8-bit (`F6`) and 16-bit
// (`0x66`-prefixed) forms are declined (return 0 -> abort) rather than risk a wrong
// partial-register write-back. Handles optional legacy + REX prefixes and
// ModRM/SIB/disp for a memory operand. Sets *width to 32 or 64.
size_t emule_decode_divlen(const uint8_t* p, int* width) {
    size_t i = 0;
    bool opsize16 = false, rexw = false;
    while (p[i] == 0x67 || p[i] == 0x66 || p[i] == 0x2e || p[i] == 0x3e || p[i] == 0x26 || p[i] == 0x64 ||
           p[i] == 0x65 || p[i] == 0x36 || p[i] == 0xf0 || p[i] == 0xf2 || p[i] == 0xf3) {
        if (p[i] == 0x66) {
            opsize16 = true;  // operand-size override
        }
        ++i;
    }
    if ((p[i] & 0xf0) == 0x40) {  // REX
        if (p[i] & 0x08) {
            rexw = true;  // REX.W
        }
        ++i;
    }
    if (p[i] != 0xf7) {  // F7 = 16/32/64-bit DIV/IDIV; F6 (8-bit) is not a RISC-V width
        return 0;
    }
    ++i;
    uint8_t modrm = p[i];
    uint8_t reg = (modrm >> 3) & 0x7;
    if (reg != 6 && reg != 7) {  // /6 = DIV, /7 = IDIV
        return 0;
    }
    uint8_t mod = modrm >> 6;
    uint8_t rm = modrm & 0x7;
    ++i;  // ModRM
    if (mod != 3) {                          // memory operand
        if (rm == 4) {                       // SIB present
            uint8_t base = p[i] & 0x7;
            ++i;
            if (mod == 0 && base == 5) {
                i += 4;  // disp32, no base
            }
        }
        if (mod == 1) {
            i += 1;  // disp8
        } else if (mod == 2) {
            i += 4;  // disp32
        } else if (mod == 0 && rm == 5) {
            i += 4;  // RIP-relative disp32
        }
    }
    *width = rexw ? 64 : (opsize16 ? 16 : 32);
    if (*width == 16) {
        return 0;  // 16-bit div: not a RISC-V width; partial-register fix-up would be unsafe
    }
    return i;
}

void emule_sigfpe_handler(int sig, siginfo_t* info, void* uc_void) {
    if (sig == SIGFPE && (info->si_code == FPE_INTDIV || info->si_code == FPE_INTOVF)) {
        auto* uc = static_cast<ucontext_t*>(uc_void);
        greg_t* regs = uc->uc_mcontext.gregs;
        auto* rip = reinterpret_cast<const uint8_t*>(regs[REG_RIP]);
        int width = 0;
        size_t len = emule_decode_divlen(rip, &width);
        if (len > 0) {
            // x86 dividend low half is in (R|E)AX; quotient lands in (R|E)AX, rem in (R|E)DX.
            const greg_t dividend = regs[REG_RAX];
            if (info->si_code == FPE_INTDIV) {
                // div/rem by zero — RISC-V: quotient = all-ones, remainder = dividend.
                if (width == 64) {
                    regs[REG_RAX] = static_cast<greg_t>(~0ULL);
                    regs[REG_RDX] = dividend;
                } else {  // 32-bit writes zero-extend the full 64-bit reg
                    regs[REG_RAX] = static_cast<greg_t>(static_cast<uint32_t>(~0U));
                    regs[REG_RDX] = static_cast<greg_t>(static_cast<uint32_t>(dividend));
                }
            } else {
                // FPE_INTOVF: signed INT_MIN / -1 — RISC-V: quotient = dividend (INT_MIN), rem = 0.
                if (width == 64) {
                    regs[REG_RAX] = dividend;
                    regs[REG_RDX] = 0;
                } else {
                    regs[REG_RAX] = static_cast<greg_t>(static_cast<uint32_t>(dividend));
                    regs[REG_RDX] = 0;
                }
            }
            regs[REG_RIP] = reinterpret_cast<greg_t>(rip + len);
            return;
        }
    }
    // Not a recoverable integer divide/overflow: fall back to default disposition.
    signal(sig, SIG_DFL);
    raise(sig);
}

// Installs the handler for the duration of kernel execution, restoring the
// previous disposition afterward so emule does not permanently alter the host.
struct EmuleSigfpeGuard {
    struct sigaction prev_ {};
    bool installed_ = false;
    EmuleSigfpeGuard() {
        struct sigaction sa {};
        sa.sa_sigaction = emule_sigfpe_handler;
        sa.sa_flags = SA_SIGINFO;  // synchronous, thread-directed; handler never re-faults
        sigemptyset(&sa.sa_mask);
        installed_ = (sigaction(SIGFPE, &sa, &prev_) == 0);
    }
    ~EmuleSigfpeGuard() {
        if (installed_) {
            sigaction(SIGFPE, &prev_, nullptr);
        }
    }
    EmuleSigfpeGuard(const EmuleSigfpeGuard&) = delete;
    EmuleSigfpeGuard& operator=(const EmuleSigfpeGuard&) = delete;
};

}  // namespace
#endif  // __x86_64__ && __linux__

// [MESH] Register/run split for concurrent multi-device dispatch: in defer mode each
// execute_program_emulated REGISTERS its fibers (spawn, no run); run_mesh_dispatch then drives ONE
// run_until_idle so all chips' fibers run concurrently. See tt-emule docs/fiber-engine.md.
static bool g_emule_mesh_defer = false;
static std::vector<std::vector<std::vector<std::unique_ptr<tt_emule::EmuleDFBInterface[]>>>>
    g_mesh_dfb_keep;

// [HOST-INTERLEAVED SOCKET] Set when run_mesh_dispatch's run_persistent() returned HostWait: the mesh
// run is parked awaiting host socket I/O, with g_mesh_dfb_keep + the scheduler's fibers kept alive.
// pump_device() drives it forward per host socket call; the pump that completes clears this + the mesh
// keepalives (the cleanup run_mesh_dispatch deferred). See tt-emule docs/socket-emulation.md §7.
static bool g_emule_host_wait = false;

// Resolved-program cache — emule's analogue of silicon's is_compiled(): collect + JIT compile + resolve
// run ONCE per program (keyed by ProgramId); every device dispatches against the shared read-only result.
// LRU-bounded as a safety net. See tt-emule docs/fiber-engine.md.
//
// Lock-free by invariant: written only from prepare_program on the sequential mesh-register path — single
// writer, never a fiber. prepare_program asserts this.
struct ResolvedProgram {
    std::map<CoreCoord, std::vector<KernelInfo>> core_kernels;
    uint32_t emule_sem_base = 0;
};
static std::unordered_map<ProgramId, ResolvedProgram> g_resolved_programs;
static std::deque<ProgramId> g_resolved_lru;
static constexpr size_t kMaxResolvedPrograms = 256;

static void launch_cores(
    std::vector<CoreSetup>& core_setups,
    uint8_t* dram_data,
    std::unordered_map<uint64_t, tt_emule::Core*>* core_map_ptr,
    ChipId device_id,
    bool defer_run,
    const EmuleOobTensorState& oob_state) {
#if defined(__x86_64__) && defined(__linux__)
    EmuleSigfpeGuard sigfpe_guard;
#endif
    // Fiber engine: one cooperatively-scheduled fiber per (core, RISC), multiplexed
    // onto a runtime-sized worker pool (TT_EMULE_FIBER_WORKERS). A blocked fiber parks
    // (yields its worker) instead of blocking an OS thread — no thread ceiling, no spin.
    // See docs/fiber-engine.md.
    auto& sched = tt::tt_metal::emule_fiber::FiberScheduler::instance();

    // The fibers borrow the per-core DFB interface arrays; own them here so they
    // outlive run_until_idle.
    std::vector<std::vector<std::unique_ptr<tt_emule::EmuleDFBInterface[]>>> dfb_keepalive;
    dfb_keepalive.reserve(core_setups.size());

    for (size_t core_idx = 0; core_idx < core_setups.size(); ++core_idx) {
        auto& cs = core_setups[core_idx];
        auto* core = cs.core;
        uint8_t* l1_data = core->l1_data();
        tt_emule::CBSyncState* cb_array = core->cb_sync_array();
        tt_emule::TileCounterArray* tc_array = cs.has_dfbs ? core->tile_counters() : nullptr;
        const uint8_t px = cs.phys_x;
        const uint8_t py = cs.phys_y;
        const uint32_t lx = cs.logical_core.x;
        const uint32_t ly = cs.logical_core.y;

        // Per-core logical coords (shared by all RISC fibers on this core).
        auto& cstate = core->core_state();
        cstate.logical_x = lx;
        cstate.logical_y = ly;

        std::vector<std::unique_ptr<tt_emule::EmuleDFBInterface[]>> per_thread_dfbs;
        if (cs.has_dfbs) {
            per_thread_dfbs = build_per_thread_dfb_interfaces(*cs.ki_list, cs.dfb_allocs);
        }

        for (size_t kidx = 0; kidx < cs.ki_list->size(); ++kidx) {
            KernelInfo* ki_ptr = &(*cs.ki_list)[kidx];
            auto& ki = *ki_ptr;
            tt_emule::EmuleDFBInterface* dfb_array = cs.has_dfbs ? per_thread_dfbs[kidx].get() : nullptr;

            // Build + populate the fiber-owned ctx (set-once identity). The scheduler
            // repoints __emule_self to this ctx on swap-in; my_x/my_y are restored from
            // the FiberIdentity (they cannot move into the ctx — silicon-named globals).
            std::unique_ptr<ThreadCommonCtx> ctx =
                ki.is_tensix ? std::unique_ptr<ThreadCommonCtx>(new ComputeThreadCtx())
                             : std::unique_ptr<ThreadCommonCtx>(new DatamovementThreadCtx());
            ctx->rt_args = (ki.rta_offset_in_kc != kRtaCrtaNoArgsSentinel)
                ? reinterpret_cast<uint32_t*>(core->l1_ptr(ki.kernel_config_base + ki.rta_offset_in_kc))
                : nullptr;
            ctx->common_rt_args = (ki.crta_offset_in_kc != kRtaCrtaNoArgsSentinel)
                ? reinterpret_cast<uint32_t*>(core->l1_ptr(ki.kernel_config_base + ki.crta_offset_in_kc))
                : nullptr;
            ctx->bridge_l1 = l1_data;
            ctx->l1_size = static_cast<uint32_t>(core->l1_size());
            ctx->bridge_dram = dram_data;
            ctx->cbs = cb_array;
            ctx->dfbs = dfb_array;
            ctx->tc_array = tc_array;
            ctx->processor_id = ki.processor_id;
            ctx->core_obj = core;
            ctx->device = nullptr;
            ctx->chip_id = static_cast<uint32_t>(device_id);
            ctx->core_map = core_map_ptr;
            ctx->neo_id = ki.is_tensix ? ki.processor_id : 0;
            ctx->trisc_id = 0;
            ctx->num_threads = ki.num_threads;
            ctx->my_thread_id = ki.thread_idx;
            ctx->core = &cstate;

            tt::tt_metal::emule_fiber::FiberIdentity id;
            id.phys_x = px;
            id.phys_y = py;
            id.logical_x = lx;
            id.logical_y = ly;
            id.proc_id = ki.processor_id;
            id.kernel_src = nullptr;

            // The fiber entry is the kernel body. __emule_self is set by the scheduler
            // on swap-in; the no-op start-barrier of the OS-thread model is gone (a
            // blocked fiber parks rather than spins, so start order is irrelevant).
            //
            // ASAN sanitizer (#44848) is armed per-kernel here: set_sanitizer_thread_locals /
            // sweep_per_kernel_dirty_cbs / clear (+ the identity globals the sanitizer .cpp reads,
            // mirroring the ctx). All inert when TT_METAL_EMULE_ASAN is off — set_sanitizer_thread_locals
            // early-returns, so the by-value oob_state view (owned by dispatch_to_device's OobStateOwner)
            // is never dereferenced. Under the fiber engine these worker-thread-locals are best-effort
            // (shared across parked fibers on a worker); per-fiber ASAN state + the per-core ObjectIntent
            // snapshot/verify are a follow-up, so ASAN is aimed at single-device runs.
            sched.spawn(
                [ki_ptr, lx, ly, cb_array, oob_state, sem_base = cs.sem_base, sem_size = cs.sem_size]() {
                    auto& ki = *ki_ptr;
                    __processor_id = ki.processor_id;
                    __emule_neo_id = ki.is_tensix ? ki.processor_id : 0;
                    __emule_trisc_id = 0;
                    __emule_kernel_name = ki.kernel_name.empty() ? nullptr : ki.kernel_name.c_str();
                    __emule_pending_noc_reads = 0;
                    set_sanitizer_thread_locals(oob_state, sem_base, sem_size);
                    try {
                        for (size_t t = 0; t < ki.variants.size(); ++t) {
                            if (ki.run_all_variants) {
                                __emule_self->trisc_id = static_cast<uint8_t>(t);
                                __emule_trisc_id = static_cast<uint8_t>(t);
                            }
                            ki.variants[t]();
                        }
                        sweep_per_kernel_dirty_cbs(oob_state, cb_array, ki.processor_id, lx, ly);
                    } catch (...) {
                        clear_sanitizer_thread_locals();
                        std::throw_with_nested(std::runtime_error(
                            "EMULE: kernel on core (" + std::to_string(lx) + "," + std::to_string(ly) +
                            ") failed"));
                    }
                    __emule_kernel_name = nullptr;
                    __emule_pending_noc_reads = 0;
                    clear_sanitizer_thread_locals();
                },
                std::move(ctx),
                id);
        }

        dfb_keepalive.push_back(std::move(per_thread_dfbs));
    }

    if (defer_run) {
        // Mesh register phase: fibers are spawned but not run yet. Keep the DFB arrays they
        // borrow alive until run_mesh_dispatch (the spawned ctx is already owned by the
        // scheduler; core_kernels is kept by execute_program_emulated). The SIGFPE guard
        // above is a no-op here since no kernel runs; run_mesh_dispatch installs its own.
        g_mesh_dfb_keep.push_back(std::move(dfb_keepalive));
        return;
    }

    // Run all registered fibers to completion; rethrows the first kernel exception,
    // throws on a quiescent deadlock, aborts with a dump on livelock/hang.
    sched.run_until_idle();
}

// ---------------------------------------------------------------------------
// prepare_program: resolve a program's kernels ONCE (collect + JIT-compile + resolve), memoized by
// ProgramId — emule's analogue of silicon's CompileProgram. The first mesh device resolves; the rest
// reuse, taking its (homogeneous-chip-identical) compile defines. See tt-emule docs/metal-integration.md.
// ---------------------------------------------------------------------------
static ResolvedProgram& prepare_program(IDevice* device, Program& program) {
    // Single-writer invariant for g_resolved_programs/g_resolved_lru: this runs only on the
    // sequential dispatch thread (register phase), never inside a fiber. __emule_self is the
    // running fiber (set on worker threads, null on the dispatch thread), so off-fiber == null.
    TT_FATAL(__emule_self == nullptr, "prepare_program must run on the dispatch path, not a fiber");
    auto& impl = program.impl();
    const ProgramId pid = impl.get_id();
    if (auto it = g_resolved_programs.find(pid); it != g_resolved_programs.end()) {
        return it->second;  // already resolved (peer mesh device or repeated invocation)
    }

    auto device_id = device->id();
    auto* sw_emu = get_sw_emulated_chip(device_id);

    tt_emule::Core* dram_core = nullptr;
    uint32_t num_dram_channels = 0;
    uint32_t num_l1_banks = 0;
    populate_bank_mapping(sw_emu, device, device_id, dram_core, num_dram_channels, num_l1_banks);

    std::string worker_col_map_str, worker_row_map_str;
    build_worker_coord_maps(device, worker_col_map_str, worker_row_map_str);

    std::string extra_inc = get_extra_include_flags();

    const auto& hal = MetalContext::instance().hal();
    uint32_t tensix_pct_index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    uint32_t kernel_config_base =
        static_cast<uint32_t>(hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG));
    const auto& prog_config = impl.get_program_config(tensix_pct_index);
    uint32_t emule_sem_base = kernel_config_base + prog_config.sem_offset;

    std::map<CoreCoord, std::vector<PendingKernelInfo>> pending_core_kernels;
    std::map<std::string, DeferredCompile> deferred_compiles;
    std::unordered_map<std::string, std::function<void()>> resolved_fns;
    std::vector<std::string> inline_src_temps;
    collect_kernels(
        impl, num_dram_channels, num_l1_banks, worker_col_map_str, worker_row_map_str,
        emule_sem_base, extra_inc, pending_core_kernels, deferred_compiles, resolved_fns, inline_src_temps);
    jit_compile_pending(deferred_compiles, resolved_fns, inline_src_temps);

    ResolvedProgram resolved;
    resolved.emule_sem_base = emule_sem_base;
    for (auto& [logical_core, pending_list] : pending_core_kernels) {
        for (auto& pk : pending_list) {
            KernelInfo ki{
                {},
                pk.run_all_variants,
                pk.processor_id,
                pk.thread_idx,
                pk.is_tensix,
                pk.num_threads,
                pk.kernel_config_base,
                pk.rta_offset_in_kc,
                pk.crta_offset_in_kc};
            ki.variants.reserve(pk.variant_cache_keys.size());
            for (const auto& key : pk.variant_cache_keys) {
                ki.variants.push_back(resolved_fns.at(key));
            }
            ki.rt_arg_values = std::move(pk.rt_arg_values);
            ki.kernel_name = std::move(pk.kernel_name);
            resolved.core_kernels[logical_core].push_back(std::move(ki));
        }
    }
    log_info(
        tt::LogMetal,
        "execute_program_emulated: program {} resolved ({} logical cores)",
        pid,
        resolved.core_kernels.size());

    // LRU-bound the cache (safety net; entries are otherwise valid for the program's life).
    if (g_resolved_programs.size() >= kMaxResolvedPrograms && !g_resolved_lru.empty()) {
        g_resolved_programs.erase(g_resolved_lru.front());
        g_resolved_lru.pop_front();
    }
    g_resolved_lru.push_back(pid);
    return g_resolved_programs.emplace(pid, std::move(resolved)).first->second;
}

// ---------------------------------------------------------------------------
// dispatch_to_device: per-device setup + launch, reusing the program's resolved kernels.
// emule's analogue of dispatching the already-compiled program to one chip. dram_core (the
// chip's DRAM backing) and the bank-table globals are per device.
// ---------------------------------------------------------------------------
static void dispatch_to_device(
    IDevice* device, Program& program, ResolvedProgram& resolved, bool defer_run) {
    auto& impl = program.impl();
    auto device_id = device->id();
    auto* sw_emu = get_sw_emulated_chip(device_id);

    tt_emule::Core* dram_core = nullptr;
    uint32_t num_dram_channels = 0;
    uint32_t num_l1_banks = 0;
    populate_bank_mapping(sw_emu, device, device_id, dram_core, num_dram_channels, num_l1_banks);

    auto* core_map_ptr = build_core_map(sw_emu, device, device_id);
    std::vector<CoreSetup> core_setups;
    setup_core_state(impl, device, sw_emu, resolved.core_kernels, resolved.emule_sem_base, core_setups);

    uint8_t* dram_data = dram_core ? dram_core->l1_data() : nullptr;

    OobStateOwner oob = build_oob_tensor_state(device, device_id);
    launch_cores(core_setups, dram_data, core_map_ptr, device_id, defer_run, oob.state);
}

// ---------------------------------------------------------------------------
// execute_program_emulated: Main entry point. Mirrors silicon's compile-once / dispatch-
// reuse: prepare_program resolves once (memoized by program id), dispatch_to_device runs
// per device against the shared resolved kernels.
// ---------------------------------------------------------------------------
void execute_program_emulated(IDevice* device, Program& program) {
    auto device_id = device->id();
    log_debug(tt::LogMetal, "execute_program_emulated: device {} starting", device_id);
    // Mark the fabric connection-route table stale: the next op's first connection record clears it, so
    // routes stay scoped to the current op (this op's builds already recorded before this launch).
    g_conn_route_dirty.store(true, std::memory_order_relaxed);

    ResolvedProgram& resolved = prepare_program(device, program);  // compile-once (memoized)

    const bool defer = g_emule_mesh_defer;  // mesh register phase (the run is deferred)
    dispatch_to_device(device, program, resolved, defer);

    if (defer) {
        log_debug(tt::LogMetal, "execute_program_emulated: device {} registered (deferred mesh run)", device_id);
        return;
    }
    log_debug(tt::LogMetal, "execute_program_emulated: device {} done", device_id);
}

// ---------------------------------------------------------------------------
// Mesh register/run split (see header). begin_mesh_dispatch puts execute_program_emulated
// into defer mode; run_mesh_dispatch drives the single concurrent run + frees kept state.
// ---------------------------------------------------------------------------
void begin_mesh_dispatch() {
    g_emule_mesh_defer = true;
    g_mesh_dfb_keep.clear();
}

void run_mesh_dispatch() {
#if defined(__x86_64__) && defined(__linux__)
    EmuleSigfpeGuard sigfpe_guard;  // the actual kernel run happens here, across all chips
#endif
    // All devices' fibers were registered (spawned) during the per-device register phase; run them
    // concurrently on the worker pool in one pass. Each fiber's ctx carries its device's
    // core_map/bridge_dram, so cross-chip NOC resolution stays correct. run_persistent (vs
    // run_until_idle) lets a host-interleaved socket program quiesce back to the host mid-run.
    tt::tt_metal::emule_fiber::RunOutcome oc = tt::tt_metal::emule_fiber::RunOutcome::Completed;
    try {
        oc = tt::tt_metal::emule_fiber::FiberScheduler::instance().run_persistent();
    } catch (...) {
        // Completed-with-exception (kernel throw / quiescent deadlock): free the kept state, rethrow.
        g_emule_mesh_defer = false;
        g_mesh_dfb_keep.clear();
        throw;
    }
    if (oc == tt::tt_metal::emule_fiber::RunOutcome::HostWait) {
        // A kernel is parked on a host-fed socket wait. Keep g_mesh_dfb_keep + the scheduler's fibers
        // ALIVE and return to the host; it streams socket tokens and pump_device() drives the run to
        // completion, which runs the deferred cleanup below (see pump_device()).
        g_emule_host_wait = true;
        return;
    }
    // Completed synchronously (no host-fed socket wait): the original register-drain cleanup.
    g_emule_mesh_defer = false;
    g_mesh_dfb_keep.clear();
}

void pump_device() {
    // Drive a parked (run_persistent) mesh run forward one scheduler quantum. No-op unless a run is
    // parked in HostWait (set by run_mesh_dispatch). The host advanced a socket credit word by a raw
    // L1 store, so pump() blanket-re-polls the parked fibers to re-check predicates. When every fiber
    // reaches Done the pump returns Completed — run the mesh cleanup run_mesh_dispatch deferred.
    if (!g_emule_host_wait) {
        return;
    }
    auto oc = tt::tt_metal::emule_fiber::FiberScheduler::instance().pump();
    if (oc == tt::tt_metal::emule_fiber::RunOutcome::Completed) {
        g_emule_host_wait = false;
        g_emule_mesh_defer = false;
        g_mesh_dfb_keep.clear();
    }
}

}  // namespace tt::tt_metal::emule
