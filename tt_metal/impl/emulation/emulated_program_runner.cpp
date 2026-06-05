// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "emulated_program_runner.hpp"
#include "emule_live_ranges.hpp"
#include "host_sanitizers.hpp"

#include <dlfcn.h>
#include <unistd.h>
#include <sys/types.h>
#include <csignal>
#if defined(__x86_64__) && defined(__linux__)
#include <ucontext.h>
#include <sys/ucontext.h>
#endif

#include <bit>
#include <atomic>
#include <cassert>
#include <tt_stl/assert.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
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
#include "tt_emule/device.hpp"
#include "tt_emule/dfb_sync_state.hpp"
#include "tt_emule/tile_counter.hpp"
// Emit the single definition of __emule_asan_panic into libtt_metal here (this
// TU has the tt-emule include path and is built with GNU features available).
#define EMULE_ASAN_IMPLEMENTATION
#include "jit_hw/emule_asan.h"
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

// Point into this core's L1 at kernel_config_base + rta_offset[proc_idx],
// where WriteRuntimeArgsToDevice already wrote the bytes. nullptr = sentinel
// (kernel declared no rt-args for this RISC).
thread_local uint32_t* __rt_args = nullptr;
thread_local uint32_t* __common_rt_args = nullptr;

thread_local tt_emule::Core* __core = nullptr;
thread_local tt_emule::Device* __device = nullptr;

// Memory bridge pointers — now non-static for -rdynamic export.
thread_local uint8_t* __emule_bridge_l1 = nullptr;
thread_local uint8_t* __emule_bridge_dram = nullptr;

// Per-core CB state array, shared between threads on the same core.
thread_local __emule_cb_state* __emule_cbs = nullptr;

// Per-thread DFB interface array (one entry per DFB on the core).
thread_local tt_emule::EmuleDFBInterface* __emule_dfbs = nullptr;

// Per-core tile counter array, shared between threads on the same core.
thread_local tt_emule::TileCounterArray* __emule_tc_array = nullptr;

// Quasar-specific per-thread identity, written by the runner at thread start
// (see launch_cores below) and read inside the JIT'd kernel .so. Each variable
// stands in for a different hardware signal; they are NOT interchangeable.
//
// __processor_id  — RISC-V mhartid analogue. DM threads: DM index 0..7.
//                   Compute threads: Neo engine index 0..3. Consumed by the
//                   JIT regex that rewrites `asm volatile("csrr %0, mhartid" ...)`
//                   into `VAR = __processor_id;` (x86 can't execute the CSR).
//
// __emule_neo_id  — Quasar NEO_ID CSR (0xBC2). Which of the 4 compute engines
//                   in a Neo is executing. Set to processor_id for compute
//                   threads, 0 for DM. Read by ckernel::csr_read<CSR::NEO_ID>().
//
// __emule_trisc_id — Quasar TRISC_ID CSR (0xBC3). Which TRISC sub-engine
//                    (0=UNPACK, 1=MATH, 2=PACK, 3=ISOLATE_SFPU) is running.
//                    Starts at 0; for Quasar compute kernels the launcher
//                    iterates it 0..3 across ki.variants (see the variant
//                    loop in launch_cores). Read by
//                    ckernel::csr_read<CSR::TRISC_ID>().
//
// __emule_num_threads  — backs get_num_threads(). Total threads this kernel
//                        runs on (DM count for DM kernels, active-engine count
//                        for compute).
//
// __emule_my_thread_id — backs get_my_thread_id(). Index of this processor within
//                        the kernel's processor list (0-based), matching the Quasar
//                        firmware's my_thread_id semantics. NOT the same as
//                        __processor_id: e.g. a consumer on RISCV_1 has
//                        __processor_id=1 but __emule_my_thread_id=0 (first consumer).
thread_local uint8_t __processor_id = 0;
thread_local uint8_t __emule_neo_id = 0;
thread_local uint8_t __emule_trisc_id = 0;
thread_local uint32_t __emule_num_threads = 1;
thread_local uint32_t __emule_my_thread_id = 0;

// Mirrored from tt-emule/src/kernel_runner.cpp — the two libs are never linked
// into the same binary, so the duplicate definitions are benign.
thread_local uint32_t __emule_sem_l1_range_start = 0;
thread_local uint32_t __emule_sem_l1_range_end = 0;
thread_local const char* __emule_kernel_name = nullptr;
thread_local uint32_t __emule_pending_noc_reads = 0;
thread_local uint32_t __emule_l1_unreserved_base = 0;
thread_local const uint64_t* __emule_l1_tensor_ranges = nullptr;
thread_local uint32_t __emule_l1_tensor_ranges_count = 0;
thread_local const uint64_t* __emule_l1_padding_ranges = nullptr;
thread_local uint32_t __emule_l1_padding_ranges_count = 0;
thread_local uint64_t* __emule_l1_resolved_ranges = nullptr;
thread_local uint32_t* __emule_l1_resolved_ranges_count = nullptr;
thread_local uint32_t __emule_l1_resolved_ranges_capacity = 0;
thread_local uint32_t __emule_cb_reserved_pages[32] = {};
thread_local uint32_t __emule_cb_waited_pages[32] = {};
thread_local const char* __emule_cb_reserve_file[32] = {};
thread_local uint32_t __emule_cb_reserve_line[32] = {};
thread_local const char* __emule_cb_wait_file[32] = {};
thread_local uint32_t __emule_cb_wait_line[32] = {};
thread_local bool __emule_cb_boundary_strict = false;

// DRAM equivalent of __emule_l1_tensor_ranges; consumed only by __emule_dram_ptr below.
thread_local uint32_t __emule_dram_unreserved_base = 0;
thread_local const uint64_t* __emule_dram_tensor_ranges = nullptr;
thread_local uint32_t __emule_dram_tensor_ranges_count = 0;

// Core map for cross-core NOC address resolution (shared across all threads).
thread_local std::unordered_map<uint64_t, tt_emule::Core*>* __emule_core_map = nullptr;

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
// Wormhole has 32 CBs; JIT header cb_api.h sizes unpack_tile_size[32].
static constexpr uint32_t EMULE_NUM_CBS = 32;
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

// NOC encoding constants (matching firmware for Blackhole/Wormhole).
static constexpr uint32_t NOC_LOCAL_BITS = 36;
static constexpr uint32_t NOC_NODE_ID_BITS = 6;
static constexpr uint64_t NOC_LOCAL_MASK = (1ULL << NOC_LOCAL_BITS) - 1;
static constexpr uint32_t NOC_NODE_MASK = (1 << NOC_NODE_ID_BITS) - 1;

// C-linkage bridge functions for JIT kernels.
extern "C" uint8_t* __emule_dram_ptr(uint64_t offset) {
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
    return __emule_bridge_dram ? __emule_bridge_dram + offset : nullptr;
}

extern "C" uint8_t* __emule_local_l1_ptr(uint32_t offset) {
    if (__emule_sem_l1_range_end > 0 &&
        offset >= __emule_sem_l1_range_start && offset < __emule_sem_l1_range_end) {
        __emule_asan_panic(
            "[ASAN ERROR] Illegal Semaphore Access: Offset 0x%x is inside the reserved Semaphore region [0x%x, 0x%x)\n",
            offset,
            __emule_sem_l1_range_start,
            __emule_sem_l1_range_end);
    }
    return __emule_bridge_l1 ? __emule_bridge_l1 + offset : nullptr;
}

extern "C" uint8_t* __emule_noc_resolve(uint32_t x, uint32_t y, uint64_t addr) {
    if (__emule_core_map) {
        uint64_t key = (uint64_t(x) << 32) | y;
        auto it = __emule_core_map->find(key);
        if (it != __emule_core_map->end()) {
            return it->second->l1_ptr(static_cast<uint32_t>(addr));
        }
    }
    return nullptr;
}

// Resolve a NOC address (encoded 64-bit) to a host pointer.
// Real firmware encoding: y in bits [47:42], x in bits [41:36], addr in bits [35:0]
//
// The L1_SLOT_MASK is applied ONLY for WORKER cores. Two reasons:
//  1. The mask handles a worker-kernel pattern where the L1 offset is a
//     truncated host pointer (from `get_write_ptr()`) rather than a
//     firmware-style L1 offset. Worker L1 slots are 2 MB-aligned, so the
//     masked low bits recover the in-slot offset.
//  2. DRAM banks are GB-scale (2 GB on Wormhole views, 4 GB on Blackhole)
//     and the kernel-side per-bank addrgen helper produces an `addr` field
//     that is the true in-bank offset (already includes
//     `bank_to_dram_offset[bank_index]`). Masking to 2 MB silently aliases
//     any DRAM access >= 2 MB to an offset within the first 2 MB of the bank.
extern "C" uint8_t* __emule_resolve_noc_addr(uint64_t noc_addr) {
    uint32_t noc_x = (noc_addr >> NOC_LOCAL_BITS) & NOC_NODE_MASK;
    uint32_t noc_y = (noc_addr >> (NOC_LOCAL_BITS + NOC_NODE_ID_BITS)) & NOC_NODE_MASK;
    uint64_t local_addr = noc_addr & NOC_LOCAL_MASK;  // 36 bits, raw

    static constexpr uint32_t L1_SLOT_SIZE = 2u * 1024 * 1024;  // 2 MB per worker L1 slot
    static constexpr uint32_t L1_SLOT_MASK = L1_SLOT_SIZE - 1;  // 0x1FFFFF

    if (__emule_core_map) {
        uint64_t key = (uint64_t(noc_x) << 32) | noc_y;
        auto it = __emule_core_map->find(key);
        if (it != __emule_core_map->end()) {
            uint32_t offset = (it->second->role() == tt_emule::CoreRole::WORKER)
                                  ? (static_cast<uint32_t>(local_addr) & L1_SLOT_MASK)
                                  : static_cast<uint32_t>(local_addr);
            return it->second->l1_ptr(offset);
        }
    }
    return nullptr;
}

extern "C" bool __emule_noc_addr_is_dram(uint64_t noc_addr) {
    if (!__emule_core_map) {
        return false;
    }
    uint32_t noc_x = (noc_addr >> NOC_LOCAL_BITS) & NOC_NODE_MASK;
    uint32_t noc_y = (noc_addr >> (NOC_LOCAL_BITS + NOC_NODE_ID_BITS)) & NOC_NODE_MASK;
    uint64_t key = (uint64_t(noc_x) << 32) | noc_y;
    auto it = __emule_core_map->find(key);
    if (it != __emule_core_map->end()) {
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
extern "C" void __emule_multicast_write(uint64_t mcast_addr, const uint8_t* src, uint32_t size, bool include_self) {
    uint32_t x_end = (mcast_addr >> NOC_LOCAL_BITS) & NOC_NODE_MASK;
    uint32_t y_end = (mcast_addr >> (NOC_LOCAL_BITS + NOC_NODE_ID_BITS)) & NOC_NODE_MASK;
    uint32_t x_start = (mcast_addr >> (NOC_LOCAL_BITS + 2 * NOC_NODE_ID_BITS)) & NOC_NODE_MASK;
    uint32_t y_start = (mcast_addr >> (NOC_LOCAL_BITS + 3 * NOC_NODE_ID_BITS)) & NOC_NODE_MASK;
    uint64_t l1_offset = mcast_addr & NOC_LOCAL_MASK;

    // The L1 offset may be a truncated host pointer (from get_write_ptr()) rather
    // than a firmware-style L1 offset.  Worker L1 slots are 2 MB-aligned, so
    // masking with SLOT_MASK extracts the true within-slot offset.  For
    // firmware-style offsets (< 2 MB) this is a no-op.
    // Multicast targets only WORKER cores (DRAM cores are skipped by the role
    // check in the delivery loop below), so the mask is L1-correct here.
    static constexpr uint32_t L1_SLOT_SIZE = 2u * 1024 * 1024;  // 2 MB per worker L1 slot
    static constexpr uint32_t L1_SLOT_MASK = L1_SLOT_SIZE - 1;  // 0x1FFFFF
    l1_offset &= L1_SLOT_MASK;

    if (!__emule_core_map) {
        return;
    }

    // Sender coordinates (from the TLS that thread launch wires up). Used to
    // skip self when include_self=false (non-loopback multicast).
    uint32_t self_x = my_x[0];
    uint32_t self_y = my_y[0];

    uint32_t delivered = 0;
    for (uint32_t x = std::min(x_start, x_end); x <= std::max(x_start, x_end); x++) {
        for (uint32_t y = std::min(y_start, y_end); y <= std::max(y_start, y_end); y++) {
            if (!include_self && x == self_x && y == self_y) {
                continue;
            }
            uint64_t key = (uint64_t(x) << 32) | y;
            auto it = __emule_core_map->find(key);
            if (it != __emule_core_map->end() && it->second->role() == tt_emule::CoreRole::WORKER) {
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
                } else {
                    std::memcpy(dst, src, size);
                    std::atomic_thread_fence(std::memory_order_release);
                }
                delivered++;
            }
        }
    }
    if (delivered == 0) {
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
    // Flat list of this kernel's runtime-arg values on its core (unique + common).
    // Buffer L1 addresses handed to the kernel appear here verbatim, so they let
    // the Object-Intent check discover which tensors are this kernel's I/O. See
    // ObjectIntentTracker::pre_launch_snapshot.
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
    // L1 start offsets of globally-allocated (persistent) CB backing buffers on
    // this core. These are legitimate kernel write targets (the CB *is* the
    // tensor), so the Object-Intent check must not flag writes to them.
    std::vector<uint32_t> persistent_cb_starts;
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
    // Bake a fingerprint of all JIT headers into the directory name. Any header
    // change produces a new directory, automatically invalidating the entire
    // cache without relying on per-file mtime comparisons at lookup time.
    static const std::string dir_path = [] {
        uint64_t max_ns = 0;
        std::error_code ec;
        for (auto& e : std::filesystem::recursive_directory_iterator(TT_EMULE_JIT_INCLUDE_DIR, ec)) {
            if (!e.is_regular_file()) {
                continue;
            }
            auto t = static_cast<uint64_t>(e.last_write_time().time_since_epoch().count());
            if (t > max_ns) {
                max_ns = t;
            }
        }
        char hex[17];
        std::snprintf(hex, sizeof(hex), "%016lx", max_ns);
        return "/tmp/tt_emule_jit_cache_" + std::to_string(getuid()) + "_" + hex;
    }();
    return dir_path;
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

// Rewrite RISC-V-specific inline asm and raw-L1-pointer idioms so the kernel
// compiles for x86 host.  The rewrites:
//   1. `asm volatile("csrr %0, mhartid" : "=r"(V));` → `V = __processor_id;`
//      (x86 assembler rejects RISC-V CSR instructions; the runner sets the
//      __processor_id TLS before each kernel launch.)
//   2. `asm volatile("fence" ::: "memory");` → `__sync_synchronize();`
//      (Host memory barrier is the closest emulation-side equivalent.)
//   3. `reinterpret_cast<T*>(get_arg_val<uint32_t>(N))` →
//      `reinterpret_cast<T*>((uintptr_t)__emule_local_l1_to_ptr(get_arg_val<uint32_t>(N)))`
//      (Quasar kernels pass raw L1 firmware offsets as runtime args; x86 needs
//      translation through the per-thread __emule_bridge_l1 base pointer.)
// Reads from `src_path`, writes the patched source to `out_path`, and throws
// on any I/O failure.
// emule_line_preserving_replace that preserves the total line count: each replacement is
// padded with as many trailing newlines as the match consumed beyond it (our
// replacements never add newlines, so this only ever pads). Keeping the emitted
// source line-aligned with the original kernel is what lets the `#line` directive
// below make debug info — and therefore ASAN backtraces — point at the real
// kernel file:line instead of the generated temp copy.
static std::string emule_line_preserving_replace(
    const std::string& input, const std::regex& re, const std::string& fmt) {
    std::string out;
    auto pos = input.cbegin();
    for (std::sregex_iterator it(input.cbegin(), input.cend(), re), end; it != end; ++it) {
        const std::smatch& m = *it;
        out.append(pos, m[0].first);
        const std::string matched = m.str();
        const std::string rep = m.format(fmt);
        const long pad = std::count(matched.begin(), matched.end(), '\n') - std::count(rep.begin(), rep.end(), '\n');
        out += rep;
        for (long k = 0; k < pad; ++k) {
            out += '\n';
        }
        pos = m[0].second;
    }
    out.append(pos, input.cend());
    return out;
}

static void preprocess_kernel_source_for_x86(const std::string& src_path, const std::string& out_path) {
    std::ifstream in(src_path);
    if (!in) {
        throw std::runtime_error("preprocess_kernel_source_for_x86: cannot read " + src_path);
    }
    std::stringstream ss;
    ss << in.rdbuf();
    std::string src = ss.str();

    static const std::regex mhartid_re(
        R"(asm\s+volatile\s*\(\s*"csrr\s+%0\s*,\s*mhartid"\s*:\s*"=r"\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\)\s*;)");
    src = emule_line_preserving_replace(src, mhartid_re, "$1 = __processor_id;");

    static const std::regex fence_re(R"(asm\s+volatile\s*\(\s*"fence"\s*:::\s*"memory"\s*\)\s*;)");
    src = emule_line_preserving_replace(src, fence_re, "__sync_synchronize();");

    static const std::regex l1_arg_ptr_re(
        R"(reinterpret_cast<([^>]+\*)>\s*\(\s*get_arg_val<uint32_t>\s*(\([^)]*\))\s*\))");
    src = emule_line_preserving_replace(
        src, l1_arg_ptr_re, "reinterpret_cast<$1>((uintptr_t)__emule_local_l1_to_ptr(get_arg_val<uint32_t>$2))");

    // Metal 2.0 named-arg pattern: reinterpret_cast<T*>(static_cast<uintptr_t>(get_arg(args::NAME)))
    static const std::regex l1_named_arg_ptr_re(
        R"(reinterpret_cast<([^>]+\*)>\s*\(\s*static_cast<uintptr_t>\s*\(\s*get_arg\s*\(\s*([^)]+)\s*\)\s*\)\s*\))");
    src = emule_line_preserving_replace(
        src,
        l1_named_arg_ptr_re,
        "reinterpret_cast<$1>((uintptr_t)__emule_local_l1_to_ptr(static_cast<uint32_t>(get_arg($2))))");

    std::ofstream out(out_path);
    if (!out) {
        throw std::runtime_error("preprocess_kernel_source_for_x86: cannot write " + out_path);
    }
    // Attribute the emitted body to the real kernel file so DWARF (and thus ASAN
    // backtraces) report `<real kernel>.cpp:<line>` rather than the generated temp
    // copy. The rewrites above are line-preserving, so line N here == line N there.
    std::error_code ec;
    std::filesystem::path abs = std::filesystem::absolute(src_path, ec);
    out << "#line 1 \"" << (ec ? src_path : abs.string()) << "\"\n";
    out << src;
}

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
    const bool has_args = !s.runtime_arg_names.empty() || !s.common_runtime_arg_names.empty() ||
                          !named_compile_args.empty();
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
    preprocess_kernel_source_for_x86(abs_kernel, patched_kernel_path);

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
        emit_metal2_namespaces(f, bindings, named_compile_args);
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
    // Under ASAN, keep -O2 (so numerics match normal runs) but add debug info +
    // frame pointers so the unified ASAN backtrace can resolve kernel-source
    // file:line via llvm-symbolizer/addr2line. The ASAN flag is folded into the
    // JIT cache key (see compute_cache_key) so these -g .so files never collide
    // with the lean non-ASAN cache.
    std::string opt_flags = " -O2";
    if (tt::tt_metal::emule::emule_asan_enabled()) {
        opt_flags += " -g -fno-omit-frame-pointer -funwind-tables";
    }
    std::ostringstream cmd;
    cmd << TT_EMULE_CXX_COMPILER << " -std=c++" << TT_EMULE_CXX_STANDARD << " -fPIC -shared" << opt_flags
        << " -Wno-c++11-narrowing"
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
    std::filesystem::remove_all(dir);

    // 11. Wrap in shared_ptr for lifetime management (dlclose on destruction).
    auto shared_handle = std::shared_ptr<void>(handle, [](void* h) { dlclose(h); });
    return [fn, shared_handle]() { fn(); };
}

// ---------------------------------------------------------------------------
// Helper: get SWEmuleChip* from MetalContext cluster for a given device_id.
// ---------------------------------------------------------------------------
static tt::umd::SWEmuleChip* get_sw_emulated_chip(ChipId device_id) {
    auto& cluster = MetalContext::instance().get_cluster();
    auto* umd_cluster = cluster.get_driver().get();
    if (!umd_cluster) {
        return nullptr;
    }
    auto* chip = umd_cluster->get_chip(device_id);
    return dynamic_cast<tt::umd::SWEmuleChip*>(chip);
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

    if (!dram_channels.empty() && !dram_channels[0].empty()) {
        auto& dc = dram_channels[0][0];
        dram_core_out = sw_emu->get_core(tt_xy_pair(dc.x, dc.y));
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
    // Populate per-NOC preferred coords separately. On Wormhole the NOC-0 and
    // NOC-1 preferred workers for a given DRAM view differ (e.g. channel 0
    // NOC0=[2,2], NOC1=[1,1]). On Blackhole they happen to match, so this is
    // a no-op change there.
    std::memset(dram_bank_to_noc_xy, 0, sizeof(dram_bank_to_noc_xy));
    std::memset(bank_to_dram_offset, 0, sizeof(bank_to_dram_offset));
    for (uint32_t ch = 0; ch < num_dram_channels_out && ch < MAX_NUM_BANKS; ch++) {
        auto dc0 = metal_soc.get_preferred_worker_core_for_dram_view(ch, 0 /* NOC 0 */);
        auto dc1 = metal_soc.get_preferred_worker_core_for_dram_view(ch, 1 /* NOC 1 */);
        uint16_t noc_xy0 = (static_cast<uint16_t>(dc0.y) << NOC_NODE_ID_BITS) | static_cast<uint16_t>(dc0.x);
        uint16_t noc_xy1 = (static_cast<uint16_t>(dc1.y) << NOC_NODE_ID_BITS) | static_cast<uint16_t>(dc1.x);
        dram_bank_to_noc_xy[0][ch] = noc_xy0;
        dram_bank_to_noc_xy[1][ch] = noc_xy1;
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
            l1_bank_to_noc_xy[0][b] = noc_xy;  // NOC 0
            l1_bank_to_noc_xy[1][b] = noc_xy;  // NOC 1 (same target in emule)
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

    // Collect CB tile sizes from program for constexpr get_tile_size().
    const auto& core_range_set = kernel.core_range_set();
    if (!core_range_set.ranges().empty()) {
        auto first_core = core_range_set.ranges().begin()->start_coord;
        auto cb_impls = impl.circular_buffers_on_core(first_core);
        uint32_t tile_sizes[EMULE_NUM_CBS] = {};
        // Per-CB data format → emule's analog of genfiles.cpp::compute_data_formats()
        // (which bakes unpack_src_format[]/pack_dst_format[] into chlkc_descriptors.h).
        // 255 == tt::DataFormat::Invalid marks unconfigured slots (mirrors the host's
        // std::optional<DataFormat> empty state); consumers fall back to the page_size
        // heuristic for those.
        uint8_t cb_formats[EMULE_NUM_CBS];
        for (uint32_t i = 0; i < EMULE_NUM_CBS; i++) {
            cb_formats[i] = static_cast<uint8_t>(tt::DataFormat::Invalid);
        }
        for (auto& cb_impl : cb_impls) {
            for (uint8_t idx : cb_impl->local_buffer_indices()) {
                if (idx < EMULE_NUM_CBS) {
                    tile_sizes[idx] = cb_impl->page_size(idx);
                    cb_formats[idx] = static_cast<uint8_t>(cb_impl->data_format(idx));
                }
            }
        }
        std::ostringstream ts, df;
        for (uint32_t i = 0; i < EMULE_NUM_CBS; i++) {
            if (i) {
                ts << ',';
                df << ',';
            }
            ts << tile_sizes[i];
            df << static_cast<uint32_t>(cb_formats[i]);
        }
        defines["EMULE_TILE_SIZES"] = ts.str();
        defines["EMULE_CB_DATA_FORMATS"] = df.str();
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

            auto compile_args = kernel->compile_time_args();
            auto named_compile_args = kernel->named_compile_time_args();
            auto defines = build_kernel_defines(
                *kernel, impl, num_dram_channels, num_l1_banks,
                worker_col_map_str, worker_row_map_str, emule_sem_base);

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
            //
            // EXCEPT for tilize kernels: emule's host-side
            // `tilize_with_val_padding` already produces tiled data, so an
            // additional kernel-side tilize would re-tilize and corrupt the
            // layout. Skip TRISC defines when the kernel source mentions
            // `llk_unpack_tilize` (the tilize compute path).
            bool is_tilize_kernel = false;
            if (is_tensix && !is_quasar_compute) {
                std::ifstream kscan(src_path);
                if (!kscan) {
                    throw std::runtime_error(
                        "collect_kernels: cannot read kernel source for TRISC-define gating: " + src_path);
                }
                std::string content((std::istreambuf_iterator<char>(kscan)), std::istreambuf_iterator<char>());
                is_tilize_kernel = content.find("llk_unpack_tilize") != std::string::npos;
            }
            if (is_tensix && !is_quasar_compute && !is_tilize_kernel) {
                defines["TRISC_UNPACK"] = "1";
                defines["TRISC_MATH"] = "1";
                defines["TRISC_PACK"] = "1";
            }

            // Metal 2.0 bindings — same across this Kernel's TRISC variants, so
            // capture the cache-key suffix once and append it to every variant key.
            Metal2BindingsSnapshot bindings = build_metal2_snapshot(*kernel);
            const std::string metal2_key_suffix = bindings.cache_key_suffix();

            // Helper: compute cache key from a defines map.
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
                // Codegen version: bump whenever preprocess_kernel_source_for_x86 /
                // the emitted wrapper changes, so stale cached .so files (whose DWARF
                // or generated source predate the change) are recompiled. pp1->pp2:
                // line-preserving rewrites + #line directive pointing at the real kernel.
                key += ":pp2";
                // ASAN builds add -g/-fno-omit-frame-pointer; keep their cached
                // .so distinct from the lean non-ASAN build of the same kernel.
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
                        deferred_compiles[key] =
                            DeferredCompile{src_path, compile_args, named_compile_args, defs, extra_inc, bindings};
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
                        // Snapshot the runtime-arg values handed to this kernel on
                        // this core (unique + common). Buffer L1 addresses passed as
                        // args appear here verbatim; the Object-Intent check matches
                        // them against live-tensor starts to discover the kernel's I/O
                        // tensors (see ObjectIntentTracker::pre_launch_snapshot). Shared
                        // across this kernel's proc_ids, so build once and copy.
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
static void jit_compile_pending(
    std::map<std::string, DeferredCompile>& deferred_compiles,
    std::unordered_map<std::string, std::function<void()>>& resolved_fns,
    std::vector<std::string>& inline_src_temps) {
    if (!deferred_compiles.empty()) {
        log_info(tt::LogMetal, "JIT parallel compile: {} unique kernels to compile", deferred_compiles.size());

        std::vector<std::pair<std::string, std::future<std::function<void()>>>> futures;
        futures.reserve(deferred_compiles.size());

        for (auto& [key, dc] : deferred_compiles) {
            std::string cache_path = disk_cache_so_path(key);
            std::string tmp_path = cache_path + ".tmp." + std::to_string(::getpid());
            futures.emplace_back(
                key, std::async(std::launch::async, [&dc, cache_path, tmp_path]() {
                    auto fn = jit_compile_kernel(
                        dc.src_path, dc.compile_args, dc.named_compile_args, dc.defines, dc.extra_inc,
                        dc.bindings, tmp_path);
                    std::filesystem::rename(tmp_path, cache_path);
                    return fn;
                }));
        }

        for (auto& [key, fut] : futures) {
            auto fn = fut.get();
            resolved_fns[key] = fn;
            std::lock_guard<std::mutex> lock(g_jit_cache_mutex);
            g_jit_cache[key] = fn;
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
static std::unordered_map<uint64_t, tt_emule::Core*>* build_core_map(
    tt::umd::SWEmuleChip* sw_emu, IDevice* device, ChipId device_id) {
    static std::mutex g_core_map_mutex;
    static std::unordered_map<uint32_t, std::shared_ptr<std::unordered_map<uint64_t, tt_emule::Core*>>>
        g_core_map_cache;

    std::lock_guard<std::mutex> lock(g_core_map_mutex);
    auto& core_map = g_core_map_cache[device_id];
    if (!core_map && sw_emu) {
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
        // Add DRAM cores (UMD SoC descriptor coords)
        auto& umd_soc = sw_emu->get_soc_descriptor();
        for (auto& dc_vec : umd_soc.get_dram_cores()) {
            for (auto& dc : dc_vec) {
                auto* core = sw_emu->get_core(tt_xy_pair(dc.x, dc.y));
                uint64_t key = (uint64_t(dc.x) << 32) | dc.y;
                (*core_map)[key] = core;
            }
        }
        // Add DRAM cores (metal_SocDescriptor preferred worker coords)
        // Register BOTH NOC0 and NOC1 preferred coords — on Wormhole they
        // differ per channel (e.g. ch0 NOC0=[2,2], NOC1=[1,1]); both must
        // be in the core_map so __emule_resolve_noc_addr can route either.
        {
            auto& msoc = MetalContext::instance().get_cluster().get_soc_desc(device_id);
            for (uint32_t ch = 0; ch < msoc.get_num_dram_views() && ch < MAX_NUM_BANKS; ch++) {
                for (uint32_t noc = 0; noc < NUM_NOCS; noc++) {
                    auto dc = msoc.get_preferred_worker_core_for_dram_view(ch, noc);
                    auto* core = sw_emu->get_core(tt_xy_pair(dc.x, dc.y));
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
// setup_core_state: Configure CBs and semaphores per core, build CoreSetup list.
// ---------------------------------------------------------------------------
// Initialize CB-sync state on a core from the program's circular buffer list.
static void init_core_cb_sync(
    tt_emule::Core* core,
    detail::ProgramImpl& impl,
    const CoreCoord& logical_core,
    std::vector<uint32_t>& persistent_cb_starts) {
    core->reset_cb_sync();
    auto cb_impls = impl.circular_buffers_on_core(logical_core);
    for (auto& cb_impl : cb_impls) {
        // A globally-allocated CB is bound to a persistent L1 buffer (its
        // address() == that buffer's address()). Record it so Object-Intent
        // exempts the kernel's legitimate writes to it.
        if (cb_impl->globally_allocated()) {
            persistent_cb_starts.push_back(cb_impl->address());
        }
        for (uint8_t idx : cb_impl->local_buffer_indices()) {
            if (idx >= EMULE_NUM_CBS) {
                continue;
            }
            uint32_t cb_addr = cb_impl->address();
            uint32_t page_size = cb_impl->page_size(idx);
            uint32_t num_pages = (page_size > 0) ? cb_impl->num_pages(idx) : 0;
            uint8_t* base = (page_size > 0) ? core->l1_ptr(cb_addr) : nullptr;
            core->init_cb_sync(idx, base, page_size, num_pages);
            log_debug(
                tt::LogMetal,
                "  Core({},{}) CB[{}]: addr=0x{:x} page_size={} num_pages={} base={:p}",
                logical_core.x,
                logical_core.y,
                idx,
                cb_addr,
                page_size,
                num_pages,
                (void*)base);
        }
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
        // Prefer the finalize-allocated L1 address (so host/test verification
        // hits the same offset); fall back to bump-alloc when absent.
        auto cl = dfb_impl->core_lookup_.find(logical_core);
        uint32_t finalize_addr = (cl != dfb_impl->core_lookup_.end()) ? core->l1_base_addr() + cl->second.second : 0;
        uint32_t base_addr;
        if (compute_is_producer && !compute_is_consumer) {
            auto it = bridge_consumer_alloc.find(dim_key);
            base_addr = (it != bridge_consumer_alloc.end()) ? it->second
                                                            : (finalize_addr ? finalize_addr : core->l1_alloc(total));
        } else {
            base_addr = finalize_addr ? finalize_addr : core->l1_alloc(total);
            if (compute_is_consumer && !compute_is_producer) {
                bridge_consumer_alloc.emplace(dim_key, base_addr);
            }
        }
        // `base_addr` is already a host pointer truncated to uint32_t — the L1 pool
        // is mmap'd with MAP_32BIT so every L1 address fits in the low 32 bits.
        // Reconstructing the host pointer is a widening cast, not a new allocation.
        // See docs/QUASAR_EMULATION.md §4.1 and IMPLEMENTATION_REPORT.md "Address Translation".
        uint8_t* base = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(base_addr));
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

        std::vector<uint32_t> persistent_cb_starts;
        init_core_cb_sync(core, impl, logical_core, persistent_cb_starts);
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
             std::move(persistent_cb_starts)});
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

// Sanitizer state threaded into each kernel thread. See SANITIZERS.md.
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
    bool object_intent_strict = false;
};

// ===========================================================================
// Sanitizer helpers. Every entry point is a no-op when ASAN is disabled, so
// launch_cores / execute_program_emulated can call them unconditionally and
// stay free of `if (asan_enabled)` clutter. See SANITIZERS.md.
// ===========================================================================

class ObjectIntentTracker {
public:
    void pre_launch_snapshot(
        const EmuleOobTensorState& oob,
        const std::vector<KernelInfo>& ki_list,
        const uint8_t* l1_data,
        const std::vector<uint32_t>& persistent_cb_starts,
        [[maybe_unused]] uint32_t lx,
        [[maybe_unused]] uint32_t ly) {
        if (!oob.object_intent_strict || oob.tensor_ranges == nullptr) {
            return;
        }
        if (ki_list.size() != 1) {
            // Per-kernel provenance relies on memcmp-after-exit to attribute byte changes to a
            // single kernel; with multiple kernels sharing a core in one launch (the normal ttnn
            // reader+compute+writer pattern) we can't tell which kernel wrote which bytes. The
            // check is simply not applicable here, so skip it without snapshotting — leaving
            // snapshots_ empty makes verify_post_launch a no-op — rather than aborting the whole
            // workload. Other sanitizers continue to run.
            return;
        }
        // I/O tensors from other kernels: a tensor whose L1 address was handed to
        // this kernel as a runtime arg is one the kernel was explicitly given to
        // operate on — even if it "belongs" to another kernel's context, this
        // kernel is allowed to write to it (in-place ops, fused producers). The
        // base address passed as a runtime arg equals the buffer's start offset,
        // so any live-tensor start that appears in the kernel's runtime args is an
        // I/O tensor and must be exempt from the snapshot. (Same space, no
        // normalization: runtime-arg buffer addresses and the live-range starts are
        // both direct offsets into the core's L1 backing.)
        std::unordered_set<uint32_t> io_arg_starts(ki_list[0].rt_arg_values.begin(), ki_list[0].rt_arg_values.end());
        snapshots_.reserve(oob.tensor_ranges_count);
        for (uint32_t i = 0; i < oob.tensor_ranges_count; ++i) {
            uint64_t packed = oob.tensor_ranges[i];
            uint32_t r_start = static_cast<uint32_t>(packed >> 32);
            uint32_t r_end = static_cast<uint32_t>(packed);
            if (r_end <= r_start) {
                continue;
            }
            // Skip persistent (globally-allocated CB) buffers: the kernel is
            // allowed to write to them, so they must not
            // be compared. Their address() == the buffer's start offset.
            if (std::find(persistent_cb_starts.begin(), persistent_cb_starts.end(), r_start) !=
                persistent_cb_starts.end()) {
                continue;
            }
            // Skip I/O tensors this kernel was handed (see above).
            if (io_arg_starts.count(r_start) != 0) {
                continue;
            }
            Snap snap;
            snap.packed = packed;
            snap.bytes.resize(r_end - r_start);
            std::memcpy(snap.bytes.data(), l1_data + r_start, r_end - r_start);
            snapshots_.push_back(std::move(snap));
        }
    }

    void setup_kernel_tls(
        const EmuleOobTensorState& oob,
        uint64_t* local_log,
        uint32_t cap,
        uint32_t* count) {
        if (!oob.object_intent_strict) {
            __emule_l1_resolved_ranges = nullptr;
            __emule_l1_resolved_ranges_count = nullptr;
            __emule_l1_resolved_ranges_capacity = 0;
            return;
        }
        __emule_l1_resolved_ranges = local_log;
        __emule_l1_resolved_ranges_count = count;
        __emule_l1_resolved_ranges_capacity = cap;
    }

    void teardown_kernel_tls(
        const EmuleOobTensorState& oob,
        const uint64_t* local_log,
        uint32_t local_count) {
        __emule_l1_resolved_ranges = nullptr;
        __emule_l1_resolved_ranges_count = nullptr;
        __emule_l1_resolved_ranges_capacity = 0;
        if (!oob.object_intent_strict || local_count == 0) {
            return;
        }
        resolved_acc_.insert(resolved_acc_.end(), local_log, local_log + local_count);
    }

    void verify_post_launch(const uint8_t* l1_data, uint32_t lx, uint32_t ly, const char* kernel_name) const {
        if (snapshots_.empty()) {
            return;
        }
        std::unordered_set<uint64_t> resolved_set(resolved_acc_.begin(), resolved_acc_.end());
        for (const auto& snap : snapshots_) {
            if (resolved_set.count(snap.packed)) {
                continue;
            }
            uint32_t r_start = static_cast<uint32_t>(snap.packed >> 32);
            uint32_t r_end = static_cast<uint32_t>(snap.packed);
            if (std::memcmp(snap.bytes.data(), l1_data + r_start, r_end - r_start) != 0) {
                // No source line: this is detected post-exit by memcmp (the stray
                // write bypassed __emule_local_l1_to_ptr, so there is no captured
                // call site). The kernel name + core + the clobbered buffer range
                // are the actionable info; the cause is typically an overrun from
                // an adjacent buffer this kernel *did* resolve.
                __emule_asan_panic(
                    "[ASAN ERROR] Object Intent Violation: Attempted to modify memory belonging to an "
                    "adjacent object context — kernel %s on core (%u, %u) changed L1 buffer [0x%x, 0x%x) "
                    "without ever resolving a pointer into it via __emule_local_l1_to_ptr (likely an overrun "
                    "from an adjacent buffer). No source line: detected post-exit by memory comparison, after "
                    "the kernel returned.\n",
                    kernel_name ? kernel_name : "(unknown)",
                    lx,
                    ly,
                    r_start,
                    r_end);
            }
        }
    }

private:
    struct Snap {
        uint64_t packed;
        std::vector<uint8_t> bytes;
    };
    std::vector<Snap> snapshots_;
    std::vector<uint64_t> resolved_acc_;
};

inline void set_sanitizer_thread_locals(
    const EmuleOobTensorState& oob, uint32_t sem_base, uint32_t sem_size) {
    __emule_sem_l1_range_start = oob.asan_enabled ? sem_base : 0;
    __emule_sem_l1_range_end = oob.asan_enabled ? (sem_base + sem_size) : 0;
    __emule_l1_unreserved_base = oob.l1_unreserved_base;
    __emule_l1_tensor_ranges = oob.tensor_ranges;
    __emule_l1_tensor_ranges_count = oob.tensor_ranges_count;
    __emule_dram_unreserved_base = oob.dram_unreserved_base;
    __emule_dram_tensor_ranges = oob.dram_tensor_ranges;
    __emule_dram_tensor_ranges_count = oob.dram_tensor_ranges_count;
    __emule_l1_padding_ranges = oob.l1_padding_ranges;
    __emule_l1_padding_ranges_count = oob.l1_padding_ranges_count;
    __emule_cb_boundary_strict = oob.cb_boundary_strict;
}

inline void clear_sanitizer_thread_locals() {
    __emule_sem_l1_range_start = 0;
    __emule_sem_l1_range_end = 0;
    __emule_l1_unreserved_base = 0;
    __emule_l1_tensor_ranges = nullptr;
    __emule_l1_tensor_ranges_count = 0;
    __emule_dram_unreserved_base = 0;
    __emule_dram_tensor_ranges = nullptr;
    __emule_dram_tensor_ranges_count = 0;
    __emule_l1_padding_ranges = nullptr;
    __emule_l1_padding_ranges_count = 0;
    for (uint32_t i = 0; i < EMULE_NUM_CBS; ++i) {
        __emule_cb_reserved_pages[i] = 0;
        __emule_cb_waited_pages[i] = 0;
        __emule_cb_reserve_file[i] = nullptr;
        __emule_cb_reserve_line[i] = 0;
        __emule_cb_wait_file[i] = nullptr;
        __emule_cb_wait_line[i] = 0;
    }
    __emule_cb_boundary_strict = false;
}

inline void abort_if_dirty_cb(
    uint32_t cb_id,
    uint32_t unpushed,
    uint32_t unpopped,
    uint32_t lx,
    uint32_t ly,
    uint32_t processor_id,
    const char* reserve_file,
    uint32_t reserve_line,
    const char* wait_file,
    uint32_t wait_line) {
    // The kernel has already returned, so there is no kernel frame to backtrace;
    // the offending file:line comes from the call site captured at reserve/wait
    // time (__emule_cb_reserve_file / __emule_cb_wait_file). Only the imbalanced
    // side(s) are reported.
    char reserve_clause[512] = "";
    if (unpushed > 0) {
        std::snprintf(
            reserve_clause,
            sizeof(reserve_clause),
            " %u page(s) reserved via cb_reserve_back at %s:%u were never committed with cb_push_back.",
            unpushed,
            reserve_file ? reserve_file : "?",
            reserve_line);
    }
    char wait_clause[512] = "";
    if (unpopped > 0) {
        std::snprintf(
            wait_clause,
            sizeof(wait_clause),
            " %u page(s) waited via cb_wait_front at %s:%u were never released with cb_pop_front.",
            unpopped,
            wait_file ? wait_file : "?",
            wait_line);
    }
    __emule_asan_panic(
        "[ASAN ERROR] Dirty CB Detected: Core (%u, %u) CB %u was not flushed! Kernel (processor %u):%s%s "
        "Every reserve must be matched by a push and every wait by a pop before the kernel exits, "
        "or the CB's read/write pointers desync on silicon.\n",
        lx,
        ly,
        cb_id,
        processor_id,
        reserve_clause,
        wait_clause);
}

// A CB is "flushed" when every cb_reserve_back was committed by a matching
// cb_push_back and every cb_wait_front was released by a matching cb_pop_front.
// The per-kernel thread-local counters hold the net unmatched amount at kernel
// exit: __emule_cb_reserved_pages[cb] is bumped by reserve_back and shrunk by
// push_back; __emule_cb_waited_pages[cb] is set by wait_front and shrunk by
// pop_front. Either being > 0 means the kernel reserved/waited without the
// matching push/pop — an un-flushed CB. This is a per-kernel property (reserve
// pairs with push within the producer, wait with pop within the consumer), so it
// is checked at each kernel's exit, before the thread-locals are cleared.
inline void sweep_per_kernel_dirty_cbs(
    const EmuleOobTensorState& oob, tt_emule::CBSyncState* cb_array, uint32_t processor_id, uint32_t lx, uint32_t ly) {
    if (!oob.asan_enabled || cb_array == nullptr) {
        return;
    }
    for (uint32_t cb_id = 0; cb_id < EMULE_NUM_CBS; ++cb_id) {
        if (cb_array[cb_id].num_pages == 0) {
            continue;
        }
        uint32_t unpushed = __emule_cb_reserved_pages[cb_id];
        uint32_t unpopped = __emule_cb_waited_pages[cb_id];
        if (unpushed > 0 || unpopped > 0) {
            abort_if_dirty_cb(
                cb_id,
                unpushed,
                unpopped,
                lx,
                ly,
                processor_id,
                __emule_cb_reserve_file[cb_id],
                __emule_cb_reserve_line[cb_id],
                __emule_cb_wait_file[cb_id],
                __emule_cb_wait_line[cb_id]);
        }
    }
}

// Owns the snapshot vectors that EmuleOobTensorState's pointers reference.
struct OobStateOwner {
    EmuleOobTensorState state;
    std::vector<uint64_t> live_ranges;
    std::vector<uint64_t> dram_live_ranges;
    std::vector<uint64_t> padding_ranges;
};

inline OobStateOwner build_oob_tensor_state(IDevice* device, int device_id) {
    OobStateOwner owner;
    const bool asan = emule_asan_enabled();
    owner.state.asan_enabled = asan;
    owner.state.cb_boundary_strict = asan;
    if (!asan) {
        return owner;
    }
    static const uint64_t kEmptyRange = 0;

    owner.live_ranges = tt::tt_metal::emule::LiveL1Ranges::snapshot(device_id);
    owner.state.l1_unreserved_base = static_cast<uint32_t>(
        device->allocator()->get_base_allocator_addr(HalMemType::L1));
    owner.state.tensor_ranges =
        owner.live_ranges.empty() ? &kEmptyRange : owner.live_ranges.data();
    owner.state.tensor_ranges_count = static_cast<uint32_t>(owner.live_ranges.size());

    owner.dram_live_ranges = tt::tt_metal::emule::LiveDramRanges::snapshot(device_id);
    owner.state.dram_unreserved_base = static_cast<uint32_t>(
        device->allocator()->get_base_allocator_addr(HalMemType::DRAM));
    owner.state.dram_tensor_ranges =
        owner.dram_live_ranges.empty() ? &kEmptyRange : owner.dram_live_ranges.data();
    owner.state.dram_tensor_ranges_count = static_cast<uint32_t>(owner.dram_live_ranges.size());
    owner.state.object_intent_strict = true;

    owner.padding_ranges = tt::tt_metal::emule::LiveL1PaddingRanges::snapshot(device_id);
    if (!owner.padding_ranges.empty()) {
        owner.state.l1_padding_ranges = owner.padding_ranges.data();
        owner.state.l1_padding_ranges_count = static_cast<uint32_t>(owner.padding_ranges.size());
    }
    return owner;
}

// ---------------------------------------------------------------------------
// launch_cores: Spawn concurrent threads per core, each runs its kernels.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// RISC-V-faithful integer divide/modulo fault recovery on x86 hosts.
//
// The real Tensix cores are RISC-V, where integer div/rem faults are DEFINED and
// non-trapping (RISC-V ISA M-extension):
//   - divide by zero: `divu x,0 -> all-ones`, `div x,0 -> -1`, `rem(u) x,0 -> x`
//   - signed overflow (`div INT_MIN,-1`): quotient = INT_MIN, remainder = 0
// emule JIT-compiles each kernel to x86, where `div`/`idiv` raises #DE -> SIGFPE
// (si_code FPE_INTDIV for /0, FPE_INTOVF for INT_MIN/-1) and aborts. Kernels
// legitimately hit /0 on idle/degenerate cores: e.g. binary_ng's no_bcast reader
// computes `start_tile_id % (D*N*C*Ht*Wt)` and the host hands idle cores all-zero
// dims, so the divisor is 0 and the (dead) result is never used. To match silicon we
// trap the SIGFPE, write RISC-V's defined result into the saved register image, and
// step the saved RIP past the faulting instruction.
//
// Scope/caveats: (1) the handler is process-global for the lifetime of launch_cores,
// so any host thread that faults in that window is also "recovered" — acceptable
// because only kernel threads run div-heavy code then; (2) a *genuine* kernel div bug
// becomes silent-wrong-output rather than a crash, but that is exactly what silicon
// would do (no trap). See docs/riscv-intdiv-by-zero.md in the tt-emule repo.
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

static void launch_cores(
    std::vector<CoreSetup>& core_setups,
    uint8_t* dram_data,
    std::unordered_map<uint64_t, tt_emule::Core*>* core_map_ptr,
    const EmuleOobTensorState& oob_state) {
#if defined(__x86_64__) && defined(__linux__)
    EmuleSigfpeGuard sigfpe_guard;
#endif
    std::vector<std::thread> core_threads;
    std::vector<std::exception_ptr> core_exceptions(core_setups.size());

    for (size_t core_idx = 0; core_idx < core_setups.size(); ++core_idx) {
        core_threads.emplace_back(
            [&cs = core_setups[core_idx], dram_data, core_map_ptr, oob_state, &core_ep = core_exceptions[core_idx]]() {
                try {
                    auto* core = cs.core;
                    uint8_t* l1_data = core->l1_data();
                    tt_emule::CBSyncState* cb_array = core->cb_sync_array();
                    tt_emule::TileCounterArray* tc_array = cs.has_dfbs ? core->tile_counters() : nullptr;
                    uint8_t px = cs.phys_x;
                    uint8_t py = cs.phys_y;

                    std::vector<std::unique_ptr<tt_emule::EmuleDFBInterface[]>> per_thread_dfbs;
                    if (cs.has_dfbs) {
                        per_thread_dfbs = build_per_thread_dfb_interfaces(*cs.ki_list, cs.dfb_allocs);
                    }

                    ObjectIntentTracker intent_tracker;
                    intent_tracker.pre_launch_snapshot(
                        oob_state,
                        *cs.ki_list,
                        l1_data,
                        cs.persistent_cb_starts,
                        static_cast<uint32_t>(cs.logical_core.x),
                        static_cast<uint32_t>(cs.logical_core.y));

                    std::vector<std::thread> threads;
                    std::vector<std::exception_ptr> kernel_exceptions(cs.ki_list->size());
                    uint32_t lx = cs.logical_core.x;
                    uint32_t ly = cs.logical_core.y;
                    uint32_t sem_base = cs.sem_base;
                    uint32_t sem_size = cs.sem_size;
                    for (size_t kidx = 0; kidx < cs.ki_list->size(); ++kidx) {
                        KernelInfo* ki_ptr = &(*cs.ki_list)[kidx];
                        tt_emule::EmuleDFBInterface* dfb_array =
                            cs.has_dfbs ? per_thread_dfbs[kidx].get() : nullptr;
                        threads.emplace_back([ki_ptr,
                                              core,
                                              l1_data,
                                              dram_data,
                                              cb_array,
                                              dfb_array,
                                              tc_array,
                                              core_map_ptr,
                                              px,
                                              py,
                                              lx,
                                              ly,
                                              sem_base,
                                              sem_size,
                                              kidx,
                                              oob_state,
                                              &intent_tracker,
                                              &kep = kernel_exceptions[kidx]]() {
                            (void)kidx;
                            auto& ki = *ki_ptr;
                            __rt_args = (ki.rta_offset_in_kc != kRtaCrtaNoArgsSentinel)
                                ? reinterpret_cast<uint32_t*>(core->l1_ptr(
                                      ki.kernel_config_base + ki.rta_offset_in_kc))
                                : nullptr;
                            __common_rt_args = (ki.crta_offset_in_kc != kRtaCrtaNoArgsSentinel)
                                ? reinterpret_cast<uint32_t*>(core->l1_ptr(
                                      ki.kernel_config_base + ki.crta_offset_in_kc))
                                : nullptr;
                            __emule_bridge_l1 = l1_data;
                            __emule_bridge_dram = dram_data;
                            __emule_cbs = cb_array;
                            __emule_dfbs = dfb_array;
                            __emule_tc_array = tc_array;
                            __processor_id = ki.processor_id;
                            __core = core;
                            __device = nullptr;
                            __emule_core_map = core_map_ptr;
                            my_x[0] = px;
                            my_x[1] = px;
                            my_y[0] = py;
                            my_y[1] = py;
                            __emule_logical_x = lx;
                            __emule_logical_y = ly;
                            __emule_kernel_name = ki.kernel_name.empty() ? nullptr : ki.kernel_name.c_str();
                            __emule_pending_noc_reads = 0;

                            // Overflow drops excess (biases toward false positives, never false negatives).
                            constexpr uint32_t kResolvedCap = 64;
                            uint64_t local_resolved[kResolvedCap] = {};
                            uint32_t local_resolved_count = 0;
                            set_sanitizer_thread_locals(oob_state, sem_base, sem_size);
                            intent_tracker.setup_kernel_tls(
                                oob_state, local_resolved, kResolvedCap, &local_resolved_count);

                            __emule_neo_id = ki.is_tensix ? ki.processor_id : 0;
                            __emule_trisc_id = 0;
                            __emule_num_threads = ki.num_threads;
                            __emule_my_thread_id = ki.thread_idx;

                            log_debug(
                                tt::LogMetal,
                                "  Launching kernel[{}] on logical ({},{}) phys ({},{}) rta_off=0x{:x} crta_off=0x{:x}",
                                kidx, lx, ly, px, py, ki.rta_offset_in_kc, ki.crta_offset_in_kc);

                            try {
                                for (size_t t = 0; t < ki.variants.size(); ++t) {
                                    if (ki.run_all_variants) {
                                        __emule_trisc_id = static_cast<uint8_t>(t);
                                    }
                                    ki.variants[t]();
                                }
                                sweep_per_kernel_dirty_cbs(oob_state, cb_array, ki.processor_id, lx, ly);
                            } catch (...) {
                                kep = std::current_exception();
                            }

                            __core = nullptr;
                            __rt_args = nullptr;
                            __common_rt_args = nullptr;
                            __emule_bridge_l1 = nullptr;
                            __emule_bridge_dram = nullptr;
                            __emule_cbs = nullptr;
                            __emule_dfbs = nullptr;
                            __emule_tc_array = nullptr;
                            __emule_core_map = nullptr;
                            __emule_kernel_name = nullptr;
                            __emule_pending_noc_reads = 0;
                            intent_tracker.teardown_kernel_tls(
                                oob_state, local_resolved, local_resolved_count);
                            clear_sanitizer_thread_locals();
                        });
                    }

                    for (auto& t : threads) {
                        t.join();
                    }

                    const char* oi_kernel_name = (cs.ki_list->size() == 1 && !cs.ki_list->front().kernel_name.empty())
                                                     ? cs.ki_list->front().kernel_name.c_str()
                                                     : nullptr;
                    intent_tracker.verify_post_launch(l1_data, lx, ly, oi_kernel_name);

                    // Rethrow first kernel exception
                    for (size_t i = 0; i < kernel_exceptions.size(); ++i) {
                        if (kernel_exceptions[i]) {
                            try {
                                std::rethrow_exception(kernel_exceptions[i]);
                            } catch (const std::exception& e) {
                                std::throw_with_nested(std::runtime_error(
                                    "EMULE: kernel[" + std::to_string(i) + "] on core (" + std::to_string(lx) + "," +
                                    std::to_string(ly) + ") failed"));
                            } catch (...) {
                                throw std::runtime_error(
                                    "EMULE: kernel[" + std::to_string(i) + "] on core (" + std::to_string(lx) + "," +
                                    std::to_string(ly) + ") threw unknown exception");
                            }
                        }
                    }
                } catch (...) {
                    core_ep = std::current_exception();
                }
            });
    }

    for (auto& t : core_threads) {
        t.join();
    }

    // Rethrow first core exception
    for (size_t i = 0; i < core_exceptions.size(); ++i) {
        if (core_exceptions[i]) {
            std::rethrow_exception(core_exceptions[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// execute_program_emulated: Main entry point.
// ---------------------------------------------------------------------------
void execute_program_emulated(IDevice* device, Program& program) {
    auto& impl = program.impl();
    auto device_id = device->id();
    log_debug(tt::LogMetal, "execute_program_emulated: device {} starting", device_id);

    auto* sw_emu = get_sw_emulated_chip(device_id);

    // Phase 0: Populate bank mapping arrays
    tt_emule::Core* dram_core = nullptr;
    uint32_t num_dram_channels = 0;
    uint32_t num_l1_banks = 0;
    populate_bank_mapping(sw_emu, device, device_id, dram_core, num_dram_channels, num_l1_banks);

    // Build worker coordinate mapping strings
    std::string worker_col_map_str, worker_row_map_str;
    build_worker_coord_maps(device, worker_col_map_str, worker_row_map_str);

    std::string extra_inc = get_extra_include_flags();

    // Compute semaphore base from HAL kernel config layout
    const auto& hal = MetalContext::instance().hal();
    uint32_t tensix_pct_index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    uint32_t kernel_config_base =
        static_cast<uint32_t>(hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG));
    const auto& prog_config = impl.get_program_config(tensix_pct_index);
    uint32_t emule_sem_base = kernel_config_base + prog_config.sem_offset;
    log_debug(
        tt::LogMetal,
        "  EMULE_SEM_BASE: 0x{:x} (kernel_config_base=0x{:x}, sem_offset=0x{:x})",
        emule_sem_base,
        kernel_config_base,
        prog_config.sem_offset);

    // Phase 1: Collect kernels and resolve/compile
    std::map<CoreCoord, std::vector<PendingKernelInfo>> pending_core_kernels;
    std::map<std::string, DeferredCompile> deferred_compiles;
    std::unordered_map<std::string, std::function<void()>> resolved_fns;
    std::vector<std::string> inline_src_temps;

    collect_kernels(
        impl,
        num_dram_channels,
        num_l1_banks,
        worker_col_map_str,
        worker_row_map_str,
        emule_sem_base,
        extra_inc,
        pending_core_kernels,
        deferred_compiles,
        resolved_fns,
        inline_src_temps);

    jit_compile_pending(deferred_compiles, resolved_fns, inline_src_temps);

    // Resolve pending kernels to function pointers
    std::map<CoreCoord, std::vector<KernelInfo>> core_kernels;
    for (auto& [logical_core, pending_list] : pending_core_kernels) {
        for (auto& pk : pending_list) {
            KernelInfo ki{
                {}, pk.run_all_variants, pk.processor_id, pk.thread_idx,
                pk.is_tensix, pk.num_threads,
                pk.kernel_config_base, pk.rta_offset_in_kc, pk.crta_offset_in_kc};
            ki.variants.reserve(pk.variant_cache_keys.size());
            for (const auto& key : pk.variant_cache_keys) {
                ki.variants.push_back(resolved_fns.at(key));
            }
            ki.rt_arg_values = std::move(pk.rt_arg_values);
            ki.kernel_name = std::move(pk.kernel_name);
            core_kernels[logical_core].push_back(std::move(ki));
        }
    }

    log_info(tt::LogMetal, "execute_program_emulated: {} logical cores", core_kernels.size());

    // Phase 2: Build core map and set up per-core state
    auto* core_map_ptr = build_core_map(sw_emu, device, device_id);

    std::vector<CoreSetup> core_setups;
    setup_core_state(impl, device, sw_emu, core_kernels, emule_sem_base, core_setups);

    // Phase 3: Launch all cores concurrently
    uint8_t* dram_data = dram_core ? dram_core->l1_data() : nullptr;

    OobStateOwner oob = build_oob_tensor_state(device, device_id);
    launch_cores(core_setups, dram_data, core_map_ptr, oob.state);

    log_debug(tt::LogMetal, "execute_program_emulated: device {} done", device_id);
}

}  // namespace tt::tt_metal::emule
