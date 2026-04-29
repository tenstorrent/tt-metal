// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "emulated_program_runner.hpp"

#include <dlfcn.h>
#include <unistd.h>
#include <sys/types.h>

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

thread_local std::vector<uint32_t> __rt_args;
thread_local std::vector<uint32_t> __common_rt_args;
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
// __emule_my_thread_id — backs get_my_thread_id(). Same value as __processor_id
//                        today, but kept separate for type (uint32_t vs uint8_t)
//                        and public-API surface.
thread_local uint8_t __processor_id = 0;
thread_local uint8_t __emule_neo_id = 0;
thread_local uint8_t __emule_trisc_id = 0;
thread_local uint32_t __emule_num_threads = 1;
thread_local uint32_t __emule_my_thread_id = 0;

// Core map for cross-core NOC address resolution (shared across all threads).
thread_local std::unordered_map<uint64_t, tt_emule::Core*>* __emule_core_map = nullptr;

// ---------------------------------------------------------------------------
// Bank mapping arrays — populated from SoC descriptor before kernel launch.
// Exported via -rdynamic so JIT .so files can resolve them at dlopen time.
// Match firmware declarations: uint16_t[NUM_NOCS][NUM_DRAM_BANKS], etc.
// ---------------------------------------------------------------------------
static constexpr uint32_t NUM_NOCS = 2;
static constexpr uint32_t MAX_NUM_BANKS = 32;
// Wormhole has 32 CBs; JIT header cb_api.h sizes unpack_tile_size[32].
static constexpr uint32_t EMULE_NUM_CBS = 32;
// Emulation simplification: each worker is its own L1 bank.
static constexpr uint32_t EMULE_NUM_L1_BANKS = 1;
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
    return __emule_bridge_dram ? __emule_bridge_dram + offset : nullptr;
}

extern "C" uint8_t* __emule_local_l1_ptr(uint32_t offset) {
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
extern "C" uint8_t* __emule_resolve_noc_addr(uint64_t noc_addr) {
    uint32_t noc_x = (noc_addr >> NOC_LOCAL_BITS) & NOC_NODE_MASK;
    uint32_t noc_y = (noc_addr >> (NOC_LOCAL_BITS + NOC_NODE_ID_BITS)) & NOC_NODE_MASK;
    uint64_t l1_offset = noc_addr & NOC_LOCAL_MASK;

    if (__emule_core_map) {
        uint64_t key = (uint64_t(noc_x) << 32) | noc_y;
        auto it = __emule_core_map->find(key);
        if (it != __emule_core_map->end()) {
            return it->second->l1_ptr(static_cast<uint32_t>(l1_offset));
        }
    }
    return nullptr;
}

// Resolve multicast: iterate over rectangle of cores and memcpy to each.
// Real firmware encoding: x_start [53:48], y_start [59:54], x_end [41:36], y_end [47:42], addr [35:0]
extern "C" void __emule_multicast_write(uint64_t mcast_addr, const uint8_t* src, uint32_t size) {
    uint32_t x_end = (mcast_addr >> NOC_LOCAL_BITS) & NOC_NODE_MASK;
    uint32_t y_end = (mcast_addr >> (NOC_LOCAL_BITS + NOC_NODE_ID_BITS)) & NOC_NODE_MASK;
    uint32_t x_start = (mcast_addr >> (NOC_LOCAL_BITS + 2 * NOC_NODE_ID_BITS)) & NOC_NODE_MASK;
    uint32_t y_start = (mcast_addr >> (NOC_LOCAL_BITS + 3 * NOC_NODE_ID_BITS)) & NOC_NODE_MASK;
    uint64_t l1_offset = mcast_addr & NOC_LOCAL_MASK;

    // The L1 offset may be a truncated host pointer (from get_write_ptr()) rather
    // than a firmware-style L1 offset.  Worker L1 slots are 2 MB-aligned, so
    // masking with SLOT_MASK extracts the true within-slot offset.  For
    // firmware-style offsets (< 2 MB) this is a no-op.
    static constexpr uint32_t L1_SLOT_SIZE = 2u * 1024 * 1024;  // 2 MB per worker L1 slot
    static constexpr uint32_t L1_SLOT_MASK = L1_SLOT_SIZE - 1;  // 0x1FFFFF
    l1_offset &= L1_SLOT_MASK;

    if (!__emule_core_map) {
        return;
    }

    uint32_t delivered = 0;
    for (uint32_t x = std::min(x_start, x_end); x <= std::max(x_start, x_end); x++) {
        for (uint32_t y = std::min(y_start, y_end); y <= std::max(y_start, y_end); y++) {
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

struct KernelInfo {
    // size 1 for normal kernels; size 4 for Quasar compute (one per TRISC).
    // Either 4 distinct compiled variants (compile-time TRISC_* guards) or 4
    // copies of one function (runtime TRISC_ID). When run_all_variants is true,
    // the launcher iterates and sets __emule_trisc_id per variant.
    std::vector<std::function<void()>> variants;
    bool run_all_variants = false;
    std::vector<uint32_t> rt_args;
    std::vector<uint32_t> common_rt_args;
    uint8_t processor_id = 0;
    bool is_tensix = false;    // true for Tensix/compute kernels (DFB mask uses bits 8-23)
    uint32_t num_threads = 1;  // number of engines (for get_num_threads())
};

struct DeferredCompile {
    std::string src_path;
    std::vector<uint32_t> compile_args;
    std::unordered_map<std::string, uint32_t> named_compile_args;
    std::map<std::string, std::string> defines;
    std::string extra_inc;
};

struct PendingKernelInfo {
    // Parallels KernelInfo::variants but holds cache keys pending compile-resolution.
    std::vector<std::string> variant_cache_keys;
    bool run_all_variants = false;
    std::vector<uint32_t> rt_args;
    std::vector<uint32_t> common_rt_args;
    uint8_t processor_id = 0;
    bool is_tensix = false;
    uint32_t num_threads = 1;
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

    // Invalidate if kernel source file is newer than cached .so
    // (skip for inline sources — their content is hashed into the cache key)
    if (!src_path.empty() && std::filesystem::exists(src_path)) {
        auto so_mtime = std::filesystem::last_write_time(so_path);
        auto src_mtime = std::filesystem::last_write_time(src_path);
        if (src_mtime > so_mtime) {
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
    src = std::regex_replace(src, mhartid_re, "$1 = __processor_id;");

    static const std::regex fence_re(R"(asm\s+volatile\s*\(\s*"fence"\s*:::\s*"memory"\s*\)\s*;)");
    src = std::regex_replace(src, fence_re, "__sync_synchronize();");

    static const std::regex l1_arg_ptr_re(
        R"(reinterpret_cast<([^>]+\*)>\s*\(\s*get_arg_val<uint32_t>\s*(\([^)]*\))\s*\))");
    src = std::regex_replace(
        src, l1_arg_ptr_re, "reinterpret_cast<$1>((uintptr_t)__emule_local_l1_to_ptr(get_arg_val<uint32_t>$2))");

    std::ofstream out(out_path);
    if (!out) {
        throw std::runtime_error("preprocess_kernel_source_for_x86: cannot write " + out_path);
    }
    out << src;
}

static std::function<void()> jit_compile_kernel(
    const std::string& kernel_src_path,
    const std::vector<uint32_t>& compile_args,
    const std::unordered_map<std::string, uint32_t>& named_compile_args,
    const std::map<std::string, std::string>& defines,
    const std::string& extra_include_flags,
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
    std::ostringstream cmd;
    cmd << TT_EMULE_CXX_COMPILER << " -std=c++" << TT_EMULE_CXX_STANDARD << " -fPIC -shared -O2 -Wno-c++11-narrowing"
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
    tt::umd::SWEmuleChip* sw_emu, ChipId device_id, tt_emule::Core*& dram_core_out, uint32_t& num_dram_channels_out) {
    dram_core_out = nullptr;
    num_dram_channels_out = 0;
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

    // noc_xy encoding: (y << 6) | x (matching Blackhole firmware encoding).
    std::memset(dram_bank_to_noc_xy, 0, sizeof(dram_bank_to_noc_xy));
    std::memset(bank_to_dram_offset, 0, sizeof(bank_to_dram_offset));
    for (uint32_t ch = 0; ch < num_dram_channels_out && ch < MAX_NUM_BANKS; ch++) {
        auto dc = metal_soc.get_preferred_worker_core_for_dram_view(ch, 0 /* NOC 0 */);
        uint16_t noc_xy = (static_cast<uint16_t>(dc.y) << NOC_NODE_ID_BITS) | static_cast<uint16_t>(dc.x);
        dram_bank_to_noc_xy[0][ch] = noc_xy;  // NOC 0
        dram_bank_to_noc_xy[1][ch] = noc_xy;  // NOC 1 (same for emulation)
        bank_to_dram_offset[ch] = static_cast<int32_t>(metal_soc.get_address_offset(ch));
        log_debug(
            tt::LogMetal,
            "  DRAM bank[{}]: core({},{}) noc_xy=0x{:04x} offset=0x{:x}",
            ch,
            dc.x,
            dc.y,
            noc_xy,
            bank_to_dram_offset[ch]);
    }

    // L1 bank mapping — for now, all worker cores use themselves as bank 0.
    std::memset(l1_bank_to_noc_xy, 0, sizeof(l1_bank_to_noc_xy));
    std::memset(bank_to_l1_offset, 0, sizeof(bank_to_l1_offset));
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
    const std::string& worker_col_map_str,
    const std::string& worker_row_map_str,
    uint32_t emule_sem_base) {
    std::map<std::string, std::string> defines;
    kernel.process_defines([&](const std::string& k, const std::string& v) { defines[k] = v; });

    auto arch = MetalContext::instance().get_cluster().arch();
    if (arch == ARCH::QUASAR) {
        defines["ARCH_QUASAR"] = "1";
    } else if (arch == ARCH::WORMHOLE_B0) {
        defines["ARCH_WORMHOLE"] = "1";
    } else if (arch == ARCH::BLACKHOLE) {
        defines["ARCH_BLACKHOLE"] = "1";
    }

    defines["NUM_DRAM_BANKS"] = std::to_string(num_dram_channels ? num_dram_channels : 1);
    defines["NUM_L1_BANKS"] = std::to_string(EMULE_NUM_L1_BANKS);
    defines["NUM_NOCS"] = std::to_string(NUM_NOCS);
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
        for (auto& cb_impl : cb_impls) {
            for (uint8_t idx : cb_impl->local_buffer_indices()) {
                if (idx < EMULE_NUM_CBS) {
                    tile_sizes[idx] = cb_impl->page_size(idx);
                }
            }
        }
        std::ostringstream ts;
        for (uint32_t i = 0; i < EMULE_NUM_CBS; i++) {
            if (i) {
                ts << ',';
            }
            ts << tile_sizes[i];
        }
        defines["EMULE_TILE_SIZES"] = ts.str();
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
    const std::string& worker_col_map_str,
    const std::string& worker_row_map_str,
    uint32_t emule_sem_base,
    const std::string& extra_inc,
    std::map<CoreCoord, std::vector<PendingKernelInfo>>& pending_core_kernels,
    std::map<std::string, DeferredCompile>& deferred_compiles,
    std::unordered_map<std::string, std::function<void()>>& resolved_fns,
    std::vector<std::string>& inline_src_temps) {
    static const char* trisc_define_names[] = {"TRISC_UNPACK", "TRISC_MATH", "TRISC_PACK", "TRISC_ISOLATE_SFPU"};

    const uint32_t num_pct = MetalContext::instance().hal().get_programmable_core_type_count();
    for (uint32_t pct = 0; pct < num_pct; ++pct) {
        auto& kernels = impl.get_kernels(pct);
        for (auto& [kernel_id, kernel] : kernels) {
            const auto& ksrc = kernel->kernel_source();
            std::string src_path = resolve_kernel_source_path(ksrc, inline_src_temps);

            auto compile_args = kernel->compile_time_args();
            auto named_compile_args = kernel->named_compile_time_args();
            auto defines = build_kernel_defines(
                *kernel, impl, num_dram_channels, worker_col_map_str, worker_row_map_str, emule_sem_base);

            auto& common_rt = kernel->common_runtime_args();

            // Tensix/compute kernels use bits 8+ in the DFB RISC mask (TENSIX_RISC_OFFSET),
            // while DM kernels use bits 0-7 directly.
            bool is_tensix = (kernel->get_kernel_processor_class() == HalProcessorClassType::COMPUTE);
            auto* qdm = dynamic_cast<experimental::quasar::QuasarDataMovementKernel*>(kernel.get());
            auto* qck = dynamic_cast<experimental::quasar::QuasarComputeKernel*>(kernel.get());
            bool is_quasar_compute = is_tensix && (qck != nullptr);

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
                            DeferredCompile{src_path, compile_args, named_compile_args, defs, extra_inc};
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

            const auto& core_range_set = kernel->core_range_set();
            for (const auto& core_range : core_range_set.ranges()) {
                for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; ++x) {
                    for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; ++y) {
                        CoreCoord logical_core(x, y);
                        auto& rt_args_data = kernel->runtime_args(logical_core);
                        for (uint8_t proc_id : procs.proc_ids) {
                            pending_core_kernels[logical_core].push_back(PendingKernelInfo{
                                variant_cache_keys,
                                run_all_variants,
                                rt_args_data,
                                common_rt,
                                proc_id,
                                is_tensix,
                                procs.num_threads});
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
                        dc.src_path, dc.compile_args, dc.named_compile_args, dc.defines, dc.extra_inc, tmp_path);
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
        {
            auto& msoc = MetalContext::instance().get_cluster().get_soc_desc(device_id);
            for (uint32_t ch = 0; ch < msoc.get_num_dram_views() && ch < MAX_NUM_BANKS; ch++) {
                auto dc = msoc.get_preferred_worker_core_for_dram_view(ch, 0);
                auto* core = sw_emu->get_core(tt_xy_pair(dc.x, dc.y));
                uint64_t key = (uint64_t(dc.x) << 32) | dc.y;
                (*core_map)[key] = core;
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
static void init_core_cb_sync(tt_emule::Core* core, detail::ProgramImpl& impl, const CoreCoord& logical_core) {
    core->reset_cb_sync();
    auto cb_impls = impl.circular_buffers_on_core(logical_core);
    for (auto& cb_impl : cb_impls) {
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
    // Reset L1 bump allocator so DFB allocations don't accumulate across runs.
    core->reset_l1_bump();
    core->reset_dfb_sync();
    if (dfb_impls.empty()) {
        return {};
    }
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

        init_core_cb_sync(core, impl, logical_core);
        init_core_semaphores(core, impl, logical_core, emule_sem_base);

        auto dfb_impls = impl.dataflow_buffers_on_core(logical_core);
        bool has_dfbs = !dfb_impls.empty();
        std::vector<DFBAllocInfo> dfb_allocs = allocate_dfbs_on_core(core, logical_core, dfb_impls);

        core_setups.push_back({logical_core, core, &ki_list, phys_x, phys_y, std::move(dfb_allocs), has_dfbs});
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

// ---------------------------------------------------------------------------
// launch_cores: Spawn concurrent threads per core, each runs its kernels.
// ---------------------------------------------------------------------------
static void launch_cores(
    std::vector<CoreSetup>& core_setups,
    uint8_t* dram_data,
    std::unordered_map<uint64_t, tt_emule::Core*>* core_map_ptr) {
    std::vector<std::thread> core_threads;
    std::vector<std::exception_ptr> core_exceptions(core_setups.size());

    for (size_t core_idx = 0; core_idx < core_setups.size(); ++core_idx) {
        core_threads.emplace_back(
            [&cs = core_setups[core_idx], dram_data, core_map_ptr, &core_ep = core_exceptions[core_idx]]() {
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

                    std::vector<std::thread> threads;
                    std::vector<std::exception_ptr> kernel_exceptions(cs.ki_list->size());
                    uint32_t lx = cs.logical_core.x;
                    uint32_t ly = cs.logical_core.y;
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
                                              kidx,
                                              &kep = kernel_exceptions[kidx]]() {
                            (void)kidx;
                            auto& ki = *ki_ptr;
                            __rt_args = ki.rt_args;
                            __common_rt_args = ki.common_rt_args;
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

                            __emule_neo_id = ki.is_tensix ? ki.processor_id : 0;
                            __emule_trisc_id = 0;
                            __emule_num_threads = ki.num_threads;
                            __emule_my_thread_id = ki.processor_id;

                            log_debug(
                                tt::LogMetal,
                                "  Launching kernel[{}] on logical ({},{}) phys ({},{}) rt_args={} common_rt_args={}",
                                kidx, lx, ly, px, py, ki.rt_args.size(), ki.common_rt_args.size());

                            try {
                                for (size_t t = 0; t < ki.variants.size(); ++t) {
                                    if (ki.run_all_variants) {
                                        __emule_trisc_id = static_cast<uint8_t>(t);
                                    }
                                    ki.variants[t]();
                                }
                            } catch (...) {
                                kep = std::current_exception();
                            }

                            __core = nullptr;
                            __emule_bridge_l1 = nullptr;
                            __emule_bridge_dram = nullptr;
                            __emule_cbs = nullptr;
                            __emule_dfbs = nullptr;
                            __emule_tc_array = nullptr;
                            __emule_core_map = nullptr;
                        });
                    }

                    for (auto& t : threads) {
                        t.join();
                    }

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
    populate_bank_mapping(sw_emu, device_id, dram_core, num_dram_channels);

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
                {}, pk.run_all_variants, pk.rt_args, pk.common_rt_args, pk.processor_id, pk.is_tensix, pk.num_threads};
            ki.variants.reserve(pk.variant_cache_keys.size());
            for (const auto& key : pk.variant_cache_keys) {
                ki.variants.push_back(resolved_fns.at(key));
            }
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
    launch_cores(core_setups, dram_data, core_map_ptr);

    log_debug(tt::LogMetal, "execute_program_emulated: device {} done", device_id);
}

}  // namespace tt::tt_metal::emule
