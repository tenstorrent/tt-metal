// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "emulated_program_runner.hpp"
#include "emulated_run_stats.hpp"

#include <dlfcn.h>
#include <unistd.h>
#include <sys/types.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <future>
#include <thread>
#include <unordered_map>
#include <vector>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/buffers/circular_buffer.hpp"
#include "impl/buffers/semaphore.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/hal_types.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "umd/device/chip/sw_emule_chip.hpp"
#include "tt_emule/device.hpp"

#include <tt-logger/tt-logger.hpp>

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

// Core map for cross-core NOC address resolution (shared across all threads).
thread_local std::unordered_map<uint64_t, tt_emule::Core*>* __emule_core_map = nullptr;

// ---------------------------------------------------------------------------
// Bank mapping arrays — populated from SoC descriptor before kernel launch.
// Exported via -rdynamic so JIT .so files can resolve them at dlopen time.
// Match firmware declarations: uint16_t[NUM_NOCS][NUM_DRAM_BANKS], etc.
// ---------------------------------------------------------------------------
uint16_t dram_bank_to_noc_xy[2][32] = {};
int32_t bank_to_dram_offset[32] = {};
uint16_t l1_bank_to_noc_xy[2][32] = {};
int32_t bank_to_l1_offset[32] = {};

// Per-core NOC coordinates — set per kernel thread (thread_local).
// On real HW these are read from NOC registers; we set them from physical coords.
thread_local uint8_t my_x[2] = {};
thread_local uint8_t my_y[2] = {};
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
    return __emule_bridge_l1 ? __emule_bridge_l1 + static_cast<uint32_t>(addr) : nullptr;
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
    static constexpr uint32_t L1_SLOT_MASK = (2u * 1024 * 1024) - 1;  // 0x1FFFFF
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
// Last-run stats (populated by execute_program_emulated)
// ---------------------------------------------------------------------------
static EmulatedRunStats g_last_run_stats;

const EmulatedRunStats& get_last_emulated_run_stats() { return g_last_run_stats; }

// ---------------------------------------------------------------------------
// JIT Compilation Cache (in-memory + persistent disk cache)
// ---------------------------------------------------------------------------

static std::mutex g_jit_cache_mutex;
static std::unordered_map<std::string, std::function<void()>> g_jit_cache;

// ---------------------------------------------------------------------------
// Disk JIT cache — survives process restarts (critical for --forked mode)
// ---------------------------------------------------------------------------

static uint64_t fnv1a_hash(const std::string& s) {
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : s) {
        h ^= static_cast<uint64_t>(c);
        h *= 1099511628211ULL;
    }
    return h;
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
    dlopen("libtt_metal.so", RTLD_NOW | RTLD_NOLOAD | RTLD_GLOBAL);
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
    char hex[17];
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
    char hex[17];
    std::snprintf(hex, sizeof(hex), "%016lx", fnv1a_hash(cache_key));
    return cache_dir + "/" + hex + ".so";
}

// ---------------------------------------------------------------------------
// JIT Kernel Compilation
// ---------------------------------------------------------------------------

std::function<void()> jit_compile_kernel(
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
        f << "#include \"" << abs_kernel << "\"\n";
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
    cmd << "g++ -std=c++17 -fPIC -shared -O3 -march=native"
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

    int rc = std::system(full_cmd.c_str());
    if (rc != 0) {
        throw std::runtime_error(
            "jit_compile_kernel: g++ failed (exit " + std::to_string(rc) + ") for kernel: " + kernel_src_path);
    }

    // 8. dlopen
    // Promote libtt_metal.so to RTLD_GLOBAL so kernel.so can resolve TLS symbols
    // (e.g. __emule_cbs) that are defined in libtt_metal.so. When loaded via
    // Python module import, shared libraries default to RTLD_LOCAL.
    dlopen("libtt_metal.so", RTLD_NOW | RTLD_NOLOAD | RTLD_GLOBAL);
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

void execute_program_emulated(IDevice* device, Program& program) {
    auto& impl = program.impl();
    auto device_id = device->id();
    log_debug(tt::LogMetal, "execute_program_emulated: device {} starting", device_id);

    // Get SWEmuleChip — memory is owned by tt_emule::Core objects inside it
    auto* sw_emu = get_sw_emulated_chip(device_id);

    // Get DRAM core for bridge pointer (first DRAM channel, for legacy __emule_dram_ptr)
    tt_emule::Core* dram_core = nullptr;
    uint32_t num_dram_channels = 0;
    if (sw_emu) {
        auto& soc = sw_emu->get_soc_descriptor();
        auto dram_channels = soc.get_dram_cores();
        num_dram_channels = static_cast<uint32_t>(dram_channels.size());

        if (!dram_channels.empty() && !dram_channels[0].empty()) {
            auto& dc = dram_channels[0][0];
            dram_core = sw_emu->get_core(tt_xy_pair(dc.x, dc.y));
        }

        // Populate bank mapping arrays using metal_SocDescriptor (matches host write path).
        // The host write path uses get_preferred_worker_core_for_dram_view() for core coords
        // and get_address_offset() for per-bank offsets, so we must match exactly.
        auto& metal_soc = MetalContext::instance().get_cluster().get_soc_desc(device_id);
        num_dram_channels = static_cast<uint32_t>(metal_soc.get_num_dram_views());

        // noc_xy encoding: (y << 6) | x (matching Blackhole firmware encoding).
        std::memset(dram_bank_to_noc_xy, 0, sizeof(dram_bank_to_noc_xy));
        std::memset(bank_to_dram_offset, 0, sizeof(bank_to_dram_offset));
        for (uint32_t ch = 0; ch < num_dram_channels && ch < 32; ch++) {
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
        // Real L1 banking is not yet emulated.
        std::memset(l1_bank_to_noc_xy, 0, sizeof(l1_bank_to_noc_xy));
        std::memset(bank_to_l1_offset, 0, sizeof(bank_to_l1_offset));
    }

    // Build worker logical→virtual coordinate mapping for JIT kernels.
    // D2M-generated kernels use convert_logical_x_to_translated / convert_logical_y_to_translated
    // which index into lookup tables. We populate these from the device's coordinate mapping.
    std::string worker_col_map_str, worker_row_map_str;
    {
        auto grid = device->compute_with_storage_grid_size();
        // Build column mapping: logical x → virtual x
        std::ostringstream col_ss;
        for (uint32_t lx = 0; lx < 64; lx++) {
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
        // Build row mapping: logical y → virtual y
        std::ostringstream row_ss;
        for (uint32_t ly = 0; ly < 64; ly++) {
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

    // Build extra include flags (project source dir for ttnn kernel includes)
#ifdef TT_EMULE_PROJECT_SOURCE_DIR
    const std::string project_src = TT_EMULE_PROJECT_SOURCE_DIR;
    std::string extra_inc;
    extra_inc += "-I\"" + project_src + "/ttnn/cpp\"";
    extra_inc += " -I\"" + project_src + "\"";
    // Real hw/inc headers for TensorAccessorArgs (pure C++17, no KERNEL_BUILD needed)
    extra_inc += " -I\"" + project_src + "/tt_metal/hw/inc\"";
    extra_inc += " -I\"" + project_src + "/tt_metal/hostdevcommon/api\"";
#else
    std::string extra_inc;
#endif

    // -----------------------------------------------------------------------
    // Phase 1: Collect all kernels grouped by logical core.
    // -----------------------------------------------------------------------

    struct KernelInfo {
        std::function<void()> fn;
        std::vector<uint32_t> rt_args;
        std::vector<uint32_t> common_rt_args;
    };

    // Map logical core → list of kernels to run on it.
    std::map<CoreCoord, std::vector<KernelInfo>> core_kernels;

    // Deferred compilation info for cache misses.
    struct DeferredCompile {
        std::string src_path;
        std::vector<uint32_t> compile_args;
        std::unordered_map<std::string, uint32_t> named_compile_args;
        std::map<std::string, std::string> defines;
        std::string extra_inc;
    };

    // Per-kernel info before function pointer resolution.
    struct PendingKernelInfo {
        std::string cache_key;
        std::vector<uint32_t> rt_args;
        std::vector<uint32_t> common_rt_args;
    };

    // Map logical core → pending kernels (resolved after parallel compile).
    std::map<CoreCoord, std::vector<PendingKernelInfo>> pending_core_kernels;

    // Unique cache misses to compile in parallel.
    std::map<std::string, DeferredCompile> deferred_compiles;

    // Cache hits resolved immediately.
    std::unordered_map<std::string, std::function<void()>> resolved_fns;

    // Inline source temp files to clean up after compilation.
    std::vector<std::string> inline_src_temps;

    // ---- Compute semaphore base from HAL kernel config layout ----
    // Matches real firmware: sem_addr = kernel_config_base + sem_offset + sem_id * L1_ALIGNMENT
    // finalize_offsets() has already been called, so ProgramConfig.sem_offset is populated.
    static constexpr uint32_t EMULE_SEM_ALIGN = 16;
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

    // ---- Phase 1a: Collect (sequential) ----
    for (uint32_t pct = 0; pct < NumHalProgrammableCoreTypes; ++pct) {
        auto& kernels = impl.get_kernels(pct);
        for (auto& [kernel_id, kernel] : kernels) {
            const auto& ksrc = kernel->kernel_source();

            // Resolve source path
            std::string src_path;
            if (ksrc.source_type_ == KernelSource::FILE_PATH) {
                src_path = ksrc.path_.string();
            } else {
                char tmpf[] = "/tmp/tt_emule_src_XXXXXX.cpp";
                int fd = mkstemps(tmpf, 4);
                if (fd < 0) {
                    throw std::runtime_error("execute_program_emulated: mkstemps failed");
                }
                std::string content = ksrc.source_;
                ssize_t written = ::write(fd, content.c_str(), content.size());
                ::close(fd);
                if (written < 0) {
                    throw std::runtime_error("execute_program_emulated: write failed");
                }
                src_path = tmpf;
                inline_src_temps.push_back(src_path);
            }

            auto compile_args = kernel->compile_time_args();
            auto named_compile_args = kernel->named_compile_time_args();
            auto defines = kernel->defines();

            // DRAM bank mapping defines for JIT kernels.
            defines["NUM_DRAM_BANKS"] = std::to_string(num_dram_channels ? num_dram_channels : 1);
            defines["NUM_L1_BANKS"] = "1";
            defines["NUM_NOCS"] = "2";
            defines["DRAM_ALIGNMENT"] = "32";
            defines["L1_ALIGNMENT"] = "16";

            // Worker coordinate mapping for D2M kernels (logical → virtual).
            defines["EMULE_WORKER_COL_MAP"] = worker_col_map_str;
            defines["EMULE_WORKER_ROW_MAP"] = worker_row_map_str;

            // Dynamic semaphore base — computed above to avoid CB/semaphore overlap.
            {
                std::ostringstream sb;
                sb << "0x" << std::hex << emule_sem_base;
                defines["EMULE_SEM_BASE"] = sb.str();
            }
            defines["EMULE_SEM_ALIGN"] = std::to_string(EMULE_SEM_ALIGN);

            // Collect CB tile sizes from program for constexpr get_tile_size().
            {
                const auto& core_range_set = kernel->core_range_set();
                if (!core_range_set.ranges().empty()) {
                    auto first_core = core_range_set.ranges().begin()->start_coord;
                    auto cb_impls = impl.circular_buffers_on_core(first_core);
                    uint32_t tile_sizes[32] = {};
                    for (auto& cb_impl : cb_impls) {
                        for (uint8_t idx : cb_impl->local_buffer_indices()) {
                            if (idx < 32) {
                                tile_sizes[idx] = cb_impl->page_size(idx);
                            }
                        }
                    }
                    std::ostringstream ts;
                    for (int i = 0; i < 32; i++) {
                        if (i) {
                            ts << ',';
                        }
                        ts << tile_sizes[i];
                    }
                    defines["EMULE_TILE_SIZES"] = ts.str();
                }
            }

            // JIT cache key — use source content hash for inline kernels (temp paths change each run)
            std::string cache_key;
            if (ksrc.source_type_ == KernelSource::FILE_PATH) {
                cache_key = src_path;
            } else {
                // Hash inline source content for stable cache key across runs
                char hex[17];
                std::snprintf(hex, sizeof(hex), "%016lx", fnv1a_hash(ksrc.source_));
                cache_key = std::string("inline:") + hex;
            }
            for (auto v : compile_args) {
                cache_key += ":" + std::to_string(v);
            }
            for (const auto& [k, v] : named_compile_args) {
                cache_key += ":N" + k + "=" + std::to_string(v);
            }
            for (const auto& [k, v] : defines) {
                cache_key += ":" + k + "=" + v;
            }

            // Check cache: in-memory → disk → defer for compilation
            {
                std::lock_guard<std::mutex> lock(g_jit_cache_mutex);
                auto it = g_jit_cache.find(cache_key);
                if (it != g_jit_cache.end()) {
                    resolved_fns[cache_key] = it->second;
                } else if (
                    resolved_fns.find(cache_key) == resolved_fns.end() &&
                    deferred_compiles.find(cache_key) == deferred_compiles.end()) {
                    // Try disk cache (survives --forked subprocess restarts)
                    std::string mtime_path = (ksrc.source_type_ == KernelSource::FILE_PATH) ? src_path : "";
                    auto disk_fn = disk_cache_lookup(cache_key, mtime_path);
                    if (disk_fn) {
                        resolved_fns[cache_key] = disk_fn;
                        g_jit_cache[cache_key] = disk_fn;
                    } else {
                        deferred_compiles[cache_key] =
                            DeferredCompile{src_path, compile_args, named_compile_args, defines, extra_inc};
                    }
                }
            }

            // Common runtime args for this kernel
            auto& common_rt = kernel->common_runtime_args();

            // Add to pending per-core map
            const auto& core_range_set = kernel->core_range_set();
            for (const auto& core_range : core_range_set.ranges()) {
                for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; ++x) {
                    for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; ++y) {
                        CoreCoord logical_core(x, y);
                        auto& rt_args_data = kernel->runtime_args(logical_core);
                        pending_core_kernels[logical_core].push_back(
                            PendingKernelInfo{cache_key, rt_args_data, common_rt});
                    }
                }
            }
        }
    }

    // ---- Phase 1b: Compile cache misses in parallel (output to disk cache) ----
    if (!deferred_compiles.empty()) {
        log_info(tt::LogMetal, "JIT parallel compile: {} unique kernels to compile", deferred_compiles.size());

        std::vector<std::pair<std::string, std::future<std::function<void()>>>> futures;
        futures.reserve(deferred_compiles.size());

        for (auto& [key, dc] : deferred_compiles) {
            std::string cache_path = disk_cache_so_path(key);
            futures.emplace_back(
                key, std::async(std::launch::async, [&dc, cache_path]() {
                    return jit_compile_kernel(
                        dc.src_path, dc.compile_args, dc.named_compile_args, dc.defines, dc.extra_inc, cache_path);
                }));
        }

        // Wait for all compilations and store results
        for (auto& [key, fut] : futures) {
            auto fn = fut.get();
            resolved_fns[key] = fn;
            std::lock_guard<std::mutex> lock(g_jit_cache_mutex);
            g_jit_cache[key] = fn;
        }
    }

    // Clean up inline source temp files (no longer needed after compilation)
    if (!std::getenv("TT_EMULE_KEEP_JIT_SRC")) {
        for (auto& tmp : inline_src_temps) {
            std::filesystem::remove(tmp);
        }
    } else {
        for (auto& tmp : inline_src_temps) {
            fprintf(stderr, "[EMULE-DBG] kept JIT source: %s\n", tmp.c_str());
        }
    }

    // ---- Phase 1c: Resolve pending kernels to function pointers ----
    for (auto& [logical_core, pending_list] : pending_core_kernels) {
        for (auto& pk : pending_list) {
            core_kernels[logical_core].push_back(
                KernelInfo{resolved_fns.at(pk.cache_key), pk.rt_args, pk.common_rt_args});
        }
    }

    // Record execution metadata for test introspection
    {
        g_last_run_stats.num_cores = static_cast<uint32_t>(core_kernels.size());
        g_last_run_stats.kernel_paths.clear();
        std::set<std::string> seen_paths;
        for (uint32_t pct = 0; pct < NumHalProgrammableCoreTypes; ++pct) {
            auto& kernels = impl.get_kernels(pct);
            for (auto& [kernel_id, kernel] : kernels) {
                const auto& ksrc = kernel->kernel_source();
                std::string basename;
                if (ksrc.source_type_ == KernelSource::FILE_PATH) {
                    basename = std::filesystem::path(ksrc.path_).filename().string();
                } else {
                    basename = "<inline>";
                }
                if (seen_paths.insert(basename).second) {
                    g_last_run_stats.kernel_paths.push_back(basename);
                }
            }
        }
        log_info(
            tt::LogMetal,
            "execute_program_emulated: {} logical cores, {} unique kernels",
            core_kernels.size(),
            g_last_run_stats.kernel_paths.size());
        for (auto& kp : g_last_run_stats.kernel_paths) {
            log_info(tt::LogMetal, "  JIT kernel: {}", kp);
        }
    }

    // -----------------------------------------------------------------------
    // Phase 2: Set up cores and launch ALL cores concurrently.
    // Core's L1 is already mmap'd below 4 GB by tt_emule::Core.
    // Concurrent launch is required for cross-core synchronization (multicast).
    // -----------------------------------------------------------------------

    static constexpr uint32_t MAX_CBS = 32;

    // Build core map: physical {x,y} → tt_emule::Core* for cross-core NOC access.
    // Shared by all kernel threads for resolving NOC addresses.
    // Must include ALL worker cores on the device, not just the current program's
    // cores, because kernels may read from cores used by previous programs (e.g.,
    // multicast sender reading tilized data stored by a prior tilize program).
    // Cached per device_id since the chip topology doesn't change between calls.
    // Not mutex-protected — safe because emulation runs in slow dispatch mode (single-threaded host).
    static std::unordered_map<uint32_t, std::shared_ptr<std::unordered_map<uint64_t, tt_emule::Core*>>>
        g_core_map_cache;

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
        // Add DRAM cores using both coordinate systems:
        // 1. UMD SoC descriptor coords (for legacy / get_dram_cores() lookups)
        auto& umd_soc = sw_emu->get_soc_descriptor();
        for (auto& dc_vec : umd_soc.get_dram_cores()) {
            for (auto& dc : dc_vec) {
                auto* core = sw_emu->get_core(tt_xy_pair(dc.x, dc.y));
                uint64_t key = (uint64_t(dc.x) << 32) | dc.y;
                (*core_map)[key] = core;
            }
        }
        // 2. metal_SocDescriptor preferred worker coords (used by host write path)
        {
            auto& msoc = MetalContext::instance().get_cluster().get_soc_desc(device_id);
            for (uint32_t ch = 0; ch < msoc.get_num_dram_views() && ch < 32; ch++) {
                auto dc = msoc.get_preferred_worker_core_for_dram_view(ch, 0);
                auto* core = sw_emu->get_core(tt_xy_pair(dc.x, dc.y));
                uint64_t key = (uint64_t(dc.x) << 32) | dc.y;
                (*core_map)[key] = core;
            }
        }
    } else if (!core_map) {
        core_map = std::make_shared<std::unordered_map<uint64_t, tt_emule::Core*>>();
    }

    // Pre-structure: collect per-core setup info before launching threads
    struct CoreSetup {
        CoreCoord logical_core;
        tt_emule::Core* core;
        std::vector<KernelInfo>* ki_list;
        uint8_t phys_x;
        uint8_t phys_y;
    };
    std::vector<CoreSetup> core_setups;

    for (auto& [logical_core, ki_list] : core_kernels) {
        tt_emule::Core* core = nullptr;
        uint8_t phys_x = 0, phys_y = 0;
        if (sw_emu) {
            auto phys = device->virtual_core_from_logical_core(logical_core, tt::CoreType::WORKER);
            core = sw_emu->get_core(tt_xy_pair(phys.x, phys.y));
            phys_x = static_cast<uint8_t>(phys.x);
            phys_y = static_cast<uint8_t>(phys.y);
        }
        if (!core) {
            continue;
        }

        // Reset CB sync state from previous run
        core->reset_cb_sync();

        // Read CB config from program and populate Core's CB sync array
        auto cb_impls = impl.circular_buffers_on_core(logical_core);
        for (auto& cb_impl : cb_impls) {
            for (uint8_t idx : cb_impl->local_buffer_indices()) {
                if (idx >= MAX_CBS) {
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

        // Initialize semaphores in Core L1 at dynamic offset
        // Semaphore region: emule_sem_base + id * EMULE_SEM_ALIGN
        auto& semaphores = impl.semaphores();
        for (auto& sem : semaphores) {
            if (!sem.initialized_on_logical_core(logical_core)) {
                continue;
            }
            uint32_t sem_id = sem.id();
            uint32_t initial_value = sem.initial_value();
            uint32_t sem_addr = emule_sem_base + sem_id * EMULE_SEM_ALIGN;
            if (sem_addr + 4 <= core->l1_size()) {
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

        core_setups.push_back({logical_core, core, &ki_list, phys_x, phys_y});
    }

    // Launch all cores concurrently — each core runner spawns kernel threads
    uint8_t* dram_data = dram_core ? dram_core->l1_data() : nullptr;
    auto core_map_ptr = core_map.get();

    std::vector<std::thread> core_threads;

    for (auto& cs : core_setups) {
        core_threads.emplace_back([&cs, dram_data, core_map_ptr]() {
            try {
                auto* core = cs.core;
                uint8_t* l1_data = core->l1_data();
                tt_emule::CBSyncState* cb_array = core->cb_sync_array();
                uint8_t px = cs.phys_x;
                uint8_t py = cs.phys_y;

                // Launch one thread per kernel on this core
                std::vector<std::thread> threads;
                uint32_t lx = cs.logical_core.x;
                uint32_t ly = cs.logical_core.y;
                uint32_t kernel_idx = 0;
                for (auto& ki : *cs.ki_list) {
                    uint32_t kidx = kernel_idx++;
                    threads.emplace_back(
                        [&ki, core, l1_data, dram_data, cb_array, core_map_ptr, px, py, lx, ly, kidx]() {
                            __rt_args = ki.rt_args;
                            __common_rt_args = ki.common_rt_args;
                            __emule_bridge_l1 = l1_data;
                            __emule_bridge_dram = dram_data;
                            __emule_cbs = cb_array;
                            __core = core;
                            __device = nullptr;
                            __emule_core_map = core_map_ptr;
                            my_x[0] = px;
                            my_x[1] = px;
                            my_y[0] = py;
                            my_y[1] = py;
                            __emule_logical_x = lx;
                            __emule_logical_y = ly;

                            log_debug(
                                tt::LogMetal,
                                "  Launching kernel[{}] on logical ({},{}) phys ({},{}) rt_args={} common_rt_args={}",
                                kidx,
                                lx,
                                ly,
                                px,
                                py,
                                ki.rt_args.size(),
                                ki.common_rt_args.size());

                            try {
                                ki.fn();
                            } catch (const std::exception& e) {
                                fprintf(
                                    stderr, "EMULE ERROR: kernel[%u] on (%u,%u) threw: %s\n", kidx, lx, ly, e.what());
                            } catch (...) {
                                fprintf(
                                    stderr,
                                    "EMULE ERROR: kernel[%u] on (%u,%u) threw unknown exception\n",
                                    kidx,
                                    lx,
                                    ly);
                            }

                            __core = nullptr;
                            __emule_bridge_l1 = nullptr;
                            __emule_bridge_dram = nullptr;
                            __emule_cbs = nullptr;
                            __emule_core_map = nullptr;
                        });
                }

                for (auto& t : threads) {
                    t.join();
                }
            } catch (const std::exception& e) {
                fprintf(
                    stderr,
                    "EMULE ERROR: core thread (%zu,%zu) exception: %s\n",
                    cs.logical_core.x,
                    cs.logical_core.y,
                    e.what());
            } catch (...) {
                fprintf(
                    stderr,
                    "EMULE ERROR: core thread (%zu,%zu) unknown exception\n",
                    cs.logical_core.x,
                    cs.logical_core.y);
            }
        });
    }

    for (auto& t : core_threads) {
        t.join();
    }

    log_debug(tt::LogMetal, "execute_program_emulated: device {} done", device_id);
}

}  // namespace tt::tt_metal::emule
