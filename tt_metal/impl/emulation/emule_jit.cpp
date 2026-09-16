// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "emule_jit.hpp"

#include <dlfcn.h>
#include <unistd.h>
#include <sys/resource.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <semaphore>
#include <sstream>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>

#include "jit_build/jit_build_utils.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_emule/kernel_patcher.hpp"
#include "emule_sanitizers.hpp"
#include "host_sanitizers.hpp"              // emule_asan_enabled
#include "tt_metal/common/stable_hash.hpp"  // tt::StableHasher

// Blaze-only experimental named-args header emit (defined in experimental/blaze/named_kernel_args.cpp).
// Removal tracked by issue #50953.
namespace tt::tt_metal::experimental::blaze {
bool emit_named_args_header(
    const std::string& dir, const NamedCTArgNamespaces& ct_namespaces, const NamedRuntimeArgNamespaces& rt_namespaces);
}  // namespace tt::tt_metal::experimental::blaze

#ifndef TT_EMULE_CXX_COMPILER
#error "TT_EMULE_CXX_COMPILER must be defined by CMake"
#endif
#ifndef TT_EMULE_CXX_STANDARD
#error "TT_EMULE_CXX_STANDARD must be defined by CMake"
#endif
#ifndef TT_EMULE_JIT_INCLUDE_DIR
#error "TT_EMULE_JIT_INCLUDE_DIR must be defined by CMake"
#endif
#ifndef TT_EMULE_INCLUDE_DIR
#error "TT_EMULE_INCLUDE_DIR must be defined by CMake"
#endif

namespace tt::tt_metal::experimental::blaze {
std::string emit_named_args_header(
    const NamedCTArgNamespaces& named_ct_arg_namespaces, const NamedRuntimeArgNamespaces& named_runtime_arg_namespaces);
}  // namespace tt::tt_metal::experimental::blaze

namespace tt::tt_metal::emule {

std::mutex g_jit_cache_mutex;
std::unordered_map<std::string, std::function<void()>> g_jit_cache;

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

uint64_t fnv1a_hash(const std::string& s) {
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
std::function<void()> disk_cache_lookup(const std::string& cache_key, const std::string& src_path) {
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

static std::function<void()> jit_compile_kernel(
    const std::string& kernel_src_path,
    const std::vector<uint32_t>& compile_args,
    const std::unordered_map<std::string, uint32_t>& named_compile_args,
    // Blaze-only experimental named args (issue #50953) — begin
    const NamedCTArgNamespaces& named_ct_arg_namespaces,
    const NamedRuntimeArgNamespaces& named_runtime_arg_namespaces,
    // Blaze-only experimental named args (issue #50953) — end
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

    // 2. Create temp directory. Honor $TMPDIR so the JIT scratch can be placed on a
    // reaper-safe filesystem (a /tmp cleanup reaper wiping these dirs mid-compile
    // causes "named_args_generated.h not found" / "ld: cannot open output" failures).
    std::string tmpl = (std::getenv("TMPDIR") ? std::string(std::getenv("TMPDIR")) : std::string("/tmp"));
    if (!tmpl.empty() && tmpl.back() == '/') {
        tmpl.pop_back();
    }
    tmpl += "/tt_emule_jit_XXXXXX";
    std::vector<char> tmpdir(tmpl.begin(), tmpl.end());
    tmpdir.push_back('\0');
    if (!mkdtemp(tmpdir.data())) {
        throw std::runtime_error("jit_compile_kernel: mkdtemp failed");
    }
    std::string dir(tmpdir.data());

    // 2b. Preprocess the kernel source for x86: rewrite RISC-V inline asm
    // (mhartid, fence) and raw L1 arg-val pointer casts. -I kernel_dir (below)
    // keeps relative includes in the patched file resolvable.
    std::string patched_kernel_path = dir + "/patched_kernel.cpp";
    const std::vector<std::string> emule_inc_roots = {jit_inc, parent_inc};
    tt::emule::patch_kernel_source(abs_kernel, patched_kernel_path, extra_include_flags, emule_inc_roots);

    ////////////////////////////////////////////////////////////
    // Blaze-only experimental named args
    // Removal is tracked by issue #50953
    // 2c. Blaze EXPERIMENTAL named kernel args.
    // Emule's JIT path bypasses genfiles; call the experimental helper to
    // emit named_args_generated.h. Included from wrapper.cpp when non-empty.
    bool has_named_args =
        experimental::blaze::emit_named_args_header(dir, named_ct_arg_namespaces, named_runtime_arg_namespaces);
    ////////////////////////////////////////////////////////////

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
        // KERNEL_COMPILE_TIME_ARG_MAP (read by api/compile_time_args.h) goes in the wrapper for the
        // same reason the kernel defines above do -- emule shells out via std::system(), so it has
        // the whole command as one argv string against MAX_ARG_STRLEN (128 KB), and this map alone
        // can exceed that. Must precede every include so the consuming header sees it.
        if (!named_compile_args.empty()) {
            f << "#define KERNEL_COMPILE_TIME_ARG_MAP "
              << tt::jit_build::utils::format_named_ct_arg_map(named_compile_args) << "\n";
        }
        f << "#include \"jit_kernel_stubs.hpp\"\n";
        // Metal-2.0 `namespace args` (base).
        emit_metal2_namespaces(f, bindings, named_compile_args);
        ////////////////////////////////////////////////////////////
        // Blaze-only experimental named args
        // Removal is tracked by issue #50953
        // Blaze experimental `blaze_ct_args::` header (additive layer on top of Metal-2.0 args).
        if (has_named_args) {
            f << "#include \"" << dir << "/named_args_generated.h\"\n";
        }
        ////////////////////////////////////////////////////////////
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

    // 5b. KERNEL_COMPILE_TIME_ARG_MAP is emitted as a #define at the top of wrapper.cpp (step 3),
    // not as a -D flag here: it is far too large for one shell command. See that site.

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
    std::filesystem::remove_all(dir);

    // 11. Wrap in shared_ptr for lifetime management (dlclose on destruction).
    auto shared_handle = std::shared_ptr<void>(handle, [](void* h) { dlclose(h); });
    return [fn, shared_handle]() { fn(); };
}

// ---------------------------------------------------------------------------
// get_extra_include_flags: Build -I flags for JIT compilation.
// ---------------------------------------------------------------------------
std::string get_extra_include_flags() {
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
std::string resolve_kernel_source_path(const KernelSource& ksrc, std::vector<std::string>& inline_src_temps) {
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

// A same-relative-path source in jit_hw is emule's implementation of a Metal file kernel.
std::string resolve_emule_kernel_source_shadow(const std::string& src_path, ContextId context_id) {
    std::error_code ec;
    const auto source = std::filesystem::weakly_canonical(src_path, ec);
    if (ec) {
        return src_path;
    }
    const auto root =
        std::filesystem::weakly_canonical(MetalContext::instance(context_id).rtoptions().get_root_dir(), ec);
    if (ec) {
        return src_path;
    }
    const auto relative = source.lexically_relative(root);
    if (relative.empty() || relative.is_absolute() || *relative.begin() == "..") {
        return src_path;
    }

    const auto shadow = std::filesystem::path(TT_EMULE_JIT_INCLUDE_DIR) / relative;
    if (!std::filesystem::is_regular_file(shadow, ec) || ec) {
        return src_path;
    }
    log_debug(tt::LogMetal, "Using emule kernel source shadow {} for {}", shadow.string(), source.string());
    return shadow.string();
}

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
    struct rlimit rl{};
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

void jit_compile_pending(
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
                              // Blaze-only experimental named args (issue #50953) are threaded through here.
                              auto fn = jit_compile_kernel(
                                  dc_copy.src_path,
                                  dc_copy.compile_args,
                                  dc_copy.named_compile_args,
                                  // Blaze named args (issue #50953) — begin
                                  dc_copy.named_ct_arg_namespaces,
                                  dc_copy.named_runtime_arg_namespaces,
                                  // Blaze named args (issue #50953) — end
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

}  // namespace tt::tt_metal::emule
