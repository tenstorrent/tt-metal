// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal-side definition of __emule_asan_panic + its diagnostic-trace facility:
// the single copy emitted into libtt_metal (host-API checks call it directly,
// JIT kernel .so files resolve it at dlopen via -rdynamic). Faithful mirror of
// tt-emule's src/kernel_runner.cpp copy — keep the two in sync. See
// SANITIZER_CHECKS.md "Diagnostic trace".

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include <dlfcn.h>
#include <execinfo.h>
#include <unistd.h>
#if defined(__linux__)
#include <fcntl.h>      // open (silence gcore output)
#include <sys/prctl.h>  // PR_SET_DUMPABLE / PR_SET_PTRACER — see emule_asan_handle_coredump
#include <sys/wait.h>   // waitpid (reap the gcore child)
#endif

// Physical-coord identity read by the trace. my_x/my_y stay worker-thread-locals
// (the fiber scheduler restores them per-fiber on swap); the kernel name, logical
// coords, processor/neo/trisc ids are read from the per-fiber context
// (__emule_self->san / __emule_self) so they aren't clobbered by a co-scheduled
// fiber across a yield — nullptr/absent when no kernel fiber is on the stack.
extern thread_local uint8_t my_x[2];
extern thread_local uint8_t my_y[2];
#include "jit_hw/internal/emule_thread_ctx.h"  // __emule_self / EmuleSanitizerState

namespace {

// Resolve one (module, file-relative offset) to a "func at file:line" string,
// flattening inlined frames onto one line with " <- ". Returns false if no real
// debug info was found.
bool emule_asan_symbolize(const char* module, uintptr_t file_offset, char* out, size_t out_sz) {
    if (module == nullptr || module[0] == '\0' || out_sz == 0) {
        return false;
    }
    // dladdr may report the JIT's transient "<hash>.so.tmp.<pid>" (renamed to
    // "<hash>.so" right after load); if that path is gone, strip the suffix.
    char modbuf[1024];
    std::snprintf(modbuf, sizeof(modbuf), "%s", module);
    if (access(modbuf, R_OK) != 0) {
        char* t = std::strstr(modbuf, ".so.tmp.");
        if (t != nullptr) {
            t[3] = '\0';  // keep "....so", drop ".tmp.<pid>"
        }
        if (access(modbuf, R_OK) != 0) {
            return false;
        }
    }
    module = modbuf;
    // Prefer llvm-symbolizer (reads clang's DWARF5; addr2line only yields "??:?"),
    // trying versioned names; addr2line is the last-ditch fallback.
    // `DEBUGINFOD_URLS=` is load-bearing: llvm-symbolizer otherwise does a blocking debuginfod
    // NETWORK lookup (the inherited DEBUGINFOD_URLS, e.g. https://debuginfod.ubuntu.com) when the
    // local DWARF is incomplete; on a host with no route it blocks forever in poll(), wedging
    // __emule_asan_panic before its abort() so the death never fires. Only bites where
    // DEBUGINFOD_URLS is set AND unreachable (dev boxes) — not CI. Disabling it keeps the trace
    // fully local. `</dev/null` additionally guards llvm-symbolizer's interactive-stdin path.
    // Keep in sync with the tt-emule mirror in include/jit_hw/asan/emule_asan.h.
    const char* tools[] = {
        "DEBUGINFOD_URLS= llvm-symbolizer --obj=\"%s\" --pretty-print --inlines --demangle 0x%lx </dev/null 2>/dev/null",
        "DEBUGINFOD_URLS= llvm-symbolizer-20 --obj=\"%s\" --pretty-print --inlines --demangle 0x%lx </dev/null 2>/dev/null",
        "DEBUGINFOD_URLS= llvm-symbolizer-19 --obj=\"%s\" --pretty-print --inlines --demangle 0x%lx </dev/null 2>/dev/null",
        "DEBUGINFOD_URLS= llvm-symbolizer-18 --obj=\"%s\" --pretty-print --inlines --demangle 0x%lx </dev/null 2>/dev/null",
        "DEBUGINFOD_URLS= addr2line -f -C -i -p -e \"%s\" 0x%lx </dev/null 2>/dev/null",
    };
    for (size_t t = 0; t < sizeof(tools) / sizeof(tools[0]); ++t) {
        char cmd[1100];
        std::snprintf(cmd, sizeof(cmd), tools[t], module, static_cast<unsigned long>(file_offset));
        FILE* pipe = popen(cmd, "r");
        if (pipe == nullptr) {
            continue;
        }
        char buf[2048];
        size_t used = 0;
        char line[512];
        while (used + 1 < sizeof(buf) && std::fgets(line, sizeof(line), pipe) != nullptr) {
            for (size_t i = 0; line[i] != '\0' && used + 1 < sizeof(buf); ++i) {
                char c = line[i];
                if (c == '\n' || c == '\r') {
                    if (used >= 4 && std::strcmp(buf + used - 4, " <- ") == 0) {
                        continue;
                    }
                    const char* sep = " <- ";
                    for (int k = 0; k < 4 && used + 1 < sizeof(buf); ++k) {
                        buf[used++] = sep[k];
                    }
                } else {
                    buf[used++] = c;
                }
            }
        }
        buf[used] = '\0';
        pclose(pipe);
        while (used >= 4 && std::strcmp(buf + used - 4, " <- ") == 0) {
            used -= 4;
            buf[used] = '\0';
        }
        if (used == 0) {
            continue;
        }
        bool has_file =
            std::strstr(buf, ".cpp") || std::strstr(buf, ".cc") || std::strstr(buf, ".h") || std::strstr(buf, ".hpp");
        if (!has_file && std::strstr(buf, "??") != nullptr) {
            continue;
        }
        std::snprintf(out, out_sz, "%s", buf);
        return true;
    }
    return false;
}

// Collapse balanced template-arg lists <...> to <> so frames stay readable
// (kernel NoC frames otherwise carry pages of TensorAccessor<...>). The " <- "
// inlined-frame separator ('<' then '-') is left intact.
void emule_asan_collapse_angles(char* s) {
    char* w = s;
    int depth = 0;
    for (const char* r = s; *r != '\0'; ++r) {
        if (*r == '<' && r[1] != '-') {
            if (depth == 0) {
                *w++ = '<';
            }
            ++depth;
        } else if (*r == '>' && depth > 0) {
            --depth;
            if (depth == 0) {
                *w++ = '>';
            }
        } else if (depth == 0) {
            *w++ = *r;
        }
    }
    *w = '\0';
}

void emule_asan_print_trace() {
    std::fflush(stdout);
    fprintf(stderr, "  --- emule ASAN context ---\n");
    const char* kn = (__emule_self != nullptr) ? __emule_self->san.kernel_name : nullptr;
    if (kn != nullptr && kn[0] != '\0') {
        fprintf(stderr, "  kernel:    %s\n", kn);
        fprintf(
            stderr,
            "  core:      logical (%u, %u)  physical (%u, %u)\n",
            __emule_self->san.logical_x,
            __emule_self->san.logical_y,
            static_cast<unsigned>(my_x[0]),
            static_cast<unsigned>(my_y[0]));
        fprintf(
            stderr,
            "  processor: %u  (neo %u, trisc %u)\n",
            static_cast<unsigned>(__emule_self->san.processor_id),
            static_cast<unsigned>(__emule_self->neo_id),
            static_cast<unsigned>(__emule_self->trisc_id));
    }

    void* frames[128];
    int n = backtrace(frames, 128);
    char** syms = backtrace_symbols(frames, n);
    fprintf(stderr, "  backtrace (most recent call first):\n");

    int printed = 0;
    bool started = false;  // suppress the leading ASAN-machinery frames
    for (int i = 0; i < n && printed < 16; ++i) {
        Dl_info info;
        bool have_info = (dladdr(frames[i], &info) != 0) && info.dli_fname != nullptr;
        const char* module = have_info ? info.dli_fname : nullptr;
        // backtrace() yields return addresses, so symbolize addr-1 to land on the
        // calling instruction's line.
        uintptr_t off =
            have_info ? (reinterpret_cast<uintptr_t>(frames[i]) - reinterpret_cast<uintptr_t>(info.dli_fbase)) : 0;
        char resolved[2048];
        bool ok =
            module != nullptr && emule_asan_symbolize(module, off > 0 ? off - 1 : off, resolved, sizeof(resolved));

        // Skip print_trace / panic so the first frame shown is the check itself.
        if (!started) {
            const char* probe = ok ? resolved : ((have_info && info.dli_sname != nullptr) ? info.dli_sname : "");
            if (std::strstr(probe, "emule_asan_") != nullptr || std::strstr(probe, "__emule_asan_") != nullptr) {
                continue;
            }
            started = true;
        }

        if (ok) {
            emule_asan_collapse_angles(resolved);
            fprintf(stderr, "    #%-2d %s\n", printed, resolved);
        } else if (syms != nullptr && syms[i] != nullptr) {
            fprintf(stderr, "    #%-2d %s\n", printed, syms[i]);
        } else {
            fprintf(stderr, "    #%-2d %p\n", printed, frames[i]);
        }
        ++printed;

        // Stop at the kernel entry — everything below is runner/thread/libc glue.
        if (ok && (std::strstr(resolved, "kernel_main") != nullptr ||
                   std::strstr(resolved, "__emule_kernel_entry") != nullptr)) {
            break;
        }
    }
    if (syms != nullptr) {
        free(syms);
    }
    std::fflush(stderr);
}

// Decide the fate of the core dump the imminent abort() would trigger, per
// TT_METAL_EMULE_ASAN_ALLOW_CORE. Unset (default): mark the process non-dumpable
// (PR_SET_DUMPABLE=0) so the ~1.4 GB L1+DRAM core is suppressed even where
// `ulimit -c 0` is ignored. Set: self-dump via gcore to ./emule_asan_core.<pid>
// (best-effort). Called once, under the panic lock. See SANITIZER_CHECKS.md "Core dumps".
void emule_asan_handle_coredump() {
#if defined(__linux__)
    if (std::getenv("TT_METAL_EMULE_ASAN_ALLOW_CORE") == nullptr) {
        prctl(PR_SET_DUMPABLE, 0, 0, 0, 0);
        return;
    }
    char pid[16];
    std::snprintf(pid, sizeof(pid), "%d", static_cast<int>(getpid()));
    prctl(PR_SET_PTRACER, -1, 0, 0, 0);  // PR_SET_PTRACER_ANY: let the gcore child attach
    pid_t child = fork();
    if (child == 0) {
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            dup2(devnull, 1);
            dup2(devnull, 2);
        }
        execlp("gcore", "gcore", "-o", "emule_asan_core", pid, static_cast<char*>(nullptr));
        _exit(127);  // gcore not on PATH
    }
    if (child > 0) {
        int status = 0;
        waitpid(child, &status, 0);
        char cwd[1024];
        const char* dir = (getcwd(cwd, sizeof(cwd)) != nullptr) ? cwd : ".";
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            fprintf(stderr, "  [emule ASAN] core written to %s/emule_asan_core.%s\n", dir, pid);
        } else {
            fprintf(stderr, "  [emule ASAN] gcore unavailable/failed; no core captured (needs gdb gcore + ptrace)\n");
        }
    }
#endif
}

}  // namespace

// Report one [ASAN ERROR] and abort. `fmt`/args are the printf-style error line
// (printed verbatim); the context+backtrace follow. Pass nullptr for those only.
// Serialized: the winning thread prints one report and aborts while the others
// block here, so only ONE error is ever emitted under a multi-core kernel bug.
extern "C" [[noreturn]] void __emule_asan_panic(const char* fmt, ...) {
    static std::mutex panic_mu;
    panic_mu.lock();  // intentionally never unlocked — the winner aborts holding it
    emule_asan_handle_coredump();
    if (fmt != nullptr) {
        va_list ap;
        va_start(ap, fmt);
        std::vfprintf(stderr, fmt, ap);
        va_end(ap);
    }
    emule_asan_print_trace();
    std::abort();
}
