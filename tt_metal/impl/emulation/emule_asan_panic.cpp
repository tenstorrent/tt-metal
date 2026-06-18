// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal-side definition of __emule_asan_panic + its diagnostic-trace facility.
//
// This is the single definition emitted into libtt_metal: host-API sanitizer
// checks call it directly, and JIT'd kernel .so files resolve it at dlopen via
// -rdynamic (like the other __emule_* symbols). It is a faithful mirror of the
// definition in tt-emule's src/kernel_runner.cpp — the two libraries are never
// linked into the same binary, so the duplicate is benign. Keep the two copies
// in sync. (Previously this was pulled in by #include "jit_hw/emule_asan.h"
// under EMULE_ASAN_IMPLEMENTATION; defining it here keeps jit_hw out of metal.)
//
// What it prints and why is documented in SANITIZER_CHECKS.md ("Diagnostic trace").

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

// Identity thread-locals read by the trace; defined in emulated_program_runner.cpp.
extern thread_local uint8_t my_x[2];
extern thread_local uint8_t my_y[2];
extern thread_local uint32_t __emule_logical_x;
extern thread_local uint32_t __emule_logical_y;
extern thread_local uint8_t __processor_id;
extern thread_local uint8_t __emule_neo_id;
extern thread_local uint8_t __emule_trisc_id;
// Source path of the kernel currently on this thread, or nullptr for host-API checks.
extern thread_local const char* __emule_kernel_name;

namespace {

// Resolve one instruction address (module + file-relative offset) to a
// "func at file:line" string via llvm-symbolizer (preferred — reads clang's
// DWARF5), falling back to addr2line. Inlined frames are flattened to one line
// joined by " <- ". Returns false if neither tool resolved real debug info.
bool emule_asan_symbolize(const char* module, uintptr_t file_offset, char* out, size_t out_sz) {
    if (module == nullptr || module[0] == '\0' || out_sz == 0) {
        return false;
    }
    // The JIT dlopen's the kernel from "<hash>.so.tmp.<pid>", then atomically
    // renames it to "<hash>.so", so dladdr hands back a path that no longer
    // exists. If the reported module is gone, strip ".tmp.<pid>" to recover the
    // real on-disk .so for the symbolizer.
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
    // Unversioned llvm-symbolizer is often absent, so try toolchain-versioned
    // names too; addr2line is the last-ditch fallback (only yields "??:?" on DWARF5).
    const char* tools[] = {
        "llvm-symbolizer --obj=\"%s\" --pretty-print --inlines --demangle 0x%lx 2>/dev/null",
        "llvm-symbolizer-20 --obj=\"%s\" --pretty-print --inlines --demangle 0x%lx 2>/dev/null",
        "llvm-symbolizer-19 --obj=\"%s\" --pretty-print --inlines --demangle 0x%lx 2>/dev/null",
        "llvm-symbolizer-18 --obj=\"%s\" --pretty-print --inlines --demangle 0x%lx 2>/dev/null",
        "addr2line -f -C -i -p -e \"%s\" 0x%lx 2>/dev/null",
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
    if (__emule_kernel_name != nullptr && __emule_kernel_name[0] != '\0') {
        fprintf(stderr, "  kernel:    %s\n", __emule_kernel_name);
        fprintf(
            stderr,
            "  core:      logical (%u, %u)  physical (%u, %u)\n",
            __emule_logical_x,
            __emule_logical_y,
            static_cast<unsigned>(my_x[0]),
            static_cast<unsigned>(my_y[0]));
        fprintf(
            stderr,
            "  processor: %u  (neo %u, trisc %u)\n",
            static_cast<unsigned>(__processor_id),
            static_cast<unsigned>(__emule_neo_id),
            static_cast<unsigned>(__emule_trisc_id));
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

}  // namespace

// Report one [ASAN ERROR] and abort. `fmt`/args are the printf-style error line
// (printed verbatim, so message text and test regexes are unchanged); the
// context+backtrace follow. emule runs one thread per core, so a kernel bug
// trips every core at once: the first thread takes the lock, prints exactly one
// full report, and aborts (tearing down the process) while every other thread
// blocks on the lock — so only ONE error is ever emitted. Pass nullptr to print
// only the context+backtrace.
extern "C" [[noreturn]] void __emule_asan_panic(const char* fmt, ...) {
    static std::mutex panic_mu;
    panic_mu.lock();  // intentionally never unlocked — the winner aborts holding it
    if (fmt != nullptr) {
        va_list ap;
        va_start(ap, fmt);
        std::vfprintf(stderr, fmt, ap);
        va_end(ap);
    }
    emule_asan_print_trace();
    std::abort();
}
