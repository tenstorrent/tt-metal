// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SIGFPE -> RISC-V divide/overflow recovery. See emule_diagnostics.hpp and
// docs/riscv-intdiv-by-zero.md. The handler is process-global for the guard's lifetime.

#include "emule_diagnostics.hpp"

#if defined(__x86_64__) && defined(__linux__)
#include <cstddef>
#include <cstdint>
#include <ucontext.h>
#include <sys/ucontext.h>

namespace tt::tt_metal::emule {
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
    ++i;                // ModRM
    if (mod != 3) {     // memory operand
        if (rm == 4) {  // SIB present
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

}  // namespace

EmuleSigfpeGuard::EmuleSigfpeGuard() {
    struct sigaction sa{};
    sa.sa_sigaction = emule_sigfpe_handler;
    sa.sa_flags = SA_SIGINFO;  // synchronous, thread-directed; handler never re-faults
    sigemptyset(&sa.sa_mask);
    installed_ = (sigaction(SIGFPE, &sa, &prev_) == 0);
}

EmuleSigfpeGuard::~EmuleSigfpeGuard() {
    if (installed_) {
        sigaction(SIGFPE, &prev_, nullptr);
    }
}

}  // namespace tt::tt_metal::emule
#endif  // __x86_64__ && __linux__
