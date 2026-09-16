// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// RISC-V-faithful integer divide/modulo fault recovery on x86 hosts. RISC-V div/rem faults
// are DEFINED and non-trapping, but the JIT-compiled x86 `div`/`idiv` raises #DE -> SIGFPE.
// EmuleSigfpeGuard installs a handler for its lifetime that writes RISC-V's defined result
// into the saved register image and steps RIP past the fault. See docs/riscv-intdiv-by-zero.md
// in the tt-emule repo. x86_64/linux only (the supported emule host).

#if defined(__x86_64__) && defined(__linux__)
#include <csignal>

namespace tt::tt_metal::emule {

// Installs the handler for the duration of kernel execution, restoring the previous
// disposition afterward so emule does not permanently alter the host.
struct EmuleSigfpeGuard {
    struct sigaction prev_{};
    bool installed_ = false;
    EmuleSigfpeGuard();
    ~EmuleSigfpeGuard();
    EmuleSigfpeGuard(const EmuleSigfpeGuard&) = delete;
    EmuleSigfpeGuard& operator=(const EmuleSigfpeGuard&) = delete;
};

}  // namespace tt::tt_metal::emule
#endif  // __x86_64__ && __linux__
