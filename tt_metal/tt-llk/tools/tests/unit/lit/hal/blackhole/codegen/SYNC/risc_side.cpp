// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %s -o %t.o
// RUN: %{blackhole_objdump} -drz %t.o > %t.dump
// RUN: %{blackhole_compare_codegen} %t.dump

// Compare each HAL function with its adjacent reference_ function. Disable
// identical-code folding so both bodies are emitted independently.
// The references are the ckernel RISC helpers these calls replace.

#include <cstdint>

#include "hal/sync.h"

namespace sync = hal::sync;

extern "C" __attribute__((noinline, used)) void wait_thread_idle()
{
    sync::wait::thread_idle();
}

extern "C" __attribute__((noinline, used)) void reference_wait_thread_idle()
{
    ckernel::tensix_sync();
}

extern "C" __attribute__((noinline, used)) void wait_mop_expander_idle()
{
    sync::wait::mop_expander_idle();
}

extern "C" __attribute__((noinline, used)) void reference_wait_mop_expander_idle()
{
    ckernel::mop_sync();
}

extern "C" __attribute__((noinline, used)) void mailbox_write(const std::uint32_t value)
{
    sync::mailbox::write<sync::Riscv::Math>(value);
}

extern "C" __attribute__((noinline, used)) void reference_mailbox_write(const std::uint32_t value)
{
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, value);
}

extern "C" __attribute__((noinline, used)) std::uint32_t mailbox_read()
{
    return sync::mailbox::read<sync::Riscv::Math>();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_mailbox_read()
{
    return ckernel::mailbox_read(ckernel::ThreadId::MathThreadId);
}

extern "C" __attribute__((noinline, used)) bool mailbox_has_data()
{
    return sync::mailbox::has_data<sync::Riscv::Pack>();
}

extern "C" __attribute__((noinline, used)) bool reference_mailbox_has_data()
{
    return ckernel::mailbox_not_empty(ckernel::ThreadId::PackThreadId);
}

extern "C" __attribute__((noinline, used)) std::uint32_t mailbox_runtime(const sync::Riscv peer, const std::uint32_t value)
{
    sync::mailbox::write(peer, value);
    return sync::mailbox::read(peer);
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_mailbox_runtime(const sync::Riscv peer, const std::uint32_t value)
{
    ckernel::mailbox_write(static_cast<std::uint8_t>(peer), value);
    return ckernel::mailbox_read(static_cast<std::uint8_t>(peer));
}
