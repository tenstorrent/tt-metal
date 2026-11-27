// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %s -o %t.o
// RUN: %{blackhole_objdump} -drz %t.o > %t.dump
// RUN: %{blackhole_compare_codegen} %t.dump

// Compare each HAL function with its adjacent reference_ function. Disable
// identical-code folding so both bodies are emitted independently.
// The references are production's RISC stores through the GPR window, from cpack_common.h.

#include <cstdint>

#include "hal/scalar.h"

namespace scalar = hal::scalar;

extern "C" __attribute__((noinline, used)) void write_tile_header(const std::uint32_t tile_size)
{
    scalar::set<scalar::Access::MMIO>(hal::gpr<ckernel::p_gpr_pack::TILE_HEADER>(), tile_size);
    scalar::set<scalar::Access::MMIO>(hal::gpr<ckernel::p_gpr_pack::TILE_HEADER + 1>(), 0);
    scalar::set<scalar::Access::MMIO>(hal::gpr<ckernel::p_gpr_pack::TILE_HEADER + 2>(), 0);
    scalar::set<scalar::Access::MMIO, scalar::Completion::Wait>(hal::gpr<ckernel::p_gpr_pack::TILE_HEADER + 3>(), 0);
}

extern "C" __attribute__((noinline, used)) void reference_write_tile_header(const std::uint32_t tile_size)
{
    ckernel::regfile[ckernel::p_gpr_pack::TILE_HEADER]     = tile_size;
    ckernel::regfile[ckernel::p_gpr_pack::TILE_HEADER + 1] = 0;
    ckernel::regfile[ckernel::p_gpr_pack::TILE_HEADER + 2] = 0;
    ckernel::regfile[ckernel::p_gpr_pack::TILE_HEADER + 3] = 0;
    static_cast<void>(ckernel::load_blocking(&ckernel::regfile[ckernel::p_gpr_pack::TILE_HEADER + 3]));
}

extern "C" __attribute__((noinline, used)) std::uint32_t read_runtime(const std::uint32_t index)
{
    return scalar::read<scalar::Access::MMIO>(hal::gpr(index));
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_read_runtime(const std::uint32_t index)
{
    return ckernel::regfile[index];
}
