// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TRISC0 adds two int32 vectors in L1 with RISC-V Vector instructions (ckernel_vector.h):
// c[i] = a[i] + b[i], with a, b and c laid out back to back at l1_address.
// Requires ComputeHardwareConfig::Compute2XXConfig::enable_unpack_rvv; the other TRISCs do nothing.

#include "api/compute/common.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"

#ifdef TRISC_UNPACK
#ifndef __riscv_vector
#error "TRISC0 was not compiled with the RISC-V Vector extension (enable_unpack_rvv not applied)"
#endif
#include "ckernel_vector.h"
#endif

void kernel_main() {
#ifdef TRISC_UNPACK
    // 32 x e32 at LMUL=8 is exactly VLMAX for zvl128b.
    constexpr uint32_t num_elements = 32;
    const uint32_t l1_address = get_arg(args::l1_address) + MEM_L1_UNCACHED_BASE;

    const int32_t* a = reinterpret_cast<const int32_t*>(l1_address);
    const int32_t* b = a + num_elements;
    int32_t* c = reinterpret_cast<int32_t*>(l1_address) + 2 * num_elements;

    (void)vsetvl<vdatasz::E32, vregmult::M8, num_elements>();
    vector_load<vreg::V0>(a);
    vector_load<vreg::V8>(b);
    vector_add<vreg::V16, vreg::V0, vreg::V8>();
    vector_store<vreg::V16>(c);
#endif
}
