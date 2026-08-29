// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Discriminator: is the PCIe destination address correct, or is the EDM the
// problem?
//
// The fabric test proved the sender does everything right (connection opened,
// 16 sends completed) and nothing arrives in host memory. Two possibilities
// remain: (a) the address I hand the EDM is wrong, (b) the EDM's receive path
// will not deliver to a PCIe tile.
//
// This kernel runs ON THE MMIO CHIP and writes to the same address directly,
// with no fabric involved. If the bytes land, the address is right and the EDM
// is the problem. If they do not, my address is wrong and fabric is exonerated.
//
// It also reports the address computed the way cq_realtime_profiler_push.cpp
// does -- NOC_XY_PCIE_ENCODING(NOC_X_PHYS_COORD(x), ...) resolved against this
// kernel's own noc_index -- so the host can diff it against the host-computed
// value numerically.
//
// Runtime args:
//   0: dest_lo   host-computed destination (NOC1-encoded PCIe | driver offset)
//   1: dest_hi
//   2: pcie_noc_x  NOC0 coords, for the in-kernel recomputation
//   3: pcie_noc_y
//   4: driver_off_lo
//   5: driver_off_hi
//   6: src_l1     scratch holding the payload
//   7: bytes

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t i = 0;
    const uint32_t dest_lo = get_arg_val<uint32_t>(i++);
    const uint32_t dest_hi = get_arg_val<uint32_t>(i++);
    const uint32_t pcie_noc_x = get_arg_val<uint32_t>(i++);
    const uint32_t pcie_noc_y = get_arg_val<uint32_t>(i++);
    const uint32_t off_lo = get_arg_val<uint32_t>(i++);
    const uint32_t off_hi = get_arg_val<uint32_t>(i++);
    const uint32_t src_l1 = get_arg_val<uint32_t>(i++);
    const uint32_t bytes = get_arg_val<uint32_t>(i++);

    const uint64_t host_dest = ((uint64_t)dest_hi << 32) | (uint64_t)dest_lo;
    const uint64_t driver_off = ((uint64_t)off_hi << 32) | (uint64_t)off_lo;

    // The profiler's recipe, resolved against THIS kernel's noc_index.
    const uint64_t kernel_dest =
        ((uint64_t)NOC_XY_PCIE_ENCODING(NOC_X_PHYS_COORD(pcie_noc_x), NOC_Y_PHYS_COORD(pcie_noc_y))) | driver_off;

    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_l1);
    p[0] = 0x51DECAFEu;
    p[1] = 0x1u;

    volatile tt_l1_ptr uint32_t* dbg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_l1 + 64);
    dbg[0] = 0xA11E0000u;
    dbg[1] = (uint32_t)(kernel_dest & 0xFFFFFFFFu);   // what THIS kernel computes
    dbg[2] = (uint32_t)(kernel_dest >> 32);
    dbg[3] = noc_index;                                // which NOC we are on

    // Write via the host-computed address first.
    noc_async_write(src_l1, host_dest, bytes);
    noc_async_write_barrier();
    dbg[0] = 0x00110000u;  // host-address write issued and flushed

    // Then via the kernel-computed one, at +128 in the host buffer, so the host
    // can see which of the two (if either) actually landed.
    p[1] = 0x2u;
    noc_async_write(src_l1, kernel_dest + 128, bytes);
    noc_async_write_barrier();
    dbg[0] = 0x00220000u;  // both writes issued
}
