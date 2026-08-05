// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Minimal device PCIe loopback kernel, modeled on cq_prefetch.cpp read_from_pcie() and
// cq_dispatch.cpp host writes:
//   1. noc_async_read from host (PCIe NOC address) into L1 staging
//   2. noc_async_write from L1 staging back to a second host (PCIe NOC) address

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "noc/noc_parameters.h"

void kernel_main() {
    constexpr uint32_t host_src_pcie_addr = get_arg(args::host_src_pcie_addr);
    constexpr uint32_t host_dst_pcie_addr = get_arg(args::host_dst_pcie_addr);
    constexpr uint32_t l1_staging_addr = get_arg(args::l1_staging_addr);
    constexpr uint32_t transfer_size_bytes = get_arg(args::transfer_size_bytes);

    const uint64_t pcie_noc_xy = uint64_t(NOC_XY_PCIE_ENCODING(PCIE_NOC_X, PCIE_NOC_Y));

    const uint64_t host_src_noc_addr = pcie_noc_xy | host_src_pcie_addr;
    noc_async_read(host_src_noc_addr, l1_staging_addr, transfer_size_bytes);
    noc_async_read_barrier();

    const uint64_t host_dst_noc_addr = pcie_noc_xy | host_dst_pcie_addr;
    noc_async_write(l1_staging_addr, host_dst_noc_addr, transfer_size_bytes);
    noc_async_write_barrier();
}
