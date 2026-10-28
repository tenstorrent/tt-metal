// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Op-to-op R/W inference coverage: exactly one FAMILY_* define is set per build, so this kernel's SOLE
// raw NoC access comes from exactly one public raw-address API. A family that fails to emit its own
// OPAQUE record (e.g. a forgotten tt_buf_rw::note_opaque() in a new/edited wrapper) shows up as this
// kernel's rw.opaque being false when the test expects true -- see test_op_to_op_hazards.cpp's
// BufRwEveryRawFamilyIndependentlyOpaque. No tensor bindings: correctness of the transfer itself is not
// checked, only the metadata. Multicast families target {self, one peer} (never self-only; see each raw
// function's own doc comment for why) via MCAST_X_START/Y_START/X_END/Y_END, set by the host.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "internal/local_mem/address_access.h"

void kernel_main() {
    Scratchpad<uint32_t> pad(scratch::pad);
    // Raw address: this kernel exercises the raw-address dataflow_api families on purpose.
    const uint32_t l1 = static_cast<uint32_t>(local_mem::detail::AddressAccess::address(pad.local_mem()));
    const uint32_t size = pad.size_in_bytes();
    const uint64_t self_addr = get_noc_addr(l1);

#if defined(FAMILY_READ_ONE_PACKET)
    noc_async_read_one_packet(self_addr, l1, size);
    noc_async_read_barrier();
#elif defined(FAMILY_READ_ANY_LEN)
    noc_async_read(self_addr, l1, size);
    noc_async_read_barrier();
#elif defined(FAMILY_READ_ONE_PACKET_SET_STATE)
    noc_async_read_one_packet_set_state(self_addr, size);
    noc_async_read_one_packet_with_state(l1, l1);
    noc_async_read_barrier();
#elif defined(FAMILY_READ_SET_STATE)
    noc_async_read_set_state(self_addr);
    noc_async_read_with_state(l1, l1, size);
    noc_async_read_barrier();
#elif defined(FAMILY_READ_SET_PCIE_STATE)
    noc_async_read_set_pcie_state(self_addr);
    noc_async_read_with_state(l1, l1, size);
    noc_async_read_clear_pcie_state();
    noc_async_read_barrier();
#elif defined(FAMILY_READ_PCIE)
    noc_async_read_pcie(self_addr, l1, size);
    noc_async_read_barrier();
#elif defined(FAMILY_WRITE_ONE_PACKET)
    noc_async_write_one_packet(l1, self_addr, size);
    noc_async_write_barrier();
#elif defined(FAMILY_WRITE_ANY_LEN)
    noc_async_write(l1, self_addr, size);
    noc_async_write_barrier();
#elif defined(FAMILY_WRITE_ONE_PACKET_SET_STATE)
    noc_async_write_one_packet_set_state(self_addr, size);
    noc_async_write_one_packet_with_state(l1, l1);
    noc_async_write_barrier();
#elif defined(FAMILY_WRITE_SET_STATE)
    noc_async_write_set_state(self_addr);
    noc_async_write_with_state(l1, l1, size);
    noc_async_write_barrier();
#elif defined(FAMILY_WRITE_SET_PCIE_STATE)
    noc_async_write_set_pcie_state(self_addr);
    noc_async_write_with_state(l1, l1, size);
    noc_async_write_clear_pcie_state();
    noc_async_write_barrier();
#elif defined(FAMILY_WRITE_PCIE)
    noc_async_write_pcie(l1, self_addr, size);
    noc_async_write_barrier();
#elif defined(FAMILY_WRITE_ONE_PACKET_WITH_TRID)
    noc_async_write_one_packet_with_trid(l1, self_addr, size, /*trid=*/1);
    noc_async_write_barrier_with_trid(1);
    // write_cmd_buf's NOC_PACKET_TAG is left at 1; the watcher requires it back at 0 before the next
    // kernel on this core runs (implicit-trid users assume they start from transaction id 0).
    noc_async_write_set_trid(0);
#elif defined(FAMILY_WRITE_ONE_PACKET_WITH_TRID_SET_STATE)
    noc_async_write_one_packet_with_trid_set_state(self_addr);
    noc_async_write_one_packet_with_trid_with_state(l1, l1, size, /*trid=*/1);
    noc_async_write_barrier_with_trid(1);
    noc_async_write_set_trid(0);  // see FAMILY_WRITE_ONE_PACKET_WITH_TRID above
#elif defined(FAMILY_INLINE_DW_WRITE)
    noc_inline_dw_write(self_addr, 0x1234);
    noc_async_write_barrier();
#elif defined(FAMILY_INLINE_DW_WRITE_SET_STATE)
    noc_inline_dw_write_set_state(self_addr);
    noc_inline_dw_write_with_state(0x1234);
    noc_async_write_barrier();
#elif defined(FAMILY_WRITE_MULTICAST_ONE_PACKET)
    noc_async_write_multicast_one_packet(
        l1, get_noc_multicast_addr(MCAST_X_START, MCAST_Y_START, MCAST_X_END, MCAST_Y_END, l1), size, /*num_dests=*/1);
    noc_async_write_barrier();
#elif defined(FAMILY_WRITE_MULTICAST)
    noc_async_write_multicast(
        l1, get_noc_multicast_addr(MCAST_X_START, MCAST_Y_START, MCAST_X_END, MCAST_Y_END, l1), size, /*num_dests=*/1);
    noc_async_write_barrier();
#elif defined(FAMILY_WRITE_MULTICAST_LOOPBACK_SRC)
    noc_async_write_multicast_loopback_src(
        l1, get_noc_multicast_addr(MCAST_X_START, MCAST_Y_START, MCAST_X_END, MCAST_Y_END, l1), size, /*num_dests=*/2);
    noc_async_write_barrier();
#elif defined(FAMILY_INLINE_MCAST_DW_WRITE)
    noc_inline_mcast_dw_write(
        get_noc_multicast_addr(MCAST_X_START, MCAST_Y_START, MCAST_X_END, MCAST_Y_END, l1), 0x1234);
    noc_async_write_barrier();
#else
#error "hazard_raw_family_coverage.cpp: no FAMILY_* define selected"
#endif
}
