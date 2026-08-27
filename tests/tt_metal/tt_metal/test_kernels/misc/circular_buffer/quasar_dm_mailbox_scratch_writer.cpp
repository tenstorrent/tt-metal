// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"
#include "internal/hw_thread.h"
#include "internal/tt-2xx/risc_common.h"
#include "internal/tt-2xx/quasar/tensix_neo_reg.h"
#include "internal/tt-2xx/quasar/overlay/meta/registers/overlay_reg_defines_core.h"

// DM-side producer for the QuasarDmToTriscMailbox test. Writes one value per receiving TRISC
// thread into the TRISC mailbox queues of this cluster's NEO0 Tensix engine:
// queue k = reader*4 + writer, reader in {T0=UNPACK, T1=MATH, T2=PACK}, writer slot T3=IsolateSfpu:
//   UNPACK: k = 0*4+3 = 3  (TRISC_MAILBOX_3,  0x0180018C)
//   MATH:   k = 1*4+3 = 7  (TRISC_MAILBOX_7,  0x0180019C)
//   PACK:   k = 2*4+3 = 11 (TRISC_MAILBOX_11, 0x018001AC)
// A DM core is not a TRISC, so it "impersonates" writer T3 (per the Quasar HW addressing model);
// each receiving thread reads its queue back with mailbox_read(IsolateSfpuThreadId). The writes
// are plain MMIO stores to the queues' NEO0 DM/NOC addresses -- that address view is directly
// load/store-accessible from the DM core.
//
// The mailbox stores are bracketed by two sentinel writes to the cluster-control scratch register
// SCRATCH_16 (0x03000080): 0x00AAAAAA before and 0x0BBBBBBB after. These act as waveform markers
// on Zebu: if a mailbox MMIO store hangs or faults, the missing 0x0BBBBBBB localizes the failure
// to the mailbox accesses themselves.
void kernel_main() {
    // The Quasar DM allocator (GetProcessorsPerClusterQuasar) skips reserved DM0/DM1, so the
    // host's request for one DM thread places this kernel on DM2 only. Gate on hartid == 2
    // defensively so exactly one core performs the writes even if the allocation grows.
    if (internal_::get_hw_thread_idx() != 2) {
        return;
    }
    constexpr std::uint32_t value_unpack = get_compile_time_arg_val(0);
    constexpr std::uint32_t value_math = get_compile_time_arg_val(1);
    constexpr std::uint32_t value_pack = get_compile_time_arg_val(2);
    volatile tt_l1_ptr std::uint32_t* const scratch = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
        TT_CLUSTER_CTRL_SCRATCH_16__REG_ADDR);  // Address 0x03000080
    volatile tt_l1_ptr std::uint32_t* const queue_unpack = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
        NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_TRISC_MAILBOX_3__REG_ADDR);  // Address 0x0180018C
    volatile tt_l1_ptr std::uint32_t* const queue_math = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
        NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_TRISC_MAILBOX_7__REG_ADDR);  // Address 0x0180019C
    volatile tt_l1_ptr std::uint32_t* const queue_pack = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
        NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_TRISC_MAILBOX_11__REG_ADDR);  // Address 0x018001AC

    scratch[0] = 0x00AAAAAA;

    queue_unpack[0] = value_unpack;
    queue_math[0] = value_math;
    queue_pack[0] = value_pack;

    scratch[0] = 0x0BBBBBBB;
}
