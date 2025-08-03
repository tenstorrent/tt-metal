// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "risc_common.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "ckernel_structs.h"
#include "stream_io_map.h"
#include "c_tensix_core.h"
#include "tdma_xmov.h"
#include "firmware_common.h"
#include "dev_msgs.h"
#include "risc_attribs.h"
#include "dataflow_api.h"
#include "ethernet/dataflow_api.h"
#include "ethernet/tunneling.h"
#include "blackhole/dev_mem_map.h"
#include "blackhole/noc_nonblocking_api.h"
#include "tt_metal/hw/inc/blackhole/core_config.h"
#include "tests/tt_metal/tt_metal/tunneling/fabric_lite.hpp"
#include "debug/dprint.h"
#include "init-fsm-basic.h"

#if !defined(tt_l1_ptr)
#define tt_l1_ptr __attribute__((rvtt_l1_ptr))
#endif

#if !defined(NUM_NOCS)
#define NUM_NOCS 2
#endif

#if !defined(NOC_INDEX)
#define NOC_INDEX 0
#endif

// Define METAL_LAUNCH if lite_fabric is being launched by Metal
#if !defined(METAL_LAUNCH)
#define IF_NOT_METAL_LAUNCH(x) x
#define IF_METAL_LAUNCH(x)
#else
#define IF_NOT_METAL_LAUNCH(x)
#define IF_METAL_LAUNCH(x) x
#endif

// Define a main function for lite_fabric being compiled as firmware or kernel
#define BEGIN_MAIN_FUNCTION()                                                           \
    IF_NOT_METAL_LAUNCH(int main())                                                     \
    IF_METAL_LAUNCH(void kernel_main()) {                                               \
        IF_NOT_METAL_LAUNCH(configure_csr();)                                           \
        IF_NOT_METAL_LAUNCH(do_crt1((uint32_t*)MEM_AERISC_INIT_LOCAL_L1_BASE_SCRATCH);) \
        IF_NOT_METAL_LAUNCH(noc_bank_table_init(MEM_AERISC_BANK_TO_NOC_SCRATCH);)       \
        IF_NOT_METAL_LAUNCH(risc_init();)                                               \
        IF_NOT_METAL_LAUNCH(noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);)                     \
        IF_NOT_METAL_LAUNCH(for (uint32_t n = 0; n < NUM_NOCS; n++) { noc_local_state_init(n); })

// End the main function
#define END_MAIN_FUNCTION()        \
    IF_NOT_METAL_LAUNCH(return 0;) \
    IF_METAL_LAUNCH(return;)       \
    }

constexpr uint32_t SOFT_RESET_ADDR = 0xFFB121B0;

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_atomics_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_posted_writes_num_issued[NUM_NOCS] __attribute__((used));

uint32_t tt_l1_ptr* rta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* crta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* sem_l1_base[ProgrammableCoreType::COUNT] __attribute__((used));

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));

// Not initialized anywhere yet
uint8_t my_logical_x_ __attribute__((used));
uint8_t my_logical_y_ __attribute__((used));
uint8_t my_relative_x_ __attribute__((used));
uint8_t my_relative_y_ __attribute__((used));

// Interface to the connected RISC1 processor via ethernet
struct ConnectedRisc1Interface {
    static constexpr uint32_t k_SoftResetAddr = 0xFFB121B0;
    static constexpr uint32_t k_ResetPcAddr = SUBORDINATE_AERISC_RESET_PC;

    template <uint32_t TXQ = 0>
    inline void send_packet(uint32_t src_addr, uint32_t dst_addr, uint32_t n) const {
        do {
            invalidate_l1_cache();
        } while (internal_::eth_txq_is_busy(TXQ));
        internal_::eth_send_packet(TXQ, src_addr >> 4, dst_addr >> 4, n);
    }

    // Put the connected RISC1 into reset
    inline void assert_connected_dm1_reset() const {
        constexpr uint32_t k_ResetValue = 0x47000;
        internal_::eth_write_remote_reg(0, k_SoftResetAddr, k_ResetValue);
    }

    // Take the connected RISC1 out of reset
    inline void deassert_connected_dm1_reset() const {
        constexpr uint32_t k_ResetValue = 0;
        internal_::eth_write_remote_reg(0, k_SoftResetAddr, k_ResetValue);
    }

    inline void set_pc(uint32_t pc) const { internal_::eth_write_remote_reg(0, k_ResetPcAddr, pc); }
};

BEGIN_MAIN_FUNCTION() {
    // Test values
    auto dptr = reinterpret_cast<volatile uint32_t*>(0x20000);
    dptr[0] = 0xdeadbeef;
    dptr[1] = 0xcafecafe;
}

auto structs = reinterpret_cast<volatile lite_fabric::LiteFabricMemoryMap*>(MEM_AERISC_FABRIC_LITE_CONFIG);
lite_fabric::routing_init(&structs->config);

END_MAIN_FUNCTION()
