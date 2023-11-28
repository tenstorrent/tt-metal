// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "eth_l1_address_map.h"
#include "risc_common.h"
#include "tt_eth_api.h"
#include "noc_nonblocking_api.h"

#define FORCE_INLINE inline __attribute__((always_inline))

inline void RISC_POST_STATUS(uint32_t status) {
    volatile uint32_t *ptr = (volatile uint32_t *)(NOC_CFG(ROUTER_CFG_2));
    ptr[0] = status;
}
struct erisc_info_t {
    volatile uint32_t num_bytes;
    volatile uint32_t mode;
    volatile uint32_t unused_arg0;
    volatile uint32_t unused_arg1;
    volatile uint32_t bytes_sent;
    volatile uint32_t reserved_0_;
    volatile uint32_t reserved_1_;
    volatile uint32_t reserved_2_;
    volatile uint32_t bytes_received;
    volatile uint32_t reserved_3_;
    volatile uint32_t reserved_4_;
    volatile uint32_t reserved_5_;
};

volatile erisc_info_t *erisc_info = (erisc_info_t *)(eth_l1_mem::address_map::ERISC_APP_SYNC_INFO_BASE);
volatile uint32_t *flag_disable = (uint32_t *)(eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG);

extern uint32_t __erisc_jump_table;
void (*rtos_context_switch_ptr)();
volatile uint32_t *RtosTable =
    (volatile uint32_t *)&__erisc_jump_table;  // Rtos Jump Table. Runtime application needs rtos function handles.;

void __attribute__((section("code_l1"))) risc_context_switch() {
    ncrisc_noc_full_sync();
    rtos_context_switch_ptr();
    ncrisc_noc_counters_init();
}

constexpr static uint32_t NUM_BYTES_PER_SEND = 16;  // internal optimization, requires testing
constexpr static uint32_t NUM_BYTES_PER_SEND_LOG2 = 4;

FORCE_INLINE
void reset_erisc_info() {
    erisc_info->bytes_sent = 0;
}

FORCE_INLINE
void disable_erisc_app() { flag_disable[0] = 0; }

FORCE_INLINE
void check_and_context_switch() {
    uint32_t start_time = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t end_time = start_time;
    while (end_time - start_time < 100000) {
        RISC_POST_STATUS(0xdeadCAFE);
        risc_context_switch();
        RISC_POST_STATUS(0xdeadFEAD);
        end_time = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    }
    // proceed
}

/**
 * Initiates an asynchronous write from a source address in L1 memory on the local ethernet core to L1 of the connected
 * remote ethernet core. Also, see \a eth_wait_for_receiver_done and \a eth_wait_for_bytes.
 *
 * Return value: None
 *
 * | Argument          | Description                                             | Type     | Valid Range | Required |
 * |-------------------|---------------------------------------------------------|----------|-------------|----------|
 * | src_addr          | Source address in local eth core L1 memory              | uint32_t | 0..256kB | True     |
 * | dst_addr          | Destination address in remote eth core L1 memory        | uint32_t | 0..256kB | True     |
 * | num_bytes         | Size of data transfer in bytes, must be multiple of 16  | uint32_t | 0..256kB | True     |
 */
FORCE_INLINE
void eth_send_bytes(uint32_t src_addr, uint32_t dst_addr, uint32_t num_bytes) {
    uint32_t num_loops = num_bytes >> NUM_BYTES_PER_SEND_LOG2;
    for (uint32_t i = 0; i < num_loops; i++) {
        eth_send_packet(0, i + (src_addr >> 4), i + (dst_addr >> 4), 1);
    }
    erisc_info->bytes_sent += num_bytes;
}

/**
 * A blocking call that waits for receiver to acknowledge that all data sent with eth_send_bytes since the last
 * reset_erisc_info call is no longer being used. Also, see \a eth_receiver_done().
 *
 * Return value: None
 *
 * | Argument          | Description                                             | Type     | Valid Range | Required |
 * |-------------------|---------------------------------------------------------|----------|-------------|----------|
 */
FORCE_INLINE
void eth_wait_for_receiver_done() {
    eth_send_packet(0, ((uint32_t)(&(erisc_info->bytes_sent))) >> 4, ((uint32_t)(&(erisc_info->bytes_sent))) >> 4, 1);
    while (erisc_info->bytes_sent != 0) {
        risc_context_switch();
    }
}

/**
 * A blocking call that waits for num_bytes of data to be sent from the remote sender ethernet core using any number of
 * eth_send_byte. User must ensure that num_bytes is equal to the total number of bytes sent. Example 1:
 * eth_send_bytes(32), eth_wait_for_bytes(32). Example 2: eth_send_bytes(16), eth_send_bytes(32),
 * eth_wait_for_bytes(48).
 *
 * Return value: None
 *
 * | Argument          | Description                                             | Type     | Valid Range | Required |
 * |-------------------|---------------------------------------------------------|----------|-------------|----------|
 * | num_bytes         | Size of data transfer in bytes, must be multiple of 16  | uint32_t | 0..256kB | True     |
 */
FORCE_INLINE
void eth_wait_for_bytes(uint32_t num_bytes) {
    while (erisc_info->bytes_sent != num_bytes) {
        risc_context_switch();
    }
}

/**
 * Initiates an asynchronous call from receiver ethernet core to tell remote sender ethernet core that data sent
 * via eth_send_bytes is no longer being used. Also, see \a eth_wait_for_receiver_done
 *
 * Return value: None
 *
 * | Argument          | Description                                             | Type     | Valid Range | Required |
 * |-------------------|---------------------------------------------------------|----------|-------------|----------|
 */
FORCE_INLINE
void eth_receiver_done() {
    erisc_info->bytes_sent = 0;
    eth_send_packet(0, ((uint32_t)(&(erisc_info->bytes_sent))) >> 4, ((uint32_t)(&(erisc_info->bytes_sent))) >> 4, 1);
}

FORCE_INLINE
void eth_send_and_wait_for_receiver_done(uint32_t src_addr, uint32_t dst_addr, uint32_t num_bytes) {
    eth_send_bytes(src_addr, dst_addr, num_bytes);
    eth_wait_for_receiver_done();
}
