// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eth_l1_address_map.h"
#include "risc_common.h"
#include "tt_eth_api.h"
#include "noc_nonblocking_api.h"
#include "../dataflow_api.h"

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
    uint32_t reserved_0_;
    uint32_t reserved_1_;
    uint32_t reserved_2_;
    volatile uint32_t bytes_received;
    uint32_t reserved_3_;
    uint32_t reserved_4_;
    uint32_t reserved_5_;
};

erisc_info_t *erisc_info = (erisc_info_t *)(eth_l1_mem::address_map::ERISC_APP_SYNC_INFO_BASE);
volatile uint32_t *flag_disable = (uint32_t *)(eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG);

extern uint32_t __erisc_jump_table;
volatile uint32_t *RtosTable =
    (volatile uint32_t *)&__erisc_jump_table;  // Rtos Jump Table. Runtime application needs rtos function handles.;

void (*rtos_context_switch_ptr)();

void __attribute__((section("code_l1"))) risc_context_switch() {
    ncrisc_noc_full_sync();
    rtos_context_switch_ptr();
    ncrisc_noc_counters_init();
}

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
 * A blocking call that waits until the value of a local L1 memory address on
 * the Tensix core executing this function becomes equal to a target value.
 * This L1 memory address is used as a semaphore of size 4 Bytes, as a
 * synchronization mechanism. Also, see *noc_semaphore_set*.
 *
 * Return value: None
 *
 * | Argument  | Description                                                    | Type     | Valid Range        | Required |
 * |-----------|----------------------------------------------------------------|----------|--------------------|----------|
 * | sem_addr  | Semaphore address in local L1 memory                           | uint32_t | 0..1MB             | True     |
 * | val       | The target value of the semaphore                              | uint32_t | Any uint32_t value | True     |
 */
FORCE_INLINE
void eth_noc_semaphore_wait(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
    while ((*sem_addr) != val) {
        risc_context_switch();
    }
}

/**
 * This blocking call waits for all the outstanding enqueued *noc_async_read*
 * calls issued on the current Tensix core to complete. After returning from
 * this call the *noc_async_read* queue will be empty for the current Tensix
 * core.
 *
 * Return value: None
 */
FORCE_INLINE
void eth_noc_async_read_barrier() {
    while (!ncrisc_noc_reads_flushed(noc_index)) {
        risc_context_switch();
    }
}

/**
 * This blocking call waits for all the outstanding enqueued *noc_async_write*
 * calls issued on the current Tensix core to complete. After returning from
 * this call the *noc_async_write* queue will be empty for the current Tensix
 * core.
 *
 * Return value: None
 */
FORCE_INLINE
void eth_noc_async_write_barrier() {
    while (!ncrisc_noc_nonposted_writes_flushed(noc_index)) {
        risc_context_switch();
    }
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
void eth_send_bytes(
    uint32_t src_addr,
    uint32_t dst_addr,
    uint32_t num_bytes,
    uint32_t num_bytes_per_send = 16,
    uint32_t num_bytes_per_send_word_size = 1) {
    uint32_t num_bytes_sent = 0;
    while (num_bytes_sent < num_bytes) {
        eth_send_packet(
            0, ((num_bytes_sent + src_addr) >> 4), ((num_bytes_sent + dst_addr) >> 4), num_bytes_per_send_word_size);
        num_bytes_sent += num_bytes_per_send;
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
 * A blocking call that waits for remote receiver to acknowledge that all data sent with eth_send_bytes since the last
 * reset_erisc_info call is no longer being used. Also, see \a eth_receiver_done().
 * This also syncs with the local receiver core using a semaphore and reads date from the receiver core
 *
 * Return value: None
 *
 * | Argument                    | Description                                               | Type     | Valid Range                                                   | Required |
 * |-----------------------------|-----------------------------------------------------------|----------|---------------------------------------------------------------|----------|
 * | sender_semaphore_addr_ptr   | Semaphore address in local L1 memory                      | uint32_t | 0..1MB                                                        | True     |
 * | receiver_semaphore_noc_addr | Encoding of the receiver semaphore location (x,y)+address | uint64_t | DOX-TODO(insert a reference to what constitutes valid coords) | True     |
 * | receiver_data_noc_addr      | Encoding of the receiver source location (x,y)+address    | uint64_t | DOX-TODO(ref to explain valid coords)                         | True     |
 * | dst_local_l1_addr           | Address in local L1 memory                                | uint32_t | 0..1MB                                                        | True     |
 * | size                        | Size of data transfer in bytes                            | uint32_t | 0..1MB                                                        | True     |
 */
FORCE_INLINE
void eth_wait_for_remote_receiver_done_and_get_local_receiver_data(
    volatile tt_l1_ptr uint32_t* sender_semaphore_addr_ptr,
    uint64_t receiver_semaphore_noc_addr,
    uint64_t receiver_data_noc_addr,
    uint32_t local_eth_l1_curr_src_addr,
    uint32_t size
) {
    eth_send_packet(0, ((uint32_t)(&(erisc_info->bytes_sent))) >> 4, ((uint32_t)(&(erisc_info->bytes_sent))) >> 4, 1);
    eth_noc_semaphore_wait(sender_semaphore_addr_ptr, 1);
    noc_async_read(receiver_data_noc_addr, local_eth_l1_curr_src_addr, size);
    eth_noc_async_read_barrier();
    noc_semaphore_set(sender_semaphore_addr_ptr, 0);
    noc_semaphore_inc(receiver_semaphore_noc_addr, 1);
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
