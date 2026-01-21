// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "risc_common.h"
#include "hostdev/dev_msgs.h"
#include "eth_l1_address_map.h"
#include "risc_common.h"
#include "internal/ethernet/tt_eth_api.h"
#include "internal/ethernet/erisc.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "noc_nonblocking_api.h"
#include "internal/ethernet/tunneling.h"
#if defined(KERNEL_BUILD)
#include "api/dataflow/dataflow_api.h"
#else
#include "internal/dataflow/dataflow_api_common.h"
#endif
/**
 * Indicates if the ethernet transaction queue is busy ingesting a command at this moment,
 *
 * Return value: bool: true if the queue is ingesting a command and cannot accept a new one
 * at this specific moment
 */
FORCE_INLINE bool eth_txq_is_busy() { return internal_::eth_txq_is_busy(0); }

/**
 * Wait until the ethernet transaction queue is no longer busy ingesting a command
 *
 * Return value: None
 *
 * | Argument  | Description                                                    | Type     | Valid Range        |
 * Required |
 * |-----------|----------------------------------------------------------------|----------|--------------------|----------|
 * | wait_min  | The number of cycles to wait before performing run_routing()   | uint32_t | Any uint32_t value | False
 * |
 */
FORCE_INLINE void wait_for_eth_txq_cmd_space(uint32_t wait_min = 0) {
    uint32_t count = 0;
    while (eth_txq_is_busy()) {
        if (count == wait_min) {
            run_routing();
            count = 0;
        } else {
            count++;
        }
    }
}

/**
 * A blocking call that waits until the value of a local L1 memory address on
 * the Tensix core executing this function becomes equal to a target value.
 * This L1 memory address is used as a semaphore of size 4 Bytes, as a
 * synchronization mechanism. Also, see *noc_semaphore_set*.
 *
 * Return value: None
 *
 * | Argument  | Description                                                    | Type     | Valid Range        |
 * Required |
 * |-----------|----------------------------------------------------------------|----------|--------------------|----------|
 * | sem_addr  | Semaphore address in local L1 memory                           | uint32_t | 0..1MB             | True |
 * | val       | The target value of the semaphore                              | uint32_t | Any uint32_t value | True |
 * | wait_min  | The number of cycles to wait before performing run_routing()   | uint32_t | Any uint32_t value | False
 * |
 */
FORCE_INLINE
void eth_noc_semaphore_wait(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val, uint32_t wait_min = 0) {
    uint32_t count = 0;
    while ((*sem_addr) != val) {
        invalidate_l1_cache();
        if (count == wait_min) {
            run_routing();
            count = 0;
        } else {
            count++;
        }
    }
}
/**
 * A blocking call that waits until the value of a local L1 memory address on
 * the Tensix core executing this function becomes equal to or greater than a target value.
 * This L1 memory address is used as a semaphore of size 4 Bytes, as a
 * synchronization mechanism. Also, see *noc_semaphore_set*.
 *
 * Return value: None
 *
 * | Argument  | Description                                                    | Type     | Valid Range        |
 * Required |
 * |-----------|----------------------------------------------------------------|----------|--------------------|----------|
 * | sem_addr  | Semaphore address in local L1 memory                           | uint32_t | 0..1MB             | True |
 * | val       | The target value of the semaphore                              | uint32_t | Any uint32_t value | True |
 * | wait_min  | The number of cycles to wait before performing run_routing()   | uint32_t | Any uint32_t value | False
 * |
 */
FORCE_INLINE
void eth_noc_semaphore_wait_min(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val, uint32_t wait_min = 0) {
    uint32_t count = 0;
    while ((*sem_addr) < val) {
        invalidate_l1_cache();
        if (count == wait_min) {
            run_routing();
            count = 0;
        } else {
            count++;
        }
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
        run_routing();
    }
    invalidate_l1_cache();
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
        run_routing();
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
        internal_::eth_send_packet(
            0, ((num_bytes_sent + src_addr) >> 4), ((num_bytes_sent + dst_addr) >> 4), num_bytes_per_send_word_size);
        num_bytes_sent += num_bytes_per_send;
    }
    erisc_info->channels[0].bytes_sent += num_bytes;
}

/**
 * Initiates an asynchronous write from a source address in L1 memory on the local ethernet core to L1 of the connected
 * remote ethernet core. However, this is only the first half of the sender's part of then transaction. It does not
 * include the sending of the write completion signature to the receiver.
 *
 * Non-blocking
 *
 * Return value: None
 *
 * | Argument                    | Description                                             | Type     | Valid Range |
 * Required |
 * |-----------------------------|---------------------------------------------------------|----------|-------------|----------|
 * | src_addr                    | Source address in local eth core L1 memory              | uint32_t | 0..256kB    |
 * True     | | dst_addr                    | Destination address in remote eth core L1 memory        | uint32_t |
 * 0..256kB    | True     | | num_bytes                   | Size of data transfer in bytes, must be multiple of 16  |
 * uint32_t | 0..256kB    | True     | | channel                     | Which transaction channel to use. Corresponds to
 * | uint32_t | 0..7        | True     | |                             | channels in erisc_info_t |          | | | |
 * num_bytes_per_send          | Number of bytes to send per packet                      | uint32_t | 16..1MB     |
 * False    | | num_bytes_per_send_word_size| num_bytes_per_send shifted right 4                      | uint32_t
 * | 1..256kB    | False    |
 */
FORCE_INLINE
void eth_send_bytes_over_channel_payload_only(
    uint32_t src_addr,
    uint32_t dst_addr,
    uint32_t num_bytes,
    uint32_t num_bytes_per_send = 16,
    uint32_t num_bytes_per_send_word_size = 1) {
    // assert(channel < 4);
    uint32_t num_bytes_sent = 0;
    while (num_bytes_sent < num_bytes) {
        internal_::eth_send_packet(
            0, ((num_bytes_sent + src_addr) >> 4), ((num_bytes_sent + dst_addr) >> 4), num_bytes_per_send_word_size);
        num_bytes_sent += num_bytes_per_send;
    }
}

// Calls the unsafe variant of eth_send_packet under the hood which is guaranteed not to context switch
// We want this for code size reasons
FORCE_INLINE
void eth_send_bytes_over_channel_payload_only_unsafe(
    uint32_t src_addr,
    uint32_t dst_addr,
    uint32_t num_bytes,
    uint32_t num_bytes_per_send = 16,
    uint32_t num_bytes_per_send_word_size = 1) {
    uint32_t num_bytes_sent = 0;
    while (num_bytes_sent < num_bytes) {
        internal_::eth_send_packet_unsafe(
            0, ((num_bytes_sent + src_addr) >> 4), ((num_bytes_sent + dst_addr) >> 4), num_bytes_per_send_word_size);
        num_bytes_sent += num_bytes_per_send;
    }
}

FORCE_INLINE
void eth_send_bytes_over_channel_payload_only_unsafe_one_packet(
    uint32_t src_addr, uint32_t dst_addr, uint32_t num_bytes) {
    internal_::eth_send_packet_bytes_unsafe(0, src_addr, dst_addr, num_bytes);
}

/*
 * Sends the write completion signal to the receiver ethernet core, for transfers where the payload was already sent.
 * The second half of a full ethernet send.
 */
FORCE_INLINE
void eth_send_payload_complete_signal_over_channel(uint32_t channel, uint32_t num_bytes) {
    erisc_info->channels[channel].bytes_sent = num_bytes;
    erisc_info->channels[channel].receiver_ack = 0;
    uint32_t addr = ((uint32_t)(&(erisc_info->channels[channel].bytes_sent))) >> 4;
    internal_::eth_send_packet(0, addr, addr, 1);
}

FORCE_INLINE
void eth_send_bytes_over_channel(
    uint32_t src_addr,
    uint32_t dst_addr,
    uint32_t num_bytes,
    uint32_t channel,
    uint32_t num_bytes_per_send = 16,
    uint32_t num_bytes_per_send_word_size = 1) {
    // assert(channel < 4);
    uint32_t num_bytes_sent = 0;
    while (num_bytes_sent < num_bytes) {
        internal_::eth_send_packet(
            0, ((num_bytes_sent + src_addr) >> 4), ((num_bytes_sent + dst_addr) >> 4), num_bytes_per_send_word_size);
        num_bytes_sent += num_bytes_per_send;
    }
    erisc_info->channels[channel].bytes_sent = num_bytes;
    erisc_info->channels[channel].receiver_ack = 0;
    uint32_t addr = ((uint32_t)(&(erisc_info->channels[channel].bytes_sent))) >> 4;
    internal_::eth_send_packet(0, addr, addr, 1);
}

/**
 * Initiates an asynchronous write from the local ethernet core to a register of the connected
 * remote ethernet core.
 *
 * Return value: None
 *
 * | Argument          | Description                                             | Type     | Valid Range | Required |
 * |-------------------|---------------------------------------------------------|----------|-------------|----------|
 * | reg_addr          | Destination address in remote eth core reg space        | uint32_t | 0xFF000000+ | True     |
 * | value             | Value to be written                                     | uint32_t | Any value   | True     |
 */
FORCE_INLINE
void eth_write_remote_reg(uint32_t reg_addr, uint32_t value) { internal_::eth_write_remote_reg(0, reg_addr, value); }

/**
 * A blocking call that waits for receiver to acknowledge that all data sent with eth_send_bytes since the last
 * reset_erisc_info call is no longer being used. Also, see \a eth_receiver_done().
 *
 * Return value: None
 *
 * | Argument   | Description                                                    | Type     | Valid Range        |
 * Required |
 * |------------|----------------------------------------------------------------|----------|--------------------|----------|
 * | wait_min   | The number of cycles to wait before performing run_routing()   | uint32_t | Any uint32_t value | False
 * |
 */
FORCE_INLINE
void eth_wait_for_receiver_done(uint32_t wait_min = 0) {
    internal_::eth_send_packet(
        0,

        ((uint32_t)(&(erisc_info->channels[0].bytes_sent))) >> 4,
        ((uint32_t)(&(erisc_info->channels[0].bytes_sent))) >> 4,
        1);
    uint32_t count = 0;
    while (erisc_info->channels[0].bytes_sent != 0) {
        invalidate_l1_cache();
        if (count == wait_min) {
            count = 0;
            run_routing();
        } else {
            count++;
        }
    }
}

/**
 * Caller is expected to be sender side. Indicates to caller that the receiver has received the last payload sent, and
 * that the local sender buffer can be cleared safely
 *
 * Non-blocking
 *
 * Return value: bool: true if the receiver has acked
 *
 * | Argument                    | Description                                             | Type     | Valid Range |
 * Required |
 * |-----------------------------|---------------------------------------------------------|----------|-------------|----------|
 * | channel                     | Which transaction channel to check. Corresponds to      | uint32_t | 0..7        |
 * True     | |                             | channels in erisc_info_t                                |          | | |
 */
FORCE_INLINE
bool eth_is_receiver_channel_send_acked(uint32_t channel) { return erisc_info->channels[channel].receiver_ack != 0; }

/**
 * Caller is expected to be sender side. Tells caller that the receiver has both received the last payload sent, and
 * also that it has cleared it to its consumers. If true, indicates that caller (sender) send safely send more data.
 *
 * Non-blocking
 *
 * Return value: bool: true if the receiver has acked and forwarded the payload.
 *
 * | Argument                    | Description                                             | Type     | Valid Range |
 * Required |
 * |-----------------------------|---------------------------------------------------------|----------|-------------|----------|
 * | channel                     | Which transaction channel to check. Corresponds to      | uint32_t | 0..7        |
 * True     | |                             | channels in erisc_info_t                                |          | | |
 */
FORCE_INLINE
bool eth_is_receiver_channel_send_done(uint32_t channel) { return erisc_info->channels[channel].bytes_sent == 0; }

/**
 * Caller is expected to be sender side. This call will block until receiver sends both levels of ack
 *
 * Blocking
 *
 * Return value: None
 *
 * | Argument                    | Description                                             | Type     | Valid Range |
 * Required |
 * |-----------------------------|---------------------------------------------------------|----------|-------------|----------|
 * | channel                     | Which transaction channel to block on                   | uint32_t | 0..7        |
 * True     |
 */
FORCE_INLINE
void eth_wait_for_receiver_channel_done(uint32_t channel) {
    uint32_t count = 0;
    uint32_t max = 100000;

    while (!eth_is_receiver_channel_send_done(channel)) {
        invalidate_l1_cache();
        count++;
        if (count > max) {
            count = 0;
            run_routing();
        }
    }
}

/**
 * Caller is expected to be sender side. This call will block until receiver sends both levels of ack
 *
 * Blocking
 *
 * Return value: None
 *
 * | Argument             | Description                                                    | Type     | Valid Range |
 * Required |
 * |----------------------|----------------------------------------------------------------|----------|--------------------|----------|
 * | channel              | Which transaction channel to block on                          | uint32_t | 0..7 | True | |
 * wait_min             | The number of cycles to wait before performing run_routing()   | uint32_t | Any uint32_t value
 * | False    |
 */
FORCE_INLINE
void eth_wait_receiver_done(uint32_t wait_min = 0) {
    uint32_t count = 0;
    while (erisc_info->channels[0].bytes_sent != 0) {
        invalidate_l1_cache();
        if (count == wait_min) {
            count = 0;
            run_routing();
        } else {
            count++;
        }
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
 * | Argument   | Description                                                    | Type     | Valid Range        |
 * Required |
 * |------------|----------------------------------------------------------------|----------|--------------------|----------|
 * | num_bytes  | Size of data transfer in bytes, must be multiple of 16         | uint32_t | 0..256kB           | True
 * | | wait_min   | The number of cycles to wait before performing run_routing()   | uint32_t | Any uint32_t value |
 * False    |
 */
FORCE_INLINE
void eth_wait_for_bytes(uint32_t num_bytes, uint32_t wait_min = 0) {
    uint32_t count = 0;
    while (erisc_info->channels[0].bytes_sent != num_bytes) {
        invalidate_l1_cache();
        if (count == wait_min) {
            count = 0;
            run_routing();
        } else {
            count++;
        }
    }
}

/**
 * Caller is expected to be receiver side. This call will tell the receiver whether or not there is payload data to in
 * the local buffer
 *
 * Non-blocking
 *
 * Return value: bool: True if payload data was sent (and not yet cleared) on the channel
 *
 * | Argument                    | Description                                             | Type     | Valid Range |
 * Required |
 * |-----------------------------|---------------------------------------------------------|----------|-------------|----------|
 * | channel                     | Which transaction channel to check                      | uint32_t | 0..7        |
 * True     |
 */
FORCE_INLINE
bool eth_bytes_are_available_on_channel(uint8_t channel) { return erisc_info->channels[channel].bytes_sent != 0; }

/**
 * Caller is expected to be receiver side. This call block until there is payload data in the local buffer associated
 * with the channel
 *
 * Blocking
 *
 * Return value: None
 *
 * | Argument                    | Description                                             | Type     | Valid Range |
 * Required |
 * |-----------------------------|---------------------------------------------------------|----------|-------------|----------|
 * | num_bytes                   | Number of bytes to receive before returning to caller   | uint32_t | 0..1MB      |
 * True     | | channel                     | Which transaction channel to check                      | uint32_t | 0..7
 * | True     |
 */
FORCE_INLINE
void eth_wait_for_bytes_on_channel_sync_addr(
    uint32_t num_bytes, volatile eth_channel_sync_t* eth_channel_syncs, uint32_t wait_min = 1000000) {
    // assert(channel < 4);
    uint32_t count = 0;
    uint32_t num_bytes_sent = eth_channel_syncs->bytes_sent;
    while (num_bytes_sent != num_bytes) {
        invalidate_l1_cache();
        uint32_t received_this_iter = eth_channel_syncs->bytes_sent;
        if (received_this_iter != num_bytes_sent) {
            // We are currently in the process of receiving data on this channel, so we just just wait a
            // bit longer instead of initiating a context switch
            num_bytes_sent = received_this_iter;
        } else if (count == wait_min) {
            count = 0;
            run_routing();
        } else {
            count++;
        }
    }
}

FORCE_INLINE
void eth_wait_for_bytes_on_channel(uint32_t num_bytes, uint8_t channel, uint32_t wait_min = 1000000) {
    // assert(channel < 4);
    eth_wait_for_bytes_on_channel_sync_addr(num_bytes, &(erisc_info->channels[channel]), wait_min);
}

/**
 * Initiates an asynchronous call from receiver ethernet core to tell remote sender ethernet core that data sent
 * via eth_send_bytes is no longer being used. Also, see \a eth_wait_for_receiver_done. Sends over channel 0
 *
 * Return value: None
 *
 * | Argument                    | Description                                             | Type     | Valid Range |
 * Required |
 * |-----------------------------|---------------------------------------------------------|----------|-------------|----------|
 */
FORCE_INLINE
void eth_receiver_done() {
    erisc_info->channels[0].bytes_sent = 0;
    internal_::eth_send_packet(
        0,
        ((uint32_t)(&(erisc_info->channels[0].bytes_sent))) >> 4,
        ((uint32_t)(&(erisc_info->channels[0].bytes_sent))) >> 4,
        1);
}

/**
 * Caller is expected to be receiver side. This call sends the second (and first) level ack to sender, indicating that
 * the receiver flushed its buffer and is able to accept more data
 *
 * Non-nlocking
 *
 * Return value: None
 *
 * | Argument                    | Description                                             | Type     | Valid Range |
 * Required |
 * |-----------------------------|---------------------------------------------------------|----------|-------------|----------|
 * | channel                     | Which transaction channel to ack                        | uint32_t | 0..7        |
 * True     |
 */

FORCE_INLINE
void send_eth_receiver_channel_done(volatile eth_channel_sync_t* channel_sync) {
    channel_sync->bytes_sent = 0;
    channel_sync->receiver_ack = 0;
    internal_::eth_send_packet(
        0, ((uint32_t)(&(channel_sync->bytes_sent))) >> 4, ((uint32_t)(&(channel_sync->bytes_sent))) >> 4, 1);
}

FORCE_INLINE
void eth_receiver_channel_done(uint32_t channel) {
    // assert(channel < 4);
    send_eth_receiver_channel_done(&(erisc_info->channels[channel]));
}

/**
 * Caller is expected to be sender side. This clears the local first level ack field. Useful when resetting on sender
 * side in preparation for next send
 *
 * Non-blocking
 *
 * Return value: None
 *
 * | Argument                    | Description                                             | Type     | Valid Range |
 * Required |
 * |-----------------------------|---------------------------------------------------------|----------|-------------|----------|
 * | channel                     | Which transaction channel to check                      | uint32_t | 0..7        |
 * True     |
 */
FORCE_INLINE
void eth_clear_sender_channel_ack(uint32_t channel) {
    // assert(channel < 4);
    erisc_info->channels[channel].receiver_ack = 0;
}

/**
 * Caller is expected to be receiver side. This sends the first level ack to sender, indicating that the last payload
 * sent on the channel was received and that sender is free to clear its buffer
 *
 * Non-blocking
 *
 * Return value: None
 *
 * | Argument                      | Description                                             | Type     | Valid Range |
 * Required |
 * |-------------------------------|---------------------------------------------------------|----------|-------------|----------|
 * | channel                       | Which transaction channel to ack                        | uint32_t | 0..7        |
 * True     | | eth_transaction_ack_word_addr | Address of 16B memory (also 16B aligned) segment with   | uint32_t | L1
 * address  | True     | |                               | to send the eth_channel_sync_t to sender for first level|
 * uint32_t | L1 address  | True     | |                               | ack. Must *not* alias
 * erisc_info->channels[channel]     | uint32_t | L1 address  | True     |
 */
FORCE_INLINE
void eth_receiver_channel_ack(uint32_t channel, uint32_t eth_transaction_ack_word_addr) {
    // assert(channel < 4);
    erisc_info->channels[channel].receiver_ack = 1;
    ASSERT(reinterpret_cast<volatile uint32_t*>(eth_transaction_ack_word_addr)[0] == 1);
    reinterpret_cast<volatile uint32_t*>(eth_transaction_ack_word_addr)[1] = 1;
    // Make sure we don't alias the erisc_info eth_channel_sync_t
    ASSERT(eth_transaction_ack_word_addr != ((uint32_t)(&(erisc_info->channels[channel].receiver_ack))) >> 4);
    internal_::eth_send_packet(
        0, eth_transaction_ack_word_addr >> 4, ((uint32_t)(&(erisc_info->channels[channel].receiver_ack))) >> 4, 1);
}

/*
 * Initiates an asynchronous call from receiver ethernet core to tell remote sender ethernet core that data sent
 * via eth_send_bytes has been received. Also, see \a eth_wait_for_receiver_done
 *
 * Return value: None
 *
 * | Argument          | Description                                             | Type     | Valid Range | Required |
 * |-------------------|---------------------------------------------------------|----------|-------------|----------|
 */
FORCE_INLINE
void eth_receiver_acknowledge(uint8_t channel = 0) {
    erisc_info->channels[channel].bytes_sent = 1;
    internal_::eth_send_packet(
        0,
        ((uint32_t)(&(erisc_info->channels[channel].bytes_sent))) >> 4,
        ((uint32_t)(&(erisc_info->channels[channel].bytes_sent))) >> 4,
        1);
}

FORCE_INLINE
void eth_wait_receiver_acknowledge(uint8_t channel = 0) {
    while (erisc_info->channels[channel].bytes_sent != 1) {
        invalidate_l1_cache();
        run_routing();
    }
}
