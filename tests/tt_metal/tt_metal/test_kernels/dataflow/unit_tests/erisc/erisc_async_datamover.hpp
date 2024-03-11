// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <type_traits>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/assert.h"
#include "eth_l1_address_map.h"
#include "ethernet/dataflow_api.h"
#include "tt_metal/hw/inc/wormhole/noc/noc.h"

namespace erisc {
namespace datamover {

/*
 * Worker coordinate, used by EDM to know which workers to signal
 */
struct WorkerXY {
    uint16_t x;
    uint16_t y;
};

/*
 * The `ChannelBuffer` is a building block of the Erisc Data Mover (EDM). For every concurrent transaction
 * channel managed by the EDM, there is a `ChannelBuffer` associated with the. The `ChannelBuffer` manages
 * state for the transaction channel, holds information such as buffer and semaphore addresses, and has helper
 * functions to more easily check semaphore and ack statuses and to send/receive data and/or semaphore updates.
 */
class ChannelBuffer final {
   public:
    enum STATE : uint8_t {
        DONE = 0,

        // For sender: means we are ready to tell the worker(s) that the buffer is available for writing into
        //
        SIGNALING_WORKER,

        // For sender: we are waiting for the payload to arrive in L1; we are checking local semaphore for worker
        // completion For receiver: we are waiting for worker to complete pull of payload from L1; we are checking local
        // semaphore for worker completion
        WAITING_FOR_WORKER,

        // For sender: means workers have signalled (via semaphores) that the buffer payload is
        //             ready in L1
        // For receiver:
        READY_FOR_ETH_TRANSFER,

        // For sender: means we are waiting for ack from receiver that payload was received
        // For receiver: means we are waitinf for a payload from sender
        WAITING_FOR_ETH,
    };

    // for default initialization in arrays
    ChannelBuffer() :
        local_semaphore_address(0),
        worker_coords(0),
        address(0),
        size_in_bytes(0),
        worker_semaphore_l1_address(0),
        num_workers(0),
        num_messages_moved(0),
        channel_bytes_sent_address(0),
        channel_bytes_acked_address(0),
        total_num_messages_to_move(0),
        state(STATE::DONE) {}

    ChannelBuffer(
        uint32_t eth_transaction_channel,
        size_t address,
        size_t size_in_bytes,
        uint32_t worker_semaphore_l1_address,
        uint32_t num_workers,
        uint32_t total_num_messages_to_move,
        volatile tt_l1_ptr uint32_t *const local_semaphore_address,
        tt_l1_ptr const WorkerXY *worker_coords,
        bool is_sender_side) :
        eth_transaction_channel(eth_transaction_channel),
        local_semaphore_address(local_semaphore_address),
        worker_coords(worker_coords),
        address(address),
        size_in_bytes(size_in_bytes),
        worker_semaphore_l1_address(worker_semaphore_l1_address),
        num_workers(num_workers),
        num_messages_moved(0),
        channel_bytes_sent_address(&erisc_info->channels[eth_transaction_channel].bytes_sent),
        channel_bytes_acked_address(&erisc_info->channels[eth_transaction_channel].receiver_ack),
        total_num_messages_to_move(total_num_messages_to_move),
        state(is_sender_side ? STATE::WAITING_FOR_WORKER : STATE::WAITING_FOR_ETH),
        is_sender_completion_pending(false) {

        clear_local_semaphore();

        if (total_num_messages_to_move != 0) {
            if (is_sender_side) {
                // Tell the sender side workers that we're ready to accept data on this channel
                increment_worker_semaphores();
            }
        } else {
            goto_state(STATE::DONE);
        }
    };

    // Resets the semaphore in local L1, which workers write to remotely.
    FORCE_INLINE void clear_local_semaphore() { noc_semaphore_set(local_semaphore_address, 0); }

    // Increment the semaphore in the remote L1s of every worker associated with this ChannelBuffer
    FORCE_INLINE void increment_worker_semaphores() {
        // We have to be careful that the worker x/y matches for the `noc_index`
        // active on the erisc
        for (std::size_t i = 0; i < this->num_workers; i++) {
            WorkerXY worker_xy = this->worker_coords[i];
            uint64_t worker_semaphore_address =
                get_noc_addr((uint32_t)worker_xy.x, (uint32_t)worker_xy.y, this->worker_semaphore_l1_address);

            // TODO: Check for (!noc_cmd_buf_ready(noc, cmd_buf)) and exit early if not ready. Try to do other processing if
            //       possible and come back to this at the next opportunity
            //       So far no solid use case as it's always single worker per channel
            noc_semaphore_inc(worker_semaphore_address, 1);
        }
    }

    [[nodiscard]] FORCE_INLINE bool is_local_semaphore_full() const {
        return *(this->local_semaphore_address) == this->num_workers;
    }

    [[nodiscard]] FORCE_INLINE bool is_active() const {
        return this->num_messages_moved < this->total_num_messages_to_move;
    }

    [[nodiscard]] STATE get_state() const { return this->state; }

    FORCE_INLINE void goto_state(STATE s) { this->state = s; }
    [[nodiscard]] FORCE_INLINE bool is_waiting_for_workers_core() const {
        return this->state == STATE::WAITING_FOR_WORKER;
    }
    [[nodiscard]] FORCE_INLINE bool is_ready_to_signal_workers() const {
        return this->state == STATE::SIGNALING_WORKER;
    }
    [[nodiscard]] FORCE_INLINE bool is_waiting_for_remote_eth_core() const {
        return this->state == STATE::WAITING_FOR_ETH;
    }
    [[nodiscard]] FORCE_INLINE bool is_ready_for_eth_transfer() const {
        return this->state == STATE::READY_FOR_ETH_TRANSFER;
    }
    [[nodiscard]] FORCE_INLINE bool is_done() const { return this->state == STATE::DONE; }

    [[nodiscard]] FORCE_INLINE uint32_t get_eth_transaction_channel() const {
        ASSERT(this->eth_transaction_channel < eth_l1_mem::address_map::MAX_NUM_CONCURRENT_TRANSACTIONS);
        return this->eth_transaction_channel;
    }
    [[nodiscard]] FORCE_INLINE std::size_t get_remote_eth_buffer_address() const {
        return this->address;
    }
    [[nodiscard]] FORCE_INLINE std::size_t get_size_in_bytes() const { return this->size_in_bytes; }
    [[nodiscard]] FORCE_INLINE std::size_t get_current_payload_size() const { return this->get_size_in_bytes(); }

    [[nodiscard]] FORCE_INLINE std::size_t get_buffer_address() const { return this->address; }

    FORCE_INLINE void increment_messages_moved() { this->num_messages_moved++; }

    [[nodiscard]] FORCE_INLINE bool all_messages_moved() {
        return this->num_messages_moved == this->total_num_messages_to_move;
    }

    FORCE_INLINE void set_send_completion_pending(bool value) { this->is_sender_completion_pending = value; }
    [[nodiscard]] FORCE_INLINE bool is_send_completion_pending() const { return this->is_sender_completion_pending; }

    FORCE_INLINE bool eth_is_receiver_channel_send_done() const { return *this->channel_bytes_sent_address == 0;}
    FORCE_INLINE bool eth_bytes_are_available_on_channel() const { return *this->channel_bytes_sent_address != 0;}
    FORCE_INLINE bool eth_is_receiver_channel_send_acked() const { return *this->channel_bytes_acked_address != 0;}
    volatile tt_l1_ptr uint32_t *const get_channel_bytes_sent_address() { return this->channel_bytes_sent_address; }
    volatile tt_l1_ptr uint32_t *const get_channel_bytes_acked_address() { return this->channel_bytes_acked_address; }

   public:
    uint32_t eth_transaction_channel;  //
    volatile tt_l1_ptr uint32_t *const local_semaphore_address;
    WorkerXY const *const worker_coords;
    std::size_t const address;
    std::size_t const size_in_bytes;
    // Even for multiple workers, this address will be the same
    std::size_t const worker_semaphore_l1_address;
    uint32_t const num_workers;
    uint32_t num_messages_moved;
    volatile tt_l1_ptr uint32_t *const channel_bytes_sent_address;
    volatile tt_l1_ptr uint32_t *const channel_bytes_acked_address;
    const uint32_t total_num_messages_to_move;
    STATE state;
    bool is_sender_completion_pending;
};

template <typename T = uint8_t>
class QueueIndexPointer {
   public:
    QueueIndexPointer(uint8_t queue_size) : ptr(0), size(queue_size), wrap_around(queue_size * 2) {
        // FWASSERT(queue_size < 128);
    }

    [[nodiscard("index was called without consuming the result. Did you mean to call it?")]] T index() const {
        return this->ptr >= this->size ? this->ptr - this->size : this->ptr;
    }
    [[nodiscard("raw_index was called without consuming the result. Did you mean to call it?")]] inline T raw_index()
        const {
        return this->ptr;
    }
    [[nodiscard("distance was called without consuming the result. Did you mean to call it?")]] inline static T
    distance(QueueIndexPointer ptr, QueueIndexPointer ackptr) {
        // FWASSERT(ptr.size == ackptr.size);
        return ackptr.ptr > ptr.ptr ? (ptr.wrap_around - ackptr.ptr) + ptr.ptr : ptr.ptr - ackptr.ptr;
    }
    [[nodiscard("full was called without consuming the result. Did you mean to call it?")]] inline static T full(
        QueueIndexPointer ptr, QueueIndexPointer ackptr) {
        // FWASSERT(ptr.size == ackptr.size);
        return distance(ptr.ptr, ackptr.ptr) >= ptr.size;
    }
    [[nodiscard("empty was called without consuming the result. Did you mean to call it?")]] inline static T empty(
        QueueIndexPointer ptr, QueueIndexPointer ackptr) {
        // FWASSERT(ptr.size == ackptr.size);
        return ptr.ptr == ackptr.ptr;
    }
    inline void increment() { this->ptr = this->next_pointer(); }
    [[nodiscard(
        "next_index was called without consuming the result. Did you mean to call it?")]] inline QueueIndexPointer
    next_index() const {
        return QueueIndexPointer(this->next_pointer(), this->size);
    }
    // Compares indices since the raw index is not visible to the user
    inline bool operator==(const QueueIndexPointer &other) const { return this->ptr == other.ptr; }
    inline bool operator!=(const QueueIndexPointer &other) const { return this->ptr != other.ptr; }

   private:
    inline T next_pointer() {
        T next_ptr = (this->ptr + 1);
        next_ptr = next_ptr == wrap_around ? 0 : next_ptr;
        return next_ptr;
    }
    QueueIndexPointer(T ptr, uint8_t queue_size) : ptr(ptr), size(queue_size), wrap_around(queue_size * 2) {}
    T ptr;
    uint8_t size;
    uint8_t wrap_around;
};

FORCE_INLINE void eth_setup_handshake(std::uint32_t handshake_register_address, bool is_sender) {
    if (is_sender) {
        eth_send_bytes(handshake_register_address, handshake_register_address, 16);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_channel_done(0);
    }
}

template <uint32_t NUM_CHANNELS>
FORCE_INLINE void initialize_transaction_buffer_addresses(
    uint32_t max_concurrent_transactions,
    uint32_t first_buffer_base_address,
    uint32_t num_bytes_per_send,
    std::array<uint32_t, NUM_CHANNELS> &transaction_channel_buffer_addresses) {
    uint32_t buffer_address = first_buffer_base_address;
    for (uint32_t i = 0; i < max_concurrent_transactions; i++) {
        transaction_channel_buffer_addresses[i] = buffer_address;
        buffer_address += num_bytes_per_send;
    }
}

/////////////////////////////////////////////
//   SENDER SIDE HELPERS
/////////////////////////////////////////////

FORCE_INLINE bool sender_eth_send_data_sequence(ChannelBuffer &sender_buffer_channel) {
    bool did_something = false;
    if (sender_buffer_channel.eth_is_receiver_channel_send_done()) {
        bool need_to_send_completion = sender_buffer_channel.is_send_completion_pending();
        if (!sender_buffer_channel.is_send_completion_pending() && !eth_txq_is_busy()) {
            static constexpr std::size_t ETH_BYTES_TO_WORDS_SHIFT = 4;
            eth_send_bytes_over_channel_payload_only(
                sender_buffer_channel.get_buffer_address(),
                sender_buffer_channel.get_remote_eth_buffer_address(),
                sender_buffer_channel.get_current_payload_size(),
                sender_buffer_channel.get_eth_transaction_channel(),
                sender_buffer_channel.get_current_payload_size(),
                sender_buffer_channel.get_current_payload_size() >> ETH_BYTES_TO_WORDS_SHIFT);

            sender_buffer_channel.set_send_completion_pending(true);
            need_to_send_completion = true;
            did_something = true;
        }

        if (need_to_send_completion && !eth_txq_is_busy()) {
            eth_send_payload_complete_signal_over_channel(sender_buffer_channel.get_eth_transaction_channel(), sender_buffer_channel.get_current_payload_size());
            sender_buffer_channel.set_send_completion_pending(false);
            sender_buffer_channel.goto_state(ChannelBuffer::WAITING_FOR_ETH);
            did_something = true;
        }
    }

    return did_something;
}

FORCE_INLINE bool sender_notify_workers_if_buffer_available_sequence(
    ChannelBuffer &sender_buffer_channel, uint32_t &num_senders_complete) {

    sender_buffer_channel.increment_worker_semaphores();

    if (!sender_buffer_channel.all_messages_moved()) {
        sender_buffer_channel.goto_state(ChannelBuffer::WAITING_FOR_WORKER);
    } else {
        sender_buffer_channel.goto_state(ChannelBuffer::DONE);
        num_senders_complete++;
    }

    return true;
}

FORCE_INLINE bool sender_eth_check_receiver_ack_sequence(ChannelBuffer &sender_buffer_channel, uint32_t &num_senders_complete) {
    bool did_something = false;

    bool transimission_acked_by_receiver =
        sender_buffer_channel.eth_is_receiver_channel_send_acked() ||
        sender_buffer_channel.eth_is_receiver_channel_send_done();
    if (transimission_acked_by_receiver) {
        eth_clear_sender_channel_ack(sender_buffer_channel.get_eth_transaction_channel());
        sender_buffer_channel.increment_messages_moved();
        sender_buffer_channel.goto_state(ChannelBuffer::SIGNALING_WORKER);
        sender_notify_workers_if_buffer_available_sequence(sender_buffer_channel, num_senders_complete);
        did_something = true;
    }

    return did_something;
}

/*
 *
 */
FORCE_INLINE bool sender_noc_receive_payload_ack_check_sequence(ChannelBuffer &sender_channel_buffer) {
    bool did_something = false;

    bool read_finished = sender_channel_buffer.is_local_semaphore_full();
    if (read_finished) {
        // We can clear the semaphore, and wait for space on receiver
        sender_channel_buffer.clear_local_semaphore();
        sender_channel_buffer.goto_state(ChannelBuffer::READY_FOR_ETH_TRANSFER);
        did_something = true;

        erisc::datamover::sender_eth_send_data_sequence(sender_channel_buffer);
    }

    return did_something;
}

/////////////////////////////////////////////
//   RECEIVER SIDE HELPERS
/////////////////////////////////////////////


/*
 *
 */
FORCE_INLINE bool receiver_eth_notify_workers_payload_available_sequence(ChannelBuffer &buffer_channel) {
    buffer_channel.increment_worker_semaphores();
    buffer_channel.goto_state(ChannelBuffer::WAITING_FOR_WORKER);

    return true;
}



/*
 * If payload received, notify (send ack to) sender so sender knows it can free up its local buffer
 *
 */
FORCE_INLINE bool receiver_eth_accept_payload_sequence(ChannelBuffer &buffer_channel) {
    bool did_something = false;

    if (buffer_channel.eth_bytes_are_available_on_channel()) {
        if (!eth_txq_is_busy()) {
            eth_receiver_channel_ack(buffer_channel.get_eth_transaction_channel());
            buffer_channel.goto_state(ChannelBuffer::SIGNALING_WORKER);
            did_something = true;

            // FIXME: Decouple these so we can still signal workers even if eth command queue is busy
            //        Prefer sending eth ack first, but notify workers even if we have to come back to
            //        send the eth ack later
            receiver_eth_notify_workers_payload_available_sequence(buffer_channel);
        }
    }

    return did_something;
}

/*
 * Does something if we are waiting for workers to complete their read and the read is complete:
 * - increment messages moved (that transfer is done)
 * - notifies sender it is safe to send next payload
 * - clear local semaphore
 */
FORCE_INLINE bool receiver_noc_read_worker_completion_check_sequence(
    ChannelBuffer &buffer_channel, uint32_t &num_receivers_complete) {
    bool did_something = false;

    bool workers_are_finished_reading = buffer_channel.is_local_semaphore_full();
    bool can_notify_sender_of_buffer_available = workers_are_finished_reading;
    if (can_notify_sender_of_buffer_available) {

        if (!eth_txq_is_busy()) {
            eth_receiver_channel_done(buffer_channel.get_eth_transaction_channel());
            buffer_channel.increment_messages_moved();
            buffer_channel.clear_local_semaphore();

            if (!buffer_channel.all_messages_moved()) {
                buffer_channel.goto_state(ChannelBuffer::WAITING_FOR_ETH);
            } else {
                buffer_channel.goto_state(ChannelBuffer::DONE);
                num_receivers_complete++;
            }

            did_something = true;
        }

    }

    return did_something;
}

////////////////////////////
//  DEPRECATED
////////////////////////////
namespace deprecated {
// This namespace exists to support non-decoupled mode microbenchmarks until those are available
// in decoupled mode

FORCE_INLINE bool sender_buffer_pool_full(
    const QueueIndexPointer<uint8_t> noc_reader_buffer_wrptr,
    const QueueIndexPointer<uint8_t> noc_reader_buffer_ackptr,
    const QueueIndexPointer<uint8_t> eth_sender_rdptr,
    const QueueIndexPointer<uint8_t> eth_sender_ackptr) {
    return QueueIndexPointer<uint8_t>::full(noc_reader_buffer_wrptr, eth_sender_ackptr);
}

FORCE_INLINE bool sender_buffer_pool_empty(
    const QueueIndexPointer<uint8_t> noc_reader_buffer_wrptr,
    const QueueIndexPointer<uint8_t> noc_reader_buffer_ackptr,
    const QueueIndexPointer<uint8_t> eth_sender_rdptr,
    const QueueIndexPointer<uint8_t> eth_sender_ackptr) {
    return QueueIndexPointer<uint8_t>::empty(eth_sender_rdptr, noc_reader_buffer_wrptr);
}

FORCE_INLINE bool sender_buffer_available_for_eth_send(
    const QueueIndexPointer<uint8_t> noc_reader_buffer_wrptr,
    const QueueIndexPointer<uint8_t> noc_reader_buffer_ackptr,
    const QueueIndexPointer<uint8_t> eth_sender_rdptr,
    const QueueIndexPointer<uint8_t> eth_sender_ackptr) {
    return eth_sender_rdptr != noc_reader_buffer_ackptr;
}

template <uint32_t MAX_NUM_CHANNELS>
FORCE_INLINE bool sender_eth_send_data_sequence(
    std::array<uint32_t, MAX_NUM_CHANNELS> &transaction_channel_sender_buffer_addresses,
    std::array<uint32_t, MAX_NUM_CHANNELS> &transaction_channel_receiver_buffer_addresses,
    uint32_t local_eth_l1_src_addr,
    uint32_t remote_eth_l1_dst_addr,
    uint32_t num_bytes,
    uint32_t num_bytes_per_send,
    uint32_t num_bytes_per_send_word_size,
    QueueIndexPointer<uint8_t> noc_reader_buffer_wrptr,
    QueueIndexPointer<uint8_t> noc_reader_buffer_ackptr,
    QueueIndexPointer<uint8_t> &eth_sender_rdptr,
    QueueIndexPointer<uint8_t> &eth_sender_ackptr) {
    bool did_something = false;
    bool data_ready_for_send = sender_buffer_available_for_eth_send(
        noc_reader_buffer_wrptr, noc_reader_buffer_ackptr, eth_sender_rdptr, eth_sender_ackptr);
    if (data_ready_for_send) {
        bool consumer_ready_to_accept = eth_is_receiver_channel_send_done(eth_sender_rdptr.index());
        if (consumer_ready_to_accept) {
            // kernel_profiler::mark_time(14);
            // Queue up another send
            uint32_t sender_buffer_address = transaction_channel_sender_buffer_addresses[eth_sender_rdptr.index()];
            uint32_t receiver_buffer_address = transaction_channel_receiver_buffer_addresses[eth_sender_rdptr.index()];

            eth_send_bytes_over_channel(
                sender_buffer_address,
                receiver_buffer_address,
                num_bytes,
                eth_sender_rdptr.index(),
                num_bytes_per_send,
                num_bytes_per_send_word_size);
            eth_sender_rdptr.increment();
            did_something = true;
        }
    }

    return did_something;
}

FORCE_INLINE bool sender_eth_check_receiver_ack_sequence(
    const QueueIndexPointer<uint8_t> noc_reader_buffer_wrptr,
    const QueueIndexPointer<uint8_t> noc_reader_buffer_ackptr,
    QueueIndexPointer<uint8_t> &eth_sender_rdptr,
    QueueIndexPointer<uint8_t> &eth_sender_ackptr,
    uint32_t &num_eth_sends_acked) {
    bool did_something = false;
    bool eth_sends_unacknowledged = QueueIndexPointer<uint8_t>::distance(eth_sender_rdptr, eth_sender_ackptr) > 0;
    if (eth_sends_unacknowledged) {
        bool transimission_acked_by_receiver = eth_is_receiver_channel_send_acked(eth_sender_ackptr.index()) ||
                                               eth_is_receiver_channel_send_done(eth_sender_ackptr.index());
        if (transimission_acked_by_receiver) {
            num_eth_sends_acked++;
            eth_sender_ackptr.increment();

            did_something = true;
        }
    }

    return did_something;
}

FORCE_INLINE bool sender_is_noc_read_in_progress(
    const QueueIndexPointer<uint8_t> noc_reader_buffer_wrptr,
    const QueueIndexPointer<uint8_t> noc_reader_buffer_ackptr) {
    return noc_reader_buffer_wrptr != noc_reader_buffer_ackptr;
}

FORCE_INLINE bool sender_noc_receive_payload_ack_check_sequence(
    QueueIndexPointer<uint8_t> &noc_reader_buffer_wrptr,
    QueueIndexPointer<uint8_t> &noc_reader_buffer_ackptr,
    const uint8_t noc_index) {
    bool did_something = false;

    bool noc_read_is_in_progress = sender_is_noc_read_in_progress(noc_reader_buffer_wrptr, noc_reader_buffer_ackptr);
    if (noc_read_is_in_progress) {
#if EMULATE_DRAM_READ_CYCLES == 1
        bool read_finished = emulated_dram_read_cycles_finished();
#else
        bool read_finished = ncrisc_noc_reads_flushed(noc_index);
#endif
        if (read_finished) {
            noc_reader_buffer_ackptr.increment();
            did_something = true;
        }
    }

    return did_something;
}

/////////////////////////////////////////////
//   RECEIVER SIDE HELPERS
/////////////////////////////////////////////

FORCE_INLINE bool receiver_is_noc_write_in_progress(
    const QueueIndexPointer<uint8_t> noc_writer_buffer_wrptr,
    const QueueIndexPointer<uint8_t> noc_writer_buffer_ackptr) {
    return noc_writer_buffer_wrptr != noc_writer_buffer_ackptr;
}

bool receiver_eth_accept_payload_sequence(
    QueueIndexPointer<uint8_t> noc_writer_buffer_wrptr,
    QueueIndexPointer<uint8_t> noc_writer_buffer_ackptr,
    QueueIndexPointer<uint8_t> &eth_receiver_ptr,
    QueueIndexPointer<uint8_t> &eth_receiver_ackptr) {
    bool did_something = false;
    bool receive_pointers_full = QueueIndexPointer<uint8_t>::full(eth_receiver_ptr, eth_receiver_ackptr);

    if (!receive_pointers_full) {
        if (eth_bytes_are_available_on_channel(eth_receiver_ptr.index())) {
            // DPRINT << "rx: accepting payload, sending receive ack on channel " << (uint32_t)eth_receiver_ptr << "\n";
            eth_receiver_channel_ack(eth_receiver_ptr.index());
            eth_receiver_ptr.increment();
            did_something = true;
        }
    }

    return did_something;
}

FORCE_INLINE bool receiver_noc_read_worker_completion_check_sequence(
    QueueIndexPointer<uint8_t> &noc_writer_buffer_wrptr,
    QueueIndexPointer<uint8_t> &noc_writer_buffer_ackptr,
    const uint8_t noc_index) {
    bool did_something = false;

    bool noc_write_is_in_progress =
        receiver_is_noc_write_in_progress(noc_writer_buffer_wrptr, noc_writer_buffer_ackptr);
    if (noc_write_is_in_progress) {
#if EMULATE_DRAM_READ_CYCLES == 1
        bool write_finished = emulated_dram_write_cycles_finished();
#else
        bool writes_finished = ncrisc_noc_nonposted_writes_sent(noc_index);
#endif
        if (writes_finished) {
            // DPRINT << "rx: accepting payload, sending receive ack on channel " << (uint32_t)noc_writer_buffer_ackptr
            // << "\n";
            noc_writer_buffer_ackptr.increment();

            did_something = true;
        }
    }

    return did_something;
}

FORCE_INLINE bool receiver_eth_send_ack_to_sender_sequence(
    const QueueIndexPointer<uint8_t> noc_writer_buffer_wrptr,
    const QueueIndexPointer<uint8_t> noc_writer_buffer_ackptr,
    QueueIndexPointer<uint8_t> &eth_receiver_rdptr,
    QueueIndexPointer<uint8_t> &eth_receiver_ackptr,
    uint32_t &num_eth_sends_acked) {
    bool did_something = false;
    bool eth_sends_unacknowledged = eth_receiver_rdptr != eth_receiver_ackptr;
    if (eth_sends_unacknowledged) {
        // If data is done being sent out of this local l1 buffer and to the destination(s),
        // then we can safely send the ack and increment the ackptr
        bool buffer_writes_flushed = ncrisc_noc_nonposted_writes_sent(noc_index);
        // bool buffer_writes_flushed = ncrisc_noc_nonposted_writes_flushed(noc_index);
        if (buffer_writes_flushed) {
            // kernel_profiler::mark_time(15);
            // DPRINT << "rx: accepting payload, sending receive ack on channel " << (uint32_t)noc_writer_buffer_wrptr
            // << "\n";
            eth_receiver_channel_done(eth_receiver_ackptr.index());
            num_eth_sends_acked++;
            eth_receiver_ackptr.increment();
            // DPRINT << "rx: Sending eth ack. ackptr incrementing to " << (uint32_t)eth_receiver_ackptr.index() <<
            // "\n";

            did_something = true;
        }
    }

    return did_something;
}

};  // namespace deprecated

};  // namespace datamover
};  // namespace erisc
