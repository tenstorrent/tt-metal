#pragma once

#include <algorithm>
#include <utility>
#include <array>
#include <cstddef>

#include "debug/assert.h"

#include "packet_queue_ctrl.hpp"
#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"
#include "tt_metal/impl/dispatch/kernels/packet_queue_remotes.hpp"

namespace packet_queue {

static constexpr uint32_t k_MaxInputOutputQueues = 10;
static_assert(k_MaxInputOutputQueues >= MAX_SWITCH_FAN_IN);
static_assert(k_MaxInputOutputQueues >= MAX_SWITCH_FAN_OUT);
static_assert(k_MaxInputOutputQueues >= MAX_TUNNEL_LANES);

/*
**********************
*                    *
*  Functions         *
*                    *
**********************
*/

// Increment a remote semaphore
inline void increment_remote_sem_val(uint32_t remote_x, uint32_t remote_y, uint32_t sem_id, uint32_t val) {
#ifdef FD_CORE_TYPE
    const uint32_t sem_l1_addr = get_semaphore<fd_core_type>(sem_id);
#else
    const uint32_t sem_l1_addr = 0;  // watcher assertion
#endif
    const uint64_t sem_noc_addr = get_noc_addr(remote_x, remote_y, sem_l1_addr);
    noc_semaphore_inc(sem_noc_addr, val);
}

// Increment a local semaphore
inline void increment_local_sem_val(uint32_t sem_id, uint32_t val) {
#ifdef FD_CORE_TYPE
    const uint32_t sem_l1_addr = get_semaphore<fd_core_type>(sem_id);
#else
    const uint32_t sem_l1_addr = 0;  // watcher assertion
#endif
    uint64_t sem_noc_addr = get_noc_addr(sem_l1_addr);
    noc_semaphore_inc(sem_noc_addr, val);
    noc_async_atomic_barrier();
}

// Get local semaphore value
inline uint32_t get_local_sem_val(uint32_t sem_id, bool& underflow) {
    invalidate_l1_cache();
#ifdef FD_CORE_TYPE
    volatile tt_l1_ptr uint32_t* local_sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<fd_core_type>(sem_id));
#else
    volatile tt_l1_ptr uint32_t* local_sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(0);
#endif
    // semaphore underflow is currently used to signal path teardown with minimal prefetcher changes
    uint32_t val = *local_sem_addr;
    if (val & 0x80000000) {
        val &= 0x7fffffff;
        underflow = true;
    }
    return val;
}

void zero_l1_buf(tt_l1_ptr uint32_t* buf, uint32_t size_bytes) {
    for (uint32_t i = 0; i < size_bytes / 4; i++) {
        buf[i] = 0;
    }
}

void write_test_results(tt_l1_ptr uint32_t* const buf, uint32_t i, uint32_t val) {
    if (buf != nullptr) {
        buf[i] = val;
    }
}

void write_kernel_status(tt_l1_ptr uint32_t* const buf, uint32_t i, uint32_t val) {
    if (buf != nullptr) {
        buf[i] = val;
    }
}

void set_64b_result(uint32_t* buf, uint64_t val, uint32_t index = 0) {
    if (buf != nullptr) {
        buf[index] = val >> 32;
        buf[index + 1] = val & 0xFFFFFFFF;
    }
}

// Get 64 bit riscv timestamp
inline uint64_t get_timestamp() {
    uint32_t timestamp_low = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t timestamp_high = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
    return (((uint64_t)timestamp_high) << 32) | timestamp_low;
}

// Get lower 32 bits of riscv timestamp
inline uint64_t get_timestamp_32b() { return reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L); }

// Helper function to call the lambda N times, passing in the template arguments and index
template <typename NetworkTypeSequence, typename CBModeSequence, typename F, size_t... Index>
bool process_queues(F&& func, std::index_sequence<Index...>) {
    static_assert(NetworkTypeSequence::size == CBModeSequence::size);
    bool result = true;
    (([&]() {
        if constexpr (NetworkTypeSequence::values[Index] == DispatchRemoteNetworkType::SKIP) {
            return true;
        } else {
            return std::forward<F>(func).template 
                operator()<NetworkTypeSequence::values[Index], CBModeSequence::values[Index], Index>(Index);
        }
    }() && (result = true)), ...);
    return result;
}

// F is a lambda that will be called over each queue from index 0 to NetworkTypeSequence::size
// If any function returns false it will stop.
template <typename NetworkTypeSequence, typename CBModeSequence, typename F>
bool process_queues(F&& func) {
    static_assert(NetworkTypeSequence::size == CBModeSequence::size);
    return process_queues<NetworkTypeSequence, CBModeSequence>(
        std::forward<F>(func), std::make_index_sequence<NetworkTypeSequence::size>());
}

/*
****************************
*                          *
*  Classes/Structs/Storage *
*                          *
****************************
*/

// CB Mode configuration info
struct cb_mode_config_t {
    uint32_t page_size_words;  // Must be a power of 2
    uint32_t page_size_mask;
    uint8_t local_sem_id;
    uint8_t remote_sem_id;

    // Initialize cb mode config options
    void init(uint32_t log_page_size, uint8_t local_sem_id, uint8_t remote_sem_id) {
        this->page_size_words = (((uint32_t)0x1) << log_page_size) / PACKET_WORD_SIZE_BYTES;
        this->page_size_mask = this->page_size_words - 1;
        this->local_sem_id = local_sem_id;
        this->remote_sem_id = remote_sem_id;
    }
};

// Dummy placeholder queue
class packet_queue_nop {};

// Base for all packet queues.
// T is the implementation. R is the remote controller for pointer updates.
template <typename T, typename R>
class packet_queue_base_t;

// Base for all input packet queues.
// T is the implementation. R is the remote controller for pointer updates.
template <typename T, typename R>
class packet_queue_input_t;

// Copies data to local L1 from a remote queue.
// R is the specialized remote control type.
template <typename R>
class input_queue_impl_t;

// Copies paged data to local L1 from a remote queue.
// R is the specialized remote control type.
template <typename R>
class input_queue_cb_mode_impl_t;

// Base for all output packet queues.
// T is the implementation. R is the remote controller for pointer updates.
template <typename T, typename R, typename InputNetworkTypeSequence, typename InputCBSequence>
class packet_queue_output_t;

// Writes paged data to a remote queue.
// R is the specialized remote control type.
template <typename R, typename InputNetworkTypeSequence, typename InputCBSequence>
class output_queue_cb_mode_impl_t;

// Writes data to a remote queue.
// R is the specialized remote control type.
template <typename R, typename InputNetworkTypeSequence, typename InputCBSequence>
class output_queue_impl_t;

template <DispatchRemoteNetworkType NetworkType>
struct select_network {
    using type = std::conditional_t<
        NetworkType == DispatchRemoteNetworkType::NOC0,
        packet_queue_remote_noc0_impl,
        std::conditional_t<
            NetworkType == DispatchRemoteNetworkType::ETH,
            packet_queue_remote_eth_impl,
            packet_queue_remote_nop_impl>>;
};

// Provides a concrete remote network implementation given a NetworkType
template <DispatchRemoteNetworkType NetworkType>
using select_network_t = typename select_network<NetworkType>::type;

template <DispatchRemoteNetworkType NetworkType, bool CBMode>
struct select_input_queue_type {
    using type = std::conditional_t<
        NetworkType == DispatchRemoteNetworkType::SKIP,
        packet_queue_nop,
        std::conditional_t<
            CBMode == true,
            input_queue_cb_mode_impl_t<select_network_t<NetworkType>>,
            input_queue_impl_t<select_network_t<NetworkType>>
            >
        >;
};

// Provides a concrete input queue implementation based on the network type and CB mode.
template <DispatchRemoteNetworkType NetworkType, bool CBMode>
using select_input_queue_t = typename select_input_queue_type<NetworkType, CBMode>::type;

template <
    DispatchRemoteNetworkType NetworkType,
    bool CBMode,
    typename InputNetworkTypeSequence,
    typename InputCBSequence>
struct select_output_queue {
    using type = std::conditional_t<
        NetworkType == DispatchRemoteNetworkType::SKIP,
        packet_queue_nop,
        std::conditional_t<
            CBMode == true,
            output_queue_cb_mode_impl_t<select_network_t<NetworkType>, InputNetworkTypeSequence, InputCBSequence>,
            output_queue_impl_t<select_network_t<NetworkType>, InputNetworkTypeSequence, InputCBSequence>
            >
        >;
};

// Provides a concrete output queue implementation based on the network type and CB mode.
template <
    DispatchRemoteNetworkType NetworkType,
    bool CBMode,
    typename InputNetworkTypeSequence,
    typename InputCBSequence>
using select_output_queue_t =
    typename select_output_queue<NetworkType, CBMode, InputNetworkTypeSequence, InputCBSequence>::type;

// Stores an input queue in raw memory. Engage must be called once
// before using the accessors.
class UnsafePacketInputQueueVariant {
private:
    // Static assert to ensure the specialized classes do not exceed this hardcoded storage size
    static constexpr size_t k_QueueSize = 128;
    alignas(16) std::byte storage[k_QueueSize];

public:
    // Initialize memory contents
    template <DispatchRemoteNetworkType NetworkType, bool CBMode>
    void engage() {
        using T = select_input_queue_t<NetworkType, CBMode>;
        static_assert(sizeof(T) <= k_QueueSize);
        new (storage) T();
    }

    // Get pointer to concrete instance
    template <DispatchRemoteNetworkType NetworkType, bool CBMode>
    inline select_input_queue_t<NetworkType, CBMode>* get() {
        return std::launder(reinterpret_cast<select_input_queue_t<NetworkType, CBMode>*>(storage));
    }

    // Get pointer to a known type
    template <typename T>
    inline T* get_known_type() {
        return std::launder(reinterpret_cast<T*>(storage));
    }
};

// Stores an output queue in raw memory. Engage must be called once
// before using the accessors.
class UnsafePacketOutputQueueVariant {
private:
    // Output queue is expected to be larger than the input queue due to the additional
    // state tracking that's needed
    static constexpr size_t k_QueueSize = 192;
    alignas(16) std::byte storage[k_QueueSize];

public:
    // Initialize memory contents
    template <
        DispatchRemoteNetworkType NetworkType,
        bool CBMode,
        typename InputNetworkTypeSequence,
        typename InputCBSequence>
    void engage() {
        using T = select_output_queue_t<NetworkType, CBMode, InputNetworkTypeSequence, InputCBSequence>;
        static_assert(sizeof(T) <= k_QueueSize);
        new (storage) T();
    }

    // Get pointer to concrete instance
    template <
        DispatchRemoteNetworkType NetworkType,
        bool CBMode,
        typename InputNetworkTypeSequence,
        typename InputCBSequence>
    inline select_output_queue_t<NetworkType, CBMode, InputNetworkTypeSequence, InputCBSequence>* get() {
        return std::launder(
            reinterpret_cast<select_output_queue_t<NetworkType, CBMode, InputNetworkTypeSequence, InputCBSequence>*>(
                storage));
    }

    // Get pointer to a known type
    template <typename T>
    inline T* get_known_type() {
        return std::launder(reinterpret_cast<T*>(storage));
    }
};

// A sequence of DispatchRemoteNetworkType's used for stamping out multiple input/output queue templates
template <DispatchRemoteNetworkType... NetworkTypes>
struct NetworkTypeSequence {
    static constexpr size_t size = sizeof...(NetworkTypes);
    static constexpr std::array<DispatchRemoteNetworkType, size> values = {NetworkTypes...};
};
// When there is no queue
using NoNetworkTypeSequence = NetworkTypeSequence<>;

// A sequence of CB Mode enabled or disabled used for stamping out multiple input/output queue templates
template <bool... CBModeEnabled>
struct CBModeTypeSequence {
    static constexpr size_t size = sizeof...(CBModeEnabled);
    static constexpr std::array<bool, size> values = {CBModeEnabled...};
};
// When there is no queue
using NoCBModeTypeSequence = CBModeTypeSequence<>;

// Common Initialization parameters for packet queues
struct init_params_t {
    // Local Queue
    uint8_t queue_id;
    uint32_t queue_start_addr_words;
    uint32_t queue_size_words;

    // Remote Queue
    uint8_t remote_queue_id;
    uint8_t remote_x;
    uint8_t remote_y;

    // Scratch Buffers
    uint32_t ptrs_addr;
    uint32_t remote_ptrs_addr;

    // CB Mode
    uint8_t local_sem_id{0};
    uint8_t remote_sem_id{0};
    uint8_t log_page_size{0};

    // Input Queue
    // bool packetizer_input{false}; can be determined using std::derived_from
    uint16_t packetizer_input_src{0}; // ex cb mode
    uint16_t packetizer_input_dest{0};

    // Output Queue
    UnsafePacketInputQueueVariant* input_queues{nullptr};
    uint32_t num_input_queues{0};     // number of elements in input_queues
    // bool unpacketizer_output{false}; can be determined using std::derived_from
    bool unpacketizer_output_remove_header{false};
};

// (CRTP) Base Class for all packet queues. Do not create instances of this directly (UB).
// T is the implementation
// R is the remote type
template <typename T, typename R>
class packet_queue_base_t {
private:
    static constexpr uint32_t k_ReservedWords = 1;

    // All pointers are in the units of words
    // to get the actual address, need to multiply it by the word size.
    volatile uint32_t* wptr;
    volatile uint32_t* rptr_sent;
    volatile uint32_t* rptr_cleared;
    volatile uint32_t* local_ready_status;  // stream reg

    uint32_t* shadow_remote_wptr;
    uint32_t* shadow_remote_rptr_sent;
    uint32_t* shadow_remote_rptr_cleared;

    uint32_t remote_scratch_buffer_wptr_addr;
    uint32_t remote_scratch_buffer_rptr_sent_addr;
    uint32_t remote_scratch_buffer_rptr_cleared_addr;

    uint32_t queue_id;
    uint32_t queue_start_addr_words;
    uint32_t queue_size_words;

    uint32_t remote_x;
    uint32_t remote_y;

    uint32_t remote_ready_status_addr;      // stream reg

    inline T* impl() { return static_cast<T*>(this); }

    // Increments that wrap over the queue size are not supported.
    // to support them we need to add a case for if value >= 2*queue_size_words then
    // use the modulus operator.
    inline uint32_t wrap_ptr(uint32_t v) const {
        if (v >= this->queue_size_words) {
            v -= this->queue_size_words;
        }
        return v;
    }

    // Returns the distance between two positions, end and start, accounting
    // for any wrapping.
    // without wrapping, end is the pointer on the right. start is the pointer on the left.
    inline uint32_t distance(uint32_t end, uint32_t start) const {
        if (end >= start) {
            return end - start;
        } else {
            return this->queue_size_words - start + end;
        }
    }

protected:
    R remote;

    packet_queue_base_t() = default;
    ~packet_queue_base_t() = default;

public:
    inline uint32_t get_queue_id() { return this->queue_id; }

    inline uint32_t get_remote_x() { return this->remote_x; }

    inline uint32_t get_remote_y() { return this->remote_y; }

    inline uint32_t get_queue_start_addr_words() const { return this->queue_start_addr_words; }

    inline uint32_t get_queue_size_words() const { return this->queue_size_words; }

    inline uint32_t get_queue_local_wptr() const { return *this->wptr; }

    inline uint32_t get_queue_local_rptr_sent() const { return *this->rptr_sent; }

    inline uint32_t get_queue_local_rptr_cleared() const { return *this->rptr_cleared; }

    inline uint32_t get_queue_data_num_words_occupied() const {
        return distance(this->get_queue_local_wptr(), this->get_queue_local_rptr_cleared());
    }

    inline uint32_t get_queue_data_num_words_free() const {
        return this->queue_size_words - this->get_queue_data_num_words_occupied() - k_ReservedWords;
    }

    inline uint32_t get_num_words_sent_not_cleared() const {
        return distance(this->get_queue_local_rptr_sent(), this->get_queue_local_rptr_cleared());
    }

    inline uint32_t get_num_words_written_not_sent() const {
        return distance(this->get_queue_local_wptr(), this->get_queue_local_rptr_sent());
    }

    inline uint32_t get_queue_rptr_sent_addr_bytes() const {
        return (this->queue_start_addr_words + this->get_queue_local_rptr_sent()) * PACKET_WORD_SIZE_BYTES;
    }

    inline uint32_t get_queue_rptr_cleared_addr_bytes() const {
        return (this->queue_start_addr_words + this->get_queue_local_rptr_cleared()) * PACKET_WORD_SIZE_BYTES;
    }

    inline uint32_t get_queue_wptr_addr_bytes() const {
        return (this->queue_start_addr_words + this->get_queue_local_wptr()) * PACKET_WORD_SIZE_BYTES;
    }

    inline uint32_t get_queue_words_before_rptr_sent_wrap() const {
        return this->queue_size_words - this->get_queue_local_rptr_sent();
    }

    inline uint32_t get_queue_words_before_rptr_cleared_wrap() const {
        return this->queue_size_words - this->get_queue_local_rptr_cleared();
    }

    inline uint32_t get_queue_words_before_wptr_wrap() const {
        return this->queue_size_words - this->get_queue_local_wptr();
    }

    inline uint32_t get_remote_ready_status() const { return *this->local_ready_status; }

    inline bool is_remote_ready() const { return *this->local_ready_status == PACKET_QUEUE_REMOTE_READY_FLAG; }

    inline bool is_remote_finished() const { return *this->local_ready_status == PACKET_QUEUE_REMOTE_FINISHED_FLAG; }

    inline void reset_ready_flag() { *this->local_ready_status = 0; }

    inline void set_queue_finished() { *this->local_ready_status = PACKET_QUEUE_REMOTE_FINISHED_FLAG; }

    inline void set_final_remote_xy(uint8_t x, uint8_t y) { this->remote.init(x, y, 0, 0); }

    inline void set_remote_ready_status_addr(uint8_t remote_queue_id) {
        this->remote_ready_status_addr = STREAM_REG_ADDR(remote_queue_id, STREAM_REMOTE_SRC_REG_INDEX);
    }

    inline void send_remote_finished_notification() {
        this->remote.reg_update(this->remote_ready_status_addr, PACKET_QUEUE_REMOTE_FINISHED_FLAG);
    }

    inline void send_remote_ready_notification() {
        this->remote.reg_update(this->remote_ready_status_addr, PACKET_QUEUE_REMOTE_READY_FLAG);
    }

    inline void advance_queue_local_wptr(uint32_t num_words) { *this->wptr = wrap_ptr(*this->wptr + num_words); }

    inline void advance_queue_local_rptr_sent(uint32_t num_words) {
        *this->rptr_sent = wrap_ptr(*this->rptr_sent + num_words);
    }

    inline void advance_queue_local_rptr_cleared(uint32_t num_words) {
        *this->rptr_cleared = wrap_ptr(*this->rptr_cleared + num_words);
    }

    inline void advance_queue_remote_wptr(uint32_t num_words) {
        *this->shadow_remote_wptr = wrap_ptr(*this->shadow_remote_wptr + num_words);
        this->remote.ptr_update((uint32_t)this->shadow_remote_wptr, this->remote_scratch_buffer_wptr_addr);
    }

    inline void advance_queue_remote_rptr_sent(uint32_t num_words) {
        *this->shadow_remote_rptr_sent = wrap_ptr(*this->shadow_remote_rptr_sent + num_words);
        this->remote.ptr_update((uint32_t)this->shadow_remote_rptr_sent, this->remote_scratch_buffer_rptr_sent_addr);
    }

    inline void advance_queue_remote_rptr_cleared(uint32_t num_words) {
        *this->shadow_remote_rptr_cleared = wrap_ptr(*this->shadow_remote_rptr_cleared + num_words);
        this->remote.ptr_update(
            (uint32_t)this->shadow_remote_rptr_cleared, this->remote_scratch_buffer_rptr_cleared_addr);
    }

    inline void handle_recv() {
        this->remote.handle_recv();
    }

    inline bool busy() const {
        return this->remote.busy();
    }

    void init(const init_params_t* params) {
        ASSERT(params->remote_ptrs_addr != 0);
        ASSERT(params->ptrs_addr != 0);
        this->queue_id = params->queue_id;
        this->queue_start_addr_words = params->queue_start_addr_words;
        this->queue_size_words = params->queue_size_words;
        this->remote_x = params->remote_x;
        this->remote_y = params->remote_y;

        this->remote_scratch_buffer_wptr_addr = reinterpret_cast<uint32_t>(
            packet_queue_scratch_buffer_layout_t::get_wptr(params->remote_ptrs_addr));
        this->remote_scratch_buffer_rptr_sent_addr = reinterpret_cast<uint32_t>(
            packet_queue_scratch_buffer_layout_t::get_rptr_sent(params->remote_ptrs_addr));
        this->remote_scratch_buffer_rptr_cleared_addr = reinterpret_cast<uint32_t>(
            packet_queue_scratch_buffer_layout_t::get_rptr_cleared(params->remote_ptrs_addr));

        this->wptr = packet_queue_scratch_buffer_layout_t::get_wptr(params->ptrs_addr);
        this->rptr_sent = packet_queue_scratch_buffer_layout_t::get_rptr_sent(params->ptrs_addr);
        this->rptr_cleared = packet_queue_scratch_buffer_layout_t::get_rptr_cleared(params->ptrs_addr);
        this->shadow_remote_wptr =
            packet_queue_scratch_buffer_layout_t::get_shadow_remote_wptr(params->ptrs_addr);
        this->shadow_remote_rptr_sent =
            packet_queue_scratch_buffer_layout_t::get_shadow_remote_rptr_sent(params->ptrs_addr);
        this->shadow_remote_rptr_cleared =
            packet_queue_scratch_buffer_layout_t::get_shadow_remote_rptr_cleared(params->ptrs_addr);

        // Init remote controller
        this->remote.init(
            params->remote_x, params->remote_y, params->ptrs_addr, params->remote_ptrs_addr);

        // Reset local values
        *this->wptr = 0;
        *this->rptr_sent = 0;
        *this->rptr_cleared = 0;
        *this->shadow_remote_wptr = 0;
        *this->shadow_remote_rptr_sent = 0;
        *this->shadow_remote_rptr_cleared = 0;

        this->remote_ready_status_addr = STREAM_REG_ADDR(params->remote_queue_id, STREAM_REMOTE_SRC_REG_INDEX);
        this->local_ready_status =
            reinterpret_cast<volatile uint32_t*>(STREAM_REG_ADDR(params->queue_id, STREAM_REMOTE_SRC_REG_INDEX));

        // Call child init
        this->impl()->_init(params);

        // Reset remote ready flag once implementations are done initializing
        this->reset_ready_flag();
    }
};  // packet_queue_base_t

// (CRTP CRTP) Base Class for all input queues. Do not create instances of this directly (UB).
// T is the implementation
// R is the remote type
template <typename T, typename R>
class packet_queue_input_t : public packet_queue_base_t<packet_queue_input_t<T, R>, R> {
private:
    friend packet_queue_base_t<packet_queue_input_t<T, R>, R>;
    friend input_queue_impl_t<R>;

    tt_l1_ptr dispatch_packet_header_t* curr_packet_header;
    bool curr_packet_valid;
    uint16_t curr_packet_src;
    uint16_t curr_packet_dest;
    uint32_t curr_packet_size_words;
    uint32_t curr_packet_words_sent;
    uint32_t curr_packet_tag;
    uint16_t curr_packet_flags;
    uint16_t end_of_cmd;

    inline T* impl() { return static_cast<T*>(this); }

    // Update the state of this queue to the next packet. This function is only valid if get_num_words_written_not_sent > 0.
    // _set_next_packet will be called with the next packet header
    inline void start_next_packet() {
        ASSERT(this->get_num_words_written_not_sent());
        const uint32_t next_header_addr =
            (this->get_queue_start_addr_words() + this->get_queue_local_rptr_sent()) * PACKET_WORD_SIZE_BYTES;
        auto next_packet_header = reinterpret_cast<tt_l1_ptr dispatch_packet_header_t*>(next_header_addr);

        this->curr_packet_header = next_packet_header;

        uint32_t packet_size_and_flags = next_packet_header->packet_size_bytes;
        uint32_t packet_size_bytes = packet_size_and_flags & 0xFFFFFFFE;  // TODO. Can we clean this up

        this->end_of_cmd = !(packet_size_and_flags & 1);
        this->curr_packet_size_words = packet_size_bytes / PACKET_WORD_SIZE_BYTES;
        this->curr_packet_words_sent = 0;

        // Round up to next word to not truncate the data
        static_assert(is_power_of_2(PACKET_WORD_SIZE_BYTES));
        if (packet_size_bytes & (PACKET_WORD_SIZE_BYTES - 1)) {
            this->curr_packet_size_words++;
        }

        // Set this to true right now so if the downstream _set_next_packet does any get calls
        // we don't advance again
        this->curr_packet_valid = true;
        this->impl()->_set_next_packet(this->curr_packet_header);
    }

protected:
    packet_queue_input_t() = default;
    ~packet_queue_input_t() = default;

    void set_curr_packet_src(uint16_t src) { this->curr_packet_src = src; }

    void set_curr_packet_dest(uint16_t dest) { this->curr_packet_dest = dest; }

    void set_curr_packet_tag(uint32_t tag) { this->curr_packet_tag = tag; }

    void set_curr_packet_flags(uint16_t flags) { this->curr_packet_flags = flags; }

    void _init(const init_params_t* params) {
        this->curr_packet_valid = false;
        this->impl()->_init(params);
    };

public:
    // Returns true if 1) current valid is valid or 2) current packet is not valid and we moved to a valid packet
    inline bool advance_if_not_valid() {
        return this->curr_packet_valid || (this->get_num_words_written_not_sent() && (this->start_next_packet(), true));
    }

    // Returns true if the current packet is valid
    inline bool get_curr_packet_valid() {
        this->impl()->_update_local_wptr_val();
        return this->curr_packet_valid;
    }

    // Returns the packet source identifier
    inline uint16_t get_curr_packet_src() const {
        return this->curr_packet_src;
    }

    // Returns the packet destination identifier
    inline uint16_t get_curr_packet_dest() const {
        return this->curr_packet_dest;
    }

    // Returns the packet size in words
    inline uint32_t get_curr_packet_size_words() const {
        return this->curr_packet_size_words;
    }

    // Returns the packet's tags.
    inline uint32_t get_curr_packet_tag() const {
        return this->curr_packet_tag;
    }

    // Returns the packet's flags
    inline uint16_t get_curr_packet_flags() const {
        return this->curr_packet_flags;
    }

    // Returns how many words left to send. This function will try to advance to the next
    // packet if the current one is not valid.
    inline uint32_t get_curr_packet_words_remaining() {
        if (!this->curr_packet_valid) {
            this->start_next_packet();
        }

        return this->curr_packet_size_words - this->curr_packet_words_sent;
    }

    // Returns a pointer to the current packet's header
    inline tt_l1_ptr dispatch_packet_header_t* get_curr_packet_header_ptr() const {
        return this->curr_packet_header;
    }

    // Get end of cmd
    inline uint16_t get_end_of_cmd() const { return this->end_of_cmd; }

    // Returns true if data for current packet is partially sent
    inline bool partial_packet_sent() const { return this->curr_packet_valid && (this->curr_packet_words_sent > 0); }

    // Returns true if data for current packet has not started to be sent yet
    inline bool curr_packet_start() const { return this->curr_packet_valid && (this->curr_packet_words_sent == 0); }

    // Returns true if the buffer has enough space for the current packet can be sent in one shot
    // num_words_available_to_send will be sent to the words to send
    inline bool full_packet_available_to_send(uint32_t& num_words_available_to_send) {
        num_words_available_to_send = this->get_num_words_written_not_sent();

        if (!num_words_available_to_send) {
            return false;
        }
        return num_words_available_to_send >= this->get_curr_packet_words_remaining();
    }

    // Returns the number of words that can be written to the destination buffer.
    inline uint32_t get_curr_packet_num_words_available_to_send() {
        this->impl()->_update_local_wptr_val();
        uint32_t num_words = this->get_num_words_written_not_sent();
        if (num_words == 0) {
            return 0;
        }
        num_words = std::min(num_words, this->get_curr_packet_words_remaining());
        return num_words;
    }

    // Advance the sent read pointer indicating copy has started
    // Returns the number of words that were potentially skipped (e.g., alignment)
    inline uint32_t advance_words_sent(uint32_t num_words) {
        if (!num_words) {
            return 0;
        }

        this->advance_queue_local_rptr_sent(num_words);
        this->advance_queue_remote_rptr_sent(num_words);
        this->curr_packet_words_sent += num_words;

        // Current packet is done. Move to next packet. Maybe there will be an adjustment
        // for alignment.
        if (!this->get_curr_packet_words_remaining()) {
            const auto adjustment = this->impl()->_align_rptr_sent(num_words);
            this->curr_packet_valid = false;
            if (this->get_num_words_written_not_sent()) {
                this->start_next_packet();
            }
            return adjustment;
        }

        // No adjustment needed
        return 0;
    }

    // Advance the cleared read pointer indicating copy is complete
    inline void advance_words_cleared(uint32_t num_words) {
        if (!num_words) {
            return;
        }

        this->advance_queue_local_rptr_cleared(num_words);
        this->advance_queue_remote_rptr_cleared(num_words);

        this->impl()->_align_rptr_cleared(num_words);
    }

    // Advance cleared pointer to match sent
    inline void clear_all_words_sent() {
        uint32_t num_words = this->get_num_words_sent_not_cleared();
        if (num_words > 0) {
            this->advance_words_cleared(num_words);
        }
    }
};  // packet_queue_input_t

// Regular Input Queue with remote R
// R is the remote type
template <typename R>
class input_queue_impl_t final : public packet_queue_input_t<input_queue_impl_t<R>, R> {
    friend packet_queue_input_t<input_queue_impl_t<R>, R>;

protected:
    void _init(const init_params_t* params) {}

    inline void _set_next_packet(tt_l1_ptr dispatch_packet_header_t* header) {
        this->set_curr_packet_dest(header->packet_dest);
        this->set_curr_packet_src(header->packet_src);
        this->set_curr_packet_tag(header->tag);
        this->set_curr_packet_flags(header->packet_flags);
    }

    inline void _update_local_wptr_val() {}

    inline uint32_t _align_rptr_sent(uint32_t num_words) { return 0; }

    inline void _align_rptr_cleared(uint32_t num_words) {}

public:
    input_queue_impl_t() = default;
    ~input_queue_impl_t() = default;
};  // input_queue_impl_t

// Paged Input Queue with remote R
// R is the remote type
template <typename R>
class input_queue_cb_mode_impl_t final : public packet_queue_input_t<input_queue_cb_mode_impl_t<R>, R> {
private:
    friend packet_queue_input_t<input_queue_cb_mode_impl_t<R>, R>;

    uint32_t packetizer_page_words_cleared;

    cb_mode_config_t config;

protected:
    inline void _init(const init_params_t* params) {
        this->packetizer_page_words_cleared = 0;
        this->set_curr_packet_src(params->packetizer_input_src);
        this->set_curr_packet_dest(params->packetizer_input_dest);
        this->set_curr_packet_flags(0);
        this->set_curr_packet_tag(0xdeadbeef);

        this->config.init(params->log_page_size, params->local_sem_id, params->remote_sem_id);
    }

    inline void _set_next_packet(tt_l1_ptr dispatch_packet_header_t* header) {
        // Update the current packet header to continue forwarding data as is
        // to the original location by updating the headers to match
        // prefetcher has size in bytes
        header->packet_dest = this->get_curr_packet_dest();
        header->packet_src = this->get_curr_packet_src();
        header->tag = this->get_curr_packet_tag();
        header->packet_flags = this->get_curr_packet_flags();
    }

    inline void _update_local_wptr_val() {
        bool underflow = false;
        uint32_t local_sem_val = get_local_sem_val(this->config.local_sem_id, underflow);

        if (underflow) {
            this->set_queue_finished();
        }

        for (uint32_t i = 0; i < local_sem_val; i++) {
            this->advance_queue_local_wptr(this->config.page_size_words);
        }
        increment_local_sem_val(this->config.local_sem_id, -1 * local_sem_val);
    }

    inline uint32_t _align_rptr_sent(uint32_t num_words) {
        uint32_t rptr_val = this->get_queue_local_rptr_sent();
        uint32_t num_words_past_page_boundary = rptr_val & this->config.page_size_mask;
        uint32_t input_pad_words_skipped = 0;

        if (num_words_past_page_boundary > 0) {
            input_pad_words_skipped = this->config.page_size_words - num_words_past_page_boundary;
            this->advance_queue_local_rptr_sent(input_pad_words_skipped);
        }

        return input_pad_words_skipped;
    }

    inline void _align_rptr_cleared(uint32_t num_words) {
        this->packetizer_page_words_cleared += num_words;
        uint32_t remote_sem_inc = 0;
        while (this->packetizer_page_words_cleared >= this->config.page_size_words) {
            remote_sem_inc++;
            this->packetizer_page_words_cleared -= this->config.page_size_words;
        }

        if (remote_sem_inc) {
            increment_remote_sem_val(
                this->get_remote_x(), this->get_remote_y(), this->config.local_sem_id, remote_sem_inc);
        }
    }
};  // input_queue_cb_mode_impl_t

// (CRTP CRTP) Base Class for all output queues. Do not create instances of this directly (UB).
// T is the implementation
// R is the remote type
// InputNetworkTypeSequence types[i] is the network type for input queue i
// InputCBSequence cb_mode_enabled[i] is cb mode enabled for input queue i
template <typename T, typename R, typename InputNetworkTypeSequence, typename InputCBSequence>
class packet_queue_output_t
    : public packet_queue_base_t<packet_queue_output_t<T, R, InputNetworkTypeSequence, InputCBSequence>, R> {
private:
    static_assert(InputNetworkTypeSequence::size == InputCBSequence::size);
    static_assert(InputNetworkTypeSequence::size <= k_MaxInputOutputQueues);

    friend packet_queue_base_t<packet_queue_output_t<T, R, InputNetworkTypeSequence, InputCBSequence>, R>;

    // How do we keep track of which variant is at each index?
    // Number of output queues and output queue configuration does not change during runtime.
    // This output_queue is templated to pass in a sequence of the input queue types.
    // Depending on which input_queues[i] is accessed, we can get the type info from the sequence
    // template args
    UnsafePacketInputQueueVariant* input_queues;
    uint32_t num_input_queues;

    uint32_t words_in_flight[2 * k_MaxInputOutputQueues]; // 2X for curr and prev
    uint32_t unpacketizer_page_words_sent;
    bool unpacketizer_remove_header;

    // Pointer to words_in_flight[current queue] and [previous queue]
    uint32_t* curr_input_queue_words_in_flight;
    uint32_t* prev_input_queue_words_in_flight;
    uint32_t curr_output_total_words_in_flight;
    uint32_t prev_output_total_words_in_flight;

    inline T* impl() { return static_cast<T*>(this); }

protected:
    packet_queue_output_t() = default;
    ~packet_queue_output_t() = default;

    void _init(const init_params_t* params) {
        ASSERT(params->num_input_queues <= k_MaxInputOutputQueues);
        ASSERT(params->num_input_queues == InputNetworkTypeSequence::size);

        this->unpacketizer_page_words_sent = 0;
        this->unpacketizer_remove_header = params->unpacketizer_output_remove_header;

        this->num_input_queues = params->num_input_queues;
        this->input_queues = params->input_queues;

        this->curr_input_queue_words_in_flight = &(this->words_in_flight[0]);
        this->prev_input_queue_words_in_flight = &(this->words_in_flight[k_MaxInputOutputQueues]);
        this->curr_output_total_words_in_flight = 0;
        this->prev_output_total_words_in_flight = 0;
        for (uint32_t i = 0; i < k_MaxInputOutputQueues; i++) {
            this->words_in_flight[i] = 0;
        }

        this->impl()->_init(params);
    }

    // Set the unpacketizer page words sent
    inline void set_unpacketizer_page_words_sent(uint32_t num_words) { this->unpacketizer_page_words_sent = num_words; }

    // Increment unpacketizer page words sent
    inline void inc_unpacketizer_page_words_sent(uint32_t num_words) {
        this->unpacketizer_page_words_sent += num_words;
    }

public:
    // Return unpacketizer remove header
    inline bool get_unpacketizer_remove_header() const { return this->unpacketizer_remove_header; }

    // Returns the page words set for unpacketize header mode
    inline uint32_t get_unpacketizer_page_words_sent() const { return this->unpacketizer_page_words_sent; }

    // Returns the total words in flight of the current output queue
    inline uint32_t get_curr_output_total_words_in_flight() const { return this->curr_output_total_words_in_flight; }

    // Return the total words in flight of the previous output queue
    inline uint32_t get_prev_output_total_words_in_flight() const { return this->prev_output_total_words_in_flight; }

    // Return the number of words that can be forwarded. The number of words that
    // can be sent is the minimum of
    // 1. words available in the input queue
    // 2. words available in the input queue before the rptr will wrap
    // 3. space available in the output buffer
    // 4. space available in the output buffer before the wptr will wrap
    // 5. maximum transmission size for the remote type
    // InputQueueIndex is the queue index
    template <size_t InputQueueIndex>
    inline uint32_t get_num_words_to_send() const {
        static_assert(InputQueueIndex < k_MaxInputOutputQueues && InputQueueIndex < InputNetworkTypeSequence::size);
        static constexpr uint32_t k_MaxSendSize = 1536;  // TODO packet_queue_remote_control_limits<R>::max_send_words;

        // Based on the InputQueueIndex, cast the variant element at that index to the correct
        // type in the InputNetworkTypeSequence X InputCBSequence combo
        auto* active_input_queue = this->input_queues[InputQueueIndex]
                                       .template get<
                                           InputNetworkTypeSequence::values[InputQueueIndex],
                                           InputCBSequence::values[InputQueueIndex]>();

        uint32_t num_words_available_in_input = active_input_queue->get_curr_packet_num_words_available_to_send();
        uint32_t num_words_before_input_rptr_wrap = active_input_queue->get_queue_words_before_rptr_sent_wrap();
        uint32_t num_words_free_in_output = this->get_queue_data_num_words_free();
        uint32_t output_buf_words_before_wptr_wrap = this->get_queue_words_before_wptr_wrap();

        uint32_t num_words_to_forward = std::min(num_words_available_in_input, num_words_before_input_rptr_wrap);
        num_words_to_forward = std::min(num_words_to_forward, num_words_free_in_output);

        if (num_words_to_forward == 0) {
            return 0;
        }

        num_words_to_forward = std::min(num_words_to_forward, output_buf_words_before_wptr_wrap);
        num_words_to_forward = std::min(num_words_to_forward, k_MaxSendSize);

        return num_words_to_forward;
    }

    // Advance words sent to cleared for all input queues
    inline uint32_t prev_words_in_flight_flush() {
        uint32_t words_flushed = this->get_prev_output_total_words_in_flight();
        if (words_flushed > 0) {
            process_queues<InputNetworkTypeSequence, InputCBSequence>(
                [&]<auto network_type, auto cbmode, auto sequence_i>(auto index) -> bool {
                    auto active_input_queue = input_queues[index].template get<network_type, cbmode>();
                    active_input_queue->advance_words_cleared(this->prev_input_queue_words_in_flight[index]);
                    this->prev_input_queue_words_in_flight[index] = 0;
                    return true;
                });
        }

        // Swapping current to previous
        std::swap(this->prev_input_queue_words_in_flight, this->curr_input_queue_words_in_flight);
        this->prev_output_total_words_in_flight = this->curr_output_total_words_in_flight;
        this->curr_output_total_words_in_flight = 0;

        return words_flushed;
    }

    // Check if any words need to be flushed and flush
    inline uint32_t prev_words_in_flight_check_flush() { return this->impl()->_prev_words_in_flight_check_flush(); }

    // Set words as in flight (sent) for an input queue
    // InputQueueIndex is the queue index
    template <size_t InputQueueIndex>
    inline void register_words_in_flight(uint32_t num_words) {
        static_assert(InputQueueIndex < k_MaxInputOutputQueues && InputQueueIndex < InputNetworkTypeSequence::size);
        auto active_input_queue = input_queues[InputQueueIndex]
                                      .template get<
                                          InputNetworkTypeSequence::values[InputQueueIndex],
                                          InputCBSequence::values[InputQueueIndex]>();

        uint32_t input_pad_words_skipped = active_input_queue->advance_words_sent(num_words);

        this->curr_input_queue_words_in_flight[InputQueueIndex] += (num_words + input_pad_words_skipped);
        this->curr_output_total_words_in_flight += num_words;
    }

    // Forward data from input to the remote output
    // InputQueueIndex is the input queue index to forward
    template <size_t InputQueueIndex>
    inline uint32_t forward_data_from_input(bool& full_packet_sent, uint16_t end_of_cmd) {
        static_assert(InputQueueIndex < k_MaxInputOutputQueues && InputQueueIndex < InputNetworkTypeSequence::size);
        uint32_t num_words = this->get_num_words_to_send<InputQueueIndex>();
        auto* active_input_queue = input_queues[InputQueueIndex]
                                       .template get<
                                           InputNetworkTypeSequence::values[InputQueueIndex],
                                           InputCBSequence::values[InputQueueIndex]>();

        // It will be possible to send the full packet one shot
        full_packet_sent = (num_words == active_input_queue->get_curr_packet_words_remaining());
        if (!num_words) {
            return 0;
        }

        if (this->get_unpacketizer_remove_header() && active_input_queue->curr_packet_start()) {
            // remove 1 word == header
            num_words--;
            this->register_words_in_flight<InputQueueIndex>(1);
            if (!num_words) {
                return 0;
            }
        }

        uint32_t src_addr =
            (active_input_queue->get_queue_start_addr_words() + active_input_queue->get_queue_local_rptr_sent()) *
            PACKET_WORD_SIZE_BYTES;  // Local
        uint32_t dest_addr =
            (this->get_queue_start_addr_words() + this->get_queue_local_wptr()) * PACKET_WORD_SIZE_BYTES;  // Remote

        this->remote.send_data(src_addr, dest_addr, num_words);

        this->register_words_in_flight<InputQueueIndex>(num_words);

        this->advance_queue_local_wptr(num_words);
        this->impl()->_forward_data_complete(num_words, full_packet_sent, end_of_cmd);

        return num_words;
    }

    // Block until all outputs are complete
    inline bool output_barrier(uint32_t timeout_cycles = 0) {
        uint32_t start_timestamp = 0;
        if (timeout_cycles > 0) {
            start_timestamp = get_timestamp_32b();
        }

        this->impl()->_barrier_setup();

        while (this->get_queue_data_num_words_occupied() > 0) {
            this->impl()->_barrier_process();

            if (timeout_cycles > 0) {
                uint32_t cycles_elapsed = get_timestamp_32b() - start_timestamp;
                if (cycles_elapsed > timeout_cycles) {
                    return false;
                }
            }
        }

        // Advance to cleared
        this->prev_words_in_flight_flush();
        return true;
    }
};  // packet_queue_output_t

// Regular Output Queue with remote R
// R is the remote type
template <typename R, typename InputNetworkTypeSequence, typename InputCBSequence>
class output_queue_impl_t final : public packet_queue_output_t<
                                      output_queue_impl_t<R, InputNetworkTypeSequence, InputCBSequence>,
                                      R,
                                      InputNetworkTypeSequence,
                                      InputCBSequence> {
    friend packet_queue_output_t<
        output_queue_impl_t<R, InputNetworkTypeSequence, InputCBSequence>,
        R,
        InputNetworkTypeSequence,
        InputCBSequence>;

protected:
    void _init(const init_params_t* params) {}

    inline uint32_t _prev_words_in_flight_check_flush() {
        if (this->get_num_words_written_not_sent() <= this->get_curr_output_total_words_in_flight()) {
            return this->prev_words_in_flight_flush();
        } else {
            return 0;
        }
    }

    inline void _forward_data_complete(uint32_t words_forwarded, bool full_packet_sent, uint16_t end_of_cmd) {
        this->advance_queue_remote_wptr(words_forwarded);
    }

    inline void _barrier_setup() {}

    inline void _barrier_process() {}
};  // output_queue_impl_t

// Paged Output Queue with remote R
// R is the remote type
template <typename R, typename InputNetworkTypeSequence, typename InputCBSequence>
class output_queue_cb_mode_impl_t final : public packet_queue_output_t<
                                              output_queue_cb_mode_impl_t<R, InputNetworkTypeSequence, InputCBSequence>,
                                              R,
                                              InputNetworkTypeSequence,
                                              InputCBSequence> {
private:
    friend packet_queue_output_t<
        output_queue_cb_mode_impl_t<R, InputNetworkTypeSequence, InputCBSequence>,
        R,
        InputNetworkTypeSequence,
        InputCBSequence>;
    cb_mode_config_t config;

    inline void local_sem_rptr_cleared_update() {
        bool underflow = false;
        uint32_t local_sem_val = get_local_sem_val(this->config.local_sem_id, underflow);

        if (underflow) {
            this->set_queue_finished();
        }

        for (uint32_t i = 0; i < local_sem_val; i++) {
            this->advance_queue_local_rptr_cleared(this->config.page_size_words);
        }
        increment_local_sem_val(this->config.local_sem_id, -1 * local_sem_val);
    }

    inline uint32_t get_local_sem_downstream_complete() const {
        bool underflow = false;  // unused
        uint32_t val = get_local_sem_val(this->config.local_sem_id, underflow);
        return (val & 0x80000000);
    }

protected:
    void _init(const init_params_t* params) {
        this->config.init(params->log_page_size, params->local_sem_id, params->remote_sem_id);
    }

    inline uint32_t _prev_words_in_flight_check_flush() {
        uint32_t words_written_not_sent = this->get_num_words_written_not_sent();
        noc_async_writes_flushed();
        this->advance_queue_local_rptr_sent(words_written_not_sent);

        uint32_t words_flushed = this->prev_words_in_flight_flush();
        this->local_sem_rptr_cleared_update();
        return words_flushed;
    }

    inline void _forward_data_complete(uint32_t words_forwarded, bool full_packet_sent, uint16_t end_of_cmd) {
        this->inc_unpacketizer_page_words_sent(words_forwarded);

        if (full_packet_sent && end_of_cmd) {
            uint32_t unpacketizer_page_words_sent_past_page_bound =
                this->get_unpacketizer_page_words_sent() & this->config.page_size_mask;
            if (unpacketizer_page_words_sent_past_page_bound > 0) {
                uint32_t pad_words = this->config.page_size_words - unpacketizer_page_words_sent_past_page_bound;
                this->inc_unpacketizer_page_words_sent(pad_words);
                this->advance_queue_local_wptr(pad_words);
            }
        }

        uint32_t remote_sem_inc = 0;
        while (this->get_unpacketizer_page_words_sent() >= this->config.page_size_words) {
            this->inc_unpacketizer_page_words_sent(-1 * this->config.page_size_words);
            remote_sem_inc++;
        }

        if (remote_sem_inc) {
            increment_remote_sem_val(
                this->get_remote_x(), this->get_remote_y(), this->config.local_sem_id, remote_sem_inc);
        }
    }

    inline void _barrier_setup() { noc_async_writes_flushed(); }

    inline void _barrier_process() {
        this->local_sem_rptr_cleared_update();
        if (this->get_local_sem_downstream_complete()) {
            // There is no guaranteed that dispatch_h will increment semaphore for all commmands
            // (specifically the final terminate command).
            // So just clear whatever remains once the completion signal is received.
            uint32_t words_occupied = this->get_queue_data_num_words_occupied();
            this->advance_queue_local_rptr_cleared(words_occupied);
        }
    }
};  // output_queue_cb_mode_impl_t

/*
**********************
*                    *
*  Functions         *
*                    *
**********************
*/

// Wait for all input and output queues and their remotes to signal Ready on the remote ready status
template <
    typename InputNetworkTypeSequence,
    typename InputCBSequence,
    typename OutputNetworkTypeSequence,
    typename OutputCBSequence>
bool wait_all_input_output_ready(
    UnsafePacketInputQueueVariant* input_queues,
    UnsafePacketOutputQueueVariant* output_queues,
    uint32_t timeout_cycles = 0) {
    bool src_ready[k_MaxInputOutputQueues];
    bool dest_ready[k_MaxInputOutputQueues];

    static_assert(InputNetworkTypeSequence::size == InputCBSequence::size);
    static_assert(OutputNetworkTypeSequence::size == OutputCBSequence::size);

    bool all_src_dest_ready = false;
    uint32_t iters = 0;

    std::fill_n(src_ready, InputNetworkTypeSequence::size, false);
    std::fill_n(dest_ready, OutputNetworkTypeSequence::size, false);

    uint32_t start_timestamp = get_timestamp_32b();
    while (!all_src_dest_ready) {
        iters++;
        if (timeout_cycles > 0) {
            uint32_t cycles_since_start = get_timestamp_32b() - start_timestamp;
            if (cycles_since_start > timeout_cycles) {
                return false;
            }
        }

        all_src_dest_ready = true;
        // checking input queues
        process_queues<InputNetworkTypeSequence, InputCBSequence>(
            [&]<auto network_type, auto cb_mode, auto sequence_i>(auto index) -> bool {
                if (!src_ready[index]) {
                    auto* active_input_queue = input_queues[index].template get<network_type, cb_mode>();
                    src_ready[index] = cb_mode || active_input_queue->is_remote_ready();
                    if (!src_ready[index]) {
                        active_input_queue->send_remote_ready_notification();
                        all_src_dest_ready = false;
                    } else {
                        // handshake with src complete
                    }
                }

                return true;  // keep looping through other queues
            });

        // checking output queues
        process_queues<OutputNetworkTypeSequence, OutputCBSequence>(
            [&]<auto network_type, auto cb_mode, auto sequence_i>(auto index) -> bool {
                if (!dest_ready[index]) {
                    auto* active_output_queue =
                        output_queues[index]
                            .template get<network_type, cb_mode, InputNetworkTypeSequence, InputCBSequence>();
                    dest_ready[index] = cb_mode || active_output_queue->is_remote_ready();
                    if (dest_ready[index]) {
                        active_output_queue->send_remote_ready_notification();
                    } else {
                        all_src_dest_ready = false;
                    }
                }
                return true;
            });

#if defined(COMPILE_FOR_ERISC)
        if ((timeout_cycles == 0) && (iters & 0xFFF) == 0) {
            // if timeout is disabled, context switch every 4096 iterations.
            // this is necessary to allow ethernet routing layer to operate.
            // this core may have pending ethernet routing work.
            internal_::risc_context_switch();
        }
#endif
    }
    return true;
}

};  // namespace packet_queue
