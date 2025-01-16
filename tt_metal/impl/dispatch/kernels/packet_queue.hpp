// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Required defines:
// FD_CORE_TYPE - Fast dispatch core type.
//
// Optional defines:
// POW2_CB - Enable power of 2 optimizations. This can be enabled if the queue size is a power of 2
//

#pragma once

#include <utility>
#include <array>

#include "debug/assert.h"
#include "risc_attribs.h"
#include "hostdevcommon/common_values.hpp"
#include "dataflow_api.h"
#include "noc_overlay_parameters.h"
#include "ethernet/dataflow_api.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"
#include "debug/dprint.h"

constexpr ProgrammableCoreType fd_core_type = static_cast<ProgrammableCoreType>(FD_CORE_TYPE);

constexpr uint32_t NUM_WR_CMD_BUFS = 4;

constexpr uint32_t DEFAULT_MAX_NOC_SEND_WORDS = (NUM_WR_CMD_BUFS-1)*(NOC_MAX_BURST_WORDS*NOC_WORD_BYTES)/PACKET_WORD_SIZE_BYTES;
constexpr uint32_t DEFAULT_MAX_ETH_SEND_WORDS = 2*1024;

constexpr uint32_t NUM_PTR_REGS_PER_INPUT_QUEUE = 1;
constexpr uint32_t NUM_PTR_REGS_PER_OUTPUT_QUEUE = 2;


inline uint64_t get_timestamp() {
    uint32_t timestamp_low = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t timestamp_high = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
    return (((uint64_t)timestamp_high) << 32) | timestamp_low;
}

inline uint64_t get_timestamp_32b() {
    return reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
}

void zero_l1_buf(tt_l1_ptr uint32_t* buf, uint32_t size_bytes) {
    for (uint32_t i = 0; i < size_bytes/4; i++) {
        buf[i] = 0;
    }
}

static FORCE_INLINE
void write_test_results(tt_l1_ptr uint32_t* const buf, uint32_t i, uint32_t val) {
    if (buf != nullptr) {
        buf[i] = val;
    }
}

static FORCE_INLINE
void write_kernel_status(tt_l1_ptr uint32_t* const buf, uint32_t i, uint32_t val) {
    if (buf != nullptr) {
        buf[i] = val;
    }
}

static FORCE_INLINE
void set_64b_result(uint32_t* buf, uint64_t val, uint32_t index = 0) {
    if (buf != nullptr) {
        buf[index] = val >> 32;
        buf[index+1] = val & 0xFFFFFFFF;
    }
}

typedef struct dispatch_packet_header_t dispatch_packet_header_t;

static_assert(sizeof(dispatch_packet_header_t) == PACKET_WORD_SIZE_BYTES);

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


// Helper function to call the lambda N times, passing in the template arguments and index
template <typename NetworkTypeSequence, typename CBModeSequence, typename F, size_t... Index>
bool process_queues(F&& func, std::index_sequence<Index...>) {
    static_assert(NetworkTypeSequence::size == CBModeSequence::size);
    bool result = true;
    (([&]() {
        if constexpr (NetworkTypeSequence::values[Index] == DispatchRemoteNetworkType::DISABLE_QUEUE) {
            return true;
        } else {
            return std::forward<F>(func)
                .template operator()<NetworkTypeSequence::values[Index], CBModeSequence::values[Index], Index>(Index);
        }
     }() &&
      (result = true)),
     ...);
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

class packet_queue_state_t;
class packet_input_queue_state_t;
class packet_output_queue_state_t;

class packet_queue_state_t {
public:
    // Credit registers (words)
    volatile uint32_t* local_wptr_val;
    volatile uint32_t* local_rptr_sent_val;
    volatile uint32_t* local_rptr_cleared_val;
    volatile uint32_t* local_wptr_update;
    volatile uint32_t* local_rptr_sent_update;
    volatile uint32_t* local_rptr_cleared_update;

    uint32_t remote_ready_status_addr;
    volatile uint32_t* local_ready_status_ptr;

    uint32_t remote_wptr_update_addr;
    uint32_t remote_rptr_sent_update_addr;
    uint32_t remote_rptr_cleared_update_addr;

    // Logical wptr and rptr offsets (words)
    uint32_t local_wptr;
    uint32_t local_rptr_sent;
    uint32_t local_rptr_cleared;

    bool cb_mode; // used in advance_next_packet. need to put into template
    uint32_t cb_mode_page_size_words;
    uint8_t cb_mode_local_sem_id;
    uint8_t cb_mode_remote_sem_id;

    uint8_t queue_id;
    uint32_t queue_start_addr_words;
    uint32_t queue_size_words;
#ifdef POW2_CB
    uint32_t ptr_offset_mask;
    uint32_t queue_size_mask;
#else
    uint32_t queue_size_words_2x;
#endif

    uint8_t remote_x, remote_y; // remote source for input queues, remote dest for output queues
    uint8_t remote_queue_id;
    DispatchRemoteNetworkType remote_update_network_type;

    void init(uint8_t queue_id,
              uint32_t queue_start_addr_words,
              uint32_t queue_size_words,
              bool queue_is_input,
              uint8_t remote_x,
              uint8_t remote_y,
              uint8_t remote_queue_id,
              DispatchRemoteNetworkType remote_update_network_type,
              bool cb_mode,
              uint8_t cb_mode_local_sem_id,
              uint8_t cb_mode_remote_sem_id,
              uint8_t cb_mode_log_page_size) {
        this->queue_id = queue_id;
        this->queue_start_addr_words = queue_start_addr_words;
        this->queue_size_words = queue_size_words;
#ifdef POW2_CB
        this->queue_size_mask = (this->queue_size_words << 1) - 1;
        this->ptr_offset_mask = this->queue_size_words - 1;
#else
        this->queue_size_words_2x = this->queue_size_words * 2;
#endif
        this->remote_x = remote_x;
        this->remote_y = remote_y;
        this->remote_queue_id = remote_queue_id;
        this->remote_update_network_type = remote_update_network_type;

        this->cb_mode = cb_mode;
        this->cb_mode_local_sem_id = cb_mode_local_sem_id;
        this->cb_mode_remote_sem_id = cb_mode_remote_sem_id;
        this->cb_mode_page_size_words = (((uint32_t)0x1) << cb_mode_log_page_size)/PACKET_WORD_SIZE_BYTES;

        this->local_rptr_sent = 0;
        this->local_rptr_cleared = 0;
        this->local_wptr = 0;
        if (queue_is_input) {
            this->local_wptr_val = reinterpret_cast<volatile uint32_t*>(
                STREAM_REG_ADDR(NUM_PTR_REGS_PER_INPUT_QUEUE*queue_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));
            this->local_rptr_sent_val = &this->local_rptr_sent;
            this->local_rptr_cleared_val = &this->local_rptr_cleared;

            this->local_wptr_update = reinterpret_cast<volatile uint32_t*>(
                STREAM_REG_ADDR(NUM_PTR_REGS_PER_INPUT_QUEUE*queue_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));

            *reinterpret_cast<volatile uint32_t*>(STREAM_REG_ADDR(NUM_PTR_REGS_PER_INPUT_QUEUE*queue_id, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX)) = 0;
        } else {
            this->local_wptr_val = &this->local_wptr;
            uint32_t adjusted_queue_id = queue_id > 15 ? queue_id - 11 : queue_id;
            this->local_rptr_sent_val = reinterpret_cast<volatile uint32_t*>(
                STREAM_REG_ADDR(NUM_PTR_REGS_PER_OUTPUT_QUEUE*adjusted_queue_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));
            this->local_rptr_cleared_val = reinterpret_cast<volatile uint32_t*>(
                STREAM_REG_ADDR(NUM_PTR_REGS_PER_OUTPUT_QUEUE*adjusted_queue_id+1, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));

            this->local_rptr_sent_update = reinterpret_cast<volatile uint32_t*>(
                STREAM_REG_ADDR(NUM_PTR_REGS_PER_OUTPUT_QUEUE*adjusted_queue_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));
            this->local_rptr_cleared_update = reinterpret_cast<volatile uint32_t*>(
                STREAM_REG_ADDR(NUM_PTR_REGS_PER_OUTPUT_QUEUE*adjusted_queue_id+1, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));

            // Setting STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX resets the credit register
            *reinterpret_cast<volatile uint32_t*>(STREAM_REG_ADDR(NUM_PTR_REGS_PER_OUTPUT_QUEUE*adjusted_queue_id, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX)) = 0;
            *reinterpret_cast<volatile uint32_t*>(STREAM_REG_ADDR(NUM_PTR_REGS_PER_OUTPUT_QUEUE*adjusted_queue_id+1, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX)) = 0;
        }

        this->remote_wptr_update_addr =
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_INPUT_QUEUE*remote_queue_id,
                            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
        uint32_t adjusted_remote_queue_id = remote_queue_id > 15 ? remote_queue_id - 11 : remote_queue_id;
        this->remote_rptr_sent_update_addr =
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_OUTPUT_QUEUE*adjusted_remote_queue_id,
                            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
        this->remote_rptr_cleared_update_addr =
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_OUTPUT_QUEUE*adjusted_remote_queue_id+1,
                            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);

        this->remote_ready_status_addr = STREAM_REG_ADDR(remote_queue_id, STREAM_REMOTE_SRC_REG_INDEX);
        this->local_ready_status_ptr = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(queue_id, STREAM_REMOTE_SRC_REG_INDEX));
    }

    inline uint8_t get_queue_id() const {
        return this->queue_id;
    }

    // Return physical offset from a logical offset
    // already_wrapped = true means to use 1X queue size
    // already_wrapped = false means to use 2X queue size
    template<bool already_wrapped>
    inline uint32_t get_physical_offset(uint32_t logical_offset) const {
#ifdef POW2_CB
        return logical_offset & this->ptr_offset_mask;
#else
        if constexpr (already_wrapped) {
            return logical_offset >= this->queue_size_words ? logical_offset - this->queue_size_words : logical_offset;
        } else {
            return logical_offset >= this->queue_size_words_2x ? logical_offset - this->queue_size_words_2x : logical_offset;
        }
#endif
    }

    // Calculate distance between left pointer and right pointer. Right pointer should be
    // ahead of left pointer. If it is behind, then that means the pointers have wrapped.
    inline uint32_t calc_distance(uint32_t left, uint32_t right) const {
#ifdef POW2_CB
        return (right - left) & this->queue_size_mask;
#else
        return right >= left ? right - left : this->queue_size_words_2x - left + right;
#endif
    }

    inline uint32_t wrap_logical_offset(uint32_t logical_offset) const {
        // Wrapping logical offset is not required for pow2. it will use AND mask.
#ifdef POW2_CB
        return logical_offset;
#else
        if (logical_offset >= this->queue_size_words_2x) {
            logical_offset -= this->queue_size_words_2x;
        }
        return logical_offset;
#endif
    }

    template<bool input_queue>
    inline uint32_t get_queue_local_wptr() {
        if constexpr (input_queue) {
            const auto wptr_creds = *this->local_wptr_val;
            *this->local_wptr_update = (-wptr_creds) << REMOTE_DEST_BUF_WORDS_FREE_INC;
            this->local_wptr = this->wrap_logical_offset(wptr_creds + this->local_wptr);
        }
        return this->local_wptr;
    }

    template<bool input_queue>
    inline uint32_t get_queue_local_rptr_sent() {
        if constexpr (!input_queue) {
            const auto sent_creds = *this->local_rptr_sent_val;
            *this->local_rptr_sent_update = (-sent_creds) << REMOTE_DEST_BUF_WORDS_FREE_INC;
            this->local_rptr_sent = this->wrap_logical_offset(sent_creds + this->local_rptr_sent);
        }
        return this->local_rptr_sent;
    }

    template<bool input_queue>
    inline uint32_t get_queue_local_rptr_cleared() {
        if constexpr (!input_queue) {
            const auto cleared_creds = *this->local_rptr_cleared_val;
            *this->local_rptr_cleared_update = (-cleared_creds) << REMOTE_DEST_BUF_WORDS_FREE_INC;
            this->local_rptr_cleared = this->wrap_logical_offset(cleared_creds + this->local_rptr_cleared);
        }
        return this->local_rptr_cleared;
    }

    template<bool is_input>
    inline void advance_queue_local_wptr(uint32_t num_words) {
        if constexpr (is_input) {
            *this->local_wptr_update = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
        } else {
            this->local_wptr = this->wrap_logical_offset(num_words + this->local_wptr);
        }
    }

    template<bool is_input>
    inline void advance_queue_local_rptr_sent(uint32_t num_words)  {
        if constexpr (is_input) {
            this->local_rptr_sent = this->wrap_logical_offset(num_words + this->local_rptr_sent);
        } else {
            *this->local_rptr_sent_update = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
        }
    }

    template<bool is_input>
    inline void advance_queue_local_rptr_cleared(uint32_t num_words)  {
        if constexpr (is_input) {
            this->local_rptr_cleared = this->wrap_logical_offset(num_words + this->local_rptr_cleared);
        } else {
            *this->local_rptr_cleared_update = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
        }
    }

    template<bool is_input>
    inline uint32_t get_queue_data_num_words_occupied() {
#ifdef POW2_CB
        return this->calc_distance(this->get_queue_local_rptr_cleared<is_input>(), this->get_queue_local_wptr<is_input>());
#else
        const auto phys_wptr = this->get_physical_offset<false>(this->get_queue_local_wptr<is_input>());
        const auto phys_rptr = this->get_physical_offset<false>(this->get_queue_local_rptr_cleared<is_input>());
        return this->calc_distance(phys_rptr, phys_wptr);
#endif
    }

    template<bool is_input>
    inline uint32_t get_queue_data_num_words_free() {
        return this->queue_size_words - this->get_queue_data_num_words_occupied<is_input>();
    }

    template<bool is_input>
    inline uint32_t get_num_words_written_not_sent() {
#ifdef POW2_CB
        return this->calc_distance(this->get_queue_local_rptr_sent<is_input>(), this->get_queue_local_wptr<is_input>());
#else
        const auto phys_wptr = this->get_physical_offset<false>(this->get_queue_local_wptr<is_input>());
        const auto phys_rptr = this->get_physical_offset<false>(this->get_queue_local_rptr_sent<is_input>());
        return this->calc_distance(phys_rptr, phys_wptr);
#endif
    }

    template<bool is_input>
    inline uint32_t get_queue_wptr_offset_words() {
        return this->get_physical_offset<true>(this->get_queue_local_wptr<is_input>());
    }

    template<bool is_input>
    inline uint32_t get_queue_rptr_sent_offset_words() {
        return this->get_physical_offset<true>(this->get_queue_local_rptr_sent<is_input>());
    }

    template<bool is_input>
    inline uint32_t get_queue_rptr_cleared_offset_words() {
        return this->get_physical_offset<true>(this->get_queue_local_rptr_cleared<is_input>());
    }

    template<bool is_input>
    inline uint32_t get_queue_rptr_sent_addr_bytes() {
        return (this->queue_start_addr_words + this->get_queue_rptr_sent_offset_words<is_input>())*PACKET_WORD_SIZE_BYTES;
    }

    template<bool is_input>
    inline uint32_t get_queue_rptr_cleared_addr_bytes() {
        return (this->queue_start_addr_words + this->get_queue_rptr_cleared_offset_words<is_input>())*PACKET_WORD_SIZE_BYTES;
    }

    template<bool is_input>
    inline uint32_t get_queue_wptr_addr_bytes() {
        return (this->queue_start_addr_words + this->get_queue_wptr_offset_words<is_input>())*PACKET_WORD_SIZE_BYTES;
    }

    template<bool is_input>
    inline uint32_t get_queue_words_before_rptr_sent_wrap() {
        return this->queue_size_words - this->get_queue_rptr_sent_offset_words<is_input>();
    }

    template<bool is_input>
    inline uint32_t get_queue_words_before_rptr_cleared_wrap() {
        return this->queue_size_words - this->get_queue_rptr_cleared_offset_words<is_input>();
    }

    template<bool is_input>
    inline uint32_t get_queue_words_before_wptr_wrap() {
        return this->queue_size_words - this->get_queue_wptr_offset_words<is_input>();
    }

    template<DispatchRemoteNetworkType network_type, bool cb_mode>
    inline void remote_reg_update(uint32_t reg_addr, uint32_t val) {
        if constexpr (network_type == DispatchRemoteNetworkType::NONE || cb_mode == true || network_type == DispatchRemoteNetworkType::DISABLE_QUEUE) {
            return;
        } else if constexpr (network_type == DispatchRemoteNetworkType::ETH) {
            eth_write_remote_reg(reg_addr, val);
        } else {
            const auto dest_addr = get_noc_addr(this->remote_x, this->remote_y, reg_addr);
            noc_inline_dw_write(dest_addr, val);
        }
    }

    template<DispatchRemoteNetworkType network_type, bool cb_mode>
    inline void advance_queue_remote_wptr(uint32_t num_words) {
        uint32_t val = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
        uint32_t reg_addr = this->remote_wptr_update_addr;
        this->remote_reg_update<network_type, cb_mode>(reg_addr, val);
    }

    template<DispatchRemoteNetworkType network_type, bool cb_mode>
    inline void advance_queue_remote_rptr_sent(uint32_t num_words) {
        uint32_t val = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
        uint32_t reg_addr = this->remote_rptr_sent_update_addr;
        this->remote_reg_update<network_type, cb_mode>(reg_addr, val);
    }

    template<DispatchRemoteNetworkType network_type, bool cb_mode>
    inline void advance_queue_remote_rptr_cleared(uint32_t num_words) {
        uint32_t val = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
        uint32_t reg_addr = this->remote_rptr_cleared_update_addr;
        this->remote_reg_update<network_type, cb_mode>(reg_addr, val);
    }

    inline void reset_ready_flag() {
        *this->local_ready_status_ptr = 0;
    }

    template<DispatchRemoteNetworkType network_type, bool cb_mode>
    inline void send_remote_ready_notification() {
        this->remote_reg_update<network_type, cb_mode>(this->remote_ready_status_addr, PACKET_QUEUE_REMOTE_READY_FLAG);
    }

    inline void set_end_remote_queue(uint8_t remote_queue_id, uint8_t remote_x, uint8_t remote_y) {
        this->remote_x = remote_x;
        this->remote_y = remote_y;
        this->remote_ready_status_addr = STREAM_REG_ADDR(remote_queue_id, STREAM_REMOTE_SRC_REG_INDEX);
    }

    template<DispatchRemoteNetworkType network_type, bool cb_mode>
    inline void send_remote_finished_notification() {
        this->remote_reg_update<network_type, cb_mode>(this->remote_ready_status_addr, PACKET_QUEUE_REMOTE_FINISHED_FLAG);
    }

    inline uint32_t get_remote_ready_status() const {
        return *this->local_ready_status_ptr;
    }

    inline bool is_remote_ready() const {
        return this->get_remote_ready_status() == PACKET_QUEUE_REMOTE_READY_FLAG;
    }

    inline bool is_remote_finished() const {
        return this->get_remote_ready_status() == PACKET_QUEUE_REMOTE_FINISHED_FLAG;
    }

    inline uint32_t cb_mode_get_local_sem_val() {
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* local_sem_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<fd_core_type>(this->cb_mode_local_sem_id));
        // semaphore underflow is currently used to signal path teardown with minimal prefetcher changes
        uint32_t val = *local_sem_addr;
        if (val & 0x80000000) { // hardcoded frmo cq_dispatch/prefetch kernel
            val &= 0x7fffffff;
            *this->local_ready_status_ptr = PACKET_QUEUE_REMOTE_FINISHED_FLAG;
        }
        return val;
    }

    inline bool cb_mode_local_sem_downstream_complete() {
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* local_sem_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<fd_core_type>(this->cb_mode_local_sem_id));
        // semaphore underflow is currently used to signal path teardown with minimal prefetcher changes
        uint32_t val = *local_sem_addr;
        return (val & 0x80000000);
    }

    inline void cb_mode_inc_local_sem_val(uint32_t val) {
        if (!val) return;
        uint32_t sem_l1_addr = get_semaphore<fd_core_type>(this->cb_mode_local_sem_id);
        uint64_t sem_noc_addr = get_noc_addr(sem_l1_addr);
        noc_semaphore_inc(sem_noc_addr, val);
        noc_async_atomic_barrier();
    }

    inline void cb_mode_inc_remote_sem_val(uint32_t val) {
        if (!val) return;
        uint32_t sem_l1_addr = get_semaphore<fd_core_type>(this->cb_mode_remote_sem_id);
        uint64_t sem_noc_addr = get_noc_addr(remote_x, remote_y, sem_l1_addr);
        noc_semaphore_inc(sem_noc_addr, val);
        // noc_async_atomic_barrier();
    }

    template<bool is_input>
    inline uint32_t cb_mode_rptr_sent_advance_page_align() {
        uint32_t rptr_val = this->get_queue_local_rptr_sent<is_input>();
        uint32_t page_size_words_mask = this->cb_mode_page_size_words - 1;
        uint32_t num_words_past_page_boundary = rptr_val & page_size_words_mask;
        uint32_t input_pad_words_skipped = 0;
        if (num_words_past_page_boundary > 0) {
            input_pad_words_skipped = this->cb_mode_page_size_words - num_words_past_page_boundary;
            this->advance_queue_local_rptr_sent<is_input>(input_pad_words_skipped);
        }
        return input_pad_words_skipped;
    }

    template<bool is_input>
    inline void cb_mode_local_sem_wptr_update() {
        const auto local_sem_val = this->cb_mode_get_local_sem_val();
        if (local_sem_val > 0) {
            this->advance_queue_local_wptr<is_input>(
                this->cb_mode_page_size_words * local_sem_val
            );
            this->cb_mode_inc_local_sem_val(-local_sem_val);
        }
    }

    template<bool is_input>
    inline void cb_mode_local_sem_rptr_cleared_update() {
        const auto local_sem_val = this->cb_mode_get_local_sem_val();
        if (local_sem_val > 0) {
            this->advance_queue_local_rptr_cleared<is_input>(
                this->cb_mode_page_size_words * local_sem_val
            );
            this->cb_mode_inc_local_sem_val(-local_sem_val);
        }
    }

    void dprint_object() {
#ifdef DEBUG_PQ
        DPRINT << "  id: " << DEC() << static_cast<uint32_t>(this->queue_id) << ENDL();
        DPRINT << "  start_addr: 0x" << HEX() << static_cast<uint32_t>(this->queue_start_addr_words*PACKET_WORD_SIZE_BYTES) << ENDL();
        DPRINT << "  size_bytes: 0x" << HEX() << static_cast<uint32_t>(this->queue_size_words*PACKET_WORD_SIZE_BYTES) << ENDL();
        DPRINT << "  remote_x: " << DEC() << static_cast<uint32_t>(this->remote_x) << ENDL();
        DPRINT << "  remote_y: " << DEC() << static_cast<uint32_t>(this->remote_y) << ENDL();
        DPRINT << "  remote_queue_id: " << DEC() << static_cast<uint32_t>(this->remote_queue_id) << ENDL();
        DPRINT << "  remote_update_network_type: " << DEC() << static_cast<uint32_t>(this->remote_update_network_type) << ENDL();
        DPRINT << "  ready_status: 0x" << HEX() << this->get_remote_ready_status() << ENDL();
        DPRINT << "  local_wptr: 0x" << HEX() << this->get_queue_local_wptr() << ENDL();
        DPRINT << "  local_rptr_sent: 0x" << HEX() << this->get_queue_local_rptr_sent() << ENDL();
        DPRINT << "  local_rptr_cleared: 0x" << HEX() << this->get_queue_local_rptr_cleared() << ENDL();
#endif
    }
};


class packet_input_queue_state_t : public packet_queue_state_t {

protected:

    bool curr_packet_valid;
    tt_l1_ptr dispatch_packet_header_t* curr_packet_header_ptr;
    uint16_t curr_packet_src;
    uint16_t curr_packet_dest;
    uint32_t curr_packet_size_words;
    uint32_t curr_packet_words_sent;
    uint32_t curr_packet_tag;
    uint16_t curr_packet_flags;
    uint16_t end_of_cmd;

    uint32_t packetizer_page_words_cleared;

    inline void advance_next_packet() {
        if(this->get_queue_data_num_words_available_to_send() > 0) {
            tt_l1_ptr dispatch_packet_header_t* next_packet_header_ptr =
                reinterpret_cast<tt_l1_ptr dispatch_packet_header_t*>(
                    (this->queue_start_addr_words + this->get_queue_rptr_sent_offset_words<true>())*PACKET_WORD_SIZE_BYTES
                );
            this->curr_packet_header_ptr = next_packet_header_ptr;
            uint32_t packet_size_and_flags = next_packet_header_ptr->packet_size_bytes;
            uint32_t packet_size_bytes = packet_size_and_flags & 0xFFFFFFFE;
            this->end_of_cmd = !(packet_size_and_flags & 1);
            this->curr_packet_size_words = packet_size_bytes/PACKET_WORD_SIZE_BYTES;
            if (packet_size_bytes % PACKET_WORD_SIZE_BYTES) {
                this->curr_packet_size_words++;
            }
            if (this->cb_mode) {
                // prefetcher has size in bytes
                next_packet_header_ptr->packet_dest = this->curr_packet_dest;
                next_packet_header_ptr->packet_src = this->curr_packet_src;
                next_packet_header_ptr->tag = this->curr_packet_tag;
                next_packet_header_ptr->packet_flags = this->curr_packet_flags;
            } else {
                this->curr_packet_dest = next_packet_header_ptr->packet_dest;
                this->curr_packet_src = next_packet_header_ptr->packet_src;
                this->curr_packet_tag = next_packet_header_ptr->tag;
                this->curr_packet_flags = next_packet_header_ptr->packet_flags;
            }
            this->curr_packet_words_sent = 0;
            this->curr_packet_valid = true;
       }
    }

public:

    void init(uint8_t queue_id,
              uint32_t queue_start_addr_words,
              uint32_t queue_size_words,
              uint8_t remote_x,
              uint8_t remote_y,
              uint8_t remote_queue_id,
              DispatchRemoteNetworkType remote_update_network_type,
              bool packetizer_input = false,
              uint8_t packetizer_input_log_page_size = 0,
              uint8_t packetizer_input_sem_id = 0,
              uint8_t packetizer_input_remote_sem_id = 0,
              uint16_t packetizer_input_src = 0,
              uint16_t packetizer_input_dest = 0) {

        packet_queue_state_t::init(queue_id, queue_start_addr_words, queue_size_words, true,
                                   remote_x, remote_y, remote_queue_id, remote_update_network_type,
                                   packetizer_input, packetizer_input_sem_id,
                                   packetizer_input_remote_sem_id,
                                   packetizer_input_log_page_size);

        tt_l1_ptr uint32_t* queue_ptr =
            reinterpret_cast<tt_l1_ptr uint32_t*>(queue_start_addr_words*PACKET_WORD_SIZE_BYTES);

        this->packetizer_page_words_cleared = 0;

        if (packetizer_input) {
            this->curr_packet_src = packetizer_input_src;
            this->curr_packet_dest = packetizer_input_dest;
            this->curr_packet_flags = 0;
            this->curr_packet_tag = 0xabcd;
        }

        this->curr_packet_valid = false;
        this->reset_ready_flag();
    }

    inline uint16_t get_end_of_cmd() const {
        return this->end_of_cmd;
    }

    inline uint32_t get_queue_data_num_words_available_to_send() {
        return this->get_num_words_written_not_sent<true>();
    }

    template<bool cb_mode>
    inline bool get_curr_packet_valid() {
        if constexpr (cb_mode) {
            this->cb_mode_local_sem_wptr_update<true>();
        }
        if (!this->curr_packet_valid && (this->get_queue_data_num_words_available_to_send() > 0)){
            this->advance_next_packet();
        }
        return this->curr_packet_valid;
    }

    inline tt_l1_ptr dispatch_packet_header_t* get_curr_packet_header_ptr() {
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        return this->curr_packet_header_ptr;
    }

    inline uint32_t get_curr_packet_dest() {
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        return this->curr_packet_dest;
    }

    inline uint32_t get_curr_packet_src() {
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        return this->curr_packet_src;
    }

    inline uint32_t get_curr_packet_size_words() {
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        return this->curr_packet_size_words;
    }

    inline uint32_t get_curr_packet_words_remaining() {
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        return this->curr_packet_size_words - this->curr_packet_words_sent;
    }

    inline uint32_t get_curr_packet_tag() {
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        return this->curr_packet_tag;
    }

    inline uint32_t get_curr_packet_flags() {
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        return this->curr_packet_flags;
    }

    inline bool partial_packet_sent() const {
        return this->curr_packet_valid && (this->curr_packet_words_sent > 0);
    }

    inline bool curr_packet_start() const {
        return this->curr_packet_valid && (this->curr_packet_words_sent == 0);
    }

    inline bool input_queue_full_packet_available_to_send(uint32_t& num_words_available_to_send) {
        num_words_available_to_send = this->get_queue_data_num_words_available_to_send();
        if (num_words_available_to_send == 0) {
            return false;
        }
        return num_words_available_to_send >= this->get_curr_packet_words_remaining();
    }

    template<bool cb_mode>
    inline uint32_t input_queue_curr_packet_num_words_available_to_send() {
        if constexpr (cb_mode) {
            this->cb_mode_local_sem_wptr_update<true>();
        }
        uint32_t num_words = this->get_queue_data_num_words_available_to_send();
        if (num_words == 0) {
            return 0;
        }
        num_words = std::min(num_words, this->get_curr_packet_words_remaining());
        return num_words;
    }

    // returns the number of words skipped for page padding if in packetizer mode
    template<DispatchRemoteNetworkType network_type, bool cb_mode>
    inline uint32_t input_queue_advance_words_sent(uint32_t num_words) {
        uint32_t input_pad_words_skipped = 0;
        if (num_words > 0) {
            this->advance_queue_local_rptr_sent<true>(num_words);
            this->advance_queue_remote_rptr_sent<network_type, cb_mode>(num_words);
            this->curr_packet_words_sent += num_words;
            uint32_t curr_packet_words_remaining = this->get_curr_packet_words_remaining();
            if (curr_packet_words_remaining == 0) {
                if constexpr (cb_mode) {
                    input_pad_words_skipped = this->cb_mode_rptr_sent_advance_page_align<true>();
                }
                this->curr_packet_valid = false;
                this->advance_next_packet();
            }
        }
        return input_pad_words_skipped;
    }

    template<DispatchRemoteNetworkType network_type, bool cb_mode>
    inline void input_queue_advance_words_cleared(uint32_t num_words) {
        if (num_words > 0) {
            this->advance_queue_local_rptr_cleared<true>(num_words);
            this->advance_queue_remote_rptr_cleared<network_type, cb_mode>(num_words);
            if constexpr (cb_mode) {
                this->packetizer_page_words_cleared += num_words;
                uint32_t remote_sem_inc = 0;
                while (this->packetizer_page_words_cleared >= this->cb_mode_page_size_words) {
                    remote_sem_inc++;
                    this->packetizer_page_words_cleared -= this->cb_mode_page_size_words;
                }
                this->cb_mode_inc_remote_sem_val(remote_sem_inc);
            }
        }
    }

    void dprint_object() {
#ifdef DEBUG_PQ
        DPRINT << "Input queue:" << ENDL();
        packet_queue_state_t::dprint_object();
        DPRINT << "  packet_valid: " << DEC() << static_cast<uint32_t>(this->curr_packet_valid) << ENDL();
        DPRINT << "  packet_tag: 0x" << HEX() << static_cast<uint32_t>(this->curr_packet_tag) << ENDL();
        DPRINT << "  packet_src: 0x" << HEX() << static_cast<uint32_t>(this->curr_packet_src) << ENDL();
        DPRINT << "  packet_dest: 0x" << HEX() << static_cast<uint32_t>(this->curr_packet_dest) << ENDL();
        DPRINT << "  packet_flags: 0x" << HEX() << static_cast<uint32_t>(this->curr_packet_flags) << ENDL();
        DPRINT << "  packet_size_words: " << DEC() << static_cast<uint32_t>(this->curr_packet_size_words) << ENDL();
        DPRINT << "  packet_words_sent: " << DEC() << static_cast<uint32_t>(this->curr_packet_words_sent) << ENDL();
#endif
    }

};


class packet_output_queue_state_t : public packet_queue_state_t {

protected:

    uint32_t output_max_send_words;

    uint32_t unpacketizer_page_words_sent;
    bool unpacketizer_remove_header;

    struct {

        packet_input_queue_state_t* input_queue_array;
        uint32_t input_queue_words_in_flight[2*MAX_SWITCH_FAN_IN];

        uint32_t* curr_input_queue_words_in_flight;
        uint32_t* prev_input_queue_words_in_flight;
        uint32_t curr_output_total_words_in_flight;
        uint32_t prev_output_total_words_in_flight;

        uint8_t num_input_queues;

        void init(packet_input_queue_state_t* input_queue_array, uint32_t num_input_queues) {
            this->num_input_queues = num_input_queues;
            this->input_queue_array = input_queue_array;
            this->curr_input_queue_words_in_flight = &(this->input_queue_words_in_flight[0]);
            this->prev_input_queue_words_in_flight = &(this->input_queue_words_in_flight[MAX_SWITCH_FAN_IN]);
            this->curr_output_total_words_in_flight = 0;
            this->prev_output_total_words_in_flight = 0;
            for (uint32_t i = 0; i < MAX_SWITCH_FAN_IN; i++) {
                this->curr_input_queue_words_in_flight[i] = 0;
                this->prev_input_queue_words_in_flight[i] = 0;
            }
        }

        inline uint32_t get_curr_output_total_words_in_flight() const {
            return this->curr_output_total_words_in_flight;
        }

        inline uint32_t get_prev_output_total_words_in_flight() const {
            return this->prev_output_total_words_in_flight;
        }

        template<typename input_network_types, typename input_cb_modes>
        inline uint32_t prev_words_in_flight_flush() {
            uint32_t words_flushed = this->prev_output_total_words_in_flight;
            if (words_flushed > 0) {
                process_queues<input_network_types, input_cb_modes>([&]<auto network_type, auto cb_mode, auto sequence_i>(auto) -> bool {
                    this->input_queue_array[sequence_i].template input_queue_advance_words_cleared<network_type, cb_mode>(this->prev_input_queue_words_in_flight[sequence_i]);
                    this->prev_input_queue_words_in_flight[sequence_i] = 0;
                    return true;
                });
            }

            uint32_t* tmp = this->prev_input_queue_words_in_flight;
            this->prev_input_queue_words_in_flight = this->curr_input_queue_words_in_flight;
            this->curr_input_queue_words_in_flight = tmp;
            this->prev_output_total_words_in_flight = this->curr_output_total_words_in_flight;
            this->curr_output_total_words_in_flight = 0;

            return words_flushed;
        }


        template<DispatchRemoteNetworkType input_network_type, bool input_cb_mode>
        inline void register_words_in_flight(uint32_t input_queue_id, uint32_t num_words) {
            uint32_t input_pad_words_skipped = this->input_queue_array[input_queue_id].input_queue_advance_words_sent<input_network_type, input_cb_mode>(num_words);
            this->curr_input_queue_words_in_flight[input_queue_id] += (num_words + input_pad_words_skipped);
            this->curr_output_total_words_in_flight += num_words;
        }

        void dprint_object() {
#ifdef DEBUG_PQ
            DPRINT << "  curr_output_total_words_in_flight: " << DEC() << this->curr_output_total_words_in_flight << ENDL();
            for (uint32_t j = 0; j < MAX_SWITCH_FAN_IN; j++) {
                DPRINT << "       from input queue id " << DEC() <<
                            (uint32_t)this->input_queue_array[j].get_queue_id() << ": "
                            << DEC() << this->curr_input_queue_words_in_flight[j]
                            << ENDL();
            }
            DPRINT << "  prev_output_total_words_in_flight: " << DEC() << this->prev_output_total_words_in_flight << ENDL();
            for (uint32_t j = 0; j < MAX_SWITCH_FAN_IN; j++) {
                DPRINT << "       from input queue id " << DEC() <<
                            (uint32_t)this->input_queue_array[j].get_queue_id() << ": "
                            << DEC() << this->prev_input_queue_words_in_flight[j]
                            << ENDL();
            }
#endif
        }

    } input_queue_status;

public:

    void init(uint8_t queue_id,
              uint32_t queue_start_addr_words,
              uint32_t queue_size_words,
              uint8_t remote_x,
              uint8_t remote_y,
              uint8_t remote_queue_id,
              DispatchRemoteNetworkType remote_update_network_type,
              packet_input_queue_state_t* input_queue_array,
              uint8_t num_input_queues,
              bool unpacketizer_output = false,
              uint16_t unpacketizer_output_log_page_size = 0,
              uint8_t unpacketizer_output_sem_id = 0,
              uint8_t unpacketizer_output_remote_sem_id = 0,
              bool unpacketizer_output_remove_header = false) {

        packet_queue_state_t::init(queue_id, queue_start_addr_words, queue_size_words, false,
                                   remote_x, remote_y, remote_queue_id, remote_update_network_type,
                                   unpacketizer_output, unpacketizer_output_sem_id,
                                   unpacketizer_output_remote_sem_id,
                                   unpacketizer_output_log_page_size);

        this->unpacketizer_remove_header = unpacketizer_output_remove_header;
        this->unpacketizer_page_words_sent = 0;
        this->input_queue_status.init(input_queue_array, num_input_queues);
        switch (remote_update_network_type) {
            case DispatchRemoteNetworkType::DISABLE_QUEUE:
            case DispatchRemoteNetworkType::NONE:
                this->output_max_send_words = 0;
                break;
            case DispatchRemoteNetworkType::ETH:
                this->output_max_send_words = DEFAULT_MAX_ETH_SEND_WORDS;
                break;
            case DispatchRemoteNetworkType::NOC0:
            case DispatchRemoteNetworkType::NOC1:
                this->output_max_send_words = DEFAULT_MAX_NOC_SEND_WORDS;
                break;
            default:
                ASSERT(false);
        }

        this->reset_ready_flag();
    }

    template<DispatchRemoteNetworkType network_type>
    inline void send_data_to_remote(uint32_t src_addr, uint32_t dest_addr, uint32_t num_words) {
        if constexpr (network_type == DispatchRemoteNetworkType::NONE) return;
        else {
            if (!num_words) return;

            if constexpr (network_type == DispatchRemoteNetworkType::ETH) {
                internal_::eth_send_packet(0, src_addr/PACKET_WORD_SIZE_BYTES, dest_addr/PACKET_WORD_SIZE_BYTES, num_words);
            } else {
                const auto noc_dest_addr = get_noc_addr(this->remote_x, this->remote_y, dest_addr);
                noc_async_write(src_addr, noc_dest_addr, num_words*PACKET_WORD_SIZE_BYTES);
            }
        }
    }

    template<bool cb_mode, typename input_network_types, typename input_cb_modes>
    inline uint32_t prev_words_in_flight_check_flush() {
        if constexpr (cb_mode) {
            uint32_t words_written_not_sent = this->get_num_words_written_not_sent<false>();
            noc_async_writes_flushed();
            this->advance_queue_local_rptr_sent<false>(words_written_not_sent);
            uint32_t words_flushed = this->input_queue_status.prev_words_in_flight_flush<input_network_types, input_cb_modes>();
            this->cb_mode_local_sem_rptr_cleared_update<false>();
            return words_flushed;
        }
        else if (this->get_num_words_written_not_sent<false>() <= this->input_queue_status.get_curr_output_total_words_in_flight()) {
            return this->input_queue_status.prev_words_in_flight_flush<input_network_types, input_cb_modes>();
        }
        else {
            return 0;
        }
    }

    template<bool cb_mode, typename input_network_types, typename input_cb_modes>
    bool output_barrier(uint32_t timeout_cycles = 0) {
        uint32_t start_timestamp = 0;
        if (timeout_cycles > 0) {
            start_timestamp = get_timestamp_32b();
        }

        if constexpr (cb_mode) {
           noc_async_writes_flushed();
        }

        while (this->get_queue_data_num_words_occupied<false>() > 0) {
            if constexpr (cb_mode) {
                this->cb_mode_local_sem_rptr_cleared_update<false>();
                if (this->cb_mode_local_sem_downstream_complete()) {
                    // There is no guaranteed that dispatch_h will increment semaphore for all commmands
                    // (specifically the final terminate command).
                    // So just clear whatever remains once the completion signal is received.
                    uint32_t words_occupied = this->get_queue_data_num_words_occupied<false>();
                    this->advance_queue_local_rptr_cleared<false>(words_occupied);
                }
            }
            if (timeout_cycles > 0) {
                uint32_t cycles_elapsed = get_timestamp_32b() - start_timestamp;
                if (cycles_elapsed > timeout_cycles) {
                    return false;
                }
            }
        }
        this->input_queue_status.prev_words_in_flight_flush<input_network_types, input_cb_modes>();
        this->input_queue_status.prev_words_in_flight_flush<input_network_types, input_cb_modes>();
        return true;
    }

    template<bool input_queue_cb_mode>
    inline uint32_t get_num_words_to_send(uint32_t input_queue_index) {
        packet_input_queue_state_t* input_queue_ptr = &(this->input_queue_status.input_queue_array[input_queue_index]);

        uint32_t num_words_available_in_input = input_queue_ptr->input_queue_curr_packet_num_words_available_to_send<input_queue_cb_mode>();
        uint32_t num_words_before_input_rptr_wrap = input_queue_ptr->get_queue_words_before_rptr_sent_wrap<true>();
        num_words_available_in_input = std::min(num_words_available_in_input, num_words_before_input_rptr_wrap);
        uint32_t num_words_free_in_output = this->get_queue_data_num_words_free<false>();
        uint32_t num_words_to_forward = std::min(num_words_available_in_input, num_words_free_in_output);
        uint32_t output_buf_words_before_wptr_wrap = this->get_queue_words_before_wptr_wrap<false>();

        num_words_to_forward = std::min(num_words_to_forward, output_buf_words_before_wptr_wrap);
        num_words_to_forward = std::min(num_words_to_forward, this->output_max_send_words);

        return num_words_to_forward;
    }

    template<DispatchRemoteNetworkType output_network_type, bool output_cb_mode, DispatchRemoteNetworkType input_network_type, bool input_cb_mode>
    inline uint32_t forward_data_from_input(uint32_t input_queue_index, bool& full_packet_sent, uint16_t end_of_cmd) {
        packet_input_queue_state_t* input_queue_ptr = &(this->input_queue_status.input_queue_array[input_queue_index]);
        uint32_t num_words_to_forward = this->get_num_words_to_send<input_cb_mode>(input_queue_index);
        full_packet_sent = (num_words_to_forward == input_queue_ptr->get_curr_packet_words_remaining());
        if (num_words_to_forward == 0) {
            return 0;
        }

        if (this->unpacketizer_remove_header && input_queue_ptr->curr_packet_start()) {
            num_words_to_forward--;
            this->input_queue_status.register_words_in_flight<input_network_type, input_cb_mode>(input_queue_index, 1);
            if (num_words_to_forward == 0) {
                return 0;
            }
        }

        const auto src_addr = input_queue_ptr->get_queue_rptr_sent_addr_bytes<true>();
        const auto dest_addr = this->get_queue_wptr_addr_bytes<false>();

        this->send_data_to_remote<output_network_type>(src_addr, dest_addr, num_words_to_forward);
        this->input_queue_status.register_words_in_flight<input_network_type, input_cb_mode>(input_queue_index, num_words_to_forward);
        this->advance_queue_local_wptr<false>(num_words_to_forward);

        if constexpr (!output_cb_mode) {
            this->advance_queue_remote_wptr<output_network_type, output_cb_mode>(num_words_to_forward);
        } else {
            this->unpacketizer_page_words_sent += num_words_to_forward;
            if (full_packet_sent && end_of_cmd) {
                uint32_t unpacketizer_page_words_sent_past_page_bound =
                    this->unpacketizer_page_words_sent & (this->cb_mode_page_size_words - 1);
                if (unpacketizer_page_words_sent_past_page_bound > 0) {
                    uint32_t pad_words = this->cb_mode_page_size_words - unpacketizer_page_words_sent_past_page_bound;
                    this->unpacketizer_page_words_sent += pad_words;
                    this->advance_queue_local_wptr<false>(pad_words);
                }
            }
            uint32_t remote_sem_inc = 0;
            while (this->unpacketizer_page_words_sent >= this->cb_mode_page_size_words) {
                this->unpacketizer_page_words_sent -= this->cb_mode_page_size_words;
                remote_sem_inc++;
            }
            this->cb_mode_inc_remote_sem_val(remote_sem_inc);
        }

        return num_words_to_forward;
    }

    void dprint_object() {
#ifdef DEBUG_PQ
        DPRINT << "Output queue:" << ENDL();
        packet_queue_state_t::dprint_object();
        this->input_queue_status.dprint_object();
#endif
    }
};


// Wait for all input and output queues and their remotes to signal Ready on the remote ready status
template <
    typename InputNetworkTypeSequence,
    typename InputCBSequence,
    typename OutputNetworkTypeSequence,
    typename OutputCBSequence>
bool wait_all_input_output_ready(packet_input_queue_state_t* input_queue_array, packet_output_queue_state_t* output_queue_array, uint32_t timeout_cycles = 0) {
    static_assert(InputNetworkTypeSequence::size == InputCBSequence::size);
    static_assert(OutputNetworkTypeSequence::size == OutputCBSequence::size);

    bool src_ready[InputNetworkTypeSequence::size];
    bool dest_ready[OutputNetworkTypeSequence::size];
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
            [&]<auto network_type, auto cb_mode, auto>(auto index) -> bool {
                if (!src_ready[index]) {
                    src_ready[index] = cb_mode || input_queue_array[index].is_remote_ready();
                    if (!src_ready[index]) {
                        input_queue_array[index].template send_remote_ready_notification<network_type, cb_mode>();
                        all_src_dest_ready = false;
                    } else {
                        // handshake with src complete
                    }
                }

                return true;  // keep looping through other queues
            });

        // checking output queues
        process_queues<OutputNetworkTypeSequence, OutputCBSequence>(
            [&]<auto network_type, auto cb_mode, auto>(auto index) -> bool {
                if (!dest_ready[index]) {
                    dest_ready[index] = cb_mode || output_queue_array[index].is_remote_ready();
                    if (dest_ready[index]) {
                        output_queue_array[index].template send_remote_ready_notification<network_type, cb_mode>();
                    } else {
                        all_src_dest_ready = false;
                    }
                }
                return true;
            });

#if defined(COMPILE_FOR_ERISC)
        // Just for init purposes it's ok to keep context switching
        internal_::risc_context_switch();
#endif
    }
    return true;
}
