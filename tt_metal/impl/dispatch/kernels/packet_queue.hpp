// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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


class packet_queue_state_t;
class packet_input_queue_state_t;
class packet_output_queue_state_t;

class packet_queue_state_t {

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
    uint32_t local_wptr;
    uint32_t local_rptr_sent;
    uint32_t local_rptr_cleared;

protected:
    volatile uint32_t* local_rptr_sent_reset;
    volatile uint32_t* local_rptr_cleared_reset;

    bool cb_mode;
    uint32_t cb_mode_page_size_words;
    uint8_t cb_mode_local_sem_id;
    uint8_t cb_mode_remote_sem_id;

public:

    uint8_t queue_id;
    uint32_t queue_start_addr_words;
    uint32_t queue_size_words;
    uint32_t ptr_offset_mask;
    uint32_t queue_size_mask;

    bool queue_is_input;

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
        this->queue_is_input = queue_is_input;
        this->remote_x = remote_x;
        this->remote_y = remote_y;
        this->remote_queue_id = remote_queue_id;
        this->remote_update_network_type = remote_update_network_type;

        this->cb_mode = cb_mode;
        this->cb_mode_local_sem_id = cb_mode_local_sem_id;
        this->cb_mode_remote_sem_id = cb_mode_remote_sem_id;
        this->cb_mode_page_size_words = (((uint32_t)0x1) << cb_mode_log_page_size)/PACKET_WORD_SIZE_BYTES;

        // Misc. register definitions below.

        // For read/write pointers, we use stream credit registers with auto-increment.
        // Pointers are in 16B units, and we assume buffer size is power of 2 so we get
        // automatic wrapping. (If needed, we can fix the pointer advance functions later
        // to handle non-power-of-2 buffer sizes.)

        // For source/destination ready synchronization signals, we use misc. registers in
        // streams that behave like scratch registers and are reset to 0.

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
            this->local_rptr_sent_reset = reinterpret_cast<volatile uint32_t*>(
                STREAM_REG_ADDR(NUM_PTR_REGS_PER_OUTPUT_QUEUE*adjusted_queue_id, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX));
            this->local_rptr_cleared_reset = reinterpret_cast<volatile uint32_t*>(
                STREAM_REG_ADDR(NUM_PTR_REGS_PER_OUTPUT_QUEUE*adjusted_queue_id+1, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX));
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

    inline uint32_t get_queue_local_wptr() const {
        return *this->local_wptr_val;
    }

    inline void advance_queue_local_wptr(uint32_t num_words) {
        if (this->queue_is_input) {
            *this->local_wptr_update = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
        } else {
            this->local_wptr += num_words;
        }
    }

    inline uint32_t get_queue_local_rptr_sent() const {
        return *this->local_rptr_sent_val;
    }

    inline uint32_t get_queue_local_rptr_cleared() const {
        return *this->local_rptr_cleared_val;
    }

    inline void advance_queue_local_rptr_sent(uint32_t num_words)  {
        if (this->queue_is_input) {
            this->local_rptr_sent += num_words;
        } else {
        *this->local_rptr_sent_update = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
        }
    }

    inline void advance_queue_local_rptr_cleared(uint32_t num_words)  {

        if (this->queue_is_input) {
            this->local_rptr_cleared += num_words;
        } else {
        *this->local_rptr_cleared_update = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
        }
    }

    inline uint32_t get_queue_data_num_words_occupied() const {
        return (this->get_queue_local_wptr() - this->get_queue_local_rptr_cleared()) & this->queue_size_mask;
    }

    inline uint32_t get_queue_data_num_words_free() const {
        return this->queue_size_words - this->get_queue_data_num_words_occupied();
    }

    inline uint32_t get_num_words_sent_not_cleared() const {
        return (this->get_queue_local_rptr_sent() - this->get_queue_local_rptr_cleared()) & this->queue_size_mask;
    }

    inline uint32_t get_num_words_written_not_sent() const {
        return (this->get_queue_local_wptr() - this->get_queue_local_rptr_sent()) & this->queue_size_mask;
    }

    inline uint32_t get_queue_wptr_offset_words() const {
        return this->get_queue_local_wptr() & this->ptr_offset_mask;
    }

    inline uint32_t get_queue_rptr_sent_offset_words() const {
        return this->get_queue_local_rptr_sent() & this->ptr_offset_mask;
    }

    inline uint32_t get_queue_rptr_cleared_offset_words() const {
        return this->get_queue_local_rptr_cleared() & this->ptr_offset_mask;
    }

    inline uint32_t get_queue_rptr_sent_addr_bytes() const {
        return (this->queue_start_addr_words + this->get_queue_rptr_sent_offset_words())*PACKET_WORD_SIZE_BYTES;
    }

    inline uint32_t get_queue_rptr_cleared_addr_bytes() const {
        return (this->queue_start_addr_words + this->get_queue_rptr_cleared_offset_words())*PACKET_WORD_SIZE_BYTES;
    }

    inline uint32_t get_queue_wptr_addr_bytes() const {
        return (this->queue_start_addr_words + this->get_queue_wptr_offset_words())*PACKET_WORD_SIZE_BYTES;
    }

    inline uint32_t get_queue_words_before_rptr_sent_wrap() const {
        return queue_size_words - this->get_queue_rptr_sent_offset_words();
    }

    inline uint32_t get_queue_words_before_rptr_cleared_wrap() const {
        return queue_size_words - this->get_queue_rptr_cleared_offset_words();
    }

    inline uint32_t get_queue_words_before_wptr_wrap() const {
        return queue_size_words - this->get_queue_wptr_offset_words();
    }

    inline void remote_reg_update(uint32_t reg_addr, uint32_t val) {

        if ((this->remote_update_network_type == DispatchRemoteNetworkType::NONE) || this->cb_mode) {
            return;
        }
        else if (this->remote_update_network_type == DispatchRemoteNetworkType::ETH) {
            eth_write_remote_reg(reg_addr, val);
        } else {
            uint64_t dest_addr = NOC_XY_ADDR(this->remote_x, this->remote_y, reg_addr);
            noc_inline_dw_write(dest_addr, val);
        }
    }

    inline void advance_queue_remote_wptr(uint32_t num_words) {
        uint32_t val = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
        uint32_t reg_addr = this->remote_wptr_update_addr;
        this->remote_reg_update(reg_addr, val);
    }

    inline void advance_queue_remote_rptr_sent(uint32_t num_words) {
        uint32_t val = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
        uint32_t reg_addr = this->remote_rptr_sent_update_addr;
        this->remote_reg_update(reg_addr, val);
    }

    inline void advance_queue_remote_rptr_cleared(uint32_t num_words) {
        uint32_t val = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
        uint32_t reg_addr = this->remote_rptr_cleared_update_addr;
        this->remote_reg_update(reg_addr, val);
    }

    inline void reset_ready_flag() {
        *this->local_ready_status_ptr = 0;
    }

    inline void send_remote_ready_notification() {
        this->remote_reg_update(this->remote_ready_status_addr,
                                PACKET_QUEUE_REMOTE_READY_FLAG);
    }

    inline void set_remote_ready_status_addr(uint8_t remote_queue_id) {
        this->remote_ready_status_addr = STREAM_REG_ADDR(remote_queue_id, STREAM_REMOTE_SRC_REG_INDEX);
    }

    inline void send_remote_finished_notification() {
        this->remote_reg_update(this->remote_ready_status_addr,
                                PACKET_QUEUE_REMOTE_FINISHED_FLAG);
    }

    inline bool is_remote_ready() const {
        return *this->local_ready_status_ptr == PACKET_QUEUE_REMOTE_READY_FLAG;
    }

    inline uint32_t get_remote_ready_status() const {
        return *this->local_ready_status_ptr;
    }

    inline bool is_remote_finished() const {
        return *this->local_ready_status_ptr == PACKET_QUEUE_REMOTE_FINISHED_FLAG;
    }

    inline uint32_t cb_mode_get_local_sem_val() {
        if (!this->cb_mode) {
            return 0;
        }
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* local_sem_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<fd_core_type>(this->cb_mode_local_sem_id));
        // semaphore underflow is currently used to signal path teardown with minimal prefetcher changes
        uint32_t val = *local_sem_addr;
        if (val & 0x80000000) {
            val &= 0x7fffffff;
            *this->local_ready_status_ptr = PACKET_QUEUE_REMOTE_FINISHED_FLAG;
        }
        return val;
    }

    inline bool cb_mode_local_sem_downstream_complete() {
        if (!this->cb_mode) {
            return false;
        }
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* local_sem_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<fd_core_type>(this->cb_mode_local_sem_id));
        // semaphore underflow is currently used to signal path teardown with minimal prefetcher changes
        uint32_t val = *local_sem_addr;
        return (val & 0x80000000);
    }

    inline void cb_mode_inc_local_sem_val(uint32_t val) {
        if (this->cb_mode) {
            uint32_t sem_l1_addr = get_semaphore<fd_core_type>(this->cb_mode_local_sem_id);
            uint64_t sem_noc_addr = get_noc_addr(sem_l1_addr);
            noc_semaphore_inc(sem_noc_addr, val);
            noc_async_atomic_barrier();
        }
    }

    inline void cb_mode_inc_remote_sem_val(uint32_t val) {
        uint32_t sem_l1_addr = get_semaphore<fd_core_type>(this->cb_mode_remote_sem_id);
        uint64_t sem_noc_addr = get_noc_addr(remote_x, remote_y, sem_l1_addr);
        if (this->cb_mode && (val > 0)) {
            noc_semaphore_inc(sem_noc_addr, val);
        }
    }

    inline uint32_t cb_mode_rptr_sent_advance_page_align() {
        uint32_t rptr_val = this->get_queue_local_rptr_sent();
        uint32_t page_size_words_mask = this->cb_mode_page_size_words - 1;
        uint32_t num_words_past_page_boundary = rptr_val & page_size_words_mask;
        uint32_t input_pad_words_skipped = 0;
        if (num_words_past_page_boundary > 0) {
            input_pad_words_skipped = this->cb_mode_page_size_words - num_words_past_page_boundary;
            this->advance_queue_local_rptr_sent(input_pad_words_skipped);
        }
        return input_pad_words_skipped;
    }

    inline void cb_mode_local_sem_wptr_update() {
        uint32_t local_sem_val = this->cb_mode_get_local_sem_val();
        for (uint32_t i = 0; i < local_sem_val; i++) {
            this->advance_queue_local_wptr(this->cb_mode_page_size_words);
        }
        this->cb_mode_inc_local_sem_val(-local_sem_val);
    }

    inline void cb_mode_local_sem_rptr_cleared_update() {
        uint32_t local_sem_val = this->cb_mode_get_local_sem_val();
        for (uint32_t i = 0; i < local_sem_val; i++) {
            this->advance_queue_local_rptr_cleared(this->cb_mode_page_size_words);
        }
        this->cb_mode_inc_local_sem_val(-local_sem_val);
    }

    void yield() {
        // TODO: implement yield for ethernet here
    }

    void dprint_object() {
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
                    (this->queue_start_addr_words + this->get_queue_rptr_sent_offset_words())*PACKET_WORD_SIZE_BYTES
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

    inline void reset_queue_local_wptr() {
        // Setting STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX resets the credit register
        volatile uint32_t* local_wptr_reset = reinterpret_cast<volatile uint32_t*>(
                STREAM_REG_ADDR(NUM_PTR_REGS_PER_INPUT_QUEUE*queue_id, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX));
        *local_wptr_reset = 0;
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
        // zero_l1_buf(queue_ptr, queue_size_words*PACKET_WORD_SIZE_BYTES);

        this->packetizer_page_words_cleared = 0;

        if (packetizer_input) {
            this->curr_packet_src = packetizer_input_src;
            this->curr_packet_dest = packetizer_input_dest;
            this->curr_packet_flags = 0;
            this->curr_packet_tag = 0xabcd;
        }

        this->ptr_offset_mask = queue_size_words - 1;
        this->queue_size_mask = (queue_size_words << 1) - 1;
        this->curr_packet_valid = false;
        this->reset_queue_local_wptr();
        this->reset_ready_flag();
    }

    inline uint16_t get_end_of_cmd() const {
        return this->end_of_cmd;
    }

    inline bool is_packetizer_input() const {
        return this->cb_mode;
    }

    inline uint32_t get_queue_data_num_words_available_to_send() const {
        return (this->get_queue_local_wptr() - this->get_queue_local_rptr_sent()) & this->queue_size_mask;
    }

    inline bool get_curr_packet_valid() {
        if (this->cb_mode) {
            this->cb_mode_local_sem_wptr_update();
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

    inline uint32_t input_queue_curr_packet_num_words_available_to_send() {
        if (this->cb_mode) {
            this->cb_mode_local_sem_wptr_update();
        }
        uint32_t num_words = this->get_queue_data_num_words_available_to_send();
        if (num_words == 0) {
            return 0;
        }
        num_words = std::min(num_words, this->get_curr_packet_words_remaining());
        return num_words;
    }

    // returns the number of words skipped for page padding if in packetizer mode
    inline uint32_t input_queue_advance_words_sent(uint32_t num_words) {
        uint32_t input_pad_words_skipped = 0;
        if (num_words > 0) {
            this->advance_queue_local_rptr_sent(num_words);
            this->advance_queue_remote_rptr_sent(num_words);
            this->curr_packet_words_sent += num_words;
            uint32_t curr_packet_words_remaining = this->get_curr_packet_words_remaining();
            if (curr_packet_words_remaining == 0) {
                if (this->is_packetizer_input()) {
                    input_pad_words_skipped = this->cb_mode_rptr_sent_advance_page_align();
                }
                this->curr_packet_valid = false;
                this->advance_next_packet();
            }
        }
        return input_pad_words_skipped;
    }

    inline void input_queue_advance_words_cleared(uint32_t num_words) {
        if (num_words > 0) {
            this->advance_queue_local_rptr_cleared(num_words);
            this->advance_queue_remote_rptr_cleared(num_words);
            if (this->is_packetizer_input()) {
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

    inline void input_queue_clear_all_words_sent() {
        uint32_t num_words = this->get_num_words_sent_not_cleared();
        if (num_words > 0) {
            this->input_queue_advance_words_cleared(num_words);
        }
    }

    void dprint_object() {
        DPRINT << "Input queue:" << ENDL();
        packet_queue_state_t::dprint_object();
        DPRINT << "  packet_valid: " << DEC() << static_cast<uint32_t>(this->curr_packet_valid) << ENDL();
        DPRINT << "  packet_tag: 0x" << HEX() << static_cast<uint32_t>(this->curr_packet_tag) << ENDL();
        DPRINT << "  packet_src: 0x" << HEX() << static_cast<uint32_t>(this->curr_packet_src) << ENDL();
        DPRINT << "  packet_dest: 0x" << HEX() << static_cast<uint32_t>(this->curr_packet_dest) << ENDL();
        DPRINT << "  packet_flags: 0x" << HEX() << static_cast<uint32_t>(this->curr_packet_flags) << ENDL();
        DPRINT << "  packet_size_words: " << DEC() << static_cast<uint32_t>(this->curr_packet_size_words) << ENDL();
        DPRINT << "  packet_words_sent: " << DEC() << static_cast<uint32_t>(this->curr_packet_words_sent) << ENDL();
    }

};


class packet_output_queue_state_t : public packet_queue_state_t {

protected:

    uint32_t max_noc_send_words;
    uint32_t max_eth_send_words;

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

        inline uint32_t prev_words_in_flight_flush() {

            uint32_t words_flushed = this->prev_output_total_words_in_flight;
            if (words_flushed > 0) {
                for (uint32_t i = 0; i < num_input_queues; i++) {
                    this->input_queue_array[i].input_queue_advance_words_cleared(this->prev_input_queue_words_in_flight[i]);
                    this->prev_input_queue_words_in_flight[i] = 0;
                }
            }

            uint32_t* tmp = this->prev_input_queue_words_in_flight;
            this->prev_input_queue_words_in_flight = this->curr_input_queue_words_in_flight;
            this->curr_input_queue_words_in_flight = tmp;
            this->prev_output_total_words_in_flight = this->curr_output_total_words_in_flight;
            this->curr_output_total_words_in_flight = 0;

            return words_flushed;
        }

        inline void register_words_in_flight(uint32_t input_queue_id, uint32_t num_words) {
            uint32_t input_pad_words_skipped = this->input_queue_array[input_queue_id].input_queue_advance_words_sent(num_words);
            this->curr_input_queue_words_in_flight[input_queue_id] += (num_words + input_pad_words_skipped);
            this->curr_output_total_words_in_flight += num_words;
        }

        void dprint_object() {
            DPRINT << "  curr_output_total_words_in_flight: " << DEC() << this->curr_output_total_words_in_flight << ENDL();
            for (uint32_t j = 0; j < MAX_SWITCH_FAN_IN; j++) {
                DPRINT << "       from input queue id " << DEC() <<
                            this->input_queue_array[j].get_queue_id() << ": "
                            << DEC() << this->curr_input_queue_words_in_flight[j]
                            << ENDL();
            }
            DPRINT << "  prev_output_total_words_in_flight: " << DEC() << this->prev_output_total_words_in_flight << ENDL();
            for (uint32_t j = 0; j < MAX_SWITCH_FAN_IN; j++) {
                DPRINT << "       from input queue id " << DEC() <<
                            this->input_queue_array[j].get_queue_id() << ": "
                            << DEC() << this->prev_input_queue_words_in_flight[j]
                            << ENDL();
            }
        }

    } input_queue_status;

    inline void reset_queue_local_rptr_sent()  {
        *this->local_rptr_sent_reset = 0;
    }

    inline void reset_queue_local_rptr_cleared()  {
        *this->local_rptr_cleared_reset = 0;
    }

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
        this->ptr_offset_mask = queue_size_words - 1;
        this->queue_size_mask = (queue_size_words << 1) - 1;
        this->max_noc_send_words = DEFAULT_MAX_NOC_SEND_WORDS;
        this->max_eth_send_words = DEFAULT_MAX_ETH_SEND_WORDS;
        this->input_queue_status.init(input_queue_array, num_input_queues);
        this->reset_queue_local_rptr_sent();
        this->reset_queue_local_rptr_cleared();
        this->reset_ready_flag();
    }

    inline bool is_unpacketizer_output() const {
        return this->cb_mode;
    }

    inline void set_max_noc_send_words(uint32_t max_noc_send_words) {
        this->max_noc_send_words = max_noc_send_words;
    }

    inline void set_max_eth_send_words(uint32_t max_eth_send_words) {
        this->max_eth_send_words = max_eth_send_words;
    }

    inline uint32_t output_max_num_words_to_forward() const {
        return (this->remote_update_network_type == DispatchRemoteNetworkType::ETH) ?
            this->max_eth_send_words : this->max_noc_send_words;
    }

    inline void send_data_to_remote(uint32_t src_addr, uint32_t dest_addr, uint32_t num_words) {
        if ((this->remote_update_network_type == DispatchRemoteNetworkType::NONE) ||
            (num_words == 0)) {
            return;
        } else if (this->remote_update_network_type == DispatchRemoteNetworkType::ETH) {
            // while(eth_txq_is_busy());
            internal_::eth_send_packet(0, src_addr/PACKET_WORD_SIZE_BYTES, dest_addr/PACKET_WORD_SIZE_BYTES, num_words);
        } else {
            uint64_t noc_dest_addr = NOC_XY_ADDR(this->remote_x, this->remote_y, dest_addr);
            noc_async_write(src_addr, noc_dest_addr, num_words*PACKET_WORD_SIZE_BYTES);
        }
    }

    inline void remote_wptr_update(uint32_t num_words) {
        this->advance_queue_remote_wptr(num_words);
    }

    inline uint32_t prev_words_in_flight_check_flush() {
        if (this->is_unpacketizer_output()) {
            uint32_t words_written_not_sent = get_num_words_written_not_sent();
            noc_async_writes_flushed();
            this->advance_queue_local_rptr_sent(words_written_not_sent);
            uint32_t words_flushed = this->input_queue_status.prev_words_in_flight_flush();
            this->cb_mode_local_sem_rptr_cleared_update();
            return words_flushed;
        }
        else if (this->get_num_words_written_not_sent() <= this->input_queue_status.get_curr_output_total_words_in_flight()) {
            return this->input_queue_status.prev_words_in_flight_flush();
        }
        else {
            return 0;
        }
    }

    bool output_barrier(uint32_t timeout_cycles = 0) {
        uint32_t start_timestamp = 0;
        if (timeout_cycles > 0) {
            start_timestamp = get_timestamp_32b();
        }
        if (this->is_unpacketizer_output()) {
           noc_async_writes_flushed();
        }
        while (this->get_queue_data_num_words_occupied() > 0) {
            if (this->is_unpacketizer_output()) {
                this->cb_mode_local_sem_rptr_cleared_update();
                if (this->cb_mode_local_sem_downstream_complete()) {
                    // There is no guaranteed that dispatch_h will increment semaphore for all commmands
                    // (specifically the final terminate command).
                    // So just clear whatever remains once the completion signal is received.
                    uint32_t words_occupied = this->get_queue_data_num_words_occupied();
                    this->advance_queue_local_rptr_cleared(words_occupied);
                }
            }
            if (timeout_cycles > 0) {
                uint32_t cycles_elapsed = get_timestamp_32b() - start_timestamp;
                if (cycles_elapsed > timeout_cycles) {
                    return false;
                }
            }
            this->yield();
        }
        this->input_queue_status.prev_words_in_flight_flush();
        this->input_queue_status.prev_words_in_flight_flush();
        return true;
    }

    inline uint32_t get_num_words_to_send(uint32_t input_queue_index) {

        packet_input_queue_state_t* input_queue_ptr = &(this->input_queue_status.input_queue_array[input_queue_index]);

        uint32_t num_words_available_in_input = input_queue_ptr->input_queue_curr_packet_num_words_available_to_send();
        uint32_t num_words_before_input_rptr_wrap = input_queue_ptr->get_queue_words_before_rptr_sent_wrap();
        num_words_available_in_input = std::min(num_words_available_in_input, num_words_before_input_rptr_wrap);
        uint32_t num_words_free_in_output = this->get_queue_data_num_words_free();
        uint32_t num_words_to_forward = std::min(num_words_available_in_input, num_words_free_in_output);

        if (num_words_to_forward == 0) {
            return 0;
        }

        uint32_t output_buf_words_before_wptr_wrap = this->get_queue_words_before_wptr_wrap();
        num_words_to_forward = std::min(num_words_to_forward, output_buf_words_before_wptr_wrap);
        num_words_to_forward = std::min(num_words_to_forward, this->output_max_num_words_to_forward());

        return num_words_to_forward;
    }

    inline uint32_t forward_data_from_input(uint32_t input_queue_index, bool& full_packet_sent, uint16_t end_of_cmd) {

        packet_input_queue_state_t* input_queue_ptr = &(this->input_queue_status.input_queue_array[input_queue_index]);
        uint32_t num_words_to_forward = this->get_num_words_to_send(input_queue_index);
        full_packet_sent = (num_words_to_forward == input_queue_ptr->get_curr_packet_words_remaining());
        if (num_words_to_forward == 0) {
            return 0;
        }

        if (this->unpacketizer_remove_header && input_queue_ptr->curr_packet_start()) {
            num_words_to_forward--;
            this->input_queue_status.register_words_in_flight(input_queue_index, 1);
        }
        if (num_words_to_forward == 0) {
            return 0;
        }

        uint32_t src_addr =
            (input_queue_ptr->queue_start_addr_words +
             input_queue_ptr->get_queue_rptr_sent_offset_words())*PACKET_WORD_SIZE_BYTES;
        uint32_t dest_addr =
            (this->queue_start_addr_words + this->get_queue_wptr_offset_words())*PACKET_WORD_SIZE_BYTES;

        this->send_data_to_remote(src_addr, dest_addr, num_words_to_forward);
        this->input_queue_status.register_words_in_flight(input_queue_index, num_words_to_forward);
        this->advance_queue_local_wptr(num_words_to_forward);

        if (!this->is_unpacketizer_output()) {
            this->remote_wptr_update(num_words_to_forward);
        } else {
            this->unpacketizer_page_words_sent += num_words_to_forward;
            if (full_packet_sent && end_of_cmd) {
                uint32_t unpacketizer_page_words_sent_past_page_bound =
                    this->unpacketizer_page_words_sent & (this->cb_mode_page_size_words - 1);
                if (unpacketizer_page_words_sent_past_page_bound > 0) {
                    uint32_t pad_words = this->cb_mode_page_size_words - unpacketizer_page_words_sent_past_page_bound;
                    this->unpacketizer_page_words_sent += pad_words;
                    this->advance_queue_local_wptr(pad_words);
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
        DPRINT << "Output queue:" << ENDL();
        packet_queue_state_t::dprint_object();
        this->input_queue_status.dprint_object();
    }
};


/**
 *  Polling for ready signal from the remote peers of all input and output queues.
 *  Blocks until all are ready, but doesn't block polling on each individual queue.
 *  Returns false in case of timeout.
 */
bool wait_all_src_dest_ready(packet_input_queue_state_t* input_queue_array, uint32_t num_input_queues,
                             packet_output_queue_state_t* output_queue_array, uint32_t num_output_queues,
                             uint32_t timeout_cycles = 0) {

    bool all_src_dest_ready = false;
    bool src_ready[num_input_queues] = {false};
    bool dest_ready[num_output_queues] = {false};

    uint32_t iters = 0;

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
        for (uint32_t i = 0; i < num_input_queues; i++) {
            if (!src_ready[i]) {
                src_ready[i] = input_queue_array[i].is_packetizer_input() ||
                               input_queue_array[i].is_remote_ready();
                if (!src_ready[i]) {
                    input_queue_array[i].send_remote_ready_notification();
                    all_src_dest_ready = false;
                } else {
                    // handshake with src complete
                }
            }
        }
        for (uint32_t i = 0; i < num_output_queues; i++) {
            if (!dest_ready[i]) {
                dest_ready[i] = output_queue_array[i].is_remote_ready() ||
                                output_queue_array[i].is_unpacketizer_output();
                if (dest_ready[i]) {
                    output_queue_array[i].send_remote_ready_notification();
                } else {
                    all_src_dest_ready = false;
                }
            }
        }
#if defined(COMPILE_FOR_ERISC)
        if ((timeout_cycles == 0) && (iters & 0xFFF) == 0) {
            //if timeout is disabled, context switch every 4096 iterations.
            //this is necessary to allow ethernet routing layer to operate.
            //this core may have pending ethernet routing work.
            internal_::risc_context_switch();
        }
#endif
    }
    return true;
}
