#pragma once

#include <cstdint>
#include "dataflow_api.h"
#include "ethernet/dataflow_api.h"

namespace packet_queue {

// Max 2 bits
enum PtrUpdateType : uint8_t {
    NONE = 0,
    WPTR = 1,
    RPTR_SENT = 2,
    RPTR_CLEARED = 3,
};

// Base for remote network controller.
// T is the implementation.
template <typename T>
class packet_queue_remote_control_t {
private:
    inline T& impl() noexcept { return static_cast<T&>(*this); }

    inline const T& impl() const noexcept { return static_cast<const T&>(*this); }

protected:
    packet_queue_remote_control_t() = default;
    ~packet_queue_remote_control_t() = default;

public:
    // Kernel init
    void init(
        uint32_t queue_id,
        uint32_t remote_queue_id,
        uint8_t remote_x,
        uint8_t remote_y,
        uint32_t local_ptrs_addr,
        uint32_t remote_ptrs_addr) {
        this->impl()._init(queue_id, remote_queue_id, remote_x, remote_y, local_ptrs_addr, remote_ptrs_addr);
    }

    // Set stream register value
    inline void reg_update(uint32_t reg_addr, uint32_t val) { this->impl()._reg_update(reg_addr, val); }

    // Update a pointer on the remote
    inline void ptr_update(uint32_t src_addr, uint32_t dest_addr, PtrUpdateType update_type) {
        this->impl()._ptr_update(src_addr, dest_addr, update_type);
    }

    // Send data to the remote
    inline void send_data(uint32_t src_addr, uint32_t dest_addr, uint32_t num_words) {
        this->impl()._send_data(src_addr, dest_addr, num_words);
    }

    // Returns true if the controller is busy and cannot be used yet
    inline bool busy() const { return this->impl()._busy(); }

    // Handle any pending acks from the remote sender
    inline void handle_recv() { this->impl()._handle_recv(); }
};  // packet_queue_remote_control_t

// Remote updates over NOC0.
class packet_queue_remote_noc0_impl final : public packet_queue_remote_control_t<packet_queue_remote_noc0_impl> {
private:
    uint8_t remote_x;
    uint8_t remote_y;

public:
    void _init(
        uint32_t queue_id,
        uint32_t remote_queue_id,
        uint8_t remote_x,
        uint8_t remote_y,
        uint32_t local_ptrs_addr,
        uint32_t remote_ptrs_addr) {
        this->remote_x = remote_x;
        this->remote_y = remote_y;
    }

    inline void _reg_update(uint32_t reg_addr, uint32_t val) {
        noc_inline_dw_write(get_noc_addr(this->remote_x, this->remote_y, reg_addr), val);
    }

    inline void _ptr_update(uint32_t src_addr, uint32_t dest_addr, PtrUpdateType update_type) {
        noc_inline_dw_write(
            get_noc_addr(this->remote_x, this->remote_y, dest_addr), *reinterpret_cast<volatile uint32_t*>(src_addr));
    }

    inline void _send_data(uint32_t src_addr, uint32_t dest_addr, uint32_t num_words) {
        noc_async_write(
            src_addr,
            get_noc_addr(this->remote_x, this->remote_y, dest_addr),
            num_words * 16  // bytes
        );
    }

    inline bool _busy() const { return false; }

    inline void _handle_recv() {}
};  // packet_queue_remote_noc0_impl

// Remote updates over Ethernet.
class packet_queue_remote_eth_impl final : public packet_queue_remote_control_t<packet_queue_remote_eth_impl> {
private:
    struct ptr_reg_fields_t {
        uint32_t ptr_value;
        uint32_t dest_addr;
    };

    union ptr_value_reg_t {
        uint64_t raw;
        ptr_reg_fields_t fields;
    };

    volatile ptr_value_reg_t* sent;
    volatile ptr_value_reg_t* recv;
    uint32_t remote_sent_addr;
    uint32_t remote_recv_addr;

    ptr_value_reg_t outgoing_data;

public:
    void _init(
        uint32_t queue_id,
        uint32_t remote_queue_id,
        uint8_t remote_x,
        uint8_t remote_y,
        uint32_t ptrs_addr,
        uint32_t remote_ptrs_addr) {
        this->sent =
            reinterpret_cast<volatile ptr_value_reg_t*>(packet_queue_ptr_buffer_layout_t::get_eth_sent(ptrs_addr));
        this->recv =
            reinterpret_cast<volatile ptr_value_reg_t*>(packet_queue_ptr_buffer_layout_t::get_eth_recv(ptrs_addr));
        this->remote_sent_addr =
            reinterpret_cast<uint32_t>(packet_queue_ptr_buffer_layout_t::get_eth_sent(remote_ptrs_addr));
        this->remote_recv_addr =
            reinterpret_cast<uint32_t>(packet_queue_ptr_buffer_layout_t::get_eth_recv(remote_ptrs_addr));
        this->sent->raw = 0;
        this->recv->raw = 0;
    }

    inline void _reg_update(uint32_t reg_addr, uint32_t val) { internal_::eth_write_remote_reg(0, reg_addr, val); }

    inline void _ptr_update(uint32_t src_addr, uint32_t dest_addr, PtrUpdateType update_type) {
        internal_::eth_send_packet(
            0,               // txq
            src_addr >> 4,   // source in words
            dest_addr >> 4,  // dest in words
            1                // words
        );
        this->outgoing_data.fields.ptr_value = *reinterpret_cast<volatile uint32_t*>(src_addr);
        this->outgoing_data.fields.dest_addr = dest_addr;
        internal_::eth_send_packet(
            0,                                    // txq
            (uint32_t)&this->outgoing_data >> 4,  // source in words
            this->remote_recv_addr >> 4,          // dest in words
            1                                     // words
        );
        this->sent->raw = 1;
    }

    inline void _send_data(uint32_t src_addr, uint32_t dest_addr, uint32_t num_words) {
        internal_::eth_send_packet(0, src_addr >> 4, dest_addr >> 4, num_words);
    }

    inline bool _busy() const { return (bool)this->sent->raw; /* there is pending data in the sent buffer */ }

    inline void _handle_recv() {
        if (!this->recv->raw) {
            return;
        }

        uint32_t value = this->recv->fields.ptr_value;
        uint32_t dest_addr = this->recv->fields.dest_addr;
        *reinterpret_cast<volatile uint32_t*>(dest_addr) = value;
        this->recv->raw = 0;
        internal_::eth_send_packet(
            0,                            // txq
            (uint32_t)this->recv >> 4,    // source in words
            this->remote_sent_addr >> 4,  // dest in words
            1                             // words
        );
    }
};  // packet_queue_remote_eth_impl

// Dummy remote update class for testing.
class packet_queue_remote_nop_impl final : public packet_queue_remote_control_t<packet_queue_remote_nop_impl> {
public:
    void _init(
        uint32_t queue_id,
        uint32_t remote_queue_id,
        uint8_t remote_x,
        uint8_t remote_y,
        uint32_t local_ptrs_addr,
        uint32_t remote_ptrs_addr) {}

    inline void _reg_update(uint32_t reg_addr, uint32_t val) {}

    inline void _ptr_update(uint32_t src_addr, uint32_t dest_addr, PtrUpdateType update_type) {}

    inline void _send_data(uint32_t src_addr, uint32_t dest_addr, uint32_t num_words) {}

    inline bool _busy() const { return false; }

    inline void _handle_recv() {}
};  // packet_queue_remote_nop_impl

};  // namespace packet_queue
