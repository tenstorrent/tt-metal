#include <cstdint>
#include "dataflow_api.h"
#include "debug/assert.h"
#include "noc_parameters.h"
#include "risc_attribs.h"
#include "socket.h"

static_assert(offsetof(sender_socket_md, bytes_acked) % L1_ALIGNMENT == 0);

static_assert(offsetof(receiver_socket_md, bytes_sent) % L1_ALIGNMENT == 0);

struct SocketSenderInterface {
    uint32_t config_addr;
    uint32_t write_ptr;
    uint32_t bytes_sent;
    uint32_t bytes_acked_addr;
    uint32_t page_size;

    // Downstream Socket Metadata
    uint32_t downstream_mesh_id;
    uint32_t downstream_chip_id;
    uint32_t downstream_noc_y;
    uint32_t downstream_noc_x;
    uint32_t downstream_bytes_sent_addr;
    uint32_t downstream_fifo_addr;
    uint32_t downstream_fifo_total_size;
    uint32_t downstream_fifo_curr_size;
};

struct SocketReceiverInterface {
    uint32_t config_addr;
    uint32_t read_ptr;
    uint32_t bytes_acked;
    uint32_t bytes_sent_addr;
    uint32_t page_size;
    uint32_t fifo_addr;
    uint32_t fifo_total_size;
    uint32_t fifo_curr_size;

    // Upstream Socket Metadata
    uint32_t upstream_mesh_id;
    uint32_t upstream_chip_id;
    uint32_t upstream_noc_y;
    uint32_t upstream_noc_x;
    uint32_t upstream_bytes_acked_addr;
};

SocketSenderInterface create_sender_socket_interface(uint32_t config_addr) {
    tt_l1_ptr sender_socket_md* socket_config = reinterpret_cast<tt_l1_ptr sender_socket_md*>(config_addr);
    SocketSenderInterface socket;
    socket.config_addr = config_addr;
    socket.write_ptr = socket_config->write_ptr;
    socket.bytes_sent = socket_config->bytes_sent;
    socket.bytes_acked_addr = config_addr + offsetof(sender_socket_md, bytes_acked);
    socket.downstream_mesh_id = socket_config->downstream_mesh_id;
    socket.downstream_chip_id = socket_config->downstream_chip_id;
    socket.downstream_noc_x = socket_config->downstream_noc_x;
    socket.downstream_noc_y = socket_config->downstream_noc_y;
    socket.downstream_fifo_addr = socket_config->downstream_fifo_addr;
    socket.downstream_bytes_sent_addr = socket_config->downstream_bytes_sent_addr;
    socket.downstream_fifo_total_size = socket_config->downstream_fifo_total_size;

    return socket;
}

void set_sender_socket_page_size(SocketSenderInterface& socket, uint32_t page_size) {
    // TODO: DRAM
    ASSERT(page_size % L1_ALIGNMENT == 0);
    uint32_t fifo_start_addr = socket.downstream_fifo_addr;
    uint32_t fifo_total_size = socket.downstream_fifo_total_size;
    ASSERT(page_size <= fifo_total_size);
    uint32_t& fifo_wr_ptr = socket.write_ptr;
    uint32_t next_fifo_wr_ptr = fifo_start_addr + align(fifo_wr_ptr - fifo_start_addr, page_size);
    uint32_t fifo_page_aligned_size = fifo_total_size - fifo_total_size % page_size;
    uint32_t fifo_limit_page_aligned = fifo_start_addr + fifo_page_aligned_size;
    if (next_fifo_wr_ptr >= fifo_limit_page_aligned) {
        socket.bytes_sent += fifo_start_addr + fifo_total_size - next_fifo_wr_ptr;
        next_fifo_wr_ptr = fifo_start_addr;
    }
    fifo_wr_ptr = next_fifo_wr_ptr;
    socket.page_size = page_size;
    socket.downstream_fifo_curr_size = fifo_limit_page_aligned;
}

void socket_reserve_pages(const SocketSenderInterface& socket, uint32_t num_pages) {
    uint32_t num_bytes = num_pages * socket.page_size;
    volatile tt_l1_ptr uint32_t* bytes_acked_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(socket.bytes_acked_addr);
    uint32_t bytes_diff;
    uint32_t bytes_free;
    do {
        bytes_diff = socket.bytes_sent - *bytes_acked_ptr;
        bytes_free = socket.downstream_fifo_curr_size - bytes_diff;
    } while (bytes_diff > socket.downstream_fifo_curr_size || bytes_free < num_bytes);
}

void socket_push_pages(SocketSenderInterface& socket, uint32_t num_pages) {
    uint32_t num_bytes = num_pages * socket.page_size;
    ASSERT(num_bytes <= socket.downstream_fifo_curr_size);
    if (socket.write_ptr + num_bytes >= socket.downstream_fifo_curr_size + socket.downstream_fifo_addr) {
        socket.write_ptr = socket.write_ptr + num_bytes - socket.downstream_fifo_curr_size;
        socket.bytes_sent += num_bytes + socket.downstream_fifo_total_size - socket.downstream_fifo_curr_size;
    } else {
        socket.write_ptr += num_bytes;
        socket.bytes_sent += num_bytes;
    }
}

// User controlled?
void socket_notify_receiver(const SocketSenderInterface& socket) {
    // TODO: Store noc encoding in struct?
    auto downstream_bytes_sent_noc_addr =
        get_noc_addr(socket.downstream_noc_x, socket.downstream_noc_y, socket.downstream_bytes_sent_addr);
    noc_inline_dw_write(downstream_bytes_sent_noc_addr, socket.bytes_sent);
}

void socket_barrier(const SocketSenderInterface& socket) {
    volatile tt_l1_ptr uint32_t* bytes_acked_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(socket.bytes_acked_addr);
    while (socket.bytes_sent != *bytes_acked_ptr);
}

void update_socket_config(const SocketSenderInterface& socket) {
    volatile tt_l1_ptr sender_socket_md* socket_config =
        reinterpret_cast<volatile tt_l1_ptr sender_socket_md*>(socket.config_addr);
    socket_config->bytes_sent = socket.bytes_sent;
    socket_config->write_ptr = socket.write_ptr;
}

SocketReceiverInterface create_receiver_socket_interface(uint32_t config_addr) {
    tt_l1_ptr receiver_socket_md* socket_config = reinterpret_cast<tt_l1_ptr receiver_socket_md*>(config_addr);
    SocketReceiverInterface socket;
    socket.config_addr = config_addr;
    socket.read_ptr = socket_config->read_ptr;
    socket.bytes_acked = socket_config->bytes_acked;
    socket.bytes_sent_addr = config_addr + offsetof(receiver_socket_md, bytes_sent);
    socket.fifo_addr = socket_config->fifo_addr;
    socket.fifo_total_size = socket_config->fifo_total_size;
    socket.upstream_mesh_id = socket_config->upstream_mesh_id;
    socket.upstream_chip_id = socket_config->upstream_chip_id;
    socket.upstream_noc_x = socket_config->upstream_noc_x;
    socket.upstream_noc_y = socket_config->upstream_noc_y;
    socket.upstream_bytes_acked_addr = socket_config->upstream_bytes_acked_addr;
    return socket;
}

void set_receiver_socket_page_size(SocketReceiverInterface& socket, uint32_t page_size) {
    // TODO: DRAM
    ASSERT(page_size % L1_ALIGNMENT == 0);
    uint32_t fifo_start_addr = socket.fifo_addr;
    uint32_t fifo_total_size = socket.fifo_total_size;
    ASSERT(page_size <= fifo_total_size);
    uint32_t& fifo_rd_ptr = socket.read_ptr;
    uint32_t next_fifo_rd_ptr = fifo_start_addr + align(fifo_rd_ptr - fifo_start_addr, page_size);
    uint32_t fifo_page_aligned_size = fifo_total_size - fifo_total_size % page_size;
    uint32_t fifo_limit_page_aligned = fifo_start_addr + fifo_page_aligned_size;
    if (next_fifo_rd_ptr >= fifo_limit_page_aligned) {
        socket.bytes_acked += fifo_start_addr + fifo_total_size - next_fifo_rd_ptr;
        next_fifo_rd_ptr = fifo_start_addr;
    }
    fifo_rd_ptr = next_fifo_rd_ptr;
    socket.page_size = page_size;
    socket.fifo_curr_size = fifo_limit_page_aligned;
}

void socket_wait_for_pages(const SocketReceiverInterface& socket, uint32_t num_pages) {
    uint32_t num_bytes = num_pages * socket.page_size;
    if (socket.read_ptr + num_bytes >= socket.fifo_curr_size + socket.fifo_addr) {
        num_bytes += socket.fifo_total_size - socket.fifo_curr_size;
    }
    volatile tt_l1_ptr uint32_t* bytes_sent_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(socket.bytes_sent_addr);
    uint32_t bytes_recv;
    do {
        bytes_recv = *bytes_sent_ptr - socket.bytes_acked;
    } while (bytes_recv < num_bytes);
}

void socket_pop_pages(SocketReceiverInterface& socket, uint32_t num_pages) {
    uint32_t num_bytes = num_pages * socket.page_size;
    ASSERT(num_bytes <= socket.fifo_curr_size);
    if (socket.read_ptr + num_bytes >= socket.fifo_curr_size + socket.fifo_addr) {
        socket.read_ptr = socket.read_ptr + num_bytes - socket.fifo_curr_size;
        socket.bytes_acked += num_bytes + socket.fifo_total_size - socket.fifo_curr_size;
    } else {
        socket.read_ptr += num_bytes;
        socket.bytes_acked += num_bytes;
    }
}

// TODO: Wrap properly
void assign_local_cb_to_socket(const SocketReceiverInterface& socket, uint32_t cb_id) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    uint32_t fifo_size = socket.fifo_curr_size >> cb_addr_shift;
    uint32_t fifo_limit = socket.fifo_addr >> cb_addr_shift + fifo_size;
    uint32_t fifo_ptr = socket.read_ptr >> cb_addr_shift;
    ASSERT(fifo_size % local_cb.fifo_page_size == 0);
    uint32_t fifo_num_pages = fifo_size / local_cb.fifo_page_size;
    local_cb.fifo_limit = fifo_limit;
    local_cb.fifo_size = fifo_size;
    local_cb.fifo_num_pages = fifo_num_pages;
    local_cb.fifo_wr_ptr = fifo_ptr;
    local_cb.fifo_rd_ptr = fifo_ptr;
}

// User controlled?
void socket_notify_sender(const SocketReceiverInterface& socket) {
    // TODO: Store noc encoding in struct?
    auto upstream_bytes_acked_noc_addr =
        get_noc_addr(socket.upstream_noc_x, socket.upstream_noc_y, socket.upstream_bytes_acked_addr);
    noc_inline_dw_write(upstream_bytes_acked_noc_addr, socket.bytes_acked);
}

void update_socket_config(const SocketReceiverInterface& socket) {
    volatile tt_l1_ptr receiver_socket_md* socket_config =
        reinterpret_cast<volatile tt_l1_ptr receiver_socket_md*>(socket.config_addr);
    socket_config->bytes_acked = socket.bytes_acked;
    socket_config->read_ptr = socket.read_ptr;
}
