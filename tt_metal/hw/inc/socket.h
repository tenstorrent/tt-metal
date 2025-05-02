#include <cstdint>

// Config Buffer on Sender Core will be populated as follows
struct sender_socket_md {
    // Standard Config Entries
    uint32_t bytes_acked;
    uint32_t write_ptr;
    uint32_t bytes_sent;

    // Downstream Socket Metadata
    uint32_t downstream_mesh_id;
    uint32_t downstream_chip_id;
    uint32_t downstream_noc_y;
    uint32_t downstream_noc_x;
    uint32_t downstream_bytes_sent_addr;
    uint32_t downstream_fifo_addr;
    uint32_t downstream_fifo_total_size;

    uint32_t is_sender;
};

// Config Buffer on Receiver Cores will be populated as follows
struct receiver_socket_md {
    // Standard Config Entries
    uint32_t bytes_sent;
    uint32_t bytes_acked;
    uint32_t read_ptr;
    uint32_t fifo_addr;
    uint32_t fifo_total_size;

    // Upstream Socket Metadata
    uint32_t upstream_mesh_id;
    uint32_t upstream_chip_id;
    uint32_t upstream_noc_y;
    uint32_t upstream_noc_x;
    uint32_t upstream_bytes_acked_addr;

    uint32_t is_sender;
};
