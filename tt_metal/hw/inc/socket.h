#include <cstdint>

// Config Buffer on Sender Core will be populated as follows
struct sender_socket_md {
    // Standard Config Entries
    uint32_t bytes_acked = 0;
    uint32_t write_ptr = 0;
    uint32_t bytes_sent = 0;

    // Downstream Socket Metadata
    uint32_t downstream_mesh_id = 0;
    uint32_t downstream_chip_id = 0;
    uint32_t downstream_noc_y = 0;
    uint32_t downstream_noc_x = 0;
    uint32_t downstream_bytes_sent_addr = 0;
    uint32_t downstream_fifo_addr = 0;
    uint32_t downstream_fifo_total_size = 0;

    uint32_t is_sender = 0;
};

// Config Buffer on Receiver Cores will be populated as follows
struct receiver_socket_md {
    // Standard Config Entries
    uint32_t bytes_sent = 0;
    uint32_t bytes_acked = 0;
    uint32_t read_ptr = 0;
    uint32_t fifo_addr = 0;
    uint32_t fifo_total_size = 0;

    // Upstream Socket Metadata
    uint32_t upstream_mesh_id = 0;
    uint32_t upstream_chip_id = 0;
    uint32_t upstream_noc_y = 0;
    uint32_t upstream_noc_x = 0;
    uint32_t upstream_bytes_acked_addr = 0;

    uint32_t is_sender = 0;
};
