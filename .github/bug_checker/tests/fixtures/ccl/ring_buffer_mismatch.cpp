// Bug checker test fixture: CCL ring buffer mismatch patterns.
// This file is intentionally buggy to validate rule ccl-ring-buffer-mismatch.

#include "ccl/edm_builder.hpp"

// --- Pattern 1: num_buffers_per_channel mismatch ---
void setup_edm_channel_bad() {
    // BUG: host config uses 4 buffers but kernel compile-time arg uses 8
    auto config = EriscDatamoverConfig(4 /* num_buffers_per_channel */);
    uint32_t ct_args[] = {0, 8, 0};  // should be 4, not 8
    set_kernel_compile_time_args(kernel, ct_args);
}

// --- Pattern 2: buffer address / semaphore count mismatch ---
void add_channel_bad() {
    // BUG: 3 buffer addresses but only 2 semaphore addresses
    std::vector<uint32_t> buf_addrs = {addr0, addr1, addr2};
    std::vector<uint32_t> sem_addrs = {sem0, sem1};
    builder.add_sender_channel(worker_semaphore, buf_addrs, sem_addrs);
}

// --- Pattern 3: ring size vs device count mismatch ---
void compute_ring_topology_bad() {
    // BUG: ring_size from one axis, neighbor lookup uses different axis
    auto ring_size = cyclic_order.size();                                     // 8 devices total
    auto neighbor = get_physical_neighbor(coord, ClusterAxis::X, ring_size);  // X axis only has 4
}

// --- Pattern 4: sender/receiver channel count asymmetry ---
void build_ring_bad() {
    // BUG: 2 sender channels but only 1 receiver channel on paired device
    for (int i = 0; i < 2; i++) {
        sender_builder.add_sender_channel(sem, bufs, sems);
    }
    receiver_builder.add_receiver_channel(sem, bufs, sems);  // missing second channel
}
