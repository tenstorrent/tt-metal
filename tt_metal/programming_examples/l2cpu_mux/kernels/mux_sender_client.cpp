// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "tt_metal/programming_examples/l2cpu_mux/kernels/tt_fabric_l2cpu_mux_interface.hpp"

// A worker that talks to the L2CPU mux exactly the way the V1 mux test client
// (tests/.../fabric_mux_sender_client.cpp) talks to a Tensix mux: build handle, wait
// for READY, connect, send N packets to a remote core, disconnect; one elected client
// waits for the others and terminates the mux.
constexpr uint8_t NUM_BUFFERS = get_compile_time_arg_val(0);
constexpr size_t slot_bytes = get_compile_time_arg_val(1);
constexpr size_t mux_status_address = get_compile_time_arg_val(2);
constexpr size_t mux_termination_address = get_compile_time_arg_val(3);
constexpr uint32_t num_mux_clients = get_compile_time_arg_val(4);

void kernel_main() {
    size_t i = 0;
    const uint8_t mux_x = get_arg_val<uint32_t>(i++);
    const uint8_t mux_y = get_arg_val<uint32_t>(i++);
    const uint32_t channel_base = get_arg_val<uint32_t>(i++);
    const uint32_t conn_info = get_arg_val<uint32_t>(i++);
    const uint32_t handshake = get_arg_val<uint32_t>(i++);
    const uint32_t flow_control = get_arg_val<uint32_t>(i++);
    const uint32_t buffer_index = get_arg_val<uint32_t>(i++);
    const uint8_t channel_id = get_arg_val<uint32_t>(i++);
    // Local scratch (raw L1 addresses laid out by the host to satisfy the 64 B offset rule)
    const uint32_t status_scratch = get_arg_val<uint32_t>(i++);
    const uint32_t local_flow_control = get_arg_val<uint32_t>(i++);
    const uint32_t local_teardown = get_arg_val<uint32_t>(i++);
    const uint32_t local_buffer_index = get_arg_val<uint32_t>(i++);
    const uint32_t sync_sem = get_semaphore(get_arg_val<uint32_t>(i++));
    const uint32_t hdr_buf = get_arg_val<uint32_t>(i++);
    const uint32_t payload_buf = get_arg_val<uint32_t>(i++);
    const uint32_t payload_size = get_arg_val<uint32_t>(i++);
    const uint32_t num_packets = get_arg_val<uint32_t>(i++);
    const uint32_t num_hops = get_arg_val<uint32_t>(i++);
    const uint32_t dst_x = get_arg_val<uint32_t>(i++);
    const uint32_t dst_y = get_arg_val<uint32_t>(i++);
    const uint32_t dst_base = get_arg_val<uint32_t>(i++);
    const uint32_t is_master = get_arg_val<uint32_t>(i++);
    const uint32_t master_x = get_arg_val<uint32_t>(i++);
    const uint32_t master_y = get_arg_val<uint32_t>(i++);
    const uint32_t client_id = get_arg_val<uint32_t>(i++);
    const uint32_t do_terminate = get_arg_val<uint32_t>(i++);  // master only: 0 = leave the mux running
    const uint32_t atomic_addr = get_arg_val<uint32_t>(i++);   // 0 = skip; else a header-only atomic inc target
    const uint32_t atomic_inc = get_arg_val<uint32_t>(i++);

    auto mux = tt::tt_fabric::build_connection_to_l2cpu_mux<NUM_BUFFERS>(
        mux_x,
        mux_y,
        channel_id,
        NUM_BUFFERS,
        slot_bytes,
        channel_base,
        conn_info,
        handshake,
        flow_control,
        buffer_index,
        local_flow_control,
        local_teardown,
        local_buffer_index);

    (void)status_scratch;  // the client owns its aligned scratch (derived from local_buffer_index)
    mux.wait_for_ready(mux_status_address);
    tt::tt_fabric::fabric_client_connect(mux);

    auto* hdr = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_buf);
    auto* payload = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(payload_buf);
    const uint32_t words = payload_size / 4;

    for (uint32_t p = 0; p < num_packets; ++p) {
        for (uint32_t w = 0; w < words; ++w) {
            payload[w] = 0xC0000000u | (client_id << 20) | (p << 12) | (w & 0xFFFu);
        }
        hdr->to_chip_unicast(static_cast<uint8_t>(num_hops));
        hdr->to_noc_unicast_write(
            tt::tt_fabric::NocUnicastCommandHeader{get_noc_addr(dst_x, dst_y, dst_base + p * payload_size)},
            payload_size);
        // slot wait -> payload (non-blocking) -> header (flushed) -> commit
        tt::tt_fabric::fabric_async_write(mux, hdr, payload_buf, payload_size);
    }

    if (atomic_addr != 0) {
        // Header-only packet through the mux: remote noc_semaphore_inc on the destination core.
        hdr->to_chip_unicast(static_cast<uint8_t>(num_hops));
        hdr->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{get_noc_addr(dst_x, dst_y, atomic_addr), atomic_inc});
        tt::tt_fabric::fabric_atomic_inc(mux, hdr);
    }

    tt::tt_fabric::fabric_client_disconnect(mux);

    if (is_master) {
        // Wait until every other client has disconnected, then (on the last round) stop the
        // mux gracefully: it drains all channels first, then reports TERMINATED.
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_sem), num_mux_clients - 1);
        if (do_terminate) {
            tt::tt_fabric::fabric_endpoint_terminate(mux_x, mux_y, mux_termination_address, /*graceful=*/true);
        }
    } else {
        noc_semaphore_inc(get_noc_addr(master_x, master_y, sync_sem), 1);
        noc_async_atomic_barrier();
    }
}
