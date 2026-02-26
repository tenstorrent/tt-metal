// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric/fabric_edm_packet_header.hpp"
#include "internal/dataflow/dataflow_cmd_bufs.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h"
#include "tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h"

#include <cstdint>
#include <cstddef>

// Latency test sender kernel - measures round-trip latency by recording timestamps
// around packet send and ack receipt

constexpr bool enable_fused_payload_with_sync = get_compile_time_arg_val(0) != 0;
constexpr bool sem_inc_only = get_compile_time_arg_val(1) != 0;
constexpr bool is_2d_fabric = get_compile_time_arg_val(2) != 0;

void kernel_main() {
    set_l1_data_cache<false>();
    using namespace tt::tt_fabric;
    size_t arg_idx = 0;

    // Common runtime args
    const size_t result_buffer_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t semaphore_address =
        get_arg_val<uint32_t>(arg_idx++);  // Shared sync address (same offset on all devices)
    const size_t payload_size_bytes = get_arg_val<uint32_t>(arg_idx++);
    const size_t num_samples = get_arg_val<uint32_t>(arg_idx++);
    const size_t send_buffer_address = get_arg_val<uint32_t>(arg_idx++);  // Sender's send buffer (write before sending)
    const size_t receive_buffer_address =
        get_arg_val<uint32_t>(arg_idx++);  // Sender's receive buffer (wait for response)
    const uint8_t responder_noc_x =
        static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));  // Responder's virtual NOC X
    const uint8_t responder_noc_y =
        static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));  // Responder's virtual NOC Y
    const size_t responder_receive_buffer_address =
        send_buffer_address;  // Responder receives from sender's send buffer

    // Topology-specific routing args
    uint32_t num_hops_to_responder = 0;
    uint32_t dst_device_id = 0;
    uint32_t dst_mesh_id = 0;

    if constexpr (!is_2d_fabric) {
        num_hops_to_responder = get_arg_val<uint32_t>(arg_idx++);
    } else {
        dst_device_id = get_arg_val<uint32_t>(arg_idx++);
        dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
    }

    // Build fabric connection
    auto fabric_connection = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);

    fabric_connection.open();

    // Allocate packet headers from pool
    auto* payload_packet_header = PacketHeaderPool::allocate_header();
    auto* sem_inc_packet_header = PacketHeaderPool::allocate_header();

    // Setup packet headers for routing
    if constexpr (!is_2d_fabric) {
        // 1D routing: use Low Latency header with hop count
        fabric_set_unicast_route<false>((LowLatencyPacketHeader*)payload_packet_header, num_hops_to_responder);
        fabric_set_unicast_route<false>((LowLatencyPacketHeader*)sem_inc_packet_header, num_hops_to_responder);
    } else {
        // 2D routing: use Hybrid Mesh header with device/mesh IDs (static routing)
        fabric_set_unicast_route((HybridMeshPacketHeader*)payload_packet_header, dst_device_id, dst_mesh_id);
        fabric_set_unicast_route((HybridMeshPacketHeader*)sem_inc_packet_header, dst_device_id, dst_mesh_id);
    }

    // Setup NOC addresses for destination (responder device)
    // Use responder's virtual core coordinates (not sender's coordinates)
    // Send payload to responder's receive buffer (not its timestamp buffer)
    auto dest_semaphore_noc_addr = safe_get_noc_addr(responder_noc_x, responder_noc_y, semaphore_address, 0);
    auto dest_payload_noc_addr =
        safe_get_noc_addr(responder_noc_x, responder_noc_y, responder_receive_buffer_address, 0);

    // Setup NOC command headers
    if constexpr (enable_fused_payload_with_sync) {
        payload_packet_header->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                dest_payload_noc_addr, dest_semaphore_noc_addr, 1, false},
            payload_size_bytes);
    } else {
        payload_packet_header->to_noc_unicast_write(NocUnicastCommandHeader{dest_payload_noc_addr}, payload_size_bytes);
        sem_inc_packet_header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{dest_semaphore_noc_addr, 1});
    }

    auto send_seminc_packet = [&fabric_connection, sem_inc_packet_header]() {
        fabric_connection.wait_for_empty_write_slot();
        fabric_connection.send_payload_flush_non_blocking_from_address(
            (uint32_t)sem_inc_packet_header, sizeof(PACKET_HEADER_TYPE));
    };

    const uint8_t noc_cmd_field_vc = NOC_UNICAST_WRITE_VC;
    uint32_t noc_cmd_field =
        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(noc_cmd_field_vc) | NOC_CMD_RESP_MARKED;

    auto send_payload_packet = [&noc_cmd_field,
                                &fabric_connection,
                                payload_packet_header,
                                send_buffer_address,
                                payload_size_bytes]() {
        // fabric_connection.wait_for_empty_write_slot();
        while (true) {
            invalidate_l1_cache();
            auto used_slots = fabric_connection.buffer_slot_write_counter.counter -
                              *fabric_connection.edm_buffer_local_free_slots_read_ptr;
            if (used_slots < fabric_connection.num_buffers_per_channel) {
                break;
            }
        }
        uint64_t buffer_address = get_noc_addr(
            fabric_connection.edm_noc_x,
            fabric_connection.edm_noc_y,
            fabric_connection.edm_buffer_addr,
            get_fabric_worker_noc());

        // fabric_connection.send_payload_without_header_non_blocking_from_address(
        //     send_buffer_address, payload_size_bytes);
        if (payload_size_bytes > 0) {
            uint32_t src_addr = send_buffer_address;
            uint64_t dest_addr = buffer_address + sizeof(PACKET_HEADER_TYPE);
            uint32_t len_bytes = payload_size_bytes;
            const uint8_t noc = get_fabric_worker_noc();
            auto cmd_buf = write_cmd_buf;

            while (len_bytes > NOC_MAX_BURST_SIZE) {
                while (!noc_cmd_buf_ready(noc, write_cmd_buf));

                NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
                NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_addr);
                NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dest_addr);
                NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dest_addr >> 32) & NOC_PCIE_MASK);
                NOC_CMD_BUF_WRITE_REG(
                    noc,
                    cmd_buf,
                    NOC_RET_ADDR_COORDINATE,
                    (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
                NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
                NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

                src_addr += NOC_MAX_BURST_SIZE;
                dest_addr += NOC_MAX_BURST_SIZE;
                len_bytes -= NOC_MAX_BURST_SIZE;
            }
            while (!noc_cmd_buf_ready(noc, write_cmd_buf));
            NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
            NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_addr);
            NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dest_addr);
            NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dest_addr >> 32) & NOC_PCIE_MASK);
            NOC_CMD_BUF_WRITE_REG(
                noc,
                cmd_buf,
                NOC_RET_ADDR_COORDINATE,
                (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
            NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
            NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        }

        // fabric_connection.send_payload_flush_non_blocking_from_address(
        //     (uint32_t)payload_packet_header, sizeof(PACKET_HEADER_TYPE));
        uint32_t src_addr = (uint32_t)payload_packet_header;
        uint64_t dest_addr = buffer_address;
        uint32_t len_bytes = sizeof(PACKET_HEADER_TYPE);

        const uint8_t noc = get_fabric_worker_noc();
        auto cmd_buf = write_cmd_buf;

        while (len_bytes > NOC_MAX_BURST_SIZE) {
            while (!noc_cmd_buf_ready(noc, write_cmd_buf));

            NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
            NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_addr);
            NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dest_addr);
            NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dest_addr >> 32) & NOC_PCIE_MASK);
            NOC_CMD_BUF_WRITE_REG(
                noc,
                cmd_buf,
                NOC_RET_ADDR_COORDINATE,
                (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
            NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
            NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

            src_addr += NOC_MAX_BURST_SIZE;
            dest_addr += NOC_MAX_BURST_SIZE;
            len_bytes -= NOC_MAX_BURST_SIZE;
        }
        while (!noc_cmd_buf_ready(noc, write_cmd_buf));
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_addr);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dest_addr);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dest_addr >> 32) & NOC_PCIE_MASK);
        NOC_CMD_BUF_WRITE_REG(
            noc, cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

        // edm update
        auto packed_val = pack_value_for_inc_on_write_stream_reg_write(-1);
        const uint64_t noc_sem_addr = get_noc_addr(
            fabric_connection.edm_noc_x,
            fabric_connection.edm_noc_y,
            fabric_connection.edm_buffer_remote_free_slots_update_addr,
            noc);

        const int vc = NOC_UNICAST_WRITE_VC;

        uint32_t noc_cmd_field = NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) | NOC_CMD_CPY | NOC_CMD_WR |
                                 NOC_CMD_WR_INLINE | NOC_CMD_RESP_MARKED;

        // If we're given a misaligned address, don't write to the bytes in the word below the address
        uint32_t be_shift = (noc_sem_addr & (NOC_WORD_BYTES - 1));
        uint32_t be32 = (0xf << be_shift);

        cmd_buf = write_at_cmd_buf;
        while (!noc_cmd_buf_ready(noc, cmd_buf));
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_DATA, packed_val);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)(noc_sem_addr));
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_MID, (uint32_t)(noc_sem_addr >> 32) & NOC_PCIE_MASK);
        NOC_CMD_BUF_WRITE_REG(
            noc,
            cmd_buf,
            NOC_TARG_ADDR_COORDINATE,
            (uint32_t)(noc_sem_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, be32);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

        // Advance buffer slot write index
        fabric_connection.buffer_slot_write_counter.counter++;
        fabric_connection.buffer_slot_index = BufferIndex{
            wrap_increment(fabric_connection.buffer_slot_index.get(), fabric_connection.num_buffers_per_channel)};
        fabric_connection.edm_buffer_addr =
            fabric_connection.edm_buffer_base_addr +
            (fabric_connection.buffer_slot_index.get() * fabric_connection.buffer_size_bytes);

        auto iterations_payload = (payload_size_bytes + NOC_MAX_BURST_SIZE - 1) >> 14;
        auto iterations_header = (sizeof(PACKET_HEADER_TYPE) + NOC_MAX_BURST_SIZE - 1) >> 14;
        // 1 for the above edm buffer idx update
        auto iterations_total = iterations_payload + iterations_header + 1;
        noc_nonposted_writes_num_issued[noc] += iterations_total;
        noc_nonposted_writes_acked[noc] += iterations_total;
    };

    auto wait_for_semaphore_then_reset = [semaphore_address](size_t target_value) {
        noc_semaphore_wait(reinterpret_cast<volatile uint32_t*>(semaphore_address), target_value);
        *reinterpret_cast<volatile uint32_t*>(semaphore_address) = 0;
    };

    // Warmup: flush the datapath
    {
        send_seminc_packet();
        wait_for_semaphore_then_reset(1);
    }

    // Store elapsed time (cycles) as uint32_t in result buffer
    volatile uint32_t* result_ptr = reinterpret_cast<volatile uint32_t*>(result_buffer_address);

    // Clear result buffer before writing elapsed times to avoid reading stale data
    uint32_t result_buffer_size = num_samples * sizeof(uint32_t);
    for (uint32_t i = 0; i < num_samples; i++) {
        result_ptr[i] = 0;
    }

    // Main latency measurement loop
    // Use separate send and receive buffers to avoid race conditions
    // Use the last word of the buffer for synchronization, indicating the entire rest of the payload has arrived before it

    const size_t payload_end_offset = payload_size_bytes - sizeof(uint32_t);
    volatile uint32_t* send_buffer_ptr =
        reinterpret_cast<volatile uint32_t*>(send_buffer_address + payload_end_offset);
    volatile uint32_t* receive_buffer_ptr =
        reinterpret_cast<volatile uint32_t*>(receive_buffer_address + payload_end_offset);

    // Initialize receive buffer to 0
    *receive_buffer_ptr = 0;

    for (size_t sample_idx = 0; sample_idx < num_samples; sample_idx++) {
        auto start_timestamp = get_timestamp();
        if constexpr (!sem_inc_only && !enable_fused_payload_with_sync) {
            // Write incrementing value to send buffer BEFORE sending
            *send_buffer_ptr = sample_idx + 1;
            *receive_buffer_ptr = 0;
        }

        // Send one message per sample
        if constexpr (enable_fused_payload_with_sync) {
            send_payload_packet();
        } else {
            if constexpr (sem_inc_only) {
                send_seminc_packet();
            } else {
                send_payload_packet();
            }
        }

        // Don't want to include noc command buffer response time in the total latency measurement
        noc_async_writes_flushed();

        // Wait on receive buffer for response from responder
        if constexpr (!sem_inc_only && !enable_fused_payload_with_sync) {
            noc_semaphore_wait(receive_buffer_ptr, sample_idx + 1);
        } else {
            noc_semaphore_wait(reinterpret_cast<volatile uint32_t*>(semaphore_address), 1);
            *reinterpret_cast<volatile uint32_t*>(semaphore_address) = 0;
        }

        auto end_timestamp = get_timestamp();

        // Store elapsed time in cycles (truncated to uint32_t, sufficient for latency measurements)
        auto elapsed_cycles = end_timestamp - start_timestamp;
        result_ptr[sample_idx] = static_cast<uint32_t>(elapsed_cycles);
    }

    fabric_connection.close();

    noc_semaphore_set(reinterpret_cast<volatile uint32_t*>(semaphore_address), 0);
    noc_async_full_barrier();
}
