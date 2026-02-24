// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "api/dataflow/dataflow_api.h"

#include <cstdint>
#include <cstddef>

// Latency test sender kernel - measures round-trip latency by recording timestamps
// around packet send and ack receipt

constexpr bool enable_fused_payload_with_sync = get_compile_time_arg_val(0) != 0;
constexpr bool sem_inc_only = get_compile_time_arg_val(1) != 0;
constexpr bool is_2d_fabric = get_compile_time_arg_val(2) != 0;
constexpr bool measure_wait_for_slot = get_compile_time_arg_val(3) != 0;
constexpr bool measure_send_payload = get_compile_time_arg_val(4) != 0;
constexpr bool measure_send_header = get_compile_time_arg_val(5) != 0;

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
    const size_t send_benchmark_buffer_address =
        get_arg_val<uint32_t>(arg_idx++);  // Buffer to store send_payload_packet timing measurements
    const size_t wait_for_slot_benchmark_buffer_address =
        get_arg_val<uint32_t>(arg_idx++);  // Buffer to store wait_for_empty_write_slot timing measurements
    const size_t send_payload_benchmark_buffer_address =
        get_arg_val<uint32_t>(arg_idx++);  // Buffer to store send_payload_without_header timing measurements
    const size_t send_header_benchmark_buffer_address =
        get_arg_val<uint32_t>(arg_idx++);  // Buffer to store send_payload_flush timing measurements
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

    // Store elapsed time (cycles) as uint32_t in result buffer
    volatile uint32_t* result_ptr = reinterpret_cast<volatile uint32_t*>(result_buffer_address);
    volatile uint32_t* send_benchmark_ptr = reinterpret_cast<volatile uint32_t*>(send_benchmark_buffer_address);
    volatile uint32_t* wait_for_slot_benchmark_ptr =
        reinterpret_cast<volatile uint32_t*>(wait_for_slot_benchmark_buffer_address);
    volatile uint32_t* send_payload_benchmark_ptr =
        reinterpret_cast<volatile uint32_t*>(send_payload_benchmark_buffer_address);
    volatile uint32_t* send_header_benchmark_ptr =
        reinterpret_cast<volatile uint32_t*>(send_header_benchmark_buffer_address);

    auto send_seminc_packet = [&fabric_connection, sem_inc_packet_header]() {
        fabric_connection.wait_for_empty_write_slot();
        fabric_connection.send_payload_flush_non_blocking_from_address(
            (uint32_t)sem_inc_packet_header, sizeof(PACKET_HEADER_TYPE));
    };

    auto send_payload_packet = [&fabric_connection, payload_packet_header, send_buffer_address, payload_size_bytes]() {
        __asm__ volatile("# abouto to wait for slot");
        fabric_connection.wait_for_empty_write_slot();
        if (payload_size_bytes > 0) {
            __asm__ volatile("# sending payload");
            fabric_connection.send_payload_without_header_non_blocking_from_address(
                send_buffer_address, payload_size_bytes);
            __asm__ volatile("# done sending payload");
        }
        __asm__ volatile("# sending header");
        fabric_connection.send_payload_flush_non_blocking_from_address(
            (uint32_t)payload_packet_header, sizeof(PACKET_HEADER_TYPE));
        __asm__ volatile("# done sending header");
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

    // Clear result buffer before writing elapsed times to avoid reading stale data
    uint32_t result_buffer_size = num_samples * sizeof(uint32_t);
    for (uint32_t i = 0; i < num_samples; i++) {
        result_ptr[i] = 0;
        send_benchmark_ptr[i] = 0;
        wait_for_slot_benchmark_ptr[i] = 0;
        send_payload_benchmark_ptr[i] = 0;
        send_header_benchmark_ptr[i] = 0;
    }

    // Main latency measurement loop
    // Use separate send and receive buffers to avoid race conditions
    // Use the last word of the buffer for synchronization, indicating the entire rest of the payload has arrived before it

    volatile uint32_t* send_buffer_ptr = nullptr;
    volatile uint32_t* receive_buffer_ptr = nullptr;

    if (payload_size_bytes > 0) {
        const size_t payload_end_offset = payload_size_bytes - sizeof(uint32_t);
        send_buffer_ptr = reinterpret_cast<volatile uint32_t*>(send_buffer_address + payload_end_offset);
        receive_buffer_ptr = reinterpret_cast<volatile uint32_t*>(receive_buffer_address + payload_end_offset);
        // Initialize receive buffer to 0
        *receive_buffer_ptr = 0;
    }

    for (size_t sample_idx = 0; sample_idx < num_samples; sample_idx++) {
        if constexpr (!sem_inc_only && !enable_fused_payload_with_sync) {
            // Write incrementing value to send buffer BEFORE sending
            *send_buffer_ptr = sample_idx + 1;
            *receive_buffer_ptr = 0;
        }

        // Initialize detailed benchmarks to 0 (in case they're not measured in this code path)
        wait_for_slot_benchmark_ptr[sample_idx] = 0;
        send_payload_benchmark_ptr[sample_idx] = 0;
        send_header_benchmark_ptr[sample_idx] = 0;

        // Send one message per sample
        auto send_start = get_timestamp();
        if constexpr (enable_fused_payload_with_sync) {
            send_payload_packet();
        } else {
            if constexpr (sem_inc_only) {
                send_seminc_packet();
            } else {
                __asm__ volatile("# about to send payload");

                // Time wait_for_empty_write_slot (inlined)
                if constexpr (measure_wait_for_slot) {
                    auto wait_start = get_timestamp();
                    // Inlined: fabric_connection.wait_for_empty_write_slot()
                    {
                        WAYPOINT("FWSW");
                        // Inlined: edm_has_space_for_packet<1>()
                        while (true) {
                            invalidate_l1_cache();
                            auto used_slots = fabric_connection.buffer_slot_write_counter.counter -
                                              *fabric_connection.edm_buffer_local_free_slots_read_ptr;
                            if (used_slots < fabric_connection.num_buffers_per_channel) {
                                break;
                            }
                        }
                        WAYPOINT("FWSD");
                    }
                    auto wait_end = get_timestamp();
                    wait_for_slot_benchmark_ptr[sample_idx] = static_cast<uint32_t>(wait_end - wait_start);
                } else {
                    // Inlined: fabric_connection.wait_for_empty_write_slot()
                    {
                        WAYPOINT("FWSW");
                        // Inlined: edm_has_space_for_packet<1>()
                        while (true) {
                            invalidate_l1_cache();
                            auto used_slots = fabric_connection.buffer_slot_write_counter.counter -
                                              *fabric_connection.edm_buffer_local_free_slots_read_ptr;
                            if (used_slots < fabric_connection.num_buffers_per_channel) {
                                break;
                            }
                        }
                        WAYPOINT("FWSD");
                    }
                }
                uint64_t buffer_address = get_noc_addr(
                    fabric_connection.edm_noc_x,
                    fabric_connection.edm_noc_y,
                    fabric_connection.edm_buffer_addr,
                    get_fabric_worker_noc());
                const uint8_t noc = get_fabric_worker_noc();

                // Track how many writes we issue so we can update counters after timing
                uint32_t num_writes_issued = 0;
                uint32_t total_num_dests = 0;

                // FULLY INLINED send_payload_without_header - includes loops and all NOC register writes
                if (payload_size_bytes > 0) {
                    if constexpr (measure_send_payload) {
                        auto payload_start = get_timestamp();
                        {
                            // Fully inline ncrisc_noc_fast_write_any_len loop
                            uint32_t src_addr = send_buffer_address;
                            uint64_t dest_addr = buffer_address + sizeof(PACKET_HEADER_TYPE);
                            uint32_t len_bytes = payload_size_bytes;
                            const uint32_t vc = 0;

                            // Split into NOC_MAX_BURST_SIZE chunks if needed
                            while (len_bytes > NOC_MAX_BURST_SIZE) {
                                while (!noc_cmd_buf_ready(noc, write_reg_cmd_buf));

                                // Fully inline ncrisc_noc_fast_write (no counter update)
                                uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC |
                                                         NOC_CMD_STATIC_VC(vc) | NOC_CMD_RESP_MARKED;
                                NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_CTRL, noc_cmd_field);
                                NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_TARG_ADDR_LO, src_addr);
                                NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dest_addr);
                                NOC_CMD_BUF_WRITE_REG(
                                    noc,
                                    write_reg_cmd_buf,
                                    NOC_RET_ADDR_MID,
                                    (uint32_t)(dest_addr >> 32) & NOC_PCIE_MASK);
                                NOC_CMD_BUF_WRITE_REG(
                                    noc,
                                    write_reg_cmd_buf,
                                    NOC_RET_ADDR_COORDINATE,
                                    (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
                                NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_AT_LEN_BE, NOC_MAX_BURST_SIZE);
                                NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
                                // Skip counter update here - will do batch update after timing

                                num_writes_issued++;
                                total_num_dests += 1;
                                src_addr += NOC_MAX_BURST_SIZE;
                                dest_addr += NOC_MAX_BURST_SIZE;
                                len_bytes -= NOC_MAX_BURST_SIZE;
                            }

                            // Final chunk
                            while (!noc_cmd_buf_ready(noc, write_reg_cmd_buf));
                            uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC |
                                                     NOC_CMD_STATIC_VC(vc) | NOC_CMD_RESP_MARKED;
                            NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_CTRL, noc_cmd_field);
                            NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_TARG_ADDR_LO, src_addr);
                            NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dest_addr);
                            NOC_CMD_BUF_WRITE_REG(
                                noc, write_reg_cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dest_addr >> 32) & NOC_PCIE_MASK);
                            NOC_CMD_BUF_WRITE_REG(
                                noc,
                                write_reg_cmd_buf,
                                NOC_RET_ADDR_COORDINATE,
                                (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
                            NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_AT_LEN_BE, len_bytes);
                            NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
                            // Skip counter update here

                            num_writes_issued++;
                            total_num_dests += 1;
                        }
                        auto payload_end = get_timestamp();
                        send_payload_benchmark_ptr[sample_idx] = static_cast<uint32_t>(payload_end - payload_start);
                    } else {
                        {
                            // Same fully inlined code for non-measured path
                            uint32_t src_addr = send_buffer_address;
                            uint64_t dest_addr = buffer_address + sizeof(PACKET_HEADER_TYPE);
                            uint32_t len_bytes = payload_size_bytes;
                            const uint32_t vc = 0;

                            while (len_bytes > NOC_MAX_BURST_SIZE) {
                                while (!noc_cmd_buf_ready(noc, write_reg_cmd_buf));
                                uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC |
                                                         NOC_CMD_STATIC_VC(vc) | NOC_CMD_RESP_MARKED;
                                NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_CTRL, noc_cmd_field);
                                NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_TARG_ADDR_LO, src_addr);
                                NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dest_addr);
                                NOC_CMD_BUF_WRITE_REG(
                                    noc,
                                    write_reg_cmd_buf,
                                    NOC_RET_ADDR_MID,
                                    (uint32_t)(dest_addr >> 32) & NOC_PCIE_MASK);
                                NOC_CMD_BUF_WRITE_REG(
                                    noc,
                                    write_reg_cmd_buf,
                                    NOC_RET_ADDR_COORDINATE,
                                    (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
                                NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_AT_LEN_BE, NOC_MAX_BURST_SIZE);
                                NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

                                num_writes_issued++;
                                total_num_dests += 1;
                                src_addr += NOC_MAX_BURST_SIZE;
                                dest_addr += NOC_MAX_BURST_SIZE;
                                len_bytes -= NOC_MAX_BURST_SIZE;
                            }

                            while (!noc_cmd_buf_ready(noc, write_reg_cmd_buf));
                            uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC |
                                                     NOC_CMD_STATIC_VC(vc) | NOC_CMD_RESP_MARKED;
                            NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_CTRL, noc_cmd_field);
                            NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_TARG_ADDR_LO, src_addr);
                            NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dest_addr);
                            NOC_CMD_BUF_WRITE_REG(
                                noc, write_reg_cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dest_addr >> 32) & NOC_PCIE_MASK);
                            NOC_CMD_BUF_WRITE_REG(
                                noc,
                                write_reg_cmd_buf,
                                NOC_RET_ADDR_COORDINATE,
                                (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
                            NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_AT_LEN_BE, len_bytes);
                            NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

                            num_writes_issued++;
                            total_num_dests += 1;
                        }
                    }
                }

                // FULLY INLINED send_payload_flush (header write)
                if constexpr (measure_send_header) {
                    auto header_start = get_timestamp();
                    {
                        // Header is always small (64 bytes), single packet
                        while (!noc_cmd_buf_ready(noc, write_reg_cmd_buf));

                        const uint32_t vc = 0;
                        uint32_t noc_cmd_field =
                            NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) | NOC_CMD_RESP_MARKED;
                        NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_CTRL, noc_cmd_field);
                        NOC_CMD_BUF_WRITE_REG(
                            noc, write_reg_cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)payload_packet_header);
                        NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_RET_ADDR_LO, (uint32_t)buffer_address);
                        NOC_CMD_BUF_WRITE_REG(
                            noc, write_reg_cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(buffer_address >> 32) & NOC_PCIE_MASK);
                        NOC_CMD_BUF_WRITE_REG(
                            noc,
                            write_reg_cmd_buf,
                            NOC_RET_ADDR_COORDINATE,
                            (uint32_t)(buffer_address >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
                        NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_AT_LEN_BE, sizeof(PACKET_HEADER_TYPE));
                        NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
                        // Skip counter update here

                        num_writes_issued++;
                        total_num_dests += 1;
                    }

                    // Inlined: update_edm_buffer_free_slots() - this signal to EDM is still in critical path
                    __asm__ volatile("# update edm buffer ");
                    auto packed_val = pack_value_for_inc_on_write_stream_reg_write(-1);
                    const uint64_t noc_sem_addr = get_noc_addr(
                        fabric_connection.edm_noc_x,
                        fabric_connection.edm_noc_y,
                        fabric_connection.edm_buffer_remote_free_slots_update_addr,
                        noc);
                    noc_inline_dw_write<InlineWriteDst::REG>(noc_sem_addr, packed_val, 0xf, noc);
                    // NOW do bookkeeping BEFORE the EDM signal (which is also critical path)
                    // Inlined: advance_buffer_slot_write_index()
                    fabric_connection.buffer_slot_write_counter.counter++;
                    fabric_connection.buffer_slot_index = BufferIndex{wrap_increment(
                        fabric_connection.buffer_slot_index.get(), fabric_connection.num_buffers_per_channel)};
                    fabric_connection.edm_buffer_addr =
                        fabric_connection.edm_buffer_base_addr +
                        (fabric_connection.buffer_slot_index.get() * fabric_connection.buffer_size_bytes);
                    auto header_end = get_timestamp();
                    send_header_benchmark_ptr[sample_idx] = static_cast<uint32_t>(header_end - header_start);
                } else {
                    {
                        // Same fully inlined header write for non-measured path
                        while (!noc_cmd_buf_ready(noc, write_reg_cmd_buf));

                        const uint32_t vc = 0;
                        uint32_t noc_cmd_field =
                            NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) | NOC_CMD_RESP_MARKED;
                        NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_CTRL, noc_cmd_field);
                        NOC_CMD_BUF_WRITE_REG(
                            noc, write_reg_cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)payload_packet_header);
                        NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_RET_ADDR_LO, (uint32_t)buffer_address);
                        NOC_CMD_BUF_WRITE_REG(
                            noc, write_reg_cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(buffer_address >> 32) & NOC_PCIE_MASK);
                        NOC_CMD_BUF_WRITE_REG(
                            noc,
                            write_reg_cmd_buf,
                            NOC_RET_ADDR_COORDINATE,
                            (uint32_t)(buffer_address >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
                        NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_AT_LEN_BE, sizeof(PACKET_HEADER_TYPE));
                        NOC_CMD_BUF_WRITE_REG(noc, write_reg_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

                        num_writes_issued++;
                        total_num_dests += 1;
                    }

                    // Bookkeeping BEFORE the EDM signal
                    // Inlined: advance_buffer_slot_write_index()
                    fabric_connection.buffer_slot_write_counter.counter++;
                    fabric_connection.buffer_slot_index = BufferIndex{wrap_increment(
                        fabric_connection.buffer_slot_index.get(), fabric_connection.num_buffers_per_channel)};
                    fabric_connection.edm_buffer_addr =
                        fabric_connection.edm_buffer_base_addr +
                        (fabric_connection.buffer_slot_index.get() * fabric_connection.buffer_size_bytes);

                    // Inlined: update_edm_buffer_free_slots()
                    __asm__ volatile("# update edm buffer ");
                    auto packed_val = pack_value_for_inc_on_write_stream_reg_write(-1);
                    const uint64_t noc_sem_addr = get_noc_addr(
                        fabric_connection.edm_noc_x,
                        fabric_connection.edm_noc_y,
                        fabric_connection.edm_buffer_remote_free_slots_update_addr,
                        noc);
                    noc_inline_dw_write<InlineWriteDst::REG>(noc_sem_addr, packed_val, 0xf, noc);
                }

                // NOW update the bookkeeping counters AFTER all critical timing is done
                // This moves the counter updates out of the critical path
                noc_nonposted_writes_num_issued[noc] += num_writes_issued;
                noc_nonposted_writes_acked[noc] += total_num_dests;
            }
        }
        auto send_end = get_timestamp();
        send_benchmark_ptr[sample_idx] = static_cast<uint32_t>(send_end - send_start);

        auto start_timestamp = get_timestamp();

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
