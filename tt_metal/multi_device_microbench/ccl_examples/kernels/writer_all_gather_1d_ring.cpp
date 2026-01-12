
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"

#include <cstdint>
#include <utility>

using namespace tt::tt_fabric::linear::experimental;
struct unicast_route_info_t {
    uint16_t dst_mesh_id;
    union {
        uint16_t dst_chip_id;       // for 2D
        uint16_t distance_in_hops;  // for 1D
    };
};

struct multicast_route_info_t {
    union {
        uint16_t dst_mesh_id;             // for 2D
        uint16_t start_distance_in_hops;  // for 1D
    };
    union {
        uint16_t dst_chip_id;  // for 2D
        uint16_t range_hops;   // for 1D
    };
    // extra hop info for 2D
    uint16_t e_num_hops;
    uint16_t w_num_hops;
    uint16_t n_num_hops;
    uint16_t s_num_hops;
};

constexpr uint32_t ring_size = get_compile_time_arg_val(0);
constexpr uint32_t my_chip_id = get_compile_time_arg_val(1);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(2);
constexpr uint32_t num_pages_to_write_per_packet = get_compile_time_arg_val(3);
constexpr uint32_t page_size = get_compile_time_arg_val(4);
constexpr uint32_t num_devices_rightside = get_compile_time_arg_val(5);
constexpr uint32_t num_devices_leftside = get_compile_time_arg_val(6);
constexpr uint8_t fabric_mux_num_buffers_per_channel = static_cast<uint8_t>(get_compile_time_arg_val(7));
constexpr uint32_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(8);
constexpr uint32_t fabric_mux_status_address = get_compile_time_arg_val(9);
constexpr uint32_t fabric_mux_termination_signal_address = get_compile_time_arg_val(10);
constexpr uint32_t num_workers_per_direction = get_compile_time_arg_val(11);  // mux clients

// Aggregate unicast routing info
constexpr unicast_route_info_t forward_unicast_route_info = {
    .dst_mesh_id = get_compile_time_arg_val(12),
    .distance_in_hops = get_compile_time_arg_val(13),
};
constexpr multicast_route_info_t forward_barrier_multicast_route_info = {
    .start_distance_in_hops = get_compile_time_arg_val(14),
    .range_hops = get_compile_time_arg_val(15),
    .e_num_hops = get_compile_time_arg_val(16),
    .w_num_hops = get_compile_time_arg_val(17),
    .n_num_hops = get_compile_time_arg_val(18),
    .s_num_hops = get_compile_time_arg_val(19),
};
constexpr unicast_route_info_t backward_unicast_route_info = {
    .dst_mesh_id = get_compile_time_arg_val(20),
    .distance_in_hops = get_compile_time_arg_val(21),
};
constexpr multicast_route_info_t backward_barrier_multicast_route_info = {
    .start_distance_in_hops = get_compile_time_arg_val(22),
    .range_hops = get_compile_time_arg_val(23),
    .e_num_hops = get_compile_time_arg_val(24),
    .w_num_hops = get_compile_time_arg_val(25),
    .n_num_hops = get_compile_time_arg_val(26),
    .s_num_hops = get_compile_time_arg_val(27),
};

constexpr uint32_t addrgen_args_base_idx = 28;

void kernel_main() {
    uint32_t arg_idx = 0;
    const uint32_t output_address = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_x = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint8_t out_ready_sem_y = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t out_ready_sem_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t barrier_sem_address = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t mapped_core_x = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint8_t mapped_core_y = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t direction = get_arg_val<uint32_t>(arg_idx++);  // 0 is forward, 1 is backward
    const uint32_t input_page_id_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_page_id_end = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pages_per_sync = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_input_pages_per_device = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_input_pages_per_direction = get_arg_val<uint32_t>(arg_idx++);
    // fabric mux runtime arguments
    bool mux_connection_valid = get_arg_val<uint32_t>(arg_idx++) == 1;
    const bool is_termination_master = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_y = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_channel_base_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_connection_info_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_connection_handshake_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_flow_control_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_buffer_index_address = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_channel_id = get_arg_val<uint32_t>(arg_idx++);

    uint32_t termination_sync_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    uint32_t termination_master_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t termination_master_noc_y = get_arg_val<uint32_t>(arg_idx++);

    // DPRINT << "=== Writer All-Gather 1D Ring Runtime Args ===" << ENDL();
    // DPRINT << "output_address: " << output_address << ENDL();
    // DPRINT << "out_ready_sem: (" << static_cast<uint32_t>(out_ready_sem_x) << "," <<
    // static_cast<uint32_t>(out_ready_sem_y) << ") @ " << out_ready_sem_address << ENDL(); DPRINT <<
    // "barrier_sem_address: " << barrier_sem_address << ENDL(); DPRINT << "mapped_core: (" <<
    // static_cast<uint32_t>(mapped_core_x) << "," << static_cast<uint32_t>(mapped_core_y) << ")" << ENDL(); DPRINT <<
    // "direction: " << (direction == 0 ? "forward" : "backward") << ENDL(); DPRINT << "input_page_id: [" <<
    // input_page_id_start << "," << input_page_id_end << ")" << ENDL(); DPRINT << "pages_per_sync: " << pages_per_sync
    // << ENDL(); DPRINT << "fabric_mux: (" << static_cast<uint32_t>(fabric_mux_x) << "," <<
    // static_cast<uint32_t>(fabric_mux_y) << ") stream=" << static_cast<uint32_t>(fabric_mux_channel_id) << ENDL();
    // DPRINT << "  channel_base: " << fabric_mux_channel_base_address << ENDL();
    // DPRINT << "  conn_info: " << fabric_mux_connection_info_address << ENDL();
    // DPRINT << "  handshake: " << fabric_mux_connection_handshake_address << ENDL();
    // DPRINT << "  flow_ctrl: " << fabric_mux_flow_control_address << ENDL();
    // DPRINT << "  buf_idx: " << fabric_mux_buffer_index_address << ENDL();
    // DPRINT << "termination_sync_address: " << termination_sync_address << ENDL();
    // DPRINT << "local addresses:" << ENDL();
    // DPRINT << "  mux_status: " << local_fabric_mux_status_address << ENDL();
    // DPRINT << "  flow_ctrl: " << local_flow_control_address << ENDL();
    // DPRINT << "  teardown: " << local_teardown_address << ENDL();
    // DPRINT << "  buf_idx: " << local_buffer_index_address << ENDL();
    // DPRINT << "termination_master_noc: (" << termination_master_noc_x << "," << termination_master_noc_y << ")" <<
    // ENDL(); DPRINT << "========================================" << ENDL();

    // semaphore ptrs
    auto out_ready_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_address);
    auto barrier_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_address);

    constexpr auto output_addrgen_args = TensorAccessorArgs<addrgen_args_base_idx>();
    const auto output_addrgen = TensorAccessor(output_addrgen_args, output_address, page_size);

    bool is_forward = (direction == 0);
    const auto& unicast_route_info = is_forward ? forward_unicast_route_info : backward_unicast_route_info;
    const auto& barrier_multicast_route_info =
        is_forward ? forward_barrier_multicast_route_info : backward_barrier_multicast_route_info;

    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel> mux_connection =
        tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
            fabric_mux_x,
            fabric_mux_y,
            fabric_mux_channel_id,
            fabric_mux_num_buffers_per_channel,
            fabric_mux_channel_buffer_size_bytes,
            fabric_mux_channel_base_address,
            fabric_mux_connection_info_address,
            fabric_mux_connection_handshake_address,
            fabric_mux_flow_control_address,
            fabric_mux_buffer_index_address,
            local_flow_control_address,
            local_teardown_address,
            local_buffer_index_address);
    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel>* mux_connection_handle = &mux_connection;

    // Wait for fabric mux to be ready to accept connections
    tt::tt_fabric::wait_for_fabric_endpoint_ready(
        fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);

    // Connect to fabric mux
    tt::tt_fabric::fabric_client_connect(*mux_connection_handle);

    // Allocate headers
    auto pkt_scatter_hdr = PacketHeaderPool::allocate_header();
    auto pkt_unicast_hdr = PacketHeaderPool::allocate_header();
    auto pkt_sem_inc_hdr = PacketHeaderPool::allocate_header();

    // 1. --------------- Sync with all devices ---------------
    // Encode multicast route
    pkt_sem_inc_hdr->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
        .start_distance_in_hops = static_cast<uint8_t>(barrier_multicast_route_info.start_distance_in_hops),
        .range_hops = static_cast<uint8_t>(barrier_multicast_route_info.range_hops),
    });
    // Encode atomic inc value and flush state
    fabric_multicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        pkt_sem_inc_hdr,
        static_cast<uint8_t>(barrier_multicast_route_info.start_distance_in_hops),
        static_cast<uint8_t>(barrier_multicast_route_info.range_hops),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0,  // ignore noc address update, it is set when the packet is sent.
            static_cast<uint32_t>(1)});

    uint64_t barrier_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_x, out_ready_sem_y, barrier_sem_address, /*noc_id*/ 0);
    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        mux_connection_handle,
        pkt_sem_inc_hdr,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});

    // Wait notify from all other devices
    noc_semaphore_wait_min(barrier_sem_ptr, ring_size - 1);
    noc_semaphore_set(barrier_sem_ptr, 0);
    // --------------------------------------------------------

    // 2. -------------- Prepare headers --------------
    static_assert(num_pages_to_write_per_packet <= 4, "pages per packet > 4 is unsupported");
    uint64_t dummy_noc_addrs[4] = {
        0,
    };
    uint16_t chunk_sizes[3] = {
        page_size,
    };

    // set scatter write state (for possible usage)
    fabric_unicast_noc_scatter_write_set_state<
        UnicastScatterWriteUpdateMask::ChunkSizes |
        UnicastScatterWriteUpdateMask::PayloadSize>(  // TODO : should update chunk and payload here?
        pkt_scatter_hdr,
        static_cast<uint8_t>(unicast_route_info.distance_in_hops),
        NocUnicastScatterCommandHeader(dummy_noc_addrs, chunk_sizes, num_pages_to_write_per_packet),
        page_size * num_pages_to_write_per_packet);
    // set unicast write state (for possible usage)
    fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        pkt_unicast_hdr, static_cast<uint8_t>(unicast_route_info.distance_in_hops), nullptr, page_size);

    // initialize semaphore header for unicast
    fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        pkt_sem_inc_hdr,
        static_cast<uint8_t>(unicast_route_info.distance_in_hops),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0,  // ignore
            static_cast<uint32_t>(1)});

    // record route for each headers
    fabric_set_unicast_route</*target_as_dev*/ false>(pkt_scatter_hdr, unicast_route_info.distance_in_hops);
    fabric_set_unicast_route</*target_as_dev*/ false>(pkt_unicast_hdr, unicast_route_info.distance_in_hops);
    fabric_set_unicast_route</*target_as_dev*/ false>(pkt_sem_inc_hdr, unicast_route_info.distance_in_hops);
    // --------------------------------------------

    // 3. -------------- Remote write local data to next device --------------
    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_x, out_ready_sem_y, out_ready_sem_address, 0);

    uint32_t num_total_input_pages = input_page_id_end - input_page_id_start;
    uint32_t page_device_offset = my_chip_id * num_input_pages_per_device;
    uint32_t page_sync_count = 0;
    uint32_t pages_read = 0;
    while (pages_read < num_total_input_pages) {
        uint32_t pages_remaining_to_read = num_total_input_pages - pages_read;
        uint32_t pages_to_put_in_current_packet = std::min(pages_remaining_to_read, num_pages_to_write_per_packet);

        // Wait local read from reader
        cb_wait_front(cb_output_id, num_pages_to_write_per_packet);
        size_t l1_read_addr = get_read_ptr(cb_output_id);

        // Prepare unicast scatter write for pages > 1
        uint16_t chunk_sizes[3] = {page_size, page_size, page_size};
        uint64_t noc_addrs[4] = {0, 0, 0, 0};
        uint64_t local_noc_addrs[4] = {0, 0, 0, 0};
        for (uint32_t i = 0; i < pages_to_put_in_current_packet; i++) {
            uint32_t page_id = page_device_offset + input_page_id_start + pages_read;
            pages_read++;

            noc_addrs[i] = tt::tt_fabric::linear::addrgen_detail::get_noc_address(output_addrgen, page_id, 0);
            local_noc_addrs[i] = get_noc_addr(page_id, output_addrgen);
        }

        if (is_forward) {
            if (num_pages_to_write_per_packet > 1) {
                // write page chunks
                fabric_unicast_noc_scatter_write_with_state<
                    UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                    UnicastScatterWriteUpdateMask::PayloadSize>(
                    mux_connection_handle,
                    pkt_scatter_hdr,
                    l1_read_addr,
                    NocUnicastScatterCommandHeader(noc_addrs, chunk_sizes, pages_to_put_in_current_packet),
                    page_size * pages_to_put_in_current_packet);
            } else {
                // write single page
                fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                    mux_connection_handle, pkt_unicast_hdr, l1_read_addr, NocUnicastCommandHeader{noc_addrs[0]});
            }

            // Write loccal data to output once in any direction.
            // Arbitrary choice to do it in forward direction.
            for (uint32_t i = 0; i < pages_to_put_in_current_packet; i++) {
                noc_async_write(l1_read_addr + i * page_size, local_noc_addrs[i], page_size);
            }
            noc_async_write_barrier();
        } else {
            if (num_pages_to_write_per_packet > 1) {
                fabric_unicast_noc_scatter_write_with_state<
                    UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                    UnicastScatterWriteUpdateMask::PayloadSize>(
                    mux_connection_handle,
                    pkt_scatter_hdr,
                    l1_read_addr,
                    NocUnicastScatterCommandHeader(noc_addrs, chunk_sizes, pages_to_put_in_current_packet),
                    page_size * pages_to_put_in_current_packet);
            } else {
                fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                    mux_connection_handle, pkt_unicast_hdr, l1_read_addr, NocUnicastCommandHeader{noc_addrs[0]});
            }
        }

        page_sync_count++;
        noc_async_writes_flushed();

        cb_pop_front(cb_output_id, num_pages_to_write_per_packet);

        if (page_sync_count % pages_per_sync == 0) {
            fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                mux_connection_handle,
                pkt_sem_inc_hdr,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, /*unused*/ 0});
        }
    }

    // Final sync if not yet synced
    if (page_sync_count % pages_per_sync != 0) {
        fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            mux_connection_handle,
            pkt_sem_inc_hdr,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, /*unused*/ 0});
    }

    // 4. -------------- Remote write received data to next device --------------
    uint32_t chunks_sent = 0;
    uint32_t chunks_to_send = 0;
    if (is_forward) {
        chunks_to_send = num_devices_leftside - 1;
    } else {
        chunks_to_send = num_devices_rightside - 1;
    }
    while (chunks_sent < chunks_to_send) {
        uint32_t chunk_origin_chip_id = UINT32_MAX;
        if (is_forward) {
            chunk_origin_chip_id = (my_chip_id - (chunks_sent + 1) + ring_size) % ring_size;
        } else {
            chunk_origin_chip_id = (my_chip_id + (chunks_sent + 1) + ring_size) % ring_size;
        }

        uint32_t page_device_offset = chunk_origin_chip_id * num_input_pages_per_device;
        // uint32_t page_direction_offset = direction * num_input_pages_per_direction;
        uint32_t page_start_offset = page_device_offset + input_page_id_start;

        uint32_t pages_read = 0;
        page_sync_count = 0;
        while (pages_read < num_total_input_pages) {
            uint32_t pages_remaining_to_read = num_total_input_pages - pages_read;
            uint32_t pages_to_put_in_current_packet = std::min(pages_remaining_to_read, num_pages_to_write_per_packet);

            cb_wait_front(cb_output_id, num_pages_to_write_per_packet);
            auto l1_read_addr = get_read_ptr(cb_output_id);

            uint16_t chunk_sizes[3] = {page_size, page_size, page_size};
            uint64_t noc_addrs[4] = {0, 0, 0, 0};
            for (uint32_t i = 0; i < pages_to_put_in_current_packet; i++) {
                uint32_t page_id = page_start_offset + pages_read;
                pages_read++;
                noc_addrs[i] = tt::tt_fabric::linear::addrgen_detail::get_noc_address(output_addrgen, page_id, 0);
            }

            if (pages_to_put_in_current_packet > 1) {
                fabric_unicast_noc_scatter_write_with_state<
                    UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                    UnicastScatterWriteUpdateMask::PayloadSize>(
                    mux_connection_handle,
                    pkt_scatter_hdr,
                    l1_read_addr,
                    NocUnicastScatterCommandHeader(noc_addrs, chunk_sizes, pages_to_put_in_current_packet),
                    page_size * pages_to_put_in_current_packet);
            } else {
                fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                    mux_connection_handle, pkt_unicast_hdr, l1_read_addr, NocUnicastCommandHeader{noc_addrs[0]});
            }

            // Flush command
            noc_async_writes_flushed();

            cb_pop_front(cb_output_id, num_pages_to_write_per_packet);

            page_sync_count++;
            if (page_sync_count % pages_per_sync == 0) {
                // 2. unicast output ready semaphore
                fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    mux_connection_handle,
                    pkt_sem_inc_hdr,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, /*unused*/ 0});
            }
            noc_async_writes_flushed();
        }

        if (page_sync_count % pages_per_sync != 0) {
            // 2. unicast output ready semaphore
            fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                mux_connection_handle,
                pkt_sem_inc_hdr,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, /*unused*/ 0});
        }

        chunks_sent++;
    }

    // Wait completion singal for processing requests.
    noc_async_write_barrier();
    noc_async_atomic_barrier();

    // Disconnect fabric mux connection
    tt::tt_fabric::fabric_client_disconnect(*mux_connection_handle);
    if (is_termination_master) {
        auto* termination_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_sync_address);
        // Wait notify from other clients
        noc_semaphore_wait(termination_sync_ptr, num_workers_per_direction - 1);  // except for me
        tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x, fabric_mux_y, fabric_mux_termination_signal_address);
    } else {
        // Notify to termination master
        uint64_t dest_addr =
            safe_get_noc_addr(termination_master_noc_x, termination_master_noc_y, termination_sync_address, 0);
        noc_semaphore_inc(dest_addr, 1);
        noc_async_atomic_barrier();
    }
    noc_async_write_barrier();
}
