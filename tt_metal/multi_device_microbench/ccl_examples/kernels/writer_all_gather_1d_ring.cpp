
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

#include <cstdint>
#include <utility>

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
constexpr uint32_t num_tiles_to_write_per_packet = get_compile_time_arg_val(3);
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
    // fabric mux runtime arguments
    const bool is_termination_master = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint8_t fabric_mux_x = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint8_t fabric_mux_y = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t fabric_mux_channel_base_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t fabric_mux_connection_info_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t fabric_mux_connection_handshake_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t fabric_mux_flow_control_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t fabric_mux_buffer_index_address = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_channel_credits_stream_id = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));

    const uint32_t termination_sync_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t local_fabric_mux_status_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t local_flow_control_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t local_teardown_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t local_buffer_index_address = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t termination_master_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t termination_master_noc_y = get_arg_val<uint32_t>(arg_idx++);

    // semaphore ptrs
    auto out_ready_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_address);
    auto barrier_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_address)

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
            fabric_mux_channel_credits_stream_id,
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

    // Sync with all devices
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
}
