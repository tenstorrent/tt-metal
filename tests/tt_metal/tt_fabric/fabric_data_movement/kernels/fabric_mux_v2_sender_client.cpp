// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#if defined(FABRIC_2D)
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#endif
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_v2_sender.hpp"

constexpr uint32_t test_results_address = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
constexpr uint32_t receiver_slots_base_address = get_compile_time_arg_val(2);
constexpr uint32_t credit_handshake_address = get_compile_time_arg_val(3);
constexpr bool is_2d_fabric = get_compile_time_arg_val(4) != 0;
constexpr bool kEagerStaging = get_compile_time_arg_val(5) != 0;
constexpr bool kUseStatefulLane = get_compile_time_arg_val(6) != 0;
constexpr uint32_t kTestPattern = get_compile_time_arg_val(7);
constexpr uint8_t kStatusReadTrid = static_cast<uint8_t>(get_compile_time_arg_val(8));
constexpr bool kRandomizePayloadSizeAndDelay = get_compile_time_arg_val(9) != 0;

constexpr uint32_t kMaxInterPacketDelayCycles = 1000;
constexpr uint32_t kDelaySeedXor = 0xA5A5A5A5u;

enum class StagingTestPattern : uint32_t {
    BasicSend = 0,
    ZeroPacket = 1,
    StageThenFlush = 2,
    OpportunisticFlush = 3,
    StageAndClose = 4,
    StageRingFull = 5,
    StageIdle = 6,
};

using Sender = tt::tt_fabric::FabricMuxV2Sender<kEagerStaging>;

struct SendContext {
    Sender& sender;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header;
    uint32_t packet_header_buffer_address;
    uint32_t payload_buffer_address;
    uint32_t max_packet_payload_size_bytes;
    uint32_t packet_payload_size_bytes;
    uint8_t receiver_noc_x;
    uint8_t receiver_noc_y;
    uint32_t receiver_slots_base_address;
    uint8_t num_hops;
    uint8_t dst_device_id;
    uint16_t dst_mesh_id;
    uint32_t seed;
    uint32_t delay_seed;
    tt_l1_ptr uint32_t* payload_start_ptr;
    volatile tt_l1_ptr uint32_t* granted_credits_ptr;
    uint32_t credits_consumed;
    uint32_t num_receiver_slots;
    uint32_t receiver_slot_id;
    uint64_t bytes_sent;
    bool has_flushed = false;
};

FORCE_INLINE void prepare_packet_header(
    SendContext& ctx, const tt::tt_fabric::NocUnicastCommandHeader& dest_command_header) {
#if defined(FABRIC_2D)
    if constexpr (is_2d_fabric) {
        fabric_set_unicast_route(ctx.packet_header, ctx.dst_device_id, ctx.dst_mesh_id);
    } else
#endif
    {
        ctx.packet_header->to_chip_unicast(ctx.num_hops);
    }
    ctx.packet_header->to_noc_unicast_write(dest_command_header, ctx.packet_payload_size_bytes);
}

FORCE_INLINE void apply_inter_packet_delay(SendContext& ctx) {
    if constexpr (!kRandomizePayloadSizeAndDelay) {
        return;
    }
    ctx.delay_seed = prng_next(ctx.delay_seed);
    const uint32_t delay_cycles = 1 + (ctx.delay_seed % kMaxInterPacketDelayCycles);
    for (volatile uint32_t delay = 0; delay < delay_cycles; ++delay) {
    }
}

template <bool Stateful>
FORCE_INLINE void send_one_packet(SendContext& ctx) {
    // Receiver slots are always strided by the configured max payload size.
    const uint64_t dest_noc_addr = get_noc_addr(
        ctx.receiver_noc_x,
        ctx.receiver_noc_y,
        ctx.receiver_slots_base_address + (ctx.receiver_slot_id * ctx.max_packet_payload_size_bytes));
    const auto dest_command_header = tt::tt_fabric::NocUnicastCommandHeader{dest_noc_addr};
    if constexpr (Stateful) {
        ctx.sender.wait_for_empty_write_slot();
        prepare_packet_header(ctx, dest_command_header);
        ctx.sender.template send_current_slot_stateful_non_blocking</*posted=*/false>(
            ctx.payload_buffer_address, ctx.packet_payload_size_bytes, ctx.packet_header_buffer_address);
    } else {
#if defined(FABRIC_2D)
        if constexpr (is_2d_fabric) {
            tt::tt_fabric::mesh::experimental::fabric_unicast_noc_unicast_write(
                &ctx.sender,
                ctx.packet_header,
                ctx.dst_device_id,
                ctx.dst_mesh_id,
                ctx.payload_buffer_address,
                ctx.packet_payload_size_bytes,
                dest_command_header);
        } else
#endif
        {
            tt::tt_fabric::linear::experimental::fabric_unicast_noc_unicast_write(
                &ctx.sender,
                ctx.packet_header,
                ctx.payload_buffer_address,
                ctx.packet_payload_size_bytes,
                dest_command_header,
                ctx.num_hops);
        }
    }
}

template <bool Stateful>
FORCE_INLINE void send_packet_steady(SendContext& ctx) {
    if (!ctx.has_flushed && ctx.sender.is_staging_ring_full()) {
        ctx.sender.template flush<true>();
        ctx.has_flushed = true;
    }
    while (ctx.granted_credits_ptr[0] == ctx.credits_consumed) {
        invalidate_l1_cache();
    }
    ctx.seed = prng_next(ctx.seed);
    if constexpr (kRandomizePayloadSizeAndDelay) {
        ctx.packet_payload_size_bytes = derive_aligned_payload_size_bytes(ctx.seed, ctx.max_packet_payload_size_bytes);
    }
    fill_packet_data(ctx.payload_start_ptr, ctx.packet_payload_size_bytes / 16, ctx.seed);

    // Receiver returns credits by monotonically incrementing granted_credits_ptr.
    // Keep consumption local so WH does not need a RISC-V atomic decrement.
    ctx.credits_consumed += 1;

    send_one_packet<Stateful>(ctx);

    ctx.bytes_sent += ctx.packet_payload_size_bytes;
    ctx.receiver_slot_id += 1;
    if (ctx.receiver_slot_id == ctx.num_receiver_slots) {
        ctx.receiver_slot_id = 0;
    }
    noc_async_writes_flushed();
    apply_inter_packet_delay(ctx);
}

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t packet_header_buffer_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t payload_buffer_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t packet_payload_size_bytes = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_packets = get_arg_val<uint32_t>(arg_idx++);
    uint32_t seed = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t receiver_noc_x = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint8_t receiver_noc_y = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t num_receiver_slots = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t num_hops = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint8_t dst_device_id = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t stage_count = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t idle_cycles = get_arg_val<uint32_t>(arg_idx++);

    auto sender = Sender::build_from_args(arg_idx);

    auto test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_address);
    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    auto packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    zero_l1_buf(reinterpret_cast<tt_l1_ptr uint32_t*>(packet_header_buffer_address), sizeof(PACKET_HEADER_TYPE));

    auto granted_credits_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(credit_handshake_address);
    granted_credits_ptr[0] = num_receiver_slots;

    auto payload_start_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(payload_buffer_address);
    uint64_t bytes_sent = 0;
    uint32_t credits_consumed = 0;
    uint32_t receiver_slot_id = 0;

    if constexpr (kUseStatefulLane) {
        sender.open(kStatusReadTrid);
        sender.template setup_stateful_send_cmd_bufs</*posted=*/false>();
    } else {
        sender.open(kStatusReadTrid);
    }

    const auto pattern = static_cast<StagingTestPattern>(kTestPattern);

    SendContext ctx{
        .sender = sender,
        .packet_header = packet_header,
        .packet_header_buffer_address = packet_header_buffer_address,
        .payload_buffer_address = payload_buffer_address,
        .max_packet_payload_size_bytes = packet_payload_size_bytes,
        .packet_payload_size_bytes = packet_payload_size_bytes,
        .receiver_noc_x = receiver_noc_x,
        .receiver_noc_y = receiver_noc_y,
        .receiver_slots_base_address = receiver_slots_base_address,
        .num_hops = num_hops,
        .dst_device_id = dst_device_id,
        .dst_mesh_id = dst_mesh_id,
        .seed = seed,
        .delay_seed = seed ^ kDelaySeedXor,
        .payload_start_ptr = payload_start_ptr,
        .granted_credits_ptr = granted_credits_ptr,
        .credits_consumed = credits_consumed,
        .num_receiver_slots = num_receiver_slots,
        .receiver_slot_id = receiver_slot_id,
        .bytes_sent = bytes_sent,
    };

    auto run_steady = [&ctx](uint32_t start, uint32_t end) {
        for (uint32_t packet_idx = start; packet_idx < end; ++packet_idx) {
            if constexpr (kUseStatefulLane) {
                send_packet_steady<true>(ctx);
            } else {
                send_packet_steady<false>(ctx);
            }
        }
    };

    if (pattern == StagingTestPattern::BasicSend || pattern == StagingTestPattern::ZeroPacket) {
        run_steady(0, num_packets);
    } else if (pattern == StagingTestPattern::StageThenFlush) {
        run_steady(0, stage_count);
        sender.template flush<true>();
        ctx.has_flushed = true;
        run_steady(stage_count, num_packets);
    } else if (pattern == StagingTestPattern::OpportunisticFlush) {
        run_steady(0, stage_count);
        sender.template flush<false>();
        sender.template flush<false>();
        sender.template flush<true>();
        ctx.has_flushed = true;
        run_steady(stage_count, num_packets);
    } else if (pattern == StagingTestPattern::StageAndClose) {
        run_steady(0, stage_count);
    } else if (pattern == StagingTestPattern::StageRingFull) {
        run_steady(0, stage_count);
        run_steady(stage_count, num_packets);
    } else if (pattern == StagingTestPattern::StageIdle) {
        run_steady(0, stage_count);
        for (volatile uint32_t delay = 0; delay < idle_cycles; ++delay) {
        }
        sender.template flush<true>();
        ctx.has_flushed = true;
        run_steady(stage_count, num_packets);
    }

    bytes_sent = ctx.bytes_sent;

    noc_async_write_barrier();

    const uint32_t expected_granted_credits = num_receiver_slots + num_packets;
    if constexpr (!kEagerStaging) {
        while (granted_credits_ptr[0] != expected_granted_credits) {
            invalidate_l1_cache();
        }
    }

    sender.close();

    if constexpr (kEagerStaging) {
        while (granted_credits_ptr[0] != expected_granted_credits) {
            invalidate_l1_cache();
        }
    }

    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    test_results[TT_FABRIC_WORD_CNT_INDEX] = static_cast<uint32_t>(bytes_sent);
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = static_cast<uint32_t>(bytes_sent >> 32);
    test_results[TX_TEST_IDX_NPKT] = num_packets;
}
