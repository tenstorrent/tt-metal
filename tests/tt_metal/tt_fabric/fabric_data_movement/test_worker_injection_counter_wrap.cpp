// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reproduces, in seconds, the failure that the worker-injection write-counter wrap causes after ~2^32
// injected packets.
//
// Background
// ----------
// A worker re-derives its buffer slot cursor on every connection open, from a free-running uint32 that
// the router persists in its own L1 across connections:
//
//   edm_fabric_worker_adapters.hpp:518-520
//     buffer_slot_write_counter.counter = *worker_teardown_addr;              // persisted, wraps at 2^32
//     buffer_slot_write_counter.index   = counter % num_buffers_per_channel;
//
// The router's cursor never passes through a uint32. It is a pure mod-N counter, bumped once per packet
// and zeroed only at fabric bring-up (fabric_erisc_datamover_channels.hpp:59 and :86); the router never
// reads the persisted value back. So after P injected packets the router sits at `P % N` while the
// producer computes `(P % 2^32) % N`. Those agree for every P if and only if N divides 2^32, i.e. if and
// only if N is a power of two.
//
// When they disagree the failure is silent: credits still balance exactly (one decrement per worker
// write, one increment per router transmit, and `get_num_free_write_slots` is a wrap-safe uint32
// difference), so nothing backpressures and nothing faults. The router simply transmits the wrong buffer
// slots, and the tail of each transfer is never transmitted at all.
//
// What this test does
// -------------------
// Waiting for 2^32 real packets is not practical, so the test injects the *consequence* of the wrap
// instead of the wrap itself. It adds `(2^32) % depth` to the persisted producer-position word --
// exactly the discontinuity a real wrap introduces, and small enough that it lands entirely in the
// word's counter field regardless of how that word is laid out.
//
// Before the fix (bare counter word, index re-derived as `counter % depth`) that addend moved the
// producer's slot cursor off the router's and the next transfer lost its tail. After the fix
// (connection_handoff carries the slot index explicitly) the same addend only perturbs the counter,
// the cursor is unaffected, and the transfer completes -- which is the whole point of the fix.
//
// The perturbation is applied while the channel is idle (between two program launches), so it does not
// disturb credit accounting: the worker's free-slot count is `depth - (write_counter - edm_read_counter)`,
// and a skew of a few slots leaves plenty of credit. That is what makes the failure silent in the field
// and hang-free here.
//
// See also FabricStaticSizedChannelsAllocatorTest.WorkerInjectionDepthFitsHandoffIndexField for the
// host-side bound on what the handoff word can represent.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <enchantum/enchantum.hpp>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

#include "fabric_fixture.hpp"
#include "hostdevcommon/fabric_common.h"
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "utils.hpp"

namespace tt::tt_fabric::fabric_router_tests {
namespace {

// Sender channel 0 is always the local-worker injection channel.
constexpr size_t kWorkerSenderChannel = 0;

// Identifies the router-side state backing the fabric connection a worker on `src` opens toward `dst`.
struct InjectionChannel {
    ChipId src_physical_chip_id = 0;
    tt::tt_metal::CoreCoord logical_eth_core;
    chan_id_t eth_channel = 0;
    // config.sender_channels_buffer_index_semaphore_address[0] -- the persisted producer write counter.
    uint32_t persisted_write_counter_addr = 0;
    uint32_t depth = 0;  // VC0 sender channel 0 buffer slots
};

// Mirrors the channel resolution in append_fabric_connection_rt_args (tt_metal/fabric/fabric.cpp) so the
// test perturbs the exact channel the sender kernel will connect to.
InjectionChannel resolve_injection_channel(
    FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id, uint32_t link_idx) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    const auto forwarding_direction = control_plane.get_forwarding_direction(src_fabric_node_id, dst_fabric_node_id);
    TT_FATAL(
        forwarding_direction.has_value(),
        "No forwarding direction from {} to {}",
        src_fabric_node_id,
        dst_fabric_node_id);

    const auto candidate_eth_chans =
        control_plane.get_active_fabric_eth_channels_in_direction(src_fabric_node_id, forwarding_direction.value());
    TT_FATAL(
        link_idx < candidate_eth_chans.size(),
        "Link index {} out of range ({} active fabric eth channels toward {})",
        link_idx,
        candidate_eth_chans.size(),
        dst_fabric_node_id);

    // append_fabric_connection_rt_args indexes this same list with the same link_idx.
    const chan_id_t eth_channel = candidate_eth_chans[link_idx];

    const ChipId src_physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(src_fabric_node_id);
    const auto& soc_desc = cluster.get_soc_desc(src_physical_chip_id);
    const auto logical_eth_core = soc_desc.get_eth_core_for_channel(eth_channel, CoordSystem::LOGICAL);

    const auto& edm_config = control_plane.get_fabric_context().get_builder_context().get_fabric_router_config();
    auto* const static_channel_allocator =
        dynamic_cast<tt::tt_fabric::FabricStaticSizedChannelsAllocator*>(edm_config.channel_allocator.get());
    TT_FATAL(static_channel_allocator != nullptr, "Channel allocator must be a FabricStaticSizedChannelsAllocator");

    return InjectionChannel{
        .src_physical_chip_id = src_physical_chip_id,
        .logical_eth_core = tt::tt_metal::CoreCoord(logical_eth_core.x, logical_eth_core.y),
        .eth_channel = eth_channel,
        .persisted_write_counter_addr =
            static_cast<uint32_t>(edm_config.sender_channels_buffer_index_semaphore_address[kWorkerSenderChannel]),
        .depth =
            static_cast<uint32_t>(static_channel_allocator->get_sender_channel_number_of_slots(kWorkerSenderChannel)),
    };
}

uint32_t read_persisted_write_counter(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& device, const InjectionChannel& channel) {
    std::vector<uint32_t> data;
    tt::tt_metal::detail::ReadFromDeviceL1(
        device->get_devices()[0],
        channel.logical_eth_core,
        channel.persisted_write_counter_addr,
        static_cast<uint32_t>(sizeof(uint32_t)),
        data,
        CoreType::ETH);
    TT_FATAL(!data.empty(), "Failed to read persisted write counter");
    return data[0];
}

void write_persisted_write_counter(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& device,
    const InjectionChannel& channel,
    uint32_t value) {
    std::vector<uint32_t> data = {value};  // WriteToDeviceL1 takes a non-const reference
    tt::tt_metal::detail::WriteToDeviceL1(
        device->get_devices()[0], channel.logical_eth_core, channel.persisted_write_counter_addr, data, CoreType::ETH);
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device->get_devices()[0]->id());
}

// Mirrors connection_handoff in edm_fabric_worker_adapters.hpp, which cannot be included here (it is
// device code). Keep these in sync with COUNTER_BITS there.
constexpr uint32_t kHandoffCounterBits = 24;
constexpr uint32_t kHandoffCounterMask = (1u << kHandoffCounterBits) - 1;

// Add `delta` to the handoff word's counter field, leaving the slot-index field untouched.
//
// Plain `handoff + delta` is wrong: the counter occupies bits [23:0], so an add that overflows it
// carries into the slot byte (and a subtract that underflows borrows from it), silently moving the
// producer's cursor. That would make the test fail for a reason unrelated to what it checks. The
// window is narrow -- only the last few counter values before 2^24 -- but it is reachable and depends
// on how much traffic the channel has already carried, so it must not be left to chance.
uint32_t perturb_handoff_counter(uint32_t handoff, int32_t delta) {
    const uint32_t slot_field = handoff & ~kHandoffCounterMask;
    const uint32_t counter_field = (handoff + static_cast<uint32_t>(delta)) & kHandoffCounterMask;
    return slot_field | counter_field;
}

// Host-side mirror of prng_next in tt_fabric_traffic_gen.hpp.
uint32_t prng_next(uint32_t n) {
    uint32_t x = n;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

// One fabric unicast transfer of `num_packets` packets from `src` to `dst`, landing at `target_address`
// in the receiver's worker L1. No receiver kernel: the destination is zeroed first and validated from the
// host afterwards, so a lost or misplaced packet fails the test instead of wedging a polling kernel.
//
// Returns, for each packet index, whether it arrived with the payload the sender generated for it.
std::vector<bool> run_unicast_transfer_and_check(
    BaseFabricFixture* fixture,
    FabricNodeId src_fabric_node_id,
    FabricNodeId dst_fabric_node_id,
    uint32_t link_idx,
    uint32_t target_address,
    uint32_t num_packets,
    uint32_t time_seed) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    const ChipId src_physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(src_fabric_node_id);
    const ChipId dst_physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(dst_fabric_node_id);
    const auto& sender_device = fixture->get_device(src_physical_chip_id);
    const auto& receiver_device = fixture->get_device(dst_physical_chip_id);

    const tt::tt_metal::CoreCoord sender_logical_core = {0, 0};
    const tt::tt_metal::CoreCoord receiver_logical_core = {1, 0};
    const auto receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    const auto topology = control_plane.get_fabric_context().get_fabric_topology();
    const bool is_2d_fabric = topology == Topology::Mesh;
    const auto worker_mem_map = fixture->generate_worker_mem_map(sender_device, topology);
    const uint32_t payload_size_bytes = worker_mem_map.packet_payload_size_bytes;

    // Zero the destination window so "did not arrive" is unambiguous.
    std::vector<uint32_t> zeros((num_packets * payload_size_bytes) / sizeof(uint32_t), 0);
    tt::tt_metal::detail::WriteToDeviceL1(
        receiver_device->get_devices()[0], receiver_logical_core, target_address, zeros, CoreType::WORKER);
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(receiver_device->get_devices()[0]->id());

    const std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        target_address,
        0 /* use_dram_dst */,
        static_cast<uint32_t>(is_2d_fabric),
        0 /* is_chip_multicast */,
        0 /* additional_dir */};

    std::map<std::string, std::string> defines = {};
    if (is_2d_fabric) {
        defines["FABRIC_2D"] = "";
    }

    auto sender_program = tt::tt_metal::CreateProgram();
    auto sender_kernel = tt::tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_tx.cpp",
        {sender_logical_core},
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args,
            .defines = defines});

    const auto mesh_shape = control_plane.get_physical_mesh_shape(src_fabric_node_id.mesh_id);

    std::vector<uint32_t> sender_runtime_args = {
        worker_mem_map.source_l1_buffer_address,
        payload_size_bytes,
        num_packets,
        time_seed,
        static_cast<uint32_t>(receiver_virtual_core.x),
        static_cast<uint32_t>(receiver_virtual_core.y),
        static_cast<uint32_t>(mesh_shape[1]),
        static_cast<uint32_t>(src_fabric_node_id.chip_id),
        1 /* num_hops */,
        1 /* fwd_range */,
        static_cast<uint32_t>(dst_fabric_node_id.chip_id),
        static_cast<uint32_t>(*dst_fabric_node_id.mesh_id)};

    append_fabric_connection_rt_args(
        src_fabric_node_id, dst_fabric_node_id, link_idx, sender_program, {sender_logical_core}, sender_runtime_args);

    tt::tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    fixture->RunProgramNonblocking(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(sender_device, sender_program);

    // tt_fabric_1d_tx.cpp advances the seed once per packet, then fill_packet_data() writes
    // `seed + k` into the last word of every 16B chunk of the payload. Checking the first marker word
    // of each packet is enough to catch both a shifted stream and a truncated one.
    constexpr uint32_t kMarkerWordOffset = (PACKET_WORD_SIZE_BYTES / sizeof(uint32_t)) - 1;
    const uint32_t payload_words = payload_size_bytes / sizeof(uint32_t);

    // The sender program completing does NOT mean the data has landed.
    //   - The source router acks the connection teardown without draining the channel
    //     (teardown_worker_connection in tt_fabric_utils.h just marks the connection unused and bumps
    //     the worker's teardown semaphore), so packets can still be staged in sender channel 0 when
    //     the sender kernel exits.
    //   - Cluster::l1_barrier is only driver_->l1_membar(), a host-side membar for host-issued L1
    //     access. It says nothing about in-flight NOC or ethernet traffic between devices.
    // So poll the destination window until every packet has arrived, bounded by a deadline. On a
    // healthy transfer this exits on the first read; on a genuine regression it costs the timeout and
    // then reports exactly which packets are missing -- deliberately not a device-side polling
    // receiver, which would turn a real failure into a hang.
    constexpr auto kDeliveryTimeout = std::chrono::seconds(5);
    const auto deadline = std::chrono::steady_clock::now() + kDeliveryTimeout;

    std::vector<bool> packet_ok(num_packets, false);
    uint32_t missing = 0;
    for (;;) {
        tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(receiver_device->get_devices()[0]->id());
        std::vector<uint32_t> received;
        tt::tt_metal::detail::ReadFromDeviceL1(
            receiver_device->get_devices()[0],
            receiver_logical_core,
            target_address,
            num_packets * payload_size_bytes,
            received,
            CoreType::WORKER);

        missing = 0;
        uint32_t seed = time_seed;
        for (uint32_t packet = 0; packet < num_packets; ++packet) {
            seed = prng_next(seed);
            const uint32_t word_idx = (packet * payload_words) + kMarkerWordOffset;
            TT_FATAL(word_idx < received.size(), "Readback too short");
            packet_ok[packet] = (received[word_idx] == seed);
            missing += packet_ok[packet] ? 0u : 1u;
        }

        if (missing == 0 || std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (missing != 0) {
        log_info(
            tt::LogTest,
            "{} of {} packets still missing after waiting {} s for delivery",
            missing,
            num_packets,
            std::chrono::duration_cast<std::chrono::seconds>(kDeliveryTimeout).count());
    }
    return packet_ok;
}

uint32_t count_missing(const std::vector<bool>& packet_ok) {
    uint32_t missing = 0;
    for (const bool ok : packet_ok) {
        missing += ok ? 0u : 1u;
    }
    return missing;
}

}  // namespace

// Emulates the 2^32 wrap of the persisted worker write counter and asserts the fabric still delivers.
//
// Passes trivially when the emulated skew is 0 (a power-of-two depth, where a real wrap is harmless
// even without the fix). At any other depth it is a genuine reproduction: it fails against the old
// bare-counter handoff and passes once the handoff carries the slot index explicitly.
TEST_F(Fabric2DFixture, WorkerInjectionSlotCursorSurvivesWriteCounterWrap) {
    FabricNodeId src_fabric_node_id(MeshId{0}, 0);
    FabricNodeId dst_fabric_node_id(MeshId{0}, 0);
    ChipId src_physical_chip_id = 0;
    ChipId dst_physical_chip_id = 0;
    // Any single-hop neighbour will do. A [32,1] line has N/S neighbours only, an [8,4] mesh has
    // both, so try every cardinal direction rather than assuming East.
    bool connection_found = false;
    for (const auto direction : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        if (find_device_with_neighbor_in_direction(
                this, src_fabric_node_id, dst_fabric_node_id, src_physical_chip_id, dst_physical_chip_id, direction)) {
            connection_found = true;
            log_info(
                tt::LogTest, "Using a {} hop for the injection channel under test", enchantum::to_string(direction));
            break;
        }
    }
    if (!connection_found) {
        GTEST_SKIP() << "No device with a single-hop neighbour in any cardinal direction";
    }

    const auto available_links = get_forwarding_link_indices(src_fabric_node_id, dst_fabric_node_id);
    ASSERT_FALSE(available_links.empty()) << "No forwarding links between the selected chips";
    const uint32_t link_idx = available_links[0];

    const auto channel = resolve_injection_channel(src_fabric_node_id, dst_fabric_node_id, link_idx);
    const auto& sender_device = get_device(channel.src_physical_chip_id);

    // The discontinuity a uint32 wrap introduces into `counter % depth`. Zero iff depth divides 2^32.
    const uint32_t wrap_skew = static_cast<uint32_t>((1ULL << 32) % channel.depth);

    log_info(
        tt::LogTest,
        "Injection channel: chip {} eth chan {} logical core ({},{}) depth {} -> emulated wrap skew {} slots",
        channel.src_physical_chip_id,
        channel.eth_channel,
        channel.logical_eth_core.x,
        channel.logical_eth_core.y,
        channel.depth,
        wrap_skew);
    if (wrap_skew != 0) {
        log_warning(
            tt::LogTest,
            "Injection channel depth {} is not a power of two, so the producer's `counter % depth` "
            "resync diverges from the router's mod-{} cursor by {} slots the first time the persisted "
            "write counter wraps at 2^32.",
            channel.depth,
            channel.depth,
            wrap_skew);
    }

    const auto worker_mem_map = generate_worker_mem_map(sender_device, Topology::Mesh);
    const uint32_t payload_size_bytes = worker_mem_map.packet_payload_size_bytes;

    // The two phases need disjoint destination windows so that the stale packets phase 1 leaves staged in
    // the router cannot land in the window phase 2 validates. Both must fit in the data space the mem map
    // reserves between target_address and test_results_address.
    const uint32_t data_space_bytes = worker_mem_map.test_results_address - worker_mem_map.target_address;
    const uint32_t max_packets_per_window = (data_space_bytes / 2) / payload_size_bytes;
    // More than one trip around the ring, and comfortably more than any plausible skew.
    const uint32_t num_packets = std::min<uint32_t>((2 * channel.depth) + 8, max_packets_per_window);
    if (num_packets < wrap_skew + 4) {
        GTEST_SKIP() << "Data space fits only " << num_packets << " packets of " << payload_size_bytes
                     << " B per window, too few to observe a " << wrap_skew << " slot skew";
    }

    const uint32_t phase1_target = worker_mem_map.target_address;
    const uint32_t phase2_target = phase1_target + (num_packets * payload_size_bytes);

    // Phase 1: an ordinary transfer. Establishes the connection and leaves the persisted counter in a
    // state consistent with the router's cursor.
    const auto phase1 = run_unicast_transfer_and_check(
        this, src_fabric_node_id, dst_fabric_node_id, link_idx, phase1_target, num_packets, 0x1234abcd);
    ASSERT_EQ(count_missing(phase1), 0u) << "Baseline transfer failed before any perturbation was applied";

    // The channel is idle here: the worker closed its connection, so the persisted counter is the
    // authoritative producer position and no one will overwrite it before the next open().
    const uint32_t counter_before = read_persisted_write_counter(sender_device, channel);
    const uint32_t counter_perturbed = perturb_handoff_counter(counter_before, static_cast<int32_t>(wrap_skew));
    write_persisted_write_counter(sender_device, channel, counter_perturbed);
    log_info(
        tt::LogTest,
        "Persisted handoff word 0x{:08x} -> 0x{:08x} (counter field +{}, emulating the 2^32 wrap)",
        counter_before,
        counter_perturbed,
        wrap_skew);

    // Phase 2: identical transfer across the perturbed connection.
    const auto phase2 = run_unicast_transfer_and_check(
        this, src_fabric_node_id, dst_fabric_node_id, link_idx, phase2_target, num_packets, 0x5678ef01);

    // Re-align the persisted counter with the router's cursor so the rest of the suite is unaffected.
    if (wrap_skew != 0) {
        const uint32_t counter_after = read_persisted_write_counter(sender_device, channel);
        write_persisted_write_counter(
            sender_device, channel, perturb_handoff_counter(counter_after, -static_cast<int32_t>(wrap_skew)));
    }

    const uint32_t missing = count_missing(phase2);
    EXPECT_EQ(missing, 0u) << missing << " of " << num_packets
                           << " packets were lost or misplaced after the persisted write counter wrapped. "
                              "Injection channel depth is "
                           << channel.depth
                           << ", which does not divide 2^32, "
                              "so a wrap moves the producer's slot cursor "
                           << wrap_skew
                           << " slots off the router's. The connection handoff must carry the slot index "
                              "explicitly instead of re-deriving it as `counter % depth` -- see "
                              "connection_handoff in edm_fabric_worker_adapters.hpp.";
}

}  // namespace tt::tt_fabric::fabric_router_tests
