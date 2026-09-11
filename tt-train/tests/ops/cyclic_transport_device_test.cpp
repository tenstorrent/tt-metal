// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Transport of the cyclic SDPA backward pass, checked with a checksum payload
// instead of tensors.
//
// The point of carrying a checksum rather than gradients is that every
// transport bug becomes an integer mismatch with a name. A packet that misses
// an update, gets one twice, or is read before its producer's write is
// visible changes a count or a sum; nothing has to be inferred from a
// tolerance on a gradient.
//
// TRANSPORT_DRAM is Algorithm 2's row traffic: load the row packet from DRAM
// every timestep, update it, write it back, and let a chip-wide barrier order
// the updates of one row across timesteps. The relay replaces those loads
// with NoC forwards in a later step; the barrier and the column-residency
// rules stay as they are here.
//
// What a failure means:
//   * row update count wrong -- the schedule visited a pair twice or not at
//     all, or a write was lost;
//   * row checksum wrong with the right count -- a core processed the wrong
//     pair, or an update was applied to the wrong row;
//   * row checksum short by a whole term -- a read saw a stale page, so the
//     barrier is not ordering the updates;
//   * column checksum containing poison -- a first visit read the column
//     gradients from DRAM instead of initialising them locally.

#include <gtest/gtest.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "metal/ops/cyclic_sdpa_bw/device/parity_snake.hpp"

namespace {

using namespace ttml::metal::ops::cyclic_sdpa_bw;
namespace tt_dist = tt::tt_metal::distributed;

constexpr uint32_t kPacketWords = 8;  // must match the kernel
constexpr uint32_t kPacketBytes = kPacketWords * sizeof(uint32_t);
constexpr uint32_t kPoison = 0xDEADBEEFu;
constexpr const char* kKernelPath =
    "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/kernels/dataflow/cyclic_transport.cpp";

constexpr uint32_t row_term(uint32_t i, uint32_t j) {
    return i * 4096u + j;
}
constexpr uint32_t column_term(uint32_t i, uint32_t j) {
    return j * 4096u + i;
}

struct Result {
    std::vector<uint32_t> packets;  // T pages of kPacketWords
    std::vector<uint32_t> columns;
    std::vector<uint32_t> stats;  // C pages: per-core traffic counters
};

// Fields of a per-core stats page, in the order the kernel writes them.
enum Stat : uint32_t {
    kRowLoads = 0,
    kRowSpills = 1,
    kForwardsSent = 2,
    kForwardsReceived = 3,
    kSelfForwards = 4,
    kColumnChanges = 5,
};

uint32_t stat_total(const Result& result, uint32_t C, Stat field) {
    uint32_t total = 0;
    for (uint32_t c = 0; c < C; ++c) {
        total += result.stats[c * kPacketWords + field];
    }
    return total;
}

// Compile-time knobs of the kernel. Defaults are the real protocol.
struct Options {
    uint32_t skew_iters = 0;      // spin at the top of each timestep on odd cores
    uint32_t rmw_spin_iters = 0;  // spin between reading a packet and writing it back
    uint32_t rmw_spin_core = 0;   // restrict that spin to one core (0 = all)
    bool no_barrier = false;      // drop the chip-wide barrier
    bool relay = false;           // forward packets over the NoC instead of via DRAM
    bool stale_tag_ok = false;    // accept any positive readiness tag
    bool no_credit_wait = false;  // forward without the receiver's permission
    bool readiness_before_payload = false;  // publish readiness before the payload lands
};

Result run_transport(uint32_t C, uint32_t grid_w, uint32_t grid_h, const Options& options = {}) {
    using namespace tt::tt_metal;

    auto& mesh = ttml::autograd::ctx().get_device();
    const CyclicSchedule sched(C);
    const uint32_t T = sched.T();

    const auto local_config = tt_dist::DeviceLocalBufferConfig{
        .page_size = kPacketBytes,
        .buffer_type = BufferType::DRAM,
    };
    const auto buffer_config = tt_dist::ReplicatedBufferConfig{.size = kPacketBytes * T};
    auto packets = tt_dist::MeshBuffer::create(buffer_config, local_config, &mesh);
    auto columns = tt_dist::MeshBuffer::create(buffer_config, local_config, &mesh);
    const auto stats_config = tt_dist::ReplicatedBufferConfig{.size = kPacketBytes * C};
    auto stats = tt_dist::MeshBuffer::create(stats_config, local_config, &mesh);

    // Row packets start with their row id and a zero accumulator. Column
    // pages start poisoned: a first visit must not read them.
    std::vector<uint32_t> packet_init(kPacketWords * T, 0u);
    std::vector<uint32_t> column_init(kPacketWords * T, kPoison);
    for (uint32_t i = 1; i <= T; ++i) {
        packet_init[(i - 1u) * kPacketWords] = i;
    }
    auto& cq = mesh.mesh_command_queue();
    tt_dist::WriteShard(cq, packets, packet_init, tt_dist::MeshCoordinate(0, 0), true);
    tt_dist::WriteShard(cq, columns, column_init, tt_dist::MeshCoordinate(0, 0), true);

    auto program = CreateProgram();
    const auto region = CoreRange(CoreCoord{0, 0}, CoreCoord{grid_w - 1, grid_h - 1});

    const auto make_cb = [&](uint32_t index, uint32_t bytes) {
        CreateCircularBuffer(
            program,
            region,
            CircularBufferConfig(bytes, {{index, tt::DataFormat::UInt32}}).set_page_size(index, bytes));
    };
    make_cb(tt::CBIndex::c_0, kPacketBytes);  // receive slot 0
    make_cb(tt::CBIndex::c_1, kPacketBytes);  // receive slot 1
    make_cb(tt::CBIndex::c_2, kPacketBytes);  // resident column state
    make_cb(tt::CBIndex::c_3, 32);            // source word for control writes
    make_cb(tt::CBIndex::c_4, kPacketBytes);  // per-core stats page

    // All semaphores start at zero. The relay's initial slot permissions are
    // granted by the receivers at startup rather than baked in here, because
    // CreateSemaphore gives every core in the range the same initial value and
    // those permissions differ per core.
    const uint32_t arrive_sem = CreateSemaphore(program, region, 0);
    const uint32_t release_sem = CreateSemaphore(program, region, 0);
    const uint32_t ready0_sem = CreateSemaphore(program, region, 0);
    const uint32_t ready1_sem = CreateSemaphore(program, region, 0);
    const uint32_t credit_prev_sem = CreateSemaphore(program, region, 0);
    const uint32_t credit_next_sem = CreateSemaphore(program, region, 0);
    const uint32_t credit_self_sem = CreateSemaphore(program, region, 0);

    std::vector<uint32_t> compile_args = {
        C, arrive_sem, release_sem, ready0_sem, ready1_sem,
        credit_prev_sem, credit_next_sem, credit_self_sem};
    tt::tt_metal::TensorAccessorArgs(*packets->get_reference_buffer()).append_to(compile_args);
    tt::tt_metal::TensorAccessorArgs(*columns->get_reference_buffer()).append_to(compile_args);
    tt::tt_metal::TensorAccessorArgs(*stats->get_reference_buffer()).append_to(compile_args);

    std::map<std::string, std::string> defines;
    defines[options.relay ? "TRANSPORT_RELAY" : "TRANSPORT_DRAM"] = "1";
    if (options.skew_iters != 0) {
        defines["SKEW_ITERS"] = std::to_string(options.skew_iters);
    }
    if (options.rmw_spin_iters != 0) {
        defines["RMW_SPIN_ITERS"] = std::to_string(options.rmw_spin_iters);
    }
    if (options.rmw_spin_core != 0) {
        defines["RMW_SPIN_CORE"] = std::to_string(options.rmw_spin_core);
    }
    if (options.no_barrier) {
        defines["FAULT_NO_BARRIER"] = "1";
    }
    if (options.stale_tag_ok) {
        defines["FAULT_STALE_TAG_OK"] = "1";
    }
    if (options.no_credit_wait) {
        defines["FAULT_NO_CREDIT_WAIT"] = "1";
    }
    if (options.readiness_before_payload) {
        defines["FAULT_READINESS_BEFORE_PAYLOAD"] = "1";
    }

    const auto kernel = CreateKernel(
        program,
        kKernelPath,
        region,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = compile_args,
            .defines = defines});

    // Logical core 1 coordinates the barrier. Every core needs its address to
    // arrive, and it needs the region's bounds to publish the release.
    const auto coordinator_logical = placement_of(C, grid_w, 1);
    const auto coordinator = mesh.worker_core_from_logical_core(
        CoreCoord{coordinator_logical.x, coordinator_logical.y});
    const auto mcast_start = mesh.worker_core_from_logical_core(CoreCoord{0, 0});
    const auto mcast_end = mesh.worker_core_from_logical_core(CoreCoord{grid_w - 1, grid_h - 1});

    // A core exchanges packets only with its snake neighbours, so those are
    // the only coordinates it needs.
    const auto noc_of_core = [&](uint32_t core) {
        const auto xy = placement_of(C, grid_w, core);
        return mesh.worker_core_from_logical_core(CoreCoord{xy.x, xy.y});
    };

    for (uint32_t c = 1; c <= C; ++c) {
        const auto xy = placement_of(C, grid_w, c);
        const auto neighbors = snake_neighbors(C, c);
        const bool has_prev = neighbors.prev != kNoCore;
        const bool has_next = neighbors.next != kNoCore;
        const auto prev = noc_of_core(has_prev ? neighbors.prev : c);
        const auto next = noc_of_core(has_next ? neighbors.next : c);
        SetRuntimeArgs(
            program,
            kernel,
            CoreCoord{xy.x, xy.y},
            {c,
             packets->address(),
             columns->address(),
             static_cast<uint32_t>(coordinator.x),
             static_cast<uint32_t>(coordinator.y),
             static_cast<uint32_t>(mcast_start.x),
             static_cast<uint32_t>(mcast_start.y),
             static_cast<uint32_t>(mcast_end.x),
             static_cast<uint32_t>(mcast_end.y),
             c == 1u ? 1u : 0u,
             stats->address(),
             has_prev ? 1u : 0u,
             static_cast<uint32_t>(prev.x),
             static_cast<uint32_t>(prev.y),
             has_next ? 1u : 0u,
             static_cast<uint32_t>(next.x),
             static_cast<uint32_t>(next.y)});
    }

    auto workload = tt_dist::MeshWorkload();
    workload.add_program(tt_dist::MeshCoordinateRange(mesh.shape()), std::move(program));
    tt_dist::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    Result out;
    out.packets.resize(kPacketWords * T, 0u);
    out.columns.resize(kPacketWords * T, 0u);
    out.stats.resize(kPacketWords * C, 0u);
    tt_dist::ReadShard(cq, out.packets, packets, tt_dist::MeshCoordinate(0, 0), true);
    tt_dist::ReadShard(cq, out.columns, columns, tt_dist::MeshCoordinate(0, 0), true);
    tt_dist::ReadShard(cq, out.stats, stats, tt_dist::MeshCoordinate(0, 0), true);
    return out;
}

// True if the transport did exactly what the schedule prescribes. Used by
// the fault tests, which need a verdict rather than a failed expectation.
bool transport_is_correct(uint32_t C, const Result& result) {
    const CyclicSchedule sched(C);
    const uint32_t T = sched.T();
    for (uint32_t i = 1; i <= T; ++i) {
        const uint32_t* page = result.packets.data() + (i - 1u) * kPacketWords;
        uint32_t expected_sum = 0;
        for (uint32_t j = 1; j <= i; ++j) {
            expected_sum += row_term(i, j);
        }
        if (page[0] != i || page[1] != i || page[2] != expected_sum) {
            return false;
        }
    }
    for (uint32_t j = 1; j <= T; ++j) {
        const uint32_t* page = result.columns.data() + (j - 1u) * kPacketWords;
        uint32_t expected_sum = 0;
        for (uint32_t i = j; i <= T; ++i) {
            expected_sum += column_term(i, j);
        }
        if (page[0] != j || page[1] != T - j + 1u || page[2] != expected_sum) {
            return false;
        }
    }
    return true;
}

bool grid_fits(uint32_t grid_w, uint32_t grid_h) {
    const auto grid = ttml::autograd::ctx().get_device().compute_with_storage_grid_size();
    return grid_w <= grid.x && grid_h <= grid.y;
}

// The traffic counters the paper predicts. These are structural: they hold
// whatever the payload is, and they catch a relay that quietly falls back to
// DRAM or forwards a packet nobody consumes.
void check_counters(uint32_t C, const Result& result, bool relay) {
    const CyclicSchedule sched(C);
    const uint32_t T = sched.T();

    // Streaks, from the schedule: every start is a DRAM load and every end a
    // spill. The paper's S = 5C - 3.
    uint32_t streak_starts = 0;
    uint32_t forwards_expected = 0;
    for (uint32_t i = 1; i <= T; ++i) {
        for (uint32_t t = 0; t <= T; ++t) {
            if (sched.is_streak_start(i, t)) {
                ++streak_starts;
            }
            if (t < T && sched.is_active(i, t) && sched.is_active(i, t + 1u)) {
                ++forwards_expected;
            }
        }
    }
    ASSERT_EQ(streak_starts, 5u * C - 3u) << "the schedule itself disagrees with 5C - 3";

    if (!relay) {
        // Every timestep goes through DRAM, and nothing is forwarded.
        EXPECT_EQ(stat_total(result, C, kRowLoads), C * (T + 1u));
        EXPECT_EQ(stat_total(result, C, kForwardsSent), 0u);
        EXPECT_EQ(stat_total(result, C, kSelfForwards), 0u);
    } else {
        EXPECT_EQ(stat_total(result, C, kRowLoads), streak_starts)
            << "row packets loaded from DRAM should equal the number of streaks";
        EXPECT_EQ(stat_total(result, C, kRowSpills), streak_starts)
            << "every streak ends in exactly one spill";
        EXPECT_EQ(stat_total(result, C, kForwardsSent), stat_total(result, C, kForwardsReceived))
            << "a forward was sent that nobody consumed, or consumed without being sent";
        EXPECT_EQ(stat_total(result, C, kSelfForwards), 1u)
            << "the schedule has exactly one self-transition";
        EXPECT_EQ(stat_total(result, C, kForwardsSent) + stat_total(result, C, kSelfForwards),
                  forwards_expected)
            << "the number of packet moves disagrees with the schedule";
    }
    // Three residency intervals per core, so two changes each.
    EXPECT_EQ(stat_total(result, C, kColumnChanges), 2u * C);
}

void check_transport(uint32_t C, uint32_t grid_w, uint32_t grid_h, const Options& options = {}) {
    if (!grid_fits(grid_w, grid_h)) {
        GTEST_SKIP() << "C=" << C << " needs " << grid_w << "x" << grid_h;
    }

    const CyclicSchedule sched(C);
    const uint32_t T = sched.T();
    const auto result = run_transport(C, grid_w, grid_h, options);

    for (uint32_t i = 1; i <= T; ++i) {
        const uint32_t* page = result.packets.data() + (i - 1u) * kPacketWords;
        uint32_t expected_sum = 0;
        for (uint32_t j = 1; j <= i; ++j) {
            expected_sum += row_term(i, j);
        }
        EXPECT_EQ(page[0], i) << "C=" << C << " row " << i << ": row id changed";
        EXPECT_EQ(page[1], i) << "C=" << C << " row " << i << ": update count";
        EXPECT_EQ(page[2], expected_sum) << "C=" << C << " row " << i << ": checksum";
    }

    for (uint32_t j = 1; j <= T; ++j) {
        const uint32_t* page = result.columns.data() + (j - 1u) * kPacketWords;
        uint32_t expected_sum = 0;
        for (uint32_t i = j; i <= T; ++i) {
            expected_sum += column_term(i, j);
        }
        EXPECT_EQ(page[0], j) << "C=" << C << " column " << j << ": column id";
        EXPECT_EQ(page[1], T - j + 1u) << "C=" << C << " column " << j << ": visit count";
        EXPECT_EQ(page[2], expected_sum) << "C=" << C << " column " << j << ": checksum";
        EXPECT_NE(page[2], kPoison) << "C=" << C << " column " << j
                                    << ": first visit read gradients from DRAM";
    }

    check_counters(C, result, options.relay);
}

void check_relay(uint32_t C, uint32_t grid_w, uint32_t grid_h, Options options = {}) {
    options.relay = true;
    check_transport(C, grid_w, grid_h, options);
}

}  // namespace

TEST(CyclicTransportDramTest, FourCores) {
    check_transport(4, 2, 2);
}

TEST(CyclicTransportDramTest, EightCores) {
    check_transport(8, 4, 2);
}

TEST(CyclicTransportDramTest, SixteenCores) {
    check_transport(16, 4, 4);
}

TEST(CyclicTransportDramTest, ThirtyTwoCores) {
    check_transport(32, 8, 4);
}

TEST(CyclicTransportDramTest, SixtyFourCores) {
    check_transport(64, 8, 8);
}

// Skew is what the simulator's jitter and scheduling policies stand in for:
// cores that do not advance together. With the barrier in place it must
// change nothing.
TEST(CyclicTransportDramTest, SkewedCoresStillAgree) {
    check_transport(8, 4, 2, Options{.skew_iters = 20000});
}

// And the barrier must be what makes that true. Without it, one row's
// updates across timesteps have nothing ordering them, so a core that runs
// ahead reads a page its predecessor has not written yet and the row
// checksums come out short.
//
// This is the device counterpart of the simulator's faults=: a protocol test
// that cannot fail when the protocol is removed is not testing the protocol.
// A long read-modify-write window is also harmless with the barrier: it is
// what the gradient computation will occupy in the real kernel.
TEST(CyclicTransportDramTest, ALongUpdateWindowIsHarmlessWithTheBarrier) {
    check_transport(8, 4, 2, Options{.rmw_spin_iters = 20000});
}

// And the barrier must be what makes that true. Without it, two cores in
// different timesteps that share a row can be inside the window at the same
// moment, and one of the two updates is lost.
//
// This is the device counterpart of the simulator's faults=: a protocol test
// that cannot fail when the protocol is removed is not testing the protocol.
TEST(CyclicTransportDramTest, OneSlowCoreIsHarmlessWithTheBarrier) {
    check_transport(8, 4, 2, Options{.rmw_spin_iters = 400000, .rmw_spin_core = 1});
}

TEST(CyclicTransportDramTest, WithoutTheBarrierAWideWindowLosesUpdates) {
    if (!grid_fits(4, 2)) {
        GTEST_SKIP() << "needs a 4x2 region";
    }
    // One core holds a packet for a long time while the others run ahead and
    // update the same rows. Its stale write-back then loses their updates.
    const auto broken = run_transport(
        8, 4, 2, Options{.rmw_spin_iters = 400000, .rmw_spin_core = 1, .no_barrier = true});
    EXPECT_FALSE(transport_is_correct(8, broken))
        << "dropping the chip-wide barrier did not corrupt the transport, so the passing tests "
           "above are not evidence that the barrier does anything; widen the window or look "
           "for ordering coming from somewhere else";
}

// ---------------------------------------------------------------- the relay
// Algorithm 3: the packet stays in L1 across an active streak and is
// forwarded to the next consumer, with DRAM touched only at streak
// boundaries. Same checksums, and now the traffic counters have to match the
// paper's streak count as well.
TEST(CyclicTransportRelayTest, FourCores) {
    check_relay(4, 2, 2);
}

TEST(CyclicTransportRelayTest, EightCores) {
    check_relay(8, 4, 2);
}

TEST(CyclicTransportRelayTest, SixteenCores) {
    check_relay(16, 4, 4);
}

TEST(CyclicTransportRelayTest, ThirtyTwoCores) {
    check_relay(32, 8, 4);
}

TEST(CyclicTransportRelayTest, SixtyFourCores) {
    check_relay(64, 8, 8);
}

// Skew and a long update window are what the credits and readiness tags exist
// to survive.
TEST(CyclicTransportRelayTest, SkewedCoresStillAgree) {
    check_relay(8, 4, 2, Options{.skew_iters = 20000});
}

TEST(CyclicTransportRelayTest, OneSlowCoreStillAgrees) {
    check_relay(8, 4, 2, Options{.rmw_spin_iters = 400000, .rmw_spin_core = 1});
}

// ------------------------------------------------------- the relay's faults
// These three tests are negative results, and the reason they are negative
// is the most useful thing this step established.
//
// In the paper's step order a core forwards for destination t + 1 at the end
// of its timestep t, after the release of t, and then arrives at barrier
// t + 1. So by the time any core begins t + 1, every forward for t + 1 has
// completed and every slot it targeted was released at t - 1. The barrier
// therefore implies both the credit and the readiness tag, and removing
// either changes nothing here.
//
// That means the barrier variant does not exercise the slot protocol. What
// the relay tests above do establish is the transport: slot addressing, the
// DRAM boundaries of a streak, the self-transition, and the traffic counts.
// The credit and the tag become load-bearing in Algorithm 4, which has no
// barrier to imply them, and in the O1 variant of Algorithm 3, which issues
// the forward before the barrier wait. Both are next; these faults get their
// corrupting counterparts there.
//
// The simulator reaches the same conclusion for the credit: benign in
// Algorithm 3's paper order, load-bearing in Algorithm 4. Pinning the
// negative result matters -- if one of these starts failing, the step order
// has changed and the reasoning above no longer applies.

// A tag left over from the slot's use at t - 2 satisfies a wait for any
// positive value, so this is the "generic binary flag without a generation
// and ownership discipline" the contract rejects. Benign only because the
// barrier has already delivered the packet.
TEST(CyclicTransportRelayTest, UnderTheBarrierAStaleTagIsNotYetObservable) {
    check_relay(8, 4, 2, Options{.stale_tag_ok = true});
}

// The credit is implied: a receiver that has arrived at t finished its
// timestep t - 1 entirely, including releasing the slot this forward targets.
TEST(CyclicTransportRelayTest, UnderTheBarrierTheCreditIsImplied) {
    check_relay(8, 4, 2, Options{.no_credit_wait = true});
}

// Payload and readiness go to the same core on the same NoC, which tt-metal
// orders, so readiness cannot overtake the payload. On hardware the barrier
// still earns its place under a different clause: it keeps the 4-byte source
// word valid until the write has consumed it. Splitting payload and control
// across NoCs -- the O9 experiment -- is what would make ordering matter.
TEST(CyclicTransportRelayTest, ReadinessCannotOvertakePayloadOnOneNoc) {
    check_relay(8, 4, 2, Options{.readiness_before_payload = true});
}
