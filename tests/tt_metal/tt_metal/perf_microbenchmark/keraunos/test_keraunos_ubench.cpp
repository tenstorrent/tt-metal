// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Keraunos CCE microbenchmarks: latency and bandwidth on one mmk package, set up like the KeraunosEmu and
// KeraunosEthSend cases (a unit MeshDevice, IDLE_ETH kernels on CCE0 of the HSIO tiles through EthernetConfig).
// Kernels and the shared layout are in tests/tt_metal/tt_metal/test_kernels/misc/keraunos_ubench/.
//
//   KeraunosUbench.TimerCheck                 rdcycle sanity: monotonic, bracket cost, add chain, load chains
//   KeraunosUbenchLatency.PingPong/*          ping-pong RTT/TURN: hart<->hart on one CCE, CCE0<->CCE0 across HSIO
//                                             tiles, CPU store or iDMA
//   KeraunosUbenchTransportBw.Run/*           iDMA (metal path, lean path) and CPU stores, 1 or 8 harts, to own SRAM
//                                             or another tile; INJECTION and DRAINED
//   KeraunosUbenchEthLatency.PingPong/*       ping-pong over the Ethernet loopback, one case per payload
//   KeraunosUbenchEthBandwidth.Run/*          streamed sends over the loopback, size x TX queues;
//                                             INJECTION/ACKED/DELIVERED
//
// Every number is a cycle count of the measuring hart's rdcycle (kernels never subtract another hart's timestamps).
// Results are emulator RTL cycles: a ratio between clock domains on the emulator (CCE, NOC, D2D, Ethernet) need not
// match silicon, so compare paths with each other, not with silicon figures.
//
// Run (after bringup; the cross-tile cases need the HSIO fabric firewalls open). The KeraunosUbenchEth* suites need
// the loopback HSIO0.eth0 <-> HSIO1.eth0 up in packet mode, as for KeraunosEthSend; without it their kernels hang
// until the host timeout, so leave them out of the filter unless bring-up started the loopback:
//   export TT_METAL_EMU_SERVER=<host>:<port>
//   export TT_METAL_EMU_SOC_DESC=$TT_METAL_HOME/tt_metal/third_party/umd/tests/soc_descs/mmk.yaml
//   export TT_METAL_SLOW_DISPATCH_MODE=1
//   B=./build/test/tt_metal/perf_microbenchmark/keraunos/test_keraunos_ubench
//   $B --gtest_filter='-*KeraunosUbenchEth*'   # on-chip
//   $B --gtest_filter='*KeraunosUbenchEth*'    # Ethernet
//
// A host timeout means a kernel is stuck in an unbounded wait: the test prints the result blocks and exits the
// process, and the emulator job needs a reset before the next run.

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <span>
#include <tuple>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "impl/dispatch/slow_dispatch.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/hal.hpp"
#include "dev_mem_map.h"
#include "env_lib.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tests/tt_metal/tt_metal/context/keraunos_session.hpp"
#include "tests/tt_metal/tt_metal/context/mimir_contract.hpp"
#include "tt_metal/tt_metal/test_kernels/misc/keraunos_ubench/keraunos_ubench.h"

namespace tt::tt_metal::keraunos_ubench_test {
namespace {

using namespace ::tt::tt_metal::mimir_contract;
using namespace ::tt::tt_metal::keraunos_session;
using namespace ::keraunos_ubench;

constexpr const char* kKernelDir = "tests/tt_metal/tt_metal/test_kernels/misc/keraunos_ubench/";

std::string kernel_path(const char* name) { return std::string(kKernelDir) + name; }

// ---------------------------------------------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------------------------------------------

constexpr uint32_t kPolls = 20'000'000;   // poll bound per device wait
constexpr uint32_t kHostTimeoutS = 1800;  // per launch
constexpr uint32_t kTimerAdds = 1000;
constexpr uint32_t kTimerLoads = 256;
constexpr uint32_t kLatencyWarmup = 4;
constexpr uint32_t kLatencySamples = 64;
constexpr uint32_t kTransfers = 64;  // per bandwidth case, after the warm-up
constexpr uint32_t kBwWarmup = 4;
// Measuring core and cross-tile peer: CCE0 of these HSIO tiles. The Ethernet loopback joins the same two.
constexpr uint32_t kOwnHsio = 0;
constexpr uint32_t kPeerHsio = 1;

// Sweep points. The kernels trust these bounds.
constexpr std::array<uint32_t, 2> kBwHarts{1, 8};
constexpr std::array<uint32_t, 3> kBwSizes{256, 4096, 65536};
// IdmaLean differs from IdmaMetal only in per-transfer command-buffer programming, visible at small sizes.
constexpr std::array<uint32_t, 2> kLeanSizes{256, 4096};
constexpr std::array<uint32_t, 4> kEthLatencyPayloads{0, 64, 1024, 4096};  // 0 = the 8-byte token alone
constexpr std::array<uint32_t, 2> kEthTxqs{1, 4};
static_assert(kLatencyWarmup >= 1 && kLatencySamples >= 1 && kLatencySamples <= kMaxSamples);
static_assert(std::ranges::all_of(kBwHarts, [](uint32_t h) { return h >= 1 && h <= kMaxHarts; }));
static_assert(std::ranges::all_of(kBwSizes, [](uint32_t b) { return b > 0 && b % 64 == 0 && b <= kMaxTransferBytes; }));
static_assert(std::ranges::all_of(kLeanSizes, [](uint32_t b) {
    return b > 0 && b % 64 == 0 && b <= kMaxTransferBytes;
}));
static_assert(std::ranges::all_of(kEthLatencyPayloads, [](uint32_t b) {
    return b % 16 == 0 && b <= kMaxTransferBytes;
}));
static_assert(std::ranges::all_of(kEthTxqs, [](uint32_t q) { return q >= 1 && q <= 4; }));

std::pair<uint32_t, uint32_t> split(uint64_t v) { return {static_cast<uint32_t>(v), static_cast<uint32_t>(v >> 32)}; }

// ---------------------------------------------------------------------------------------------------------------
// Statistics and formatting
// ---------------------------------------------------------------------------------------------------------------

// Floor-index percentile of a sorted sample set: element floor(p * (n - 1) / 100).
uint64_t percentile(const std::vector<uint32_t>& sorted, uint32_t p) {
    return sorted.empty() ? 0 : sorted[(static_cast<uint64_t>(p) * (sorted.size() - 1)) / 100];
}

std::string cycles_str(double cycles) { return fmt::format("{:.0f} cyc", cycles); }

std::string rate_str(double bytes, double cycles) {
    return cycles == 0 ? std::string("n/a") : fmt::format("{:.3f} B/cyc", bytes / cycles);
}

const char* status_name(uint32_t s) {
    switch (s) {
        case kStatusDone: return "done";
        case kStatusRunning: return "running";
        case kStatusTimeout: return "timeout";
        case 0: return "not-started";
        default: return "unknown";
    }
}

std::string format_result(const Result& r) {
    return fmt::format(
        "status {} (0x{:08x}) detail {} iterations {} cycles [{}] aux [{}]",
        status_name(r.status),
        r.status,
        r.detail,
        r.iterations_done,
        fmt::join(r.cycles, ","),
        fmt::join(r.aux, ","));
}

// ---------------------------------------------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------------------------------------------

// One hart of a CCE core.
struct Watched {
    KeraunosCore core;
    uint32_t hart = 0;
};

// The shared Session plus this benchmark's layout: result blocks, the control region, and the three addresses of a
// flat SRAM offset.
class UbenchSession : public Session {
public:
    UbenchSession() : Session(kRegionBytes) {}

    // The iDMA address a copy issued on @p from uses for @p off on @p to: the chiplet-local view on the same tile, the
    // system address on another tile (as metal's CCE NOC routes, quasar_cce_noc_routes()).
    uint64_t idma_addr(const KeraunosCore& from, const KeraunosCore& to, uint32_t off) const {
        return from.hsio == to.hsio ? hsio_local_addr(to.hsio, tile_stride(), off) : global_addr(to, off);
    }
    uint64_t global_addr(const KeraunosCore& core, uint32_t off) const {
        return hsio_global_addr(core.hsio, tile_stride(), off);
    }
    // The CPU address a kernel on @p from uses for @p off on @p to: the uncached alias on the same core, the global
    // address on another tile.
    uint64_t cpu_addr(const KeraunosCore& from, const KeraunosCore& to, uint32_t off) const {
        return from.hsio == to.hsio ? static_cast<uint64_t>(MEM_IERISC_UNCACHED_BASE) + off : global_addr(to, off);
    }

    Result result(const KeraunosCore& core, uint32_t hart) {
        return read_struct<Result>(core, base() + result_offset(hart));
    }

    // Zeroes the result blocks, gate and mailboxes of @p core.
    void clear_control(const KeraunosCore& core) {
        write(core, base(), std::vector<uint32_t>(kSamplesOffset / sizeof(uint32_t), 0));
    }

    // Session::run over the watched harts' status words; a hang prints every watched result block.
    void run(Program& program, const std::vector<Watched>& watched) {
        std::vector<StatusWord> words;
        for (const Watched& w : watched) {
            words.push_back({w.core, base() + result_offset(w.hart) + static_cast<uint32_t>(offsetof(Result, status))});
        }
        Session::run(program, words, kStatusRunning, std::chrono::seconds(kHostTimeoutS), [&] {
            for (const Watched& d : watched) {
                fmt::print(stderr, "  hsio {} {}\n", d.core.hsio, format_result(result(d.core, d.hart)));
            }
        });
    }
};

KernelHandle create(Program& program, const char* name, const KeraunosCore& core, uint32_t hart) {
    return CreateKernel(
        program,
        kernel_path(name),
        core.logical,
        EthernetConfig{
            .eth_mode = Eth::IDLE, .noc = NOC::NOC_0, .processor = static_cast<DataMovementProcessor>(hart)});
}

std::vector<uint32_t> pattern(uint32_t hart, uint32_t bytes) {
    std::vector<uint32_t> words(bytes / sizeof(uint32_t));
    for (uint32_t j = 0; j < words.size(); j++) {
        words[j] = pattern_word(hart, j);
    }
    return words;
}

uint32_t count_mismatches(const std::vector<uint32_t>& got, const std::vector<uint32_t>& want) {
    uint32_t bad = 0;
    for (size_t j = 0; j < want.size(); j++) {
        bad += got.at(j) != want[j];
    }
    return bad;
}

// ---------------------------------------------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------------------------------------------

class KeraunosUbench : public ::testing::Test {
public:
    // The device stays open across every suite of this file (an open costs about 90 s on the emulator); the global
    // environment below closes it once, after the last test.
    static void close_session() { session_.reset(); }

protected:
    void SetUp() override {
        skip_unless_ethernet_package();
        if (!session_) {
            session_ = std::make_unique<UbenchSession>();
        }
    }

    UbenchSession& s() { return *session_; }

    KeraunosCore require_core(uint32_t hsio) {
        const auto c = s().core(hsio);
        if (!c) {
            throw std::runtime_error(fmt::format("descriptor does not list CCE0 of HSIO tile {}", hsio));
        }
        return *c;
    }

    static inline std::unique_ptr<UbenchSession> session_;
};

class SessionEnvironment : public ::testing::Environment {
public:
    void TearDown() override { KeraunosUbench::close_session(); }
};
[[maybe_unused]] ::testing::Environment* const kSessionEnvironment =
    ::testing::AddGlobalTestEnvironment(new SessionEnvironment);

// ---------------------------------------------------------------------------------------------------------------
// Timer check
// ---------------------------------------------------------------------------------------------------------------

TEST_F(KeraunosUbench, TimerCheck) {
    const KeraunosCore own = require_core(kOwnHsio);
    const auto peer = s().core(kPeerHsio);
    const uint32_t base = s().base();
    const uint32_t adds = kTimerAdds;
    const uint32_t loads = kTimerLoads;
    const uint64_t remote = peer ? s().global_addr(*peer, base + staging_offset(1)) : 0;

    s().clear_control(own);
    Program program = CreateProgram();
    const KernelHandle k = create(program, "timer_check.cpp", own, 0);
    const auto [rlo, rhi] = split(remote);
    SetRuntimeArgs(program, k, own.logical, {base, 0, adds, loads, rlo, rhi});
    s().run(program, {{own, 0}});

    const Result r = s().result(own, 0);
    namespace tc = timer_check;
    log_info(tt::LogTest, "TimerCheck: {}", format_result(r));
    ASSERT_EQ(r.status, kStatusDone) << format_result(r);
    log_info(
        tt::LogTest,
        "TimerCheck: bracket {}..{} cyc | {:.2f} cyc/add | {:.1f} cyc/local uncached load | {} | backwards {} of {} "
        "reads, {} zero deltas",
        r.cycles[tc::kEmptyMin],
        r.cycles[tc::kEmptyMax],
        static_cast<double>(r.cycles[tc::kAddLoop]) / adds,
        static_cast<double>(r.cycles[tc::kLocalLoadChase]) / loads,
        peer ? fmt::format(
                   "{:.1f} cyc/remote load (hsio {})",
                   static_cast<double>(r.cycles[tc::kRemoteLoadChase]) / loads,
                   peer->hsio)
             : std::string("remote load skipped"),
        r.aux[tc::kMonotonicBackwards],
        tc::kMonotonicReadCount,
        r.aux[tc::kMonotonicZeroDelta]);

    // A counter that goes backwards cannot bracket anything.
    EXPECT_EQ(r.aux[tc::kMonotonicBackwards], 0u);
    // A counter that does not advance across adds is not counting cycles.
    EXPECT_GT(r.cycles[tc::kAddLoop], r.cycles[tc::kEmptyMax]);
}

// ---------------------------------------------------------------------------------------------------------------
// Latency
// ---------------------------------------------------------------------------------------------------------------

struct LatencyRun {
    std::string name;
    LatencyTransport transport;
    uint32_t payload_bytes = 0;
};

// Runs one ping-pong between @p a (initiator) and @p b (responder) and logs it.
void run_latency(UbenchSession& s, const Watched& a, const Watched& b, const LatencyRun& run) {
    const uint32_t base = s.base();

    s.clear_control(a.core);
    if (b.core.hsio != a.core.hsio) {
        s.clear_control(b.core);
    }
    if (run.transport == LatencyTransport::Eth) {
        s.write_eth_link(a.core, b.core);
        s.write_eth_link(b.core, a.core);
        if (run.payload_bytes != 0) {
            // The initiator sends its pattern; the responder starts from zeros (and echoes zeros back), so the
            // check below sees the initiator's payload only if it arrived.
            s.write(a.core, base + src_offset(a.hart), pattern(a.hart, run.payload_bytes));
            s.write(b.core, base + src_offset(b.hart), std::vector<uint32_t>(run.payload_bytes / 4, 0));
        }
    }

    Program program = CreateProgram();
    auto add = [&](const Watched& me, const Watched& peer, LatencyRole role) {
        const KernelHandle k = create(program, "latency.cpp", me.core, me.hart);
        const uint32_t peer_mailbox = base + mailbox_offset(peer.hart);
        const auto [cpu_lo, cpu_hi] = split(s.cpu_addr(me.core, peer.core, peer_mailbox));
        const auto [dma_lo, dma_hi] = split(s.idma_addr(me.core, peer.core, peer_mailbox));
        const auto [stg_lo, stg_hi] = split(s.idma_addr(me.core, me.core, base + staging_offset(me.hart)));
        SetRuntimeArgs(
            program,
            k,
            me.core.logical,
            {base,
             me.hart,
             static_cast<uint32_t>(role),
             static_cast<uint32_t>(run.transport),
             kLatencyWarmup,
             kLatencySamples,
             kPolls,
             cpu_lo,
             cpu_hi,
             dma_lo,
             dma_hi,
             peer_mailbox,
             stg_lo,
             stg_hi,
             run.payload_bytes});
    };
    add(a, b, LatencyRole::Initiator);
    add(b, a, LatencyRole::Responder);
    s.run(program, {{a.core, a.hart}, {b.core, b.hart}});

    const Result ra = s.result(a.core, a.hart);
    const Result rb = s.result(b.core, b.hart);
    const std::string trace = "initiator " + format_result(ra) + "\nresponder " + format_result(rb);
    ASSERT_EQ(ra.status, kStatusDone) << run.name << "\n" << trace;
    ASSERT_EQ(rb.status, kStatusDone) << run.name << "\n" << trace;

    auto rtt_sorted = s.read(a.core, base + samples_offset(a.hart), kLatencySamples);
    auto turn_sorted = s.read(b.core, base + samples_offset(b.hart), kLatencySamples);
    std::sort(rtt_sorted.begin(), rtt_sorted.end());
    std::sort(turn_sorted.begin(), turn_sorted.end());

    log_info(
        tt::LogTest,
        "Latency {:<28} RTT p50 {} p90 {} p99 {} | TURN p50 {} p90 {} p99 {}",
        run.name,
        cycles_str(percentile(rtt_sorted, 50)),
        cycles_str(percentile(rtt_sorted, 90)),
        cycles_str(percentile(rtt_sorted, 99)),
        cycles_str(percentile(turn_sorted, 50)),
        cycles_str(percentile(turn_sorted, 90)),
        cycles_str(percentile(turn_sorted, 99)));

    if (run.transport == LatencyTransport::Eth && run.payload_bytes != 0) {
        // The initiator's payload landed at the same offset on the responder.
        EXPECT_EQ(
            count_mismatches(
                s.read(b.core, base + src_offset(b.hart), run.payload_bytes / 4), pattern(a.hart, run.payload_bytes)),
            0u)
            << run.name << ": responder data region is not the initiator's pattern";
    }
}

struct OnChipLatencyCase {
    const char* name;
    bool same_core;
    LatencyTransport transport;
};

class KeraunosUbenchLatency : public KeraunosUbench, public ::testing::WithParamInterface<OnChipLatencyCase> {};

TEST_P(KeraunosUbenchLatency, PingPong) {
    const OnChipLatencyCase& c = GetParam();
    const KeraunosCore own = require_core(kOwnHsio);
    Watched a{own, 0};
    Watched b{own, 1};
    if (!c.same_core) {
        const auto peer = s().core(kPeerHsio);
        if (!peer) {
            GTEST_SKIP() << "Descriptor does not list CCE0 of HSIO tile " << kPeerHsio;
        }
        b = {*peer, 0};
    }
    run_latency(s(), a, b, {fmt::format("{}_h{}_h{}", c.name, a.core.hsio, b.core.hsio), c.transport});
}

INSTANTIATE_TEST_SUITE_P(
    KeraunosUbench,
    KeraunosUbenchLatency,
    ::testing::Values(
        OnChipLatencyCase{"SameCoreCpuStore", true, LatencyTransport::CpuStore},
        OnChipLatencyCase{"SameCoreIdma", true, LatencyTransport::Idma},
        OnChipLatencyCase{"CrossTileCpuStore", false, LatencyTransport::CpuStore},
        OnChipLatencyCase{"CrossTileIdma", false, LatencyTransport::Idma}),
    [](const ::testing::TestParamInfo<OnChipLatencyCase>& info) { return std::string(info.param.name); });

// The loopback's ends (KeraunosEthSend): HSIO0.eth0 <-> HSIO1.eth0, on TX queue 0.
class KeraunosUbenchEthLatency : public KeraunosUbench, public ::testing::WithParamInterface<uint32_t> {};

TEST_P(KeraunosUbenchEthLatency, PingPong) {
    const uint32_t payload = GetParam();
    const KeraunosCore tx = require_core(kOwnHsio);
    const KeraunosCore rx = require_core(kPeerHsio);
    run_latency(s(), {tx, 0}, {rx, 0}, {fmt::format("Eth_p{}", payload), LatencyTransport::Eth, payload});
}

INSTANTIATE_TEST_SUITE_P(
    KeraunosUbench,
    KeraunosUbenchEthLatency,
    ::testing::ValuesIn(kEthLatencyPayloads),
    [](const ::testing::TestParamInfo<uint32_t>& info) { return fmt::format("{}B", info.param); });

// ---------------------------------------------------------------------------------------------------------------
// Transport bandwidth
// ---------------------------------------------------------------------------------------------------------------

const char* mech_name(TransportMech mech) {
    switch (mech) {
        case TransportMech::IdmaMetal: return "IdmaMetal";
        case TransportMech::IdmaLean: return "IdmaLean";
        case TransportMech::CpuStore64: return "CpuStore64";
    }
    TT_THROW("Unknown TransportMech {}", static_cast<uint32_t>(mech));
}

// Mechanism, destination on the own core (true) or the peer tile (false), harts, bytes per transfer.
using TransportBwCase = std::tuple<TransportMech, bool, uint32_t, uint32_t>;

std::string transport_bw_name(const TransportBwCase& c) {
    const auto& [mech, local, harts, size] = c;
    return fmt::format("{}_{}_{}Harts_{}B", mech_name(mech), local ? "Local" : "Tile", harts, size);
}

std::vector<TransportBwCase> transport_bw_cases() {
    std::vector<TransportBwCase> cases;
    for (const TransportMech mech : {TransportMech::IdmaMetal, TransportMech::IdmaLean, TransportMech::CpuStore64}) {
        for (const bool local : {true, false}) {
            for (const uint32_t harts : kBwHarts) {
                for (const uint32_t size : mech == TransportMech::IdmaLean ? std::span<const uint32_t>(kLeanSizes)
                                                                           : std::span<const uint32_t>(kBwSizes)) {
                    cases.emplace_back(mech, local, harts, size);
                }
            }
        }
    }
    return cases;
}

class KeraunosUbenchTransportBw : public KeraunosUbench, public ::testing::WithParamInterface<TransportBwCase> {};

TEST_P(KeraunosUbenchTransportBw, Run) {
    const auto [mech, local, harts, size] = GetParam();
    const bool cpu = mech == TransportMech::CpuStore64;
    const std::string name = transport_bw_name(GetParam());
    const KeraunosCore own = require_core(kOwnHsio);
    std::optional<KeraunosCore> peer;
    if (!local) {
        peer = s().core(kPeerHsio);
        if (!peer) {
            GTEST_SKIP() << "Descriptor does not list CCE0 of HSIO tile " << kPeerHsio;
        }
    }
    const KeraunosCore dst_core = local ? own : *peer;
    const uint32_t base = s().base();

    s().clear_control(own);
    Program program = CreateProgram();
    std::vector<Watched> watched;
    for (uint32_t h = 0; h < harts; h++) {
        // Destination slice: on a local copy kDstOffset above the source; on another tile the same slice there.
        const uint32_t dst_off = base + local_dst_offset(h);
        if (!cpu) {  // CPU stores build the pattern in registers
            s().write(own, base + src_offset(h), pattern(h, size));
        }
        s().write(dst_core, dst_off, std::vector<uint32_t>(size / 4, 0));
        const KernelHandle k = create(program, "transport_bw.cpp", own, h);
        const auto [src_lo, src_hi] = split(s().idma_addr(own, own, base + src_offset(h)));
        const auto [dst_lo, dst_hi] = split(s().idma_addr(own, dst_core, dst_off));
        const auto [cpu_lo, cpu_hi] = split(s().cpu_addr(own, dst_core, dst_off));
        SetRuntimeArgs(
            program,
            k,
            own.logical,
            {base,
             h,
             harts,
             static_cast<uint32_t>(mech),
             size,
             kTransfers,
             kBwWarmup,
             kPolls,
             src_lo,
             src_hi,
             dst_lo,
             dst_hi,
             cpu_lo,
             cpu_hi});
        watched.push_back({own, h});
    }
    s().run(program, watched);

    std::vector<uint64_t> inj, drained;
    uint32_t data_errors = 0;
    for (uint32_t h = 0; h < harts; h++) {
        const Result r = s().result(own, h);
        ASSERT_EQ(r.status, kStatusDone) << name << ": " << format_result(r);
        inj.push_back(r.cycles[transport::kInjection]);
        drained.push_back(r.cycles[transport::kDrained]);
        data_errors += count_mismatches(s().read(dst_core, base + local_dst_offset(h), size / 4), pattern(h, size));
    }
    const double bytes_per_hart = static_cast<double>(kTransfers) * size;
    const uint64_t span_inj = *std::max_element(inj.begin(), inj.end());
    const uint64_t span_drained = *std::max_element(drained.begin(), drained.end());
    const double run_bytes = bytes_per_hart * harts;
    if (cpu) {
        // INJECTION is the stores' retirement; there is no separate drain.
        log_info(
            tt::LogTest,
            "TransportBw {:<34} per hart (hart 0) RETIRED {} | run ({} B over slowest hart) RETIRED {}",
            name,
            rate_str(bytes_per_hart, inj[0]),
            static_cast<uint64_t>(run_bytes),
            rate_str(run_bytes, span_inj));
    } else {
        log_info(
            tt::LogTest,
            "TransportBw {:<34} per hart (hart 0) INJ {} DRAINED {} | run ({} B over slowest hart) INJ {} DRAINED {}",
            name,
            rate_str(bytes_per_hart, inj[0]),
            rate_str(bytes_per_hart, drained[0]),
            static_cast<uint64_t>(run_bytes),
            rate_str(run_bytes, span_inj),
            rate_str(run_bytes, span_drained));
    }
    EXPECT_EQ(data_errors, 0u) << name << ": destination is not the source pattern";
}

INSTANTIATE_TEST_SUITE_P(
    KeraunosUbench,
    KeraunosUbenchTransportBw,
    ::testing::ValuesIn(transport_bw_cases()),
    [](const ::testing::TestParamInfo<TransportBwCase>& info) { return transport_bw_name(info.param); });

// ---------------------------------------------------------------------------------------------------------------
// Ethernet bandwidth
// ---------------------------------------------------------------------------------------------------------------

// Bytes per send, TX queues.
using EthBwCase = std::tuple<uint32_t, uint32_t>;

class KeraunosUbenchEthBandwidth : public KeraunosUbench, public ::testing::WithParamInterface<EthBwCase> {};

TEST_P(KeraunosUbenchEthBandwidth, Run) {
    const auto [size, txqs] = GetParam();
    const std::string name = fmt::format("Eth_S{}_Q{}", size, txqs);
    const KeraunosCore tx = require_core(kOwnHsio);
    const KeraunosCore rx = require_core(kPeerHsio);
    const uint32_t base = s().base();
    const auto [ack_lo, ack_hi] = split(s().global_addr(tx, base + kEthAckOffset));

    s().clear_control(tx);
    s().clear_control(rx);
    s().write_eth_link(tx, rx);
    s().write_eth_link(rx, tx);
    s().write(tx, base + src_offset(0), pattern(0, size));
    s().write(rx, base + src_offset(0), std::vector<uint32_t>(size / 4, 0));

    Program program = CreateProgram();
    const KernelHandle ks = create(program, "eth_bw_sender.cpp", tx, 0);
    const KernelHandle kr = create(program, "eth_bw_receiver.cpp", rx, 0);
    SetRuntimeArgs(program, ks, tx.logical, {base, size, kTransfers, kBwWarmup, txqs, kPolls});
    SetRuntimeArgs(program, kr, rx.logical, {base, kPolls, ack_lo, ack_hi});
    s().run(program, {{tx, 0}, {rx, 0}});

    const Result rs = s().result(tx, 0);
    const Result rr = s().result(rx, 0);
    const std::string trace = "sender " + format_result(rs) + "\nreceiver " + format_result(rr);
    ASSERT_EQ(rs.status, kStatusDone) << name << "\n" << trace;
    ASSERT_EQ(rr.status, kStatusDone) << name << "\n" << trace;
    const uint32_t data_errors = count_mismatches(s().read(rx, base + src_offset(0), size / 4), pattern(0, size));

    const double bytes = static_cast<double>(kTransfers) * size;
    log_info(
        tt::LogTest,
        "EthBw {:<16} INJ {} ACKED {} DELIVERED {} | idle RTT {} | busy-queue polls {}",
        name,
        rate_str(bytes, rs.cycles[eth_bw::kInjection]),
        rate_str(bytes, rs.cycles[eth_bw::kAcked]),
        rate_str(bytes, rs.cycles[eth_bw::kDelivered]),
        cycles_str(rs.cycles[eth_bw::kHandshakeRtt]),
        rs.aux[eth_bw::kTxqBusySpins]);
    EXPECT_EQ(data_errors, 0u) << name << ": receiver data region is not the sender's pattern";
}

INSTANTIATE_TEST_SUITE_P(
    KeraunosUbench,
    KeraunosUbenchEthBandwidth,
    ::testing::Combine(::testing::ValuesIn(kBwSizes), ::testing::ValuesIn(kEthTxqs)),
    [](const ::testing::TestParamInfo<EthBwCase>& info) {
        return fmt::format("{}B_{}Txq", std::get<0>(info.param), std::get<1>(info.param));
    });

}  // namespace
}  // namespace tt::tt_metal::keraunos_ubench_test
