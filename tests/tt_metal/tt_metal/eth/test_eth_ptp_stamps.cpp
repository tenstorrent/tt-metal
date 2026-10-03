// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The Blackhole Ethernet 1588 layer on real links. On each active eth link between two chips, an initiator and an echo
// kernel restart their PTP timers and exchange 256 frames with one-step egress stamps and an RX stamp rule; the
// initiator then sends 16 two-step stamped frames. It checks:
//   - each timer's restart offset against its clocks;
//   - every frame was stamped on egress, and the RX FIFO held exactly its one ingress stamp;
//   - the stamps are causal round by round;
//   - each round's one-way time is within 40 ns of the median, and each offset within 40 ns of a fitted line;
//   - every two-step frame's TX FIFO entry carried its tag and a time inside its send;
//   - both kernels restore the TX header row and RX settings they changed.
// Needs slow dispatch (TT_METAL_SLOW_DISPATCH_MODE=1) on a multi-chip Blackhole system.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>

#include "device_fixture.hpp"
#include "eth_test_common.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_ptp_stamps.hpp"

namespace tt::tt_metal {
namespace {

using eth_ptp_stamps::Result;

constexpr const char* kKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_ptp_stamps.cpp";
constexpr double kStampTickNs = 20.0;

struct LinkFit {
    std::vector<double> one_way;
    double one_way_p10 = 0, one_way_p50 = 0, one_way_p90 = 0, one_way_rms_ns = 0;
    double slope = 0;
    double residual_max_ns = 0, residual_rms_ns = 0;
};

// Round i: t0 initiator egress, t1 echo ingress, t1b echo egress, t2 initiator ingress, each end in its own PTP ns.
void expect_causal(const Result& initiator, const Result& echo, const std::string& link) {
    for (uint32_t i = 0; i < eth_ptp_stamps::kRounds; i++) {
        const uint64_t t0 = echo.stamps[i].peer_egress, t1 = echo.stamps[i].ingress;
        const uint64_t t1b = initiator.stamps[i].peer_egress, t2 = initiator.stamps[i].ingress;
        EXPECT_LT(t0, t2) << link << " round " << i;
        EXPECT_LT(t1, t1b) << link << " round " << i;
        if (i > 0) {
            EXPECT_GT(t0, initiator.stamps[i - 1].ingress) << link << " round " << i;
            EXPECT_GT(t1, initiator.stamps[i - 1].peer_egress) << link << " round " << i;
        }
    }
}

LinkFit fit(const Result& initiator, const Result& echo) {
    struct Round {
        double at, offset;
    };
    LinkFit link_fit;
    std::vector<Round> rounds;
    for (uint32_t i = 0; i < eth_ptp_stamps::kRounds; i++) {
        const auto t0 = static_cast<double>(echo.stamps[i].peer_egress);
        const auto t1 = static_cast<double>(echo.stamps[i].ingress);
        const auto t1b = static_cast<double>(initiator.stamps[i].peer_egress);
        const auto t2 = static_cast<double>(initiator.stamps[i].ingress);
        link_fit.one_way.push_back(0.5 * ((t2 - t0) - (t1b - t1)));
        rounds.push_back({.at = 0.5 * (t0 + t2), .offset = 0.5 * ((t1 + t1b) - (t0 + t2))});
    }
    double at_sum = 0, offset_sum = 0;
    for (const Round& round : rounds) {
        at_sum += round.at;
        offset_sum += round.offset;
    }
    const double at_mean = at_sum / static_cast<double>(rounds.size());
    const double offset_mean = offset_sum / static_cast<double>(rounds.size());
    double at_sq = 0, at_offset = 0;
    for (const Round& round : rounds) {
        at_sq += (round.at - at_mean) * (round.at - at_mean);
        at_offset += (round.at - at_mean) * (round.offset - offset_mean);
    }
    link_fit.slope = at_offset / at_sq;
    double residual_sq = 0;
    for (const Round& round : rounds) {
        const double residual = round.offset - (offset_mean + link_fit.slope * (round.at - at_mean));
        link_fit.residual_max_ns = std::max(link_fit.residual_max_ns, std::abs(residual));
        residual_sq += residual * residual;
    }
    link_fit.residual_rms_ns = std::sqrt(residual_sq / static_cast<double>(rounds.size()));
    std::vector<double>& one_way = link_fit.one_way;
    std::ranges::sort(one_way);
    const auto percentile = [&](size_t percent) {
        return one_way[std::min(one_way.size() - 1, one_way.size() * percent / 100)];
    };
    link_fit.one_way_p10 = percentile(10);
    link_fit.one_way_p50 = percentile(50);
    link_fit.one_way_p90 = percentile(90);
    double one_way_sq = 0;
    for (const double round_one_way : one_way) {
        one_way_sq += (round_one_way - link_fit.one_way_p50) * (round_one_way - link_fit.one_way_p50);
    }
    link_fit.one_way_rms_ns = std::sqrt(one_way_sq / static_cast<double>(one_way.size()));
    return link_fit;
}

void run_link(
    MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_a,
    const std::shared_ptr<distributed::MeshDevice>& mesh_b,
    const CoreCoord& eth_a,
    const CoreCoord& eth_b) {
    IDevice* dev_a = mesh_a->get_devices()[0];
    IDevice* dev_b = mesh_b->get_devices()[0];
    const uint32_t base =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t result_addr = base + eth_ptp_stamps::kResultOffset;
    std::vector<uint32_t> zero(sizeof(Result) / sizeof(uint32_t), 0);
    detail::WriteToDeviceL1(dev_a, eth_a, result_addr, zero, CoreType::ETH);
    detail::WriteToDeviceL1(dev_b, eth_b, result_addr, zero, CoreType::ETH);

    const auto make_workload = [&](const CoreCoord& core, bool initiator) {
        Program program;
        EthernetConfig config{.compile_args = {initiator ? 1u : 0u}};
        eth_test_common::set_arch_specific_eth_config(config);
        const KernelHandle kernel = CreateKernel(program, kKernel, core, config);
        SetRuntimeArgs(program, kernel, core, {base});
        distributed::MeshWorkload workload;
        const distributed::MeshCoordinate zero_coord(0, 0);
        workload.add_program(distributed::MeshCoordinateRange(zero_coord, zero_coord), std::move(program));
        return workload;
    };
    distributed::MeshWorkload initiator_workload = make_workload(eth_a, true);
    distributed::MeshWorkload echo_workload = make_workload(eth_b, false);
    std::thread initiator_thread([&] { fixture->RunProgram(mesh_a, initiator_workload); });
    std::thread echo_thread([&] { fixture->RunProgram(mesh_b, echo_workload); });
    initiator_thread.join();
    echo_thread.join();

    const std::string link =
        fmt::format("chip {} eth {} -> chip {} eth {}", dev_a->id(), eth_a.str(), dev_b->id(), eth_b.str());
    const auto read = [&](IDevice* dev, const CoreCoord& core) {
        Result result{};
        detail::ReadFromDeviceL1(
            dev, core, result_addr, std::span(reinterpret_cast<uint8_t*>(&result), sizeof(result)), CoreType::ETH);
        return result;
    };
    const Result initiator = read(dev_a, eth_a);
    const Result echo = read(dev_b, eth_b);

    for (const auto& [result, end] : {std::pair{&initiator, "initiator"}, std::pair{&echo, "echo"}}) {
        ASSERT_EQ(result->done, eth_ptp_stamps::kDone) << link << ": the " << end << " kernel did not finish";
        EXPECT_EQ(result->header_select_after, result->header_select_before)
            << link << ": the " << end << "'s TX header row selection was not restored";
        EXPECT_EQ(result->no_match_after, result->no_match_before)
            << link << ": the " << end << "'s RX no-match actions were not restored";
        // Both clocks are read just after a refclk update, so they agree to the ns when the offset is right.
        EXPECT_EQ(result->restart_error_ns, 0) << link << ": the " << end << "'s PTP time is "
                                               << result->restart_error_ns << " ns off what the restart's offset says";
        ASSERT_EQ(result->rounds, eth_ptp_stamps::kRounds)
            << link << ": the " << end << " stopped waiting for its peer";
        EXPECT_EQ(result->unstamped, 0u) << link << ": frames to the " << end << " without an egress stamp";
        EXPECT_EQ(result->ingress_mismatched, 0u)
            << link << ": " << end << " frames whose ingress stamp wasn't the RX FIFO's one entry";
    }
    EXPECT_EQ(initiator.two_step_good, eth_ptp_stamps::kTwoStepFrames)
        << link << ": two-step frames without a matching, in-window TX FIFO entry";

    expect_causal(initiator, echo, link);
    const LinkFit link_fit = fit(initiator, echo);
    log_info(
        tt::LogTest,
        "{}: one way inside the stamps {:.1f} ns (p10-p90 {:.1f}, rms {:.2f}), "
        "offset residual {:.2f} ns rms {:.2f} max, rate {:+.3f} ppm, restart error {} / {} ns",
        link,
        link_fit.one_way_p50,
        link_fit.one_way_p90 - link_fit.one_way_p10,
        link_fit.one_way_rms_ns,
        link_fit.residual_rms_ns,
        link_fit.residual_max_ns,
        link_fit.slope * 1e6,
        initiator.restart_error_ns,
        echo.restart_error_ns);
    EXPECT_GT(link_fit.one_way_p50, 0.0) << link;
    // Each stamp is its true time rounded down to the 20 ns grid, so a correctly paired round is under a tick from the
    // truth and none strays two ticks from the median or the line; a missing or mispaired stamp is off by far more. The
    // stamps' precision is gated end to end by the sync gate's link precision check, with the link's send dither.
    for (const double one_way : link_fit.one_way) {
        EXPECT_LT(std::abs(one_way - link_fit.one_way_p50), 2 * kStampTickNs) << link;
    }
    EXPECT_LT(link_fit.residual_max_ns, 2 * kStampTickNs) << link;
}

}  // namespace

TEST_F(MeshDeviceFixture, ActiveEthPtpStamps) {
    if (arch_ != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "1588 stamping is Blackhole's";
    }
    auto& cluster = MetalContext::instance().get_cluster();
    std::map<ChipId, std::shared_ptr<distributed::MeshDevice>> mesh_of;
    for (const auto& mesh : devices_) {
        mesh_of.emplace(mesh->get_device_ids()[0], mesh);
    }
    size_t links = 0;
    for (const auto& [chip_a, mesh_a] : mesh_of) {
        for (const auto& [chip_b, cores] : cluster.get_ethernet_cores_grouped_by_connected_chips(chip_a)) {
            const auto mesh_b = mesh_of.find(chip_b);
            if (chip_b <= chip_a || mesh_b == mesh_of.end()) {
                continue;
            }
            for (const CoreCoord& eth_a : cores) {
                const CoreCoord eth_b =
                    std::get<1>(cluster.get_connected_ethernet_core(std::make_tuple(chip_a, eth_a)));
                run_link(this, mesh_a, mesh_b->second, eth_a, eth_b);
                links++;
            }
        }
    }
    if (links == 0) {
        GTEST_SKIP() << "no ethernet link between local chips";
    }
}

}  // namespace tt::tt_metal
