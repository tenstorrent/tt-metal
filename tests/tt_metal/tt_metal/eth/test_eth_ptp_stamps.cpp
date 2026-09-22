// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The Blackhole eth tile's IEEE-1588 stamping (hw/inc/internal/ethernet/eth_ptp.hpp) on every link between local
// chips: a StampSession opens and closes on both ends and gives back what it borrowed, every frame of a two-way
// exchange gets its egress and ingress stamps, and the stamps describe a plausible link.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
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

double quantile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(q * static_cast<double>(v.size())))];
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
    const uint32_t res_addr = base + eth_ptp_stamps::kResultOffset;
    std::vector<uint32_t> zero(sizeof(Result) / sizeof(uint32_t), 0);
    ASSERT_TRUE(detail::WriteToDeviceL1(dev_a, eth_a, res_addr, zero, CoreType::ETH));
    ASSERT_TRUE(detail::WriteToDeviceL1(dev_b, eth_b, res_addr, zero, CoreType::ETH));

    const auto workload = [&](const CoreCoord& core, bool initiator) {
        Program program;
        EthernetConfig config{.compile_args = {initiator ? 1u : 0u}};
        eth_test_common::set_arch_specific_eth_config(config);
        const KernelHandle k = CreateKernel(program, kKernel, core, config);
        SetRuntimeArgs(program, k, core, {base});
        distributed::MeshWorkload w;
        const distributed::MeshCoordinate zero_coord(0, 0);
        w.add_program(distributed::MeshCoordinateRange(zero_coord, zero_coord), std::move(program));
        return w;
    };
    distributed::MeshWorkload wa = workload(eth_a, true);
    distributed::MeshWorkload wb = workload(eth_b, false);
    if (fixture->IsSlowDispatch()) {
        std::thread ta([&] { fixture->RunProgram(mesh_a, wa); });
        std::thread tb([&] { fixture->RunProgram(mesh_b, wb); });
        ta.join();
        tb.join();
    } else {
        fixture->RunProgram(mesh_b, wb, true);
        fixture->RunProgram(mesh_a, wa, true);
        fixture->FinishCommands(mesh_a);
        fixture->FinishCommands(mesh_b);
    }

    const auto read = [&](IDevice* dev, const CoreCoord& core) {
        std::vector<uint32_t> words;
        EXPECT_TRUE(detail::ReadFromDeviceL1(dev, core, res_addr, sizeof(Result), words, CoreType::ETH));
        Result r{};
        std::memcpy(&r, words.data(), std::min(sizeof(r), words.size() * sizeof(uint32_t)));
        return r;
    };
    const Result a = read(dev_a, eth_a);
    const Result b = read(dev_b, eth_b);
    const std::string link =
        fmt::format("chip {} eth {} -> chip {} eth {}", dev_a->id(), eth_a.str(), dev_b->id(), eth_b.str());

    for (const auto& [r, end] : {std::pair{&a, "initiator"}, std::pair{&b, "echo"}}) {
        ASSERT_EQ(r->done, eth_ptp_stamps::kDone) << link << ": the " << end << " kernel did not finish";
        EXPECT_EQ(r->timer_ok, 1u) << link << ": the " << end << "'s PTP timer did not acknowledge its rate";
        EXPECT_EQ(r->sel_after, r->sel_before)
            << link << ": the " << end << "'s TX header row selection was not restored";
        EXPECT_EQ(r->no_match_after, r->no_match_before)
            << link << ": the " << end << "'s RX no-match actions were not restored";
        ASSERT_EQ(r->rounds, eth_ptp_stamps::kRounds) << link << ": the " << end << " stopped waiting for its peer";
        EXPECT_EQ(r->tx_missing, 0u) << link << ": " << end << " frames without an egress stamp";
        EXPECT_EQ(r->rx_missing, 0u) << link << ": " << end << " frames without an ingress stamp";
        EXPECT_LE(r->tx_extra + r->rx_extra, eth_ptp_stamps::kRounds / 64)
            << link << ": " << end << " stamps beyond one per frame";
    }

    // Round i: t0 initiator egress, t1 echo ingress, t1b echo egress, t2 initiator ingress, each end in its own PTP ns.
    std::vector<double> one_way, offset, at;
    for (uint32_t i = 0; i < eth_ptp_stamps::kRounds; i++) {
        const auto t0 = static_cast<double>(a.stamps[i][0]), t2 = static_cast<double>(a.stamps[i][1]);
        const auto t1 = static_cast<double>(b.stamps[i][0]), t1b = static_cast<double>(b.stamps[i][1]);
        EXPECT_LT(t0, t2) << link << " round " << i;
        EXPECT_LT(t1, t1b) << link << " round " << i;
        if (i > 0) {
            EXPECT_GT(t0, static_cast<double>(a.stamps[i - 1][1])) << link << " round " << i;
            EXPECT_GT(t1, static_cast<double>(b.stamps[i - 1][1])) << link << " round " << i;
        }
        one_way.push_back(0.5 * ((t2 - t0) - (t1b - t1)));
        offset.push_back(0.5 * ((t1 + t1b) - (t0 + t2)));
        at.push_back(0.5 * (t0 + t2));
    }
    // The two ends' clocks drift apart by a few ppm over the exchange: offsets against a line through them.
    const double n = static_cast<double>(at.size());
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (size_t i = 0; i < at.size(); i++) {
        const double x = at[i] - at.front();
        sx += x, sy += offset[i], sxx += x * x, sxy += x * offset[i];
    }
    const double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx), inter = (sy - slope * sx) / n;
    double ss = 0;
    for (size_t i = 0; i < at.size(); i++) {
        const double r = offset[i] - (inter + slope * (at[i] - at.front()));
        ss += r * r;
    }
    const double resid = std::sqrt(ss / n);
    const double ow = quantile(one_way, 0.5), spread = quantile(one_way, 0.9) - quantile(one_way, 0.1);
    log_info(
        tt::LogTest,
        "{}: one way inside the stamps {:.1f} ns (p10-p90 {:.1f}), offset residual {:.2f} ns rms, rate {:+.3f} ppm, "
        "extra stamps {}/{}",
        link,
        ow,
        spread,
        resid,
        slope * 1e6,
        a.tx_extra + a.rx_extra,
        b.tx_extra + b.rx_extra);
    // A passive DAC between two p150s: ~34 ns. A single frame's stamps step by the timer's 20 ns tick.
    EXPECT_GT(ow, 20.0) << link;
    EXPECT_LT(ow, 60.0) << link;
    EXPECT_LT(spread, 20.0) << link;
    EXPECT_LT(resid, 10.0) << link;
}

}  // namespace

TEST_F(MeshDeviceFixture, ActiveEthPtpStamps) {
    if (arch_ != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "1588 stamping is Blackhole's";
    }
    auto& cluster = MetalContext::instance().get_cluster();
    size_t links = 0;
    for (const auto& mesh_a : devices_) {
        IDevice* dev_a = mesh_a->get_devices()[0];
        for (const auto& [chip_b, cores] : cluster.get_ethernet_cores_grouped_by_connected_chips(dev_a->id())) {
            if (chip_b <= dev_a->id()) {
                continue;
            }
            for (const auto& mesh_b : devices_) {
                if (mesh_b->get_device_ids()[0] != chip_b) {
                    continue;
                }
                for (const CoreCoord& eth_a : cores) {
                    const CoreCoord eth_b =
                        std::get<1>(cluster.get_connected_ethernet_core(std::make_tuple(dev_a->id(), eth_a)));
                    run_link(this, mesh_a, mesh_b, eth_a, eth_b);
                    links++;
                }
            }
        }
    }
    if (links == 0) {
        GTEST_SKIP() << "no ethernet link between local chips";
    }
}

}  // namespace tt::tt_metal
