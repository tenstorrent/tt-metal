// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Blackhole hang reproducer (traffic only): both DRISC-usable DRAM tiles of one bank stream GDDR DMA reads on the
// SAME TX stream with ~7 reads in flight each, while Tensix cores stream NOC1 DRAM reads from that bank (its NOC1
// endpoint is one of the two tiles). Seen 2026-10-14 in the flat routed expert's DRISC weight senders (bh-lb-09): the
// Tensix NOC1 reads never returned (tt-triage: the readers' NOC1 reads outstanding), 3 of 3 runs. Tile 1 alone,
// tile 2 alone, no DMA, split streams (tile 0 -> stream 0, tile 1 -> stream 1) or <= 2 reads in flight all passed.
// The DRISC tiles also NoC-wrote every block to Tensix cores there ("forward").
//
// Nothing waits on another core: fixed counts, the DRISC kernels wait only for their own DMA queue / NoC flushes,
// the Tensix readers for their own reads. A hang shows as the dispatch timeout of a Finish.
//
// DISABLED by default (expected to hang the chip). Run one case at a time through the pytest wrapper
// tests/ttnn/unit_tests/operations/debug/noc_wedge/test_drisc_dma_wedge.py (scripts/run_safe_pytest.sh), or by hand:
//   ./build/test/tt_metal/unit_tests_api --gtest_also_run_disabled_tests \
//       --gtest_filter='DriscNocWedge/DriscNocWedgeFixture.DISABLED_Run/same_stream_depth7'

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "device_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr const char* kDriscKernel = "tests/ttnn/unit_tests/operations/debug/noc_wedge/kernels/drisc_dma_stream.cpp";
constexpr const char* kTrafficKernel = "tests/ttnn/unit_tests/operations/debug/noc_wedge/kernels/noc_traffic.cpp";

struct DriscWedgeCase {
    std::string name;
    bool noc1_endpoint_tile;  // the bank's NOC1-endpoint tile streams
    bool free_tile;           // the bank's other DRISC-usable tile streams
    bool split_streams;       // NOC1-endpoint tile on stream 0, free tile on stream 1 (else both on stream 0)
    uint32_t depth;           // DMA reads in flight per tile
    bool forward;             // each landed block NoC-written to a Tensix core
    bool noc1_reads;          // Tensix cores stream NOC1 DRAM reads from the bank
    uint32_t launches;
};

// Logical DRAM endpoint whose NIU serves `noc`'s reads of bank `view` (see test_dram_kernels.cpp).
CoreCoord endpoint_for_noc(const metal_SocDescriptor& soc, uint32_t view, NOC noc) {
    CoreCoord pref = soc.get_preferred_worker_core_for_dram_view(view, static_cast<uint8_t>(noc));
    const auto& eps = soc.dram_bank_endpoint_coords.at(view);
    for (uint32_t i = 0; i < eps.size(); i++) {
        if (eps[i] == pref) {
            return CoreCoord{view, i};
        }
    }
    TT_FATAL(false, "preferred DRAM endpoint of bank {} not found", view);
    return {};
}

}  // namespace

class DriscNocWedgeFixture : public BlackholeSingleCardFixture, public testing::WithParamInterface<DriscWedgeCase> {
protected:
    void SetUp() override {
        BlackholeSingleCardFixture::SetUp();
        if (devices_.empty()) {
            return;
        }
        if (!MetalContext::instance().hal().has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
            GTEST_SKIP() << "DRAM programmable cores not enabled";
        }
    }
};

TEST_P(DriscNocWedgeFixture, DISABLED_Run) {
    const DriscWedgeCase& c = GetParam();
    auto* mesh = devices_[0].get();
    const auto& hal = MetalContext::instance().hal();
    const auto& soc = MetalContext::instance().get_cluster().get_soc_desc(mesh->build_id());
    constexpr uint32_t bank = 0;
    constexpr uint32_t block = 9216;      // the flat expert's DRISC weight block
    constexpr uint32_t region = 4 << 20;  // GDDR scratch in the bank
    constexpr uint32_t dma_iters = 20000;
    constexpr uint32_t read_bytes = 8192, reads_in_flight = 4, read_iters = 2000;

    const uint32_t drisc_l1 = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint32_t tensix_l1 = mesh->allocator()->get_base_allocator_addr(HalMemType::L1);
    ASSERT_LE(c.depth * block, 96u * 1024u);

    // a single-page buffer: the whole region in `bank`
    auto dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = region},
        {.page_size = region, .buffer_type = BufferType::DRAM},
        mesh);
    const uint32_t dram_addr = dram->address();

    const CoreCoord noc0_ep = endpoint_for_noc(soc, bank, NOC::NOC_0);
    const CoreCoord noc1_ep = endpoint_for_noc(soc, bank, NOC::NOC_1);
    std::vector<CoreCoord> usable;
    for (uint32_t sub = 0; sub < soc.get_dram_compute_grid_size().y; ++sub) {
        if (sub != noc0_ep.y) {
            usable.push_back(CoreCoord{bank, sub});
        }
    }
    ASSERT_EQ(usable.size(), 2u) << "expected two DRISC-usable tiles in bank " << bank;
    ASSERT_NE(noc1_ep.y, noc0_ep.y) << "the NOC1 endpoint is the firmware-owned NOC0 endpoint";
    const CoreCoord free_tile = usable[0] == noc1_ep ? usable[1] : usable[0];
    log_info(
        LogTest,
        "DRISC_WEDGE {}: bank {} NOC1-endpoint tile {} free tile {} (NOC0 endpoint {})",
        c.name,
        bank,
        noc1_ep.str(),
        free_tile.str(),
        noc0_ep.str());

    // Tensix forward targets: logical (2..5, 2..3); NOC1 DRAM readers: every other worker
    const CoreCoord grid = mesh->compute_with_storage_grid_size();
    std::vector<uint32_t> targets;
    for (uint32_t x = 2; x <= 5; ++x) {
        for (uint32_t y = 2; y <= 3; ++y) {
            const CoreCoord v = mesh->worker_core_from_logical_core({x, y});
            targets.push_back((v.x << 16) | v.y);
        }
    }

    Program program = CreateProgram(*mesh);
    auto add_drisc = [&](CoreCoord tile, uint32_t stream, NOC noc) {
        auto k = CreateKernel(program, kDriscKernel, tile, DramConfig{.noc = noc});
        std::vector<uint32_t> args = {
            dma_iters,
            dram_addr,
            region,
            block,
            drisc_l1,
            stream,
            c.depth,
            static_cast<uint32_t>(c.forward),
            tensix_l1 + (256u << 10),
            static_cast<uint32_t>(targets.size())};
        args.insert(args.end(), targets.begin(), targets.end());
        SetRuntimeArgs(program, k, tile, args);
    };
    // the NOC1-endpoint tile's NOC1 NIU is the bank's NOC2AXI endpoint: its kernel uses NOC0; the free tile NOC1
    if (c.noc1_endpoint_tile) {
        add_drisc(noc1_ep, 0, NOC::NOC_0);
    }
    if (c.free_tile) {
        add_drisc(free_tile, c.split_streams ? 1 : 0, NOC::NOC_1);
    }
    if (c.noc1_reads) {
        std::vector<CoreCoord> readers;
        for (uint32_t y = 0; y < grid.y; ++y) {
            for (uint32_t x = 0; x < grid.x; ++x) {
                if (!(x >= 2 && x <= 5 && y >= 2 && y <= 3)) {
                    readers.push_back({x, y});
                }
            }
        }
        CoreRangeSet rs(std::vector<CoreRange>(readers.begin(), readers.end()));
        auto k = CreateKernel(
            program,
            kTrafficKernel,
            rs,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::NOC_1,
                .compile_args = {/*ROLE_DRAM_READ*/ 0, reads_in_flight, 0, 0}});
        // iters, dram, region, bytes, first bank, nbanks, l1, pinned
        SetRuntimeArgs(program, k, rs, {read_iters, dram_addr, region, read_bytes, bank, 1u, tensix_l1, 1u});
    }

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0)), std::move(program));
    auto& cq = mesh->mesh_command_queue();
    for (uint32_t i = 0; i < c.launches; ++i) {
        distributed::EnqueueMeshWorkload(cq, workload, false);
        if (i % 10 == 9) {
            distributed::Finish(cq);
            if (i % 50 == 49) {
                log_info(LogTest, "DRISC_WEDGE {}: launch {}/{}", c.name, i + 1, c.launches);
            }
        }
    }
    distributed::Finish(cq);
    log_info(LogTest, "DRISC_WEDGE {}: done, {} launches, no hang", c.name, c.launches);
}

INSTANTIATE_TEST_SUITE_P(
    DriscNocWedge,
    DriscNocWedgeFixture,
    testing::Values(
        // name, NOC1-endpoint tile, free tile, split streams, depth, forward, NOC1 reads, launches
        DriscWedgeCase{"same_stream_depth7", true, true, false, 7, true, true, 200},               // hang (3/3)
        DriscWedgeCase{"same_stream_depth7_noforward", true, true, false, 7, false, true, 200},    // open
        DriscWedgeCase{"split_streams_depth7", true, true, true, 7, true, true, 200},              // clean
        DriscWedgeCase{"same_stream_depth2", true, true, false, 2, true, true, 200},               // clean
        DriscWedgeCase{"noc1_endpoint_tile_only_depth7", true, false, false, 7, true, true, 200},  // clean
        DriscWedgeCase{"free_tile_only_depth7", false, true, false, 7, true, true, 200},           // clean
        DriscWedgeCase{"same_stream_depth7_no_noc_reads", true, true, false, 7, true, false, 200}  // clean
        ),
    [](const testing::TestParamInfo<DriscWedgeCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------------------------------------------
// Data check (not a hang): ONE DRISC tile running two DMA read jobs (job j on TX stream j), each with `depth` 9 KB
// reads in flight, consuming a slot once its stream's outstanding count drops to depth - 1. In the flat routed
// expert's DRISC senders (GLM-Flash, 2026-11-02) that sent wrong weights with depth >= 4 (y differed) and was
// bit-exact with depth 3; the op capped DMA reads in flight at 2 (SD_DMA_CAP, flat_routed_expert_plan.cpp
// "4 or more DMA reads in flight a job sent wrong weights"). The kernel checks every landed block against the
// pattern the host wrote and the test fails on any mismatch.
// ---------------------------------------------------------------------------------------------------------------

namespace {

constexpr const char* kDriscCheckKernel =
    "tests/ttnn/unit_tests/operations/debug/noc_wedge/kernels/drisc_dma_check.cpp";

struct DriscCheckCase {
    std::string name;
    bool noc1_endpoint_tile;  // run on the bank's NOC1-endpoint tile (else its free tile)
    uint32_t njobs;
    uint32_t depth;
    bool forward;     // each checked block NoC-written to a Tensix core (as the senders did)
    bool noc1_reads;  // Tensix cores stream NOC1 DRAM reads from the bank meanwhile
    uint32_t launches;
};

}  // namespace

class DriscDmaCheckFixture : public BlackholeSingleCardFixture, public testing::WithParamInterface<DriscCheckCase> {
protected:
    void SetUp() override {
        BlackholeSingleCardFixture::SetUp();
        if (devices_.empty()) {
            return;
        }
        if (!MetalContext::instance().hal().has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
            GTEST_SKIP() << "DRAM programmable cores not enabled";
        }
    }
};

TEST_P(DriscDmaCheckFixture, DISABLED_Run) {
    const DriscCheckCase& c = GetParam();
    auto* mesh = devices_[0].get();
    const auto& hal = MetalContext::instance().hal();
    const auto& soc = MetalContext::instance().get_cluster().get_soc_desc(mesh->build_id());
    constexpr uint32_t bank = 0;
    constexpr uint32_t block = 9216;
    constexpr uint32_t slots = 64;  // blocks per job region
    constexpr uint32_t region = slots * block;
    constexpr uint32_t iters = 20000;  // reads per job per launch
    constexpr uint32_t stride = 64;    // words between checked words

    const uint32_t drisc_l1 = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint64_t drisc_l1_noc = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint32_t drisc_l1_size = hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint32_t res_off = c.njobs * c.depth * block;
    ASSERT_LE(res_off + 64, drisc_l1_size) << "rings do not fit the DRISC L1";
    const uint32_t tensix_l1 = mesh->allocator()->get_base_allocator_addr(HalMemType::L1);

    const CoreCoord noc0_ep = endpoint_for_noc(soc, bank, NOC::NOC_0);
    const CoreCoord noc1_ep = endpoint_for_noc(soc, bank, NOC::NOC_1);
    CoreCoord tile = noc1_ep;
    if (!c.noc1_endpoint_tile) {
        for (uint32_t sub = 0; sub < soc.get_dram_compute_grid_size().y; ++sub) {
            if (sub != noc0_ep.y && sub != noc1_ep.y) {
                tile = CoreCoord{bank, sub};
            }
        }
    }
    ASSERT_NE(tile.y, noc0_ep.y);

    auto dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = 2 * region},
        {.page_size = 2 * region, .buffer_type = BufferType::DRAM},
        mesh);
    const uint32_t dram_addr = dram->address();
    std::vector<uint32_t> pattern(2 * region / 4);
    for (uint32_t j = 0; j < 2; ++j) {
        for (uint32_t b = 0; b < slots; ++b) {
            for (uint32_t k = 0; k < block / 4; ++k) {
                pattern[(j * region + b * block) / 4 + k] = (j << 30) | (b << 14) | k;
            }
        }
    }
    slow_dispatch::WriteToDRAMChannel(*mesh, mesh->dram_channel_from_logical_core(tile), dram_addr, pattern);

    Program program = CreateProgram(*mesh);
    const CoreCoord fwd = mesh->worker_core_from_logical_core({3, 3});
    // the NOC1-endpoint tile's NOC1 NIU is the bank's NOC2AXI endpoint: its kernel uses NOC0; the free tile NOC1
    auto k = CreateKernel(
        program, kDriscCheckKernel, tile, DramConfig{.noc = c.noc1_endpoint_tile ? NOC::NOC_0 : NOC::NOC_1});
    SetRuntimeArgs(
        program,
        k,
        tile,
        {iters,
         dram_addr,
         region,
         block,
         drisc_l1,
         c.depth,
         c.njobs,
         stride,
         drisc_l1 + res_off,
         static_cast<uint32_t>(c.forward),
         (static_cast<uint32_t>(fwd.x) << 16) | static_cast<uint32_t>(fwd.y),
         tensix_l1 + (256u << 10)});
    if (c.noc1_reads) {
        const CoreCoord grid = mesh->compute_with_storage_grid_size();
        std::vector<CoreRange> readers;
        for (uint32_t y = 0; y < grid.y; ++y) {
            for (uint32_t x = 0; x < grid.x; ++x) {
                if (!(x == 3 && y == 3)) {
                    readers.push_back(CoreRange(CoreCoord{x, y}));
                }
            }
        }
        CoreRangeSet rs(readers);
        auto kr = CreateKernel(
            program,
            kTrafficKernel,
            rs,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1, .compile_args = {0, 4, 0, 0}});
        SetRuntimeArgs(program, kr, rs, {2000u, dram_addr, 2 * region, 8192u, bank, 1u, tensix_l1, 1u});
    }
    log_info(
        LogTest,
        "DRISC_CHECK {}: tile {} (NOC1 endpoint {}), {} jobs x {} reads in flight x {} B",
        c.name,
        tile.str(),
        noc1_ep.str(),
        c.njobs,
        c.depth,
        block);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0)), std::move(program));
    auto& cq = mesh->mesh_command_queue();
    const CoreCoord vtile = mesh->virtual_core_from_logical_core(tile, CoreType::DRAM);
    uint64_t total_bad = 0, total_checked = 0, bad_launches = 0;
    for (uint32_t i = 0; i < c.launches; ++i) {
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        std::vector<uint32_t> r(7);
        MetalContext::instance().get_cluster().read_core(
            r.data(), 7 * sizeof(uint32_t), tt_cxy_pair(mesh->build_id(), vtile), drisc_l1_noc + res_off);
        total_bad += r[0];
        total_checked += r[1];
        if (r[0]) {
            if (bad_launches++ < 5) {
                log_info(
                    LogTest,
                    "DRISC_CHECK {}: launch {}: {} bad words; first at iteration {} job {} word {}: got {:#x} want "
                    "{:#x}",
                    c.name,
                    i,
                    r[0],
                    r[2],
                    r[3],
                    r[4],
                    r[5],
                    r[6]);
            }
        }
    }
    log_info(
        LogTest,
        "DRISC_CHECK {}: {} launches, {} blocks checked, {} bad words in {} launches",
        c.name,
        c.launches,
        total_checked,
        total_bad,
        bad_launches);
    EXPECT_GT(total_checked, 0u);
    EXPECT_EQ(total_bad, 0u);
}

INSTANTIATE_TEST_SUITE_P(
    DriscDmaCheck,
    DriscDmaCheckFixture,
    testing::Values(
        // name, NOC1-endpoint tile, jobs, depth, forward, NOC1 reads, launches
        DriscCheckCase{"free_tile_2jobs_depth3", false, 2, 3, true, false, 20},  // bit-exact in the op
        DriscCheckCase{"free_tile_2jobs_depth4", false, 2, 4, true, false, 20},  // wrong weights in the op
        DriscCheckCase{"free_tile_2jobs_depth5", false, 2, 5, true, false, 20},
        DriscCheckCase{"free_tile_2jobs_depth4_noforward", false, 2, 4, false, false, 20},
        DriscCheckCase{"free_tile_1job_depth5", false, 1, 5, true, false, 20},
        DriscCheckCase{"endpoint_tile_2jobs_depth4", true, 2, 4, true, false, 20},
        DriscCheckCase{"free_tile_2jobs_depth4_noc1_reads", false, 2, 4, true, true, 20}),
    [](const testing::TestParamInfo<DriscCheckCase>& info) { return info.param.name; });
