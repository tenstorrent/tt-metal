// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side checks of the signaler that llama_rs_matmul fuses into its matmul: every semaphore it allocates and every
// core its signal reaches must belong to the op. A program initializes a semaphore on every core of its range, and the
// privileged matmul core multicasts the signal over NOC rectangles, so a range or a rectangle that grows to the RS
// cores' bounding box writes into cores the op does not own. On Wormhole Galaxy that box holds the DRAM prefetcher's
// sender cores, whose live kernel text sits at the same L1 offsets.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>

#include "impl/buffers/semaphore.hpp"
#include "impl/program/program_impl.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/unit_tests/gtests/ttnn_test_fixtures.hpp"

namespace {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::NOC;
using tt::tt_metal::Program;
using tt::tt_metal::Semaphore;
using ttnn::experimental::ccl::MatmulFusedOpSignaler;
using ttnn::experimental::ccl::MatmulFusedOpSignalerType;

// Shaped like the RS cores of the Wormhole Galaxy decode MLP: disjoint rectangles whose bounding box also covers cores
// the op does not own, the prefetcher's sender column (x = 4) among them. The Galaxy layout's last rectangle sits on
// row 8; it is moved to row 6 so the layout fits every Wormhole worker grid.
CoreRangeSet rs_cores() {
    return CoreRangeSet(std::vector<CoreRange>{
        CoreRange(CoreCoord(1, 1), CoreCoord(3, 2)),
        CoreRange(CoreCoord(1, 3), CoreCoord(2, 3)),
        CoreRange(CoreCoord(5, 3), CoreCoord(6, 3)),
        CoreRange(CoreCoord(2, 6), CoreCoord(3, 6)),
    });
}

// The matmul ring, outside the RS bounding box. Its first core becomes the privileged core.
CoreRangeSet matmul_cores() { return CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(6, 0))); }

struct SignalRect {
    CoreCoord first;  // NOC coordinates
    CoreCoord last;
    uint32_t num_dests = 0;
};

// One matmul core's signaling runtime args, in the order do_signaling in
// reader_bmm_tile_layout_in1_ring_all_gather.cpp reads them.
struct SignalArgs {
    CoreCoord privileged_core;  // NOC coordinates
    uint32_t privileged_semaphore = 0;
    bool is_privileged = false;
    uint32_t target = 0;
    uint32_t signaled_semaphore = 0;
    std::vector<SignalRect> rects;
};

SignalArgs parse_signal_args(const std::vector<uint32_t>& args) {
    // at() throws on args shorter than the layout they describe; the caller turns that into a test failure.
    std::size_t i = 0;
    SignalArgs parsed;
    parsed.privileged_core = CoreCoord(args.at(i), args.at(i + 1));
    i += 2;
    parsed.privileged_semaphore = args.at(i++);
    parsed.is_privileged = args.at(i++) == 1;
    if (parsed.is_privileged) {
        parsed.target = args.at(i++);
        parsed.signaled_semaphore = args.at(i++);
        const uint32_t num_rects = args.at(i++);
        for (uint32_t r = 0; r < num_rects; ++r, i += 5) {
            parsed.rects.push_back(
                {CoreCoord(args.at(i), args.at(i + 1)), CoreCoord(args.at(i + 2), args.at(i + 3)), args.at(i + 4)});
        }
    }
    EXPECT_EQ(i, args.size()) << "runtime args beyond the signaling layout";
    return parsed;
}

bool has_semaphore(const Program& program, uint32_t id, const CoreCoord& core) {
    const std::vector<Semaphore>& semaphores = program.impl().semaphores();
    return std::any_of(semaphores.begin(), semaphores.end(), [&](const Semaphore& semaphore) {
        return semaphore.id() == id && semaphore.initialized_on_logical_core(core);
    });
}

// Sorted core names, so a mismatch prints which cores differ.
std::vector<std::string> sorted_names(const std::vector<CoreCoord>& cores) {
    std::vector<std::string> names;
    names.reserve(cores.size());
    for (const CoreCoord& core : cores) {
        names.push_back(core.str());
    }
    std::sort(names.begin(), names.end());
    return names;
}

class LlamaRsSignalerTest : public ttnn::TTNNFixtureWithDevice {
protected:
    // Built in llama_rs_matmul's order: the RS half first, then the matmul half.
    MatmulFusedOpSignaler make_signaler(Program& program) const {
        MatmulFusedOpSignaler signaler(MatmulFusedOpSignalerType::LLAMA_REDUCE_SCATTER);
        signaler.init_llama_rs_cores_rs(rs_cores(), program);
        signaler.init_llama_rs_cores_mm(matmul_cores(), program, device_);
        return signaler;
    }

    std::vector<CoreCoord> all_logical_cores() const {
        const CoreCoord grid = device_->logical_grid_size();
        std::vector<CoreCoord> cores;
        for (std::size_t y = 0; y < grid.y; ++y) {
            for (std::size_t x = 0; x < grid.x; ++x) {
                cores.emplace_back(x, y);
            }
        }
        return cores;
    }

    // The worker cores whose NOC coordinates fall inside the rectangle: every core a multicast over it reaches.
    std::vector<CoreCoord> cores_in(const SignalRect& rect) const {
        const auto [min_x, max_x] = std::minmax(rect.first.x, rect.last.x);
        const auto [min_y, max_y] = std::minmax(rect.first.y, rect.last.y);
        std::vector<CoreCoord> cores;
        for (const CoreCoord& core : all_logical_cores()) {
            const CoreCoord noc = device_->worker_core_from_logical_core(core);
            if (noc.x >= min_x && noc.x <= max_x && noc.y >= min_y && noc.y <= max_y) {
                cores.push_back(core);
            }
        }
        return cores;
    }
};

}  // namespace

TEST_F(LlamaRsSignalerTest, SemaphoresStayOnOwnedCores) {
    Program program = tt::tt_metal::CreateProgram();
    const MatmulFusedOpSignaler signaler = make_signaler(program);
    const CoreRangeSet owned = rs_cores().merge(matmul_cores());

    for (const Semaphore& semaphore : program.impl().semaphores()) {
        for (const CoreCoord& core : all_logical_cores()) {
            if (!owned.contains(core)) {
                EXPECT_FALSE(semaphore.initialized_on_logical_core(core))
                    << "semaphore " << semaphore.id() << " initialized on core " << core.str()
                    << ", not owned by the op";
            }
        }
    }

    // The RS kernels wait on this semaphore, so every RS core needs it.
    std::vector<uint32_t> rs_args;
    signaler.push_llama_rs_rt_args_for_rs(rs_args);
    ASSERT_EQ(rs_args.size(), 1u);
    for (const CoreCoord& core : tt::tt_metal::corerange_to_cores(rs_cores())) {
        EXPECT_TRUE(has_semaphore(program, rs_args.at(0), core)) << "RS semaphore missing on " << core.str();
    }
}

TEST_F(LlamaRsSignalerTest, SignalReachesEachRsCoreOnceAndNothingElse) {
    Program program = tt::tt_metal::CreateProgram();
    const MatmulFusedOpSignaler signaler = make_signaler(program);
    std::vector<uint32_t> rs_args;
    signaler.push_llama_rs_rt_args_for_rs(rs_args);
    ASSERT_EQ(rs_args.size(), 1u);
    const uint32_t rs_semaphore = rs_args.at(0);

    const std::vector<std::string> expected = sorted_names(tt::tt_metal::corerange_to_cores(rs_cores()));
    const std::vector<CoreCoord> mm_cores = tt::tt_metal::corerange_to_cores(matmul_cores());

    for (const NOC noc : {NOC::NOC_0, NOC::NOC_1}) {
        SCOPED_TRACE(noc == NOC::NOC_0 ? "NOC_0" : "NOC_1");
        std::vector<SignalArgs> per_core;
        for (const CoreCoord& core : mm_cores) {
            std::vector<uint32_t> args;
            signaler.push_llama_rs_rt_args_for_mm(args, core, noc, device_);
            ASSERT_NO_THROW(per_core.push_back(parse_signal_args(args)))
                << "runtime args of core " << core.str() << " end before the layout do_signaling reads";
        }
        const auto is_privileged = [](const SignalArgs& args) { return args.is_privileged; };
        ASSERT_EQ(std::count_if(per_core.begin(), per_core.end(), is_privileged), 1);
        const auto privileged = std::find_if(per_core.begin(), per_core.end(), is_privileged);
        const CoreCoord privileged_core = mm_cores.at(std::distance(per_core.begin(), privileged));

        // Every other matmul core increments a semaphore on the privileged core, which waits until all have arrived.
        EXPECT_EQ(privileged->target, mm_cores.size() - 1);
        EXPECT_TRUE(has_semaphore(program, privileged->privileged_semaphore, privileged_core));
        for (const SignalArgs& args : per_core) {
            EXPECT_EQ(args.privileged_core.str(), device_->worker_core_from_logical_core(privileged_core).str());
            EXPECT_EQ(args.privileged_semaphore, privileged->privileged_semaphore);
        }

        // The privileged core then signals the semaphore the RS kernels wait on, one rectangle at a time.
        EXPECT_EQ(privileged->signaled_semaphore, rs_semaphore);
        std::vector<CoreCoord> reached;
        for (const SignalRect& rect : privileged->rects) {
            // NOC1 walks the grid the other way, so its rectangles start at the far corner.
            if (noc == NOC::NOC_0) {
                EXPECT_TRUE(rect.first.x <= rect.last.x && rect.first.y <= rect.last.y);
            } else {
                EXPECT_TRUE(rect.first.x >= rect.last.x && rect.first.y >= rect.last.y);
            }
            const std::vector<CoreCoord> cores = cores_in(rect);
            EXPECT_EQ(rect.num_dests, cores.size()) << "multicast destination count";
            reached.insert(reached.end(), cores.begin(), cores.end());
        }
        EXPECT_EQ(sorted_names(reached), expected) << "cores the signal reaches vs the RS cores";
    }
}
