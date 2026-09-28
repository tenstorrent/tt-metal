// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// JIT-compile smoke test for the TT_KERNEL compute-path kernel_main() shim (mock Wormhole).

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1ComputeKernel;
using test_helpers::MakeMinimalReaderDMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;

// ============================================================================
// TT_KERNEL ("1st world arguments") compute-path shim — JIT compile smoke test
// ============================================================================
//
// Compiles a TT_KERNEL compute kernel through genfiles + the RISC-V compiler on a MOCK Wormhole
// device (no silicon, no dispatch) via program.impl().compile. This is the only no-hardware
// coverage that the generated kernel_main() shim is emitted on the COMPUTE (TRISC) compile path
// and actually compiles: the on-hardware compute test (TtKernelNamedArgsLoopbackCompute) skips in
// CI, and the shim unit tests only check the generated string, not its genfiles wiring.
//
// (A Quasar variant would also exercise the 4th TRISC, isolate_sfpu, but mock-Quasar JIT-compile
// isn't wired up in this checkout — the Quasar TRISC firmware objects aren't built, so the link
// step fails. The fix is arch-correct by construction regardless: the shim is appended to the same
// source for every TRISC, and run_kernel() calls kernel_main() on Quasar too.)

// Minimal TT_KERNEL compute entry: CTAs as template params, RTA/CRTA as function params, producing
// into a DFB. The point is solely that the kernel_main() shim is generated on the TRISC path and
// the whole thing compiles; the body avoids any arch-specific raw-L1 pokes.
constexpr const char* kTtKernelComputeShimSource = R"(
#include "api/compute/common.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
template <uint32_t magic, uint32_t entry_size>                             // CTAs
TT_KERNEL void compute_entry(uint32_t input_offset, uint32_t num_tiles) {  // RTA, CRTA
    DataflowBuffer out(dfb::out_dfb);
    out.reserve_back(num_tiles);
    out.push_back(num_tiles);
    volatile uint32_t sink = magic ^ entry_size ^ input_offset;
    (void)sink;
}

static_assert(dfb::get_token_if_present<"out_dfb">() == &dfb::out_dfb);
static_assert(dfb::get_token_if_present<"not_a_dfb">() == nullptr);
)";

TEST_F(ProgramSpecTestGen1, CPU_TtKernelComputeShimCompiles) {
    const NodeCoord node{0, 0};
    constexpr uint32_t entry_size = 1024;

    // Compute kernel authored in TT_KERNEL form, producing into a DFB drained by a trivial consumer.
    auto compute = MakeMinimalGen1ComputeKernel("compute");
    compute.source = KernelSpec::SourceCode{kTtKernelComputeShimSource};
    compute.runtime_arg_schema.runtime_arg_names = {"input_offset"};
    compute.runtime_arg_schema.common_runtime_arg_names = {"num_tiles"};
    compute.compile_time_args = {{"magic", 0xCAFE0001u}, {"entry_size", entry_size}};

    auto consumer = MakeMinimalReaderDMKernel("consumer");  // trivial drain kernel

    auto out_dfb = MakeMinimalDFB("out_dfb", entry_size, 4);
    out_dfb.data_format_metadata = tt::DataFormat::Float16_b;  // required for a compute DFB endpoint
    compute.dfb_bindings.push_back(ProducerOf(DFBSpecName{"out_dfb"}, "out_dfb"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"out_dfb"}, "out_dfb"));

    ProgramSpec spec;
    spec.name = "tt_kernel_compute_shim_compile";
    spec.kernels = {compute, consumer};
    spec.dataflow_buffers = {out_dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("wu", node, {"compute", "consumer"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
