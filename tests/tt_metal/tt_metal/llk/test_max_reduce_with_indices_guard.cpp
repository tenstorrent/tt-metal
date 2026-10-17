// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compile-only guard test for max_reduce_with_indices<num_rows, layout, accumulate>.
//
// The num_rows <= 9 path of the SFPU kernel never reads `accumulate` or `chunk` and never
// touches the accumulator tiles at idst + 1 / idst_idx + 1, so a kernel asking for
// <9, ROW_MAJOR, true> compiled and silently returned only the last chunk's max. The kernel
// now static_asserts !accumulate on that path. This test compiles a probe kernel with the
// three relevant template argument sets: the two supported ones must build (so a rejection
// is attributable to the guard and not to the probe itself), and the guarded one must fail
// with the guard's message. Nothing runs on the device; it is only needed for the JIT build.

#include <gtest/gtest.h>

#include <exception>
#include <string>
#include <string_view>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include "impl/program/program_impl.hpp"
#include "llk_device_fixture.hpp"

namespace tt::tt_metal {
namespace {

const std::string kProbeKernel = "tests/tt_metal/tt_metal/test_kernels/compute/max_reduce_with_indices_probe.cpp";
constexpr std::string_view kGuardMessage = "accumulate is only implemented for the 32-row ROW_MAJOR path";

// Builds the probe kernel for one <num_rows, accumulate> pair. Returns the build failure's
// message (which carries the compiler log), or an empty string when the kernel compiled.
std::string compile_probe(distributed::MeshDevice& device, int num_rows, bool accumulate) {
    Program program = CreateProgram();
    CreateKernel(
        program,
        kProbeKernel,
        CoreCoord(0, 0),
        ComputeConfig{
            .defines = {
                {"MPWI_NUM_ROWS", std::to_string(num_rows)}, {"MPWI_ACCUMULATE", accumulate ? "true" : "false"}}});
    try {
        program.impl().compile(&device);
    } catch (const std::exception& e) {
        return e.what();
    }
    return {};
}

}  // namespace

TEST_F(LLKMeshDeviceSingleCardFixture, MaxReduceWithIndicesAccumulateRejectedOnNineRowPath) {
    // Controls: the supported combinations build, so a rejection below is the guard's doing.
    EXPECT_EQ(compile_probe(this->device(), /*num_rows=*/9, /*accumulate=*/false), "");
    EXPECT_EQ(compile_probe(this->device(), /*num_rows=*/32, /*accumulate=*/true), "");

    const std::string failure = compile_probe(this->device(), /*num_rows=*/9, /*accumulate=*/true);
    EXPECT_FALSE(failure.empty()) << "max_reduce_with_indices<9, ROW_MAJOR, true> compiled; the 9-row path ignores "
                                     "accumulate, so this must be a compile error";
    EXPECT_NE(failure.find(kGuardMessage), std::string::npos)
        << "the build failed for a reason other than the accumulate guard:\n"
        << failure;
}

}  // namespace tt::tt_metal
