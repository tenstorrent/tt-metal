// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariants of PrefetcherPipeParameter geometry (prefetcher_pipe_parameter.hpp).

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "metal2_host_api/prefetcher_pipe_test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeSenderOnlySpec;
using test_helpers::pipe_ring_size;
using test_helpers::PrefetcherPipeSpecTestQuasar;

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_EmptyReceiversFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.advanced_options.prefetcher_pipe_parameters[0].receivers = NodeRangeSet{};
    EXPECT_SPEC_REJECTED(spec, "PrefetcherPipeParameter 'weights' has no receiver nodes");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_ZeroRingSizeFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.advanced_options.prefetcher_pipe_parameters[0].ring_size = 0;
    EXPECT_SPEC_REJECTED(spec, "PrefetcherPipeParameter 'weights' has ring_size = 0");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_ZeroEntrySizeFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.advanced_options.prefetcher_pipe_parameters[0].entry_size = 0;
    EXPECT_SPEC_REJECTED(spec, "PrefetcherPipeParameter 'weights' has entry_size = 0");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_UnalignedEntrySizeFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.advanced_options.prefetcher_pipe_parameters[0].entry_size = 2048 + 4;
    EXPECT_SPEC_REJECTED(spec, "entry_size 2052 must be a multiple of the L1 alignment");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_EntrySizeLargerThanRingFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.advanced_options.prefetcher_pipe_parameters[0].entry_size = pipe_ring_size * 2;
    EXPECT_SPEC_REJECTED(spec, "exceeds ring_size");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_OutOfBoundsReceiverFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.advanced_options.prefetcher_pipe_parameters[0].receivers = NodeCoord{1000, 1000};
    EXPECT_SPEC_REJECTED(spec, "PrefetcherPipeParameter 'weights' targets node (1000,1000), which is out of bounds");
}

}  // namespace
}  // namespace tt::tt_metal::experimental
