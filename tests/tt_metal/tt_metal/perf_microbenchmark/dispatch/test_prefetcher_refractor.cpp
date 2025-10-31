// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TODO: need to remove all the includes that are not needed for this test
#include "gtest/gtest.h"
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <hostdevcommon/common_values.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/control_plane.hpp>
#include "command_queue_fixture.hpp"

// test_prefetcher_refractor.cpp:
// get num of available devices
// tt_metal::CreateDevice(test_device_id_g)
// tt_metal::CreateProgram() x 2
// reserve l1 buffer and make sure we dont exceed the l1 size
/// get cq_start -> not sure what this is
// hugepage base
// call configure_for_single_chip
// configure dispatch/prefetch d core
// call worker_core_from_logical_core for above
// some l1 base stuff + get size of prefetch d buffer
// some important asserts related to sync
// get some info like mmio device id, host_hugepage_base
// create semaphores for prefetch/dispatch core
// prefetch_defines
// noc indices for prefetch/dispatch core
// if split prefetcher is true, then configure prefetch d kernels and dispatch d kernels
// get some wr pointers
// dispatch_defines
// if split dispatcher is true, then configure dispatch h kernels and dispatch d kernels
// some more ifs test type 2, 3, 4...
// umd::writer -> prefetch_q_writer -> what does this do?
// call to  DeviceData device_data
// l1 and dram barriers for each device
// call to gen_terminate_cmds
// call to gen_prefetcher_cmds
// -> lots of switches here to generate different test types
// 1: gen_smoke_test
// 2: gen_rnd_test
// 3: gen_pcie_test
// 4: gen_paged_read_dram_test
// 5: gen_paged_write_read_dram_test
// 6: gen_host_test
// 7: gen_packed_read_test
// 8: gen_ringbuffer_read_test
// 9: gen_relay_linear_h_test
// if (warmup_g) -> warm up cache
// if readback_every_iteration_g -> run gen_prefetcher_cmds and
// validate test for iterations_g times and check if pass is true ->>> else -> run gen_prefetcher_cmds and validate test
// for 1 time see if close device passes

namespace tt::tt_dispatch {
namespace prefetcher_tests {
class PrefetcherTestFixture : public tt_metal::MeshDispatchFixture {
public:
};

TEST_F(PrefetcherTestFixture, PrefetcherTest) {
    log_info(tt::LogTest, "PrefetcherTestFixture - Test Start");
    log_info(tt::LogTest, "PrefetcherTestFixture - Test End");
}
}  // namespace prefetcher_tests
}  // namespace tt::tt_dispatch
