// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "command_queue_fixture.hpp"
#include "command_queue_test_utils.hpp"
#include "gtest/gtest.h"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt::tt_metal;

TEST_F(CommandQueueFixture, TestEnqueueRestart) {
    GTEST_SKIP() << "Skipping restart test until restart support added back in";

    Program program;
    CoreRange cr = {.start = {0, 0}, .end = {0, 0}};
    CoreRangeSet cr_set({cr});
    // Add an NCRISC blank manually, but in compile program, the BRISC blank will be
    // added separately
    auto dummy_reader_kernel = CreateKernel(
        program, "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/arbiter_hang.cpp", cr_set, DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    CommandQueue cq(this->device_, 0);

    uint32_t starting_issue_read_ptr;
    uint32_t starting_issue_write_ptr;
    tt_cxy_pair physical_producer_core(this->device_->id(), this->device_->worker_core_from_logical_core(cq.issue_queue_reader_core));
    tt::Cluster::instance().read_core(&starting_issue_read_ptr, 4, physical_producer_core, CQ_ISSUE_READ_PTR, false);
    tt::Cluster::instance().read_core(&starting_issue_write_ptr, 4, physical_producer_core, CQ_ISSUE_WRITE_PTR, false);

    uint32_t starting_completion_read_ptr;
    uint32_t starting_completion_write_ptr;
    tt_cxy_pair physical_consumer_core(this->device_->id(), this->device_->worker_core_from_logical_core(cq.completion_queue_writer_core));
    tt::Cluster::instance().read_core(&starting_completion_read_ptr, 4, physical_consumer_core, CQ_COMPLETION_READ_PTR, false);
    tt::Cluster::instance().read_core(&starting_completion_write_ptr, 4, physical_consumer_core, CQ_COMPLETION_WRITE_PTR, false);
    EnqueueProgram(cq, program, false);
    Finish(cq);
    ::detail::EnqueueRestart(cq);

    // Assert that the device and system memory pointers successfully
    // restarted
    uint32_t ending_issue_read_ptr;
    uint32_t ending_issue_write_ptr;
    tt::Cluster::instance().read_core(&ending_issue_read_ptr, 4, physical_producer_core, CQ_ISSUE_READ_PTR, false);
    tt::Cluster::instance().read_core(&ending_issue_write_ptr, 4, physical_producer_core, CQ_ISSUE_WRITE_PTR, false);

    uint32_t ending_completion_read_ptr;
    uint32_t ending_completion_write_ptr;
    tt::Cluster::instance().read_core(&ending_completion_read_ptr, 4, physical_consumer_core, CQ_COMPLETION_READ_PTR, false);
    tt::Cluster::instance().read_core(&ending_completion_write_ptr, 4, physical_consumer_core, CQ_COMPLETION_WRITE_PTR, false);
    EXPECT_EQ(starting_issue_read_ptr, ending_issue_read_ptr);
    EXPECT_EQ(starting_issue_write_ptr, ending_issue_write_ptr);
    EXPECT_EQ(starting_completion_read_ptr, ending_completion_read_ptr);
    EXPECT_EQ(starting_completion_write_ptr, ending_completion_write_ptr);
}
