// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "gtest/gtest.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/common.h"
#include "tests/tt_metal/tt_metal/unit_tests/common/device_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/common/math.hpp"

using namespace tt::tt_metal;

// TODO: Remove dependency on "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/common.h" and remove globals
bool debug_g = false;
// Page size 4096 bytes
uint32_t log_dispatch_buffer_page_size_g = 12;
uint32_t dispatch_buffer_page_size_g = 1 << log_dispatch_buffer_page_size_g;
bool use_coherent_data_g = false;
uint32_t hugepage_buffer_size_g = 256 * 1024 * 1024;
uint32_t dev_hugepage_base = dispatch_buffer_page_size_g;
std::pair<uint32_t, uint32_t> default_ptrs = std::make_pair(dev_hugepage_base, 0);
uint32_t hugepage_issue_buffer_size_g;

inline void gen_dispatcher_pad_to_page(vector<uint32_t>& cmds, uint32_t page_size) {
    uint32_t num_words_in_page = page_size / sizeof(uint32_t);
    uint32_t num_pad_words = tt::round_up(cmds.size(), num_words_in_page) - cmds.size();
    for (uint32_t i = 0; i < num_pad_words; ++i) {
        cmds.push_back(0);
    }
}

inline bool validate_results(
    std::vector<uint32_t>& dev_data,
    uint32_t num_words,
    void *host_hugepage_base,
    uint32_t dev_hugepage_base,
    uint32_t dev_hugepage_start,
    uint32_t hugepage_buffer_size_g) {
    bool failed = false;

    log_info(tt::LogTest, "Validating {} bytes from hugepage", num_words * sizeof(uint32_t));

    uint32_t *results = ((uint32_t *)host_hugepage_base);  // 8 = 32B / sizeof(uint32_t)
    uint32_t dev_hugepage_start_diff_uint = (dev_hugepage_start - dev_hugepage_base) / sizeof(uint32_t);
    uint32_t hugepage_buffer_size_g_uint = hugepage_buffer_size_g / sizeof(uint32_t);
    int fail_count = 0;

    for (int i = 0; i < num_words; ++i) {
        uint32_t hugepage_idx = (dev_hugepage_start_diff_uint + i) % hugepage_buffer_size_g_uint;
        if (results[hugepage_idx] != dev_data[i]) {
            if (!failed) {
                tt::log_fatal("Data mismatch");
                fprintf(stderr, "First 20 failures for each core: [idx] expected->read\n");
            }
            if (fail_count == 0) {
                fprintf(stderr, "Failures reading hugepage\n");
            }

            fprintf(stderr, "  [%02d] 0x%08x->0x%08x\n", i, (unsigned int)dev_data[i], (unsigned int)results[hugepage_idx]);

            failed = true;
            fail_count++;
            if (fail_count > 20) {
                break;
            }
        }
    }

    return !failed;
}

namespace local_test_functions {

bool test_write_host(Device *device, uint32_t data_size, std::pair<uint32_t, uint32_t> write_ptr_start = default_ptrs, std::pair<uint32_t, uint32_t> read_ptr_start = default_ptrs, std::optional<std::pair<uint32_t, uint32_t>> read_ptr_update = std::nullopt) {
    CoreCoord spoof_prefetch_core = {0, 0};
    CoreCoord dispatch_core = {4, 0};
    CoreCoord phys_spoof_prefetch_core = device->worker_core_from_logical_core(spoof_prefetch_core);
    CoreCoord phys_dispatch_core = device->worker_core_from_logical_core(dispatch_core);

    auto program = tt::tt_metal::CreateScopedProgram();

    uint32_t dispatch_buffer_size_blocks_g = 4;

    uint32_t total_size = data_size + sizeof(CQDispatchCmd);

    // NOTE: this test hijacks hugepage
    // Start after first page since ptrs are at the start of hugepage

    void *host_hugepage_base;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
    host_hugepage_base = (void *)tt::Cluster::instance().host_dma_address(0, mmio_device_id, channel);
    host_hugepage_base = (void *)((uint8_t *)host_hugepage_base + dev_hugepage_base);

    uint32_t l1_unreserved_base = devices_.at(id)->get_base_allocator_addr(HalMemType::L1);
    uint32_t l1_buf_base = align(l1_unreserved_base, dispatch_buffer_page_size_g);

    std::vector<uint32_t> dispatch_cmds;
    CQDispatchCmd cmd;
    memset(&cmd, 0, sizeof(CQDispatchCmd));
    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST;
    cmd.write_linear_host.length = data_size + sizeof(CQDispatchCmd);
    add_dispatcher_cmd(dispatch_cmds, cmd, data_size);
    gen_dispatcher_pad_to_page(dispatch_cmds, dispatch_buffer_page_size_g);
    uint32_t dev_output_num_words = total_size / sizeof(uint32_t);
    gen_dispatcher_terminate_cmd(dispatch_cmds);

    uint32_t cmd_cb_pages = tt::div_up(dispatch_cmds.size() * sizeof(uint32_t), dispatch_buffer_page_size_g);

    // Make full blocks
    uint32_t dispatch_buffer_pages = tt::round_up(cmd_cb_pages, dispatch_buffer_size_blocks_g);
    uint32_t dispatch_buffer_size_g = dispatch_buffer_pages * dispatch_buffer_page_size_g;
    TT_FATAL(l1_buf_base + dispatch_buffer_size_g <= device->l1_size_per_core(), "Does not fit in L1");

    std::vector<uint32_t> write_ptr_val = {(write_ptr_start.first >> 4) | (write_ptr_start.second << 31)};
    std::vector<uint32_t> read_ptr_val = {(read_ptr_start.first >> 4) | (read_ptr_start.second << 31)};

    uint32_t completion_q_wr_ptr = dispatch_constants::get(CoreType::WORKER).get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_WR);
    uint32_t completion_q_rd_ptr = dispatch_constants::get(dispatch_core_type).get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_RD);
    // Write the read and write ptrs
    tt::llrt::write_hex_vec_to_core(
        device->id(), phys_dispatch_core, write_ptr_val, completion_q_wr_ptr);
    tt::llrt::write_hex_vec_to_core(
        device->id(), phys_dispatch_core, read_ptr_val, completion_q_rd_ptr);

    tt::llrt::write_hex_vec_to_core(device->id(), phys_spoof_prefetch_core, dispatch_cmds, l1_buf_base);
    tt::Cluster::instance().l1_barrier(device->id());

    const uint32_t spoof_prefetch_core_sem_0_id =
        tt::tt_metal::CreateSemaphore(program, {spoof_prefetch_core}, dispatch_buffer_pages);
    const uint32_t dispatch_core_sem_id = tt::tt_metal::CreateSemaphore(program, {dispatch_core}, 0);
    TT_ASSERT(spoof_prefetch_core_sem_0_id == dispatch_core_sem_id);
    const uint32_t dispatch_cb_sem = spoof_prefetch_core_sem_0_id;

    const uint32_t spoof_prefetch_core_sem_1_id = tt::tt_metal::CreateSemaphore(program, {spoof_prefetch_core}, 0);
    const uint32_t prefetch_sync_sem = spoof_prefetch_core_sem_1_id;

    std::vector<uint32_t> dispatch_compile_args = {
        l1_buf_base,
        log_dispatch_buffer_page_size_g,
        dispatch_buffer_pages,
        dispatch_cb_sem,
        dispatch_cb_sem, // ugly, share an address
        dispatch_buffer_size_blocks_g,
        prefetch_sync_sem,
        default_ptrs.second,
        dev_hugepage_base,
        hugepage_buffer_size_g,
        0,    // unused downstream_cb_base
        0,    // unused downstream_cb_size
        0,    // unused my_downstream_cb_sem_id
        0,    // unused downstream_cb_sem_id
        0,    // unused split_dispatch_page_preamble_size
        true,
        true};
    std::vector<uint32_t> spoof_prefetch_compile_args = {
        l1_buf_base,
        log_dispatch_buffer_page_size_g,
        dispatch_buffer_pages,
        dispatch_cb_sem,
        l1_buf_base,
        cmd_cb_pages,
        // Hardcode page_batch_size to 1 to force the inner loops to only run once
        1,
        prefetch_sync_sem,
        };

    std::map<string, string> prefetch_defines = {
        {"MY_NOC_X", std::to_string(phys_spoof_prefetch_core.x)},
        {"MY_NOC_Y", std::to_string(phys_spoof_prefetch_core.y)},
        {"DISPATCH_NOC_X", std::to_string(phys_dispatch_core.x)},
        {"DISPATCH_NOC_Y", std::to_string(phys_dispatch_core.y)},
    };

    auto sp1 = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/spoof_prefetch.cpp",
        {spoof_prefetch_core},
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = spoof_prefetch_compile_args,
            .defines = prefetch_defines});

    // Hardcode outer loop to 1
    vector<uint32_t> args = {1};
    tt::tt_metal::SetRuntimeArgs(program, sp1, spoof_prefetch_core, args);

    constexpr NOC my_noc_index = NOC::NOC_0;
    constexpr NOC dispatch_upstream_noc_index = NOC::NOC_1;

    auto* program_ptr = tt::tt_metal::ProgramPool::instance().get_program(program);
    configure_kernel_variant<true, true>(program,
        "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",
        dispatch_compile_args,
        dispatch_core,
        phys_dispatch_core,
        phys_spoof_prefetch_core,
        {0, 0},
        device,
        my_noc_index,
        my_noc_index,
        my_noc_index);

    // Need a separate thread for SD
    if (read_ptr_update.has_value()) {
        std::thread t1 ([&]() {
            uint64_t run_mailbox_address = GET_MAILBOX_ADDRESS_HOST(launch.run);
            std::vector<uint32_t> run_mailbox_read_val;
            uint8_t run;
            do {
                run_mailbox_read_val = tt::llrt::read_hex_vec_from_core(device->id(), phys_dispatch_core, run_mailbox_address & ~0x3, sizeof(uint32_t));
                run = run_mailbox_read_val[0] >> (8 * (offsetof(launch_msg_t, run) & 3));
            } while (run != RUN_MSG_GO);
            sleep(1);
            std::vector<uint32_t> read_ptr_update_val = {(read_ptr_update.value().first >> 4) | (read_ptr_update.value().second << 31)};
            tt::llrt::write_hex_vec_to_core(
                device->id(), phys_dispatch_core, read_ptr_update_val, completion_q_rd_ptr);
        });
        tt::tt_metal::detail::LaunchProgram(device, *program_ptr);
        t1.join();
    } else {
        tt::tt_metal::detail::LaunchProgram(device, *program_ptr);
    }

    // Validation
    bool pass = validate_results(
        dispatch_cmds, dev_output_num_words, host_hugepage_base, dev_hugepage_base, write_ptr_start.first, hugepage_buffer_size_g);
    return pass;
}

}  // end namespace local_test_functions

namespace basic_tests {

TEST_F(DeviceSingleCardFixture, TestWriteHostBasic) {
    EXPECT_TRUE(local_test_functions::test_write_host(device_, dispatch_buffer_page_size_g - sizeof(CQDispatchCmd)));
    EXPECT_TRUE(local_test_functions::test_write_host(device_, dispatch_buffer_page_size_g));
    EXPECT_TRUE(local_test_functions::test_write_host(device_, 256));
    EXPECT_TRUE(local_test_functions::test_write_host(device_, 3 * dispatch_buffer_page_size_g));
    EXPECT_TRUE(local_test_functions::test_write_host(device_, 10 * dispatch_buffer_page_size_g));
}

TEST_F(DeviceSingleCardFixture, TestWriteHostWrap) {
    EXPECT_TRUE(local_test_functions::test_write_host(device_, 10 * dispatch_buffer_page_size_g, {hugepage_buffer_size_g - 1 * dispatch_buffer_page_size_g + dev_hugepage_base, 0}, {hugepage_buffer_size_g - 1 * dispatch_buffer_page_size_g + dev_hugepage_base, 0}));
    EXPECT_TRUE(local_test_functions::test_write_host(device_, 10 * dispatch_buffer_page_size_g, {hugepage_buffer_size_g - 2 * dispatch_buffer_page_size_g + dev_hugepage_base, 0}, {hugepage_buffer_size_g - 2 * dispatch_buffer_page_size_g + dev_hugepage_base, 0}));
    EXPECT_TRUE(local_test_functions::test_write_host(device_, 10 * dispatch_buffer_page_size_g, {hugepage_buffer_size_g - 3 * dispatch_buffer_page_size_g + dev_hugepage_base, 0}, {hugepage_buffer_size_g - 3 * dispatch_buffer_page_size_g + dev_hugepage_base, 0}));
}

TEST_F(DeviceSingleCardFixture, TestWriteHostStall) {
    EXPECT_TRUE(local_test_functions::test_write_host(device_, 10 * dispatch_buffer_page_size_g, {dev_hugepage_base, 1}, {dev_hugepage_base, 0}, std::make_pair(dev_hugepage_base + 11 * dispatch_buffer_page_size_g, 0)));
    EXPECT_TRUE(local_test_functions::test_write_host(device_, 10 * dispatch_buffer_page_size_g, {dev_hugepage_base, 1}, {dev_hugepage_base + 5 * dispatch_buffer_page_size_g, 0}, std::make_pair(dev_hugepage_base + 11 * dispatch_buffer_page_size_g, 0)));
    EXPECT_TRUE(local_test_functions::test_write_host(device_, 10 * dispatch_buffer_page_size_g, {dev_hugepage_base + 3 * dispatch_buffer_page_size_g, 1}, {dev_hugepage_base + 3 * dispatch_buffer_page_size_g, 0}, std::make_pair(dev_hugepage_base + 3 * dispatch_buffer_page_size_g, 1)));
}

}  // namespace basic_tests
