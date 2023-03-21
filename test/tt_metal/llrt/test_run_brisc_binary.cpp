#include <chrono>
#include <fstream>
#include <errno.h>
#include <random>

#include "tt_cluster.hpp"
#include "utils.hpp"
#include "common/logger.hpp"
#include "tensix.h"

#include "llrt.hpp"

bool run_brisc(tt_cluster *cluster, int chip_id, const tt_xy_pair& core) {

    uint64_t test_mailbox_addr = l1_mem::address_map::FIRMWARE_BASE + TEST_MAILBOX_ADDRESS;
    constexpr int INIT_VALUE = 69;
    constexpr int DONE_VALUE = 1;

    std::vector<uint32_t> test_mailbox_init_val = {INIT_VALUE};
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, core, test_mailbox_init_val, test_mailbox_addr);
    log_info(tt::LogVerif, "initialized test_maxilbox");

    std::vector<uint32_t> test_mailbox_init_val_check;
    test_mailbox_init_val_check = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, test_mailbox_addr, sizeof(uint32_t));  // read a single uint32_t
    TT_ASSERT(test_mailbox_init_val_check[0] == INIT_VALUE);
    log_info(tt::LogVerif, "checked test_mailbox is correctly initialized to value = {}", test_mailbox_init_val_check[0]);

    tt::llrt::disable_ncrisc(cluster, chip_id, core);
    tt::llrt::disable_triscs(cluster, chip_id, core);

    tt::llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, tt::llrt::TensixRiscsOptions::BRISC_NCRISC, {core});
    tt::llrt::internal_::enable_cores(cluster, chip_id, {core});

    tt::llrt::deassert_brisc_reset_for_all_chips_all_cores(cluster);

    std::vector<uint32_t> test_mailbox_read_val = {0};
    bool brisc_done = false;
    // TODO: add time-out to this loop
    while(!brisc_done) {
        test_mailbox_read_val = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, test_mailbox_addr, sizeof(uint32_t));  // read a single uint32_t

        TT_ASSERT(test_mailbox_read_val[0] == INIT_VALUE || test_mailbox_read_val[0] == DONE_VALUE); // ensure no corruption

        brisc_done = test_mailbox_read_val[0] == DONE_VALUE;
        tt::llrt::internal_::assert_enable_core_mailbox_is_valid_for_core(cluster, chip_id, core);
    }
    log_info(tt::LogVerif, "brisc on core {} finished", core.str());
    log_info(tt::LogVerif, "test_mailbox_read_val = {}", test_mailbox_read_val[0]);

    // TODO: if timed out return false;
    return true;
}

int main(int argc, char** argv)
{
    bool pass = true;

    const TargetDevice target_type = TargetDevice::Silicon;
    const tt::ARCH arch = tt::ARCH::GRAYSKULL;
    const std::string sdesc_file = get_soc_description_file(arch, target_type);


    try {
        tt_device_params default_params;
        tt_cluster *cluster = new tt_cluster;
        cluster->open_device(arch, target_type, {0}, sdesc_file);
        cluster->start_device(default_params); // use default params
        tt::llrt::utils::log_current_ai_clk(cluster);

        // tt::llrt::print_worker_cores(cluster);

        // the first worker core starts at (1,1)
        pass = tt::llrt::test_load_write_read_risc_binary(cluster, "built_kernels/test_op/brisc/brisc.hex", 0, {1,1}, 0);
        if (pass) {
            pass = run_brisc(cluster, 0, {1,1});
        }

        cluster->close_device();
        delete cluster;

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(tt::LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(tt::LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(tt::LogTest, "Test Passed");
    } else {
        log_fatal(tt::LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
