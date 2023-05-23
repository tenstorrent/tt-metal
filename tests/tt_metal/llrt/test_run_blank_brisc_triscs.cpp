#include <chrono>
#include <fstream>
#include <errno.h>
#include <random>

#include "tt_cluster.hpp"
#include "utils.hpp"
#include "common/logger.hpp"
#include "tensix.h"

#include "llrt.hpp"

bool run_brisc_and_triscs(tt_cluster *cluster, int chip_id, const CoreCoord& core) {

   uint64_t brisc_test_mailbox_addr = MEM_TEST_MAILBOX_ADDRESS + MEM_MAILBOX_BRISC_OFFSET;

    constexpr int INIT_VALUE = 42;
    constexpr int DONE_VALUE = 1;

    std::vector<uint32_t> test_mailbox_init_val = {INIT_VALUE};
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, core, test_mailbox_init_val, brisc_test_mailbox_addr);
    log_info(tt::LogVerif, "initialized test_maxilbox");

    std::vector<uint32_t> test_mailbox_init_val_check;
    test_mailbox_init_val_check = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, brisc_test_mailbox_addr, sizeof(uint32_t));  // read a single uint32_t
    TT_ASSERT(test_mailbox_init_val_check[0] == INIT_VALUE);
    log_info(tt::LogVerif, "checked brisc_test_mailbox is correctly initialized to value = {}", test_mailbox_init_val_check[0]);

    for (int trisc_id = 0; trisc_id <= 2; trisc_id++) {
        std::vector<uint32_t> test_mailbox_init_val = {INIT_VALUE};
        tt::llrt::write_hex_vec_to_core(cluster, chip_id, core, test_mailbox_init_val, trisc_mailbox_addresses[trisc_id]);
        log_info(tt::LogVerif, "initialized test_maxilbox for trisc_id = {}", trisc_id);

        std::vector<uint32_t> test_mailbox_init_val_check;
        test_mailbox_init_val_check = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, trisc_mailbox_addresses[trisc_id], sizeof(uint32_t));  // read a single uint32_t
        TT_ASSERT(test_mailbox_init_val_check[0] == INIT_VALUE);
        log_info(tt::LogVerif, "checked test_mailbox for trisc{} is correctly initialized to value = {}", trisc_id, test_mailbox_init_val_check[0]);
    }

    tt::llrt::disable_ncrisc(cluster, chip_id, core);

    tt::llrt::enable_triscs(cluster, chip_id, core);
    //tt::llrt::disable_triscs(cluster, chip_id, core); // use this to make the test hang

    tt::llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, tt::llrt::TensixRiscsOptions::BRISC_NCRISC, {core});
    tt::llrt::internal_::enable_cores(cluster, chip_id, {core});

    tt::llrt::deassert_brisc_reset_for_all_chips_all_cores(cluster);

    std::vector<uint32_t> test_mailbox_read_val = {0};
    bool brisc_done = false;
    bool triscs_done = false;
    // TODO: add time-out to this loop
    while(!brisc_done || !triscs_done) {
        test_mailbox_read_val = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, brisc_test_mailbox_addr, sizeof(uint32_t));  // read a single uint32_t
        TT_ASSERT(test_mailbox_read_val[0] == INIT_VALUE || test_mailbox_read_val[0] == DONE_VALUE); // ensure no corruption
        brisc_done = test_mailbox_read_val[0] == DONE_VALUE;

        int trisc_id = 0;
        for (trisc_id = 0; trisc_id <= 2; trisc_id++) {
            std::vector<uint32_t> test_trisc_mailbox_read_val = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, trisc_mailbox_addresses[trisc_id], sizeof(uint32_t));  // read a single uint32_t
            if (test_trisc_mailbox_read_val[0] != DONE_VALUE) {
                break;
            }
            log_info(tt::LogVerif, "trisc_id = {} is done, read mailbox value of {}", trisc_id, test_trisc_mailbox_read_val[0]);
        }
        triscs_done = trisc_id == 3;

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

    std::vector<std::string> input_args(argv, argv + argc);
    string arch_name = "";
    try {
        std::tie(arch_name, input_args) =
            test_args::get_command_option_and_remaining_args(input_args, "--arch", "grayskull");
    } catch (const std::exception& e) {
        log_fatal(tt::LogTest, "Command line arguments found exception", e.what());
    }

    const TargetDevice target_type = TargetDevice::Silicon;
    const tt::ARCH arch = tt::get_arch_from_string(arch_name);
    const std::string sdesc_file = get_soc_description_file(arch, target_type);


    try {
        tt_device_params default_params;
        tt_cluster *cluster = new tt_cluster;
        cluster->open_device(arch, target_type, {0}, sdesc_file);
        cluster->start_device(default_params); // use default params
        tt::llrt::utils::log_current_ai_clk(cluster);

        // tt::llrt::print_worker_cores(cluster);

        // the first worker core starts at (1,1)
        string op_path = "built_kernels/blank_op";
        pass = tt::llrt::test_load_write_read_risc_binary(cluster, op_path + "/brisc/brisc.hex", 0, {1,1}, 0);
        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread0/tensix_thread0.hex", 0, {1,1}, 0);
        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread1/tensix_thread1.hex", 0, {1,1}, 1);
        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread2/tensix_thread2.hex", 0, {1,1}, 2);

        if (pass) {
            pass = run_brisc_and_triscs(cluster, 0, {1,1});
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
