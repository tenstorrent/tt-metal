#include <chrono>
#include <fstream>
#include <errno.h>
#include <random>

#include "tt_cluster.hpp"
#include "utils.hpp"
#include "common/logger.hpp"
#include "tensix.h"

#include "llrt.hpp"

std::mt19937 rand_gen(0);

int tt_rnd_int(int min, int max) {
    tt::log_assert((min <= max), "min is greater than max!");
    std::uniform_int_distribution<int> distrib(min, max);
    return distrib(rand_gen);
}

void tt_rnd_set_seed(int seed) {
    log_info(tt::LogVerif, "Setting Test Seed = {}", (uint32_t)seed);
    rand_gen.seed(seed);
}

bool run_dram_copy_ncrisc(tt_cluster *cluster, int chip_id, const tt_xy_pair& core) {

    uint64_t test_mailbox_addr_brisc = MEM_BRISC_FIRMWARE_BASE + MEM_TEST_MAILBOX_ADDRESS;
    uint64_t test_mailbox_addr_ncrisc = MEM_NCRISC_FIRMWARE_BASE + MEM_TEST_MAILBOX_ADDRESS;

    constexpr int INIT_VALUE = 69;
    constexpr int DONE_VALUE = 1;
    std::vector<uint32_t> test_mailbox_init_val = {INIT_VALUE};
    std::vector<uint32_t> test_mailbox_init_val_check;

    tt::llrt::write_hex_vec_to_core(cluster, chip_id, core, test_mailbox_init_val, test_mailbox_addr_brisc);
    log_info(tt::LogVerif, "initialized test_maxilbox for brisc");

    test_mailbox_init_val_check = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, test_mailbox_addr_brisc, sizeof(uint32_t));  // read a single uint32_t
    TT_ASSERT(test_mailbox_init_val_check[0] == INIT_VALUE);
    log_info(tt::LogVerif, "checked test_mailbox for brisc is correctly initialized to value = {}", test_mailbox_init_val_check[0]);

    tt::llrt::write_hex_vec_to_core(cluster, chip_id, core, test_mailbox_init_val, test_mailbox_addr_ncrisc);
    log_info(tt::LogVerif, "initialized test_maxilbox for ncrisc");

    test_mailbox_init_val_check = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, test_mailbox_addr_ncrisc, sizeof(uint32_t));  // read a single uint32_t
    TT_ASSERT(test_mailbox_init_val_check[0] == INIT_VALUE);
    log_info(tt::LogVerif, "checked test_mailbox for ncrisc is correctly initialized to value = {}", test_mailbox_init_val_check[0]);

    tt::llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, tt::llrt::TensixRiscsOptions::BRISC_NCRISC, {core});
    tt::llrt::internal_::enable_cores(cluster, chip_id, {core});

    tt::llrt::enable_ncrisc(cluster, chip_id, core);
    tt::llrt::disable_triscs(cluster, chip_id, core);

    int dram_src_channel_id = 0;
    int dram_dst_channel_id = 0;

    // Kernel arguments
    std::uint32_t l1_buffer_addr = 200 * 1024;

    std::uint32_t dram_buffer_src_addr = 0;
    tt_xy_pair dram_src_noc_xy = tt::llrt::get_core_for_dram_channel(cluster, dram_src_channel_id);
    log_info(tt::LogVerif, "dram_src_noc_xy = {}", dram_src_noc_xy.str());

    std::uint32_t dram_buffer_dst_addr = 512 * 1024;
    tt_xy_pair dram_dst_noc_xy = tt::llrt::get_core_for_dram_channel(cluster, dram_dst_channel_id);
    log_info(tt::LogVerif, "dram_dst_noc_xy = {}", dram_dst_noc_xy.str());

    std::uint32_t dram_buffer_size = 100 * 1024;

    // blast ncrisc kernel arguments to L1 in one-shot
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, core,
        {l1_buffer_addr,

        dram_buffer_src_addr,
        (std::uint32_t)dram_src_noc_xy.x,
        (std::uint32_t)dram_src_noc_xy.y,

        dram_buffer_dst_addr,
       (std::uint32_t) dram_dst_noc_xy.x,
       (std::uint32_t) dram_dst_noc_xy.y,

        dram_buffer_size
        }, NCRISC_L1_ARG_BASE);

    // blast the src buffer to DRAM
    TT_ASSERT(dram_buffer_size % sizeof(std::uint32_t) == 0);
    int init_value_for_buffer = tt_rnd_int(0,100000);
    log_info(tt::LogVerif, "init_value_for_buffer = {}", init_value_for_buffer);
    std::vector<std::uint32_t> src_vec(dram_buffer_size/sizeof(std::uint32_t), init_value_for_buffer);
    cluster->write_dram_vec(src_vec, tt_target_dram{chip_id, dram_src_channel_id, 0}, dram_buffer_src_addr); // write to address

    tt::llrt::deassert_brisc_reset_for_all_chips_all_cores(cluster);

    std::vector<uint32_t> test_mailbox_read_val_brisc = {0};
    std::vector<uint32_t> test_mailbox_read_val_ncrisc = {0};
    bool brisc_done = false;
    bool ncrisc_done = false;

    // TODO: add time-out to this loop
    while(!brisc_done || !ncrisc_done) {
        test_mailbox_read_val_brisc = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, test_mailbox_addr_brisc, sizeof(uint32_t));  // read a single uint32_t
        test_mailbox_read_val_ncrisc = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, test_mailbox_addr_ncrisc, sizeof(uint32_t));  // read a single uint32_t

        TT_ASSERT(test_mailbox_read_val_brisc[0] == INIT_VALUE || test_mailbox_read_val_brisc[0] == DONE_VALUE); // ensure no corruption
        TT_ASSERT(test_mailbox_read_val_ncrisc[0] == INIT_VALUE || test_mailbox_read_val_ncrisc[0] == DONE_VALUE); // ensure no corruption

        brisc_done = test_mailbox_read_val_brisc[0] == DONE_VALUE;
        ncrisc_done = test_mailbox_read_val_ncrisc[0] == DONE_VALUE;

        tt::llrt::internal_::assert_enable_core_mailbox_is_valid_for_core(cluster, chip_id, core);
    }
    log_info(tt::LogVerif, "brisc/ncrisc on core {} finished", core.str());
    log_info(tt::LogVerif, "test_mailbox_read_val_brisc = {}",  test_mailbox_read_val_brisc[0]);
    log_info(tt::LogVerif, "test_mailbox_read_val_ncrisc = {}", test_mailbox_read_val_ncrisc[0]);

    // read the dst buffer from DRAM
    vector<std::uint32_t> dst_vec;
    cluster->read_dram_vec(dst_vec, tt_target_dram{chip_id, dram_dst_channel_id, 0}, dram_buffer_dst_addr, dram_buffer_size); // read size is in bytes

    return src_vec == dst_vec;
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


    tt_rnd_set_seed(std::random_device()());

    try {
        tt_device_params default_params;
        tt_cluster *cluster = new tt_cluster;
        cluster->open_device(arch, target_type, {0}, sdesc_file);
        cluster->start_device(default_params); // use default params
        tt::llrt::utils::log_current_ai_clk(cluster);

        // tt::llrt::print_worker_cores(cluster);

        // the first worker core starts at (1,1)
        // need to load BRISC FW + blank brisc kernel because BRISC FW loads NCRISC from L1 to NCRISC IRAM and also deasserts reset for NCRISC
        pass = tt::llrt::test_load_write_read_risc_binary(cluster, "built_kernels/dram_copy_ncrisc/brisc/brisc.hex", 0, {5,10}, 0);
        // load NCRISC FW
        pass = pass & tt::llrt::test_load_write_read_risc_binary(cluster, "built_kernels/dram_copy_ncrisc/ncrisc/ncrisc.hex", 0, {5,10}, 1);

        if (pass) {
            pass = run_dram_copy_ncrisc(cluster, 0, {5,10});
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
