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

struct dram_copy_kernel_args {
    int dram_src_channel_id;
    std::uint32_t dram_src_buffer_addr;
    tt_xy_pair dram_src_noc_xy;

    int dram_dst_channel_id;
    std::uint32_t dram_dst_buffer_addr = 512 * 1024;
    tt_xy_pair dram_dst_noc_xy;

    std::uint32_t dram_buffer_size;
    std::uint32_t l1_buffer_addr;
    std::uint32_t arg_base_addr;
};

void write_dram_copy_kernel_args(tt_cluster* cluster, int chip_id, tt_xy_pair core, const dram_copy_kernel_args& kernel_args) {
    // blast dram copy kernel arguments to L1 in one-shot
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, core,
        {kernel_args.l1_buffer_addr,

        kernel_args.dram_src_buffer_addr,
        (std::uint32_t)kernel_args.dram_src_noc_xy.x,
        (std::uint32_t)kernel_args.dram_src_noc_xy.y,

        kernel_args.dram_dst_buffer_addr,
       (std::uint32_t) kernel_args.dram_dst_noc_xy.x,
       (std::uint32_t) kernel_args.dram_dst_noc_xy.y,

        kernel_args.dram_buffer_size,
        },
        kernel_args.arg_base_addr);
}

bool run_dram_copy_brisc_ncrisc(tt_cluster *cluster, int chip_id, const tt_xy_pair& core) {
    const tt::llrt::TensixRiscsOptions riscs_options = tt::llrt::TensixRiscsOptions::BRISC_NCRISC;

    tt::llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, tt::llrt::TensixRiscsOptions::BRISC_NCRISC, {core});
    tt::llrt::internal_::setup_riscs_on_specified_core(cluster, chip_id, riscs_options, core);

    // channel 0
    dram_copy_kernel_args brisc_kernel_args = {
        .dram_src_channel_id = 0,
        .dram_src_buffer_addr = 0,
        .dram_src_noc_xy = tt::llrt::get_core_for_dram_channel(cluster,0),

        .dram_dst_channel_id = 0,
        .dram_dst_buffer_addr = 512*1024,
        .dram_dst_noc_xy = tt::llrt::get_core_for_dram_channel(cluster,0),

        .dram_buffer_size = 100 * 1024,
        .l1_buffer_addr = 200 * 1024, // addr=200KB
        .arg_base_addr = BRISC_L1_ARG_BASE
    };

    log_info(tt::LogVerif, "brisc dram_src_noc_xy = {}", brisc_kernel_args.dram_src_noc_xy.str());
    log_info(tt::LogVerif, "brisc dram_dst_noc_xy = {}", brisc_kernel_args.dram_dst_noc_xy.str());

    // channel 1
    dram_copy_kernel_args ncrisc_kernel_args = {
        .dram_src_channel_id = 1,
        .dram_src_buffer_addr = 0,
        .dram_src_noc_xy = tt::llrt::get_core_for_dram_channel(cluster,1),

        .dram_dst_channel_id = 1,
        .dram_dst_buffer_addr = 512*1024,
        .dram_dst_noc_xy = tt::llrt::get_core_for_dram_channel(cluster,1),

        .dram_buffer_size = 100 * 1024,
        .l1_buffer_addr = 300 * 1024, // addr=300KB
        .arg_base_addr = NCRISC_L1_ARG_BASE
    };

    log_info(tt::LogVerif, "ncrisc dram_src_noc_xy = {}", ncrisc_kernel_args.dram_src_noc_xy.str());
    log_info(tt::LogVerif, "ncrisc dram_dst_noc_xy = {}", ncrisc_kernel_args.dram_dst_noc_xy.str());


    write_dram_copy_kernel_args(cluster, chip_id, core, brisc_kernel_args);
    write_dram_copy_kernel_args(cluster, chip_id, core, ncrisc_kernel_args);

    TT_ASSERT(ncrisc_kernel_args.dram_buffer_size == brisc_kernel_args.dram_buffer_size);
    TT_ASSERT(ncrisc_kernel_args.dram_buffer_size % sizeof(std::uint32_t) == 0);

    int brisc_init_value_for_buffer = tt_rnd_int(0,100000);
    log_info(tt::LogVerif, "brisc_init_value_for_buffer = {}", brisc_init_value_for_buffer);
    std::vector<std::uint32_t> brisc_src_vec(brisc_kernel_args.dram_buffer_size/sizeof(std::uint32_t), brisc_init_value_for_buffer);
    // blast the BRISC's src buffer to DRAM
    cluster->write_dram_vec(brisc_src_vec, tt_target_dram{chip_id, brisc_kernel_args.dram_src_channel_id, 0}, brisc_kernel_args.dram_src_buffer_addr); // write to address

    int ncrisc_init_value_for_buffer = tt_rnd_int(0,100000);
    log_info(tt::LogVerif, "ncrisc_init_value_for_buffer = {}", ncrisc_init_value_for_buffer);
    std::vector<std::uint32_t> ncrisc_src_vec(ncrisc_kernel_args.dram_buffer_size/sizeof(std::uint32_t), ncrisc_init_value_for_buffer);
    // blast the NCRISC's src buffer to DRAM
    cluster->write_dram_vec(ncrisc_src_vec, tt_target_dram{chip_id, ncrisc_kernel_args.dram_src_channel_id, 0}, ncrisc_kernel_args.dram_src_buffer_addr); // write to address

    tt::llrt::deassert_brisc_reset_for_all_chips_all_cores(cluster);

    bool riscs_are_done = false;
    // TODO: add time-out to this loop
    while(!riscs_are_done) {
        riscs_are_done = true;
        riscs_are_done &= tt::llrt::internal_::check_if_riscs_on_specified_core_done(cluster, chip_id, tt::llrt::TensixRiscsOptions::BRISC_NCRISC, core);
    }
    log_info(tt::LogVerif, "brisc/ncrisc on core {} finished", core.str());

    // read the dst buffer from DRAM
    vector<std::uint32_t> brisc_dst_vec;
    cluster->read_dram_vec(brisc_dst_vec, tt_target_dram{chip_id, brisc_kernel_args.dram_dst_channel_id, 0}, brisc_kernel_args.dram_dst_buffer_addr, brisc_kernel_args.dram_buffer_size); // read size is in bytes
    vector<std::uint32_t> ncrisc_dst_vec;
    cluster->read_dram_vec(ncrisc_dst_vec, tt_target_dram{chip_id, ncrisc_kernel_args.dram_dst_channel_id, 0}, ncrisc_kernel_args.dram_dst_buffer_addr, ncrisc_kernel_args.dram_buffer_size); // read size is in bytes

    tt::llrt::internal_::cleanup_risc_on_specified_core(cluster, chip_id, riscs_options, core);
    tt::llrt::assert_reset_for_all_chips(cluster);

    return brisc_src_vec == brisc_dst_vec && ncrisc_src_vec == ncrisc_dst_vec;
}

int main(int argc, char** argv)
{
    bool pass = true;

    const TargetDevice target_type = TargetDevice::Silicon;
    const tt::ARCH arch = tt::ARCH::GRAYSKULL;
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
        // load BRISC FW (this also load NCRISC from L1 to IRAM and also deasserts reset for NCRISC)
        pass = tt::llrt::test_load_write_read_risc_binary(cluster, "built_kernels/dram_copy_brisc_ncrisc/brisc/brisc.hex", 0, {3,10}, 0);
        // load NCRISC FW
        pass = pass & tt::llrt::test_load_write_read_risc_binary(cluster, "built_kernels/dram_copy_brisc_ncrisc/ncrisc/ncrisc.hex", 0, {3,10}, 1);

        if (pass) {
            pass = run_dram_copy_brisc_ncrisc(cluster, 0, {3,10});
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
        TT_ASSERT(pass);
    }

    return 0;
}
