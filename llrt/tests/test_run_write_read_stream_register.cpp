#include <chrono>
#include <fstream>
#include <errno.h>
#include <random>

#include "tt_cluster.hpp"
#include "utils.hpp"
#include "common/logger.hpp"
#include "tensix.h"

#include "llrt.hpp"
#include "test_libs/tiles.hpp"
#include "hostdevcommon/registers.hpp"

std::vector<uint32_t> run_stream_register_write(tt_cluster *cluster, int chip_id, const tt_xy_pair& src_core, const tt_xy_pair& dst_core, const std::uint32_t buffer_dst_addr, std::vector<uint32_t>& src_vec) {
    tt::llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, tt::llrt::TensixRiscsOptions::BRISC_NCRISC, {});
    const int ncrisc_id = 1;
    if (!tt::llrt::test_load_write_read_risc_binary(cluster, "built_kernels/l1_to_dram_copy/ncrisc/ncrisc.hex", chip_id, src_core, ncrisc_id)) {
        TT_ASSERT(false);
    }

    // Kernel arguments
    std::uint32_t buffer_src_addr = 200 * 1024;
    std::uint32_t buffer_size = 4;
    // blast kernel arguments to L1 in one-shot
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, src_core,
        {buffer_dst_addr,
        (std::uint32_t)dst_core.x,
        (std::uint32_t)dst_core.y,
        buffer_src_addr,
        buffer_size
        }, NCRISC_L1_ARG_BASE);

    tt::llrt::write_hex_vec_to_core(cluster, chip_id, src_core, src_vec, buffer_src_addr);

    const tt::llrt::TensixRiscsOptions riscs_options = tt::llrt::TensixRiscsOptions::BRISC_NCRISC;
    tt::llrt::internal_::setup_riscs_on_specified_cores(cluster, chip_id, riscs_options, {src_core});
    tt::llrt::internal_::run_riscs_on_specified_cores(cluster, chip_id, riscs_options, {src_core});
    vector<std::uint32_t> dst_vec = tt::llrt::read_hex_vec_from_core(cluster, chip_id, dst_core, buffer_dst_addr, buffer_size);  // read L1
    return dst_vec;
}

std::vector<uint32_t> run_stream_register_read(tt_cluster *cluster, int chip_id, const tt_xy_pair& src_core, const tt_xy_pair& dst_core, const std::uint32_t buffer_src_addr) {
    tt::llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, tt::llrt::TensixRiscsOptions::BRISC_ONLY, {});
    const int brisc_id = 0;
    if (!tt::llrt::test_load_write_read_risc_binary(cluster, "built_kernels/dram_to_l1_copy/brisc/brisc.hex", chip_id, dst_core, brisc_id)) {
        TT_ASSERT(false);
    }

    // Kernel arguments
    std::uint32_t buffer_dst_addr = 300 * 1024;
    std::uint32_t buffer_size = 4;
    // blast kernel arguments to L1 in one-shot
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, dst_core,
        {buffer_src_addr,
        (std::uint32_t)src_core.x,
        (std::uint32_t)src_core.y,
        buffer_dst_addr,
        buffer_size
        }, BRISC_L1_ARG_BASE);

    const tt::llrt::TensixRiscsOptions riscs_options = tt::llrt::TensixRiscsOptions::BRISC_ONLY;
    tt::llrt::internal_::setup_riscs_on_specified_cores(cluster, chip_id, riscs_options, {dst_core});
    tt::llrt::internal_::run_riscs_on_specified_cores(cluster, chip_id, riscs_options, {dst_core});
    vector<std::uint32_t> dst_vec = tt::llrt::read_hex_vec_from_core(cluster, chip_id, dst_core, buffer_dst_addr, buffer_size);  // read L1
    return dst_vec;
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

        for(std::uint32_t stream_id = 0; stream_id < NUM_STREAMS; stream_id++) {
            for(std::uint32_t register_id: REGISTERS) {
                std::uint32_t stream_register_address = STREAM_REG_ADDR(stream_id, register_id);
                std::cout<<stream_id<<", "<<register_id<<std::endl;
                for(std::uint32_t i = 0; i < 10; i++) {
                    std::vector<uint32_t> src_vec = {i};
                    // we can only write/read 16B aligned
                    auto dst_vec = run_stream_register_write(cluster, 0, {11,3}, {1,1}, stream_register_address, src_vec);
                    auto dst_vec1 = run_stream_register_read(cluster, 0, {1,1}, {11, 3}, stream_register_address);
                    pass &= (src_vec == dst_vec && src_vec == dst_vec1);
                }
            }
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
