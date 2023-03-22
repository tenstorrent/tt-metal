#include "tt_cluster.hpp"
#include "utils.hpp"
#include "common/logger.hpp"
#include "tensix.h"

#include "llrt.hpp"
#include "test_libs/tiles.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "test_libs/conv_pattern.hpp"
#include "hostdevcommon/registers.hpp"

int main(int argc, char** argv)
{
    bool pass = true;

    const TargetDevice target_type = TargetDevice::Silicon;
    const tt::ARCH arch = tt::ARCH::GRAYSKULL;
    const std::string sdesc_file = get_soc_description_file(arch, target_type);

    try {
        tt_device_params default_params;
        tt_cluster *cluster = new tt_cluster;
        const int chip_id = 0;
        cluster->open_device(arch, target_type, {chip_id}, sdesc_file);
        cluster->start_device(default_params); // use default params
        tt::llrt::utils::log_current_ai_clk(cluster);
        tt::llrt::LoadFirmwareFlag load_firmware_flag = true;
        tt_xy_pair core = {11, 3};

        for(std::uint32_t stream_id = 0; stream_id < NUM_STREAMS; stream_id ++) {
            for(std::uint32_t register_id: REGISTERS) {
                for(std::uint32_t i = 0; i < 10; i++) {
                    std::vector<uint32_t> src_vec = {i};
                    std::uint32_t stream_register_address = STREAM_REG_ADDR(stream_id, register_id);
                    tt::llrt::write_hex_vec_to_core(cluster, chip_id, core, src_vec, stream_register_address);
                    vector<std::uint32_t> dst_vec = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, stream_register_address, src_vec.size() * sizeof(uint32_t)); // read size is in bytes
                    if(src_vec != dst_vec) {
                        std::cout<<"Failed["<<stream_id<<","<<register_id<<"]: "<<src_vec[0]<<", "<<dst_vec[0]<<std::endl;
                    }
                    pass &= (src_vec == dst_vec);
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
