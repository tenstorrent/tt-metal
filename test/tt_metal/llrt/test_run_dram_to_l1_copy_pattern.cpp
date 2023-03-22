#include "tt_cluster.hpp"
#include "utils.hpp"
#include "common/logger.hpp"
#include "tensix.h"

#include "llrt.hpp"
#include "test_libs/tiles.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "test_libs/conv_pattern.hpp"

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
        tt_xy_pair dram = {1, 0};
        std::uint32_t starting_dram_address = 0;


        SHAPE shape = {1, 16, 25, 20};
        std::vector<std::tuple<SHAPE, ConvParameters>> test_cases = {
            std::make_tuple(shape, ConvParameters(7, 7, 2, 2, 3, 3)),
            std::make_tuple(shape, ConvParameters(5, 5, 1, 1, 0, 0)),
            std::make_tuple(shape, ConvParameters(3, 7, 1, 2, 5, 3)),
            std::make_tuple(shape, ConvParameters(7, 3, 2, 5, 1, 3)),
            std::make_tuple(shape, ConvParameters(1, 1, 1, 1, 0, 0)),
            std::make_tuple(shape, ConvParameters(3, 3, 1, 1, 1, 1)),
            std::make_tuple(shape, ConvParameters(5, 5, 2, 2, 2, 2)),
            std::make_tuple(shape, ConvParameters(1, 1, 2, 2, 0, 0))
        };

        for(auto test_case: test_cases) {
            auto shape = std::get<0>(test_case);
            auto conv_params = std::get<1>(test_case);

            tt::deprecated::Tensor<std::uint32_t> tensor = tt::deprecated::initialize_tensor<std::uint32_t>(shape, tt::deprecated::Initialize::RANDOM, tt::tiles_test::get_seed_from_systime());
            std::array<std::array<uint32_t, 2>, 4> pad_size = {{{0, 0}, {0, 0}, {conv_params.PadH, conv_params.PadH}, {conv_params.PadW, conv_params.PadW}}};
            tt::deprecated::Tensor<std::uint32_t> tensor_padded = tt::deprecated::pad(tensor, pad_size);
            auto tensor_p = tt::deprecated::permute(tensor_padded, {0, 2, 3, 1}); // NHWC
            auto src_vec = tensor_p.get_values();

            // This will create the 2D matrix by modeling what dram to l1 read patterns are
            auto golden_matrix = move_act_dram_to_l1(tensor_p, conv_params);
            // This would be the actual golden that we compare the L1 data against
            auto golden_vector = flatten(golden_matrix);
            tt::llrt::write_hex_vec_to_core(cluster, chip_id, dram, src_vec, starting_dram_address);

            std::uint32_t starting_l1_address = 250 * 1024;
            double total_l1_used = (double)(starting_l1_address + golden_vector.size() * sizeof(uint32_t)) / 1024 / 1024;
            log_info(tt::LogVerif, "Total L1 used: {}MB", total_l1_used);
            TT_ASSERT(total_l1_used < 1);

            auto padded_shape = tensor_padded.get_shape();
            std::array<std::uint32_t, 4> nchw = {padded_shape[0], padded_shape[1], padded_shape[2], padded_shape[3]};
            std::vector<tt::llrt::CopyPatternSpec> specs = {
                tt::llrt::create_copy_pattern_spec(
                    core,
                    starting_l1_address,
                    dram,
                    starting_dram_address,
                    nchw,
                    {conv_params.R, conv_params.S, conv_params.U, conv_params.V}, // RSUV
                    4,
                    1,
                    load_firmware_flag
                )
            };
            bool load_blanks = load_firmware_flag;
            tt::llrt::run_copy_pattern_kernel_with_specs(cluster, chip_id, specs, load_blanks);

            vector<std::uint32_t> dst_vec = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, starting_l1_address, golden_vector.size() * sizeof(uint32_t)); // read size is in bytes

            pass &= golden_vector == dst_vec;
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
