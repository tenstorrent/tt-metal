#include <chrono>

#include "tt_cluster.hpp"
#include "utils.hpp"
#include "common/logger.hpp"
#include "tensix.h"

#include "llrt.hpp"
#include "test_libs/tiles.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "common/bfloat16.hpp"
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
        int dram_channel_id = 0;
        tt_xy_pair dram = tt::llrt::get_core_for_dram_channel(cluster, dram_channel_id);
        std::uint32_t starting_dram_address = 0;


        std::vector<uint32_t> range_C = { 16, 32, 64, 128, 256, 512, 1024};
        std::vector<uint32_t> range_H = { 5};
        std::vector<uint32_t> range_R = { 3};
        std::vector<uint32_t> range_U = { 1};
        std::vector<uint32_t> range_Pad = { 1};
        std::uint32_t num_repetitions = 100000;
        std::vector<std::tuple<SHAPE, ConvParameters>> test_cases;
        for(auto c : range_C) {
            for(auto h: range_H) {
                for(auto r: range_R) {
                    for(auto u: range_U) {
                        for(auto pad: range_Pad) {
                            SHAPE shape = {1, c, h, h};
                            test_cases.push_back(std::make_tuple(shape, ConvParameters(r, r, u, u, pad, pad)));
                        }
                    }
                }
            }
        }

        // std::ofstream results_file;
        // results_file.open ("sweep.csv");
        // results_file<<"N, C, H, W, R, S, U, V, PadH, PadW, Total L1 (MB), Read BW (GB/sec), L1 to Dram (sec)\n";
        for(auto test_case: test_cases) {
            auto shape = std::get<0>(test_case);
            auto conv_params = std::get<1>(test_case);

            tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, tt::tiles_test::get_seed_from_systime());
            std::array<std::array<uint32_t, 2>, 4> pad_size = {{{0, 0}, {0, 0}, {conv_params.PadH, conv_params.PadH}, {conv_params.PadW, conv_params.PadW}}};
            tt::deprecated::Tensor<bfloat16> tensor_padded = tt::deprecated::pad(tensor, pad_size, bfloat16((std::uint32_t)0));
            auto tensor_p = tt::deprecated::permute(tensor_padded, {0, 2, 3, 1}); // NHWC
            // This will create the 2D matrix by modeling what dram to l1 read patterns are
            auto golden_matrix = move_act_dram_to_l1(tensor_p, conv_params);
            // This would be the actual golden that we compare the L1 data against
            auto golden_vector = flatten(golden_matrix);

            auto src_vec = pack_bfloat16_vec_into_uint32_vec(tensor_p.get_values());

            tt::llrt::write_hex_vec_to_core(cluster, chip_id, dram, src_vec, starting_dram_address);

            std::uint32_t starting_l1_address = 250 * 1024;
            double total_l1_used = (double)(starting_l1_address + golden_vector.size() * 2) / 1024 / 1024;
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
                    2,
                    num_repetitions,
                    load_firmware_flag
                )
            };
            bool load_blanks = load_firmware_flag;

            auto start = std::chrono::steady_clock::now();
            tt::llrt::run_copy_pattern_kernel_with_specs(cluster, chip_id, specs, load_blanks);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> dram_to_l1_seconds = end-start;
            log_info(tt::LogVerif, "Dram To L1 time: {}s", dram_to_l1_seconds.count());
            std::uint32_t total_buffer_size = golden_vector.size() * 2;

            uint64_t total_bytes = (uint64_t)total_buffer_size * num_repetitions;
            double total_GB = (double)total_bytes / (1024*1024*1024);
            double read_bw_gbps = total_GB/dram_to_l1_seconds.count();
            log_info(tt::LogVerif, "Bytes read: {}, GB read: {}", total_bytes, total_GB);
            log_info(tt::LogVerif, "Read speed GB/s: {}", read_bw_gbps);

            std::uint32_t dst_dram_address = total_buffer_size;
            std::vector<tt::llrt::L1ToDramCopySpec> l1_to_dram_specs = {tt::llrt::create_l1_to_dram_copy_spec(core, dram_channel_id, total_buffer_size, dst_dram_address, starting_l1_address, load_firmware_flag)};
            start = std::chrono::steady_clock::now();
            tt::llrt::run_l1_to_dram_copy_kernel_with_specs(cluster, chip_id, l1_to_dram_specs, load_blanks);
            end = std::chrono::steady_clock::now();
            std::chrono::duration<double> l1_to_dram_seconds = end-start;
            log_info(tt::LogVerif, "L1 to Dram time: {}s", l1_to_dram_seconds.count());

            // results_file<<shape[0]<<","<<shape[1]<<","<<shape[2]<<","<<shape[3]<<","<<conv_params.R<<","<<conv_params.S<<","<<conv_params.U<<","<<conv_params.V<<","<<conv_params.PadH<<","<<conv_params.PadW<<","<<total_l1_used<<","<<read_bw_gbps<<","<<l1_to_dram_seconds.count()<<std::endl;

            std::vector<std::uint32_t> dst_vec_dram = tt::llrt::read_hex_vec_from_core(cluster, chip_id, dram, dst_dram_address, total_buffer_size);
            auto dst_vec = unpack_uint32_vec_into_bfloat16_vec(dst_vec_dram);
            TT_ASSERT(golden_vector.size() == dst_vec.size());

            pass &= golden_vector == dst_vec;
            log_info(tt::LogVerif, "Test passed {}", golden_vector == dst_vec);
            log_info(tt::LogVerif, "---------------------------------------------");
        }
        log_info(tt::LogVerif, "Finished sweep");
        // results_file.close();
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
