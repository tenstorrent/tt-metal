#include "tt_cluster.hpp"
#include "utils.hpp"
#include "common/logger.hpp"
#include "tensix.h"

#include "llrt.hpp"
#include "test_libs/tiles.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "test_libs/conv_pattern.hpp"
#include "common/bfloat16.hpp"

using tt::llrt::CircularBufferConfigVec;

bool run_copy_pattern_multi_tile(
    tt_cluster* cluster,
    int chip_id,
    const tt_xy_pair& core,
    std::uint32_t num_tiles_r,
    std::uint32_t num_tiles_c,
    std::vector<uint32_t>& src_vec,
    std::vector<bfloat16>& expected_dst_vec,
    std::array<std::uint32_t, 4>& nchw,
    std::array<std::uint32_t, 4>& rsuv
) {

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_input_tiles = num_tiles_r * num_tiles_c;
    uint32_t num_output_tiles = num_input_tiles;
    uint32_t dram_buffer_size = single_tile_size * num_output_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_src_addr = 0;
    uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
    int dram_src_channel_id = 0;
    int dram_dst_channel_id = 0;
    log_info(tt::LogVerif, "num_tiles_r = {}", num_tiles_r);
    log_info(tt::LogVerif, "num_tiles_c = {}", num_tiles_c);
    log_info(tt::LogVerif, "single_tile_size = {} B", single_tile_size);
    log_info(tt::LogVerif, "dram_bufer_size = {} B", dram_buffer_size);

    tt::llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, {core});

    tt_xy_pair dram_src_noc_xy = tt::llrt::get_core_for_dram_channel(cluster, dram_src_channel_id);
    log_info(tt::LogVerif, "dram_src_noc_xy = {}", dram_src_noc_xy.str());
    tt_xy_pair dram_dst_noc_xy = tt::llrt::get_core_for_dram_channel(cluster, dram_dst_channel_id);
    log_info(tt::LogVerif, "dram_dst_noc_xy = {}", dram_dst_noc_xy.str());

    // BufferConfigVec -- common across all kernels, so written once to the core
    CircularBufferConfigVec circular_buffer_config_vec = tt::llrt::create_circular_buffer_config_vector();
    tt::llrt::set_config_for_circular_buffer(circular_buffer_config_vec, 0, 200*1024, single_tile_size * num_tiles_c, num_tiles_c); // input buf
    tt::llrt::set_config_for_circular_buffer(circular_buffer_config_vec, 16, 300*1024, single_tile_size, 1); // output buf (starts at index 16)
    TT_ASSERT(200 * 1024 + single_tile_size * num_tiles_c < 300 * 1024);
    // buffer_config_vec written in one-shot
    tt::llrt::write_circular_buffer_config_vector_to_core(cluster, chip_id, core, circular_buffer_config_vec);

    // NCRISC kernel arguments to L1 in one-shot
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, core,
        {
            dram_buffer_src_addr,
            (std::uint32_t)dram_src_noc_xy.x,
            (std::uint32_t)dram_src_noc_xy.y,
            nchw[0], //N
            nchw[1], //C
            nchw[2], //H
            nchw[3], //W
            rsuv[0], //R
            rsuv[1], //S
            rsuv[2], //U (vertical stride)
            rsuv[3], //V (horizontal stride)
            nchw[3] * nchw[1] * 2, // C * W * 2B
            nchw[1] * 2, // C * 2B
            num_tiles_c,
            num_tiles_c * 1024 * 2,
            1
        },
        NCRISC_L1_ARG_BASE);

    // BRISC kernel arguments to L1 in one-shot
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, core,
        { dram_buffer_dst_addr, (std::uint32_t)dram_dst_noc_xy.x, (std::uint32_t)dram_dst_noc_xy.y, (std::uint32_t)(num_tiles_r * num_tiles_c) },
        BRISC_L1_ARG_BASE);

    TT_ASSERT(dram_buffer_size % sizeof(std::uint32_t) == 0);

    // Write tiles sequentially to DRAM
    cluster->write_dram_vec(src_vec, tt_target_dram{chip_id, dram_src_channel_id, 0}, dram_buffer_src_addr); // write to address

    tt::llrt::internal_::setup_riscs_on_specified_cores(cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, {core});
    tt::llrt::internal_::run_riscs_on_specified_cores(cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, {core});

    std::vector<std::uint32_t> dst_vec_dram;
    cluster->read_dram_vec(dst_vec_dram, tt_target_dram{chip_id, dram_dst_channel_id, 0}, dram_buffer_dst_addr, dram_buffer_size);
    auto dst_vec = unpack_uint32_vec_into_bfloat16_vec(dst_vec_dram);

    bool pass = (dst_vec == expected_dst_vec);

    return pass;
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
        const int chip_id = 0;
        tt_xy_pair core = {1, 1};
        cluster->open_device(arch, target_type, {chip_id}, sdesc_file);
        cluster->start_device(default_params); // use default params
        tt::llrt::utils::log_current_ai_clk(cluster);
        string op_path = "built_kernels/copy_pattern_tilized";
        pass = tt::llrt::test_load_write_read_risc_binary(cluster, op_path + "/brisc/brisc.hex", 0, core, 0); // brisc
        pass = pass & tt::llrt::test_load_write_read_risc_binary(cluster, op_path + "/ncrisc/ncrisc.hex", 0, core, 1); // ncrisc

        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread0/tensix_thread0.hex", 0, core, 0); // trisc0
        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread1/tensix_thread1.hex", 0, core, 1); // trisc1
        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread2/tensix_thread2.hex", 0, core, 2); // trisc2



        SHAPE shape = {1, 32, 16, 4};
        auto conv_params = ConvParameters(3, 3, 1, 1, 1, 1);

        tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, tt::tiles_test::get_seed_from_systime()); // TODO: make randomized!
        std::array<std::array<uint32_t, 2>, 4> pad_size = {{{0, 0}, {0, 0}, {conv_params.PadH, conv_params.PadH}, {conv_params.PadW, conv_params.PadW}}};
        tt::deprecated::Tensor<bfloat16> tensor_padded = tt::deprecated::pad(tensor, pad_size, bfloat16((std::uint32_t)0));
        auto tensor_p = tt::deprecated::permute(tensor_padded, {0, 2, 3, 1}); // NHWC

        // This will create the 2D matrix by modeling what dram to l1 read patterns are
        auto golden_matrix = move_act_dram_to_l1(tensor_p, conv_params);
        // This would be the actual golden that we compare the L1 data against
        // auto golden_vector = flatten(golden_matrix);
        auto golden_vector = tilize_2d_matrix(golden_matrix);
        auto src_vec = pack_bfloat16_vec_into_uint32_vec(tensor_p.get_values());

        auto padded_shape = tensor_padded.get_shape();
        std::array<std::uint32_t, 4> nchw = {padded_shape[0], padded_shape[1], padded_shape[2], padded_shape[3]};
        std::array<std::uint32_t, 4> rsuv = {conv_params.R, conv_params.S, conv_params.U, conv_params.V};
        std::uint32_t num_tiles_c = rsuv[0] * rsuv[1];
        std::uint32_t num_tiles_r = (shape[3] * shape[2] * shape[0]) / 32;
        pass &= run_copy_pattern_multi_tile(
            cluster,
            chip_id,
            core,
            num_tiles_r,
            num_tiles_c,
            src_vec,
            golden_vector,
            nchw,
            rsuv
        );
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
