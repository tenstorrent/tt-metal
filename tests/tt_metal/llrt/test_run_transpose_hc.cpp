#include <chrono>
#include <fstream>
#include <errno.h>
#include <random>
#include <iomanip>

#include "tt_cluster.hpp"
#include "utils.hpp"
#include "common/logger.hpp"
#include "tensix.h"

#include "llrt.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/llrt/test_libs/debug_mailbox.hpp"
#include "tt_metal/llrt/test_libs/tiles.hpp"

#include "llrt/tt_debug_print_server.hpp"

using tt::llrt::CircularBufferConfigVec;
using std::vector;
using std::endl;
using u32 = std::uint32_t;
using u16 = std::uint16_t;
using tt::tiles_test::untilize_nchw;
using tt::tiles_test::tilize_nchw;

uint32_t prod(vector<uint32_t> &shape) {
    uint32_t shape_prod = 1;

    for (uint32_t shape_i: shape) {
        shape_prod *= shape_i;
    }

    return shape_prod;
}

// input shape.x is assumed to have the full number of elements in bfloat16
// src_vec is expected to be untilized
// result is also untilized
inline vector<u16> gold_transpose_hc(std::vector<u16> src_vec, vector<uint32_t> shape) {
    struct TensLinAddr {
        vector<uint32_t> sh;
        TensLinAddr(vector<uint32_t> shape) : sh(shape) {}
        int offs(int n, int c, int h, int w) {
            TT_ASSERT(u32(n) < sh[0] && u32(c) < sh[1] && u32(h) < sh[2] && u32(w) < sh[3]);
            return w + sh[3]*h + sh[2]*sh[3]*c + sh[1]*sh[2]*sh[3]*n;
        }
    };

    vector<uint32_t> shapeT{shape[0], shape[2], shape[1], shape[3]};
    TensLinAddr addr(shape);
    TensLinAddr addrt(shapeT);

    vector<u16> transposed(src_vec.size());
    for (int n = 0; n < shape[0]; n++)
    for (int c = 0; c < shape[1]; c++)
    for (int h = 0; h < shape[2]; h++)
    for (int w = 0; w < shape[3]; w++) {
        auto toffs = addrt.offs(n, h, c, w);
        auto offs = addr.offs(n, c, h, w);
        TT_ASSERT(toffs < transposed.size() && offs < src_vec.size());
        transposed[toffs] = src_vec[offs];
    }
    //log_info(tt::LogVerif, "Prior size = {}", transposed.size());

    return transposed;
};

// Test
bool run_transpose_hc(
    tt_cluster* cluster,
    int chip_id,
    const tt_xy_pair& core,
    u32 num_tiles,
    const vector<u32>& shape
) {

    std::uint32_t single_tile_size = 2 * 1024;
    std::uint32_t dram_buffer_size = single_tile_size * num_tiles;
    std::uint32_t dram_buffer_src_addr = 0;
    std::uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
    int dram_src_channel_id = 0;
    int dram_dst_channel_id = 0;
    log_info(tt::LogVerif, "num_tiles = {}", num_tiles);
    log_info(tt::LogVerif, "single_tile_size = {} B", single_tile_size);
    log_info(tt::LogVerif, "dram_bufer_size = {} B", dram_buffer_size);

    tt::llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(
        cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, {core});

    tt_xy_pair dram_src_noc_xy = tt::llrt::get_core_for_dram_channel(cluster, dram_src_channel_id);
    log_info(tt::LogVerif, "dram_src_noc_xy = {}", dram_src_noc_xy.str());
    tt_xy_pair dram_dst_noc_xy = tt::llrt::get_core_for_dram_channel(cluster, dram_dst_channel_id);
    log_info(tt::LogVerif, "dram_dst_noc_xy = {}", dram_dst_noc_xy.str());

    // BufferConfigVec -- common across all kernels, so written once to the core
    CircularBufferConfigVec circular_buffer_config_vec = tt::llrt::create_circular_buffer_config_vector();
    tt::llrt::set_config_for_circular_buffer(circular_buffer_config_vec, 0, 200*1024, 2*single_tile_size, 2); // input buf
    tt::llrt::set_config_for_circular_buffer(circular_buffer_config_vec, 16, 300*1024, 2*single_tile_size, 2); // output buf (starts at index 16)
    // buffer_config_vec written in one-shot
    tt::llrt::write_circular_buffer_config_vector_to_core(cluster, chip_id, core, circular_buffer_config_vec);

    // NCRISC kernel arguments to L1 in one-shot
    u32 W = shape[3], H = shape[2], C = shape[1];
    u32 HW = H*W;
    TT_ASSERT(shape[0] == 1); // TODO(AP): batch 1 only for now
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, core,
        { dram_buffer_src_addr, u32(dram_src_noc_xy.x), u32(dram_src_noc_xy.y), W, H, C, HW },
        NCRISC_L1_ARG_BASE);

    // BRISC kernel arguments to L1 in one-shot
    u32 num_output_tiles = num_tiles;
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, core,
        { dram_buffer_dst_addr, u32(dram_dst_noc_xy.x), u32(dram_dst_noc_xy.y), num_output_tiles },
        BRISC_L1_ARG_BASE);

    // Note: TRISC 0/1/2 kernel args are hard-coded
    TT_ASSERT(dram_buffer_size % sizeof(std::uint32_t) == 0);

    // Write input tensor from device DRAM
    constexpr int SEED = 0x1234;
    constexpr float MAXVAL = 4096.0f;
    vector<u32> src_vec = create_random_vector_of_bfloat16(dram_buffer_size, MAXVAL, SEED);
    cluster->write_dram_vec(
        src_vec, tt_target_dram{chip_id, dram_src_channel_id, 0}, dram_buffer_src_addr);

    tt::llrt::internal_::setup_riscs_on_specified_cores(cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, {core});
    tt::llrt::internal_::run_riscs_on_specified_cores(cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, {core});

    // Read output tensor from device DRAM
    vector<u32> dst_vec(src_vec.size());
    cluster->read_dram_vec(
        dst_vec, tt_target_dram{chip_id, dram_dst_channel_id, 0}, dram_buffer_dst_addr, dram_buffer_size);

    // untilize input vector for consumption by gold_transpose_hc
    vector<u16> src_untilized16 = untilize_nchw(u16_from_u32_vector(src_vec), shape);
    auto gold_transposed = gold_transpose_hc(src_untilized16, shape); // result is u16 untilized

    // Tilize from row major and convert to pairs (u32)
    vector<uint32_t> shapeT{shape[0], shape[2], shape[1], shape[3]};
    auto expected32 = u32_from_u16_vector(tilize_nchw(gold_transposed, shapeT));

    auto comparison_function = [](float a, float b) { return a == b; };
    int argfail = -1;
    bool pass = packed_uint32_t_vector_comparison(dst_vec, expected32, comparison_function, &argfail);
    if (!pass)
        log_error(tt::LogTest, "Failure position={}", argfail);
    return pass;
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
        int chip_id = 0;
        tt_xy_pair xy = {1,1};
        cluster->open_device(arch, target_type, {0}, sdesc_file);
        cluster->start_device(default_params); // use default params
        tt::llrt::utils::log_current_ai_clk(cluster);

        string op_path = "built_kernels/transpose_hc_op";

        pass = tt::llrt::test_load_write_read_risc_binary(cluster, op_path + "/brisc/brisc.hex", chip_id, xy, 0); // brisc
        pass = pass & tt::llrt::test_load_write_read_risc_binary(cluster, op_path + "/ncrisc/ncrisc.hex", chip_id, xy, 1); // ncrisc

        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread0/tensix_thread0.hex", chip_id, xy, 0); // trisc0
        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread1/tensix_thread1.hex", chip_id, xy, 1); // trisc1
        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread2/tensix_thread2.hex", chip_id, xy, 2); // trisc2

        if (pass) {
            vector<uint32_t> shape = {1, 96, 32*4, 32*5};
            uint32_t num_tiles = shape.at(0) * shape.at(1) * shape.at(2) * shape.at(3) / (32*32);
            pass &= run_transpose_hc(cluster, 0, {1, 1}, num_tiles, shape);
        }
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
