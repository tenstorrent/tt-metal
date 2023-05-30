#include "tt_cluster.hpp"
#include "utils.hpp"
#include "common/logger.hpp"
#include "tensix.h"

#include "llrt.hpp"
#include "test_libs/tiles.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "test_libs/conv_pattern.hpp"
#include "common/bfloat16.hpp"
#include "hostdevcommon/registers.hpp"

using tt::llrt::CircularBufferConfigVec;

constexpr static std::uint32_t INVALID = 0x4321;

bool run_eltwise_sync(
    tt_cluster* cluster,
    int chip_id,
    const CoreCoord& loader_core,
    const CoreCoord& compute_core,
    std::vector<uint32_t>& src_vec,
    const std::vector<bfloat16>& golden
) {

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_input_tiles = 8 * 1;
    uint32_t num_output_tiles = num_input_tiles;
    uint32_t dram_buffer_size = single_tile_size * num_output_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_src_addr = 0;
    uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
    uint32_t loader_buffer_address = 500 * 1024;
    uint32_t transient_buffer_size_tiles = 1;
    uint32_t transient_buffer_size_bytes = transient_buffer_size_tiles * single_tile_size;
    std::uint32_t stream_register_address = STREAM_REG_ADDR(0, 12);
    int dram_channel_id = 0;

    // Producer core
    tt::llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, {compute_core});
    string loader_op_path = "built_kernels/dram_loader_sync";
    bool pass = tt::llrt::test_load_write_read_risc_binary(cluster, loader_op_path + "/brisc/brisc.hex", 0, loader_core, 0); // brisc
    // Consumer core
    string compute_op_path = "built_kernels/eltwise_sync";
    pass = tt::llrt::test_load_write_read_risc_binary(cluster, compute_op_path + "/brisc/brisc.hex", 0, compute_core, 0); // brisc
    pass = pass & tt::llrt::test_load_write_read_risc_binary(cluster, compute_op_path + "/ncrisc/ncrisc.hex", 0, compute_core, 1); // ncrisc

    pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, compute_op_path + "/tensix_thread0/tensix_thread0.hex", 0, compute_core, 0); // trisc0
    pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, compute_op_path + "/tensix_thread1/tensix_thread1.hex", 0, compute_core, 1); // trisc1
    pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, compute_op_path + "/tensix_thread2/tensix_thread2.hex", 0, compute_core, 2); // trisc2

    // Src/Dst dram noc addresses
    CoreCoord dram_src_noc_xy = tt::llrt::get_core_for_dram_channel(cluster, dram_channel_id);
    CoreCoord dram_dst_noc_xy = tt::llrt::get_core_for_dram_channel(cluster, dram_channel_id);
    // ---------------------------------------------------------------------------------------
    // Producer core:
    // BRISC kernel arguments to L1 in one-shot
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, loader_core,
        {
            dram_buffer_src_addr,
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            loader_buffer_address,
            (std::uint32_t)compute_core.x,
            (std::uint32_t)compute_core.y,
            stream_register_address,
            num_output_tiles,
            transient_buffer_size_tiles,
            transient_buffer_size_bytes
        },
        BRISC_L1_ARG_BASE);
    // ---------------------------------------------------------------------------------------
    // Consumer core:
    // BufferConfigVec -- common across all kernels, so written once to the core
    CircularBufferConfigVec circular_buffer_config_vec = tt::llrt::create_circular_buffer_config_vector();
    tt::llrt::set_config_for_circular_buffer(circular_buffer_config_vec, 0, 200*1024, single_tile_size * 1, 1); // input buf
    tt::llrt::set_config_for_circular_buffer(circular_buffer_config_vec, 16, 300*1024, single_tile_size * 1, 1); // output buf (starts at index 16)
    // buffer_config_vec written in one-shot
    tt::llrt::write_circular_buffer_config_vector_to_core(cluster, chip_id, compute_core, circular_buffer_config_vec);

    // NCRISC kernel arguments to L1 in one-shot
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, compute_core,
        {
            loader_buffer_address,
            (std::uint32_t)loader_core.x,
            (std::uint32_t)loader_core.y,
            stream_register_address,
            num_output_tiles,
            transient_buffer_size_tiles,
            transient_buffer_size_bytes
        },
        NCRISC_L1_ARG_BASE);

    // BRISC kernel arguments to L1 in one-shot
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, compute_core,
        { dram_buffer_dst_addr, (std::uint32_t)dram_dst_noc_xy.x, (std::uint32_t)dram_dst_noc_xy.y, (std::uint32_t)(num_output_tiles) },
        BRISC_L1_ARG_BASE);
    // ---------------------------------------------------------------------------------------

    // Initialize producer/consumer registers to "INVALID"
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, loader_core, {INVALID}, stream_register_address);
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, compute_core, {INVALID}, stream_register_address);

    // Write tiles sequentially to DRAM
    cluster->write_dram_vec(src_vec, tt_target_dram{chip_id, dram_channel_id, 0}, dram_buffer_src_addr); // write to address

    tt::llrt::internal_::setup_riscs_on_specified_cores(cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, {loader_core, compute_core});
    tt::llrt::internal_::run_riscs_on_specified_cores(cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, {loader_core, compute_core});

    std::vector<std::uint32_t> dst_vec_dram;
    cluster->read_dram_vec(dst_vec_dram, tt_target_dram{chip_id, dram_channel_id, 0}, dram_buffer_dst_addr, dram_buffer_size);
    auto dst_vec = unpack_uint32_vec_into_bfloat16_vec(dst_vec_dram);

    pass &= (dst_vec == golden);

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
        const int chip_id = 0;
        CoreCoord loader_core = {1, 1};
        CoreCoord compute_core = {2, 1};
        cluster->open_device(arch, target_type, {chip_id}, sdesc_file);
        cluster->start_device(default_params); // use default params
        tt::llrt::utils::log_current_ai_clk(cluster);

        SHAPE shape = {1, 1, 32, 256};

        tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, tt::tiles_test::get_seed_from_systime()); // TODO: make randomized!
        auto src_vec = pack_bfloat16_vec_into_uint32_vec(tensor.get_values());
        pass &= run_eltwise_sync(
            cluster,
            chip_id,
            loader_core,
            compute_core,
            src_vec,
            tensor.get_values()
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
