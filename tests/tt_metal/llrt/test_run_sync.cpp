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

bool run_sync(
    tt_cluster* cluster,
    int chip_id,
    const tt_xy_pair& loader_core,
    const tt_xy_pair& writer_core,
    std::vector<uint32_t>& src_vec,
    const std::vector<bfloat16>& golden
) {

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_input_tiles = 1024 * 1;
    uint32_t num_output_tiles = num_input_tiles;
    uint32_t dram_buffer_size = single_tile_size * num_output_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_src_addr = 0;
    uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
    uint32_t loader_buffer_address = 500 * 1024;
    uint32_t writer_buffer_address = 500 * 1024;
    uint32_t transient_buffer_size_tiles = 4;
    uint32_t transient_buffer_size_bytes = transient_buffer_size_tiles * single_tile_size;
    std::uint32_t stream_register_address = STREAM_REG_ADDR(0, 12);
    int dram_channel_id = 0;

    TT_ASSERT(num_output_tiles % transient_buffer_size_tiles == 0);
    // Producer core
    tt::llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, tt::llrt::TensixRiscsOptions::BRISC_NCRISC, {});
    string loader_op_path = "built_kernels/dram_loader_sync";
    bool pass = tt::llrt::test_load_write_read_risc_binary(cluster, loader_op_path + "/brisc/brisc.hex", 0, loader_core, 0); // brisc
    // Consumer core
    string compute_op_path = "built_kernels/remote_read_remote_write_sync";
    pass = tt::llrt::test_load_write_read_risc_binary(cluster, compute_op_path + "/brisc/brisc.hex", 0, writer_core, 0); // brisc
    pass = pass & tt::llrt::test_load_write_read_risc_binary(cluster, compute_op_path + "/ncrisc/ncrisc.hex", 0, writer_core, 1); // ncrisc

    // Src/Dst dram noc addresses
    tt_xy_pair dram_src_noc_xy = tt::llrt::get_core_for_dram_channel(cluster, dram_channel_id);
    tt_xy_pair dram_dst_noc_xy = tt::llrt::get_core_for_dram_channel(cluster, dram_channel_id);
    // ---------------------------------------------------------------------------------------
    // Producer core:
    // BRISC kernel arguments to L1 in one-shot
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, loader_core,
        {
            dram_buffer_src_addr,
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            loader_buffer_address,
            (std::uint32_t)writer_core.x,
            (std::uint32_t)writer_core.y,
            stream_register_address,
            num_output_tiles,
            transient_buffer_size_tiles,
            transient_buffer_size_bytes
        },
        BRISC_L1_ARG_BASE);
    // ---------------------------------------------------------------------------------------

    // NCRISC kernel arguments to L1 in one-shot
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, writer_core,
        {
            loader_buffer_address,
            (std::uint32_t)loader_core.x,
            (std::uint32_t)loader_core.y,
            dram_buffer_dst_addr,
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            writer_buffer_address,
            stream_register_address,
            num_output_tiles,
            transient_buffer_size_tiles,
            transient_buffer_size_bytes
        },
        NCRISC_L1_ARG_BASE);

    // ---------------------------------------------------------------------------------------

    // Initialize producer/consumer registers to "INVALID"
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, loader_core, {INVALID}, stream_register_address);
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, writer_core, {INVALID}, stream_register_address);

    // Write tiles sequentially to DRAM
    cluster->write_dram_vec(src_vec, tt_target_dram{chip_id, dram_channel_id, 0}, dram_buffer_src_addr); // write to address

    tt::llrt::internal_::setup_riscs_on_specified_cores(cluster, chip_id, tt::llrt::TensixRiscsOptions::BRISC_NCRISC, {loader_core, writer_core});
    tt::llrt::internal_::run_riscs_on_specified_cores(cluster, chip_id, tt::llrt::TensixRiscsOptions::BRISC_NCRISC, {loader_core, writer_core});

    std::vector<std::uint32_t> dst_vec_dram;
    cluster->read_dram_vec(dst_vec_dram, tt_target_dram{chip_id, dram_channel_id, 0}, dram_buffer_dst_addr, dram_buffer_size);
    auto dst_vec = unpack_uint32_vec_into_bfloat16_vec(dst_vec_dram);

    pass &= (dst_vec == golden);

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
        tt_xy_pair loader_core = {1, 1};
        tt_xy_pair compute_core = {2, 1};
        cluster->open_device(arch, target_type, {chip_id}, sdesc_file);
        cluster->start_device(default_params); // use default params
        tt::llrt::utils::log_current_ai_clk(cluster);

        SHAPE shape = {1, 1, 32, 1024 * 32};

        tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, tt::tiles_test::get_seed_from_systime()); // TODO: make randomized!
        auto src_vec = pack_bfloat16_vec_into_uint32_vec(tensor.get_values());
        log_info(tt::LogTest, "Started: run_sync");
        pass &= run_sync(
            cluster,
            chip_id,
            loader_core,
            compute_core,
            src_vec,
            tensor.get_values()
        );
        log_info(tt::LogTest, "Finished: run_sync");
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
