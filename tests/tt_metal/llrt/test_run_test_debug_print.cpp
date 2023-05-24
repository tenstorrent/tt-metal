#include <chrono>
#include <fstream>
#include <errno.h>
#include <random>
#include <iomanip>
#include <cstdio>
#include <thread>
#include <unistd.h>
#include <filesystem>
#include <chrono>

#include "tt_cluster.hpp"
#include "utils.hpp"
#include "common/logger.hpp"
#include "tensix.h"

#include "llrt.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/llrt/test_libs/debug_mailbox.hpp"
#include "tt_metal/llrt/test_libs/tiles.hpp"

#include "llrt/tt_debug_print_server.hpp"
#include "hostdevcommon/debug_print_common.h"

using tt::llrt::CircularBufferConfigVec;
using std::vector;
using std::endl;
using std::filesystem::file_size;
using u32 = std::uint32_t;
using u16 = std::uint16_t;
using std::size_t;
using std::string;
using std::to_string;
using std::cout;
using std::endl;

constexpr bool REMOVE_TEMP_DEBUG_PRINT_FILE = true;

void launch_on_cores(
    tt_cluster* cluster,
    int chip_id,
    vector<CoreCoord> cores,
    u32 num_tiles,
    u32 num_underscores_to_print,
    bool multicore_sync,
    u32 end_x = 2
) {

    bool pass = true;
    { // load the op to cores
        string op_path = "test_debug_print_op";
        for (auto xy: cores) {
            pass = tt::llrt::test_load_write_read_risc_binary(cluster, op_path + "/brisc/brisc.hex", chip_id, xy, 0); // brisc
            pass = pass & tt::llrt::test_load_write_read_risc_binary(cluster, op_path + "/ncrisc/ncrisc.hex", chip_id, xy, 1); // ncrisc
            pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread0/tensix_thread0.hex", chip_id, xy, 0); // trisc0
            pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread1/tensix_thread1.hex", chip_id, xy, 1); // trisc1
            pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread2/tensix_thread2.hex", chip_id, xy, 2); // trisc2
        }
    }

    std::uint32_t single_tile_size = 2 * 1024;
    std::uint32_t dram_buffer_src_addr = 0;
    std::uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
    int dram_src_channel_id = 0;
    int dram_dst_channel_id = 0;
    log_info(tt::LogVerif, "num_tiles = {}", num_tiles);
    tt::llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(
        cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, cores);

    CoreCoord dram_src_noc_xy = tt::llrt::get_core_for_dram_channel(cluster, dram_src_channel_id);
    log_info(tt::LogVerif, "dram_src_noc_xy = {}", dram_src_noc_xy.str());
    CoreCoord dram_dst_noc_xy = tt::llrt::get_core_for_dram_channel(cluster, dram_dst_channel_id);
    log_info(tt::LogVerif, "dram_dst_noc_xy = {}", dram_dst_noc_xy.str());

    // BufferConfigVec -- common across all kernels, so written once to the core
    CircularBufferConfigVec circular_buffer_config_vec = tt::llrt::create_circular_buffer_config_vector();
    tt::llrt::set_config_for_circular_buffer(circular_buffer_config_vec, 0, 200*1024, 2*single_tile_size, 2); // input buf
    tt::llrt::set_config_for_circular_buffer(circular_buffer_config_vec, 16, 300*1024, 2*single_tile_size, 2); // output buf (starts at index 16)
    // buffer_config_vec written in one-shot
    for (auto core: cores) {
        tt::llrt::write_circular_buffer_config_vector_to_core(cluster, chip_id, core, circular_buffer_config_vec);

        // NCRISC kernel arguments to L1 in one-shot
        u32 ARG0 = 4, ARG1 = 2, ARG2 = int(multicore_sync), ARG3 = num_underscores_to_print;
        // ARG0 = setw arg
        // ARG1 = u32 val
        // bool(ARG2) = print endl
        // print("_"*ARG3)
        uint32_t x = core.x, y = core.y;
        tt::llrt::write_hex_vec_to_core(cluster, chip_id, core,
            { dram_buffer_src_addr, u32(dram_src_noc_xy.x), u32(dram_src_noc_xy.y), ARG0, ARG1, ARG2, ARG3, x, y, end_x },
            NCRISC_L1_ARG_BASE);

        // BRISC kernel arguments to L1 in one-shot
        u32 num_output_tiles = 0;
        u32 num_plus_signs_to_print = 4;
        tt::llrt::write_hex_vec_to_core(cluster, chip_id, core,
            { dram_buffer_dst_addr, u32(dram_dst_noc_xy.x), u32(dram_dst_noc_xy.y), num_output_tiles, num_plus_signs_to_print, x, y, multicore_sync },
            BRISC_L1_ARG_BASE);
    }

    tt::llrt::internal_::setup_riscs_on_specified_cores(cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, cores);
    tt::llrt::internal_::run_riscs_on_specified_cores(cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, cores);
}

std::string test_tmpnam()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto nanosec = now.time_since_epoch().count();
    return std::string("/tmp/___debug_print_test_tempfile__pid") + std::to_string(getpid()) + "_" + std::to_string(nanosec);
}

bool run_test_debug_print(
    tt_cluster* cluster,
    int chip_id,
    const CoreCoord& core_start,
    const CoreCoord& core_end,
    u32 num_tiles,
    u32 num_underscores_to_print,
    const char* gold_string
) {
    vector<CoreCoord> cores;
    for (size_t x = core_start.x; x < core_end.x; x++)
    for (size_t y = core_start.y; y < core_end.y; y++)
        cores.push_back(CoreCoord{x, y});

    int hart_mask = DPRINT_HART_NC | DPRINT_HART_BR;
    std::string debug_outfile_str = test_tmpnam();
    const char* debug_outfile = debug_outfile_str.c_str();
    tt_start_debug_print_server(cluster, {chip_id}, cores, hart_mask, debug_outfile);

    bool multicore = (core_end.x > core_start.x+1) || (core_end.y > core_start.y+1);
    launch_on_cores(
        cluster, chip_id, cores, num_tiles, num_underscores_to_print, multicore, core_end.x);

    bool pass = true;
    // now try to read the generated debug printed file in /tmp and compare with ref result
    constexpr int max_iters = 1000; // wait up to 10s
    for (int i = 0; i < max_iters; i ++) {
        // TODO(AP): add print order checking
        if (access(debug_outfile, F_OK) == 0 && file_size(debug_outfile) >= strlen(gold_string)) // file exists
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // sleep for a few ms
        if (i == max_iters-1) {
            log_error(tt::LogTest, "Timed out waiting on debug print file to be generated");
            pass = false;
        }
    }

    tt_stop_debug_print_server();

    if (pass) {
        // compare to reference string
        std::ifstream t(debug_outfile);
        std::string str((std::istreambuf_iterator<char>(t)),
                        std::istreambuf_iterator<char>());

        for (int j = 0; j < str.size(); j++)
            if (str[j] != gold_string[j])
                log_info(tt::LogTest, "Test string mismatch at {}\n", j);
        pass &= (str == gold_string);
    }

    // delete our temp file
    if (REMOVE_TEMP_DEBUG_PRINT_FILE)
        ::remove(debug_outfile);

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
        CoreCoord xy_start = {1,1};
        CoreCoord xy_end1x1 = {2,2};
        CoreCoord xy_end5x5 = {6,6};
        cluster->open_device(arch, target_type, {0}, sdesc_file);
        cluster->start_device(default_params); // use default params
        tt::llrt::utils::log_current_ai_clk(cluster);
        if (pass) {
            uint32_t num_tiles = 1;
            const char* expected1 = "   2\nTestConstCharStrNC{1,1}\n0.1235\n0.1200\n0.1226\n____\nTestStrBR{1,1}\n++++\n";
            pass &= run_test_debug_print(cluster, 0, xy_start, xy_end1x1, num_tiles, 4, expected1);
            // run_test will wait for a flush of buffer, so we don't need to check for a race condition for the next run_test

            // this test case will try to force the buffer flush without overflowing the buffer
            // in this second test we test for cumulative string in the output to avoid recreating the server
            int max_payload_size = sizeof(DebugPrintMemLayout::data);
            std::string n_underscores(max_payload_size*2, '_');
            std::string expected2 = "   2\nTestConstCharStrNC{1,1}\n0.1235\n0.1200\n0.1226\n" + n_underscores + "\nTestStrBR{1,1}\n++++\n";
            pass &= run_test_debug_print(cluster, 0, xy_start, xy_end1x1, num_tiles, max_payload_size*2, expected2.c_str());


            // Run a test on 5x5 tiles, expect ordered output due to RAISE/WAIT signals in debug prints
            string expected3_1 = "   2\nTestConstCharStrNC{";
            string expected3_2 = "}\n0.1235\n0.1200\n0.1226\n_____\nTestStrBR{";
            string expected3_3 = "}\n++++\n";
            string expected3 = "";
            for (int y = xy_start.y; y < xy_end5x5.y; y++)
            for (int x = xy_start.x; x < xy_end5x5.x; x++) {
                string digits = to_string(x) + "," + to_string(y);
                expected3 += expected3_1 + digits + expected3_2 + digits + expected3_3;
            }
            pass &= run_test_debug_print(cluster, 0, xy_start, xy_end5x5, num_tiles, 5, expected3.c_str());

            // this test case will try to overflow the buffer with a single large message and check that
            // the debug diagnostic message gets printed.
            // We should fail starting at max_payload_size-2 string length, due to 2 other bytes used for size and code
            // 0xFFFF is a value for ARG3 that the code in test_debug_print_nc.cpp expects to see to trigger this failure case.
            std::string expected4 = string(debug_print_overflow_error_message) + "TestStrBR{1,1}\n++++\n";
            pass &= run_test_debug_print(cluster, 0, xy_start, xy_end1x1, num_tiles, 0xFFFF, expected4.c_str());
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
