// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "common/bfloat16.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"

using namespace tt;
using namespace tt::tt_metal;
using std::chrono::duration_cast;
using std::chrono::microseconds;

////////////////////////////////////////////////////////////////////////////////
// This test measures the performance of adjacent NOC data transfer. Every
// Tensix cores read from or write to the L1 of the neighbor Tensix core. The
// direction of the transfer is fixed within a test. A user can change the
// direction of the transfer by giving an input argument.
//
// Usage example:
//   ./test_noc_adjacent
//     --cores-r <number of cores in a row>
//     --cores-c <number of cores in a column>
//     --num-tiles <total number of tiles each core transfers>
//     --tiles-per-transfer <number of tiles for each transfer>
//     --noc-index <NOC index to use>
//     --noc-direction <direction of data transfer:
//                      0 for +x, 1 for -y, 2 for -x, and 3 for +y>
//     --access-type <0 for read access, 1 for write access>
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
    log_fatal("Test not supported w/ slow dispatch, exiting");
  }

  bool pass = true;
  double measured_bandwidth = 0;

  ////////////////////////////////////////////////////////////////////////////
  //                      Initial Runtime Args Parse
  ////////////////////////////////////////////////////////////////////////////
  std::vector<std::string> input_args(argv, argv + argc);
  uint32_t num_cores_r;
  uint32_t num_cores_c;
  uint32_t num_tiles;
  uint32_t noc_index;
  uint32_t noc_direction;
  uint32_t access_type;
  uint32_t tiles_per_transfer;
  try {
    std::tie(num_cores_r, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args,
                                                                "--cores-r", 9);
    std::tie(num_cores_c, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(
            input_args, "--cores-c", 12);

    std::tie(num_tiles, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(
            input_args, "--num-tiles", 256);

    std::tie(tiles_per_transfer, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(
            input_args, "--tiles-per-transfer", 1);

    std::tie(noc_index, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(
            input_args, "--noc-index", 0);

    std::tie(noc_direction, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(
            input_args, "--noc-direction", 0);

    std::tie(access_type, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(
            input_args, "--access-type", 0);

    test_args::validate_remaining_args(input_args);
  } catch (const std::exception& e) {
    log_fatal(tt::LogTest, "Command line arguments found exception", e.what());
  }

  if (num_tiles % tiles_per_transfer != 0) {
    log_fatal(
        tt::LogTest,
        "Total number of tiles each core transfers ({}) must be the multiple "
        "of number of tiles for each transfer ({})",
        num_tiles, tiles_per_transfer);
  }

  if (num_tiles < tiles_per_transfer) {
    log_fatal(tt::LogTest,
              "Total number of tiles each core transfers ({}) must be bigger "
              "than or equal to the number of tiles for each transfer ({})",
              num_tiles, tiles_per_transfer);
  }

  try {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    int device_id = 0;
    tt_metal::Device* device = tt_metal::CreateDevice(device_id);
    CommandQueue& cq = *tt::tt_metal::detail::GLOBAL_CQ;

    tt_cluster* cluster = device->cluster();
    int clock_freq_mhz = cluster->get_device_aiclk(0);

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();

    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {(std::size_t)num_cores_c - 1,
                          (std::size_t)num_cores_r - 1};
    CoreRange all_cores{.start = start_core, .end = end_core};

    uint32_t cb_tiles = 32;
    uint32_t single_tile_size = 2 * 1024;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program, 0, all_cores, cb_tiles, cb_tiles * single_tile_size,
        tt::DataFormat::Float16_b);

    auto cb_src1 = tt_metal::CreateCircularBuffers(
        program, 1, all_cores, cb_tiles, cb_tiles * single_tile_size,
        tt::DataFormat::Float16_b);

    auto noc_kernel = tt_metal::CreateDataMovementKernel(
        program,
        (access_type == 0) ? "tests/tt_metal/tt_metal/perf_microbenchmark/"
                             "2_noc_adjacent/kernels/noc_read.cpp"
                           : "tests/tt_metal/tt_metal/perf_microbenchmark/"
                             "2_noc_adjacent/kernels/noc_write.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = (noc_index == 0)
                             ? tt_metal::DataMovementProcessor::RISCV_0
                             : tt_metal::DataMovementProcessor::RISCV_1,
            .noc = (noc_index == 0) ? tt_metal::NOC::RISCV_0_default
                                    : tt_metal::NOC::RISCV_1_default});

    for (int i = 0; i < num_cores_r; i++) {
      for (int j = 0; j < num_cores_c; j++) {
        CoreCoord logical_core = {(std::size_t)j, (std::size_t)i};

        CoreCoord adjacent_core_logical = {(std::size_t)j, (std::size_t)i};
        if (noc_direction == 0) {
          // right (+x direction)
          adjacent_core_logical.x = (adjacent_core_logical.x + 1) % num_cores_c;
        } else if (noc_direction == 1) {
          // down (-y direction)
          adjacent_core_logical.y =
              (adjacent_core_logical.y + num_cores_r - 1) % num_cores_r;
        } else if (noc_direction == 2) {
          // left (-x direction)
          adjacent_core_logical.x =
              (adjacent_core_logical.x + num_cores_c - 1) % num_cores_c;
        } else {
          // up (+y direction)
          adjacent_core_logical.y = (adjacent_core_logical.y + 1) % num_cores_r;
        }

        CoreCoord adjacent_core_noc =
            device->worker_core_from_logical_core(adjacent_core_logical);

        vector<uint32_t> noc_runtime_args = {
            (uint32_t)adjacent_core_noc.x, (uint32_t)adjacent_core_noc.y,
            cb_src1.address(), num_tiles / tiles_per_transfer,
            tiles_per_transfer};
        SetRuntimeArgs(program, noc_kernel, logical_core, noc_runtime_args);
      }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::detail::CompileProgram(device, program);

    auto t_begin = std::chrono::steady_clock::now();
    EnqueueProgram(cq, program, false);
    Finish(cq);
    auto t_end = std::chrono::steady_clock::now();
    unsigned long elapsed_us =
        duration_cast<microseconds>(t_end - t_begin).count();
    unsigned long elapsed_cc = clock_freq_mhz * elapsed_us;

    log_info(LogTest, "Time elapsed for NOC transfers: {}us ({}cycles)",
             elapsed_us, elapsed_cc);

    // total transfer amount per core = tile size * number of tiles
    // NOC bandwidth = total transfer amount per core / elapsed clock cycle
    measured_bandwidth = (double)single_tile_size * num_tiles / elapsed_cc;

    log_info(LogTest, "Measured NOC bandwidth: {:.3f}B/cc", measured_bandwidth);

    pass &= tt_metal::CloseDevice(device);
  } catch (const std::exception& e) {
    pass = false;
    log_error(LogTest, "{}", e.what());
    log_error(LogTest, "System error message: {}", std::strerror(errno));
  }

  // Determine if it passes performance goal
  if (pass) {
    // goal is 95% of theoretical peak using a single NOC channel
    // theoretical peak: 32bytes per clock cycle
    double target_bandwidth = 32 * 0.9;

    if (measured_bandwidth > target_bandwidth) {
      pass = false;
      log_error(LogTest,
                "The NOC bandwidth does not meet the criteria. "
                "Current: {:.3f}B/cc, goal: <{:.3f}B/cc",
                measured_bandwidth, target_bandwidth);
    }
  }

  if (pass) {
    log_info(LogTest, "Test Passed");
  } else {
    log_fatal(LogTest, "Test Failed");
  }

  TT_ASSERT(pass);

  return 0;
}
