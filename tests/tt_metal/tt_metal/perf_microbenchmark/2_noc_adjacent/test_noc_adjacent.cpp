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
// TODO
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
    log_fatal("Test not supported w/ slow dispatch, exiting");
  }

  bool pass = true;
  unsigned long elapsed_us;

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

    std::tie(noc_index, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(
            input_args, "--noc-index", 0);

    std::tie(noc_direction, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(
            input_args, "--noc-direction", 0);

    std::tie(access_type, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(
            input_args, "--access-type", 0);
  } catch (const std::exception& e) {
    log_fatal(tt::LogTest, "Command line arguments found exception", e.what());
  }

  try {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    int device_id = 0;
    tt_metal::Device* device = tt_metal::CreateDevice(device_id);
    CommandQueue& cq = *tt::tt_metal::detail::GLOBAL_CQ;

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

        vector<uint32_t> noc_runtime_args = {(uint32_t)adjacent_core_noc.x,
                                             (uint32_t)adjacent_core_noc.y,
                                             cb_src1.address(), num_tiles};
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
    elapsed_us = duration_cast<microseconds>(t_end - t_begin).count();

    log_info(LogTest, "Time elapsed for NOC transfers: {}us", elapsed_us);

    pass &= tt_metal::CloseDevice(device);
  } catch (const std::exception& e) {
    pass = false;
    log_error(LogTest, "{}", e.what());
    log_error(LogTest, "System error message: {}", std::strerror(errno));
  }

  // Determine if it passes performance goal
  if (pass) {
    // TODO: Numbers are TBD in SoW
    ;
  }

  if (pass) {
    log_info(LogTest, "Test Passed");
  } else {
    log_fatal(LogTest, "Test Failed");
  }

  TT_ASSERT(pass);

  return 0;
}
