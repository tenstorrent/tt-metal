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
// This test measures the performance of random-to-random NOC data transfer. It
// creates a L1 buffer and every Tensix cores read from or write to the L1
// buffer. The memory access pattern of each Tensix core is a stride where the
// stride length is equal to the number of Tensix cores multiplied by page size.
// Since each page of the L1 buffer is scattered across all the Tensix cores and
// their mapping is random, the random-to-random data transfer between NOCs
// occurs.
//
// TODO: Currently, this benchmark uses only 1 NOC for communication.
// Ultimately, it should use two NOCs together through Metal's channel grouping
// interface
//
// Usage example:
//   ./test_noc_rtor --cores-r <number of cores in a row> --cores-c <number
//   of cores in a column> --num-tiles <number of tiles each core accesses>
//   --noc-index <NOC index to use> --access-type <0 for read access, 1 for
//   write access>
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

    uint32_t single_tile_size = 2 * 1024;
    uint32_t page_size = single_tile_size;
    uint32_t l1_buffer_size = num_cores_r * num_cores_c * num_tiles * page_size;
    auto l1_buffer = tt_metal::Buffer(device, l1_buffer_size, page_size,
                                      tt_metal::BufferType::L1);

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();

    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {(std::size_t)num_cores_c - 1,
                          (std::size_t)num_cores_r - 1};
    CoreRange all_cores{.start = start_core, .end = end_core};

    for (int i = 0; i < num_cores_r; i++) {
      for (int j = 0; j < num_cores_c; j++) {
        int core_index = i * num_cores_c + j;
        CoreCoord core = {(std::size_t)j, (std::size_t)i};

        uint32_t cb_index = 0;
        uint32_t cb_tiles = 32;
        auto cb_src0 = tt_metal::CreateCircularBuffer(
            program, cb_index, core, cb_tiles, cb_tiles * single_tile_size,
            tt::DataFormat::Float16_b);
      }
    }

    auto noc_kernel = tt_metal::CreateDataMovementKernel(
        program,
        (access_type == 0) ? "tests/tt_metal/tt_metal/perf_microbenchmark/"
                             "2_noc_rtor/kernels/noc_read.cpp"
                           : "tests/tt_metal/tt_metal/perf_microbenchmark/"
                             "2_noc_rtor/kernels/noc_write.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = (noc_index == 0)
                             ? tt_metal::DataMovementProcessor::RISCV_0
                             : tt_metal::DataMovementProcessor::RISCV_1,
            .noc = (noc_index == 0) ? tt_metal::NOC::RISCV_0_default
                                    : tt_metal::NOC::RISCV_1_default});

    for (int i = 0; i < num_cores_r; i++) {
      for (int j = 0; j < num_cores_c; j++) {
        CoreCoord core = {(std::size_t)j, (std::size_t)i};
        uint32_t core_index = i * num_cores_c + j;
        uint32_t l1_buffer_addr = l1_buffer.address();

        vector<uint32_t> noc_runtime_args = {
            core_index, l1_buffer_addr, num_tiles, num_cores_r * num_cores_c};
        SetRuntimeArgs(program, noc_kernel, core, noc_runtime_args);
      }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
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
