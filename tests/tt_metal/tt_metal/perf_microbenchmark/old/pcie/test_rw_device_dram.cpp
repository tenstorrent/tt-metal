// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>
#include <string>

#include "common/bfloat16.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt;
using namespace tt::tt_metal;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::steady_clock;

int main(int argc, char** argv) {
  bool pass = true;

  try {
    // Initial Runtime Args Parse
    std::vector<std::string> input_args(argv, argv + argc);

    string size_string = "";
    uint32_t iter;
    try {
      std::tie(size_string, input_args) =
          test_args::get_command_option_and_remaining_args(input_args,
                                                           "--size");
      std::tie(iter, input_args) =
          test_args::get_command_option_uint32_and_remaining_args(input_args,
                                                           "--iter", 1);
    } catch (const std::exception& e) {
      TT_THROW("Please input test size with \"--size <size to test>\"",
                e.what());
    }
    uint64_t buffer_size = stoul(size_string);

    log_info(LogTest, "Measuring performance for size={}bytes", buffer_size);

    // Device Setup
    int device_id = 0;
    tt_metal::Device* device = tt_metal::CreateDevice(device_id);

    // Application Setup
    srand(time(0));
    uint32_t dram_addr = 64;
    uint32_t dram_channel = rand() % 8;
    log_info(LogTest, "Target DRAM channel = {}", dram_channel);

    // Execute Application
    log_info(LogTest, "iter {}", iter);
    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        buffer_size, 100,
        std::chrono::system_clock::now().time_since_epoch().count());

    {
      auto begin = std::chrono::steady_clock::now();
      auto end = std::chrono::steady_clock::now();
      auto elapsed_sum = end - begin;

      for (int i=0; i<iter; i++) {
        begin = std::chrono::steady_clock::now();
        pass &= tt_metal::detail::WriteToDeviceDRAMChannel(device, dram_channel,
                                                         dram_addr, src_vec);
        end = std::chrono::steady_clock::now();
        elapsed_sum += end - begin;
      }

      auto elapsed_us = duration_cast<microseconds>(elapsed_sum / iter).count();
      auto bw = (buffer_size / 1024.0 / 1024.0 / 1024.0) /
                  (elapsed_us / 1000.0 / 1000.0);
      log_info(LogTest, "WriteToDeviceDRAMChannel {:.3f}ms, {:.3f}GB/s",
               elapsed_us / 1000.0, bw);
    }

    std::vector<uint32_t> result_vec;
    {
      auto begin = std::chrono::steady_clock::now();
      auto end = std::chrono::steady_clock::now();
      auto elapsed_sum = end - begin;

      for (int i=0; i<iter; i++) {
        begin = std::chrono::steady_clock::now();
        tt_metal::detail::ReadFromDeviceDRAMChannel(
          device, dram_channel, dram_addr, buffer_size, result_vec);
        end = std::chrono::steady_clock::now();
        elapsed_sum += end - begin;
      }

      auto elapsed_us = duration_cast<microseconds>(elapsed_sum / iter).count();
      auto bw = (buffer_size / 1024.0 / 1024.0 / 1024.0) /
                  (elapsed_us / 1000.0 / 1000.0);
      log_info(LogTest, "ReadFromDeviceDRAMChannel {:.3f}ms, {:.3f}GB/s",
               elapsed_us / 1000.0, bw);
    }

    // Validation & Teardown
    pass &= (src_vec == result_vec);
    pass &= tt_metal::CloseDevice(device);
  } catch (const std::exception& e) {
    pass = false;
    log_error(LogTest, "{}", e.what());
    log_error(LogTest, "System error message: {}", std::strerror(errno));
  }

  if (pass) {
    log_info(LogTest, "Test Passed");
  } else {
    TT_THROW("Test Failed");
  }

  TT_FATAL(pass, "Error");
  return 0;
}
