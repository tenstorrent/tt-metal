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
using std::chrono::steady_clock;

int main(int argc, char** argv) {
  bool pass = true;

  try {
    // Initial Runtime Args Parse
    std::vector<std::string> input_args(argv, argv + argc);

    string buffer_type_string = "";
    uint32_t iter;
    try {
      std::tie(iter, input_args) =
          test_args::get_command_option_uint32_and_remaining_args(input_args,
                                                           "--iter", 1);
      std::tie(buffer_type_string, input_args) =
          test_args::get_command_option_and_remaining_args(input_args,
                                                           "--buffer_type");
    } catch (const std::exception& e) {
      TT_THROW("Please input type of the buffer with \"--buffer_type <0: "
                "DRAM, 1: L1>\"",
                e.what());
    }
    int buffer_type = stoi(buffer_type_string);

    string size_string = "";
    try {
      std::tie(size_string, input_args) =
          test_args::get_command_option_and_remaining_args(input_args,
                                                           "--size");
    } catch (const std::exception& e) {
      TT_THROW("Please input test size with \"--size <size to test>\"",
                e.what());
    }
    uint64_t buffer_size = stoul(size_string);

    log_info(LogTest, "Measuring performance for buffer_type={}, size={}bytes",
             buffer_type == 0 ? "DRAM" : "L1", buffer_size);

    // Device Setup
    int device_id = 0;
    tt_metal::Device* device = tt_metal::CreateDevice(device_id);

    // Application Setup
    uint32_t single_tile_size = 2 * 1024;
    auto page_size = single_tile_size;
    BufferType buff_type = buffer_type == 0 ? tt_metal::BufferType::DRAM
                                              : tt_metal::BufferType::L1;
    tt_metal::InterleavedBufferConfig buff_config{
                    .device=device,
                    .size = buffer_size,
                    .page_size = page_size,
                    .buffer_type = buff_type
        };
    auto buffer = CreateBuffer(buff_config);

    // Execute Application
    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        buffer_size, 100,
        std::chrono::system_clock::now().time_since_epoch().count());
    {
      auto begin = std::chrono::steady_clock::now();
      auto end = std::chrono::steady_clock::now();
      auto elapsed_sum = end - begin;

      for (int i=0; i<iter; i++) {
        begin = std::chrono::steady_clock::now();
        tt_metal::detail::WriteToBuffer(buffer, src_vec);
        end = std::chrono::steady_clock::now();
        elapsed_sum += end - begin;
      }

      auto elapsed_us = duration_cast<microseconds>(elapsed_sum / iter).count();
      auto bw = (buffer_size / 1024.0 / 1024.0 / 1024.0) /
                  (elapsed_us / 1000.0 / 1000.0);
      log_info(LogTest, "detail::WriteToBuffer {}: {:.3f}ms, {:.3f}GB/s",
                buffer_type == 0 ? "DRAM" : "L1", elapsed_us / 1000.0, bw);
    }

    std::vector<uint32_t> result_vec;
    {
      auto begin = std::chrono::steady_clock::now();
      auto end = std::chrono::steady_clock::now();
      auto elapsed_sum = end - begin;

      for (int i=0; i<iter; i++) {
        begin = std::chrono::steady_clock::now();
        tt_metal::detail::ReadFromBuffer(buffer, result_vec);
        end = std::chrono::steady_clock::now();
        elapsed_sum += end - begin;
      }
      auto elapsed_us = duration_cast<microseconds>(elapsed_sum / iter).count();
      auto bw = (buffer_size / 1024.0 / 1024.0 / 1024.0) /
                  (elapsed_us / 1000.0 / 1000.0);
      log_info(LogTest, "detail::ReadFromBuffer {}: {:.3f}ms, {:.3f}GB/s",
                buffer_type == 0 ? "DRAM" : "L1", elapsed_us / 1000.0, bw);
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
