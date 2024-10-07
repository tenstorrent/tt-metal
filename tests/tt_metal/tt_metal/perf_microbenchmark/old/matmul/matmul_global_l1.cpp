// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <functional>
#include <random>
#include <thread>

#include "common/bfloat16.hpp"
#include "test_tiles.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/debug/dprint_server.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"

using namespace tt;

// took from bmm_op.cpp
CoreCoord get_core_range(uint32_t num_blocks_rows, uint32_t num_blocks_cols,
                         uint32_t max_num_rows, uint32_t max_num_cols) {
  CoreCoord core_range(0, 0);
  if (!(num_blocks_rows == 1 && num_blocks_cols == 1) &&
      num_blocks_rows <= max_num_rows && num_blocks_cols <= max_num_cols) {
    core_range.x = num_blocks_cols;
    core_range.y = num_blocks_rows;
  }
  return core_range;
}

// took & revise from bmm_op_multi_core_reuse_mcast_2d_optimized.cpp
tt_metal::ScopedProgramHandle create_program_mcast_in0_in1(
    tt_metal::Device* device, MathFidelity math_fidelity, CoreCoord core_range,
    uint32_t B, uint32_t M, uint32_t N, uint32_t K, bool bcast_batch,
    uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w,
    uint32_t per_core_M, uint32_t per_core_N, std::shared_ptr<tt_metal::Buffer> in0_buffer,
    std::shared_ptr<tt_metal::Buffer> in1_buffer, std::shared_ptr<tt_metal::Buffer> bias_buffer,
    std::shared_ptr<tt_metal::Buffer> out_buffer, tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format, tt::DataFormat bias_data_format,
    tt::DataFormat output_data_format) {
  auto program = tt::tt_metal::CreateScopedProgram();

  uint32_t in0_single_tile_size = tt_metal::detail::TileSize(in0_data_format);
  uint32_t in1_single_tile_size = tt_metal::detail::TileSize(in1_data_format);
  uint32_t bias_single_tile_size = tt_metal::detail::TileSize(bias_data_format);
  uint32_t output_single_tile_size =
      tt_metal::detail::TileSize(output_data_format);

  uint32_t in0_block_tiles = per_core_M * in0_block_w;
  uint32_t in0_CB_tiles = in0_block_tiles * 2;  // double buffer
  uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;
  uint32_t in1_block_tiles = per_core_N * in0_block_w;
  uint32_t in1_CB_tiles = in1_block_tiles * 2;  // double buffer
  uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;
  uint32_t out_block_tiles = per_core_M * per_core_N;
  uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
  uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;

  uint32_t in3_block_tiles = per_core_N;
  uint32_t in3_CB_tiles = in3_block_tiles;  // No double buffer
  uint32_t in3_CB_size = in3_CB_tiles * bias_single_tile_size;

  uint32_t interm1_block_tiles = out_subblock_h * out_subblock_w;
  uint32_t interm1_CB_tiles = interm1_block_tiles;  // No double buffer
  uint32_t interm1_CB_size = interm1_CB_tiles * output_single_tile_size;

  uint32_t start_core_x = 0;
  uint32_t start_core_y = 0;
  uint32_t num_cores_c = core_range.x;
  uint32_t num_cores_r = core_range.y;

  CoreRange all_cores(
      {(std::size_t)start_core_x, (std::size_t)start_core_y},
      {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});

  CoreRange top_left_corner(
      {(std::size_t)start_core_x, (std::size_t)start_core_y}, {(std::size_t)start_core_x, (std::size_t)start_core_y});

  CoreRange left_column(
      {(std::size_t)start_core_x, (std::size_t)start_core_y},
      {(std::size_t)start_core_x, (std::size_t)start_core_y + num_cores_r - 1});

  CoreRange top_row(
      {(std::size_t)start_core_x, (std::size_t)start_core_y},
      {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y});

  CoreRange all_except_left_column(
      {(std::size_t)start_core_x + 1, (std::size_t)start_core_y},
      {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});

  CoreRange all_except_top_row(
      {(std::size_t)start_core_x, (std::size_t)start_core_y + 1},
      {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});

  CoreRange in0_sender_in1_receiver(
      {(std::size_t)start_core_x, (std::size_t)start_core_y + 1},
      {(std::size_t)start_core_x, (std::size_t)start_core_y + num_cores_r - 1});

  CoreRange in0_receiver_in1_sender(
      {(std::size_t)start_core_x + 1, (std::size_t)start_core_y},
      {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y});

  // Not exactly half-half; this seems to get slightly better perf for fused qkv
  // and selfout
  // TODO: Experiment with different splits?

  uint32_t half_core = (num_cores_c) / 2;
  bool split_half = num_cores_c > 2;

  CoreRange in0_receiver_in1_receiver_left_half(
      {(std::size_t)start_core_x + 1, (std::size_t)start_core_y + 1},
      {(std::size_t)start_core_x + half_core, (std::size_t)start_core_y + num_cores_r - 1});

  CoreRange in0_receiver_in1_receiver_right_half({0, 0}, {0, 0});

  if (split_half) {
    in0_receiver_in1_receiver_right_half = CoreRange(
        {(std::size_t)start_core_x + 1 + half_core, (std::size_t)start_core_y + 1},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});
  }

  /* Uncomment if we don't checkerboard
  CoreRange in0_receiver_in1_receiver(
      {(std::size_t) start_core_x + 1, (std::size_t) start_core_y + 1},
      {(std::size_t) start_core_x + num_cores_c - 1, (std::size_t)
  start_core_y + num_cores_r - 1});
  */

  /* Checkerboard logic
  std::set<CoreRange> in0_receiver_in1_receiver_ckb_white_set;
  std::set<CoreRange> in0_receiver_in1_receiver_ckb_black_set;
  bool white = true;
  for (std::size_t y = start_core_y + 1; y < start_core_y + num_cores_r; y++) {
      for (std::size_t x = start_core_x + 1; x < start_core_x + num_cores_c;
  x++) { CoreCoord core_coord(x, y); CoreRange
  dummy_core_range(core_coord, core_coord); if (white) {
              in0_receiver_in1_receiver_ckb_white_set.insert(dummy_core_range);
              white = false;
          }
          else {
              in0_receiver_in1_receiver_ckb_black_set.insert(dummy_core_range);
              white = true;
          }
      }
  }
  CoreRangeSet
  in0_receiver_in1_receiver_ckb_white(in0_receiver_in1_receiver_ckb_white_set);
  CoreRangeSet
  in0_receiver_in1_receiver_ckb_black(in0_receiver_in1_receiver_ckb_black_set);
  */

  // Mcast args
  auto in0_mcast_sender_semaphore_id =
      tt_metal::CreateSemaphore(program, all_cores, INVALID);
  auto in0_mcast_receiver_semaphore_id =
      tt_metal::CreateSemaphore(program, all_cores, INVALID);
  auto in1_mcast_sender_semaphore_id =
      tt_metal::CreateSemaphore(program, all_cores, INVALID);
  auto in1_mcast_receiver_semaphore_id =
      tt_metal::CreateSemaphore(program, all_cores, INVALID);
  uint32_t in3_mcast_sender_semaphore_id = 0;
  uint32_t in3_mcast_receiver_semaphore_id = 0;
  if (bias_buffer != nullptr) {
    in3_mcast_sender_semaphore_id = in1_mcast_sender_semaphore_id;
    in3_mcast_receiver_semaphore_id = in1_mcast_receiver_semaphore_id;
  }
  CoreCoord top_left_core = {(std::size_t)start_core_x,
                             (std::size_t)start_core_y};
  CoreCoord top_left_core_plus_one = {(std::size_t)start_core_x + 1,
                                      (std::size_t)start_core_y + 1};
  CoreCoord bottom_right_core = {(std::size_t)start_core_x + num_cores_c - 1,
                                 (std::size_t)start_core_y + num_cores_r - 1};
  auto top_left_core_physical =
      device->worker_core_from_logical_core(top_left_core);
  auto top_left_core_plus_one_physical =
      device->worker_core_from_logical_core(top_left_core_plus_one);
  auto bottom_right_core_physical =
      device->worker_core_from_logical_core(bottom_right_core);

  bool in0_is_dram =
      in0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
  bool in1_is_dram =
      in1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
  bool in3_is_dram = true;
  if (bias_buffer != nullptr) {
    in3_is_dram =
        bias_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
  }
  bool out_is_dram =
      out_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
  std::vector<uint32_t> in0_sender_compile_time_args = {
      // interleaved accessor args
      (std::uint32_t)in0_is_dram,

      // in0 tensor args
      (std::uint32_t)1,            // in0_tensor_stride_w
      (std::uint32_t)K,            // in0_tensor_stride_h
      (std::uint32_t)in0_block_w,  // in0_tensor_next_block_stride
      // in0 block args
      (std::uint32_t)in0_block_w,               // in0_block_w
      (std::uint32_t)per_core_M,                // in0_block_h
      (std::uint32_t)in0_block_w * per_core_M,  // in0_block_num_tiles
      // in0/in1 common args
      (std::uint32_t)K / in0_block_w,  // num_blocks
      // in0 mcast args
      (std::uint32_t)
          bottom_right_core_physical.x,  // in0_mcast_dest_noc_start_x
      (std::uint32_t)
          top_left_core_plus_one_physical.x,  // in0_mcast_dest_noc_end_x
      (std::uint32_t)in0_mcast_sender_semaphore_id,
      (std::uint32_t)in0_mcast_receiver_semaphore_id,
      (std::uint32_t)(num_cores_c - 1),  // in0_mcast_num_dests
      // batch args
      (std::uint32_t)M * K,  // MtKt
      (std::uint32_t)B       // batch
  };
  std::vector<uint32_t> in1_sender_writer_compile_time_args = {
      // interleaved accessor args
      (std::uint32_t)in1_is_dram, (std::uint32_t)out_is_dram,

      // READER
      // in1 tensor args
      (std::uint32_t)1,                // in1_tensor_stride_w
      (std::uint32_t)N,                // in1_tensor_stride_h
      (std::uint32_t)in0_block_w * N,  // in1_tensor_next_block_stride
      // in1 block args
      (std::uint32_t)per_core_N,                // in1_block_w
      (std::uint32_t)in0_block_w,               // in1_block_h
      (std::uint32_t)per_core_N * in0_block_w,  // in1_block_num_tiles
      // in0/in1 common args
      (std::uint32_t)K / in0_block_w,  // num_blocks
      // in1 mcast args
      (std::uint32_t)
          bottom_right_core_physical.y,  // in1_mcast_dest_noc_start_y
      (std::uint32_t)
          top_left_core_plus_one_physical.y,  // in1_mcast_dest_noc_end_y
      (std::uint32_t)in1_mcast_sender_semaphore_id,
      (std::uint32_t)in1_mcast_receiver_semaphore_id,
      (std::uint32_t)(num_cores_r - 1),  // in1_mcast_num_dests
      // batch args
      (std::uint32_t)K * N,        // KtNt
      (std::uint32_t)B,            // batch
      (std::uint32_t)bcast_batch,  // bcast_B

      // WRITER
      // out tensor args
      (std::uint32_t)1,                   // out_tensor_stride_w
      (std::uint32_t)N,                   // out_tensor_stride_h
      (std::uint32_t)out_subblock_w,      // out_tensor_next_subblock_stride_w
      (std::uint32_t)out_subblock_h * N,  // out_tensor_next_subblock_stride_h
      // out subblock args
      (std::uint32_t)out_subblock_w,  // out_subblock_w
      (std::uint32_t)out_subblock_h,  // out_subblock_h
      (std::uint32_t)(out_subblock_w *
                      out_subblock_h),  // out_subblocks_w * out_subblocks_h
      // batch args
      (std::uint32_t)M * N  // MtNt
  };
  if (bias_buffer != nullptr) {
    // in3 mcast args
    in1_sender_writer_compile_time_args.push_back((std::uint32_t)in3_is_dram);
    // in1 tensor args
    in1_sender_writer_compile_time_args.push_back((std::uint32_t)1);
    in1_sender_writer_compile_time_args.push_back(
        (std::uint32_t)
            bottom_right_core_physical.y);  // in1_mcast_dest_noc_start_y
    in1_sender_writer_compile_time_args.push_back(
        (std::uint32_t)
            top_left_core_plus_one_physical.y);  // in1_mcast_dest_noc_end_y
    in1_sender_writer_compile_time_args.push_back(
        (std::uint32_t)in1_mcast_sender_semaphore_id);
    in1_sender_writer_compile_time_args.push_back(
        (std::uint32_t)in1_mcast_receiver_semaphore_id);
    in1_sender_writer_compile_time_args.push_back(
        (std::uint32_t)(num_cores_r - 1));  // in1_mcast_num_dests
  }
  std::vector<uint32_t> in0_receiver_compile_time_args = {
      // in0 block args
      (std::uint32_t)in0_block_w * per_core_M,  // in0_block_num_tiles
      // in0/in1 common args
      (std::uint32_t)K / in0_block_w,  // num_blocks
      // in0 mcast args
      (std::uint32_t)top_left_core_physical.x,  // in0_mcast_sender_noc_x
      (std::uint32_t)in0_mcast_sender_semaphore_id,
      (std::uint32_t)in0_mcast_receiver_semaphore_id,
      // batch args
      (std::uint32_t)B  // batch
  };
  std::vector<uint32_t> in1_receiver_writer_compile_time_args = {
      // interleaved accessor args
      (std::uint32_t)out_is_dram,

      // READER
      // in1 block args
      (std::uint32_t)per_core_N * in0_block_w,  // in1_block_num_tiles
      // in0/in1 common args
      (std::uint32_t)K / in0_block_w,  // num_blocks
      // in1 mcast args
      (std::uint32_t)top_left_core_physical.y,  // in1_mcast_sender_noc_y
      (std::uint32_t)in1_mcast_sender_semaphore_id,
      (std::uint32_t)in1_mcast_receiver_semaphore_id,
      // batch args
      (std::uint32_t)B,  // batch

      // WRITER
      // out tensor args
      (std::uint32_t)1,                   // out_tensor_stride_w
      (std::uint32_t)N,                   // out_tensor_stride_h
      (std::uint32_t)out_subblock_w,      // out_tensor_next_subblock_stride_w
      (std::uint32_t)out_subblock_h * N,  // out_tensor_next_subblock_stride_h
      // out subblock args
      (std::uint32_t)out_subblock_w,  // out_subblock_w
      (std::uint32_t)out_subblock_h,  // out_subblock_h
      (std::uint32_t)(out_subblock_w *
                      out_subblock_h),  // out_subblocks_w * out_subblocks_h
      // batch args
      (std::uint32_t)M * N  // MtNt
  };
  if (bias_buffer != nullptr) {
    // in3 mcast args
    in1_receiver_writer_compile_time_args.push_back((std::uint32_t)per_core_N);
    in1_receiver_writer_compile_time_args.push_back(
        (std::uint32_t)top_left_core_physical.y);  // in1_mcast_sender_noc_y
    in1_receiver_writer_compile_time_args.push_back(
        (std::uint32_t)in3_mcast_sender_semaphore_id);
    in1_receiver_writer_compile_time_args.push_back(
        (std::uint32_t)in3_mcast_receiver_semaphore_id);
  }

  std::map<string, string> mm_kernel_defines;
  std::map<string, string> mm_kernel_in1_sender_writer_defines;
  std::map<string, string> mm_kernel_in1_receiver_writer_defines;
  std::map<string, string>
      mm_kernel_in1_receiver_writer_other_noc_setup_defines;
  if (bias_buffer != nullptr) {
    mm_kernel_defines["FUSE_BIAS"] = "1";
    mm_kernel_in1_sender_writer_defines["FUSE_BIAS"] = "1";
    mm_kernel_in1_receiver_writer_defines["FUSE_BIAS"] = "1";
    mm_kernel_in1_receiver_writer_other_noc_setup_defines["FUSE_BIAS"] = "1";
  }

  auto mm_kernel_in0_sender_id = tt_metal::CreateKernel(
      program,
      "tests/tt_metal/tt_metal/perf_microbenchmark/old/matmul/kernels/"
      "reader_bmm_tile_layout_in0_sender_padding.cpp",
      left_column,
      tt_metal::DataMovementConfig{
          .processor = tt_metal::DataMovementProcessor::RISCV_1,
          .noc = tt_metal::NOC::RISCV_0_default,
          .compile_args = in0_sender_compile_time_args});

  auto mm_kernel_in1_sender_writer_id = tt_metal::CreateKernel(
      program,
      "tests/tt_metal/tt_metal/perf_microbenchmark/old/matmul/kernels/"
      "reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
      top_row,
      tt_metal::DataMovementConfig{
          .processor = tt_metal::DataMovementProcessor::RISCV_0,
          .noc = tt_metal::NOC::RISCV_1_default,
          .compile_args = in1_sender_writer_compile_time_args,
          .defines = mm_kernel_in1_sender_writer_defines});

  auto mm_kernel_in1_receiver_writer_id = tt_metal::CreateKernel(
      program,
      "tests/tt_metal/tt_metal/perf_microbenchmark/old/matmul/kernels/"
      "reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
      /* in0_sender_in1_receiver, // If not using half-half noc setup */
      (CoreRangeSet)(std::set<CoreRange>){in0_sender_in1_receiver,
                                          in0_receiver_in1_receiver_left_half},
      tt_metal::DataMovementConfig{
          .processor = tt_metal::DataMovementProcessor::RISCV_0,
          .noc = tt_metal::NOC::RISCV_1_default,
          .compile_args = in1_receiver_writer_compile_time_args,
          .defines = mm_kernel_in1_receiver_writer_defines});

  auto mm_kernel_in0_receiver_id = tt_metal::CreateKernel(
      program,
      "tests/tt_metal/tt_metal/perf_microbenchmark/old/matmul/kernels/"
      "reader_bmm_tile_layout_in0_receiver.cpp",
      /* in0_receiver_in1_sender, // If not using half-half noc setup */
      (CoreRangeSet)(std::set<CoreRange>){in0_receiver_in1_sender,
                                          in0_receiver_in1_receiver_left_half},
      tt_metal::DataMovementConfig{
          .processor = tt_metal::DataMovementProcessor::RISCV_1,
          .noc = tt_metal::NOC::RISCV_0_default,
          .compile_args = in0_receiver_compile_time_args});

  KernelHandle mm_kernel_in1_receiver_writer_other_noc_setup_id = 0;
  KernelHandle mm_kernel_in0_receiver_other_noc_setup_id = 0;

  if (split_half) {
    mm_kernel_in1_receiver_writer_other_noc_setup_id =
        tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/old/matmul/kernels/"
            "reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
            in0_receiver_in1_receiver_right_half,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = in1_receiver_writer_compile_time_args,
                .defines =
                    mm_kernel_in1_receiver_writer_other_noc_setup_defines});

    mm_kernel_in0_receiver_other_noc_setup_id =
        tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/old/matmul/kernels/"
            "reader_bmm_tile_layout_in0_receiver.cpp",
            in0_receiver_in1_receiver_right_half,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
                .compile_args = in0_receiver_compile_time_args});
  }
  /* Checkerboard logic
  auto mm_kernel_in0_receiver_ckb_white = tt_metal::CreateKernel(
      program,
      "tests/tt_metal/tt_metal/perf_microbenchmark/matmul/kernels/reader_bmm_tile_layout_in0_receiver.cpp",
      in0_receiver_in1_receiver_ckb_white,
      in0_receiver_compile_time_args,
      tt_metal::DataMovementProcessor::RISCV_1,
      tt_metal::NOC::RISCV_1_default);

  auto mm_kernel_in1_receiver_writer_ckb_white =
  tt_metal::CreateKernel( program,
      "tests/tt_metal/tt_metal/perf_microbenchmark/matmul/kernels/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
      in0_receiver_in1_receiver_ckb_white,
      in1_receiver_writer_compile_time_args,
      tt_metal::DataMovementProcessor::RISCV_0,
      tt_metal::NOC::RISCV_0_default);

  auto mm_kernel_in0_receiver_ckb_black = tt_metal::CreateKernel(
      program,
      "tests/tt_metal/tt_metal/perf_microbenchmark/matmul/kernels/reader_bmm_tile_layout_in0_receiver.cpp",
      in0_receiver_in1_receiver_ckb_black,
      in0_receiver_compile_time_args,
      tt_metal::DataMovementProcessor::RISCV_1,
      tt_metal::NOC::RISCV_0_default);

  auto mm_kernel_in1_receiver_writer_ckb_black =
  tt_metal::CreateKernel( program,
      "tests/tt_metal/tt_metal/perf_microbenchmark/matmul/kernels/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
      in0_receiver_in1_receiver_ckb_black,
      in1_receiver_writer_compile_time_args,
      tt_metal::DataMovementProcessor::RISCV_0,
      tt_metal::NOC::RISCV_1_default);
  */

  /* Uncomment if we don't checkerboard
  auto mm_kernel_checkerboard_in0_receiver = tt_metal::CreateKernel(
      program,
      "tests/tt_metal/tt_metal/perf_microbenchmark/matmul/kernels/reader_bmm_tile_layout_in0_receiver.cpp",
      in0_receiver_in1_receiver,
      reader_writer_compile_time_args,
      tt_metal::DataMovementProcessor::RISCV_1,
      tt_metal::NOC::RISCV_1_default);

  auto mm_kernel_checkerboard_in1_receiver_writer =
  tt_metal::CreateKernel( program,
      "tests/tt_metal/tt_metal/perf_microbenchmark/matmul/kernels/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
      in0_receiver_in1_receiver,
      reader_writer_compile_time_args,
      tt_metal::DataMovementProcessor::RISCV_0,
      tt_metal::NOC::RISCV_0_default);
  */

  // Compute kernel compile time args
  uint32_t num_blocks = (K / in0_block_w);

  uint32_t in0_num_subblocks = (per_core_M / out_subblock_h);
  uint32_t in0_block_num_tiles =
      out_subblock_h * in0_block_w * in0_num_subblocks;
  uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

  uint32_t in1_num_subblocks = (per_core_N / out_subblock_w);
  uint32_t in1_block_num_tiles =
      out_subblock_w * in0_block_w * in1_num_subblocks;
  uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

  uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

  vector<uint32_t> compute_kernel_args = {
      in0_block_w,             // in0_block_w
      in0_num_subblocks,       // in0_num_subblocks
      in0_block_num_tiles,     // in0_block_num_tiles
      in0_subblock_num_tiles,  // in0_subblock_num_tiles

      in1_num_subblocks,    // in1_num_subblocks
      in1_block_num_tiles,  // in1_block_num_tiles
      in1_per_core_w,       // in1_per_core_w

      num_blocks,  // num_blocks

      out_subblock_h,          // out_subblock_h
      out_subblock_w,          // out_subblock_w
      out_subblock_num_tiles,  // out_subblock_num_tiles
      B                        // batch
  };

  // Create compute kernel
  bool fp32_dest_acc_en = false;
  // Gelu currently has better accuracy when run in approx mode
  bool math_approx_mode = false;
  auto mm_kernel = tt_metal::CreateKernel(
      program,
      "tests/tt_metal/tt_metal/perf_microbenchmark/old/matmul/kernels/"
      "bmm_large_block_zm_fused_bias_activation.cpp",
      all_cores,
      tt_metal::ComputeConfig{.math_fidelity = math_fidelity,
                              .fp32_dest_acc_en = fp32_dest_acc_en,
                              .math_approx_mode = math_approx_mode,
                              .compile_args = compute_kernel_args,
                              .defines = mm_kernel_defines});

  // Create circular buffers
  uint32_t src0_cb_index = 0;
  tt_metal::CircularBufferConfig src_cb0_config = tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
    .set_page_size(src0_cb_index, in0_single_tile_size);
  auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src_cb0_config);

  uint32_t src1_cb_index = 1;
  tt_metal::CircularBufferConfig src_cb1_config = tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}})
    .set_page_size(src1_cb_index, in1_single_tile_size);
  auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, src_cb1_config);

  uint32_t output_cb_index = 16;  // output operands start at index 16
  uint32_t interm0_cb_index = 24;
  std::map<uint8_t, tt::DataFormat> interim_and_out_data_format_spec = {
    {output_cb_index, output_data_format},
    {interm0_cb_index, output_data_format}
  };
  tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(out_CB_size, interim_and_out_data_format_spec)
    .set_page_size(output_cb_index, output_single_tile_size)
    .set_page_size(interm0_cb_index, output_single_tile_size);
  auto cb_output = tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}), cb_output_config);

  // CB for bias
  if (bias_buffer != nullptr) {
    uint32_t src3_cb_index = 3;
    tt_metal::CircularBufferConfig cb_src3_config = tt_metal::CircularBufferConfig(in3_CB_size, {{src3_cb_index, bias_data_format}})
        .set_page_size(src3_cb_index, bias_single_tile_size);
    auto cb_src3 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src3_config);

    uint32_t interm1_cb_index = 25;
    tt_metal::CircularBufferConfig cb_interm1_config = tt_metal::CircularBufferConfig(interm1_CB_size, {{interm1_cb_index, output_data_format}})
        .set_page_size(interm1_cb_index, output_single_tile_size);
    auto cb_interm1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_interm1_config);
  }

  // Parameters for last row, col, or block
  uint32_t last_block_h = M % per_core_M == 0 ? per_core_M : M % per_core_M;
  uint32_t last_block_w = N % per_core_N == 0 ? per_core_N : N % per_core_N;
  uint32_t last_block_num_nonzero_subblocks_h =
      (last_block_h - 1) / out_subblock_h + 1;
  uint32_t last_block_num_nonzero_subblocks_w =
      (last_block_w - 1) / out_subblock_w + 1;
  uint32_t last_subblock_of_last_block_h = last_block_h % out_subblock_h == 0
                                               ? out_subblock_h
                                               : last_block_h % out_subblock_h;
  uint32_t last_subblock_of_last_block_w = last_block_w % out_subblock_w == 0
                                               ? out_subblock_w
                                               : last_block_w % out_subblock_w;
  uint32_t last_block_padded_subblock_tiles_addr_skip =
      output_single_tile_size *
      (out_subblock_w - last_subblock_of_last_block_w);
  uint32_t last_block_padded_block_tiles_w_skip =
      (out_subblock_w * out_subblock_h) *
      (per_core_N / out_subblock_w - last_block_num_nonzero_subblocks_w);
  uint32_t last_block_padded_block_tiles_h_skip =
      (per_core_M / out_subblock_h - last_block_num_nonzero_subblocks_h) *
      (per_core_N * out_subblock_h);

  std::vector<KernelHandle> reader_kernel_ids;
  std::vector<KernelHandle> writer_kernel_ids;
  for (int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
    for (int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
      CoreCoord core = {(std::size_t)start_core_x + core_idx_x,
                        (std::size_t)start_core_y + core_idx_y};
      CoreCoord left_core = {(std::size_t)start_core_x, (std::size_t)core.y};
      CoreCoord left_core_plus_one = {(std::size_t)start_core_x + 1,
                                      (std::size_t)core.y};
      CoreCoord right_core = {(std::size_t)start_core_x + num_cores_c - 1,
                              (std::size_t)core.y};
      CoreCoord top_core = {(std::size_t)core.x, (std::size_t)start_core_y};
      CoreCoord top_core_plus_one = {(std::size_t)core.x,
                                     (std::size_t)start_core_y + 1};
      CoreCoord bottom_core = {(std::size_t)core.x,
                               (std::size_t)start_core_y + num_cores_r - 1};

      auto left_core_physical =
          device->worker_core_from_logical_core(left_core);
      auto left_core_plus_one_physical =
          device->worker_core_from_logical_core(left_core_plus_one);
      auto right_core_physical =
          device->worker_core_from_logical_core(right_core);
      auto top_core_physical = device->worker_core_from_logical_core(top_core);
      auto top_core_plus_one_physical =
          device->worker_core_from_logical_core(top_core_plus_one);
      auto bottom_core_physical =
          device->worker_core_from_logical_core(bottom_core);

      // in0 sender and in1 sender
      if (core_idx_x == 0 and core_idx_y == 0) {
        std::vector<uint32_t> mm_in0_sender_args = {
            // in0 tensor args
            (std::uint32_t)in0_buffer->address(),
            (std::uint32_t)K * per_core_M *
                core_idx_y,  // in0_tensor_start_tile_id
            // in0 mcast args
            (std::uint32_t)right_core_physical.y,  // in0_mcast_dest_noc_start_y
            (std::uint32_t)
                left_core_plus_one_physical.y,  // in0_mcast_dest_noc_end_y

            // padding args
            (std::uint32_t)per_core_M  // last_block_h
        };
        std::vector<uint32_t> mm_in1_sender_writer_args = {
            // READER
            // in1 tensor args
            (std::uint32_t)in1_buffer->address(),
            (std::uint32_t)per_core_N * core_idx_x,  // in1_tensor_start_tile_id
            // in1 mcast args
            (std::uint32_t)
                bottom_core_physical.x,  // in1_mcast_dest_noc_start_x
            (std::uint32_t)
                top_core_plus_one_physical.x,  // in1_mcast_dest_noc_end_x

            // WRITER
            // out tensor args
            (std::uint32_t)out_buffer->address(),
            (std::uint32_t)core_idx_x * per_core_N +
                core_idx_y * per_core_M * N,  // out_tensor_start_tile_id

            // padding args (READER)
            (std::uint32_t)per_core_N,  // last_block_w
            // padding args (WRITER)
            (std::uint32_t)per_core_M / out_subblock_h,
            (std::uint32_t)out_subblock_h, (std::uint32_t)0,
            (std::uint32_t)per_core_N / out_subblock_w,
            (std::uint32_t)out_subblock_w, (std::uint32_t)0, (std::uint32_t)0};

        if (bias_buffer != nullptr) {
          mm_in1_sender_writer_args.push_back(
              (std::uint32_t)bias_buffer->address());
          mm_in1_sender_writer_args.push_back(
              (std::uint32_t)per_core_N *
              core_idx_x);  // in1_tensor_start_tile_id
          mm_in1_sender_writer_args.push_back(
              (std::uint32_t)
                  bottom_core_physical.x);  // in1_mcast_dest_noc_start_x
          mm_in1_sender_writer_args.push_back(
              (std::uint32_t)
                  top_core_plus_one_physical.x);  // in1_mcast_dest_noc_end_x
        }

        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_sender_id, core,
                                 mm_in0_sender_args);  // RISCV_0_default
        tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core,
                                 mm_in1_sender_writer_args);  // RISCV_1_default
        reader_kernel_ids.push_back(mm_kernel_in0_sender_id);
        writer_kernel_ids.push_back(mm_kernel_in1_sender_writer_id);
      }
      // in0 sender and in1 receiver
      else if (core_idx_x == 0 and core_idx_y != 0) {
        std::vector<uint32_t> mm_in0_sender_args = {
            // in0 tensor args
            (std::uint32_t)in0_buffer->address(),
            (std::uint32_t)K * per_core_M *
                core_idx_y,  // in0_tensor_start_tile_id
            // in0 mcast args
            (std::uint32_t)right_core_physical.y,  // in0_mcast_dest_noc_start_y
            (std::uint32_t)
                left_core_plus_one_physical.y  // in0_mcast_dest_noc_end_y
        };

        std::vector<uint32_t> mm_in1_receiver_writer_args = {
            // READER
            // in1 mcast args
            (std::uint32_t)top_core_physical.x,  // in1_mcast_sender_noc_x

            // WRITER
            // out tensor args
            (std::uint32_t)out_buffer->address(),
            (std::uint32_t)core_idx_x * per_core_N +
                core_idx_y * per_core_M * N  // out_tensor_start_tile_id
        };

        if (core_idx_y == num_cores_r - 1) {
          // padding args (READER)
          mm_in0_sender_args.push_back(last_block_h);  // last_block_h

          // padding args (WRITER)
          mm_in1_receiver_writer_args.push_back(
              last_block_num_nonzero_subblocks_h);
          mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_h);
          mm_in1_receiver_writer_args.push_back(
              last_block_padded_block_tiles_h_skip);
          mm_in1_receiver_writer_args.push_back(per_core_N / out_subblock_w);
          mm_in1_receiver_writer_args.push_back(out_subblock_w);
          mm_in1_receiver_writer_args.push_back(0);
          mm_in1_receiver_writer_args.push_back(0);
        } else {
          // padding args (READER)
          mm_in0_sender_args.push_back(per_core_M);

          // padding args (WRITER)
          mm_in1_receiver_writer_args.push_back(per_core_M / out_subblock_h);
          mm_in1_receiver_writer_args.push_back(out_subblock_h);
          mm_in1_receiver_writer_args.push_back(0);
          mm_in1_receiver_writer_args.push_back(per_core_N / out_subblock_w);
          mm_in1_receiver_writer_args.push_back(out_subblock_w);
          mm_in1_receiver_writer_args.push_back(0);
          mm_in1_receiver_writer_args.push_back(0);
        }

        if (bias_buffer != nullptr) {
          mm_in1_receiver_writer_args.push_back(
              (std::uint32_t)top_core_physical.x);  // in1_mcast_sender_noc_x
        }

        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_sender_id, core,
                                 mm_in0_sender_args);  // RISCV_0_default
        tt_metal::SetRuntimeArgs(
            program, mm_kernel_in1_receiver_writer_id, core,
            mm_in1_receiver_writer_args);  // RISCV_1_default
        reader_kernel_ids.push_back(mm_kernel_in0_sender_id);
        writer_kernel_ids.push_back(mm_kernel_in1_receiver_writer_id);
      }
      // in0 receiver and in 1 sender
      else if (core_idx_x != 0 and core_idx_y == 0) {
        std::vector<uint32_t> mm_in0_receiver_args = {
            // in0 mcast args
            (std::uint32_t)left_core_physical.y  // in0_mcast_sender_noc_y
        };
        std::vector<uint32_t> mm_in1_sender_writer_args = {
            // READER
            // in1 tensor args
            (std::uint32_t)in1_buffer->address(),
            (std::uint32_t)per_core_N * core_idx_x,  // in1_tensor_start_tile_id
            // in1 mcast args
            (std::uint32_t)
                bottom_core_physical.x,  // in1_mcast_dest_noc_start_x
            (std::uint32_t)
                top_core_plus_one_physical.x,  // in1_mcast_dest_noc_end_x

            // WRITER
            // out tensor args
            (std::uint32_t)out_buffer->address(),
            (std::uint32_t)core_idx_x * per_core_N +
                core_idx_y * per_core_M * N  // out_tensor_start_tile_id
        };

        if (core_idx_x == num_cores_c - 1) {
          // padding args (READER)
          mm_in1_sender_writer_args.push_back(last_block_w);

          // padding args (WRITER)
          mm_in1_sender_writer_args.push_back(per_core_M / out_subblock_h);
          mm_in1_sender_writer_args.push_back(out_subblock_h);
          mm_in1_sender_writer_args.push_back(0);
          mm_in1_sender_writer_args.push_back(
              last_block_num_nonzero_subblocks_w);
          mm_in1_sender_writer_args.push_back(last_subblock_of_last_block_w);
          mm_in1_sender_writer_args.push_back(
              last_block_padded_subblock_tiles_addr_skip);
          mm_in1_sender_writer_args.push_back(
              last_block_padded_block_tiles_w_skip);
        } else {
          // padding args (READER)
          mm_in1_sender_writer_args.push_back(per_core_N);

          // padding args (WRITER)
          mm_in1_sender_writer_args.push_back(per_core_M / out_subblock_h);
          mm_in1_sender_writer_args.push_back(out_subblock_h);
          mm_in1_sender_writer_args.push_back(0);
          mm_in1_sender_writer_args.push_back(per_core_N / out_subblock_w);
          mm_in1_sender_writer_args.push_back(out_subblock_w);
          mm_in1_sender_writer_args.push_back(0);
          mm_in1_sender_writer_args.push_back(0);
        }

        if (bias_buffer != nullptr) {
          mm_in1_sender_writer_args.push_back(
              (std::uint32_t)bias_buffer->address());
          mm_in1_sender_writer_args.push_back(
              (std::uint32_t)per_core_N *
              core_idx_x);  // in1_tensor_start_tile_id
          mm_in1_sender_writer_args.push_back(
              (std::uint32_t)
                  bottom_core_physical.x);  // in1_mcast_dest_noc_start_x
          mm_in1_sender_writer_args.push_back(
              (std::uint32_t)
                  top_core_plus_one_physical.x);  // in1_mcast_dest_noc_end_x
        }
        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_receiver_id, core,
                                 mm_in0_receiver_args);  // RISCV_1_default
        tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core,
                                 mm_in1_sender_writer_args);  // RISCV_0_default
        reader_kernel_ids.push_back(mm_kernel_in0_receiver_id);
        writer_kernel_ids.push_back(mm_kernel_in1_sender_writer_id);
      }
      // in0 receiver and in 1 receiver
      else {
        std::vector<uint32_t> mm_in0_receiver_args = {
            // in0 mcast args
            (std::uint32_t)left_core_physical.y  // in0_mcast_sender_noc_y
        };
        std::vector<uint32_t> mm_in1_receiver_writer_args = {
            // READER
            // in1 mcast args
            (std::uint32_t)top_core_physical.x,  // in1_mcast_sender_noc_x

            // WRITER
            // out tensor args
            (std::uint32_t)out_buffer->address(),  // out_tensor_addr
            (std::uint32_t)core_idx_x * per_core_N +
                core_idx_y * per_core_M * N  // out_tensor_start_tile_id
        };

        if (core_idx_x == num_cores_c - 1 and core_idx_y == num_cores_r - 1) {
          // padding args (WRITER)
          mm_in1_receiver_writer_args.push_back(
              last_block_num_nonzero_subblocks_h);
          mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_h);
          mm_in1_receiver_writer_args.push_back(
              last_block_padded_block_tiles_h_skip);
          mm_in1_receiver_writer_args.push_back(
              last_block_num_nonzero_subblocks_w);
          mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_w);
          mm_in1_receiver_writer_args.push_back(
              last_block_padded_subblock_tiles_addr_skip);
          mm_in1_receiver_writer_args.push_back(
              last_block_padded_block_tiles_w_skip);
        } else if (core_idx_y == num_cores_r - 1) {
          // padding args (WRITER)
          mm_in1_receiver_writer_args.push_back(
              last_block_num_nonzero_subblocks_h);
          mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_h);
          mm_in1_receiver_writer_args.push_back(
              last_block_padded_block_tiles_h_skip);
          mm_in1_receiver_writer_args.push_back(per_core_N / out_subblock_w);
          mm_in1_receiver_writer_args.push_back(out_subblock_w);
          mm_in1_receiver_writer_args.push_back(0);
          mm_in1_receiver_writer_args.push_back(0);
        } else if (core_idx_x == num_cores_c - 1) {
          // padding args (WRITER)
          mm_in1_receiver_writer_args.push_back(per_core_M / out_subblock_h);
          mm_in1_receiver_writer_args.push_back(out_subblock_h);
          mm_in1_receiver_writer_args.push_back(0);
          mm_in1_receiver_writer_args.push_back(
              last_block_num_nonzero_subblocks_w);
          mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_w);
          mm_in1_receiver_writer_args.push_back(
              last_block_padded_subblock_tiles_addr_skip);
          mm_in1_receiver_writer_args.push_back(
              last_block_padded_block_tiles_w_skip);
        } else {
          // padding args (WRITER)
          mm_in1_receiver_writer_args.push_back(per_core_M / out_subblock_h);
          mm_in1_receiver_writer_args.push_back(out_subblock_h);
          mm_in1_receiver_writer_args.push_back(0);
          mm_in1_receiver_writer_args.push_back(per_core_N / out_subblock_w);
          mm_in1_receiver_writer_args.push_back(out_subblock_w);
          mm_in1_receiver_writer_args.push_back(0);
          mm_in1_receiver_writer_args.push_back(0);
        }

        if (bias_buffer != nullptr) {
          mm_in1_receiver_writer_args.push_back(
              (std::uint32_t)top_core_physical.x);  // in1_mcast_sender_noc_x
        }

        // left half
        if (core_idx_x <= half_core) {
          tt_metal::SetRuntimeArgs(program, mm_kernel_in0_receiver_id, core,
                                   mm_in0_receiver_args);
          tt_metal::SetRuntimeArgs(program, mm_kernel_in1_receiver_writer_id,
                                   core, mm_in1_receiver_writer_args);
          reader_kernel_ids.push_back(mm_kernel_in0_receiver_id);
          writer_kernel_ids.push_back(mm_kernel_in1_receiver_writer_id);
        }
        // right half
        else {
          tt_metal::SetRuntimeArgs(program,
                                   mm_kernel_in0_receiver_other_noc_setup_id,
                                   core, mm_in0_receiver_args);
          tt_metal::SetRuntimeArgs(
              program, mm_kernel_in1_receiver_writer_other_noc_setup_id, core,
              mm_in1_receiver_writer_args);
          reader_kernel_ids.push_back(
              mm_kernel_in0_receiver_other_noc_setup_id);
          writer_kernel_ids.push_back(
              mm_kernel_in1_receiver_writer_other_noc_setup_id);
        }
        /* Checkerboard logic
        // white
        if ((core_idx_x + core_idx_y) % 2 == 0) {
            tt_metal::SetRuntimeArgs(mm_kernel_in0_receiver_ckb_white, core,
        mm_in0_receiver_args); // RISCV_1_default
            tt_metal::SetRuntimeArgs(mm_kernel_in1_receiver_writer_ckb_white,
        core, mm_in1_receiver_writer_args); // RISCV_0_default
        }
        // black
        else {
            tt_metal::SetRuntimeArgs(mm_kernel_in0_receiver_ckb_black, core,
        mm_in0_receiver_args); // RISCV_1_default
            tt_metal::SetRuntimeArgs(mm_kernel_in1_receiver_writer_ckb_black,
        core, mm_in1_receiver_writer_args); // RISCV_0_default
        }
        */

        /* Uncomment if we don't checkerboard
        tt_metal::SetRuntimeArgs(mm_kernel_checkerboard_in0_receiver, core,
        mm_in0_receiver_args); // RISCV_1_default
        tt_metal::SetRuntimeArgs(mm_kernel_checkerboard_in1_receiver_writer,
        core, mm_in1_receiver_writer_args); // RISCV_0_default
        */
      }
    }
  }
  return std::move(program);
}

// Given a tensor that is row-major datums, make it tilized
// so that its row major within a tile, and each tile's data
// is contiguous
template <typename T>
std::vector<T> tilize(std::vector<T> data, int rows, int cols) {
  TT_FATAL(rows % 32 == 0, "Error");
  TT_FATAL(cols % 32 == 0, "Error");
  int num_tiles_r = rows / 32;
  int num_tiles_c = cols / 32;
  std::vector<T> result;
  for (auto r = 0; r < num_tiles_r; r++) {
    for (auto c = 0; c < num_tiles_c; c++) {
      for (auto j = 0; j < 32; j++) {    // tile rows
        for (auto i = 0; i < 32; i++) {  // tile cols
          // each row of tiles is 32x32 * num_tiles_c
          // each row within the row of tiles is cols
          // each col of tiles is 32
          // pick row of tiles, pick the row within the tile, pick col tile
          int index = r * 32 * 32 * num_tiles_c + j * cols + c * 32 + i;
          result.push_back(data.at(index));
        }
      }
    }
  }
  return result;
}

// Given a tilized data (each tile's data is contiguous and row major within the
// tile) transform it back to row major full tensor. (This function inverts the
// tilize() function)
template <typename T>
std::vector<T> untilize(std::vector<T> data, int rows, int cols) {
  TT_FATAL(rows % 32 == 0, "Error");
  TT_FATAL(cols % 32 == 0, "Error");
  int num_tiles_r = rows / 32;
  int num_tiles_c = cols / 32;
  std::vector<T> result;
  for (auto r = 0; r < num_tiles_r; r++) {
    for (auto i = 0; i < 32; i++) {
      for (auto c = 0; c < num_tiles_c; c++) {
        int offset = r * 32 * 32 * num_tiles_c + c * 32 * 32 + i * 32;
        for (auto j = 0; j < 32; j++) {
          result.push_back(data.at(offset + j));
        }
      }
    }
  }

  return result;
}

std::vector<bfloat16> select_columns(std::vector<bfloat16> data, int M, int K,
                                     int N) {
  if (N == K) {
    return data;
  }
  std::vector<bfloat16> result;
  if (N > K) {
    for (int i = 0; i < M * 32; i++) {
      for (int j = 0; j < K * 32; j++) {
        int offset = i * K * 32;
        result.push_back(data.at(offset + j));
      }
      for (int j = 0; j < (N - K) * 32; j++) {
        result.push_back((float)0);
      }
    }
  } else {
    for (int i = 0; i < M * 32; i++) {
      for (int j = 0; j < N * 32; j++) {
        int offset = i * K * 32;
        result.push_back(data.at(offset + j));
      }
    }
  }
  return result;
}

int main(int argc, char** argv) {
  if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
    TT_THROW("Test not supported w/ slow dispatch, exiting");
  }

  bool pass = true;
  try {
    ////////////////////////////////////////////////////////////////////////////
    //                      Initial Runtime Args Parse
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::string> input_args(argv, argv + argc);
    uint32_t dprint;
    uint32_t print_tensor;
    uint32_t debug;
    uint32_t Mt;
    uint32_t Nt;
    uint32_t Kt;
    uint32_t num_cores_y;
    uint32_t num_cores_x;
    uint32_t in0_block_w;
    uint32_t out_subblock_h;
    uint32_t out_subblock_w;
    uint32_t per_core_Mt;
    uint32_t per_core_Nt;
    uint32_t l1_in0;
    uint32_t l1_in1;
    uint32_t l1_out;
    uint32_t validation;
    try {
      std::tie(debug, input_args) =
          test_args::get_command_option_uint32_and_remaining_args(input_args,
                                                                  "--debug", 0);
      std::tie(dprint, input_args) =
          test_args::get_command_option_uint32_and_remaining_args(
              input_args, "--dprint", 0);
      std::tie(print_tensor, input_args) =
          test_args::get_command_option_uint32_and_remaining_args(
              input_args, "--print_tensor", 0);
      std::tie(Mt, input_args) =
          test_args::get_command_option_uint32_and_remaining_args(input_args,
                                                                  "--mt", 72);
      std::tie(Nt, input_args) =
          test_args::get_command_option_uint32_and_remaining_args(input_args,
                                                                  "--nt", 96);
      std::tie(Kt, input_args) =
          test_args::get_command_option_uint32_and_remaining_args(input_args,
                                                                  "--kt", 24);
      std::tie(num_cores_y, input_args) =
          test_args::get_command_option_uint32_and_remaining_args(input_args,
                                                                  "--r", 9);
      std::tie(num_cores_x, input_args) =
          test_args::get_command_option_uint32_and_remaining_args(input_args,
                                                                  "--c", 12);
      std::tie(in0_block_w, input_args) =
          test_args::get_command_option_uint32_and_remaining_args(
              input_args, "--in0_block_w", 4);
      std::tie(out_subblock_h, input_args) =
          test_args::get_command_option_uint32_and_remaining_args(
              input_args, "--out_subblock_h", 4);
      std::tie(out_subblock_w, input_args) =
          test_args::get_command_option_uint32_and_remaining_args(
              input_args, "--out_subblock_w", 2);
      std::tie(per_core_Mt, input_args) =
          test_args::get_command_option_uint32_and_remaining_args(
              input_args, "--per_core_mt", 8);
      std::tie(per_core_Nt, input_args) =
          test_args::get_command_option_uint32_and_remaining_args(
              input_args, "--per_core_nt", 8);
      std::tie(l1_in0, input_args) =
          test_args::get_command_option_uint32_and_remaining_args(
              input_args, "--l1_in0", 0);
      std::tie(l1_in1, input_args) =
          test_args::get_command_option_uint32_and_remaining_args(
              input_args, "--l1_in1", 0);
      std::tie(l1_out, input_args) =
          test_args::get_command_option_uint32_and_remaining_args(
              input_args, "--l1_out", 0);
      std::tie(validation, input_args) =
          test_args::get_command_option_uint32_and_remaining_args(
              input_args, "--validation", 1);
    } catch (const std::exception& e) {
      TT_THROW("Command line arguments found exception",
                e.what());
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    int device_id = 0;
    tt_metal::Device* device = tt_metal::CreateDevice(device_id);

    ////////////////////////////////////////////////////////////////////////////
    //                      Inputs Setup
    ////////////////////////////////////////////////////////////////////////////
    if (debug) {
      log_info(LogTest, "row {} x col {} = {} cores", num_cores_y, num_cores_x,
               num_cores_y * num_cores_x);
      log_info(LogTest, "in0_block_w {}", in0_block_w);
      log_info(LogTest, "out_subblock_h {}", out_subblock_h);
      log_info(LogTest, "out_subblock_w {}", out_subblock_w);
      log_info(LogTest, "per_core_mt {}", per_core_Mt);
      log_info(LogTest, "per_core_nt {}", per_core_Nt);
      log_info(LogTest, "l1_in0 {}", l1_in0);
      log_info(LogTest, "l1_in1 {}", l1_in1);
      log_info(LogTest, "l1_out {}", l1_out);
    }

    log_info(LogTest, "Mt = {}, Nt = {}, Kt = {}", Mt, Nt, Kt);
    log_info(LogTest, "activations = {}x{}", Mt * 32, Kt * 32);
    log_info(LogTest, "weights = {}x{}", Kt * 32, Nt * 32);
    log_info(LogTest, "output = {}x{}", Mt * 32, Nt * 32);

    tt::DataFormat data_format = tt::DataFormat::Float16_b;
    uint32_t single_tile_size = 2 * 1024;

    // buffer creation
    uint32_t in0_buffer_size = single_tile_size * Mt * Kt;
    uint32_t in1_buffer_size = single_tile_size * Kt * Nt;
    uint32_t out_buffer_size = single_tile_size * Mt * Nt;
    BufferType in0_buffer_type =
        (l1_in0 == 0) ? (BufferType::DRAM) : (BufferType::L1);
    BufferType in1_buffer_type =
        (l1_in1 == 0) ? (BufferType::DRAM) : (BufferType::L1);
    BufferType out_buffer_type =
        (l1_out == 0) ? (BufferType::DRAM) : (BufferType::L1);

    tt_metal::InterleavedBufferConfig in0_config{
                    .device=device,
                    .size = in0_buffer_size,
                    .page_size = single_tile_size,
                    .buffer_type = in0_buffer_type
    };
    tt_metal::InterleavedBufferConfig in1_config{
                    .device=device,
                    .size = in1_buffer_size,
                    .page_size = single_tile_size,
                    .buffer_type = in1_buffer_type
    };
    tt_metal::InterleavedBufferConfig out_config{
                    .device=device,
                    .size = out_buffer_size,
                    .page_size = single_tile_size,
                    .buffer_type = out_buffer_type
    };

    auto in0_buffer = CreateBuffer(in0_config);
    auto in1_buffer = CreateBuffer(in1_config);
    auto out_buffer = CreateBuffer(out_config);

    SHAPE in0_shape = {1, 1, Mt * 32, Kt * 32};
    tt::deprecated::Tensor<bfloat16> tensor =
        tt::deprecated::initialize_tensor<bfloat16>(
            in0_shape, tt::deprecated::Initialize::RANDOM, 100,
            std::chrono::system_clock::now().time_since_epoch().count());
    auto activations_tilized = tilize(tensor.get_values(), Mt * 32, Kt * 32);
    auto activations_tile_layout = convert_to_tile_layout(activations_tilized);
    auto activations =
        pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
    tt_metal::detail::WriteToBuffer(in0_buffer, activations);

    auto identity =
        create_identity_matrix(Kt * 32, Nt * 32, std::min(Kt, Nt) * 32);
    auto identity_tilized = tilize(identity, Kt * 32, Nt * 32);
    auto weights_tile_layout = convert_to_tile_layout(identity_tilized);
    auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
    tt_metal::detail::WriteToBuffer(in1_buffer, weights);

    ////////////////////////////////////////////////////////////////////////////
    //                      Matmul Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t B = 1;
    TT_FATAL(Kt % in0_block_w == 0, "Error");

    uint32_t num_blocks_y = (Mt - 1) / per_core_Mt + 1;
    uint32_t num_blocks_x = (Nt - 1) / per_core_Nt + 1;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
    TT_FATAL(num_blocks_total <= num_cores_x * num_cores_y, "Error");
    CoreCoord core_range =
        get_core_range(num_blocks_y, num_blocks_x, num_cores_y, num_cores_x);
    TT_FATAL(num_cores_y == core_range.y && num_cores_x == core_range.x, "Error");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    TT_FATAL(core_range.x > 1 && core_range.y > 1, "Error");
    MathFidelity math_fidelity = MathFidelity::HiFi4;

    auto program = create_program_mcast_in0_in1(
        device, math_fidelity, core_range, B, Mt, Nt, Kt, false, in0_block_w,
        out_subblock_h, out_subblock_w, per_core_Mt, per_core_Nt, in0_buffer,
        in1_buffer, nullptr, out_buffer, tt::DataFormat::Float16_b,
        tt::DataFormat::Float16_b, tt::DataFormat::Float16_b,
        tt::DataFormat::Float16_b);

    std::chrono::duration<double, std::nano> duration;

    // took from run_operation.cpp
    auto start = std::chrono::high_resolution_clock::now();
    EnqueueProgram(device->command_queue(), program, false);
    Finish(device->command_queue());
    auto end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    tt_metal::DumpDeviceProfileResults(device, program);

    uint64_t num_of_matmul_ops =
        (2 * static_cast<uint64_t>(Kt) * 32 - 1) *
        (static_cast<uint64_t>(Mt) * static_cast<uint64_t>(Nt) * 1024);
    if (debug) {
      log_info(LogTest, "number of matmul ops: {}", num_of_matmul_ops);
    }

    double tflops =
        static_cast<double>(num_of_matmul_ops) / duration.count() / 1000;
    log_info(LogTest, "time duration: {} ns, TFLOPS {}", duration.count(),
             tflops);

    ////////////////////////////////////////////////////////////////////////////
    //                      Validation & Teardown
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> result_vec;
    tt_metal::detail::ReadFromBuffer(out_buffer, result_vec);
    auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
    auto result_flat_layout = convert_to_flat_layout(result_bfp16);
    auto result_untilized = untilize(result_flat_layout, Mt * 32, Nt * 32);

    auto golden = select_columns(tensor.get_values(), Mt, Kt, Nt);
    pass &= (golden == result_untilized);
    pass &= tt_metal::CloseDevice(device);

  } catch (const std::exception& e) {
    pass = false;
    // Capture the exception error message
    log_error(LogTest, "{}", e.what());
    // Capture system call errors that may have returned from driver/kernel
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
