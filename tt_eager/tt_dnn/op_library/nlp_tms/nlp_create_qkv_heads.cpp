// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/nlp_tms/nlp_tms.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;
using namespace tt;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads(const Tensor &input_tensor, std::optional<const Tensor> input_tensor_kv, const uint32_t num_q_heads, const uint32_t num_kv_heads, const uint32_t head_dim, const bool transpose_k_heads, std::vector<Tensor>& output, CoreCoord compute_with_storage_grid_size) {

    const auto& input_shape = input_tensor.shape();

    tt_metal::Device *device = input_tensor.device();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    const bool read_from_input_tensor_kv = input_tensor_kv.has_value();

    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    tt_metal::Buffer *in0_buffer = input_tensor.buffer();
    TT_ASSERT(in0_buffer->size() % single_tile_size == 0);

    tt_metal::Buffer *in1_buffer;
    uint32_t in1_buffer_addr = 0;
    tt_metal::BufferType in1_buffer_type = tt_metal::BufferType::DRAM;
    if (read_from_input_tensor_kv) {
        in1_buffer = input_tensor_kv.value().buffer();
        TT_ASSERT(in1_buffer->size() % single_tile_size == 0);
        in1_buffer_addr = in1_buffer->address();
        in1_buffer_type = in1_buffer->buffer_type();
    }


    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t in0_w_tiles = input_shape[3] / TILE_WIDTH;
    uint32_t in1_w_tiles = 0;
    if (read_from_input_tensor_kv) {
        in1_w_tiles = input_tensor_kv.value().shape()[3] / TILE_WIDTH;
    }

    // Per output tensor args
    // Output shape for Q is: [B, num_q_heads, s, head_dim], shuffled from [B, 1, s, num_q_heads * head_dim]
    // Output shape for K/V is: [B, num_kv_heads, s, head_dim], shuffled from [B, 1, s, num_kv_heads * head_dim]
    // NOTE: Output h and w dims are identical for Q, K, V, so any arg that is related to these dims for q_* can be shared for K, V
    uint32_t q_out_h_tiles = input_shape[2] / TILE_HEIGHT;
    uint32_t q_out_w_tiles = head_dim / TILE_WIDTH; // tiles along head_dim
    uint32_t q_out_HtWt = q_out_h_tiles * q_out_w_tiles;
    uint32_t q_out_CHtWt = num_q_heads * q_out_HtWt;
    uint32_t kv_out_CHtWt = num_kv_heads * q_out_HtWt;
    uint32_t q_num_tiles = num_q_heads * q_out_w_tiles;
    uint32_t kv_num_tiles = num_kv_heads * q_out_w_tiles;

    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // Block is a unit of work; ie. num of in0_w_tiles per core
    uint32_t num_blocks = input_shape[0] * input_shape[1] * input_shape[2] / TILE_HEIGHT;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_blocks);


    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    TT_ASSERT((output.size() == 3), "Output vector must be size 3 for split fused qkv!");
    tt_metal::Tensor& q = output[0];
    tt_metal::Tensor& k = output[1];
    tt_metal::Tensor& v = output[2];

    tt_metal::Buffer *q_buffer = q.buffer();
    TT_ASSERT(q_buffer != nullptr, "Output q buffer should be allocated on device!");
    tt_metal::Buffer *k_buffer = k.buffer();
    TT_ASSERT(k_buffer != nullptr, "Output k buffer should be allocated on device!");
    tt_metal::Buffer *v_buffer = v.buffer();
    TT_ASSERT(v_buffer != nullptr, "Output v buffer should be allocated on device!");


    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::CreateProgram();

    bool tile_dtype_is_bfloat16 = input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16;
    bool in0_is_dram = in0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool in1_is_dram = false;
    if (read_from_input_tensor_kv) {
        in1_is_dram = in1_buffer_type == tt_metal::BufferType::DRAM ? 1 : 0;
    }

    // TODO: Q, K, V doesn't necessarily need to be the same output mem config
    bool out_is_dram = q_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
            // interleaved accessor args
            (std::uint32_t) in0_is_dram,
            (std::uint32_t) in1_is_dram,
            (std::uint32_t) q_num_tiles,
            (std::uint32_t) kv_num_tiles,
    };
    std::vector<uint32_t> writer_compile_time_args = {
            // interleaved accessor args
            (std::uint32_t) out_is_dram,
            (std::uint32_t) q_out_h_tiles,
            (std::uint32_t) q_out_w_tiles,
            (std::uint32_t) q_out_HtWt,
            (std::uint32_t) num_q_heads, // q_out_c
            (std::uint32_t) num_kv_heads, // kv_out_c
    };

    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;
    if (transpose_k_heads) {
        std::vector<uint32_t> compute_args_core_group_1 = {num_blocks_per_core_group_1 * kv_num_tiles};
        auto compute_kernel_id_group_1 = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/kernels/compute/transpose_wh.cpp",
            core_group_1,
            tt_metal::ComputeConfig{.compile_args = compute_args_core_group_1}
        );

        if (core_group_2.num_cores() > 0) {
            std::vector<uint32_t> compute_args_core_group_2 = {num_blocks_per_core_group_2 * kv_num_tiles};
            auto compute_kernel_id_group_2 = tt_metal::CreateKernel(
                program,
                "tt_eager/tt_dnn/kernels/compute/transpose_wh.cpp",
                core_group_2,
                tt_metal::ComputeConfig{.compile_args = compute_args_core_group_2}
            );
        }

        reader_defines["TRANSPOSE_K_HEADS"] = "1";
        writer_defines["TRANSPOSE_K_HEADS"] = "1";
    }
    if (read_from_input_tensor_kv) {
        reader_defines["READ_FROM_INPUT_TENSOR_KV"] = "1";
    }

    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig{.compile_args = reader_compile_time_args, .defines = reader_defines});
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/writer_tm_tile_layout_nlp_create_qkv_heads.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig{.compile_args = writer_compile_time_args, .defines = writer_defines});


    // Create circular buffers
    uint32_t micro_block_size = 1; // Num tiles to read/wait for in reader and writer
    uint32_t cb_num_tiles = micro_block_size * 4; // Quadruple buffer everything

    // TODO: Investigate perf allocating full in0_w_tiles with double buffer
    // uint32_t cb1_num_tiles = in0_w_tiles * 2; // double buffer; this runs out of space for generic shapes
    uint32_t src1_cb_index = 1; // cb0 is needed for compute if we want to use generic transpose_wh compute kernel
    uint32_t cb1_num_tiles = cb_num_tiles;
    tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(cb1_num_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
		.set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    // If we transpose_k_heads:
    // - reader will write to cb0, instead of cb1
    // - compute will wait on cb0 and write to cb16
    // - writer will wait on cb 16, instead of cb1
    if (transpose_k_heads) {
        uint32_t src0_cb_index = 0;
        uint32_t cb0_num_tiles = cb_num_tiles;
        tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(cb0_num_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		    .set_page_size(src0_cb_index, single_tile_size);
        auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

        uint32_t out_cb_index = 16;
        uint32_t out_cb_num_tiles = cb_num_tiles;
        tt_metal::CircularBufferConfig cb_out_config = tt_metal::CircularBufferConfig(out_cb_num_tiles * single_tile_size, {{out_cb_index, cb_data_format}})
		    .set_page_size(out_cb_index, single_tile_size);
        auto cb_out = tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);
    }

    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++){
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_blocks_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_blocks_per_core = num_blocks_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        std::vector<uint32_t> reader_runtime_args = {
            (std::uint32_t) in0_buffer->address(),
            (std::uint32_t) in1_buffer_addr,
            num_blocks_per_core,
            num_blocks_written * in0_w_tiles,
            num_blocks_written * in1_w_tiles,
        };

        uint32_t q_out_h_dim = num_blocks_written % q_out_h_tiles;
        uint32_t q_out_tensor_tile_id = num_blocks_written / q_out_h_tiles * q_out_CHtWt + q_out_h_dim * q_out_w_tiles;
        uint32_t v_out_tensor_tile_id = num_blocks_written / q_out_h_tiles * kv_out_CHtWt + q_out_h_dim * q_out_w_tiles;
        uint32_t k_out_tensor_tile_id = transpose_k_heads ? num_blocks_written / q_out_h_tiles * kv_out_CHtWt + q_out_h_dim : v_out_tensor_tile_id;

        std::vector<uint32_t> writer_runtime_args = {
            (std::uint32_t) q_buffer->address(), // q_tensor_addr
            (std::uint32_t) k_buffer->address(), // k_tensor_addr
            (std::uint32_t) v_buffer->address(), // v_tensor_addr
            num_blocks_per_core, // num_blocks
            q_out_h_dim, // q_out_h_dim
            q_out_tensor_tile_id, // q_out_tensor_tile_id
            k_out_tensor_tile_id, // k_out_tensor_tile_id
            v_out_tensor_tile_id, // v_out_tensor_tile_id
        };

        tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        num_blocks_written += num_blocks_per_core;
    }

    auto override_runtime_arguments_callback = [
            reader_kernel_id,
            writer_kernel_id,
            num_cores,
            num_cores_y,
            read_from_input_tensor_kv=read_from_input_tensor_kv
        ]
    (
        const void* operation,
        Program &program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_buffer = input_tensors.at(0).buffer();

        uint32_t src_kv_buffer_addr = 0;
        if (read_from_input_tensor_kv) {
            src_kv_buffer_addr = optional_input_tensors.at(0).value().buffer()->address();
        }

        auto dst_buffer_query = output_tensors.at(0).buffer();
        auto dst_buffer_key = output_tensors.at(1).buffer();
        auto dst_buffer_value = output_tensors.at(2).buffer();

        for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++){
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();

                if (read_from_input_tensor_kv) {
                    runtime_args[1] = src_kv_buffer_addr;
                }
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = dst_buffer_query->address();
                runtime_args[1] = dst_buffer_key->address();
                runtime_args[2] = dst_buffer_value->address();
            }
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

} // namespace tt_metal

} // namespace tt
