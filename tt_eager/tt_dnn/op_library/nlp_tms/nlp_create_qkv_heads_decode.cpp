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

operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads_decode(const Tensor &input_tensor, const uint32_t num_q_heads, const uint32_t num_kv_heads, const uint32_t head_dim, std::vector<Tensor>& output, CoreCoord compute_with_storage_grid_size) {

    tt_metal::Program program = tt_metal::CreateProgram();

    const auto& input_shape = input_tensor.get_legacy_shape();

    tt_metal::Device *device = input_tensor.device();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    uint32_t head_tiles = head_dim / TILE_WIDTH;
    uint32_t head_size = head_tiles * single_tile_size;

    auto q_shard_spec = output[0].shard_spec().value();
    auto q_cores = q_shard_spec.grid;
    auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / TILE_HW;

    uint32_t per_core_out_q_heads = num_q_heads / q_cores.num_cores();
    uint32_t per_risc0_out_q_heads = div_up(per_core_out_q_heads, 2);
    uint32_t per_risc1_out_q_heads = per_core_out_q_heads / 2;
    uint32_t per_core_in_q_heads = num_q_heads / input_tensor.shard_spec().value().num_cores();

    uint32_t q_output_cb_index = CB::c_out0;
    tt_metal::CircularBufferConfig cb_q_output_config =
        tt_metal::CircularBufferConfig(
            q_num_tiles * single_tile_size, {{q_output_cb_index, cb_data_format}})
            .set_page_size(q_output_cb_index, single_tile_size).set_globally_allocated_address(*output[0].buffer());
    auto cb_q_output = tt_metal::CreateCircularBuffer(program, q_cores, cb_q_output_config);

    auto k_shard_spec = output[1].shard_spec().value();
    auto k_cores = k_shard_spec.grid;
    auto k_num_tiles = k_shard_spec.shape[0] * k_shard_spec.shape[1] / TILE_HW;

    uint32_t k_output_cb_index = CB::c_out1;
    tt_metal::CircularBufferConfig cb_k_output_config =
        tt_metal::CircularBufferConfig(
            k_num_tiles * single_tile_size, {{k_output_cb_index, cb_data_format}})
            .set_page_size(k_output_cb_index, single_tile_size).set_globally_allocated_address(*output[1].buffer());
    auto cb_k_output = tt_metal::CreateCircularBuffer(program, k_cores, cb_k_output_config);

    auto v_shard_spec = output[0].shard_spec().value();
    auto v_cores = q_shard_spec.grid;
    auto v_num_tiles = v_shard_spec.shape[0] * v_shard_spec.shape[1] / TILE_HW;

    uint32_t v_output_cb_index = CB::c_out2;
    tt_metal::CircularBufferConfig cb_v_output_config =
        tt_metal::CircularBufferConfig(
            v_num_tiles * single_tile_size, {{v_output_cb_index, cb_data_format}})
            .set_page_size(v_output_cb_index, single_tile_size).set_globally_allocated_address(*output[2].buffer());
    auto cb_v_output = tt_metal::CreateCircularBuffer(program, v_cores, cb_v_output_config);

    uint32_t per_core_out_kv_heads = num_kv_heads / k_cores.num_cores();
    uint32_t per_core_in_kv_heads = num_kv_heads / input_tensor.shard_spec().value().num_cores();

    uint32_t q_base_addr = input_tensor.buffer()->address();
    uint32_t k_base_addr = q_base_addr + per_core_in_q_heads * head_tiles * single_tile_size;
    uint32_t v_base_addr = k_base_addr + per_core_in_kv_heads * head_tiles * single_tile_size;

    std::vector<uint32_t> reader_compile_time_args = {
        q_output_cb_index,
        k_output_cb_index
    };
    std::vector<uint32_t> writer_compile_time_args = {
        q_output_cb_index,
        v_output_cb_index
    };
    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/nlp_tms/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp",
        q_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/nlp_tms/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp",
        q_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    uint32_t num_cores = std::max(q_cores.num_cores(), k_cores.num_cores());

    auto core_grid = q_cores.bounding_box();
    uint32_t num_cores_x = core_grid.end.x + 1, num_cores_y = core_grid.end.y + 1;
    uint32_t num_kv_cores = k_cores.num_cores();

    const auto &cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, true);

    std::vector<uint32_t> noc_x_coords;
    noc_x_coords.reserve(num_cores_x);
    for (uint32_t x = 0; x < num_cores_x; ++x) {
        noc_x_coords.push_back(device->worker_core_from_logical_core({x, 0}).x);
    }
    std::vector<uint32_t> noc_y_coords;
    noc_y_coords.reserve(num_cores_y);
    for (uint32_t y = 0; y < num_cores_y; ++y) {
        noc_y_coords.push_back(device->worker_core_from_logical_core({0, y}).y);
    }

    uint32_t remote_q_head_start_idx = 0;
    uint32_t remote_kv_head_start_idx = 0;
    uint32_t q_x = 0, q_y = 0, kv_x = 0, kv_y = 0;
    uint32_t q_start_addr = q_base_addr;
    uint32_t k_start_addr = k_base_addr;
    uint32_t v_start_addr = v_base_addr;

    uint32_t remote_q_read = 0;
    uint32_t remote_kv_read = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        bool read_kv_heads = i < k_cores.num_cores();
        std::vector<uint32_t> reader_runtime_args;
        reader_runtime_args.reserve(18 + num_cores_x + num_cores_y);
        reader_runtime_args = {
            head_size,
            per_risc0_out_q_heads,
            per_core_in_q_heads,
            remote_q_head_start_idx,
            q_x,
            q_y,
            q_base_addr,
            q_start_addr,
            0,
            read_kv_heads,
            per_core_out_kv_heads,
            per_core_in_kv_heads,
            remote_kv_head_start_idx,
            kv_x,
            kv_y,
            k_base_addr,
            k_start_addr,
            k_num_tiles,
            num_cores_x,
        };
        reader_runtime_args.insert(reader_runtime_args.end(), noc_x_coords.begin(), noc_x_coords.end());
        reader_runtime_args.insert(reader_runtime_args.end(), noc_y_coords.begin(), noc_y_coords.end());

        remote_q_read += per_risc0_out_q_heads;
        q_y = (remote_q_read / per_core_in_q_heads) / num_cores_x;
        q_x = (remote_q_read / per_core_in_q_heads) % num_cores_x;
        remote_q_head_start_idx = (remote_q_head_start_idx + per_risc0_out_q_heads) % per_core_in_q_heads;
        q_start_addr = q_base_addr + remote_q_head_start_idx * head_size;

        tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);

        reader_runtime_args[1] = per_risc1_out_q_heads;
        reader_runtime_args[3] = remote_q_head_start_idx;
        reader_runtime_args[4] = q_x;
        reader_runtime_args[5] = q_y;
        reader_runtime_args[7] = q_start_addr;
        reader_runtime_args[8] = per_risc0_out_q_heads * head_size;

        if (per_risc1_out_q_heads > 0) {
            remote_q_read += per_risc1_out_q_heads;
            q_y = (remote_q_read / per_core_in_q_heads) / num_cores_x;
            q_x = (remote_q_read / per_core_in_q_heads) % num_cores_x;
            remote_q_head_start_idx = (per_risc1_out_q_heads + remote_q_head_start_idx) % per_core_in_q_heads;
            q_start_addr = q_base_addr + remote_q_head_start_idx * head_size;
        }

        if (read_kv_heads) {
            reader_runtime_args[15] = v_base_addr;
            reader_runtime_args[16] = v_start_addr;
            remote_kv_read += per_core_out_kv_heads;
            kv_y = (remote_kv_read / per_core_in_kv_heads) / num_cores_x;
            kv_x = (remote_kv_read / per_core_in_kv_heads) % num_cores_x;
            remote_kv_head_start_idx = (remote_kv_head_start_idx + per_core_out_kv_heads) % per_core_in_kv_heads;
            k_start_addr = k_base_addr + remote_kv_head_start_idx * head_size;
            v_start_addr = v_base_addr + remote_kv_head_start_idx * head_size;
        }

        tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, reader_runtime_args);
    }

    auto override_runtime_arguments_callback = [
            reader_kernel_id,
            writer_kernel_id,
            num_cores,
            num_cores_y,
            cb_q_output,
            cb_k_output,
            cb_v_output,
            cores,
            head_size,
            per_risc0_out_q_heads,
            per_risc1_out_q_heads,
            per_core_in_q_heads,
            per_core_out_kv_heads,
            per_core_in_kv_heads,
            head_tiles,
            num_kv_cores,
            single_tile_size
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

        auto dst_buffer_query = output_tensors.at(0).buffer();
        auto dst_buffer_key = output_tensors.at(1).buffer();
        auto dst_buffer_value = output_tensors.at(2).buffer();

        UpdateDynamicCircularBufferAddress(program, cb_q_output, *dst_buffer_query);
        UpdateDynamicCircularBufferAddress(program, cb_k_output, *dst_buffer_key);
        UpdateDynamicCircularBufferAddress(program, cb_v_output, *dst_buffer_value);

        uint32_t q_base_addr = input_tensors[0].buffer()->address();
        uint32_t k_base_addr = q_base_addr + per_core_in_q_heads * head_tiles * single_tile_size;
        uint32_t v_base_addr = k_base_addr + per_core_in_kv_heads * head_tiles * single_tile_size;

        uint32_t remote_q_head_start_idx = 0;
        uint32_t remote_kv_head_start_idx = 0;
        uint32_t q_start_addr = q_base_addr;
        uint32_t k_start_addr = k_base_addr;
        uint32_t v_start_addr = v_base_addr;

        for (uint32_t i = 0; i < num_cores; ++i) {
            const auto& core = cores[i];
            bool read_kv_heads = i < num_kv_cores;
            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[6] = q_base_addr;
                runtime_args[7] = q_start_addr;
                runtime_args[15] = k_base_addr;
                runtime_args[16] = k_start_addr;
                remote_q_head_start_idx = (remote_q_head_start_idx + per_risc0_out_q_heads) % per_core_in_q_heads;
                q_start_addr = q_base_addr + remote_q_head_start_idx * head_size;
            }
            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[6] = q_base_addr;
                runtime_args[15] = v_base_addr;
                if (per_risc1_out_q_heads > 0) {
                    runtime_args[7] = q_start_addr;
                    remote_q_head_start_idx = (remote_q_head_start_idx + per_risc1_out_q_heads) % per_core_in_q_heads;
                    q_start_addr = q_base_addr + remote_q_head_start_idx * head_size;
                }
                if (read_kv_heads) {
                    runtime_args[16] = v_start_addr;
                    remote_kv_head_start_idx = (remote_kv_head_start_idx + per_core_out_kv_heads) % per_core_in_kv_heads;
                    k_start_addr = k_base_addr + remote_kv_head_start_idx * head_size;
                    v_start_addr = v_base_addr + remote_kv_head_start_idx * head_size;
                }
            }
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

} // namespace tt_metal

} // namespace tt
