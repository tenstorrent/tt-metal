// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/reshape/reshape_op.hpp"
#include "tt_dnn/op_library/copy/copy_op.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include <tt_eager/tt_numpy/functions.hpp>
#include <tt_eager/tensor/tensor_impl.hpp>

#include "tensor/tensor_utils.hpp"

#include <algorithm>

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks reshape_tile_single_core(const Tensor &a, Tensor &output, int N, int C, int H, int W) {

    tt_metal::Program program = tt_metal::CreateProgram();

    CoreRange core({0, 0}, {0, 0});

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    tt_metal::Buffer *src0_buffer = a.buffer();

    uint32_t num_tiles = a.volume() / TILE_HW;

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    Shape output_shape = output.get_legacy_shape();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t) src0_is_dram};

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) src0_cb_index,
        (std::uint32_t) dst_is_dram
    };

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/reshape/kernels/dataflow/reader_unary_reshape_interleaved.cpp",
        core,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        {src0_buffer->address(),
        a.get_legacy_shape()[3] / TILE_WIDTH,
        (uint32_t) output_shape[0],
        (uint32_t) output_shape[1],
        (uint32_t) output_shape[2] / TILE_HEIGHT,
        (uint32_t) output_shape[3] / TILE_WIDTH }
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        {dst_buffer->address(),
        num_tiles, 0 }
    );

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);

        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks reshape_rm_single_core(const Tensor &a, Tensor& output, int N, int C, int H, int W) {

    tt_metal::Program program = tt_metal::CreateProgram();
    CoreRange core({0, 0}, {0, 0});

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    Shape output_shape = output.get_legacy_shape();
    tt_metal::Buffer *src0_buffer = a.buffer();
    tt_metal::Buffer *dst_buffer = output.buffer();

    uint32_t num_old_sticks = a.get_legacy_shape()[0] * a.get_legacy_shape()[1] * a.get_legacy_shape()[2];
    uint32_t num_new_sticks = output_shape[0] * output_shape[1] * output_shape[2];

    uint32_t old_stick_size = a.get_legacy_shape()[3] * 2; // Assuming bfloat16 data format
    uint32_t new_stick_size = output_shape[3] * 2; // Assuming bfloat16 data format

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = (a.get_legacy_shape()[1] * a.get_legacy_shape()[2] * a.get_legacy_shape()[3] / TILE_HW);
    uint32_t num_output_tiles = (output_shape[1] * output_shape[2] * output_shape[3] / TILE_HW);

    // Currently added to support Bert large, TODO: Make op more generic, parallelize
    uint32_t available_l1 = device->l1_size_per_core() - L1_UNRESERVED_BASE;
    if (num_input_tiles * single_tile_size + num_output_tiles * single_tile_size > available_l1) {
        if (old_stick_size >= new_stick_size) {
            if (old_stick_size % new_stick_size == 0) {
                // Maximize L1 usage. Is this needed or do we just need to double buffer 32 sticks (64)
                // Evenly divide L1 between input/output
                uint32_t w_tiles = a.get_legacy_shape()[3] / TILE_WIDTH;
                num_input_tiles = ((available_l1 / 2) / single_tile_size) / w_tiles * w_tiles;
                num_output_tiles = num_input_tiles;
            } else {
                // Not needed for Bert large at the moment so will trigger L1 OOM assert
            }
        } else {
            if (new_stick_size % old_stick_size == 0) {
                // Maximize L1 usage. Is this needed or do we just need to double buffer 32 sticks (64)
                // Evenly divide L1 between input/output
                uint32_t w_tiles = (output_shape[3] / TILE_WIDTH);
                num_output_tiles = ((available_l1 / 2) / single_tile_size) / w_tiles * w_tiles;
                num_input_tiles = num_output_tiles;
            } else {
                // Not needed for Bert large at the moment so will trigger L1 OOM assert
            }
        }
        TT_ASSERT(num_input_tiles > 0 && num_output_tiles > 0, "Cannot fit input/output rows into L1");
    }

    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t output_cb_index = 16; // output operands start at index 16
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
		.set_page_size(output_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    // Reader compile-time args
    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool old_stick_size_is_power_of_two = is_power_of_two_at_least_32(old_stick_size);
    vector<uint32_t> reader_kernel_args = {src0_buffer->address(), num_old_sticks, old_stick_size};
    std::vector<uint32_t> reader_compile_time_args = {src0_is_dram};
    if (old_stick_size_is_power_of_two) {
        reader_kernel_args.push_back(log2(old_stick_size));

        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        reader_compile_time_args.push_back(1);
    } else {
        reader_compile_time_args.push_back(0);
    }

    // Writer compile-time args
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool new_stick_size_is_power_of_two = is_power_of_two_at_least_32(new_stick_size);
    vector<uint32_t> writer_kernel_args = {dst_buffer->address(), num_new_sticks, new_stick_size};
    std::vector<uint32_t> writer_compile_time_args {dst_is_dram};
    if (new_stick_size_is_power_of_two) {
        writer_kernel_args.push_back(log2(new_stick_size));
        writer_compile_time_args.push_back(1);
    } else {
        writer_compile_time_args.push_back(0);
    }

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/reshape/kernels/dataflow/reader_unary_reshape_stick_layout_interleaved.cpp",
        core,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/reshape/kernels/dataflow/writer_unary_reshape_stick_layout_interleaved.cpp",
        core,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // No compute required, so using blank kernel
    vector<uint32_t> compute_args = {
        uint(a.volume() / TILE_HW), // per_core_block_cnt
        1 // per_core_block_size
    };

    auto eltwise_unary_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/compute/eltwise_copy.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_args}
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        reader_kernel_args
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        writer_kernel_args
    );

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);

        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

void Reshape::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to reshape need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr , "Operands to reshape need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.get_dtype() == DataType::BFLOAT16);

    TT_FATAL(input_tensor_a.get_layout() == Layout::TILE || input_tensor_a.get_layout() == Layout::ROW_MAJOR, "Only tile and row major reshape supported!");

    auto output_shape = infer_dims_for_reshape(this->N, this->C, this->H, this->W, input_tensor_a.volume());
    TT_FATAL(input_tensor_a.volume() == output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3], "New shape volume must match old shape volume");

    TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Reshape does not currently support sharding");
    TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED, "Reshape does not currently support sharding");

    if (input_tensor_a.get_layout() == Layout::TILE) {
        TT_FATAL(input_tensor_a.volume() % TILE_HW == 0);
        TT_FATAL(output_shape[2] % TILE_HEIGHT == 0 && output_shape[3] % TILE_WIDTH == 0 && "Expected a multiple of 32 for H, W (or -1 evaluating to such) for reshape!");
    } else if (input_tensor_a.get_layout() == Layout::ROW_MAJOR) {
        TT_FATAL(input_tensor_a.get_legacy_shape()[3] % TILE_WIDTH == 0 && output_shape[3] % TILE_WIDTH == 0, "Operand/target width must be a multiple of 32");
        uint32_t num_old_sticks = input_tensor_a.get_legacy_shape()[0] * input_tensor_a.get_legacy_shape()[1] * input_tensor_a.get_legacy_shape()[2];
        uint32_t num_new_sticks = output_shape[0] * output_shape[1] * output_shape[2];
        TT_FATAL(num_old_sticks % TILE_HEIGHT == 0 && num_new_sticks % TILE_HEIGHT == 0, "Operand/target number of rows must be a multiple of 32");
    } else {
        TT_FATAL(false, "Unsupported layout for reshape");
    }
}

std::vector<Shape> Reshape::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    return {infer_dims_for_reshape(this->N, this->C, this->H, this->W, input_tensor_a.volume())};
}

std::vector<Tensor> Reshape::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor_a.get_dtype(), input_tensor_a.get_layout(), this->output_mem_config);
}

operation::ProgramWithCallbacks Reshape::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    if (input_tensor_a.get_layout() == Layout::ROW_MAJOR) {
        return {reshape_rm_single_core(input_tensor_a, output_tensor, this->N, this->C, this->H, this->W)};
    } else if (input_tensor_a.get_layout() == Layout::TILE) {
        return {reshape_tile_single_core(input_tensor_a, output_tensor, this->N, this->C, this->H, this->W)};
    } else {
        TT_ASSERT(false, "Unsupported layout for reshape");
        return {};
    }
}

tt::stl::reflection::Attributes Reshape::attributes() const {
    return {
        {"N", this->N},
        {"C", this->C},
        {"H", this->H},
        {"W", this->W},
        {"output_mem_config", this->output_mem_config},
    };
}

Tensor reshape (const Tensor &input_tensor_a, int N, int C, int H, int W, const MemoryConfig& output_mem_config) {
    // No-op (Will do a tensor copy)
    auto output_shape = infer_dims_for_reshape(N, C, H, W, input_tensor_a.volume());
    if (
        ((input_tensor_a.get_layout() == Layout::TILE or input_tensor_a.get_layout() == Layout::ROW_MAJOR) && output_shape[3] == input_tensor_a.get_legacy_shape()[3])
    ) {
        // Don't need to do a check here to see the H and W both divisible by 32
        // since handled within the tensor reshape method
        return input_tensor_a.reshape(N, C, H, W);
    }
    if (input_tensor_a.get_legacy_shape() == output_shape) {
        return AutoFormat::move_tensor_to_mem_config(input_tensor_a, output_mem_config);
    }
    if (input_tensor_a.get_layout() == Layout::ROW_MAJOR && ((compute_volume(output_shape) / output_shape[-1]) % TILE_HEIGHT != 0 || output_shape[-1] % TILE_WIDTH != 0 || input_tensor_a.get_legacy_shape()[-1] % TILE_WIDTH != 0 || (input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1]) % TILE_HEIGHT != 0)) {
        TT_FATAL(input_tensor_a.get_dtype()==DataType::BFLOAT16);
        return tt::numpy::manual_insertion<bfloat16>(input_tensor_a, output_shape, DataType::BFLOAT16, Layout::ROW_MAJOR, input_tensor_a.device(), output_mem_config);
    }
    return operation::run_without_autoformat(Reshape{N, C, H, W, output_mem_config}, {input_tensor_a}).at(0);
}

} // namespace tt_metal
} // namespace tt
