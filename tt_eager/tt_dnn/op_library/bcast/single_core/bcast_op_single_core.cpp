// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"


using namespace tt::tt_metal;
using namespace tt::constants;


namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks bcast_single_core(const Tensor &a, const Tensor &b, const Tensor& output, BcastOpMath bcast_math, BcastOpDim bcast_dim) {

    const auto ashape = a.get_legacy_shape();
    const auto bshape = b.get_legacy_shape();
    uint32_t N  = ashape[0], C  = ashape[1], H  = ashape[2], W  = ashape[3];
    uint32_t bN = bshape[0], bC = bshape[1], bH = bshape[2], bW = bshape[3];
    uint32_t NC = N*C;
    uint32_t HW = H*W;

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = NC*Ht*Wt;
    uint32_t num_btensor_tiles = NC*bH*bW / TILE_HW;

    tt_metal::Program program = tt_metal::CreateProgram();

    CoreRange core({0, 0}, {0, 0});

    auto src0_buffer = a.buffer();
	auto src1_buffer = b.buffer();
	auto dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());

    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig src0_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
	auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, src0_cb_config);

	uint32_t src1_cb_index = 1;
	tt_metal::CircularBufferConfig src1_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
		.set_page_size(src1_cb_index, single_tile_size);
	auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, src1_cb_config);

	uint32_t output_cb_index = 16; // output operands start at index 16
	uint32_t num_output_tiles = 2;
	tt_metal::CircularBufferConfig output_cb_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
		.set_page_size(output_cb_index, single_tile_size);
	auto cb_output = tt_metal::CreateCircularBuffer(program, core, output_cb_config);

    uint32_t bnc1 = (bN*bC == 1);

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) dst_is_dram
    };

    std::map<string, string> reader_defines;
	std::map<string, string> bcast_compute_defines = bcast_op_utils::get_defines(bcast_dim, bcast_math);

    if(bcast_dim == BcastOpDim::HW && bnc1) {
		reader_defines["BCAST_SCALAR"] = "1";
		bcast_compute_defines["BCAST_SCALAR"] = "1";
	}

    const char* reader_name = bcast_op_utils::get_reader_name(bcast_dim, BcastOpParallelizationStrategy::SINGLE_CORE);
    KernelHandle binary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        reader_name,
        core,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    const char* compute_name = bcast_op_utils::get_compute_name(bcast_dim);

    auto bcast_kernel_id = tt_metal::CreateKernel(
        program,
        compute_name,
        core,
        tt_metal::ComputeConfig{.compile_args = {}, .defines = bcast_compute_defines}
    );

    tt_metal::SetRuntimeArgs(
        program,
        binary_reader_kernel_id,
        core,
        {
            a.buffer()->address(), // 0
            0, // 1
            0, // 2
            num_tensor_tiles, // 3
            b.buffer()->address(), // 4
            0, // 5
            0, // 6
            num_btensor_tiles, NC*Ht*Wt, NC, Ht, Wt, bnc1  // 7 8 9 10 11 12
        }
    );


    tt_metal::SetRuntimeArgs(
        program,
        bcast_kernel_id,
        core,
        {
            NC, // B
            Ht, // Ht
            Wt  // Wt
        }
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        {
            output.buffer()->address(),
            num_tensor_tiles, 0
        }
    );

    auto override_runtime_arguments_callback = [
            binary_reader_kernel_id,
            unary_writer_kernel_id,
            bcast_kernel_id
        ]
    (
        const void* operation,
        const Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_dram_buffer_a = input_tensors.at(0).buffer();
        auto src_dram_buffer_b = input_tensors.at(1).buffer();

        auto dst_dram_buffer = output_tensors.at(0).buffer();

        const auto ashape = input_tensors.at(0).get_legacy_shape();
        const auto bshape = input_tensors.at(1).get_legacy_shape();
        uint32_t N  = ashape[0], C  = ashape[1], H  = ashape[2], W  = ashape[3];
        uint32_t bN = bshape[0], bC = bshape[1], bH = bshape[2], bW = bshape[3];
        uint32_t NC = N*C;
        uint32_t HW = H*W;

        uint32_t Wt = W/TILE_WIDTH;
        uint32_t Ht = H/TILE_HEIGHT;

        uint32_t num_tensor_tiles = NC*Ht*Wt;
        uint32_t num_btensor_tiles = NC*bH*bW / TILE_HW;

        uint32_t bnc1 = (bN*bC == 1);

        CoreCoord core = {0, 0};

        tt_metal::SetRuntimeArgs(
            program,
            binary_reader_kernel_id,
            core,
            {
                src_dram_buffer_a->address(), // 0
                0, // 1
                0, // 2
                num_tensor_tiles, // 3
                src_dram_buffer_b->address(), // 4
                0, // 5
                0, // 6
                num_btensor_tiles, NC*Ht*Wt, NC, Ht, Wt, bnc1  // 7 8 9 10 11 12
            }
        );


        tt_metal::SetRuntimeArgs(
            program,
            bcast_kernel_id,
            core,
            {
                NC, // B
                Ht, // Ht
                Wt  // Wt
            }
        );

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                dst_dram_buffer->address(),
                num_tensor_tiles,
                0
            }
        );
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
