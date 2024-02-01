#include <math.h>

#include "tt_dnn/op_library/upsample/upsample_op.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/math.hpp"

#include "tt_metal/tt_stl/reflection.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks upsample_single_core(const Tensor &a, Tensor& output) {

    tt_metal::Program program = tt_metal::CreateProgram();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    tt_metal::Buffer *src0_buffer = a.buffer();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    auto output_shape = output.shape();
    auto input_shape = a.shape();
    std::cout <<  "Log testing input shape: --> "  <<  input_shape[0] << " " << input_shape[1] << " " << input_shape[2] << " "  << input_shape[3]; //<< std::endl;
    std::cout <<  "Log testing output shape: --> " <<  output_shape[0] << " " << output_shape[1] << " " << output_shape[2] << " "  << output_shape[3] << std::endl;

    /*for(int i=0; i<output_shape.size(); i++) {
        std::cout << output_shape[i] << " ";
    }*/
    std::cout << std::endl;
    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);
    std::cout << "input_single_tile_size: " << input_single_tile_size << std::endl;

    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_cb_data_format);
    std::cout << "output_single_tile_size: " << output_single_tile_size << std::endl;
    std::cout <<  "Log testing output shape: _1" << std::endl;

    int32_t num_tiles = a.volume() / TILE_HW;

    auto width = a.shape()[-1];
    uint32_t stick_s =  width;
    uint32_t num_sticks = a.volume() / width;
    uint32_t stick_size = stick_s * a.element_size(); // Assuming bfloat16 dataformat
    std::cout << "num_tiles " << num_tiles << " " << stick_s << "   " << std::endl;

    uint32_t num_tiles_in_row = stick_s / TILE_WIDTH;
    // Ensure we don't intrude into storage space
    uint32_t max_l1_size = a.device()->l1_size_per_core() / 2 - L1_UNRESERVED_BASE;
    uint32_t max_tiles = max_l1_size / (input_single_tile_size + output_single_tile_size); // 2 CBs
    // Currently need the number of tiles in a row to be divisible by tiles in a block
    std::cout << "num_tiles_in_row: " << num_tiles_in_row << "max_tiles: " << max_tiles << std::endl;
    uint32_t num_tiles_per_block = 1;

        if (num_tiles_in_row <= max_tiles) {
        num_tiles_per_block = num_tiles_in_row;
    } else {
        for(uint32_t n_t = max_tiles; n_t > 0; n_t--) {
            if (num_tiles_in_row % n_t == 0) {
                num_tiles_per_block = n_t;
                break;
            }
        }
    }

    std::cout <<  "Log testing output shape: _3" << std::endl;
    uint32_t block_width_size = num_tiles_per_block * TILE_WIDTH * a.element_size();
    std::cout << "block_width_size: " << block_width_size << std::endl;
    std::cout <<  "Log testing output shape: _3.1.1" << std::endl;
    std::cout << "num_tiles_per_block --> "  << num_tiles_per_block << std::endl;
    uint32_t num_full_blocks_in_row = num_tiles_in_row / num_tiles_per_block;
        std::cout <<  "Log testing output shape: _3.1.2" << std::endl;
    uint32_t num_leftover_tiles = num_tiles_in_row % num_tiles_per_block;
        std::cout <<  "Log testing output shape: _3.1.3" << std::endl;
    uint32_t leftover_width_in_row = num_leftover_tiles * a.element_size();
    std::cout <<  "Log testing output shape: _3.1" << std::endl;
    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = num_tiles_per_block;
    std::cout <<  "Log testing output shape: _3.2" << std::endl;

    tt_metal::CircularBufferConfig src0_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * input_single_tile_size, {{src0_cb_index, input_cb_data_format}})
		.set_page_size(src0_cb_index, input_single_tile_size);
	auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, src0_cb_config);
    std::cout <<  "Log testing output shape: _3.3" << std::endl;

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = num_tiles_per_block;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * output_single_tile_size, {{output_cb_index, output_cb_data_format}})
		.set_page_size(output_cb_index, output_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);
    std::cout <<  "output_cb_index: _3.4 " << output_cb_index << std::endl;

    vector<uint32_t> reader_kernel_args = {
        src0_buffer->address(),
        num_sticks,
        stick_size,
        num_tiles_per_block,
        block_width_size,
        num_full_blocks_in_row,
        num_leftover_tiles,
        leftover_width_in_row,
        0                       // row_start_id
    };
    std::cout <<  "Log testing output shape: _4" << std::endl;
    // Reader compile-time args
    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)log2(stick_size) : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_is_dram,
        (std::uint32_t) stick_size_is_power_of_two,
        (std::uint32_t) log2_stick_size,
    };

    bool out_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) out_is_dram
    };
    std::cout <<  "Log testing output shape: 5" << std::endl;
    // Tilized reader
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/tilize/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp",
        core,
        tt_metal::ReaderDataMovementConfig{.compile_args = reader_compile_time_args});

    // Tilized writer
    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/writer_twice_interleaved_start_id.cpp",
        core,
        tt_metal::WriterDataMovementConfig{.compile_args = writer_compile_time_args});

    vector<uint32_t> compute_args = {
        uint32_t(num_tiles / num_tiles_per_block), // per_core_block_cnt
        uint32_t(num_tiles_per_block) // per_core_block_tile_cnt
    };
    std::cout <<  "Log testing output shape: 6" << std::endl;

    auto upsample_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/compute/upsample_wh.cpp",
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
        {dst_buffer->address(),
        (uint32_t) num_tiles,
        (uint32_t) 0}
    );
    std::cout <<  "Log testing output shape: 7" << std::endl;

    auto override_runtime_args_callback = [
        reader_kernel_id=unary_reader_kernel_id,
        writer_kernel_id=unary_writer_kernel_id
    ](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);

        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    };
    std::cout <<  "Log testing output shape: _11 " << std::endl;
    return {std::move(program), override_runtime_args_callback};

}

}  // namespace tt_metal

}  // namespace tt
