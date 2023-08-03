#include "tt_dnn/op_library/concat/concat_op.hpp"

#include <algorithm>

#include "common/test_tiles.hpp"
#include "tensor/tensor_utils.hpp"

#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/reshape/reshape_op.hpp"
#include "tt_dnn/op_library/transpose/transpose_op.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

ConcatOpParallelizationStrategy Concat2::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) {
    // TODO: add multi-code implementation and then pick stratergy based on volume of tensors.
    return ConcatOpParallelizationStrategy::SINGLE_CORE;
}

inline operation::ProgramWithCallbacks concat2_dim3_single_core(const Tensor &a, const Tensor &b, Tensor &output) {

    constexpr u32 dim = 3;
    const auto shapeA = a.shape(), shapeB = b.shape();


    tt::tt_metal::Layout layout = a.layout();
    //both source tensor layouts should be same (layout == b.layout())
    //all dimensions of A, B tensor outside the concat dimension should be same
    u32 commonH = shapeA[2];
    u32 wA = shapeA[3];
    u32 wB = shapeB[3];

    auto shape_out = shapeA;
    shape_out[dim] += shapeB[dim];  // we only grow the output dimension
    //TT_ASSERT(output.volume() == (a.volume() + b.volume()), "output tensor is not of right size");
    tt_metal::Program program = tt_metal::Program();
    CoreRange core = {.start = {0, 0}, .end = {0, 0}};
    CoreRangeSet cs({core});

    tt_metal::Buffer *srcA_dram_buffer = a.buffer();
    tt_metal::Buffer *srcB_dram_buffer = b.buffer();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    tt_metal::Shape output_shape = output.shape();

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    bool is_srcA_dram = ( srcA_dram_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM );
    bool is_srcB_dram = ( srcB_dram_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM );
    bool is_dst_dram = ( dst_dram_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM );

    if (layout == tt::tt_metal::Layout::TILE) {
        tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
        std::uint32_t tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
        std::uint32_t num_tiles_A = a.volume() / TILE_HW;
        std::uint32_t num_tiles_B = b.volume() / TILE_HW;
        std::uint32_t num_tiles = num_tiles_A + num_tiles_B;

        const std::uint32_t cb_num_tiles = 2;
        const std::uint32_t cb_size = cb_num_tiles * tile_size;
        constexpr std::uint32_t cb_id_0 = 0;
        constexpr std::uint32_t cb_id_out_0 = 16;

        const CircularBuffer &cb0 = tt_metal::CreateCircularBuffers(
            program, cb_id_0, cs, cb_num_tiles, cb_size, src0_cb_data_format);

        const CircularBuffer &cb_out0 = tt_metal::CreateCircularBuffers(
            program, cb_id_out_0, cs, cb_num_tiles, cb_size, src0_cb_data_format);

        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t) is_srcA_dram, //are tensors on DRAM or in L1 memory ?
            (std::uint32_t) is_srcB_dram,
        };

        // read contents of tensors A, B, into CB ID 0 one tile at a time.
        tt_metal::KernelID concat_read_kernel_id = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/concat1d_reader.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

        tt_metal::SetRuntimeArgs(
            program,
            concat_read_kernel_id,
            core,
            {2,  // 2-buffers
             a.shape()[0] * a.shape()[1] * a.shape()[2] / TILE_HEIGHT, // Number of rows is equal for all tensors for concat last dim
             cb_id_0,
             srcA_dram_buffer->address(),
             a.shape()[3] / TILE_WIDTH,
             srcB_dram_buffer->address(),
             b.shape()[3] / TILE_WIDTH});

        // write the contents of CB ID OUT 0 into the destination buffer for output tensor.
        std::vector<uint32_t> writer_compile_time_args = {
            (std::uint32_t) cb_id_out_0,
            (std::uint32_t) is_dst_dram
        };

        tt_metal::KernelID concat_write_kernel_id = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

        tt_metal::SetRuntimeArgs(
            program,
            concat_write_kernel_id,
            core,
            {dst_dram_buffer->address(),
             num_tiles, 0});

        // eltwise copy kernel
        auto per_core_tile_cnt = num_tiles;
        std::vector<std::uint32_t> compute_kernel_args = {per_core_tile_cnt};
        auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/eltwise_copy.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args});

        auto override_runtime_args_callback = [core, concat_read_kernel_id, concat_write_kernel_id](
                                                  const tt_metal::Program &program,
                                                  const std::vector<Buffer *> &input_buffers,
                                                  const std::vector<Buffer *> &output_buffers) {
            auto srcA_dram_buffer = input_buffers.at(0);
            auto srcB_dram_buffer = input_buffers.at(1);
            auto dst_0_dram_buffer = output_buffers.at(0);

            {
                auto runtime_args = GetRuntimeArgs(program, concat_read_kernel_id, core.start);
                runtime_args[3] = srcA_dram_buffer->address();
                runtime_args[5] = srcB_dram_buffer->address();

                tt_metal::SetRuntimeArgs(program, concat_read_kernel_id, core, runtime_args);
            }

            {
                auto runtime_args = GetRuntimeArgs(program, concat_write_kernel_id, core.start);
                runtime_args[0] = dst_0_dram_buffer->address();
                SetRuntimeArgs(program, concat_write_kernel_id, core, runtime_args);
            }

            auto runtime_args = GetRuntimeArgs(program, concat_write_kernel_id, core.start);
        };
        return {std::move(program), override_runtime_args_callback};
    } else {
        TT_ASSERT(false && "ROW MAJOR, or other layouts not supported for concat right now!");
    }
    return {std::move(program)};
}

///////////////// Concat2 operator implementation ////////////////////////

// Concat new operator world-view.
void Concat2::validate(const std::vector<Tensor> &input_tensors) const {
    TT_ASSERT(dim == 3, "concat2 will always work on dim3");
    TT_ASSERT(input_tensors.size() == 2 && "only 2 supported now");
    tt::tt_metal::Shape shape_first = input_tensors[0].shape();
    shape_first[dim] = 0;

    for (const Tensor &in_ref : input_tensors) {
        TT_ASSERT((in_ref.layout() == Layout::TILE) && "Only tile layout supported.");
        TT_ASSERT(in_ref.device() && "Operand to concat needs to be on device.");
        TT_ASSERT(in_ref.buffer() && "Operand to concat needs to be allocated in a buffer on device.");
        TT_ASSERT(in_ref.layout() == input_tensors.at(0).layout() && "All Tensors should have same layouts.");
        tt::tt_metal::Shape curr_shape = in_ref.shape();
        curr_shape[dim] = 0;
        TT_ASSERT(curr_shape == shape_first, "concat tensors differ in shape across non-concat dimensions.");
    }
}

std::vector<tt::tt_metal::Shape> Concat2::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    tt::tt_metal::Shape shape_out = input_tensors[0].shape();
    shape_out[dim] = 0;
    for (const Tensor &in_ref : input_tensors) {
        tt::tt_metal::Shape curr_shape = in_ref.shape();
        shape_out[dim] += curr_shape[dim];
    }
    return {shape_out};
}

std::vector<Tensor> Concat2::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const Tensor &ref_in_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, ref_in_tensor.dtype(), Layout::TILE, this->output_mem_config);
}

tt::stl::reflection::Attributes Concat2::attributes() const {
    return {
        {"dim", this->dim},
    };
}

operation::ProgramWithCallbacks Concat2::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    return concat2_dim3_single_core(input_tensors[0], input_tensors[1], output_tensors[0]);
}

///////////////////////// API functions //////////////////////////
Tensor concat(Tensor &input_tensor_a, Tensor &input_tensor_b, uint32_t dim, const MemoryConfig& output_mem_config) {
    TT_ASSERT(input_tensor_a.shape()[dim] % TILE_WIDTH == 0, "Does not support padding on concat dim");
    TT_ASSERT(input_tensor_b.shape()[dim] % TILE_WIDTH == 0, "Does not support padding on concat dim");
    if (dim != 3) {
        input_tensor_a = transpose(input_tensor_a, dim, 3, output_mem_config);
        input_tensor_b = transpose(input_tensor_b, dim, 3, output_mem_config);
    }
    Tensor output = operation::run_with_autoformat(Concat2{3}, {input_tensor_a, input_tensor_b}).at(0);
    if (dim != 3) {
        return transpose(output, dim, 3, output_mem_config);
    }
    return output;
}

// note: all tensors are expected to be in same layout.
Tensor concat(std::vector<Tensor> &in_t, uint32_t dim, const MemoryConfig& output_mem_config) {
    TT_ASSERT(in_t.size() > 0, "need 1 or more tensors");
    Tensor result(in_t.at(0));
    for (int idx = 1; idx < in_t.size(); idx++) {
        TT_ASSERT((in_t.at(0).layout() == in_t.at(idx).layout()) && "Layout of all input tensors should be identical.");
    }
    for (int idx = 1; idx < in_t.size(); idx++) {
        result = concat(result, in_t.at(idx), dim, output_mem_config);
    }
    return result;
}

}  // namespace tt_metal
}  // namespace tt
