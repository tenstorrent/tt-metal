// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "tt_eager/tt_dnn/op_library/embeddings/embeddings_op.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/work_split.hpp"


using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

#define RISC_CORES_PER_TENSIX 2

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks embeddings_tilized(
    const Tensor &a,
    const Tensor &weights,
    Tensor & output,
    bool split_weights
){

    ////////////////////////////////////////////////////////////////////////////
    //                 Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::Buffer *a_buffer = a.buffer();
    tt_metal::Buffer *weights_buffer = weights.buffer();
    tt_metal::Buffer *out_buffer = output.buffer();


    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    Device *device = a.device();
    auto dst_addr = output.buffer()->address();

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program{};

    uint32_t num_tiles_per_cb = 1;
    bool in0_is_dram = a.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool weights_is_dram = weights.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool weights_dtype_is_bfloat16 = weights.dtype() == tt::tt_metal::DataType::BFLOAT16;
    bool out_is_dram = output.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    uint32_t last_dim = 3;
    uint32_t element_size_in_bytes = 2; // size of float
    if(!weights_dtype_is_bfloat16)
        element_size_in_bytes = 1; //bfp8_b

    // row major, page size is last dim
    uint32_t single_page_size = weights.shape()[last_dim]*element_size_in_bytes;

    // weights shape is [1, 1, num_embeddings, num_dim]
    uint32_t num_embeddings = weights.shape()[last_dim-1];

    uint32_t batch_size = a.shape()[0];
    uint32_t num_output_rows_per_batch = a.shape()[last_dim - 1];
    uint32_t num_output_rows = num_output_rows_per_batch * batch_size;
    uint32_t num_blocks = num_output_rows/TILE_HEIGHT;

    auto num_embedding_dims = weights.shape()[3];


    //setup problem and grid size
    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    uint32_t problem_size;
    if(split_weights){
        problem_size = num_embeddings;
    }
    else{
        problem_size = num_blocks;
    }

    uint32_t embedding_risc_cores_per_tensix = 1;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1,
        core_group_2, num_blocks_per_core_group_1,
        num_blocks_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, problem_size);

    tt_metal::Buffer dst_buffer_l1 = CreateBuffer(device,
                                                        sizeof(uint32_t)*2,
                                                        sizeof(uint32_t)*2,
                                                        tt_metal::BufferType::L1);

    std::vector<tt_metal::DataMovementProcessor> risc_procs = {tt_metal::DataMovementProcessor::RISCV_0, tt_metal::DataMovementProcessor::RISCV_1};
    std::vector<tt_metal::NOC> noc_ports = {tt_metal::NOC::RISCV_0_default, tt_metal::NOC::RISCV_1_default};

    //Create Buffers

    uint32_t num_tiles_per_block = weights.shape()[3] / TILE_WIDTH;
    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    uint32_t src0_cb_index = 0;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_tiles_per_block * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = 16; // output operands start at index 16
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_tiles_per_block * single_tile_size, {{output_cb_index, cb_data_format}})
		.set_page_size(output_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    bool weight_stick_size_is_power_of_two = is_power_of_two_at_least_32(single_page_size);
    uint32_t weight_log2_stick_size = weight_stick_size_is_power_of_two ? (std::uint32_t)log2(single_page_size) : 0;

    // Create Kernels
    //reader
    std::vector<uint32_t> embedding_compile_time_args= {
        (std::uint32_t) in0_is_dram,
        (std::uint32_t) weights_is_dram,
        (std::uint32_t) single_page_size,
        (std::uint32_t) dst_buffer_l1.address(),
        (std::uint32_t) num_tiles_per_block,
        (std::uint32_t) weight_stick_size_is_power_of_two,
        (std::uint32_t) weight_log2_stick_size
        };

    auto reader_kernel_id = tt_metal::CreateKernel(
                                        program,
                                        "tt_eager/tt_dnn/kernels/dataflow/embeddings_tilize.cpp",
                                        all_cores,
                                        tt_metal::DataMovementConfig{.processor = risc_procs[0],
                                                                    .noc = noc_ports[0],
                                                                    .compile_args = embedding_compile_time_args});

    if(num_blocks_per_core_group_1 > 0){
        vector<uint32_t> compute_args_1 = {
            uint32_t(num_blocks_per_core_group_1), // per_core_block_cnt
            uint32_t(num_tiles_per_block) // per_core_block_tile_cnt
        };
        auto tilize_kernel_id_1 = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/kernels/compute/tilize.cpp",
            core_group_1,
            tt_metal::ComputeConfig{.compile_args = compute_args_1}
        );
    }

    if(num_blocks_per_core_group_2 > 0){
        vector<uint32_t> compute_args_2 = {
            uint32_t(num_blocks_per_core_group_2), // per_core_block_cnt
            uint32_t(num_tiles_per_block) // per_core_block_tile_cnt
        };
        auto tilize_kernel_id_2 = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/kernels/compute/tilize.cpp",
            core_group_2,
            tt_metal::ComputeConfig{.compile_args = compute_args_2}
        );
    }

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) out_is_dram
    };

    // Tilized writer
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = risc_procs[1],
                                    .noc = noc_ports[1],
                                    .compile_args = writer_compile_time_args});




    uint32_t input_offset = 0;
    uint32_t weight_offset = 0;
    uint32_t tile_offset = 0;

    for(uint32_t idx=0; idx<num_cores_x; idx++){
        for(uint32_t idy=0; idy<num_cores_y; idy++){
            uint32_t local_num_blocks;
            uint32_t core_id = idx*num_cores_y +
                               idy;

            uint32_t local_num_embeddings;
            uint32_t local_weight_offset;
            uint32_t local_input_offset = input_offset;
            CoreCoord core(idx, idy);
            uint32_t local_problem_size = num_blocks_per_core_group_1;
            bool core_used = core_group_1.core_coord_in_core_ranges(core)
                            || core_group_2.core_coord_in_core_ranges(core);
            bool core_0 = true;
            if(core_group_2.core_coord_in_core_ranges(core)){
                local_problem_size = num_blocks_per_core_group_2;
                core_0 = false;
            }
            else{
                bool valid_core = core_used && core_group_1.core_coord_in_core_ranges(core) || (!core_used);
                TT_ASSERT(valid_core, "Not valid core");
            }
            if(core_used){
                if(split_weights){
                    local_weight_offset = weight_offset;
                    local_input_offset = input_offset;
                    local_num_blocks = num_blocks;
                    local_num_embeddings = local_problem_size * TILE_HEIGHT;

                }
                else{
                    local_input_offset = input_offset;
                    local_weight_offset = weight_offset;
                    local_num_embeddings = num_embeddings;
                    local_num_blocks = local_problem_size;
                }

                std::vector<uint32_t>runtime_args;

                //Reader
                {
                    std::vector<uint32_t> runtime_args = {
                    (std::uint32_t) local_input_offset*TILE_HEIGHT,
                    (std::uint32_t) (local_num_blocks),
                    (std::uint32_t) a.buffer()->address(),
                    (std::uint32_t) weights.buffer()->address()
                    };
                    tt_metal::SetRuntimeArgs(program, reader_kernel_id, core,runtime_args);
                }

                //Writer
                {
                    std::vector<uint32_t> runtime_args = {output.buffer()->address(),(uint32_t) num_tiles_per_block*local_num_blocks, tile_offset};
                    tile_offset += local_num_blocks * num_tiles_per_block;
                    tt_metal::SetRuntimeArgs(program, writer_kernel_id, core,runtime_args);
                }

                if(split_weights){
                    weight_offset += local_problem_size;
                }
                else{
                    input_offset+=local_problem_size;
                }

            }
        }
    }


    auto override_runtime_args_callback =
        [num_cores_x, num_cores_y, reader_kernel_id, writer_kernel_id, all_cores, core_group_1, core_group_2](
            const Program &program, const std::vector<Buffer *> &input_buffers,
            const std::vector<Buffer *> &output_buffers) {
            for(uint32_t idx=0; idx<num_cores_x; idx++){
                for(uint32_t idy=0; idy<num_cores_y; idy++){
                    CoreCoord core = {(std::size_t)idx, (std::size_t)idy};
                    bool core_used = core_group_1.core_coord_in_core_ranges(core)
                                    || core_group_2.core_coord_in_core_ranges(core);
                    if(core_used){
                        auto output_dram_buffer = output_buffers.at(0);
                        auto input_dram_buffer = input_buffers.at(0);
                        auto weights_dram_buffer = input_buffers.at(1);
                        {

                            auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                            runtime_args[2] = input_dram_buffer->address();
                            runtime_args[3] = weights_dram_buffer->address();
                            SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
                        }

                        {
                            auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                            runtime_args[0] = output_dram_buffer->address();
                            SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
                        }
                    }
                }
            }
    };

    return {std::move(program), override_runtime_args_callback};
}


operation::ProgramWithCallbacks embeddings_rm(
    const Tensor &a,
    const Tensor &weights,
    Tensor & output,
    bool split_weights
){

    ////////////////////////////////////////////////////////////////////////////
    //                 Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::Buffer *a_buffer = a.buffer();
    tt_metal::Buffer *weights_buffer = weights.buffer();
    tt_metal::Buffer *out_buffer = output.buffer();


    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    Device *device = a.device();
    auto dst_addr = output.buffer()->address();

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program{};


    uint32_t cb_id = 0;
    uint32_t num_tiles_per_cb = 1;
    bool in0_is_dram = a.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool weights_is_dram = weights.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool weights_dtype_is_bfloat16 = weights.dtype() == tt::tt_metal::DataType::BFLOAT16;
    bool out_is_dram = output.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    uint32_t last_dim = 3;
    uint32_t element_size_in_bytes = 2; // size of float
    if(!weights_dtype_is_bfloat16)
        element_size_in_bytes = 1; //bfp8_b

    // row major, page size is last dim
    uint32_t single_page_size = weights.shape()[last_dim]*element_size_in_bytes;

    // weights shape is [1, 1, num_embeddings, num_dim]
    uint32_t num_embeddings = weights.shape()[last_dim-1];

    uint32_t batch_size = a.shape()[0];
    uint32_t num_output_rows_per_batch = a.shape()[last_dim - 1];
    uint32_t num_output_rows = num_output_rows_per_batch * batch_size;


    //setup problem and grid size
    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    uint32_t problem_size;
    if(split_weights){
        problem_size = num_embeddings;
    }
    else{
        problem_size = num_output_rows;
    }



    // if tilized, then we will use one risc core per tensix for data movement of embedding and the other to read out from the tilized kernel
    // else both risc cores will be used for lookup of the embedding table
    uint32_t embedding_risc_cores_per_tensix = RISC_CORES_PER_TENSIX;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1,
        core_group_2, num_blocks_per_core_group_1,
        num_blocks_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, problem_size);


    // Create Kernels

    std::vector <tt_metal::Buffer> dst_buffer_l1 (embedding_risc_cores_per_tensix,
                                        CreateBuffer(device,
                                                        sizeof(uint32_t)*2,
                                                        sizeof(uint32_t)*2,
                                                        tt_metal::BufferType::L1));
    std::vector <tt_metal::Buffer> weights_buffer_l1 (embedding_risc_cores_per_tensix,
                                        CreateBuffer(device,
                                                        single_page_size*2,
                                                        single_page_size,
                                                        tt_metal::BufferType::L1));

    std::vector< std::vector<uint32_t> > compile_time_args(embedding_risc_cores_per_tensix);
    std::vector<tt_metal::DataMovementProcessor> risc_procs = {tt_metal::DataMovementProcessor::RISCV_0, tt_metal::DataMovementProcessor::RISCV_1};
    std::vector<tt_metal::NOC> noc_ports = {tt_metal::NOC::RISCV_0_default, tt_metal::NOC::RISCV_1_default};


    std::vector<tt::tt_metal::KernelID> kernIds(RISC_CORES_PER_TENSIX);

    for(int risc_id =0; risc_id < embedding_risc_cores_per_tensix; risc_id++){
       std::vector<uint32_t> embedding_compile_time_args= {  (std::uint32_t) in0_is_dram,
                                                     (std::uint32_t) weights_is_dram,
                                                     (std::uint32_t) out_is_dram,
                                                     (std::uint32_t) single_page_size,
                                                     (std::uint32_t) dst_buffer_l1[risc_id].address(),
                                                     (std::uint32_t) weights_buffer_l1[risc_id].address(),
                                                     (std::uint32_t) false,
                                                     (std::uint32_t) 0,
                                                     (std::uint32_t) 0

                                                     };

        kernIds[risc_id] = tt_metal::CreateKernel(
                                    program,
                                    "tt_eager/tt_dnn/kernels/dataflow/embeddings.cpp",
                                    all_cores,
                                    tt_metal::DataMovementConfig{.processor = risc_procs[risc_id],
                                                                .noc = noc_ports[risc_id],
                                                                .compile_args = embedding_compile_time_args});
    }


    uint32_t input_offset = 0;
    uint32_t weight_offset = 0;
    uint32_t tile_offset = 0;

    for(uint32_t idx=0; idx<num_cores_x; idx++){
        for(uint32_t idy=0; idy<num_cores_y; idy++){
            uint32_t local_num_output_rows;
            for(uint32_t idc=0; idc<embedding_risc_cores_per_tensix; idc++){
                uint32_t core_id = idx*num_cores_y*embedding_risc_cores_per_tensix +
                                   idy*embedding_risc_cores_per_tensix +
                                   idc;

                uint32_t local_num_embeddings;
                uint32_t local_weight_offset;
                uint32_t local_input_offset = input_offset;
                CoreCoord core = {(std::size_t)idx, (std::size_t)idy};
                uint32_t local_problem_size = num_blocks_per_core_group_1;
                bool core_used = core_group_1.core_coord_in_core_ranges(core)
                                || core_group_2.core_coord_in_core_ranges(core);
                bool core_0 = true;
                if(core_group_2.core_coord_in_core_ranges(core)){
                    local_problem_size = num_blocks_per_core_group_2;
                    core_0 = false;
                }
                else{
                    bool valid_core = core_used && core_group_1.core_coord_in_core_ranges(core) || (!core_used);
                    TT_ASSERT(valid_core, "Not valid core");
                }

                if(core_used){
                    if(split_weights){
                        local_weight_offset = weight_offset;
                        weight_offset+= local_problem_size;
                        local_input_offset = input_offset;
                        local_num_output_rows = num_output_rows;
                        local_num_embeddings= local_problem_size;
                    }
                    else{
                        local_input_offset = input_offset;
                        input_offset+= local_problem_size;
                        local_weight_offset = weight_offset;
                        local_num_embeddings = num_embeddings;
                        local_num_output_rows = local_problem_size;
                    }

                    std::vector<uint32_t>runtime_args;

                    runtime_args = {
                        (std::uint32_t) local_input_offset,
                        (std::uint32_t) local_weight_offset,
                        (std::uint32_t) local_num_embeddings,
                        (std::uint32_t) local_num_output_rows,
                        (std::uint32_t) a.buffer()->address(),
                        (std::uint32_t) weights.buffer()->address(),
                        (std::uint32_t) output.buffer()->address()
                        };
                    tt_metal::SetRuntimeArgs(program, kernIds[idc], core,runtime_args);
                }

            }
        }
    }

    auto override_runtime_args_callback =
        [num_cores_x, num_cores_y, kernIds, all_cores, embedding_risc_cores_per_tensix, core_group_1, core_group_2](
            const Program &program, const std::vector<Buffer *> &input_buffers,
            const std::vector<Buffer *> &output_buffers) {
            for(uint32_t idx=0; idx<num_cores_x; idx++){
                for(uint32_t idy=0; idy<num_cores_y; idy++){
                    CoreCoord core = {(std::size_t)idx, (std::size_t)idy};
                    bool core_used = core_group_1.core_coord_in_core_ranges(core)
                                    || core_group_2.core_coord_in_core_ranges(core);
                    if(core_used){
                        for(uint32_t idc=0; idc<embedding_risc_cores_per_tensix; idc++){
                            auto input_dram_buffer = input_buffers.at(0);
                            auto weights_dram_buffer = input_buffers.at(1);
                            auto output_dram_buffer = output_buffers.at(0);
                            {

                                auto runtime_args = GetRuntimeArgs(program, kernIds[idc], core);
                                runtime_args[4] = input_dram_buffer->address();
                                runtime_args[5] = weights_dram_buffer->address();
                                runtime_args[6] = output_dram_buffer->address();
                                SetRuntimeArgs(program, kernIds[idc], core, runtime_args);
                            }
                        }
                    }
                }
            }
    };

    return {std::move(program), override_runtime_args_callback};
}




operation::ProgramWithCallbacks embeddings_(
    const Tensor &a,
    const Tensor &weights,
    Tensor & output,
    bool split_weights,
    bool tilized
){

    if(tilized){
        return embeddings_tilized(a, weights, output, split_weights);
    }
    else{
        return embeddings_rm(a, weights, output, split_weights);
    }

}



void Embeddings::validate(const std::vector<Tensor> &input_tensors) const  {
    TT_FATAL(input_tensors.size() == 2 , "Must have between 2 input tensors");
    auto& a = input_tensors.at(0);
    const auto& weights = input_tensors.at(1);
    TT_FATAL(a.layout() == Layout::ROW_MAJOR);
    TT_FATAL(weights.layout() == Layout::ROW_MAJOR);
    TT_FATAL(a.dtype() == DataType::UINT32, "Input must be UIN32");
    TT_FATAL(weights.dtype() == DataType::BFLOAT16 ||
            weights.dtype() == DataType::BFLOAT8_B);


    TT_FATAL(weights.shape()[0] == 1 && weights.shape()[1] == 1,
                "First two dimensions for the weights must be 1");
    if(this->tilized){
        TT_FATAL(weights.shape()[3] % TILE_WIDTH == 0, "Number of columns in table must be factor of tile width");
    }
    TT_FATAL(a.shape()[1] == 1 && a.shape()[3] == 1, "Only dim 0 && 2 for the input can be non 1");
}

std::vector<Shape> Embeddings::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& weight_tensor = input_tensors.at(1);
    auto num_output_embeddings = input_tensor.shape()[2];
    auto batch_num = input_tensor.shape()[0];
    auto num_embedding_dims = weight_tensor.shape()[3];

    Shape output_shape({batch_num, 1, num_output_embeddings, num_embedding_dims});
    return {output_shape};
}

std::vector<Tensor> Embeddings::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& weight_tensor = input_tensors.at(1);
    if(!tilized){
    return operation::generic_create_output_tensors(*this, input_tensors,
                                                weight_tensor.dtype(), Layout::ROW_MAJOR,
                                                this->output_mem_config);
    }
    else{
    return operation::generic_create_output_tensors(*this, input_tensors,
                                                weight_tensor.dtype(), Layout::TILE,
                                                this->output_mem_config);
    }
}

operation::ProgramWithCallbacks Embeddings::create_program(
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    const auto& a = input_tensors.at(0);
    const auto& weights = input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);
    return embeddings_(a, weights, output_tensor, this->split_weights, this->tilized);
}

tt::stl::reflection::Attributes Embeddings::attributes() const {
    return {
        {"output_mem_config", this->output_mem_config},
    };
}

} // namespace tt_metal
} // namespace tt
