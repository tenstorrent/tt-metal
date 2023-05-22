#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "common/test_tiles.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"

namespace blank_hlk {
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    std::int32_t dummy;
};
}

using namespace tt::constants;

namespace tt {

namespace tt_metal {

Tensor transpose_hc_rm(const Tensor &a) {

    TT_ASSERT(a.shape()[3] <= 16*1024 && "transpose_hc_rm kernel doesn't support W>=16k elems yet.");
    tt_metal::Device *device = a.device();
    tt_metal::Program program = tt_metal::Program();
    tt_xy_pair core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to eltwise unary needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to eltwise unary needs to be allocated in a buffer on device!");

    uint32_t single_tile_size = 2 * TILE_HW;
    tt_metal::Buffer *src0_dram_buffer = a.buffer();
    auto ashape = a.shape();
    int N = ashape[0], C = ashape[1], H = ashape[2], W = ashape[3];

    auto bshape = ashape;
    bshape[1] = ashape[2];
    bshape[2] = ashape[1];

    TT_ASSERT(a.layout() == tt::tt_metal::Layout::ROW_MAJOR, "This transpose assumes that the data layout is row major!");

    tt_metal::Tensor output = tt_metal::Tensor(bshape, a.dtype(), tt::tt_metal::Layout::ROW_MAJOR, device);
    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto l1_bank_ids = device->bank_ids_from_logical_core(core);
    TT_ASSERT(not l1_bank_ids.empty());
    auto l1_bank_id = l1_bank_ids.at(0);
    auto l1_b0 = tt_metal::Buffer(device, src0_dram_buffer->size(), l1_bank_id, src0_dram_buffer->size(), tt_metal::BufferType::L1);

    uint32_t num_cb_tiles = 16;
    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        0, // cb index
        core,
        num_cb_tiles, num_cb_tiles * single_tile_size,
        DataFormat::Float16_b);
    auto cb_src1 = tt_metal::CreateCircularBuffer(
        program,
        device,
        1, // cb index
        core,
        num_cb_tiles, num_cb_tiles * single_tile_size,
        DataFormat::Float16_b);

    tt_metal::DataMovementKernel *binary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/transpose_hc_rm_8bank_l1.cpp",
        core, tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::RISCV_1_default);

    // Compile kernels

    bool profile_kernel = true;
    tt_metal::CompileProgram(device, program, profile_kernel);
    tt_metal::ConfigureDeviceWithProgram(device, program);
    tt_metal::WriteRuntimeArgsToDevice(
        device,
        binary_reader_kernel,
        core,
        {src0_dram_buffer->address(),
        dst_dram_buffer->address(),
        l1_b0.address(),
        uint32_t(N),
        uint32_t(C),
        uint32_t(H),
        uint32_t(W),
        uint32_t(C*H)
        }
    );

    tt_metal::LaunchKernels(device, program);

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

Tensor transpose_hc_rm_multi_core(const Tensor &a) {

    TT_ASSERT(a.shape()[3] <= 16*1024 && "transpose_hc_rm kernel doesn't support W>=16k elems yet.");
    tt_metal::Device *device = a.device();
    tt_metal::Program program = tt_metal::Program();
    auto num_cores_c = a.shape()[1];
    auto num_cores_r = a.shape()[2];
    tt_xy_pair start_core = {0, 0};
    tt_xy_pair end_core = {(std::size_t)num_cores_c - 1, (std::size_t)num_cores_r - 1};;
    tt_metal::CoreRange all_cores(start_core, end_core);

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to eltwise unary needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to eltwise unary needs to be allocated in a buffer on device!");

    uint32_t single_tile_size = 2 * TILE_HW;
    tt_metal::Buffer *src0_dram_buffer = a.buffer();
    auto ashape = a.shape();
    int N = ashape[0], C = ashape[1], H = ashape[2], W = ashape[3];

    auto bshape = ashape;
    bshape[1] = ashape[2];
    bshape[2] = ashape[1];

    TT_ASSERT(a.layout() == tt::tt_metal::Layout::ROW_MAJOR, "This transpose assumes that the data layout is row major!");

    tt_metal::Tensor output = tt_metal::Tensor(bshape, a.dtype(), tt::tt_metal::Layout::ROW_MAJOR, device);
    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    uint32_t l1_buffer_addr = 400 * 1024;
    assert(src0_dram_buffer->size() % (num_cores_r * num_cores_c) == 0);
    uint32_t per_core_l1_size = src0_dram_buffer->size() / (num_cores_r * num_cores_c);
    for(int i = 0; i < num_cores_r; i++) {
        for(int j = 0; j < num_cores_c; j++) {
            tt_xy_pair core = {(std::size_t) j, (std::size_t) i};
            auto l1_bank_ids = device->bank_ids_from_logical_core(core);
            TT_ASSERT(not l1_bank_ids.empty());
            auto l1_bank_id = l1_bank_ids.at(0);
            auto l1_b0 = tt_metal::Buffer(device, per_core_l1_size, l1_buffer_addr, l1_bank_id, per_core_l1_size, tt_metal::BufferType::L1);
        }
    }
    std::cout << "Creating kernels " << std::endl;
    std::vector<tt_metal::DataMovementKernel*> binary_reader_kernels;
    for(uint32_t i = 0; i < num_cores_r; i++) {
        tt_metal::CoreRange core_range({0,i}, {(std::size_t)num_cores_c - 1,i});
        if(i%2 == 0) {
            binary_reader_kernels.push_back(tt_metal::CreateDataMovementKernel(
                program, "tt_metal/kernels/dataflow/transpose_hc_rm_8bank_partitioned.cpp",
                core_range, tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::RISCV_1_default));
        }
        else {
            binary_reader_kernels.push_back(tt_metal::CreateDataMovementKernel(
                program, "tt_metal/kernels/dataflow/transpose_hc_rm_8bank_partitioned.cpp",
                core_range, tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::RISCV_0_default));
        }
    }

    // for(uint32_t j = 0; j < num_cores_c; j++) {
    //     tt_metal::CoreRange core_range({j,0}, {j, (std::size_t)num_cores_r - 1});
    //     if(j%2 == 0) {
    //         binary_reader_kernels.push_back(tt_metal::CreateDataMovementKernel(
    //             program, "tt_metal/kernels/dataflow/transpose_hc_rm_8bank_partitioned.cpp",
    //             core_range, tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::RISCV_1_default));
    //     }
    //     else {
    //         binary_reader_kernels.push_back(tt_metal::CreateDataMovementKernel(
    //             program, "tt_metal/kernels/dataflow/transpose_hc_rm_8bank_partitioned.cpp",
    //             core_range, tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::RISCV_0_default));
    //     }
    // }

    // Compile kernels

    std::cout << "Compiling kernels " << std::endl;
    bool profile_kernel = true;
    tt_metal::CompileProgram(device, program, profile_kernel);
    std::cout << "Configure device with program " << std::endl;
    tt_metal::ConfigureDeviceWithProgram(device, program);
    std::cout << "Num cores " << num_cores_r * num_cores_c << std::endl;

    for(int i = 0; i < num_cores_r; i++) {
        for(int j = 0; j < num_cores_c; j++) {
            int core_index = i * num_cores_c + j;
            tt_xy_pair core = {(std::size_t) j, (std::size_t) i};
            tt_metal::WriteRuntimeArgsToDevice(
                device,
                binary_reader_kernels[i],
                //binary_reader_kernels[j],
                core,
                {src0_dram_buffer->address(),
                dst_dram_buffer->address(),
                l1_buffer_addr,
                uint32_t(N),
                uint32_t(C),
                uint32_t(H),
                uint32_t(W),
                uint32_t(C*H),
                uint32_t(i+j*H),
                uint32_t(j+i*C),
                1,
                1,
                uint32_t(H)
                }
            );
        }
    }

    // uint32_t core_index = 0;
    // for(int i = 0; i < ashape[2]; i++) {
    //     for(int j = 0; j < ashape[1]; j++) {
    //         tt_xy_pair core = {(std::size_t) core_index, (std::size_t) 0};
    //         tt_metal::WriteRuntimeArgsToDevice(
    //             device,
    //             binary_reader_kernel,
    //             core,
    //             {src0_dram_buffer->address(),
    //             dst_dram_buffer->address(),
    //             l1_buffer_addr,
    //             uint32_t(N),
    //             uint32_t(C),
    //             uint32_t(H),
    //             uint32_t(W),
    //             uint32_t(C*H),
    //             uint32_t(i+j*H),
    //             uint32_t(j+i*C),
    //             1,
    //             1,
    //             uint32_t(H)
    //             }
    //         );
    //         core_index++;
    //     }
    // }
    std::cout << "Launching kernels" << std::endl;
    tt_metal::LaunchKernels(device, program);

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

}  // namespace tt_metal

}  // namespace tt
