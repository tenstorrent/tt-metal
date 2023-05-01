#include "libs/tt_dnn/op_library/softmax/softmax_op.hpp"
#include "libs/tt_dnn/op_library/work_split.hpp"
#include "tile_math.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"

#include <iostream>

using u32 = std::uint32_t;
using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

Tensor softmax(const Tensor &a) {

    bool profile = false;
    const auto shape = a.shape();
    u32 W = shape[3], H = shape[2], NC = shape[1]*shape[0];
    u32 HW = H*W;
    TT_ASSERT(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
    TT_ASSERT(H > 0 && W > 0 && NC > 0);
    u32 Wt = W/TILE_WIDTH;
    u32 Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = NC*H*W / TILE_HW;

    Program *program = new Program();

    tt_xy_pair core = {0, 0};

    TT_ASSERT(a.device() != nullptr, "Operand to transpose_wh op needs to be on device!");

    uint32_t TBYTES = 2 * 1024;

    Buffer *src0_dram_buffer = a.buffer();

    TT_ASSERT(a.volume() % TILE_HW == 0);
    int32_t num_tiles = a.volume()/TILE_HW;

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    // This should allocate a DRAM buffer on the device
    Device *device = a.device();

    std::array<uint32_t, 4> output_shape = {shape[0], shape[1], H, W};
    Tensor output = Tensor(output_shape, a.dtype(), Layout::TILE, device);

    Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    int block_size = find_max_divisor(Wt, 8);
    //if (getenv("FORCE_BLOCK_SIZE") != nullptr) block_size = std::stoi( getenv("FORCE_BLOCK_SIZE") );
    //std::cout << "Block size=" << block_size << std::endl;

    TT_ASSERT(Wt % block_size == 0);
    TT_ASSERT((block_size != -1) && "Wt must be divisible by one of the numbers in the range from 8 to 1.");

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t in0_t  = block_size == 8 ? 16 : 12;
    uint32_t out0_t = block_size == 8 ? 16 : 12;
    uint32_t im1_t  = block_size == 8 ? 16 : 12;
    uint32_t in2_t  = 2; // scaler for reduce coming from reader
    uint32_t im2_t  = 2; // recip result
    uint32_t im0_t  = (block_size == 8) ? 128 : 120; // buffer for exps

    TT_ASSERT(im0_t % block_size == 0 && "Size of cb must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(out0_t % block_size == 0 && "Size of cb must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(im1_t % block_size == 0 && "Size of cb buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(W <= TILE_WIDTH*im0_t && "W exceeds the maximum supported size of tile buffer (kernel limitation right now).");
    //std::cout << "Block size=" << block_size << std::endl;

    uint32_t NCHt = NC*Ht;
    CoreGridDesc grid(a.device());
    uint32_t num_cores = grid.numcores_dividing_numtiles(NCHt);
    TT_ASSERT(NCHt % num_cores == 0);

    // we are actually splitting blocks of Wt tiles, not tiles, so no checking for bank alignment is needed
    TilesSplit ts(num_cores, NCHt);
    auto tpc = ts.get_tpc();
    TT_ASSERT(NCHt % tpc == 0);

    //cout << "NUM CORES=" << num_cores << " TPC=" << tpc << endl;

    vector<DataMovementKernel*> readers, writers;
    readers.reserve(num_cores);
    writers.reserve(num_cores);
    for (uint32_t icore = 0; icore < num_cores; icore++) {
        auto core = grid.wrap_core(icore);
        // see softmax.cpp for which buffers are needed
        CreateCircularBuffer( program, device, CB::c_in0,       core, in0_t,  in0_t *TBYTES,  DataFormat::Float16_b );
        CreateCircularBuffer( program, device, CB::c_out0,      core, out0_t, out0_t*TBYTES,  DataFormat::Float16_b );
        CreateCircularBuffer( program, device, CB::c_intermed1, core, im1_t,  im1_t *TBYTES,  DataFormat::Float16_b );
        CreateCircularBuffer( program, device, CB::c_in2,       core, in2_t,  in2_t *TBYTES,  DataFormat::Float16_b );
        CreateCircularBuffer( program, device, CB::c_intermed2, core, im2_t,  im2_t *TBYTES,  DataFormat::Float16_b );
        CreateCircularBuffer( program, device, CB::c_intermed0, core, im0_t,  im0_t *TBYTES,  DataFormat::Float16_b );

        DataMovementKernel *reader_kernel = CreateDataMovementKernel(
            program, "tt_metal/kernels/dataflow/reader_unary_8bank_sm.cpp", core,
            DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);

        DataMovementKernel *writer_kernel = CreateDataMovementKernel(
            program, "tt_metal/kernels/dataflow/writer_unary_8bank_sm.cpp", core,
            DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);

        vector<uint32_t> compute_args = { tpc, Wt };
        ComputeKernelArgs *softmax_args = InitializeCompileTimeComputeKernelArgs(core, compute_args);

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = true;
        auto eltwise_binary_kernel = CreateComputeKernel(
            program, "kernels/compute/softmax.cpp", core, softmax_args,
            MathFidelity::HiFi4, fp32_dest_acc_en, math_approx_mode);

        eltwise_binary_kernel->add_define("BLOCK_SIZE", block_size);
        reader_kernel->add_define("BLOCK_SIZE", block_size);
        writer_kernel->add_define("BLOCK_SIZE", block_size);
        readers.push_back(reader_kernel);
        writers.push_back(writer_kernel);
    }

    CompileProgram(device, program, profile);
    ConfigureDeviceWithProgram(device, program);

    const int bank_size = 8;
    for (uint32_t icore = 0; icore < num_cores; icore++) {
        auto core = grid.wrap_core(icore);
        uint32_t src_addr = src0_dram_buffer->address();
        uint32_t dst_addr = dst_dram_buffer->address();
        uint32_t tile_offset = tpc*Wt*icore;
        //cout << "TPC=" << tpc << endl;
        //cout << "tile_offset=" << tile_offset << endl;
        WriteRuntimeArgsToDevice(device, readers[icore], core, { src_addr, 0, 0, tpc*Wt, tile_offset, 0, 0, 0, 0x3f800000 }); // [8]=1.0f is scaler
        WriteRuntimeArgsToDevice(device, writers[icore], core, { dst_addr, 0, 0, tpc*Wt, tile_offset });
    }

    LaunchKernels(device, program);
    if (profile)
        tt_metal::DumpDeviceProfileResults(device, program);

    delete program;

    return output;
} // softmax

}  // namespace ll_buda

}  // namespace tt
