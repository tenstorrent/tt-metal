#include "libs/tt_dnn/op_library/layernorm/layernorm_op.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"
#include "libs/tt_dnn/op_library/work_split.hpp"

#include "../op_config.hpp"

#include <iostream>

using u32 = std::uint32_t;
using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

// computes layernorm(a+*b)*gamma + beta
// if b is nullptr it's treated as zero (no addition)
Tensor layernorm_(const Tensor &a, const Tensor* b, float eps, const Tensor* gamma, const Tensor* beta) {

    const auto shape = a.shape();
    u32 W = shape[3], H = shape[2], NC = shape[1]*shape[0];
    u32 HW = H*W;
    TT_ASSERT(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
    TT_ASSERT(H > 0 && W > 0 && NC > 0);
    u32 Wt = W/TILE_WIDTH;
    u32 Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = NC*H*W / TILE_HW;

    Program *program = new Program();

    TT_ASSERT(a.device() != nullptr, "Operand to transpose_wh op needs to be on device!");
    int block_size = find_max_divisor(Wt, 8);
    //if (getenv("FORCE_BLOCK_SIZE") != nullptr) block_size = std::stoi( getenv("FORCE_BLOCK_SIZE") );
    //std::cout << "Block size=" << block_size << std::endl;

    uint32_t single_tile_size = 2 * 1024;

    Buffer *a_dram_buf = a.buffer();
    auto b_dram_addr = b ? b->buffer()->address() : 0;
    auto gamma_dram_addr = gamma ? gamma->buffer()->address() : 0;
    auto beta_dram_addr = beta ? beta->buffer()->address() : 0;

    TT_ASSERT(b == nullptr || a.shape() == b->shape());
    TT_ASSERT(a.volume() % TILE_HW == 0);
    uint32_t num_tiles = a.volume()/TILE_HW;
    uint32_t num_gamma_tiles = gamma ? gamma->volume()/TILE_HW : 0;
    uint32_t num_beta_tiles = beta ? beta->volume()/TILE_HW : 0;

    // This should allocate a DRAM buffer on the device
    Device *device = a.device();

    std::array<uint32_t, 4> output_shape = {shape[0], shape[1], H, W};
    Tensor output = Tensor(output_shape, a.dtype(), Layout::TILE, device);

    Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    // TODO(AP): this will not work for all Wts possibly, but should work for Wt=8, 12, 16, 32
    // TODO(AP): can also add support for block_size=7 -> 63, 28
    uint32_t WtB    =  divup(Wt, block_size)*block_size; // Wt padded to be divisible by block size
    uint32_t in0_t  =  WtB+2*block_size; // cb_x for no pre-add variant, x=a+b for fused pre-add, extra space for some buffering
    uint32_t in1_t  =  block_size*2; // buffer for fused pre-add b tensor
    uint32_t out0_t =  block_size*2;
    uint32_t im0_t  =  WtB; // buffer for saving xmm
    uint32_t im3_t  =  WtB; // buffer for xmm^2
    uint32_t in5_t  =  WtB; // buffer for gamma
    uint32_t in6_t  =  WtB; // buffer for beta
    uint32_t im6_t  =  block_size*2; // x=a+b reuse for x-E[x] computation plus a bit extra for buffering
    if (b) {
        im6_t = in0_t;
        in0_t = 2*block_size;
    }
    uint32_t im5_t  =  2*block_size; // for buffering to/from *gamma/+beta
    uint32_t im4_t  =  8; // 8 just in case, 4 would prob suffice
    uint32_t in4_t  =  2; // ones column mask
    uint32_t im1_t  =  2;
    uint32_t in2_t  =  2; // scaler for reduce coming from reader
    uint32_t in3_t  =  2; // epsilon coming from reader
    uint32_t im2_t  =  2; //

    TT_ASSERT(W <= TILE_WIDTH*im0_t && "W exceeds the maximum supported size of tile buffer (kernel limitation right now).");
    TT_ASSERT(in0_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(in1_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(out0_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(im0_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(im3_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(in5_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(in6_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(im6_t % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(Wt % block_size == 0);

    uint32_t NCHt = NC*Ht;
    CoreGridDesc grid(a.device());
    uint32_t num_cores = grid.numcores_dividing_numtiles(NCHt);
    OpEnvConfig::update_num_cores(&num_cores);
    TT_ASSERT(NCHt % num_cores == 0);

    // we are actually splitting blocks of Wt tiles, not tiles, so no checking for bank alignment is needed
    TilesSplit ts(num_cores, NCHt);
    auto tpc = ts.get_tpc(); // Wt*tpc per core
    TT_ASSERT(NCHt % tpc == 0);


    vector<DataMovementKernel*> readers, writers;
    readers.reserve(num_cores);
    writers.reserve(num_cores);
    for (uint32_t icore = 0; icore < num_cores; icore++) {
        auto core = grid.wrap_core(icore);
        CreateCircularBuffer( program, device, CB::c_in0,       core, in0_t,  in0_t*single_tile_size,  DataFormat::Float16_b );
        CreateCircularBuffer( program, device, CB::c_out0,      core, out0_t, out0_t*single_tile_size, DataFormat::Float16_b );
        CreateCircularBuffer( program, device, CB::c_intermed1, core, im1_t,  im1_t*single_tile_size,  DataFormat::Float16_b );
        CreateCircularBuffer( program, device, CB::c_in2,       core, in2_t,  in2_t*single_tile_size,  DataFormat::Float16_b );
        CreateCircularBuffer( program, device, CB::c_in3,       core, in3_t,  in3_t*single_tile_size,  DataFormat::Float16_b );
        CreateCircularBuffer( program, device, CB::c_in4,       core, in4_t,  in4_t*single_tile_size,  DataFormat::Float16_b );
        CreateCircularBuffer( program, device, CB::c_intermed2, core, im2_t,  im2_t*single_tile_size,  DataFormat::Float16_b );
        CreateCircularBuffer( program, device, CB::c_intermed0, core, im0_t,  im0_t*single_tile_size,  DataFormat::Float16_b );
        CreateCircularBuffer( program, device, CB::c_intermed3, core, im3_t,  im3_t*single_tile_size,  DataFormat::Float16_b );
        CreateCircularBuffer( program, device, CB::c_intermed4, core, im4_t,  im4_t*single_tile_size,  DataFormat::Float16_b );
        CreateCircularBuffer( program, device, CB::c_intermed5, core, im5_t,  im5_t*single_tile_size,  DataFormat::Float16_b );
        CreateCircularBuffer( program, device, CB::c_in5,       core, in5_t,  in5_t*single_tile_size,  DataFormat::Float16_b );
        CreateCircularBuffer( program, device, CB::c_in6,       core, in6_t,  in6_t*single_tile_size,  DataFormat::Float16_b );
        if (b) {
            // x = a+b in this notation
            // result = ln(x)*gamma + beta
            // if there's no pre-add we use cb_in0 for x, otherwise a is pre-buffered into in0, added into im6, then im6 is used as x
            // b is buffered into c_in1
            CreateCircularBuffer( program, device, CB::c_intermed6, core, im6_t,  im6_t*single_tile_size,  DataFormat::Float16_b );
            // c_in1 is input buffer for b
            CreateCircularBuffer( program, device, CB::c_in1,       core, in1_t,  in1_t*single_tile_size,  DataFormat::Float16_b );
        }

        DataMovementKernel *reader_kernel = CreateDataMovementKernel(
            program, "tt_metal/kernels/dataflow/reader_unary_8bank_ln.cpp", core,
            DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);

        DataMovementKernel *writer_kernel = CreateDataMovementKernel(
            program, "tt_metal/kernels/dataflow/writer_unary_8bank_ln.cpp", core,
            DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);

        vector<uint32_t> compute_args = { tpc, Wt, num_gamma_tiles>0, num_beta_tiles>0 };
        KernelArgs softmax_args = KernelArgs(core, compute_args);

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = true;
        auto eltwise_binary_kernel = CreateComputeKernel(
            program,
            "kernels/compute/layernorm.cpp",
            core,
            softmax_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );
        eltwise_binary_kernel->add_define("BLOCK_SIZE", block_size);
        reader_kernel->add_define("BLOCK_SIZE", block_size);
        writer_kernel->add_define("BLOCK_SIZE", block_size);
        if (b) {
            reader_kernel->add_define("FUSE_PRE_ADD", "1");
            eltwise_binary_kernel->add_define("FUSE_PRE_ADD", "1");
        }
        readers.push_back(reader_kernel);
        writers.push_back(writer_kernel);
    }

    bool profile = false;
    OpEnvConfig::update_profile(&profile);
    CompileProgram(device, program, profile);
    ConfigureDeviceWithProgram(device, program);

    for (uint32_t icore = 0; icore < num_cores; icore++) {
        auto core = grid.wrap_core(icore);
        union { float f; uint32_t u; } winv; winv.f = 1.0f / W; // bcast-w scaler
        union { float f; uint32_t u; } e; e.f = eps; // epsilon
        uint32_t gamma_tiles = gamma ? gamma->volume() / TILE_HW : 0;
        uint32_t beta_tiles = beta ? beta->volume() / TILE_HW : 0;
        //std::cout << "Num gamma=" << num_gamma_tiles << " addr=" << gamma_dram_addr << std::endl;
        //std::cout << "Num beta=" << num_beta_tiles << " addr=" << beta_dram_addr << std::endl;
        uint32_t tile_offset = tpc*Wt*icore;
        WriteRuntimeArgsToDevice( device, readers[icore], core,
            { a_dram_buf->address(), 0, 0, tpc*Wt, tile_offset, 0, 0, 0, winv.u, e.u, // 0-9
              num_gamma_tiles, gamma_dram_addr, num_beta_tiles, beta_dram_addr, b_dram_addr } // 10-14
        );
        WriteRuntimeArgsToDevice( device, writers[icore], core, { dst_dram_buffer->address(), 0, 0, tpc*Wt, tile_offset } );
    }
    LaunchKernels(device, program);

    if (profile)
        tt_metal::DumpDeviceProfileResults(device, program);

    delete program;

    return output;
} // softmax

Tensor layernorm(const Tensor &a, float eps) { return layernorm_(a, nullptr, eps, nullptr, nullptr); }
Tensor layernorm_gamma(const Tensor &a, float eps, const Tensor& gamma) { return layernorm_(a, nullptr, eps, &gamma, nullptr); }
Tensor layernorm_gamma_beta(const Tensor &a, float eps, const Tensor& gamma, const Tensor& beta) { return layernorm_(a, nullptr, eps, &gamma, &beta); }
Tensor add_layernorm_gamma_beta(const Tensor &a, const Tensor& b, float eps, const Tensor& gamma, const Tensor& beta) { return layernorm_(a, &b, eps, &gamma, &beta); }

}  // namespace ll_buda

}  // namespace tt
