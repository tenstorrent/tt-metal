#include "libs/tt_dnn/op_library/softmax/softmax_op.hpp"
#include "libs/tt_dnn/op_library/work_split.hpp"
#include "tile_math.hpp"

#include "../op_config.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"

#include <iostream>

using u32 = std::uint32_t;
using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

inline bool is_dram(const Tensor& a) { return a.buffer_type() == BufferType::DRAM; }
inline bool is_dram(const Tensor* a) { return a ? a->buffer_type() == BufferType::DRAM : true; } // if nullptr doesn't matter
inline bool is_dram(const Buffer* b) { return b->buffer_type() == BufferType::DRAM; }

// implementation of softmax with optional scale/mask (see the header for a more detailed description)
Tensor scale_mask_softmax_(float scale, const Tensor* mask, Tensor &a) {

    bool profile = false;
    OpEnvConfig::update_profile(&profile);

    const auto shape = a.shape();
    u32 W = shape[3], H = shape[2], NC = shape[1]*shape[0];
    u32 HW = H*W;
    TT_ASSERT(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
    TT_ASSERT(H > 0 && W > 0 && NC > 0);
    TT_ASSERT(a.dtype() == DataType::BFLOAT16);
    TT_ASSERT(mask == nullptr || mask->dtype() == DataType::BFLOAT16);
    u32 Wt = W/TILE_WIDTH;
    u32 Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = NC*H*W / TILE_HW;

    Program program = Program();

    TT_ASSERT(a.device() != nullptr, "Operand to transpose_wh op needs to be on device!");

    uint32_t TBYTES = 2 * 1024;

    auto src0_dram_buffer = a.buffer();

    TT_ASSERT(a.volume() % TILE_HW == 0);
    int32_t num_tiles = a.volume()/TILE_HW;

    // This should allocate a DRAM buffer on the device
    Device *device = a.device();

    uint32_t block_size = find_max_divisor(Wt, 8);
    OpEnvConfig::update_block_size(&block_size);

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t in0_t  = block_size*2;
    uint32_t out0_t = block_size*2;
    uint32_t im1_t  = 2;
    uint32_t in2_t  = 2; // scaler for reduce coming from reader
    uint32_t in3_t  = 2; // 1/sqrt() scaler tile cb for fused scale/mask/softmax variant
    uint32_t in4_t  = divup(Wt, block_size)*block_size; // attention mask (N,C,32,W) - Wt is reused for each Ht, NC is cycled
    uint32_t im2_t  = 2; // recip result

    // cb_exps - keeps exps in CB in L1 to avoid recomputing
    uint32_t im0_t  = block_size*divup(Wt, block_size);
    TT_ASSERT(im0_t == Wt);

    // used for buffering scale-mask
    // can't easily reuse im0_t because cumulative wait for Wt needs to have Wt tiles contiguous free
    uint32_t im3_t  = block_size*(divup(Wt, block_size)+1);
    TT_ASSERT(im3_t == Wt+block_size);

    TT_ASSERT(Wt % block_size == 0);
    TT_ASSERT((block_size != -1) && "Wt must be divisible by one of the numbers in the range from 8 to 1.");
    TT_ASSERT(im0_t % block_size == 0 && "Size of cb must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(out0_t % block_size == 0 && "Size of cb must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(in4_t % block_size == 0);
    TT_ASSERT(W <= TILE_WIDTH*im0_t && "W exceeds the maximum supported size of tile buffer (kernel limitation right now).");

    uint32_t NCHt = NC*Ht;
    CoreGridDesc grid(a.device());
    uint32_t num_cores = grid.numcores_dividing_numtiles(NCHt);
    OpEnvConfig::update_num_cores(&num_cores);
    uint32_t partHt = NCHt/num_cores; // only used by fused_scale_mask variant

    // we are actually splitting blocks of Wt tiles, not tiles, so no checking for bank alignment is needed
    TilesSplit ts(num_cores, NCHt);
    auto wtpc = ts.get_tpc();
    TT_ASSERT(NCHt % wtpc == 0);
    TT_ASSERT(NCHt % num_cores == 0);
    TT_ASSERT(wtpc < Ht || (wtpc % Ht == 0));
    TT_ASSERT(NCHt % num_cores == 0);
    TT_ASSERT(partHt >= Ht || Ht % partHt == 0);
    //cout << "NUM CORES=" << num_cores << " WTPC=" << wtpc << " partHt=" << partHt << endl;

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
        if (mask != nullptr) {
            CreateCircularBuffer( program, device, CB::c_intermed3, core, im3_t,  im3_t *TBYTES,  DataFormat::Float16_b );
            CreateCircularBuffer( program, device, CB::c_in3, core, in3_t,  in3_t *TBYTES,  DataFormat::Float16_b );
            CreateCircularBuffer( program, device, CB::c_in4, core, in4_t,  in4_t *TBYTES,  DataFormat::Float16_b );
        }

        DataMovementKernel *reader_kernel = CreateDataMovementKernel(
            program, "tt_metal/kernels/dataflow/reader_unary_8bank_sm.cpp", core,
            DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);
            //DataMovementProcessor::RISCV_1, core.x < 6 ? NOC::RISCV_1_default : NOC::RISCV_0_default);

        DataMovementKernel *writer_kernel = CreateDataMovementKernel(
            program, "tt_metal/kernels/dataflow/writer_unary_8bank_sm.cpp", core,
            DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);
            //DataMovementProcessor::RISCV_0, core.x < 6 ? NOC::RISCV_0_default : NOC::RISCV_1_default);

        // for broadcasting in H direction we need to
        // NCHt, Nt, Wt
        // if wtpc < Ht then since we pass tpc to the kernel as Ht, the broadcasts should be correct
        // if wtpc >= Ht then tpc should be a multiple of Ht
        vector<uint32_t> compute_args = { wtpc, partHt, Wt };

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = true;
        auto softmax_kernel = CreateComputeKernel(
            program, "kernels/compute/softmax.cpp", core, compute_args,
            MathFidelity::HiFi4, fp32_dest_acc_en, math_approx_mode);

        softmax_kernel->add_define("BLOCK_SIZE", block_size);
        reader_kernel->add_define("BLOCK_SIZE", block_size);
        reader_kernel->add_define("A_DRAM", is_dram(a) ? "true" : "false");
        reader_kernel->add_define("MASK_DRAM", is_dram(mask) ? "true" : "false");
        writer_kernel->add_define("BLOCK_SIZE", block_size);
        writer_kernel->add_define("OUTPUT_DRAM", is_dram(a) ? "true" : "false");
        if (scale != 0.0f) {
            reader_kernel->add_define("FUSED_SCALE_MASK", "1");
            softmax_kernel->add_define("FUSED_SCALE_MASK", "1");
        }
        readers.push_back(reader_kernel);
        writers.push_back(writer_kernel);
    }

    CompileProgram(device, program, profile);
    ConfigureDeviceWithProgram(device, program);

    for (uint32_t icore = 0; icore < num_cores; icore++) {
        auto core = grid.wrap_core(icore);
        uint32_t src_addr = src0_dram_buffer->address();
        //uint32_t dst_addr = dst_dram_buffer->address();
        uint32_t mask_addr = mask ? mask->buffer()->address() : 0;
        uint32_t tile_offset = wtpc*Wt*icore;
        //cout << "WTPC=" << wtpc << endl;
        //cout << "Wt=" << Wt << endl;
        //cout << "tile_offset=" << tile_offset << endl;
        union { float f; uint32_t u; } s; s.f = scale; // scale for fused scale-mask-softmax
        // always in-place
        //                                                              0  1    2       3            4   5       6          7           8
        WriteRuntimeArgsToDevice(device, readers[icore], core, { src_addr, 0, s.u, wtpc*Wt, tile_offset, partHt, Wt, mask_addr, 0x3f800000 }); // [8]=1.0f is scaler
        WriteRuntimeArgsToDevice(device, writers[icore], core, { src_addr, 0,   0, wtpc*Wt, tile_offset });
    }

    LaunchKernels(device, program);
    if (profile)
        tt_metal::DumpDeviceProfileResults(device, program);

    return std::move(a);
} // scale_mask_softmax_

Tensor scale_mask_softmax_in_place(float scale, const Tensor& mask, Tensor& a) {
    return std::move(scale_mask_softmax_(scale, &mask, a));
}

Tensor softmax_in_place(Tensor &a) {
    return std::move(scale_mask_softmax_(0.0f, nullptr, a)); // 0.0f means unused scale
}

}  // namespace ll_buda

}  // namespace tt
