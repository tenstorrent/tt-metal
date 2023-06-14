#include "libs/tt_dnn/op_library/layernorm/layernorm_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "libs/tt_dnn/op_library/work_split.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

#include "../op_config.hpp"

#include <optional>

using u32 = std::uint32_t;
using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

inline bool is_dram(const Tensor& input_tensor) { return input_tensor.buffer_type() == BufferType::DRAM; }
inline bool is_dram(const std::optional<std::reference_wrapper<const Tensor>> input_tensor) {
     return input_tensor.has_value() ? is_dram(input_tensor.value().get()) : true;
}
inline bool is_dram(const Buffer* b) { return b->buffer_type() == BufferType::DRAM; }

// computes layernorm(a+*b)*gamma + beta
// if b is nullptr it's treated as zero (no addition)
Program layernorm_(
    const Tensor &a,
    const std::optional<std::reference_wrapper<const Tensor>> b,
    const std::optional<std::reference_wrapper<const Tensor>> gamma,
    const std::optional<std::reference_wrapper<const Tensor>> beta,
    Tensor& output,
    float eps,
    bool output_dram
) {

    const auto shape = a.shape();
    u32 W = shape[3], H = shape[2], NC = shape[1]*shape[0];
    u32 HW = H*W;
    TT_ASSERT(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
    TT_ASSERT(H > 0 && W > 0 && NC > 0);
    TT_ASSERT(a.dtype() == DataType::BFLOAT16);
    TT_ASSERT(not b.has_value() || b.value().get().dtype() == DataType::BFLOAT16);
    TT_ASSERT(not gamma.has_value() || gamma.value().get().dtype() == DataType::BFLOAT16);
    TT_ASSERT(not beta.has_value() || beta.value().get().dtype() == DataType::BFLOAT16);
    u32 Wt = W/TILE_WIDTH;
    u32 Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = NC*H*W / TILE_HW;

    Program program = Program();

    TT_ASSERT(a.device() != nullptr, "Operand to transpose_wh op needs to be on device!");
    uint32_t block_size = find_max_divisor(Wt, 8);
    OpEnvConfig::update_block_size(&block_size);

    uint32_t single_tile_size = 2 * 1024;

    auto a_addr = a.buffer()->address();
    auto b_dram_addr = b ? b.value().get().buffer()->address() : 0;
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().get().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().get().buffer()->address() : 0;

    TT_ASSERT(not b.has_value() || a.shape() == b.value().get().shape());
    TT_ASSERT(a.volume() % TILE_HW == 0);
    uint32_t num_tiles = a.volume()/TILE_HW;
    uint32_t num_gamma_tiles = gamma.has_value() ? gamma.value().get().volume()/TILE_HW : 0;
    uint32_t num_beta_tiles = beta.has_value() ? beta.value().get().volume()/TILE_HW : 0;
    TT_ASSERT(num_gamma_tiles == Wt || num_gamma_tiles == 0);
    TT_ASSERT(num_beta_tiles == Wt || num_beta_tiles == 0);

    // This should allocate a DRAM buffer on the device
    Device *device = a.device();

    auto memcfg = tt::tt_metal::MemoryConfig{true, -1, output_dram ? BufferType::DRAM : BufferType::L1};
    auto dst_addr = output.buffer()->address();

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    // TODO(AP): this will not work for all Wts possibly, but should work for Wt=8, 12, 16, 32
    // TODO(AP): can also add support for block_size=7 -> 63, 28
    uint32_t WtB    =  divup(Wt, block_size)*block_size; // Wt padded to be divisible by block size
    uint32_t in0_t  =  WtB; // cb_x for no pre-add variant, x=a+b for fused pre-add, extra space for some buffering
    uint32_t in1_t  =  block_size*2; // buffer for fused pre-add b tensor
    uint32_t out0_t =  block_size*2;
    uint32_t im0_t  =  WtB; // buffer for saving xmm
    uint32_t im3_t  =  WtB; // buffer for xmm^2
    uint32_t in5_t  =  WtB; // buffer for gamma
    uint32_t in6_t  =  WtB; // buffer for beta
    uint32_t im6_t  =  block_size*2; // x=a+b reuse for x-E[x] computation plus a bit extra for buffering
    if (b) {
        im6_t = WtB;
        //cout << "im6_t=WtB=" << WtB << endl;
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
    TT_ASSERT(num_gamma_tiles % block_size == 0);
    TT_ASSERT(num_beta_tiles % block_size == 0);

    uint32_t NCHt = NC*Ht;
    CoreGridDesc grid(a.device());
    uint32_t num_cores = grid.numcores_dividing_numtiles(NCHt);
    OpEnvConfig::update_num_cores(&num_cores);
    TT_ASSERT(NCHt % num_cores == 0);

    // we are actually splitting blocks of Wt tiles, not tiles, so no checking for bank alignment is needed
    TilesSplit ts(num_cores, NCHt);
    auto wtpc = ts.get_tpc(); // Wt*tpc per core
    TT_ASSERT(NCHt % wtpc == 0);

    //cout << "WTPC=" << wtpc << "Wt=" << Wt << " num_cores=" << num_cores << endl;

    // Parallelize across rows
    // TODO: Refactor by calling utility function?
    uint32_t num_full_rows = num_cores / grid.x_;
    uint32_t last_row_cores = num_cores % grid.x_;

    std::set<CoreRange> all_cores_set;
    if (num_full_rows) {
        all_cores_set.insert((CoreRange) {
            .start={0, 0}, .end={grid.x_ - 1, num_full_rows - 1}
        });
    }
    if (last_row_cores) {
        all_cores_set.insert((CoreRange) {
            .start={0, num_full_rows}, .end={last_row_cores - 1, num_full_rows}
        });
    }
    CoreRangeSet all_cores(all_cores_set);

    auto reader_kernels = CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/reader_unary_8bank_ln.cpp", all_cores,
        DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);
        //DataMovementProcessor::RISCV_1, NOC::RISCV_0_default);
        //DataMovementProcessor::RISCV_1, core.x < 6 ? NOC::RISCV_1_default : NOC::RISCV_0_default);
        //DataMovementProcessor::RISCV_1, core.x % 6 == 1 ? NOC::RISCV_1_default : NOC::RISCV_0_default);
        //DataMovementProcessor::RISCV_1, core.y < 5 ? NOC::RISCV_1_default : NOC::RISCV_0_default);

    auto writer_kernels = CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/writer_unary_8bank_ln.cpp", all_cores,
        DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);
        //DataMovementProcessor::RISCV_0, NOC::RISCV_1_default);
        //DataMovementProcessor::RISCV_0, core.x < 6 ? NOC::RISCV_0_default : NOC::RISCV_1_default);
        //DataMovementProcessor::RISCV_0, core.y < 5 ? NOC::RISCV_0_default : NOC::RISCV_1_default);
        //DataMovementProcessor::RISCV_0, core.x % 6 == 1 ? NOC::RISCV_0_default : NOC::RISCV_1_default);
        //DataMovementProcessor::RISCV_0, NOC::NOC_1);

    vector<uint32_t> compute_args = { wtpc, Wt, num_gamma_tiles>0, num_beta_tiles>0 };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = true;
    auto eltwise_binary_kernels = CreateComputeKernel(
        program,
        "kernels/compute/layernorm.cpp",
        all_cores,
        compute_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    // Add defines
    eltwise_binary_kernels->add_define("BLOCK_SIZE", block_size);
    reader_kernels->add_define("BLOCK_SIZE", block_size);
    reader_kernels->add_define("A_DRAM", is_dram(a) ? "true" : "false");
    reader_kernels->add_define("B_DRAM", is_dram(b) ? "true" : "false");
    reader_kernels->add_define("GAMMA_DRAM", is_dram(gamma) ? "true" : "false");
    reader_kernels->add_define("BETA_DRAM", is_dram(beta) ? "true" : "false");

    writer_kernels->add_define("BLOCK_SIZE", block_size);
    writer_kernels->add_define("OUTPUT_DRAM", is_dram(output) ? "true" : "false");
    if (b) {
        reader_kernels->add_define("FUSE_PRE_ADD", "1");
        eltwise_binary_kernels->add_define("FUSE_PRE_ADD", "1");
    }

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

        union { float f; uint32_t u; } winv; winv.f = 1.0f / W; // bcast-w scaler
        union { float f; uint32_t u; } e; e.f = eps; // epsilon
        uint32_t gamma_tiles = gamma ? gamma.value().get().volume() / TILE_HW : 0;
        uint32_t beta_tiles = beta ? beta.value().get().volume() / TILE_HW : 0;
        //std::cout << "Num gamma=" << num_gamma_tiles << " addr=" << gamma_dram_addr << std::endl;
        //std::cout << "Num beta=" << num_beta_tiles << " addr=" << beta_dram_addr << std::endl;
        uint32_t tile_offset = wtpc*Wt*icore;
        //std::cout << "icore=" << icore << "TO=" << tile_offset << endl;
        WriteRuntimeArgsToDevice( device, reader_kernels, core,
            { a_addr, wtpc, Wt, wtpc*Wt, tile_offset, 0, 0, 0, winv.u, e.u, // 0-9
            num_gamma_tiles, gamma_dram_addr, num_beta_tiles, beta_dram_addr, b_dram_addr } // 10-14
        );
        WriteRuntimeArgsToDevice( device, writer_kernels, core, { dst_addr, 0, 0, wtpc*Wt, tile_offset } );
    }

    return program;
}

void ResidualLayerNorm::validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    TT_ASSERT(input_tensors.size() > 0 and input_tensors.size() <= 4, "Must have between 1 to 4 input tensors");
}

std::vector<Shape> ResidualLayerNorm::compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0).get();
    return {input_tensor.shape()};
}

std::vector<Tensor> ResidualLayerNorm::create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    return operation::generic_create_output_tensors(*this, input_tensors);
}

Program ResidualLayerNorm::create_program(
    const std::vector<std::reference_wrapper<const Tensor>>& input_tensors,
    const std::vector<std::optional<std::reference_wrapper<const Tensor>>>& optional_input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    auto& a = input_tensors.at(0).get();
    const auto b = optional_input_tensors.at(0);
    const auto gamma = optional_input_tensors.at(1);
    const auto beta = optional_input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);
    return layernorm_(a, b, gamma, beta, output_tensor, this->eps, this->out_dram);

}

Tensor layernorm(const Tensor &a, float eps, bool out_dram) {
    return std::move(operation::run(ResidualLayerNorm{.eps=eps, .out_dram=out_dram}, {a}, {std::nullopt, std::nullopt, std::nullopt}).at(0));
}
Tensor layernorm_gamma(const Tensor &a, float eps, const Tensor& gamma, bool out_dram) {
    return std::move(operation::run(ResidualLayerNorm{.eps=eps, .out_dram=out_dram}, {a}, {std::nullopt, gamma, std::nullopt}).at(0));
}
Tensor layernorm_gamma_beta(const Tensor &a, float eps, const Tensor& gamma, const Tensor& beta, bool out_dram) {
    return std::move(operation::run(ResidualLayerNorm{.eps=eps, .out_dram=out_dram}, {a}, {std::nullopt, gamma, beta}).at(0));
}
Tensor add_layernorm_gamma_beta(const Tensor &a, const Tensor& b, float eps, const Tensor& gamma, const Tensor& beta, bool out_dram) {
    return std::move(operation::run(ResidualLayerNorm{.eps=eps, .out_dram=out_dram}, {a}, {b, gamma, beta}).at(0));
}



}  // namespace ll_buda

}  // namespace tt
