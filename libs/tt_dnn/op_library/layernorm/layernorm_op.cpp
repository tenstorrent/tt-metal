#include "libs/tt_dnn/op_library/layernorm/layernorm_op.hpp"
#include "libs/tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

#include <optional>

using u32 = std::uint32_t;
using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

inline bool is_dram(const Tensor& input_tensor) { return input_tensor.memory_config().buffer_type == BufferType::DRAM; }
inline bool is_dram(const std::optional<const Tensor> input_tensor) {
     return input_tensor.has_value() ? is_dram(input_tensor.value()) : true;
}
inline bool is_dram(const Buffer* b) { return b->buffer_type() == BufferType::DRAM; }

// computes layernorm(a+*b)*gamma + beta
// if b is nullptr it's treated as zero (no addition)
operation::ProgramWithCallbacks layernorm_(
    const Tensor &a,
    const std::optional<const Tensor> b,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    Tensor& output,
    float eps,
    bool rm_gb = false,
    bool rms_norm = false
) {

    const auto shape = a.shape();
    u32 W = shape[3], H = shape[2], NC = shape[1]*shape[0];
    u32 HW = H*W;

    // Kernels are configured to support BFLOAT8_B, but bad pcc so we need mixed precision support in compute
    const auto& a_dtype = a.dtype();

    u32 Wt = W/TILE_WIDTH;
    u32 Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = NC*H*W / TILE_HW;

    uint32_t block_size = find_max_divisor(Wt, 8);

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt_metal::TileSize(cb_data_format);
    uint32_t bfloat16_tile_size = tt_metal::TileSize(tt::DataFormat::Float16_b);

    auto a_addr = a.buffer()->address();
    auto b_dram_addr = b ? b.value().buffer()->address() : 0;
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().buffer()->address() : 0;

    uint32_t num_tiles = a.volume()/TILE_HW;
    uint32_t num_gamma_tiles = gamma.has_value() ? gamma.value().volume()/TILE_HW : 0;
    uint32_t num_beta_tiles = beta.has_value() ? beta.value().volume()/TILE_HW : 0;

    // For bert, tensor is packed as RM with width 32
    if (rm_gb) {
        num_gamma_tiles = gamma.has_value() ? gamma.value().volume()/TILE_WIDTH : 0;
        num_beta_tiles = beta.has_value() ? beta.value().volume()/TILE_WIDTH : 0;
    }


    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    Device *device = a.device();
    auto dst_addr = output.buffer()->address();


    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
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
    TT_ASSERT(NCHt % num_cores == 0);

    // we are actually splitting blocks of Wt tiles, not tiles, so no checking for bank alignment is needed
    TilesSplit ts(num_cores, NCHt);
    auto wtpc = ts.get_tpc(); // Wt*tpc per core
    TT_ASSERT(NCHt % wtpc == 0);


    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = Program();

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

    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) static_cast<uint32_t>(cb_data_format),
        (std::uint32_t) is_dram(a),
        (std::uint32_t) is_dram(b),
        (std::uint32_t) is_dram(gamma),
        (std::uint32_t) is_dram(beta),
        (std::uint32_t) block_size
    };
    if (rm_gb) {
        if (gamma.has_value()) {
            auto gamma_stick_size = gamma.value().shape()[3] * gamma.value().element_size();
            bool gamma_stick_size_is_power_of_two = is_power_of_two_at_least_32(gamma_stick_size);
            reader_compile_time_args.push_back((std::uint32_t) gamma_stick_size_is_power_of_two);
            if (gamma_stick_size_is_power_of_two) {
                uint32_t gamma_log2_stick_size = gamma_stick_size_is_power_of_two ? (std::uint32_t)log2(gamma_stick_size) : 0;
                reader_compile_time_args.push_back((std::uint32_t) gamma_log2_stick_size);
            } else {
                reader_compile_time_args.push_back(gamma_stick_size);
            }
        } else if (beta.has_value()) {
            auto beta_stick_size = beta.value().shape()[3] * beta.value().element_size();
            bool beta_stick_size_is_power_of_two = is_power_of_two_at_least_32(beta_stick_size);
            reader_compile_time_args.push_back((std::uint32_t) beta_stick_size_is_power_of_two);
            if (beta_stick_size_is_power_of_two) {
                uint32_t beta_log2_stick_size = beta_stick_size_is_power_of_two ? (std::uint32_t)log2(beta_stick_size) : 0;
                reader_compile_time_args.push_back((std::uint32_t) beta_log2_stick_size);
            } else {
                reader_compile_time_args.push_back(beta_stick_size);
            }
        } else {
            reader_compile_time_args.push_back(0);
            reader_compile_time_args.push_back(0);
        }
    }

    std::vector<uint32_t> writer_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) static_cast<uint32_t>(cb_data_format),
        (std::uint32_t) is_dram(output),
        (std::uint32_t) block_size
    };


    CoreRangeSet all_cores(all_cores_set);

    bool tile_dtype_is_bfloat16 = a.dtype() == tt::tt_metal::DataType::BFLOAT16;
    auto reader_kernels = CreateDataMovementKernel(
        program,
        rm_gb ? "tt_metal/kernels/dataflow/reader_unary_interleaved_ln_rm_gb.cpp" : "tt_metal/kernels/dataflow/reader_unary_interleaved_ln.cpp",
        all_cores,
        reader_compile_time_args,
        DataMovementProcessor::RISCV_1,
        NOC::RISCV_1_default
    );

    auto writer_kernels = CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp",
        all_cores,
        writer_compile_time_args,
        DataMovementProcessor::RISCV_0,
        NOC::RISCV_0_default
    );

    vector<uint32_t> compute_args = { wtpc, Wt, block_size, gamma.has_value(), beta.has_value() };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = true;
    auto eltwise_binary_kernels = CreateComputeKernel(
        program,
        rms_norm ? "kernels/compute/rmsnorm.cpp" : "kernels/compute/layernorm.cpp",
        all_cores,
        compute_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    if (b) {
        reader_kernels->add_define("FUSE_PRE_ADD", "1");
        eltwise_binary_kernels->add_define("FUSE_PRE_ADD", "1");
    }
    if (gamma.has_value()) {
        reader_kernels->add_define("FUSE_GAMMA", "1");
    }
    if (beta.has_value()) {
        reader_kernels->add_define("FUSE_BETA", "1");
    }
    // Create circular buffers
    CreateCircularBuffers( program, CB::c_in0,       all_cores, in0_t,  in0_t*single_tile_size,  cb_data_format );
    CreateCircularBuffers( program, CB::c_out0,      all_cores, out0_t, out0_t*single_tile_size, cb_data_format );
    CreateCircularBuffers( program, CB::c_intermed1, all_cores, im1_t,  im1_t*single_tile_size,  cb_data_format );
    CreateCircularBuffers( program, CB::c_in2,       all_cores, in2_t,  in2_t*bfloat16_tile_size,  DataFormat::Float16_b );
    CreateCircularBuffers( program, CB::c_in3,       all_cores, in3_t,  in3_t*bfloat16_tile_size,  DataFormat::Float16_b );
    CreateCircularBuffers( program, CB::c_intermed2, all_cores, im2_t,  im2_t*single_tile_size,  cb_data_format );
    if (!rms_norm) {
        CreateCircularBuffers( program, CB::c_intermed0, all_cores, im0_t,  im0_t*single_tile_size,  cb_data_format );
    }
    CreateCircularBuffers( program, CB::c_intermed3, all_cores, im3_t,  im3_t*single_tile_size,  cb_data_format );
    CreateCircularBuffers( program, CB::c_intermed4, all_cores, im4_t,  im4_t*single_tile_size,  cb_data_format );
    if (gamma.has_value() || beta.has_value()) {
        CreateCircularBuffers( program, CB::c_intermed5, all_cores, im5_t,  im5_t*single_tile_size,  cb_data_format );
    }
    if (gamma.has_value()) {
        CreateCircularBuffers( program, CB::c_in5,       all_cores, in5_t,  in5_t*(rm_gb ? bfloat16_tile_size : single_tile_size),  rm_gb ? DataFormat::Float16_b : cb_data_format );
    }
    if (beta.has_value()) {
        CreateCircularBuffers( program, CB::c_in6,       all_cores, in6_t,  in6_t*(rm_gb ? bfloat16_tile_size : single_tile_size),  rm_gb ? DataFormat::Float16_b : cb_data_format );
    }
    if (b) {
        // x = a+b in this notation
        // result = ln(x)*gamma + beta
        // if there's no pre-add we use cb_in0 for x, otherwise a is pre-buffered into in0, added into im6, then im6 is used as x
        // b is buffered into c_in1
        CreateCircularBuffers( program, CB::c_intermed6, all_cores, im6_t,  im6_t*single_tile_size,  cb_data_format );
        // c_in1 is input buffer for b
        CreateCircularBuffers( program, CB::c_in1,       all_cores, in1_t,  in1_t*single_tile_size,  cb_data_format );
    }

    union { float f; uint32_t u; } winv; winv.f = 1.0f / W; // bcast-w scaler
    union { float f; uint32_t u; } e; e.f = eps; // epsilon
    for (uint32_t icore = 0; icore < num_cores; icore++) {
        auto core = grid.wrap_core(icore);

        uint32_t tile_offset = wtpc*Wt*icore;
        SetRuntimeArgs(reader_kernels, core,
            { a_addr, wtpc, Wt, tile_offset, winv.u, e.u, // 0-5
            gamma_dram_addr, beta_dram_addr, b_dram_addr } // 6-8
        );
        SetRuntimeArgs(writer_kernels, core, { dst_addr, wtpc*Wt, tile_offset } );
    }

    auto override_runtime_args_callback = [
            reader_kernel=reader_kernels,
            writer_kernel=writer_kernels,
            num_cores,
            grid
        ]
    (
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_a_dram_buffer = input_buffers.at(0);
        auto src_b_dram_buffer = input_buffers.at(1);
        auto gamma_dram_buffer = input_buffers.at(2);
        auto beta_dram_buffer = input_buffers.at(3);

        auto dst_dram_buffer = output_buffers.at(0);

        for (uint32_t icore = 0; icore < num_cores; icore++) {
            auto core = grid.wrap_core(icore);

            {
                auto runtime_args = GetRuntimeArgs(reader_kernel, core);
                runtime_args[0] = src_a_dram_buffer->address();
                if (src_b_dram_buffer != nullptr) {
                    runtime_args[8] = src_b_dram_buffer->address();
                }
                if (gamma_dram_buffer != nullptr) {
                    runtime_args[6] = gamma_dram_buffer->address();
                }
                if (beta_dram_buffer != nullptr) {
                    runtime_args[7] = beta_dram_buffer->address();
                }
                SetRuntimeArgs(reader_kernel, core, runtime_args);
            }

            {
                auto runtime_args = GetRuntimeArgs(writer_kernel, core);
                runtime_args[0] = dst_dram_buffer->address();
                SetRuntimeArgs(writer_kernel, core, runtime_args);
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

void LayerNorm::validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_ASSERT(input_tensors.size() == 1 and optional_input_tensors.size() <= 3, "Must have between 1 to 4 input tensors");
    auto& a = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    TT_ASSERT(a.layout() == Layout::TILE);
    TT_ASSERT(a.dtype() == DataType::BFLOAT16 or a.dtype() == DataType::BFLOAT8_B);
    TT_ASSERT(a.storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
    if (b.has_value()) {
        TT_ASSERT(b.value().layout() == Layout::TILE);
        TT_ASSERT(a.shape() == b.value().shape());
        TT_ASSERT(a.device() == b.value().device());
        TT_ASSERT(b.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
    }
    if (gamma.has_value()) {
        TT_ASSERT(gamma.value().layout() == Layout::TILE);
        TT_ASSERT(a.shape()[3] == gamma.value().shape()[3]);
        TT_ASSERT(a.device() == gamma.value().device());
        TT_ASSERT(gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        TT_ASSERT(gamma.value().shape()[2] == TILE_HEIGHT);
    }
    if (beta.has_value()) {
        TT_ASSERT(beta.value().layout() == Layout::TILE);
        TT_ASSERT(a.shape()[3] == beta.value().shape()[3]);
        TT_ASSERT(a.device() == beta.value().device());
        TT_ASSERT(beta.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        TT_ASSERT(beta.value().shape()[2] == TILE_HEIGHT);
    }

}

std::vector<Shape> LayerNorm::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.shape()};
}

std::vector<Tensor> LayerNorm::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    return operation::generic_create_output_tensors(*this, input_tensors, Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks LayerNorm::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    const auto& a = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);
    return layernorm_(a, b, gamma, beta, output_tensor, this->eps);

}

operation::Hash LayerNorm::compute_program_hash(
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors
) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    return fmt::format("{}_{}_{}_{}_{}", *this, input_tensor, b, gamma, beta);
}

tt::stl::reflection::Attributes LayerNorm::attributes() const {
    return {
        {"eps", this->eps},
        {"output_mem_config", this->output_mem_config},
    };
}


void BertLargeLayerNorm::validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_ASSERT(input_tensors.size() == 1 and optional_input_tensors.size() <= 3, "Must have between 1 to 4 input tensors");
    auto& a = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    TT_ASSERT(a.layout() == Layout::TILE);
    TT_ASSERT(a.dtype() == DataType::BFLOAT16 or a.dtype() == DataType::BFLOAT8_B);
    TT_ASSERT(a.storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
    if (b.has_value()) {
        TT_ASSERT(b.value().layout() == Layout::TILE);
        TT_ASSERT(a.shape() == b.value().shape());
        TT_ASSERT(a.device() == b.value().device());
        TT_ASSERT(b.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
    }
    if (gamma.has_value()) {
        TT_ASSERT(gamma.value().layout() == Layout::ROW_MAJOR);
        TT_ASSERT((gamma.value().shape()[3] == TILE_WIDTH && gamma.value().volume() / TILE_WIDTH == a.shape()[3] / TILE_WIDTH));
        TT_ASSERT(a.device() == gamma.value().device());
        TT_ASSERT(gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        TT_ASSERT(gamma.value().dtype() == DataType::BFLOAT16);
    }
    if (beta.has_value()) {
        TT_ASSERT(beta.value().layout() == Layout::ROW_MAJOR);
        TT_ASSERT((beta.value().shape()[3] == TILE_WIDTH && beta.value().volume() / TILE_WIDTH == a.shape()[3] / TILE_WIDTH));
        TT_ASSERT(a.device() == beta.value().device());
        TT_ASSERT(beta.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        TT_ASSERT(beta.value().dtype() == DataType::BFLOAT16);
    }

}

std::vector<Shape> BertLargeLayerNorm::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.shape()};
}

std::vector<Tensor> BertLargeLayerNorm::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    return operation::generic_create_output_tensors(*this, input_tensors, Layout::TILE, this->output_mem_config);
}


operation::ProgramWithCallbacks BertLargeLayerNorm::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    const auto& a = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);
    return layernorm_(a, b, gamma, beta, output_tensor, this->eps, true);

}

operation::Hash BertLargeLayerNorm::compute_program_hash(
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors
) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    return fmt::format("{}_{}_{}_{}_{}", *this, input_tensor, b, gamma, beta);
}

tt::stl::reflection::Attributes BertLargeLayerNorm::attributes() const {
    return {
        {"eps", this->eps},
        {"output_mem_config", this->output_mem_config},
    };
}

void RMSNorm::validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_ASSERT(input_tensors.size() == 1 and optional_input_tensors.size() <= 3, "Must have between 1 to 4 input tensors");
    auto& a = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    TT_ASSERT(a.layout() == Layout::TILE);
    TT_ASSERT(a.dtype() == DataType::BFLOAT16 or a.dtype() == DataType::BFLOAT8_B);
    TT_ASSERT(a.storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
    if (b.has_value()) {
        TT_ASSERT(b.value().layout() == Layout::TILE);
        TT_ASSERT(a.shape() == b.value().shape());
        TT_ASSERT(a.device() == b.value().device());
        TT_ASSERT(b.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
    }
    if (gamma.has_value()) {
        TT_ASSERT(gamma.value().layout() == Layout::TILE);
        TT_ASSERT(a.shape()[3] == gamma.value().shape()[3]);
        TT_ASSERT(a.device() == gamma.value().device());
        TT_ASSERT(gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        TT_ASSERT(gamma.value().shape()[2] == TILE_HEIGHT);
    }
    if (beta.has_value()) {
        TT_ASSERT(beta.value().layout() == Layout::TILE);
        TT_ASSERT(a.shape()[3] == beta.value().shape()[3]);
        TT_ASSERT(a.device() == beta.value().device());
        TT_ASSERT(beta.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        TT_ASSERT(beta.value().shape()[2] == TILE_HEIGHT);
    }

}

std::vector<Shape> RMSNorm::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.shape()};
}

std::vector<Tensor> RMSNorm::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    return operation::generic_create_output_tensors(*this, input_tensors, Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks RMSNorm::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    const auto& a = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);
    return layernorm_(a, b, gamma, beta, output_tensor, this->eps, false, true);

}

operation::Hash RMSNorm::compute_program_hash(
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors
) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    return fmt::format("{}_{}_{}_{}_{}", *this, input_tensor, b, gamma, beta);
}

tt::stl::reflection::Attributes RMSNorm::attributes() const {
    return {
        {"eps", fmt::format("{}", this->eps)},
        {"output_mem_config", fmt::format("{}", this->output_mem_config)},
    };
}

}  // namespace ll_buda

}  // namespace tt
