#include "tt_dnn/op_library/transpose/transpose_op.hpp"
#include "tt_dnn/op_library/permute/permute_op.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

using u32 = std::uint32_t;
using namespace tt::constants;

namespace tt {

namespace tt_metal {

void Transpose::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operands to transpose need to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr , "Operands to transpose need to be allocated in buffers on device!");
    const auto shape = input_tensor.shape();
    u32 W = shape[3], H = shape[2], C = shape[1], N = shape[0];
    u32 HW = H*W;
    TT_ASSERT(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
    TT_ASSERT(input_tensor.volume() % TILE_HW == 0);
    if (this->dim == TransposeOpDim::HC) {
        TT_ASSERT(C % TILE_HEIGHT == 0);
        TT_ASSERT(input_tensor.dtype() == DataType::BFLOAT16);
    } else if (this->dim == TransposeOpDim::CW) {
        TT_ASSERT(C % TILE_WIDTH == 0);
        TT_ASSERT(input_tensor.dtype() == DataType::BFLOAT16);
    } else if (this->dim == TransposeOpDim::NH) {
        TT_ASSERT(N % TILE_HEIGHT == 0);
        TT_ASSERT(input_tensor.dtype() == DataType::BFLOAT16);
    } else if (this->dim == TransposeOpDim::NW) {
        TT_ASSERT(N % TILE_WIDTH == 0);
        TT_ASSERT(input_tensor.dtype() == DataType::BFLOAT16);
    }
}


std::vector<Shape> Transpose::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto out_shape = input_tensor.shape();
    switch (this->dim){
        case TransposeOpDim::CN:
            out_shape[0] = input_tensor.shape()[1];
            out_shape[1] = input_tensor.shape()[0];
            break;
        case TransposeOpDim::HC:
            out_shape[1] = input_tensor.shape()[2];
            out_shape[2] = input_tensor.shape()[1];
            break;
        case TransposeOpDim::WH:
            out_shape[2] = input_tensor.shape()[3];
            out_shape[3] = input_tensor.shape()[2];
            break;
        case TransposeOpDim::NH:
            out_shape[0] = input_tensor.shape()[2];
            out_shape[2] = input_tensor.shape()[0];
            break;
        case TransposeOpDim::NW:
            out_shape[3] = input_tensor.shape()[0];
            out_shape[0] = input_tensor.shape()[3];
            break;
        case TransposeOpDim::CW:
            out_shape[1] = input_tensor.shape()[3];
            out_shape[3] = input_tensor.shape()[1];
            break;
    }
    return {out_shape};
}


std::vector<Tensor> Transpose::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks Transpose::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    auto parallelization_strategy = this->get_parallelization_strategy(input_tensors);

    switch (parallelization_strategy) {
        case TransposeOpParallelizationStrategy::MULTI_CORE_WH:
            return transpose_wh_multi_core(input_tensor, output_tensor);
            break;
        case TransposeOpParallelizationStrategy::MULTI_CORE_HC:
            return transpose_hc_multi_core(input_tensor, output_tensor);
            break;
        default:
            return transpose_single_core(input_tensor, output_tensor, this->dim);
    }
}

operation::Hash Transpose::compute_program_hash(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return fmt::format("{}_{}", *this, input_tensor);
}

TransposeOpParallelizationStrategy::Enum Transpose::get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto ashape = input_tensor.shape();
    uint32_t num_tiles = input_tensor.volume() / TILE_HW;
    if (this->dim == TransposeOpDim::WH && num_tiles > 1) {
        return TransposeOpParallelizationStrategy::MULTI_CORE_WH;
    } else if (this->dim == TransposeOpDim::HC && num_tiles > 1) { // Always true for legal shape until requirement on tile size IO is no longer required
        return TransposeOpParallelizationStrategy::MULTI_CORE_HC;
    } else {
        return TransposeOpParallelizationStrategy::SINGLE_CORE;
    }
}

tt::stl::reflection::Attributes Transpose::attributes() const {
    return {
        {"dim", this->dim},
    };
}

inline Tensor transpose_(const Tensor &a, TransposeOpDim::Enum transpose_dim, const MemoryConfig& output_mem_config) {

    bool pad_c = false;
    bool pad_n = false;
    switch (transpose_dim) {
        case TransposeOpDim::CN:
            if (a.shape()[0] == 1 && a.shape()[1] == 1) {
                return a;
            }
            break;
        case TransposeOpDim::HC:
            if (a.shape()[1] == 1 && a.shape()[2] == 1) {
                return a;
            }
            pad_c = true;
            break;
        case TransposeOpDim::WH:
            if (a.shape()[2] == 1 && a.shape()[3] == 1) {
                return a;
            }
            break;
        case TransposeOpDim::NH:
            if (a.shape()[0] == 1 && a.shape()[2] == 1) {
                return a;
            }
            return transpose_nh(a);
            pad_n = true;
            break;
        case TransposeOpDim::NW:
            if (a.shape()[0] == 1 && a.shape()[3] == 1) {
                return a;
            }
            return transpose_nw(a);
            pad_n = true;
            break;
        case TransposeOpDim::CW:
            if (a.shape()[1] == 1 && a.shape()[3] == 1) {
                return a;
            }
            return transpose_cw(a);
            pad_c = true;
            break;
        default:
            TT_ASSERT( false && "unexpected operator mode for transpose ");
    }

    // TODO: Add pad_n to run_with_autoformat when needed
    return operation::run_with_autoformat(Transpose{transpose_dim, output_mem_config}, {a}, {}, 0, pad_c /*, pad_n */).at(0);
}

// TODO: Don't bind transpose as transpose_wh, should explicitly bind like the others
// Alternatively, bind only 1 transpose function and take 2 dims to transpose
Tensor transpose(const Tensor &a, const MemoryConfig& output_mem_config) { return transpose_(a, TransposeOpDim::WH, output_mem_config); }
// 4 choose 2 = 6 transposes on NCHW rank-4 tensors without order.
// Unique transposes : ('n', 'c'), ('n', 'h'), ('n', 'w'), ('c', 'h'), ('c', 'w'), ('h', 'w')
Tensor transpose_wh(const Tensor &a, const MemoryConfig& output_mem_config) { return transpose_(a, TransposeOpDim::WH, output_mem_config); }
Tensor transpose_hc(const Tensor &a, const MemoryConfig& output_mem_config) { return transpose_(a, TransposeOpDim::HC, output_mem_config); }
Tensor transpose_cn(const Tensor &a, const MemoryConfig& output_mem_config) { return transpose_(a, TransposeOpDim::CN, output_mem_config); }

Tensor transpose_nh(const Tensor &a, const MemoryConfig& output_mem_config) { return permute(a,2,1,0,3, output_mem_config); }
Tensor transpose_nw(const Tensor &a, const MemoryConfig& output_mem_config) { return permute(a,3,1,2,0, output_mem_config); }
Tensor transpose_cw(const Tensor &a, const MemoryConfig& output_mem_config) { return permute(a,0,3,2,1, output_mem_config); }

Tensor transpose(const Tensor &a, uint dim1, uint dim2, const MemoryConfig& output_mem_config) {
    TT_ASSERT( dim1 <= 3, "dimension have to be 0-3 only corresponding to N,C,H,W");
    TT_ASSERT( dim2 <= 3, "dimension have to be 0-3 only corresponding to N,C,H,W");

    if ( dim1 == dim2 ) {
        return a;
    }

    uint _t = dim1;
    if ( dim1 > dim2 ) {
        dim1 = dim2;
        dim2 = _t;
    }

    TT_ASSERT(dim2 > dim1,"expected order");

    // N C H W
    // 0 1 2 3

    if ( dim2 == 3 && dim1 == 0 ) {
        return transpose_nw(a);
    } else if (dim2 == 3 && dim1 == 1) {
        return transpose_cw(a);
    } else if (dim2 == 3 && dim1 == 2) {
        return transpose_wh(a);
    } else if (dim2 == 2 && dim1 == 0) {
        return transpose_nh(a);
    } else if (dim2 == 2 && dim1 == 1) {
        return transpose_hc(a);
    } else if (dim2 == 1 && dim1 == 0) {
        return transpose_cn(a);
    }
    TT_ASSERT(false,"unreachable");
    return a;
}

Tensor transpose(const Tensor &a, std::array<uint32_t,2> dim_a_b, const MemoryConfig& output_mem_config) {
    return transpose(a, dim_a_b[0], dim_a_b[1], output_mem_config);
}

// provide access to transposes on a [n,c,h,w] ranked tensor @a
Tensor transpose(const Tensor &a, char dim_a, char dim_b, const MemoryConfig& output_mem_config) {

    auto char_to_int_dim = [](char &char_dim) {
        switch(tolower(char_dim)) {
            case 'w': return 3;
            case 'h': return 2;
            case 'c': return 1;
            case 'n': return 0;
            default: TT_ASSERT("Unknown dim specified");
        }
        return 0;
    };

    uint32_t dim1 = char_to_int_dim(dim_a), dim2 = char_to_int_dim(dim_b);
    return transpose(a, dim1, dim2, output_mem_config);
}

}  // namespace tt_metal

}  // namespace tt
