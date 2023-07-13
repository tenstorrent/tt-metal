#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include<string>

namespace tt {

namespace tt_metal {

struct TransposeOpDim {
    enum Enum { WH = 0, HC = 1, CN = 2, NH = 3, NW = 4, CW = 5 };
    static const vector<Enum> all() { return { WH, HC, CN, NH, NW, CW }; }
};

struct TransposeOpParallelizationStrategy {
    enum Enum { MULTI_CORE_WH = 0, MULTI_CORE_HC = 1, SINGLE_CORE = 2 };
    static const vector<Enum> all() { return { MULTI_CORE_WH, MULTI_CORE_HC, SINGLE_CORE }; }
};

struct Transpose {
    const TransposeOpDim::Enum dim;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
    TransposeOpParallelizationStrategy::Enum get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

// TODO: Accept parallelization
Tensor transpose_(const Tensor &a, TransposeOpDim::Enum transpose_dim=TransposeOpDim::WH);
// TODO: Don't bind transpose as transpose_wh, should explicitly bind like the others
// Alternatively, bind only 1 transpose function and take 2 dims to transpose
Tensor transpose(const Tensor &a);
Tensor transpose(const Tensor &a,uint dim_a, uint dim_b);
// 4 choose 2 = 6 transposes on NCHW rank-4 tensors without order.
// Unique transposes : ('n', 'c'), ('n', 'h'), ('n', 'w'), ('c', 'h'), ('c', 'w'), ('h', 'w')
Tensor transpose_wh(const Tensor &a);
Tensor transpose_hc(const Tensor &a);
Tensor transpose_cn(const Tensor &a);
Tensor transpose_nh(const Tensor &a);
Tensor transpose_nw(const Tensor &a);
Tensor transpose_cw(const Tensor &a);
// transpose with tensor and dimensions
Tensor transpose(const Tensor &a, uint dim1, uint dim2);

// provide access to transposes on a [n,c,h,w] ranked tensor @a
Tensor transpose_(const Tensor &a,char dim_a, char dim_b);
Tensor transpose_(const Tensor &a,std::array<uint32_t,2> dim_a_b);

operation::ProgramWithCallbacks transpose_single_core(const Tensor &a, Tensor &output, TransposeOpDim::Enum transpose_dim);
operation::ProgramWithCallbacks transpose_wh_multi_core(const Tensor &a, Tensor &output);
operation::ProgramWithCallbacks transpose_hc_multi_core(const Tensor &a, Tensor &output);

}  // namespace tt_metal

}  // namespace tt
