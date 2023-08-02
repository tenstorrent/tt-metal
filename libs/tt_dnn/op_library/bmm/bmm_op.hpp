#pragma once
#include <optional>

#include "tensor/tensor.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization
enum class BmmOpParallelizationStrategy {
    MULTI_CORE = 0,
    MULTI_CORE_REUSE = 1,
    MULTI_CORE_REUSE_MCAST = 2,
    MULTI_CORE_REUSE_GENERALIZED = 3,
    MULTI_CORE_REUSE_MCAST_GENERALIZED = 4,
    MULTI_CORE_REUSE_PADDING = 5,
    MULTI_CORE_REUSE_MCAST_PADDING = 6,
    SINGLE_CORE = 7
};


/*
 * GENERAL MATMUL AND BMM
 */
operation::ProgramWithCallbacks matmul_single_core  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse_generalized  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_generalized  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse_padding (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);
operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_padding (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, bool bcast_batch);

struct Matmul {
    bool bcast_batch;
    const MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    BmmOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};


inline Tensor matmul (const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    TT_ASSERT(input_tensor_a.shape()[3] == input_tensor_b.shape()[2] && "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in bmm_op"); // A.K == B.K
    TT_ASSERT(input_tensor_b.shape()[0]*input_tensor_b.shape()[1] == 1 && "matmul (batch bcast variant) expects input tensors of shapes BCMK*11KN=BCMN");
    return operation::run_with_autoformat(Matmul{.bcast_batch=true, .output_mem_config=mem_config}, {input_tensor_a, input_tensor_b}).at(0);
}
inline Tensor bmm    (const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    TT_ASSERT(input_tensor_a.shape()[3] == input_tensor_b.shape()[2] && "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in bmm_op"); // A.K == B.K
    TT_ASSERT(input_tensor_a.shape()[1] == input_tensor_b.shape()[1] && input_tensor_a.shape()[0] == input_tensor_b.shape()[0]
        && "bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN");
    return operation::run_with_autoformat(Matmul{.bcast_batch=false, .output_mem_config=mem_config}, {input_tensor_a, input_tensor_b}).at(0);
}


/*
 * BERT LARGE MATMUL AND BMM
 */
enum class BertLargeMatmulOpType {
    FUSED_QKV = 0,
    FF1 = 1,
    FF2 = 2,
    SELFOUT = 3,
    PRE_SOFTMAX_BMM = 4,
    POST_SOFTMAX_BMM = 5,
};

operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_optimized_bert_large(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const std::optional<const Tensor> bias, Tensor &output_tensor, CoreCoord compute_with_storage_grid_size, tt::tt_metal::DataType output_dtype, MathFidelity math_fidelity, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch, bool gelu=false);
operation::ProgramWithCallbacks bmm_multi_core_reuse_optimized_bert_large(const Tensor& input_tensor_a, const Tensor& input_tensor_b, const Shape &ashape, const Shape &bshape, Tensor &output_tensor, CoreCoord compute_with_storage_grid_size, tt::tt_metal::DataType output_dtype, MathFidelity math_fidelity, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch);


struct BertLargeMatmul {
    const BertLargeMatmulOpType bert_large_matmul_op_type;
    const MemoryConfig output_mem_config;
    const DataType output_dtype;
    const bool fuse_gelu_activation;

    void validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
    tt::stl::reflection::Attributes attributes() const;
};


inline Tensor bert_large_fused_qkv_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype=std::nullopt) {
    return operation::run(BertLargeMatmul{BertLargeMatmulOpType::FUSED_QKV, mem_config, output_dtype.value_or(input_tensor_a.dtype()), false}, {input_tensor_a, input_tensor_b}, {bias}).at(0);
}
inline Tensor bert_large_ff1_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, bool fuse_gelu_activation, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype=std::nullopt) {
    return operation::run(BertLargeMatmul{BertLargeMatmulOpType::FF1, mem_config, output_dtype.value_or(input_tensor_a.dtype()), fuse_gelu_activation}, {input_tensor_a, input_tensor_b}, {bias}).at(0);
}
inline Tensor bert_large_ff2_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype=std::nullopt) {
    return operation::run(BertLargeMatmul{BertLargeMatmulOpType::FF2, mem_config, output_dtype.value_or(input_tensor_a.dtype()), false}, {input_tensor_a, input_tensor_b}, {bias}).at(0);
}
inline Tensor bert_large_selfout_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype=std::nullopt) {
    return operation::run(BertLargeMatmul{BertLargeMatmulOpType::SELFOUT, mem_config, output_dtype.value_or(input_tensor_a.dtype()), false}, {input_tensor_a, input_tensor_b}, {bias}).at(0);
}
inline Tensor bert_large_pre_softmax_bmm(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype=std::nullopt) {
    return operation::run(BertLargeMatmul{BertLargeMatmulOpType::PRE_SOFTMAX_BMM, mem_config, output_dtype.value_or(input_tensor_a.dtype()), false}, {input_tensor_a, input_tensor_b}, {std::nullopt}).at(0);
}
inline Tensor bert_large_post_softmax_bmm(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype=std::nullopt) {
    return operation::run(BertLargeMatmul{BertLargeMatmulOpType::POST_SOFTMAX_BMM, mem_config, output_dtype.value_or(input_tensor_a.dtype()), false}, {input_tensor_a, input_tensor_b}, {std::nullopt}).at(0);
}

/**
 * Generalized blocked matmul with support for tilize and untilize and mixed-prec
 */
struct BMMTilizeUntilize {
    const DataType out_dt_;
    const uint32_t in0_nblocks_h_, in0_nblocks_w_, in1_nblocks_w_;
    const uint32_t in0_block_ntiles_h_, in0_block_ntiles_w_, in1_block_ntiles_w_;
    const uint32_t out_subblock_ntiles_h_, out_subblock_ntiles_w_;
    const bool tilize_in0_, untilize_out_;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    stl::reflection::Attributes attributes() const;
};

/**
 * Blocked Matmul, with support for tilize a and untilize output.
 * NOTE: Takes blocks and subblock information as arguments.
 */
Tensor bmm_tilize_untilize(const Tensor& a, const Tensor& b, DataType out_dt,
                           uint32_t a_height_nblocks, uint32_t a_width_nblocks, uint32_t b_width_nblocks,
                           uint32_t a_block_height_ntiles, uint32_t a_block_width_ntiles, uint32_t b_block_width_ntiles,
                           uint32_t out_subblock_height_ntiles, uint32_t out_subblock_width_ntiles,
                           bool tilize_in0, bool untilize_out);
operation::ProgramWithCallbacks bmm_single_core_tilize_untilize(
                                    const Tensor &in0, const Tensor &in1, DataType out_dt,
                                    uint32_t in0_height_nblocks, uint32_t in0_width_nblocks, uint32_t in1_width_nblocks,
                                    uint32_t in0_block_height_ntiles, uint32_t in0_block_width_ntiles, uint32_t in1_block_width_ntiles,
                                    uint32_t out_subblock_height_ntiles, uint32_t out_subblock_width_ntiles,
                                    bool tilize_in0, bool untilize_out,
                                    Tensor &out);

}  // namespace tt_metal

}  // namespace tt

namespace bmm_op_utils {
using namespace tt::tt_metal;

tuple<uint32_t, uint32_t, uint32_t, uint32_t> get_large_matmul_params(uint32_t Mt, uint32_t Nt, uint32_t num_cores_y, uint32_t num_cores_x, uint32_t in0_block_w);

CoreCoord get_core_range(uint32_t num_blocks_rows, uint32_t num_blocks_cols, uint32_t max_num_rows, uint32_t max_num_cols);

}
