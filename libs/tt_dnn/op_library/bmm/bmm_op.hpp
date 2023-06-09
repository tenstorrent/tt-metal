#pragma once

#include "tensor/tensor.hpp"

#include "tt_dnn/op_library/operation.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization
struct BmmOpParallelizationStrategy {
    enum Enum {
        MULTI_CORE = 0,
        MULTI_CORE_REUSE = 1,
        MULTI_CORE_REUSE_MCAST = 2,
        MULTI_CORE_REUSE_GENERALIZED = 3,
        MULTI_CORE_REUSE_MCAST_GENERALIZED = 4,
        MULTI_CORE_REUSE_PADDING = 5,
        MULTI_CORE_REUSE_MCAST_PADDING = 6,
        SINGLE_CORE = 7
    };
    static const vector<Enum> all() {
        return {
            MULTI_CORE,
            MULTI_CORE_REUSE,
            MULTI_CORE_REUSE_MCAST,
            MULTI_CORE_REUSE_GENERALIZED,
            MULTI_CORE_REUSE_MCAST_GENERALIZED,
            MULTI_CORE_REUSE_PADDING,
            MULTI_CORE_REUSE_MCAST_PADDING,
            SINGLE_CORE
        };
    }
};


/*
 * GENERAL MATMUL AND BMM
 */
Program matmul_single_core  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // broadcasts batch, expects N=1 for now
Program bmm_single_core     (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // doesn't broadcast batch, expects batch to match in A and B
Program matmul_multi_core  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // broadcasts batch, expects N=1 for now
Program bmm_multi_core     (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // doesn't broadcast batch, expects batch to match in A and B
Program matmul_multi_core_reuse  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now
Program bmm_multi_core_reuse  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now
Program matmul_multi_core_reuse_mcast  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now
Program bmm_multi_core_reuse_mcast  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now
Program matmul_multi_core_reuse_generalized  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now
Program bmm_multi_core_reuse_generalized  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now
Program matmul_multi_core_reuse_mcast_generalized  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now
Program bmm_multi_core_reuse_mcast_generalized  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now
Program matmul_multi_core_reuse_padding (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now
Program bmm_multi_core_reuse_padding  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now
Program matmul_multi_core_reuse_mcast_padding (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now
Program bmm_multi_core_reuse_mcast_padding  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now


struct Matmul : Operation {

    Matmul() {}
    Matmul(const Matmul&) = delete;
    Matmul& operator=(const Matmul&) = delete;
    ~Matmul() {}

    void validate(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const override;
    std::vector<Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const override;
    std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const override;
    Program create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const override;
};


struct BatchedMatmul : Operation {

    BatchedMatmul() {}
    BatchedMatmul(const BatchedMatmul&) = delete;
    BatchedMatmul& operator=(const BatchedMatmul&) = delete;
    ~BatchedMatmul() {}

    void validate(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const override;
    std::vector<Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const override;
    std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const override;
    Program create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const override;
};


inline Tensor matmul (const Tensor &input_tensor_a, const Tensor &input_tensor_b) {
    return detail::run_with_autopad(Matmul(), input_tensor_a, input_tensor_b);
}
inline Tensor bmm    (const Tensor &input_tensor_a, const Tensor &input_tensor_b) {
    return detail::run_with_autopad(BatchedMatmul(), input_tensor_a, input_tensor_b);
}


/*
 * BERT LARGE MATMUL AND BMM
 */
enum class BertLargeMatmulOpType {
    FUSED_QKV = 0,
    FF1 = 1,
    FF2 = 2,
    SELFOUT = 3
};

Program matmul_multi_core_reuse_mcast_optimized_bert_large(const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor &output_tensor, CoreCoord compute_and_storage_grid_size, tt::DataFormat output_cb_data_format, MathFidelity math_fidelity, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch);
Tensor bmm_multi_core_reuse_optimized_bert_large(const Tensor& input_tensor_a, const Tensor& input_tensor_b, const std::array<uint32_t, 4> &ashape, const std::array<uint32_t, 4> &bshape, const std::array<uint32_t, 4> &cshape, const MemoryConfig& mem_config, CoreCoord compute_and_storage_grid_size, tt::DataFormat output_cb_data_format, MathFidelity math_fidelity, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch);


struct BertLargeMatmul : Operation {
    BertLargeMatmulOpType bert_large_matmul_op_type;
    MemoryConfig output_mem_config;

    BertLargeMatmul(BertLargeMatmulOpType bert_large_matmul_op_type, MemoryConfig output_mem_config) : bert_large_matmul_op_type(bert_large_matmul_op_type), output_mem_config(output_mem_config) {}
    BertLargeMatmul(const BertLargeMatmul&) = delete;
    BertLargeMatmul& operator=(const BertLargeMatmul&) = delete;
    ~BertLargeMatmul() {}

    void validate(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const override;
    std::vector<Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const override;
    std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const override;
    Program create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const override;
};


inline Tensor bert_large_fused_qkv_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config) {
    return std::move(BertLargeMatmul(BertLargeMatmulOpType::FUSED_QKV, mem_config).run({std::cref(input_tensor_a), std::cref(input_tensor_b)}).at(0));
}
inline Tensor bert_large_ff1_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config) {
    return std::move(BertLargeMatmul(BertLargeMatmulOpType::FF1, mem_config).run({std::cref(input_tensor_a), std::cref(input_tensor_b)}).at(0));
}
inline Tensor bert_large_ff2_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config) {
    return std::move(BertLargeMatmul(BertLargeMatmulOpType::FF2, mem_config).run({std::cref(input_tensor_a), std::cref(input_tensor_b)}).at(0));
}
inline Tensor bert_large_selfout_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config) {
    return std::move(BertLargeMatmul(BertLargeMatmulOpType::SELFOUT, mem_config).run({std::cref(input_tensor_a), std::cref(input_tensor_b)}).at(0));
}
Tensor bert_large_pre_softmax_bmm(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config);
Tensor bert_large_post_softmax_bmm(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config);


// TODO: Refactor and uplift these bmms
Tensor large_bmm(const Tensor &input_tensor_a, const Tensor &input_tensor_b, bool tilize_act, bool untilize_out); // Tilizes, untilizes b
Tensor large_bmm_single_block(const Tensor &input_tensor_a, const Tensor &input_tensor_b, bool tilize_a, bool untilize_out); // Allows support for tilizing a, untilize b
Tensor bmm_tilize_untilize(const Tensor& a, const Tensor& b,
                           uint32_t a_height_nblocks, uint32_t a_width_nblocks, uint32_t b_width_nblocks,
                           uint32_t a_block_height_ntiles, uint32_t a_block_width_ntiles, uint32_t b_block_width_ntiles,
                           uint32_t out_subblock_height_ntiles, uint32_t out_subblock_width_ntiles,
                           bool tilize_b, bool untilize_out);
Tensor bmm_single_core_tilize_untilize(const Tensor &input_tensor_a, const Tensor &input_tensor_b,
                                       uint32_t a_height_nblocks, uint32_t a_width_nblocks, uint32_t b_width_nblocks,
                                       uint32_t a_block_height_ntiles, uint32_t a_block_width_ntiles, uint32_t b_block_width_ntiles,
                                       uint32_t out_subblock_height_ntiles, uint32_t out_subblock_width_ntiles,
                                       bool tilize_b, bool untilize_out);
Tensor large_bmm_single_core(const Tensor &input_tensor_a, const Tensor &input_tensor_b, bool tilize_act, bool untilize_out); // Tilizes a, untilizes b
Tensor large_bmm_single_core_single_block(const Tensor &input_tensor_a, const Tensor &input_tensor_b, bool tilize_a, bool untilize_out); // Allows support for tilizing a, untilize b


// TODO: Merge/delete these (un)used matmuls/bmms
Tensor matmul_multi_core_reuse_mcast_padding_generalized(const Tensor &input_tensor_a, const Tensor &input_tensor_b, CoreCoord compute_and_storage_grid_size, tt::DataFormat output_cb_data_format, MathFidelity math_fidelity, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch);
Tensor bmm_multi_core_reuse_mcast_padding_generalized(const Tensor &input_tensor_a, const Tensor &input_tensor_b, CoreCoord compute_and_storage_grid_size, tt::DataFormat output_cb_data_format, MathFidelity math_fidelity, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch);
Tensor matmul_multi_core_reuse_generalized_bert_large  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, CoreCoord compute_and_storage_grid_size, tt::DataFormat output_cb_data_format, MathFidelity math_fidelity, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch); // No actual padding
Tensor bmm_multi_core_reuse_generalized_bert_large  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, CoreCoord compute_and_storage_grid_size, tt::DataFormat output_cb_data_format, MathFidelity math_fidelity, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch); // No actual padding


}  // namespace tt_metal

}  // namespace tt

namespace bmm_op_utils {
using namespace tt::tt_metal;

tuple<uint32_t, uint32_t, uint32_t, uint32_t> get_large_matmul_params(uint32_t Mt, uint32_t Nt, uint32_t num_cores_y, uint32_t num_cores_x, uint32_t in0_block_w);

CoreCoord get_core_range(uint32_t num_blocks_rows, uint32_t num_blocks_cols, uint32_t max_num_rows, uint32_t max_num_cols);

BmmOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a, const Tensor &b);

}
