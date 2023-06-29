#pragma once
#include <optional>

#include "tensor/tensor.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

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
operation::ProgramWithCallbacks matmul_single_core  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // broadcasts batch, expects N=1 for now
operation::ProgramWithCallbacks bmm_single_core     (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // doesn't broadcast batch, expects batch to match in A and B
operation::ProgramWithCallbacks matmul_multi_core  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // broadcasts batch, expects N=1 for now
operation::ProgramWithCallbacks bmm_multi_core     (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // doesn't broadcast batch, expects batch to match in A and B
operation::ProgramWithCallbacks matmul_multi_core_reuse  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now
operation::ProgramWithCallbacks bmm_multi_core_reuse  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now
operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now
operation::ProgramWithCallbacks bmm_multi_core_reuse_mcast  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now
operation::ProgramWithCallbacks matmul_multi_core_reuse_generalized  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now
operation::ProgramWithCallbacks bmm_multi_core_reuse_generalized  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now
operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_generalized  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now
operation::ProgramWithCallbacks bmm_multi_core_reuse_mcast_generalized  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now
operation::ProgramWithCallbacks matmul_multi_core_reuse_padding (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now
operation::ProgramWithCallbacks bmm_multi_core_reuse_padding  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now
operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_padding (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now
operation::ProgramWithCallbacks bmm_multi_core_reuse_mcast_padding  (const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor); // Only supports 2D matmul expects N=1 for now

struct Matmul {
    const MemoryConfig output_mem_config;
    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
    BmmOpParallelizationStrategy::Enum get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
};


struct BatchedMatmul {
    const MemoryConfig output_mem_config;
    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
    BmmOpParallelizationStrategy::Enum get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
};


inline Tensor matmul (const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config = MemoryConfig{.interleaved = true}) {
    TT_ASSERT(input_tensor_a.shape()[3] == input_tensor_b.shape()[2] && "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in bmm_op"); // A.K == B.K
    TT_ASSERT(input_tensor_b.shape()[0]*input_tensor_b.shape()[1] == 1 && "matmul (batch bcast variant) expects input tensors of shapes BCMK*11KN=BCMN");
    return operation::run_with_autoformat(Matmul{.output_mem_config=mem_config}, input_tensor_a, input_tensor_b);
}
inline Tensor bmm    (const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config = MemoryConfig{.interleaved = true}) {
    TT_ASSERT(input_tensor_a.shape()[3] == input_tensor_b.shape()[2] && "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in bmm_op"); // A.K == B.K
    TT_ASSERT(input_tensor_a.shape()[1] == input_tensor_b.shape()[1] && input_tensor_a.shape()[0] == input_tensor_b.shape()[0]
        && "bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN");
    return operation::run_with_autoformat(BatchedMatmul{.output_mem_config=mem_config}, input_tensor_a, input_tensor_b);
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

operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_optimized_bert_large(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const std::optional<const Tensor> bias, Tensor &output_tensor, CoreCoord compute_and_storage_grid_size, tt::DataFormat output_cb_data_format, MathFidelity math_fidelity, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch, bool gelu=false);
operation::ProgramWithCallbacks bmm_multi_core_reuse_optimized_bert_large(const Tensor& input_tensor_a, const Tensor& input_tensor_b, const Shape &ashape, const Shape &bshape, Tensor &output_tensor, CoreCoord compute_and_storage_grid_size, tt::DataFormat output_cb_data_format, MathFidelity math_fidelity, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch);


struct BertLargeMatmul {
    const BertLargeMatmulOpType bert_large_matmul_op_type;
    const MemoryConfig output_mem_config;
    const bool fuse_gelu_activation;

    void validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
    operation::Hash compute_program_hash(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors
    ) const;
};

std::ostream& operator<<(std::ostream& os, const BertLargeMatmul& op);


inline Tensor bert_large_fused_qkv_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config) {
    return operation::run(BertLargeMatmul{BertLargeMatmulOpType::FUSED_QKV, mem_config, false}, {input_tensor_a, input_tensor_b}, {bias}).at(0);
}
inline Tensor bert_large_ff1_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, bool fuse_gelu_activation, const MemoryConfig& mem_config) {
    return operation::run(BertLargeMatmul{BertLargeMatmulOpType::FF1, mem_config, fuse_gelu_activation}, {input_tensor_a, input_tensor_b}, {bias}).at(0);
}
inline Tensor bert_large_ff2_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config) {
    return operation::run(BertLargeMatmul{BertLargeMatmulOpType::FF2, mem_config, false}, {input_tensor_a, input_tensor_b}, {bias}).at(0);
}
inline Tensor bert_large_selfout_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config) {
    return operation::run(BertLargeMatmul{BertLargeMatmulOpType::SELFOUT, mem_config, false}, {input_tensor_a, input_tensor_b}, {bias}).at(0);
}
inline Tensor bert_large_pre_softmax_bmm(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config) {
    return operation::run(BertLargeMatmul{BertLargeMatmulOpType::PRE_SOFTMAX_BMM, mem_config, false}, {input_tensor_a, input_tensor_b}, {std::nullopt}).at(0);
}
inline Tensor bert_large_post_softmax_bmm(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config) {
    return operation::run(BertLargeMatmul{BertLargeMatmulOpType::POST_SOFTMAX_BMM, mem_config, false}, {input_tensor_a, input_tensor_b}, {std::nullopt}).at(0);
}


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

}  // namespace tt_metal

}  // namespace tt

namespace bmm_op_utils {
using namespace tt::tt_metal;

tuple<uint32_t, uint32_t, uint32_t, uint32_t> get_large_matmul_params(uint32_t Mt, uint32_t Nt, uint32_t num_cores_y, uint32_t num_cores_x, uint32_t in0_block_w);

CoreCoord get_core_range(uint32_t num_blocks_rows, uint32_t num_blocks_cols, uint32_t max_num_rows, uint32_t max_num_cols);

}
