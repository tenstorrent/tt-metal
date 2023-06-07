#include <optional>
#include "tt_dnn/op_library/bmm/bmm_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

using namespace tt::constants;

vector<uint32_t> _get_prime_factors(uint32_t n) {
    uint32_t i = 2;

    vector<uint32_t> prime_factors;
    while (i * i <= n) {
        if (n % i != 0) i++;
        else {
            n /= i;
            prime_factors.push_back(i);
        }
    }
    if (n > 1) prime_factors.push_back(n);

    return prime_factors;
}

vector<uint32_t> _get_possible_products(vector<uint32_t> factors) {
    if (factors.size() == 0) return {1};

    vector<uint32_t> products;
    for (uint32_t& fac : factors) {
        vector<uint32_t> new_products;
        if (not std::count(products.begin(), products.end(), fac))
            new_products.push_back(fac);
        for (uint32_t& prod : products) {
            if (not std::count(products.begin(), products.end(), fac * prod))
                new_products.push_back(fac * prod);
        }

        // Insert all new products to product
        products.reserve(products.size() + distance(new_products.begin(), new_products.end()));
        products.insert(products.end(), new_products.begin(), new_products.end());
    }

    // Sort products
    std::sort(products.begin(), products.end());

    return products;
}

uint32_t _get_maximum_block_dim(int32_t block_dim, int32_t in0_block_w) {
    int32_t other_dim = (400 - 2 * in0_block_w * block_dim) / (2 * in0_block_w + block_dim);
    if (other_dim > 0)
        return other_dim;
    return 0;
}

namespace bmm_op_utils {
using namespace tt::tt_metal;


tuple<uint32_t, uint32_t, uint32_t, uint32_t> get_large_matmul_params(uint32_t Mt, uint32_t Nt, uint32_t num_cores_y, uint32_t num_cores_x, uint32_t in0_block_w) {
    auto Nt_fac = _get_prime_factors(Nt);
    auto Mt_fac = _get_prime_factors(Mt);
    uint32_t Npc_min = 1;
    uint32_t Mpc_min = 1;

    for (auto it = Nt_fac.begin(); it != Nt_fac.end(); ++it) {
        auto ele = *it;
        if (ele > num_cores_x) {
            Npc_min *= ele;
            Nt_fac.erase(it);
            --it;
        }
    }
    for (auto it = Mt_fac.begin(); it != Mt_fac.end(); ++it) {
        auto ele = *it;
        if (ele > num_cores_y) {
            Mpc_min *= ele;
            Mt_fac.erase(it);
            --it;
        }
    }

    if (Npc_min > _get_maximum_block_dim(Mpc_min, in0_block_w))
        return {0, 0, 0, 0};

    uint32_t Mpc = Mpc_min;
    uint32_t Npc = Npc_min;
    vector<tuple<uint32_t, uint32_t>> SUBBLOCK_HW_CHOICES = {
        {4, 2}, {2, 4}, {8, 1}, {1, 8},
        {7, 1}, {1, 7},
        {3, 2}, {2, 3}, {6, 1}, {1, 6},
        {5, 1}, {1, 5},
        {2, 2}, {4, 1}, {1, 4},
        {3, 1}, {1, 3},
        {2, 1}, {1, 2},
        {1, 1},
    };
    if (Mpc_min > 1) {
        auto Npc_choices = _get_possible_products(Nt_fac);
        auto Npc_max = _get_maximum_block_dim(Mpc_min, in0_block_w);
        for (auto &ele : Npc_choices) {
            if (ele *  Npc_min <= Npc_max)
                Npc = ele * Npc_min;
            else
                break;
        }

        if (Mt / Mpc > num_cores_y or Nt / Npc > num_cores_x)
            return {0, 0, 0, 0};

        for (auto &subblock_hw : SUBBLOCK_HW_CHOICES) {
            auto subblock_h = std::get<0>(subblock_hw);
            auto subblock_w = std::get<1>(subblock_hw);
            if (Mpc % subblock_h == 0 and Npc % subblock_w == 0)
                return {Mpc, Npc, subblock_h, subblock_w};
        }
    }

    else if (Npc_min > 1) {
        auto Mpc_choices = _get_possible_products(Mt_fac);
        auto Mpc_max = _get_maximum_block_dim(Npc_min, in0_block_w);
        for (auto &ele : Mpc_choices) {
            if (ele *  Mpc_min <= Mpc_max)
                Mpc = ele * Mpc_min;
            else
                break;
        }

        if (Mt / Mpc > num_cores_y or Nt / Npc > num_cores_x) {
            return {0, 0, 0, 0};
        }

        for (auto &subblock_hw : SUBBLOCK_HW_CHOICES) {
            auto subblock_h = std::get<0>(subblock_hw);
            auto subblock_w = std::get<1>(subblock_hw);
            if (Mpc % subblock_h == 0 and Npc % subblock_w == 0)
                return {Mpc, Npc, subblock_h, subblock_w};
        }
    }

    else {
        auto Mpc_choices = _get_possible_products(Mt_fac);
        auto Npc_choices = _get_possible_products(Nt_fac);
        for (auto &Npc : Npc_choices) {
            auto Mpc_max = _get_maximum_block_dim(Npc, in0_block_w);
            for (auto &ele : Mpc_choices) {
                if (ele <= Mpc_max)
                    Mpc = ele;
            }

            if (Mt / Mpc > num_cores_y or Nt / Npc > num_cores_x)
                continue;

            for (auto &subblock_hw : SUBBLOCK_HW_CHOICES) {
                auto subblock_h = std::get<0>(subblock_hw);
                auto subblock_w = std::get<1>(subblock_hw);
                if (Mpc % subblock_h == 0 and Npc % subblock_w == 0)
                    return {Mpc, Npc, subblock_h, subblock_w};
            }
        }
    }

    return {0, 0, 0, 0};
}


CoreCoord get_core_range(uint32_t num_blocks_rows, uint32_t num_blocks_cols, uint32_t max_num_rows, uint32_t max_num_cols) {
    CoreCoord core_range(0, 0);
    if (!(num_blocks_rows == 1 && num_blocks_cols == 1) && num_blocks_rows <= max_num_rows && num_blocks_cols <= max_num_cols) {
        core_range.x = num_blocks_cols;
        core_range.y = num_blocks_rows;
    }
    return core_range;
}

BmmOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a, const Tensor &b){
    const auto& ashape = a.shape(), bshape = b.shape();
    uint32_t num_output_tiles = ashape[0] * ashape[1] * ashape[2] * bshape[3] / TILE_HW; // Output M x N

    // Parameters for large matmul with reuse
    uint32_t B = ashape[0] * ashape[1];
    uint32_t Mt = ashape[2]/TILE_HEIGHT;
    uint32_t Kt = ashape[3]/TILE_WIDTH;
    uint32_t Nt = bshape[3]/TILE_WIDTH;
    uint32_t in0_block_w = 2;

    tt::tt_metal::Device *device = a.device();
    auto compute_and_storage_grid_size = device->compute_and_storage_grid_size();
    uint32_t num_cores_x = compute_and_storage_grid_size.x;
    uint32_t num_cores_y = compute_and_storage_grid_size.y;

    bool use_general_large_matmul_params = false; // Hard force to use default 16, 16, 4, 2
    uint32_t per_core_M, per_core_N, out_subblock_h, out_subblock_w;
    uint32_t num_blocks_x, num_blocks_y;
    if (use_general_large_matmul_params) {
        // Get large matmul params
        auto matmul_params = bmm_op_utils::get_large_matmul_params(Mt, Nt, num_cores_y, num_cores_x, in0_block_w);
        per_core_M = std::get<0>(matmul_params);
        per_core_N = std::get<1>(matmul_params);
        out_subblock_h = std::get<2>(matmul_params);
        out_subblock_w = std::get<3>(matmul_params);
    }
    else {
        // out_subblock h/w doesn't matter
        per_core_M = 16;
        per_core_N = 16;

        // Calculate number of blocks along x and y; tensor dims are padded up to 512
        num_blocks_y = (Mt - 1) / per_core_M + 1;
        num_blocks_x = (Nt - 1) / per_core_N + 1;
    }

    // If no possible params, matmul_params will be (0, 0, 0, 0)
    if (use_general_large_matmul_params and per_core_M > 0 and Kt % in0_block_w == 0 and B == 1) {
        CoreCoord core_range = get_core_range((Mt / per_core_M), (Nt / per_core_N), num_cores_y, num_cores_x);
        // If matmul params are (16, 16, 4, 2), use the default mcast op
        if (
            per_core_M == 16 and
            per_core_N == 16 and
            out_subblock_h == 4 and
            out_subblock_w == 2
        ) {
            if (core_range.y > 0)
                return BmmOpParallelizationStrategy::MULTI_CORE_REUSE_MCAST;
            return BmmOpParallelizationStrategy::MULTI_CORE_REUSE;
        }
        else if (core_range.y > 0)
            return BmmOpParallelizationStrategy::MULTI_CORE_REUSE_MCAST_GENERALIZED;
        return BmmOpParallelizationStrategy::MULTI_CORE_REUSE_GENERALIZED;
    }
    else if (num_blocks_x * num_blocks_y <= num_cores_x * num_cores_y and Kt % in0_block_w == 0) {
        CoreCoord core_range = get_core_range(num_blocks_y, num_blocks_x, num_cores_y, num_cores_x);
        // If we don't need padding, use the default multi_core reuse/reuse_mcast
        if (Mt % per_core_M == 0 and Nt % per_core_N == 0) {
            if (core_range.y > 0)
                return BmmOpParallelizationStrategy::MULTI_CORE_REUSE_MCAST;
            return BmmOpParallelizationStrategy::MULTI_CORE_REUSE;
        }
        else if (core_range.y > 0)
            return BmmOpParallelizationStrategy::MULTI_CORE_REUSE_MCAST_PADDING;
        return BmmOpParallelizationStrategy::MULTI_CORE;
    }
    else if (num_output_tiles > 1) {
        return BmmOpParallelizationStrategy::MULTI_CORE;
    }else {
        return BmmOpParallelizationStrategy::SINGLE_CORE;
    }
}

}

namespace tt {

namespace tt_metal {

Tensor large_bmm(const Tensor& a, const Tensor& b, bool tilize_act, bool untilize_out) {
    if (bmm_op_utils::get_parallelization_strategy(a, b) != BmmOpParallelizationStrategy::SINGLE_CORE) {
        log_warning("WARNING: Only single core mode supported for large_bmm. Falling back to single core.");
    }
    return large_bmm_single_core(a, b, tilize_act, untilize_out);
}

/**
 * Blocked Matmul, with tilize a and untilize output.
 * NOTE: Takes blocks and subblock information as arguments.
 */
Tensor bmm_tilize_untilize(const Tensor& a, const Tensor& b,
                           uint32_t a_height_nblocks, uint32_t a_width_nblocks, uint32_t b_width_nblocks,
                           uint32_t a_block_height_ntiles, uint32_t a_block_width_ntiles, uint32_t b_block_width_ntiles,
                           uint32_t out_subblock_height_ntiles, uint32_t out_subblock_width_ntiles,
                           bool tilize_a, bool untilize_out) {
    return bmm_single_core_tilize_untilize(a, b,
                                           a_height_nblocks, a_width_nblocks, b_width_nblocks,
                                           a_block_height_ntiles, a_block_width_ntiles, b_block_width_ntiles,
                                           out_subblock_height_ntiles, out_subblock_width_ntiles,
                                           tilize_a, untilize_out);
}

Tensor large_bmm_single_block(const Tensor& a, const Tensor& b, bool tilize_a, bool untilize_out) {
    return large_bmm_single_core_single_block(a, b, tilize_a, untilize_out);
}

void Matmul::validate(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    const auto& input_tensor_b = input_tensors.at(1).get();
    TT_ASSERT(input_tensor_a.shape()[3] == input_tensor_b.shape()[2] && "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in bmm_op"); // A.K == B.K
    TT_ASSERT(input_tensor_b.shape()[0] * input_tensor_b.shape()[1] == 1 && "matmul (batch bcast variant) expects input tensors of shapes BCMK*11KN=BCMN");
}

std::vector<Shape> Matmul::compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    const auto& input_tensor_b = input_tensors.at(1).get();
    auto output_shape = input_tensor_a.shape();
    output_shape.back() = input_tensor_b.shape().back();
    return {output_shape};
}

std::vector<Tensor> Matmul::create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const {
    const auto output_shape = this->compute_output_shapes(input_tensors).at(0);
    const auto& input_tensor = input_tensors.at(0).get();
    std::vector<Tensor> output_tensors;
    output_tensors.emplace_back(tt_metal::Tensor(output_shape, input_tensor.dtype(), tt::tt_metal::Layout::TILE, input_tensor.device()));
    return output_tensors;
}

operation::ProgramWithCallbacks Matmul::create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    const auto& input_tensor_b = input_tensors.at(1).get();
    auto& output_tensor = output_tensors.at(0);

    auto parallelization_strategy = bmm_op_utils::get_parallelization_strategy(input_tensor_a, input_tensor_b);
    op_profiler::set_parallelization_strategy (parallelization_strategy);

    switch (parallelization_strategy){
        case BmmOpParallelizationStrategy::MULTI_CORE:
            return {matmul_multi_core(input_tensor_a, input_tensor_b, output_tensor)};
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE:
            return {matmul_multi_core_reuse(input_tensor_a, input_tensor_b, output_tensor)};
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE_MCAST:
            return {matmul_multi_core_reuse_mcast(input_tensor_a, input_tensor_b, output_tensor)};
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE_GENERALIZED:
            return {matmul_multi_core_reuse_generalized(input_tensor_a, input_tensor_b, output_tensor)};
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE_MCAST_GENERALIZED:
            return {matmul_multi_core_reuse_mcast_generalized(input_tensor_a, input_tensor_b, output_tensor)};
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE_PADDING:
            return {matmul_multi_core_reuse_padding(input_tensor_a, input_tensor_b, output_tensor)};
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE_MCAST_PADDING:
            return {matmul_multi_core_reuse_mcast_padding(input_tensor_a, input_tensor_b, output_tensor)};
            break;
        case BmmOpParallelizationStrategy::SINGLE_CORE:
        default:
            return {matmul_single_core(input_tensor_a, input_tensor_b, output_tensor)};
    }

}

void BatchedMatmul::validate(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    const auto& input_tensor_b = input_tensors.at(1).get();
    TT_ASSERT(input_tensor_a.shape()[3] == input_tensor_b.shape()[2] && "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in bmm_op"); // A.K == B.K
    TT_ASSERT(input_tensor_a.shape()[1] == input_tensor_b.shape()[1] && input_tensor_a.shape()[0] == input_tensor_b.shape()[0]
        && "bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN");
}

std::vector<Shape> BatchedMatmul::compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    const auto& input_tensor_b = input_tensors.at(1).get();
    auto output_shape = input_tensor_a.shape();
    output_shape.back() = input_tensor_b.shape().back();
    return {output_shape};
}

std::vector<Tensor> BatchedMatmul::create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const {
    const auto output_shape = this->compute_output_shapes(input_tensors).at(0);
    const auto& input_tensor = input_tensors.at(0).get();
    std::vector<Tensor> output_tensors;
    output_tensors.emplace_back(tt_metal::Tensor(output_shape, input_tensor.dtype(), tt::tt_metal::Layout::TILE, input_tensor.device()));
    return output_tensors;
}

operation::ProgramWithCallbacks BatchedMatmul::create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    const auto& input_tensor_b = input_tensors.at(1).get();
    auto& output_tensor = output_tensors.at(0);

    auto parallelization_strategy = bmm_op_utils::get_parallelization_strategy(input_tensor_a, input_tensor_b);
    op_profiler::set_parallelization_strategy (parallelization_strategy);

    switch (parallelization_strategy){
        case BmmOpParallelizationStrategy::MULTI_CORE:
            return {bmm_multi_core(input_tensor_a, input_tensor_b, output_tensor)};
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE:
            return {bmm_multi_core_reuse(input_tensor_a, input_tensor_b, output_tensor)};
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE_MCAST:
            return {bmm_multi_core_reuse_mcast(input_tensor_a, input_tensor_b, output_tensor)};
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE_GENERALIZED:
            return {bmm_multi_core_reuse_generalized(input_tensor_a, input_tensor_b, output_tensor)};
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE_MCAST_GENERALIZED:
            return {bmm_multi_core_reuse_mcast_generalized(input_tensor_a, input_tensor_b, output_tensor)};
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE_PADDING:
            return {bmm_multi_core_reuse_padding(input_tensor_a, input_tensor_b, output_tensor)};
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE_MCAST_PADDING:
            return {bmm_multi_core_reuse_mcast_padding(input_tensor_a, input_tensor_b, output_tensor)};
            break;
        case BmmOpParallelizationStrategy::SINGLE_CORE:
        default:
            return {bmm_single_core(input_tensor_a, input_tensor_b, output_tensor)};
    }

}

/*
 * BERT LARGE MATMUL AND BMM
 */
void BertLargeMatmul::validate(
    const std::vector<std::reference_wrapper<const Tensor>>& input_tensors,
    const std::vector<std::optional<std::reference_wrapper<const Tensor>>>& optional_input_tensors
) const {

    TT_ASSERT(input_tensors.size() == 2);
    const auto& input_tensor_a = input_tensors.at(0).get();
    const auto& input_tensor_b = input_tensors.at(1).get();

    TT_ASSERT(input_tensor_a.layout() == Layout::TILE, "Unsupported input layout");
    TT_ASSERT(input_tensor_b.layout() == Layout::TILE, "Unsupported input layout");

    switch (this->bert_large_matmul_op_type) {
        case BertLargeMatmulOpType::FUSED_QKV:
            TT_ASSERT((input_tensor_a.shape() == std::array<uint32_t, 4>({9, 1, 384, 1024})), "Unsupported input shape");
            TT_ASSERT((input_tensor_b.shape() == std::array<uint32_t, 4>({1, 1, 1024, 3072})), "Unsupported input shape");
            break;
        case BertLargeMatmulOpType::FF1:
            TT_ASSERT((input_tensor_a.dtype() != DataType::BFLOAT16) or (this->output_mem_config.buffer_type == BufferType::DRAM) or (input_tensor_a.buffer_type() == BufferType::DRAM and input_tensor_b.buffer_type() == BufferType::DRAM), "For BFLOAT16, if output is on L1, one of in0 or in1 must be on DRAM!");
            TT_ASSERT((input_tensor_a.shape() == std::array<uint32_t, 4>({9, 1, 384, 1024})), "Unsupported input shape");
            TT_ASSERT((input_tensor_b.shape() == std::array<uint32_t, 4>({1, 1, 1024, 4096})), "Unsupported input shape");
            break;
        case BertLargeMatmulOpType::FF2:
            TT_ASSERT((input_tensor_a.shape() == std::array<uint32_t, 4>({9, 1, 384, 4096})), "Unsupported input shape");
            TT_ASSERT((input_tensor_b.shape() == std::array<uint32_t, 4>({1, 1, 4096, 1024})), "Unsupported input shape");
            break;
        case BertLargeMatmulOpType::SELFOUT:
            TT_ASSERT((input_tensor_a.shape() == std::array<uint32_t, 4>({9, 1, 384, 1024})), "Unsupported input shape");
            TT_ASSERT((input_tensor_b.shape() == std::array<uint32_t, 4>({1, 1, 1024, 1024})), "Unsupported input shape");
            break;
        case BertLargeMatmulOpType::PRE_SOFTMAX_BMM:
            TT_ASSERT((input_tensor_a.shape() == std::array<uint32_t, 4>({9, 16, 384, 64})), "Unsupported input shape");
            TT_ASSERT((input_tensor_b.shape() == std::array<uint32_t, 4>({9, 16, 64, 384})), "Unsupported input shape");
            break;
        case BertLargeMatmulOpType::POST_SOFTMAX_BMM:
            TT_ASSERT((input_tensor_a.shape() == std::array<uint32_t, 4>({9, 1, 16 * 384, 384})), "Unsupported input shape");
            TT_ASSERT((input_tensor_b.shape() == std::array<uint32_t, 4>({9, 16, 384, 64})), "Unsupported input shape");
            break;
        default:
            TT_ASSERT(false, "Unknown bert large matmul op in validate!");
    }
    TT_ASSERT(optional_input_tensors.size() == 1);
    const auto& optional_bias = optional_input_tensors.at(0);
    if (
        this->bert_large_matmul_op_type == BertLargeMatmulOpType::PRE_SOFTMAX_BMM ||
        this->bert_large_matmul_op_type == BertLargeMatmulOpType::POST_SOFTMAX_BMM
    ) {
        TT_ASSERT(!optional_bias.has_value(), "Specified matmul does not take bias");
    }
    else {
        if (optional_bias.has_value()) {
            const auto& bias = optional_bias.value().get();
            TT_ASSERT(bias.layout() == Layout::TILE, "Unsupported input layout");
            TT_ASSERT(bias.shape() == Shape({1, 1, TILE_HEIGHT, input_tensor_b.shape()[3]}), "Unsupported bias shape");
        }
    }
}

std::vector<Shape> BertLargeMatmul::compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const {
    Shape output_shape;
    const auto& input_tensor_a = input_tensors.at(0).get();
    const auto& input_tensor_b = input_tensors.at(1).get();
    auto ashape = input_tensor_a.shape();
    const auto& bshape = input_tensor_b.shape();
    switch (this->bert_large_matmul_op_type) {
        case BertLargeMatmulOpType::FUSED_QKV:
        case BertLargeMatmulOpType::FF1:
        case BertLargeMatmulOpType::FF2:
        case BertLargeMatmulOpType::SELFOUT:
            output_shape = ashape;
            output_shape.back() = bshape.back();
            break;
        case BertLargeMatmulOpType::PRE_SOFTMAX_BMM:
            output_shape = {ashape[0], 1, ashape[1] * ashape[2], bshape[3]};
            break;
        case BertLargeMatmulOpType::POST_SOFTMAX_BMM:
            ashape = {9, 16, 384, 384};
            output_shape = ashape;
            output_shape.back() = bshape.back();
            break;
        default:
            TT_ASSERT(false, "Unknown bert large matmul op in compute_output_shapes!");
    }
    return {output_shape};
}

std::vector<Tensor> BertLargeMatmul::create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const {
    return operation::generic_create_output_tensors(*this, input_tensors, Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks BertLargeMatmul::create_program(
    const std::vector<std::reference_wrapper<const Tensor>>& input_tensors,
    const std::vector<std::optional<std::reference_wrapper<const Tensor>>>& optional_input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    const auto& input_tensor_b = input_tensors.at(1).get();
    const auto bias = optional_input_tensors.empty() ? std::nullopt : optional_input_tensors.at(0);
    auto ashape = input_tensor_a.shape();
    const auto& bshape = input_tensor_b.shape();
    auto& output_tensor = output_tensors.at(0);

    Program program;
    auto device_compute_and_storage_grid_size = input_tensor_a.device()->compute_and_storage_grid_size();
    CoreCoord compute_and_storage_grid_size;
    tt::DataFormat output_cb_data_format = tt::DataFormat::Bfp8_b;
    MathFidelity math_fidelity = MathFidelity::LoFi;
    uint32_t in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N;
    bool fuse_batch = true;

    op_profiler::set_preferred_name(this->bert_large_matmul_op_type);

    switch (this->bert_large_matmul_op_type) {
        case BertLargeMatmulOpType::FUSED_QKV:
            compute_and_storage_grid_size = {12, 9};
            TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");
            in0_block_w = 4;
            out_subblock_h = 4;
            out_subblock_w = 2;
            per_core_M = 12;
            per_core_N = 8;
            program = matmul_multi_core_reuse_mcast_optimized_bert_large(input_tensor_a, input_tensor_b, bias, output_tensor, compute_and_storage_grid_size, output_cb_data_format, math_fidelity, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch);
            break;
        case BertLargeMatmulOpType::FF1:
            compute_and_storage_grid_size = {12, 9};
            TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");
            in0_block_w = 4;
            out_subblock_h = 6;
            out_subblock_w = 1;
            per_core_M = 12;
            per_core_N = 11;
            program = matmul_multi_core_reuse_mcast_optimized_bert_large(input_tensor_a, input_tensor_b, bias, output_tensor, compute_and_storage_grid_size, output_cb_data_format, math_fidelity, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch, this->fuse_gelu_activation);
            break;
        case BertLargeMatmulOpType::FF2:
            compute_and_storage_grid_size = {11, 9};
            TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");
            in0_block_w = 4;
            out_subblock_h = 6;
            out_subblock_w = 1;
            per_core_M = 12;
            per_core_N = 3;
            program = matmul_multi_core_reuse_mcast_optimized_bert_large(input_tensor_a, input_tensor_b, bias, output_tensor, compute_and_storage_grid_size, output_cb_data_format, math_fidelity, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch);
            break;
        case BertLargeMatmulOpType::SELFOUT:
            compute_and_storage_grid_size = {11, 9};
            TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");
            in0_block_w = 4;
            out_subblock_h = 6;
            out_subblock_w = 1;
            per_core_M = 12;
            per_core_N = 3;
            program = matmul_multi_core_reuse_mcast_optimized_bert_large(input_tensor_a, input_tensor_b, bias, output_tensor, compute_and_storage_grid_size, output_cb_data_format, math_fidelity, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch);
            break;
        case BertLargeMatmulOpType::PRE_SOFTMAX_BMM:
            compute_and_storage_grid_size = {12, 9};
            TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");
            in0_block_w = 1;
            out_subblock_h = 4;
            out_subblock_w = 2;
            per_core_M = 12;
            per_core_N = 12;
            program = bmm_multi_core_reuse_optimized_bert_large(input_tensor_a, input_tensor_b, ashape, bshape, output_tensor, compute_and_storage_grid_size, output_cb_data_format, math_fidelity, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch);
            break;
        case BertLargeMatmulOpType::POST_SOFTMAX_BMM:
            compute_and_storage_grid_size = {12, 9};
            TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");
            ashape = {9, 16, 384, 384};
            in0_block_w = 2;
            out_subblock_h = 4;
            out_subblock_w = 2;
            per_core_M = 12;
            per_core_N = 2;
            program = bmm_multi_core_reuse_optimized_bert_large(input_tensor_a, input_tensor_b, ashape, bshape, output_tensor, compute_and_storage_grid_size, output_cb_data_format, math_fidelity, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch);
            break;
        default:
            TT_ASSERT(false, "Unknown bert large matmul op in create_program!");
    }
    return {std::move(program)};
}

}  // namespace tt_metal

}  // namespace tt
