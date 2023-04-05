#include "tt_dnn/op_library/bmm/bmm_op.hpp"

#include "tt_metal/host_api.hpp"
#include "common/constants.hpp"

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


tt_xy_pair get_core_range(uint32_t num_blocks_rows, uint32_t num_blocks_cols, uint32_t max_num_rows, uint32_t max_num_cols) {
    tt_xy_pair core_range(0, 0);
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
    auto logical_grid_size = device->logical_grid_size();
    uint32_t num_cores_x = logical_grid_size.x;
    uint32_t num_cores_y = logical_grid_size.y;

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
        tt_xy_pair core_range = get_core_range((Mt / per_core_M), (Nt / per_core_N), num_cores_y, num_cores_x);
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
        tt_xy_pair core_range = get_core_range(num_blocks_y, num_blocks_x, num_cores_y, num_cores_x);
        // If we don't need padding, use the default multi_core reuse/reuse_mcast
        if (Mt % per_core_M == 0 and Nt % per_core_N == 0) {
            if (core_range.y > 0)
                return BmmOpParallelizationStrategy::MULTI_CORE_REUSE_MCAST;
            return BmmOpParallelizationStrategy::MULTI_CORE_REUSE;
        }
        else if (core_range.y > 0)
            return BmmOpParallelizationStrategy::MULTI_CORE;
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


Tensor matmul(const Tensor& a, const Tensor& b) {
    switch (bmm_op_utils::get_parallelization_strategy(a, b)){
        case BmmOpParallelizationStrategy::MULTI_CORE:
            return matmul_multi_core(a, b);
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE:
            return matmul_multi_core_reuse(a, b);
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE_MCAST:
            return matmul_multi_core_reuse_mcast(a, b);
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE_GENERALIZED:
            return matmul_multi_core_reuse_generalized(a, b);
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE_MCAST_GENERALIZED:
            return matmul_multi_core_reuse_mcast_generalized(a, b);
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE_PADDING:
            return matmul_multi_core_reuse_padding(a, b);
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE_MCAST_PADDING:
            return matmul_multi_core_reuse_mcast_padding(a, b);
            break;
        case BmmOpParallelizationStrategy::SINGLE_CORE:
        default:
            return matmul_single_core(a, b);
    }
}

Tensor bmm(const Tensor& a, const Tensor& b) {
    switch (bmm_op_utils::get_parallelization_strategy(a, b)){
        case BmmOpParallelizationStrategy::MULTI_CORE:
            return bmm_multi_core(a, b);
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE:
            return bmm_multi_core_reuse(a, b);
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE_MCAST:
            return bmm_multi_core_reuse_mcast(a, b);
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE_GENERALIZED:
            return bmm_multi_core_reuse_generalized(a, b);
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE_MCAST_GENERALIZED:
            return bmm_multi_core_reuse_mcast_generalized(a, b);
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE_PADDING:
            return bmm_multi_core_reuse_padding(a, b);
            break;
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE_MCAST_PADDING:
            return bmm_multi_core_reuse_mcast_padding(a, b);
            break;
        case BmmOpParallelizationStrategy::SINGLE_CORE:
        default:
            return bmm_single_core(a, b);
    }
}

Tensor large_bmm(const Tensor& a, const Tensor& b) {
    // TT_ASSERT(
    //     bmm_op_utils::get_parallelization_strategy(a, b) == BmmOpParallelizationStrategy::SINGLE_CORE,
    //     "Only single core large_bmm supported so far");
    if (bmm_op_utils::get_parallelization_strategy(a, b) != BmmOpParallelizationStrategy::SINGLE_CORE) {
        std::cout << "WARNING: Only single core mode supported for large_bmm. Falling back to single core." << std::endl;
    }
    return large_bmm_single_core(a, b);
}

Tensor large_bmm_single_block(const Tensor& a, const Tensor& b, bool tilize_a, bool untilize_out) {
    return large_bmm_single_core_single_block(a, b, tilize_a, untilize_out);
}

}  // namespace tt_metal

}  // namespace tt
