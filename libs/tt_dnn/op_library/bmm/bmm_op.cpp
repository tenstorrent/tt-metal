#include "tt_dnn/op_library/bmm/bmm_op.hpp"

#include "tt_metal/host_api.hpp"
#include "common/constants.hpp"

#include "tt_dnn/op_library/auto_pad.hpp"

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


Tensor matmul_(const Tensor& a, const Tensor& b) {
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

Tensor matmul(const Tensor& a, const Tensor& b) {

    Device * device;

    if (a.on_host() && b.on_host()) {
        device = AutoPad::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else if (!a.on_host()){
        device = a.device();
    } else {
        device = b.device();
    }

    TT_ASSERT(a.shape()[3] == b.shape()[2] && "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in bmm_op"); // A.K == B.K
    TT_ASSERT(b.shape()[0]*b.shape()[1] == 1 && "matmul (batch bcast variant) expects input tensors of shapes BCMK*11KN=BCMN");

    auto a_pad_shape = AutoPad::pad_to_tile_shape(a.shape());
    auto b_pad_shape = AutoPad::pad_to_tile_shape(b.shape());
    auto out_shape = a.shape();
    out_shape[3] = b.shape()[3];
    auto no_pad_a = AutoPad::check_input_tensor_format(a, a_pad_shape);
    auto no_pad_b = AutoPad::check_input_tensor_format(b, b_pad_shape);
    if (no_pad_a && no_pad_b) {
        return matmul_(a, b);
    } else if (no_pad_a) {
        auto output = matmul_(a, AutoPad::format_input_tensor(b, device, b_pad_shape, 0));
        AutoPad::format_output_tensor(a, output, out_shape, device);
        return output;
    } else if (no_pad_b) {
        auto output = matmul_(AutoPad::format_input_tensor(a, device, a_pad_shape, 0), b);
        AutoPad::format_output_tensor(a, output, out_shape, device);
        return output;
    } else {
        auto output = matmul_(AutoPad::format_input_tensor(a, device, a_pad_shape, 0), AutoPad::format_input_tensor(b, device, b_pad_shape, 0));
        AutoPad::format_output_tensor(a, output, out_shape, device);
        return output;
    }
}

Tensor bmm_(const Tensor& a, const Tensor& b) {
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

Tensor bmm(const Tensor& a, const Tensor& b) {

    Device * device;

    if (a.on_host() && b.on_host()) {
        device = AutoPad::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else if (!a.on_host()){
        device = a.device();
    } else {
        device = b.device();
    }

    TT_ASSERT(a.shape()[3] == b.shape()[2] && "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in bmm_op"); // A.K == B.K
    TT_ASSERT(a.shape()[1] == b.shape()[1] && a.shape()[0] == b.shape()[0]
        && "bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN");

    auto a_pad_shape = AutoPad::pad_to_tile_shape(a.shape());
    auto b_pad_shape = AutoPad::pad_to_tile_shape(b.shape());
    auto out_shape = a.shape();
    out_shape[3] = b.shape()[3];

    auto no_pad_a = AutoPad::check_input_tensor_format(a, a_pad_shape);
    auto no_pad_b = AutoPad::check_input_tensor_format(b, b_pad_shape);
    if (no_pad_a && no_pad_b) {
        return bmm_(a, b);
    } else if (no_pad_a) {
        auto output = bmm_(a, AutoPad::format_input_tensor(b, device, b_pad_shape, 0));
        AutoPad::format_output_tensor(a, output, out_shape, device);
        return output;
    } else if (no_pad_b) {
        auto output = bmm_(AutoPad::format_input_tensor(a, device, a_pad_shape, 0), b);
        AutoPad::format_output_tensor(a, output, out_shape, device);
        return output;
    } else {
        auto output = bmm_(AutoPad::format_input_tensor(a, device, a_pad_shape, 0), AutoPad::format_input_tensor(b, device, b_pad_shape, 0));
        AutoPad::format_output_tensor(a, output, out_shape, device);
        return output;
    }
}

Tensor large_bmm(const Tensor& a, const Tensor& b, bool tilize_act, bool untilize_out) {
    // TT_ASSERT(
    //     bmm_op_utils::get_parallelization_strategy(a, b) == BmmOpParallelizationStrategy::SINGLE_CORE,
    //     "Only single core large_bmm supported so far");
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
                           uint32_t out_subblock_height_ntiles, uint32_t out_subblock_width_ntiles) {
    // TT_ASSERT(bmm_op_utils::get_parallelization_strategy(a, b) == BmmOpParallelizationStrategy::SINGLE_CORE,
    //             "Only single core mode supported for bmm_tilize_untilize.");
    return bmm_single_core_tilize_untilize(a, b,
                                           a_height_nblocks, a_width_nblocks, b_width_nblocks,
                                           a_block_height_ntiles, a_block_width_ntiles, b_block_width_ntiles,
                                           out_subblock_height_ntiles, out_subblock_width_ntiles);
}

Tensor large_bmm_single_block(const Tensor& a, const Tensor& b, bool tilize_a, bool untilize_out) {
    return large_bmm_single_core_single_block(a, b, tilize_a, untilize_out);
}

Tensor bert_large_fused_qkv_matmul(const Tensor& a, const Tensor& b, const MemoryConfig& mem_config) {
    TT_ASSERT((a.shape() == std::array<uint32_t, 4>({9, 1, 384, 1024})), "Unsupported input shape");
    TT_ASSERT((b.shape() == std::array<uint32_t, 4>({1, 1, 1024, 3072})), "Unsupported input shape");
    CoreCoord compute_and_storage_grid_size = {12, 9};
    auto device_compute_and_storage_grid_size = a.device()->compute_and_storage_grid_size();
    TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");
    tt::DataFormat output_cb_data_format = tt::DataFormat::Bfp8_b;
    MathFidelity math_fidelity = MathFidelity::LoFi;
    uint32_t in0_block_w = 4;
    uint32_t out_subblock_h = 4;
    uint32_t out_subblock_w = 2;
    uint32_t per_core_M = 12;
    uint32_t per_core_N = 8;
    bool fuse_batch = true;
    Tensor output = matmul_multi_core_reuse_mcast_optimized_bert_large(a, b, mem_config, compute_and_storage_grid_size, output_cb_data_format, math_fidelity, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch);
    return output;
    // Old matmul:
    // return matmul_multi_core_reuse_mcast_padding_generalized(a, b, compute_and_storage_grid_size, output_cb_data_format, math_fidelity, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch);
}

Tensor bert_large_ff1_matmul(const Tensor& a, const Tensor& b, const MemoryConfig& mem_config) {
    TT_ASSERT((a.dtype() != DataType::BFLOAT16) or (mem_config.buffer_type == BufferType::DRAM) or (a.buffer_type() == BufferType::DRAM and b.buffer_type() == BufferType::DRAM), "For BFLOAT16, if output is on L1, one of in0 or in1 must be on DRAM!");
    TT_ASSERT((a.shape() == std::array<uint32_t, 4>({9, 1, 384, 1024})), "Unsupported input shape");
    TT_ASSERT((b.shape() == std::array<uint32_t, 4>({1, 1, 1024, 4096})), "Unsupported input shape");
    CoreCoord compute_and_storage_grid_size = {12, 9};
    auto device_compute_and_storage_grid_size = a.device()->compute_and_storage_grid_size();
    TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");
    tt::DataFormat output_cb_data_format = tt::DataFormat::Bfp8_b;
    MathFidelity math_fidelity = MathFidelity::LoFi;
    uint32_t in0_block_w = 4;
    uint32_t out_subblock_h = 6;
    uint32_t out_subblock_w = 1;
    uint32_t per_core_M = 12;
    uint32_t per_core_N = 11;
    bool fuse_batch = true;
    Tensor output = matmul_multi_core_reuse_mcast_optimized_bert_large(a, b, mem_config, compute_and_storage_grid_size, output_cb_data_format, math_fidelity, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch);
    return output;
    // Old matmul:
    // return matmul_multi_core_reuse_mcast_padding_generalized(a, b, compute_and_storage_grid_size, output_cb_data_format, math_fidelity, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch);
}

Tensor bert_large_ff2_matmul(const Tensor& a, const Tensor& b, const MemoryConfig& mem_config) {
    TT_ASSERT((a.shape() == std::array<uint32_t, 4>({9, 1, 384, 4096})), "Unsupported input shape");
    TT_ASSERT((b.shape() == std::array<uint32_t, 4>({1, 1, 4096, 1024})), "Unsupported input shape");
    CoreCoord compute_and_storage_grid_size = {11, 9};
    auto device_compute_and_storage_grid_size = a.device()->compute_and_storage_grid_size();
    TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");
    tt::DataFormat output_cb_data_format = tt::DataFormat::Bfp8_b;
    MathFidelity math_fidelity = MathFidelity::LoFi;
    uint32_t in0_block_w = 4;
    uint32_t out_subblock_h = 6;
    uint32_t out_subblock_w = 1;
    uint32_t per_core_M = 12;
    uint32_t per_core_N = 3;
    bool fuse_batch = true;
    Tensor output = matmul_multi_core_reuse_mcast_optimized_bert_large(a, b, mem_config, compute_and_storage_grid_size, output_cb_data_format, math_fidelity, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch);
    return output;
    // Old matmul:
    // return matmul_multi_core_reuse_mcast_padding_generalized(a, b, compute_and_storage_grid_size, output_cb_data_format, math_fidelity, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch);
}

Tensor bert_large_selfout_matmul(const Tensor& a, const Tensor& b, const MemoryConfig& mem_config) {
    TT_ASSERT((a.shape() == std::array<uint32_t, 4>({9, 1, 384, 1024})), "Unsupported input shape");
    TT_ASSERT((b.shape() == std::array<uint32_t, 4>({1, 1, 1024, 1024})), "Unsupported input shape");
    CoreCoord compute_and_storage_grid_size = {11, 9};
    auto device_compute_and_storage_grid_size = a.device()->compute_and_storage_grid_size();
    TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");
    tt::DataFormat output_cb_data_format = tt::DataFormat::Bfp8_b;
    MathFidelity math_fidelity = MathFidelity::LoFi;
    uint32_t in0_block_w = 4;
    uint32_t out_subblock_h = 6;
    uint32_t out_subblock_w = 1;
    uint32_t per_core_M = 12;
    uint32_t per_core_N = 3;
    bool fuse_batch = true;
    Tensor output = matmul_multi_core_reuse_mcast_optimized_bert_large(a, b, mem_config, compute_and_storage_grid_size, output_cb_data_format, math_fidelity, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch);
    return output;
    // Old matmul:
    // return matmul_multi_core_reuse_mcast_padding_generalized(a, b, compute_and_storage_grid_size, output_cb_data_format, math_fidelity, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch);
}

Tensor bert_large_pre_softmax_bmm(const Tensor& a, const Tensor& b, const MemoryConfig& mem_config) {
    TT_ASSERT((a.shape() == std::array<uint32_t, 4>({9, 16, 384, 64})), "Unsupported input shape");
    TT_ASSERT((b.shape() == std::array<uint32_t, 4>({9, 16, 64, 384})), "Unsupported input shape");
    CoreCoord compute_and_storage_grid_size = {12, 9};
    auto device_compute_and_storage_grid_size = a.device()->compute_and_storage_grid_size();
    TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");
    tt::DataFormat output_cb_data_format = tt::DataFormat::Bfp8_b;
    MathFidelity math_fidelity = MathFidelity::LoFi;
    uint32_t in0_block_w = 1;
    uint32_t out_subblock_h = 4;
    uint32_t out_subblock_w = 2;
    uint32_t per_core_M = 12;
    uint32_t per_core_N = 12;
    bool fuse_batch = true;
    Tensor output = bmm_multi_core_reuse_optimized_bert_large(a, b, mem_config, compute_and_storage_grid_size, output_cb_data_format, math_fidelity, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch);
    return output;
    // Old matmul:
    // return bmm_multi_core_reuse_generalized_bert_large(a, b, compute_and_storage_grid_size, output_cb_data_format, math_fidelity, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch);
}

Tensor bert_large_post_softmax_bmm(const Tensor& a, const Tensor& b, const MemoryConfig& mem_config) {
    TT_ASSERT((a.shape() == std::array<uint32_t, 4>({9, 16, 384, 384})), "Unsupported input shape");
    TT_ASSERT((b.shape() == std::array<uint32_t, 4>({9, 16, 384, 64})), "Unsupported input shape");
    CoreCoord compute_and_storage_grid_size = {12, 9};
    auto device_compute_and_storage_grid_size = a.device()->compute_and_storage_grid_size();
    TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");
    tt::DataFormat output_cb_data_format = tt::DataFormat::Bfp8_b;
    MathFidelity math_fidelity = MathFidelity::LoFi;
    uint32_t in0_block_w = 2;
    uint32_t out_subblock_h = 4;
    uint32_t out_subblock_w = 2;
    uint32_t per_core_M = 12;
    uint32_t per_core_N = 2;
    bool fuse_batch = true;
    Tensor output = bmm_multi_core_reuse_optimized_bert_large(a, b, mem_config, compute_and_storage_grid_size, output_cb_data_format, math_fidelity, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch);
    return output;
    // Old matmul:
    // return bmm_multi_core_reuse_generalized_bert_large(a, b, compute_and_storage_grid_size, output_cb_data_format, math_fidelity, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch);
}

}  // namespace tt_metal

}  // namespace tt
