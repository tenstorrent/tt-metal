// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include <array>

#include <algorithm>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>

#include <umd/device/tt_xy_pair.h>

#include <tt-metalium/work_split.hpp>

using std::pair;
using CoreCoord = tt_xy_pair;

using namespace tt::constants;

std::vector<uint32_t> _get_prime_factors(uint32_t n) {
    uint32_t i = 2;

    std::vector<uint32_t> prime_factors;
    while (i * i <= n) {
        if (n % i != 0) {
            i++;
        } else {
            n /= i;
            prime_factors.push_back(i);
        }
    }
    if (n > 1) {
        prime_factors.push_back(n);
    }

    return prime_factors;
}

std::vector<uint32_t> _get_possible_products(std::vector<uint32_t> factors) {
    if (factors.size() == 0) {
        return {1};
    }

    std::vector<uint32_t> products;
    for (uint32_t& fac : factors) {
        std::vector<uint32_t> new_products;
        if (not std::count(products.begin(), products.end(), fac)) {
            new_products.push_back(fac);
        }
        for (uint32_t& prod : products) {
            if (not std::count(products.begin(), products.end(), fac * prod)) {
                new_products.push_back(fac * prod);
            }
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
    if (other_dim > 0) {
        return other_dim;
    }
    return 0;
}

inline float check_bfloat16_vector_pcc(const std::vector<bfloat16>& vec_a, const std::vector<bfloat16>& vec_b) {
    // Calculate the mean of x and y values
    float x_mean = 0.0f;
    float y_mean = 0.0f;

    for (size_t i = 0; i < vec_a.size(); i++) {
        x_mean += vec_a[i].to_float();
        y_mean += vec_b[i].to_float();
    }

    x_mean /= vec_a.size();
    y_mean /= vec_b.size();

    // Calculate the covariance and standard deviation of x and y values
    float covariance = 0.0f;
    float x_stddev = 0.0f;
    float y_stddev = 0.0f;

    for (size_t i = 0; i < vec_a.size(); i++) {
        float x_diff = vec_a[i].to_float() - x_mean;
        float y_diff = vec_b[i].to_float() - y_mean;

        covariance += x_diff * y_diff;
        x_stddev += x_diff * x_diff;
        y_stddev += y_diff * y_diff;
    }

    covariance /= vec_a.size();
    x_stddev /= vec_a.size();
    y_stddev /= vec_b.size();

    // Calculate the correlation coefficient
    float correlation_coefficient_ = covariance / (std::sqrt(x_stddev) * std::sqrt(y_stddev));
    return correlation_coefficient_;
}

namespace bmm_op_utils {

constexpr std::array<std::tuple<uint32_t, uint32_t>, 20> SUBBLOCK_HW_CHOICES = {{
    {4, 2}, {2, 4}, {8, 1}, {1, 8}, {7, 1}, {1, 7}, {3, 2}, {2, 3}, {6, 1}, {1, 6},
    {5, 1}, {1, 5}, {2, 2}, {4, 1}, {1, 4}, {3, 1}, {1, 3}, {2, 1}, {1, 2}, {1, 1},
}};

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> get_large_matmul_params(
    uint32_t Mt, uint32_t Nt, uint32_t num_cores_y, uint32_t num_cores_x, uint32_t in0_block_w) {
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

    if (Npc_min > _get_maximum_block_dim(Mpc_min, in0_block_w)) {
        return {0, 0, 0, 0};
    }

    uint32_t Mpc = Mpc_min;
    uint32_t Npc = Npc_min;
    if (Mpc_min > 1) {
        auto Npc_choices = _get_possible_products(Nt_fac);
        auto Npc_max = _get_maximum_block_dim(Mpc_min, in0_block_w);
        for (auto& ele : Npc_choices) {
            if (ele * Npc_min <= Npc_max) {
                Npc = ele * Npc_min;
            } else {
                break;
            }
        }

        if (Mt / Mpc > num_cores_y or Nt / Npc > num_cores_x) {
            return {0, 0, 0, 0};
        }

        for (auto& subblock_hw : SUBBLOCK_HW_CHOICES) {
            auto subblock_h = std::get<0>(subblock_hw);
            auto subblock_w = std::get<1>(subblock_hw);
            if (Mpc % subblock_h == 0 and Npc % subblock_w == 0) {
                return {Mpc, Npc, subblock_h, subblock_w};
            }
        }
    }

    else if (Npc_min > 1) {
        auto Mpc_choices = _get_possible_products(Mt_fac);
        auto Mpc_max = _get_maximum_block_dim(Npc_min, in0_block_w);
        for (auto& ele : Mpc_choices) {
            if (ele * Mpc_min <= Mpc_max) {
                Mpc = ele * Mpc_min;
            } else {
                break;
            }
        }

        if (Mt / Mpc > num_cores_y or Nt / Npc > num_cores_x) {
            return {0, 0, 0, 0};
        }

        for (auto& subblock_hw : SUBBLOCK_HW_CHOICES) {
            auto subblock_h = std::get<0>(subblock_hw);
            auto subblock_w = std::get<1>(subblock_hw);
            if (Mpc % subblock_h == 0 and Npc % subblock_w == 0) {
                return {Mpc, Npc, subblock_h, subblock_w};
            }
        }
    }

    else {
        auto Mpc_choices = _get_possible_products(Mt_fac);
        auto Npc_choices = _get_possible_products(Nt_fac);
        for (auto& Npc : Npc_choices) {
            auto Mpc_max = _get_maximum_block_dim(Npc, in0_block_w);
            for (auto& ele : Mpc_choices) {
                if (ele <= Mpc_max) {
                    Mpc = ele;
                }
            }

            if (Mt / Mpc > num_cores_y or Nt / Npc > num_cores_x) {
                continue;
            }

            for (auto& subblock_hw : SUBBLOCK_HW_CHOICES) {
                auto subblock_h = std::get<0>(subblock_hw);
                auto subblock_w = std::get<1>(subblock_hw);
                if (Mpc % subblock_h == 0 and Npc % subblock_w == 0) {
                    return {Mpc, Npc, subblock_h, subblock_w};
                }
            }
        }
    }

    return {0, 0, 0, 0};
}

CoreCoord get_core_range(
    uint32_t num_blocks_rows, uint32_t num_blocks_cols, uint32_t max_num_rows, uint32_t max_num_cols) {
    CoreCoord core_range(0, 0);
    if (!(num_blocks_rows == 1 && num_blocks_cols == 1) && num_blocks_rows <= max_num_rows &&
        num_blocks_cols <= max_num_cols) {
        core_range.x = num_blocks_cols;
        core_range.y = num_blocks_rows;
    }
    return core_range;
}

}  // namespace bmm_op_utils
