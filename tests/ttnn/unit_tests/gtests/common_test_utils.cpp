// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common_test_utils.hpp"

#include <stdexcept>
#include <cmath>
#include <variant>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include <tt-metalium/host_api.hpp>

namespace ttnn::test_utils {

float pcc(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vectors must be of the same length.");
    }
    int n = x.size();
    float mean_x = 0, mean_y = 0;
    for (int i = 0; i < n; ++i) {
        mean_x += x[i];
        mean_y += y[i];
    }
    mean_x /= n;
    mean_y /= n;

    float numerator = 0, sum_sq_x = 0, sum_sq_y = 0;
    for (int i = 0; i < n; ++i) {
        float diff_x = x[i] - mean_x;
        float diff_y = y[i] - mean_y;
        numerator += diff_x * diff_y;
        sum_sq_x += diff_x * diff_x;
        sum_sq_y += diff_y * diff_y;
    }

    float denominator = std::sqrt(sum_sq_x * sum_sq_y);
    if (denominator == 0) {
        return 0;
    }

    return numerator / denominator;
}

Tensor dispatch_ops_to_device(Tensor input_tensor, QueueId cq_id) {
    using ttnn::operations::unary::UnaryOpType;
    using ttnn::operations::unary::UnaryWithParam;

    auto guard = ttnn::with_command_queue_id(cq_id);

    Tensor output_tensor = ttnn::mul_sfpu(input_tensor, 2);
    for (int i = 0; i < 3; i++) {
        output_tensor = ttnn::neg(output_tensor);
        output_tensor = ttnn::neg(output_tensor);
        output_tensor = ttnn::mul_sfpu(output_tensor, 2);
    }
    output_tensor = ttnn::neg(output_tensor);
    output_tensor = ttnn::mul_sfpu(output_tensor, 2);
    output_tensor = ttnn::add_sfpu(output_tensor, 128);

    return output_tensor;
}

}  // namespace ttnn::test_utils
