// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "generators.hpp"

#include <numeric>

#include "autograd/auto_context.hpp"
namespace ttml::datasets {
InMemoryFloatVecDataset make_regression(MakeRegressionParams params) {
    std::normal_distribution<float> dist(0.0, 1.0);

    std::vector<std::vector<float>> data(params.n_samples, std::vector<float>(params.n_features));
    std::vector<std::vector<float>> targets(
        params.n_samples, std::vector<float>(params.n_targets));  // Targets are vectors of size n_targets

    // Generate random coefficients for each target
    std::vector<std::vector<float>> coefficients(params.n_targets, std::vector<float>(params.n_features));
    // Generate biases for each target
    std::vector<float> biases(params.n_targets, 0.0F);

    auto generate_sample = [&](auto& sample_data) {
        std::ranges::generate(
            sample_data, [&]() { return dist(autograd::AutoContext::get_instance().get_generator()); });
    };

    auto compute_target = [&](const auto& sample_data, const auto& coeff) {
        return std::transform_reduce(
            sample_data.begin(), sample_data.end(), coeff.begin(), 0.0F, std::plus<>(), std::multiplies<>());
    };

    auto add_bias_and_noise = [&](float target, float bias) {
        if (params.bias) {
            target += bias;  // Add bias
        }
        target += params.noise * dist(autograd::AutoContext::get_instance().get_generator());  // Add noise
        return target;
    };

    generate_sample(biases);
    std::ranges::for_each(coefficients, [&](auto& target_coeffs) { generate_sample(target_coeffs); });

    for (size_t i = 0; i < params.n_samples; ++i) {
        generate_sample(data[i]);

        for (size_t j = 0; j < params.n_targets; ++j) {
            float target = compute_target(data[i], coefficients[j]);
            targets[i][j] = add_bias_and_noise(target, biases[j]);
        }
    }

    return {data, targets};
}
}  // namespace ttml::datasets
