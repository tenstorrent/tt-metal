// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/rope_op.hpp"

#include <numbers>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::ops {

void validate_rope_input_and_params(const autograd::TensorPtr& input, const RotaryEmbeddingParams& params) {
    if (input->get_rank() != 4U) {
        throw std::runtime_error(
            fmt::format("RoPE only supports rank-4 input tensors, but got rank {}.", input->get_rank()));
    }
    auto input_shape = input->get_shape();

    auto input_seq_len = input_shape[-2];
    auto input_head_dim = input_shape[-1];

    if (input_head_dim != params.head_dim) {
        throw std::runtime_error(fmt::format(
            "RoPE input tensor's head dimension ({}) must match the head dimension in the params ({})",
            input_head_dim,
            params.head_dim));
    }

    if (input_seq_len != params.sequence_length) {
        throw std::runtime_error(fmt::format(
            "RoPE input tensor's sequence length ({}) must match the sequence length in the params ({})",
            input_seq_len,
            params.sequence_length));
    }

    auto trans_mat_shape = params.trans_mat.logical_shape();
    auto trig_param_shapes = std::array{
        params.cos_cache.logical_shape(),
        params.sin_cache.logical_shape(),
        params.neg_cos_cache.logical_shape(),
        params.neg_sin_cache.logical_shape()};

    auto expected_trig_shape = ttnn::Shape{1U, 1U, input_seq_len, input_head_dim};
    if (!std::ranges::all_of(
            trig_param_shapes, [&expected_trig_shape](auto shape) { return shape == expected_trig_shape; })) {
        throw std::runtime_error(fmt::format(
            "All trigonometric rotary embedding parameters must have shape [1, 1, {}, {}], but got shapes: "
            "cos_cache: {}, sin_cache: {}, neg_cos_cache: {}, neg_sin_cache: {}",
            input_seq_len,
            input_head_dim,
            params.cos_cache.logical_shape(),
            params.sin_cache.logical_shape(),
            params.neg_cos_cache.logical_shape(),
            params.neg_sin_cache.logical_shape()));
    }

    auto expected_trans_mat_shape = ttnn::Shape{1U, 1U, ttnn::TILE_SIZE, ttnn::TILE_SIZE};
    if (trans_mat_shape != expected_trans_mat_shape) {
        throw std::runtime_error(fmt::format(
            "RoPE trans_mat must be of shape {}, but has shape {}", expected_trans_mat_shape, trans_mat_shape));
    }
}

template <typename E>
E apply_rope_scaling(const E& freqs, const RopeScalingParams& scaling_params) {
    // Typical values for low_freq_factor and high_freq_factor are 1 and 4, respectively.
    assert(scaling_params.low_freq_factor != scaling_params.high_freq_factor);
    assert(scaling_params.low_freq_factor < scaling_params.high_freq_factor);

    // These wavelengths are used as thresholds to determine whether to scale the frequency.
    // For example, if the low_freq_factor is 1, then every frequency after the
    auto low_freq_wavelength = scaling_params.original_context_length / scaling_params.low_freq_factor;
    auto high_freq_wavelength = scaling_params.original_context_length / scaling_params.high_freq_factor;
    if (low_freq_wavelength == high_freq_wavelength) {
        throw std::invalid_argument("RoPE scaling requires low and high frequency wavelengths to be different.");
    }

    auto wavelengths = 2 * std::numbers::pi / freqs;

    // for high frequencies, we're capturing short-range dependencies and needn't scale.
    auto high_freqs = freqs;
    // for low frequencies, we're capturing long-range dependencies and need to scale by the full scaling factor.
    auto low_freqs = freqs / scaling_params.scaling_factor;

    // for frequencies in between, we smoothly interpolate.
    auto smooths = (scaling_params.original_context_length / wavelengths - scaling_params.low_freq_factor) /
                   (scaling_params.high_freq_factor - scaling_params.low_freq_factor);
    auto mid_freqs = (1.0F - smooths) * freqs / scaling_params.scaling_factor + smooths * freqs;

    // if we're between the low and high freqs, use the smoothly interpolated mid-freqs,
    // otherwise use low_freqs for the low freq wavelengths and high_freqs otherwise.
    return xt::where(
        (wavelengths < low_freq_wavelength) && (wavelengths > high_freq_wavelength),
        mid_freqs,
        xt::where(wavelengths > low_freq_wavelength, low_freqs, high_freqs));
}

// trans_mat, sin_cache, cos_cache are all precomputed and stored somewhere in
// the module hierarchy and passed to the operation.
autograd::TensorPtr rope(const autograd::TensorPtr& input, const RotaryEmbeddingParams& params) {
    validate_rope_input_and_params(input, params);

    auto input_logical_shape = input->get_value().logical_shape();
    auto num_batch = input_logical_shape[0];
    auto num_heads = input_logical_shape[1];
    auto seq_len = input_logical_shape[2];
    auto head_dim = input_logical_shape[3];
    auto device = &autograd::ctx().get_device();

    auto squish_batch = [num_batch, num_heads, seq_len, head_dim](const ttnn::Tensor& input) {
        auto shape = input.logical_shape();
        auto seq_len = shape[2];
        auto head_dim = shape[3];
        auto unbatched_input = ttnn::reshape(input, ttnn::Shape{1U, num_batch * num_heads, seq_len, head_dim});
        return unbatched_input;
    };

    auto unsquish_batch = [num_batch, num_heads, seq_len, head_dim](const ttnn::Tensor& input) {
        auto unbatched_input = ttnn::reshape(input, ttnn::Shape{num_batch, num_heads, seq_len, head_dim});
        return unbatched_input;
    };

    auto out_tensor = ttnn::experimental::rotary_embedding_llama(
        squish_batch(input->get_value()),
        params.cos_cache,
        params.sin_cache,
        params.trans_mat,
        /*is_decode_mode=*/false,
        /*memory_config=*/std::nullopt,
        /*compute_kernel_config=*/core::ComputeKernelConfig::precise());
    auto batched_output = unsquish_batch(out_tensor);
    auto out = autograd::create_tensor(batched_output);

    // In the backward pass we rotate by -θ, so we need negated cos and sin
    // caches. Note: we can just reuse trans_mat here since the data movement
    // should be the same on the backward pass (we use the same trick to speed
    // up the matmul, and the matrix used is specified by the cos/sin caches.)
    autograd::GradFunction grad_fn = [squish_batch, unsquish_batch, input, params, out]() {
        auto dL_dout = out->get_grad();

        auto dL_dinput = ttnn::experimental::rotary_embedding_llama(
            squish_batch(dL_dout),
            params.neg_cos_cache,
            params.neg_sin_cache,
            params.trans_mat,
            /*is_decode_mode=*/false,
            /*memory_config=*/std::nullopt,
            /*compute_kernel_config=*/core::ComputeKernelConfig::precise());
        auto unsquished = unsquish_batch(dL_dinput);
        input->add_grad(unsquished);
    };

    auto links = autograd::get_links(input);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad_fn), links));

    return out;
}

std::pair<ttnn::Tensor, ttnn::Tensor> gen_freqs(
    uint32_t head_dim, uint32_t sequence_length, float theta, const RopeScalingParams& scaling_params) {
    // compute freqs: 1.0 / (theta ** (2 * (i-1) / head_dim)) for i in [1, head_dim/2]
    xt::xarray<uint32_t> expt_data = xt::arange(0, static_cast<int>(head_dim)) / 2;
    xt::xarray<float> expt_xt = xt::cast<float>(expt_data);

    expt_xt *= 2.0F / static_cast<float>(head_dim);
    xt::xarray<float> theta_pow = xt::pow(theta, expt_xt);

    auto freqs = xt::ones_like(theta_pow) / theta_pow;

    xt::xarray<float> seq_pos = xt::arange<float>(sequence_length);
    xt::xarray<float> seq_pos_repeated_to_head = xt::repeat(seq_pos, head_dim, seq_pos.dimension() - 1U);
    xt::xarray<float> scales = seq_pos_repeated_to_head.reshape({sequence_length, head_dim});

    xt::xarray<float> scaled_freqs = scales * freqs;

    if (scaling_params.scaling_factor != 0.0F) {
        scaled_freqs = apply_rope_scaling(scaled_freqs, scaling_params);
    }

    // take the scaled freqs mod 2π to satisfy ttnn inputs constraints for sin/cos
    auto pi = static_cast<float>(std::numbers::pi);
    scaled_freqs = xt::fmod(scaled_freqs, 2.0F * pi);
    scaled_freqs = scaled_freqs.reshape({1, 1, sequence_length, head_dim});

    xt::xarray<float> sin_freqs = xt::sin(scaled_freqs);
    xt::xarray<float> cos_freqs = xt::cos(scaled_freqs);

    auto* device = &autograd::ctx().get_device();
    return {core::from_xtensor(sin_freqs, device), core::from_xtensor(cos_freqs, device)};
}

ttnn::Tensor gen_trans_mat() {
    xt::xarray<float> trans_mat = xt::zeros<float>({1, 1, ttnn::TILE_SIZE, ttnn::TILE_SIZE});
    for (int i = 0; i < ttnn::TILE_SIZE; i += 2) {
        trans_mat(0, 0, i, i + 1) = 1.0F;
    }
    for (int j = 1; j < ttnn::TILE_SIZE; j += 2) {
        trans_mat(0, 0, j, j - 1) = -1.0F;
    }

    auto device = &autograd::ctx().get_device();
    return core::from_xtensor(trans_mat, device);
}

RotaryEmbeddingParams build_rope_params(
    uint32_t sequence_length, uint32_t head_dim, float theta, RopeScalingParams scaling_params) {
    if (head_dim % 32U != 0U) {
        throw std::invalid_argument(fmt::format("RoPE head_dim must be divisible by 32, but is {}", head_dim));
    }
    if (head_dim > 256U) {
        throw std::invalid_argument(
            fmt::format("RoPE head_dim must be less than or equal to 256, but is {}", head_dim));
    }
    if (head_dim == 0U) {
        throw std::invalid_argument("RoPE head_dim must be non-zero.");
    }
    auto [sin_freqs, cos_freqs] = gen_freqs(head_dim, sequence_length, theta, scaling_params);
    auto trans_mat = gen_trans_mat();
    return {
        .cos_cache = cos_freqs,
        .sin_cache = sin_freqs,
        .neg_cos_cache = cos_freqs,             // cos(θ) = cos(-θ): symmetry over x-axis
        .neg_sin_cache = ttnn::neg(sin_freqs),  // sin(-θ) = -sin(θ)
        .trans_mat = trans_mat,

        .sequence_length = sequence_length,
        .head_dim = head_dim,

        .rope_scaling_params = scaling_params,
    };
}

}  // namespace ttml::ops
