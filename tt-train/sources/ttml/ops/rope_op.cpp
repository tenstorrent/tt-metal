// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

    auto input_head_dim = input_shape[-1];

    if (input_head_dim != params.head_dim) {
        throw std::runtime_error(fmt::format(
            "RoPE input tensor's head dimension ({}) must match the head dimension in the params ({})",
            input_head_dim,
            params.head_dim));
    }

    auto trans_mat_shape = params.trans_mat.logical_shape();
    auto trig_param_shapes = std::array{
        params.cos_cache.logical_shape(),
        params.sin_cache.logical_shape(),
        params.neg_cos_cache.logical_shape(),
        params.neg_sin_cache.logical_shape()};

    auto expected_trig_shape = ttnn::Shape{1U, 1U, params.sequence_length, params.head_dim};
    if (!std::ranges::all_of(
            trig_param_shapes, [&expected_trig_shape](auto shape) { return shape == expected_trig_shape; })) {
        throw std::runtime_error(fmt::format(
            "All trigonometric rotary embedding parameters must have shape [1, 1, {}, {}], but got shapes: "
            "cos_cache: {}, sin_cache: {}, neg_cos_cache: {}, neg_sin_cache: {}",
            params.sequence_length,
            params.head_dim,
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
E apply_rope_scaling(const E& freqs, const RopeScalingParams& p) {
    using T = typename std::decay_t<E>::value_type;

    // Typical: low=1, high=4, factor>=1
    assert(p.low_freq_factor != p.high_freq_factor);
    assert(p.low_freq_factor < p.high_freq_factor);
    assert(p.scaling_factor > T(0));

    // Period (wavelength) in tokens for each channel: 2π / ω
    const E wavelengths = T(2) * T(M_PI) / freqs;

    // Threshold wavelengths (tokens). Cast to T to avoid integer division.
    const T low_wl = T(p.original_context_length) / T(p.low_freq_factor);    // e.g., N / 1 = N
    const T high_wl = T(p.original_context_length) / T(p.high_freq_factor);  // e.g., N / 4

    if (low_wl == high_wl) {
        throw std::invalid_argument("RoPE scaling requires distinct low/high wavelengths.");
    }

    // Unscaled (high-frequency / short-range) and fully-scaled (low-frequency / long-range)
    const E high_freqs = freqs;
    const E low_freqs = freqs / T(p.scaling_factor);

    // Mid-band linear blend between fully-scaled and unscaled.
    // Using cycles across the original context: c = original_ctx / period
    // period == wavelengths, so c = N / wavelength.
    E cycles = T(p.original_context_length) / wavelengths;  // dimensionless
    E smooths = (cycles - T(p.low_freq_factor)) / (T(p.high_freq_factor) - T(p.low_freq_factor));

    // (Optional) clamp for numerical safety
    smooths = xt::clip(smooths, T(0), T(1));

    const E mid_freqs = (T(1) - smooths) * (freqs / T(p.scaling_factor)) + smooths * freqs;

    // Masks
    const auto in_mid = (wavelengths < low_wl) & (wavelengths > high_wl);
    const auto is_low = (wavelengths > low_wl);  // very long periods → fully scaled
    // else: high band (wavelengths <= high_wl) → unscaled

    return xt::where(in_mid, mid_freqs, xt::where(is_low, low_freqs, high_freqs));
}

// trans_mat, sin_cache, cos_cache are all precomputed and stored somewhere in
// the module hierarchy and passed to the operation.
autograd::TensorPtr rope(
    const autograd::TensorPtr& input, const RotaryEmbeddingParams& params, const uint32_t token_position) {
    validate_rope_input_and_params(input, params);

    auto input_logical_shape = input->get_value().logical_shape();
    auto num_batch = input_logical_shape[0];
    auto num_heads = input_logical_shape[1];
    auto seq_len = input_logical_shape[2];
    auto head_dim = input_logical_shape[3];

    auto squish_batch = [num_batch, num_heads](const ttnn::Tensor& input) {
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

    // Slice cos/sin caches to the specific position if provided (for decode mode)
    ttnn::Tensor cos_cache_to_use = params.cos_cache;
    ttnn::Tensor sin_cache_to_use = params.sin_cache;
    ttnn::Tensor neg_cos_cache_to_use = params.neg_cos_cache;
    ttnn::Tensor neg_sin_cache_to_use = params.neg_sin_cache;

    if (token_position > 0U) {
        auto pos = token_position;
        ttnn::SmallVector<uint32_t> start = {0, 0, pos, 0};
        ttnn::SmallVector<uint32_t> end = {1, 1, pos + seq_len, head_dim};
        ttnn::SmallVector<uint32_t> step = {1, 1, 1, 1};

        cos_cache_to_use = ttnn::slice(params.cos_cache, start, end, step);
        sin_cache_to_use = ttnn::slice(params.sin_cache, start, end, step);
        neg_cos_cache_to_use = ttnn::slice(params.neg_cos_cache, start, end, step);
        neg_sin_cache_to_use = ttnn::slice(params.neg_sin_cache, start, end, step);
    }

    // after setting is_decode_mode to and converting all required tensors' memory layout to SHARDED, receiving DRAM OOM
    // errors
    auto out_tensor = ttnn::experimental::rotary_embedding_llama(
        squish_batch(input->get_value()),
        cos_cache_to_use,
        sin_cache_to_use,
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
    autograd::GradFunction grad_fn =
        [squish_batch, unsquish_batch, input, params, out, neg_cos_cache_to_use, neg_sin_cache_to_use]() {
            auto dL_dout = out->get_grad();

            auto dL_dinput = ttnn::experimental::rotary_embedding_llama(
                squish_batch(dL_dout),
                neg_cos_cache_to_use,
                neg_sin_cache_to_use,
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
    uint32_t head_dim, uint32_t sequence_length, float theta, const RopeScalingParams& p) {
    // pair indices: 0,0,1,1,2,2,... size=head_dim
    xt::xarray<uint32_t> pair_idx_u = xt::arange<uint32_t>(head_dim) / 2u;  // integer divide
    xt::xarray<float> pair_idx = xt::cast<float>(pair_idx_u);

    // exponent = 2 * floor(i/2) / head_dim
    pair_idx *= 2.0F / static_cast<float>(head_dim);

    // inv_freq[i] = 1 / (theta ** exponent[i])
    xt::xarray<float> inv_freq = 1.0f / xt::pow(theta, pair_idx);  // [D]

    // Apply NTK scaling to base frequencies (recommended)
    if (p.scaling_factor != 1.0f) {
        inv_freq = apply_rope_scaling(inv_freq, p);  // [D]
    }

    // positions column vector [L,1]
    // positions column vector [L,1]
    xt::xarray<float> pos = xt::arange<float>(static_cast<float>(sequence_length));
    pos.reshape({sequence_length, 1u});  // member reshape

    // θ = pos * inv_freq  -> broadcast to [L,D]
    xt::xarray<float> theta_mat = pos * inv_freq;  // [L,D]

    // keep in principal range
    theta_mat = xt::fmod(theta_mat, 2.0F * std::numbers::pi_v<float>);

    // expand to [1,1,L,D]
    xt::xarray<float> theta_4d = theta_mat;                 // copy shape metadata from theta_mat
    theta_4d.reshape({1u, 1u, sequence_length, head_dim});  // member reshape

    // trig caches
    xt::xarray<float> sin_freqs = xt::sin(theta_4d);
    xt::xarray<float> cos_freqs = xt::cos(theta_4d);

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
