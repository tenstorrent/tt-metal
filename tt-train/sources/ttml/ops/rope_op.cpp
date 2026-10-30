// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/rope_op.hpp"

#include <numbers>
#include <unordered_map>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/embedding/embedding.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama.hpp"
#include "ttnn/types.hpp"

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

namespace {

ttnn::Tensor rotate_per_head(
    const ttnn::Tensor& x, const ttnn::Tensor& cos, const ttnn::Tensor& sin, const ttnn::Tensor& trans_mat) {
    const auto shape = x.logical_shape();
    const auto flat = ttnn::reshape(x, ttnn::Shape{1U, shape[0] * shape[1], shape[2], shape[3]});
    // after setting is_decode_mode to and converting all required tensors' memory layout to SHARDED, receiving DRAM OOM
    // errors
    const auto rotated = ttnn::experimental::rotary_embedding_llama(
        flat,
        cos,
        sin,
        trans_mat,
        /*is_decode_mode=*/false,
        /*memory_config=*/std::nullopt,
        /*compute_kernel_config=*/core::ComputeKernelConfig::precise());
    return ttnn::reshape(rotated, shape);
}

autograd::TensorPtr rope_impl(
    const autograd::TensorPtr& input,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& neg_sin,
    const ttnn::Tensor& trans_mat) {
    auto out = autograd::create_tensor(rotate_per_head(input->get_value(), cos, sin, trans_mat));

    // The backward pass rotates by -θ. cos(-θ) = cos θ, so it reuses cos and only sin is negated.
    // trans_mat is the same in both directions.
    autograd::GradFunction grad_fn = [input, out, cos, neg_sin, trans_mat]() {
        input->add_grad(rotate_per_head(out->get_grad(), cos, neg_sin, trans_mat));
    };
    out->set_node(autograd::add_backward_node(std::move(grad_fn), out, input));
    return out;
}

}  // namespace

// trans_mat, sin_cache, cos_cache are all precomputed and stored somewhere in
// the module hierarchy and passed to the operation.
autograd::TensorPtr rope(
    const autograd::TensorPtr& input, const RotaryEmbeddingParams& params, const uint32_t token_position) {
    validate_rope_input_and_params(input, params);

    auto input_logical_shape = input->get_value().logical_shape();
    auto seq_len = input_logical_shape[2];
    auto head_dim = input_logical_shape[3];

    // Slice cos/sin caches to the specific position if provided (for decode mode)
    ttnn::Tensor cos_cache_to_use = params.cos_cache;
    ttnn::Tensor sin_cache_to_use = params.sin_cache;
    ttnn::Tensor neg_sin_cache_to_use = params.neg_sin_cache;

    if (token_position > 0U) {
        auto pos = token_position;
        ttsl::SmallVector<uint32_t> start = {0, 0, pos, 0};
        ttsl::SmallVector<uint32_t> end = {1, 1, pos + seq_len, head_dim};
        ttsl::SmallVector<uint32_t> step = {1, 1, 1, 1};

        cos_cache_to_use = ttnn::slice(params.cos_cache, start, end, step);
        sin_cache_to_use = ttnn::slice(params.sin_cache, start, end, step);
        neg_sin_cache_to_use = ttnn::slice(params.neg_sin_cache, start, end, step);
    }

    return rope_impl(input, cos_cache_to_use, sin_cache_to_use, neg_sin_cache_to_use, params.trans_mat);
}

uint64_t RotaryEmbeddingPositions::s_version_counter = 0;

RotaryEmbeddingPositions::RotaryEmbeddingPositions(autograd::TensorPtr ids) :
    m_ids(std::move(ids)), m_version(++s_version_counter) {
}

void RotaryEmbeddingPositions::update(autograd::TensorPtr ids) {
    m_ids = std::move(ids);
    m_version = ++s_version_counter;
}

const autograd::TensorPtr& RotaryEmbeddingPositions::get_ids() const {
    return m_ids;
}

struct RopeRows {
    ttnn::Tensor cos;
    ttnn::Tensor sin;
    ttnn::Tensor neg_sin;
};

// cos/sin rows gathered for one rope_positions version, one entry per head count.
// Every layer of a forward passes the same rope_positions, so only the first call per head count gathers;
// a new rope_positions version (the next step, or update()) clears the entries.
struct RopeRowsCache {
    uint64_t rope_positions_version = 0;
    // Q and K need the same cos/sin values, but in tensors of different shapes. The kernel needs one cos/sin row
    // for each flattened (batch, head) row of its input, and Q and K have different head counts. The head count is
    // the only thing that changes the shape, and the positions are the only thing that changes the values. So
    // within one step, the head count is enough to tell the two cached tensors apart.
    std::unordered_map<uint32_t, RopeRows> rows_by_head_count;
};

namespace {

RopeRows gather_rows(const RotaryEmbeddingParams& params, const ttnn::Tensor& ids, uint32_t num_heads) {
    const uint32_t batch = ids.logical_shape()[0];
    const uint32_t seq_len = ids.logical_shape()[1];

    auto gather = [&](const ttnn::Tensor& table) {
        auto rows = ttnn::embedding(ids, table, std::nullopt, ttnn::Layout::TILE);
        rows = ttnn::reshape(rows, ttnn::Shape{batch, 1U, seq_len, params.head_dim});
        rows = ttnn::repeat(rows, ttnn::Shape{1U, num_heads, 1U, 1U});
        return ttnn::reshape(rows, ttnn::Shape{1U, batch * num_heads, seq_len, params.head_dim});
    };
    auto sin = gather(params.sin_table);
    return {gather(params.cos_table), sin, ttnn::neg(sin)};
}

}  // namespace

const RopeRows& get_rope_rows(
    const RotaryEmbeddingParams& params, const RotaryEmbeddingPositions& rope_positions, uint32_t num_heads) {
    const auto& ids = rope_positions.m_ids->get_value();
    auto& cache = *params.rows_cache;
    if (cache.rope_positions_version != rope_positions.m_version) {
        cache.rope_positions_version = rope_positions.m_version;
        cache.rows_by_head_count.clear();
    }
    auto it = cache.rows_by_head_count.find(num_heads);
    if (it == cache.rows_by_head_count.end()) {
        it = cache.rows_by_head_count.emplace(num_heads, gather_rows(params, ids, num_heads)).first;
    }
    return it->second;
}

autograd::TensorPtr rope(
    const autograd::TensorPtr& input,
    const RotaryEmbeddingParams& params,
    const RotaryEmbeddingPositions& rope_positions) {
    validate_rope_input_and_params(input, params);

    const auto input_shape = input->get_value().logical_shape();
    const auto& ids = rope_positions.get_ids()->get_value();
    const auto ids_shape = ids.logical_shape();
    TT_FATAL(
        ids_shape.rank() == 2U && ids_shape[0] == input_shape[0] && ids_shape[1] == input_shape[2],
        "RoPE positions must have shape [{}, {}], but got {}",
        input_shape[0],
        input_shape[2],
        ids_shape);
    TT_FATAL(
        ids.dtype() == ttnn::DataType::UINT32 && ids.layout() == ttnn::Layout::ROW_MAJOR,
        "RoPE positions must be UINT32 ROW_MAJOR, but got {} {}",
        ids.dtype(),
        ids.layout());
    TT_FATAL(
        !autograd::ctx().is_parallelism_context_initialized() ||
            autograd::ctx().get_parallelism_context().get_cp_size() <= 1,
        "RoPE positions are not supported with context parallelism");
    TT_FATAL(params.rows_cache, "RoPE positions require rope_params from build_rope_params");

    const auto& rows = get_rope_rows(params, rope_positions, input_shape[1]);
    return rope_impl(input, rows.cos, rows.sin, rows.neg_sin, params.trans_mat);
}

std::pair<ttnn::Tensor, ttnn::Tensor> gen_freqs(
    uint32_t head_dim,
    uint32_t sequence_length,
    float theta,
    const RopeScalingParams& p,
    const ttnn::distributed::TensorToMesh* mesh_mapper) {
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
    return {
        core::from_xtensor(sin_freqs, device, ttnn::Layout::TILE, mesh_mapper),
        core::from_xtensor(cos_freqs, device, ttnn::Layout::TILE, mesh_mapper)};
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

    // If ParallelismContext is initialized and CP is enabled, shard freqs across the CP axis.
    std::shared_ptr<ttnn::distributed::TensorToMesh> mapper;
    uint32_t local_seq_len = sequence_length;

    if (autograd::ctx().is_parallelism_context_initialized()) {
        auto& pctx = autograd::ctx().get_parallelism_context();
        std::optional<uint32_t> cp_axis = pctx.get_cp_axis();
        const uint32_t cp_size = pctx.get_cp_size();

        if (cp_axis.has_value() && cp_size > 1) {
            TT_FATAL(
                sequence_length % cp_size == 0,
                "sequence_length ({}) must be divisible by cp_size ({})",
                sequence_length,
                cp_size);
            local_seq_len = sequence_length / cp_size;

            auto* device = &autograd::ctx().get_device();
            mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis.value());
        }
    }

    auto [sin_freqs, cos_freqs] = gen_freqs(head_dim, sequence_length, theta, scaling_params, mapper.get());
    auto trans_mat = gen_trans_mat();
    auto to_table = [local_seq_len, head_dim](const ttnn::Tensor& freqs) {
        return ttnn::to_layout(ttnn::reshape(freqs, ttnn::Shape{local_seq_len, head_dim}), ttnn::Layout::ROW_MAJOR);
    };
    return {
        .cos_cache = cos_freqs,
        .sin_cache = sin_freqs,
        .neg_cos_cache = cos_freqs,             // cos(θ) = cos(-θ): symmetry over x-axis
        .neg_sin_cache = ttnn::neg(sin_freqs),  // sin(-θ) = -sin(θ)
        .cos_table = to_table(cos_freqs),
        .sin_table = to_table(sin_freqs),
        .trans_mat = trans_mat,
        .rows_cache = std::make_shared<RopeRowsCache>(),

        .sequence_length = local_seq_len,
        .head_dim = head_dim,

        .rope_scaling_params = scaling_params,
    };
}

}  // namespace ttml::ops
