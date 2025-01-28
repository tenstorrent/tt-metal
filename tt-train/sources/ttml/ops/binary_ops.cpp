// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ops.hpp"

#include <core/compute_kernel_config.hpp>
#include <core/ttnn_all_includes.hpp>
#include <memory>
#include <ranges>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/binary_backward/binary_backward.hpp>
#include <ttnn/tensor/types.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

namespace {

struct BroadcastResult {
    ttnn::Tensor fst;
    ttnn::Tensor snd;
    std::optional<std::vector<uint32_t>> fst_stretch_factors;
    std::optional<std::vector<uint32_t>> snd_stretch_factors;
};

BroadcastResult broadcast(const ttnn::Tensor& fst, const ttnn::Tensor& snd) {
    auto fst_shape = fst.get_logical_shape();
    auto snd_shape = snd.get_logical_shape();
    if (fst_shape == snd_shape) {
        return BroadcastResult{
            .fst = fst,
            .snd = snd,
            .fst_stretch_factors = std::nullopt,
            .snd_stretch_factors = std::nullopt};  // no stretch required
    }

    auto fst_broadcasted = fst;
    auto snd_broadcasted = snd;
    if (fst_shape.rank() < snd_shape.rank()) {
        fst_broadcasted = core::unsqueeze_to_rank(fst, snd_shape.rank());
        fst_shape = fst_broadcasted.get_logical_shape();
    }
    if (snd_shape.rank() < fst_shape.rank()) {
        snd_broadcasted = core::unsqueeze_to_rank(snd, fst_shape.rank());
        snd_shape = snd_broadcasted.get_logical_shape();
    }

    bool fst_is_stretched = false;
    bool snd_is_stretched = false;

    std::vector<uint32_t> fst_stretch_factors;
    std::vector<uint32_t> snd_stretch_factors;
    for (size_t i = 0; i < fst_shape.size(); ++i) {
        if (fst_shape[i] == snd_shape[i]) {
            fst_stretch_factors.push_back(1);
            snd_stretch_factors.push_back(1);
        } else if (fst_shape[i] == 1) {
            fst_stretch_factors.push_back(snd_shape[i]);
            fst_is_stretched = true;
        } else if (snd_shape[i] == 1) {
            snd_stretch_factors.push_back(fst_shape[i]);
            snd_is_stretched = true;
        } else {
            TT_THROW("Incompatible shapes for broadcasting: {} and {}, at dim {}", fst_shape, snd_shape, i);
        }
    }

    if (fst_is_stretched) {
        fst_broadcasted = ttnn::repeat(fst_broadcasted, ttnn::Shape{fst_stretch_factors});
    }
    if (snd_is_stretched) {
        snd_broadcasted = ttnn::repeat(snd_broadcasted, ttnn::Shape{snd_stretch_factors});
    }
    return BroadcastResult{
        .fst = fst_broadcasted,
        .snd = snd_broadcasted,
        .fst_stretch_factors = fst_is_stretched ? std::optional{fst_stretch_factors} : std::nullopt,
        .snd_stretch_factors = snd_is_stretched ? std::optional{snd_stretch_factors} : std::nullopt};
}

bool is_batch_broadcasted(const autograd::TensorPtr& a, const ttnn::Tensor& grad) {
    auto a_shape = a->get_value().get_logical_shape();
    auto b_shape = grad.get_logical_shape();
    if (a_shape.rank() != b_shape.rank()) {
        return false;
    }
    for (size_t i = 1; i < a_shape.size(); ++i) {
        if (a_shape[i] != b_shape[i]) {
            return false;
        }
    }

    if (a_shape[0] == 1 && b_shape[0] != 1) {
        return true;
    }

    return false;
}

}  // namespace

autograd::TensorPtr operator+(const autograd::TensorPtr& a, const autograd::AutocastTensor& b) {
    auto out = autograd::create_tensor(ttnn::add(a->get_value(), b.get_tensor()));
    autograd::GradFunction grad = [a, out]() { a->add_grad(out->get_grad()); };
    auto links = autograd::get_links(a);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr operator+(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    /* TT_ASSERT(a->get_value().shape().rank() == b->get_value().shape().rank(), "Ranks must match for addition"); */
    auto out = autograd::create_tensor();

    out->set_value(ttnn::add(a->get_value(), b->get_value()));
    autograd::GradFunction grad = [a, b, out]() {
        auto broadcast_result_a = broadcast(out->get_grad(), a->get_value());
        auto broadcast_result_b = broadcast(out->get_grad(), b->get_value());

        auto a_stretch_factors = broadcast_result_a.fst_stretch_factors.value_or(std::vector<uint32_t>{});
        auto b_stretch_factors = broadcast_result_b.snd_stretch_factors.value_or(std::vector<uint32_t>{});

        ttnn::SmallVector<int64_t> a_sum_dims_vec;
        ttnn::SmallVector<int64_t> b_sum_dims_vec;
        a_sum_dims_vec.reserve(a_stretch_factors.size());
        b_sum_dims_vec.reserve(b_stretch_factors.size());

        uint32_t i = 0;
        for (const auto& dim : a_stretch_factors) {
            if (dim != 1) {
                a_sum_dims_vec.push_back(i);
            }
            ++i;
        }

        i = 0;
        for (const auto& dim : b_stretch_factors) {
            if (dim != 1) {
                b_sum_dims_vec.push_back(i);
            }
            ++i;
        }

        fmt::print("a_stretch_factors: {}\n", a_stretch_factors);
        fmt::print("b_stretch_factors: {}\n", b_stretch_factors);
        fmt::print("a_sum_dims_vec: {}\n", a_sum_dims_vec);
        fmt::print("b_sum_dims_vec: {}\n", b_sum_dims_vec);
        fmt::print("broadcast_result_a.fst: {}\n", broadcast_result_a.fst.get_logical_shape());
        fmt::print("broadcast_result_b.fst: {}\n", broadcast_result_b.fst.get_logical_shape());
        fmt::print("out->get_grad(): {}\n", out->get_grad().get_logical_shape());
        fmt::print("a->get_value(): {}\n", a->get_value().get_logical_shape());
        fmt::print("b->get_value(): {}\n", b->get_value().get_logical_shape());

        a->add_grad(ttnn_fixed::sum_moreh(broadcast_result_a.fst, a_sum_dims_vec, /* keep_dim */ true));
        b->add_grad(ttnn_fixed::sum_moreh(broadcast_result_b.fst, b_sum_dims_vec, /* keep_dim */ true));

        // if (is_batch_broadcasted(a, out->get_grad())) {
        //     a->add_grad(ttnn_fixed::sum_over_dim(out->get_grad(), /* axis */ 0));
        // } else {
        //     a->add_grad(out->get_grad());
        // }

        // if (is_batch_broadcasted(b, out->get_grad())) {
        //     b->add_grad(ttnn_fixed::sum_over_dim(out->get_grad(), /* axis */ 0));
        // } else {
        //     b->add_grad(out->get_grad());
        // }
    };
    auto links = autograd::get_links(a, b);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr operator-(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    auto out = autograd::create_tensor();

    out->set_value(ttnn::subtract(a->get_value(), b->get_value()));
    autograd::GradFunction grad = [a, b, out]() {
        tt::tt_metal::MemoryConfig mem_config;
        // TODO: support broadcasting
        a->add_grad(out->get_grad());
        b->add_grad(ttnn::neg(out->get_grad()));
    };
    auto links = autograd::get_links(a, b);

    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr operator*(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    auto out = autograd::create_tensor();

    out->set_value(ttnn::multiply(a->get_value(), b->get_value()));
    autograd::GradFunction grad = [a, b, out]() {
        tt::tt_metal::MemoryConfig mem_config;
        // TODO: support broadcasting (or not)
        auto a_grad = ttnn::multiply(out->get_grad(), b->get_value());
        auto b_grad = ttnn::multiply(out->get_grad(), a->get_value());

        auto clamp_to_rank = [](const ttnn::Tensor& tensor, size_t rank) {
            auto tensor_rank = tensor.shape().rank();
            if (tensor_rank == rank) {
                return tensor;
            } else if (tensor_rank > rank) {
                return ttml::core::squeeze_to_rank(tensor, rank);
            } else {
                return ttml::core::unsqueeze_to_rank(tensor, rank);
            }
        };

        auto logical_suffixes_match = [](const ttnn::Tensor& a, const ttnn::Tensor& b) {
            auto a_shape = a.get_logical_shape();
            auto b_shape = b.get_logical_shape();

            auto suffix_len = std::min(a_shape.size(), b_shape.size());
            for (auto i = -1; i >= -suffix_len; i--) {
                if (a_shape[i] != b_shape[i]) {
                    return false;
                }
            }
            return true;
        };

        if (a->get_value().shape().rank() != a_grad.shape().rank()) {
            if (logical_suffixes_match(a->get_value(), a_grad)) {
                a_grad = clamp_to_rank(a_grad, a->get_value().shape().rank());
            }
        }

        if (b->get_value().shape().rank() != b_grad.shape().rank()) {
            if (logical_suffixes_match(b->get_value(), b_grad)) {
                b_grad = clamp_to_rank(b_grad, b->get_value().shape().rank());
            }
        }

        a->add_grad(a_grad);
        b->add_grad(b_grad);
    };
    auto links = autograd::get_links(a, b);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr operator*(const autograd::TensorPtr& a, float b) {
    auto out = autograd::create_tensor(ttnn::multiply(a->get_value(), b));
    autograd::GradFunction grad = [a, b, out]() {
        auto a_grad = ttnn::multiply(out->get_grad(), b);

        a->add_grad(a_grad);
    };
    auto links = autograd::get_links(a);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr operator*(float a, const autograd::TensorPtr& b) {
    return b * a;
}

autograd::TensorPtr operator/(const autograd::TensorPtr& a, float b) {
    return a * (1.F / b);
}

autograd::TensorPtr operator/(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    auto out = autograd::create_tensor();

    out->set_value(ttnn::divide(a->get_value(), b->get_value()));
    autograd::GradFunction grad = [a, b, out]() {
        auto res = ttnn::div_bw(out->get_grad(), a->get_value(), b->get_value());
        a->add_grad(res[0].value());
        b->add_grad(res[1].value());
    };
    auto links = autograd::get_links(a, b);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr add(const autograd::TensorPtr& a, const autograd::AutocastTensor& b) {
    return a + b;
}

autograd::TensorPtr add(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    return a + b;
}

autograd::TensorPtr sub(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    return a - b;
}

autograd::TensorPtr mul(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    return a * b;
}

autograd::TensorPtr mul(const autograd::TensorPtr& a, float b) {
    return a * b;
}

autograd::TensorPtr mul(float a, const autograd::TensorPtr& b) {
    return b * a;
}

autograd::TensorPtr div(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    return a / b;
}

autograd::TensorPtr div(const autograd::TensorPtr& a, float b) {
    return a / b;
}

tt::tt_metal::Tensor ttnn_matmul(
    const tt::tt_metal::Tensor& a, const tt::tt_metal::Tensor& b, bool transpose_a, bool transpose_b) {
    return ttnn::matmul(
        a,
        b,
        transpose_a,
        transpose_b,
        /* memory_config */ std::nullopt,
        /* dtype */ std::nullopt,
        /* program_config */ std::nullopt,
        /* activation */ std::nullopt,
        /* compute_kernel_config */ core::ComputeKernelConfig::matmul(),
        /* core_grid */ std::nullopt,  // NOTE: I believe matmul will use the
                                       // core grid for the device it ends up
                                       // running on, but should confirm.
        /* output_tile */ std::nullopt);
}

autograd::TensorPtr matmul(
    const autograd::TensorPtr& a, const autograd::TensorPtr& b, bool transpose_a, bool transpose_b) {
    auto out = autograd::create_tensor();
    out->set_value(ttnn_matmul(a->get_value(), b->get_value(), transpose_a, transpose_b));

    autograd::GradFunction grad = [a, b, out, transpose_a, transpose_b]() {
        // For loss function L and matmul C = AB:
        // dL/dA = dL/dC * B^T
        // dL/dB = A^T * dL/dC

        // where L is the loss function
        auto grad_a = ttnn_matmul(
            out->get_grad(),
            b->get_value(),
            /* transpose_a */ transpose_a,
            /* transpose_b */ !transpose_b);
        auto grad_b = ttnn_matmul(
            a->get_value(),
            out->get_grad(),
            /* transpose_a */ !transpose_a,
            /* transpose_b */ transpose_b);

        a->add_grad(grad_a);
        b->add_grad(grad_b);
    };

    auto links = autograd::get_links(a, b);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops
