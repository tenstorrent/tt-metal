// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <memory>
#include <optional>

#include "autocast_tensor.hpp"
#include "graph.hpp"

namespace ttml::autograd {

class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    AutocastTensor m_value;
    ttnn::Tensor m_grad;
    bool m_requires_grad = false;
    std::optional<NodeId> m_node_id;
    bool m_accumulate_grad_in_fp32 = false;

public:
    Tensor() = default;
    Tensor(const Tensor &) = default;
    Tensor(Tensor &&) noexcept = default;
    Tensor &operator=(const Tensor &) = default;
    Tensor &operator=(Tensor &&) noexcept = default;
    explicit Tensor(const ttnn::Tensor &value, bool requires_grad = false);
    ~Tensor() = default;

    void set_value(const ttnn::Tensor &value);
    // Sets the value in the dtype this tensor is stored in, for loading values into it; see AutocastTensor::assign().
    void assign(const ttnn::Tensor &value);
    void set_grad(const ttnn::Tensor &grad);
    void set_node(const std::optional<NodeId> &node);
    void clean_node();
    // Stores the first gradient as it comes, and adds every later one into it: in fp32 when this tensor accumulates its
    // gradient in fp32 or the gradient is already fp32, in the gradient's dtype otherwise.
    void add_grad(const ttnn::Tensor &grad);
    void set_requires_grad(bool requires_grad);
    // Whether add_grad() sums this tensor's gradients in fp32. When set, the stored gradient and each incoming one are
    // cast to fp32 from the second add on, so micro-batches and a weight used twice sum in fp32 and the gradient stays
    // fp32 until something replaces it; a gradient that is never added to keeps the dtype it came in.
    // core::apply_precision_config() sets it on trainable parameters with reduce_dtype float32.
    void set_accumulate_grad_in_fp32(bool accumulate_in_fp32);
    [[nodiscard]] bool accumulates_grad_in_fp32() const;

    const ttnn::Tensor &get_value(PreferredPrecision preferred_precision = PreferredPrecision::HALF) const;
    // The only way to update the value in place; see AutocastTensor.
    [[nodiscard]] MutableTensorView get_value_for_update(PreferredPrecision precision = PreferredPrecision::NATIVE);
    const ttnn::Tensor &get_grad() const;
    ttnn::Tensor &get_grad();
    bool get_requires_grad() const;
    const std::optional<NodeId> &get_node() const;
    const ttnn::Shape &get_shape() const;
    uint32_t get_rank() const;

    void backward(bool retain_graph = false);

    bool is_grad_initialized() const;

private:
    void try_init_grad(bool init_ones = false);
};
using TensorPtr = std::shared_ptr<Tensor>;

// TODO: In future implement create tensor without variadic templates to help with code hints in IDE
template <typename... Args>
TensorPtr create_tensor(Args &&...args) {
    return std::make_shared<Tensor>(std::forward<Args>(args)...);
}

void print_tensor_stats(const autograd::TensorPtr &tensor, const std::string &name);

}  // namespace ttml::autograd
