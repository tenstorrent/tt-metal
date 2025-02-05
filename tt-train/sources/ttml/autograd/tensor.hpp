// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
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
    tt::tt_metal::Tensor m_grad;
    bool m_requires_grad = true;
    std::optional<NodeId> m_node_id;

public:
    Tensor() = default;
    Tensor(const Tensor &) = default;
    Tensor(Tensor &&) noexcept = default;
    Tensor &operator=(const Tensor &) = default;
    Tensor &operator=(Tensor &&) noexcept = default;
    explicit Tensor(const tt::tt_metal::Tensor &value, bool requires_grad = true);
    ~Tensor() = default;

    void set_value(const tt::tt_metal::Tensor &value);
    void set_grad(const tt::tt_metal::Tensor &grad);
    void set_node(const std::optional<NodeId> &node);
    void clean_node();
    void add_grad(const tt::tt_metal::Tensor &grad);
    void set_requires_grad(bool requires_grad);

    const tt::tt_metal::Tensor &get_value(PreferredPrecision preferred_precision = PreferredPrecision::HALF) const;
    const tt::tt_metal::Tensor &get_grad() const;
    tt::tt_metal::Tensor &get_grad();
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
