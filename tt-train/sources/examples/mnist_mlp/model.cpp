// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "model.hpp"

#include "ops/unary_ops.hpp"

MNISTTensorParallel::MNISTTensorParallel() {
    m_linear1 = std::make_shared<ttml::modules::distributed::ColumnParallelLinear>(
        784, 128, /* has_bias */ true, /* gather_output */ false);
    m_linear2 = std::make_shared<ttml::modules::distributed::RowParallelLinear>(
        128, 10, /* has_bias */ true, /* input_is_parallel */ true);
    create_name("mlp");
    register_module(m_linear1, "linear1");
    register_module(m_linear2, "linear2");
}

ttml::autograd::TensorPtr MNISTTensorParallel::operator()(ttml::autograd::TensorPtr tensor) {
    tensor = (*m_linear1)(tensor);
    tensor = ttml::ops::relu(tensor);
    tensor = (*m_linear2)(tensor);
    return tensor;
}
