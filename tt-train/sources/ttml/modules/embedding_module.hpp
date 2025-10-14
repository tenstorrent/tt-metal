// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "modules/module_base.hpp"

namespace ttml::modules {

class Embedding : public ModuleBase {
    autograd::TensorPtr m_weight;

    void initialize_tensors(uint32_t num_embeddings, uint32_t embedding_dim);

public:
    Embedding(uint32_t num_embeddings, uint32_t embedding_dim);
    Embedding(const autograd::TensorPtr& weight);
    [[nodiscard]] autograd::TensorPtr get_weight() const;

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& tensor) override;
};

}  // namespace ttml::modules
