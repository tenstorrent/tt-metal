// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"

namespace ttml::modules {

class Embedding : public autograd::ModuleBase {
    autograd::TensorPtr m_weight;

    void initialize_tensors(uint32_t num_embeddings, uint32_t embedding_dim);

public:
    Embedding(uint32_t num_embeddings, uint32_t embedding_dim);
    void set_weight(const autograd::TensorPtr& weight);
    [[nodiscard]] autograd::TensorPtr get_weight() const;

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& tensor);
};

}  // namespace ttml::modules
