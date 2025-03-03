// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "autograd/tensor.hpp"
#include "linear.hpp"
#include "modules/distributed/linear.hpp"
#include "modules/dropout_module.hpp"

namespace ttml::modules::distributed {

class DistributedMultiHeadAttention : public ttml::autograd::ModuleBase {
private:
    uint32_t m_embedding_dim{};
    uint32_t m_num_heads{};
    uint32_t m_local_num_heads{};
    std::shared_ptr<ColumnParallelLinear> m_qkv_linear;
    std::shared_ptr<RowParallelLinear> m_out_linear;
    std::shared_ptr<DropoutLayer> m_dropout;

public:
    explicit DistributedMultiHeadAttention(uint32_t embedding_dim, uint32_t num_heads, float dropout_prob);

    autograd::TensorPtr operator()(const autograd::TensorPtr& x, const autograd::TensorPtr& mask) override;
};

}  // namespace ttml::modules::distributed
