// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "modules/distributed/linear.hpp"
#include "modules/module_base.hpp"

class MNISTTensorParallel : public ttml::modules::ModuleBase {
public:
    MNISTTensorParallel();
    ttml::autograd::TensorPtr operator()(ttml::autograd::TensorPtr tensor);

private:
    std::shared_ptr<ttml::modules::distributed::ColumnParallelLinear> m_linear1;
    std::shared_ptr<ttml::modules::distributed::RowParallelLinear> m_linear2;
};
