// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/module_base.hpp"
#include "modules/distributed/linear.hpp"

class MNISTTensorParallel : public ttml::autograd::ModuleBase {
public:
    MNISTTensorParallel();
    ttml::autograd::TensorPtr operator()(ttml::autograd::TensorPtr tensor);

private:
    std::shared_ptr<ttml::modules::distributed::ColumnParallelLinear> m_linear1;
    std::shared_ptr<ttml::modules::distributed::RowParallelLinear> m_linear2;
};
