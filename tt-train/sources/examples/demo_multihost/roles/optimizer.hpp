// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "core/distributed/distributed.hpp"
#include "optimizers/sgd.hpp"
#include "worker.hpp"
namespace roles {
class Optimizer {
public:
    Optimizer(std::shared_ptr<ttml::modules::LinearLayer> model, int rank, int worker_rank);
    void optimization_step();
    void send_weights();

private:
    std::shared_ptr<ttml::modules::LinearLayer> m_model;
    std::shared_ptr<ttml::optimizers::OptimizerBase> m_optimizer;
    SortedParameters m_sorted_parameters;
    int m_rank = 0;
    int m_worker_rank = 0;
};

}  // namespace roles
