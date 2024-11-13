// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/module_base.hpp"

namespace ttml::optimizers {

class OptimizerBase {
public:
    explicit OptimizerBase(autograd::NamedParameters&& parameters);
    OptimizerBase(const OptimizerBase&) = delete;
    OptimizerBase& operator=(const OptimizerBase&) = delete;
    OptimizerBase(OptimizerBase&&) = delete;
    OptimizerBase& operator=(OptimizerBase&&) = delete;
    virtual ~OptimizerBase() = default;

    virtual void zero_grad() = 0;

    virtual void step() = 0;

    [[nodiscard]] virtual autograd::NamedParameters get_state_dict() const = 0;
    virtual void set_state_dict(const autograd::NamedParameters& dict) = 0;

    [[nodiscard]] virtual size_t get_steps() const = 0;
    virtual void set_steps(size_t steps) = 0;

    virtual void print_stats() const;

protected:
    autograd::NamedParameters m_parameters;
};

}  // namespace ttml::optimizers
