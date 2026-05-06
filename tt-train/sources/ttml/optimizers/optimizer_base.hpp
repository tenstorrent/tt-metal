// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>

#include "serialization/serializable.hpp"

namespace ttml::optimizers {

class OptimizerBase {
public:
    [[nodiscard]] virtual std::string get_name() const = 0;

    explicit OptimizerBase(serialization::NamedParameters&& parameters);
    OptimizerBase(const OptimizerBase&) = delete;
    OptimizerBase& operator=(const OptimizerBase&) = delete;
    OptimizerBase(OptimizerBase&&) = delete;
    OptimizerBase& operator=(OptimizerBase&&) = delete;
    virtual ~OptimizerBase() = default;

    virtual void zero_grad() = 0;

    virtual void step() = 0;

    [[nodiscard]] virtual serialization::StateDict get_state_dict() const = 0;
    virtual void set_state_dict(const serialization::StateDict& dict) = 0;

    // Maps ValueType keys to a zero-initialized sentinel of the correct variant
    // alternative. The nanobind binding uses this to cast Python values to the
    // right C++ type without hardcoding key names. Populated by each subclass
    // constructor via m_state_dict_schema.
    [[nodiscard]] const std::unordered_map<std::string, serialization::ValueType>& get_state_dict_schema() const;

    [[nodiscard]] virtual size_t get_steps() const = 0;
    virtual void set_steps(size_t steps) = 0;

    virtual void set_lr(float lr) = 0;
    [[nodiscard]] virtual float get_lr() const = 0;

    virtual void print_stats() const;

protected:
    serialization::NamedParameters m_parameters;
    std::unordered_map<std::string, serialization::ValueType> m_state_dict_schema;
};

}  // namespace ttml::optimizers
