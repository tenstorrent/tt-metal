// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>

#include "serialization/serializable.hpp"
#include "tensor.hpp"

namespace ttml::autograd {

enum class RunMode { TRAIN, EVAL };

class ModuleBase;
using ModuleBasePtr = std::shared_ptr<ModuleBase>;

class ModuleBase {
private:
    std::string m_name;
    RunMode m_run_mode = RunMode::TRAIN;

    // Do not change map to unordered_map, as we need to keep order of iteration for serialization
    // special case for weight tying in transformers
    // for model save and load we need to make sure that stored/loaded name is the same between different runs
    // unordered_map does not guarantee the order of iteration
    std::map<std::string, TensorPtr> m_named_tensors;
    std::map<std::string, ModuleBasePtr> m_named_modules;

protected:
    void create_name(const std::string& name);
    void register_tensor(const TensorPtr& tensor_ptr, const std::string& name);
    void register_module(const ModuleBasePtr& module_ptr, const std::string& name);
    void override_tensor(const TensorPtr& tensor_ptr, const std::string& name);
    void override_module(const ModuleBasePtr& module_ptr, const std::string& name);

public:
    ModuleBase() = default;
    virtual ~ModuleBase() = default;
    ModuleBase(const ModuleBase&) = default;
    ModuleBase(ModuleBase&&) = default;
    ModuleBase& operator=(const ModuleBase&) = default;
    ModuleBase& operator=(ModuleBase&&) = default;

    [[nodiscard]] const std::string& get_name() const;
    [[nodiscard]] serialization::NamedParameters parameters() const;

    void train();
    void eval();
    void set_run_mode(RunMode mode);
    [[nodiscard]] RunMode get_run_mode() const;

    // Forward pass for the module. All posible overloads
    [[nodiscard]] virtual autograd::TensorPtr operator()(const autograd::TensorPtr& tensor);
    [[nodiscard]] virtual autograd::TensorPtr operator()(
        const autograd::TensorPtr& tensor, const autograd::TensorPtr& other);
};

}  // namespace ttml::autograd
