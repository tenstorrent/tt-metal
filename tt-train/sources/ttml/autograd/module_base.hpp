// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <unordered_map>

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

    std::unordered_map<std::string, TensorPtr> m_named_tensors;
    std::unordered_map<std::string, ModuleBasePtr> m_named_modules;

protected:
    void create_name(const std::string& name);
    void register_tensor(const TensorPtr& tensor_ptr, const std::string& name);
    void register_module(const ModuleBasePtr& module_ptr, const std::string& name);

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
};

}  // namespace ttml::autograd
