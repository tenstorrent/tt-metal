// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "module_base.hpp"

#include <queue>
#include <string>
#include <unordered_set>

namespace ttml::autograd {

void ModuleBase::register_tensor(const TensorPtr& tensor_ptr, const std::string& name) {
    auto [_, is_inserted] = m_named_tensors.emplace(name, tensor_ptr);
    if (!is_inserted) {
        throw std::logic_error("Names of two tensors coincide");
    }
}

void ModuleBase::register_module(const ModuleBasePtr& module_ptr, const std::string& name) {
    if (module_ptr == nullptr) {
        throw std::runtime_error(fmt::format("Module {} is uninitialized.", name));
    }
    auto [_, is_inserted] = m_named_modules.emplace(name, module_ptr);
    if (!is_inserted) {
        throw std::logic_error(fmt::format("Names of two modules coincide: {}", name));
    }
}

void ModuleBase::override_tensor(const TensorPtr& tensor_ptr, const std::string& name) {
    if (auto it = m_named_tensors.find(name); it != m_named_tensors.end()) {
        it->second = tensor_ptr;
    } else {
        throw std::logic_error(fmt::format("Tensor with such name does not exist. Name {}", name));
    }
}

void ModuleBase::override_module(const ModuleBasePtr& module_ptr, const std::string& name) {
    if (auto it = m_named_modules.find(name); it != m_named_modules.end()) {
        it->second = module_ptr;
    } else {
        throw std::logic_error(fmt::format("Module with such name does not exist. Name {}", name));
    }
}

void ModuleBase::create_name(const std::string& name) {
    m_name = name;
}

const std::string& ModuleBase::get_name() const {
    return m_name;
}

serialization::NamedParameters ModuleBase::parameters() const {
    serialization::NamedParameters params;

    std::queue<std::pair<const ModuleBase*, std::string>> modules_to_process;
    modules_to_process.emplace(this, get_name() + "/");

    std::unordered_set<std::string> modules_in_queue;
    modules_in_queue.insert(get_name());

    // We need to store the address of the tensor to avoid duplicates
    // as the same tensor can be registered in different modules
    // and we need to store it only once
    // Usecase: weight tying in transformers (embedding + output layer)
    // std::uintptr_t is used to store the address of the tensor, and system dependent (32 or 64 bit)
    std::unordered_set<std::uintptr_t> tensors_in_params;

    while (!modules_to_process.empty()) {
        auto [module_ptr, name_prefix] = modules_to_process.front();
        modules_to_process.pop();

        for (const auto& [tensor_name, tensor_ptr] : module_ptr->m_named_tensors) {
            auto tensor_ptr_address = reinterpret_cast<std::uintptr_t>(tensor_ptr.get());
            if (!tensors_in_params.contains(tensor_ptr_address)) {
                tensors_in_params.insert(tensor_ptr_address);
                params.emplace(name_prefix + tensor_name, tensor_ptr);
            }
        }

        for (const auto& [module_name, next_module_ptr] : module_ptr->m_named_modules) {
            const auto module_name_with_prefix = name_prefix + module_name;
            if (!modules_in_queue.contains(module_name_with_prefix)) {
                modules_to_process.emplace(next_module_ptr.get(), name_prefix + module_name + "/");
                modules_in_queue.insert(module_name_with_prefix);
            }
        }
    }

    return params;
}

void ModuleBase::set_run_mode(RunMode mode) {
    m_run_mode = mode;
    for (auto& [_, module] : this->m_named_modules) {
        module->set_run_mode(mode);
    }
}

[[nodiscard]] RunMode ModuleBase::get_run_mode() const {
    return m_run_mode;
}

void ModuleBase::train() {
    set_run_mode(RunMode::TRAIN);
}

void ModuleBase::eval() {
    set_run_mode(RunMode::EVAL);
}
autograd::TensorPtr ModuleBase::operator()(const autograd::TensorPtr& tensor) {
    throw std::logic_error("ModuleBase::operator()(const autograd::TensorPtr& tensor) is Not implemented");
}
autograd::TensorPtr ModuleBase::operator()(const autograd::TensorPtr& tensor, const autograd::TensorPtr& other) {
    throw std::logic_error(
        "ModuleBase::operator()(const autograd::TensorPtr& tensor, const autograd::TensorPtr& other) is Not "
        "implemented");
}

}  // namespace ttml::autograd
