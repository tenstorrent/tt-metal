// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <vector>

#include "autograd/tensor.hpp"
#include "serialization/serializable.hpp"

namespace ttml::modules {

enum class RunMode { TRAIN, EVAL };

class ModuleBase;
using ModuleBasePtr = std::shared_ptr<ModuleBase>;

// Hook type definitions for FSDP and other use cases
// PreForwardHook: Called before forward pass with (module, input)
// PostForwardHook: Called after forward pass with (module, input, output)
using PreForwardHook = std::function<void(ModuleBase*, const autograd::TensorPtr&)>;
using PostForwardHook = std::function<void(ModuleBase*, const autograd::TensorPtr&, const autograd::TensorPtr&)>;

// Hook handle for removing hooks
using HookHandle = size_t;

class ModuleBase {
private:
    std::string m_name;
    RunMode m_run_mode = RunMode::TRAIN;

    // Do not change map to unordered_map, as we need to keep order of iteration for serialization
    // special case for weight tying in transformers
    // for model save and load we need to make sure that stored/loaded name is the same between different runs
    // unordered_map does not guarantee the order of iteration
    std::map<std::string, autograd::TensorPtr> m_named_tensors;
    std::map<std::string, ModuleBasePtr> m_named_modules;

    // Hook storage for pre/post forward hooks
    std::vector<std::pair<HookHandle, PreForwardHook>> m_pre_forward_hooks;
    std::vector<std::pair<HookHandle, PostForwardHook>> m_post_forward_hooks;
    HookHandle m_next_hook_handle = 0;

protected:
    void create_name(const std::string& name);
    void register_tensor(const autograd::TensorPtr& tensor_ptr, const std::string& name);
    void register_module(const ModuleBasePtr& module_ptr, const std::string& name);
    void override_tensor(const autograd::TensorPtr& tensor_ptr, const std::string& name);
    void override_module(const ModuleBasePtr& module_ptr, const std::string& name);

    // Run all registered pre-forward hooks
    void run_pre_forward_hooks(const autograd::TensorPtr& input);
    // Run all registered post-forward hooks
    void run_post_forward_hooks(const autograd::TensorPtr& input, const autograd::TensorPtr& output);

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

    // Hook registration methods
    // Returns a handle that can be used to remove the hook
    HookHandle register_pre_forward_hook(PreForwardHook hook);
    HookHandle register_post_forward_hook(PostForwardHook hook);

    // Remove a hook by its handle
    void remove_pre_forward_hook(HookHandle handle);
    void remove_post_forward_hook(HookHandle handle);

    // Clear all hooks
    void clear_pre_forward_hooks();
    void clear_post_forward_hooks();
    void clear_all_hooks();

    // Check if hooks are registered
    [[nodiscard]] bool has_pre_forward_hooks() const;
    [[nodiscard]] bool has_post_forward_hooks() const;

    // Forward pass for the module. All possible overloads
    [[nodiscard]] virtual autograd::TensorPtr operator()(const autograd::TensorPtr& tensor);
    [[nodiscard]] virtual autograd::TensorPtr operator()(
        const autograd::TensorPtr& tensor, const autograd::TensorPtr& other);

    // Call method that wraps operator() with hooks
    // This is the recommended way to invoke modules when hooks are needed
    [[nodiscard]] autograd::TensorPtr call_with_hooks(const autograd::TensorPtr& tensor);
    [[nodiscard]] autograd::TensorPtr call_with_hooks(
        const autograd::TensorPtr& tensor, const autograd::TensorPtr& other);
};

}  // namespace ttml::modules
