// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "serialization.hpp"

#include <core/ttnn_all_includes.hpp>
#include <cstdint>
#include <ttnn/tensor/types.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/module_base.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "msgpack_file.hpp"
#include "optimizers/optimizer_base.hpp"
#include "optimizers/sgd.hpp"
namespace ttml::serialization {

// demangle type name

// trivial type to the std::string
template <typename T>
std::string to_bytes(const T& value) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
    std::string bytes(sizeof(T), '\0');
    std::memcpy(bytes.data(), &value, sizeof(T));
    return bytes;
}

template <typename T>
void from_bytes(const std::string& bytes, T& value) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");

    if (bytes.size() != sizeof(T)) {
        throw std::invalid_argument(fmt::format(
            "Invalid byte size for conversion to type T. Expected: {} Actual: {}, type: {} ",
            sizeof(T),
            bytes.size(),
            core::demangle(typeid(T).name())));
    }
    std::memcpy(&value, bytes.data(), sizeof(T));
}

template <typename T>
void get_enum(MsgPackFile& file, std::string_view name, T& value) {
    int int_value = 0;
    file.get(std::string(name), int_value);
    value = static_cast<T>(int_value);
}

void write_ttnn_tensor(MsgPackFile& file, std::string_view name, const tt::tt_metal::Tensor& tensor) {
    auto shape = tensor.get_shape();
    auto data_type = tensor.get_dtype();
    auto layout = tensor.get_layout();
    auto storage_type = tensor.storage_type();

    file.put(std::string(name) + "/shape", to_bytes(shape));
    file.put(std::string(name) + "/data_type", static_cast<int>(data_type));
    file.put(std::string(name) + "/layout", static_cast<int>(layout));
    file.put(std::string(name) + "/storage_type", static_cast<int>(storage_type));

    if (data_type == tt::tt_metal::DataType::BFLOAT16) {
        auto data = ttml::core::to_vector<float>(tensor);
        file.put(std::string(name) + "/data", std::span<const float>(data.data(), data.size()));
    } else if (data_type == tt::tt_metal::DataType::UINT32) {
        auto data = ttml::core::to_vector<uint32_t>(tensor);
        file.put(std::string(name) + "/data", std::span<const uint32_t>(data.data(), data.size()));
    } else {
        throw std::runtime_error(fmt::format("Unsupported data type: {}", magic_enum::enum_name(data_type)));
    }
}

void read_ttnn_tensor(MsgPackFile& file, std::string_view name, tt::tt_metal::Tensor& tensor) {
    tt::tt_metal::DataType data_type{};
    tt::tt_metal::Layout layout{};
    tt::tt_metal::StorageType storage_type{};

    auto shape = core::create_shape({1, 1, 1, 1});
    std::string bytes;
    file.get(std::string(name) + "/shape", bytes);
    from_bytes<ttnn::Shape>(bytes, shape);

    get_enum(file, std::string(name) + "/data_type", data_type);
    get_enum(file, std::string(name) + "/layout", layout);
    get_enum(file, std::string(name) + "/storage_type", storage_type);

    if (data_type == tt::tt_metal::DataType::BFLOAT16) {
        std::vector<float> data;
        file.get(std::string(name) + "/data", data);
        tensor = core::from_vector(data, shape, &ttml::autograd::ctx().get_device(), layout);
    } else if (data_type == tt::tt_metal::DataType::UINT32) {
        std::vector<uint32_t> data;
        file.get(std::string(name) + "/data", data);
        tensor =
            core::from_vector<uint32_t, DataType::UINT32>(data, shape, &ttml::autograd::ctx().get_device(), layout);
    } else {
        throw std::runtime_error(fmt::format("Unsupported data type: {}", magic_enum::enum_name(data_type)));
    }
}

void write_autograd_tensor(
    MsgPackFile& file, std::string_view name, const ttml::autograd::TensorPtr& tensor, bool save_grads) {
    write_ttnn_tensor(file, std::string(name) + "/value", tensor->get_value());
    auto& grad = tensor->get_grad();
    bool has_grads = save_grads && core::is_tensor_initialized(grad);
    file.put(std::string(name) + "/requires_grads", tensor->get_requires_grad());
    file.put(std::string(name) + "/has_grads", has_grads);
    if (has_grads) {
        write_ttnn_tensor(file, std::string(name) + "/grad", tensor->get_grad());
    }
}

void read_autograd_tensor(MsgPackFile& file, std::string_view name, ttml::autograd::TensorPtr& tensor) {
    tt::tt_metal::Tensor value;
    bool has_grads = false;
    bool requires_grads = false;
    read_ttnn_tensor(file, std::string(name) + "/value", value);
    tensor->set_value(value);
    file.get(std::string(name) + "/requires_grads", requires_grads);
    file.get(std::string(name) + "/has_grads", has_grads);
    tensor->set_requires_grad(requires_grads);
    if (has_grads) {
        tt::tt_metal::Tensor grad;
        read_ttnn_tensor(file, std::string(name) + "/grad", grad);
        tensor->set_grad(grad);
    }
}

void write_named_parameters(MsgPackFile& file, std::string_view name, const ttml::autograd::NamedParameters& params) {
    for (const auto& [key, value] : params) {
        write_autograd_tensor(file, std::string(name) + "/" + key, value);
    }
}
void read_named_parameters(MsgPackFile& file, std::string_view name, ttml::autograd::NamedParameters& params) {
    for (auto& [key, value] : params) {
        read_autograd_tensor(file, std::string(name) + "/" + key, value);
    }
}

void write_optimizer(MsgPackFile& file, std::string_view name, const optimizers::OptimizerBase* optimizer) {
    assert(optimizer);
    auto state_dict = optimizer->get_state_dict();
    for (const auto& [key, value] : state_dict) {
        ttml::serialization::write_autograd_tensor(file, std::string(name) + "/" + key, value);
    }
    file.put(std::string(name) + "/steps", optimizer->get_steps());
}

void read_optimizer(MsgPackFile& file, std::string_view name, optimizers::OptimizerBase* optimizer) {
    assert(optimizer);
    size_t steps = 0;
    auto state_dict = optimizer->get_state_dict();
    for (auto& [key, value] : state_dict) {
        ttml::serialization::read_autograd_tensor(file, std::string(name) + "/" + key, value);
    }
    optimizer->set_state_dict(state_dict);
    file.get(std::string(name) + "/steps", steps);
    optimizer->set_steps(steps);
}

void write_module(MsgPackFile& file, std::string_view name, const autograd::ModuleBase* module) {
    assert(module);
    auto named_parameters = module->parameters();
    write_named_parameters(file, name, named_parameters);
}

void read_module(MsgPackFile& file, std::string_view name, autograd::ModuleBase* module) {
    assert(module);
    auto named_parameters = module->parameters();
    read_named_parameters(file, name, named_parameters);
}

}  // namespace ttml::serialization
