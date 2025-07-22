// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "serialization.hpp"
#include <enchantum/enchantum.hpp>

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
std::span<const uint8_t> to_bytes(T& value) {
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
    auto ptr = reinterpret_cast<uint8_t*>(&value);
    return std::span<const uint8_t>(ptr, sizeof(T));
}

template <>
std::span<const uint8_t> to_bytes(ttnn::Shape& value) {
    auto ptr = reinterpret_cast<const uint8_t*>(value.view().data());
    return std::span<const uint8_t>(ptr, sizeof(value[0]) * value.rank());
}

template <typename T>
void from_bytes(std::span<const uint8_t> bytes, T& value) {
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    if (bytes.size() != sizeof(T)) {
        std::ostringstream oss;
        oss << "Invalid byte size for conversion to type T. Expected: " << sizeof(T) << " Actual: " << bytes.size()
            << ", type: " << typeid(T).name();
        throw std::invalid_argument(oss.str());
    }

    std::memcpy(&value, bytes.data(), sizeof(T));
}

template <>
void from_bytes(std::span<const uint8_t> bytes, ttnn::Shape& value) {
    if (bytes.size() % sizeof(uint32_t) != 0) {
        std::ostringstream oss;
        oss << "Invalid byte size for conversion to type T. Expected divisible by" << sizeof(uint32_t)
            << " Actual: " << bytes.size() << ", type: " << typeid(ttnn::Shape).name();
        throw std::invalid_argument(oss.str());
    }
    ttnn::SmallVector<uint32_t> data(bytes.size() / sizeof(uint32_t));
    std::memcpy(data.data(), bytes.data(), bytes.size());
    value = ttnn::Shape(std::move(data));
}

template <typename T>
void get_enum(MsgPackFile& file, std::string_view name, T& value) {
    int int_value = 0;
    file.get(std::string(name), int_value);
    value = static_cast<T>(int_value);
}

void write_ttnn_tensor(MsgPackFile& file, std::string_view name, const tt::tt_metal::Tensor& tensor) {
    auto shape = tensor.logical_shape();
    auto data_type = tensor.dtype();
    auto layout = tensor.layout();
    auto storage_type = tensor.storage_type();

    file.put(std::string(name) + "/shape", to_bytes(shape));
    file.put(std::string(name) + "/data_type", static_cast<int>(data_type));
    file.put(std::string(name) + "/layout", static_cast<int>(layout));
    file.put(std::string(name) + "/storage_type", static_cast<int>(storage_type));

    // we currently assume that there are two types of runs: single device and DDP
    // once we decide to use other parallelization techniques (tensor parallel, FSDP) we need to update this code
    if (data_type == tt::tt_metal::DataType::BFLOAT16) {
        auto* device = &ttml::autograd::ctx().get_device();
        ttml::core::MeshToXTensorVariant<float> composer = ttml::core::VectorMeshToXTensor<float>(device->shape());
        auto data_all_devices = ttml::core::to_xtensor<float>(tensor, composer);
        // pick weights from first device
        auto data = data_all_devices.front();
        file.put(std::string(name) + "/data", std::span<const float>(data.data(), data.size()));
    } else if (data_type == tt::tt_metal::DataType::UINT32) {
        auto* device = &ttml::autograd::ctx().get_device();
        ttml::core::MeshToXTensorVariant<uint32_t> composer =
            ttml::core::VectorMeshToXTensor<uint32_t>(device->shape());
        auto data_all_devices = ttml::core::to_xtensor<uint32_t>(tensor, composer);
        // pick weights from first device
        auto data = data_all_devices.front();
        file.put(std::string(name) + "/data", std::span<const uint32_t>(data.data(), data.size()));
    } else {
        throw std::runtime_error(fmt::format("Unsupported data type: {}", enchantum::to_string(data_type)));
    }
}

void read_ttnn_tensor(MsgPackFile& file, std::string_view name, tt::tt_metal::Tensor& tensor) {
    tt::tt_metal::DataType data_type{};
    tt::tt_metal::Layout layout{};
    tt::tt_metal::StorageType storage_type{};

    auto shape = ttnn::Shape({1, 1, 1, 1});
    std::vector<uint8_t> bytes;
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
        tensor = core::from_vector<uint32_t, tt::tt_metal::DataType::UINT32>(
            data, shape, &ttml::autograd::ctx().get_device(), layout);
    } else {
        throw std::runtime_error(fmt::format("Unsupported data type: {}", enchantum::to_string(data_type)));
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

void write_named_parameters(
    MsgPackFile& file, std::string_view name, const ttml::serialization::NamedParameters& params) {
    for (const auto& [key, value] : params) {
        write_autograd_tensor(file, std::string(name) + "/" + key, value);
    }
}
void read_named_parameters(MsgPackFile& file, std::string_view name, ttml::serialization::NamedParameters& params) {
    for (auto& [key, value] : params) {
        read_autograd_tensor(file, std::string(name) + "/" + key, value);
    }
}

void write_optimizer(MsgPackFile& file, std::string_view name, const optimizers::OptimizerBase* optimizer) {
    assert(optimizer);
    auto state_dict = optimizer->get_state_dict();
    write_state_dict(file, std::string(name), state_dict);
}

void read_optimizer(MsgPackFile& file, std::string_view name, optimizers::OptimizerBase* optimizer) {
    assert(optimizer);
    size_t steps = 0;
    auto state_dict = optimizer->get_state_dict();
    read_state_dict(file, name, state_dict);
    optimizer->set_state_dict(state_dict);
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

void write_state_dict(MsgPackFile& file, std::string_view name, const serialization::StateDict& state_dict) {
    for (const auto& [key, value] : state_dict) {
        if (std::holds_alternative<ValueType>(value)) {
            file.put(std::string(name) + "/" + key, std::get<ValueType>(value));
        } else if (std::holds_alternative<ttnn::Tensor>(value)) {
            write_ttnn_tensor(file, std::string(name) + "/" + key, std::get<ttnn::Tensor>(value));
        } else if (std::holds_alternative<ttml::autograd::TensorPtr>(value)) {
            write_autograd_tensor(file, std::string(name) + "/" + key, std::get<ttml::autograd::TensorPtr>(value));
        } else if (std::holds_alternative<NamedParameters>(value)) {
            write_named_parameters(file, std::string(name) + "/" + key, std::get<NamedParameters>(value));
        } else {
            throw std::runtime_error("Unsupported type in state dict");
        }
    }
}
void read_state_dict(MsgPackFile& file, std::string_view name, serialization::StateDict& state_dict) {
    for (auto& [key, value] : state_dict) {
        if (std::holds_alternative<ValueType>(value)) {
            file.get(std::string(name) + "/" + key, std::get<ValueType>(value));
        } else if (std::holds_alternative<ttnn::Tensor>(value)) {
            read_ttnn_tensor(file, std::string(name) + "/" + key, std::get<ttnn::Tensor>(value));
        } else if (std::holds_alternative<ttml::autograd::TensorPtr>(value)) {
            read_autograd_tensor(file, std::string(name) + "/" + key, std::get<ttml::autograd::TensorPtr>(value));
        } else if (std::holds_alternative<NamedParameters>(value)) {
            read_named_parameters(file, std::string(name) + "/" + key, std::get<NamedParameters>(value));
        } else {
            throw std::runtime_error("Unsupported type in state dict");
        }
    }
}

}  // namespace ttml::serialization
