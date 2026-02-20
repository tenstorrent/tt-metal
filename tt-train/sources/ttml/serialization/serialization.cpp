// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "serialization.hpp"

#include <core/ttnn_all_includes.hpp>
#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <filesystem>
#include <fstream>
#include <ttnn/tensor/types.hpp>

#include "autograd/auto_context.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "flatbuffer_file.hpp"
#include "modules/module_base.hpp"
#include "optimizers/optimizer_base.hpp"
#include "optimizers/sgd.hpp"
#include "ttnn/tensor/serialization.hpp"

namespace ttml::serialization {

// Concept for trivially copyable types
template <typename T>
concept TriviallyCopyable = std::is_trivially_copyable_v<T>;

// demangle type name

// trivial type to the std::string
template <TriviallyCopyable T>
std::span<const uint8_t> to_bytes(T& value) {
    auto ptr = reinterpret_cast<uint8_t*>(&value);
    return std::span<const uint8_t>(ptr, sizeof(T));
}

// Specialization for ttnn::Shape (not trivially copyable, handled specially)
inline std::span<const uint8_t> to_bytes(ttnn::Shape& value) {
    auto ptr = reinterpret_cast<const uint8_t*>(value.view().data());
    return std::span<const uint8_t>(ptr, sizeof(value[0]) * value.rank());
}

template <TriviallyCopyable T>
void from_bytes(std::span<const uint8_t> bytes, T& value) {
    if (bytes.size() != sizeof(T)) {
        std::ostringstream oss;
        oss << "Invalid byte size for conversion to type T. Expected: " << sizeof(T) << " Actual: " << bytes.size()
            << ", type: " << typeid(T).name();
        throw std::invalid_argument(oss.str());
    }

    std::memcpy(&value, bytes.data(), sizeof(T));
}

// Specialization for ttnn::Shape (not trivially copyable, handled specially)
inline void from_bytes(std::span<const uint8_t> bytes, ttnn::Shape& value) {
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
void get_enum(FlatBufferFile& file, std::string_view name, T& value) {
    int int_value = file.get_int(std::string(name));
    value = static_cast<T>(int_value);
}

void write_ttnn_tensor(FlatBufferFile& file, std::string_view name, const tt::tt_metal::Tensor& tensor) {
    // Store tensor in m_tensors - it will be written to file during serialize()
    // This ensures tensors are written to the correct directory
    file.put(name, tensor);

    // Store metadata needed to restore tensor properties (layout, shape, storage_type) on deserialization
    auto shape = tensor.logical_shape();
    auto data_type = tensor.dtype();
    auto layout = tensor.layout();
    auto storage_type = tensor.storage_type();

    file.put(std::string(name) + "/shape", to_bytes(shape));
    file.put(std::string(name) + "/data_type", static_cast<int>(data_type));
    file.put(std::string(name) + "/layout", static_cast<int>(layout));
    file.put(std::string(name) + "/storage_type", static_cast<int>(storage_type));
}

void read_ttnn_tensor(FlatBufferFile& file, std::string_view name, tt::tt_metal::Tensor& tensor) {
    // Read tensor directly from m_tensors (loaded during deserialize)
    tensor = file.get_tensor(name);

    // Restore original layout and shape if needed
    tt::tt_metal::Layout original_layout{};
    get_enum(file, std::string(name) + "/layout", original_layout);

    ttnn::Shape original_shape{};
    std::vector<uint8_t> shape_bytes = file.get_vector_uint8(std::string(name) + "/shape");
    from_bytes(shape_bytes, original_shape);

    // If tensor was padded to TILE layout (for ROW_MAJOR device tensors), unpad it
    if (tensor.layout() == tt::tt_metal::Layout::TILE && original_layout == tt::tt_metal::Layout::ROW_MAJOR) {
        // Convert to CPU first, then unpad and convert layout
        tensor = tensor.cpu();
        if (tensor.logical_shape() != original_shape) {
            tensor = tensor.unpad_from_tile(original_shape);
        }
        tensor = tensor.to_layout(original_layout);
    } else if (tensor.layout() != original_layout) {
        // For other cases, just convert layout
        tensor = tensor.cpu().to_layout(original_layout);
    }

    // Restore storage type if needed
    tt::tt_metal::StorageType storage_type{};
    get_enum(file, std::string(name) + "/storage_type", storage_type);

    if (storage_type == tt::tt_metal::StorageType::DEVICE &&
        tensor.storage_type() != tt::tt_metal::StorageType::DEVICE) {
        tensor = tensor.to_device(&ttml::autograd::ctx().get_device());
    }
}

void write_autograd_tensor(
    FlatBufferFile& file, std::string_view name, const ttml::autograd::TensorPtr& tensor, bool save_grads) {
    write_ttnn_tensor(file, std::string(name) + "/value", tensor->get_value(ttml::autograd::PreferredPrecision::FULL));
    auto& grad = tensor->get_grad();
    bool has_grads = save_grads && core::is_tensor_initialized(grad);
    file.put(std::string(name) + "/requires_grads", tensor->get_requires_grad());
    file.put(std::string(name) + "/has_grads", has_grads);
    if (has_grads) {
        write_ttnn_tensor(file, std::string(name) + "/grad", tensor->get_grad());
    }
}

void read_autograd_tensor(FlatBufferFile& file, std::string_view name, ttml::autograd::TensorPtr& tensor) {
    tt::tt_metal::Tensor value;
    bool has_grads = false;
    bool requires_grads = false;
    read_ttnn_tensor(file, std::string(name) + "/value", value);
    tensor->set_value(value);
    requires_grads = file.get_bool(std::string(name) + "/requires_grads");
    has_grads = file.get_bool(std::string(name) + "/has_grads");
    tensor->set_requires_grad(requires_grads);
    if (has_grads) {
        tt::tt_metal::Tensor grad;
        read_ttnn_tensor(file, std::string(name) + "/grad", grad);
        tensor->set_grad(grad);
    }
}

void write_named_parameters(
    FlatBufferFile& file, std::string_view name, const ttml::serialization::NamedParameters& params) {
    for (const auto& [key, value] : params) {
        write_autograd_tensor(file, std::string(name) + "/" + key, value);
    }
}
void read_named_parameters(FlatBufferFile& file, std::string_view name, ttml::serialization::NamedParameters& params) {
    for (auto& [key, value] : params) {
        read_autograd_tensor(file, std::string(name) + "/" + key, value);
    }
}

void write_optimizer(FlatBufferFile& file, std::string_view name, const optimizers::OptimizerBase* optimizer) {
    assert(optimizer);
    auto state_dict = optimizer->get_state_dict();
    write_state_dict(file, std::string(name), state_dict);
}

void read_optimizer(FlatBufferFile& file, std::string_view name, optimizers::OptimizerBase* optimizer) {
    assert(optimizer);
    auto state_dict = optimizer->get_state_dict();
    read_state_dict(file, name, state_dict);
    optimizer->set_state_dict(state_dict);
}

void write_module(FlatBufferFile& file, std::string_view name, const modules::ModuleBase* module) {
    assert(module);
    assert(!name.empty());
    if (name.empty()) {
        throw std::runtime_error("Model name must be non-empty when writing module");
    }
    auto named_parameters = module->parameters();
    write_named_parameters(file, name, named_parameters);
}

void read_module(FlatBufferFile& file, std::string_view name, modules::ModuleBase* module) {
    assert(module);
    assert(!name.empty());
    if (name.empty()) {
        throw std::runtime_error("Model name must be non-empty when reading module");
    }
    auto named_parameters = module->parameters();
    read_named_parameters(file, name, named_parameters);
}

void write_state_dict(FlatBufferFile& file, std::string_view name, const serialization::StateDict& state_dict) {
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
void read_state_dict(FlatBufferFile& file, std::string_view name, serialization::StateDict& state_dict) {
    for (auto& [key, value] : state_dict) {
        if (std::holds_alternative<ValueType>(value)) {
            std::get<ValueType>(value) = file.get_value_type(std::string(name) + "/" + key);
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
