// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "autograd/graph.hpp"
#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"
#include "models/gpt2.hpp"
#include "models/linear_regression.hpp"
#include "modules/gpt_block.hpp"
#include "modules/linear_module.hpp"
#include "modules/llama_block.hpp"
#include "ops/losses.hpp"
#include "optimizers/sgd.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"
#include "ttnn/operations/loss/loss.hpp"

namespace nanobind::detail {
template <>
struct dtype_traits<bfloat16> {
    static constexpr dlpack::dtype value{
        (uint8_t)dlpack::dtype_code::Float,  // type code
        16,                                  // size in bits
        1                                    // lanes (simd), usually set to 1
    };
    static constexpr auto name = const_name("bfloat16");
};
}  // namespace nanobind::detail
namespace ttml::autograd {

nb::ndarray<nb::numpy> make_numpy_tensor(const tt::tt_metal::Tensor& tensor) {
    const auto tensor_spec = tensor.tensor_spec();
    const auto impl = [&tensor_spec]<typename T>(const tt::tt_metal::Tensor& t) {
        const tt::tt_metal::Shape& tensor_shape = tensor_spec.logical_shape();

        const auto tensor_shape_rank = tensor_shape.rank();
        std::vector<size_t> numpy_shape(tensor_shape_rank);
        std::copy(tensor_shape.cbegin(), tensor_shape.cend(), numpy_shape.begin());

        const auto tensor_strides = t.strides();
        std::vector<int64_t> numpy_strides(tensor_strides.rank());
        std::copy(tensor_strides.cbegin(), tensor_strides.cend(), numpy_strides.begin());

        const auto tensor_data = t.template to_vector<T>();
        T* numpy_data = new T[tensor_data.size()];
        std::copy(tensor_data.cbegin(), tensor_data.cend(), numpy_data);

        const nb::capsule owner(numpy_data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
        return nb::ndarray<nb::numpy>(
            numpy_data, tensor_shape_rank, numpy_shape.data(), owner, numpy_strides.data(), nb::dtype<T>());
    };

    const auto ensure_row_major = [&impl]<typename T>(const tt::tt_metal::Tensor& t) {
        if (t.layout() != tt::tt_metal::Layout::ROW_MAJOR) {
            return impl.template operator()<T>(
                ttnn::untilize_with_unpadding(ttnn::DefaultQueueId, t, t.logical_shape(), std::nullopt));
        }
        return impl.template operator()<T>(t);
    };

    switch (tensor_spec.data_type()) {
        case tt::tt_metal::DataType::INT32: return ensure_row_major.template operator()<int32_t>(tensor);
        case tt::tt_metal::DataType::UINT32: return ensure_row_major.template operator()<uint32_t>(tensor);
        case tt::tt_metal::DataType::FLOAT32: return ensure_row_major.template operator()<float>(tensor);
        case tt::tt_metal::DataType::BFLOAT16: return ensure_row_major.template operator()<bfloat16>(tensor);
        case tt::tt_metal::DataType::BFLOAT8_B: TT_THROW("Unsupported type: BFLOAT8_B"); break;
        case tt::tt_metal::DataType::BFLOAT4_B: TT_THROW("Unsupported type: BFLOAT4_B"); break;
        case tt::tt_metal::DataType::UINT8: TT_THROW("Unsupported type: UINT8"); break;
        case tt::tt_metal::DataType::UINT16: TT_THROW("Unsupported type: UINT16"); break;
        case tt::tt_metal::DataType::INVALID: TT_THROW("Unsupported type: INVALID"); break;
    }

    TT_THROW("Unsupported type: unknown");
}

tt::tt_metal::Tensor make_metal_tensor(nb::ndarray<> data) {
    const auto data_type = data.dtype();
    TT_FATAL(!(data_type.bits % 8), "Unsupported precision: {} bits", data_type.bits);

    const auto rank = data.ndim();
    tt::tt_metal::ShapeBase::Container shape_container(rank);
    for (size_t dimension = 0; dimension < rank; ++dimension) {
        const auto dimension_size = data.shape(dimension);
        TT_FATAL(
            dimension_size >= std::numeric_limits<uint32_t>::min(),
            "Invalid shape parameter for dimension {}: {} is too small",
            dimension,
            dimension_size);
        TT_FATAL(
            dimension_size <= std::numeric_limits<uint32_t>::max(),
            "Invalid shape parameter for dimension {}: {} is too large",
            dimension,
            dimension_size);
        shape_container[dimension] = dimension_size;
    }
    const tt::tt_metal::Shape tensor_shape(shape_container);
    const tt::tt_metal::MemoryConfig tensor_memory_config{};
    const tt::tt_metal::PageConfig tensor_page_config(tt::tt_metal::Layout::ROW_MAJOR);

    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();

    const auto impl = [&]<typename T>(tt::tt_metal::DataType tensor_data_type) {
        TT_FATAL(
            data_type.bits == (sizeof(T) * 8),
            "Unsupported precision: expected {} bits, got {} bits",
            sizeof(T) * 8,
            data_type.bits);

        tt::tt_metal::TensorLayout tensor_layout(tensor_data_type, tensor_page_config, tensor_memory_config);
        tt::tt_metal::TensorSpec tensor_spec(tensor_shape, tensor_layout);
        return ttnn::tilize_with_zero_padding(
                   ttnn::DefaultQueueId,
                   tt::tt_metal::Tensor::from_span(
                       tt::stl::Span<const T>(static_cast<const T*>(data.data()), data.size()), tensor_spec, device))
            .to_device(device, tensor_memory_config);
    };

    switch (static_cast<nb::dlpack::dtype_code>(data_type.code)) {
        case nb::dlpack::dtype_code::Int:
            return impl.template operator()<int32_t>(tt::tt_metal::DataType::INT32);
            break;
        case nb::dlpack::dtype_code::UInt:
            return impl.template operator()<uint32_t>(tt::tt_metal::DataType::UINT32);
            break;
        case nb::dlpack::dtype_code::Float:
            return impl.template operator()<float>(tt::tt_metal::DataType::FLOAT32);
            break;
        case nb::dlpack::dtype_code::Bfloat:
            return impl.template operator()<bfloat16>(tt::tt_metal::DataType::BFLOAT16);
            break;
        case nb::dlpack::dtype_code::Complex: TT_THROW("Unsupported type: Complex"); break;
        case nb::dlpack::dtype_code::Bool: TT_THROW("Unsupported type: Bool"); break;
    }

    TT_THROW("Unsupported type: unknown");
}

void py_module_types(nb::module_& m) {
    nb::export_enum<ttnn::DataType>(m);
    nb::export_enum<GradMode>(m);
    nb::export_enum<PreferredPrecision>(m);
    nb::export_enum<ttml::ops::ReduceType>(m);
    nb::export_enum<RunMode>(m);

    nb::class_<GraphNode>(m, "GraphNode");
    nb::class_<Graph>(m, "Graph");
    nb::class_<ModuleBase>(m, "ModuleBase");
    nb::class_<modules::GPTMLP, ModuleBase>(m, "GPTMLP");
    nb::class_<modules::LinearLayer, ModuleBase>(m, "LinearLayer");
    nb::class_<modules::LlamaMLP, ModuleBase>(m, "LlamaMLP");
    nb::class_<Tensor>(m, "Tensor");
    nb::class_<AutocastTensor>(m, "AutocastTensor");
    nb::class_<AutoContext>(m, "AutoContext");
    nb::class_<optimizers::OptimizerBase>(m, "OptimizerBase");
    nb::class_<optimizers::SGDConfig>(m, "SGDConfig");
    nb::class_<optimizers::SGD, optimizers::OptimizerBase>(m, "SGD");

    m.def("create_linear_regression_model", &ttml::models::linear_regression::create);
    m.def("load_gpt2_model_from_safetensors", &ttml::models::gpt2::load_model_from_safetensors);
    m.def("cross_entropy_loss", &ttml::ops::cross_entropy_loss);
    m.def("mse_loss", &ttml::ops::mse_loss);
}

void py_module(nb::module_& m) {
    {
        auto py_graph_node = static_cast<nb::class_<GraphNode>>(m.attr("GraphNode"));
        py_graph_node.def(nb::init<>());
        py_graph_node.def_rw("grad_function", &GraphNode::grad_function);
    }

    {
        auto py_graph = static_cast<nb::class_<Graph>>(m.attr("Graph"));
        py_graph.def(nb::init<>());
        py_graph.def("get_edges", &Graph::get_edges);
        py_graph.def("get_graph_nodes", &Graph::get_graph_nodes);
        py_graph.def("add_node", &Graph::add_node);
    }

    {
        auto py_module_base = static_cast<nb::class_<ModuleBase>>(m.attr("ModuleBase"));
        py_module_base.def(nb::init<>());
        py_module_base.def(nb::init<const ModuleBase&>());
        py_module_base.def(nb::init<ModuleBase&&>());
        py_module_base.def("get_name", &ModuleBase::get_name);
        py_module_base.def("parameters", &ModuleBase::parameters);
        py_module_base.def("train", &ModuleBase::train);
        py_module_base.def("eval", &ModuleBase::eval);
        py_module_base.def("set_run_mode", &ModuleBase::set_run_mode);
        py_module_base.def("get_run_mode", &ModuleBase::get_run_mode);
    }

    {
        auto py_gpt_mlp = static_cast<nb::class_<modules::GPTMLP, ModuleBase>>(m.attr("GPTMLP"));
        py_gpt_mlp.def(nb::init<uint32_t, float>());
        py_gpt_mlp.def("__call__", &modules::GPTMLP::operator());
    }

    {
        auto py_linear_layer = static_cast<nb::class_<modules::LinearLayer, ModuleBase>>(m.attr("LinearLayer"));
        py_linear_layer.def(nb::init<uint32_t, uint32_t, bool>());
        py_linear_layer.def("__call__", &modules::LinearLayer::operator());
        py_linear_layer.def("get_weight", &modules::LinearLayer::get_weight);
        py_linear_layer.def("get_weight2", [](const modules::LinearLayer& layer) {
            auto const w = layer.get_weight();
            return make_numpy_tensor(w->get_value(PreferredPrecision::FULL));
        });
    }

    {
        auto py_llama_mlp = static_cast<nb::class_<modules::LlamaMLP, ModuleBase>>(m.attr("LlamaMLP"));
        py_llama_mlp.def(nb::init<uint32_t, std::optional<uint32_t>, float>());
        py_llama_mlp.def("__call__", &modules::LlamaMLP::operator());
    }

    {
        auto py_tensor = static_cast<nb::class_<Tensor>>(m.attr("Tensor"));
        py_tensor.def(nb::init<>());
        py_tensor.def(nb::init<const Tensor&>());
        py_tensor.def(nb::init<Tensor&&>());
        py_tensor.def(nb::init<const tt::tt_metal::Tensor&, bool>());
        py_tensor.def("set_value", &Tensor::set_value);
        py_tensor.def("set_grad", &Tensor::set_grad);
        py_tensor.def("set_node", &Tensor::set_node);
        py_tensor.def("clean_node", &Tensor::clean_node);
        py_tensor.def("add_grad", &Tensor::add_grad);
        py_tensor.def("set_requires_grad", &Tensor::set_requires_grad);
        py_tensor.def("get_value", &Tensor::get_value);
        py_tensor.def("get_grad", nb::overload_cast<>(&Tensor::get_grad, nb::const_));
        py_tensor.def("get_grad_rw", nb::overload_cast<>(&Tensor::get_grad));
        py_tensor.def("get_requires_grad", &Tensor::get_requires_grad);
        py_tensor.def("get_node", &Tensor::get_node);
        py_tensor.def("get_shape", &Tensor::get_shape);
        py_tensor.def("get_rank", &Tensor::get_rank);
        py_tensor.def("backward", &Tensor::backward);
        py_tensor.def("is_grad_initialized", &Tensor::is_grad_initialized);
        py_tensor.def_static(
            "from_numpy", [](nb::ndarray<> numpy_tensor) { return Tensor(make_metal_tensor(numpy_tensor)); });
        py_tensor.def("to_numpy", [](const Tensor& tensor) {
            return make_numpy_tensor(tensor.get_value(PreferredPrecision::FULL));
        });
        py_tensor.def("to_string", [](const Tensor& tensor) {
            return tensor.get_value(PreferredPrecision::FULL).write_to_string();
        });
        py_tensor.def("shape", [](const Tensor& tensor) {
            const tt::tt_metal::Shape& shape = tensor.get_value(PreferredPrecision::FULL).logical_shape();
            nb::list ret;
            for (auto it = shape.cbegin(); it != shape.cend(); ++it) {
                ret.append(*it);
            }
            return ret;
        });
        py_tensor.def("dtype", [](const Tensor& tensor) { return tensor.get_value(PreferredPrecision::FULL).dtype(); });
    }

    {
        auto py_autocast_tensor = static_cast<nb::class_<AutocastTensor>>(m.attr("AutocastTensor"));
        py_autocast_tensor.def(nb::init<>());
        // py_autocast_tensor.def(nb::init<const<ScrollWheelDown>tt::tt_metal::Tensor&>());
        py_autocast_tensor.def(nb::init<const AutocastTensor&>());
        py_autocast_tensor.def(nb::init<AutocastTensor&&>());
        py_autocast_tensor.def("set_tensor", &AutocastTensor::set_tensor);
        py_autocast_tensor.def("get_tensor", &AutocastTensor::get_tensor);
    }

    {
        auto py_auto_context = static_cast<nb::class_<AutoContext>>(m.attr("AutoContext"));
        py_auto_context.def_static("get_instance", &AutoContext::get_instance);
        py_auto_context.def("get_generator", &AutoContext::get_generator);
        py_auto_context.def("set_generator", &AutoContext::set_generator);
        py_auto_context.def("set_seed", &AutoContext::set_seed);
        py_auto_context.def("get_seed", &AutoContext::get_seed);
        py_auto_context.def("add_backward_node", &AutoContext::add_backward_node);
        py_auto_context.def("reset_graph", &AutoContext::reset_graph);
        py_auto_context.def("set_gradient_mode", &AutoContext::set_gradient_mode);
        py_auto_context.def("open_device", &AutoContext::open_device);
        py_auto_context.def("close_device", &AutoContext::close_device);
        py_auto_context.def("get_device", &AutoContext::get_device);
        // TODO: argv's char** not supported
        // py_auto_context.def("initialize_distributed_context", &AutoContext::initialize_distributed_context);
        py_auto_context.def("initialize_distributed_context", [](AutoContext& auto_context, nb::args args) {
            const auto argc = args.size();
            std::vector<const char*> argv(argc);

            for (const auto& arg : args) {
                argv.push_back(nb::str(arg).c_str());
            }
            argv.push_back(nullptr);

            auto_context.initialize_distributed_context(argc, const_cast<char**>(argv.data()));
        });
        py_auto_context.def("get_distributed_context", &AutoContext::get_distributed_context);
        py_auto_context.def("get_profiler", &AutoContext::get_profiler);
        py_auto_context.def("close_profiler", &AutoContext::close_profiler);
        py_auto_context.def("get_ccl_resources", &AutoContext::get_ccl_resources);
    }

    {
        auto py_sgd_config = static_cast<nb::class_<optimizers::SGDConfig>>(m.attr("SGDConfig"));
        py_sgd_config.def_static(
            "make", [](float lr, float momentum, float dampening, float weight_decay, bool nesterov) {
                return optimizers::SGDConfig{
                    .lr = lr,
                    .momentum = momentum,
                    .dampening = dampening,
                    .weight_decay = weight_decay,
                    .nesterov = nesterov};
            });

        auto py_sgd = static_cast<nb::class_<optimizers::SGD, optimizers::OptimizerBase>>(m.attr("SGD"));
        py_sgd.def(nb::init<serialization::NamedParameters, const optimizers::SGDConfig&>());
        py_sgd.def("zero_grad", &optimizers::SGD::zero_grad);
        py_sgd.def("step", &optimizers::SGD::step);
        py_sgd.def("get_state_dict", &optimizers::SGD::get_state_dict);
        py_sgd.def("get_state_dict", &optimizers::SGD::get_state_dict);
    }
}

}  // namespace ttml::autograd
