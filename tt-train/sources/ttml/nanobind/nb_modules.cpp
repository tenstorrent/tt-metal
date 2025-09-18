// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "models/base_transformer.hpp"
#include "models/gpt2.hpp"
#include "models/linear_regression.hpp"
#include "models/llama.hpp"
#include "models/mlp.hpp"
#include "modules/linear_module.hpp"
#include "modules/module_base.hpp"
#include "modules/multi_layer_perceptron.hpp"
#include "nb_export_enum.hpp"
#include "nb_fwd.hpp"
#include "nb_util.hpp"

namespace ttml::modules {

void py_module_types(nb::module_& m) {
    nb::export_enum<RunMode>(m);
    nb::export_enum<models::common::transformer::RunnerType>(m);
    nb::export_enum<models::common::transformer::WeightTyingType>(m);

    nb::export_enum<models::gpt2::PositionalEmbeddingType>(m);

    nb::class_<ModuleBase>(m, "ModuleBase");
    nb::class_<models::BaseTransformer, ModuleBase>(m, "BaseTransformer");

    nb::class_<models::gpt2::TransformerConfig>(m, "GPT2TransformerConfig");
    nb::class_<models::gpt2::Transformer>(m, "GPT2Transformer");

    nb::class_<LinearLayer, ModuleBase>(m, "LinearLayer");

    nb::class_<models::llama::LlamaConfig>(m, "LlamaConfig");
    nb::class_<models::llama::Llama, models::BaseTransformer>(m, "Llama");

    nb::class_<MultiLayerPerceptronParameters>(m, "MultiLayerPerceptronParameters");
    nb::class_<MultiLayerPerceptron, ModuleBase>(m, "MultiLayerPerceptron");
}

void py_module(nb::module_& m) {
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
        auto py_base_transformer =
            static_cast<nb::class_<models::BaseTransformer, ModuleBase>>(m.attr("BaseTransformer"));
        py_base_transformer.def("load_from_safetensors", &models::BaseTransformer::load_from_safetensors);
    }

    {
        auto py_gpt2_transformer_config =
            static_cast<nb::class_<models::gpt2::TransformerConfig>>(m.attr("GPT2TransformerConfig"));
        py_gpt2_transformer_config.def(
            nb::init<
                uint32_t,
                uint32_t,
                float,
                uint32_t,
                uint32_t,
                uint32_t,
                models::gpt2::RunnerType,
                models::gpt2::WeightTyingType,
                models::gpt2::PositionalEmbeddingType,
                bool>(),
            nb::arg("num_heads") = 6,
            nb::arg("embedding_dim") = 384,
            nb::arg("dropout_prob") = 0.2F,
            nb::arg("num_blocks") = 32,
            nb::arg("vocab_size") = 256,
            nb::arg("max_sequence_length") = 256,
            nb::arg("runner_type") = models::gpt2::RunnerType::Default,
            nb::arg("weight_tying") = models::gpt2::WeightTyingType::Disabled,
            nb::arg("positional_embedding_type") = models::gpt2::PositionalEmbeddingType::Trainable,
            nb::arg("use_composite_layernorm") = false);
        auto py_gpt2 =
            static_cast<nb::class_<models::gpt2::Transformer, models::BaseTransformer>>(m.attr("GPT2Transformer"));
        py_gpt2.def(nb::init<const models::gpt2::TransformerConfig&>());
        py_gpt2.def_static(
            "create", [](const models::gpt2::TransformerConfig& config) { return models::gpt2::create(config); });
    }

    {
        auto py_linear_layer = static_cast<nb::class_<LinearLayer, ModuleBase>>(m.attr("LinearLayer"));
        py_linear_layer.def(nb::init<uint32_t, uint32_t, bool>());
        py_linear_layer.def("__call__", &LinearLayer::operator());
        py_linear_layer.def("get_weight", &LinearLayer::get_weight);
        py_linear_layer.def("get_weight_numpy", [](const LinearLayer& layer) {
            auto const w = layer.get_weight();
            return make_numpy_tensor(w->get_value(autograd::PreferredPrecision::FULL));
        });
    }

    {
        auto py_llama_config = static_cast<nb::class_<models::llama::LlamaConfig>>(m.attr("LlamaConfig"));
        py_llama_config.def(
            nb::init<
                uint32_t,
                uint32_t,
                uint32_t,
                std::optional<uint32_t>,
                float,
                float,
                uint32_t,
                uint32_t,
                uint32_t,
                models::llama::RunnerType,
                models::llama::WeightTyingType,
                float,
                float,
                float,
                uint32_t>(),
            nb::arg("num_heads") = 6U,
            nb::arg("num_groups") = 3U,
            nb::arg("embedding_dim") = 384U,
            nb::arg("intermediate_dim") = std::nullopt,
            nb::arg("dropout_prop") = 0.0F,
            nb::arg("theta") = 10000.0F,
            nb::arg("num_blocks") = 6U,
            nb::arg("vocab_size") = 256U,
            nb::arg("max_sequence_length") = 256U,
            nb::arg("runner_type") = models::llama::RunnerType::Default,
            nb::arg("weight_tying") = models::llama::WeightTyingType::Disabled,
            nb::arg("scaling_factor") = 0.0F,
            nb::arg("high_freq_factor") = 4.0F,
            nb::arg("low_freq_factor") = 0.0F,
            nb::arg("original_context_length") = 0U);

        auto py_llama = static_cast<nb::class_<models::llama::LlamaConfig>>(m.attr("Llama"));
        py_llama.def(nb::init<models::llama::LlamaConfig>());
        py_llama.def_static(
            "create", [](const models::llama::LlamaConfig& config) { return models::llama::create(config); });
    }

    {
        auto py_mlp_params =
            static_cast<nb::class_<MultiLayerPerceptronParameters>>(m.attr("MultiLayerPerceptronParameters"));
        py_mlp_params.def(
            "create",
            [](uint32_t input_features, const std::vector<uint32_t>& hidden_features, uint32_t output_features) {
                return MultiLayerPerceptronParameters{
                    .input_features = input_features,
                    .hidden_features = hidden_features,
                    .output_features = output_features};
            });

        auto py_mlp = static_cast<nb::class_<MultiLayerPerceptron, ModuleBase>>(m.attr("MultiLayerPerceptron"));
        py_mlp.def(nb::init<const MultiLayerPerceptronParameters&>());
        py_mlp.def_static("create", [](const modules::MultiLayerPerceptronParameters& config) {
            return models::mlp::create(config);
        });
    }

    m.def("create_linear_regression_model", &models::linear_regression::create);
    m.def("load_gpt2_model_from_safetensors", &models::gpt2::load_model_from_safetensors);
    m.def("load_gpt2_model_from_safetensors", &models::gpt2::load_model_from_safetensors);
}

}  // namespace ttml::modules
