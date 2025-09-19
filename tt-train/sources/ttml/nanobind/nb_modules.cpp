// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>

#include <ttnn/operations/experimental/dropout/dropout.hpp>

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
        auto py_gpt2_transformer_config_experimental =
            static_cast<nb::class_<models::gpt2::TransformerConfig::Experimental>>(
                m.attr("GPT2TransformerConfigExperimental"));
        py_gpt2_transformer_config_experimental.def(nb::init<>());
        py_gpt2_transformer_config_experimental.def_rw(
            "use_composite_layernorm", &models::gpt2::TransformerConfig::Experimental::use_composite_layernorm);

        auto py_gpt2_transformer_config =
            static_cast<nb::class_<models::gpt2::TransformerConfig>>(m.attr("GPT2TransformerConfig"));
        py_gpt2_transformer_config.def(nb::init<>());
        py_gpt2_transformer_config.def_rw("num_heads", &models::gpt2::TransformerConfig::num_heads);
        py_gpt2_transformer_config.def_rw("embedding_dim", &models::gpt2::TransformerConfig::embedding_dim);
        py_gpt2_transformer_config.def_rw("dropout_prob", &models::gpt2::TransformerConfig::dropout_prob);
        py_gpt2_transformer_config.def_rw("num_blocks", &models::gpt2::TransformerConfig::num_blocks);
        py_gpt2_transformer_config.def_rw("vocab_size", &models::gpt2::TransformerConfig::vocab_size);
        py_gpt2_transformer_config.def_rw("max_sequence_length", &models::gpt2::TransformerConfig::max_sequence_length);
        py_gpt2_transformer_config.def_rw("runner_type", &models::gpt2::TransformerConfig::runner_type);
        py_gpt2_transformer_config.def_rw("weight_tying", &models::gpt2::TransformerConfig::weight_tying);
        py_gpt2_transformer_config.def_rw(
            "positional_embedding_type", &models::gpt2::TransformerConfig::positional_embedding_type);
        py_gpt2_transformer_config.def_rw("experimental", &models::gpt2::TransformerConfig::experimental);

        auto py_gpt2 =
            static_cast<nb::class_<models::gpt2::Transformer, models::BaseTransformer>>(m.attr("GPT2Transformer"));
        py_gpt2.def(nb::init<const models::gpt2::TransformerConfig&>());
        py_gpt2.def("__call__", &models::gpt2::Transformer::operator());
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
        py_llama_config.def(nb::init<>());
        py_llama_config.def_rw("num_heads", &models::llama::LlamaConfig::num_heads);
        py_llama_config.def_rw("num_groups", &models::llama::LlamaConfig::num_groups);
        py_llama_config.def_rw("embedding_dim", &models::llama::LlamaConfig::embedding_dim);
        py_llama_config.def_rw("intermediate_dim", &models::llama::LlamaConfig::intermediate_dim);
        py_llama_config.def_rw("dropout_prob", &models::llama::LlamaConfig::dropout_prob);
        py_llama_config.def_rw("theta", &models::llama::LlamaConfig::theta);
        py_llama_config.def_rw("num_blocks", &models::llama::LlamaConfig::num_blocks);
        py_llama_config.def_rw("vocab_size", &models::llama::LlamaConfig::vocab_size);
        py_llama_config.def_rw("max_sequence_length", &models::llama::LlamaConfig::max_sequence_length);
        py_llama_config.def_rw("runner_type", &models::llama::LlamaConfig::runner_type);
        py_llama_config.def_rw("weight_tying", &models::llama::LlamaConfig::weight_tying);
        py_llama_config.def_rw("scaling_factor", &models::llama::LlamaConfig::scaling_factor);
        py_llama_config.def_rw("high_freq_factor", &models::llama::LlamaConfig::high_freq_factor);
        py_llama_config.def_rw("low_freq_factor", &models::llama::LlamaConfig::low_freq_factor);

        auto py_llama = static_cast<nb::class_<models::llama::Llama>>(m.attr("Llama"));
        py_llama.def(nb::init<models::llama::LlamaConfig>());
        py_llama.def("__call__", &models::llama::Llama::operator());
        py_llama.def_static(
            "create", [](const models::llama::LlamaConfig& config) { return models::llama::create(config); });
    }

    {
        auto py_mlp_params =
            static_cast<nb::class_<MultiLayerPerceptronParameters>>(m.attr("MultiLayerPerceptronParameters"));
        py_mlp_params.def(nb::init<>());
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
        py_mlp.def("__call__", &modules::MultiLayerPerceptron::operator());
        py_mlp.def_static("create", [](const modules::MultiLayerPerceptronParameters& config) {
            return models::mlp::create(config);
        });
    }

    m.def("create_linear_regression_model", &models::linear_regression::create);
}

}  // namespace ttml::modules
