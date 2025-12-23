// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <optional>

#include "models/base_transformer.hpp"
#include "models/common/transformer_common.hpp"
#include "models/distributed/gpt2.hpp"
#include "models/distributed/llama.hpp"
#include "models/distributed/pipeline_parallel_llama.hpp"
#include "models/gpt2.hpp"
#include "models/linear_regression.hpp"
#include "models/llama.hpp"
#include "models/mlp.hpp"
#include "modules/module_base.hpp"
#include "modules/multi_layer_perceptron.hpp"
#include "nb_export_enum.hpp"
#include "nb_fwd.hpp"
#include "nb_modules.hpp"

namespace ttml::nanobind::models {
using namespace ttml::models;

void py_module_types(nb::module_& m, nb::module_& m_modules) {
    ttml::nanobind::modules::py_module_types(m_modules);

    ttml::nanobind::util::export_enum<models::common::transformer::RunnerType>(m);
    ttml::nanobind::util::export_enum<models::common::transformer::WeightTyingType>(m);

    // Export KvCacheConfig class
    {
        auto py_kv_cache_config = nb::class_<models::common::transformer::KvCacheConfig>(m, "KvCacheConfig");
        py_kv_cache_config.def(
            nb::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>(),
            nb::arg("num_layers"),
            nb::arg("batch_size"),
            nb::arg("num_groups"),
            nb::arg("max_seq_len"),
            nb::arg("head_dim"),
            "KV Cache configuration");
        py_kv_cache_config.def_rw("num_layers", &models::common::transformer::KvCacheConfig::num_layers);
        py_kv_cache_config.def_rw("batch_size", &models::common::transformer::KvCacheConfig::batch_size);
        py_kv_cache_config.def_rw("num_groups", &models::common::transformer::KvCacheConfig::num_groups);
        py_kv_cache_config.def_rw("max_seq_len", &models::common::transformer::KvCacheConfig::max_seq_len);
        py_kv_cache_config.def_rw("head_dim", &models::common::transformer::KvCacheConfig::head_dim);
    }

    // Export KvCache class
    {
        auto py_kv_cache = nb::class_<models::common::transformer::KvCache>(m, "KvCache");
        py_kv_cache.def(
            nb::init<const models::common::transformer::KvCacheConfig&>(),
            nb::arg("config"),
            "Construct KV cache from config");
        py_kv_cache.def(
            nb::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>(),
            nb::arg("num_layers"),
            nb::arg("batch_size"),
            nb::arg("num_groups"),
            nb::arg("max_seq_len"),
            nb::arg("head_dim"),
            "Construct KV cache (convenience constructor)");
        py_kv_cache.def(
            "update",
            &models::common::transformer::KvCache::update,
            nb::arg("layer_idx"),
            nb::arg("key_states"),
            nb::arg("value_states"),
            nb::arg("new_tokens"),
            "Update KV cache with new key and value states");
        py_kv_cache.def(
            "get_k_cache",
            &models::common::transformer::KvCache::get_k_cache,
            nb::arg("layer_idx"),
            "Get K cache tensor for a layer");
        py_kv_cache.def(
            "get_v_cache",
            &models::common::transformer::KvCache::get_v_cache,
            nb::arg("layer_idx"),
            "Get V cache tensor for a layer");
        py_kv_cache.def(
            "get_cache_position",
            &models::common::transformer::KvCache::get_cache_position,
            "Get current cache position");
        py_kv_cache.def("reset", &models::common::transformer::KvCache::reset, "Reset cache position for new sequence");
        py_kv_cache.def("clear", &models::common::transformer::KvCache::clear, "Clear all cache data");
        py_kv_cache.def("num_layers", &models::common::transformer::KvCache::num_layers, "Get number of layers");
        py_kv_cache.def("empty", &models::common::transformer::KvCache::empty, "Check if cache is empty");
    }

    nb::class_<models::BaseTransformer, ttml::modules::ModuleBase>(m, "BaseTransformer");

    {
        auto py_gpt2_module = m.def_submodule("gpt2");
        ttml::nanobind::util::export_enum<models::gpt2::PositionalEmbeddingType>(py_gpt2_module);
        nb::class_<models::gpt2::TransformerConfig::Experimental>(py_gpt2_module, "GPT2TransformerConfigExperimental");
        nb::class_<models::gpt2::TransformerConfig>(py_gpt2_module, "GPT2TransformerConfig");
        nb::class_<models::gpt2::Transformer, models::BaseTransformer>(py_gpt2_module, "GPT2Transformer");
    }

    m.def_submodule("linear_regression");

    // Distributed models: register classes so return types can be wrapped
    m.def_submodule("distributed");
    {
        auto m_distributed = static_cast<nb::module_>(m.attr("distributed"));
        auto m_distributed_gpt2 = m_distributed.def_submodule("gpt2");
        nb::class_<ttml::models::distributed::gpt2::DistributedTransformer, models::BaseTransformer>(
            m_distributed_gpt2, "DistributedGPT2Transformer");

        auto m_distributed_llama = m_distributed.def_submodule("llama");
        nb::class_<ttml::models::distributed::llama::DistributedLlama, models::BaseTransformer>(
            m_distributed_llama, "DistributedLlama");

        auto m_distributed_pp = m_distributed.def_submodule("pipeline_parallel");
        auto m_distributed_pp_llama = m_distributed_pp.def_submodule("llama");
        nb::class_<ttml::models::distributed::pipeline_parallel_llama::PipelineParallelLlama, models::BaseTransformer>(
            m_distributed_pp_llama, "PipelineParallelLlama");
        nb::class_<ttml::models::distributed::pipeline_parallel_llama::PipelineParallelConfig>(
            m_distributed_pp_llama, "PipelineParallelConfig");
    }

    {
        auto py_llama_module = m.def_submodule("llama");
        nb::class_<models::llama::LlamaConfig>(py_llama_module, "LlamaConfig");
        nb::class_<models::llama::Llama, models::BaseTransformer>(py_llama_module, "Llama");
        py_llama_module.def("create_llama_model", [](const models::llama::LlamaConfig& config) {
            return models::llama::create(config);
        });
    }

    {
        auto py_mlp_module = m.def_submodule("mlp");
        nb::class_<ttml::modules::MultiLayerPerceptronParameters>(py_mlp_module, "MultiLayerPerceptronParameters");
        nb::class_<ttml::modules::MultiLayerPerceptron, ttml::modules::ModuleBase>(
            py_mlp_module, "MultiLayerPerceptron");
    }
}

void py_module(nb::module_& m, nb::module_& m_modules) {
    ttml::nanobind::modules::py_module(m_modules);

    {
        auto py_base_transformer =
            static_cast<nb::class_<models::BaseTransformer, ttml::modules::ModuleBase>>(m.attr("BaseTransformer"));
        py_base_transformer.def("load_from_safetensors", &models::BaseTransformer::load_from_safetensors);
    }

    {
        auto py_gpt2_module = static_cast<nb::module_>(m.attr("gpt2"));
        py_gpt2_module.def(
            "create_gpt2_model",
            [](const models::gpt2::TransformerConfig& config) { return models::gpt2::create(config); },
            "Create GPT2 model");

        auto py_gpt2_transformer_config_experimental =
            static_cast<nb::class_<models::gpt2::TransformerConfig::Experimental>>(
                py_gpt2_module.attr("GPT2TransformerConfigExperimental"));
        py_gpt2_transformer_config_experimental.def(nb::init<>());
        py_gpt2_transformer_config_experimental.def_rw(
            "use_composite_layernorm",
            &models::gpt2::TransformerConfig::Experimental::use_composite_layernorm,
            "Use composite layernorm");

        auto py_gpt2_transformer_config =
            static_cast<nb::class_<models::gpt2::TransformerConfig>>(py_gpt2_module.attr("GPT2TransformerConfig"));
        py_gpt2_transformer_config.def(nb::init<>());
        py_gpt2_transformer_config.def_rw("num_heads", &models::gpt2::TransformerConfig::num_heads, "Number of heads");
        py_gpt2_transformer_config.def_rw(
            "embedding_dim", &models::gpt2::TransformerConfig::embedding_dim, "Embedding dimensions");
        py_gpt2_transformer_config.def_rw(
            "dropout_prob", &models::gpt2::TransformerConfig::dropout_prob, "Dropout probability");
        py_gpt2_transformer_config.def_rw(
            "num_blocks", &models::gpt2::TransformerConfig::num_blocks, "Number of blocks");
        py_gpt2_transformer_config.def_rw(
            "vocab_size", &models::gpt2::TransformerConfig::vocab_size, "Vocabulary size");
        py_gpt2_transformer_config.def_rw(
            "max_sequence_length", &models::gpt2::TransformerConfig::max_sequence_length, "Max sequence length");
        py_gpt2_transformer_config.def_rw("runner_type", &models::gpt2::TransformerConfig::runner_type, "Runner type");
        py_gpt2_transformer_config.def_rw(
            "weight_tying", &models::gpt2::TransformerConfig::weight_tying, "Weight tying");
        py_gpt2_transformer_config.def_rw(
            "positional_embedding_type",
            &models::gpt2::TransformerConfig::positional_embedding_type,
            "Positional embedding type");
        py_gpt2_transformer_config.def_rw(
            "experimental", &models::gpt2::TransformerConfig::experimental, "Experimental config options");

        auto py_gpt2 = static_cast<nb::class_<models::gpt2::Transformer, models::BaseTransformer>>(
            py_gpt2_module.attr("GPT2Transformer"));
        py_gpt2.def(nb::init<const models::gpt2::TransformerConfig&>());
    }

    {
        auto py_linear_regression_module = static_cast<nb::module_>(m.attr("linear_regression"));
        py_linear_regression_module.def(
            "create_linear_regression_model", &models::linear_regression::create, "Create linear regression model");
    }

    {
        // Distributed creators
        auto py_distributed = static_cast<nb::module_>(m.attr("distributed"));
        auto py_distributed_gpt2 = py_distributed.def_submodule("gpt2");
        py_distributed_gpt2.def(
            "create_gpt2_model",
            [](const models::gpt2::TransformerConfig& config) {
                return ttml::models::distributed::gpt2::create(config);
            },
            "Create GPT2 model");

        auto py_distributed_llama = py_distributed.def_submodule("llama");
        py_distributed_llama.def(
            "create_llama_model",
            [](const models::llama::LlamaConfig& config) { return ttml::models::distributed::llama::create(config); },
            "Create Llama model");

        auto py_distributed_pp = py_distributed.def_submodule("pipeline_parallel");
        auto py_distributed_pp_llama = py_distributed_pp.def_submodule("llama");
        py_distributed_pp_llama.def(
            "create_llama_model",
            [](const models::llama::LlamaConfig& config,
               const ttml::models::distributed::pipeline_parallel_llama::PipelineParallelConfig& pp_config,
               bool is_tensor_parallel) {
                return ttml::models::distributed::pipeline_parallel_llama::create(
                    config, pp_config, is_tensor_parallel);
            },
            nb::arg("config"),
            nb::arg("pipeline_parallel_config"),
            nb::arg("is_tensor_parallel") = false,
            "Create Pipeline Parallel Llama model");
        auto py_distributed_pp_llama_cfg =
            static_cast<nb::class_<ttml::models::distributed::pipeline_parallel_llama::PipelineParallelConfig>>(
                py_distributed_pp_llama.attr("PipelineParallelConfig"));
        py_distributed_pp_llama_cfg.def(nb::init<>());
        py_distributed_pp_llama_cfg.def_rw(
            "num_blocks",
            &ttml::models::distributed::pipeline_parallel_llama::PipelineParallelConfig::num_blocks,
            "Number of blocks");
        py_distributed_pp_llama_cfg.def_rw(
            "blocks_per_rank",
            &ttml::models::distributed::pipeline_parallel_llama::PipelineParallelConfig::blocks_per_rank,
            "Blocks per rank");
    }

    {
        auto py_llama_module = static_cast<nb::module_>(m.attr("llama"));

        auto py_llama_config = static_cast<nb::class_<models::llama::LlamaConfig>>(py_llama_module.attr("LlamaConfig"));
        py_llama_config.def(nb::init<>());
        py_llama_config.def_rw("num_heads", &models::llama::LlamaConfig::num_heads, "Number of heads");
        py_llama_config.def_rw("num_groups", &models::llama::LlamaConfig::num_groups, "Number of groups");
        py_llama_config.def_rw("embedding_dim", &models::llama::LlamaConfig::embedding_dim, "Embedding dimensions");
        py_llama_config.def_rw(
            "intermediate_dim", &models::llama::LlamaConfig::intermediate_dim, "Intermediate dimensions");
        py_llama_config.def_rw("dropout_prob", &models::llama::LlamaConfig::dropout_prob, "Dropout probability");
        py_llama_config.def_rw("theta", &models::llama::LlamaConfig::theta, "Theta");
        py_llama_config.def_rw("num_blocks", &models::llama::LlamaConfig::num_blocks, "Number of blocks");
        py_llama_config.def_rw("vocab_size", &models::llama::LlamaConfig::vocab_size, "Vocabulary size");
        py_llama_config.def_rw(
            "max_sequence_length", &models::llama::LlamaConfig::max_sequence_length, "Max sequence length");
        py_llama_config.def_rw("runner_type", &models::llama::LlamaConfig::runner_type, "Runner type");
        py_llama_config.def_rw("weight_tying", &models::llama::LlamaConfig::weight_tying, "Weight tying");
        py_llama_config.def_rw("scaling_factor", &models::llama::LlamaConfig::scaling_factor, "Scaling factor");
        py_llama_config.def_rw(
            "high_freq_factor", &models::llama::LlamaConfig::high_freq_factor, "High frequency factor");
        py_llama_config.def_rw("low_freq_factor", &models::llama::LlamaConfig::low_freq_factor, "Low frequency factor");
        py_llama_config.def_rw(
            "original_context_length", &models::llama::LlamaConfig::original_context_length, "Original context length");

        auto py_llama = static_cast<nb::class_<models::llama::Llama>>(py_llama_module.attr("Llama"));
        py_llama.def(nb::init<models::llama::LlamaConfig>());

        // Overload 1: Without KV cache
        py_llama.def(
            "__call__",
            [](models::llama::Llama& self,
               const ttml::autograd::TensorPtr& tensor,
               const ttml::autograd::TensorPtr& mask) { return self(tensor, mask); },
            nb::arg("tensor"),
            nb::arg("mask"),
            "Model forward pass without KV cache.");

        // Overload 2: With KV cache and new_tokens
        py_llama.def(
            "__call__",
            [](models::llama::Llama& self,
               const ttml::autograd::TensorPtr& tensor,
               const ttml::autograd::TensorPtr& mask,
               std::shared_ptr<models::common::transformer::KvCache> kv_cache,
               uint32_t new_tokens) { return self(tensor, mask, kv_cache, new_tokens); },
            nb::arg("tensor"),
            nb::arg("mask"),
            nb::arg("kv_cache"),
            nb::arg("new_tokens"), /* Danik: number of new tokens to process (we must pad the sequence
            to a multiple of 32 for perfomance of models' training step due to an inefficient implementation of linear
            backward) */
            "Model forward pass with KV cache for inference.");
        py_llama.def(
            "get_original_vocab_size",
            &models::llama::Llama::get_original_vocab_size,
            "Get the original vocabulary size");
    }

    {
        auto py_mlp_module = static_cast<nb::module_>(m.attr("mlp"));
        py_mlp_module.def("create_mlp_model", [](const ttml::modules::MultiLayerPerceptronParameters& config) {
            return models::mlp::create(config);
        });

        auto py_mlp_params = static_cast<nb::class_<ttml::modules::MultiLayerPerceptronParameters>>(
            py_mlp_module.attr("MultiLayerPerceptronParameters"));
        py_mlp_params.def(nb::init<>());
        py_mlp_params.def_static(
            "create",
            [](uint32_t input_features, const std::vector<uint32_t>& hidden_features, uint32_t output_features) {
                return ttml::modules::MultiLayerPerceptronParameters{
                    .input_features = input_features,
                    .hidden_features = hidden_features,
                    .output_features = output_features};
            },
            nb::arg("input_features"),
            nb::arg("hidden_features"),
            nb::arg("output_features"),
            "Create multilayer perceptron parameters");
        py_mlp_params.def_rw("input_features", &ttml::modules::MultiLayerPerceptronParameters::input_features);
        py_mlp_params.def_rw("hidden_features", &ttml::modules::MultiLayerPerceptronParameters::hidden_features);
        py_mlp_params.def_rw("output_features", &ttml::modules::MultiLayerPerceptronParameters::output_features);

        auto py_mlp = static_cast<nb::class_<ttml::modules::MultiLayerPerceptron, ttml::modules::ModuleBase>>(
            py_mlp_module.attr("MultiLayerPerceptron"));
        py_mlp.def(nb::init<const ttml::modules::MultiLayerPerceptronParameters&>());
    }
}

}  // namespace ttml::nanobind::models
