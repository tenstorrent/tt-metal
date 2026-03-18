#include "deit_test_infra.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/functions.hpp"
#include <torch/torch.h>
#include <torch/script.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

namespace deit_inference {

// Helper to convert torch::Tensor to ttnn::Tensor
ttnn::Tensor from_torch_tensor(
    torch::Tensor t,
    ttnn::DataType dtype,
    ttnn::Layout layout,
    ttnn::MeshDevice* device = nullptr,
    std::optional<ttnn::MemoryConfig> memory_config = std::nullopt) {
    t = t.contiguous().cpu();

    std::vector<uint32_t> shape_vec;
    for (int i = 0; i < t.dim(); ++i) {
        shape_vec.push_back(t.size(i));
    }
    ttnn::Shape shape(shape_vec);

    size_t num_elements = t.numel();
    std::vector<bfloat16> data(num_elements);

    if (t.scalar_type() == torch::kFloat32) {
        float* ptr = t.data_ptr<float>();
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = bfloat16(ptr[i]);
        }
    } else if (t.scalar_type() == torch::kBFloat16) {
        auto* ptr = t.data_ptr<at::BFloat16>();
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = bfloat16(static_cast<float>(ptr[i]));
        }
    } else {
        auto t_float = t.to(torch::kFloat32);
        float* ptr = t_float.data_ptr<float>();
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = bfloat16(ptr[i]);
        }
    }

    auto host_buffer = tt::tt_metal::HostBuffer{data};
    auto tt_tensor = ttnn::Tensor(host_buffer, shape, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR);

    if (layout == ttnn::Layout::TILE) {
        tt_tensor = ttnn::to_layout(tt_tensor, ttnn::Layout::TILE, dtype, memory_config);
    } else {
        if (dtype != ttnn::DataType::BFLOAT16) {
            tt_tensor = ttnn::typecast(tt_tensor, dtype);
        }
    }
    if (device != nullptr) {
        tt_tensor = ttnn::to_device(tt_tensor, device, memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG));
    }

    return tt_tensor;
}

DeitTestInfra::DeitTestInfra(ttnn::MeshDevice* device, int batch_size, const std::string& model_name) :
    device_(device), batch_size_(batch_size) {
    // Initialize config
    update_model_config(config_, batch_size);

    // Load model weights
    std::string weights_path = model_name;
    std::unordered_map<std::string, torch::Tensor> state_dict;

    try {
        torch::jit::script::Module module = torch::jit::load(weights_path);
        for (const auto& param : module.named_parameters()) {
            std::string name = param.name;
            // Remove "model." prefix if present (artifact of tracing wrapper)
            if (name.rfind("model.", 0) == 0) {
                name = name.substr(6);
            }
            state_dict[name] = param.value;
        }
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model from " << weights_path << ": " << e.what() << std::endl;
        throw;
    }

    // --- Process weights for ttnn ---

    // 1. Special tokens (cls, distillation, position)
    auto process_special_token = [&](const std::string& name, bool is_position = false) {
        torch::Tensor t = state_dict[name];
        if (batch_size_ > 1) {
            std::vector<int64_t> expand_shape = t.sizes().vec();
            expand_shape[0] = batch_size_;
            t = t.expand(expand_shape).contiguous();
        }
        ttnn::DataType dtype = is_position ? ttnn::DataType::BFLOAT8_B : ttnn::DataType::BFLOAT16;
        ttnn::Layout layout = is_position ? ttnn::Layout::TILE : ttnn::Layout::ROW_MAJOR;
        return from_torch_tensor(t, dtype, layout, device_);
    };

    cls_token_ = process_special_token("deit.embeddings.cls_token");
    distillation_token_ = process_special_token("deit.embeddings.distillation_token");
    position_embeddings_ = process_special_token("deit.embeddings.position_embeddings", true);

    // 2. Linear weights helper
    // Transpose [out, in] -> [in, out] -> reshape [1, 1, in, out] for ttnn::linear
    auto to_ttnn_linear_weight = [&](torch::Tensor t) {
        // t is [out, in]
        auto t_transposed = t.t().contiguous();  // [in, out]
        auto t_reshaped = t_transposed.reshape({1, 1, t_transposed.size(0), t_transposed.size(1)});
        return from_torch_tensor(t_reshaped, ttnn::DataType::BFLOAT8_B, ttnn::Layout::TILE, device_);
    };

    auto to_ttnn_bias = [&](torch::Tensor t) {
        // t is [out] -> [1, 1, 1, out]
        auto t_reshaped = t.reshape({1, 1, 1, t.size(0)});
        return from_torch_tensor(t_reshaped, ttnn::DataType::BFLOAT8_B, ttnn::Layout::TILE, device_);
    };

    // 3. Patch Embeddings
    {
        // PyTorch: [192, 3, 16, 16]
        // Target: Compatible with folded input [N, 196, 1024] (16*16*4)
        // We need to pad channels 3->4, then flatten to [192, 1024], then transpose to [1024, 192]
        torch::Tensor w = state_dict["deit.embeddings.patch_embeddings.projection.weight"];
        torch::Tensor b = state_dict["deit.embeddings.patch_embeddings.projection.bias"];

        // Pad channel dim (1) from 3 to 4
        // PadFuncOptions padding is (left, right, top, bottom, front, back, ...) for last 3 dims?
        // Actually it's (W_left, W_right, H_top, H_bottom, C_front, C_back, ...) working backwards.
        // Input is [192, 3, 16, 16].
        // We want to pad dim 1 (channels).
        // 16, 16 are dims 2, 3.
        // Padding args: (0,0) for dim 3, (0,0) for dim 2, (0,1) for dim 1.
        w = torch::constant_pad_nd(w, {0, 0, 0, 0, 0, 1}, 0);

        // Now w is [192, 4, 16, 16]
        // Permute to [192, 16, 16, 4] to match NHWC folding?
        // Wait, input preprocessing: Permute NHWC -> Pad C -> Reshape (N, H, W/P, 4*P).
        // It seems complex to match exactly without running data.
        // But assuming the standard transformation:
        // We want a linear weight that maps the 1024-sized input vector to 192.
        // Input vector corresponds to a 16x16 patch with 4 channels.
        // We flatten the weight to [192, 1024].
        // But the order of 1024 elements must match the input folding order.
        // Input folding: NHWC -> Pad C -> Reshape.
        // So pixels are ordered by channel last?
        // Yes, permute(0, 2, 3, 1) makes it NHWC.
        // So we should permute weight to [192, 16, 16, 4] then flatten.
        w = w.permute({0, 2, 3, 1}).contiguous().reshape({192, -1});  // [192, 1024]

        parameters_["deit.embeddings.patch_embeddings.projection.weight"] = to_ttnn_linear_weight(w);
        parameters_["deit.embeddings.patch_embeddings.projection.bias"] = to_ttnn_bias(b);
    }

    // 4. Layers
    for (int i = 0; i < 12; ++i) {
        std::string prefix = "deit.encoder.layer." + std::to_string(i) + ".";

        // QKV
        auto qw = state_dict[prefix + "attention.attention.query.weight"];
        auto kw = state_dict[prefix + "attention.attention.key.weight"];
        auto vw = state_dict[prefix + "attention.attention.value.weight"];
        auto qb = state_dict[prefix + "attention.attention.query.bias"];
        auto kb = state_dict[prefix + "attention.attention.key.bias"];
        auto vb = state_dict[prefix + "attention.attention.value.bias"];

        // Combine Q, K, V weights with interleaved head layout to match Python:
        //   torch.cat([q.reshape(num_heads, head_size, -1),
        //              k.reshape(num_heads, head_size, -1),
        //              v.reshape(num_heads, head_size, -1)], dim=1)
        //        .reshape(hidden_size, -1)
        // This produces [Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, ...] layout
        // required by split_query_key_value_and_split_heads.
        int num_heads = 3;
        int head_size = 64;
        int hidden_size = num_heads * head_size * 3;
        auto qkv_w = torch::cat(
                         {qw.reshape({num_heads, head_size, -1}),
                          kw.reshape({num_heads, head_size, -1}),
                          vw.reshape({num_heads, head_size, -1})},
                         1)
                         .reshape({hidden_size, -1});  // [576, 192]
        auto qkv_b = torch::cat(
                         {qb.reshape({num_heads, head_size}),
                          kb.reshape({num_heads, head_size}),
                          vb.reshape({num_heads, head_size})},
                         1)
                         .reshape({hidden_size});  // [576]

        parameters_[prefix + "attention.attention.qkv.weight"] = to_ttnn_linear_weight(qkv_w);
        parameters_[prefix + "attention.attention.qkv.bias"] = to_ttnn_bias(qkv_b);

        // Output Dense
        parameters_[prefix + "attention.output.dense.weight"] =
            to_ttnn_linear_weight(state_dict[prefix + "attention.output.dense.weight"]);
        parameters_[prefix + "attention.output.dense.bias"] =
            to_ttnn_bias(state_dict[prefix + "attention.output.dense.bias"]);

        // MLP
        parameters_[prefix + "intermediate.dense.weight"] =
            to_ttnn_linear_weight(state_dict[prefix + "intermediate.dense.weight"]);
        parameters_[prefix + "intermediate.dense.bias"] = to_ttnn_bias(state_dict[prefix + "intermediate.dense.bias"]);
        parameters_[prefix + "output.dense.weight"] = to_ttnn_linear_weight(state_dict[prefix + "output.dense.weight"]);
        parameters_[prefix + "output.dense.bias"] = to_ttnn_bias(state_dict[prefix + "output.dense.bias"]);

        // LayerNorms
        auto ln_w_before = state_dict[prefix + "layernorm_before.weight"].reshape({1, 1, 1, 192});
        auto ln_b_before = state_dict[prefix + "layernorm_before.bias"].reshape({1, 1, 1, 192});
        parameters_[prefix + "layernorm_before.weight"] =
            from_torch_tensor(ln_w_before, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, device_);
        parameters_[prefix + "layernorm_before.bias"] =
            from_torch_tensor(ln_b_before, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, device_);

        auto ln_w_after = state_dict[prefix + "layernorm_after.weight"].reshape({1, 1, 1, 192});
        auto ln_b_after = state_dict[prefix + "layernorm_after.bias"].reshape({1, 1, 1, 192});
        parameters_[prefix + "layernorm_after.weight"] =
            from_torch_tensor(ln_w_after, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, device_);
        parameters_[prefix + "layernorm_after.bias"] =
            from_torch_tensor(ln_b_after, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, device_);
    }

    // 5. Final LN
    auto final_ln_w = state_dict["deit.layernorm.weight"].reshape({1, 1, 1, 192});
    auto final_ln_b = state_dict["deit.layernorm.bias"].reshape({1, 1, 1, 192});
    parameters_["deit.layernorm.weight"] =
        from_torch_tensor(final_ln_w, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, device_);
    parameters_["deit.layernorm.bias"] =
        from_torch_tensor(final_ln_b, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, device_);

    // 6. Classifiers
    // Need to pad 1000 -> 1152 (or aligned size)
    // 1152 is used in deit_inference.cpp
    auto process_classifier = [&](const std::string& name) {
        torch::Tensor w = state_dict[name + ".weight"];
        torch::Tensor b = state_dict[name + ".bias"];
        // Pad output dim (0) from 1000 to 1152
        int padding = 1152 - 1000;
        if (padding > 0) {
            w = torch::constant_pad_nd(
                w, {0, 0, 0, padding}, 0);  // Pad last dim (in) 0,0; second last (out) 0,padding?
            // w is [1000, 192]. Dim 0 is out.
            // constant_pad_nd operates from last dim.
            // {0, 0, 0, padding} -> dim 1 (192) +0,+0; dim 0 (1000) +0,+padding.
            b = torch::constant_pad_nd(b, {0, padding}, 0);
        }
        parameters_[name + ".weight"] = to_ttnn_linear_weight(w);
        parameters_[name + ".bias"] = to_ttnn_bias(b);
    };

    process_classifier("cls_classifier");
    process_classifier("distillation_classifier");

    // Setup head masks (attention masks)
    // Python: torch_attention_mask = torch.ones(self.config.num_hidden_layers, sequence_size, dtype=torch.float32)
    // Sequence size for attention mask usually matches the sequence length (198 for DeiT tiny)
    // Python code uses 224.
    // The C++ implementation pads the sequence to 224 (multiple of 32).
    int sequence_size = 224;

    // Create ones tensor
    // Shape: [batch, 1, 1, sequence_size]
    auto torch_mask = torch::ones({batch_size, 1, 1, sequence_size}, torch::dtype(torch::kFloat32));

    for (int i = 0; i < config_.num_layers; ++i) {
        auto tt_mask = from_torch_tensor(
            torch_mask, ttnn::DataType::BFLOAT8_B, ttnn::Layout::TILE, device_, ttnn::L1_MEMORY_CONFIG);
        head_masks_.push_back(tt_mask);
    }

    // Create synthetic data for initialization
    torch_pixel_values_ = torch::randn({batch_size, 3, 224, 224}, torch::dtype(torch::kFloat32));
}

std::pair<ttnn::Tensor, ttnn::MemoryConfig> DeitTestInfra::setup_l1_sharded_input(
    const std::optional<torch::Tensor>& torch_pixel_values) {
    torch::Tensor x_raw = torch_pixel_values.has_value() ? torch_pixel_values.value() : torch_pixel_values_;

    // NHWC permutation
    torch::Tensor x = x_raw.permute({0, 2, 3, 1});  // [N, C, H, W] -> [N, H, W, C]

    // Channel padding (3 -> 4)
    // x is [N, H, W, 3]
    // Pad last dim by 1.
    x = torch::constant_pad_nd(x, {0, 1}, 0);  // [N, H, W, 4]

    // Patching reshape
    // Python: x.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)
    int batch = x.size(0);
    int h = x.size(1);
    int w = x.size(2);
    int patch_size = 16;
    x = x.reshape({batch, h, w / patch_size, 4 * patch_size});

    // Convert to ttnn tensor (DRAM)
    ttnn::Tensor tt_inputs_host = from_torch_tensor(x, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR);

    // Shard spec configuration
    // Matches Python logic exactly:
    // shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(batch_size - 1, 3))})
    int N_dim = x.size(0);
    int H_dim = x.size(1);
    int W_dim = x.size(2);
    int C_dim = x.size(3);

    // Python: ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(batch_size - 1, 3))
    // Note: This grid (batch_size, 4) is larger than the compute grid (3, batch_size) used in the model.
    // However, we strictly follow the Python test infrastructure here as requested.
    ttnn::CoreRangeSet shard_grid({ttnn::CoreRange(ttnn::CoreCoord(0, 0), ttnn::CoreCoord(batch - 1, 3))});

    int n_cores = batch * 3;
    std::array<uint32_t, 2> shard_shape = {
        static_cast<uint32_t>((N_dim * H_dim * W_dim) / n_cores), static_cast<uint32_t>(C_dim)};

    tt::tt_metal::ShardSpec shard_spec(shard_grid, shard_shape, ttnn::ShardOrientation::ROW_MAJOR);
    ttnn::MemoryConfig input_mem_config(ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ttnn::BufferType::L1, shard_spec);

    return {tt_inputs_host, input_mem_config};
}

std::tuple<ttnn::Tensor, ttnn::MemoryConfig, ttnn::MemoryConfig> DeitTestInfra::setup_dram_sharded_input(
    const std::optional<torch::Tensor>& torch_input_tensor) {
    torch::Tensor input = torch_input_tensor.has_value() ? torch_input_tensor.value() : torch_pixel_values_;

    auto [tt_inputs_host, input_mem_config] = setup_l1_sharded_input(input);

    auto dram_grid_size = device_->dram_grid_size();
    ttnn::CoreRangeSet dram_grid(
        {ttnn::CoreRange(ttnn::CoreCoord(0, 0), ttnn::CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))});

    // DRAM shard spec
    // Python: [divup(volume // width, dram_grid.x), width]
    // width is shape[-1] (64)
    uint32_t width = tt_inputs_host.logical_shape()[-1];
    uint32_t volume = tt_inputs_host.logical_volume();
    uint32_t height = volume / width;
    uint32_t shard_height = (height + dram_grid_size.x - 1) / dram_grid_size.x;  // divup

    tt::tt_metal::ShardSpec dram_shard_spec(dram_grid, {shard_height, width}, ttnn::ShardOrientation::ROW_MAJOR);
    ttnn::MemoryConfig sharded_mem_config_DRAM(
        ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ttnn::BufferType::DRAM, dram_shard_spec);

    return {tt_inputs_host, sharded_mem_config_DRAM, input_mem_config};
}

ttnn::Tensor DeitTestInfra::run(const std::optional<ttnn::Tensor>& tt_input_tensor) {
    if (tt_input_tensor.has_value()) {
        input_tensor = tt_input_tensor.value();
    }

    output_tuple =
        deit(config_, input_tensor, head_masks_, cls_token_, distillation_token_, position_embeddings_, parameters_);

    // Unpack tuple
    logits = std::get<0>(output_tuple);
    cls_logits = std::get<1>(output_tuple);
    distillation_logits = std::get<2>(output_tuple);

    output_tensor = logits;
    return output_tensor;
}

std::shared_ptr<DeitTestInfra> create_test_infra(
    ttnn::MeshDevice* device, int batch_size, const std::string& model_name) {
    return std::make_shared<DeitTestInfra>(device, batch_size, model_name);
}

}  // namespace deit_inference
