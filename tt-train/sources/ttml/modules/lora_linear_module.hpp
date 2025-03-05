#pragma once
#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"

namespace ttml::modules {

// LoRaConfig represents the configuration parameters for LoRA fine-tuning,
// similar to what Hugging Face provides in their implementations.
// but without
struct LoRaConfig {
    // The rank (r) determines the size of the low-rank matrices inserted into target layers.
    // A smaller rank reduces the number of additional parameters but may limit the adaptation capacity.
    int r = 32;

    // The scaling factor (alpha) controls the magnitude of the low-rank updates.
    // It balances the impact of the LoRA modifications relative to the original weights.
    float alpha = 1.0F;

    // Dropout rate applied to the LoRA layers.
    // Introducing dropout can help regularize the adaptation and prevent overfitting.
    float dropout = 0.0;

    // Bias configuration specifies how biases are handled during fine-tuning.
    bool is_bias_trainable = false;
};

class LoRALinearLayer : public autograd::ModuleBase {
private:
    autograd::TensorPtr m_weight;
    autograd::TensorPtr m_bias;

    void initialize_tensors(uint32_t in_features, uint32_t out_features, bool has_bias = true);
    void register_tensors();

public:
    LoRALinearLayer(uint32_t in_features, uint32_t out_features, bool has_bias = true);
    LoRALinearLayer(const autograd::TensorPtr& weight, const autograd::TensorPtr& bias);
    LoRALinearLayer(const autograd::TensorPtr& weight, bool has_bias = true);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& tensor) override;
};

}  // namespace ttml::modules
