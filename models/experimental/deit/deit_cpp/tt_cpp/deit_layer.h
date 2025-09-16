#ifndef DEIT_CPP_TT_CPP_DEIT_LAYER_H
#define DEIT_CPP_TT_CPP_DEIT_LAYER_H

#include "deit_config.h"
#include "deit_attention.h"
#include "deit_intermediate.h"
#include "deit_output.h"
#include "../helper_funcs.h"
#include <memory>
#include <unordered_map>
#include <string>
#include <optional>
#include <tuple>
#include <torch/torch.h>
#include <ttnn/tensor/tensor.hpp>


/**
 * TtDeiTLayer class - C++ implementation of DeiT layer (Block)
 * Equivalent to the Python TtDeiTLayer class
 * 
 * This layer combines attention, intermediate, and output modules with layer normalization.
 * It implements the transformer block with pre-norm architecture used in DeiT.
 */
class TtDeiTLayer {
public:
    /**
     * Constructor
     * @param config DeiT configuration
     * @param device TTNN device pointer
     * @param state_dict Model parameters dictionary
     * @param base_address Base address for parameter loading
     */
    TtDeiTLayer(
        const DeiTConfig& config,
        std::shared_ptr<ttnn::MeshDevice> device,
        const std::unordered_map<std::string, torch::Tensor>& state_dict,
        const std::string& base_address = ""
    );

    /**
     * Forward pass
     * @param hidden_states Input hidden states tensor
     * @param head_mask Optional attention head mask
     * @param output_attentions Whether to output attention weights
     * @return Tuple containing layer output and optionally attention weights
     */
    std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> forward(
        const ttnn::Tensor& hidden_states,
        const std::optional<ttnn::Tensor>& head_mask = std::nullopt,
        bool output_attentions = false
    );

    /**
     * Destructor
     */
    ~TtDeiTLayer() = default;

private:
    // Configuration and device
    DeiTConfig config;
    std::shared_ptr<ttnn::MeshDevice> device;
    
    // Sub-modules
    std::unique_ptr<TtDeiTAttention> attention;
    std::unique_ptr<TtDeiTIntermediate> intermediate;
    std::unique_ptr<TtDeiTOutput> output;
    
    // Layer normalization parameters
    ttnn::Tensor layernorm_before_weight;
    ttnn::Tensor layernorm_before_bias;
    ttnn::Tensor layernorm_after_weight;
    ttnn::Tensor layernorm_after_bias;
    
    /**
     * Apply layer normalization
     * @param input Input tensor
     * @param weight Normalization weight
     * @param bias Normalization bias
     * @param eps Layer norm epsilon
     * @return Normalized tensor
     */
    ttnn::Tensor apply_layernorm(
        const ttnn::Tensor& input,
        const ttnn::Tensor& weight,
        const ttnn::Tensor& bias,
        float eps
    );
};


#endif // DEIT_CPP_TT_CPP_DEIT_LAYER_H