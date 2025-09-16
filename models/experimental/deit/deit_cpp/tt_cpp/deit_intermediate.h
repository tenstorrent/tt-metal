#ifndef DEIT_CPP_TT_CPP_DEIT_INTERMEDIATE_H
#define DEIT_CPP_TT_CPP_DEIT_INTERMEDIATE_H

#include "deit_config.h"
#include "../helper_funcs.h"
#include <memory>
#include <unordered_map>
#include <string>
#include <optional>
#include <torch/torch.h>
#include <ttnn/tensor/tensor.hpp>


/**
 * TtDeiTIntermediate class - C++ implementation of DeiT intermediate layer
 * Equivalent to the Python TtDeiTIntermediate class
 * 
 * This layer applies a linear transformation followed by an activation function
 * (typically GELU) to expand the hidden dimension to intermediate_size.
 */
class TtDeiTIntermediate {
public:
    /**
     * Constructor
     * @param config DeiT configuration
     * @param device TTNN device pointer
     * @param state_dict Model parameters dictionary
     * @param base_address Base address for parameter loading
     */
    TtDeiTIntermediate(
        const DeiTConfig& config,
        std::shared_ptr<ttnn::MeshDevice> device,
        const std::unordered_map<std::string, torch::Tensor>& state_dict,
        const std::string& base_address = ""
    );

    /**
     * Forward pass
     * @param hidden_states Input hidden states tensor
     * @return Output tensor after dense transformation and activation
     */
    ttnn::Tensor forward(const ttnn::Tensor& hidden_states);

    /**
     * Destructor
     */
    ~TtDeiTIntermediate() = default;

private:
    /**
     * Apply activation function based on configuration
     * @param input Input tensor
     * @return Tensor after activation
     */
    ttnn::Tensor apply_activation(const ttnn::Tensor& input);

    // Configuration and device
    DeiTConfig config;
    std::shared_ptr<ttnn::MeshDevice> device;
    
    // Dense layer weights and bias
    ttnn::Tensor dense_weight;
    std::optional<ttnn::Tensor> dense_bias;
    
    // Activation function type
    std::string activation_type;
};


#endif // DEIT_CPP_TT_CPP_DEIT_INTERMEDIATE_H