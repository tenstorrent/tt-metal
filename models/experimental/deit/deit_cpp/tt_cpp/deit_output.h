#ifndef DEIT_CPP_TT_CPP_DEIT_OUTPUT_H
#define DEIT_CPP_TT_CPP_DEIT_OUTPUT_H

#include "deit_config.h"
#include "helper_funcs.h"
#include <memory>
#include <unordered_map>
#include <string>
#include <optional>
#include <torch/torch.h>
#include <ttnn/tensor/tensor.hpp>

namespace deit_cpp {

/**
 * TtDeiTOutput class - C++ implementation of DeiT output layer
 * Equivalent to the Python TtDeiTOutput class
 * 
 * This layer applies a linear transformation to reduce the intermediate dimension
 * back to hidden_size and adds a residual connection.
 */
class TtDeiTOutput {
public:
    /**
     * Constructor
     * @param config DeiT configuration
     * @param device TTNN device pointer
     * @param state_dict Model parameters dictionary
     * @param base_address Base address for parameter loading
     */
    TtDeiTOutput(
        const DeiTConfig& config,
        std::shared_ptr<ttnn::MeshDevice> device,
        const std::unordered_map<std::string, torch::Tensor>& state_dict,
        const std::string& base_address = ""
    );

    /**
     * Forward pass
     * @param hidden_states Input hidden states tensor
     * @param input_tensor Input tensor for residual connection
     * @return Output tensor after dense transformation and residual addition
     */
    ttnn::Tensor forward(const ttnn::Tensor& hidden_states, const ttnn::Tensor& input_tensor);

    /**
     * Destructor
     */
    ~TtDeiTOutput() = default;

private:
    // Configuration and device
    DeiTConfig config;
    std::shared_ptr<ttnn::MeshDevice> device;
    
    // Dense layer parameters
    ttnn::Tensor dense_weight;
    std::optional<ttnn::Tensor> dense_bias;
};

} // namespace deit_cpp

#endif // DEIT_CPP_TT_CPP_DEIT_OUTPUT_H