#ifndef DEIT_CPP_TT_CPP_DEIT_ENCODER_H
#define DEIT_CPP_TT_CPP_DEIT_ENCODER_H

#include "deit_config.h"
#include "deit_layer.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <optional>
#include <tuple>
#include <torch/torch.h>
#include <ttnn/tensor/tensor.hpp>

namespace deit_cpp {

/**
 * TtDeiTEncoder class - C++ implementation of DeiT encoder
 * Equivalent to the Python TtDeiTEncoder class
 * 
 * This class manages multiple DeiT layers and handles the forward pass
 * through all transformer layers in the encoder.
 */
class TtDeiTEncoder {
public:
    /**
     * Constructor
     * @param config DeiT configuration
     * @param device TTNN device pointer
     * @param state_dict Model parameters dictionary
     * @param base_address Base address for parameter loading
     */
    TtDeiTEncoder(
        const DeiTConfig& config,
        std::shared_ptr<ttnn::MeshDevice> device,
        const std::unordered_map<std::string, torch::Tensor>& state_dict,
        const std::string& base_address = ""
    );

    /**
     * Forward pass through all encoder layers
     * @param hidden_states Input hidden states tensor
     * @param head_mask Optional attention head mask for all layers
     * @param output_attentions Whether to output attention weights
     * @param output_hidden_states Whether to output hidden states from all layers
     * @param return_dict Whether to return structured output (not used in C++)
     * @return Tuple containing final hidden states, all hidden states (optional), and all attention weights (optional)
     */
    std::tuple<ttnn::Tensor, std::optional<std::vector<ttnn::Tensor>>, std::optional<std::vector<ttnn::Tensor>>> forward(
        const ttnn::Tensor& hidden_states,
        const std::optional<std::vector<ttnn::Tensor>>& head_mask = std::nullopt,
        bool output_attentions = false,
        bool output_hidden_states = false,
        bool return_dict = true
    );

    /**
     * Get the number of layers in the encoder
     * @return Number of layers
     */
    size_t get_num_layers() const { return layers.size(); }

    /**
     * Destructor
     */
    ~TtDeiTEncoder() = default;

private:
    // Configuration and device
    DeiTConfig config;
    std::shared_ptr<ttnn::MeshDevice> device;
    
    // Encoder layers
    std::vector<std::unique_ptr<TtDeiTLayer>> layers;
    
    // Gradient checkpointing flag (not used in inference)
    bool gradient_checkpointing;
};

} // namespace deit_cpp

#endif // DEIT_CPP_TT_CPP_DEIT_ENCODER_H