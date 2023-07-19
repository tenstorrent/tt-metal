import torch
from torch import nn, Tensor


class PytorchLlamaDecoderModelStacked(torch.nn.Module):
    def __init__(self, hf_reference_model, decoder_ids):
        super().__init__()
        self.decoder_list = torch.nn.Sequential(
            *[
                hf_reference_model.model.layers[decoder_idx]
                for decoder_idx in decoder_ids
            ]
        )
        # get final norm layer
        self.final_layer_norm = hf_reference_model.model.norm
        # Disable dropout
        self.final_layer_norm.eval()

        # get linear layer
        self.linear_layer = hf_reference_model.lm_head
        self.linear_layer.eval()

    def forward(self, x, y, has_layer_norm=False, is_causal=False):
        result = x
        for idx, decoder_layer in enumerate(self.decoder_list):
            result = decoder_layer(hidden_states=result, position_ids=y)[0]

        # layer norm is always present in HF pytorch model
        if has_layer_norm:
            result = self.final_layer_norm(result)

        if is_causal:
            result = self.linear_layer(result)

        return result
