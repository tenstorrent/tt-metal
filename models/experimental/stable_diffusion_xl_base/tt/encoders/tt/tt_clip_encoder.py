from models.experimental.stable_diffusion_xl_base.tt.encoders.tt.tt_clip_encoder_layer import TtClipEncoderLayer
import torch.nn as nn


class TtClipEncoder(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        model_config,
        num_attention_heads,
        hidden_size,
        num_layers,
    ):
        super().__init__()
        self.device = device

        self.layers = []
        for i in range(num_layers):
            self.layers.append(
                TtClipEncoderLayer(
                    device,
                    state_dict,
                    f"{module_path}.layers.{i}",
                    model_config,
                    num_attention_heads,
                    hidden_size,
                )
            )

    def forward(self, inputs_embeds, causal_attention_mask):
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_attention_mask)
        return hidden_states
