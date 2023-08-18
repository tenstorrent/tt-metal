import torch.nn as nn

import tt_lib
from models.squeezebert.tt.squeezebert_self_attention import TtSqueezeBertSelfAttention
from models.squeezebert.tt.squeezebert_conv_layernorm import TtConvLayerNorm
from models.squeezebert.tt.squeezebert_conv_activation import TtConvActivation


class TtSqueezeBertModule(nn.Module):
    def __init__(self, config, base_address="", state_dict=None, device=None) -> None:
        super().__init__()
        self.config = config
        self.base_address = base_address
        self.state_dict = state_dict
        self.device = device

        c0 = self.config.hidden_size
        c1 = self.config.hidden_size
        c2 = self.config.intermediate_size
        c3 = self.config.hidden_size

        self.attention = TtSqueezeBertSelfAttention(
            self.config,
            cin=c0,
            q_groups=self.config.q_groups,
            k_groups=self.config.k_groups,
            v_groups=self.config.v_groups,
            base_address=f"{self.base_address}.attention",
            state_dict=self.state_dict,
            device=self.device,
        )

        self.post_attention = TtConvLayerNorm(
            self.config,
            cin=c0,
            cout=c1,
            groups=self.config.post_attention_groups,
            base_address=f"{self.base_address}.post_attention",
            state_dict=self.state_dict,
            device=self.device,
        )

        self.intermediate = TtConvActivation(
            self.config,
            cin=c1,
            cout=c2,
            groups=self.config.intermediate_groups,
            base_address=f"{self.base_address}.intermediate",
            state_dict=self.state_dict,
            device=self.device,
        )

        self.output = TtConvLayerNorm(
            self.config,
            cin=c2,
            cout=c3,
            groups=self.config.output_groups,
            base_address=f"{self.base_address}.output",
            state_dict=self.state_dict,
            device=self.device,
        )

    def forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        attention_mask: tt_lib.tensor.Tensor,
        output_attentions: bool = False,
    ):
        att = self.attention(hidden_states, attention_mask, output_attentions)
        attention_output = att["context_layer"]

        post_attention_output = self.post_attention(attention_output, hidden_states)
        intermediate_output = self.intermediate(post_attention_output)
        layer_output = self.output(intermediate_output, post_attention_output)

        output_dict = {"feature_map": layer_output}
        if output_attentions:
            output_dict["attention_score"] = att["attention_score"]
        return output_dict
