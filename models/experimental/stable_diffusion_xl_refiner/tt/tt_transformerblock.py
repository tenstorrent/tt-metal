import ttnn
from models.experimental.stable_diffusion_xl_refiner.tt.tt_attention import TtAttention
from models.experimental.stable_diffusion_xl_refiner.tt.components.tt_components import TransformerBlockLayerNorm
from models.experimental.stable_diffusion_xl_refiner.tt.tt_feedforward import TtFeedForward


class TtBasicTransformerBlock:
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        num_attn_heads,  # TODO: replace with predetermined values
    ):
        super().__init__()

        self.device = device
        self.module_path = module_path
        self.num_attn_heads = num_attn_heads

        # Configuration setup
        # self.block_config = get_downblock_config(module_path)

        self._initialize_components(state_dict)

    def _initialize_components(self, state_dict):
        # Order of layers:
        # 1. layer_norm_1
        # 2. attention_1
        # 3. layer_norm_2
        # 4. attention_2
        # 5. layer_norm_3
        # 6. feedforward

        self.layer_norm_1 = TransformerBlockLayerNorm(
            self.device,
            state_dict[f"{self.module_path}.norm1.weight"],
            state_dict[f"{self.module_path}.norm1.bias"],
        )

        self.attention_1 = TtAttention(
            device=self.device,
            state_dict=state_dict,
            module_path=f"{self.module_path}.attn1",
            num_attn_heads=self.num_attn_heads,
            # self.block_config.attn1_num_heads,
        )

        self.layer_norm_2 = TransformerBlockLayerNorm(
            self.device,
            state_dict[f"{self.module_path}.norm2.weight"],
            state_dict[f"{self.module_path}.norm2.bias"],
        )

        self.attention_2 = TtAttention(
            device=self.device,
            state_dict=state_dict,
            module_path=f"{self.module_path}.attn2",
            num_attn_heads=self.num_attn_heads,
            # self.block_config.attn2_num_heads,
        )

        self.layer_norm_3 = TransformerBlockLayerNorm(
            self.device,
            state_dict[f"{self.module_path}.norm3.weight"],
            state_dict[f"{self.module_path}.norm3.bias"],
        )

        self.feedforward = TtFeedForward(
            device=self.device,
            state_dict=state_dict,
            module_path=f"{self.module_path}.ff",
        )

    def forward(self, input_tensor, encoder_tensor=None):
        hidden_states = input_tensor

        # Self-attention layer
        hidden_states = self.layer_norm_1.apply(hidden_states)
        hidden_states = self.attention_1.forward(hidden_states, None)
        attn_result = ttnn.add(input_tensor, hidden_states, use_legacy=False)

        ttnn.deallocate(input_tensor)

        # Cross-attention layer
        hidden_states = self.layer_norm_2.apply(attn_result)
        hidden_states = self.attention_2.forward(hidden_states, encoder_tensor)
        attn_result = ttnn.add(attn_result, hidden_states, use_legacy=False)

        # Feedforward layer
        hidden_states = self.layer_norm_3.apply(attn_result)
        hidden_states = self.feedforward.forward(hidden_states)
        attn_result = ttnn.add(attn_result, hidden_states, use_legacy=False)

        return attn_result
