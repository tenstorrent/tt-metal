import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.rope import TTNNRotaryPositionEmbedding

# from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm
from models.experimental.tt_symbiote.modules.normalization import TTNNRMSNorm
from models.experimental.tt_symbiote.modules.attention import TTNNSDPAAttention


class TTNNQwen3OmniAttention(TTNNModule):
    """
    TTNN implementation of Qwen3 Omni Attention (without sliding window support)
    """

    def __init__(self):
        super().__init__()

        # Attention kernel
        self.sdpa = TTNNSDPAAttention()

        # Rotary positional embedding
        self.rope = TTNNRotaryPositionEmbedding()

        # Hardware grid configuration
        self.core_grid = ttnn.CoreGrid(y=8, x=8)

    def move_weights_to_device_impl(self):
        """
        Initialize SDPA configuration when device becomes available
        """
        super().move_weights_to_device_impl()

        if self.sdpa.program_config is None:
            self.sdpa.program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=256,
                k_chunk_size=256,
                exp_approx_mode=False,
            )

            self.sdpa.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

    # -----------------------
    # Initialize parameters
    # -----------------------

    def init_parameters(self):
        self.q_proj = TTNNLinear.from_torch(self.torch_layer.q_proj)
        self.k_proj = TTNNLinear.from_torch(self.torch_layer.k_proj)
        self.v_proj = TTNNLinear.from_torch(self.torch_layer.v_proj)

        self.o_proj = TTNNLinear.from_torch(self.torch_layer.o_proj)

        # Qwen3 specific normalization
        self.q_norm = TTNNRMSNorm.from_torch(self.torch_layer.q_norm)
        self.k_norm = TTNNRMSNorm.from_torch(self.torch_layer.k_norm)

    @classmethod
    def from_torch(cls, torch_layer):
        new_attn = cls()

        # Keep reference for fallback/debugging
        new_attn._fallback_torch_layer = torch_layer

        new_attn.head_dim = torch_layer.head_dim
        new_attn.scaling = torch_layer.scaling
        new_attn.is_causal = torch_layer.is_causal
        new_attn.num_key_value_groups = getattr(torch_layer, "num_key_value_groups", 1)

        # Sliding window removed → always full attention
        new_attn.init_parameters()

        return new_attn

    # -----------------------
    # Forward pass
    # -----------------------

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = list(hidden_states.shape)[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # QKV projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        # Q/K normalization (Qwen3 specific)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Rotary positional embedding
        cos, sin = position_embeddings
        query_states, key_states = self.rope(
            query_states,
            key_states,
            cos,
            sin,
        )

        # KV cache update
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.torch_layer.layer_idx,
            )

        # Attention computation (full causal attention)
        attn_output = self.sdpa(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=self.is_causal,
            transpose_output=False,
        )

        # Merge attention heads
        attn_output = ttnn.experimental.nlp_concat_heads(attn_output.to_ttnn)
        attn_output = ttnn.squeeze(attn_output, 1)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, None
