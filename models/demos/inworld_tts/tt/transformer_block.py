"""TTNN implementation of VocosBackbone TransformerBlock.

Pre-norm architecture: RMSNorm -> Attention + residual, RMSNorm -> MLP + residual.
All activations in bf16 L1. Matmuls use fp32 accumulation via compute_kernel_config.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.inworld_tts.tt.attention import TtAttention
from models.demos.inworld_tts.tt.mlp import TtMLP
from models.demos.inworld_tts.tt.model_config import (
    VOCOS_DIM,
    VOCOS_HEADS,
    VOCOS_POS_EMB_DIM,
    get_compute_kernel_config_hifi4,
)

L1 = ttnn.L1_MEMORY_CONFIG


class TtTransformerBlock(LightweightModule):
    """VocosBackbone transformer block. bf16 activations in L1, full core grid."""

    def __init__(
        self,
        device,
        state_dict,
        layer_num,
        dim=VOCOS_DIM,
        n_heads=VOCOS_HEADS,
        pos_emb_dim=VOCOS_POS_EMB_DIM,
        dtype=ttnn.bfloat16,
        state_dict_prefix="",
    ):
        super().__init__()
        self.device = device

        prefix = f"{state_dict_prefix}transformers.{layer_num}."

        self.att_norm_weight = ttnn.from_torch(
            state_dict[prefix + "att_norm.weight"].reshape(1, 1, dim // 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.ffn_norm_weight = ttnn.from_torch(
            state_dict[prefix + "ffn_norm.weight"].reshape(1, 1, dim // 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.attention = TtAttention(
            device=device,
            state_dict=state_dict,
            layer_num=layer_num,
            n_heads=n_heads,
            dim=dim,
            pos_emb_dim=pos_emb_dim,
            dtype=dtype,
            state_dict_prefix=state_dict_prefix,
        )
        self.mlp = TtMLP(
            device=device,
            state_dict=state_dict,
            layer_num=layer_num,
            dim=dim,
            dtype=dtype,
            state_dict_prefix=state_dict_prefix,
        )

        self.compute_kernel_config = get_compute_kernel_config_hifi4()

    def forward(self, x):
        """Forward pass. bf16 activations in L1. fp32 accumulation in matmuls.

        Args:
            x: [1, 1, T, dim] bf16 TILE_LAYOUT
        Returns:
            [1, 1, T, dim] bf16 L1
        """
        # Attention: pre-norm + residual
        h = ttnn.rms_norm(
            x, weight=self.att_norm_weight, memory_config=L1, compute_kernel_config=self.compute_kernel_config
        )
        h = self.attention(h)
        x = ttnn.add(x, h, memory_config=L1)

        # MLP: pre-norm + residual
        h = ttnn.rms_norm(
            x, weight=self.ffn_norm_weight, memory_config=L1, compute_kernel_config=self.compute_kernel_config
        )
        h = self.mlp(h)
        x = ttnn.add(x, h, memory_config=L1)

        return x
