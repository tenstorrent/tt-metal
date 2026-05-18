# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


class Wav2Vec2Config:
    """
    Configuration for Wav2Vec2Encoder, mirroring the HuggingFace
    `facebook/wav2vec2-base-960h` model architecture used by the WAN 2.2 S2V
    pipeline's reference AudioEncoder.

    Args:
        hidden_size: Transformer-encoder hidden dimension (768 for base, 1024 for large).
        num_hidden_layers: Number of transformer encoder layers (12 for base, 24 for large).
        num_attention_heads: Number of self-attention heads.
        intermediate_size: FFN inner dimension.
        conv_dim: Per-conv-layer output channel sizes for the feature extractor.
        conv_stride: Strides for each conv layer.
        conv_kernel: Kernel sizes for each conv layer.
        conv_bias: Whether the conv layers use bias.
        feat_extract_norm: "group" (only first layer has GroupNorm) or "layer"
            (every layer has LayerNorm). "group" is the base default.
        num_conv_pos_embeddings: Kernel size of the positional convolution applied
            in the encoder before the transformer stack.
        num_conv_pos_embedding_groups: Number of groups in that positional conv.
        layer_norm_eps: Eps for all LayerNorms.
        hidden_act: Activation for the FFN ("gelu").
        feat_proj_layer_norm: Whether feature_projection applies LayerNorm before
            the Linear projection. True for both base and large.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        conv_dim: tuple = (512, 512, 512, 512, 512, 512, 512),
        conv_stride: tuple = (5, 2, 2, 2, 2, 2, 2),
        conv_kernel: tuple = (10, 3, 3, 3, 3, 2, 2),
        conv_bias: bool = False,
        feat_extract_norm: str = "group",
        num_conv_pos_embeddings: int = 128,
        num_conv_pos_embedding_groups: int = 16,
        layer_norm_eps: float = 1e-5,
        hidden_act: str = "gelu",
        feat_proj_layer_norm: bool = True,
        do_stable_layer_norm: bool = False,
    ):
        assert len(conv_dim) == len(conv_stride) == len(conv_kernel)
        assert feat_extract_norm in ("group", "layer")
        assert hidden_act == "gelu"

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.conv_dim = tuple(conv_dim)
        self.conv_stride = tuple(conv_stride)
        self.conv_kernel = tuple(conv_kernel)
        self.conv_bias = conv_bias
        self.feat_extract_norm = feat_extract_norm
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.feat_proj_layer_norm = feat_proj_layer_norm
        # `wav2vec2-base-960h`: False (post-LN).  `wav2vec2-large-xlsr-53`: True (pre-LN).
        self.do_stable_layer_norm = do_stable_layer_norm

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_hf(cls, hf_config) -> "Wav2Vec2Config":
        return cls(
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            intermediate_size=hf_config.intermediate_size,
            conv_dim=tuple(hf_config.conv_dim),
            conv_stride=tuple(hf_config.conv_stride),
            conv_kernel=tuple(hf_config.conv_kernel),
            conv_bias=bool(hf_config.conv_bias),
            feat_extract_norm=hf_config.feat_extract_norm,
            num_conv_pos_embeddings=hf_config.num_conv_pos_embeddings,
            num_conv_pos_embedding_groups=hf_config.num_conv_pos_embedding_groups,
            layer_norm_eps=hf_config.layer_norm_eps,
            hidden_act=hf_config.hidden_act,
            feat_proj_layer_norm=getattr(hf_config, "feat_proj_layer_norm", True),
            do_stable_layer_norm=getattr(hf_config, "do_stable_layer_norm", False),
        )
