# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


class Wav2Vec2Config:
    """
    Configuration for Wav2Vec2Encoder, mirroring the HuggingFace
    `facebook/wav2vec2-large-xlsr-53` model architecture used by the WAN 2.2
    S2V pipeline's reference AudioEncoder. Only the pre-LN / per-layer-LN
    variant (``do_stable_layer_norm=True``, ``feat_extract_norm="layer"``) is
    supported.

    Args:
        hidden_size: Transformer-encoder hidden dimension.
        num_hidden_layers: Number of transformer encoder layers.
        num_attention_heads: Number of self-attention heads.
        intermediate_size: FFN inner dimension.
        conv_dim: Per-conv-layer output channel sizes for the feature extractor.
        conv_stride: Strides for each conv layer.
        conv_kernel: Kernel sizes for each conv layer.
        conv_bias: Whether the conv layers use bias.
        layer_norm_eps: Eps for all LayerNorms.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        conv_dim: tuple = (512, 512, 512, 512, 512, 512, 512),
        conv_stride: tuple = (5, 2, 2, 2, 2, 2, 2),
        conv_kernel: tuple = (10, 3, 3, 3, 3, 2, 2),
        conv_bias: bool = True,
        feat_extract_norm: str = "layer",
        layer_norm_eps: float = 1e-5,
        hidden_act: str = "gelu",
        do_stable_layer_norm: bool = True,
    ):
        assert len(conv_dim) == len(conv_stride) == len(conv_kernel)
        assert feat_extract_norm == "layer", "only feat_extract_norm='layer' (large-xlsr-53) is supported"
        assert do_stable_layer_norm, "only pre-LN (large-xlsr-53) wav2vec2 is supported"
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
        self.layer_norm_eps = layer_norm_eps
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
            layer_norm_eps=hf_config.layer_norm_eps,
            hidden_act=hf_config.hidden_act,
            do_stable_layer_norm=getattr(hf_config, "do_stable_layer_norm", False),
        )
