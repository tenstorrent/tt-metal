# ------------------------------------------------------------------------
# RF-DETR-base reference (CPU) configuration.
#
# Self-contained config object built from the official Roboflow/rf-detr-base
# config.json. This is a plain dataclass-like container (no transformers
# dependency) holding every hyper-parameter the reference forward needs.
#
# Derived from / cross-checked against:
#   - Roboflow/rf-detr-base config.json (HF cache)
#   - transformers (main) src/transformers/models/rf_detr/configuration_rf_detr.py
#   - rfdetr (develop) src/rfdetr/config.py RFDETRBaseConfig
# ------------------------------------------------------------------------
"""Configuration for the RF-DETR-base reference model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RfDetrBackboneConfig:
    """Windowed DINOv2-with-registers (num_register_tokens=0) backbone config."""

    hidden_size: int = 384
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    mlp_ratio: int = 4
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    image_size: int = 518  # native DINOv2 grid (37x37 patches at patch_size 14)
    patch_size: int = 14
    num_channels: int = 3
    qkv_bias: bool = True
    layerscale_value: float = 1.0
    use_swiglu_ffn: bool = False
    num_register_tokens: int = 0
    use_mask_token: bool = True
    apply_layernorm: bool = True
    reshape_hidden_states: bool = True
    num_windows: int = 4  # windows-per-side -> num_windows**2 = 16 windows total
    # out_indices select which encoder hidden states become feature maps.
    # stage_names = ["stem", "stage1", ..., "stage12"]; out_features ->
    # indices into the hidden_states list (which starts with the embedding output).
    out_indices: tuple[int, ...] = (2, 5, 8, 11)
    # window_block_indexes are derived: every block up to the last out index that
    # is NOT an out index uses windowed attention; out-index blocks use global.
    window_block_indexes: tuple[int, ...] = field(default=(0, 1, 3, 4, 6, 7, 9, 10))


@dataclass
class RfDetrConfig:
    """Top-level RF-DETR-base configuration."""

    # backbone
    backbone_config: RfDetrBackboneConfig = field(default_factory=RfDetrBackboneConfig)

    # projector (C2f) — matches rfdetr MultiScaleProjector with scale P4 (==1.0)
    d_model: int = 256
    hidden_expansion: float = 0.5
    c2f_num_blocks: int = 3
    activation_function: str = "silu"
    projector_scale_factors: tuple[float, ...] = (1.0,)
    # The projector ConvNorm uses a channels-first LayerNorm (NOT BatchNorm);
    # the published checkpoint stores only affine weight/bias under `.bn`.
    projector_norm_eps: float = 1e-6

    # decoder / transformer
    decoder_layers: int = 3
    decoder_self_attention_heads: int = 8
    decoder_cross_attention_heads: int = 16
    decoder_n_points: int = 2
    decoder_ffn_dim: int = 2048
    decoder_activation_function: str = "relu"
    num_feature_levels: int = 1
    num_queries: int = 300
    group_detr: int = 13
    layer_norm_eps: float = 1e-5

    # heads
    num_labels: int = 91  # class_embed outputs 91 logits

    # behaviour flags (RF-DETR-base): bbox reparametrisation + lite refine
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    two_stage: bool = True

    # preprocessing
    image_resolution: int = 560
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    rescale_factor: float = 1.0 / 255.0

    # COCO id->label (filled by weights.load_rf_detr_base from config.json)
    id2label: dict | None = None

    @property
    def head_dim_self(self) -> int:
        return self.d_model // self.decoder_self_attention_heads
