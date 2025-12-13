# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest

try:
    from transformers import MaskFormerForInstanceSegmentation  # noqa: F401

    _TRANSFORMERS_AVAILABLE = True
except ModuleNotFoundError:
    _TRANSFORMERS_AVAILABLE = False

import torch

from models.experimental.maskformer_swin.backbone_swin import MaskFormerSwinBackbone
from models.experimental.maskformer_swin.pixel_decoder import MaskFormerPixelDecoder, PixelDecoderConfig
from models.experimental.maskformer_swin.transformer_decoder import (
    MaskFormerTransformerDecoder,
    TransformerDecoderConfig,
)
from models.experimental.maskformer_swin.heads import MaskFormerHeads, MaskFormerHeadsConfig
from models.experimental.maskformer_swin.fallback import MaskFormerFallbackPipeline
from models.experimental.maskformer_swin.weights import (
    WeightConversionConfig,
    convert_state_dict_to_tt,
    download_reference_weights,
)


@pytest.mark.skip(reason="MaskFormer module parity tests are not implemented yet.")
def test_backbone_stage_parity():
    """Placeholder test for Swin stage PCC metrics (see `parity.GOLDEN_TAPS`)."""
    raise NotImplementedError


@pytest.mark.skip(reason="MaskFormer module parity tests are not implemented yet.")
def test_pixel_decoder_parity():
    """Placeholder test for pixel decoder PCC metrics (see `parity.GOLDEN_TAPS`)."""
    raise NotImplementedError


@pytest.mark.skip(reason="MaskFormer module parity tests are not implemented yet.")
def test_transformer_decoder_parity():
    """Placeholder test for transformer decoder PCC metrics (see `parity.GOLDEN_TAPS`)."""
    raise NotImplementedError


@pytest.mark.skipif(not _TRANSFORMERS_AVAILABLE, reason="Requires transformers package for fallback execution.")
@pytest.mark.skipif(
    os.environ.get("MASKFORMER_RUN_WEIGHT_TESTS") != "1",
    reason="Set MASKFORMER_RUN_WEIGHT_TESTS=1 to run integration tests that pull HF weights.",
)
def test_backbone_fallback_emits_four_scales():
    """Ensure the HuggingFace fallback returns four feature maps."""

    cfg = WeightConversionConfig()
    ref = download_reference_weights(cfg)
    state = convert_state_dict_to_tt(ref.state_dict, cfg)
    backbone_cfg = ref.config.get("backbone_config", {})
    backbone = MaskFormerSwinBackbone.from_huggingface(state, device=None, config_dict=backbone_cfg)

    images = torch.randn(1, 3, backbone.config.image_size[0], backbone.config.image_size[1])
    features, hidden_states = backbone.forward(images)
    assert len(features) == 4
    assert len(hidden_states) == 4
    shapes = [tuple(f.shape) for f in features]
    expected_channels = [
        backbone.config.embed_dim,
        backbone.config.embed_dim * 2,
        backbone.config.embed_dim * 4,
        backbone.config.embed_dim * 8,
    ]
    h = backbone.config.image_size[0] // backbone.config.patch_size
    w = backbone.config.image_size[1] // backbone.config.patch_size
    expected_hw = [
        (h, w),
        (h // 2, w // 2),
        (h // 4, w // 4),
        (h // 8, w // 8),
    ]
    for idx, shape in enumerate(shapes):
        _, c, height, width = shape
        assert c == expected_channels[idx]
        assert (height, width) == expected_hw[idx]


@pytest.mark.skipif(not _TRANSFORMERS_AVAILABLE, reason="Requires transformers package for fallback execution.")
@pytest.mark.skipif(
    os.environ.get("MASKFORMER_RUN_WEIGHT_TESTS") != "1",
    reason="Set MASKFORMER_RUN_WEIGHT_TESTS=1 to run integration tests that pull HF weights.",
)
def test_pixel_decoder_fallback_shapes():
    cfg = WeightConversionConfig()
    ref = download_reference_weights(cfg)
    state = convert_state_dict_to_tt(ref.state_dict, cfg)
    backbone_cfg = ref.config.get("backbone_config", {})
    backbone = MaskFormerSwinBackbone.from_huggingface(state, device=None, config_dict=backbone_cfg)
    feature_maps, _ = backbone.forward(torch.randn(1, 3, 384, 384))

    pixel_decoder = MaskFormerPixelDecoder.from_huggingface(state, config=PixelDecoderConfig(), device=None)
    mask_features, hidden = pixel_decoder.forward(feature_maps)

    assert tuple(mask_features.shape) == (1, 256, 96, 96)
    assert [tuple(f.shape) for f in hidden] == [(1, 256, 24, 24), (1, 256, 48, 48), (1, 256, 96, 96)]


@pytest.mark.skipif(not _TRANSFORMERS_AVAILABLE, reason="Requires transformers package for fallback execution.")
@pytest.mark.skipif(
    os.environ.get("MASKFORMER_RUN_WEIGHT_TESTS") != "1",
    reason="Set MASKFORMER_RUN_WEIGHT_TESTS=1 to run integration tests that pull HF weights.",
)
def test_transformer_decoder_fallback_shapes():
    cfg = WeightConversionConfig()
    ref = download_reference_weights(cfg)
    state = convert_state_dict_to_tt(ref.state_dict, cfg)
    backbone_cfg = ref.config.get("backbone_config", {})
    backbone = MaskFormerSwinBackbone.from_huggingface(state, device=None, config_dict=backbone_cfg)
    feature_maps, _ = backbone.forward(torch.randn(1, 3, 384, 384))

    pixel_decoder_cfg = PixelDecoderConfig()
    pixel_decoder = MaskFormerPixelDecoder.from_huggingface(state, config=pixel_decoder_cfg, device=None)
    mask_features, _ = pixel_decoder.forward(feature_maps)

    transformer_cfg = TransformerDecoderConfig(
        num_layers=ref.config.get("num_hidden_layers", 6),
        num_attention_heads=ref.config.get("num_attention_heads", 8),
        hidden_dim=ref.config.get("fpn_feature_size", 256),
        dim_feedforward=ref.config["decoder_config"].get("decoder_ffn_dim", 2048),
        dropout=ref.config["decoder_config"].get("dropout", 0.0),
        activation=ref.config["decoder_config"].get("activation_function", "relu"),
        in_features=pixel_decoder_cfg.input_channels[-1],
        maskformer_config=ref.config,
    )
    transformer_decoder = MaskFormerTransformerDecoder.from_huggingface(state, config=transformer_cfg, device=None)
    image_features = feature_maps[-1]
    last_hidden, hidden_states, attentions = transformer_decoder.forward(
        image_features,
        output_hidden_states=True,
        output_attentions=True,
    )

    assert tuple(last_hidden.shape) == (1, ref.config.get("num_queries", 100), transformer_cfg.hidden_dim)
    assert len(hidden_states) == transformer_cfg.num_layers
    if attentions:
        assert attentions[0].shape[1] == transformer_cfg.num_attention_heads

    try:
        heads = MaskFormerHeads.from_huggingface(
            state,
            config=MaskFormerHeadsConfig(
                num_classes=len(ref.config.get("id2label", {})),
                hidden_dim=transformer_cfg.hidden_dim,
                mask_dim=pixel_decoder_cfg.fpn_dim,
            ),
            device=None,
        )
    except NotImplementedError:
        pytest.skip("transformers not available for heads fallback.")
    class_logits, mask_logits = heads.forward(last_hidden, mask_features)
    assert tuple(class_logits.shape) == (
        1,
        ref.config.get("num_queries", 100),
        len(ref.config.get("id2label", {})) + 1,
    )
    assert tuple(mask_logits.shape) == (1, ref.config.get("num_queries", 100), 96, 96)


@pytest.mark.skipif(not _TRANSFORMERS_AVAILABLE, reason="Requires transformers package for fallback execution.")
@pytest.mark.skipif(
    os.environ.get("MASKFORMER_RUN_WEIGHT_TESTS") != "1",
    reason="Set MASKFORMER_RUN_WEIGHT_TESTS=1 to run integration tests that pull HF weights.",
)
def test_fallback_pipeline_forward():
    cfg = WeightConversionConfig()
    ref = download_reference_weights(cfg)
    state = convert_state_dict_to_tt(ref.state_dict, cfg)
    pipeline = MaskFormerFallbackPipeline.from_reference(ref, state)

    pixel_values = torch.randn(1, 3, 384, 384)
    outputs = pipeline.forward(pixel_values)

    expected_classes = len(ref.config.get("id2label", {})) + 1
    assert outputs.class_logits.shape == (1, ref.config.get("num_queries", 100), expected_classes)
    assert outputs.mask_logits.shape[0] == 1
    assert outputs.mask_logits.shape[1] == ref.config.get("num_queries", 100)


@pytest.mark.skipif(not _TRANSFORMERS_AVAILABLE, reason="Requires transformers package for fallback execution.")
@pytest.mark.skipif(
    os.environ.get("MASKFORMER_RUN_WEIGHT_TESTS") != "1",
    reason="Set MASKFORMER_RUN_WEIGHT_TESTS=1 to run integration tests that pull HF weights.",
)
def test_tt_pipeline_forward_shapes():
    """Smoke-test TT decoder + heads (and optional TT mask projection) end-to-end."""

    from models.experimental.maskformer_swin.ttnn_compat import ttnn as _ttnn

    if _ttnn is None or not hasattr(_ttnn, "open_device"):
        pytest.skip("TTNN runtime with open_device is required for TT pipeline tests.")

    try:
        device = _ttnn.open_device(device_id=0)
    except Exception:
        pytest.skip("Unable to open TT device for MaskFormer TT pipeline test.")

    try:
        cfg = WeightConversionConfig()
        ref = download_reference_weights(cfg)
        state = convert_state_dict_to_tt(ref.state_dict, cfg)

        # Enable TT decoder and mask projection paths.
        os.environ["MASKFORMER_TT_DECODER"] = "1"
        os.environ.setdefault("MASKFORMER_TT_MASK_PROJ", "1")

        pipeline = MaskFormerFallbackPipeline.from_reference(ref, state, device=device)

        pixel_values = torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            outputs = pipeline.forward(pixel_values)

        expected_classes = len(ref.config.get("id2label", {})) + 1
        assert outputs.class_logits.shape == (1, ref.config.get("num_queries", 100), expected_classes)
        assert outputs.mask_logits.shape[0] == 1
        assert outputs.mask_logits.shape[1] == ref.config.get("num_queries", 100)
        assert torch.isfinite(outputs.class_logits).all()
        assert torch.isfinite(outputs.mask_logits).all()
    finally:
        os.environ["MASKFORMER_TT_DECODER"] = "0"
        try:
            _ttnn.close_device(device)
        except Exception:
            pass
