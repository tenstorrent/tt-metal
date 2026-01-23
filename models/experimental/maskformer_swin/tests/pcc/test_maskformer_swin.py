# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maskformer_swin_b_pcc(device, reset_seeds):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from transformers import MaskFormerForInstanceSegmentation

    import ttnn

    from models.experimental.maskformer_swin.tt.backbone_swin import MaskFormerSwinBackbone
    from models.experimental.maskformer_swin.tt.heads import MaskFormerHeads, MaskFormerHeadsConfig
    from models.experimental.maskformer_swin.tt.pixel_decoder import MaskFormerPixelDecoder, PixelDecoderConfig
    from models.experimental.maskformer_swin.tt.transformer_decoder import (
        MaskFormerTransformerDecoder,
        TransformerDecoderConfig,
    )
    from tests.ttnn.utils_for_testing import assert_with_pcc

    model_id = "facebook/maskformer-swin-base-coco"
    ref_model = MaskFormerForInstanceSegmentation.from_pretrained(model_id)
    ref_model.eval()

    # Use reference model's weights/config to avoid double downloads
    state_dict = ref_model.state_dict()
    ref_cfg = ref_model.config.to_dict() if hasattr(ref_model, "config") else {}

    backbone_cfg = ref_cfg.get("backbone_config", {}) if isinstance(ref_cfg, dict) else {}
    decoder_cfg = ref_cfg.get("decoder_config", {}) if isinstance(ref_cfg, dict) else {}

    backbone = MaskFormerSwinBackbone.from_huggingface(state_dict, device=device, config_dict=backbone_cfg)

    pixel_cfg = PixelDecoderConfig(
        fpn_dim=int(ref_cfg.get("fpn_feature_size", 256)),
        mask_dim=int(ref_cfg.get("mask_feature_size", 256)),
    )
    pixel_decoder = MaskFormerPixelDecoder.from_huggingface(state_dict, config=pixel_cfg, device=device)

    transformer_cfg = TransformerDecoderConfig(
        num_layers=int(ref_cfg.get("num_hidden_layers", 6)),
        num_attention_heads=int(ref_cfg.get("num_attention_heads", 8)),
        hidden_dim=int(ref_cfg.get("fpn_feature_size", 256)),
        dim_feedforward=int(decoder_cfg.get("decoder_ffn_dim", 2048)) if isinstance(decoder_cfg, dict) else 2048,
        activation=str(decoder_cfg.get("activation_function", "relu")) if isinstance(decoder_cfg, dict) else "relu",
        in_features=int(pixel_cfg.input_channels[-1]),
    )
    transformer_decoder = MaskFormerTransformerDecoder.from_huggingface(
        state_dict, config=transformer_cfg, device=device
    )

    heads_cfg = MaskFormerHeadsConfig(
        num_classes=len(ref_cfg.get("id2label", {})),
        hidden_dim=transformer_cfg.hidden_dim,
        mask_dim=pixel_cfg.mask_dim,
    )
    heads = MaskFormerHeads(config=heads_cfg, device=device)
    heads.load_weights(state_dict)

    pixel_values = torch.randn(1, 3, 320, 320, dtype=torch.float32)

    with torch.no_grad():
        ref_out = ref_model(pixel_values)

    features, _ = backbone.forward(pixel_values)
    mask_features, _ = pixel_decoder.forward(features)
    decoder_last, _, _ = transformer_decoder.forward_tt(features[-1])
    class_logits, mask_logits = heads.forward(decoder_last, mask_features)

    if hasattr(ttnn, "synchronize_device"):
        ttnn.synchronize_device(device)

    # Reference output names in HF transformers
    ref_class = getattr(ref_out, "class_queries_logits", None)
    ref_masks = getattr(ref_out, "masks_queries_logits", None)
    assert ref_class is not None, "Reference model output missing class_queries_logits"
    assert ref_masks is not None, "Reference model output missing masks_queries_logits"

    assert_with_pcc(ref_class, class_logits, pcc=0.97)
    assert_with_pcc(ref_masks, mask_logits, pcc=0.97)
