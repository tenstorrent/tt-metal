# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
End-to-end SAM3 image inference pipeline on ttnn.

Injects ttnn-accelerated ViT backbone into the SAM3 model and runs the full
forward pass (ViT → FPN neck → text encoder → geometry encoder → transformer
encoder/decoder → segmentation head) end-to-end.

Components on device:
  - ViT backbone (32 blocks): ttnn linear, SDPA, layer_norm, RoPE
Components on CPU (to be progressively ported):
  - FPN neck convolutions
  - Text encoder (CLIP, runs once per prompt)
  - Geometry encoder
  - Transformer encoder/decoder
  - Segmentation head
"""

import os
import unittest.mock as mock

import torch

BPE_PATH = os.environ.get(
    "SAM3_BPE_PATH",
    os.path.join(
        os.environ.get("TT_METAL_HOME", os.path.dirname(os.path.abspath(__file__)).split("/models/")[0]),
        "python_env/lib/python3.10/site-packages/open_clip/bpe_simple_vocab_16e6.txt.gz",
    ),
)


def _patch_cuda_to_cpu():
    """Context-manager-style patches redirecting CUDA tensors to CPU."""
    orig = {
        n: getattr(torch, n)
        for n in [
            "zeros",
            "ones",
            "arange",
            "empty",
            "full",
            "randn",
            "rand",
            "tensor",
            "linspace",
            "logspace",
            "eye",
        ]
    }

    def _redirect(fn):
        def wrapper(*args, **kwargs):
            if "device" in kwargs and kwargs["device"] is not None and "cuda" in str(kwargs["device"]):
                kwargs["device"] = "cpu"
            return fn(*args, **kwargs)

        return wrapper

    patches = [mock.patch("torch.cuda.is_available", return_value=False)]
    for name, fn in orig.items():
        patches.append(mock.patch(f"torch.{name}", _redirect(fn)))
    return patches


def _patch_pin_memory():
    """Make pin_memory a no-op when CUDA is unavailable."""
    _orig = torch.Tensor.pin_memory

    def _safe(self, device=None):
        try:
            return _orig(self, device=device)
        except RuntimeError:
            return self

    torch.Tensor.pin_memory = _safe


def build_sam3_model(use_pretrained=True):
    """Build SAM3 model on CPU with pretrained weights.

    Args:
        use_pretrained: If True, download pretrained weights from HuggingFace.

    Returns:
        Sam3Image model on CPU in eval mode.
    """
    patches = _patch_cuda_to_cpu()
    for p in patches:
        p.start()
    try:
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(
            bpe_path=BPE_PATH,
            device="cpu",
            eval_mode=True,
            load_from_HF=use_pretrained,
            checkpoint_path=None,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
    finally:
        for p in patches:
            p.stop()
    return model


def make_batched_datapoint(pixel_values, text_prompts=None):
    """Construct a BatchedDatapoint for SAM3 model.forward().

    Args:
        pixel_values: (B, 3, 1008, 1008) preprocessed image tensor.
        text_prompts: list of text strings (default: ["object", "visual"]).

    Returns:
        BatchedDatapoint ready for model.forward().
    """
    from sam3.model.data_misc import BatchedDatapoint, FindStage

    if text_prompts is None:
        text_prompts = ["object", "visual"]

    find_stage = FindStage(
        img_ids=torch.tensor([0], dtype=torch.long),
        text_ids=torch.tensor([0], dtype=torch.long),
        input_boxes=torch.zeros(0, 1, 4),
        input_boxes_mask=torch.zeros(1, 0, dtype=torch.bool),
        input_boxes_label=torch.zeros(0, 1, dtype=torch.long),
        input_points=torch.empty(0),
        input_points_mask=torch.empty(0),
        object_ids=[],
    )

    return BatchedDatapoint(
        img_batch=pixel_values,
        find_text_batch=text_prompts,
        find_inputs=[find_stage],
        find_targets=[None],
        find_metadatas=[None],
    )


def extract_predictions(sam3_output):
    """Extract pred_masks and pred_logits from SAM3Output.

    Args:
        sam3_output: SAM3Output from model.forward().

    Returns:
        dict with 'pred_masks' and 'pred_logits' tensors, or None if not found.
    """
    try:
        for item in sam3_output:
            if isinstance(item, dict) and "pred_masks" in item:
                return item
            if isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict) and "pred_masks" in sub:
                        return sub
    except TypeError:
        pass
    for attr in dir(sam3_output):
        if attr.startswith("_"):
            continue
        val = getattr(sam3_output, attr, None)
        if not isinstance(val, list):
            continue
        for v in val:
            if isinstance(v, list):
                for sv in v:
                    if isinstance(sv, dict) and "pred_masks" in sv:
                        return sv
            if isinstance(v, dict) and "pred_masks" in v:
                return v
    return None


class TtSam3ImagePipeline:
    """End-to-end SAM3 image inference pipeline with ttnn-accelerated ViT backbone.

    Patches the ViT backbone inside the SAM3 model with a ttnn implementation,
    then delegates to the original model.forward() for the complete pipeline.
    """

    def __init__(self, sam3_model, device):
        from sam3.model.vitdet import ViT

        from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_encoder import (
            _make_compute_config,
            move_encoder_params_to_device,
            preprocess_encoder_weights,
            tt_encoder_forward,
        )
        from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_vitdet import (
            move_backbone_params_to_device,
            preprocess_vit_backbone_weights,
            tt_vit_backbone,
        )

        self.device = device
        self.sam3_model = sam3_model

        vit_backbone = None
        for _, module in sam3_model.named_modules():
            if isinstance(module, ViT):
                vit_backbone = module
                break
        assert vit_backbone is not None, "Could not find ViT backbone in SAM3 model"

        self.vit_backbone = vit_backbone
        self.backbone_params = preprocess_vit_backbone_weights(vit_backbone)
        self.backbone_params = move_backbone_params_to_device(self.backbone_params, device)
        self._tt_vit_backbone = tt_vit_backbone

        self._orig_vit_forward = vit_backbone.forward
        vit_backbone.forward = self._patched_vit_forward

        self._orig_forward_text = sam3_model.backbone.forward_text
        self._text_cache = {}
        sam3_model.backbone.forward_text = self._cached_forward_text

        sam3_model.requires_grad_(False)

        neck = sam3_model.backbone.vision_backbone
        self._orig_pos_enc_forward = neck.position_encoding.forward
        pos_enc_module = neck.position_encoding

        def _fast_pos_enc(x):
            key = (x.shape[-2], x.shape[-1])
            if key in pos_enc_module.cache:
                return pos_enc_module.cache[key].unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            return self._orig_pos_enc_forward(x)

        self._pos_enc_module = pos_enc_module
        self._fast_pos_enc = _fast_pos_enc
        neck.position_encoding.forward = _fast_pos_enc

        encoder = sam3_model.transformer.encoder
        self._encoder = encoder
        self._enc_compute_config = _make_compute_config(device)
        enc_params = preprocess_encoder_weights(encoder)
        self._enc_params = move_encoder_params_to_device(enc_params, device)
        self._tt_encoder_forward = tt_encoder_forward

        self._orig_encoder_forward = encoder.forward

        def _patched_encoder_forward(
            src,
            prompt,
            src_key_padding_mask=None,
            src_pos=None,
            prompt_key_padding_mask=None,
            prompt_pos=None,
            feat_sizes=None,
            encoder_extra_kwargs=None,
        ):
            from sam3.model.encoder import pool_text_feat

            bs = src[0].shape[1]
            if feat_sizes is not None:
                if src_key_padding_mask is None:
                    src_key_padding_mask = [None] * len(src)
                for i, (h, w) in enumerate(feat_sizes):
                    src[i] = src[i].reshape(h, w, bs, -1).permute(2, 3, 0, 1)
                    src_pos[i] = src_pos[i].reshape(h, w, bs, -1).permute(2, 3, 0, 1)
                    if src_key_padding_mask[i] is not None:
                        src_key_padding_mask[i] = src_key_padding_mask[i].reshape(h, w, bs).permute(2, 0, 1)

            if encoder.add_pooled_text_to_img_feat:
                pooled_text = pool_text_feat(prompt, prompt_key_padding_mask, encoder.pool_text_with_mask)
                pooled_text = encoder.text_pooling_proj(pooled_text)[..., None, None]
                src = [x.add_(pooled_text) for x in src]

            (
                src_flatten,
                key_padding_masks_flatten,
                lvl_pos_embed_flatten,
                level_start_index,
                valid_ratios,
                spatial_shapes,
            ) = encoder._prepare_multilevel_features(src, src_key_padding_mask, src_pos)

            memory_batch_first = prompt.transpose(0, 1)
            output = self._tt_encoder_forward(
                src_flatten,
                memory_batch_first,
                lvl_pos_embed_flatten,
                prompt_key_padding_mask,
                self._enc_params,
                self.device,
                self._enc_compute_config,
            )

            return {
                "memory": output.transpose(0, 1),
                "padding_mask": (
                    key_padding_masks_flatten.transpose(0, 1) if key_padding_masks_flatten is not None else None
                ),
                "pos_embed": lvl_pos_embed_flatten.transpose(0, 1),
                "memory_text": prompt,
                "level_start_index": level_start_index,
                "spatial_shapes": spatial_shapes,
                "valid_ratios": valid_ratios,
            }

        self._patched_encoder_forward = _patched_encoder_forward
        encoder.forward = _patched_encoder_forward

        pixel_decoder = sam3_model.segmentation_head.pixel_decoder
        self._pixel_decoder = pixel_decoder
        self._orig_pd_forward = pixel_decoder.forward

        pd_cw = [c.weight.to(torch.bfloat16, memory_format=torch.channels_last) for c in pixel_decoder.conv_layers]
        pd_cb = [c.bias.to(torch.bfloat16) if c.bias is not None else None for c in pixel_decoder.conv_layers]
        pd_nw = [n.weight.to(torch.bfloat16) for n in pixel_decoder.norms]
        pd_nb = [n.bias.to(torch.bfloat16) for n in pixel_decoder.norms]
        pd_ng = [n.num_groups for n in pixel_decoder.norms]
        pd_shared = pixel_decoder.shared_conv

        def _fast_pd_forward(backbone_feats):
            feats_b = [f.to(torch.bfloat16, memory_format=torch.channels_last) for f in backbone_feats]
            prev = feats_b[-1]
            for i, bb in enumerate(feats_b[:-1][::-1]):
                prev = bb + torch.nn.functional.interpolate(
                    prev, size=bb.shape[-2:], mode=pixel_decoder.interpolation_mode
                )
                ci = 0 if pd_shared else i
                prev = torch.nn.functional.conv2d(prev, pd_cw[ci], pd_cb[ci], padding=1)
                prev = torch.nn.functional.relu(torch.nn.functional.group_norm(prev, pd_ng[ci], pd_nw[ci], pd_nb[ci]))
            return prev.float().contiguous()

        self._fast_pd_forward = _fast_pd_forward
        pixel_decoder.forward = _fast_pd_forward

    def _patched_vit_forward(self, x):
        if self._encoder.forward is not self._patched_encoder_forward:
            self._encoder.forward = self._patched_encoder_forward
        if self._pixel_decoder.forward is not self._fast_pd_forward:
            self._pixel_decoder.forward = self._fast_pd_forward
        return self._tt_vit_backbone(x, self.backbone_params, self.device)

    def _cached_forward_text(self, captions, input_boxes=None, additional_text=None, device="cuda"):
        cache_key = (tuple(captions), input_boxes is not None, additional_text is not None)
        if cache_key in self._text_cache:
            return self._text_cache[cache_key]
        result = self._orig_forward_text(
            captions, input_boxes=input_boxes, additional_text=additional_text, device=device
        )
        self._text_cache[cache_key] = result
        return result

    def restore(self):
        """Restore original forwards (for cleanup / PCC comparison)."""
        self.vit_backbone.forward = self._orig_vit_forward
        self.sam3_model.backbone.forward_text = self._orig_forward_text
        self._pos_enc_module.forward = self._orig_pos_enc_forward
        self._encoder.forward = self._orig_encoder_forward
        self._pixel_decoder.forward = self._orig_pd_forward

    @torch.inference_mode()
    def forward(self, input_batch):
        """Run the full SAM3 forward pass with ttnn ViT backbone.

        Args:
            input_batch: BatchedDatapoint from make_batched_datapoint().

        Returns:
            SAM3Output with pred_masks, pred_logits, etc.
        """
        return self.sam3_model(input_batch)

    @torch.inference_mode()
    def forward_image(self, pixel_values, text_prompts=None):
        """Convenience: preprocess + forward + extract predictions.

        Args:
            pixel_values: (B, 3, 1008, 1008) preprocessed image tensor.
            text_prompts: list of text strings.

        Returns:
            dict with 'pred_masks' (B, Q, H, W) and 'pred_logits' (B, Q, 1).
        """
        input_batch = make_batched_datapoint(pixel_values, text_prompts)
        output = self.forward(input_batch)
        return extract_predictions(output)


def preprocess_image(image: torch.Tensor, target_size: int = 1008) -> torch.Tensor:
    """Preprocess image tensor for SAM3: resize + normalize to [-1, 1]."""
    import torch.nn.functional as F

    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.max() > 1.0:
        image = image.float() / 255.0
    if image.shape[-2:] != (target_size, target_size):
        image = F.interpolate(image, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return (image - 0.5) / 0.5
