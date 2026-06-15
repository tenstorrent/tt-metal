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
        torch.set_num_threads(17)

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

        mask_pred_module = sam3_model.segmentation_head.mask_predictor
        self._mask_predictor = mask_pred_module
        self._orig_mp_forward = mask_pred_module.forward

        me = mask_pred_module.mask_embed
        mp_w = [l.weight.to(torch.bfloat16) for l in me.layers]
        mp_b = [l.bias.to(torch.bfloat16) if l.bias is not None else None for l in me.layers]
        n_me_layers = me.num_layers

        def _fast_mp_forward(obj_queries, pixel_embed):
            x = obj_queries.to(torch.bfloat16)
            for i in range(n_me_layers):
                x = torch.nn.functional.linear(x, mp_w[i], mp_b[i])
                if i < n_me_layers - 1:
                    x = torch.nn.functional.relu(x)
            pe = pixel_embed if pixel_embed.dtype == torch.bfloat16 else pixel_embed.to(torch.bfloat16)
            if pe.ndim == 3:
                return torch.einsum("bqc,chw->bqhw", x, pe).float()
            return torch.einsum("bqc,bchw->bqhw", x, pe).float()

        self._fast_mp_forward = _fast_mp_forward
        mask_pred_module.forward = _fast_mp_forward

        from sam3.model.box_ops import box_cxcywh_to_xyxy as _box_cxcywh_to_xyxy

        decoder = sam3_model.transformer.decoder
        _log2_8 = 3.0

        def _fast_rpb_matrix(reference_boxes, feat_size):
            H, W = feat_size
            boxes_xyxy = _box_cxcywh_to_xyxy(reference_boxes).transpose(0, 1)
            bs, num_queries, _ = boxes_xyxy.shape

            if decoder.compilable_cord_cache is None:
                decoder.compilable_cord_cache = decoder._get_coords(H, W, reference_boxes.device)
                decoder.compilable_stored_size = (H, W)

            if decoder.compilable_stored_size == (H, W):
                coords_h, coords_w = decoder.compilable_cord_cache
            else:
                if feat_size not in decoder.coord_cache:
                    decoder.coord_cache[feat_size] = decoder._get_coords(H, W, reference_boxes.device)
                coords_h, coords_w = decoder.coord_cache[feat_size]

            boxes_flat = boxes_xyxy.reshape(-1, 1, 4)
            deltas_y = coords_h.view(1, -1, 1) - boxes_flat[:, :, 1:4:2]
            deltas_y = deltas_y.view(bs, num_queries, -1, 2)
            deltas_x = coords_w.view(1, -1, 1) - boxes_flat[:, :, 0:3:2]
            deltas_x = deltas_x.view(bs, num_queries, -1, 2)

            deltas_x = deltas_x * 8
            deltas_x = torch.sign(deltas_x) * torch.log2(torch.abs(deltas_x) + 1.0) / _log2_8
            deltas_y = deltas_y * 8
            deltas_y = torch.sign(deltas_y) * torch.log2(torch.abs(deltas_y) + 1.0) / _log2_8

            deltas_x = decoder.boxRPB_embed_x(x=deltas_x)
            deltas_y = decoder.boxRPB_embed_y(x=deltas_y)

            deltas_y = deltas_y.permute(0, 3, 1, 2).contiguous()
            deltas_x = deltas_x.permute(0, 3, 1, 2).contiguous()
            B = deltas_y.unsqueeze(-1) + deltas_x.unsqueeze(-2)
            return B.flatten(-2, -1)

        decoder._get_rpb_matrix = _fast_rpb_matrix

        self._decoder = decoder
        self._orig_decoder_forward = decoder.forward
        self._orig_layer_forwards = [l.forward for l in decoder.layers]

        import math as _math

        from sam3.model.model_misc import inverse_sigmoid as _inv_sig

        _dim_t_cache = {}

        def _cached_sineembed(pos_tensor, d_model):
            nf = d_model // 2
            key = (nf, pos_tensor.device.type)
            if key not in _dim_t_cache:
                dt = torch.arange(nf, dtype=torch.float32, device=pos_tensor.device)
                _dim_t_cache[key] = 10000 ** (2 * (torch.div(dt, 2, rounding_mode="floor")) / nf)
            dim_t = _dim_t_cache[key]
            scale = 2 * _math.pi
            x_embed = pos_tensor[:, :, 0] * scale
            y_embed = pos_tensor[:, :, 1] * scale
            pos_x = x_embed[:, :, None] / dim_t
            pos_y = y_embed[:, :, None] / dim_t
            pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
            pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
            if pos_tensor.size(-1) == 2:
                return torch.cat((pos_y, pos_x), dim=2)
            w_embed = pos_tensor[:, :, 2] * scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
            h_embed = pos_tensor[:, :, 3] * scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
            return torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)

        def _make_opt_layer_fwd(lyr):
            _sa, _ca = lyr.self_attn, lyr.cross_attn
            _n1, _n2, _d1, _d2 = lyr.norm1, lyr.norm2, lyr.dropout1, lyr.dropout2
            _ffn = lyr.forward_ffn
            _txt = lyr.use_text_cross_attention
            if _txt:
                _ca_txt, _ct_d, _ct_n = lyr.ca_text, lyr.catext_dropout, lyr.catext_norm

            def _fwd(
                tgt=None,
                tgt_query_pos=None,
                tgt_query_sine_embed=None,
                tgt_key_padding_mask=None,
                tgt_reference_points=None,
                memory_text=None,
                text_attention_mask=None,
                memory=None,
                memory_key_padding_mask=None,
                memory_level_start_index=None,
                memory_spatial_shapes=None,
                memory_pos=None,
                self_attn_mask=None,
                cross_attn_mask=None,
                dac=False,
                dac_use_selfatt_ln=True,
                presence_token=None,
                identity=0.0,
                **kwargs,
            ):
                if _sa is not None:
                    if dac:
                        h = tgt.shape[0] // 2
                        tgt_o2o, tgt_o2m = tgt[:h], tgt[h:]
                        qp_o2o = tgt_query_pos[:h]
                    else:
                        tgt_o2o = tgt
                        qp_o2o = tgt_query_pos
                    if presence_token is not None:
                        tgt_o2o = torch.cat([presence_token, tgt_o2o], dim=0)
                        zp = torch.zeros_like(presence_token)
                        qp_o2o = torch.cat([zp, qp_o2o], dim=0)
                        tgt_query_pos = torch.cat([zp, tgt_query_pos], dim=0)
                    q = k = tgt_o2o + qp_o2o
                    tgt2 = _sa(q, k, tgt_o2o, attn_mask=self_attn_mask)[0]
                    tgt_o2o = tgt_o2o + _d2(tgt2)
                    if dac:
                        if not dac_use_selfatt_ln:
                            tgt_o2o = _n2(tgt_o2o)
                        tgt = torch.cat((tgt_o2o, tgt_o2m), dim=0)
                        if dac_use_selfatt_ln:
                            tgt = _n2(tgt)
                    else:
                        tgt = _n2(tgt_o2o)
                if _txt:
                    tgt2 = _ca_txt(tgt + tgt_query_pos, memory_text, memory_text, key_padding_mask=text_attention_mask)[
                        0
                    ]
                    tgt = tgt + _ct_d(tgt2)
                    tgt = _ct_n(tgt)
                if presence_token is not None and cross_attn_mask is not None:
                    pp = getattr(lyr, "_pp_mask", None)
                    if pp is not None:
                        cross_attn_mask = pp
                    else:
                        cross_attn_mask = torch.cat(
                            [torch.zeros_like(cross_attn_mask[:, :1, :]), cross_attn_mask], dim=1
                        )
                tgt2 = _ca(
                    query=tgt + tgt_query_pos,
                    key=(memory + memory_pos) if memory_pos is not None else memory,
                    value=memory,
                    attn_mask=cross_attn_mask,
                    key_padding_mask=(
                        memory_key_padding_mask.transpose(0, 1) if memory_key_padding_mask is not None else None
                    ),
                )[0]
                tgt = tgt + _d1(tgt2)
                tgt = _n1(tgt)
                tgt = _ffn(tgt)
                pt_out = None
                if presence_token is not None:
                    pt_out = tgt[:1]
                    tgt = tgt[1:]
                return tgt, pt_out

            return _fwd

        for _lyr in decoder.layers:
            _lyr._pp_mask = None
            _lyr.forward = _make_opt_layer_fwd(_lyr)

        _mask_buf = [None]

        def _opt_dec_fwd(
            self_d,
            tgt,
            memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
            pos=None,
            reference_boxes=None,
            level_start_index=None,
            spatial_shapes=None,
            valid_ratios=None,
            memory_text=None,
            text_attention_mask=None,
            apply_dac=None,
            is_instance_prompt=False,
            decoder_extra_kwargs=None,
            obj_roi_memory_feat=None,
            obj_roi_memory_mask=None,
            box_head_trk=None,
        ):
            if memory_mask is not None:
                assert self_d.boxRPB == "none"
            apply_dac = apply_dac if apply_dac is not None else self_d.dac
            if apply_dac:
                tgt = tgt.repeat(2, 1, 1)
                if reference_boxes is not None:
                    reference_boxes = reference_boxes.repeat(2, 1, 1)
            bs = tgt.shape[1]
            intermediate = []
            intermediate_presence_logits = []
            presence_feats = None
            if self_d.box_refine:
                if reference_boxes is None:
                    reference_boxes = self_d.reference_points.weight.unsqueeze(1)
                    reference_boxes = (
                        reference_boxes.repeat(2, bs, 1) if apply_dac else reference_boxes.repeat(1, bs, 1)
                    ).sigmoid()
                intermediate_ref_boxes = [reference_boxes]
            else:
                reference_boxes = None
                intermediate_ref_boxes = None
            output = tgt
            presence_out = None
            if self_d.presence_token is not None and not is_instance_prompt:
                presence_out = self_d.presence_token.weight[None].expand(1, bs, -1)
            box_head = self_d.bbox_embed
            if is_instance_prompt and self_d.instance_bbox_embed is not None:
                box_head = self_d.instance_bbox_embed
            out_norm = self_d.norm
            if is_instance_prompt and self_d.instance_norm is not None:
                out_norm = self_d.instance_norm
            vr2 = torch.cat([valid_ratios, valid_ratios], -1)[None, :] if valid_ratios is not None else None
            fs = (
                (spatial_shapes[0, 0], spatial_shapes[0, 1])
                if spatial_shapes is not None and self_d.boxRPB != "none"
                else None
            )

            for li, layer in enumerate(self_d.layers):
                rpi = reference_boxes[:, :, None] * vr2
                qse = _cached_sineembed(rpi[:, :, 0, :], self_d.d_model)
                qp = self_d.ref_point_head(qse)
                if self_d.boxRPB != "none" and reference_boxes is not None:
                    memory_mask = self_d._get_rpb_matrix(reference_boxes, fs).flatten(0, 1)
                    if presence_out is not None:
                        nq = memory_mask.shape[1]
                        if _mask_buf[0] is None or _mask_buf[0].shape[1] != nq + 1:
                            _mask_buf[0] = torch.zeros(
                                memory_mask.shape[0], nq + 1, memory_mask.shape[2], dtype=memory_mask.dtype
                            )
                        _mask_buf[0][:, 1:, :] = memory_mask
                        layer._pp_mask = _mask_buf[0]
                    else:
                        layer._pp_mask = None
                output, presence_out = layer(
                    tgt=output,
                    tgt_query_pos=qp,
                    tgt_query_sine_embed=qse,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    tgt_reference_points=rpi,
                    memory_text=memory_text,
                    text_attention_mask=text_attention_mask,
                    memory=memory,
                    memory_key_padding_mask=memory_key_padding_mask,
                    memory_level_start_index=level_start_index,
                    memory_spatial_shapes=spatial_shapes,
                    memory_pos=pos,
                    self_attn_mask=tgt_mask,
                    cross_attn_mask=memory_mask,
                    dac=apply_dac,
                    dac_use_selfatt_ln=self_d.dac_use_selfatt_ln,
                    presence_token=presence_out,
                    **(decoder_extra_kwargs or {}),
                    obj_roi_memory_feat=obj_roi_memory_feat,
                    obj_roi_memory_mask=obj_roi_memory_mask,
                )
                layer._pp_mask = None
                if self_d.box_refine:
                    ref_bs = _inv_sig(reference_boxes)
                    normed = out_norm(output)
                    if box_head_trk is None:
                        du = box_head(normed) if self_d.use_normed_output_consistently else box_head(output)
                    else:
                        Q_det = decoder_extra_kwargs["Q_det"]
                        du = torch.cat([self_d.bbox_embed(output[:Q_det]), box_head_trk(output[Q_det:])], dim=0)
                    new_rp = (du + ref_bs).sigmoid()
                    reference_boxes = new_rp.detach()
                    if li != self_d.num_layers - 1:
                        intermediate_ref_boxes.append(new_rp)
                else:
                    normed = out_norm(output)
                intermediate.append(normed)
                if self_d.presence_token is not None and not is_instance_prompt:
                    ipl = self_d.presence_token_head(self_d.presence_token_out_norm(presence_out)).squeeze(-1)
                    if self_d.clamp_presence_logits:
                        ipl.clamp_(min=-self_d.clamp_presence_logit_max_val, max=self_d.clamp_presence_logit_max_val)
                    intermediate_presence_logits.append(ipl)
                    presence_feats = presence_out.clone()
            return (
                torch.stack(intermediate),
                torch.stack(intermediate_ref_boxes) if intermediate_ref_boxes is not None else None,
                (
                    torch.stack(intermediate_presence_logits)
                    if self_d.presence_token is not None and not is_instance_prompt
                    else None
                ),
                presence_feats,
            )

        import types

        decoder.forward = types.MethodType(_opt_dec_fwd, decoder)
        decoder.compiled = True

    def _patched_vit_forward(self, x):
        if self._encoder.forward is not self._patched_encoder_forward:
            self._encoder.forward = self._patched_encoder_forward
        if self._pixel_decoder.forward is not self._fast_pd_forward:
            self._pixel_decoder.forward = self._fast_pd_forward
        if self._mask_predictor.forward is not self._fast_mp_forward:
            self._mask_predictor.forward = self._fast_mp_forward
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
        self._mask_predictor.forward = self._orig_mp_forward
        self._decoder.forward = self._orig_decoder_forward
        for i, layer in enumerate(self._decoder.layers):
            layer.forward = self._orig_layer_forwards[i]

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
