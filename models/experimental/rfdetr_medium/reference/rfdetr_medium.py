# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch reference for RF-DETR Medium (detection only) with staged forward.
Stages are separated for component-level PCC comparison against TTNN implementation.

Stages:
  1. Backbone: DINOv2-ViT-S with windowed attention → 4 multi-scale feature maps
  2. Projector: MultiScaleProjector → fused feature map(s) at P4
  3. Two-stage proposal: enc_output → top-K query selection (300 queries)
  4. Decoder: 4 layers of self-attn + deformable cross-attn + FFN
  5. Detection heads: class_embed + bbox_embed → logits + boxes
  6. Post-processing: sigmoid, top-K, box decode
"""

import torch

from models.experimental.rfdetr_medium.common import (
    NUM_QUERIES,
)


def staged_forward_backbone(model, image_tensor):
    """
    Run backbone (DINOv2-ViT-S + projector) to get multi-scale features.

    Args:
        model: LWDETR model
        image_tensor: [B, 3, 576, 576] float32

    Returns:
        features: list of NestedTensor (src, mask) from backbone
        poss: list of positional encodings
        srcs: list of source feature tensors [B, C, H, W]
    """
    from rfdetr.util.misc import nested_tensor_from_tensor_list

    if isinstance(image_tensor, (list, torch.Tensor)):
        samples = nested_tensor_from_tensor_list(image_tensor)
    else:
        samples = image_tensor

    features, poss = model.backbone(samples)

    srcs = []
    masks = []
    for feat in features:
        src, mask = feat.decompose()
        srcs.append(src)
        masks.append(mask)

    return features, poss, srcs, masks, samples


def staged_forward_flatten(model, srcs, masks, poss):
    """
    Flatten multi-scale features for the transformer.

    Args:
        model: LWDETR model
        srcs: list of [B, C, H, W] feature tensors
        masks: list of [B, H, W] masks
        poss: list of positional encodings

    Returns:
        memory: [B, sum(H*W), C] flattened features
        mask_flatten: [B, sum(H*W)] flattened masks
        lvl_pos_embed_flatten: [B, sum(H*W), C] positional embeddings
        spatial_shapes: [num_levels, 2] tensor
        level_start_index: [num_levels] tensor
        valid_ratios: [B, num_levels, 2] tensor
    """
    transformer = model.transformer

    src_flatten = []
    mask_flatten = [] if masks[0] is not None else None
    lvl_pos_embed_flatten = []
    spatial_shapes = []
    valid_ratios = [] if masks[0] is not None else None

    for lvl, (src, pos_embed) in enumerate(zip(srcs, poss)):
        bs, c, h, w = src.shape
        spatial_shapes.append((h, w))
        src = src.flatten(2).transpose(1, 2)
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        lvl_pos_embed_flatten.append(pos_embed)
        src_flatten.append(src)
        if masks[lvl] is not None:
            mask = masks[lvl].flatten(1)
            mask_flatten.append(mask)

    memory = torch.cat(src_flatten, 1)
    if mask_flatten is not None:
        mask_flatten = torch.cat(mask_flatten, 1)
        valid_ratios = torch.stack([transformer.get_valid_ratio(m) for m in masks], 1)
    lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=memory.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

    return (
        memory,
        mask_flatten,
        lvl_pos_embed_flatten,
        spatial_shapes,
        level_start_index,
        valid_ratios,
    )


def staged_forward_two_stage(
    model,
    memory,
    mask_flatten,
    spatial_shapes,
):
    """
    Two-stage proposal generation: enc_output → top-K.

    Args:
        model: LWDETR model
        memory: [B, sum(H*W), C]
        mask_flatten: [B, sum(H*W)]
        spatial_shapes: [num_levels, 2]

    Returns:
        refpoint_embed_ts: [B, num_queries, 4] - reference points (unsigmoid)
        memory_ts: [B, num_queries, C] - selected memory features
        boxes_ts: [B, num_queries, 4] - proposal boxes
    """
    from rfdetr.models.transformer import gen_encoder_output_proposals

    transformer = model.transformer

    output_memory, output_proposals = gen_encoder_output_proposals(
        memory,
        mask_flatten,
        spatial_shapes,
        unsigmoid=not transformer.bbox_reparam,
    )

    refpoint_embed_ts_list, memory_ts_list, boxes_ts_list = [], [], []
    group_detr = 1  # inference uses 1 group

    for g_idx in range(group_detr):
        output_memory_gidx = transformer.enc_output_norm[g_idx](transformer.enc_output[g_idx](output_memory))
        enc_outputs_class = transformer.enc_out_class_embed[g_idx](output_memory_gidx)

        if transformer.bbox_reparam:
            enc_delta = transformer.enc_out_bbox_embed[g_idx](output_memory_gidx)
            enc_cxcy = enc_delta[..., :2] * output_proposals[..., 2:] + output_proposals[..., :2]
            enc_wh = enc_delta[..., 2:].exp() * output_proposals[..., 2:]
            enc_coord = torch.cat([enc_cxcy, enc_wh], dim=-1)
        else:
            enc_coord = transformer.enc_out_bbox_embed[g_idx](output_memory_gidx) + output_proposals

        topk = min(transformer.num_queries, enc_outputs_class.shape[-2])
        topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]

        refpoint_embed_gidx = torch.gather(enc_coord, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).detach()

        tgt_undetach_gidx = torch.gather(
            output_memory_gidx,
            1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, transformer.d_model),
        )

        refpoint_embed_ts_list.append(refpoint_embed_gidx)
        memory_ts_list.append(tgt_undetach_gidx)
        boxes_ts_list.append(torch.gather(enc_coord, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)))

    refpoint_embed_ts = torch.cat(refpoint_embed_ts_list, dim=1)
    memory_ts = torch.cat(memory_ts_list, dim=1)
    boxes_ts = torch.cat(boxes_ts_list, dim=1)

    return refpoint_embed_ts, memory_ts, boxes_ts


def staged_forward_decoder(
    model,
    memory,
    mask_flatten,
    lvl_pos_embed_flatten,
    spatial_shapes,
    level_start_index,
    valid_ratios,
    refpoint_embed_ts,
):
    """
    Run decoder layers.

    Args:
        model: LWDETR model
        memory, mask_flatten, etc.: from staged_forward_flatten
        refpoint_embed_ts: [B, num_queries, 4] from two-stage

    Returns:
        hs: [num_dec_layers, B, num_queries, hidden_dim] decoder outputs
        references: reference points per layer
    """
    transformer = model.transformer
    bs = memory.shape[0]

    query_feat_weight = model.query_feat.weight[:NUM_QUERIES]
    refpoint_embed_weight = model.refpoint_embed.weight[:NUM_QUERIES]

    tgt = query_feat_weight.unsqueeze(0).repeat(bs, 1, 1)
    refpoint_embed = refpoint_embed_weight.unsqueeze(0).repeat(bs, 1, 1)

    if transformer.two_stage:
        ts_len = refpoint_embed_ts.shape[-2]
        refpoint_embed_ts_subset = refpoint_embed[..., :ts_len, :]
        refpoint_embed_subset = refpoint_embed[..., ts_len:, :]

        if transformer.bbox_reparam:
            re_cxcy = refpoint_embed_ts_subset[..., :2] * refpoint_embed_ts[..., 2:] + refpoint_embed_ts[..., :2]
            re_wh = refpoint_embed_ts_subset[..., 2:].exp() * refpoint_embed_ts[..., 2:]
            refpoint_embed_ts_subset = torch.cat([re_cxcy, re_wh], dim=-1)
        else:
            refpoint_embed_ts_subset = refpoint_embed_ts_subset + refpoint_embed_ts

        refpoint_embed = torch.cat([refpoint_embed_ts_subset, refpoint_embed_subset], dim=-2)

    hs, references = transformer.decoder(
        tgt,
        memory,
        memory_key_padding_mask=mask_flatten,
        pos=lvl_pos_embed_flatten,
        refpoints_unsigmoid=refpoint_embed,
        level_start_index=level_start_index,
        spatial_shapes=spatial_shapes,
        valid_ratios=(valid_ratios.to(memory.dtype) if valid_ratios is not None else None),
    )

    return hs, references


def staged_forward_detection_heads(model, hs, references):
    """
    Run detection heads on decoder output.

    Args:
        model: LWDETR model
        hs: [num_dec_layers, B, num_queries, hidden_dim]
        references: reference points from decoder

    Returns:
        outputs_class: [num_dec_layers, B, num_queries, num_classes+1] logits
        outputs_coord: [num_dec_layers, B, num_queries, 4] boxes
    """
    if model.bbox_reparam:
        outputs_coord_delta = model.bbox_embed(hs)
        outputs_coord_cxcy = outputs_coord_delta[..., :2] * references[..., 2:] + references[..., :2]
        outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * references[..., 2:]
        outputs_coord = torch.cat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
    else:
        outputs_coord = (model.bbox_embed(hs) + references).sigmoid()

    outputs_class = model.class_embed(hs)

    return outputs_class, outputs_coord


def staged_forward_postprocess(outputs_class, outputs_coord, img_shape, score_thr=0.3):
    """
    Post-process detection outputs to get final bboxes.

    Args:
        outputs_class: [B, num_queries, num_classes+1] logits (last decoder layer)
        outputs_coord: [B, num_queries, 4] boxes (cxcywh format, normalized)
        img_shape: (H, W) original image size
        score_thr: score threshold

    Returns:
        list of dicts with 'boxes' [N, 4], 'scores' [N], 'labels' [N]
    """
    prob = outputs_class.sigmoid()
    scores, labels = prob.max(-1)

    results = []
    for b in range(outputs_class.shape[0]):
        mask = scores[b] > score_thr
        b_scores = scores[b][mask]
        b_labels = labels[b][mask]
        b_boxes = outputs_coord[b][mask]

        # cxcywh → xyxy
        cx, cy, w, h = b_boxes.unbind(-1)
        x1 = (cx - w / 2) * img_shape[1]
        y1 = (cy - h / 2) * img_shape[0]
        x2 = (cx + w / 2) * img_shape[1]
        y2 = (cy + h / 2) * img_shape[0]
        b_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        results.append(
            {
                "boxes": b_boxes_xyxy,
                "scores": b_scores,
                "labels": b_labels,
            }
        )

    return results


@torch.no_grad()
def full_reference_forward(model, image_tensor, score_thr=0.3):
    """
    Complete reference forward pass through all stages.

    Args:
        model: LWDETR model in eval mode
        image_tensor: [B, 3, 576, 576] float32

    Returns:
        dict with all intermediate results for PCC comparison
    """
    model.eval()

    # Stage 1: Backbone
    features, poss, srcs, masks, samples = staged_forward_backbone(model, image_tensor)

    # Stage 2: Flatten
    (
        memory,
        mask_flatten,
        lvl_pos_embed_flatten,
        spatial_shapes,
        level_start_index,
        valid_ratios,
    ) = staged_forward_flatten(model, srcs, masks, poss)

    # Stage 3: Two-stage proposal
    refpoint_embed_ts, memory_ts, boxes_ts = staged_forward_two_stage(model, memory, mask_flatten, spatial_shapes)

    # Stage 4: Decoder
    hs, references = staged_forward_decoder(
        model,
        memory,
        mask_flatten,
        lvl_pos_embed_flatten,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        refpoint_embed_ts,
    )

    # Stage 5: Detection heads
    outputs_class, outputs_coord = staged_forward_detection_heads(model, hs, references)

    # Stage 6: Post-processing
    img_h, img_w = image_tensor.shape[-2:]
    detections = staged_forward_postprocess(outputs_class[-1], outputs_coord[-1], (img_h, img_w), score_thr)

    return {
        "features": features,
        "srcs": srcs,
        "masks": masks,
        "poss": poss,
        "memory": memory,
        "mask_flatten": mask_flatten,
        "spatial_shapes": spatial_shapes,
        "level_start_index": level_start_index,
        "valid_ratios": valid_ratios,
        "refpoint_embed_ts": refpoint_embed_ts,
        "memory_ts": memory_ts,
        "boxes_ts": boxes_ts,
        "hs": hs,
        "references": references,
        "all_cls_scores": outputs_class,
        "all_bbox_preds": outputs_coord,
        "detections": detections,
    }
