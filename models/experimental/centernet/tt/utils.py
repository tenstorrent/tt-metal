# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from typing import List, Optional, Tuple
from typing import Dict, List, Optional, Tuple
from torch import Tensor
import torch
from mmengine.structures import InstanceData
import torch.nn.functional as F


def predict_by_feat(
    center_heatmap_preds: List[Tensor],
    wh_preds: List[Tensor],
    offset_preds: List[Tensor],
    batch_img_metas: Optional[List[dict]] = None,
    rescale: bool = True,
    with_nms: bool = False,
):
    assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1
    result_list = []
    for img_id in range(len(batch_img_metas)):
        result_list.append(
            _predict_by_feat_single(
                center_heatmap_preds[0][img_id : img_id + 1, ...],
                wh_preds[0][img_id : img_id + 1, ...],
                offset_preds[0][img_id : img_id + 1, ...],
                batch_img_metas[img_id],
                rescale=rescale,
                with_nms=with_nms,
            )
        )
    return result_list


def _predict_by_feat_single(
    center_heatmap_pred: Tensor,
    wh_pred: Tensor,
    offset_pred: Tensor,
    img_meta: dict,
    rescale: bool = True,
    with_nms: bool = False,
) -> InstanceData:
    batch_det_bboxes, batch_labels = _decode_heatmap(
        center_heatmap_pred,
        wh_pred,
        offset_pred,
        img_meta["batch_input_shape"],
        k=100,
        kernel=3,
    )

    det_bboxes = batch_det_bboxes.view([-1, 5])
    det_labels = batch_labels.view(-1)

    batch_border = det_bboxes.new_tensor(img_meta["border"])[..., [2, 0, 2, 0]]
    det_bboxes[..., :4] -= batch_border

    if rescale and "scale_factor" in img_meta:
        det_bboxes[..., :4] /= det_bboxes.new_tensor(img_meta["scale_factor"]).repeat((1, 2))

    results = InstanceData()
    results.bboxes = det_bboxes[..., :4]
    results.scores = det_bboxes[..., 4]
    results.labels = det_labels
    return results


def _decode_heatmap(
    center_heatmap_pred: Tensor, wh_pred: Tensor, offset_pred: Tensor, img_shape: tuple, k: int = 100, kernel: int = 3
) -> Tuple[Tensor, Tensor]:
    height, width = center_heatmap_pred.shape[2:]
    inp_h, inp_w = img_shape

    center_heatmap_pred = get_local_maximum(center_heatmap_pred, kernel=kernel)

    *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(center_heatmap_pred, k=k)
    batch_scores, batch_index, batch_topk_labels = batch_dets

    wh = transpose_and_gather_feat(wh_pred, batch_index)
    offset = transpose_and_gather_feat(offset_pred, batch_index)
    topk_xs = topk_xs + offset[..., 0]
    topk_ys = topk_ys + offset[..., 1]
    tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
    tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
    br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
    br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

    batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
    batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]), dim=-1)
    return batch_bboxes, batch_topk_labels


def get_local_maximum(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def get_topk_from_heatmap(scores, k=20):
    batch, _, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
    topk_clses = topk_inds // (height * width)
    topk_inds = topk_inds % (height * width)
    topk_ys = topk_inds // width
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat


def gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).repeat(1, 1, dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def batched_nms(
    boxes: Tensor, scores: Tensor, idxs: Tensor, nms_cfg: Optional[Dict], class_agnostic: bool = False
) -> Tuple[Tensor, Tensor]:
    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, inds = scores.sort(descending=True)
        boxes = boxes[inds]
        return torch.cat([boxes, scores[:, None]], -1), inds

    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop("class_agnostic", class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        if boxes.size(-1) == 5:
            max_coordinate = boxes[..., :2].max() + boxes[..., 2:4].max()
            offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
            boxes_ctr_for_nms = boxes[..., :2] + offsets[:, None]
            boxes_for_nms = torch.cat([boxes_ctr_for_nms, boxes[..., 2:5]], dim=-1)
        else:
            max_coordinate = boxes.max()
            offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
            boxes_for_nms = boxes + offsets[:, None]

    nms_op = nms_cfg_.pop("type", "nms")
    if isinstance(nms_op, str):
        nms_op = eval(nms_op)

    split_thr = nms_cfg_.pop("split_thr", 10000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_thr:
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]

        # This assumes `dets` has arbitrary dimensions where
        # the last dimension is score.
        # Currently it supports bounding boxes [x1, y1, x2, y2, score] or
        # rotated boxes [cx, cy, w, h, angle_radian, score].

        scores = dets[:, -1]
    else:
        max_num = nms_cfg_.pop("max_num", -1)
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        # Some type of nms would reweight the score, such as SoftNMS
        scores_after_nms = scores.new_zeros(scores.size())
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero(as_tuple=False).view(-1)

        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    boxes = torch.cat([boxes, scores[:, None]], -1)
    return boxes, keep
