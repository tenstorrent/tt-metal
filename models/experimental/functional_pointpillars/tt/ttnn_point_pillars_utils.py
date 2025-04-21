from typing import List, Tuple, Union, Optional, Sequence
import torch
from torch import Tensor
import mmengine
from abc import ABCMeta, abstractmethod
from functools import partial
import importlib
import numpy as np
from mmengine.structures import InstanceData
import ttnn

from torch import nn as nn


def load_ext(name, funcs):
    ext = importlib.import_module("mmcv." + name)
    for fun in funcs:
        assert hasattr(ext, fun), f"{fun} miss in module {name}"
    return ext


ext_module = load_ext("_ext", ["nms", "softnms", "nms_match", "nms_rotated", "nms_quadri"])

# def get_paddings_indicator(actual_num, max_num, axis: int = 0):
#     device = actual_num.device()
#     actual_num = ttnn.unsqueeze(actual_num, axis + 1)
#     # tiled_actual_num: [N, M, 1]
#     print(actual_num.shape)
#     max_num_shape = [1] * len(actual_num.shape)
#     print(max_num_shape)
#     max_num_shape[axis + 1] = -1
#     max_num = ttnn.reshape(ttnn.arange(0, max_num, device=actual_num.device(), dtype=ttnn.uint32), max_num_shape)

#     actual_num = ttnn.to_layout(actual_num, layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32)

#     # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
#     # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
#     # return max_num
#     actual_num = ttnn.to_torch(actual_num)
#     max_num = ttnn.to_torch(max_num)
#     paddings_indicator = actual_num > max_num

#     # paddings_indicator shape: [batch_size, max_num]
#     paddings_indicator = ttnn.from_torch(paddings_indicator, device=device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)
#     return paddings_indicator


def get_paddings_indicator(actual_num: Tensor, max_num: Tensor, axis: int = 0) -> Tensor:
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


def nms_normal_bev(boxes: Tensor, scores: Tensor, thresh: float) -> Tensor:
    assert boxes.shape[1] == 5, "Input boxes shape should be [N, 5]"
    return nms(boxes[:, :-1], scores, thresh)[1]


def nms_rotated(
    dets: Tensor, scores: Tensor, iou_threshold: float, labels: Optional[Tensor] = None, clockwise: bool = True
) -> Tuple[Tensor, Tensor]:
    if dets.shape[0] == 0:
        return dets, None
    if not clockwise:
        flip_mat = dets.new_ones(dets.shape[-1])
        flip_mat[-1] = -1
        dets_cw = dets * flip_mat
    else:
        dets_cw = dets
    multi_label = labels is not None
    if labels is None:
        input_labels = scores.new_empty(0, dtype=torch.int)
    else:
        input_labels = labels
    if dets.device.type in ("npu", "mlu"):
        order = scores.new_empty(0, dtype=torch.long)
        if dets.device.type == "npu":
            coefficient = 57.29578  # 180 / PI
            for i in range(dets.size()[0]):
                dets_cw[i][4] *= coefficient  # radians to angle
        keep_inds = nms_rotated(dets_cw, scores, order, dets_cw, input_labels, iou_threshold, multi_label)
        dets = torch.cat((dets[keep_inds], scores[keep_inds].reshape(-1, 1)), dim=1)
        return dets, keep_inds

    if multi_label:
        dets_wl = torch.cat((dets_cw, labels.unsqueeze(1)), 1)  # type: ignore
    else:
        dets_wl = dets_cw
    _, order = scores.sort(0, descending=True)
    dets_sorted = dets_wl.index_select(0, order)

    if torch.__version__ == "parrots":
        keep_inds = ext_module.nms_rotated(
            dets_wl, scores, order, dets_sorted, input_labels, iou_threshold=iou_threshold, multi_label=multi_label
        )
    else:
        keep_inds = ext_module.nms_rotated(
            dets_wl, scores, order, dets_sorted, input_labels, iou_threshold, multi_label
        )
    dets = torch.cat((dets[keep_inds], scores[keep_inds].reshape(-1, 1)), dim=1)
    return dets, keep_inds


class NMSop(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, bboxes: Tensor, scores: Tensor, iou_threshold: float, offset: int, score_threshold: float, max_num: int
    ) -> Tensor:
        is_filtering_by_score = score_threshold > 0
        if is_filtering_by_score:
            valid_mask = scores > score_threshold
            bboxes, scores = bboxes[valid_mask], scores[valid_mask]
            valid_inds = torch.nonzero(valid_mask, as_tuple=False).squeeze(dim=1)

        inds = nms(bboxes, scores, iou_threshold=float(iou_threshold), offset=offset)

        if max_num > 0:
            inds = inds[:max_num]
        if is_filtering_by_score:
            inds = valid_inds[inds]
        return inds


def nms(boxes, scores, iou_threshold: float, offset: int = 0, score_threshold: float = 0, max_num: int = -1):
    assert isinstance(boxes, (Tensor, np.ndarray))
    assert isinstance(scores, (Tensor, np.ndarray))
    is_numpy = False
    if isinstance(boxes, np.ndarray):
        is_numpy = True
        boxes = torch.from_numpy(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    assert boxes.size(1) == 4
    assert boxes.size(0) == scores.size(0)
    assert offset in (0, 1)

    inds = NMSop.apply(boxes, scores, iou_threshold, offset, score_threshold, max_num)
    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
    if is_numpy:
        dets = dets.cpu().numpy()
        inds = inds.cpu().numpy()
    return dets, inds


def nms_bev(
    boxes: Tensor,
    scores: Tensor,
    thresh: float,
    pre_max_size: Optional[int] = None,
    post_max_size: Optional[int] = None,
) -> Tensor:
    assert boxes.size(1) == 5, "Input boxes shape should be [N, 5]"
    order = scores.sort(0, descending=True)[1]
    if pre_max_size is not None:
        order = order[:pre_max_size]
    boxes = boxes[order].contiguous()
    scores = scores[order]

    # xyxyr -> back to xywhr
    # note: better skip this step before nms_bev call in the future
    boxes = torch.stack(
        (
            (boxes[:, 0] + boxes[:, 2]) / 2,
            (boxes[:, 1] + boxes[:, 3]) / 2,
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
            boxes[:, 4],
        ),
        dim=-1,
    )

    keep = nms_rotated(boxes, scores, thresh)[1]
    keep = order[keep]
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep


def box3d_multiclass_nms(
    mlvl_bboxes: Tensor,
    mlvl_bboxes_for_nms: Tensor,
    mlvl_scores: Tensor,
    score_thr: float,
    max_num: int,
    cfg: dict,
    mlvl_dir_scores: Optional[Tensor] = None,
    mlvl_attr_scores: Optional[Tensor] = None,
    mlvl_bboxes2d: Optional[Tensor] = None,
) -> Tuple[Tensor]:
    # do multi class nms
    # the fg class id range: [0, num_classes-1]
    num_classes = mlvl_scores.shape[1] - 1
    bboxes = []
    scores = []
    labels = []
    dir_scores = []
    attr_scores = []
    bboxes2d = []
    for i in range(0, num_classes):
        # get bboxes and scores of this class
        cls_inds = mlvl_scores[:, i] > score_thr
        if not cls_inds.any():
            continue

        _scores = mlvl_scores[cls_inds, i]
        _bboxes_for_nms = mlvl_bboxes_for_nms[cls_inds, :]

        if cfg["use_rotate_nms"]:
            nms_func = nms_bev
        else:
            nms_func = nms_normal_bev

        selected = nms_func(_bboxes_for_nms, _scores, cfg["nms_thr"])
        _mlvl_bboxes = mlvl_bboxes[cls_inds, :]
        bboxes.append(_mlvl_bboxes[selected])
        scores.append(_scores[selected])
        cls_label = mlvl_bboxes.new_full((len(selected),), i, dtype=torch.long)
        labels.append(cls_label)

        if mlvl_dir_scores is not None:
            _mlvl_dir_scores = mlvl_dir_scores[cls_inds]
            dir_scores.append(_mlvl_dir_scores[selected])
        if mlvl_attr_scores is not None:
            _mlvl_attr_scores = mlvl_attr_scores[cls_inds]
            attr_scores.append(_mlvl_attr_scores[selected])
        if mlvl_bboxes2d is not None:
            _mlvl_bboxes2d = mlvl_bboxes2d[cls_inds]
            bboxes2d.append(_mlvl_bboxes2d[selected])

    if bboxes:
        bboxes = torch.cat(bboxes, dim=0)
        scores = torch.cat(scores, dim=0)
        labels = torch.cat(labels, dim=0)
        if mlvl_dir_scores is not None:
            dir_scores = torch.cat(dir_scores, dim=0)
        if mlvl_attr_scores is not None:
            attr_scores = torch.cat(attr_scores, dim=0)
        if mlvl_bboxes2d is not None:
            bboxes2d = torch.cat(bboxes2d, dim=0)
        if bboxes.shape[0] > max_num:
            _, inds = scores.sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            scores = scores[inds]
            if mlvl_dir_scores is not None:
                dir_scores = dir_scores[inds]
            if mlvl_attr_scores is not None:
                attr_scores = attr_scores[inds]
            if mlvl_bboxes2d is not None:
                bboxes2d = bboxes2d[inds]
    else:
        bboxes = mlvl_scores.new_zeros((0, mlvl_bboxes.size(-1)))
        scores = mlvl_scores.new_zeros((0,))
        labels = mlvl_scores.new_zeros((0,), dtype=torch.long)
        if mlvl_dir_scores is not None:
            dir_scores = mlvl_scores.new_zeros((0,))
        if mlvl_attr_scores is not None:
            attr_scores = mlvl_scores.new_zeros((0,))
        if mlvl_bboxes2d is not None:
            bboxes2d = mlvl_scores.new_zeros((0, 4))

    results = (bboxes, scores, labels)

    if mlvl_dir_scores is not None:
        results = results + (dir_scores,)
    if mlvl_attr_scores is not None:
        results = results + (attr_scores,)
    if mlvl_bboxes2d is not None:
        results = results + (bboxes2d,)

    return results


def limit_period(
    val: Union[np.ndarray, Tensor], offset: float = 0.5, period: float = np.pi
) -> Union[np.ndarray, Tensor]:
    limited_val = val - torch.floor(val / period + offset) * period
    return limited_val


def xywhr2xyxyr(boxes_xywhr: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[..., 2] / 2
    half_h = boxes_xywhr[..., 3] / 2

    boxes[..., 0] = boxes_xywhr[..., 0] - half_w
    boxes[..., 1] = boxes_xywhr[..., 1] - half_h
    boxes[..., 2] = boxes_xywhr[..., 0] + half_w
    boxes[..., 3] = boxes_xywhr[..., 1] + half_h
    boxes[..., 4] = boxes_xywhr[..., 4]
    return boxes


def select_single_mlvl(mlvl_tensors, batch_id, detach=True):
    assert isinstance(mlvl_tensors, (list, tuple))
    num_levels = len(mlvl_tensors)

    if detach:
        mlvl_tensor_list = [mlvl_tensors[i][batch_id].detach() for i in range(num_levels)]
    else:
        mlvl_tensor_list = [mlvl_tensors[i][batch_id] for i in range(num_levels)]
    return mlvl_tensor_list


class TtAnchor3DRangeGenerator(object):
    def __init__(
        self,
        ranges: List[List[float]],
        sizes: List[List[float]] = [[3.9, 1.6, 1.56]],
        scales: List[int] = [1],
        rotations: List[float] = [0, 1.5707963],
        custom_values: Tuple[float] = (),
        reshape_out: bool = True,
        size_per_range: bool = True,
    ) -> None:
        assert mmengine.is_list_of(ranges, list)
        if size_per_range:
            if len(sizes) != len(ranges):
                assert len(ranges) == 1
                ranges = ranges * len(sizes)
            assert len(ranges) == len(sizes)
        else:
            assert len(ranges) == 1
        assert mmengine.is_list_of(sizes, list)
        assert isinstance(scales, list)

        self.sizes = sizes
        self.scales = scales
        self.ranges = ranges
        self.rotations = rotations
        self.custom_values = custom_values
        self.cached_anchors = None
        self.reshape_out = reshape_out
        self.size_per_range = size_per_range

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += f"anchor_range={self.ranges},\n"
        s += f"scales={self.scales},\n"
        s += f"sizes={self.sizes},\n"
        s += f"rotations={self.rotations},\n"
        s += f"reshape_out={self.reshape_out},\n"
        s += f"size_per_range={self.size_per_range})"
        return s

    @property
    def num_base_anchors(self) -> int:
        """int: Total number of base anchors in a feature grid."""
        num_rot = len(self.rotations)
        num_size = torch.tensor(self.sizes).reshape(-1, 3).size(0)
        return num_rot * num_size

    @property
    def num_levels(self) -> int:
        """int: Number of feature levels that the generator is applied to."""
        return len(self.scales)

    def grid_anchors(self, featmap_sizes: List[Tuple[int]], device: Union[str, torch.device] = "cuda") -> List[Tensor]:
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(featmap_sizes[i], self.scales[i], device=device)
            if self.reshape_out:
                anchors = anchors.reshape(-1, anchors.size(-1))
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_anchors(
        self, featmap_size: Tuple[int], scale: int, device: Union[str, torch.device] = "cuda"
    ) -> Tensor:
        # We reimplement the anchor generator using torch in cuda
        # torch: 0.6975 s for 1000 times
        # numpy: 4.3345 s for 1000 times
        # which is ~5 times faster than the numpy implementation
        if not self.size_per_range:
            return self.anchors_single_range(
                featmap_size, self.ranges[0], scale, self.sizes, self.rotations, device=device
            )

        mr_anchors = []
        for anchor_range, anchor_size in zip(self.ranges, self.sizes):
            mr_anchors.append(
                self.anchors_single_range(featmap_size, anchor_range, scale, anchor_size, self.rotations, device=device)
            )
        mr_anchors = torch.cat(mr_anchors, dim=-3)
        return mr_anchors

    def anchors_single_range(
        self,
        feature_size: Tuple[int],
        anchor_range: Union[Tensor, List[float]],
        scale: int = 1,
        sizes: Union[List[List[float]], List[float]] = [[3.9, 1.6, 1.56]],
        rotations: List[float] = [0, 1.5707963],
        device: Union[str, torch.device] = "cuda",
    ) -> Tensor:
        if len(feature_size) == 2:
            feature_size = [1, feature_size[0], feature_size[1]]
        anchor_range = torch.tensor(anchor_range, device=device)
        z_centers = torch.linspace(anchor_range[2], anchor_range[5], feature_size[0], device=device)
        y_centers = torch.linspace(anchor_range[1], anchor_range[4], feature_size[1], device=device)
        x_centers = torch.linspace(anchor_range[0], anchor_range[3], feature_size[2], device=device)
        sizes = torch.tensor(sizes, device=device).reshape(-1, 3) * scale
        rotations = torch.tensor(rotations, device=device)

        # torch.meshgrid default behavior is 'id', np's default is 'xy'
        rets = torch.meshgrid(x_centers, y_centers, z_centers, rotations)
        # torch.meshgrid returns a tuple rather than list
        rets = list(rets)
        tile_shape = [1] * 5
        tile_shape[-2] = int(sizes.shape[0])
        for i in range(len(rets)):
            rets[i] = rets[i].unsqueeze(-2).repeat(tile_shape).unsqueeze(-1)

        sizes = sizes.reshape([1, 1, 1, -1, 1, 3])
        tile_size_shape = list(rets[0].shape)
        tile_size_shape[3] = 1
        sizes = sizes.repeat(tile_size_shape)
        rets.insert(3, sizes)

        ret = torch.cat(rets, dim=-1).permute([2, 1, 0, 3, 4, 5])
        # [1, 200, 176, N, 2, 7] for kitti after permute

        if len(self.custom_values) > 0:
            custom_ndim = len(self.custom_values)
            custom = ret.new_zeros([*ret.shape[:-1], custom_ndim])
            # custom[:] = self.custom_values
            ret = torch.cat([ret, custom], dim=-1)
            # [1, 200, 176, N, 2, 9] for nus dataset after permute
        return ret


class BaseBBoxCoder(metaclass=ABCMeta):
    # The size of the last of dimension of the encoded tensor.
    encode_size = 4

    def __init__(self, use_box_type: bool = False, **kwargs):
        self.use_box_type = use_box_type

    @abstractmethod
    def encode(self, bboxes, gt_bboxes):
        """Encode deltas between bboxes and ground truth boxes."""

    @abstractmethod
    def decode(self, bboxes, bboxes_pred):
        """Decode the predicted bboxes according to prediction and base
        boxes."""


class TtDeltaXYZWLHRBBoxCoder(BaseBBoxCoder):
    """Bbox Coder for 3D boxes.

    Args:
        code_size (int): The dimension of boxes to be encoded.
    """

    def __init__(self, code_size: int = 7) -> None:
        super(TtDeltaXYZWLHRBBoxCoder, self).__init__()
        self.code_size = code_size

    @staticmethod
    def encode(src_boxes: Tensor, dst_boxes: Tensor) -> Tensor:
        box_ndim = src_boxes.shape[-1]
        cas, cgs, cts = [], [], []
        if box_ndim > 7:
            xa, ya, za, wa, la, ha, ra, *cas = torch.split(src_boxes, 1, dim=-1)
            xg, yg, zg, wg, lg, hg, rg, *cgs = torch.split(dst_boxes, 1, dim=-1)
            cts = [g - a for g, a in zip(cgs, cas)]
        else:
            xa, ya, za, wa, la, ha, ra = torch.split(src_boxes, 1, dim=-1)
            xg, yg, zg, wg, lg, hg, rg = torch.split(dst_boxes, 1, dim=-1)
        za = za + ha / 2
        zg = zg + hg / 2
        diagonal = torch.sqrt(la**2 + wa**2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha
        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)
        rt = rg - ra
        return torch.cat([xt, yt, zt, wt, lt, ht, rt, *cts], dim=-1)

    @staticmethod
    def decode(anchors: Tensor, deltas: Tensor) -> Tensor:
        cas, cts = [], []
        box_ndim = anchors.shape[-1]
        if box_ndim > 7:
            xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
            xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(deltas, 1, dim=-1)
        else:
            xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
            xt, yt, zt, wt, lt, ht, rt = torch.split(deltas, 1, dim=-1)

        za = za + ha / 2
        diagonal = torch.sqrt(la**2 + wa**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, wg, lg, hg, rg, *cgs], dim=-1)


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class TtBase3DDenseHead(nn.Module, metaclass=ABCMeta):
    def __init__(self, init_cfg=None) -> None:
        super().__init__()

    def predict(self, x: Tuple[Tensor], batch_data_samples, rescale: bool = False):
        # batch_input_metas = [data_samples.metainfo for data_samples in batch_data_samples]
        batch_input_metas = batch_data_samples  # modified and passed the input
        outs = self(x)
        predictions = self.predict_by_feat(*outs, batch_input_metas=batch_input_metas, rescale=rescale)
        return predictions

    def predict_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        dir_cls_preds: List[Tensor],
        batch_input_metas: Optional[List[dict]] = None,
        cfg=None,
        rescale: bool = False,
        **kwargs,
    ):
        for i in range(len(cls_scores)):
            cls_scores[i] = ttnn.to_torch(cls_scores[i])
            cls_scores[i] = cls_scores[i].permute(0, 3, 1, 2)
            cls_scores[i] = cls_scores[i].to(dtype=torch.float)
        for i in range(len(bbox_preds)):
            bbox_preds[i] = ttnn.to_torch(bbox_preds[i])
            bbox_preds[i] = bbox_preds[i].permute(0, 3, 1, 2)
            bbox_preds[i] = bbox_preds[i].to(dtype=torch.float)
        for i in range(len(dir_cls_preds)):
            dir_cls_preds[i] = ttnn.to_torch(dir_cls_preds[i])
            dir_cls_preds[i] = dir_cls_preds[i].permute(0, 3, 1, 2)

        assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(dir_cls_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_anchors(featmap_sizes, device=cls_scores[0].device)
        mlvl_priors = [prior.reshape(-1, self.box_code_size) for prior in mlvl_priors]

        result_list = []

        for input_id in range(len(batch_input_metas)):
            input_meta = batch_input_metas[input_id]
            cls_score_list = select_single_mlvl(cls_scores, input_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, input_id)
            dir_cls_pred_list = select_single_mlvl(dir_cls_preds, input_id)

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                dir_cls_pred_list=dir_cls_pred_list,
                mlvl_priors=mlvl_priors,
                input_meta=input_meta,
                cfg=cfg,
                rescale=rescale,
                **kwargs,
            )
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(
        self,
        cls_score_list: List[Tensor],
        bbox_pred_list: List[Tensor],
        dir_cls_pred_list: List[Tensor],
        mlvl_priors: List[Tensor],
        input_meta: dict,
        cfg,
        rescale: bool = False,
        **kwargs,
    ):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_priors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        for cls_score, bbox_pred, dir_cls_pred, priors in zip(
            cls_score_list, bbox_pred_list, dir_cls_pred_list, mlvl_priors
        ):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert cls_score.size()[-2:] == dir_cls_pred.size()[-2:]
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]

            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, self.box_code_size)

            nms_pre = cfg.get("nms_pre", -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                priors = priors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]

            bboxes = self.bbox_coder.decode(priors, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta["box_type_3d"](mlvl_bboxes, box_dim=self.box_code_size).bev)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)

        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        score_thr = cfg.get("score_thr", 0)
        results = box3d_multiclass_nms(
            mlvl_bboxes, mlvl_bboxes_for_nms, mlvl_scores, score_thr, cfg["max_num"], cfg, mlvl_dir_scores
        )
        bboxes, scores, labels, dir_scores = results
        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset, self.dir_limit_offset, np.pi)
            bboxes[..., 6] = dir_rot + self.dir_offset + np.pi * dir_scores.to(bboxes.dtype)
        bboxes = input_meta["box_type_3d"](bboxes, box_dim=self.box_code_size)
        results = InstanceData()
        results.bboxes_3d = bboxes
        results.scores_3d = scores
        results.labels_3d = labels

        return results


class TtAlignedAnchor3DRangeGenerator(TtAnchor3DRangeGenerator):
    def __init__(self, align_corner: bool = False, **kwargs) -> None:
        super(TtAlignedAnchor3DRangeGenerator, self).__init__(**kwargs)
        self.align_corner = align_corner

    def anchors_single_range(
        self,
        feature_size: List[int],
        anchor_range: List[float],
        scale: int,
        sizes: Union[List[List[float]], List[float]] = [[3.9, 1.6, 1.56]],
        rotations: List[float] = [0, 1.5707963],
        device: Union[str, torch.device] = "cuda",
    ):
        if len(feature_size) == 2:
            feature_size = [1, feature_size[0], feature_size[1]]
        anchor_range = torch.tensor(anchor_range, device=device)
        z_centers = torch.linspace(anchor_range[2], anchor_range[5], feature_size[0] + 1, device=device)
        y_centers = torch.linspace(anchor_range[1], anchor_range[4], feature_size[1] + 1, device=device)
        x_centers = torch.linspace(anchor_range[0], anchor_range[3], feature_size[2] + 1, device=device)
        sizes = torch.tensor(sizes, device=device).reshape(-1, 3) * scale
        rotations = torch.tensor(rotations, device=device)

        # shift the anchor center
        if not self.align_corner:
            z_shift = (z_centers[1] - z_centers[0]) / 2
            y_shift = (y_centers[1] - y_centers[0]) / 2
            x_shift = (x_centers[1] - x_centers[0]) / 2
            z_centers += z_shift
            y_centers += y_shift
            x_centers += x_shift

        # torch.meshgrid default behavior is 'id', np's default is 'xy'
        rets = torch.meshgrid(
            x_centers[: feature_size[2]], y_centers[: feature_size[1]], z_centers[: feature_size[0]], rotations
        )

        # torch.meshgrid returns a tuple rather than list
        rets = list(rets)
        tile_shape = [1] * 5
        tile_shape[-2] = int(sizes.shape[0])
        for i in range(len(rets)):
            rets[i] = rets[i].unsqueeze(-2).repeat(tile_shape).unsqueeze(-1)

        sizes = sizes.reshape([1, 1, 1, -1, 1, 3])
        tile_size_shape = list(rets[0].shape)
        tile_size_shape[3] = 1
        sizes = sizes.repeat(tile_size_shape)
        rets.insert(3, sizes)

        ret = torch.cat(rets, dim=-1).permute([2, 1, 0, 3, 4, 5])

        if len(self.custom_values) > 0:
            custom_ndim = len(self.custom_values)
            custom = ret.new_zeros([*ret.shape[:-1], custom_ndim])
            # TODO: check the support of custom values
            # custom[:] = self.custom_values
            ret = torch.cat([ret, custom], dim=-1)
        return ret


# taken from mmdet3d/structures/bbox_3d/base_box3d.py
class TtBaseInstance3DBoxes:
    YAW_AXIS: int = 0

    def __init__(
        self,
        tensor: Union[Tensor, np.ndarray, Sequence[Sequence[float]]],
        box_dim: int = 7,
        with_yaw: bool = True,
        origin: Tuple[float, float, float] = (0.5, 0.5, 0),
    ) -> None:
        if isinstance(tensor, Tensor):
            device = tensor.device
        else:
            device = torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does
            # not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, box_dim))
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, (
            "The box dimension must be 2 and the length of the last "
            f"dimension must be {box_dim}, but got boxes with shape "
            f"{tensor.shape}."
        )

        if tensor.shape[-1] == 6:
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 1)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = tensor.clone()

        if origin != (0.5, 0.5, 0):
            dst = self.tensor.new_tensor((0.5, 0.5, 0))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    def __getitem__(self, item: Union[int, slice, np.ndarray, Tensor]) -> "TtBaseInstance3DBoxes":
        original_type = type(self)
        if isinstance(item, int):
            return original_type(self.tensor[item].view(1, -1), box_dim=self.box_dim, with_yaw=self.with_yaw)
        b = self.tensor[item]
        assert b.dim() == 2, f"Indexing on Boxes with {item} failed to return a matrix!"
        return original_type(b, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def to(self, device: Union[str, torch.device], *args, **kwargs) -> "TtBaseInstance3DBoxes":
        original_type = type(self)
        return original_type(self.tensor.to(device, *args, **kwargs), box_dim=self.box_dim, with_yaw=self.with_yaw)

    @property
    def bev(self) -> Tensor:
        """Tensor: 2D BEV box of each box with rotation in XYWHR format, in
        shape (N, 5)."""
        return self.tensor[:, [0, 1, 3, 4, 6]]

    def __len__(self) -> int:
        """int: Number of boxes in the current object."""
        return self.tensor.shape[0]


# taken from mmdet3d/structures/bbox_3d/lidar_box3d.py, removed all the functions from this class as they are not  invoked
class TtLiDARInstance3DBoxes(TtBaseInstance3DBoxes):
    YAW_AXIS = 2
