import os
import cv2
import torch
import mmcv
from copy import deepcopy
from mmdet3d.registry import TRANSFORMS
import mmengine
import numpy as np
from mmdet3d.datasets.transforms.loading import LoadAnnotations3D
from mmengine.dataset import Compose
from models.experimental.uniad.demo.data_container import DataContainer as DC
from mmdet3d.datasets.transforms.formating import to_tensor


def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height

    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position Bird's-eye view first element
        bev_dimension Bird's-eye view tensor spatial dimension
    """
    bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor(
        [(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.long
    )

    return bev_resolution, bev_start_position, bev_dimension


@TRANSFORMS.register_module()
class CustomCollect3D(object):
    """Collect data from the loader relevant to the specific task.
    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".
    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:
        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.
    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(
        self,
        keys,
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "lidar2img",
            "depth2img",
            "cam2img",
            "pad_shape",
            "scale_factor",
            "flip",
            "pcd_horizontal_flip",
            "pcd_vertical_flip",
            "box_mode_3d",
            "box_type_3d",
            "img_norm_cfg",
            "pcd_trans",
            "sample_idx",
            "prev_idx",
            "next_idx",
            "pcd_scale_factor",
            "pcd_rotation",
            "pts_filename",
            "transformation_3d_flow",
            "scene_token",
            "can_bus",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        # print("self.keysself.keys",self.keys[0])
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.
        Args:
            results (dict): Result dict contains the data to collect.
        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """

        data = {}
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data["img_metas"] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + f"(keys={self.keys}, meta_keys={self.meta_keys})"


class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __init__(
        self,
    ):
        return

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if "img" in results:
            if isinstance(results["img"], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results["img"]]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results["img"] = DC(to_tensor([imgs]), stack=False)

            else:
                img = np.ascontiguousarray(results["img"].transpose(2, 0, 1))
                results["img"] = DC(to_tensor([img]), stack=True)
        for key in [
            "proposals",
            "gt_bboxes",
            "gt_bboxes_ignore",
            "gt_labels",
            "gt_labels_3d",
            "attr_labels",
            "pts_instance_mask",
            "pts_semantic_mask",
            "centers2d",
            "depths",
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))
        if "gt_bboxes_3d" in results:
            if isinstance(results["gt_bboxes_3d"], BaseInstance3DBoxes):
                results["gt_bboxes_3d"] = DC(results["gt_bboxes_3d"], cpu_only=True)
            else:
                results["gt_bboxes_3d"] = DC(to_tensor(results["gt_bboxes_3d"]))

        if "gt_masks" in results:
            results["gt_masks"] = DC(results["gt_masks"], cpu_only=True)
        if "gt_semantic_seg" in results:
            results["gt_semantic_seg"] = DC(to_tensor(results["gt_semantic_seg"][None, ...]), stack=True)

        return results

    def __repr__(self):
        return self.__class__.__name__


@TRANSFORMS.register_module()
class CustomDefaultFormatBundle3D(DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, class_names, with_gt=True, with_label=True):
        super(CustomDefaultFormatBundle3D, self).__init__()
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if "points" in results:
            assert isinstance(results["points"], BasePoints)
            results["points"] = DC(results["points"].tensor)

        for key in ["voxels", "coors", "voxel_centers", "num_points"]:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]), stack=False)

        if self.with_gt:
            # Clean GT bboxes in the final
            if "gt_bboxes_3d_mask" in results:
                gt_bboxes_3d_mask = results["gt_bboxes_3d_mask"]
                results["gt_bboxes_3d"] = results["gt_bboxes_3d"][gt_bboxes_3d_mask]
                if "gt_names_3d" in results:
                    results["gt_names_3d"] = results["gt_names_3d"][gt_bboxes_3d_mask]
                if "centers2d" in results:
                    results["centers2d"] = results["centers2d"][gt_bboxes_3d_mask]
                if "depths" in results:
                    results["depths"] = results["depths"][gt_bboxes_3d_mask]
            if "gt_bboxes_mask" in results:
                gt_bboxes_mask = results["gt_bboxes_mask"]
                if "gt_bboxes" in results:
                    results["gt_bboxes"] = results["gt_bboxes"][gt_bboxes_mask]
                results["gt_names"] = results["gt_names"][gt_bboxes_mask]
            if self.with_label:
                if "gt_names" in results and len(results["gt_names"]) == 0:
                    results["gt_labels"] = np.array([], dtype=np.int64)
                    results["attr_labels"] = np.array([], dtype=np.int64)
                elif "gt_names" in results and isinstance(results["gt_names"][0], list):
                    # gt_labels might be a list of list in multi-view setting
                    results["gt_labels"] = [
                        np.array([self.class_names.index(n) for n in res], dtype=np.int64)
                        for res in results["gt_names"]
                    ]
                elif "gt_names" in results:
                    results["gt_labels"] = np.array(
                        [self.class_names.index(n) for n in results["gt_names"]], dtype=np.int64
                    )
                # we still assume one pipeline for one frame LiDAR
                # thus, the 3D name is list[string]
                if "gt_names_3d" in results:
                    results["gt_labels_3d"] = np.array(
                        [self.class_names.index(n) for n in results["gt_names_3d"]], dtype=np.int64
                    )
        results = super(CustomDefaultFormatBundle3D, self).__call__(results)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(class_names={self.class_names}, "
        repr_str += f"with_gt={self.with_gt}, with_label={self.with_label})"
        return repr_str


@TRANSFORMS.register_module()
class CustomMultiScaleFlipAug3D(object):
    """Test-time augmentation with multiple scales and flipping.

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple]: Images scales for resizing.
        pts_scale_ratio (float | list[float]): Points scale ratios for
            resizing.
        flip (bool, optional): Whether apply flip augmentation.
            Defaults to False.
        flip_direction (str | list[str], optional): Flip augmentation
            directions for images, options are "horizontal" and "vertical".
            If flip_direction is list, multiple flip augmentations will
            be applied. It has no effect when ``flip == False``.
            Defaults to "horizontal".
        pcd_horizontal_flip (bool, optional): Whether apply horizontal
            flip augmentation to point cloud. Defaults to True.
            Note that it works only when 'flip' is turned on.
        pcd_vertical_flip (bool, optional): Whether apply vertical flip
            augmentation to point cloud. Defaults to True.
            Note that it works only when 'flip' is turned on.
    """

    def __init__(
        self,
        transforms,
        img_scale,
        pts_scale_ratio,
        flip=False,
        flip_direction="horizontal",
        pcd_horizontal_flip=False,
        pcd_vertical_flip=False,
    ):
        self.transforms = Compose(transforms)
        self.img_scale = img_scale if isinstance(img_scale, list) else [img_scale]
        self.pts_scale_ratio = pts_scale_ratio if isinstance(pts_scale_ratio, list) else [float(pts_scale_ratio)]

        assert mmengine.is_list_of(self.img_scale, tuple)
        assert mmengine.is_list_of(self.pts_scale_ratio, float)

        self.flip = flip
        self.pcd_horizontal_flip = pcd_horizontal_flip
        self.pcd_vertical_flip = pcd_vertical_flip

        self.flip_direction = flip_direction if isinstance(flip_direction, list) else [flip_direction]
        assert mmengine.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ["horizontal"]:
            warnings.warn("flip_direction has no effect when flip is set to False")
        if self.flip and not any([(t["type"] == "RandomFlip3D" or t["type"] == "RandomFlip") for t in transforms]):
            warnings.warn("flip has no effect when RandomFlip is not in transforms")

    def __call__(self, results):
        """Call function to augment common fields in results.

        Args:
            results (dict): Result dict contains the data to augment.

        Returns:
            dict: The result dict contains the data that is augmented with
                different scales and flips.
        """
        aug_data = []

        # modified from `flip_aug = [False, True] if self.flip else [False]`
        # to reduce unnecessary scenes when using double flip augmentation
        # during test time
        flip_aug = [True] if self.flip else [False]
        pcd_horizontal_flip_aug = [False, True] if self.flip and self.pcd_horizontal_flip else [False]
        pcd_vertical_flip_aug = [False, True] if self.flip and self.pcd_vertical_flip else [False]
        for scale in self.img_scale:
            for pts_scale_ratio in self.pts_scale_ratio:
                for flip in flip_aug:
                    for pcd_horizontal_flip in pcd_horizontal_flip_aug:
                        for pcd_vertical_flip in pcd_vertical_flip_aug:
                            for direction in self.flip_direction:
                                # results.copy will cause bug
                                # since it is shallow copy
                                _results = deepcopy(results)
                                _results["scale"] = scale
                                _results["flip"] = flip
                                _results["pcd_scale_factor"] = pts_scale_ratio
                                _results["flip_direction"] = direction
                                _results["pcd_horizontal_flip"] = pcd_horizontal_flip
                                _results["pcd_vertical_flip"] = pcd_vertical_flip
                                data = self.transforms(_results)
                                aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(transforms={self.transforms}, "
        repr_str += f"img_scale={self.img_scale}, flip={self.flip}, "
        repr_str += f"pts_scale_ratio={self.pts_scale_ratio}, "
        repr_str += f"flip_direction={self.flip_direction})"
        return repr_str


@TRANSFORMS.register_module()
class CustomGenerateOccFlowLabels(object):
    def __init__(self, grid_conf, ignore_index=255, only_vehicle=True, filter_invisible=True, deal_instance_255=False):
        self.grid_conf = grid_conf
        self.bev_resolution, self.bev_start_position, self.bev_dimension = calculate_birds_eye_view_parameters(
            grid_conf["xbound"],
            grid_conf["ybound"],
            grid_conf["zbound"],
        )
        # convert numpy
        self.bev_resolution = self.bev_resolution.numpy()
        self.bev_start_position = self.bev_start_position.numpy()
        self.bev_dimension = self.bev_dimension.numpy()
        self.spatial_extent = (grid_conf["xbound"][1], grid_conf["ybound"][1])
        self.ignore_index = ignore_index
        self.only_vehicle = only_vehicle
        self.filter_invisible = filter_invisible
        self.deal_instance_255 = deal_instance_255
        assert self.deal_instance_255 is False

        nusc_classes = [
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone",
        ]
        vehicle_classes = ["car", "bus", "construction_vehicle", "bicycle", "motorcycle", "truck", "trailer"]
        plan_classes = vehicle_classes + ["pedestrian"]

        self.vehicle_cls_ids = np.array([nusc_classes.index(cls_name) for cls_name in vehicle_classes])

        self.plan_cls_ids = np.array([nusc_classes.index(cls_name) for cls_name in plan_classes])

        if only_vehicle:
            self.filter_cls_ids = self.vehicle_cls_ids
        else:
            self.filter_cls_ids = self.plan_cls_ids

    def reframe_boxes(self, boxes, t_init, t_curr):
        l2e_r_mat_curr = t_curr["l2e_r"]
        l2e_t_curr = t_curr["l2e_t"]
        e2g_r_mat_curr = t_curr["e2g_r"]
        e2g_t_curr = t_curr["e2g_t"]

        l2e_r_mat_init = t_init["l2e_r"]
        l2e_t_init = t_init["l2e_t"]
        e2g_r_mat_init = t_init["e2g_r"]
        e2g_t_init = t_init["e2g_t"]

        # to bbox under curr ego frame  # TODO: Uncomment
        boxes.rotate(l2e_r_mat_curr.T)
        boxes.translate(l2e_t_curr)

        # to bbox under world frame
        boxes.rotate(e2g_r_mat_curr.T)
        boxes.translate(e2g_t_curr)

        # to bbox under initial ego frame, first inverse translate, then inverse rotate
        boxes.translate(-e2g_t_init)
        m1 = np.linalg.inv(e2g_r_mat_init)
        boxes.rotate(m1.T)

        # to bbox under curr ego frame, first inverse translate, then inverse rotate
        boxes.translate(-l2e_t_init)
        m2 = np.linalg.inv(l2e_r_mat_init)
        boxes.rotate(m2.T)

        return boxes

    def __call__(self, results):
        """
        # Given lidar frame bboxes for curr frame and each future frame,
        # generate segmentation, instance, centerness, offset, and fwd flow map
        """
        # Avoid ignoring obj with index = self.ignore_index
        SPECIAL_INDEX = -20
        all_gt_bboxes_3d = results["future_gt_bboxes_3d"]
        all_gt_labels_3d = results["future_gt_labels_3d"]
        all_gt_inds = results["future_gt_inds"]
        all_vis_tokens = results["future_gt_vis_tokens"]
        num_frame = len(all_gt_bboxes_3d)

        # motion related transforms, of seq lengths
        l2e_r_mats = results["occ_l2e_r_mats"]
        l2e_t_vecs = results["occ_l2e_t_vecs"]
        e2g_r_mats = results["occ_e2g_r_mats"]
        e2g_t_vecs = results["occ_e2g_t_vecs"]

        # reference frame transform
        t_ref = dict(l2e_r=l2e_r_mats[0], l2e_t=l2e_t_vecs[0], e2g_r=e2g_r_mats[0], e2g_t=e2g_t_vecs[0])

        segmentations = []
        instances = []
        gt_future_boxes = []
        gt_future_labels = []

        # num_frame is 5
        for i in range(num_frame):
            # bbox, label, index of curr frame
            gt_bboxes_3d, gt_labels_3d = all_gt_bboxes_3d[i], all_gt_labels_3d[i]
            ins_inds = all_gt_inds[i]
            vis_tokens = all_vis_tokens[i]

            if gt_bboxes_3d is None:
                # for invalid samples, no loss calculated
                segmentation = np.ones((self.bev_dimension[1], self.bev_dimension[0])) * self.ignore_index
                instance = np.ones((self.bev_dimension[1], self.bev_dimension[0])) * self.ignore_index
            else:
                # reframe bboxes to reference frame
                t_curr = dict(l2e_r=l2e_r_mats[i], l2e_t=l2e_t_vecs[i], e2g_r=e2g_r_mats[i], e2g_t=e2g_t_vecs[i])
                ref_bboxes_3d = self.reframe_boxes(gt_bboxes_3d, t_ref, t_curr)
                gt_future_boxes.append(ref_bboxes_3d)
                gt_future_labels.append(gt_labels_3d)

                # for valid samples
                segmentation = np.zeros((self.bev_dimension[1], self.bev_dimension[0]))
                instance = np.zeros((self.bev_dimension[1], self.bev_dimension[0]))

                if self.only_vehicle:
                    vehicle_mask = np.isin(gt_labels_3d, self.filter_cls_ids)
                    ref_bboxes_3d = ref_bboxes_3d[vehicle_mask]
                    gt_labels_3d = gt_labels_3d[vehicle_mask]
                    ins_inds = ins_inds[vehicle_mask]
                    if vis_tokens is not None:
                        vis_tokens = vis_tokens[vehicle_mask]

                if self.filter_invisible:
                    assert vis_tokens is not None
                    visible_mask = vis_tokens != 1  # obj are filtered out with visibility(1) between 0 and 40%
                    ref_bboxes_3d = ref_bboxes_3d[visible_mask]
                    gt_labels_3d = gt_labels_3d[visible_mask]
                    ins_inds = ins_inds[visible_mask]

                # valid sample and has objects
                if len(ref_bboxes_3d.tensor) > 0:
                    bbox_corners = ref_bboxes_3d.corners[:, [0, 3, 7, 4], :2].numpy()
                    bbox_corners = np.round(
                        (bbox_corners - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0)
                        / self.bev_resolution[:2]
                    ).astype(np.int32)

                    for index, gt_ind in enumerate(ins_inds):
                        if gt_ind == self.ignore_index:
                            gt_ind = SPECIAL_INDEX  # 255 -> -20
                        poly_region = bbox_corners[index]

                        cv2.fillPoly(segmentation, [poly_region], 1.0)
                        cv2.fillPoly(instance, [poly_region], int(gt_ind))

            segmentations.append(segmentation)
            instances.append(instance)

        # segmentation = 1 where objects are located
        segmentations = torch.from_numpy(np.stack(segmentations, axis=0)).long()
        instances = torch.from_numpy(np.stack(instances, axis=0)).long()

        # generate heatmap & offset from segmentation & instance
        instance_centerness, instance_offset, instance_flow, instance_backward_flow = self.center_offset_flow(
            instances,
            all_gt_inds,
            ignore_index=255,
        )

        invalid_mask = segmentations[:, 0, 0] == self.ignore_index
        instance_centerness[invalid_mask] = self.ignore_index
        results["gt_occ_has_invalid_frame"] = results.pop("occ_has_invalid_frame")
        results["gt_occ_img_is_valid"] = results.pop("occ_img_is_valid")
        results.update(
            {
                "gt_segmentation": segmentations,
                "gt_instance": instances,
                "gt_centerness": instance_centerness,
                "gt_offset": instance_offset,
                "gt_flow": instance_flow,
                "gt_backward_flow": instance_backward_flow,
                "gt_future_boxes": gt_future_boxes,
                "gt_future_labels": gt_future_labels,
            }
        )
        return results

    def center_offset_flow(self, instance_img, all_gt_inds, ignore_index=255, sigma=3.0):
        seq_len, h, w = instance_img.shape
        # heatmap
        center_label = torch.zeros(seq_len, 1, h, w)
        # offset from parts to centers
        offset_label = ignore_index * torch.ones(seq_len, 2, h, w)
        # future flow
        future_displacement_label = ignore_index * torch.ones(seq_len, 2, h, w)

        # backward flow
        backward_flow = ignore_index * torch.ones(seq_len, 2, h, w)

        # x is vertical displacement, y is horizontal displacement
        x, y = torch.meshgrid(torch.arange(h, dtype=torch.float), torch.arange(w, dtype=torch.float))

        gt_inds_all = []
        for ins_inds_per_frame in all_gt_inds:
            if ins_inds_per_frame is None:
                continue
            for ins_ind in ins_inds_per_frame:
                gt_inds_all.append(ins_ind)
        gt_inds_unique = np.unique(np.array(gt_inds_all))

        # iterate over all instances across this sequence
        for instance_id in gt_inds_unique:
            instance_id = int(instance_id)
            prev_xc = None
            prev_yc = None
            prev_mask = None
            for t in range(seq_len):
                instance_mask = instance_img[t] == instance_id
                if instance_mask.sum() == 0:
                    # this instance is not in this frame
                    prev_xc = None
                    prev_yc = None
                    prev_mask = None
                    continue

                # the Bird-Eye-View center of the instance
                xc = x[instance_mask].mean()
                yc = y[instance_mask].mean()

                off_x = xc - x
                off_y = yc - y
                g = torch.exp(-(off_x**2 + off_y**2) / sigma**2)
                center_label[t, 0] = torch.maximum(center_label[t, 0], g)
                offset_label[t, 0, instance_mask] = off_x[instance_mask]
                offset_label[t, 1, instance_mask] = off_y[instance_mask]

                if prev_xc is not None and instance_mask.sum() > 0:
                    delta_x = xc - prev_xc
                    delta_y = yc - prev_yc
                    future_displacement_label[t - 1, 0, prev_mask] = delta_x
                    future_displacement_label[t - 1, 1, prev_mask] = delta_y
                    backward_flow[t - 1, 0, instance_mask] = -1 * delta_x
                    backward_flow[t - 1, 1, instance_mask] = -1 * delta_y

                prev_xc = xc
                prev_yc = yc
                prev_mask = instance_mask

        return center_label, offset_label, future_displacement_label, backward_flow

    def visualize_instances(self, instances, vis_root=""):
        if vis_root is not None and vis_root != "":
            os.makedirs(vis_root, exist_ok=True)

        for i, ins in enumerate(instances):
            ins_c = ins.astype(np.uint8)
            ins_c = cv2.applyColorMap(ins_c, cv2.COLORMAP_JET)
            save_path = os.path.join(vis_root, "{}.png".format(i))
            cv2.imwrite(save_path, ins_c)

        vid_path = os.path.join(vis_root, "vid_ins.avi")
        height, width = instances[0].shape
        size = (height, width)
        v_out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"DIVX"), 4, size)
        for i in range(len(instances)):
            ins_c = instances[i].astype(np.uint8)
            ins_c = cv2.applyColorMap(ins_c, cv2.COLORMAP_JET)
            v_out.write(ins_c)
        v_out.release()
        return


@TRANSFORMS.register_module()
class CustomLoadAnnotations3D_E2E(LoadAnnotations3D):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(
        self,
        with_future_anns=False,
        with_ins_inds_3d=False,
        ins_inds_add_1=False,  # NOTE: make ins_inds start from 1, not 0
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.with_future_anns = with_future_anns
        self.with_ins_inds_3d = with_ins_inds_3d

        self.ins_inds_add_1 = ins_inds_add_1

    def _load_future_anns(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """

        gt_bboxes_3d = []
        gt_labels_3d = []
        gt_inds_3d = []
        # gt_valid_flags = []
        gt_vis_tokens = []

        for ann_info in results["occ_future_ann_infos"]:
            if ann_info is not None:
                gt_bboxes_3d.append(ann_info["gt_bboxes_3d"])
                gt_labels_3d.append(ann_info["gt_labels_3d"])

                ann_gt_inds = ann_info["gt_inds"]
                if self.ins_inds_add_1:
                    ann_gt_inds += 1
                    # NOTE: sdc query is changed from -10 -> -9
                gt_inds_3d.append(ann_gt_inds)

                # gt_valid_flags.append(ann_info['gt_valid_flag'])
                gt_vis_tokens.append(ann_info["gt_vis_tokens"])
            else:
                # invalid frame
                gt_bboxes_3d.append(None)
                gt_labels_3d.append(None)
                gt_inds_3d.append(None)
                # gt_valid_flags.append(None)
                gt_vis_tokens.append(None)

        results["future_gt_bboxes_3d"] = gt_bboxes_3d
        # results['future_bbox3d_fields'].append('gt_bboxes_3d')  # Field is used for augmentations, not needed here
        results["future_gt_labels_3d"] = gt_labels_3d
        results["future_gt_inds"] = gt_inds_3d
        # results['future_gt_valid_flag'] = gt_valid_flags
        results["future_gt_vis_tokens"] = gt_vis_tokens

        return results

    def _load_ins_inds_3d(self, results):
        ann_gt_inds = results["ann_info"]["gt_inds"].copy()  # TODO: note here

        # NOTE: Avoid gt_inds generated twice
        results["ann_info"].pop("gt_inds")

        if self.ins_inds_add_1:
            ann_gt_inds += 1
        results["gt_inds"] = ann_gt_inds
        return results

    def __call__(self, results):
        results = super().__call__(results)

        if self.with_future_anns:
            results = self._load_future_anns(results)
        if self.with_ins_inds_3d:
            results = self._load_ins_inds_3d(results)

        # Generate ann for plan
        if "occ_future_ann_infos_for_plan" in results.keys():
            results = self._load_future_anns_plan(results)

        return results

    def __repr__(self):
        repr_str = super().__repr__()
        indent_str = "    "
        repr_str += f"{indent_str}with_future_anns={self.with_future_anns}, "
        repr_str += f"{indent_str}with_ins_inds_3d={self.with_ins_inds_3d}, "

        return repr_str


@TRANSFORMS.register_module()
class CustomPadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(img, shape=self.size, pad_val=self.pad_val) for img in results["img"]]
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val) for img in results["img"]
            ]

        results["ori_shape"] = [img.shape for img in results["img"]]
        results["img"] = padded_img
        results["img_shape"] = [img.shape for img in padded_img]
        results["pad_shape"] = [img.shape for img in padded_img]
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"size_divisor={self.size_divisor}, "
        repr_str += f"pad_val={self.pad_val})"
        return repr_str


@TRANSFORMS.register_module()
class CustomLoadMultiViewImageFromFilesInCeph(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type="unchanged", file_client_args=dict(backend="disk"), img_root=""):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = mmengine.FileClient(**self.file_client_args)
        self.img_root = img_root

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (list of str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        images_multiView = []
        filename = results["img_filename"]
        for img_path in filename:
            if img_path.startswith("./mmdetection3d/data/nuscenes/"):
                img_path = img_path.replace("./mmdetection3d/data/nuscenes/", "")
            img_path = os.path.join(self.img_root, img_path)
            if self.file_client_args["backend"] == "petrel":
                img_bytes = self.file_client.get(img_path)
                img = mmcv.imfrombytes(img_bytes)
            elif self.file_client_args["backend"] == "disk":
                img = mmcv.imread(img_path, self.color_type)
            images_multiView.append(img)
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            # [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
            images_multiView,
            axis=-1,
        )
        if self.to_float32:
            img = img.astype(np.float32)
        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results["img"] = [img[..., i] for i in range(img.shape[-1])]
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        # Set initial values for default meta_keys
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32), std=np.ones(num_channels, dtype=np.float32), to_rgb=False
        )
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@TRANSFORMS.register_module()
class CustomNormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results["img"] = [mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) for img in results["img"]]
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str
