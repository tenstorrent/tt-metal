import copy

#  from mmdet3d.datasets import NuScenesDataset
from models.experimental.BEVFormerV2.projects.mmdet3d_plugin.dependency import NuScenesDataset
import mmcv
from os import path as osp

# from mmdet.datasets import DATASETS
from models.experimental.BEVFormerV2.projects.mmdet3d_plugin.dependency import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import Quaternion
from .nuscnes_eval import NuScenesEval_custom

# from mmcv.parallel import DataContainer as DC
from models.experimental.BEVFormerV2.projects.mmdet3d_plugin.dependency import DataContainer as DC
from collections import defaultdict, OrderedDict


@DATASETS.register_module()
class CustomNuScenesDatasetV2(NuScenesDataset):
    def __init__(self, frames=(), mono_cfg=None, overlap_test=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames = frames
        self.queue_length = len(frames)
        self.overlap_test = overlap_test
        self.mono_cfg = mono_cfg
        if not self.test_mode and mono_cfg is not None:
            from models.experimental.BEVFormerV2.projects.mmdet3d_plugin.dd3d.datasets.nuscenes import (
                NuscenesDataset as DD3DNuscenesDataset,
            )

            self.mono_dataset = DD3DNuscenesDataset(**mono_cfg)

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        data_queue = OrderedDict()
        input_dict = self.get_data_info(index)
        cur_scene_token = input_dict["scene_token"]
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        data_queue[0] = example

        for frame_idx in self.frames:
            chosen_idx = index + frame_idx
            if frame_idx == 0 or chosen_idx < 0 or chosen_idx >= len(self.data_infos):
                continue
            info = self.data_infos[chosen_idx]
            input_dict = self.prepare_input_dict(info)
            if input_dict["scene_token"] == cur_scene_token:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                data_queue[frame_idx] = example

        data_queue = OrderedDict(sorted(data_queue.items()))
        ret = defaultdict(list)
        # for i in range(len(data_queue[0]["img"])):
        #     single_aug_data_queue = {}
        #     for t in data_queue.keys():
        #         single_example = {}
        #         for key, value in data_queue[t].items():
        #             single_example[key] = value[i]
        #         single_aug_data_queue[t] = single_example
        #     single_aug_data_queue = OrderedDict(sorted(single_aug_data_queue.items()))
        #     single_aug_sample = self.union2one(single_aug_data_queue)

        #     for key, value in single_aug_sample.items():
        #         ret[key].append(value)
        ret = self.union2one(data_queue)
        return ret

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        data_queue = OrderedDict()
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        cur_scene_token = input_dict["scene_token"]
        # cur_frame_idx = input_dict['frame_idx']
        ann_info = copy.deepcopy(input_dict["ann_info"])
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and (example is None or ~(example["gt_labels_3d"]._data != -1).any()):
            return None
        data_queue[0] = example
        aug_param = copy.deepcopy(example["aug_param"]) if "aug_param" in example else {}

        # frame_idx_to_idx = self.scene_to_frame_idx_to_idx[cur_scene_token]
        for frame_idx in self.frames:
            chosen_idx = index + frame_idx
            if frame_idx == 0 or chosen_idx < 0 or chosen_idx >= len(self.data_infos):
                continue
            info = self.data_infos[chosen_idx]
            input_dict = self.prepare_input_dict(info)
            if input_dict["scene_token"] == cur_scene_token:
                input_dict["ann_info"] = copy.deepcopy(ann_info)  # only for pipeline, should never be used
                self.pre_pipeline(input_dict)
                input_dict["aug_param"] = copy.deepcopy(aug_param)
                example = self.pipeline(input_dict)
                data_queue[frame_idx] = example

        data_queue = OrderedDict(sorted(data_queue.items()))
        return self.union2one(data_queue)

    def union2one(self, queue: dict):
        """
        convert sample queue into one single sample.
        """
        # Handle both DataContainer and plain list cases for img
        imgs_list = []
        for each in queue.values():
            img = each["img"]
            if hasattr(img, "data"):
                # DataContainer object
                img_data = img.data
            else:
                # Plain list or tensor
                img_data = img

            # If img_data is a list of numpy arrays (from loading pipeline), convert to tensor
            if isinstance(img_data, list) and len(img_data) > 0:
                if isinstance(img_data[0], np.ndarray):
                    # List of numpy arrays (one per camera view), stack them into a tensor
                    img_data = torch.from_numpy(np.stack(img_data))
                elif isinstance(img_data[0], torch.Tensor):
                    # List of tensors, stack them
                    img_data = torch.stack(img_data)
                # If it's already a tensor or other type, use as-is
            elif isinstance(img_data, np.ndarray):
                # Single numpy array, convert to tensor
                img_data = torch.from_numpy(img_data)

            imgs_list.append(img_data)

        lidar2ego = np.eye(4, dtype=np.float32)
        lidar2ego[:3, :3] = Quaternion(queue[0]["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = queue[0]["lidar2ego_translation"]

        egocurr2global = np.eye(4, dtype=np.float32)
        egocurr2global[:3, :3] = Quaternion(queue[0]["ego2global_rotation"]).rotation_matrix
        egocurr2global[:3, 3] = queue[0]["ego2global_translation"]
        metas_map = {}
        for i, each in queue.items():
            # Handle both DataContainer and plain dict cases for img_metas
            # img_metas might not exist in test mode, so create it from available data
            if "img_metas" in each:
                img_metas = each["img_metas"]
                if hasattr(img_metas, "data"):
                    # DataContainer object
                    metas_map[i] = img_metas.data
                else:
                    # Plain dict
                    metas_map[i] = img_metas
            else:
                # Create img_metas dict from available metadata
                metas_map[i] = {}
                # Copy relevant metadata fields if they exist
                for key in ["lidar2img", "cam2img", "lidar2cam", "img_shape", "ori_shape", "img_norm_cfg"]:
                    if key in each:
                        metas_map[i][key] = each[key]

            # Add timestamp if available
            if "timestamp" in each:
                metas_map[i]["timestamp"] = each["timestamp"]
            if "aug_param" in each:
                metas_map[i]["aug_param"] = each["aug_param"]
            if i == 0:
                metas_map[i]["lidaradj2lidarcurr"] = None
            else:
                egoadj2global = np.eye(4, dtype=np.float32)
                egoadj2global[:3, :3] = Quaternion(each["ego2global_rotation"]).rotation_matrix
                egoadj2global[:3, 3] = each["ego2global_translation"]

                lidaradj2lidarcurr = (
                    np.linalg.inv(lidar2ego) @ np.linalg.inv(egocurr2global) @ egoadj2global @ lidar2ego
                )
                metas_map[i]["lidaradj2lidarcurr"] = lidaradj2lidarcurr
                # Only update lidar2img if it exists
                if "lidar2img" in metas_map[i] and isinstance(metas_map[i]["lidar2img"], (list, np.ndarray)):
                    for i_cam in range(len(metas_map[i]["lidar2img"])):
                        metas_map[i]["lidar2img"][i_cam] = metas_map[i]["lidar2img"][i_cam] @ np.linalg.inv(
                            lidaradj2lidarcurr
                        )
        queue[0]["img"] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[0]["img_metas"] = DC(metas_map, cpu_only=True)
        queue = queue[0]
        return queue

    def prepare_input_dict(self, info):
        # standard protocal modified from SECOND.Pytorch
        # Replace hardcoded ./data/nuscenes/ or data/nuscenes/ with actual data_root for lidar_path
        lidar_path = info["lidar_path"]
        # Check for both lowercase and uppercase variants (annotation has lowercase, config may have uppercase)
        if lidar_path.startswith("./data/nuscenes/"):
            # Strip ./data/nuscenes/ prefix to get relative path like samples/...
            lidar_path = osp.join(self.data_root, lidar_path[len("./data/nuscenes/") :])
        elif lidar_path.startswith("./data/nuScenes/"):
            lidar_path = osp.join(self.data_root, lidar_path[len("./data/nuScenes/") :])
        elif lidar_path.startswith("data/nuscenes/"):
            lidar_path = osp.join(self.data_root, lidar_path[len("data/nuscenes/") :])
        elif lidar_path.startswith("data/nuScenes/"):
            lidar_path = osp.join(self.data_root, lidar_path[len("data/nuScenes/") :])
        elif not osp.isabs(lidar_path):
            lidar_path = osp.join(self.data_root, lidar_path)
        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=lidar_path,
            sweeps=info["sweeps"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
            lidar2ego_translation=info["lidar2ego_translation"],
            lidar2ego_rotation=info["lidar2ego_rotation"],
            prev=info["prev"],
            next=info["next"],
            scene_token=info["scene_token"],
            frame_idx=info["frame_idx"],
            timestamp=info["timestamp"] / 1e6,
        )

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info["cams"].items():
                # Replace hardcoded ./data/nuscenes/ or data/nuscenes/ with actual data_root
                data_path = cam_info["data_path"]
                # Check for both lowercase and uppercase variants (annotation has lowercase, config may have uppercase)
                if data_path.startswith("./data/nuscenes/"):
                    # Strip ./data/nuscenes/ prefix to get relative path like samples/CAM_FRONT/...
                    data_path = osp.join(self.data_root, data_path[len("./data/nuscenes/") :])
                elif data_path.startswith("./data/nuScenes/"):
                    data_path = osp.join(self.data_root, data_path[len("./data/nuScenes/") :])
                elif data_path.startswith("data/nuscenes/"):
                    data_path = osp.join(self.data_root, data_path[len("data/nuscenes/") :])
                elif data_path.startswith("data/nuScenes/"):
                    data_path = osp.join(self.data_root, data_path[len("data/nuScenes/") :])
                elif not osp.isabs(data_path):
                    # If path is relative but doesn't match the pattern, join with data_root
                    data_path = osp.join(self.data_root, data_path)
                image_paths.append(data_path)
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
                lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info["cam_intrinsic"]
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt.T
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam2img=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                )
            )

        return input_dict

    def filter_crowd_annotations(self, data_dict):
        for ann in data_dict["annotations"]:
            if ann.get("iscrowd", 0) == 0:
                return True
        return False

    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = self.prepare_input_dict(info)
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict["ann_info"] = annos

        if not self.test_mode and self.mono_cfg is not None:
            if input_dict is None:
                return None
            info = self.data_infos[index]
            img_ids = []
            for cam_type, cam_info in info["cams"].items():
                img_ids.append(cam_info["sample_data_token"])

            mono_input_dict = []
            mono_ann_index = []
            for i, img_id in enumerate(img_ids):
                tmp_dict = self.mono_dataset.getitem_by_datumtoken(img_id)
                if tmp_dict is not None:
                    if self.filter_crowd_annotations(tmp_dict):
                        mono_input_dict.append(tmp_dict)
                        mono_ann_index.append(i)

            # filter empth annotation
            if len(mono_ann_index) == 0:
                return None

            mono_ann_index = DC(mono_ann_index, cpu_only=True)
            input_dict["mono_input_dict"] = mono_input_dict
            input_dict["mono_ann_idx"] = mono_ann_index
        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _evaluate_single(self, result_path, logger=None, metric="bbox", result_name="pts_bbox"):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes

        self.nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=True)

        output_dir = osp.join(*osp.split(result_path)[:-1])

        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
        }
        self.nusc_eval = NuScenesEval_custom(
            self.nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
            overlap_test=self.overlap_test,
            data_infos=self.data_infos,
        )
        self.nusc_eval.main(plot_examples=0, render_curves=False)
        # record metrics
        metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
        detail = dict()
        metric_prefix = f"{result_name}_NuScenes"
        for name in self.CLASSES:
            for k, v in metrics["label_aps"][name].items():
                val = float("{:.4f}".format(v))
                detail["{}/{}_AP_dist_{}".format(metric_prefix, name, k)] = val
            for k, v in metrics["label_tp_errors"][name].items():
                val = float("{:.4f}".format(v))
                detail["{}/{}_{}".format(metric_prefix, name, k)] = val
            for k, v in metrics["tp_errors"].items():
                val = float("{:.4f}".format(v))
                detail["{}/{}".format(metric_prefix, self.ErrNameMapping[k])] = val
        detail["{}/NDS".format(metric_prefix)] = metrics["nd_score"]
        detail["{}/mAP".format(metric_prefix)] = metrics["mean_ap"]
        return detail
