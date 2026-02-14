import os
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union, Dict, Any, Callable
import argparse
import mmcv
import mmengine
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
import numpy as np
from pyquaternion import Quaternion
from loguru import logger
from nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw
from models.experimental.uniad.demo.my_dataset import CustomNuScenesDataset

from mmdet3d.structures import points_cam2img

MICROSECONDS_PER_SECOND = 1e6
BUFFER = 0.15  # seconds
Record = Dict[str, Any]

nus_categories = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
)

nus_attributes = (
    "cycle.with_rider",
    "cycle.without_rider",
    "pedestrian.moving",
    "pedestrian.standing",
    "pedestrian.sitting_lying_down",
    "vehicle.moving",
    "vehicle.parked",
    "vehicle.stopped",
    "None",
)


def generate_record(
    ann_rec: dict, x1: float, y1: float, x2: float, y2: float, sample_data_token: str, filename: str
) -> OrderedDict:
    repro_rec = OrderedDict()
    repro_rec["sample_data_token"] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        "attribute_tokens",
        "category_name",
        "instance_token",
        "next",
        "num_lidar_pts",
        "num_radar_pts",
        "prev",
        "sample_annotation_token",
        "sample_data_token",
        "visibility_token",
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec["bbox_corners"] = [x1, y1, x2, y2]
    repro_rec["filename"] = filename

    coco_rec["file_name"] = filename
    coco_rec["image_id"] = sample_data_token
    coco_rec["area"] = (y2 - y1) * (x2 - x1)

    if repro_rec["category_name"] not in CustomNuScenesDataset.NameMapping:
        return None
    cat_name = CustomNuScenesDataset.NameMapping[repro_rec["category_name"]]
    coco_rec["category_name"] = cat_name
    coco_rec["category_id"] = nus_categories.index(cat_name)
    coco_rec["bbox"] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec["iscrowd"] = 0

    return coco_rec


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def get_2d_boxes(nusc, sample_data_token: str, visibilities: List[str], mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera \
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get("sample_data", sample_data_token)

    assert sd_rec["sensor_modality"] == "camera", "Error: get_2d_boxes only works" " for camera sample_data!"
    if not sd_rec["is_key_frame"]:
        raise ValueError("The 2D re-projections are available only for keyframes.")

    s_rec = nusc.get("sample", sd_rec["sample_token"])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_rec = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    camera_intrinsic = np.array(cs_rec["camera_intrinsic"])

    # Get all the annotation with the specified visibilties.
    ann_recs = [nusc.get("sample_annotation", token) for token in s_rec["anns"]]
    ann_recs = [ann_rec for ann_rec in ann_recs if (ann_rec["visibility_token"] in visibilities)]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec["sample_annotation_token"] = ann_rec["token"]
        ann_rec["sample_data_token"] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec["token"])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec["translation"]))
        box.rotate(Quaternion(pose_rec["rotation"]).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec["translation"]))
        box.rotate(Quaternion(cs_rec["rotation"]).inverse)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y, sample_data_token, sd_rec["filename"])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # convert the rot to our cam coordinate

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec["rotation"]).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec["rotation"]).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec["bbox_cam3d"] = loc + dim + rot
            repro_rec["velo_cam3d"] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(center3d, camera_intrinsic, with_depth=True)
            repro_rec["center2d"] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec["center2d"][2] <= 0:
                continue

            ann_token = nusc.get("sample_annotation", box.token)["attribute_tokens"]
            if len(ann_token) == 0:
                attr_name = "None"
            else:
                attr_name = nusc.get("attribute", ann_token[0])["name"]
            attr_id = nus_attributes.index(attr_name)
            repro_rec["attribute_name"] = attr_name
            repro_rec["attribute_id"] = attr_id

        repro_recs.append(repro_rec)

    return repro_recs


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool): Whether to export mono3d annotation. Default: True.
    """
    # get bbox annotations for camera
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    nusc_infos = mmengine.load(info_path)["infos"]
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    # info_2d_list = []
    cat2Ids = [dict(id=nus_categories.index(cat_name), name=cat_name) for cat_name in nus_categories]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmengine.track_iter_progress(nusc_infos):
        for cam in camera_types:
            cam_info = info["cams"][cam]
            coco_infos = get_2d_boxes(
                nusc, cam_info["sample_data_token"], visibilities=["", "1", "2", "3", "4"], mono3d=mono3d
            )
            (height, width, _) = mmcv.imread(cam_info["data_path"]).shape
            coco_2d_dict["images"].append(
                dict(
                    file_name=cam_info["data_path"].split("data/nuscenes/")[-1],
                    id=cam_info["sample_data_token"],
                    token=info["token"],
                    cam2ego_rotation=cam_info["sensor2ego_rotation"],
                    cam2ego_translation=cam_info["sensor2ego_translation"],
                    ego2global_rotation=info["ego2global_rotation"],
                    ego2global_translation=info["ego2global_translation"],
                    cam_intrinsic=cam_info["cam_intrinsic"],
                    width=width,
                    height=height,
                )
            )
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info["segmentation"] = []
                coco_info["id"] = coco_ann_id
                coco_2d_dict["annotations"].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f"{info_path[:-4]}_mono3d"
    else:
        json_prefix = f"{info_path[:-4]}"
    mmengine.dump(coco_2d_dict, f"{json_prefix}.coco.json")


def angle_of_rotation(yaw: float) -> float:
    """
    Given a yaw angle (measured from x axis), find the angle needed to rotate by so that
    the yaw is aligned with the y axis (pi / 2).
    :param yaw: Radians. Output of quaternion_yaw function.
    :return: Angle in radians.
    """
    return (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)


def make_2d_rotation_matrix(angle_in_radians: float) -> np.ndarray:
    """
    Makes rotation matrix to rotate point in x-y plane counterclockwise
    by angle_in_radians.
    """

    return np.array(
        [[np.cos(angle_in_radians), -np.sin(angle_in_radians)], [np.sin(angle_in_radians), np.cos(angle_in_radians)]]
    )


def convert_global_coords_to_local(
    coordinates: np.ndarray, translation: Tuple[float, float, float], rotation: Tuple[float, float, float, float]
) -> np.ndarray:
    """
    Converts global coordinates to coordinates in the frame given by the rotation quaternion and
    centered at the translation vector. The rotation is meant to be a z-axis rotation.
    :param coordinates: x,y locations. array of shape [n_steps, 2].
    :param translation: Tuple of (x, y, z) location that is the center of the new frame.
    :param rotation: Tuple representation of quaternion of new frame.
        Representation - cos(theta / 2) + (xi + yi + zi)sin(theta / 2).
    :return: x,y locations in frame stored in array of share [n_times, 2].
    """
    yaw = angle_of_rotation(quaternion_yaw(Quaternion(rotation)))

    transform = make_2d_rotation_matrix(angle_in_radians=yaw)

    coords = (coordinates - np.atleast_2d(np.array(translation)[:2])).T

    return np.dot(transform, coords).T[:, :2]


class PredictHelper:
    """Wrapper class around NuScenes to help retrieve data for the prediction task."""

    def __init__(self, nusc: NuScenes):
        """
        Inits PredictHelper
        :param nusc: Instance of NuScenes class.
        """
        self.data = nusc
        self.inst_sample_to_ann = self._map_sample_and_instance_to_annotation()

    def _map_sample_and_instance_to_annotation(self) -> Dict[Tuple[str, str], str]:
        """
        Creates mapping to look up an annotation given a sample and instance in constant time.
        :return: Mapping from (sample_token, instance_token) -> sample_annotation_token.
        """
        mapping = {}

        for record in self.data.sample_annotation:
            mapping[(record["sample_token"], record["instance_token"])] = record["token"]

        return mapping

    def _timestamp_for_sample(self, sample_token: str) -> float:
        """
        Gets timestamp from sample token.
        :param sample_token: Get the timestamp for this sample.
        :return: Timestamp (microseconds).
        """
        return self.data.get("sample", sample_token)["timestamp"]

    def _absolute_time_diff(self, time1: float, time2: float) -> float:
        """
        Helper to compute how much time has elapsed in _iterate method.
        :param time1: First timestamp (microseconds since unix epoch).
        :param time2: Second timestamp (microseconds since unix epoch).
        :return: Absolute Time difference in floats.
        """
        return abs(time1 - time2) / MICROSECONDS_PER_SECOND

    def _iterate(self, starting_annotation: Dict[str, Any], seconds: float, direction: str) -> List[Dict[str, Any]]:
        """
        Iterates forwards or backwards in time through the annotations for a given amount of seconds.
        :param starting_annotation: Sample annotation record to start from.
        :param seconds: Number of seconds to iterate.
        :param direction: 'prev' for past and 'next' for future.
        :return: List of annotations ordered by time.
        """
        if seconds < 0:
            raise ValueError(f"Parameter seconds must be non-negative. Received {seconds}.")

        # Need to exit early because we technically _could_ return data in this case if
        # the first observation is within the BUFFER.
        if seconds == 0:
            return []

        seconds_with_buffer = seconds + BUFFER
        starting_time = self._timestamp_for_sample(starting_annotation["sample_token"])

        next_annotation = starting_annotation

        time_elapsed = 0.0

        annotations = []

        expected_samples_per_sec = 2
        max_annotations = int(expected_samples_per_sec * seconds)
        while time_elapsed <= seconds_with_buffer and len(annotations) < max_annotations:
            if next_annotation[direction] == "":
                break

            next_annotation = self.data.get("sample_annotation", next_annotation[direction])
            current_time = self._timestamp_for_sample(next_annotation["sample_token"])

            time_elapsed = self._absolute_time_diff(current_time, starting_time)

            if time_elapsed < seconds_with_buffer:
                annotations.append(next_annotation)

        return annotations

    def get_sample_annotation(self, instance_token: str, sample_token: str) -> Record:
        """
        Retrieves an annotation given an instance token and its sample.
        :param instance_token: Instance token.
        :param sample_token: Sample token for instance.
        :return: Sample annotation record.
        """
        return self.data.get("sample_annotation", self.inst_sample_to_ann[(sample_token, instance_token)])

    def get_annotations_for_sample(self, sample_token: str) -> List[Record]:
        """
        Gets a list of sample annotation records for a sample.
        :param sample_token: Sample token.
        """

        sample_record = self.data.get("sample", sample_token)
        annotations = []

        for annotation_token in sample_record["anns"]:
            annotation_record = self.data.get("sample_annotation", annotation_token)
            annotations.append(annotation_record)

        return annotations

    def _get_past_or_future_for_agent(
        self,
        instance_token: str,
        sample_token: str,
        seconds: float,
        in_agent_frame: bool,
        direction: str,
        just_xy: bool = True,
    ) -> Union[List[Record], np.ndarray]:
        """
        Helper function to reduce code duplication between get_future and get_past for agent.
        :param instance_token: Instance of token.
        :param sample_token: Sample token for instance.
        :param seconds: How many seconds of data to retrieve.
        :param in_agent_frame: Whether to rotate the coordinates so the
            heading is aligned with the y-axis. Only relevant if just_xy = True.
        :param direction: 'next' for future or 'prev' for past.
        :return: array of shape [n_timesteps, 2].
        """
        starting_annotation = self.get_sample_annotation(instance_token, sample_token)
        sequence = self._iterate(starting_annotation, seconds, direction)

        if not just_xy:
            return sequence

        coords = np.array([r["translation"][:2] for r in sequence])

        if coords.size == 0:
            return coords

        if in_agent_frame:
            coords = convert_global_coords_to_local(
                coords, starting_annotation["translation"], starting_annotation["rotation"]
            )

        return coords

    def get_future_for_agent(
        self, instance_token: str, sample_token: str, seconds: float, in_agent_frame: bool, just_xy: bool = True
    ) -> Union[List[Record], np.ndarray]:
        """
        Retrieves the agent's future x,y locations.
        :param instance_token: Instance token.
        :param sample_token: Sample token.
        :param seconds: How much future data to retrieve.
        :param in_agent_frame: If true, locations are rotated to the agent frame.
            Only relevant if just_xy = True.
        :param just_xy: If true, returns an np.array of x,y locations as opposed to the
            entire record.
        :return: If just_xy, np.ndarray. Else, List of records.
            The rows increate with time, i.e the last row occurs the farthest in the future.
        """
        return self._get_past_or_future_for_agent(
            instance_token, sample_token, seconds, in_agent_frame, direction="next", just_xy=just_xy
        )

    def get_past_for_agent(
        self, instance_token: str, sample_token: str, seconds: float, in_agent_frame: bool, just_xy: bool = True
    ) -> Union[List[Record], np.ndarray]:
        """
        Retrieves the agent's past sample annotation records.
        :param instance_token: Instance token.
        :param sample_token: Sample token.
        :param seconds: How much past data to retrieve.
        :param in_agent_frame: If true, locations are rotated to the agent frame.
            Only relevant if just_xy = True.
        :param just_xy: If true, returns an np.array of x,y locations as opposed to the
            entire record.
        :return: If just_xy, np.ndarray. Else, List of records.
            The rows decrease with time, i.e the last row occurs the farthest in the past.
        """
        return self._get_past_or_future_for_agent(
            instance_token, sample_token, seconds, in_agent_frame, direction="prev", just_xy=just_xy
        )

    def _get_past_or_future_for_sample(
        self,
        sample_token: str,
        seconds: float,
        in_agent_frame: bool,
        direction: str,
        just_xy: bool,
        function: Callable[[str, str, float, bool, str, bool], np.ndarray],
    ) -> Union[Dict[str, np.ndarray], Dict[str, List[Record]]]:
        """
        Helper function to reduce code duplication between get_future and get_past for sample.
        :param sample_token: Sample token.
        :param seconds: How much past or future data to retrieve.
        :param in_agent_frame: Whether to rotate each agent future.
            Only relevant if just_xy = True.
        :param just_xy: If true, returns an np.array of x,y locations as opposed to the
            entire record.
        :param function: _get_past_or_future_for_agent.
        :return: Dictionary mapping instance token to np.array or list of records.
        """
        sample_record = self.data.get("sample", sample_token)
        sequences = {}
        for annotation in sample_record["anns"]:
            annotation_record = self.data.get("sample_annotation", annotation)
            sequence = function(
                annotation_record["instance_token"],
                annotation_record["sample_token"],
                seconds,
                in_agent_frame,
                direction,
                just_xy=just_xy,
            )

            sequences[annotation_record["instance_token"]] = sequence

        return sequences

    def get_future_for_sample(
        self, sample_token: str, seconds: float, in_agent_frame: bool, just_xy: bool = True
    ) -> Union[Dict[str, np.ndarray], Dict[str, List[Record]]]:
        """
        Retrieves the the future x,y locations of all agents in the sample.
        :param sample_token: Sample token.
        :param seconds: How much future data to retrieve.
        :param in_agent_frame: If true, locations are rotated to the agent frame.
            Only relevant if just_xy = True.
        :param just_xy: If true, returns an np.array of x,y locations as opposed to the
            entire record.
        :return: If just_xy, Mapping of instance token to np.ndarray.
            Else, the mapping is from instance token to list of records.
            The rows increase with time, i.e the last row occurs the farthest in the future.
        """
        return self._get_past_or_future_for_sample(
            sample_token, seconds, in_agent_frame, "next", just_xy, function=self._get_past_or_future_for_agent
        )

    def get_past_for_sample(
        self, sample_token: str, seconds: float, in_agent_frame: bool, just_xy: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Retrieves the the past x,y locations of all agents in the sample.
        :param sample_token: Sample token.
        :param seconds: How much past data to retrieve.
        :param in_agent_frame: If true, locations are rotated to the agent frame.
                Only relevant if just_xy = True.
        :param just_xy: If true, returns an np.array of x,y locations as opposed to the
            entire record.
        :return: If just_xy, Mapping of instance token to np.ndarray.
            Else, the mapping is from instance token to list of records.
            The rows decrease with time, i.e the last row occurs the farthest in the past.
        """
        return self._get_past_or_future_for_sample(
            sample_token, seconds, in_agent_frame, "prev", just_xy, function=self._get_past_or_future_for_agent
        )

    def _compute_diff_between_sample_annotations(
        self, instance_token: str, sample_token: str, max_time_diff: float, with_function, **kwargs
    ) -> float:
        """
        Grabs current and previous annotation and computes a float from them.
        :param instance_token: Instance token.
        :param sample_token: Sample token.
        :param max_time_diff: If the time difference between now and the most recent annotation is larger
            than this param, function will return np.nan.
        :param with_function: Function to apply to the annotations.
        :param **kwargs: Keyword arguments to give to with_function.

        """
        annotation = self.get_sample_annotation(instance_token, sample_token)

        if annotation["prev"] == "":
            return np.nan

        prev = self.data.get("sample_annotation", annotation["prev"])

        current_time = 1e-6 * self.data.get("sample", sample_token)["timestamp"]
        prev_time = 1e-6 * self.data.get("sample", prev["sample_token"])["timestamp"]
        time_diff = current_time - prev_time

        if time_diff <= max_time_diff:
            return with_function(annotation, prev, time_diff, **kwargs)

        else:
            return np.nan

    def get_velocity_for_agent(self, instance_token: str, sample_token: str, max_time_diff: float = 1.5) -> float:
        """
        Computes velocity based on the difference between the current and previous annotation.
        :param instance_token: Instance token.
        :param sample_token: Sample token.
        :param max_time_diff: If the time difference between now and the most recent annotation is larger
            than this param, function will return np.nan.
        """
        return self._compute_diff_between_sample_annotations(
            instance_token, sample_token, max_time_diff, with_function=velocity
        )

    def get_heading_change_rate_for_agent(
        self, instance_token: str, sample_token: str, max_time_diff: float = 1.5
    ) -> float:
        """
        Computes heading change rate based on the difference between the current and previous annotation.
        :param instance_token: Instance token.
        :param sample_token: Sample token.
        :param max_time_diff: If the time difference between now and the most recent annotation is larger
            than this param, function will return np.nan.
        """
        return self._compute_diff_between_sample_annotations(
            instance_token, sample_token, max_time_diff, with_function=heading_change_rate
        )

    def get_acceleration_for_agent(self, instance_token: str, sample_token: str, max_time_diff: float = 1.5) -> float:
        """
        Computes heading change rate based on the difference between the current and previous annotation.
        :param instance_token: Instance token.
        :param sample_token: Sample token.
        :param max_time_diff: If the time difference between now and the most recent annotation is larger
            than this param, function will return np.nan.
        """
        return self._compute_diff_between_sample_annotations(
            instance_token,
            sample_token,
            max_time_diff,
            with_function=acceleration,
            instance_token_for_velocity=instance_token,
            helper=self,
        )

    def get_map_name_from_sample_token(self, sample_token: str) -> str:
        sample = self.data.get("sample", sample_token)
        scene = self.data.get("scene", sample["scene_token"])
        log = self.data.get("log", scene["log_token"])
        return log["location"]


def _get_future_traj_info(nusc, sample, predict_steps=16):
    sample_token = sample["token"]
    ann_tokens = np.array(sample["anns"])
    sd_rec = nusc.get("sample", sample_token)
    fut_traj_all = []
    fut_traj_valid_mask_all = []
    _, boxes, _ = nusc.get_sample_data(sd_rec["data"]["LIDAR_TOP"], selected_anntokens=ann_tokens)
    predict_helper = PredictHelper(nusc)
    for i, ann_token in enumerate(ann_tokens):
        box = boxes[i]
        instance_token = nusc.get("sample_annotation", ann_token)["instance_token"]
        fut_traj_local = predict_helper.get_future_for_agent(
            instance_token, sample_token, seconds=predict_steps // 2, in_agent_frame=True
        )

        fut_traj = np.zeros((predict_steps, 2))
        fut_traj_valid_mask = np.zeros((predict_steps, 2))
        if fut_traj_local.shape[0] > 0:
            # trans = box.center
            # trans = np.array([0, 0, 0])
            # rot = Quaternion(matrix=box.rotation_matrix)
            # fut_traj_scence_centric = convert_local_coords_to_global(fut_traj_local, trans, rot)
            fut_traj_scence_centric = fut_traj_local
            fut_traj[: fut_traj_scence_centric.shape[0], :] = fut_traj_scence_centric
            fut_traj_valid_mask[: fut_traj_scence_centric.shape[0], :] = 1
        fut_traj_all.append(fut_traj)
        fut_traj_valid_mask_all.append(fut_traj_valid_mask)
    if len(ann_tokens) > 0:
        fut_traj_all = np.stack(fut_traj_all, axis=0)
        fut_traj_valid_mask_all = np.stack(fut_traj_valid_mask_all, axis=0)
    else:
        fut_traj_all = np.zeros((0, predict_steps, 2))
        fut_traj_valid_mask_all = np.zeros((0, predict_steps, 2))
    return fut_traj_all, fut_traj_valid_mask_all


def obtain_sensor2top(nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f"{os.getcwd()}/")[-1]  # relative path
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }

    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep


def _get_can_bus_info(nusc, nusc_can_bus, sample):
    scene_name = nusc.get("scene", sample["scene_token"])["name"]
    sample_timestamp = sample["timestamp"]
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, "pose")
    except:
        return np.zeros(18)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose["utime"] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop("utime")  # useless
    pos = last_pose.pop("pos")
    rotation = last_pose.pop("orientation")
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 16 elements
    can_bus.extend([0.0, 0.0])
    return np.array(can_bus)


def _fill_trainval_infos(nusc, nusc_can_bus, train_scenes, val_scenes, test=False, max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []
    frame_idx = 0
    for sample in mmengine.track_iter_progress(nusc.sample):
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
        print("lidar_pathlidar_path", lidar_path)
        mmengine.check_file_exist(lidar_path)
        can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)
        ##
        info = {
            "lidar_path": lidar_path,
            "token": sample["token"],
            "prev": sample["prev"],
            "next": sample["next"],
            "can_bus": can_bus,
            "frame_idx": frame_idx,  # temporal related info
            "sweeps": [],
            "cams": dict(),
            "scene_token": sample["scene_token"],  # temporal related info
            "lidar2ego_translation": cs_record["translation"],
            "lidar2ego_rotation": cs_record["rotation"],
            "ego2global_translation": pose_record["translation"],
            "ego2global_rotation": pose_record["rotation"],
            "timestamp": sample["timestamp"],
        }

        if sample["next"] == "":
            frame_idx = 0
        else:
            frame_idx += 1

        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
        for cam in camera_types:
            cam_token = sample["data"][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info["cams"].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec["prev"] == "":
                sweep = obtain_sensor2top(nusc, sd_rec["prev"], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, "lidar")
                sweeps.append(sweep)
                sd_rec = nusc.get("sample_data", sd_rec["prev"])
            else:
                break
        info["sweeps"] = sweeps
        # obtain annotation
        if not test:
            annotations = [nusc.get("sample_annotation", token) for token in sample["anns"]]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
            velocity = np.array([nusc.box_velocity(token)[:2] for token in sample["anns"]])
            valid_flag = np.array(
                [(anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0 for anno in annotations], dtype=bool
            ).reshape(-1)
            instance_inds = [nusc.getind("instance", ann["instance_token"]) for ann in annotations]
            future_traj_all, future_traj_valid_mask_all = _get_future_traj_info(nusc, sample)
            instance_tokens = [ann["instance_token"] for ann in annotations]  # dtype('<U[length_of_str]')

            # TODO: Add traj in next dataset_version
            # future_traj_all, future_traj_valid_mask_all = _get_future_traj_info(nusc, sample)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in CustomNuScenesDataset.NameMapping:
                    names[i] = CustomNuScenesDataset.NameMapping[names[i]]
            names = np.array(names)
            # instance_inds = [nusc.getind('instance', ann['instance_token']) for ann in annotations]
            # TODO(box3d): convert gt_boxes to mmdet3d 1.0.0rc6 LiDARInstance3DBoxes format. [DONE]
            gt_boxes = np.concatenate([locs, dims, rots], axis=1)
            # gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            assert len(gt_boxes) == len(annotations), f"{len(gt_boxes)}, {len(annotations)}"
            info["gt_boxes"] = gt_boxes
            info["gt_names"] = names
            info["gt_velocity"] = velocity.reshape(-1, 2)
            info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
            info["num_radar_pts"] = np.array([a["num_radar_pts"] for a in annotations])
            info["valid_flag"] = valid_flag
            info["gt_inds"] = np.array(instance_inds)
            info["gt_ins_tokens"] = np.array(instance_tokens)
            info["fut_traj"] = future_traj_all
            info["fut_traj_valid_mask"] = future_traj_valid_mask_all

            # add visibility_tokens
            visibility_tokens = [int(anno["visibility_token"]) for anno in annotations]
            info["visibility_tokens"] = np.array(visibility_tokens)

        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print("total scene num: {}".format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f"{os.getcwd()}/")[-1]
                # relative path
            if not mmengine.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num: {}".format(len(available_scenes)))
    return available_scenes


def create_nuscenes_infos(root_path, out_path, can_bus_root_path, info_prefix, version, max_sweeps=10):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus

    version = "v1.0-mini"
    print(version, root_path)
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
    from nuscenes.utils import splits

    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers
    if version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError("unknown")

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]["token"] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes])

    test = "test" in version
    if test:
        print("test scene: {}".format(len(train_scenes)))
    else:
        print("train scene: {}, val scene: {}".format(len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, nusc_can_bus, train_scenes, val_scenes, test, max_sweeps=max_sweeps
    )

    metadata = dict(version=version)
    if test:
        print("test sample: {}".format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path, "{}_infos_temporal_test.pkl".format(info_prefix))
        mmengine.dump(data, info_path)
    else:
        print("train sample: {}, val sample: {}".format(len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path, "{}_infos_temporal_train.pkl".format(info_prefix))
        mmengine.dump(data, info_path)
        data["infos"] = val_nusc_infos
        info_val_path = osp.join(out_path, "{}_infos_temporal_val.pkl".format(info_prefix))
        logger.info(f".pkl file generated here: {info_val_path}")
        mmengine.dump(data, info_val_path)


def nuscenes_data_prep(root_path, can_bus_root_path, info_prefix, version, dataset_name, out_dir, max_sweeps=10):
    create_nuscenes_infos(root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    if version == "v1.0-test":
        info_test_path = osp.join(out_dir, f"{info_prefix}_infos_temporal_test_uniad2.0.pkl")
        export_2d_annotation(root_path, info_test_path, version=version)
    else:
        info_train_path = osp.join(out_dir, f"{info_prefix}_infos_temporal_train.pkl")
        info_val_path = osp.join(out_dir, f"{info_prefix}_infos_temporal_val.pkl")
        export_2d_annotation(root_path, info_train_path, version=version)
        export_2d_annotation(root_path, info_val_path, version=version)


parser = argparse.ArgumentParser(description="Data converter arg parser")
parser.add_argument("dataset", metavar="kitti", help="name of the dataset")
parser.add_argument("--root-path", type=str, default="./data/kitti", help="specify the root path of dataset")
parser.add_argument("--canbus", type=str, default="./data", help="specify the root path of nuScenes canbus")
parser.add_argument(
    "--version", type=str, default="v1.0", required=False, help="specify the dataset version, no need for kitti"
)
parser.add_argument("--max-sweeps", type=int, default=10, required=False, help="specify sweeps of lidar per example")
parser.add_argument("--out-dir", type=str, default="./data/kitti", required=False, help="name of info pkl")
parser.add_argument("--extra-tag", type=str, default="kitti")
parser.add_argument("--workers", type=int, default=4, help="number of threads to be used")
args = parser.parse_args()

if __name__ == "__main__":
    print("args.version:", args.version)
    if args.dataset == "nuscenes" and args.version != "v1.0-mini":
        train_version = f"{args.version}-trainval"
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name="CustomNuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
        )
        test_version = f"{args.version}-test"
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name="CustomNuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
        )
    elif args.dataset == "nuscenes" and args.version == "v1.0-mini":
        train_version = f"{args.version}"
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name="CustomNuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
        )
