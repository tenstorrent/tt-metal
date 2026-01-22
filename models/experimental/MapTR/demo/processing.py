# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
import glob
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.utils.data_classes import Box, LidarPointCloud

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(SCRIPT_DIR, "..", "resources", "nuScenes")

IMG_KEYS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]

SHOW_CLASSES = [
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

map_name_from_general_to_detection = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}


def generate_infos():
    """Generate sample information with embedded calibration and annotation data."""
    cam_names = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
    lidar_names = ["LIDAR_TOP"]

    cam_infos = {
        "CAM_FRONT": {
            "sample_token": "e93e98b63d3b40209056d129dc53ceee",
            "timestamp": 1533151603512404,
            "is_key_frame": True,
            "height": 900,
            "width": 1600,
            "filename": "samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg",
            "calibrated_sensor": {
                "token": "d3ab655f3cc540a88491ec218751f9c6",
                "sensor_token": "725903f5b62f56118f4094b46a4470d8",
                "translation": [1.72200568478, 0.00475453292289, 1.49491291905],
                "rotation": [0.5077241387638071, -0.4973392230703816, 0.49837167536166627, -0.4964832014373754],
                "camera_intrinsic": [
                    [1252.8131021185304, 0.0, 826.588114781398],
                    [0.0, 1252.8131021185304, 469.9846626224581],
                    [0.0, 0.0, 1.0],
                ],
            },
            "ego_pose": {
                "token": "4f5e35aa6c6a426ca945e206fb2f4921",
                "timestamp": 1533151603512404,
                "rotation": [-0.9687876119182126, -0.004506968075376869, -0.00792272203393983, 0.24772460658591755],
                "translation": [599.849775495386, 1647.6411294309523, 0.0],
            },
        },
        "CAM_FRONT_RIGHT": {
            "sample_token": "e93e98b63d3b40209056d129dc53ceee",
            "timestamp": 1533151603520482,
            "is_key_frame": True,
            "height": 900,
            "width": 1600,
            "filename": "samples/CAM_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151603520482.jpg",
            "calibrated_sensor": {
                "token": "69356358127a4756bafab3d45ec7d6b4",
                "sensor_token": "2f7ad058f1ac5557bf321c7543758f43",
                "translation": [1.58082565783, -0.499078711449, 1.51749368405],
                "rotation": [0.20335173766558642, -0.19146333228946724, 0.6785710044972951, -0.6793609166212989],
                "camera_intrinsic": [
                    [1256.7485116440405, 0.0, 817.7887570959712],
                    [0.0, 1256.7485116440403, 451.9541780095127],
                    [0.0, 0.0, 1.0],
                ],
            },
            "ego_pose": {
                "token": "5ed84fb1dbe24efcb00eb766a22d69d6",
                "timestamp": 1533151603520482,
                "rotation": [-0.9687599514054591, -0.004456697153369989, -0.007899682341935369, 0.2478343991908144],
                "translation": [599.9118549287866, 1647.606633933739, 0.0],
            },
        },
        "CAM_BACK_RIGHT": {
            "sample_token": "e93e98b63d3b40209056d129dc53ceee",
            "timestamp": 1533151603528113,
            "is_key_frame": True,
            "height": 900,
            "width": 1600,
            "filename": "samples/CAM_BACK_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151603528113.jpg",
            "calibrated_sensor": {
                "token": "b37d2a134cf24b4283e74e0654ae6a4b",
                "sensor_token": "ca7dba2ec9f95951bbe67246f7f2c3f7",
                "translation": [1.05945173053, -0.46720294852, 1.55050857555],
                "rotation": [0.13819187705364147, -0.13796718183628456, -0.6893329941542625, 0.697630335509333],
                "camera_intrinsic": [
                    [1249.9629280788233, 0.0, 825.3768045375984],
                    [0.0, 1249.9629280788233, 462.54816385708756],
                    [0.0, 0.0, 1.0],
                ],
            },
            "ego_pose": {
                "token": "8fafdaa824b74553b1a08011d29baf20",
                "timestamp": 1533151603528113,
                "rotation": [-0.9687345485285538, -0.0043670388304257405, -0.007816404838658813, 0.24793791011951208],
                "translation": [599.9705034252927, 1647.574034904777, 0.0],
            },
        },
        "CAM_BACK": {
            "sample_token": "e93e98b63d3b40209056d129dc53ceee",
            "timestamp": 1533151603537558,
            "is_key_frame": True,
            "height": 900,
            "width": 1600,
            "filename": "samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg",
            "calibrated_sensor": {
                "token": "78056a17635540eb9ed0d980d3e24520",
                "sensor_token": "ce89d4f3050b5892b33b3d328c5e82a3",
                "translation": [0.05524611077, 0.0107882366898, 1.56794286957],
                "rotation": [0.5067997344989889, -0.4977567019405021, -0.4987849934090844, 0.496594225837321],
                "camera_intrinsic": [
                    [796.8910634503094, 0.0, 857.7774326863696],
                    [0.0, 796.8910634503094, 476.8848988407415],
                    [0.0, 0.0, 1.0],
                ],
            },
            "ego_pose": {
                "token": "1908fe7dc09c474ebc6ea23b4c1c5401",
                "timestamp": 1533151603537558,
                "rotation": [-0.9687030311295038, -0.0042154863536825755, -0.007752028981545582, 0.24806566308536676],
                "translation": [600.0430992523302, 1647.5336699861132, 0.0],
            },
        },
        "CAM_BACK_LEFT": {
            "sample_token": "e93e98b63d3b40209056d129dc53ceee",
            "timestamp": 1533151603547405,
            "is_key_frame": True,
            "height": 900,
            "width": 1600,
            "filename": "samples/CAM_BACK_LEFT/n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151603547405.jpg",
            "calibrated_sensor": {
                "token": "d884789aacbc44649be792a2e203b9bf",
                "sensor_token": "a89643a5de885c6486df2232dc954da2",
                "translation": [1.04852047718, 0.483058131052, 1.56210154484],
                "rotation": [0.7048620297871717, -0.6907306801461466, -0.11209091960167808, 0.11617345743327073],
                "camera_intrinsic": [
                    [1254.9860565800168, 0.0, 829.5769333630991],
                    [0.0, 1254.9860565800168, 467.1680561863987],
                    [0.0, 0.0, 1.0],
                ],
            },
            "ego_pose": {
                "token": "e4233736f4ba4fd5989684f0f1e84377",
                "timestamp": 1533151603547405,
                "rotation": [-0.9686660835660069, -0.004081555849799428, -0.007697727348287311, 0.24821382806848222],
                "translation": [600.1185731195969, 1647.4917138239566, 0.0],
            },
        },
        "CAM_FRONT_LEFT": {
            "sample_token": "e93e98b63d3b40209056d129dc53ceee",
            "timestamp": 1533151603504799,
            "is_key_frame": True,
            "height": 900,
            "width": 1600,
            "filename": "samples/CAM_FRONT_LEFT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151603504799.jpg",
            "calibrated_sensor": {
                "token": "51406a6af1e34c6b80c1abe1b0304aca",
                "sensor_token": "ec4b5d41840a509984f7ec36419d4c09",
                "translation": [1.5752559464, 0.500519383135, 1.50696032589],
                "rotation": [0.6812088525125634, -0.6687507165046241, 0.2101702448905517, -0.21108161122114324],
                "camera_intrinsic": [
                    [1257.8625342125129, 0.0, 827.2410631095686],
                    [0.0, 1257.8625342125129, 450.915498205774],
                    [0.0, 0.0, 1.0],
                ],
            },
            "ego_pose": {
                "token": "27f02b3e285d4ca18015535511520b3e",
                "timestamp": 1533151603504799,
                "rotation": [-0.9688136386550925, -0.004554290680191179, -0.007944423174925015, 0.24762123926008034],
                "translation": [599.7913353051094, 1647.6735927814666, 0.0],
            },
        },
    }

    lidar_infos = {
        "LIDAR_TOP": {
            "sample_token": "e93e98b63d3b40209056d129dc53ceee",
            "timestamp": 1533151603547590,
            "filename": "samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin",
            "calibrated_sensor": {
                "token": "d051cafdd9fe4d999b413462364d44a0",
                "sensor_token": "dc8b396651c05aedbb9cdaae573bb567",
                "translation": [0.985793, 0.0, 1.84019],
                "rotation": [0.706749235646644, -0.015300993788500868, 0.01739745181256607, -0.7070846669051719],
                "camera_intrinsic": [],
            },
            "ego_pose": {
                "token": "fdddd75ee1d94f14a09991988dab8b3e",
                "timestamp": 1533151603547590,
                "rotation": [-0.968669701688471, -0.004043399262151301, -0.007666594265959211, 0.24820129589817977],
                "translation": [600.1202137947669, 1647.490776275174, 0.0],
            },
        }
    }

    ann_infos = [
        {
            "category_name": "human.pedestrian.adult",
            "translation": [637.141, 1636.252, -0.235],
            "size": [0.621, 0.647, 1.778],
            "rotation": [0.3495370229501108, 0.0, 0.0, 0.9369225526088982],
        },
        {
            "category_name": "human.pedestrian.adult",
            "translation": [612.719, 1632.142, 0.491],
            "size": [0.688, 0.944, 1.904],
            "rotation": [0.3598258294147673, 0.0, 0.0, 0.9330194920182401],
        },
        {
            "category_name": "human.pedestrian.adult",
            "translation": [619.03, 1648.941, 0.413],
            "size": [0.578, 0.613, 1.752],
            "rotation": [0.2267278025893585, 0.0, 0.0, 0.9739581631327913],
        },
        {
            "category_name": "human.pedestrian.adult",
            "translation": [635.378, 1674.253, -0.163],
            "size": [0.909, 1.105, 2.0],
            "rotation": [0.859218955073936, 0.0, 0.0, 0.5116080406342863],
        },
        {
            "category_name": "human.pedestrian.adult",
            "translation": [622.35, 1624.018, -0.016],
            "size": [0.751, 1.03, 1.975],
            "rotation": [0.3270447225565966, 0.0, 0.0, 0.9450088621001809],
        },
        {
            "category_name": "vehicle.car",
            "translation": [635.447, 1620.546, -0.326],
            "size": [2.001, 4.734, 1.481],
            "rotation": [0.902945631703063, 0.0, 0.0, -0.4297547977490846],
        },
        {
            "category_name": "human.pedestrian.adult",
            "translation": [598.103, 1642.075, 1.029],
            "size": [0.631, 0.61, 1.929],
            "rotation": [0.9842156913650645, 0.0, 0.0, -0.1769730851592639],
        },
        {
            "category_name": "human.pedestrian.adult",
            "translation": [622.249, 1646.081, 0.321],
            "size": [0.697, 0.498, 1.761],
            "rotation": [0.24369118456274716, 0.0, 0.0, 0.9698528788256522],
        },
        {
            "category_name": "human.pedestrian.adult",
            "translation": [619.603, 1624.655, 0.071],
            "size": [0.665, 0.736, 1.89],
            "rotation": [0.9500782963408733, 0.0, 0.0, -0.31201158764062575],
        },
        {
            "category_name": "vehicle.car",
            "translation": [583.549, 1656.391, 1.267],
            "size": [1.871, 4.488, 1.515],
            "rotation": [0.9708735025408489, 0.0, 0.0, -0.23959265861888215],
        },
        {
            "category_name": "human.pedestrian.adult",
            "translation": [627.008, 1617.877, -0.387],
            "size": [0.546, 0.439, 1.622],
            "rotation": [0.5418842462178632, 0.0, 0.0, 0.8404531299845923],
        },
        {
            "category_name": "human.pedestrian.adult",
            "translation": [640.863, 1643.013, -0.285],
            "size": [0.64, 0.395, 1.807],
            "rotation": [0.7735748318721248, 0.0, 0.0, 0.6337049624975442],
        },
        {
            "category_name": "human.pedestrian.adult",
            "translation": [602.152, 1626.301, 0.234],
            "size": [0.489, 0.491, 1.851],
            "rotation": [0.8737340236499682, 0.0, 0.0, 0.4864040048318236],
        },
        {
            "category_name": "human.pedestrian.adult",
            "translation": [631.815, 1636.973, 0.074],
            "size": [0.612, 0.736, 1.877],
            "rotation": [-0.21480351118958524, 0.0, 0.0, 0.9766572846094098],
        },
        {
            "category_name": "vehicle.car",
            "translation": [660.851, 1604.404, -0.423],
            "size": [1.803, 4.495, 1.56],
            "rotation": [0.35256038671077344, 0.0, 0.0, 0.935789064758907],
        },
        {
            "category_name": "human.pedestrian.adult",
            "translation": [650.737, 1625.23, -0.3],
            "size": [0.585, 0.681, 1.711],
            "rotation": [0.9770697356884313, 0.0, 0.0, -0.21291954255478537],
        },
        {
            "category_name": "human.pedestrian.adult",
            "translation": [582.275, 1668.561, 1.473],
            "size": [0.908, 1.109, 2.211],
            "rotation": [0.2657664441485724, 0.0, 0.0, 0.9640374459348681],
        },
        {
            "category_name": "human.pedestrian.adult",
            "translation": [597.73, 1641.374, 0.993],
            "size": [0.712, 0.601, 1.891],
            "rotation": [0.9715759823183562, 0.0, 0.0, -0.23672792522666428],
        },
        {
            "category_name": "human.pedestrian.adult",
            "translation": [636.797, 1680.132, -0.198],
            "size": [0.755, 1.235, 2.083],
            "rotation": [0.8724192438664237, 0.0, 0.0, 0.4887582868162314],
        },
        {
            "category_name": "vehicle.car",
            "translation": [582.374, 1660.997, 1.38],
            "size": [2.037, 4.958, 1.639],
            "rotation": [0.25477072568470893, 0.0, 0.0, 0.9670014877620855],
        },
        {
            "category_name": "human.pedestrian.adult",
            "translation": [639.585, 1606.675, -0.122],
            "size": [0.724, 0.828, 1.835],
            "rotation": [0.940133570865004, 0.0, 0.0, -0.3408062043634423],
        },
        {
            "category_name": "human.pedestrian.adult",
            "translation": [631.807, 1644.622, -0.459],
            "size": [0.738, 0.783, 1.52],
            "rotation": [0.8805848980472477, 0.0, 0.0, 0.47388842287095206],
        },
        {
            "category_name": "human.pedestrian.adult",
            "translation": [637.791, 1636.674, -0.011],
            "size": [0.699, 0.738, 1.95],
            "rotation": [0.36583533995688255, 0.0, 0.0, 0.9306795925766462],
        },
    ]

    info = {
        "sample_token": "e93e98b63d3b40209056d129dc53ceee",
        "timestamp": 1533151603547590,
        "scene_token": "bebf5f5b2a674631ab5c88fd1aa9e87a",
        "cam_infos": cam_infos,
        "lidar_infos": lidar_infos,
        "cam_sweeps": [],
        "lidar_sweeps": [],
        "ann_infos": ann_infos,
    }

    for cam_name in cam_names:
        cam_pattern = os.path.join(RESOURCES_DIR, "samples", cam_name, "*.jpg")
        cam_files = glob.glob(cam_pattern)
        if cam_files:
            rel_path = os.path.relpath(cam_files[0], RESOURCES_DIR)
            info["cam_infos"][cam_name]["filename"] = rel_path

    for lidar_name in lidar_names:
        lidar_pattern = os.path.join(RESOURCES_DIR, "samples", lidar_name, "*.bin")
        lidar_files = glob.glob(lidar_pattern)
        if lidar_files:
            rel_path = os.path.relpath(lidar_files[0], RESOURCES_DIR)
            info["lidar_infos"][lidar_name]["filename"] = rel_path

    return [info]


def generate_map_annotations():
    """
    Generate map annotations data structure that replaces the JSON map_ann_file.

    Returns:
        dict: Map annotations in the format expected by CustomNuScenesLocalMapDataset
        Format: {"GTs": [{"sample_token": str, "vectors": [...]}]}
    """

    # Map classes: divider, ped_crossing, boundary
    # For this sample, we'll create some example map vectors
    sample_token = "e93e98b63d3b40209056d129dc53ceee"

    # Example map vectors (divider lines)
    # These are in ego frame (centered at vehicle)
    # Format: each vector has pts (list of [x, y] coordinates), pts_num, cls_name, type
    vectors = [
        {
            "pts": [[-5.0, -10.0], [-5.0, 10.0]],  # Left divider line
            "pts_num": 2,
            "cls_name": "divider",
            "type": 0,
        },
        {
            "pts": [[5.0, -10.0], [5.0, 10.0]],  # Right divider line
            "pts_num": 2,
            "cls_name": "divider",
            "type": 0,
        },
        {
            "pts": [[-2.0, 0.0], [2.0, 0.0]],  # Pedestrian crossing
            "pts_num": 2,
            "cls_name": "ped_crossing",
            "type": 1,
        },
    ]

    gt_anno = {
        "sample_token": sample_token,
        "vectors": vectors,
    }

    return {"GTs": [gt_anno]}


def write_map_annotations_to_file(output_path):
    """
    Generate map annotations and write them to a JSON file.

    Args:
        output_path: Path where the JSON file should be written
    """
    import json
    import os

    map_ann_data = generate_map_annotations()

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(map_ann_data, f, indent=2)

    return output_path


def generate_sample_info_data():
    """
    Generate sample info data structure that replaces the pickle ann_file.

    Returns:
        dict: Sample info data in the format expected by NuScenesDataset
        Format: {"infos": [...], "metadata": {...}}
    """
    from pyquaternion import Quaternion

    infos = generate_infos()

    # Convert to the format expected by the dataset
    data_infos = []
    for info in infos:
        lidar_info = info["lidar_infos"]["LIDAR_TOP"]

        # Get ego pose from lidar (most recent/relevant)
        ego2global_translation = lidar_info["ego_pose"]["translation"]
        ego2global_rotation = lidar_info["ego_pose"]["rotation"]

        # Get lidar to ego transformation
        lidar2ego_translation = lidar_info["calibrated_sensor"]["translation"]
        lidar2ego_rotation = lidar_info["calibrated_sensor"]["rotation"]

        # Build the data info structure expected by NuScenesDataset
        data_info = {
            "token": info["sample_token"],
            "timestamp": info["timestamp"],
            "scene_token": info["scene_token"],
            "lidar_path": lidar_info["filename"],
            "sweeps": info.get("lidar_sweeps", []),
            "ego2global_translation": ego2global_translation,
            "ego2global_rotation": ego2global_rotation,
            "lidar2ego_translation": lidar2ego_translation,
            "lidar2ego_rotation": lidar2ego_rotation,
            "prev": None,  # Previous sample index (None for single sample)
            "next": None,  # Next sample index (None for single sample)
            "can_bus": np.zeros(18, dtype=np.float32),  # CAN bus data (18 features)
            "frame_idx": 0,  # Frame index
            "map_location": "singapore-onenorth",  # Default map location
            "cams": {},
        }

        # Add camera information
        for cam_name, cam_info in info["cam_infos"].items():
            # Get camera intrinsic - ensure it's a 2D array
            cam_intrinsic_raw = cam_info["calibrated_sensor"]["camera_intrinsic"]
            cam_intrinsic = np.array(cam_intrinsic_raw)
            if cam_intrinsic.ndim == 1:
                # Reshape if 1D (shouldn't happen but handle it)
                cam_intrinsic = cam_intrinsic.reshape(-1, 3)

            # Compute sensor2lidar transformation (inverse of lidar2sensor)
            # sensor2lidar = lidar2ego^-1 @ sensor2ego
            # First compute sensor2ego rotation matrix
            # Quaternion expects [w, x, y, z] format
            sensor2ego_rot_quat = Quaternion(cam_info["calibrated_sensor"]["rotation"])
            sensor2ego_rot_mat = sensor2ego_rot_quat.rotation_matrix

            # Compute lidar2ego rotation matrix
            lidar2ego_rot_quat = Quaternion(lidar2ego_rotation)
            lidar2ego_rot_mat = lidar2ego_rot_quat.rotation_matrix

            # sensor2lidar = lidar2ego^-1 @ sensor2ego
            sensor2lidar_rot_mat = lidar2ego_rot_mat.T @ sensor2ego_rot_mat

            # Compute sensor2lidar translation
            # sensor2lidar_t = lidar2ego_R^-1 @ (sensor2ego_t - lidar2ego_t)
            sensor2lidar_translation = lidar2ego_rot_mat.T @ (
                np.array(cam_info["calibrated_sensor"]["translation"]) - np.array(lidar2ego_translation)
            )

            data_info["cams"][cam_name] = {
                "data_path": cam_info["filename"],
                "sensor2lidar_rotation": sensor2lidar_rot_mat.tolist(),  # 3x3 rotation matrix
                "sensor2lidar_translation": sensor2lidar_translation.tolist(),
                "sensor2ego_translation": cam_info["calibrated_sensor"]["translation"],
                "sensor2ego_rotation": cam_info["calibrated_sensor"]["rotation"],
                "cam_intrinsic": cam_intrinsic,  # Keep as numpy array for pickle
                "ego2global_translation": cam_info["ego_pose"]["translation"],
                "ego2global_rotation": cam_info["ego_pose"]["rotation"],
            }

        data_infos.append(data_info)

    metadata = {
        "version": "v1.0-mini",
        "use_camera": True,
        "use_lidar": True,
        "use_radar": False,
        "use_map": True,
        "use_external": True,
    }

    return {"infos": data_infos, "metadata": metadata}


def write_sample_info_to_file(output_path):
    """
    Generate sample info data and write it to a pickle file.

    Args:
        output_path: Path where the pickle file should be written
    """
    import pickle
    import os

    sample_data = generate_sample_info_data()

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(sample_data, f)

    return output_path


def load_images_and_mats(info):
    """
    Load and preprocess camera images and transformation matrices.

    Args:
        info: Sample information dictionary containing camera info

    Returns:
        imgs: Preprocessed image tensor [B, num_sweeps, num_cameras, C, H, W]
        mats_dict: Dictionary containing transformation matrices
        ego2global_rotation: Ego to global rotation
        ego2global_translation: Ego to global translation
    """
    img_mean = np.array([123.675, 116.28, 103.53], np.float32)
    img_std = np.array([58.395, 57.12, 57.375], np.float32)

    sweep_imgs = []
    sweep_sensor2ego_mats = []
    sweep_intrin_mats = []
    sweep_ida_mats = []
    sweep_sensor2sensor_mats = []

    cam_info = info["cam_infos"]

    ida_aug_conf = {
        "H": 900,
        "W": 1600,
        "final_dim": [256, 704],
        "bot_pct_lim": [0.0, 0.0],
        "resize_lim": [0.386, 0.55],
        "rot_lim": [0.0, 0.0],
        "rand_flip": False,
    }

    H, W = ida_aug_conf["H"], ida_aug_conf["W"]
    fH, fW = ida_aug_conf["final_dim"]
    resize = max(fH / H, fW / W)
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    crop_h = int((1 - np.mean(ida_aug_conf["bot_pct_lim"])) * newH) - fH
    crop_w = int(max(0, newW - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

    for cam in IMG_KEYS:
        img_path = os.path.join(RESOURCES_DIR, cam_info[cam]["filename"])
        img = Image.open(img_path)
        img = img.resize(resize_dims)
        img = img.crop(crop)

        ida_mat = torch.eye(4)
        ida_mat[0, 0] = resize
        ida_mat[1, 1] = resize
        ida_mat[0, 3] = -crop[0]
        ida_mat[1, 3] = -crop[1]

        img_np = np.array(img, dtype=np.float32)
        img_np = (img_np - img_mean) / img_std
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        # Sensor to ego transformation (camera to vehicle frame)
        w, x, y, z = cam_info[cam]["calibrated_sensor"]["rotation"]
        sensor2ego_rot = torch.Tensor(Quaternion(w, x, y, z).rotation_matrix)
        sensor2ego_tran = torch.Tensor(cam_info[cam]["calibrated_sensor"]["translation"])
        sensor2ego = torch.eye(4)
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, 3] = sensor2ego_tran

        # For key frame, sensor2ego_mats is just the sensor2ego transform
        sweepsensor2keyego = sensor2ego

        intrin_mat = torch.zeros((4, 4))
        intrin_mat[3, 3] = 1
        intrin_mat[:3, :3] = torch.Tensor(cam_info[cam]["calibrated_sensor"]["camera_intrinsic"])

        sweep_imgs.append(img_tensor)
        sweep_sensor2ego_mats.append(sweepsensor2keyego)
        sweep_intrin_mats.append(intrin_mat)
        sweep_ida_mats.append(ida_mat)
        sweep_sensor2sensor_mats.append(torch.eye(4))

    # Stack and reshape to [B, num_sweeps, num_cameras, ...]
    imgs = torch.stack(sweep_imgs).unsqueeze(0).unsqueeze(0)  # [1, 1, 6, 3, H, W]
    sensor2ego_mats = torch.stack(sweep_sensor2ego_mats).unsqueeze(0).unsqueeze(0)  # [1, 1, 6, 4, 4]
    intrin_mats = torch.stack(sweep_intrin_mats).unsqueeze(0).unsqueeze(0)
    ida_mats = torch.stack(sweep_ida_mats).unsqueeze(0).unsqueeze(0)
    sensor2sensor_mats = torch.stack(sweep_sensor2sensor_mats).unsqueeze(0).unsqueeze(0)

    # For 2-key model, duplicate sweep to have 2 sweeps
    imgs = imgs.repeat(1, 2, 1, 1, 1, 1)
    sensor2ego_mats = sensor2ego_mats.repeat(1, 2, 1, 1, 1)
    intrin_mats = intrin_mats.repeat(1, 2, 1, 1, 1)
    ida_mats = ida_mats.repeat(1, 2, 1, 1, 1)
    sensor2sensor_mats = sensor2sensor_mats.repeat(1, 2, 1, 1, 1)

    mats_dict = {
        "sensor2ego_mats": sensor2ego_mats,
        "intrin_mats": intrin_mats,
        "ida_mats": ida_mats,
        "sensor2sensor_mats": sensor2sensor_mats,
        "bda_mat": torch.eye(4).unsqueeze(0),
    }

    lidar_info = info["lidar_infos"]["LIDAR_TOP"]
    ego2global_rotation = lidar_info["ego_pose"]["rotation"]
    ego2global_translation = lidar_info["ego_pose"]["translation"]

    return imgs, mats_dict, ego2global_rotation, ego2global_translation


def load_lidar_points(info):
    """
    Load and transform LiDAR points to ego frame.

    Args:
        info: Sample information dictionary containing LiDAR info

    Returns:
        points: LiDAR points in ego frame [N, 4]
    """
    lidar_path = info["lidar_infos"]["LIDAR_TOP"]["filename"]
    lidar_points = np.fromfile(os.path.join(RESOURCES_DIR, lidar_path), dtype=np.float32, count=-1).reshape(-1, 5)[
        ..., :4
    ]
    lidar_calibrated_sensor = info["lidar_infos"]["LIDAR_TOP"]["calibrated_sensor"]

    pts = LidarPointCloud(lidar_points.T)
    pts.rotate(Quaternion(lidar_calibrated_sensor["rotation"]).rotation_matrix)
    pts.translate(np.array(lidar_calibrated_sensor["translation"]))
    return pts.points.T


def get_ego_box(box_dict, ego2global_rotation, ego2global_translation):
    """
    Transform a box from global frame to ego frame.

    Args:
        box_dict: Dictionary containing box translation, size, and rotation
        ego2global_rotation: Ego to global rotation quaternion
        ego2global_translation: Ego to global translation

    Returns:
        box_ego: Box in ego frame [x, y, z, dx, dy, dz, yaw, vx, vy]
    """
    box = Box(
        box_dict["translation"],
        box_dict["size"],
        Quaternion(box_dict["rotation"]),
    )
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    box.translate(trans)
    box.rotate(rot)
    box_xyz = np.array(box.center)
    box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
    box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
    box_velo = np.array(box.velocity[:2])
    return np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])


def rotate_points_along_z(points, angle):
    """
    Rotate points along the z-axis.

    Args:
        points: Points to rotate [N, 8, 3]
        angle: Rotation angle [N]

    Returns:
        points_rot: Rotated points [N, 8, 3]
    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0])
    ones = np.ones(points.shape[0])
    rot_matrix = np.stack((cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), axis=1).reshape(-1, 3, 3)
    points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot


def get_corners(boxes3d):
    """
    Convert 3D boxes to corner points.

    Args:
        boxes3d: 3D boxes [N, 7] (x, y, z, dx, dy, dz, yaw)

    Returns:
        corners3d: Corner points [N, 8, 3]
    """
    template = (
        np.array(
            (
                [1, 1, -1],
                [1, -1, -1],
                [-1, -1, -1],
                [-1, 1, -1],
                [1, 1, 1],
                [1, -1, 1],
                [-1, -1, 1],
                [-1, 1, 1],
            )
        )
        / 2
    )
    corners3d = np.tile(boxes3d[:, None, 3:6], [1, 8, 1]) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.reshape(-1, 8, 3), boxes3d[:, 6]).reshape(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]
    return corners3d


def get_bev_lines(corners):
    """
    Extract BEV (bird's eye view) lines from corner points.

    Args:
        corners: Corner points [8, 3]

    Returns:
        lines: List of line segments for BEV visualization
    """
    return [[[corners[i, 0], corners[(i + 1) % 4, 0]], [corners[i, 1], corners[(i + 1) % 4, 1]]] for i in range(4)]


def get_3d_lines(corners):
    """
    Extract 3D lines from corner points for camera view visualization.

    Args:
        corners: Corner points [8, 3]

    Returns:
        lines: List of line segments for 3D visualization
    """
    ret = []
    for st, ed in [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
        if corners[st, -1] > 0 and corners[ed, -1] > 0:
            ret.append([[corners[st, 0], corners[ed, 0]], [corners[st, 1], corners[ed, 1]]])
    return ret


def get_cam_corners(corners, translation, rotation, cam_intrinsics):
    """
    Transform corner points to camera frame and project to image plane.

    Args:
        corners: Corner points in global/ego frame [8, 3]
        translation: Camera translation
        rotation: Camera rotation quaternion
        cam_intrinsics: Camera intrinsic matrix [3, 3]

    Returns:
        cam_corners: Projected corner points [8, 3] (x, y, depth)
    """
    cam_corners = corners.copy()
    cam_corners -= np.array(translation)
    cam_corners = cam_corners @ Quaternion(rotation).inverse.rotation_matrix.T
    cam_corners = cam_corners @ np.array(cam_intrinsics).T
    valid = cam_corners[:, -1] > 0
    cam_corners /= cam_corners[:, 2:3]
    cam_corners[~valid] = 0
    return cam_corners


def decode_predictions(preds, class_names, score_threshold=0.3):
    """
    Decode model predictions to bounding boxes.

    Args:
        preds: Model predictions (list of task predictions)
        class_names: Class names for each task
        score_threshold: Score threshold for filtering detections

    Returns:
        boxes_list: List of boxes [x, y, z, dx, dy, dz, yaw, vx, vy]
        classes_list: List of class names
        scores_list: List of detection scores
    """
    boxes_list = []
    classes_list = []
    scores_list = []

    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    voxel_size = [0.8, 0.8, 8.0]
    out_size_factor = 1

    for task_idx, task_pred in enumerate(preds):
        if isinstance(task_pred, list):
            pred_dict = task_pred[0]
        else:
            pred_dict = task_pred

        heatmap = pred_dict["heatmap"].sigmoid()
        reg = pred_dict["reg"]
        height = pred_dict["height"]
        dim = pred_dict["dim"]
        rot = pred_dict["rot"]
        vel = pred_dict.get("vel", None)

        batch_size, num_classes, H, W = heatmap.shape
        num_task_classes = len(class_names[task_idx])

        for b in range(batch_size):
            for c in range(min(num_classes, num_task_classes)):
                heat = heatmap[b, c]
                mask = heat > score_threshold

                if mask.sum() == 0:
                    continue

                ys, xs = torch.where(mask)
                scores = heat[mask]

                for i in range(len(xs)):
                    x_idx = xs[i].item()
                    y_idx = ys[i].item()
                    score = scores[i].item()

                    # Decode position
                    x = (x_idx + reg[b, 0, y_idx, x_idx].item()) * voxel_size[0] * out_size_factor + pc_range[0]
                    y = (y_idx + reg[b, 1, y_idx, x_idx].item()) * voxel_size[1] * out_size_factor + pc_range[1]
                    z = height[b, 0, y_idx, x_idx].item()

                    # Decode dimensions
                    dx = dim[b, 0, y_idx, x_idx].item()
                    dy = dim[b, 1, y_idx, x_idx].item()
                    dz = dim[b, 2, y_idx, x_idx].item()

                    # Decode rotation
                    rot_sin = rot[b, 0, y_idx, x_idx].item()
                    rot_cos = rot[b, 1, y_idx, x_idx].item()
                    yaw = np.arctan2(rot_sin, rot_cos)

                    # Velocity
                    vx, vy = 0, 0
                    if vel is not None:
                        vx = vel[b, 0, y_idx, x_idx].item()
                        vy = vel[b, 1, y_idx, x_idx].item()

                    boxes_list.append([x, y, z, dx, dy, dz, yaw, vx, vy])
                    classes_list.append(class_names[task_idx][c])
                    scores_list.append(score)

    return boxes_list, classes_list, scores_list


def boxes_to_corners(boxes_list, classes_list, show_range):
    """
    Convert boxes to corner format for visualization.

    Args:
        boxes_list: List of boxes [x, y, z, dx, dy, dz, yaw, vx, vy]
        classes_list: List of class names
        show_range: Maximum range to show (meters)

    Returns:
        pred_corners: List of corner points
        pred_classes: List of filtered class names
    """
    pred_corners = []
    pred_classes = []

    for box, cls in zip(boxes_list, classes_list):
        if cls not in SHOW_CLASSES:
            continue
        box_np = np.array(box[:9])
        if np.linalg.norm(box_np[:2]) <= show_range:
            corners = get_corners(box_np[None])[0]
            pred_corners.append(corners)
            pred_classes.append(cls)

    return pred_corners, pred_classes


def get_gt_corners(info, ego2global_rotation, ego2global_translation, show_range):
    """
    Get ground truth corners from annotations.

    Args:
        info: Sample information dictionary
        ego2global_rotation: Ego to global rotation
        ego2global_translation: Ego to global translation
        show_range: Maximum range to show (meters)

    Returns:
        gt_corners: List of ground truth corner points
    """
    gt_corners = []
    for ann in info["ann_infos"]:
        if map_name_from_general_to_detection.get(ann["category_name"], "ignore") in SHOW_CLASSES:
            box = get_ego_box(
                dict(
                    size=ann["size"],
                    rotation=ann["rotation"],
                    translation=ann["translation"],
                ),
                ego2global_rotation,
                ego2global_translation,
            )
            if np.linalg.norm(box[:2]) <= show_range:
                corners = get_corners(box[None])[0]
                gt_corners.append(corners)

    return gt_corners


def visualize_results(
    info,
    pts,
    pred_corners_torch,
    pred_classes_torch,
    pred_corners_ttnn,
    pred_classes_ttnn,
    gt_corners,
    output_path,
    show_range=60,
):
    """
    Visualize BEVDepth predictions and ground truth.

    Args:
        info: Sample information dictionary
        pts: LiDAR points
        pred_corners_torch: Torch prediction corners
        pred_classes_torch: Torch prediction classes
        pred_corners_ttnn: TTNN prediction corners (can be None)
        pred_classes_ttnn: TTNN prediction classes (can be None)
        gt_corners: Ground truth corners
        output_path: Output file path for visualization
        show_range: Maximum range to show (meters)
    """
    import cv2
    import matplotlib.pyplot as plt

    cam_info = info["cam_infos"]

    show_torch = len(pred_corners_torch) > 0

    if pred_corners_ttnn is not None:
        if show_torch:
            fig = plt.figure(figsize=(24, 16))
            num_rows = 4
        else:
            fig = plt.figure(figsize=(24, 8))
            num_rows = 2
    else:
        fig = plt.figure(figsize=(24, 8))
        num_rows = 2

    if show_torch:
        for i, k in enumerate(IMG_KEYS):
            fig_idx = i + 1 if i < 3 else i + 2
            ax = plt.subplot(num_rows, 4, fig_idx)
            ax.set_title(f"{k} (Torch)" if pred_corners_ttnn is not None else k)
            ax.axis("off")
            ax.set_xlim(0, 1600)
            ax.set_ylim(900, 0)

            img_path = os.path.join(RESOURCES_DIR, cam_info[k]["filename"])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)

            for corners, cls in zip(pred_corners_torch, pred_classes_torch):
                cam_corners = get_cam_corners(
                    corners,
                    cam_info[k]["calibrated_sensor"]["translation"],
                    cam_info[k]["calibrated_sensor"]["rotation"],
                    cam_info[k]["calibrated_sensor"]["camera_intrinsic"],
                )
                lines = get_3d_lines(cam_corners)
                for line in lines:
                    ax.plot(
                        line[0],
                        line[1],
                        c=plt.colormaps["tab10"](SHOW_CLASSES.index(cls) if cls in SHOW_CLASSES else 0),
                    )

        ax_bev_torch = plt.subplot(num_rows, 4, 4)
        ax_bev_torch.set_title("BEV (Torch)")
        ax_bev_torch.axis("equal")
        ax_bev_torch.set_xlim(-40, 40)
        ax_bev_torch.set_ylim(-40, 40)

        ax_bev_torch.scatter(-pts[:, 1], pts[:, 0], s=0.01, c=pts[:, -1], cmap="gray")

        for corners in gt_corners:
            lines = get_bev_lines(corners)
            for line in lines:
                ax_bev_torch.plot([-x for x in line[1]], line[0], c="r", label="ground truth")

        for corners in pred_corners_torch:
            lines = get_bev_lines(corners)
            for line in lines:
                ax_bev_torch.plot([-x for x in line[1]], line[0], c="g", label="torch prediction")

        handles, labels = ax_bev_torch.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_bev_torch.legend(by_label.values(), by_label.keys(), loc="upper right", framealpha=1)

    if pred_corners_ttnn is not None:
        start_idx = 9 if show_torch else 1

        for i, k in enumerate(IMG_KEYS):
            if show_torch:
                fig_idx = i + 9 if i < 3 else i + 10
            else:
                fig_idx = i + 1 if i < 3 else i + 2
            ax = plt.subplot(num_rows, 4, fig_idx)
            ax.set_title(f"{k} (TTNN)" if show_torch else k)
            ax.axis("off")
            ax.set_xlim(0, 1600)
            ax.set_ylim(900, 0)

            img_path = os.path.join(RESOURCES_DIR, cam_info[k]["filename"])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)

            for corners, cls in zip(pred_corners_ttnn, pred_classes_ttnn):
                cam_corners = get_cam_corners(
                    corners,
                    cam_info[k]["calibrated_sensor"]["translation"],
                    cam_info[k]["calibrated_sensor"]["rotation"],
                    cam_info[k]["calibrated_sensor"]["camera_intrinsic"],
                )
                lines = get_3d_lines(cam_corners)
                for line in lines:
                    ax.plot(
                        line[0],
                        line[1],
                        c=plt.colormaps["tab10"](SHOW_CLASSES.index(cls) if cls in SHOW_CLASSES else 0),
                    )

        # BEV for TTNN
        bev_fig_idx = 12 if show_torch else 4
        ax_bev_ttnn = plt.subplot(num_rows, 4, bev_fig_idx)
        ax_bev_ttnn.set_title("BEV (TTNN)")
        ax_bev_ttnn.axis("equal")
        ax_bev_ttnn.set_xlim(-40, 40)
        ax_bev_ttnn.set_ylim(-40, 40)

        ax_bev_ttnn.scatter(-pts[:, 1], pts[:, 0], s=0.01, c=pts[:, -1], cmap="gray")

        # Always show ground truth in TTNN visualization
        for corners in gt_corners:
            lines = get_bev_lines(corners)
            for line in lines:
                ax_bev_ttnn.plot([-x for x in line[1]], line[0], c="r", label="ground truth")

        for corners in pred_corners_ttnn:
            lines = get_bev_lines(corners)
            for line in lines:
                ax_bev_ttnn.plot([-x for x in line[1]], line[0], c="b", label="ttnn prediction")

        handles, labels = ax_bev_ttnn.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_bev_ttnn.legend(by_label.values(), by_label.keys(), loc="upper right", framealpha=1)

    plt.tight_layout(w_pad=0, h_pad=2)
    plt.savefig(output_path, dpi=150)
