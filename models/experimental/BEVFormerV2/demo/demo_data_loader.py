# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import numpy as np


SAMPLE_0_INFO = {
    "token": "3e8750f331d7499e9b5123e9eb70f2e2",
    "can_bus": [0.0] * 18,
    "ego2global_translation": [600.1202137947669, 1647.490776275174, 0.0],
    "ego2global_rotation": [-0.968669701688471, -0.004043399262151301, -0.007666594265959211, 0.24820129589817977],
    "lidar2ego_translation": [0.985793, 0.0, 1.84019],
    "lidar2ego_rotation": [0.706749235646644, -0.015300993788500868, 0.01739745181256607, -0.7070846669051719],
    "timestamp": 1533151603547590.0,
    "cams": {
        "CAM_FRONT": {
            "data_path": "./data/nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg",
            "sensor2lidar_rotation": [
                [0.9998801270116863, -0.01013819478916953, -0.01170250458296505],
                [0.012232575354279787, 0.05390463505646997, 0.9984711585316994],
                [-0.009491875857770705, -0.9984946205793245, 0.054022189578496575],
            ],
            "sensor2lidar_translation": [-0.006275140311572613, 0.4437230287236389, -0.3316126716045531],
            "cam_intrinsic": [
                [1252.8131021185304, 0.0, 826.588114781398],
                [0.0, 1252.8131021185304, 469.9846626224581],
                [0.0, 0.0, 1.0],
            ],
        },
        "CAM_FRONT_RIGHT": {
            "data_path": "./data/nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151603520482.jpg",
            "sensor2lidar_rotation": [
                [0.5372736788664327, -0.001367748330040076, 0.843406855119068],
                [-0.8417394728983844, 0.0620003137815353, 0.5363120554823875],
                [-0.05302502958114645, -0.9980751927362483, 0.0321598489178931],
            ],
            "sensor2lidar_translation": [0.49830135431261624, 0.3730319110657092, -0.30971646952113474],
            "cam_intrinsic": [
                [1256.7485116440405, 0.0, 817.7887570959712],
                [0.0, 1256.7485116440403, 451.9541780095127],
                [0.0, 0.0, 1.0],
            ],
        },
        "CAM_FRONT_LEFT": {
            "data_path": "./data/nuscenes/samples/CAM_FRONT_LEFT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151603504799.jpg",
            "sensor2lidar_rotation": [
                [0.5672581511644471, -0.014333426843678844, -0.8234152918257058],
                [0.8228127923991059, 0.051874023077259974, 0.5659400978850749],
                [0.03460200285939523, -0.9985507691673464, 0.041219689390163995],
            ],
            "sensor2lidar_translation": [-0.5023761049415043, 0.22914751525587462, -0.3316580138974885],
            "cam_intrinsic": [
                [1257.8625342125129, 0.0, 827.2410631095686],
                [0.0, 1257.8625342125129, 450.915498205774],
                [0.0, 0.0, 1.0],
            ],
        },
        "CAM_BACK": {
            "data_path": "./data/nuscenes/samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg",
            "sensor2lidar_rotation": [
                [-0.9999283380727066, -0.008594852305998796, -0.008333500644689665],
                [0.007990712947773468, 0.039174287069874525, -0.9992004422232587],
                [0.008914439171549637, -0.9991954282053173, -0.03910280076735698],
            ],
            "sensor2lidar_translation": [-0.009512203183021484, -1.0046424905119693, -0.3205656017442351],
            "cam_intrinsic": [
                [796.8910634503094, 0.0, 857.7774326863696],
                [0.0, 796.8910634503094, 476.8848988407415],
                [0.0, 0.0, 1.0],
            ],
        },
        "CAM_BACK_LEFT": {
            "data_path": "./data/nuscenes/samples/CAM_BACK_LEFT/n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151603547405.jpg",
            "sensor2lidar_rotation": [
                [-0.31910314470327766, -0.015891215448150118, -0.9475867518660553],
                [0.9468607653579226, 0.037220814098564134, -0.31948286655727137],
                [0.04034692139792285, -0.999180704512163, 0.0031694895944351597],
            ],
            "sensor2lidar_translation": [-0.48218189331873873, 0.07357368426630728, -0.27649453910384736],
            "cam_intrinsic": [
                [1254.9860565800168, 0.0, 829.5769333630991],
                [0.0, 1254.9860565800168, 467.1680561863987],
                [0.0, 0.0, 1.0],
            ],
        },
        "CAM_BACK_RIGHT": {
            "data_path": "./data/nuscenes/samples/CAM_BACK_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151603528113.jpg",
            "sensor2lidar_rotation": [
                [-0.38201342410869077, 0.013854058151105826, 0.9240529253638575],
                [-0.923050639707394, 0.043186672761450676, -0.38224655372098476],
                [-0.04520243728526039, -0.9989709587212956, -0.003709891498541602],
            ],
            "sensor2lidar_translation": [0.4673898584139806, -0.08280982434399675, -0.2960748534933302],
            "cam_intrinsic": [
                [1249.9629280788233, 0.0, 825.3768045375984],
                [0.0, 1249.9629280788233, 462.54816385708756],
                [0.0, 0.0, 1.0],
            ],
        },
    },
}


def convert_to_numpy(info):
    result = {}
    for key, value in info.items():
        if key == "cams":
            result[key] = {}
            for cam_name, cam_data in value.items():
                result[key][cam_name] = {}
                for cam_key, cam_value in cam_data.items():
                    if cam_key in ["sensor2lidar_rotation", "cam_intrinsic"]:
                        result[key][cam_name][cam_key] = np.array(cam_value, dtype=np.float32)
                    elif cam_key == "sensor2lidar_translation":
                        result[key][cam_name][cam_key] = np.array(cam_value, dtype=np.float32)
                    else:
                        result[key][cam_name][cam_key] = cam_value
        elif key in ["can_bus", "ego2global_translation", "lidar2ego_translation"]:
            result[key] = np.array(value, dtype=np.float32)
        elif key in ["ego2global_rotation", "lidar2ego_rotation"]:
            # These are quaternions (4 elements) or rotation matrices (3x3)
            result[key] = np.array(value, dtype=np.float32)
        elif key == "timestamp":
            result[key] = float(value)
        else:
            result[key] = value
    return result


def load_demo_data(sample_idx=0):
    """
    Load demo data generated on-the-fly.

    Args:
        sample_idx: Sample index to load (default: 0, the only sample with images).

    Returns:
        List of info dictionaries (compatible with original pkl format).
    """
    if sample_idx == 0:
        info = convert_to_numpy(SAMPLE_0_INFO.copy())
        return [info]
    else:
        raise ValueError(f"Sample {sample_idx} not available. Only sample 0 is available in demo data.")
