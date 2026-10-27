# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Convert an MMDetection3D-style camera info pickle to the demo's neutral manifest.

Pickle can execute code while loading. Only convert metadata files from a trusted source.
"""

import argparse
import json
import pickle
from pathlib import Path


def _projection(camera):
    """The lidar-to-image composition used by the upstream custom dataset."""
    import numpy as np

    sensor_to_lidar_rotation = np.asarray(camera["sensor2lidar_rotation"], dtype=np.float64)
    sensor_to_lidar_translation = np.asarray(camera["sensor2lidar_translation"], dtype=np.float64)
    lidar_to_camera_rotation = np.linalg.inv(sensor_to_lidar_rotation)
    lidar_to_camera_translation = sensor_to_lidar_translation @ lidar_to_camera_rotation.T
    lidar_to_camera = np.eye(4, dtype=np.float64)
    lidar_to_camera[:3, :3] = lidar_to_camera_rotation.T
    lidar_to_camera[3, :3] = -lidar_to_camera_translation
    intrinsic = np.asarray(camera["cam_intrinsic"], dtype=np.float64)
    viewpad = np.eye(4, dtype=np.float64)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    return (viewpad @ lidar_to_camera.T).tolist()


def _relative_path(path, strip_prefix):
    path = Path(path)
    if strip_prefix is None:
        return str(path)
    prefix = Path(strip_prefix)
    try:
        return str(path.relative_to(prefix))
    except ValueError:
        return str(path)


def _json_matrix(value):
    return value.tolist() if hasattr(value, "tolist") else value


def convert_info(
    input_path: str | Path,
    output_path: str | Path,
    *,
    camera_order: tuple[str, ...],
    can_bus_dims: int,
    strip_prefix=None,
):
    input_path, output_path = Path(input_path), Path(output_path)
    with input_path.open("rb") as file:
        payload = pickle.load(file)  # noqa: S301 - explicitly documented trusted-input CLI.
    infos = payload.get("infos") if isinstance(payload, dict) else payload
    if not isinstance(infos, list):
        raise ValueError(f"{input_path}: expected a list or an object containing an 'infos' list")

    frames = []
    for index, info in enumerate(infos):
        if not isinstance(info, dict) or not isinstance(info.get("cams"), dict):
            raise ValueError(f"{input_path}: info {index} has no camera dictionary")
        missing_cameras = set(camera_order) - set(info["cams"])
        if missing_cameras:
            raise ValueError(f"{input_path}: info {index} is missing cameras {sorted(missing_cameras)}")
        cameras = {name: info["cams"][name] for name in camera_order}
        can_bus = info.get("can_bus")
        if can_bus is None or len(can_bus) != can_bus_dims:
            raise ValueError(f"{input_path}: info {index} must contain a {can_bus_dims}-value can_bus")
        frames.append(
            {
                "token": str(info.get("token", index)),
                "scene_token": str(info.get("scene_token", info.get("scene", "default"))),
                "timestamp": int(info.get("timestamp", index)),
                "cameras": {
                    name: {
                        "image": _relative_path(camera["data_path"], strip_prefix),
                        "lidar2img": (
                            _json_matrix(camera["lidar2img"]) if "lidar2img" in camera else _projection(camera)
                        ),
                    }
                    for name, camera in cameras.items()
                },
                "can_bus": [float(value) for value in can_bus],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"camera_order": list(camera_order), "frames": frames}, indent=2), encoding="utf-8"
    )
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert camera infos to a BEVFormer demo manifest")
    parser.add_argument("--config", required=True, help="Normalized model-config JSON path or HTTPS URL")
    parser.add_argument("--config-sha256", help="Expected SHA-256 for a remote config")
    parser.add_argument("--config-cache", help="Directory used to cache remote configs")
    parser.add_argument("--refresh-config", action="store_true", help="Refetch a cached remote config")
    parser.add_argument("--input", required=True, help="Trusted MMDetection3D info pickle")
    parser.add_argument("--output", required=True, help="Output manifest JSON")
    parser.add_argument(
        "--strip-prefix",
        help="Remove this leading directory from relative camera paths before writing the manifest",
    )
    args = parser.parse_args(argv)
    from models.experimental.bevformer.config import load_model_config

    config = load_model_config(
        args.config,
        sha256=args.config_sha256,
        cache_dir=args.config_cache,
        refresh=args.refresh_config,
    )
    output = convert_info(
        args.input,
        args.output,
        camera_order=config.camera_order,
        can_bus_dims=config.can_bus_dims,
        strip_prefix=args.strip_prefix,
    )
    print(output)


if __name__ == "__main__":
    main()
