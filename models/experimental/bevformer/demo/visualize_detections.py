# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Render BEVFormer detections on cameras and a top-down BEV plane."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


CAMERA_LAYOUT = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)
BOX_EDGES = (
    (0, 1),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 7),
    (5, 6),
    (6, 7),
)
BOX_TEMPLATE = np.asarray(
    [
        (-0.5, -0.5, 0.0),
        (-0.5, -0.5, 1.0),
        (-0.5, 0.5, 1.0),
        (-0.5, 0.5, 0.0),
        (0.5, -0.5, 0.0),
        (0.5, -0.5, 1.0),
        (0.5, 0.5, 1.0),
        (0.5, 0.5, 0.0),
    ],
    dtype=np.float64,
)
COLORS = (
    (76, 177, 34),
    (32, 165, 218),
    (187, 113, 58),
    (0, 215, 255),
    (180, 105, 255),
    (255, 144, 30),
    (80, 80, 255),
    (208, 224, 64),
    (255, 102, 153),
    (128, 255, 255),
)


def _parse_args():
    workspace = Path(__file__).resolve().parents[4]
    dataset_workspace = workspace.parent / "BEVFormer"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Folder containing detections.json, or the JSON path itself",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=dataset_workspace / "data/manifest/val.json",
        help="Sequence manifest used for inference",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=dataset_workspace / "data/data/nuscenes",
        help="Root prepended to relative manifest image paths",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=800,
        help="Width of each camera tile in the output mosaic",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Additional visualization score threshold",
    )
    parser.add_argument(
        "--bev-size",
        type=int,
        default=1000,
        help="Width and height of each top-down BEV image",
    )
    parser.add_argument(
        "--bev-range",
        type=float,
        default=51.2,
        help="BEV half-range in meters around the ego vehicle",
    )
    return parser.parse_args()


def _resolve_detections_path(value):
    if value is None:
        value = input("Folder containing detections.json: ").strip()
    if not value:
        raise ValueError("No detections folder was provided")
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path /= "detections.json"
    if not path.is_file():
        raise FileNotFoundError(f"Detection output does not exist: {path}")
    return path


def _load_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _box_corners(box):
    center = np.asarray(box[:3], dtype=np.float64)
    dimensions = np.asarray(box[3:6], dtype=np.float64)
    yaw = float(box[6])
    corners = BOX_TEMPLATE * dimensions
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)))
    corners[:, :2] = corners[:, :2] @ rotation.T
    return corners + center


def _project_corners(box, lidar2img):
    corners = _box_corners(box)
    homogeneous = np.concatenate((corners, np.ones((8, 1))), axis=1)
    projected = homogeneous @ np.asarray(lidar2img, dtype=np.float64).reshape(4, 4).T
    depth = projected[:, 2]
    pixels = np.zeros((8, 2), dtype=np.float64)
    visible = depth > 1e-5
    pixels[visible] = projected[visible, :2] / depth[visible, None]
    return pixels, visible


def _draw_detection(image, detection, lidar2img, thickness=2):
    pixels, visible = _project_corners(detection["box"], lidar2img)
    color = COLORS[int(detection.get("label", 0)) % len(COLORS)]
    height, width = image.shape[:2]
    drew_edge = False

    for start, end in BOX_EDGES:
        if not visible[start] or not visible[end]:
            continue
        point_a = tuple(np.clip(np.rint(pixels[start]), -1_000_000, 1_000_000).astype(int))
        point_b = tuple(np.clip(np.rint(pixels[end]), -1_000_000, 1_000_000).astype(int))
        intersects, clipped_a, clipped_b = cv2.clipLine((0, 0, width, height), point_a, point_b)
        if intersects:
            cv2.line(image, clipped_a, clipped_b, color, thickness, cv2.LINE_AA)
            drew_edge = True

    if drew_edge:
        in_frame = (
            visible & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        )
        if np.any(in_frame):
            anchor = np.rint(pixels[in_frame].min(axis=0)).astype(int)
            class_name = detection.get("class_name", str(detection.get("label", "?")))
            label = f"{class_name} {float(detection['score']):.2f}"
            cv2.putText(
                image,
                label,
                (max(0, int(anchor[0])), max(18, int(anchor[1]) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )


def _render_camera(image_path, camera_name, lidar2img, detections, camera_width):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read camera image: {image_path}")
    for detection in detections:
        _draw_detection(image, detection, lidar2img)
    scale = camera_width / image.shape[1]
    image = cv2.resize(image, (camera_width, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cv2.rectangle(image, (0, 0), (camera_width, 36), (0, 0, 0), -1)
    cv2.putText(image, camera_name, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def _make_mosaic(camera_images):
    ordered_names = [name for name in CAMERA_LAYOUT if name in camera_images]
    ordered_names.extend(name for name in camera_images if name not in ordered_names)
    columns = 3
    tile = next(iter(camera_images.values()))
    blank = np.zeros_like(tile)
    while len(ordered_names) % columns:
        ordered_names.append(None)
    rows = []
    for offset in range(0, len(ordered_names), columns):
        rows.append(np.hstack([camera_images.get(name, blank) for name in ordered_names[offset : offset + columns]]))
    return np.vstack(rows)


def _bev_pixels(points, bev_size, bev_range, margin):
    points = np.asarray(points, dtype=np.float64)
    scale = (bev_size - 2 * margin) / (2 * bev_range)
    pixels = np.empty_like(points)
    pixels[:, 0] = bev_size / 2 - points[:, 1] * scale
    pixels[:, 1] = bev_size / 2 - points[:, 0] * scale
    return np.rint(pixels).astype(np.int32)


def _render_bev(detections, frame_index, token, bev_size, bev_range):
    canvas = np.full((bev_size, bev_size, 3), 24, dtype=np.uint8)
    margin = max(40, bev_size // 20)
    grid_limit = math.floor(bev_range / 10) * 10
    for coordinate in range(-grid_limit, grid_limit + 1, 10):
        horizontal = _bev_pixels(((-bev_range, coordinate), (bev_range, coordinate)), bev_size, bev_range, margin)
        vertical = _bev_pixels(((coordinate, -bev_range), (coordinate, bev_range)), bev_size, bev_range, margin)
        color = (75, 75, 75) if coordinate == 0 else (45, 45, 45)
        cv2.line(canvas, tuple(horizontal[0]), tuple(horizontal[1]), color, 1, cv2.LINE_AA)
        cv2.line(canvas, tuple(vertical[0]), tuple(vertical[1]), color, 1, cv2.LINE_AA)

    border = _bev_pixels(
        (
            (-bev_range, -bev_range),
            (-bev_range, bev_range),
            (bev_range, bev_range),
            (bev_range, -bev_range),
        ),
        bev_size,
        bev_range,
        margin,
    )
    cv2.polylines(canvas, [border], True, (100, 100, 100), 1, cv2.LINE_AA)

    for detection in detections:
        box = detection["box"]
        corners = _box_corners(box)[[0, 3, 7, 4], :2]
        polygon = _bev_pixels(corners, bev_size, bev_range, margin)
        color = COLORS[int(detection.get("label", 0)) % len(COLORS)]
        cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)

        center = np.asarray(box[:2], dtype=np.float64)
        yaw = float(box[6])
        heading_length = max(float(box[3]), float(box[4])) * 0.75
        heading = center + heading_length * np.asarray((math.cos(yaw), math.sin(yaw)))
        heading_pixels = _bev_pixels((center, heading), bev_size, bev_range, margin)
        cv2.arrowedLine(
            canvas,
            tuple(heading_pixels[0]),
            tuple(heading_pixels[1]),
            color,
            2,
            cv2.LINE_AA,
            tipLength=0.25,
        )

        velocity = np.asarray(box[7:9], dtype=np.float64)
        if np.linalg.norm(velocity) > 0.1:
            velocity_pixels = _bev_pixels((center, center + velocity), bev_size, bev_range, margin)
            cv2.arrowedLine(
                canvas,
                tuple(velocity_pixels[0]),
                tuple(velocity_pixels[1]),
                (0, 255, 255),
                1,
                cv2.LINE_AA,
                tipLength=0.2,
            )

        center_pixel = _bev_pixels((center,), bev_size, bev_range, margin)[0]
        if margin <= center_pixel[0] < bev_size - margin and margin <= center_pixel[1] < bev_size - margin:
            class_name = detection.get("class_name", str(detection.get("label", "?")))
            label = f"{class_name} {float(detection['score']):.2f}"
            cv2.putText(
                canvas,
                label,
                (int(center_pixel[0]) + 4, int(center_pixel[1]) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

    ego_corners = np.asarray(((2.3, 0.95), (-2.3, 0.95), (-2.3, -0.95), (2.3, -0.95)))
    ego_polygon = _bev_pixels(ego_corners, bev_size, bev_range, margin)
    cv2.polylines(canvas, [ego_polygon], True, (255, 255, 255), 2, cv2.LINE_AA)
    ego_heading = _bev_pixels(((0.0, 0.0), (4.0, 0.0)), bev_size, bev_range, margin)
    cv2.arrowedLine(
        canvas,
        tuple(ego_heading[0]),
        tuple(ego_heading[1]),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
        tipLength=0.2,
    )

    title = f"frame {frame_index:03d}  {len(detections)} detections  range +/-{bev_range:g} m"
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "+X FORWARD",
        (bev_size // 2 - 55, margin - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "+Y LEFT",
        (margin, bev_size // 2 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        token,
        (12, bev_size - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (140, 140, 140),
        1,
        cv2.LINE_AA,
    )
    return canvas


def visualize(detections_path, manifest_path, dataset_root, camera_width, score_threshold, bev_size, bev_range):
    results = _load_json(detections_path)
    if results.get("summary", {}).get("dataset_format") != "manifest":
        raise ValueError("Only manifest-backed outputs have camera images and calibration")
    manifest = _load_json(manifest_path)
    frames_by_token = {str(frame["token"]): frame for frame in manifest["frames"]}
    camera_order = manifest["camera_order"]

    for result_frame in results["frames"]:
        token = str(result_frame["token"])
        if token not in frames_by_token:
            raise KeyError(f"Result token is absent from the manifest: {token}")
        manifest_frame = frames_by_token[token]
        detections = [
            detection for detection in result_frame["detections"] if float(detection["score"]) >= score_threshold
        ]
        camera_images = {}
        for camera_name in camera_order:
            camera = manifest_frame["cameras"][camera_name]
            image_path = Path(camera["image"])
            if not image_path.is_absolute():
                image_path = dataset_root / image_path
            camera_images[camera_name] = _render_camera(
                image_path, camera_name, camera["lidar2img"], detections, camera_width
            )
        mosaic = _make_mosaic(camera_images)
        frame_index = int(result_frame["frame_index"])
        output_path = detections_path.parent / f"frame_{frame_index:03d}_{token}.png"
        if not cv2.imwrite(str(output_path), mosaic):
            raise OSError(f"Could not write visualization: {output_path}")
        print(f"Wrote {output_path}")

        bev = _render_bev(detections, frame_index, token, bev_size, bev_range)
        bev_output_path = detections_path.parent / f"frame_{frame_index:03d}_{token}_bev.png"
        if not cv2.imwrite(str(bev_output_path), bev):
            raise OSError(f"Could not write BEV visualization: {bev_output_path}")
        print(f"Wrote {bev_output_path}")


def main():
    args = _parse_args()
    if args.camera_width <= 0:
        raise ValueError("--camera-width must be positive")
    if args.bev_size < 200:
        raise ValueError("--bev-size must be at least 200")
    if args.bev_range <= 0:
        raise ValueError("--bev-range must be positive")
    detections_path = _resolve_detections_path(args.input_path)
    visualize(
        detections_path=detections_path,
        manifest_path=args.manifest.expanduser().resolve(),
        dataset_root=args.dataset_root.expanduser().resolve(),
        camera_width=args.camera_width,
        score_threshold=args.score_threshold,
        bev_size=args.bev_size,
        bev_range=args.bev_range,
    )


if __name__ == "__main__":
    main()
