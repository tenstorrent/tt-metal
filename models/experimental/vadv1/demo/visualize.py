# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Render the VAD demo's detections.json, one PNG per frame, without a Tenstorrent device.

Each figure shows the frame's camera images, when a manifest provides them, above a bird's-eye view
in the lidar frame (x right, y forward): the map elements, the detected agents with their predicted
trajectories, and the ego plan for the frame's driving command.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from models.experimental.vadv1.demo.dataset import SequenceManifest  # noqa: E402

CAMERA_ORDER = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")
MANIFEST_ORDER = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
MAP_COLORS = {"divider": "tab:orange", "ped_crossing": "tab:blue", "boundary": "tab:red"}
BEV_RANGE_X, BEV_RANGE_Y = 15.0, 30.0


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Render VAD demo results")
    parser.add_argument("--results", required=True, help="The demo's detections.json")
    parser.add_argument("--manifest", help="The manifest the demo ran on, for the camera images")
    parser.add_argument("--dataset-root", help="Root prepended to relative image paths in the manifest")
    parser.add_argument("--score-threshold", type=float, default=0.3, help="Draw detections at or above this score")
    parser.add_argument("--map-score-threshold", type=float, default=0.3)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def box_footprint(box):
    """A (cx, cy, z, dx, dy, dz, yaw, ...) box's bottom rectangle, closed, as mmdet3d 0.17's
    LiDARInstance3DBoxes rotates it: x' = x cos(yaw) + y sin(yaw), y' = -x sin(yaw) + y cos(yaw)."""
    cx, cy, _, dx, dy, _, yaw = box[:7]
    local = np.array([[-dx, -dy], [-dx, dy], [dx, dy], [dx, -dy], [-dx, -dy]]) / 2
    cos, sin = np.cos(yaw), np.sin(yaw)
    x = local[:, 0] * cos + local[:, 1] * sin
    y = -local[:, 0] * sin + local[:, 1] * cos
    return np.stack([x + cx, y + cy], axis=1)


def _draw_bev(ax, frame, score_threshold, map_score_threshold):
    for element in frame["map_elements"]:
        if element["score"] < map_score_threshold:
            continue
        pts = np.asarray(element["points"])
        ax.plot(pts[:, 0], pts[:, 1], color=MAP_COLORS[element["class_name"]], linewidth=1.5, alpha=0.8)
    for detection in frame["detections"]:
        if detection["score"] < score_threshold:
            continue
        corners = box_footprint(detection["box"])
        ax.plot(corners[:, 0], corners[:, 1], color="tab:green", linewidth=1.2)
        for trajectory in detection["trajectories"]:
            traj = np.asarray([detection["box"][:2]] + trajectory)
            ax.plot(traj[:, 0], traj[:, 1], color="tab:purple", linewidth=0.6, alpha=0.5)
        ax.text(
            detection["box"][0], detection["box"][1], f"{detection['class_name']} {detection['score']:.2f}", fontsize=5
        )
    plan = frame["ego_plan"] or frame["ego_plans"]["go_straight"]
    plan = np.asarray([[0.0, 0.0]] + plan)
    ax.plot(plan[:, 0], plan[:, 1], color="black", linewidth=2.5, marker="o", markersize=3, label="ego plan")
    ax.add_patch(plt.Rectangle((-0.95, -2.4), 1.9, 4.8, fill=True, color="black", alpha=0.6))
    ax.set_xlim(-BEV_RANGE_X, BEV_RANGE_X)
    ax.set_ylim(-BEV_RANGE_Y, BEV_RANGE_Y)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    command = frame["command"] or "go_straight (no command given)"
    ax.set_title(f"BEV, command {command}", fontsize=9)


def render(results_path, manifest, output_dir, score_threshold, map_score_threshold):
    results = json.loads(Path(results_path).read_text())
    frames_by_token = {} if manifest is None else {frame.token: frame for frame in manifest.frames}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for frame in results["frames"]:
        source = frames_by_token.get(frame["token"])
        fig = plt.figure(figsize=(16, 9))
        if source is not None:
            images = dict(zip(MANIFEST_ORDER, source.image_paths))
            for index, name in enumerate(CAMERA_ORDER):
                ax = fig.add_axes([0.01 + (index % 3) * 0.2, 0.52 - (index // 3) * 0.24, 0.195, 0.22])
                ax.imshow(plt.imread(images[name]))
                ax.set_title(name, fontsize=7)
                ax.axis("off")
            bev_ax = fig.add_axes([0.62, 0.05, 0.36, 0.9])
        else:
            bev_ax = fig.add_axes([0.3, 0.05, 0.4, 0.9])
        _draw_bev(bev_ax, frame, score_threshold, map_score_threshold)
        fig.suptitle(
            f"frame {frame['frame_index']} {frame['token']}: device {frame['device_latency_ms']:.1f} ms",
            fontsize=10,
        )
        path = output_dir / f"frame_{frame['frame_index']:03d}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)
    return written


def main(argv=None):
    args = _parse_args(argv)
    manifest = None
    if args.manifest is not None:
        manifest = SequenceManifest.load(args.manifest, args.dataset_root, camera_order=MANIFEST_ORDER, can_bus_dims=18)
    for path in render(args.results, manifest, args.output_dir, args.score_threshold, args.map_score_threshold):
        print(path)


if __name__ == "__main__":
    main()
