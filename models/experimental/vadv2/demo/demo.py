# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run TTNN VAD over one preparation frame and a configurable output sequence.

Weights and dataset are independent CLI inputs. The synthetic path is the default: random camera
images and a synthetic camera rig and ego motion, which measure the model's latency and FPS and
whose outputs have no semantic meaning. The real-data path consumes a source-neutral manifest
(demo/dataset.py) and a checkpoint.
"""

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
from loguru import logger

from models.experimental.vadv2.demo.dataset import SequenceManifest

CAMERA_ORDER = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
CLASS_NAMES = (
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
)
MAP_CLASS_NAMES = ("divider", "ped_crossing", "boundary")
COMMAND_NAMES = ("turn_right", "turn_left", "go_straight")
CAN_BUS_DIMS = 18
# VAD-Tiny's input: each camera image scaled by 0.4 and padded to a multiple of 32.
IMAGE_SCALE = 0.4
IMAGE_HEIGHT, IMAGE_WIDTH = 384, 640
# The released VAD-Tiny weights were trained on BGR images with only the mean subtracted, unlike the
# RGB mean/std normalization in the published config.
IMAGE_MEAN_BGR = (103.53, 116.28, 123.675)
# The synthetic ego: constant speed and turn rate, frames 0.5 s apart.
SYNTHETIC_SPEED_MPS = 5.0
SYNTHETIC_YAW_RATE_DEG = 2.0
FRAME_INTERVAL_S = 0.5


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run TTNN VAD over a camera-frame sequence")
    parser.add_argument("--checkpoint", default="random", help="Checkpoint path, or 'random' for dummy weights")
    parser.add_argument("--dataset-format", choices=("synthetic", "manifest"), default="synthetic")
    parser.add_argument("--dataset", help="Sequence manifest JSON (required for --dataset-format manifest)")
    parser.add_argument("--dataset-root", help="Root prepended to relative image paths in the manifest")
    parser.add_argument("--scene-token", help="Scene token from the manifest")
    parser.add_argument("--scene-index", type=int, default=0, help="Scene index when --scene-token is omitted")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=10, help="Output frames, excluding one preparation frame")
    parser.add_argument("--seed", type=int, default=0, help="Synthetic input and weight seed")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output-dir", default="vad_demo_output")
    return parser.parse_args(argv)


def _yaw_quaternion(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def synthetic_rig():
    """Six pinhole cameras around the ego at VAD's input size, as (num_cams, 4, 4) lidar2img. The
    lidar frame has x to the right, y forward and z up; each camera looks along its yaw."""
    # The image content is 640x360; the padding below it holds no pixels.
    fx = fy = 500.0
    cx, cy = 320.0, 180.0
    intrinsic = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    matrices = []
    for yaw_deg in (0.0, -55.0, 55.0, 180.0, 110.0, -110.0):
        yaw = math.radians(yaw_deg)
        forward = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        right = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        down = np.array([0.0, 0.0, -1.0])
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = np.stack([right, down, forward])
        matrices.append(intrinsic @ extrinsic)
    return matrices


def synthetic_frames(count, seed):
    """count frames of random images, the synthetic rig and an ego driving a gentle curve."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    rig = synthetic_rig()
    frames = []
    x = y = yaw = 0.0
    for index in range(count):
        can_bus = np.zeros(CAN_BUS_DIMS)
        can_bus[:3] = (x, y, 0.0)
        can_bus[3:7] = _yaw_quaternion(yaw)
        angle = math.degrees(yaw) % 360
        can_bus[-2], can_bus[-1] = math.radians(angle), angle
        images = torch.randn(len(CAMERA_ORDER), 3, IMAGE_HEIGHT, IMAGE_WIDTH, generator=generator)
        frames.append((images, rig, can_bus, None, f"synthetic-{index}"))
        x += SYNTHETIC_SPEED_MPS * FRAME_INTERVAL_S * math.cos(yaw)
        y += SYNTHETIC_SPEED_MPS * FRAME_INTERVAL_S * math.sin(yaw)
        yaw += math.radians(SYNTHETIC_YAW_RATE_DEG)
    return frames


def _load_manifest_images(frame):
    """VAD's test pipeline: normalize, scale by IMAGE_SCALE and pad; scale lidar2img with the images."""
    import cv2
    import torch
    from PIL import Image

    mean = np.array(IMAGE_MEAN_BGR, dtype=np.float32)
    tensors, matrices = [], []
    for path, source_matrix in zip(frame.image_paths, frame.lidar2img, strict=True):
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32)[..., ::-1]
        source_height, source_width = array.shape[:2]
        width, height = int(source_width * IMAGE_SCALE + 0.5), int(source_height * IMAGE_SCALE + 0.5)
        resized = cv2.resize(array - mean, (width, height), interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
        padded[:height, :width] = resized
        tensors.append(torch.from_numpy(padded).permute(2, 0, 1))
        scale = np.eye(4)
        scale[0, 0], scale[1, 1] = width / source_width, height / source_height
        matrices.append(scale @ np.asarray(source_matrix))
    return torch.stack(tensors), matrices


def _select_manifest_frames(args):
    if args.dataset is None:
        raise ValueError("--dataset is required with --dataset-format manifest")
    manifest = SequenceManifest.load(
        args.dataset, args.dataset_root, camera_order=CAMERA_ORDER, can_bus_dims=CAN_BUS_DIMS
    )
    return manifest.sequence(
        num_output_frames=args.num_frames,
        start_frame=args.start_frame,
        scene_token=args.scene_token,
        scene_index=args.scene_index,
    )


def _metas(lidar2img, can_bus, scene_token):
    """One frame's img_metas as TtVAD.prepare_frame reads them; can_bus is absolute."""
    return [
        [
            [
                {
                    "scene_token": scene_token,
                    "can_bus": np.array(can_bus, dtype=np.float64),
                    "lidar2img": [np.asarray(m) for m in lidar2img],
                    "img_shape": [(IMAGE_HEIGHT, IMAGE_WIDTH, 3)] * len(lidar2img),
                }
            ]
        ]
    ]


def build_reference_vad(checkpoint, seed):
    """The PyTorch VAD-Tiny the TT model takes its parameters and shapes from."""
    import torch

    from models.experimental.vadv2.reference import vad

    torch.manual_seed(seed)
    model = vad.VAD(
        use_grid_mask=False,
        img_backbone=True,
        img_neck=True,
        pts_bbox_head=True,
        video_test_mode=True,
        fut_ts=6,
        fut_mode=6,
    )
    if checkpoint != "random":
        state = torch.load(checkpoint, map_location="cpu")
        state = state.get("state_dict", state)
        state = {
            k.replace("lateral_convs.0.", "lateral_convs.").replace("fpn_convs.0.", "fpn_convs."): v
            for k, v in state.items()
        }
        model.load_state_dict(state)
    return model.eval().requires_grad_(False)


def _serialize(results):
    """A frame's decoded results as JSON: positions in the lidar frame (x right, y forward), in metres."""
    boxes = results["boxes_3d"].tensor
    detections = []
    for box, score, label, traj in zip(boxes, results["scores_3d"], results["labels_3d"], results["trajs_3d"]):
        # Each mode's trajectory as offsets per step from the box centre, summed to positions.
        trajectories = (traj.reshape(traj.shape[0], -1, 2).cumsum(dim=1) + box[:2]).tolist()
        detections.append(
            {
                "box": box.tolist(),
                "score": float(score),
                "label": int(label),
                "class_name": CLASS_NAMES[int(label)],
                "trajectories": trajectories,
            }
        )
    map_elements = [
        {"points": pts.tolist(), "score": float(score), "label": int(label), "class_name": MAP_CLASS_NAMES[int(label)]}
        for pts, score, label in zip(results["map_pts_3d"], results["map_scores_3d"], results["map_labels_3d"])
    ]
    ego_plans = {name: plan.cumsum(dim=0).tolist() for name, plan in zip(COMMAND_NAMES, results["ego_fut_preds"])}
    return detections, map_elements, ego_plans


def _latency_summary(device_latencies, end_to_end_latencies):
    mean_device_latency = statistics.mean(device_latencies)
    return {
        "mean_device_latency_ms": mean_device_latency * 1e3,
        "p50_device_latency_ms": statistics.median(device_latencies) * 1e3,
        "mean_end_to_end_latency_ms": statistics.mean(end_to_end_latencies) * 1e3,
        "p50_end_to_end_latency_ms": statistics.median(end_to_end_latencies) * 1e3,
        "device_fps": 1.0 / mean_device_latency,
        "end_to_end_fps": 1.0 / statistics.mean(end_to_end_latencies),
    }


def _resolve_tt_metal_commit():
    repo_root = Path(__file__).resolve().parents[4]
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def run_demo(args):
    if args.num_frames < 1:
        raise ValueError("--num-frames must be positive")

    import ttnn

    from models.common.device_utils import get_device_name
    from models.experimental.vadv2.tt.model_preprocessing import create_vadv2_model_parameters_vad
    from models.experimental.vadv2.num_layers import log_layer_counts, num_layers_from_env, reduce_layers
    from models.experimental.vadv2.tt.tt_vad import TtVAD

    if args.dataset_format == "synthetic":
        if args.dataset is not None:
            raise ValueError("--dataset is only used with --dataset-format manifest")
        synthetic = synthetic_frames(args.num_frames + 1, args.seed)

        def load_frame(index):
            images, lidar2img, can_bus, command, token = synthetic[index]
            return images, _metas(lidar2img, can_bus, "synthetic"), command, token

        frame_count = len(synthetic)
    else:
        selected = _select_manifest_frames(args)

        def load_frame(index):
            frame = selected[index]
            images, lidar2img = _load_manifest_images(frame)
            return images, _metas(lidar2img, frame.can_bus, frame.scene_token), frame.command, frame.token

        frame_count = len(selected)

    parameter_model = build_reference_vad(args.checkpoint, args.seed)
    num_layers = num_layers_from_env()
    if num_layers is not None:
        reduce_layers(parameter_model, num_layers)
    log_layer_counts(parameter_model, num_layers)
    first_images, first_metas, _, first_token = load_frame(0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"VAD-Tiny: {args.num_frames} output frames plus one preparation frame; "
        f"checkpoint={args.checkpoint}, dataset={args.dataset_format}"
    )

    device = ttnn.open_device(device_id=args.device_id, l1_small_size=20 * 1024)
    results, device_latencies, end_to_end_latencies = [], [], []
    try:
        hardware = f"{ttnn.get_arch_name()} {get_device_name(device, num_devices=1)}"
        logger.info(f"Hardware: {hardware}")
        # The PyTorch module is a checkpoint-compatible parameter and shape source only.
        parameters = create_vadv2_model_parameters_vad(parameter_model, [None, [first_images[None]]], device)
        tt_model = TtVAD(
            device,
            parameters,
            img_shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
            num_cams=len(CAMERA_ORDER),
            img_backbone=True,
            img_neck=True,
            pts_bbox_head=True,
            video_test_mode=True,
            fut_ts=6,
            fut_mode=6,
        )

        def run_frame(images, metas):
            tt_model.prepare_frame(metas)
            image = ttnn.from_torch(images[None], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            outs = tt_model.predict(image)
            return outs, tt_model.detect(outs)

        # The preparation frame starts the scene's temporal state; the next one compiles the path
        # that reads a previous BEV. Both stay out of the measurements, and the warmup frame's BEV
        # is not kept, so the measured sequence starts from the preparation frame's.
        outs, detected = run_frame(first_images, first_metas)
        tt_model.compact(outs, detected)
        ttnn.synchronize_device(device)
        logger.info(f"Preparation frame {first_token} completed outside measurements")
        state = dict(tt_model.prev_frame_info)
        warm_images, warm_metas, _, warm_token = load_frame(1)
        outs, detected = run_frame(warm_images, warm_metas)
        tt_model.compact(outs, detected)
        ttnn.synchronize_device(device)
        tt_model.prev_frame_info = state
        logger.info(f"Temporal warmup on frame {warm_token} completed outside measurements")

        for index in range(1, frame_count):
            end_to_end_start = time.perf_counter()
            images, metas, command, token = load_frame(index)
            # The device latency spans the frame's preparation, forward and on-device decoding; the
            # image load and the results' readback count in the end-to-end latency only.
            device_start = time.perf_counter()
            outs, detected = run_frame(images, metas)
            ttnn.synchronize_device(device)
            device_latency = time.perf_counter() - device_start
            decoded = tt_model.compact(outs, detected)
            end_to_end_latency = time.perf_counter() - end_to_end_start
            detections, map_elements, ego_plans = _serialize(decoded)
            command_name = None if command is None else COMMAND_NAMES[int(np.argmax(command))]
            results.append(
                {
                    "token": token,
                    "frame_index": index - 1,
                    "device_latency_ms": device_latency * 1e3,
                    "end_to_end_latency_ms": end_to_end_latency * 1e3,
                    "command": command_name,
                    "ego_plan": None if command_name is None else ego_plans[command_name],
                    "ego_plans": ego_plans,
                    "detections": detections,
                    "map_elements": map_elements,
                }
            )
            device_latencies.append(device_latency)
            end_to_end_latencies.append(end_to_end_latency)
            logger.info(
                f"frame {index - 1:03d} token={token}: {len(detections)} detections, {len(map_elements)} map "
                f"elements, device {device_latency * 1e3:.1f} ms, end-to-end {end_to_end_latency * 1e3:.1f} ms"
            )
    finally:
        ttnn.close_device(device)

    summary = {
        "model": "VAD-Tiny",
        "seed": args.seed,
        "hardware": hardware,
        "tt_metal_commit": _resolve_tt_metal_commit(),
        "checkpoint": args.checkpoint,
        "dataset_format": args.dataset_format,
        "preparation_frame": first_token,
        "num_frames": len(results),
        "execution": "eager",
        **_latency_summary(device_latencies, end_to_end_latencies),
    }
    output_path = output_dir / "detections.json"
    output_path.write_text(json.dumps({"summary": summary, "frames": results}, indent=2), encoding="utf-8")
    logger.info(f"Mean device latency: {summary['mean_device_latency_ms']:.1f} ms ({summary['device_fps']:.2f} FPS)")
    logger.info(
        f"Mean end-to-end latency: {summary['mean_end_to_end_latency_ms']:.1f} ms "
        f"({summary['end_to_end_fps']:.2f} FPS)"
    )
    logger.info(f"Results saved to {output_path}")
    return output_path


def main(argv=None):
    return run_demo(_parse_args(argv))


if __name__ == "__main__":
    main()
