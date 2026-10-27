# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run TTNN BEVFormer over one preparation frame and a configurable output sequence.

Model configuration, weights and dataset are independent CLI inputs. The real-data path consumes a
source-neutral manifest; the synthetic path is an explicit smoke/demo mode whose detections have no
semantic meaning.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger

from models.experimental.bevformer.config import activate_config, load_builtin_config, load_model_config
from models.experimental.bevformer.config.convert_upstream import convert_upstream_config, variant_from_source
from models.experimental.bevformer.config.loader import read_source_bytes
from models.experimental.bevformer.config.schema import SUPPORTED_VARIANTS
from models.experimental.bevformer.demo.dataset import SequenceManifest


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run TTNN BEVFormer over a camera-frame sequence")
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--config", help="Normalized model-config JSON path or HTTPS URL")
    model.add_argument("--variant", choices=SUPPORTED_VARIANTS, default="base", help="Built-in architecture variant")
    parser.add_argument(
        "--config-format",
        choices=("auto", "json", "upstream"),
        default="auto",
        help="Config syntax; auto recognizes official-style .py URLs and paths",
    )
    parser.add_argument("--config-sha256", help="Expected SHA-256 for a remote config")
    parser.add_argument("--config-cache", help="Directory used to cache remote configs")
    parser.add_argument("--refresh-config", action="store_true", help="Refetch a cached remote config")
    parser.add_argument("--weights", default="random", help="Checkpoint path, or 'random' for dummy weights")
    parser.add_argument("--checkpoint-format", choices=("mmdet3d", "local"), default="mmdet3d")
    parser.add_argument("--dataset-format", choices=("synthetic", "manifest"), default="synthetic")
    parser.add_argument("--dataset", help="Sequence manifest JSON (required for --dataset-format manifest)")
    parser.add_argument("--dataset-root", help="Root prepended to relative image paths in the manifest")
    parser.add_argument("--scene-token", help="Scene token from the manifest")
    parser.add_argument("--scene-index", type=int, default=0, help="Scene index when --scene-token is omitted")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=10, help="Output frames, excluding one preparation frame")
    parser.add_argument("--score-threshold", type=float, help="Optional device-side detection score threshold")
    parser.add_argument("--seed", type=int, default=0, help="Synthetic input and weight seed")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output-dir", default="bevformer_demo_output")
    return parser.parse_args(argv)


def _model_config(args):
    if args.config:
        config_format = args.config_format
        if config_format == "auto":
            config_format = "upstream" if urlparse(args.config).path.endswith(".py") else "json"
        if config_format == "upstream":
            content, origin = read_source_bytes(
                args.config,
                sha256=args.config_sha256,
                cache_dir=args.config_cache,
                refresh=args.refresh_config,
            )
            variant = variant_from_source(origin)
            if variant is None:
                raise ValueError("could not infer tiny/small/base from the config name")
            try:
                source = content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"{origin}: upstream config must be UTF-8 Python source") from error
            return convert_upstream_config(source, variant)
        return load_model_config(
            args.config,
            sha256=args.config_sha256,
            cache_dir=args.config_cache,
            refresh=args.refresh_config,
        )
    if args.config_format != "auto":
        raise ValueError("--config-format requires --config")
    if args.config_sha256 or args.config_cache or args.refresh_config:
        raise ValueError("remote config options require --config")
    return load_builtin_config(args.variant)


def _select_manifest_frames(args, config):
    if args.dataset is None:
        raise ValueError("--dataset is required with --dataset-format manifest")
    manifest = SequenceManifest.load(
        args.dataset,
        args.dataset_root,
        camera_order=config.camera_order,
        can_bus_dims=config.can_bus_dims,
    )
    return manifest.sequence(
        num_output_frames=args.num_frames,
        start_frame=args.start_frame,
        scene_token=args.scene_token,
        scene_index=args.scene_index,
    )


def _load_manifest_image(frame):
    """Load, resize, normalize and pad camera images; scale their projection matrices with them."""
    import numpy as np
    import torch
    from PIL import Image

    from models.experimental.bevformer.model_config import (
        IMAGE_HEIGHT,
        IMAGE_MEAN,
        IMAGE_STD,
        IMAGE_WIDTH,
        IMAGE_COLOR_ORDER,
        RAW_IMAGE_SIZE,
    )

    tensors, matrices = [], []
    raw_width, raw_height = RAW_IMAGE_SIZE
    mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD, dtype=torch.float32).view(3, 1, 1)
    for path, source_matrix in zip(frame.image_paths, frame.lidar2img, strict=True):
        with Image.open(path) as image:
            image = image.convert("RGB")
            source_width, source_height = image.size
            image = image.resize((raw_width, raw_height), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        if IMAGE_COLOR_ORDER == "BGR":
            tensor = tensor.flip(0)
        tensor = (tensor - mean) / std
        padded = torch.zeros(3, IMAGE_HEIGHT, IMAGE_WIDTH, dtype=torch.float32)
        padded[:, :raw_height, :raw_width] = tensor
        tensors.append(padded)

        matrix = torch.tensor(source_matrix, dtype=torch.float32)
        scale = torch.eye(4)
        scale[0, 0] = raw_width / source_width
        scale[1, 1] = raw_height / source_height
        matrices.append((scale @ matrix).tolist())

    meta = {
        "img_shape": [(IMAGE_HEIGHT, IMAGE_WIDTH, 3)] * len(tensors),
        "lidar2img": matrices,
        # Made relative by the caller, after scene ordering is known.
        "can_bus": frame.can_bus,
    }
    return torch.stack(tensors), meta


def _serialize_detection(boxes, scores, labels, class_names):
    boxes, scores, labels = boxes.tolist(), scores.tolist(), labels.tolist()
    result = []
    for box, score, label in zip(boxes, scores, labels, strict=True):
        item = {"box": box, "score": float(score), "label": int(label)}
        if 0 <= label < len(class_names):
            item["class_name"] = class_names[label]
        result.append(item)
    return result


def run_demo(args):
    if args.num_frames < 1:
        raise ValueError("--num-frames must be positive")
    config = _model_config(args)
    activate_config(config)

    # model_config derives all shapes when first imported. Refuse an already-imported conflicting
    # variant rather than silently constructing a mixed architecture.
    module_name = "models.experimental.bevformer.model_config"
    if module_name in sys.modules and sys.modules[module_name].MODEL_CONFIG != config:
        raise RuntimeError("BEVFormer was imported before CLI parsing with a different model config")

    import torch
    import ttnn

    from models.experimental.bevformer.model_config import BEV_H, BEV_W, CAN_BUS_DIMS, NUM_CAMS
    from models.experimental.bevformer.reference.bevformer import build_bevformer, relative_can_bus
    from models.experimental.bevformer.tests.common import (
        build_reference_bevformer,
        frame_metas,
        random_image_batch,
        to_conv_layout,
    )
    from models.experimental.bevformer.tt.model_preprocessing import create_bevformer_parameters
    from models.experimental.bevformer.tt.tt_bevformer import TTBEVFormer

    torch.manual_seed(args.seed)
    assert (config.num_cameras, config.can_bus_dims) == (NUM_CAMS, CAN_BUS_DIMS)
    if args.weights == "random":
        parameter_model = build_reference_bevformer((BEV_H, BEV_W))
    else:
        from models.experimental.bevformer.demo.checkpoint import load_model_checkpoint

        parameter_model = build_bevformer()
        load_model_checkpoint(parameter_model, args.weights, args.checkpoint_format)
        parameter_model.eval().requires_grad_(False)

    if config.class_names and len(config.class_names) != parameter_model.bbox_coder.num_classes:
        raise ValueError(
            f"config has {len(config.class_names)} class names, but the selected model has "
            f"{parameter_model.bbox_coder.num_classes} output classes"
        )
    if args.score_threshold is not None:
        if not 0.0 <= args.score_threshold <= 1.0:
            raise ValueError("--score-threshold must be between 0 and 1")
        parameter_model.bbox_coder.score_threshold = args.score_threshold

    if args.dataset_format == "synthetic":
        generator = torch.Generator().manual_seed(args.seed)
        metas = frame_metas(1, args.num_frames + 1, generator)

        def load_frame(index):
            # Unlike the numerical perf test, the demo presents a fresh camera batch per frame.
            return random_image_batch(), metas[index], f"synthetic-{index}"

        frame_count = args.num_frames + 1
    else:
        selected = _select_manifest_frames(args, config)
        previous_can_bus = None
        relative_can_buses = []
        for frame in selected:
            relative_can_buses.append(relative_can_bus(frame.can_bus, previous_can_bus).numpy())
            previous_can_bus = frame.can_bus

        def load_frame(index):
            images, meta = _load_manifest_image(selected[index])
            meta["can_bus"] = relative_can_buses[index]
            return images, [meta], selected[index].token

        frame_count = len(selected)

    first_images, first_metas, first_token = load_frame(0)
    first_batch = first_images[None]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"BEVFormer {config.variant}: {args.num_frames} output frames plus one preparation frame; "
        f"weights={args.weights}, dataset={args.dataset_format}"
    )
    device = ttnn.open_device(device_id=args.device_id, l1_small_size=32 * 1024)
    results, device_latencies, end_to_end_latencies = [], [], []
    try:
        # The PyTorch module is a checkpoint-compatible parameter and shape source. It is not run
        # as a per-frame golden model by this demo.
        tt_model = TTBEVFormer(create_bevformer_parameters(parameter_model, first_batch, device), device)

        tt_image = to_conv_layout(first_images, device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        prepared = tt_model.prepare_frame(first_metas)
        *_, prev_bev = tt_model(tt_image, prepared)
        ttnn.synchronize_device(device)
        ttnn.deallocate(tt_image)
        logger.info(f"Preparation frame {first_token} completed outside measurements")

        # Compile the path that consumes a previous BEV. Keep the preparation frame's BEV as the
        # timed sequence's initial state, so warming frame 1 does not advance temporal history.
        warm_images, warm_metas, warm_token = load_frame(1)
        warm_image = to_conv_layout(warm_images, device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        prepared = tt_model.prepare_frame(warm_metas, prepared)
        warm_outputs = tt_model(warm_image, prepared, prev_bev=prev_bev)
        warm_detections = tt_model.detect(*warm_outputs[:2])
        ttnn.synchronize_device(device)
        ttnn.deallocate(warm_image)
        del warm_outputs, warm_detections
        logger.info(f"Temporal warmup on frame {warm_token} completed outside measurements")

        for index in range(1, frame_count):
            end_to_end_start = time.perf_counter()
            images, metas, token = load_frame(index)
            tt_image = to_conv_layout(images, device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            prepared = tt_model.prepare_frame(metas, prepared)

            device_start = time.perf_counter()
            cls_scores, bbox_preds, next_bev = tt_model(tt_image, prepared, prev_bev=prev_bev)
            scores, labels, _, boxes, valid = tt_model.detect(cls_scores, bbox_preds)
            ttnn.synchronize_device(device)
            ttnn.deallocate(tt_image)
            device_latency = time.perf_counter() - device_start
            compacted = tt_model.get_bboxes(scores, labels, boxes, valid)[0]
            end_to_end_latency = time.perf_counter() - end_to_end_start
            prev_bev = next_bev

            sample_boxes, sample_scores, sample_labels = compacted
            detections = _serialize_detection(sample_boxes, sample_scores, sample_labels, config.class_names)
            results.append(
                {
                    "token": token,
                    "frame_index": index - 1,
                    "device_latency_ms": device_latency * 1e3,
                    "end_to_end_latency_ms": end_to_end_latency * 1e3,
                    "detections": detections,
                }
            )
            device_latencies.append(device_latency)
            end_to_end_latencies.append(end_to_end_latency)
            logger.info(
                f"frame {index - 1:03d} token={token}: {len(detections)} detections, "
                f"device {device_latency * 1e3:.1f} ms, end-to-end {end_to_end_latency * 1e3:.1f} ms"
            )
    finally:
        ttnn.close_device(device)

    summary = {
        "variant": config.variant,
        "weights": args.weights,
        "dataset_format": args.dataset_format,
        "score_threshold": parameter_model.bbox_coder.score_threshold,
        "preparation_frame": first_token,
        "num_frames": len(results),
        "mean_device_latency_ms": statistics.mean(device_latencies) * 1e3,
        "mean_end_to_end_latency_ms": statistics.mean(end_to_end_latencies) * 1e3,
        "device_fps": 1.0 / statistics.mean(device_latencies),
    }
    output = {"summary": summary, "frames": results}
    output_path = output_dir / "detections.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info(f"Mean device latency: {summary['mean_device_latency_ms']:.1f} ms ({summary['device_fps']:.2f} FPS)")
    logger.info(f"Results saved to {output_path}")
    return output_path


def main(argv=None):
    args = _parse_args(argv)
    if args.dataset_format == "synthetic" and args.dataset is not None:
        raise ValueError("--dataset is only used with --dataset-format manifest")
    return run_demo(args)


if __name__ == "__main__":
    main()
