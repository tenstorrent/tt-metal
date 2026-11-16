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
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger

from models.experimental.bevformer.config import activate_config, load_builtin_config, load_model_config
from models.experimental.bevformer.config.convert_upstream import convert_upstream_config, variant_from_source
from models.experimental.bevformer.config.loader import read_source_bytes
from models.experimental.bevformer.config.schema import SUPPORTED_VARIANTS
from models.experimental.bevformer.demo.dataset import SequenceManifest


# Frames the pipelined run's host thread prepares ahead of the one on device: one ready for the next
# upload while the thread already works on the one after it, so the thread never waits on the loop.
PREFETCH_FRAMES = 2


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
    parser.add_argument("--checkpoint", default="random", help="Checkpoint path, or 'random' for dummy weights")
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
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Replay each frame from traces (tt_bevformer_trace) instead of dispatching it op by op",
    )
    parser.add_argument(
        "--trace-region-size", type=int, default=256 << 20, help="Device trace region in bytes, with --trace"
    )
    parser.add_argument(
        "--num-command-queues",
        type=int,
        choices=(1, 2),
        default=1,
        help="With --trace, 2 prepares frames on a host thread ahead of the device and uploads each frame's "
        "images on a second command queue while the previous frame runs; the latencies then measure that "
        "pipeline (see demo/README.md)",
    )
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

    import cv2

    tensors, matrices = [], []
    raw_width, raw_height = RAW_IMAGE_SIZE
    mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD, dtype=torch.float32).view(3, 1, 1)
    for path, source_matrix in zip(frame.image_paths, frame.lidar2img, strict=True):
        with Image.open(path) as image:
            image = image.convert("RGB")
            source_width, source_height = image.size
            array = np.asarray(image, dtype=np.float32).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        if IMAGE_COLOR_ORDER == "BGR":
            tensor = tensor.flip(0)
        tensor = (tensor - mean) / std
        resized = cv2.resize(tensor.permute(1, 2, 0).numpy(), (raw_width, raw_height), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1)
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


def _hardware_metadata(architecture, device_card):
    architecture_label = {
        "blackhole": "BH",
        "wormhole_b0": "WH",
    }.get(architecture, architecture.upper())
    return {
        "architecture": architecture,
        "device_card": device_card,
        "hardware": f"{architecture_label} {device_card}",
    }


def _resolve_tt_metal_commit():
    repo_root = Path(__file__).resolve().parents[4]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
    except Exception:
        commit = ""
    return commit or "unknown"


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

    from models.common.device_utils import get_device_name
    from models.experimental.bevformer.model_config import BEV_H, BEV_W, CAN_BUS_DIMS, NUM_CAMS
    from models.experimental.bevformer.reference.bevformer import build_bevformer, relative_can_bus
    from models.experimental.bevformer.tests.common import (
        build_reference_bevformer,
        frame_metas,
        random_image_batch,
        to_image_layout,
    )
    from models.experimental.bevformer.tt.model_preprocessing import create_bevformer_parameters
    from models.experimental.bevformer.tt.tt_bevformer import TTBEVFormer
    from models.experimental.bevformer.tt.tt_bevformer_trace import TTBEVFormerTrace

    torch.manual_seed(args.seed)
    assert (config.num_cameras, config.can_bus_dims) == (NUM_CAMS, CAN_BUS_DIMS)
    if args.checkpoint == "random":
        parameter_model = build_reference_bevformer((BEV_H, BEV_W), seed=args.seed)
    else:
        from models.experimental.bevformer.demo.checkpoint import load_model_checkpoint

        parameter_model = build_bevformer()
        load_model_checkpoint(parameter_model, args.checkpoint, args.checkpoint_format)
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
        metadata_generator = torch.Generator().manual_seed(args.seed)
        image_generator = torch.Generator().manual_seed(args.seed)
        synthetic_metas = frame_metas(1, args.num_frames + 1, metadata_generator)

        def load_frame(index):
            # Unlike the numerical perf test, the demo presents a fresh camera batch per frame.
            return random_image_batch(generator=image_generator), synthetic_metas[index], f"synthetic-{index}"

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
        f"checkpoint={args.checkpoint}, dataset={args.dataset_format}"
    )
    if args.num_command_queues == 2 and not args.trace:
        raise ValueError("--num-command-queues 2 requires --trace")
    if args.num_command_queues == 2 and not ttnn.CONFIG.enable_fast_runtime_mode:
        # The slow runtime tracks ttnn calls on one process-wide stack, which the prefetch thread's
        # from_torch calls would interleave with the main loop's.
        raise ValueError("--num-command-queues 2 needs ttnn's fast runtime mode")
    device = ttnn.open_device(
        device_id=args.device_id,
        l1_small_size=32 * 1024,
        trace_region_size=args.trace_region_size if args.trace else 0,
        num_command_queues=args.num_command_queues,
    )
    results, device_latencies, end_to_end_latencies = [], [], []
    traced = None
    try:
        hardware_metadata = _hardware_metadata(
            ttnn.get_arch_name(),
            get_device_name(device, num_devices=1),
        )
        logger.info(f"Hardware: {hardware_metadata['hardware']}")

        # The PyTorch module is a checkpoint-compatible parameter and shape source. It is not run
        # as a per-frame golden model by this demo.
        tt_model = TTBEVFormer(create_bevformer_parameters(parameter_model, first_batch, device), device)

        def detect_frame(cls_scores, bbox_preds):
            scores, labels, _, boxes, valid = tt_model.detect(cls_scores, bbox_preds, args.score_threshold)
            return tt_model.get_bboxes(scores, labels, boxes, valid)

        tt_image = to_image_layout(first_images, device)
        prepared = tt_model.prepare_frame(first_metas)
        *_, prev_bev = tt_model(tt_image, prepared)
        ttnn.synchronize_device(device)
        for camera in tt_image:
            ttnn.deallocate(camera)
        logger.info(f"Preparation frame {first_token} completed outside measurements")

        # Compile the path that consumes a previous BEV. Keep the preparation frame's BEV as the
        # timed sequence's initial state, so warming frame 1 does not advance temporal history.
        warm_images, warm_metas, warm_token = load_frame(1)
        warm_image = to_image_layout(warm_images, device)
        prepared = tt_model.prepare_frame(warm_metas, prepared)
        warm_outputs = tt_model(warm_image, prepared, prev_bev=prev_bev)
        warm_detections = detect_frame(*warm_outputs[:2])
        ttnn.synchronize_device(device)
        for camera in warm_image:
            ttnn.deallocate(camera)
        del warm_outputs, warm_detections
        logger.info(f"Temporal warmup on frame {warm_token} completed outside measurements")

        if args.trace:
            # Takes over the preparation frame's BEV and the prepared frame's buffers.
            traced = TTBEVFormerTrace(
                tt_model,
                device,
                to_image_layout(warm_images, device, upload=False),
                prepared,
                prev_bev,
                score_threshold=args.score_threshold,
                num_command_queues=args.num_command_queues,
            )
            logger.info(f"Traces captured in {traced.captures[0] * 1e3:.1f} ms")

        def host_inputs(index):
            # Host work only, so the prefetch thread can run it: the frame's images in the image
            # buffer's layout, its metas and its token.
            images, metas, token = load_frame(index)
            return to_image_layout(images, device, upload=False), metas, token

        def prefetched_inputs(index):
            # host_inputs plus the frame's other inputs as host tensors, all ahead of the device.
            host_image, metas, token = host_inputs(index)
            return host_image, metas, token, tt_model.host_frame(metas)

        def record(index, token, compacted, device_latency, end_to_end_latency):
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

        def run_eager():
            nonlocal prepared, prev_bev
            for index in range(1, frame_count):
                end_to_end_start = time.perf_counter()
                host_image, metas, token = host_inputs(index)
                tt_image = [
                    ttnn.to_device(camera, device, memory_config=camera.memory_config()) for camera in host_image
                ]
                # The write is queued, not done: as in run_traced, the upload stays out of the device latency.
                ttnn.synchronize_device(device)
                # Preparing the frame projects the cameras on device and queues their visible counts' readback.
                device_start = time.perf_counter()
                prepared = tt_model.prepare_frame(metas, prepared)
                cls_scores, bbox_preds, prev_bev = tt_model(tt_image, prepared, prev_bev=prev_bev)
                detections = tt_model.detect(cls_scores, bbox_preds, args.score_threshold)
                ttnn.synchronize_device(device)
                device_latency = time.perf_counter() - device_start
                scores, labels, _, boxes, valid = detections
                compacted = tt_model.get_bboxes(scores, labels, boxes, valid)[0]
                for camera in tt_image:
                    ttnn.deallocate(camera)
                record(index, token, compacted, device_latency, time.perf_counter() - end_to_end_start)

        def run_traced():
            for index in range(1, frame_count):
                end_to_end_start = time.perf_counter()
                host_image, metas, token = host_inputs(index)
                traced.upload(host_image)
                # As in eager, the device latency spans the frame from its other inputs to its last
                # device work; the images' upload and the detections' readback are outside it.
                ttnn.synchronize_device(device)
                device_start = time.perf_counter()
                outputs = traced(metas)
                ttnn.event_synchronize(traced.frame_done)
                device_latency = time.perf_counter() - device_start
                compacted = tt_model.get_bboxes(*outputs)[0]
                end_to_end_latency = time.perf_counter() - end_to_end_start
                # After a frame that grew the plan, which ran live; timed apart in trace_capture_ms.
                traced.capture_if_stale()
                record(index, token, compacted, device_latency, end_to_end_latency)

        def run_pipelined():
            # A host thread prepares frames PREFETCH_FRAMES ahead, and frame N + 1's images upload
            # on the second command queue while frame N runs, so a frame's end-to-end latency is the
            # interval since the previous frame finished. One thread keeps the frames in order. A
            # thread, not a process: the image decode, torch's cast and permute and from_torch's
            # packing release the GIL, so they overlap the main loop without copying each image batch
            # between processes.
            pool = ThreadPoolExecutor(max_workers=1)
            # Times each frame's device work: the main thread is busy with the next frame when the
            # device finishes. event_synchronize releases the GIL while it waits.
            watcher = ThreadPoolExecutor(max_workers=1)

            def completion_time(event):
                ttnn.event_synchronize(event)
                return time.perf_counter()

            try:
                pending = deque()

                def prefetch(index):
                    if index < frame_count:
                        pending.append(pool.submit(prefetched_inputs, index))

                for index in range(1, 1 + PREFETCH_FRAMES):
                    prefetch(index)
                host_image, metas, token, host_frame = pending.popleft().result()
                traced.upload(host_image)
                # Filling the pipeline is not measured, as the warmup is not: the first frame's
                # interval starts once its inputs are ready, as every later frame's does.
                frame_end = time.perf_counter()
                for index in range(1, frame_count):
                    device_start = time.perf_counter()
                    outputs = traced(metas, host_frame)
                    device_done = watcher.submit(completion_time, traced.frame_done)
                    prefetch(index + PREFETCH_FRAMES)
                    current = token
                    if pending:
                        host_image, metas, token, host_frame = pending.popleft().result()
                        traced.upload(host_image)
                    # Reading the detections back waits for this frame only, not for the next upload.
                    compacted = tt_model.get_bboxes(*outputs)[0]
                    device_latency = device_done.result() - device_start
                    end_to_end_latency = time.perf_counter() - frame_end
                    traced.capture_if_stale()
                    record(index, current, compacted, device_latency, end_to_end_latency)
                    # The demo's own bookkeeping stays out of the next interval, as it stays out of
                    # every window when not pipelined.
                    frame_end = time.perf_counter()
            finally:
                # On a failure, drop the frames not yet started rather than prepare them first.
                pool.shutdown(wait=True, cancel_futures=True)
                watcher.shutdown(wait=True, cancel_futures=True)

        if traced is None:
            run_eager()
        elif args.num_command_queues == 2:
            run_pipelined()
        else:
            run_traced()
    finally:
        if traced is not None:
            traced.release()
        ttnn.close_device(device)

    summary = {
        "variant": config.variant,
        "model_config": config.to_dict(),
        "seed": args.seed,
        **hardware_metadata,
        "tt_metal_commit": _resolve_tt_metal_commit(),
        "checkpoint": args.checkpoint,
        "dataset_format": args.dataset_format,
        "score_threshold": parameter_model.bbox_coder.score_threshold,
        "preparation_frame": first_token,
        "num_frames": len(results),
        "execution": ("trace" if args.trace else "eager") + ("_2cq_pipelined" if args.num_command_queues == 2 else ""),
        **_latency_summary(device_latencies, end_to_end_latencies),
    }
    if traced is not None:
        # The first capture, then one per growth of the spatial cross-attention's plan.
        summary["trace_capture_ms"] = [seconds * 1e3 for seconds in traced.captures]
    output = {"summary": summary, "frames": results}
    output_path = output_dir / "detections.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info(f"Mean device latency: {summary['mean_device_latency_ms']:.1f} ms ({summary['device_fps']:.2f} FPS)")
    logger.info(
        f"Mean end-to-end latency: {summary['mean_end_to_end_latency_ms']:.1f} ms "
        f"({summary['end_to_end_fps']:.2f} FPS)"
    )
    if traced is not None:
        logger.info(f"Trace captures (ms): {', '.join(f'{ms:.1f}' for ms in summary['trace_capture_ms'])}")
    logger.info(
        f"P50 latency: device {summary['p50_device_latency_ms']:.1f} ms, "
        f"end-to-end {summary['p50_end_to_end_latency_ms']:.1f} ms"
    )
    logger.info(f"Results saved to {output_path}")
    return output_path


def main(argv=None):
    args = _parse_args(argv)
    if args.dataset_format == "synthetic" and args.dataset is not None:
        raise ValueError("--dataset is only used with --dataset-format manifest")
    return run_demo(args)


if __name__ == "__main__":
    main()
