#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ethernet dispatch must be selected before ttnn/metal runtime reads env (common_demo_utils imports ttnn).
if __name__ == "__main__" and "--tt-eth-dispatch" in sys.argv:
    os.environ.setdefault("TT_METAL_GTEST_ETH_DISPATCH", "1")

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, UnidentifiedImageError
from sahi import AutoDetectionModel
from sahi.postprocess.combine import GreedyNMMPostprocess, LSNMSPostprocess, NMMPostprocess, NMSPostprocess
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.prediction import ObjectPrediction, PredictionResult
from sahi.slicing import slice_image
from sahi.utils.cv import read_image_as_pil, visualize_object_predictions

from models.demos.utils.common_demo_utils import get_mesh_mappers, load_coco_class_names, postprocess, preprocess

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _r4(x: float) -> float:
    return round(float(x), 4)


def _timing_pre_device_post(t: dict[str, Any]) -> tuple[float, float, float]:
    """Unified triple: (pre-processing, device inference, post-processing) in seconds."""
    pre = float(t.get("host_slice_and_preprocess_sec", 0) or 0)
    dev = float(t.get("device_inference_sec", 0) or 0)
    post = float(t.get("host_postprocess_and_sahi_merge_sec", 0) or 0)
    return pre, dev, post


def _format_pre_device_post_line(label: str, t: dict[str, Any]) -> str:
    pre, dev, post = _timing_pre_device_post(t)
    return f"  {label:16} pre={pre:.4f}s  device={dev:.4f}s  post={post:.4f}s"


def _format_timing_detail_line(label: str, t: dict[str, Any]) -> str:
    """Second line: granular fields when non-zero (TT + SAHI paths)."""
    pairs = []
    order = [
        ("host_read_image_sec", "read"),
        ("host_sahi_slice_sec", "sahi_slice"),
        ("host_cpu_prep_letterbox_sec", "cpu_prep"),
        ("device_ttnn_run_sec", "tt_run"),
        ("host_ttnn_to_torch_sec", "to_torch"),
        ("host_tt_torch_postprocess_sec", "tt_post"),
        ("sahi_ultralytics_per_slice_host_sec", "sahi_per_slice"),
        ("sahi_shift_to_full_sec", "sahi_shift"),
        ("sahi_merge_sec", "sahi_merge"),
        ("host_slice_image_export_sec", "slice_export"),
    ]
    for key, short in order:
        v = float(t.get(key, 0) or 0)
        if v > 1e-7:
            if key == "host_cpu_prep_letterbox_sec":
                extra = t.get("extra") if isinstance(t.get("extra"), dict) else {}
                n_sl = int(extra.get("tt_num_slices", 0) or 0)
                mean_v = float(t.get("host_cpu_prep_mean_per_slice_sec", 0) or 0)
                if n_sl > 1 and mean_v > 1e-7:
                    pairs.append(f"{short}={v:.4f}s({n_sl}t~{mean_v:.4f}s/t)")
                else:
                    pairs.append(f"{short}={v:.4f}s")
            else:
                pairs.append(f"{short}={v:.4f}s")
    # SAHI get_sliced_prediction API: "prediction" timer = model + per-slice host work (excludes merge).
    if float(t.get("device_ttnn_run_sec", 0) or 0) < 1e-7 and float(t.get("host_sahi_slice_sec", 0) or 0) > 1e-7:
        v = float(t.get("device_inference_sec", 0) or 0)
        if v > 1e-7:
            pairs.append(f"sahi_pred={v:.4f}s")
    # Ultralytics full image: no TTNN / no tiling; post bucket is SAHI+Ultralytics host decode only.
    if (
        float(t.get("device_ttnn_run_sec", 0) or 0) < 1e-7
        and float(t.get("host_ttnn_to_torch_sec", 0) or 0) < 1e-7
        and float(t.get("host_sahi_slice_sec", 0) or 0) < 1e-7
    ):
        v = float(t.get("host_postprocess_and_sahi_merge_sec", 0) or 0)
        if v > 1e-7:
            pairs.append(f"ultra_decode={v:.4f}s")
    if not pairs:
        return ""
    return f"  {label:16} " + "  ".join(pairs)


@dataclass
class SlicedPipelineTiming:
    """Wall-clock style breakdown for one pipeline run (seconds)."""

    total_wall_sec: float = 0.0
    host_sahi_slice_sec: float = 0.0
    host_preprocess_before_device_sec: float = 0.0
    device_inference_sec: float = 0.0
    host_postprocess_and_sahi_merge_sec: float = 0.0
    host_slice_image_export_sec: float = 0.0
    # Granular (TT): read / letterbox / on-device / D2H to torch / NMS+scale_boxes on host
    host_read_image_sec: float = 0.0
    host_cpu_prep_letterbox_sec: float = 0.0
    # TT sliced: host_cpu_prep_letterbox_sec is summed over tiles; this is mean per tile for fair vs single full-image run.
    host_cpu_prep_mean_per_slice_sec: float = 0.0
    device_ttnn_run_sec: float = 0.0
    host_ttnn_to_torch_sec: float = 0.0
    host_tt_torch_postprocess_sec: float = 0.0
    # SAHI merge path (shift to full-image coords + NMM/NMS merge)
    sahi_shift_to_full_sec: float = 0.0
    sahi_merge_sec: float = 0.0
    # Ultralytics+SAHI: sum of get_prediction "postprocess" timers (convert preds on host), when known
    sahi_ultralytics_per_slice_host_sec: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def host_slice_and_preprocess_sec(self) -> float:
        # Full-image path: load is inside host_preprocess_before_device_sec only.
        # Sliced TT path: host_read_image_sec + slice-only + per-slice letterbox are separate timers.
        if self.host_sahi_slice_sec > 0:
            return _r4(self.host_read_image_sec + self.host_sahi_slice_sec + self.host_preprocess_before_device_sec)
        return _r4(self.host_sahi_slice_sec + self.host_preprocess_before_device_sec)

    def to_summary_dict(self) -> dict[str, Any]:
        out = {
            "total_wall_sec": _r4(self.total_wall_sec),
            "host_sahi_slice_sec": _r4(self.host_sahi_slice_sec),
            "host_preprocess_before_device_sec": _r4(self.host_preprocess_before_device_sec),
            "host_slice_and_preprocess_sec": self.host_slice_and_preprocess_sec(),
            "device_inference_sec": _r4(self.device_inference_sec),
            "host_postprocess_and_sahi_merge_sec": _r4(self.host_postprocess_and_sahi_merge_sec),
            "host_slice_image_export_sec": _r4(self.host_slice_image_export_sec),
            "host_read_image_sec": _r4(self.host_read_image_sec),
            "host_cpu_prep_letterbox_sec": _r4(self.host_cpu_prep_letterbox_sec),
            "host_cpu_prep_mean_per_slice_sec": _r4(self.host_cpu_prep_mean_per_slice_sec),
            "device_ttnn_run_sec": _r4(self.device_ttnn_run_sec),
            "host_ttnn_to_torch_sec": _r4(self.host_ttnn_to_torch_sec),
            "host_tt_torch_postprocess_sec": _r4(self.host_tt_torch_postprocess_sec),
            "sahi_shift_to_full_sec": _r4(self.sahi_shift_to_full_sec),
            "sahi_merge_sec": _r4(self.sahi_merge_sec),
            "sahi_ultralytics_per_slice_host_sec": _r4(self.sahi_ultralytics_per_slice_host_sec),
        }
        if self.extra:
            out["extra"] = {k: _r4(v) if isinstance(v, (int, float)) else v for k, v in self.extra.items()}
        return out


def _timing_from_sahi_durations(d: dict[str, Any]) -> SlicedPipelineTiming:
    """Map SAHI get_sliced_prediction durations: slice=slice_image; prediction=loop incl. model+per-slice host; postprocess=merge only."""
    slice_t = float(d.get("slice", 0) or 0)
    pred_t = float(d.get("prediction", 0) or 0)
    post_t = float(d.get("postprocess", 0) or 0)
    return SlicedPipelineTiming(
        host_sahi_slice_sec=slice_t,
        host_preprocess_before_device_sec=0.0,
        device_inference_sec=pred_t,
        host_postprocess_and_sahi_merge_sec=post_t,
        sahi_merge_sec=post_t,
    )


def _timing_full_ultralytics(d: dict[str, Any]) -> SlicedPipelineTiming:
    """Full-image get_prediction: prediction=model forward; postprocess=Ultralytics+SAHI host decode."""
    pred_t = float(d.get("prediction", 0) or 0)
    post_t = float(d.get("postprocess", 0) or 0)
    return SlicedPipelineTiming(
        host_sahi_slice_sec=0.0,
        host_preprocess_before_device_sec=0.0,
        device_inference_sec=pred_t,
        host_postprocess_and_sahi_merge_sec=post_t,
    )


def load_image_bgr_sahi(image_path) -> np.ndarray:
    """BGR uint8 image matching SAHI / PIL (EXIF orientation applied). OpenCV imread does not rotate by EXIF."""
    pil = read_image_as_pil(str(image_path), exif_fix=True)
    rgb = np.asarray(pil.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def parallel_slice_chunk_bounds(num_slices: int, parallel: int):
    """Yield (start_index, num_valid_in_chunk) for batching SAHI slices across `parallel` devices."""
    i = 0
    while i < num_slices:
        n_valid = min(parallel, num_slices - i)
        yield i, n_valid
        i += parallel


def _tt_device_open_kwargs(args):
    return {
        "l1_small_size": args.tt_l1_small_size,
        "trace_region_size": args.tt_trace_region_size,
        "num_command_queues": 2,
    }


def _system_or_configured_mesh_device_count(ttnn, args) -> int:
    """Upper bound on devices we can use (from --tt-mesh-shape or system descriptor)."""
    if args.tt_mesh_shape is not None:
        rows, cols = args.tt_mesh_shape
        return int(rows) * int(cols)
    sys_shape = tuple(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
    if len(sys_shape) != 2:
        raise RuntimeError(f"Unexpected system mesh shape tuple: {sys_shape}")
    return int(sys_shape[0]) * int(sys_shape[1])


def _open_tt_yolo_device(ttnn, args):
    """
    Open a mesh device when multiple chips are available (e.g. T3K 1x8); otherwise open_device.
    Matches models/demos/yolov8s/demo/demo.py (test_demo vs test_demo_dp).

    Uses SystemMeshDescriptor for single-vs-multi detection instead of ttnn.get_num_devices():
    GetNumAvailableDevices() can block for minutes and then throw (e.g. arc core start timeout)
    on a bad driver state after an unclean process exit.
    """
    kwargs = _tt_device_open_kwargs(args)
    if getattr(args, "tt_force_single_device", False):
        return ttnn.open_device(device_id=args.tt_device_id, **kwargs), False

    if args.tt_mesh_shape is not None:
        rows, cols = args.tt_mesh_shape
        mesh_shape = ttnn.MeshShape(int(rows), int(cols))
        return ttnn.open_mesh_device(mesh_shape=mesh_shape, **kwargs), True

    sys_shape = tuple(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
    if len(sys_shape) != 2:
        raise RuntimeError(f"Unexpected system mesh shape tuple: {sys_shape}")
    mesh_rows, mesh_cols = int(sys_shape[0]), int(sys_shape[1])
    if mesh_rows * mesh_cols <= 1:
        return ttnn.open_device(device_id=args.tt_device_id, **kwargs), False

    mesh_shape = ttnn.MeshShape(mesh_rows, mesh_cols)
    return ttnn.open_mesh_device(mesh_shape=mesh_shape, **kwargs), True


def _open_tt_slice_parallel_device(ttnn, args, n_par: int):
    """
    Open exactly n_par chips as a 1×n row mesh (or open_device when n_par==1).

    Do not use create_submesh() off a larger parent mesh: closing the child can fail with
    "MeshDevice cq ID ... is in use by parent mesh ..." because parent and submesh share CQs.
    """
    kwargs = _tt_device_open_kwargs(args)
    if getattr(args, "tt_force_single_device", False):
        if n_par != 1:
            raise ValueError("--tt-force-single-device is incompatible with --tt-slice-parallel-devices > 1")
        return ttnn.open_device(device_id=args.tt_device_id, **kwargs), False

    max_devices = _system_or_configured_mesh_device_count(ttnn, args)
    if n_par > max_devices:
        raise ValueError(
            f"--tt-slice-parallel-devices {n_par} exceeds available mesh devices ({max_devices}); "
            "adjust --tt-mesh-shape or hardware."
        )
    if n_par < 1:
        raise ValueError("--tt-slice-parallel-devices must be >= 1")

    if n_par == 1:
        return ttnn.open_device(device_id=args.tt_device_id, **kwargs), False

    mesh_shape = ttnn.MeshShape(1, n_par)
    return ttnn.open_mesh_device(mesh_shape=mesh_shape, **kwargs), True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Ultralytics full-image inference vs SAHI sliced inference.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to an image file or directory containing images.",
    )
    parser.add_argument(
        "--model",
        default="yolov8s.pt",
        help="Ultralytics model path/name (e.g. yolov8s.pt, yolov8x.pt). Default matches YOLOv8s (640 native imgsz).",
    )
    parser.add_argument(
        "--backend",
        default="ultralytics",
        choices=["ultralytics", "tt"],
        help="Inference backend: ultralytics (host) or tt (Tenstorrent).",
    )
    parser.add_argument(
        "--tt-model",
        default="yolov8s",
        choices=["yolov8s", "yolov8x"],
        help="TT model variant. Both use 640x640 internal letterbox (see models/demos/yolov8s|yolov8x).",
    )
    parser.add_argument(
        "--tt-device-id",
        type=int,
        default=0,
        help="TT device id for ttnn.open_device.",
    )
    parser.add_argument(
        "--tt-l1-small-size",
        type=int,
        default=24576,
        help="TT device l1_small_size (YOLOv8s/YOLOv8x demos default 24576).",
    )
    parser.add_argument(
        "--tt-trace-region-size",
        type=int,
        default=6434816,
        help="TT device trace_region_size (YOLOv8s demo uses 6434816; override if your device requires it).",
    )
    parser.add_argument(
        "--tt-mesh-shape",
        type=int,
        nargs=2,
        metavar=("ROWS", "COLS"),
        default=None,
        help="TT mesh shape for multi-chip systems (e.g. 1 8 for T3K). Default: system mesh from SystemMeshDescriptor. "
        "Use this if device open fails (e.g. arc core timeout) so shape is explicit.",
    )
    parser.add_argument(
        "--tt-force-single-device",
        action="store_true",
        help="Use open_device(device_id) even when multiple chips exist (may fail on mesh-only setups).",
    )
    parser.add_argument(
        "--tt-eth-dispatch",
        action="store_true",
        help="Set TT_METAL_GTEST_ETH_DISPATCH=1 before device init (Ethernet cores for dispatch; useful on multi-chip). "
        "Equivalent: export TT_METAL_GTEST_ETH_DISPATCH=1 before python.",
    )
    parser.add_argument(
        "--tt-slice-parallel-devices",
        type=int,
        default=None,
        metavar="N",
        help="TT mesh width for SAHI sliced mode: use a 1×N submesh so each of N simultaneous 640×640 inputs is sharded "
        "to one wormhole chip (data parallel on batch dim). Requires a parent mesh with ≥N devices (e.g. T3K 1×8 with "
        "N=4). Omit for legacy behavior (full parent mesh, same slice replicated on all devices per forward).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Inference device (e.g. "cpu", "cuda:0").',
    )
    parser.add_argument(
        "--slice-height",
        type=int,
        default=512,
        help="SAHI slice height.",
    )
    parser.add_argument(
        "--slice-width",
        type=int,
        default=512,
        help="SAHI slice width.",
    )
    parser.add_argument(
        "--overlap-height-ratio",
        type=float,
        default=0.2,
        help="Vertical overlap ratio between slices.",
    )
    parser.add_argument(
        "--overlap-width-ratio",
        type=float,
        default=0.2,
        help="Horizontal overlap ratio between slices.",
    )
    parser.add_argument(
        "--perform-standard-pred",
        action="store_true",
        help="Also run standard full-image prediction pass during sliced inference merge.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/demos/yolo_eval/sahi_outputs",
        help="Directory for visuals and JSON summary.",
    )
    parser.add_argument(
        "--save-visuals",
        action="store_true",
        help="Save annotated prediction images for both modes.",
    )
    parser.add_argument(
        "--postprocess-type",
        default="NMS",
        choices=["NMM", "GREEDYNMM", "NMS", "LSNMS"],
        help="SAHI postprocess strategy for merging sliced predictions.",
    )
    parser.add_argument(
        "--postprocess-match-metric",
        default="IOU",
        choices=["IOU", "IOS"],
        help="Overlap metric for box matching in SAHI postprocess.",
    )
    parser.add_argument(
        "--postprocess-match-threshold",
        type=float,
        default=0.5,
        help="Threshold for postprocess match metric.",
    )
    parser.add_argument(
        "--postprocess-class-agnostic",
        action="store_true",
        help="Merge boxes across classes during SAHI postprocess.",
    )
    parser.add_argument(
        "--save-slice-grid-overlay",
        action="store_true",
        help="Save original image with SAHI slice boundaries drawn on top.",
    )
    parser.add_argument(
        "--save-slice-images",
        action="store_true",
        help="After per-slice inference, save each crop with detections drawn to OUTPUT_DIR/<image_stem>_slices/*.png.",
    )
    parser.add_argument(
        "--pre-resize-to",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Resize each input image to WIDTH HEIGHT before slicing/inference.",
    )
    return parser.parse_args()


def collect_images(input_path: Path) -> list[Path]:
    if str(input_path) in {"/path/to/image.jpg", "/path/to/images", "/path/to/images_dir"}:
        raise FileNotFoundError(
            f"Placeholder input path used: {input_path}\n"
            "Replace it with a real file or directory path, for example:\n"
            "  --input docs/source/common/images/MfB-Fig3a.png"
        )
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted([p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])
    raise FileNotFoundError(
        f"Input path does not exist: {input_path}\n" "Pass a valid image file or directory path via --input."
    )


def split_readable_images(images: list[Path]) -> tuple[list[Path], list[Path]]:
    readable = []
    unreadable = []
    for image in images:
        try:
            with Image.open(image) as im:
                im.verify()
            readable.append(image)
        except (UnidentifiedImageError, OSError):
            unreadable.append(image)
    return readable, unreadable


def run_prediction(detection_model, image_path: str):
    start = time.perf_counter()
    result = get_prediction(image=image_path, detection_model=detection_model)
    elapsed = time.perf_counter() - start
    timing = _timing_full_ultralytics(result.durations_in_seconds)
    timing.total_wall_sec = elapsed
    return result, elapsed, timing.to_summary_dict()


class TTYoloBackend:
    """Tenstorrent YOLO runner. Ultralytics YOLOv8s/YOLOv8x are trained for 640 imgsz; TT demos match that."""

    _TT_INPUT_RES = (640, 640)

    def __init__(self, args):
        import ttnn
        from models.demos.yolov8s.runner.performant_runner import YOLOv8sPerformantRunner
        from models.demos.yolov8x.runner.performant_runner import YOLOv8xPerformantRunner

        self.ttnn = ttnn
        self.tt_model_name = args.tt_model

        n_par = getattr(args, "tt_slice_parallel_devices", None)
        if n_par is not None:
            n_par = int(n_par)
            max_devices = _system_or_configured_mesh_device_count(ttnn, args)
            if n_par > 1 and max_devices <= 1:
                raise ValueError(
                    "--tt-slice-parallel-devices > 1 requires a multi-chip system (or set --tt-mesh-shape to a multi-device shape)."
                )
            self.device, self._is_mesh_device = _open_tt_slice_parallel_device(ttnn, args, n_par)
        else:
            self.device, self._is_mesh_device = _open_tt_yolo_device(ttnn, args)

        self.num_devices = self.device.get_num_devices()
        inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(self.device)
        self.output_mesh_composer = output_mesh_composer
        batch_size = self.num_devices

        if args.tt_model == "yolov8s":
            self.runner = YOLOv8sPerformantRunner(
                self.device,
                device_batch_size=batch_size,
                mesh_mapper=inputs_mesh_mapper,
                mesh_composer=output_mesh_composer,
                weights_mesh_mapper=weights_mesh_mapper,
            )
        elif args.tt_model == "yolov8x":
            self.runner = YOLOv8xPerformantRunner(
                self.device,
                device_batch_size=batch_size,
                inputs_mesh_mapper=inputs_mesh_mapper,
                weights_mesh_mapper=weights_mesh_mapper,
                outputs_mesh_composer=output_mesh_composer,
            )
        else:
            raise ValueError(f"Unsupported --tt-model: {args.tt_model}")
        self.names = load_coco_class_names()
        self.confidence_threshold = args.confidence_threshold
        self.slice_parallel_devices = getattr(args, "tt_slice_parallel_devices", None)

        mesh_shape_list = None
        try:
            mesh_shape_list = list(self.device.shape)
        except Exception:
            pass
        n_par_cfg = getattr(args, "tt_slice_parallel_devices", None)
        if n_par_cfg is not None and int(n_par_cfg) > 1 and not getattr(args, "tt_force_single_device", False):
            opened_desc = f"open_mesh_device MeshShape(1, {int(n_par_cfg)})"
        elif self._is_mesh_device:
            opened_desc = "open_mesh_device (full or explicit --tt-mesh-shape)"
        else:
            opened_desc = "open_device (single chip)"
        self.device_verify_info = {
            "num_devices": self.num_devices,
            "is_mesh_device": self._is_mesh_device,
            "device_batch_size": batch_size,
            "tt_slice_parallel_devices": n_par_cfg,
            "mesh_shape": mesh_shape_list,
            "opened_as": opened_desc,
        }
        print(
            "TT device verify: "
            f"num_devices={self.num_devices}, batch_dim_sharded={batch_size}, "
            f"mesh={self._is_mesh_device}, mesh_shape={mesh_shape_list!r}, "
            f"{opened_desc}, --tt-slice-parallel-devices={n_par_cfg!r}"
        )

    def close(self):
        try:
            self.runner.release()
        finally:
            try:
                self.ttnn.synchronize_device(self.device)
            except Exception:
                pass
            if self._is_mesh_device:
                self.ttnn.close_mesh_device(self.device)
            else:
                self.ttnn.close_device(self.device)

    def _forward_preprocessed_batch(self, im_tensor: torch.Tensor, orig_imgs: list, paths: tuple):
        """im_tensor shape (num_devices, 3, H, W); orig_imgs length num_devices."""
        preds = self.runner.run(im_tensor)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        preds = self.ttnn.to_torch(preds, dtype=torch.float32, mesh_composer=self.output_mesh_composer)
        return postprocess(preds, im_tensor, orig_imgs, paths, self.names)

    def _forward_preprocessed_batch_timed(self, im_tensor: torch.Tensor, orig_imgs: list, paths: tuple):
        """Returns (results list) and per-forward device vs host post timing."""
        t0 = time.perf_counter()
        preds = self.runner.run(im_tensor)
        t1 = time.perf_counter()
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        preds = self.ttnn.to_torch(preds, dtype=torch.float32, mesh_composer=self.output_mesh_composer)
        t2 = time.perf_counter()
        results = postprocess(preds, im_tensor, orig_imgs, paths, self.names)
        t3 = time.perf_counter()
        part = {
            "device_run_sec": t1 - t0,
            "host_ttnn_to_torch_sec": t2 - t1,
            "host_tt_torch_postprocess_sec": t3 - t2,
            "host_d2h_and_tt_postprocess_sec": t3 - t1,
        }
        return results, part

    def infer_bgr_image(self, image_bgr, image_id: str):
        im_tensor = preprocess([image_bgr], res=self._TT_INPUT_RES)
        if self.num_devices > 1:
            im_tensor = im_tensor.repeat(self.num_devices, 1, 1, 1)
        orig_imgs = [image_bgr] * self.num_devices
        paths = ([image_id] * self.num_devices,)
        results = self._forward_preprocessed_batch(im_tensor, orig_imgs, paths)
        return results[0]

    def infer_bgr_image_timed(self, image_bgr, image_id: str):
        t0 = time.perf_counter()
        im_tensor = preprocess([image_bgr], res=self._TT_INPUT_RES)
        if self.num_devices > 1:
            im_tensor = im_tensor.repeat(self.num_devices, 1, 1, 1)
        t1 = time.perf_counter()
        orig_imgs = [image_bgr] * self.num_devices
        paths = ([image_id] * self.num_devices,)
        results, part = self._forward_preprocessed_batch_timed(im_tensor, orig_imgs, paths)
        part["host_preprocess_sec"] = t1 - t0
        return results[0], part

    def infer_bgr_images_distinct(self, images_bgr: list, image_id: str):
        """
        One 640×640 (letterboxed) input per mesh device; batch dim is sharded so each chip runs a different slice.
        """
        if len(images_bgr) != self.num_devices:
            raise ValueError(f"Expected {self.num_devices} images, got {len(images_bgr)}")
        parts = [preprocess([im], res=self._TT_INPUT_RES) for im in images_bgr]
        im_tensor = torch.cat(parts, dim=0)
        paths = ([f"{image_id}#slot{i}" for i in range(self.num_devices)],)
        return self._forward_preprocessed_batch(im_tensor, images_bgr, paths)

    def infer_bgr_images_distinct_timed(self, images_bgr: list, image_id: str):
        t0 = time.perf_counter()
        parts = [preprocess([im], res=self._TT_INPUT_RES) for im in images_bgr]
        im_tensor = torch.cat(parts, dim=0)
        t1 = time.perf_counter()
        paths = ([f"{image_id}#slot{i}" for i in range(self.num_devices)],)
        results, part = self._forward_preprocessed_batch_timed(im_tensor, images_bgr, paths)
        part["host_preprocess_sec"] = t1 - t0
        return results, part


def result_to_object_predictions(result_dict, shift_xy=(0, 0), full_shape=None, confidence_threshold=0.25):
    preds = []
    boxes = result_dict["boxes"]["xyxy"].tolist()
    confs = result_dict["boxes"]["conf"].tolist()
    clss = result_dict["boxes"]["cls"].tolist()
    for box, conf, cls in zip(boxes, confs, clss):
        if conf < confidence_threshold:
            continue
        cls_int = int(cls)
        preds.append(
            ObjectPrediction(
                bbox=[int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                category_id=cls_int,
                category_name=result_dict["names"][cls_int],
                score=float(conf),
                shift_amount=[int(shift_xy[0]), int(shift_xy[1])],
                full_shape=full_shape,
            )
        )
    return preds


def _tagged_slice_dir(parent: Path, image_stem: str) -> Path:
    d = parent / f"{image_stem}_slices"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_tagged_slice_png(
    slice_rgb: np.ndarray,
    object_prediction_list: list,
    slice_dir: Path,
    index: int,
    start_x: int,
    start_y: int,
):
    """Draw SAHI ObjectPredictions (slice coordinates) on slice_rgb and save PNG."""
    visualize_object_predictions(
        np.ascontiguousarray(slice_rgb),
        object_prediction_list,
        output_dir=str(slice_dir),
        file_name=f"slice_{index:03d}_x{start_x}_y{start_y}",
        hide_labels=False,
        hide_conf=False,
    )


def save_tagged_tt_slice_from_result(
    slice_rgb: np.ndarray,
    tt_result: dict,
    confidence_threshold: float,
    slice_dir: Path,
    index: int,
    start_x: int,
    start_y: int,
):
    preds = result_to_object_predictions(
        tt_result,
        shift_xy=(0, 0),
        full_shape=[int(slice_rgb.shape[0]), int(slice_rgb.shape[1])],
        confidence_threshold=confidence_threshold,
    )
    save_tagged_slice_png(slice_rgb, preds, slice_dir, index, start_x, start_y)


def build_postprocess(postprocess_type: str, match_metric: str, match_threshold: float, class_agnostic: bool):
    cls_map = {
        "NMS": NMSPostprocess,
        "NMM": NMMPostprocess,
        "GREEDYNMM": GreedyNMMPostprocess,
        "LSNMS": LSNMSPostprocess,
    }
    return cls_map[postprocess_type](
        match_threshold=match_threshold,
        match_metric=match_metric,
        class_agnostic=class_agnostic,
    )


def run_tt_full_prediction(tt_backend: TTYoloBackend, image_path: Path):
    start = time.perf_counter()
    t0 = time.perf_counter()
    try:
        image_bgr = load_image_bgr_sahi(image_path)
    except Exception as e:
        raise ValueError(f"Failed to read image: {image_path}") from e
    t_load = time.perf_counter() - t0
    result_dict, fwd = tt_backend.infer_bgr_image_timed(image_bgr, str(image_path))
    obj_preds = result_to_object_predictions(
        result_dict,
        shift_xy=(0, 0),
        full_shape=[image_bgr.shape[0], image_bgr.shape[1]],
        confidence_threshold=tt_backend.confidence_threshold,
    )
    elapsed = time.perf_counter() - start
    timing = SlicedPipelineTiming(
        total_wall_sec=elapsed,
        host_sahi_slice_sec=0.0,
        host_preprocess_before_device_sec=t_load + fwd["host_preprocess_sec"],
        host_read_image_sec=t_load,
        host_cpu_prep_letterbox_sec=fwd["host_preprocess_sec"],
        device_inference_sec=fwd["device_run_sec"],
        device_ttnn_run_sec=fwd["device_run_sec"],
        host_ttnn_to_torch_sec=fwd["host_ttnn_to_torch_sec"],
        host_tt_torch_postprocess_sec=fwd["host_tt_torch_postprocess_sec"],
        host_postprocess_and_sahi_merge_sec=fwd["host_d2h_and_tt_postprocess_sec"],
    )
    return PredictionResult(object_prediction_list=obj_preds, image=str(image_path)), elapsed, timing.to_summary_dict()


def run_tt_sliced_prediction(
    tt_backend: TTYoloBackend,
    image_path: Path,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
    postprocess_type: str,
    postprocess_match_metric: str,
    postprocess_match_threshold: float,
    postprocess_class_agnostic: bool,
    tagged_slice_parent: Path | None = None,
    slice_file_stem: str | None = None,
):
    start = time.perf_counter()
    tagged_dir = None
    if tagged_slice_parent is not None and slice_file_stem is not None:
        tagged_dir = _tagged_slice_dir(tagged_slice_parent, slice_file_stem)

    t_read0 = time.perf_counter()
    image_pil = read_image_as_pil(str(image_path), exif_fix=True)
    host_read_image_sec = time.perf_counter() - t_read0
    t_sl0 = time.perf_counter()
    slice_result = slice_image(
        image=image_pil,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        auto_slice_resolution=False,
    )
    host_sahi_slice_sec = time.perf_counter() - t_sl0
    print(f"Performing prediction on {len(slice_result.images)} slices.")
    all_obj_preds = []
    full_shape = [slice_result.original_image_height, slice_result.original_image_width]
    use_distinct_slice_batch = tt_backend.slice_parallel_devices is not None
    parallel = tt_backend.num_devices if use_distinct_slice_batch else 1
    black_bgr = np.zeros((TTYoloBackend._TT_INPUT_RES[1], TTYoloBackend._TT_INPUT_RES[0], 3), dtype=np.uint8)

    host_preprocess_sec = 0.0
    device_inference_sec = 0.0
    host_ttnn_to_torch_sec = 0.0
    host_tt_torch_postprocess_sec = 0.0
    slice_export_sec = 0.0

    slice_entries = list(zip(slice_result.starting_pixels, slice_result.images))
    slice_global_idx = 0
    if not use_distinct_slice_batch:
        for (start_x, start_y), slice_img in slice_entries:
            t_h0 = time.perf_counter()
            slice_bgr = cv2.cvtColor(slice_img, cv2.COLOR_RGB2BGR)
            host_preprocess_sec += time.perf_counter() - t_h0
            result_dict, fwd = tt_backend.infer_bgr_image_timed(slice_bgr, str(image_path))
            host_preprocess_sec += fwd["host_preprocess_sec"]
            device_inference_sec += fwd["device_run_sec"]
            host_ttnn_to_torch_sec += fwd["host_ttnn_to_torch_sec"]
            host_tt_torch_postprocess_sec += fwd["host_tt_torch_postprocess_sec"]
            if tagged_dir is not None:
                t_e0 = time.perf_counter()
                save_tagged_tt_slice_from_result(
                    slice_img,
                    result_dict,
                    tt_backend.confidence_threshold,
                    tagged_dir,
                    slice_global_idx,
                    start_x,
                    start_y,
                )
                slice_export_sec += time.perf_counter() - t_e0
            slice_global_idx += 1
            all_obj_preds.extend(
                result_to_object_predictions(
                    result_dict,
                    shift_xy=(start_x, start_y),
                    full_shape=full_shape,
                    confidence_threshold=tt_backend.confidence_threshold,
                )
            )
    else:
        for idx, n_valid in parallel_slice_chunk_bounds(len(slice_entries), parallel):
            chunk = slice_entries[idx : idx + n_valid]
            shifts = []
            batch_bgr = []
            slice_rgbs = []
            t_h0 = time.perf_counter()
            for (start_x, start_y), slice_img in chunk:
                shifts.append((start_x, start_y))
                batch_bgr.append(cv2.cvtColor(slice_img, cv2.COLOR_RGB2BGR))
                slice_rgbs.append(slice_img)
            for _ in range(parallel - len(chunk)):
                shifts.append((0, 0))
                batch_bgr.append(black_bgr.copy())
                slice_rgbs.append(None)
            host_preprocess_sec += time.perf_counter() - t_h0
            batch_results, fwd = tt_backend.infer_bgr_images_distinct_timed(batch_bgr, str(image_path))
            host_preprocess_sec += fwd["host_preprocess_sec"]
            device_inference_sec += fwd["device_run_sec"]
            host_ttnn_to_torch_sec += fwd["host_ttnn_to_torch_sec"]
            host_tt_torch_postprocess_sec += fwd["host_tt_torch_postprocess_sec"]
            for j in range(n_valid):
                if tagged_dir is not None:
                    t_e0 = time.perf_counter()
                    sx, sy = shifts[j]
                    save_tagged_tt_slice_from_result(
                        slice_rgbs[j],
                        batch_results[j],
                        tt_backend.confidence_threshold,
                        tagged_dir,
                        slice_global_idx,
                        sx,
                        sy,
                    )
                    slice_export_sec += time.perf_counter() - t_e0
                slice_global_idx += 1
                all_obj_preds.extend(
                    result_to_object_predictions(
                        batch_results[j],
                        shift_xy=shifts[j],
                        full_shape=full_shape,
                        confidence_threshold=tt_backend.confidence_threshold,
                    )
                )
    # Match sahi.predict.get_sliced_prediction: shift each slice to full-image coords before NMS/NMM.
    t_sh0 = time.perf_counter()
    all_obj_preds = [p.get_shifted_object_prediction() for p in all_obj_preds]
    host_shift_sec = time.perf_counter() - t_sh0
    postprocess = build_postprocess(
        postprocess_type=postprocess_type,
        match_metric=postprocess_match_metric,
        match_threshold=postprocess_match_threshold,
        class_agnostic=postprocess_class_agnostic,
    )
    t_m0 = time.perf_counter()
    merged_obj_preds = postprocess(all_obj_preds)
    sahi_merge_sec = time.perf_counter() - t_m0
    elapsed = time.perf_counter() - start
    host_post_incl_merge = host_ttnn_to_torch_sec + host_tt_torch_postprocess_sec + host_shift_sec + sahi_merge_sec
    n_slices = len(slice_result.images)
    prep_mean = (host_preprocess_sec / n_slices) if n_slices else 0.0
    timing = SlicedPipelineTiming(
        total_wall_sec=elapsed,
        host_read_image_sec=host_read_image_sec,
        host_sahi_slice_sec=host_sahi_slice_sec,
        host_preprocess_before_device_sec=host_preprocess_sec,
        host_cpu_prep_letterbox_sec=host_preprocess_sec,
        host_cpu_prep_mean_per_slice_sec=prep_mean,
        device_inference_sec=device_inference_sec,
        device_ttnn_run_sec=device_inference_sec,
        host_ttnn_to_torch_sec=host_ttnn_to_torch_sec,
        host_tt_torch_postprocess_sec=host_tt_torch_postprocess_sec,
        host_postprocess_and_sahi_merge_sec=host_post_incl_merge,
        host_slice_image_export_sec=slice_export_sec,
        sahi_shift_to_full_sec=host_shift_sec,
        sahi_merge_sec=sahi_merge_sec,
        extra={"tt_num_slices": n_slices} if n_slices else {},
    )
    if tagged_dir is not None:
        print(f"Saved {len(slice_result.images)} per-slice images with detections (pre-merge) to: {tagged_dir}")
    return (
        PredictionResult(object_prediction_list=merged_obj_preds, image=str(image_path)),
        elapsed,
        timing.to_summary_dict(),
    )


def run_sliced_prediction(
    detection_model,
    image_path: str,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
    perform_standard_pred: bool,
    postprocess_type: str,
    postprocess_match_metric: str,
    postprocess_match_threshold: float,
    postprocess_class_agnostic: bool,
    tagged_slice_parent: Path | None = None,
    slice_file_stem: str | None = None,
):
    """
    Ultralytics + SAHI sliced path. When tagged_slice_parent is set, mirrors get_sliced_prediction but
    saves each crop with slice-local boxes immediately after get_prediction (before shift + merge).
    """
    start = time.perf_counter()
    if tagged_slice_parent is None or slice_file_stem is None:
        result = get_sliced_prediction(
            image=image_path,
            detection_model=detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            perform_standard_pred=perform_standard_pred,
            postprocess_type=postprocess_type,
            postprocess_match_metric=postprocess_match_metric,
            postprocess_match_threshold=postprocess_match_threshold,
            postprocess_class_agnostic=postprocess_class_agnostic,
            auto_slice_resolution=False,
        )
        elapsed = time.perf_counter() - start
        timing = _timing_from_sahi_durations(result.durations_in_seconds)
        timing.total_wall_sec = elapsed
        return result, elapsed, timing.to_summary_dict()

    from sahi.models.ultralytics import UltralyticsDetectionModel

    tagged_dir = _tagged_slice_dir(tagged_slice_parent, slice_file_stem)
    t_read0 = time.perf_counter()
    image_pil = read_image_as_pil(str(image_path), exif_fix=True)
    host_read_image_sec = time.perf_counter() - t_read0
    t_sl0 = time.perf_counter()
    slice_image_result = slice_image(
        image=image_pil,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        auto_slice_resolution=False,
    )
    host_sahi_slice_sec = time.perf_counter() - t_sl0
    num_slices = len(slice_image_result.images)
    print(f"Performing prediction on {num_slices} slices.")

    pt = postprocess_type
    if isinstance(detection_model, UltralyticsDetectionModel) and detection_model.is_obb:
        pt = "NMS"
    postprocess_merge = build_postprocess(
        pt, postprocess_match_metric, postprocess_match_threshold, postprocess_class_agnostic
    )

    device_inference_sec = 0.0
    host_post_per_slice_sec = 0.0
    slice_export_sec = 0.0
    object_prediction_list = []
    for slice_ind in range(num_slices):
        slice_img = slice_image_result.images[slice_ind]
        sx, sy = slice_image_result.starting_pixels[slice_ind]
        prediction_result = get_prediction(
            image=np.ascontiguousarray(slice_img),
            detection_model=detection_model,
            shift_amount=[sx, sy],
            full_shape=[
                slice_image_result.original_image_height,
                slice_image_result.original_image_width,
            ],
        )
        d = prediction_result.durations_in_seconds
        device_inference_sec += float(d.get("prediction", 0) or 0)
        host_post_per_slice_sec += float(d.get("postprocess", 0) or 0)
        t_e0 = time.perf_counter()
        save_tagged_slice_png(
            slice_img,
            prediction_result.object_prediction_list,
            tagged_dir,
            slice_ind,
            sx,
            sy,
        )
        slice_export_sec += time.perf_counter() - t_e0
        for object_prediction in prediction_result.object_prediction_list:
            if object_prediction:
                object_prediction_list.append(object_prediction.get_shifted_object_prediction())

    if num_slices > 1 and perform_standard_pred:
        prediction_result = get_prediction(
            image=image_path,
            detection_model=detection_model,
            shift_amount=[0, 0],
            full_shape=[
                slice_image_result.original_image_height,
                slice_image_result.original_image_width,
            ],
        )
        d = prediction_result.durations_in_seconds
        device_inference_sec += float(d.get("prediction", 0) or 0)
        host_post_per_slice_sec += float(d.get("postprocess", 0) or 0)
        object_prediction_list.extend(prediction_result.object_prediction_list)

    t_m0 = time.perf_counter()
    if len(object_prediction_list) > 1:
        object_prediction_list = postprocess_merge(object_prediction_list)
    sahi_merge_sec = time.perf_counter() - t_m0

    elapsed = time.perf_counter() - start
    host_post_total = host_post_per_slice_sec + sahi_merge_sec
    timing = SlicedPipelineTiming(
        total_wall_sec=elapsed,
        host_read_image_sec=host_read_image_sec,
        host_sahi_slice_sec=host_sahi_slice_sec,
        host_preprocess_before_device_sec=0.0,
        device_inference_sec=device_inference_sec,
        host_postprocess_and_sahi_merge_sec=host_post_total,
        host_slice_image_export_sec=slice_export_sec,
        sahi_merge_sec=sahi_merge_sec,
        sahi_ultralytics_per_slice_host_sec=host_post_per_slice_sec,
    )
    print(f"Saved {num_slices} per-slice images with detections (pre-merge) to: {tagged_dir}")
    return (
        PredictionResult(image=image_path, object_prediction_list=object_prediction_list),
        elapsed,
        timing.to_summary_dict(),
    )


def maybe_export_visuals(result, export_dir: Path, file_stem: str):
    result.export_visuals(export_dir=str(export_dir), file_name=file_stem, hide_labels=False, hide_conf=False)


def save_slice_grid_overlay(
    image_path: Path,
    export_dir: Path,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
):
    with Image.open(image_path) as original:
        overlay = original.convert("RGB")

    slice_result = slice_image(
        image=str(image_path),
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        auto_slice_resolution=False,
    )

    draw = ImageDraw.Draw(overlay)
    for index, ((start_x, start_y), slice_arr) in enumerate(zip(slice_result.starting_pixels, slice_result.images)):
        slice_h = int(slice_arr.shape[0])
        slice_w = int(slice_arr.shape[1])
        end_x = start_x + slice_w - 1
        end_y = start_y + slice_h - 1
        draw.rectangle([(start_x, start_y), (end_x, end_y)], outline=(255, 0, 0), width=2)
        label = f"{slice_w}x{slice_h}"
        text_x = start_x + 4
        text_y = start_y + 4
        bbox = draw.textbbox((text_x, text_y), label)
        draw.rectangle(bbox, fill=(255, 255, 255))
        draw.text((text_x, text_y), label, fill=(255, 0, 0))

    out_path = export_dir / f"{image_path.stem}_slice_grid.png"
    overlay.save(out_path)


def resize_images_for_processing(
    images: list[Path], target_width: int, target_height: int, temp_dir: Path
) -> list[Path]:
    resized_images = []
    for image_path in images:
        out_path = temp_dir / f"{image_path.stem}_{target_width}x{target_height}{image_path.suffix.lower() or '.jpg'}"
        with Image.open(image_path) as image:
            resized = image.convert("RGB").resize((target_width, target_height))
            resized.save(out_path)
        resized_images.append(out_path)
    return resized_images


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_path)
    if not images:
        raise ValueError(f"No images found at: {input_path}")
    images, unreadable_images = split_readable_images(images)
    if unreadable_images:
        print(f"Skipping {len(unreadable_images)} unreadable image(s):")
        for bad_path in unreadable_images:
            print(f"  - {bad_path}")
    if not images:
        raise ValueError("No readable images were found in --input. " "Check file format/content and try again.")

    detection_model = None
    tt_backend = None
    if args.backend == "ultralytics":
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=args.model,
            confidence_threshold=args.confidence_threshold,
            device=args.device,
        )
    else:
        try:
            tt_backend = TTYoloBackend(args)
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize TT YOLO backend. "
                "If the cause mentions arc core timeout or Ethernet flush, reset the board (e.g. `tt-smi -r`) "
                "after an unclean exit, or pass an explicit mesh, e.g. `--tt-mesh-shape 1 8`. "
                "Otherwise this may be an API/runtime mismatch: run "
                "`pytest --disable-warnings models/demos/yolov8s/tests/pcc/test_yolov8s.py::test_yolov8s_640` "
                "or `pytest --disable-warnings models/demos/yolov8x/tests/pcc/test_yolov8x.py::test_yolov8x_640`."
            ) from e

    processing_images = images
    temp_resize_dir = None
    if args.pre_resize_to is not None:
        resize_width, resize_height = args.pre_resize_to
        temp_resize_dir = Path(tempfile.mkdtemp(prefix="sahi_preresize_"))
        processing_images = resize_images_for_processing(images, resize_width, resize_height, temp_resize_dir)

    slice_save_parent = output_dir if args.save_slice_images else None
    rows = []
    for image in processing_images:
        if args.backend == "ultralytics":
            full_result, full_time, full_timing = run_prediction(detection_model, str(image))
            sliced_result, sliced_time, sliced_timing = run_sliced_prediction(
                detection_model=detection_model,
                image_path=str(image),
                slice_height=args.slice_height,
                slice_width=args.slice_width,
                overlap_height_ratio=args.overlap_height_ratio,
                overlap_width_ratio=args.overlap_width_ratio,
                perform_standard_pred=args.perform_standard_pred,
                postprocess_type=args.postprocess_type,
                postprocess_match_metric=args.postprocess_match_metric,
                postprocess_match_threshold=args.postprocess_match_threshold,
                postprocess_class_agnostic=args.postprocess_class_agnostic,
                tagged_slice_parent=slice_save_parent,
                slice_file_stem=image.stem if slice_save_parent else None,
            )
        else:
            full_result, full_time, full_timing = run_tt_full_prediction(tt_backend, image)
            sliced_result, sliced_time, sliced_timing = run_tt_sliced_prediction(
                tt_backend=tt_backend,
                image_path=image,
                slice_height=args.slice_height,
                slice_width=args.slice_width,
                overlap_height_ratio=args.overlap_height_ratio,
                overlap_width_ratio=args.overlap_width_ratio,
                postprocess_type=args.postprocess_type,
                postprocess_match_metric=args.postprocess_match_metric,
                postprocess_match_threshold=args.postprocess_match_threshold,
                postprocess_class_agnostic=args.postprocess_class_agnostic,
                tagged_slice_parent=slice_save_parent,
                slice_file_stem=image.stem if slice_save_parent else None,
            )

        full_count = len(full_result.object_prediction_list)
        sliced_count = len(sliced_result.object_prediction_list)

        row = {
            "image": str(image),
            "full_time_sec": round(full_time, 4),
            "sliced_time_sec": round(sliced_time, 4),
            "full_detections": full_count,
            "sliced_detections": sliced_count,
            "time_ratio_sliced_over_full": round(sliced_time / full_time, 4) if full_time > 0 else None,
            "delta_detections_sliced_minus_full": sliced_count - full_count,
            "full_timing_sec": full_timing,
            "sliced_timing_sec": sliced_timing,
        }
        rows.append(row)

        if args.save_visuals:
            maybe_export_visuals(full_result, output_dir, f"{image.stem}_full")
            maybe_export_visuals(sliced_result, output_dir, f"{image.stem}_sliced")
        if args.save_slice_grid_overlay:
            save_slice_grid_overlay(
                image_path=image,
                export_dir=output_dir,
                slice_height=args.slice_height,
                slice_width=args.slice_width,
                overlap_height_ratio=args.overlap_height_ratio,
                overlap_width_ratio=args.overlap_width_ratio,
            )

        ft = full_timing
        st = sliced_timing
        detail_lines = []
        d_full = _format_timing_detail_line("full (single)", ft)
        d_sliced = _format_timing_detail_line("sliced", st)
        if d_full:
            detail_lines.append(d_full)
        if d_sliced:
            detail_lines.append(d_sliced)
        detail_block = ("\n" + "\n".join(detail_lines)) if detail_lines else ""
        print(
            f"[{image.name}] full={full_count} ({full_time:.3f}s), "
            f"sliced={sliced_count} ({sliced_time:.3f}s), "
            f"delta={sliced_count - full_count}\n"
            f"{_format_pre_device_post_line('full (single)', ft)}\n"
            f"{_format_pre_device_post_line('sliced', st)}"
            f"{detail_block}"
        )

    summary = {
        "config": {
            "model": args.model if args.backend == "ultralytics" else args.tt_model,
            "backend": args.backend,
            "tt_model": args.tt_model if args.backend == "tt" else None,
            "tt_slice_parallel_devices": args.tt_slice_parallel_devices if args.backend == "tt" else None,
            "tt_mesh_shape": list(args.tt_mesh_shape) if args.backend == "tt" and args.tt_mesh_shape else None,
            "tt_device_verify": tt_backend.device_verify_info if args.backend == "tt" and tt_backend else None,
            "confidence_threshold": args.confidence_threshold,
            "device": args.device if args.backend == "ultralytics" else f"tt:{args.tt_device_id}",
            "slice_height": args.slice_height,
            "slice_width": args.slice_width,
            "overlap_height_ratio": args.overlap_height_ratio,
            "overlap_width_ratio": args.overlap_width_ratio,
            "perform_standard_pred": args.perform_standard_pred,
            "postprocess_type": args.postprocess_type,
            "postprocess_match_metric": args.postprocess_match_metric,
            "postprocess_match_threshold": args.postprocess_match_threshold,
            "postprocess_class_agnostic": args.postprocess_class_agnostic,
            "pre_resize_to": args.pre_resize_to,
            "save_slice_images": args.save_slice_images,
            "num_images": len(images),
        },
        "per_image": rows,
    }

    if rows:
        n = len(rows)
        timing_keys = [
            "total_wall_sec",
            "host_sahi_slice_sec",
            "host_preprocess_before_device_sec",
            "host_slice_and_preprocess_sec",
            "device_inference_sec",
            "host_postprocess_and_sahi_merge_sec",
            "host_slice_image_export_sec",
            "host_read_image_sec",
            "host_cpu_prep_letterbox_sec",
            "host_cpu_prep_mean_per_slice_sec",
            "device_ttnn_run_sec",
            "host_ttnn_to_torch_sec",
            "host_tt_torch_postprocess_sec",
            "sahi_shift_to_full_sec",
            "sahi_merge_sec",
            "sahi_ultralytics_per_slice_host_sec",
        ]

        def mean_sliced_timing(key: str) -> float:
            return round(sum(r["sliced_timing_sec"].get(key, 0) for r in rows) / n, 4)

        def mean_full_timing(key: str) -> float:
            return round(sum(r["full_timing_sec"].get(key, 0) for r in rows) / n, 4)

        summary["aggregate"] = {
            "mean_full_time_sec": round(sum(r["full_time_sec"] for r in rows) / n, 4),
            "mean_sliced_time_sec": round(sum(r["sliced_time_sec"] for r in rows) / n, 4),
            "mean_delta_detections_sliced_minus_full": round(
                sum(r["delta_detections_sliced_minus_full"] for r in rows) / n, 4
            ),
            "mean_full_timing_sec": {k: mean_full_timing(k) for k in timing_keys},
            "mean_sliced_timing_sec": {k: mean_sliced_timing(k) for k in timing_keys},
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary to: {summary_path}")

    if temp_resize_dir is not None and temp_resize_dir.exists():
        shutil.rmtree(temp_resize_dir, ignore_errors=True)
    if tt_backend is not None:
        tt_backend.close()


if __name__ == "__main__":
    main()
