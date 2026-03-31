#!/usr/bin/env python3
import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import torch
from PIL import Image, ImageDraw, UnidentifiedImageError
from sahi import AutoDetectionModel
from sahi.postprocess.combine import GreedyNMMPostprocess, LSNMSPostprocess, NMMPostprocess, NMSPostprocess
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.prediction import ObjectPrediction, PredictionResult
from sahi.slicing import slice_image

from models.demos.utils.common_demo_utils import load_coco_class_names, postprocess, preprocess

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Ultralytics full-image inference vs SAHI sliced inference.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to an image file or directory containing images.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Ultralytics model path/name (e.g. yolo11n.pt, yolov8s.pt).",
    )
    parser.add_argument(
        "--backend",
        default="ultralytics",
        choices=["ultralytics", "tt"],
        help="Inference backend: ultralytics (host) or tt (Tenstorrent).",
    )
    parser.add_argument(
        "--tt-model",
        default="yolov8x",
        help="TT model name. Currently supported: yolov8x.",
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
        help="TT device l1_small_size for YOLOv8x runner.",
    )
    parser.add_argument(
        "--tt-trace-region-size",
        type=int,
        default=6434816,
        help="TT device trace_region_size for YOLOv8x runner.",
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
    return result, elapsed


class TTYoloBackend:
    def __init__(self, args):
        if args.tt_model != "yolov8x":
            raise ValueError("Only --tt-model yolov8x is currently supported.")

        import ttnn
        from models.demos.yolov8x.runner.performant_runner import YOLOv8xPerformantRunner

        self.ttnn = ttnn
        self.device = ttnn.open_device(
            device_id=args.tt_device_id,
            l1_small_size=args.tt_l1_small_size,
            trace_region_size=args.tt_trace_region_size,
            num_command_queues=2,
        )
        self.runner = YOLOv8xPerformantRunner(self.device, device_batch_size=1)
        self.names = load_coco_class_names()
        self.confidence_threshold = args.confidence_threshold

    def close(self):
        try:
            self.runner.release()
        finally:
            self.ttnn.close_device(self.device)

    def infer_bgr_image(self, image_bgr, image_id: str):
        im_tensor = preprocess([image_bgr], res=(640, 640))
        preds = self.runner.run(im_tensor)
        preds = self.ttnn.to_torch(preds, dtype=torch.float32)
        results = postprocess(preds, im_tensor, [image_bgr], ([image_id],), self.names)
        return results[0]


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
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    result_dict = tt_backend.infer_bgr_image(image_bgr, str(image_path))
    obj_preds = result_to_object_predictions(
        result_dict,
        shift_xy=(0, 0),
        full_shape=[image_bgr.shape[0], image_bgr.shape[1]],
        confidence_threshold=tt_backend.confidence_threshold,
    )
    elapsed = time.perf_counter() - start
    return PredictionResult(object_prediction_list=obj_preds, image=str(image_path)), elapsed


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
):
    start = time.perf_counter()
    slice_result = slice_image(
        image=str(image_path),
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        auto_slice_resolution=False,
    )
    print(f"Performing prediction on {len(slice_result.images)} slices.")
    all_obj_preds = []
    full_shape = [slice_result.original_image_height, slice_result.original_image_width]
    for (start_x, start_y), slice_img in zip(slice_result.starting_pixels, slice_result.images):
        slice_bgr = cv2.cvtColor(slice_img, cv2.COLOR_RGB2BGR)
        result_dict = tt_backend.infer_bgr_image(slice_bgr, str(image_path))
        all_obj_preds.extend(
            result_to_object_predictions(
                result_dict,
                shift_xy=(start_x, start_y),
                full_shape=full_shape,
                confidence_threshold=tt_backend.confidence_threshold,
            )
        )
    postprocess = build_postprocess(
        postprocess_type=postprocess_type,
        match_metric=postprocess_match_metric,
        match_threshold=postprocess_match_threshold,
        class_agnostic=postprocess_class_agnostic,
    )
    merged_obj_preds = postprocess(all_obj_preds)
    # Normalize merged predictions to global coordinates before visualization/export.
    merged_obj_preds = [pred.get_shifted_object_prediction() for pred in merged_obj_preds]
    elapsed = time.perf_counter() - start
    return PredictionResult(object_prediction_list=merged_obj_preds, image=str(image_path)), elapsed


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
):
    start = time.perf_counter()
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
    )
    elapsed = time.perf_counter() - start
    return result, elapsed


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
                "Failed to initialize TT backend for YOLOv8x. "
                "This is often due to an API/runtime mismatch between current tt-metal and the YOLOv8x demo runner. "
                "Please verify the standard TT YOLOv8x demo command works first: "
                "`pytest --disable-warnings models/demos/yolov8x/tests/pcc/test_yolov8x.py::test_yolov8x_640`"
            ) from e

    processing_images = images
    temp_resize_dir = None
    if args.pre_resize_to is not None:
        resize_width, resize_height = args.pre_resize_to
        temp_resize_dir = Path(tempfile.mkdtemp(prefix="sahi_preresize_"))
        processing_images = resize_images_for_processing(images, resize_width, resize_height, temp_resize_dir)

    rows = []
    for image in processing_images:
        if args.backend == "ultralytics":
            full_result, full_time = run_prediction(detection_model, str(image))
            sliced_result, sliced_time = run_sliced_prediction(
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
            )
        else:
            full_result, full_time = run_tt_full_prediction(tt_backend, image)
            sliced_result, sliced_time = run_tt_sliced_prediction(
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

        print(
            f"[{image.name}] full={full_count} ({full_time:.3f}s), "
            f"sliced={sliced_count} ({sliced_time:.3f}s), "
            f"delta={sliced_count - full_count}"
        )

    summary = {
        "config": {
            "model": args.model if args.backend == "ultralytics" else args.tt_model,
            "backend": args.backend,
            "tt_model": args.tt_model if args.backend == "tt" else None,
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
            "num_images": len(images),
        },
        "per_image": rows,
    }

    if rows:
        summary["aggregate"] = {
            "mean_full_time_sec": round(sum(r["full_time_sec"] for r in rows) / len(rows), 4),
            "mean_sliced_time_sec": round(sum(r["sliced_time_sec"] for r in rows) / len(rows), 4),
            "mean_delta_detections_sliced_minus_full": round(
                sum(r["delta_detections_sliced_minus_full"] for r in rows) / len(rows), 4
            ),
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
