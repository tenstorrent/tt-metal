# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import statistics
import time
import uuid
from datetime import datetime, timezone
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import av
import cv2
import numpy as np
import requests
import torch
from loguru import logger
from PIL import Image, ImageDraw
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, RTDetrV2ForObjectDetection
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.vision.detection.rtdetr.common.preprocessing import custom_preprocessor
from models.demos.vision.detection.rtdetr.tt.model import TtRTDetrModel

MODEL_CONFIGS = {
    1: ("PekingU/rtdetr_r50vd", RTDetrForObjectDetection),
    2: ("PekingU/rtdetr_v2_r50vd", RTDetrV2ForObjectDetection),
}

SAMPLE_TEST_IMAGES = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000397133.jpg",
    "http://images.cocodataset.org/val2017/000000037777.jpg",
    "http://images.cocodataset.org/val2017/000000252219.jpg",
    "http://images.cocodataset.org/val2017/000000087038.jpg",
    "http://images.cocodataset.org/val2017/000000174482.jpg",
    "http://images.cocodataset.org/val2017/000000403385.jpg",
    "http://images.cocodataset.org/val2017/000000006818.jpg",
    "http://images.cocodataset.org/val2017/000000480985.jpg",
    "http://images.cocodataset.org/val2017/000000331352.jpg",
]

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}
MAX_VIDEO_DURATION_SECONDS = 20


def load_image(source: str) -> Image.Image:
    if source.startswith(("http://", "https://")):
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    return Image.open(source).convert("RGB")


def create_output_directory() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_directory = OUTPUT_ROOT / f"{timestamp}_{uuid.uuid4().hex[:8]}"
    output_directory.mkdir(parents=True)
    return output_directory


def label_name(id2label: dict, label: int) -> str:
    return id2label.get(label, id2label.get(str(label), str(label)))


def draw_detections(image: Image.Image, detections: dict, id2label: dict) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    line_width = max(2, min(image.size) // 300)

    for score, label, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
        score = float(score)
        label = int(label)
        x_min, y_min, x_max, y_max = [float(value) for value in box]
        color = (
            64 + (label * 47) % 192,
            64 + (label * 89) % 192,
            64 + (label * 131) % 192,
        )
        text = f"{label_name(id2label, label)} {score:.2f}"

        draw.rectangle((x_min, y_min, x_max, y_max), outline=color, width=line_width)
        text_box = draw.textbbox((x_min, y_min), text)
        text_height = text_box[3] - text_box[1]
        text_y = max(0, y_min - text_height - 4)
        text_box = draw.textbbox((x_min + 2, text_y + 2), text)
        draw.rectangle(
            (text_box[0] - 2, text_box[1] - 2, text_box[2] + 2, text_box[3] + 2),
            fill=color,
        )
        draw.text((x_min + 2, text_y + 2), text, fill="black")

    return output


def save_detection_images(
    images: list[Image.Image],
    torch_detections: list[dict],
    tt_detections: list[dict],
    id2label: dict,
    output_directory: Path,
) -> None:
    for index, (image, torch_result, tt_result) in enumerate(zip(images, torch_detections, tt_detections)):
        torch_image = draw_detections(image, torch_result, id2label)
        tt_image = draw_detections(image, tt_result, id2label)

        if len(images) == 1:
            torch_path = output_directory / "torch.png"
            tt_path = output_directory / "tt.png"
        else:
            torch_path = output_directory / f"torch_{index}.png"
            tt_path = output_directory / f"tt_{index}.png"

        torch_image.save(torch_path)
        tt_image.save(tt_path)
        logger.info(f"Saved Torch output to {torch_path}")
        logger.info(f"Saved TT output to {tt_path}")


def run_demo(image_sources: list[str], threshold: float, device_id: int, version: int = 1) -> Path:
    if len(image_sources) != 1:
        raise ValueError("The current demo supports batch size 1")

    output_directory = create_output_directory()
    images = [load_image(source) for source in image_sources]

    model_name, model_class = MODEL_CONFIGS[version]

    logger.info(f"Loading RT-DETR v{version}: {model_name}")
    image_processor = RTDetrImageProcessor.from_pretrained(model_name)
    torch_model = model_class.from_pretrained(model_name).eval()
    pixel_values = image_processor(images=images, return_tensors="pt").pixel_values
    _, _, input_height, input_width = pixel_values.shape
    target_sizes = torch.tensor([(image.height, image.width) for image in images])

    with torch.no_grad():
        torch_outputs = torch_model(pixel_values=pixel_values)

    torch_detections = image_processor.post_process_object_detection(
        torch_outputs,
        threshold=threshold,
        target_sizes=target_sizes,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model.model,
        custom_preprocessor=custom_preprocessor,
    )

    logger.info(f"Opening TT device {device_id}")
    device = ttnn.open_device(device_id=device_id, l1_small_size=16384)
    try:
        tt_model = TtRTDetrModel(
            config=torch_model.config,
            parameters=parameters,
            device=device,
            dtype=ttnn.bfloat16,
            input_height=input_height,
            input_width=input_width,
        )
        tt_pixel_values = ttnn.from_torch(
            pixel_values,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        _, _, _, _, tt_logits, tt_pred_boxes = tt_model(tt_pixel_values)
        tt_outputs = SimpleNamespace(
            logits=ttnn.to_torch(tt_logits).float(),
            pred_boxes=ttnn.to_torch(tt_pred_boxes).float(),
        )
    finally:
        ttnn.close_device(device)

    tt_detections = image_processor.post_process_object_detection(
        tt_outputs,
        threshold=threshold,
        target_sizes=target_sizes,
    )

    save_detection_images(
        images=images,
        torch_detections=torch_detections,
        tt_detections=tt_detections,
        id2label=torch_model.config.id2label,
        output_directory=output_directory,
    )

    logger.info(f"Demo outputs saved in {output_directory}")
    return output_directory


def validate_video_source(source: str) -> None:
    if source.startswith(("http://", "https://")):
        return

    video_path = Path(source)
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise ValueError(f"Unsupported video extension {video_path.suffix!r}; expected one of: {supported}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")


def frame_to_image(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def image_to_frame(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def run_video_demo(source: str, threshold: float, device_id: int, version: int = 1) -> Path:
    validate_video_source(source)
    output_directory = create_output_directory()
    output_path = output_directory / "tt_output.mp4"

    video_capture = cv2.VideoCapture(source)
    if not video_capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    source_fps = video_capture.get(cv2.CAP_PROP_FPS)
    output_fps = source_fps if source_fps > 0 else 30.0
    maximum_frames = int(output_fps * MAX_VIDEO_DURATION_SECONDS)
    progress_interval = max(1, int(output_fps * 5))
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    success, first_frame = video_capture.read()
    if not success:
        video_capture.release()
        raise RuntimeError(f"Video source contains no readable frames: {source}")

    frame_height, frame_width = first_frame.shape[:2]
    try:
        output_container = av.open(str(output_path), mode="w")
        output_stream = output_container.add_stream(
            "libx264",
            rate=Fraction(output_fps).limit_denominator(1001),
        )
        output_stream.width = frame_width
        output_stream.height = frame_height
        output_stream.pix_fmt = "yuv420p"
        output_stream.options = {"preset": "ultrafast", "crf": "23"}
    except Exception:
        video_capture.release()
        raise

    model_name, model_class = MODEL_CONFIGS[version]
    logger.info(f"Loading RT-DETR v{version}: {model_name}")
    logger.info(
        f"Input video: {frame_width}x{frame_height}, {source_fps:.2f} FPS, "
        f"{frame_count if frame_count > 0 else 'unknown'} frames"
    )
    logger.info(f"Processing at most the first {MAX_VIDEO_DURATION_SECONDS} seconds")

    image_processor = RTDetrImageProcessor.from_pretrained(model_name)
    torch_model = model_class.from_pretrained(model_name).eval()
    first_image = frame_to_image(first_frame)
    first_pixel_values = image_processor(images=first_image, return_tensors="pt").pixel_values
    _, _, input_height, input_width = first_pixel_values.shape

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model.model,
        custom_preprocessor=custom_preprocessor,
    )

    logger.info(f"Opening TT device {device_id}")
    device = ttnn.open_device(
        device_id=device_id,
        l1_small_size=16384,
        trace_region_size=1 << 26,
    )
    trace_id = None
    trace_outputs = None
    frame_latencies = []
    frames_written = 0

    try:
        device.enable_program_cache()
        tt_model = TtRTDetrModel(
            config=torch_model.config,
            parameters=parameters,
            device=device,
            dtype=ttnn.bfloat16,
            input_height=input_height,
            input_width=input_width,
        )
        tt_pixel_values = ttnn.from_torch(
            first_pixel_values,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        logger.info("Running model warmup and capturing device trace")
        warmup_outputs = tt_model(tt_pixel_values)
        ttnn.synchronize_device(device)
        del warmup_outputs

        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        trace_outputs = tt_model(tt_pixel_values)
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        ttnn.synchronize_device(device)

        def process_frame(frame: np.ndarray) -> None:
            nonlocal frames_written

            image = frame_to_image(frame)
            pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
            target_sizes = torch.tensor([(image.height, image.width)])
            host_input = ttnn.from_torch(
                pixel_values,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(host_input, tt_pixel_values, cq_id=0)
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)

            outputs = SimpleNamespace(
                logits=ttnn.to_torch(trace_outputs[4]).float(),
                pred_boxes=ttnn.to_torch(trace_outputs[5]).float(),
            )
            detections = image_processor.post_process_object_detection(
                outputs,
                threshold=threshold,
                target_sizes=target_sizes,
            )[0]
            annotated_image = draw_detections(image, detections, torch_model.config.id2label)
            output_frame = av.VideoFrame.from_ndarray(image_to_frame(annotated_image), format="bgr24")
            for packet in output_stream.encode(output_frame):
                output_container.mux(packet)
            frames_written += 1
            if frames_written % progress_interval == 0:
                logger.info(f"Processed {frames_written}/{maximum_frames} frames")

        process_frame(first_frame)

        while frames_written < maximum_frames:
            frame_start = time.perf_counter()
            success, frame = video_capture.read()
            if not success:
                break

            process_frame(frame)
            frame_latencies.append(time.perf_counter() - frame_start)
    finally:
        if trace_id is not None:
            ttnn.release_trace(device, trace_id)
        for packet in output_stream.encode():
            output_container.mux(packet)
        output_container.close()
        video_capture.release()
        ttnn.close_device(device)

    if frame_latencies:
        frame_fps = [1.0 / latency for latency in frame_latencies]
        logger.info(f"Frames written: {frames_written}")
        logger.info(f"Mean FPS: {statistics.mean(frame_fps):.2f}")
        logger.info(f"Minimum FPS: {min(frame_fps):.2f}")
        logger.info(f"Maximum FPS: {max(frame_fps):.2f}")
    else:
        logger.info(f"Frames written: {frames_written}; no frames remained after the setup frame for FPS measurement")

    logger.info(f"Annotated video saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run TTNN RT-DETR inference on images or video")
    parser.add_argument("--image", action="append", dest="images", help="Input image path or URL")
    parser.add_argument("--video", help="Local MP4, MOV, MKV, or AVI file, or a direct video URL")
    parser.add_argument("--sample_image", type=int, default=0, help="Sample input image number from 0-9")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--device-id", type=int, default=0, help="TT device ID")
    parser.add_argument(
        "--model_version",
        type=int,
        choices=(1, 2),
        default=1,
        help="RT-DETR model version",
    )

    args = parser.parse_args()

    if args.video is not None:
        run_video_demo(
            source=args.video,
            threshold=args.threshold,
            device_id=args.device_id,
            version=args.model_version,
        )
    else:
        run_demo(
            image_sources=args.images or [SAMPLE_TEST_IMAGES[args.sample_image]],
            threshold=args.threshold,
            device_id=args.device_id,
            version=args.model_version,
        )


if __name__ == "__main__":
    main()
