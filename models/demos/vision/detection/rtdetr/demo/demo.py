import argparse
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import requests
import torch
from loguru import logger
from PIL import Image, ImageDraw
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.vision.detection.rtdetr.common.preprocessing import custom_preprocessor
from models.demos.vision.detection.rtdetr.tt.model import TtRTDetrModel

MODEL_NAME = "PekingU/rtdetr_r50vd"

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


def run_demo(image_sources: list[str], threshold: float, device_id: int) -> Path:
    if len(image_sources) != 1:
        raise ValueError("The current demo supports batch size 1")

    output_directory = create_output_directory()
    images = [load_image(source) for source in image_sources]

    logger.info(f"Loading {MODEL_NAME}")
    image_processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    torch_model = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).eval()
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


def main():
    parser = argparse.ArgumentParser(description="Run Torch and TTNN RT-DETR inference and save annotated images")
    parser.add_argument("--image", action="append", dest="images", help="Input image path or URL")
    parser.add_argument("--sample_image", type=int, default=0, help="Sample input image number from 0-9")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--device-id", type=int, default=0, help="TT device ID")

    args = parser.parse_args()

    run_demo(
        image_sources=args.images or [SAMPLE_TEST_IMAGES[args.sample_image]],
        threshold=args.threshold,
        device_id=args.device_id,
    )


if __name__ == "__main__":
    main()
