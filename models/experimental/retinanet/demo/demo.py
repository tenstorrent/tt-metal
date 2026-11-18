# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from pathlib import Path
from typing import Any, List, Optional

import torch
import ttnn
from PIL import Image
from loguru import logger

import torchvision.transforms as transforms
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torch import Tensor
from models.experimental.retinanet.tt.tt_regression_head import ttnn_retinanet_regression_head
from models.experimental.retinanet.tt.tt_classification_head import ttnn_retinanet_classification_head
from models.experimental.retinanet.tt.tt_backbone import TTBackbone
from models.experimental.retinanet.tt.tt_postprocess import TTDetectionDecoder, select_detections
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.retinanet.tt.custom_preprocessor import (
    create_custom_mesh_preprocessor,
    preprocess_regression_head_parameters,
    preprocess_classification_head_parameters,
)

# The input size the model runs at, (H, W); every image is resized to it.
IMAGE_SIZE = (512, 512)

# torchvision's 91 COCO category slots, which the labels index; unused slots are N/A.
CATEGORY_NAMES = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"]

device = ttnn.open_device(device_id=0, l1_small_size=24576)


class Demo:
    """RetinaNet detection on one image: the TTNN model on device, its detections drawn on the image."""

    def __init__(self) -> None:
        self.ttnn_model: Optional[Any] = None
        self.ttnn_device: Optional[Any] = None

        # Mesh mappers for TTNN
        self.inputs_mesh_mapper = None
        self.weights_mesh_mapper = None
        self.output_mesh_composer = None

    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------

    def prepare_model(self, image_size, model_config):
        """Prepares the TT model's parameters and its detection decoder for ``image_size`` (H, W)
        from the pretrained torchvision RetinaNet, whose postprocessing settings the demo keeps."""
        retinanet = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
        retinanet.eval()

        parameters = preprocess_model_parameters(
            initialize_model=lambda: retinanet,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )
        self.backbone_parameters = parameters.get("backbone", parameters)
        self.regression_parameters = preprocess_regression_head_parameters(
            torch_head=retinanet.head.regression_head,
            device=device,
            mesh_mapper=self.weights_mesh_mapper,
            model_config=model_config,
        )
        self.classification_parameters = preprocess_classification_head_parameters(
            torch_head=retinanet.head.classification_head,
            device=device,
            mesh_mapper=self.weights_mesh_mapper,
            model_config=model_config,
        )

        self.decoder = TTDetectionDecoder(device, image_size, topk_candidates=retinanet.topk_candidates)
        self.postprocess_settings = {
            "score_thresh": retinanet.score_thresh,
            "nms_thresh": retinanet.nms_thresh,
            "detections_per_img": retinanet.detections_per_img,
        }

    def run_ttnn_inference(self, input_tensor, model_config, device):
        """Run the TTNN model on ``input_tensor`` (NHWC) and return the device candidates of every
        pyramid level: per-level lists of boxes, scores and labels for select_detections."""
        self.ttnn_model = TTBackbone(parameters=self.backbone_parameters, model_config=model_config)

        backbone_output = self.ttnn_model(input_tensor, device)
        fpn_features = [backbone_output[key] for key in ["0", "1", "2", "p6", "p7"]]
        input_shapes = [
            (backbone_output["0"].shape[1], backbone_output["0"].shape[2]),
            (backbone_output["1"].shape[1], backbone_output["1"].shape[2]),
            (backbone_output["2"].shape[1], backbone_output["2"].shape[2]),
            (backbone_output["p6"].shape[1], backbone_output["p6"].shape[2]),
            (backbone_output["p7"].shape[1], backbone_output["p7"].shape[2]),
        ]
        # Run regression head
        regression_output = ttnn_retinanet_regression_head(
            feature_maps=fpn_features,
            parameters=self.regression_parameters,
            device=device,
            in_channels=256,
            num_anchors=9,
            batch_size=1,
            input_shapes=input_shapes,
            model_config=model_config,
            optimization_profile="optimized",
        )
        logger.debug("Regression head completed")
        # Run classification head
        classification_output = ttnn_retinanet_classification_head(
            feature_maps=fpn_features,
            parameters=self.classification_parameters,
            device=device,
            in_channels=256,
            num_anchors=9,
            batch_size=1,
            input_shapes=input_shapes,
            model_config=model_config,
            optimization_profile="optimized",
        )

        return self.decoder(classification_output, regression_output)

    def preprocess_image(self, image_path: str, output_dir: str, target_size=IMAGE_SIZE):
        """The image resized to ``target_size`` and normalized, as (1, 3, H, W), and its original
        (H, W). Saves the resized image, denormalized, as restored.jpg in ``output_dir``."""
        image = Image.open(image_path).convert("RGB")
        # PIL sizes are (W, H); resize_boxes takes (H, W).
        og_size = [tuple(reversed(image.size))]

        # Resize image to the target size
        image = image.resize(target_size, Image.BILINEAR)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)

        denorm = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )

        img = image_tensor.squeeze(0)
        img = denorm(img)
        img = img.clamp(0, 1)

        to_pil = transforms.ToPILImage()
        restored_image = to_pil(img)
        output_path = os.path.join(output_dir, "restored.jpg")
        restored_image.save(output_path)
        logger.info(f"Saved restored image to {output_path}")

        return image_tensor, og_size

    def visualize_detections(self, image_path, detections, output_dir, threshold):
        from PIL import Image, ImageDraw, ImageFont
        import os

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        dets = detections[0]
        boxes = dets["boxes"]
        scores = dets["scores"]
        labels = dets["labels"]

        for box, score, label in zip(boxes, scores, labels):
            if score < threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

            class_name = CATEGORY_NAMES[label.item()]
            print(f"{class_name} : {score}")

            text = f"{class_name}: {score:.2f}"
            draw.text((x1 + 5, y1 + 5), text, fill="yellow", font=font)

        out_path = os.path.join(output_dir, "result.jpg")
        image.save(out_path)
        logger.info(f"Saved visualization to {out_path}")

    # ---------------------------------------------------------------------
    # Run demo
    # ---------------------------------------------------------------------
    def run_demo(self, image_path: str, output_dir: str, threshold: float) -> None:
        """Run the full demo pipeline end-to-end."""
        logger.info("Starting demo for image: {}", image_path)

        # Preprocess image
        self.torch_input, original_image_sizes = self.preprocess_image(image_path, output_dir)

        self.ttnn_input = ttnn.from_torch(self.torch_input.permute(0, 2, 3, 1), dtype=ttnn.bfloat16)
        self.ttnn_input = ttnn.to_device(self.ttnn_input, device)

        # Run inference
        model_config = {
            "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
            "WEIGHTS_DTYPE": ttnn.bfloat16,
            "ACTIVATIONS_DTYPE": ttnn.bfloat16,
        }

        self.prepare_model(IMAGE_SIZE, model_config)
        boxes, scores, labels = self.run_ttnn_inference(self.ttnn_input, model_config, device)
        detections = [select_detections(boxes, scores, labels, **self.postprocess_settings)]
        image_sizes = [tuple(self.torch_input.shape[-2:])]
        for i, (pred, im_s, o_im_s) in enumerate(zip(detections, image_sizes, original_image_sizes)):
            detections[i]["boxes"] = Demo.resize_boxes(pred["boxes"], im_s, o_im_s)

        logger.info(detections)
        self.visualize_detections(image_path, detections, output_dir, threshold)

        logger.info("Demo completed. Output dir: {}", output_dir)

    @staticmethod
    def resize_boxes(boxes: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
        ratios = [
            torch.tensor(s, dtype=torch.float32, device=boxes.device)
            / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
            for s, s_orig in zip(new_size, original_size)
        ]
        ratio_height, ratio_width = ratios
        xmin, ymin, xmax, ymax = boxes.unbind(1)

        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        return torch.stack((xmin, ymin, xmax, ymax), dim=1)

    def cleanup(self) -> None:
        """Release device resources."""
        if self.ttnn_device is not None:
            try:
                ttnn.close_device(self.ttnn_device)
                logger.info("TTNN device closed.")
            finally:
                self.ttnn_device = None


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TT RetinaNet Demo")
    parser.add_argument("--input", "-i", required=True, help="Path to input image")
    parser.add_argument(
        "--output",
        "-o",
        default="models/experimental/retinanet/resources/outputs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum score of a drawn detection; the model drops scores below 0.05 before drawing",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    # Validate input file
    if not args.input or not os.path.exists(args.input):
        logger.error("Input image not found: {}", args.input)
        return 1

    # Prepare output directory
    out_dir = args.output or "models/experimental/retinanet/resources/outputs"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    demo: Optional[Demo] = None

    logger.info("=== RetinaNet Demo ===")
    try:
        demo = Demo()
        demo.run_demo(args.input, out_dir, args.threshold)
        return 0
    except Exception as e:
        logger.exception("Demo failed: {}", e)
        return 1
    finally:
        if demo is not None:
            try:
                demo.cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
