# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from pathlib import Path
from typing import Any, Optional

import torch
import ttnn
from PIL import Image
from loguru import logger

from PIL import Image
import torchvision.transforms as transforms
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.image_list import ImageList
from typing import Any, Dict, List, Optional
from torch import Tensor
from models.experimental.retinanet.tt.tt_regression_head import ttnn_retinanet_regression_head
from models.experimental.retinanet.tt.tt_classification_head import ttnn_retinanet_classification_head
from models.experimental.retinanet.tt.tt_backbone import TTBackbone
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.retinanet.tt.custom_preprocessor import (
    create_custom_mesh_preprocessor,
    preprocess_regression_head_parameters,
    preprocess_classification_head_parameters,
)

# LUT
COCO_INSTANCE_CATEGORY_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "sheep",
    "horse",
    "dog",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sport ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wineglass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hotdog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

device = ttnn.open_device(device_id=0, l1_small_size=24576)


class Demo:
    """Panoptic-DeepLab demo supporting both PyTorch and TTNN pipelines."""

    def __init__(self) -> None:
        self.torch_model: Optional[Any] = None
        self.ttnn_model: Optional[Any] = None
        self.ttnn_device: Optional[Any] = None

        # Mesh mappers for TTNN
        self.inputs_mesh_mapper = None
        self.weights_mesh_mapper = None
        self.output_mesh_composer = None

    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------

    def run_torch_inference(self, input_tensor, model_config):
        """Run PyTorch inference."""
        retinanet = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
        retinanet.eval()
        self.torch_backbone = retinanet.backbone
        self.torch_regression_head = retinanet.head.regression_head
        self.torch_classification_head = retinanet.head.classification_head

        self.parameters = preprocess_model_parameters(
            initialize_model=lambda: retinanet,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )
        self.backbone_parameters = self.parameters.get("backbone", self.parameters)

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

        backbone_features = self.torch_backbone(input_tensor)
        torch_regression_output = self.torch_regression_head(list(backbone_features.values()))
        torch_classification_output = self.torch_classification_head(list(backbone_features.values()))

        output = {
            "backbone_features": backbone_features,  # FPN levels: "0", "1", "2", "p6", "p7"
            "regression": torch_regression_output,
            "classification": torch_classification_output,
        }

        return output

    def run_ttnn_inference(self, input_tensor, model_config, device):
        """Run TTNN inference."""
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
        logger.debug("✅✅✅ REGRESSION HEAD Complete ✅✅✅")
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

        for key in backbone_output:
            backbone_output[key] = ttnn.to_torch(backbone_output[key], dtype=torch.float32).permute((0, 3, 1, 2))

        regression_output = ttnn.to_torch(regression_output, dtype=torch.float32)
        classification_output = ttnn.to_torch(classification_output, dtype=torch.float32)

        output = {
            "backbone_features": backbone_output,  # FPN levels: "0", "1", "2", "p6", "p7"
            "regression": regression_output,
            "classification": classification_output,
        }

        return output

    def preprocess_image(self, image_path: str, device, target_size=(800, 800)):
        """PREPROCESS IMAGE"""
        image = Image.open(image_path).convert("RGB")
        og_size = [tuple(image.size)]

        # Resize image to the target size
        image = image.resize(target_size, Image.BILINEAR)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert image to tensor (range [0, 1])
                normalize,  # Normalize with ImageNet stats
            ]
        )

        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)

        denorm = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )

        # Remove batch dimension if it exists
        img = image_tensor.squeeze(0)

        # Denormalize
        img = denorm(img)

        # Clamp to [0,1] just to avoid out-of-range values after denorm
        img = img.clamp(0, 1)

        # Convert to PIL
        to_pil = transforms.ToPILImage()
        restored_image = to_pil(img)

        # Save
        restored_image.save("models/experimental/retinanet/resources/outputs/restored.jpg")
        print("Saved restored image to restored.jpg")

        return image_tensor, og_size

    def decode_boxes(anchors, deltas):
        """
        Decode regression deltas to boxes.
        anchors: [N, 4] (x1, y1, x2, y2)
        deltas:  [N, 4] (dx, dy, dw, dh)
        """
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights

        dx, dy, dw, dh = deltas.unbind(dim=1)
        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h

        return torch.stack((x1, y1, x2, y2), dim=1)

    def postprocess_detections(self, head_outputs, anchors, image_shapes):
        # type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]

        num_images = len(image_shapes)

        detections: List[Dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            anchors_per_image, image_shape = anchors[index], image_shapes[index]

            image_boxes = []
            image_scores = []
            image_labels = []

            for box_regression_per_level, logits_per_level, anchors_per_level in zip(
                box_regression_per_image, logits_per_image, anchors_per_image
            ):
                num_classes = logits_per_level.shape[-1]

                # remove low scoring boxes
                scores_per_level = torch.sigmoid(logits_per_level).flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                num_topk = det_utils._topk_min(topk_idxs, self.topk_candidates, 0)
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode_single(
                    box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
                )
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )

        return detections

    def visualize_detections(self, image_path, detections, output_dir):
        from PIL import Image, ImageDraw, ImageFont
        import os

        label_map = {i: name for i, name in enumerate(COCO_INSTANCE_CATEGORY_NAMES)}

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        dets = detections[0]
        boxes = dets["boxes"]
        scores = dets["scores"]
        labels = dets["labels"]

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

            if label_map:
                class_name = label_map.get(label.item(), str(label.item()))
                print(f"{class_name} : {score}")
            else:
                class_name = str(label.item())

            text = f"{class_name}: {score:.2f}"
            draw.text((x1 + 5, y1 + 5), text, fill="yellow", font=font)

        out_path = os.path.join(output_dir, "result.jpg")
        image.save(out_path)
        logger.info(f"Saved visualization to {out_path}")

    # ---------------------------------------------------------------------
    # Run demo
    # ---------------------------------------------------------------------
    def run_demo(self, image_path: str, output_dir: str) -> None:
        """Run the full demo pipeline end-to-end."""
        logger.info("Starting demo for image: {}", image_path)

        # Preprocess image
        self.torch_input, original_image_sizes = self.preprocess_image(
            image_path, self.ttnn_device, target_size=(512, 512)
        )

        self.ttnn_input = ttnn.from_torch(self.torch_input.permute(0, 2, 3, 1), dtype=ttnn.bfloat16)
        self.ttnn_input = ttnn.to_device(self.ttnn_input, device)

        # Run inference
        model_config = {
            "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
            "WEIGHTS_DTYPE": ttnn.bfloat16,
            "ACTIVATIONS_DTYPE": ttnn.bfloat16,
        }

        torch_output = self.run_torch_inference(self.torch_input, model_config)
        torch_output = self.run_ttnn_inference(self.ttnn_input, model_config, device)
        # Posstprocess detections
        anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        self.score_thresh = 0.05
        self.nms_thresh = 0.5
        self.detections_per_img = 1
        fg_iou_thresh = 0.5
        bg_iou_thresh = 0.4
        self.topk_candidates = 1000

        proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        image_list = ImageList(self.torch_input, [(self.torch_input.shape[-2], self.torch_input.shape[-1])])
        features = list(torch_output["backbone_features"].values())
        anchors = anchor_generator(image_list, features)

        detections: List[Dict[str, Tensor]] = []

        num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
        HW = 0
        for v in num_anchors_per_level:
            HW += v
        HWA = torch_output["classification"].size(1)
        A = HWA // HW
        num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

        # split outputs per level
        split_head_outputs: Dict[str, List[Tensor]] = {}
        for k, y in zip(["classification", "regression"], ["cls_logits", "bbox_regression"]):
            split_head_outputs[y] = list(torch_output[k].split(num_anchors_per_level, dim=1))
        split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

        # Postprocess to comparable outputs
        detections = self.postprocess_detections(split_head_outputs, split_anchors, image_list.image_sizes)
        # original_image_sizes = [(900, 600)]
        print(f"image_list.image_sizes: {image_list.image_sizes}, original_image_sizes: {original_image_sizes}")
        for i, (pred, im_s, o_im_s) in enumerate(zip(detections, image_list.image_sizes, original_image_sizes)):
            print(f"im_s:{im_s}, o_im_s:{o_im_s}")
            boxes = pred["boxes"]
            boxes = Demo.resize_boxes(boxes, im_s, o_im_s)
            detections[i]["boxes"] = boxes

        print(detections)
        self.visualize_detections(image_path, detections, output_dir)

        logger.info("Demo completed. Output dir: {}", output_dir)

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
        demo.run_demo(args.input, out_dir)
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
