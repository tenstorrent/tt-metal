# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import ttnn
from loguru import logger
from torch import Tensor
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList

from models.experimental.retinanet.demo.processing import preprocess_image, postprocess_detections, visualize_detections
from models.experimental.retinanet.tests.pcc.test_resnet50_fpn import infer_ttnn_module_args as infer_module_args
from models.experimental.retinanet.tt.custom_preprocessor import create_custom_mesh_preprocessor
from models.experimental.retinanet.tt.tt_retinanet import TTRetinaNet
from ttnn.model_preprocessing import infer_ttnn_module_args, preprocess_model_parameters

device = ttnn.open_device(device_id=0, l1_small_size=24576)


class RetinaNetDemo:
    """RetinaNet demo for TTNN inference."""

    def __init__(self) -> None:
        self.ttnn_model: Optional[Any] = None
        self.ttnn_device: Optional[Any] = None
        self.parameters = None
        self.model_args = None
        self.weights_mesh_mapper = None
        self.torch_backbone = None
        self.torch_regression_head = None
        self.torch_classification_head = None

    def setup_model(self, input_tensor: Tensor, model_config: Dict[str, Any]) -> None:
        """Setup model parameters and configurations."""
        retinanet = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
        retinanet.eval()

        self.torch_backbone = retinanet.backbone
        self.torch_regression_head = retinanet.head.regression_head
        self.torch_classification_head = retinanet.head.classification_head

        backbone = retinanet.backbone.body
        fpn_model = retinanet.backbone.fpn
        torch_fpn_input_tensor = backbone(input_tensor)

        conv_args = infer_ttnn_module_args(
            model=self.torch_backbone,
            run_model=lambda model: self.torch_backbone(input_tensor),
            device=device,
        )
        fpn_args = infer_module_args(
            model=fpn_model,
            run_model=lambda model: fpn_model(torch_fpn_input_tensor),
            device=device,
        )

        self.model_args = {
            "stem": {
                "conv1": conv_args["body"]["conv1"],
                "maxpool": conv_args["body"]["maxpool"],
            },
            "fpn": {
                "inner_blocks": {
                    0: fpn_args["fpn"]["fpn"]["inner_blocks"][0]["fpn"]["inner_blocks"][0][0],
                    1: fpn_args["fpn"]["fpn"]["inner_blocks"][1]["fpn"]["inner_blocks"][1][0],
                    2: fpn_args["fpn"]["fpn"]["inner_blocks"][2]["fpn"]["inner_blocks"][2][0],
                },
                "layer_blocks": {
                    0: fpn_args["fpn"]["fpn"]["layer_blocks"][0]["fpn"]["layer_blocks"][0][0],
                    1: fpn_args["fpn"]["fpn"]["layer_blocks"][1]["fpn"]["layer_blocks"][1][0],
                    2: fpn_args["fpn"]["fpn"]["layer_blocks"][2]["fpn"]["layer_blocks"][2][0],
                },
                "extra_blocks": {
                    "p6": fpn_args["fpn"]["fpn"]["extra_blocks"]["fpn"]["extra_blocks"]["p6"],
                    "p7": fpn_args["fpn"]["fpn"]["extra_blocks"]["fpn"]["extra_blocks"]["p7"],
                },
            },
            "layer1": conv_args["body"]["layer1"],
            "layer2": conv_args["body"]["layer2"],
            "layer3": conv_args["body"]["layer3"],
            "layer4": conv_args["body"]["layer4"],
        }

        self.parameters = preprocess_model_parameters(
            initialize_model=lambda: retinanet,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

    def run_torch_inference(self, input_tensor: Tensor) -> Dict[str, Any]:
        """Run PyTorch inference for reference."""
        if self.torch_backbone is None:
            raise RuntimeError("Model not setup. Call setup_model() first.")

        backbone_features = self.torch_backbone(input_tensor)
        torch_regression_output = self.torch_regression_head(list(backbone_features.values()))
        torch_classification_output = self.torch_classification_head(list(backbone_features.values()))

        return {
            "backbone_features": backbone_features,
            "regression": torch_regression_output,
            "classification": torch_classification_output,
        }

    def run_inference(self, input_tensor: Tensor, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run TTNN inference."""
        if self.ttnn_model is None:
            self.ttnn_model = TTRetinaNet(
                parameters=self.parameters,
                model_config=model_config,
                device=device,
                model_args=self.model_args,
            )

        output_dict = self.ttnn_model(input_tensor, device)

        backbone_output = {key: output_dict[key] for key in ["0", "1", "2", "p6", "p7"]}

        for key in backbone_output:
            raw_tensor = ttnn.to_torch(backbone_output[key], dtype=torch.float32)
            permuted = raw_tensor.permute((0, 3, 1, 2))
            if permuted.shape[2] == 1:
                size = int(permuted.shape[3] ** 0.5)
                permuted = permuted.reshape(permuted.shape[0], permuted.shape[1], size, size)
            backbone_output[key] = permuted

        regression_output = ttnn.to_torch(output_dict["regression"], dtype=torch.float32)
        classification_output = ttnn.to_torch(output_dict["classification"], dtype=torch.float32)

        return {
            "backbone_features": backbone_output,
            "regression": regression_output,
            "classification": classification_output,
        }

    def run_demo(self, image_path: str, output_dir: str) -> None:
        """Run the full demo pipeline."""
        logger.info("Starting RetinaNet demo for image: {}", image_path)

        torch_input, _ = preprocess_image(image_path, target_size=(512, 512))

        model_config = {
            "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
            "WEIGHTS_DTYPE": ttnn.bfloat16,
            "ACTIVATIONS_DTYPE": ttnn.bfloat16,
        }

        logger.info("Setting up model...")
        self.setup_model(torch_input, model_config)

        ttnn_input = ttnn.from_torch(torch_input.permute(0, 2, 3, 1), dtype=ttnn.bfloat16)
        ttnn_input = ttnn.to_device(ttnn_input, device)

        logger.info("Running inference...")
        output = self.run_inference(ttnn_input, model_config)

        anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        score_thresh = 0.05
        nms_thresh = 0.5
        detections_per_img = 1
        topk_candidates = 1000
        box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        image_list = ImageList(torch_input, [(torch_input.shape[-2], torch_input.shape[-1])])
        features = list(output["backbone_features"].values())
        anchors = anchor_generator(image_list, features)

        num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
        HW = sum(num_anchors_per_level)
        HWA = output["classification"].size(1)
        A = HWA // HW
        num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

        split_head_outputs = {}
        for k, y in zip(["classification", "regression"], ["cls_logits", "bbox_regression"]):
            split_head_outputs[y] = [t.float() for t in output[k].split(num_anchors_per_level, dim=1)]
        split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

        logger.info("Postprocessing detections...")
        detections = postprocess_detections(
            split_head_outputs,
            split_anchors,
            image_list.image_sizes,
            score_thresh,
            nms_thresh,
            detections_per_img,
            topk_candidates,
            box_coder,
        )

        output_path = os.path.join(output_dir, "result.jpg")
        visualize_detections(image_path, detections, output_path, target_size=(512, 512))

        logger.info("Demo completed. Output saved to: {}", output_path)

    def cleanup(self) -> None:
        """Release device resources."""
        if self.ttnn_device is not None:
            try:
                ttnn.close_device(self.ttnn_device)
                logger.info("TTNN device closed.")
            finally:
                self.ttnn_device = None


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TT RetinaNet Demo")
    parser.add_argument(
        "--input",
        "-i",
        default="models/experimental/retinanet/resources/dog_800x800.jpg",
        help="Path to input image",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="models/experimental/retinanet/resources/outputs",
        help="Output directory for results",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the demo."""
    args = _parse_args(argv)

    if not args.input or not os.path.exists(args.input):
        logger.error("Input image not found: {}", args.input)
        return 1

    out_dir = args.output or "models/experimental/retinanet/resources/outputs"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    demo: Optional[RetinaNetDemo] = None

    logger.info("=== RetinaNet Demo ===")
    try:
        demo = RetinaNetDemo()
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
