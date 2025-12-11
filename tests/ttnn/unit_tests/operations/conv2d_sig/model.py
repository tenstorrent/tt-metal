import math
import os.path

import os
import shutil
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import backbone as backbones_mod
import requests


def download_nvidia_model(variant_name):
    """Download and extract NVIDIA RetinaNet model."""
    url = f"https://github.com/NVIDIA/retinanet-examples/releases/download/19.04/{variant_name}.zip"
    local_zip_path = f"{variant_name}.zip"

    # Download the model
    response = requests.get(url)
    with open(local_zip_path, "wb") as f:
        f.write(response.content)

    # Extract the zip file
    extracted_path = variant_name
    with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
        zip_ref.extractall(".")

    # Find the .pth file
    checkpoint_path = ""
    for root, _, files in os.walk(extracted_path):
        for file in files:
            if file.endswith(".pth"):
                checkpoint_path = os.path.join(root, file)
                break

    return checkpoint_path


class FocalLoss(nn.Module):
    "Focal Loss - https://arxiv.org/abs/1708.02002"

    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits, target):
        pred = pred_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
        alpha = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        pt = torch.where(target == 1, pred, 1 - pred)
        return alpha * (1.0 - pt) ** self.gamma * ce


class SmoothL1Loss(nn.Module):
    "Smooth L1 Loss"

    def __init__(self, beta=0.11):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        x = (pred - target).abs()
        l1 = x - 0.5 * self.beta
        l2 = 0.5 * x**2 / self.beta
        return torch.where(x >= self.beta, l1, l2)


class Model(nn.Module):
    "RetinaNet - https://arxiv.org/abs/1708.02002"

    def __init__(
        self,
        backbones="ResNet50FPN",
        classes=80,
        ratios=[1.0, 2.0, 0.5],
        scales=[4 * 2 ** (i / 3) for i in range(3)],
        angles=None,
        rotated_bbox=False,
        anchor_ious=[0.4, 0.5],
        config={},
    ):
        super().__init__()

        if not isinstance(backbones, list):
            backbones = [backbones]

        self.backbones = nn.ModuleDict({b: getattr(backbones_mod, b)() for b in backbones})
        self.name = "RetinaNet"
        self.unused_modules = []
        for b in backbones:
            self.unused_modules.extend(getattr(self.backbones, b).features.unused_modules)
        self.exporting = False
        self.rotated_bbox = rotated_bbox
        self.anchor_ious = anchor_ious

        self.ratios = ratios
        self.scales = scales
        self.angles = angles if angles is not None else [-np.pi / 6, 0, np.pi / 6] if self.rotated_bbox else None
        self.anchors = {}
        self.classes = classes

        self.threshold = config.get("threshold", 0.05)
        self.top_n = config.get("top_n", 1000)
        self.nms = config.get("nms", 0.5)
        self.detections = config.get("detections", 100)

        self.stride = max([b.stride for _, b in self.backbones.items()])

        # classification and box regression heads
        def make_head(out_size):
            layers = []
            for _ in range(4):
                layers += [nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()]
            layers += [nn.Conv2d(256, out_size, 3, padding=1)]
            return nn.Sequential(*layers)

        self.num_anchors = len(self.ratios) * len(self.scales)
        self.num_anchors = self.num_anchors if not self.rotated_bbox else (self.num_anchors * len(self.angles))
        self.cls_head = make_head(classes * self.num_anchors)
        self.box_head = (
            make_head(4 * self.num_anchors) if not self.rotated_bbox else make_head(6 * self.num_anchors)
        )  # theta -> cos(theta), sin(theta)

        self.cls_criterion = FocalLoss()
        self.box_criterion = SmoothL1Loss(beta=0.11)

    def __repr__(self):
        return "\n".join(
            [
                "     model: {}".format(self.name),
                "  backbone: {}".format(", ".join([k for k, _ in self.backbones.items()])),
                "   classes: {}, anchors: {}".format(self.classes, self.num_anchors),
            ]
        )

    def initialize(self, pre_trained):
        if pre_trained:
            # Initialize using weights from pre-trained model
            if not os.path.isfile(pre_trained):
                raise ValueError("No checkpoint {}".format(pre_trained))

            print("Fine-tuning weights from {}...".format(os.path.basename(pre_trained)))
            state_dict = self.state_dict()
            chk = torch.load(pre_trained, map_location=lambda storage, loc: storage)
            ignored = ["cls_head.8.bias", "cls_head.8.weight"]
            if self.rotated_bbox:
                ignored += ["box_head.8.bias", "box_head.8.weight"]
            weights = {k: v for k, v in chk["state_dict"].items() if k not in ignored}
            state_dict.update(weights)
            self.load_state_dict(state_dict)

            del chk, weights
            torch.cuda.empty_cache()

        else:
            # Initialize backbone(s)
            for _, backbone in self.backbones.items():
                backbone.initialize()

            # Initialize heads
            def initialize_layer(layer):
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)

            self.cls_head.apply(initialize_layer)
            self.box_head.apply(initialize_layer)

        # Initialize class head prior
        def initialize_prior(layer):
            pi = 0.01
            b = -math.log((1 - pi) / pi)
            nn.init.constant_(layer.bias, b)
            nn.init.normal_(layer.weight, std=0.01)

        self.cls_head[-1].apply(initialize_prior)
        if self.rotated_bbox:
            self.box_head[-1].apply(initialize_prior)

    def forward(self, x):
        # Backbones forward pass
        features = []
        for _, backbone in self.backbones.items():
            features.extend(backbone(x))

        # Heads forward pass
        cls_heads = [self.cls_head(t) for t in features]
        box_heads = [self.box_head(t) for t in features]

        cls_heads = [cls_head.sigmoid() for cls_head in cls_heads]

        combined_heads = cls_heads + box_heads

        return combined_heads

    @classmethod
    def load(cls, filename, rotated_bbox=False):
        if not os.path.isfile(filename):
            raise ValueError("No checkpoint {}".format(filename))

        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
        kwargs = {}
        for i in ["ratios", "scales", "angles"]:
            if i in checkpoint:
                kwargs[i] = checkpoint[i]
        if ("angles" in checkpoint) or rotated_bbox:
            kwargs["rotated_bbox"] = True
        # Recreate model from checkpoint instead of from individual backbones
        model = cls(backbones=checkpoint["backbone"], classes=checkpoint["classes"], **kwargs)
        model.load_state_dict(checkpoint["state_dict"])

        return model
