# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CPU reference implementation for ED-Pose inference.

Loads the official ED-Pose checkpoint (with CUDA deformable attention replaced
by pure PyTorch grid_sample fallback) and runs inference on CPU.
Used as ground truth for validating TT-Metalium implementations.

Requires:
  - ED-Pose repo cloned at EDPOSE_ROOT (default: ~/ttwork/ED-Pose)
  - Patched ms_deform_attn_func.py (CUDA → PyTorch fallback)
  - Swin-L 5scale checkpoint at EDPOSE_ROOT/weights/edpose_swinl_5scale_coco.pth
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

EDPOSE_ROOT = os.environ.get("EDPOSE_ROOT", str(Path.home() / "ttwork" / "ED-Pose"))


def _ensure_edpose_on_path():
    if EDPOSE_ROOT not in sys.path:
        sys.path.insert(0, EDPOSE_ROOT)


def load_edpose_model(
    checkpoint_path=None,
    config_path=None,
    device="cpu",
):
    """Build ED-Pose model and load Swin-L 5scale checkpoint.

    Returns:
        model: nn.Module in eval mode
        postprocessor: PostProcess module for decoding outputs
    """
    _ensure_edpose_on_path()

    from util.config import Config
    from models.edpose.edpose import build_edpose

    if config_path is None:
        config_path = os.path.join(EDPOSE_ROOT, "config", "edpose.cfg.py")
    if checkpoint_path is None:
        checkpoint_path = os.path.join(EDPOSE_ROOT, "weights", "edpose_swinl_5scale_coco.pth")

    cfg = Config.fromfile(config_path)
    cfg.backbone = "swin_L_384_22k"
    cfg.return_interm_indices = [0, 1, 2, 3]
    cfg.num_feature_levels = 5
    cfg.batch_size = 1
    cfg.epochs = 60
    cfg.lr_drop = 55
    cfg.num_body_points = 17
    cfg.device = device
    cfg.use_dn = False
    cfg.masks = False

    model, _, postprocessors = build_edpose(cfg)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_ema", ckpt.get("model", ckpt))
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)

    model.eval()
    model.to(device)
    return model, postprocessors["bbox"]


def preprocess_image(image_path, target_size=800, max_size=1333):
    """Preprocess image for ED-Pose inference.

    Returns:
        samples: NestedTensor (tensor + mask)
        target_sizes: tensor of [H, W] for post-processing
        original_size: (orig_W, orig_H)
    """
    _ensure_edpose_on_path()
    from util.misc import NestedTensor

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    scale = target_size / min(orig_w, orig_h)
    if max(orig_w, orig_h) * scale > max_size:
        scale = max_size / max(orig_w, orig_h)

    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = normalize(img)

    pad_h = (32 - new_h % 32) % 32
    pad_w = (32 - new_w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h))

    mask = torch.ones(tensor.shape[1], tensor.shape[2], dtype=torch.bool)
    mask[:new_h, :new_w] = False

    samples = NestedTensor(tensor.unsqueeze(0), mask.unsqueeze(0))
    target_sizes = torch.tensor([[new_h, new_w]])
    return samples, target_sizes, (orig_w, orig_h)


def make_synthetic_input(height=800, width=1216):
    """Create synthetic NestedTensor for testing without an image.

    Returns:
        samples: NestedTensor
        target_sizes: tensor
    """
    _ensure_edpose_on_path()
    from util.misc import NestedTensor

    tensor = torch.randn(1, 3, height, width)
    mask = torch.zeros(1, height, width, dtype=torch.bool)
    samples = NestedTensor(tensor, mask)
    target_sizes = torch.tensor([[height, width]])
    return samples, target_sizes


@torch.no_grad()
def run_inference(model, postprocessor, samples, target_sizes, score_threshold=0.3):
    """Run ED-Pose inference and return detected persons with keypoints.

    Returns:
        list of dicts, each with:
          - score: float
          - bbox: [x1, y1, x2, y2]
          - keypoints: tensor (17, 3) — x, y, visibility
    """
    outputs = model(samples)
    results = postprocessor(outputs, target_sizes)

    detections = []
    for res in results:
        scores = res["scores"]
        mask = scores > score_threshold
        for idx in torch.where(mask)[0]:
            detections.append({
                "score": scores[idx].item(),
                "bbox": res["boxes"][idx].tolist(),
                "keypoints": res["keypoints"][idx].reshape(-1, 3),
            })

    return detections, outputs
