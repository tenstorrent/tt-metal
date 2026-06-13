# ------------------------------------------------------------------------
# RF-DETR-base reference weight loading + preprocessing.
# ------------------------------------------------------------------------
"""Build the reference model, load the published checkpoint (strict), and
provide the matching image preprocessing."""

from __future__ import annotations

import json
import os

import torch
from safetensors.torch import load_file

from .configuration_rf_detr import RfDetrConfig
from .modeling_rf_detr import RfDetrForObjectDetection

HF_SNAPSHOT = (
    "/home/ttuser/.cache/huggingface/hub/models--Roboflow--rf-detr-base/"
    "snapshots/7b95b089788e6c7db56d5ea9b0a07ca08ea6ac0a"
)
WEIGHTS_PATH = os.path.join(HF_SNAPSHOT, "model.safetensors")
CONFIG_PATH = os.path.join(HF_SNAPSHOT, "config.json")


def _load_id2label(config_path: str = CONFIG_PATH) -> dict[int, str]:
    with open(config_path, "r") as f:
        cfg = json.load(f)
    return {int(k): v for k, v in cfg["id2label"].items()}


def load_rf_detr_base(
    weights_path: str = WEIGHTS_PATH,
    config_path: str = CONFIG_PATH,
    device: str = "cpu",
) -> tuple[RfDetrForObjectDetection, RfDetrConfig]:
    """Build RfDetrForObjectDetection, load weights strictly, return (model, config).

    Asserts zero missing and zero unexpected keys.
    """
    cfg = RfDetrConfig()
    cfg.id2label = _load_id2label(config_path)

    model = RfDetrForObjectDetection(cfg)
    model.eval()

    state_dict = load_file(weights_path)

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)

    print(f"[load_rf_detr_base] checkpoint tensors: {len(ckpt_keys)}")
    print(f"[load_rf_detr_base] model tensors:      {len(model_keys)}")
    print(f"[load_rf_detr_base] missing keys:       {len(missing)}")
    print(f"[load_rf_detr_base] unexpected keys:    {len(unexpected)}")
    if missing:
        print("  MISSING (first 20):")
        for k in missing[:20]:
            print("   ", k)
    if unexpected:
        print("  UNEXPECTED (first 20):")
        for k in unexpected[:20]:
            print("   ", k)

    result = model.load_state_dict(state_dict, strict=True)
    assert len(result.missing_keys) == 0, f"missing keys: {result.missing_keys[:20]}"
    assert len(result.unexpected_keys) == 0, f"unexpected keys: {result.unexpected_keys[:20]}"
    print("[load_rf_detr_base] strict load OK (0 missing, 0 unexpected)")

    model.to(device)
    model.eval()
    return model, cfg


# ----------------------------------------------------------------------------
# Preprocessing: resize 560x560 (bilinear), rescale 1/255, ImageNet normalize.
# Matches RfDetrImageProcessor (do_resize, do_rescale, do_normalize).
# ----------------------------------------------------------------------------
class RfDetrPreprocessor:
    def __init__(self, cfg: RfDetrConfig):
        self.size = cfg.image_resolution
        self.mean = torch.tensor(cfg.image_mean).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg.image_std).view(1, 3, 1, 1)
        self.rescale_factor = cfg.rescale_factor

    def __call__(self, image) -> torch.Tensor:
        return self.preprocess(image)

    def preprocess(self, image) -> torch.Tensor:
        """PIL.Image (RGB) -> pixel_values [1,3,size,size] float32."""
        import torch.nn.functional as F

        if image.mode != "RGB":
            image = image.convert("RGB")
        # HWC uint8 -> CHW float
        import numpy as np

        arr = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0).float()
        # resize to (size, size) bilinear, align_corners=False (PIL BILINEAR equivalent
        # for the torchvision "use_fast" path the RfDetrImageProcessor uses).
        arr = F.interpolate(
            arr, size=(self.size, self.size), mode="bilinear", align_corners=False, antialias=True
        )
        arr = arr * self.rescale_factor
        arr = (arr - self.mean) / self.std
        return arr


def get_preprocessor(cfg: RfDetrConfig) -> RfDetrPreprocessor:
    return RfDetrPreprocessor(cfg)
