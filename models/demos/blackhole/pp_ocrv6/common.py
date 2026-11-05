# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import os

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

DET_REPO = "PaddlePaddle/PP-OCRv6_small_det_safetensors"
REC_REPO = "PaddlePaddle/PP-OCRv6_small_rec_safetensors"
DET_L1_SMALL_SIZE = 32768

DET_MEAN_BGR = (0.406, 0.456, 0.485)
DET_STD_BGR = (0.225, 0.224, 0.229)


def _snapshot(repo):
    local = os.environ.get("PPOCRV6_WEIGHTS_DIR")
    if local:
        return os.path.join(local, repo.split("/")[-1])
    return snapshot_download(repo)


def _load(model, repo):
    state = load_file(os.path.join(_snapshot(repo), "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    assert not missing and not unexpected, (missing, unexpected)
    return model.eval()


def load_torch_det():
    from models.demos.blackhole.pp_ocrv6.reference.det import PPOCRv6SmallDet

    return _load(PPOCRv6SmallDet(), DET_REPO)


def load_torch_rec():
    from models.demos.blackhole.pp_ocrv6.reference.rec import PPOCRv6SmallRec

    return _load(PPOCRv6SmallRec(), REC_REPO)


def load_rec_charset():
    with open(os.path.join(_snapshot(REC_REPO), "preprocessor_config.json")) as f:
        return json.load(f)["character_list"]


def synthetic_gate_image(text="TRLU 482913", size=(1280, 720)):
    """BGR uint8 image of a trailer number on a plate; stand-in until real gate images are available."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", size, (90, 95, 100))
    d = ImageDraw.Draw(img)
    d.rectangle([300, 250, 980, 470], fill=(235, 235, 230))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 96)
    d.text((340, 300), text, fill=(20, 20, 20), font=font)
    return np.ascontiguousarray(np.array(img)[:, :, ::-1])


REC_HEIGHT = 48
REC_WIDTH = 320


def rec_normalize_bgr(bgr_uint8_hwc, height=REC_HEIGHT, width=REC_WIDTH):
    """Resize a text-line crop to `height`, keep aspect up to `width`, right-pad with zeros: [1, 3, H, W] in [-1, 1]."""
    import cv2
    import numpy as np

    h, w = bgr_uint8_hwc.shape[:2]
    rw = min(width, int(np.ceil(height * w / h)))
    img = cv2.resize(bgr_uint8_hwc, (rw, height)).astype(np.float32) / 127.5 - 1.0
    out = np.zeros((height, width, 3), np.float32)
    out[:, :rw] = img
    return torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0).contiguous()


def ctc_greedy_decode(probs, charset):
    """probs [T, C] -> (text, mean confidence). Index 0 is the CTC blank."""
    conf, idx = probs.max(-1)
    chars, scores, prev = [], [], 0
    for i, c in zip(idx.tolist(), conf.tolist()):
        if i != prev and i != 0:
            chars.append(charset[i])
            scores.append(c)
        prev = i
    return "".join(chars), (sum(scores) / len(scores) if scores else 0.0)


def det_normalize_bgr(bgr_uint8_hwc):
    x = torch.as_tensor(bgr_uint8_hwc, dtype=torch.float32) / 255.0
    x = (x - torch.tensor(DET_MEAN_BGR)) / torch.tensor(DET_STD_BGR)
    return x.permute(2, 0, 1).unsqueeze(0).contiguous()
