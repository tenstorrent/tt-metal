# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Goldens for tests/test_vision_accuracy.py: Hugging Face's MiniMax-M3 vision tower and projector with the vision
weights of the checkpoint in HF_MODEL, run on the CPU in fp32 (the goldens) and in bf16 (the error of a plain bf16
run, which sets the test's tower bar).

Each image of IMAGES goes through the checkpoint's image processor, then MiniMaxM3VLModel.get_image_features.
<out>/<image>.safetensors holds
    pixel_values       (L, 1176) fp32      the processor's output, the test's input
    image_grid_thw     (1, 3) int64        the (t, h, w) patch grid
    tower_out          (L, 1280) fp32      the tower output (last_hidden_state)
    final              (L / 4, 6144) fp32  the vision tokens (pooler_output)
    pre_layrnorm_rows  (64, 1280) bf16     the first rows of the pre_layrnorm output, which the test recomputes from
                                           HF_MODEL's weights to check that HF_MODEL is this checkpoint
<out>/manifest.json names the checkpoint, the library versions, the CPU and the number of torch threads, and gives, per
image, its file and the PCC of the bf16 run against the fp32 one; <out>/SHA256SUMS checks a copy
(sha256sum -c SHA256SUMS).

Standalone (no test_ prefix, so pytest does not collect it). CPU only, no ttnn; needs transformers with
minimax_m3_vl (tt-metal pins 5.12.1). From the tt-metal root:
    HF_MODEL=<checkpoint> python -m models.demos.minimax_m3.tests.golden_hf_vision <out dir>
then set M3_VISION_GOLDEN_DIR=<out dir>, or copy <out dir> to $HF_MODEL/golden/vision.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file

from models.demos.minimax_m3.tt.vision.loader import load_vision_state_dict

# transformers' minimax_m3_vl classes are imported where they are used: they take seconds to import, and
# tests/test_vision_accuracy.py imports IMAGES from this module without needing them.

REPO = Path(__file__).resolve().parents[4]
LLAMA_PROMPTS = "models/tt_transformers/demo/sample_prompts/llama_models"
# The test's images: repo path, and the side of the top-left square kept (None: the whole image). Each makes 2304
# patches (576 vision tokens). On the device, pasta and perf_report (48x48) had the lowest tower and the lowest final
# PCC of six real images at 672x672; ocr_image (48x48) is a page of text and house (36x64) is not square.
IMAGES = {
    "pasta": (f"{LLAMA_PROMPTS}/pasta.jpeg", None),
    "perf_report": ("models/perf/images/example_perf_report.png", 1166),
    "ocr_image": (f"{LLAMA_PROMPTS}/ocr_image.jpeg", 941),
    "house": ("models/sample_data/house_in_field_1080p.jpg", None),
}
PRE_LAYRNORM_ROWS = 64


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_case(out: str, case: str, tensors: dict, metadata: dict) -> dict:
    """Write <out>/<case>.safetensors under a temporary name, then rename it; return its file name, size and sha256."""
    path = os.path.join(out, f"{case}.safetensors")
    save_file({name: tensor.contiguous() for name, tensor in tensors.items()}, path + ".tmp", metadata=metadata)
    os.replace(path + ".tmp", path)
    return {"file": os.path.basename(path), "bytes": os.path.getsize(path), "sha256": sha256(path)}


def save_manifest(out: str, manifest: dict) -> None:
    """Write <out>/manifest.json, then <out>/SHA256SUMS for it and every case file."""
    path = os.path.join(out, "manifest.json")
    with open(path + ".tmp", "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    os.replace(path + ".tmp", path)
    names = ["manifest.json"] + [entry["file"] for entry in manifest["cases"].values()]
    path = os.path.join(out, "SHA256SUMS")
    with open(path + ".tmp", "w") as f:
        f.writelines(f"{sha256(os.path.join(out, name))}  {name}\n" for name in names)
    os.replace(path + ".tmp", path)


def load_image(case: str) -> tuple:
    """The image of IMAGES[case] in RGB, and where it came from. The crop must fit the image: PIL would pad it with
    black."""
    path, side = IMAGES[case]
    with Image.open(REPO / path) as image:
        source = {
            "image": path,
            "image_sha256": sha256(REPO / path),
            "image_mode": image.mode,
            "image_size": list(image.size),
            "crop_box": [0, 0, side, side] if side else None,
        }
        if side:
            if side > min(image.size):
                raise ValueError(f"{case}: crop {side}x{side} does not fit the {image.size[0]}x{image.size[1]} image")
            image = image.crop((0, 0, side, side))
        return image.convert("RGB"), source


def load_hf_model(hf_model: str):
    """MiniMaxM3VLModel with the checkpoint's vision tower and projector in fp32, read by tt/vision/loader.py as for
    the TT model. The language model stays on the meta device: it takes no memory and never runs."""
    from transformers import MiniMaxM3VLConfig, MiniMaxM3VLModel

    config = MiniMaxM3VLConfig.from_pretrained(hf_model)
    with torch.device("meta"):
        model = MiniMaxM3VLModel(config)
    missing, unexpected = model.load_state_dict(load_vision_state_dict(hf_model), strict=False, assign=True)
    missing = [key for key in missing if not key.startswith("language_model.")]
    if missing or unexpected:
        raise ValueError(f"{hf_model}: vision weights missing {missing}, unexpected {unexpected}")
    return model.eval()


def image_features(model, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> tuple:
    """Tower output (L, 1280) and vision tokens (L / 4, 6144) from get_image_features, as fp32."""
    features = model.get_image_features(pixel_values=pixel_values, image_grid_thw=grid_thw)
    return features.last_hidden_state.squeeze(0).float(), features.pooler_output.float()


def pcc(ref: torch.Tensor, out: torch.Tensor) -> float:
    """Pearson correlation in float64, to 7 significant digits."""
    pair = torch.stack([ref.flatten().double(), out.flatten().double()])
    return float(f"{torch.corrcoef(pair)[0, 1].item():.7g}")


def cpu_name() -> str | None:
    """The CPU's model name from /proc/cpuinfo, or None if there is none."""
    try:
        with open("/proc/cpuinfo") as f:
            return next((line.split(":", 1)[1].strip() for line in f if line.startswith("model name")), None)
    except OSError:
        return None


@torch.no_grad()
def make_goldens(hf_model: str, out: str) -> dict:
    """Write the goldens of every image of IMAGES, the manifest and SHA256SUMS to out; return the manifest."""
    import transformers
    from transformers import MiniMaxM3VLImageProcessor

    processor = MiniMaxM3VLImageProcessor.from_pretrained(hf_model)
    model = load_hf_model(hf_model)
    goldens, cases = {}, {}
    for case in IMAGES:
        image, source = load_image(case)
        inputs = processor(images=[image], return_tensors="pt")
        pixel_values, grid_thw = inputs["pixel_values"].float(), inputs["image_grid_thw"].long()
        tower, final = image_features(model, pixel_values, grid_thw)
        if not (tower.isfinite().all() and final.isfinite().all()):
            raise ValueError(f"{case}: the fp32 output of Hugging Face's model is not finite")
        vision = model.vision_tower
        goldens[case] = {
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
            "tower_out": tower,
            "final": final,
            "pre_layrnorm_rows": vision.pre_layrnorm(vision.embeddings(pixel_values[:PRE_LAYRNORM_ROWS])).bfloat16(),
        }
        cases[case] = {
            "file": f"{case}.safetensors",
            **source,
            "grid_thw": grid_thw[0].tolist(),
            "patches": tower.shape[0],
            "vision_tokens": final.shape[0],
        }

    model.to(torch.bfloat16)
    for case, golden in goldens.items():
        tower, final = image_features(model, golden["pixel_values"].bfloat16(), golden["image_grid_thw"])
        bf16_pcc = {"final": pcc(golden["final"], final), "tower": pcc(golden["tower_out"], tower)}
        if not all(math.isfinite(value) for value in bf16_pcc.values()):
            raise ValueError(f"{case}: the bf16 PCC of Hugging Face's model is not finite: {bf16_pcc}")
        cases[case]["hf_bf16_pcc"] = bf16_pcc

    os.makedirs(out, exist_ok=True)
    checkpoint = os.path.basename(os.path.normpath(hf_model))
    for case, entry in cases.items():
        metadata = {
            "case": case,
            "image": entry["image"],
            "grid_thw": json.dumps(entry["grid_thw"]),
            "checkpoint": checkpoint,
        }
        entry.update(save_case(out, case, goldens[case], metadata))
    manifest = {
        "format": "minimax_m3 vision goldens 1",
        "checkpoint": checkpoint,
        "checkpoint_path": os.path.abspath(hf_model),
        "reference": {
            "what": "Hugging Face transformers MiniMaxM3VLModel vision tower and projector "
            "(get_image_features: last_hidden_state = tower_out, pooler_output = final), "
            "fp32 weights and activations on the CPU",
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "attn_implementation": model.vision_tower.config._attn_implementation,
            "cpu": cpu_name(),
            "torch_threads": torch.get_num_threads(),
            "image_processor": type(processor).__name__,
        },
        "hf_bf16_pcc": "the same Hugging Face modules cast to bf16 on the same CPU, against these fp32 goldens",
        "pre_layrnorm_rows": PRE_LAYRNORM_ROWS,
        "made": {"by": str(Path(__file__).resolve().relative_to(REPO)), "date": datetime.date.today().isoformat()},
        "cases": cases,
    }
    save_manifest(out, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Hugging Face goldens for tests/test_vision_accuracy.py")
    parser.add_argument("out", help="output directory, created if missing")
    out = parser.parse_args().out
    hf_model = os.getenv("HF_MODEL")
    if not (hf_model and os.path.isdir(hf_model)):
        sys.exit("set HF_MODEL to the checkpoint directory")
    manifest = make_goldens(hf_model, out)
    for case, entry in manifest["cases"].items():
        bf16 = entry["hf_bf16_pcc"]
        print(f"{case}: grid {entry['grid_thw']}, Hugging Face bf16 PCC final {bf16['final']} tower {bf16['tower']}")
    print(f"wrote {out}: {len(manifest['cases'])} images of {manifest['checkpoint']}, manifest.json and SHA256SUMS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
