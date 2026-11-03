# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Goldens for tests/test_vision_accuracy.py: Hugging Face's MiniMax-M3 vision tower and projector with the vision
weights of the checkpoint in HF_MODEL, run on the CPU in fp32 (the goldens) and in bf16 (the error of a plain bf16
run, which sets the test's tower bar and which the test logs beside its own).

Each image of IMAGES goes through the checkpoint's image processor, then MiniMaxM3VLModel.get_image_features. The
model's vision weights are the checkpoint's, named by transformers' own rules for it (vision_weights), not through
tt/vision/loader.py, which the TT model loads with: a rename there that swapped two tensors of the same shape would
otherwise reach the TT model and the goldens alike, and the test would pass.
<out>/<image>.safetensors holds
    pixel_values       (L, 1176) fp32      the processor's output, the test's input
    image_grid_thw     (1, 3) int64        the (t, h, w) patch grid
    tower_out          (L, 1280) fp32      the tower output (last_hidden_state)
    final              (L / 4, 6144) fp32  the vision tokens (pooler_output)
    pre_layrnorm_rows  (64, 1280) bf16     the first rows of the pre_layrnorm output, which the test recomputes from
                                           HF_MODEL's weights to check that HF_MODEL is this checkpoint
<out>/manifest.json names the checkpoint, the library versions, the CPU and the number of torch threads, and gives, per
image, its file and its sha256, which the test checks, and the error of the bf16 run against the fp32 one on each
output: PCC, relative L2 error and the 1st percentile of the cosine between rows; <out>/SHA256SUMS checks a copy
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
from safetensors import safe_open
from safetensors.torch import save_file

# transformers' minimax_m3_vl classes are imported where they are used: they take seconds to import, and
# tests/test_vision_accuracy.py imports IMAGES from this module without needing them.

REPO = Path(__file__).resolve().parents[4]
LLAMA_PROMPTS = "models/tt_transformers/demo/sample_prompts/llama_models"
# The test's images, in the order it runs them: repo path, and the side of the top-left square kept (None: the whole
# image), with their patch grids. The first four make 2304 patches (576 vision tokens, whole tiles): on the device,
# pasta and perf_report (48x48) had the lowest tower and the lowest final PCC of six real images at 672x672, ocr_image
# (48x48) is a page of text, and house (36x64) runs new RoPE tables on the programs the three before it built. cat
# (34x46: 1564 patches, 391 tokens) takes the padded path most photos take: partial tiles of rows and of tokens and
# masked keys in SDPA. cluster_cabling (42x54: 2268 patches, 567 tokens) is a 4:3 photo: on a 12x10 chip the per-patch
# matmuls split their 71 tiles of rows 6 to a core, so the last core takes 5.
IMAGES = {
    "pasta": (f"{LLAMA_PROMPTS}/pasta.jpeg", None),
    "perf_report": ("models/perf/images/example_perf_report.png", 1166),
    "ocr_image": (f"{LLAMA_PROMPTS}/ocr_image.jpeg", 941),
    "house": ("models/sample_data/house_in_field_1080p.jpg", None),
    "cat": ("models/sample_data/huggingface_cat_image.jpg", None),
    "cluster_cabling": ("tests/scale_out/4x_bh_quietbox/images/cluster_cabling.jpg", None),
}
GOLDEN_FORMAT = "minimax_m3 vision goldens 2"
PRE_LAYRNORM_ROWS = 64
# Name prefixes of MiniMaxM3VLModel's vision tower and projector weights.
VISION_PREFIXES = ("vision_tower.", "multi_modal_projector.")


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


def load_image(case: str, mode: str = "RGB") -> tuple:
    """The image of IMAGES[case], converted from its decoded mode to mode (RGB for the processor), and where it came
    from. The crop must fit the image: PIL would pad it with black."""
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
        return image.convert(mode), source


def vision_weights(hf_model: str, model) -> dict:
    """The vision tower and projector weights of the checkpoint in hf_model, in fp32, under the names of model (a
    MiniMaxM3VLModel): every key of the checkpoint's model.safetensors.index.json is renamed by transformers' rules
    for the model, the conversion mapping and rename_source_key that from_pretrained applies, and the keys that land on
    a vision weight of the model are read, from their shards only. Raises ValueError for a vision weight of the model
    that no key lands on, two keys that land on one, or a key that transformers would convert rather than rename."""
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key

    transforms = get_model_conversion_mapping(model)
    renamings = [transform for transform in transforms if isinstance(transform, WeightRenaming)]
    converters = [transform for transform in transforms if isinstance(transform, WeightConverter)]
    params = model.state_dict()
    vision = {name for name in params if name.startswith(VISION_PREFIXES)}
    with open(os.path.join(hf_model, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    keys_by_shard, sources = {}, {}
    for key, shard in weight_map.items():
        name, converter = rename_source_key(key, renamings, converters, model.base_model_prefix, params)
        if name not in vision:
            continue
        if converter is not None:
            raise ValueError(f"{hf_model}: transformers converts {key} into {name} ({converter}), not only renames it")
        if name in sources:
            raise ValueError(f"{hf_model}: {sources[name]} and {key} are both named {name} by transformers")
        sources[name] = key
        keys_by_shard.setdefault(shard, []).append(key)
    missing = sorted(vision - set(sources))
    if missing:
        raise ValueError(f"{hf_model}: no checkpoint weight for {len(missing)} vision weights, such as {missing[0]}")
    names = {key: name for name, key in sources.items()}
    weights = {}
    for shard, keys in sorted(keys_by_shard.items()):
        with safe_open(os.path.join(hf_model, shard), framework="pt") as f:
            for key in keys:
                weights[names[key]] = f.get_tensor(key).float()
    return weights


def load_hf_model(hf_model: str):
    """MiniMaxM3VLModel with the checkpoint's vision tower and projector in fp32 (vision_weights). The language model
    stays on the meta device: it takes no memory and never runs."""
    from transformers import MiniMaxM3VLConfig, MiniMaxM3VLModel

    config = MiniMaxM3VLConfig.from_pretrained(hf_model)
    with torch.device("meta"):
        model = MiniMaxM3VLModel(config)
    missing, unexpected = model.load_state_dict(vision_weights(hf_model, model), strict=False, assign=True)
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


def rel_l2(ref: torch.Tensor, out: torch.Tensor) -> float:
    """Relative L2 error, |out - ref| / |ref| over all values, in float64, to 7 significant digits."""
    error = torch.linalg.vector_norm((out.double() - ref.double()).flatten()) / torch.linalg.vector_norm(ref.double())
    return float(f"{error.item():.7g}")


def gain(ref: torch.Tensor, out: torch.Tensor) -> float:
    """The least-squares scale of out against ref, sum(out * ref) / sum(ref * ref), in float64, to 7 significant
    digits: 1.0 when out has no scale error. PCC cannot see a scale. The goldens do not record it; the test logs it
    for the encoder's outputs."""
    ref, out = ref.double(), out.double()
    return float(f"{((out * ref).sum() / (ref * ref).sum()).item():.7g}")


def token_cosine_p01(ref: torch.Tensor, out: torch.Tensor) -> float:
    """The 1st percentile of the cosine similarity between each row of out and the same row of ref (a vision token or a
    patch), in float64, to 7 significant digits: the worst rows, which a PCC over all values averages away."""
    ref, out = ref.double().flatten(1), out.double().flatten(1)
    cosine = (ref * out).sum(1) / (torch.linalg.vector_norm(ref, dim=1) * torch.linalg.vector_norm(out, dim=1))
    return float(f"{torch.quantile(cosine, 0.01).item():.7g}")


def errors(ref: torch.Tensor, out: torch.Tensor) -> dict:
    """The manifest's errors of out against ref: pcc, rel_l2 and cos_p01."""
    return {"pcc": pcc(ref, out), "rel_l2": rel_l2(ref, out), "cos_p01": token_cosine_p01(ref, out)}


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
        bf16 = {"final": errors(golden["final"], final), "tower": errors(golden["tower_out"], tower)}
        if not all(math.isfinite(value) for output in bf16.values() for value in output.values()):
            raise ValueError(f"{case}: the bf16 errors of Hugging Face's model are not finite: {bf16}")
        cases[case]["hf_bf16"] = bf16

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
        "format": GOLDEN_FORMAT,
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
        "hf_bf16": "the same Hugging Face modules cast to bf16 on the same CPU, against these fp32 goldens: PCC, "
        "relative L2 error and 1st percentile of the per-row cosine",
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
        final, tower = entry["hf_bf16"]["final"], entry["hf_bf16"]["tower"]
        print(
            f"{case}: grid {entry['grid_thw']}, Hugging Face bf16 final PCC {final['pcc']} relL2 {final['rel_l2']},"
            f" tower PCC {tower['pcc']} relL2 {tower['rel_l2']}"
        )
    print(f"wrote {out}: {len(manifest['cases'])} images of {manifest['checkpoint']}, manifest.json and SHA256SUMS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
