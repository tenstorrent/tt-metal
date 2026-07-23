# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Export a real denoised latent via host PyTorch denoise -> .pt for VAE isolation.
#
# Run:
#   python_env/bin/python models/experimental/hunyuan_image_3_0/demo/export_latent.py \
#     "a photo of a cat, studio lighting"

import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = str(Path(__file__).resolve().parents[4])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.experimental.hunyuan_image_3_0.ref.host_denoise import HostDenoiseRunner, denoise_loop_host
from models.experimental.hunyuan_image_3_0.ref.image_processor import HunyuanImage3ImageProcessor
from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer
from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import (
    bundle_to_denoise_cond,
    prepare_gen_image_inputs,
)
from models.experimental.hunyuan_image_3_0.ref.weights import ensure_base_weights
from models.experimental.hunyuan_image_3_0.ref.model_config import (
    IMAGE_BASE_SIZE,
    NUM_HIDDEN_LAYERS,
    VAE_SCALING_FACTOR,
)

PROMPT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HY_PROMPT", "a photo of a cat, studio lighting")
STEPS = int(os.environ.get("HY_STEPS", "50"))
NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", str(NUM_HIDDEN_LAYERS)))
GUIDANCE = float(os.environ.get("HY_GUIDANCE", "5.0"))
SEED = int(os.environ.get("HY_SEED", "0"))
IMAGE_SIZE = int(os.environ.get("HY_IMAGE_SIZE", str(IMAGE_BASE_SIZE)))
OUT_LATENT = os.environ.get("HY_OUT_LATENT", f"real_latent_{IMAGE_BASE_SIZE}.pt")

WEIGHTS = ensure_base_weights()
os.environ.setdefault("HUNYUAN_MODEL_DIR", str(WEIGHTS))

_INDEX = WEIGHTS / "model.safetensors.index.json"
if not _INDEX.is_file():
    raise SystemExit(
        f"Weights not found at {_INDEX}\nDownload: hf download tencent/HunyuanImage-3.0\nOr set HUNYUAN_MODEL_DIR"
    )
_WMAP = json.load(open(_INDEX))["weight_map"]
_OPEN = {}


def _load(key):
    shard = _WMAP[key]
    f = _OPEN.get(shard) or _OPEN.setdefault(shard, safe_open(WEIGHTS / shard, framework="pt"))
    return f.get_tensor(key)


def _load_prefix(prefix):
    return {k[len(prefix) + 1 :]: _load(k) for k in _WMAP if k.startswith(prefix + ".")}


def _cfg():
    c = json.load(open(WEIGHTS / "config.json"))
    first = lambda v: v if isinstance(v, int) else v[0]
    return dict(
        H=c["hidden_size"],
        HEADS=c["num_attention_heads"],
        KV=c.get("num_key_value_heads", c["num_attention_heads"]),
        HD=c.get("attention_head_dim", c["hidden_size"] // c["num_attention_heads"]),
        E=first(c["num_experts"]),
        K=first(c["moe_topk"]),
        NORM=c.get("norm_topk_prob", True),
        MIXED=c.get("use_mixed_mlp_moe", True),
        QKN=c.get("use_qk_norm", True),
        EPS=c.get("rms_norm_eps", 1e-5),
        MAX_SEQ=int(c["max_position_embeddings"]),
    )


def _pe_dims(down_sd):
    hid, latent = down_sd["model.0.weight"].shape[:2]
    hsz = down_sd["model.1.in_layers.2.weight"].shape[0]
    return int(latent), int(hid), int(hsz)


class _WeightLoader:
    model_dir = WEIGHTS

    @staticmethod
    def load_prefix(prefix):
        return _load_prefix(prefix)


def main():
    print(
        f"[export_latent] prompt={PROMPT!r}  image_size={IMAGE_SIZE}  steps={STEPS}  "
        f"layers={NUM_LAYERS}  guidance={GUIDANCE}  seed={SEED}",
        flush=True,
    )
    c = _cfg()
    down_sd = _load_prefix("patch_embed")
    up_sd = _load_prefix("final_layer")
    LATENT, _, _ = _pe_dims(down_sd)

    tok = HunyuanTokenizer.from_pretrained()
    wte = _load("model.wte.weight").float()
    proc = HunyuanImage3ImageProcessor(json.load(open(WEIGHTS / "config.json")))

    bundle = prepare_gen_image_inputs(tok, PROMPT, image_size=IMAGE_SIZE)
    span = bundle.rope_image_info[0][0][0]
    grid = bundle.rope_image_info[0][0][1]
    assert grid[0] == grid[1], f"expected square grid, got {grid}"
    assert IMAGE_SIZE == grid[0] * 16, f"image_size {IMAGE_SIZE} != grid*16 ({grid[0] * 16})"
    print(
        f"[export_latent] seq_len={bundle.seq_len}  grid={grid}  image={IMAGE_SIZE}x{IMAGE_SIZE}",
        flush=True,
    )

    torch.manual_seed(SEED)
    init_latent = torch.randn(1, LATENT, grid[0], grid[1])
    cond = bundle_to_denoise_cond(bundle, wte, proc, row=0)
    uncond = bundle_to_denoise_cond(bundle, wte, proc, row=1)

    t0 = time.time()
    runner = HostDenoiseRunner(
        _WeightLoader(),
        WEIGHTS,
        num_layers=NUM_LAYERS,
        down_sd=down_sd,
        up_sd=up_sd,
        model_cfg=c,
    )
    latent = denoise_loop_host(
        runner,
        init_latent=init_latent,
        cond=cond,
        uncond=uncond,
        img_slice=span,
        steps=STEPS,
        guidance_scale=GUIDANCE,
    )
    print(f"[export_latent] denoise done in {time.time() - t0:.0f}s  latent={tuple(latent.shape)}", flush=True)

    payload = {
        "latent": latent.cpu().float(),
        "prompt": PROMPT,
        "seed": SEED,
        "steps": STEPS,
        "num_layers": NUM_LAYERS,
        "guidance": GUIDANCE,
        "image_size": IMAGE_SIZE,
        "grid": grid,
        "scaling_factor": VAE_SCALING_FACTOR,
    }
    out_path = Path(OUT_LATENT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"[export_latent] saved -> {out_path.resolve()}", flush=True)
    print(
        f"[export_latent] decode with:\n  HY_LATENT={out_path} "
        f"python_env/bin/python models/experimental/hunyuan_image_3_0/demo/vae_decode_random.py",
        flush=True,
    )


if __name__ == "__main__":
    main()
