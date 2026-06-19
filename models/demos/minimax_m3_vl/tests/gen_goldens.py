# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Generate MiniMax-M3-VL reference goldens. RUN IN THE transformers-5.12 VENV:

    MINIMAX_M3_SNAPSHOT=... \
    /localdev/zbaczewski/m3_ref_venv/bin/python \
        models/demos/minimax_m3_vl/tests/gen_goldens.py

Loads the vision tower + projector from the 2 vision shards (LLM stays on
meta), captures per-submodule + tower + final activations via forward hooks
for a few seeded synthetic images, and writes <grid_tag>.safetensors goldens
+ manifest.json into tests/goldens/. The main-env ttnn PCC tests load these.
"""
import glob
import json
import os

import numpy as np
import torch
from accelerate import init_empty_weights
from PIL import Image
from safetensors.torch import save_file

SNAP = (
    os.environ.get("MINIMAX_M3_SNAPSHOT")
    or sorted(glob.glob("/localdev/zbaczewski/hf_cache/hub/models--MiniMaxAI--MiniMax-M3/snapshots/*"))[-1]
)
GOLD = os.path.join(os.path.dirname(__file__), "goldens")
os.makedirs(GOLD, exist_ok=True)

from transformers import AutoImageProcessor, MiniMaxM3VLConfig, MiniMaxM3VLModel  # noqa: E402

_WANT = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")


def _remap(k):
    if k.startswith("patch_merge_mlp."):
        return "multi_modal_projector.merge_" + k[len("patch_merge_mlp.") :]
    if k.startswith("vision_tower.vision_model."):
        rest = k[len("vision_tower.vision_model.") :]
        rest = rest.replace("embeddings.patch_embedding", "embeddings.proj").replace("encoder.layers.", "layers.")
        return "vision_tower." + rest
    return k


def load_model():
    from safetensors import safe_open

    cfg = MiniMaxM3VLConfig.from_pretrained(SNAP)
    with init_empty_weights():
        model = MiniMaxM3VLModel(cfg)
    model.eval()
    msd = model.state_dict()
    sd = {}
    for shard in ("model-00026-of-00059.safetensors", "model-00059-of-00059.safetensors"):
        with safe_open(os.path.join(SNAP, shard), framework="pt") as f:
            for k in f.keys():
                if k.startswith(_WANT):
                    nk = _remap(k)
                    t = f.get_tensor(k).to(torch.float32)
                    if nk in msd and tuple(msd[nk].shape) != tuple(t.shape):
                        t = t.reshape(msd[nk].shape)
                    sd[nk] = t
    miss, unexp = model.load_state_dict(sd, strict=False, assign=True)
    assert not [m for m in miss if m.startswith(_WANT)], "vision params missing"
    assert not unexp, f"unexpected: {unexp[:4]}"
    return model, cfg


def main():
    torch.manual_seed(0)
    model, cfg = load_model()
    proc = AutoImageProcessor.from_pretrained(SNAP, trust_remote_code=True)

    # Hook the submodules we want goldens for.
    cap = {}

    def hook(name):
        def fn(mod, inp, out):
            cap[name + ".in"] = inp[0].detach().float() if inp and torch.is_tensor(inp[0]) else None
            o = out[0] if isinstance(out, tuple) else out
            cap[name + ".out"] = o.detach().float() if torch.is_tensor(o) else None

        return fn

    vt = model.vision_tower
    handles = [
        vt.pre_layrnorm.register_forward_hook(hook("pre_layrnorm")),
        vt.layers[0].register_forward_hook(hook("block0")),
        vt.layers[0].layer_norm1.register_forward_hook(hook("layer_norm1")),
        vt.layers[0].mlp.register_forward_hook(hook("mlp")),
        vt.layers[0].self_attn.register_forward_hook(hook("attn")),
    ]
    # rotary: capture (cos,sin)
    rot = dict(vt.named_modules()).get("rotary_emb")
    if rot is not None:

        def rope_hook(mod, inp, out):
            cos, sin = out if isinstance(out, tuple) else (out, None)
            cap["rope.cos"] = cos.detach().float()
            if sin is not None:
                cap["rope.sin"] = sin.detach().float()

        handles.append(rot.register_forward_hook(rope_hook))

    manifest = {}
    for h, w in [(224, 224), (448, 448)]:
        cap.clear()
        img = Image.fromarray(np.random.default_rng(h * 31 + w).integers(0, 256, (h, w, 3), dtype=np.uint8))
        pv = proc(images=[img], return_tensors="pt")
        pixel_values = pv["pixel_values"].to(torch.float32)
        grid = pv["image_grid_thw"]
        with torch.no_grad():
            feats = model.get_image_features(pixel_values=pixel_values, image_grid_thw=grid)
        tower_out = feats.last_hidden_state.squeeze(0).detach().float()
        final = feats.pooler_output.detach().float()
        if final.dim() == 3:
            final = final.squeeze(0)

        tag = f"{h}x{w}"
        tensors = {
            "pixel_values": pixel_values,
            "image_grid_thw": grid.to(torch.int32),
            "tower_out": tower_out,
            "final": final,
        }
        for k, v in cap.items():
            if v is not None:
                tensors[k] = v.squeeze(0) if v.dim() == 3 and v.shape[0] == 1 else v
        # safetensors requires distinct storage (captured activations alias each other).
        tensors = {k: v.contiguous().clone() for k, v in tensors.items()}
        save_file(tensors, os.path.join(GOLD, f"{tag}.safetensors"))
        manifest[tag] = {k: list(v.shape) for k, v in tensors.items()}
        print(f"[{tag}] saved {len(tensors)} tensors; tower {tuple(tower_out.shape)} final {tuple(final.shape)}")

    for h in handles:
        h.remove()
    json.dump(manifest, open(os.path.join(GOLD, "manifest.json"), "w"), indent=2)
    print("DONE; goldens in", GOLD)


if __name__ == "__main__":
    main()
