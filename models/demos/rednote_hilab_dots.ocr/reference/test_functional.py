# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC tests: pure-PyTorch reference blocks vs official dots.ocr HF modules
(real checkpoint weights). Also writes golden tensors when GOLDEN_DIR is set.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest  # noqa: F401
import torch
from safetensors import safe_open

REPO = "rednote-hilab/dots.ocr"
HERE = Path(__file__).resolve().parent
GOLDEN_DIR = HERE / "golden"

# Dir name contains a dot -> not importable as a package; load by path.
_spec = importlib.util.spec_from_file_location("dots_ocr_reference_functional", HERE / "functional.py")
fn = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fn
_spec.loader.exec_module(fn)


def _snapshot():
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(REPO, allow_patterns=["*.json", "*.py", "*.safetensors"]))


SNAP = _snapshot()


def _remote_module(name):
    pkg = sys.modules.setdefault("dots_remote", types.ModuleType("dots_remote"))
    pkg.__path__ = [str(SNAP)]
    full = f"dots_remote.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, SNAP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


def load_weights(prefix, keys):
    idx = json.load(open(SNAP / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for k in keys:
        full = f"{prefix}.{k}"
        with safe_open(SNAP / idx[full], framework="pt") as f:
            out[k] = f.get_tensor(full).float()
    return out


def vision_config():
    cfg_mod = _remote_module("configuration_dots")
    raw = json.load(open(SNAP / "config.json"))["vision_config"]
    raw["attn_implementation"] = "eager"
    return cfg_mod.DotsVisionConfig(**{k: v for k, v in raw.items() if k != "model_type"})


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _save_golden(name, payload):
    GOLDEN_DIR.mkdir(exist_ok=True)
    torch.save(payload, GOLDEN_DIR / f"{name}.pt")


GRID = torch.tensor([[1, 28, 28]])  # 784 patches
SEQ = 784


def test_vision_rmsnorm():
    torch.manual_seed(0)
    w = load_weights("vision_tower.blocks.0", ["norm1.weight"])["norm1.weight"]
    x = torch.randn(SEQ, 1536)
    vis = _remote_module("modeling_dots_vision")
    hf = vis.RMSNorm(1536, eps=1e-5)
    hf.weight.data.copy_(w)
    ref = fn.vision_rmsnorm_forward(x, w, eps=1e-5)
    p = pcc(ref, hf(x))
    _save_golden("vision_rmsnorm", {"input": x, "output": ref, "weight": w, "pcc_vs_hf": p, "eps": 1e-5})
    assert p > 0.99, p


def test_vision_patch_embed():
    torch.manual_seed(0)
    sd = load_weights("vision_tower.patch_embed.patchifier", ["proj.weight", "proj.bias", "norm.weight"])
    x = torch.randn(SEQ, 3 * 14 * 14)
    vis = _remote_module("modeling_dots_vision")
    hf = vis.DotsPatchEmbed(vision_config())
    hf.proj.weight.data.copy_(sd["proj.weight"])
    hf.proj.bias.data.copy_(sd["proj.bias"])
    hf.norm.weight.data.copy_(sd["norm.weight"])
    ref = fn.vision_patch_embed_forward(x, sd)
    p = pcc(ref, hf(x))
    _save_golden("vision_patch_embed", {"input": x, "output": ref, "pcc_vs_hf": p, "grid_thw": GRID})
    assert p > 0.99, p


def test_vision_attention():
    torch.manual_seed(0)
    sd = load_weights("vision_tower.blocks.0.attn", ["qkv.weight", "proj.weight"])
    x = torch.randn(SEQ, 1536)
    rope = fn.vision_rot_pos_emb(GRID)
    cu = torch.tensor([0, SEQ], dtype=torch.int32)
    vis = _remote_module("modeling_dots_vision")
    hf = vis.VisionAttention(vision_config(), 1536, num_heads=12, bias=False)
    hf.qkv.weight.data.copy_(sd["qkv.weight"])
    hf.proj.weight.data.copy_(sd["proj.weight"])
    ref = fn.vision_attention_forward(x, sd, cu, rope, num_heads=12)
    p = pcc(ref, hf(x, cu, rope))
    _save_golden(
        "vision_attention",
        {"input": x, "output": ref, "rotary_pos_emb": rope, "cu_seqlens": cu, "grid_thw": GRID, "pcc_vs_hf": p},
    )
    assert p > 0.99, p


def test_vision_mlp():
    torch.manual_seed(0)
    sd = load_weights("vision_tower.blocks.0.mlp", ["fc1.weight", "fc2.weight", "fc3.weight"])
    x = torch.randn(SEQ, 1536)
    vis = _remote_module("modeling_dots_vision")
    hf = vis.DotsSwiGLUFFN(vision_config())
    hf.fc1.weight.data.copy_(sd["fc1.weight"])
    hf.fc2.weight.data.copy_(sd["fc2.weight"])
    hf.fc3.weight.data.copy_(sd["fc3.weight"])
    ref = fn.vision_mlp_forward(x, sd)
    p = pcc(ref, hf(x))
    _save_golden("vision_mlp", {"input": x, "output": ref, "pcc_vs_hf": p})
    assert p > 0.99, p
