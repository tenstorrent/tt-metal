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


BLOCK_KEYS = [
    "norm1.weight",
    "attn.qkv.weight",
    "attn.proj.weight",
    "norm2.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
    "mlp.fc3.weight",
]


def test_vision_block():
    torch.manual_seed(0)
    sd = load_weights("vision_tower.blocks.0", BLOCK_KEYS)
    x = torch.randn(SEQ, 1536)
    rope = fn.vision_rot_pos_emb(GRID)
    cu = torch.tensor([0, SEQ], dtype=torch.int32)
    vis = _remote_module("modeling_dots_vision")
    hf = vis.DotsVisionBlock(vision_config(), "eager")
    hf.load_state_dict(sd)
    ref = fn.vision_block_forward(x, sd, cu, rope, num_heads=12, eps=1e-5)
    p = pcc(ref, hf(x, cu_seqlens=cu, rotary_pos_emb=rope))
    _save_golden(
        "vision_block",
        {"input": x, "output": ref, "rotary_pos_emb": rope, "cu_seqlens": cu, "grid_thw": GRID, "pcc_vs_hf": p},
    )
    assert p > 0.99, p


MERGER_KEYS = ["ln_q.weight", "ln_q.bias", "mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias"]


def test_patch_merger():
    torch.manual_seed(0)
    sd = load_weights("vision_tower.merger", MERGER_KEYS)
    x = torch.randn(SEQ, 1536)
    vis = _remote_module("modeling_dots_vision")
    hf = vis.PatchMerger(dim=1536, context_dim=1536, spatial_merge_size=2)
    hf.load_state_dict(sd)
    ref = fn.patch_merger_forward(x, sd, spatial_merge_size=2, eps=1e-6)
    p = pcc(ref, hf(x))
    _save_golden("patch_merger", {"input": x, "output": ref, "pcc_vs_hf": p})
    assert p > 0.99, p


def test_vision_transformer():
    torch.manual_seed(0)
    idx = json.load(open(SNAP / "model.safetensors.index.json"))["weight_map"]
    vt_keys = sorted(k for k in idx if k.startswith("vision_tower."))
    sd = {}
    by_file = {}
    for k in vt_keys:
        by_file.setdefault(idx[k], []).append(k)
    for fname, keys in by_file.items():
        with safe_open(SNAP / fname, framework="pt") as f:
            for k in keys:
                sd[k[len("vision_tower.") :]] = f.get_tensor(k).float()

    x = torch.randn(SEQ, 3 * 14 * 14)
    vis = _remote_module("modeling_dots_vision")
    hf = vis.DotsVisionTransformer(vision_config())
    hf.load_state_dict(sd, assign=True)
    hf.eval()
    with torch.no_grad():
        hf_out = hf(x, GRID, bf16=False)
        ref = fn.vision_transformer_forward(x, sd, GRID, num_layers=42, num_heads=12, eps=1e-5)
    p = pcc(ref, hf_out)
    _save_golden("vision_transformer", {"input": x, "output": ref, "grid_thw": GRID, "pcc_vs_hf": p})
    assert p > 0.99, p


def test_embedding():
    torch.manual_seed(0)
    w = load_weights("model", ["embed_tokens.weight"])["embed_tokens.weight"]
    ids = torch.randint(0, w.shape[0], (1, 128))
    hf = torch.nn.Embedding(w.shape[0], w.shape[1])
    hf.weight.data.copy_(w)
    ref = fn.embedding_forward(ids, w)
    p = pcc(ref, hf(ids))
    _save_golden("embedding", {"input": ids, "output": ref, "pcc_vs_hf": p})
    assert p > 0.99, p


# ---------------------------------------------------------------------------
# text side: plain Qwen2 (DotsOCRForCausalLM subclasses Qwen2ForCausalLM)
# ---------------------------------------------------------------------------
TEXT_SEQ = 128


def text_config():
    from transformers.models.qwen2 import Qwen2Config

    raw = json.load(open(SNAP / "config.json"))
    drop = {"vision_config", "architectures", "auto_map", "model_type"}
    cfg = Qwen2Config(**{k: v for k, v in raw.items() if k not in drop})
    cfg._attn_implementation = "eager"
    return cfg


def _text_pos_embeds(cfg, x):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    pos_ids = torch.arange(TEXT_SEQ).unsqueeze(0)
    hf_cos, hf_sin = Qwen2RotaryEmbedding(cfg)(x, pos_ids)
    ref_cos, ref_sin = fn.text_rope_cos_sin(pos_ids, head_dim=cfg.hidden_size // cfg.num_attention_heads)
    assert torch.allclose(ref_cos, hf_cos, atol=1e-5) and torch.allclose(ref_sin, hf_sin, atol=1e-5)
    return ref_cos, ref_sin


def _causal_mask_4d(dtype):
    mask = torch.triu(torch.full((TEXT_SEQ, TEXT_SEQ), torch.finfo(dtype).min, dtype=dtype), diagonal=1)
    return mask[None, None, :, :]


def test_text_rmsnorm():
    torch.manual_seed(0)
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    w = load_weights("model.layers.0", ["input_layernorm.weight"])["input_layernorm.weight"]
    x = torch.randn(1, TEXT_SEQ, 1536)
    hf = Qwen2RMSNorm(1536, eps=1e-6)
    hf.weight.data.copy_(w)
    ref = fn.text_rmsnorm_forward(x, w, eps=1e-6)
    p = pcc(ref, hf(x))
    _save_golden("text_rmsnorm", {"input": x, "output": ref, "weight": w, "pcc_vs_hf": p, "eps": 1e-6})
    assert p > 0.99, p


ATTN_KEYS = [
    "q_proj.weight",
    "q_proj.bias",
    "k_proj.weight",
    "k_proj.bias",
    "v_proj.weight",
    "v_proj.bias",
    "o_proj.weight",
]


def test_text_attention():
    torch.manual_seed(0)
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

    cfg = text_config()
    sd = load_weights("model.layers.0.self_attn", ATTN_KEYS)
    x = torch.randn(1, TEXT_SEQ, cfg.hidden_size)
    cos, sin = _text_pos_embeds(cfg, x)
    hf = Qwen2Attention(cfg, layer_idx=0)
    hf.load_state_dict({k: v for k, v in sd.items()})
    with torch.no_grad():
        hf_out = hf(x, position_embeddings=(cos, sin), attention_mask=_causal_mask_4d(x.dtype))[0]
        ref = fn.text_attention_forward(x, sd, cos, sin, num_heads=12, num_kv_heads=2)
    p = pcc(ref, hf_out)
    _save_golden("text_attention", {"input": x, "output": ref, "cos": cos, "sin": sin, "pcc_vs_hf": p})
    assert p > 0.99, p


def test_text_mlp():
    torch.manual_seed(0)
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    cfg = text_config()
    sd = load_weights("model.layers.0.mlp", ["gate_proj.weight", "up_proj.weight", "down_proj.weight"])
    x = torch.randn(1, TEXT_SEQ, cfg.hidden_size)
    hf = Qwen2MLP(cfg)
    hf.load_state_dict(sd)
    with torch.no_grad():
        hf_out = hf(x)
        ref = fn.text_mlp_forward(x, sd)
    p = pcc(ref, hf_out)
    _save_golden("text_mlp", {"input": x, "output": ref, "pcc_vs_hf": p})
    assert p > 0.99, p


LAYER_KEYS = (
    ["input_layernorm.weight", "post_attention_layernorm.weight"]
    + [f"self_attn.{k}" for k in ATTN_KEYS]
    + ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
)


def test_decoder_layer():
    torch.manual_seed(0)
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    cfg = text_config()
    sd = load_weights("model.layers.0", LAYER_KEYS)
    x = torch.randn(1, TEXT_SEQ, cfg.hidden_size)
    cos, sin = _text_pos_embeds(cfg, x)
    hf = Qwen2DecoderLayer(cfg, layer_idx=0)
    hf.load_state_dict(sd)
    with torch.no_grad():
        hf_out = hf(x, attention_mask=_causal_mask_4d(x.dtype), position_embeddings=(cos, sin))
        hf_out = hf_out[0] if isinstance(hf_out, tuple) else hf_out
        ref = fn.decoder_layer_forward(x, sd, cos, sin, num_heads=12, num_kv_heads=2, eps=1e-6)
    p = pcc(ref, hf_out)
    _save_golden("decoder_layer", {"input": x, "output": ref, "cos": cos, "sin": sin, "pcc_vs_hf": p})
    assert p > 0.99, p


def test_lm_head():
    torch.manual_seed(0)
    cfg = text_config()
    w = load_weights("lm_head", ["weight"])["weight"]
    assert w.shape == (cfg.vocab_size, cfg.hidden_size)
    x = torch.randn(1, TEXT_SEQ, cfg.hidden_size)
    hf = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    hf.weight.data.copy_(w)
    with torch.no_grad():
        hf_out = hf(x)
        ref = fn.lm_head_forward(x, w)
    p = pcc(ref, hf_out)
    _save_golden("lm_head", {"input": x, "output": ref, "pcc_vs_hf": p})
    assert p > 0.99, p
