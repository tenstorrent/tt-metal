"""Build the HF MiMoV2DecoderLayer (from the checkpoint's own modeling_mimo_v2.py) with real, dequantized weights. CPU, one layer at a time."""
import importlib.util
import os
import sys
import types

import torch

from .loader import Ckpt, deinterleave_qkv, dequant_fp8_qkv_ranked, load_layer, qkv_dims  # noqa: F401


def import_hf(ckpt_dir=None):
    ckpt_dir = ckpt_dir or os.environ["MIMO_CKPT"]
    if "mimo_hf" in sys.modules:
        return sys.modules["mimo_hf.modeling_mimo_v2"], sys.modules["mimo_hf.configuration_mimo_v2"]
    pkg = types.ModuleType("mimo_hf")
    pkg.__path__ = [ckpt_dir]
    sys.modules["mimo_hf"] = pkg
    mods = []
    for n in ("configuration_mimo_v2", "modeling_mimo_v2"):
        spec = importlib.util.spec_from_file_location(f"mimo_hf.{n}", os.path.join(ckpt_dir, f"{n}.py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"mimo_hf.{n}"] = m
        spec.loader.exec_module(m)
        mods.append(m)
    return mods[1], mods[0]


def hf_config(ckpt_dir=None):
    mod, cfgmod = import_hf(ckpt_dir)
    cfg = cfgmod.MiMoV2Config.from_pretrained(ckpt_dir or os.environ["MIMO_CKPT"])
    cfg._attn_implementation = "eager"
    return cfg


def build_layer(i, dtype=torch.float32, qkv_layout="rank", sd=None, ckpt=None, cfg=None):
    """qkv_layout: 'rank' (de-interleave per-TP4-rank blocks; the decision) or 'naive' (read checkpoint rows as contiguous Q|K|V with contiguous
    128-row scale blocks). Returns the HF decoder layer in eval mode."""
    mod, _ = import_hf()
    cfg = cfg or hf_config()
    with torch.device("meta"):
        layer = mod.MiMoV2DecoderLayer(cfg, i, "fused_qkv")
    if sd is None:
        sd = load_layer(i, dtype, ckpt=ckpt, qkv_layout=qkv_layout)
    layer.load_state_dict(
        {k: v.to(dtype) if v.is_floating_point() and "e_score" not in k else v for k, v in sd.items()},
        assign=True,
        strict=True,
    )
    return layer.eval()


def rotary(cfg, is_swa, device="cpu"):
    """HF rotary embedding for the layer type (theta 1e7 global / 1e4 SWA, rope dim 64)."""
    mod, _ = import_hf()
    return mod, cfg


def masks_and_rope(cfg, h, position_ids=None):
    """HF causal / sliding-window masks (transformers masking_utils, as MiMoV2Model.forward builds them) and both rotary tables."""
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    mod, _ = import_hf()
    B, S, _ = h.shape
    cache_position = torch.arange(S)
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    kw = dict(config=cfg, inputs_embeds=h, attention_mask=None, past_key_values=None, position_ids=position_ids)
    cfg.layer_types = ["sliding_attention" if p == 1 else "full_attention" for p in cfg.hybrid_layer_pattern]
    masks = {
        "full_attention": create_causal_mask(**kw),
        "sliding_window_attention": create_sliding_window_causal_mask(**kw),
    }
    rope = {
        "full_attention": mod.MiMoV2RotaryEmbedding(cfg, is_swa=False)(h, position_ids),
        "sliding_window_attention": mod.MiMoV2RotaryEmbedding(cfg, is_swa=True)(h, position_ids),
    }
    return masks, rope, position_ids


def run_layer(layer, h, cfg, tm=None):
    masks, rope, pos = tm or masks_and_rope(cfg, h)
    t = layer.attention_type
    with torch.no_grad():
        return layer(h, attention_mask=masks[t], position_ids=pos, position_embeddings=rope[t], use_cache=False)
