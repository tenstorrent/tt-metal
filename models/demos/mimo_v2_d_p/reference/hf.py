# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden reference: the checkpoint's own ``modeling_mimo_v2.py`` decoder layer, fed dequantized weights.

The remote-code files live next to the downloaded config (``MIMO_V2_CKPT``); they are imported as a
synthetic package so their relative imports resolve. Eager attention (the only path with sink bias).
"""

import importlib.util
import sys
import types

import torch

from models.demos.mimo_v2_d_p.reference.remote_st import LOCAL

_PKG = "mimo_v2_hf"


def hf_modules():
    if _PKG in sys.modules:
        return sys.modules[_PKG + ".configuration_mimo_v2"], sys.modules[_PKG + ".modeling_mimo_v2"]
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(LOCAL)]
    sys.modules[_PKG] = pkg
    mods = []
    for name in ("configuration_mimo_v2", "modeling_mimo_v2"):
        spec = importlib.util.spec_from_file_location(f"{_PKG}.{name}", LOCAL / f"{name}.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        mods.append(m)
    return tuple(mods)


def hf_config():
    conf, _ = hf_modules()
    cfg = conf.MiMoV2Config.from_pretrained(str(LOCAL))
    cfg = getattr(cfg, "text_config", cfg)
    cfg._attn_implementation = "eager"
    return cfg


def decoder_layer(layer_idx: int, sd: dict, cfg=None, dtype=torch.float32):
    """HF ``MiMoV2DecoderLayer`` with ``sd`` (layer_state names) loaded."""
    _, mod = hf_modules()
    cfg = cfg or hf_config()
    layer = mod.MiMoV2DecoderLayer(cfg, layer_idx, attention_projection_layout="fused_qkv")
    missing, unexpected = layer.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
    assert not unexpected, unexpected
    assert not missing, missing
    return layer.to(dtype).eval()


def rotary(is_swa: bool, cfg=None):
    _, mod = hf_modules()
    return mod.MiMoV2RotaryEmbedding(cfg or hf_config(), is_swa)


def mask(q_pos: torch.Tensor, k_len: int, window: int | None, dtype=torch.float32):
    """Additive [1,1,Sq,Sk] mask: causal, plus HF sliding semantics (k > q - window) when windowed."""
    k = torch.arange(k_len)[None, :]
    q = q_pos[:, None]
    ok = k <= q
    if window is not None:
        ok &= k > q - window
    m = torch.zeros(ok.shape, dtype=dtype)
    m[~ok] = torch.finfo(dtype).min
    return m[None, None]


@torch.no_grad()
def run_layer(layer, x: torch.Tensor, is_swa: bool, cfg=None, start: int = 0, window=None):
    """Full-sequence forward of one decoder layer. x [1, S, H] (positions start..start+S)."""
    S = x.shape[1]
    pos = torch.arange(start, start + S)[None]
    cos, sin = rotary(is_swa, cfg)(x, pos)
    m = mask(pos[0], S, window, x.dtype)
    return layer(x, attention_mask=m, position_ids=pos, position_embeddings=(cos, sin))


# --------------------------------------------------------------------------- long-sequence golden
KV_CAPTURE: dict = {}  # layer_idx -> (key [1,nkv,S,192] post-rope, value [1,nkv,S,128] scaled) when capturing


def blocked_eager_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, sinks=None, q_block=1024, **kwargs):
    """Exactly ``eager_attention_forward`` (incl. sinks) computed over query blocks, so S=8k+ fits in RAM."""
    _, mod = hf_modules()
    KV_CAPTURE[module.layer_idx] = (key.detach().clone(), value.detach().clone())
    k = mod.repeat_kv(key, module.num_key_value_groups)
    v = mod.repeat_kv(value, module.num_key_value_groups)
    outs = []
    for q0 in range(0, query.shape[2], q_block):
        q = query[:, :, q0 : q0 + q_block]
        w = torch.matmul(q, k.transpose(2, 3)) * scaling
        if attention_mask is not None:
            w = w + attention_mask[:, :, q0 : q0 + q_block, : k.shape[-2]]
        if sinks is not None:
            s = module.attention_sink_bias.reshape(1, -1, 1, 1).expand(q.shape[0], -1, q.shape[-2], -1)
            w = torch.cat([w, s], dim=-1)
        w = w - w.max(dim=-1, keepdim=True).values
        p = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        if sinks is not None:
            p = p[..., :-1]
        outs.append(torch.matmul(p, v))
    return torch.cat(outs, 2).transpose(1, 2).contiguous(), None


def use_blocked_attention():
    _, mod = hf_modules()
    mod.eager_attention_forward = blocked_eager_attention


def tokenize_prompt(n_tokens: int, path=None) -> torch.Tensor:
    from pathlib import Path

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(LOCAL), trust_remote_code=True)
    text = Path(path or Path(__file__).parents[1] / "tests" / "prompt.txt").read_text()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    while len(ids) < n_tokens:
        ids = ids + ids
    return torch.tensor(ids[:n_tokens], dtype=torch.long)
