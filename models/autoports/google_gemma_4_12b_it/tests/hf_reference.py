# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF reference, checkpoint access and synthetic weights for the Gemma4 12B functional decoder tests.

The reference is the installed ``Gemma4UnifiedTextDecoderLayer`` run in float32 on the BF16 checkpoint
values, with an explicit per-request ``DynamicCache`` (full retention) and an explicit additive mask
built from absolute positions: causal, plus ``kv > q - sliding_window`` on sliding layers. That is the
same mask HF's ``create_sliding_window_causal_mask`` produces for a single text request.
"""

import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedTextDecoderLayer,
    Gemma4UnifiedTextRotaryEmbedding,
    apply_rotary_pos_emb,
)

HF_MODEL = "google/gemma-4-12B-it"
HF_REVISION = os.environ.get("HF_REVISION", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7")
HERE = Path(__file__).resolve().parent
WEIGHT_STATS = HERE / "stats" / "weight_stats.json"
ACTIVATION_STATS = HERE / "stats" / "activation_stats.json"
LAYER_KINDS = {"sliding": 0, "full": 5}


def snapshot_dir():
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            HF_MODEL, revision=HF_REVISION, allow_patterns=["*.json", "*.safetensors"], local_files_only=True
        )
    )


def text_config():
    cfg = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION).text_config
    cfg._attn_implementation = "eager"
    return cfg


def real_layer_state_dict(layer_idx):
    """BF16 tensors of one decoder layer, keys relative to the layer."""
    from safetensors import safe_open

    prefix = f"model.language_model.layers.{layer_idx}."
    out = {}
    with safe_open(str(snapshot_dir() / "model.safetensors"), "pt") as f:
        for k in f.keys():
            if k.startswith(prefix):
                out[k[len(prefix) :]] = f.get_tensor(k)
    assert out, f"no checkpoint tensors for layer {layer_idx}"
    return out


def synthetic_layer_state_dict(layer_idx, seed=0):
    """Deterministic BF16 tensors with the checkpoint's key/shape contract and per-tensor mean/std."""
    stats = json.loads(WEIGHT_STATS.read_text())[str(layer_idx)]
    g = torch.Generator().manual_seed(1000 * layer_idx + seed)
    out = {}
    for name, s in sorted(stats.items()):
        t = torch.randn(s["shape"], generator=g, dtype=torch.float32) * s["std"] + s["mean"]
        out[name] = t.to(torch.bfloat16)
    return out


def layer_state_dict(layer_idx, weights):
    return real_layer_state_dict(layer_idx) if weights == "real" else synthetic_layer_state_dict(layer_idx)


def synthetic_hidden(layer_idx, seq_len, seed=0, batch=1):
    """Activations shaped like real inputs to ``layer_idx``: per-channel mean/std from a real prompt."""
    stats = json.loads(ACTIVATION_STATS.read_text())
    mean, std = torch.tensor(stats[f"{layer_idx}.mean"]), torch.tensor(stats[f"{layer_idx}.std"])
    g = torch.Generator().manual_seed(7919 * (layer_idx + 1) + seed)
    x = torch.randn(batch, seq_len, mean.numel(), generator=g) * std + mean
    return x.to(torch.bfloat16)


class HFLayer:
    """Float32 HF decoder layer + rotary embedding with explicit absolute-position masks."""

    def __init__(self, cfg, layer_idx, state_dict):
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.layer_type = cfg.layer_types[layer_idx]
        self.window = cfg.sliding_window if self.layer_type == "sliding_attention" else None
        with torch.device("meta"):
            layer = Gemma4UnifiedTextDecoderLayer(cfg, layer_idx)
        layer.load_state_dict({k: v.float() for k, v in state_dict.items()}, assign=True, strict=True)
        self.layer = layer.eval()
        self.rope = Gemma4UnifiedTextRotaryEmbedding(cfg)

    def new_cache(self):
        return DynamicCache()

    def cached_len(self, cache):
        return cache.get_seq_length(self.layer_idx) if len(cache.layers) > self.layer_idx else 0

    def _mask(self, q_pos, kv_len):
        k_pos = torch.arange(kv_len)[None, :]
        allowed = k_pos <= q_pos[:, None]
        if self.window is not None:
            allowed &= k_pos > q_pos[:, None] - self.window
        return torch.where(allowed, 0.0, float("-inf"))[None, None]

    @torch.no_grad()
    def forward(self, x, cache, start_pos):
        """x ``[1, S, H]`` at absolute positions ``start_pos..``; appends its K/V to ``cache``."""
        assert self.cached_len(cache) == start_pos, "reference cache must hold exactly the prefix"
        x = x.float()
        seq = x.shape[1]
        pos = torch.arange(start_pos, start_pos + seq)
        pe = self.rope(x, pos[None], self.layer_type)
        return self.layer(
            x,
            shared_kv_states={},
            position_embeddings=pe,
            attention_mask=self._mask(pos, start_pos + seq),
            position_ids=pos[None],
            past_key_values=cache,
        )

    def prefill(self, x, cache, start_pos=0, slice_len=1024):
        """HF chunked prefill (bounded memory); same math as one call."""
        outs = []
        for s in range(0, x.shape[1], slice_len):
            outs.append(self.forward(x[:, s : s + slice_len], cache, start_pos + s))
        return torch.cat(outs, dim=1)

    @torch.no_grad()
    def inject_prefix_kv(self, x, cache, start_pos=0, slice_len=8192):
        """Append K/V of ``x`` exactly as HF attention computes them, without evaluating its queries.

        Used only to build very long prefixes (the full-length context tests); the checked rows always
        run through the real HF decoder layer forward. Mirrors Gemma4UnifiedTextAttention.forward.
        """
        attn = self.layer.self_attn
        for s in range(0, x.shape[1], slice_len):
            xs = x[:, s : s + slice_len].float()
            pos = torch.arange(start_pos + s, start_pos + s + xs.shape[1])
            cos, sin = self.rope(xs, pos[None], self.layer_type)
            h = self.layer.input_layernorm(xs)
            shape = (*h.shape[:-1], -1, attn.head_dim)
            k = attn.k_proj(h).view(shape)
            v = attn.v_proj(h).view(shape) if attn.v_proj is not None else k
            k = apply_rotary_pos_emb(attn.k_norm(k), cos, sin, unsqueeze_dim=2).transpose(1, 2)
            v = attn.v_norm(v).transpose(1, 2)
            cache.update(k, v, self.layer_idx)


def pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def real_prompt_ids(num_tokens):
    """Token ids of real English/technical text from this repository, chat-formatted as a user turn."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_MODEL, revision=HF_REVISION)
    root = HERE.parents[3]
    text = "\n\n".join((root / p).read_text() for p in ("README.md", "tech_reports/LLMs/llms.md", "METALIUM_GUIDE.md"))
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=False, return_dict=False
    )
    assert len(ids) >= num_tokens, f"prompt has only {len(ids)} tokens"
    return torch.tensor(ids[:num_tokens])


@torch.no_grad()
def real_hidden(cfg, layer_idx, ids):
    """Real input activations of ``layer_idx``: scaled embeddings then HF layers ``0..layer_idx-1`` (fp32)."""
    from safetensors import safe_open

    with safe_open(str(snapshot_dir() / "model.safetensors"), "pt") as f:
        emb = f.get_slice("model.language_model.embed_tokens.weight")
        x = torch.stack([emb[int(i) : int(i) + 1][0] for i in ids]).float()
    # Gemma4UnifiedTextScaledWordEmbedding in the float32 reference: embedding * sqrt(hidden_size).
    x = (x * cfg.hidden_size**0.5)[None]
    for i in range(layer_idx):
        ref = HFLayer(cfg, i, real_layer_state_dict(i))
        x = ref.prefill(x, ref.new_cache())
    return x
