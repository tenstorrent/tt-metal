# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4-31B checkpoint -> per-chip weights of the resident prototype, converted once and cached on disk.

TP over 4 chips, as models/demos/gemma4_31b_qb2/tt/decoder.py: chip c holds q heads [8c, 8c+8), sliding KV heads
[4c, 4c+4) / full KV head c (K=V, stored once), MLP columns [c*I/4, (c+1)*I/4) and vocab rows [c*V/4, (c+1)*V/4) of
the tied embedding as its LM-head slice. Folded caches (step 1) fold the input and pre-feedforward norm weights into
the rows of the projection after them and the final norm into the LM head (adapted from changh95/qwen3.8-27b-p300x2
code/qwen36_resident/weights.py, Apache-2.0). Raw caches (--raw) quantize the checkpoint matrices as tt/decoder.py
does, so every BF4 / BF8 block equals the ttnn path's, and the slices apply those norm weights to the activations
(vectors.pt: "folded" False, "pre" per layer). Every matrix is laid out per DRAM bank (common.BankLayout) and
quantized (BF4 projections, BF8 LM head) on the host; the cache holds the device-ready host tensors.

Run as a module to build the cache (host only; set TT_METAL_MOCK_CLUSTER_DESC_PATH so ttnn does not open the chips):
    python -m models.demos.gemma4_31b_qb2.resident_m3_2.weights --cache DIR [--layers N]
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

from models.demos.gemma4_31b_qb2.resident_m3_2.common import (
    E_DOWN,
    E_GU,
    E_HEAD,
    E_O_F,
    E_O_S,
    E_QKV_F,
    E_QKV_S,
    ENTRIES,
    HIDDEN,
    IC,
    LAYERS,
    N_CHIPS,
    Q_F,
    Q_S,
    SLOT_BYTES,
    VC,
    is_full,
    layout_of,
)

PREFIX = "model.language_model."


class Checkpoint:
    def __init__(self, path=None):
        self.path = Path(path or os.environ["HF_MODEL"])
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        self.config = json.loads((self.path / "config.json").read_text())["text_config"]

    def get(self, name):
        from safetensors import safe_open

        with safe_open(self.path / self.index[name], framework="pt") as f:
            return f.get_tensor(name)

    def layer(self, i, name):
        return self.get(f"{PREFIX}layers.{i}.{name}")


def _fold(norm_w, w_t):
    """rows of w_t [K, N] scaled by norm_w [K] (Gemma4 RMSNorm multiplies by w, not 1 + w), as bf16"""
    return (norm_w.float()[:, None] * w_t.float()).bfloat16()


def layer_mats(ck, l, fold=True):
    """per chip {entry: [K, N] bf16} of layer l, plus (post-attention norm, post-feedforward norm, layer scalar)"""
    full = is_full(l)
    ln_in = ck.layer(l, "input_layernorm.weight")
    ln_ff = ck.layer(l, "pre_feedforward_layernorm.weight")
    q = ck.layer(l, "self_attn.q_proj.weight")
    k = ck.layer(l, "self_attn.k_proj.weight")
    v = None if full else ck.layer(l, "self_attn.v_proj.weight")
    o = ck.layer(l, "self_attn.o_proj.weight")
    g, u, d = (ck.layer(l, f"mlp.{p}_proj.weight") for p in ("gate", "up", "down"))
    qc = Q_F if full else Q_S
    kc = k.shape[0] // N_CHIPS
    chips = []
    for c in range(N_CHIPS):
        rows = [q[c * qc : (c + 1) * qc], k[c * kc : (c + 1) * kc]]
        if v is not None:
            rows.append(v[c * kc : (c + 1) * kc])
        cols = slice(c * IC, (c + 1) * IC)
        if fold:
            qkv = _fold(ln_in, torch.cat(rows).T)
            gu = torch.cat([_fold(ln_ff, g[cols].T), _fold(ln_ff, u[cols].T)], dim=1)
        else:
            qkv = torch.cat(rows).T.contiguous()
            gu = torch.cat([g[cols].T, u[cols].T], dim=1).contiguous()
        chips.append(
            {
                E_QKV_F if full else E_QKV_S: qkv,
                E_O_F if full else E_O_S: o[:, c * qc : (c + 1) * qc].T.contiguous(),
                E_GU: gu,
                E_DOWN: d[:, cols].T.contiguous(),
            }
        )
    post = (ck.layer(l, "post_attention_layernorm.weight"), ck.layer(l, "post_feedforward_layernorm.weight"))
    scalar = float(ck.layer(l, "layer_scalar").float().item())
    return chips, post, scalar


def head_mats(ck, fold=True):
    emb = ck.get(PREFIX + "embed_tokens.weight")  # [V, H]
    ln = ck.get(PREFIX + "norm.weight")
    if not fold:
        return [emb[c * VC : (c + 1) * VC].T.contiguous() for c in range(N_CHIPS)], emb
    return [_fold(ln, emb[c * VC : (c + 1) * VC].T.contiguous()) for c in range(N_CHIPS)], emb


def entry_path(cache, l, e, c):
    name = ENTRIES[e][0]
    return Path(cache) / ("head" if e == E_HEAD else f"L{l:02d}") / f"{name}_c{c}.tensorbin"


def quantize_layout(e, w):
    import ttnn

    dt = ttnn.bfloat4_b if ENTRIES[e][2] == "bf4" else ttnn.bfloat8_b
    return ttnn.from_torch(layout_of(e)(w), dtype=dt, layout=ttnn.TILE_LAYOUT)


def convert(cache, layers=LAYERS, head=True, embed=True, fold=True, reuse=None):
    """reuse: a cache of the same kind (or any, for the raw o_proj / down entries, which are never folded) whose files
    are linked instead of converted again"""
    import ttnn

    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    ck = Checkpoint()
    vec_path = cache / "vectors.pt"
    vectors = torch.load(vec_path) if vec_path.exists() else {"post": {}, "scalar": {}}
    assert vectors.get("folded", fold) == fold, "cache was built with the other folding"
    vectors["folded"] = fold
    vectors.setdefault("pre", {})
    for l in range(layers):
        if l not in vectors["pre"]:
            vectors["pre"][l] = tuple(
                ck.layer(l, n).bfloat16() for n in ("input_layernorm.weight", "pre_feedforward_layernorm.weight")
            )
    for l in range(layers):
        if all(entry_path(cache, l, e, c).exists() for e in _layer_entries(l) for c in range(N_CHIPS)) and (
            l in vectors["scalar"]
        ):
            continue
        t0 = time.time()
        chips, post, scalar = layer_mats(ck, l, fold)
        (cache / f"L{l:02d}").mkdir(parents=True, exist_ok=True)
        for c, mats in enumerate(chips):
            for e, w in mats.items():
                dst = entry_path(cache, l, e, c)
                if dst.exists():
                    continue
                if reuse is not None and e in (E_O_S, E_O_F, E_DOWN) and entry_path(reuse, l, e, c).exists():
                    os.symlink(entry_path(reuse, l, e, c), dst)
                    continue
                ttnn.dump_tensor(str(dst), quantize_layout(e, w))
        vectors["post"][l] = tuple(p.bfloat16() for p in post)
        vectors["scalar"][l] = scalar
        torch.save(vectors, vec_path)
        print(f"layer {l}: {time.time() - t0:.1f} s", flush=True)
    if head and not all(entry_path(cache, 0, E_HEAD, c).exists() for c in range(N_CHIPS)):
        t0 = time.time()
        heads, emb = head_mats(ck, fold)
        (cache / "head").mkdir(parents=True, exist_ok=True)
        for c, w in enumerate(heads):
            ttnn.dump_tensor(str(entry_path(cache, 0, E_HEAD, c)), quantize_layout(E_HEAD, w))
        print(f"head: {time.time() - t0:.1f} s", flush=True)
    else:
        emb = None
    if embed and not (cache / "embed_scaled.pt").exists() and reuse is not None and (Path(reuse) / "embed_scaled.pt").exists():
        os.symlink(Path(reuse) / "embed_scaled.pt", cache / "embed_scaled.pt")
    if embed and not (cache / "embed_scaled.pt").exists():
        emb = ck.get(PREFIX + "embed_tokens.weight") if emb is None else emb
        # Gemma4TextScaledWordEmbedding: embedding * sqrt(hidden) in the weight dtype (bf16(73.32) = 73.5)
        scale = torch.tensor(HIDDEN**0.5, dtype=torch.bfloat16)
        torch.save((emb * scale).contiguous(), cache / "embed_scaled.pt")
    if "final_norm" not in vectors:
        vectors["final_norm"] = ck.get(PREFIX + "norm.weight").bfloat16()
        torch.save(vectors, vec_path)


def _layer_entries(l):
    return (E_QKV_F, E_O_F, E_GU, E_DOWN) if is_full(l) else (E_QKV_S, E_O_S, E_GU, E_DOWN)


def load_host(cache, l, e):
    """the 4 chips' cached host tensors of (layer, entry)"""
    import ttnn

    return [ttnn.load_tensor(str(entry_path(cache, l, e, c))) for c in range(N_CHIPS)]


def dequantized(cache, l, e):
    """the 4 chips' [K, N] matrices as the device holds them (bf16, exact)"""
    import ttnn

    lay = layout_of(e)
    N = ENTRIES[e][3] * 32
    return [lay.inverse(ttnn.to_torch(t).bfloat16(), N) for t in load_host(cache, l, e)]


def post_norm_rows(vectors, layers):
    """[2 * layers, SLOT_BYTES / 2] bf16: round 2l + 1 (post-attention) and 2l + 2 (post-feedforward) of the hub,
    stored as rows 2l and 2l + 1, zero padded to the hub's 6 views"""
    out = torch.zeros(2 * layers, SLOT_BYTES // 2, dtype=torch.bfloat16)
    for l in range(layers):
        pa, pf = vectors["post"][l]
        out[2 * l, :HIDDEN] = pa
        out[2 * l + 1, :HIDDEN] = pf
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--layers", type=int, default=LAYERS)
    p.add_argument("--no-head", action="store_true")
    p.add_argument("--no-embed", action="store_true")
    p.add_argument("--raw", action="store_true", help="unfolded norms: the ttnn path's exact BF4 / BF8 blocks")
    p.add_argument("--reuse", default=None, help="link the o_proj / down entries and the embedding from this cache")
    a = p.parse_args()
    convert(a.cache, a.layers, head=not a.no_head, embed=not a.no_embed, fold=not a.raw, reuse=a.reuse)
