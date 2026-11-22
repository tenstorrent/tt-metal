# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DFlash2 checkpoint -> the resident drafter's per-chip, per-bank BF8 weight cache (M6b; drafter_common.py).

Every matrix is laid out per DRAM bank in streamer_reader's shape (common.BankLayout, as the target's raw cache) and
quantized to BF8 on the host; the cache holds device-ready host tensors, one file per (entry, layer, chip). The BF8
blocks equal tt/dflash2.py's (bfloat8_b of the same bf16 matrices; the layout permutes whole tiles). vectors.pt holds
the norm weights, the conv base kernels and the RoPE check; selector.pt the codebooks (bf16, host selector).

    TT_METAL_MOCK_CLUSTER_DESC_PATH=... python -m models.demos.gemma4_31b_qb2.resident_m7.drafter_weights --cache DIR
"""
import argparse
import time
from pathlib import Path

import torch

from models.demos.gemma4_31b_qb2.resident_m7.common import HIDDEN, IC, N_CHIPS
from models.demos.gemma4_31b_qb2.resident_m7.drafter_common import (
    D_ENTRIES,
    D_FC,
    D_GU,
    D_HP,
    D_KP_ATTN,
    D_KP_MLP,
    D_KV,
    D_KV_E,
    D_LAYERS,
    D_O,
    D_Q,
    D_QKV,
    D_DOWN,
    GROUPS,
    KP_COLS,
    PER_LAYER,
    SECTION,
    TAP_COLS,
    D_TAPS,
    checkpoint_dir,
    d_layout,
    drafter_config,
)


def load_state(path=None):
    from safetensors import safe_open

    sd = {}
    with safe_open(Path(path or checkpoint_dir()) / "model.safetensors", framework="pt") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
    return sd


def kp_matrix(w):
    """kernel_projection [4 * GROUPS, H] (rows io, tap, group) -> [H, KP_COLS]: section s = 2 io + tap at columns
    [s * SECTION, s * SECTION + GROUPS), zero padded"""
    out = torch.zeros(HIDDEN, KP_COLS, dtype=w.dtype)
    sec = w.reshape(4, GROUPS, HIDDEN)
    for s in range(4):
        out[:, s * SECTION : s * SECTION + GROUPS] = sec[s].T
    return out


def layer_mats(sd, i):
    """per chip {entry: [K, N] bf16} of drafter layer i"""
    p = f"layers.{i}."
    q, k, v = (sd[p + f"self_attn.{n}_proj.weight"] for n in "qkv")
    o = sd[p + "self_attn.o_proj.weight"]
    g, u, d = (sd[p + f"mlp.{n}_proj.weight"] for n in ("gate", "up", "down"))
    kpa = kp_matrix(sd[p + "attention_conv.kernel_projection.weight"])
    kpm = kp_matrix(sd[p + "mlp_conv.kernel_projection.weight"])
    chips = []
    for c in range(N_CHIPS):
        qc, kc, vc = q[c * D_Q : (c + 1) * D_Q], k[c * D_KV : (c + 1) * D_KV], v[c * D_KV : (c + 1) * D_KV]
        cols = slice(c * IC, (c + 1) * IC)
        chips.append(
            {
                D_QKV: torch.cat([qc, kc, vc]).T.contiguous(),
                D_O: o[:, c * D_Q : (c + 1) * D_Q].T.contiguous(),
                D_GU: torch.cat([g[cols].T, u[cols].T], dim=1).contiguous(),
                D_DOWN: d[:, cols].T.contiguous(),
                D_KP_ATTN: kpa,
                D_KP_MLP: kpm,
                D_KV_E: torch.cat([kc, vc]).T.contiguous(),
            }
        )
    return chips


def fc_rows(c):
    """the fc input rows chip c owns: its TAP_COLS columns of each tap, tap-major"""
    return torch.cat([torch.arange(t * HIDDEN + c * TAP_COLS, t * HIDDEN + (c + 1) * TAP_COLS) for t in range(D_TAPS)])


def global_mats(sd):
    fc_t = sd["fc.weight"].T  # [5 H, H]
    hp = sd["candidate_selector.hidden_projection.weight"].T.contiguous()  # [H, rank]
    return [{D_FC: fc_t[fc_rows(c)].contiguous(), D_HP: hp} for c in range(N_CHIPS)]


def entry_path(cache, e, c, layer=None):
    name = D_ENTRIES[e][0]
    return Path(cache) / ("global" if layer is None else f"L{layer}") / f"{name}_c{c}.tensorbin"


def quantize_layout(e, w):
    import ttnn

    return ttnn.from_torch(d_layout(e)(w), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)


def vectors_of(sd, cfg):
    tl = cfg["transformer_layer_config"]
    v = {"layers": [], "eps": tl["rms_norm_eps"], "theta": float(tl["rope_parameters"]["rope_theta"])}
    for i in range(D_LAYERS):
        p = f"layers.{i}."
        v["layers"].append(
            {
                "input_norm": sd[p + "input_layernorm.weight"].bfloat16(),
                "post_norm": sd[p + "post_attention_layernorm.weight"].bfloat16(),
                "q_norm": sd[p + "self_attn.q_norm.weight"].bfloat16(),
                "k_norm": sd[p + "self_attn.k_norm.weight"].bfloat16(),
                "attn_base": sd[p + "attention_conv.base_kernel"].bfloat16(),  # [io, tap, H]
                "mlp_base": sd[p + "mlp_conv.base_kernel"].bfloat16(),
            }
        )
    v["hidden_norm"] = sd["hidden_norm.weight"].bfloat16()
    v["final_norm"] = sd["norm.weight"].bfloat16()
    v["mask_token"] = cfg["mask_token_id"]
    return v


def convert(cache):
    import ttnn

    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    cfg = drafter_config()
    sd = load_state()
    t0 = time.time()
    for i in range(D_LAYERS):
        (cache / f"L{i}").mkdir(exist_ok=True)
        if all(entry_path(cache, e, c, i).exists() for e in PER_LAYER for c in range(N_CHIPS)):
            continue
        for c, mats in enumerate(layer_mats(sd, i)):
            for e, w in mats.items():
                ttnn.dump_tensor(str(entry_path(cache, e, c, i)), quantize_layout(e, w.bfloat16()))
        print(f"layer {i}: {time.time() - t0:.0f} s", flush=True)
    (cache / "global").mkdir(exist_ok=True)
    for c, mats in enumerate(global_mats(sd)):
        for e, w in mats.items():
            if not entry_path(cache, e, c).exists():
                ttnn.dump_tensor(str(entry_path(cache, e, c)), quantize_layout(e, w.bfloat16()))
    torch.save(vectors_of(sd, cfg), cache / "vectors.pt")
    torch.save(
        {
            "pred_cb": sd["candidate_selector.predecessor_codebook"].bfloat16(),
            "succ_cb": sd["candidate_selector.successor_codebook"].bfloat16(),
        },
        cache / "selector.pt",
    )
    print(f"drafter cache {cache}: {time.time() - t0:.0f} s", flush=True)


def load_host(cache, e, layer=None):
    """the 4 chips' cached host tensors of (entry, layer)"""
    import ttnn

    return [ttnn.load_tensor(str(entry_path(cache, e, c, layer))) for c in range(N_CHIPS)]


def dequantized(cache, e, layer=None):
    """the 4 chips' [K, N] matrices as the device holds them (bf16, exact)"""
    import ttnn

    lay = d_layout(e)
    N = D_ENTRIES[e][2] * 32
    return [lay.inverse(ttnn.to_torch(t).bfloat16(), N) for t in load_host(cache, e, layer)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    convert(ap.parse_args().cache)
