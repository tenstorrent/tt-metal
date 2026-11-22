# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dimensions and per-bank weight layout of the resident DFlash2 drafter (M6b; bench/resident/m6/M6A_DESIGN.md + its
addenda). RedHatAI/gemma-4-31B-it-speculator.dflash2 @ b0cd01b4: five Qwen3 layers (32 Q / 16 KV heads x 256, MLP
21504, sliding window 2048 anchor-relative, non-causal in the block, RoPE theta 1e4), two dynamic 2-tap block convs per
layer, fc over five target taps, a candidate selector over the target's LM head.

TP4 as tt/dflash2.py: chip c holds q heads [8c, 8c+8), KV heads [4c, 4c+4) and MLP columns [5376c, 5376(c+1)). The
projections have the target's sliding-layer shapes, so they stream exactly as the target's (common.bank_core_cols /
gate_up_core_cols, BankLayout), in BF8 (matmul_block at HiFi2; custom_mm is LoFi-only). The fc is input-sharded
(addendum 1 §2): chip c multiplies its 1344 columns of each tap, and the chips' partials are all-reduced. The conv
kernel projections and the selector's hidden projection are replicated (every chip holds the whole residual)."""
import json
from pathlib import Path

from models.demos.gemma4_31b_qb2.resident_m7.common import (
    BF8_TILE,
    GU_EXTRA,
    HIDDEN,
    HT,
    IT,
    N_CHIPS,
    TILE,
    BankLayout,
    bank_core_cols,
    block_geometry,
    gate_up_core_cols,
)

DFLASH2_ID = "RedHatAI/gemma-4-31B-it-speculator.dflash2"
DFLASH2_REVISION = "b0cd01b4918aae6b6388272950c4f9851734674b"

D_LAYERS = 5
D_HEADS, D_KV_HEADS, D_HEAD_DIM = 8, 4, 256  # per chip
D_Q = D_HEADS * D_HEAD_DIM  # 2048 q columns per chip
D_KV = D_KV_HEADS * D_HEAD_DIM  # 1024 k (and v) columns per chip
D_TAPS = 5
TAP_COLS = HIDDEN // N_CHIPS  # 1344: a chip's quarter of one tap (the sharded lazy ring's row width)
GROUP, GROUPS = 16, HIDDEN // 16  # conv groups: 336 of 16 channels
SECTION = 384  # one (io, tap) section of the conv kernel projection, 336 groups padded to 12 tiles
KP_COLS = 4 * SECTION  # 1536: sections (in tap 0, in tap 1, out tap 0, out tap 1)
RANK = 256  # selector rank
TOP_K = 16
WINDOW = 2048
RING = 2304  # drafter ring positions per (slot, layer, kv head): 72 pages of 32 (M6A addendum 1 §4.5)
RING_PAGES = RING // TILE

# drafter weight entries: (name, Kt, N tiles); all BF8
D_ENTRIES = [
    ("d_qkv", HT, (D_Q + 2 * D_KV) // TILE),  # 168 x 128 (the target's qkv_s shape)
    ("d_o", D_Q // TILE, HT),  # 64 x 168
    ("d_gu", HT, 2 * IT),  # 168 x 336: gate | up
    ("d_down", IT, HT),  # 168 x 168
    ("d_kp_attn", HT, KP_COLS // TILE),  # 168 x 48, replicated
    ("d_kp_mlp", HT, KP_COLS // TILE),  # 168 x 48, replicated
    ("d_kv", HT, 2 * D_KV // TILE),  # 168 x 64: the ingest's k | v columns (a copy of d_qkv's)
    ("d_fc", D_TAPS * TAP_COLS // TILE, HT),  # 210 x 168, input-sharded
    ("d_hp", HT, RANK // TILE),  # 168 x 8, replicated
]
D_QKV, D_O, D_GU, D_DOWN, D_KP_ATTN, D_KP_MLP, D_KV_E, D_FC, D_HP = range(len(D_ENTRIES))
PER_LAYER = (D_QKV, D_O, D_GU, D_DOWN, D_KP_ATTN, D_KP_MLP, D_KV_E)
GLOBAL = (D_FC, D_HP)


def d_core_cols(e, banks=8):
    """per bank, per streamer core: the column tiles it streams (as common.entry_core_cols)"""
    _, _, N = D_ENTRIES[e]
    if e == D_GU:
        return gate_up_core_cols(IT, banks, GU_EXTRA)
    return bank_core_cols(N, banks)


def d_geometry(e):
    """(sb, pages, page, block) of entry e (common.block_geometry, BF8 tiles)"""
    return block_geometry(D_ENTRIES[e][1], BF8_TILE)


def d_layout(e, banks=8):
    return BankLayout(D_ENTRIES[e][1], d_core_cols(e, banks), d_geometry(e)[0])


def kp_columns(io, tap, group):
    """column of the conv kernel projection holding (io, tap, group) (io 0: conv in, 1: conv out)"""
    return (2 * io + tap) * SECTION + group


def checkpoint_dir():
    """the pinned snapshot in the local HF cache (serving runs with HF_HUB_OFFLINE=1)"""
    from huggingface_hub.constants import HF_HUB_CACHE

    path = Path(HF_HUB_CACHE) / f"models--{DFLASH2_ID.replace('/', '--')}" / "snapshots" / DFLASH2_REVISION
    if not (path / "model.safetensors").exists():
        raise FileNotFoundError(f"{DFLASH2_ID} @ {DFLASH2_REVISION} is not in the local HF cache: {path}")
    return path


def drafter_config(path=None):
    cfg = json.loads((Path(path or checkpoint_dir()) / "config.json").read_text())
    tl = cfg["transformer_layer_config"]
    assert tl["num_hidden_layers"] == D_LAYERS and tl["hidden_size"] == HIDDEN
    assert tl["num_attention_heads"] == N_CHIPS * D_HEADS and tl["num_key_value_heads"] == N_CHIPS * D_KV_HEADS
    assert tl["head_dim"] == D_HEAD_DIM and tl["sliding_window"] == WINDOW and tl["intermediate_size"] == N_CHIPS * IT * TILE
    assert cfg["conv_group_size"] == GROUP and cfg["conv_kernel_size"] == 2 and cfg["selector_rank"] == RANK
    assert cfg["selector_top_k"] == TOP_K and len(cfg["aux_hidden_state_layer_ids"]) == D_TAPS
    return cfg


def tap_layers(cfg):
    """target layers whose output residual the drafter reads (speculators ids are HF hidden_states indices)"""
    return [layer - 1 for layer in cfg["aux_hidden_state_layer_ids"]]
