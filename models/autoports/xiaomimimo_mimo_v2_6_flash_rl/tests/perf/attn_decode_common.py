# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared setup for the MiMo attention decode perf / correctness scripts (Galaxy row 0, TP=8, 4 users).

The measured unit is ``TtMiMoDecoderLayer.attention_decode``: input RMSNorm, attention (qkv projection,
heads, RoPE, paged KV update, SDPA decode, o projection, row all-reduce) and the residual add. Two layers
cover both kinds: layer 1 (sliding window 128, 8 KV heads: one per chip) and layer 5 (global, 4 KV heads:
each chip pair splits the users). Users are prefilled with random activations to 96..704 tokens, so the
SWA layer sees its window and its 512-token ring wrap.
"""

import os

import torch

import ttnn
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tests.perf import moe_decode_common as common
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tt import kv as kv_lib
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tt.common import MeshPlan
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tt.decoder import TtMiMoDecoderLayer
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tt.rope import MiMoRotarySetup

LAYERS = (1, 5)  # SWA, global
USERS = 4
MAX_BATCH = 32
BLOCK = 64
MAX_SEQ = 4096
PROMPT_LENS = (96, 224, 416, 704)
HIDDEN_STD = 0.5


def layer_weights(cfg):
    """Share of SWA and global layers in the model (39 and 9 of 48 for MiMo-V2.6-Flash)."""
    n = cfg.num_hidden_layers
    swa = sum(cfg.is_swa(i) for i in range(n)) / n
    return {LAYERS[0]: swa, LAYERS[1]: 1.0 - swa}


def build(row, cfg, layer_idx, decode_users=None):
    """Decoder layer ``layer_idx`` from the model's tensor cache, its paged KV cache and page table.
    ``decode_users``: set up the MoE's decode path for that many users (the full-layer scripts)."""
    plan = MeshPlan(row)
    dims = cfg.attention_dims(layer_idx)
    ropes = {
        "swa": MiMoRotarySetup(row, MAX_BATCH, dims.head_dim, dims.rope_dim, cfg.swa_rope_theta),
        "full": MiMoRotarySetup(row, MAX_BATCH, dims.head_dim, dims.rope_dim, cfg.rope_theta),
    }
    layer = TtMiMoDecoderLayer(
        plan,
        cfg,
        layer_idx,
        None,
        ropes,
        cache_dir=os.path.join(common.CACHE_ROOT, "model_bfp4experts"),
        expert_dtype=ttnn.bfloat4_b,
    )
    blocks = layer.attn.kv_tokens_per_user(MAX_SEQ) // BLOCK
    tables, num_blocks = kv_lib.page_tables(USERS, MAX_BATCH, blocks, plan.tp, layer.attn.rep, seed=layer_idx)
    kv = layer.attn.allocate_kv_cache(num_blocks, BLOCK, ttnn.bfloat8_b)
    page_table = kv_lib.per_chip(plan, tables, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    if decode_users is not None and layer.is_moe:
        layer.mlp.set_decode_users(decode_users)
    return dict(plan=plan, ropes=ropes, layer=layer, kv=kv, page_table=page_table, row=row)


def to_tt(state, x):
    return ttnn.from_torch(
        x.reshape(1, 1, *x.shape[-2:]).bfloat16(),
        device=state["row"],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=state["plan"].replicate(),
    )


def prefill(state, cfg, seed):
    """Fill every user's KV cache: input norm, attention prefill, residual add. Returns the outputs [len, hidden]."""
    g = torch.Generator().manual_seed(seed)
    layer, outs = state["layer"], []
    for u, n in enumerate(PROMPT_LENS):
        x = (torch.randn(n, cfg.hidden_size, generator=g) * HIDDEN_STD).bfloat16()
        x_tt = to_tt(state, x)
        h = layer.input_norm(x_tt)
        attn = layer.attn.forward_prefill(h, u, layer._page_table(state["page_table"]), state["kv"])
        y = ttnn.add(x_tt, attn)
        outs.append(state["plan"].to_torch_replicated(y).float()[0, 0, :n])
        for t in (x_tt, h, attn, y):
            t.deallocate(True)
    return outs


def prefill_layer(state, cfg, seed, return_inputs=False):
    """Fill every user's KV cache through the whole layer (TtMiMoDecoderLayer.forward_prefill: attention and FFN).
    Returns the outputs [len, hidden] (and the inputs with ``return_inputs``)."""
    g = torch.Generator().manual_seed(seed)
    layer, outs, xs = state["layer"], [], []
    for u, n in enumerate(PROMPT_LENS):
        x = (torch.randn(n, cfg.hidden_size, generator=g) * HIDDEN_STD).bfloat16()
        x_tt = to_tt(state, x)
        y = layer.forward_prefill(x_tt, u, state["page_table"], state["kv"])
        outs.append(state["plan"].to_torch_replicated(y).float()[0, 0, :n])
        xs.append(x.float())
        x_tt.deallocate(True)
        y.deallocate(True)
    return (outs, xs) if return_inputs else outs


def token_mask(state):
    """[1, 1, 32, 1] bf16: 1 for the USERS active decode slots."""
    m = torch.zeros(1, 1, MAX_BATCH, 1)
    m[0, 0, :USERS] = 1
    return ttnn.from_torch(
        m, device=state["row"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=state["plan"].replicate()
    )


def decode_layer(state, x_tt, rot, cur_pos, mask):
    return state["layer"].forward_decode(x_tt, rot, cur_pos, state["page_table"], state["kv"], mask)


def decode_inputs(state, cfg, step, seed):
    """Device inputs of decode step ``step`` (users at PROMPT_LENS + step): (x, rot mats, positions), host x."""
    plan, row = state["plan"], state["row"]
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(MAX_BATCH, cfg.hidden_size)
    x[:USERS] = torch.randn(USERS, cfg.hidden_size, generator=g) * HIDDEN_STD
    x = x.bfloat16()
    pos = torch.full((MAX_BATCH,), -1, dtype=torch.int32)
    pos[:USERS] = torch.tensor(PROMPT_LENS, dtype=torch.int32) + step
    rot = {
        True: state["ropes"]["swa"].get_rot_mats(pos.clamp(min=0)),
        False: state["ropes"]["full"].get_rot_mats(pos.clamp(min=0)),
    }
    rep = state["layer"].attn.rep
    cur_pos = {
        "pos": ttnn.from_torch(
            pos, device=row, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=plan.replicate()
        ),
        "pos_owned": kv_lib.per_chip(
            plan, kv_lib.owned_positions(pos, plan.tp, rep), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
        ),
        "owner_mask": kv_lib.per_chip(plan, kv_lib.owner_mask(pos, plan.tp, rep), ttnn.bfloat16, ttnn.TILE_LAYOUT),
    }
    return to_tt(state, x), rot, cur_pos, x.float()


def decode(state, x_tt, rot, cur_pos):
    return state["layer"].attention_decode(x_tt, rot, cur_pos, state["page_table"], state["kv"])
