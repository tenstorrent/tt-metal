# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Chunked prefill for one MiMo decoder layer of each type on the Galaxy (TP=8 x DP=4).

Every replica prefills the same four prompts (200 to 1333 tokens), each in chunks of a different size:
64, 128, 256 and 512 tokens on rows 0-3, so chunks are both shorter and longer than the 128-token window.
Every chunk after the first reads the earlier ones from the paged KV cache. The test checks:

* the chunked output against the fp32 reference (which sees the whole prompt at once) and against a
  one-shot prefill of the same prompt on the same row;
* the K/V the chunks wrote, position by position, against the reference K/V;
* a few decode steps that read that cache, against the reference.

The caches start out full of random values instead of zeros, so a kernel that reads a block the user
never wrote, or a key outside the window, shows up as an error. SWA layers get vLLM ``SlidingWindowSpec``
tables: before each chunk and decode step, every block wholly behind the window points at a shared null
block (block 0), and blocks past the chunk are not mapped yet.
"""

import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.reference import model as ref
from models.experimental.mimo_v2_6_flash.tests.common import (
    DP,
    GALAXY_DEVICE_PARAMS,
    TP,
    cache_dir,
    dp_submeshes,
    load_config_and_ckpt,
    pcc,
)
from models.experimental.mimo_v2_6_flash.tt import kv as kv_lib
from models.experimental.mimo_v2_6_flash.tt.attention import kv_heads_for_chip
from models.experimental.mimo_v2_6_flash.tt.common import MeshPlan
from models.experimental.mimo_v2_6_flash.tt.decoder import TtMiMoDecoderLayer
from models.experimental.mimo_v2_6_flash.tt.model import prefill_padded_len
from models.experimental.mimo_v2_6_flash.tt.rope import MiMoRotarySetup, meta_rope_permutation

MAX_SEQ = 2048
MAX_BATCH = 32
BLOCK = 64
LOGICAL_BLOCKS = MAX_SEQ // BLOCK
PROMPT_LENS = (200, 517, 960, 1333)
CHUNKS = (64, 128, 256, 512)  # chunk size on replica rows 0..3
DECODE_STEPS = 3
HIDDEN_STD = 0.5
NULL_BLOCK = 0
KV_DTYPE = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[os.environ.get("MIMO_KV_DTYPE", "bfp8")]


def to_tt(x, plan):
    return ttnn.from_torch(
        x.reshape(1, 1, *x.shape[-2:]).bfloat16(),
        device=plan.mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=plan.replicate(),
    )


def replicated(plan, t, dtype, layout):
    return ttnn.from_torch(t, device=plan.mesh_device, dtype=dtype, layout=layout, mesh_mapper=plan.replicate())


def garbage_cache(plan, num_blocks, head_dim, seed):
    """[K, V] caches filled with random values: reading a block nobody wrote gives a visible error."""
    g = torch.Generator().manual_seed(seed)
    return [
        ttnn.from_torch(
            torch.randn(num_blocks, 1, BLOCK, head_dim, generator=g) * 4,
            device=plan.mesh_device,
            dtype=KV_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=plan.replicate(),
        )
        for _ in range(2)
    ]


def swa_physical(slot, block):
    """Physical block of a SWA slot's logical block: no reuse, so every written position stays readable."""
    return 1 + slot * LOGICAL_BLOCKS + block


def set_swa_row(table, slot, lo, hi, window):
    """vLLM SlidingWindowSpec row: blocks overlapping [lo - window + 1, hi) are mapped, the rest are null."""
    first = max(0, lo - window + 1) // BLOCK
    last = math.ceil(hi / BLOCK)
    table[:, slot] = NULL_BLOCK
    for b in range(first, last):
        table[:, slot, b] = swa_physical(slot, b)


class Row:
    """One replica (Galaxy row) with its layer, one-shot and chunked caches, and their page tables."""

    def __init__(self, sm, cfg, layer_idx, weights, dims, r):
        self.plan = plan = MeshPlan(sm)
        self.ropes = {
            "swa": MiMoRotarySetup(sm, MAX_BATCH, dims.head_dim, dims.rope_dim, cfg.swa_rope_theta),
            "full": MiMoRotarySetup(sm, MAX_BATCH, dims.head_dim, dims.rope_dim, cfg.rope_theta),
        }
        # Replica 0 converts the weights and fills the cache; the other rows load the cached tensors.
        self.layer = TtMiMoDecoderLayer(plan, cfg, layer_idx, weights, self.ropes, cache_dir=cache_dir("layer_bfp8"))
        self.window = dims.sliding_window
        self.chunk = CHUNKS[r]
        n = len(PROMPT_LENS)
        if self.window is None:
            tables, num_blocks = kv_lib.page_tables(n, MAX_BATCH, LOGICAL_BLOCKS, TP, self.layer.attn.rep, seed=r)
            self.tables = {"oneshot": tables, "chunked": tables}
        else:
            num_blocks = 1 + n * LOGICAL_BLOCKS
            full = torch.full((TP, MAX_BATCH, LOGICAL_BLOCKS), NULL_BLOCK, dtype=torch.int32)
            for u in range(n):
                set_swa_row(full, u, MAX_SEQ, MAX_SEQ, MAX_SEQ)
            self.tables = {"oneshot": full, "chunked": full.clone().fill_(NULL_BLOCK)}
        self.kv = {
            mode: garbage_cache(plan, num_blocks, dims.head_dim, seed=100 * r + i)
            for i, mode in enumerate(("oneshot", "chunked"))
        }
        self.tables_tt = {}
        for mode in self.tables:
            self.upload(mode)

    def upload(self, mode):
        if mode in self.tables_tt:
            self.tables_tt[mode].deallocate(True)
        self.tables_tt[mode] = kv_lib.per_chip(self.plan, self.tables[mode], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

    def prefill(self, mode, x, slot, start_pos):
        n = x.shape[0]
        padded = prefill_padded_len(n)
        if mode == "chunked" and self.window is not None:
            set_swa_row(self.tables[mode], slot, start_pos, start_pos + padded, self.window)
            self.upload(mode)
        x_tt = to_tt(torch.nn.functional.pad(x, (0, 0, 0, padded - n)), self.plan)
        out = self.layer.forward_prefill(
            x_tt, slot, self.tables_tt[mode], self.kv[mode], real_len=n, start_pos=start_pos
        )
        x_tt.deallocate(True)
        return out, n

    def read_kv(self, slot, length, ref_k, ref_v, num_kv_heads):
        """Worst PCC of the chunked cache's K and V vs the reference over the chips that hold this slot."""
        perm = meta_rope_permutation(ref_k.shape[-1], self.layer.attn.dims.rope_dim)
        caches = [[ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(c)] for c in self.kv["chunked"]]
        worst_k, worst_v, pad_max = 1.0, 1.0, 0.0
        rep = self.layer.attn.rep
        for chip in range(TP):
            if slot % rep != chip % rep:
                continue  # the other chip of the pair does not hold this user
            if self.window is None:
                blocks = self.tables["chunked"][chip, slot]
            else:
                blocks = torch.tensor([swa_physical(slot, b) for b in range(LOGICAL_BLOCKS)])
            head = kv_heads_for_chip(chip, TP, num_kv_heads)[0]
            k = caches[0][chip][blocks, 0].reshape(-1, caches[0][chip].shape[-1])[:length]
            v = caches[1][chip][blocks, 0].reshape(-1, caches[1][chip].shape[-1])[:length]
            vd = ref_v.shape[-1]
            worst_k = min(worst_k, pcc(k, ref_k[head][:, perm]))
            worst_v = min(worst_v, pcc(v[:, :vd], ref_v[head]))
            pad_max = max(pad_max, v[:, vd:].abs().max().item())
        return worst_k, worst_v, pad_max


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 1, 5])
def test_chunked_prefill_tp8_dp4(mesh_device, layer_idx):
    torch.manual_seed(4321 + layer_idx)
    cfg, ckpt = load_config_and_ckpt()
    weights = ref.load_layer_weights(ckpt, cfg, layer_idx)
    ref_layer = ref.ReferenceDecoderLayer(cfg, layer_idx, weights=weights)
    dims = cfg.attention_dims(layer_idx)
    logger.info(f"layer {layer_idx}: swa={dims.is_swa} moe={cfg.is_moe(layer_idx)} kv_heads={dims.num_kv_heads}")

    rows = [
        Row(sm, cfg, layer_idx, weights if r == 0 else None, dims, r) for r, sm in enumerate(dp_submeshes(mesh_device))
    ]

    # The same prompts on every row; the reference sees each whole prompt once.
    users = []
    for length in PROMPT_LENS:
        x = (torch.randn(length, cfg.hidden_size) * HIDDEN_STD).bfloat16().float()
        hist = ref.KVHistory()
        with torch.no_grad():
            y = ref_layer(x, torch.arange(length), hist)
        users.append(dict(x=x, ref=y, hist=hist, ref_k=hist.k.clone(), ref_v=hist.v.clone()))

    # ---- prefill: one shot and in chunks, every row enqueued before anything is read
    pending = []
    for u, usr in enumerate(users):
        for r, row in enumerate(rows):
            outs = [row.prefill("oneshot", usr["x"], u, 0)]
            for s in range(0, usr["x"].shape[0], row.chunk):
                outs.append(row.prefill("chunked", usr["x"][s : s + row.chunk], u, s))
            pending.append((r, u, outs))

    worst = dict(chunked=1.0, chunked_delta=1.0, vs_oneshot=1.0, vs_oneshot_delta=1.0, decode=1.0, decode_delta=1.0)
    for r, u, outs in pending:
        plan, usr = rows[r].plan, users[u]
        host = [plan.to_torch_replicated(o).float()[0, 0, :n] for o, n in outs]
        for o, _ in outs:
            o.deallocate(True)
        oneshot, chunked = host[0], torch.cat(host[1:])
        x, y = usr["x"], usr["ref"]
        result = dict(
            chunked=pcc(chunked, y),
            chunked_delta=pcc(chunked - x, y - x),
            vs_oneshot=pcc(chunked, oneshot),
            vs_oneshot_delta=pcc(chunked - x, oneshot - x),
        )
        for key, value in result.items():
            worst[key] = min(worst[key], value)
        logger.info(
            f"row {r} chunk {rows[r].chunk} slot {u} len {x.shape[0]}: "
            + ", ".join(f"{k} {v:.5f}" for k, v in result.items())
            + f" (one-shot vs ref {pcc(oneshot, y):.5f})"
        )

    # ---- the K/V the chunks wrote
    worst_kv = dict(k=1.0, v=1.0, v_pad=0.0)
    for r, row in enumerate(rows):
        for u, usr in enumerate(users):
            k, v, pad = row.read_kv(u, usr["x"].shape[0], usr["ref_k"], usr["ref_v"], dims.num_kv_heads)
            worst_kv["k"], worst_kv["v"] = min(worst_kv["k"], k), min(worst_kv["v"], v)
            worst_kv["v_pad"] = max(worst_kv["v_pad"], pad)
    logger.info(f"chunked KV vs reference, worst over rows/users/chips: {worst_kv}")

    # ---- decode from the chunked cache; the same inputs on every row, so one reference serves all four
    decode_pool = {r: ([], []) for r in range(DP)}
    for step in range(DECODE_STEPS):
        pos = torch.full((MAX_BATCH,), -1, dtype=torch.int32)
        x = torch.zeros(MAX_BATCH, cfg.hidden_size)
        for u, usr in enumerate(users):
            p = usr["x"].shape[0] + step
            xu = (torch.randn(cfg.hidden_size) * HIDDEN_STD).bfloat16().float()
            with torch.no_grad():
                usr["ref_decode"] = ref_layer(xu[None], torch.tensor([p]), usr["hist"])[0]
            usr["x_decode"] = xu
            x[u], pos[u] = xu, p
        outs = []
        for row in rows:
            plan = row.plan
            if row.window is not None:
                for u in range(len(users)):
                    set_swa_row(row.tables["chunked"], u, int(pos[u]), int(pos[u]) + 1, row.window)
                row.upload("chunked")
            rot = {
                True: row.ropes["swa"].get_rot_mats(pos.clamp(min=0)),
                False: row.ropes["full"].get_rot_mats(pos.clamp(min=0)),
            }
            rep = row.layer.attn.rep
            cur_pos = {
                "pos": replicated(plan, pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                "pos_owned": kv_lib.per_chip(
                    plan, kv_lib.owned_positions(pos, TP, rep), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
                ),
                "owner_mask": kv_lib.per_chip(plan, kv_lib.owner_mask(pos, TP, rep), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            }
            mask = replicated(plan, (pos >= 0).bfloat16().reshape(1, 1, -1, 1), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            outs.append(
                row.layer.forward_decode(
                    to_tt(x, plan), rot, cur_pos, row.tables_tt["chunked"], row.kv["chunked"], mask
                )
            )
        for r, out in enumerate(outs):
            got = rows[r].plan.to_torch_replicated(out).float()[0, 0]
            for u, usr in enumerate(users):
                xd, rd = usr["x_decode"], usr["ref_decode"]
                p, pd = pcc(got[u], rd), pcc(got[u] - xd, rd - xd)
                worst["decode"] = min(worst["decode"], p)
                decode_pool[r][0].append(got[u] - xd)
                decode_pool[r][1].append(rd - xd)
                logger.info(f"decode step {step} row {r} slot {u} pos {int(pos[u])}: pcc {p:.5f} delta {pd:.5f}")
    # One decode token's delta is dominated by its MoE output, and a token whose 8th and 9th expert scores
    # are within the router matmul's error (~2^-12 relative) picks a different expert on device. As in
    # test_decoder_tp8_dp4.py, the delta is checked over all of a row's decode tokens together.
    for r, (got_d, ref_d) in decode_pool.items():
        worst["decode_delta"] = min(worst["decode_delta"], pcc(torch.stack(got_d), torch.stack(ref_d)))

    logger.info(f"layer {layer_idx} worst over 4 rows x {len(users)} prompts: {worst}")
    # Same bounds as test_decoder_tp8_dp4.py: ~1-2% of MoE tokens flip one expert at a near-tie.
    assert worst["chunked"] > 0.995 and worst["decode"] > 0.995, worst
    assert worst["chunked_delta"] > 0.99 and worst["decode_delta"] > 0.99, worst
    assert worst["vs_oneshot"] > 0.995 and worst["vs_oneshot_delta"] > 0.99, worst
    assert worst_kv["k"] > 0.999 and worst_kv["v"] > 0.999, worst_kv
    assert worst_kv["v_pad"] == 0.0, "V padding columns must stay zero"
