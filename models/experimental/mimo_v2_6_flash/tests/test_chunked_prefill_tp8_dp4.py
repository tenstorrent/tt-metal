# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Chunked prefill for one MiMo decoder layer of each type on the Galaxy (TP=8 x DP=4).

Two cases:

* ``short``: every replica prefills the same four prompts (200 to 1333 tokens), each in chunks of a
  different size: 64, 128, 256 and 512 tokens on rows 0-3, so chunks are both shorter and longer than
  the 128-token window.
* ``32k``: every replica prefills one 32768-token prompt, in chunks of 1024, 2048, 4096 and 8192 tokens on
  rows 0-3 (#61416). The fp32 reference runs attention in query blocks (``reference.model.QUERY_BLOCK``),
  so it fits on the host; set ``MIMO_REF_CACHE`` to a directory to keep its outputs between runs.

Every chunk after the first reads the earlier ones from the paged KV cache. The test checks:

* the chunked output against the fp32 reference (which sees the whole prompt at once) and against a
  one-shot prefill of the same prompt on the same row;
* the K/V the chunks wrote, position by position, against the reference K/V;
* a few decode steps that read that cache, against the reference.

The caches start out full of random values instead of zeros, so a kernel that reads a block the user
never wrote, or a key outside the window, shows up as an error. SWA layers get vLLM ``SlidingWindowSpec``
tables: before each chunk and decode step, every block wholly behind the window points at a shared null
block (block 0), and blocks past the chunk are not mapped yet.

SWA layers also run a third time, on the tables ``TtMiMoModel`` itself uses (``kv.layer_page_tables``),
sized for 4 users: each user gets the blocks for the window plus one of that row's chunks, and no fewer
than its share of the table width (the pool must be at least as wide as the table), and reuses them as
the window moves. ``short``: 8 blocks (512 tokens) on rows 0-2, so the 1333-token prompt cycles through
them 2-3 times. ``32k``: 132 blocks (8448 tokens) on every row, about 4 cycles. That run must match the
reference, and the no-reuse run, in prefill and decode.
"""

import hashlib
import math
import os
from dataclasses import dataclass

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

MAX_BATCH = 32
BLOCK = 64
HIDDEN_STD = 0.5
NULL_BLOCK = 0
CYCLIC_USERS = 4  # users the cyclic SWA tables are sized for, so their blocks are reused
KV_DTYPE = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[os.environ.get("MIMO_KV_DTYPE", "bfp8")]


@dataclass(frozen=True)
class Case:
    max_seq: int  # page table width in tokens; room for the prompts plus the decode steps
    prompt_lens: tuple
    chunks: tuple  # chunk size on replica rows 0..3
    # Decode tokens per user. The decode delta is pooled over a row's tokens, and one token whose 8th and
    # 9th expert scores tie within the router's error moves its output a lot, so a row needs enough tokens
    # that one such tie cannot decide the score (layer 5 at 32K: one tie in 3 tokens pooled to 0.978).
    decode_steps: int

    @property
    def logical_blocks(self):
        return self.max_seq // BLOCK


CASES = {
    "short": Case(max_seq=2048, prompt_lens=(200, 517, 960, 1333), chunks=(64, 128, 256, 512), decode_steps=3),
    "32k": Case(max_seq=32768 + 1024, prompt_lens=(32768,), chunks=(1024, 2048, 4096, 8192), decode_steps=32),
}


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


def swa_physical(slot, block, logical):
    """Physical block of a SWA slot's logical block: no reuse, so every written position stays readable."""
    return 1 + slot * logical + block


def set_swa_row(table, slot, lo, hi, window):
    """vLLM SlidingWindowSpec row: blocks overlapping [lo - window + 1, hi) are mapped, the rest are null."""
    logical = table.shape[-1]
    first = max(0, lo - window + 1) // BLOCK
    last = math.ceil(hi / BLOCK)
    table[:, slot] = NULL_BLOCK
    for b in range(first, last):
        table[:, slot, b] = swa_physical(slot, b, logical)


def reference_prefill(ref_layer, layer_idx, x):
    """fp32 reference output and K/V history for one whole prompt, from ``MIMO_REF_CACHE`` when it is there."""
    root = os.environ.get("MIMO_REF_CACHE")
    path = None
    if root:
        digest = hashlib.sha1(x.numpy().tobytes()).hexdigest()[:16]
        path = os.path.join(root, f"layer{layer_idx}_len{x.shape[0]}_{digest}.pt")
        if os.path.exists(path):
            saved = torch.load(path)
            hist = ref.KVHistory()
            hist.k, hist.v, hist.pos = saved["k"], saved["v"], saved["pos"]
            return saved["y"], hist
    hist = ref.KVHistory()
    with torch.no_grad():
        y = ref_layer(x, torch.arange(x.shape[0]), hist)
    if path:
        os.makedirs(root, exist_ok=True)
        torch.save(dict(y=y, k=hist.k, v=hist.v, pos=hist.pos), path)
    return y, hist


class Row:
    """One replica (Galaxy row) with its layer and one cache + page table per mode: "oneshot", "chunked",
    and on SWA layers "cyclic" (the model's own reused blocks)."""

    def __init__(self, sm, cfg, layer_idx, weights, dims, r, case):
        self.plan = plan = MeshPlan(sm)
        self.ropes = {
            "swa": MiMoRotarySetup(sm, MAX_BATCH, dims.head_dim, dims.rope_dim, cfg.swa_rope_theta),
            "full": MiMoRotarySetup(sm, MAX_BATCH, dims.head_dim, dims.rope_dim, cfg.rope_theta),
        }
        # Replica 0 converts the weights and fills the cache; the other rows load the cached tensors.
        self.layer = TtMiMoDecoderLayer(plan, cfg, layer_idx, weights, self.ropes, cache_dir=cache_dir("layer_bfp8"))
        self.window = dims.sliding_window
        self.chunk = case.chunks[r]
        self.logical = logical = case.logical_blocks
        n = len(case.prompt_lens)
        if self.window is None:
            tables, num_blocks = kv_lib.page_tables(n, MAX_BATCH, logical, TP, self.layer.attn.rep, seed=r)
            self.tables = {"oneshot": tables, "chunked": tables}
            num_blocks = {"oneshot": num_blocks, "chunked": num_blocks}
        else:
            full = torch.full((TP, MAX_BATCH, logical), NULL_BLOCK, dtype=torch.int32)
            for u in range(n):
                set_swa_row(full, u, case.max_seq, case.max_seq, case.max_seq)
            cyclic, cyclic_blocks = kv_lib.layer_page_tables(
                max(n, CYCLIC_USERS),
                MAX_BATCH,
                case.max_seq,
                BLOCK,
                TP,
                1,
                sliding_window=self.window,
                max_chunk=self.chunk,
                seed=r,
            )
            self.tables = {"oneshot": full, "chunked": full.clone().fill_(NULL_BLOCK), "cyclic": cyclic}
            num_blocks = {"oneshot": 1 + n * logical, "chunked": 1 + n * logical, "cyclic": cyclic_blocks}
        self.kv = {
            mode: garbage_cache(plan, num_blocks[mode], dims.head_dim, seed=100 * r + i)
            for i, mode in enumerate(self.tables)
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
                blocks = torch.tensor([swa_physical(slot, b, self.logical) for b in range(self.logical)])
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
@pytest.mark.parametrize("case_name", list(CASES))
@pytest.mark.parametrize("layer_idx", [0, 1, 5])
def test_chunked_prefill_tp8_dp4(mesh_device, layer_idx, case_name):
    case = CASES[case_name]
    torch.manual_seed(4321 + layer_idx)
    cfg, ckpt = load_config_and_ckpt()
    weights = ref.load_layer_weights(ckpt, cfg, layer_idx)
    ref_layer = ref.ReferenceDecoderLayer(cfg, layer_idx, weights=weights)
    dims = cfg.attention_dims(layer_idx)
    logger.info(
        f"layer {layer_idx} ({case_name}): swa={dims.is_swa} moe={cfg.is_moe(layer_idx)} kv_heads={dims.num_kv_heads}"
    )

    rows = [
        Row(sm, cfg, layer_idx, weights if r == 0 else None, dims, r, case)
        for r, sm in enumerate(dp_submeshes(mesh_device))
    ]

    # The same prompts on every row; the reference sees each whole prompt once.
    users = []
    for length in case.prompt_lens:
        x = (torch.randn(length, cfg.hidden_size) * HIDDEN_STD).bfloat16().float()
        y, hist = reference_prefill(ref_layer, layer_idx, x)
        users.append(dict(x=x, ref=y, hist=hist, ref_k=hist.k.clone(), ref_v=hist.v.clone()))

    # ---- prefill: one shot and in chunks, every row enqueued before anything is read
    chunk_modes = [m for m in rows[0].tables if m != "oneshot"]
    pending = []
    for u, usr in enumerate(users):
        for r, row in enumerate(rows):
            outs = {"oneshot": [row.prefill("oneshot", usr["x"], u, 0)]}
            for mode in chunk_modes:
                outs[mode] = [
                    row.prefill(mode, usr["x"][s : s + row.chunk], u, s) for s in range(0, usr["x"].shape[0], row.chunk)
                ]
            pending.append((r, u, outs))

    worst = dict(chunked=1.0, chunked_delta=1.0, vs_oneshot=1.0, vs_oneshot_delta=1.0, decode=1.0, decode_delta=1.0)
    worst.update(cyclic=1.0, cyclic_delta=1.0, cyclic_vs_chunked=1.0)  # stay 1.0 on global layers
    for r, u, outs in pending:
        plan, usr = rows[r].plan, users[u]
        host = {}
        for mode, chunk_outs in outs.items():
            host[mode] = torch.cat([plan.to_torch_replicated(o).float()[0, 0, :n] for o, n in chunk_outs])
            for o, _ in chunk_outs:
                o.deallocate(True)
        oneshot, chunked = host["oneshot"], host["chunked"]
        x, y = usr["x"], usr["ref"]
        result = dict(
            chunked=pcc(chunked, y),
            chunked_delta=pcc(chunked - x, y - x),
            vs_oneshot=pcc(chunked, oneshot),
            vs_oneshot_delta=pcc(chunked - x, oneshot - x),
        )
        if "cyclic" in host:
            cyclic = host["cyclic"]
            result.update(
                cyclic=pcc(cyclic, y),
                cyclic_delta=pcc(cyclic - x, y - x),
                cyclic_vs_chunked=pcc(cyclic - x, chunked - x),
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

    # ---- decode from each chunked cache; the same inputs on every row, so one reference serves all four
    decode_pool = {(r, mode): ([], [], [], []) for r in range(DP) for mode in chunk_modes}
    for step in range(case.decode_steps):
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
            for mode in chunk_modes:
                outs.append(
                    (
                        mode,
                        row.layer.forward_decode(to_tt(x, plan), rot, cur_pos, row.tables_tt[mode], row.kv[mode], mask),
                    )
                )
        for i, (mode, out) in enumerate(outs):
            r = i // len(chunk_modes)
            got = rows[r].plan.to_torch_replicated(out).float()[0, 0]
            for u, usr in enumerate(users):
                xd, rd = usr["x_decode"], usr["ref_decode"]
                p, pd = pcc(got[u], rd), pcc(got[u] - xd, rd - xd)
                pool = decode_pool[(r, mode)]
                pool[0].append(got[u])
                pool[1].append(rd)
                pool[2].append(got[u] - xd)
                pool[3].append(rd - xd)
                logger.info(f"decode step {step} row {r} {mode} slot {u} pos {int(pos[u])}: pcc {p:.5f} delta {pd:.5f}")
    # One decode token is dominated by its MoE output, and a token whose 8th and 9th expert scores are within
    # the router matmul's error (~2^-12 relative) picks a different expert on device: layer 5 at 32K has one
    # at position 32769 (gap 1.0e-4), output PCC 0.94. As for the delta in test_decoder_tp8_dp4.py, both
    # decode scores are taken over all of a row's decode tokens together; each token's own score is logged.
    for got, ref_out, got_d, ref_d in decode_pool.values():
        worst["decode"] = min(worst["decode"], pcc(torch.stack(got), torch.stack(ref_out)))
        worst["decode_delta"] = min(worst["decode_delta"], pcc(torch.stack(got_d), torch.stack(ref_d)))

    logger.info(f"layer {layer_idx} ({case_name}) worst over 4 rows x {len(users)} prompts: {worst}")
    # Same bounds as test_decoder_tp8_dp4.py: ~1-2% of MoE tokens flip one expert at a near-tie.
    assert worst["chunked"] > 0.995 and worst["decode"] > 0.995, worst
    assert worst["chunked_delta"] > 0.99 and worst["decode_delta"] > 0.99, worst
    assert worst["vs_oneshot"] > 0.995 and worst["vs_oneshot_delta"] > 0.99, worst
    assert worst["cyclic"] > 0.995 and worst["cyclic_delta"] > 0.99, worst
    # Same math as the no-reuse run, only other physical blocks: the results should agree to the last bit.
    assert worst["cyclic_vs_chunked"] > 0.9999, worst
    assert worst_kv["k"] > 0.999 and worst_kv["v"] > 0.999, worst_kv
    assert worst_kv["v_pad"] == 0.0, "V padding columns must stay zero"
