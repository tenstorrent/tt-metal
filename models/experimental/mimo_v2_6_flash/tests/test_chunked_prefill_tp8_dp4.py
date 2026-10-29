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

Every chunk after the first reads the earlier ones from the paged KV cache: a SWA layer only the window
before the chunk (kv.SwaBlockTable keeps kv.swa_blocks_per_user blocks per user), a global layer the whole
context on the chip that owns the user (chunked MLA prefill over the K and 128-wide V caches). ``kv_both``
(global layers): both chips of the pair keep every user's K / V, as vLLM's caches do, and a later chunk runs
8 heads on each (TtMiMoAttention._prefill_pair_chunk). The test checks:

* the chunked output against the fp32 reference (which sees the whole prompt at once) and against a
  one-shot prefill of the same prompt on the same row;
* the K/V the chunks left in the cache against the reference K/V: every position on a global layer's
  owner chip, the last window on a SWA layer;
* a few decode steps that read that cache, against the reference.

The caches start out full of random values instead of zeros, so reading a block the user never wrote, or a
key outside the window, shows up as an error.
"""

import hashlib
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
    model_path,
    pcc,
)
from models.experimental.mimo_v2_6_flash.tt import kv as kv_lib
from models.experimental.mimo_v2_6_flash.tt.attention import kv_heads_for_chip
from models.experimental.mimo_v2_6_flash.tt.common import MeshPlan, pad_to
from models.experimental.mimo_v2_6_flash.tt.decoder import TtMiMoDecoderLayer
from models.experimental.mimo_v2_6_flash.tt.model import prefill_padded_len
from models.experimental.mimo_v2_6_flash.tt.rope import MiMoRotarySetup, meta_rope_permutation

MAX_BATCH = 32
BLOCK = 64
HIDDEN_STD = 0.5
MODES = ("oneshot", "chunked")
KV_DTYPE = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[os.environ.get("MIMO_KV_DTYPE", "bfp8")]


@dataclass(frozen=True)
class Case:
    max_seq: int  # page table width in tokens; room for the prompts plus the decode steps
    prompt_lens: tuple
    chunks: tuple  # chunk size on replica rows 0..3
    # Decode tokens per user. The decode scores are pooled over a row's tokens, and one token whose 8th and
    # 9th expert scores tie within the router's error moves its output a lot, so a row needs enough tokens
    # that one such tie cannot decide the score (layer 5 at 32K: one tie in 3 tokens pooled to 0.978).
    decode_steps: int

    @property
    def logical_blocks(self):
        return self.max_seq // BLOCK

    @property
    def context_blocks(self):
        """As TtMiMoModel.prefill_context_blocks."""
        return pad_to(-(-(self.max_seq + kv_lib.PREFILL_MAX_PAD) // BLOCK), 8)


CASES = {
    "short": Case(max_seq=2048, prompt_lens=(200, 517, 960, 1333), chunks=(64, 128, 256, 512), decode_steps=3),
    "32k": Case(max_seq=32768 + 1024, prompt_lens=(32768,), chunks=(1024, 2048, 4096, 8192), decode_steps=32),
}


def reference_prefill(ref_layer, layer_idx, x):
    """fp32 reference output and K/V history for one whole prompt, from ``MIMO_REF_CACHE`` when it is there. Entries
    are kept per checkpoint, as the TTNN weight cache is (the inputs are seeded the same for every checkpoint), and
    one that records a different checkpoint, or none, is recomputed."""
    root = os.environ.get("MIMO_REF_CACHE")
    path = None
    checkpoint = os.path.realpath(model_path())
    if root:
        digest = hashlib.sha1(x.numpy().tobytes()).hexdigest()[:16]
        path = os.path.join(root, os.path.basename(checkpoint), f"layer{layer_idx}_len{x.shape[0]}_{digest}.pt")
        if os.path.exists(path):
            saved = torch.load(path)
            if saved.get("checkpoint") == checkpoint:
                hist = ref.KVHistory()
                hist.k, hist.v, hist.pos = saved["k"], saved["v"], saved["pos"]
                return saved["y"], hist
            logger.warning(f"{path} was made from {saved.get('checkpoint')}, not {checkpoint}; recomputing")
    hist = ref.KVHistory()
    with torch.no_grad():
        y = ref_layer(x, torch.arange(x.shape[0]), hist)
    if path:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(dict(y=y, k=hist.k, v=hist.v, pos=hist.pos, checkpoint=checkpoint), path)
        except OSError as e:  # a read-only cache (CI mounts /mnt/MLPerf read-only) is still read above
            logger.warning(f"could not save the reference to {path}: {e}")
    return y, hist


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


def garbage_cache(plan, num_blocks, dims, seed):
    """[K, V] caches filled with random values: reading a block nobody wrote gives a visible error."""
    g = torch.Generator().manual_seed(seed)
    return [
        ttnn.from_torch(
            torch.randn(num_blocks, 1, BLOCK, dim, generator=g) * 4,
            device=plan.mesh_device,
            dtype=KV_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=plan.replicate(),
        )
        for dim in dims
    ]


class Row:
    """One replica (Galaxy row) with its layer and a one-shot and a chunked cache with their page tables."""

    def __init__(self, sm, cfg, layer_idx, weights, dims, r, case, kv_both=False):
        self.plan = plan = MeshPlan(sm)
        self.ropes = {
            "swa": MiMoRotarySetup(sm, MAX_BATCH, dims.head_dim, dims.rope_dim, cfg.swa_rope_theta),
            "full": MiMoRotarySetup(sm, MAX_BATCH, dims.head_dim, dims.rope_dim, cfg.rope_theta),
        }
        # Replica 0 converts the weights and fills the cache; the other rows load the cached tensors.
        self.layer = TtMiMoDecoderLayer(plan, cfg, layer_idx, weights, self.ropes, cache_dir=cache_dir("layer_bfp8"))
        attn = self.layer.attn
        self.kv_both = kv_both
        if kv_both:
            attn.keep_kv_on_both_chips()
        self.chunk = case.chunks[r]
        self.context_blocks = case.context_blocks
        logical = case.logical_blocks
        n = len(case.prompt_lens)
        self.swa, self.tables = {}, {}
        for mode in MODES:
            if attn.is_swa:
                self.swa[mode] = kv_lib.SwaBlockTable(
                    n,
                    MAX_BATCH,
                    logical,
                    BLOCK,
                    attn.sliding_window,
                    kv_lib.swa_blocks_per_user(attn.sliding_window, BLOCK),
                    min_blocks=logical,  # decode here runs paged_update_cache (no SwaDecodeQkv)
                )
                self.tables[mode], num_blocks = self.swa[mode].host_table(TP), self.swa[mode].num_blocks
                self.scratch = 0
            elif kv_both:  # both chips of the pair keep every user's K / V (vLLM's layout), plus a scratch block
                self.swa[mode] = None
                self.tables[mode], num_blocks = kv_lib.page_tables(n, MAX_BATCH, logical, TP, 1, seed=r)
                self.scratch, num_blocks = num_blocks, num_blocks + 1
            else:
                self.swa[mode] = None
                self.tables[mode], num_blocks = kv_lib.page_tables(n, MAX_BATCH, logical, TP, attn.rep, seed=r)
                self.scratch = num_blocks - 1  # the chip pair's scratch block (kv.page_tables)
        self.kv = {
            mode: garbage_cache(plan, num_blocks, (attn.head_dim, attn.v_head_dim), seed=100 * r + i)
            for i, mode in enumerate(MODES)
        }
        self.tables_tt = {mode: self._device_table(mode) for mode in MODES}
        if self.layer.is_moe:
            self.layer.mlp.set_decode_users(n)

    def _device_table(self, mode):
        return kv_lib.per_chip(self.plan, self.tables[mode], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

    def sync(self, mode):
        """Copy a SWA block table to its device page table if it changed (ordered with the queued work)."""
        tbl = self.swa[mode]
        if tbl is not None and tbl.dirty:
            self.tables[mode] = tbl.host_table(TP)
            host = kv_lib.per_chip(self.plan, self.tables[mode], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, False)
            ttnn.copy_host_to_device_tensor(host, self.tables_tt[mode])
            tbl.dirty = False

    def prefill(self, mode, x, slot, start_pos):
        """One chunk through the layer with its kv.PrefillTables, as TtMiMoModel._prefill_chunk builds them."""
        n = x.shape[0]
        padded = prefill_padded_len(n)
        tbl = self.swa[mode]
        if tbl is not None:
            tbl.prefill(slot, start_pos, start_pos + n, start_pos + padded)
            self.sync(mode)
        later = start_pos > 0
        start = (
            kv_lib.replicated(self.plan, torch.tensor([start_pos], dtype=torch.int32), ttnn.int32) if later else None
        )
        tables = kv_lib.PrefillTables(
            self.plan,
            self.tables[mode],
            slot,
            start_pos,
            padded,
            BLOCK,
            self.scratch,
            context_blocks=self.context_blocks if later and tbl is None else None,
            window=tbl.window_blocks(slot, start_pos) if later and tbl is not None else None,
            start_tensor=start,
        )
        x_tt = to_tt(torch.nn.functional.pad(x, (0, 0, 0, padded - n)), self.plan)
        out = self.layer.forward_prefill(
            x_tt, slot, self.tables_tt[mode], self.kv[mode], real_len=n, start_pos=start_pos, tables=tables
        )
        x_tt.deallocate(True)
        tables.release()
        if start is not None:
            start.deallocate(True)
        return out, n

    def read_kv(self, slot, length, ref_k, ref_v, num_kv_heads):
        """Worst PCC of the chunked cache's K and V vs the reference, over the chips that hold this slot and the
        positions a later step can read (all of them on a global layer, the last window on a SWA layer)."""
        attn = self.layer.attn
        perm = meta_rope_permutation(ref_k.shape[-1], attn.dims.rope_dim)
        caches = [[ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(c)] for c in self.kv["chunked"]]
        first = 0 if attn.sliding_window is None else max(0, length - attn.sliding_window + 1)
        positions = torch.arange(first, length)
        worst_k, worst_v, pad_max = 1.0, 1.0, 0.0
        vd = ref_v.shape[-1]
        for chip in range(TP):
            if slot % attn.rep != chip % attn.rep and not self.kv_both:
                continue  # the other chip of the pair does not hold this user
            blocks = self.tables["chunked"][chip, slot, positions // BLOCK]
            assert (blocks != 0).all() or attn.rep > 1, "a readable SWA position sits in the scratch block"
            head = kv_heads_for_chip(chip, TP, num_kv_heads)[0]
            k = caches[0][chip][blocks, 0, positions % BLOCK]
            v = caches[1][chip][blocks, 0, positions % BLOCK]
            worst_k = min(worst_k, pcc(k, ref_k[head][positions][:, perm]))
            worst_v = min(worst_v, pcc(v[:, :vd], ref_v[head][positions]))
            if v.shape[-1] > vd:
                pad_max = max(pad_max, v[:, vd:].abs().max().item())
        return worst_k, worst_v, pad_max


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("case_name", list(CASES))
@pytest.mark.parametrize(
    "layer_idx, kv_both",
    [(0, False), (1, False), (5, False), (0, True), (5, True)],
    ids=["0", "1", "5", "0-kvboth", "5-kvboth"],
)
def test_chunked_prefill_tp8_dp4(mesh_device, layer_idx, kv_both, case_name):
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
        Row(sm, cfg, layer_idx, weights if r == 0 else None, dims, r, case, kv_both)
        for r, sm in enumerate(dp_submeshes(mesh_device))
    ]

    # The same prompts on every row; the reference sees each whole prompt once.
    users = []
    for length in case.prompt_lens:
        x = (torch.randn(length, cfg.hidden_size) * HIDDEN_STD).bfloat16().float()
        y, hist = reference_prefill(ref_layer, layer_idx, x)
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

    # ---- the K/V the chunks left in the cache
    worst_kv = dict(k=1.0, v=1.0, v_pad=0.0)
    for row in rows:
        for u, usr in enumerate(users):
            k, v, pad = row.read_kv(u, usr["x"].shape[0], usr["ref_k"], usr["ref_v"], dims.num_kv_heads)
            worst_kv["k"], worst_kv["v"] = min(worst_kv["k"], k), min(worst_kv["v"], v)
            worst_kv["v_pad"] = max(worst_kv["v_pad"], pad)
    logger.info(f"chunked KV vs reference, worst over rows/users/chips: {worst_kv}")

    # ---- decode from the chunked cache; the same inputs on every row, so one reference serves all four
    decode_pool = {r: ([], [], [], []) for r in range(DP)}
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
            tbl = row.swa["chunked"]
            if tbl is not None:
                for u in range(len(users)):
                    tbl.advance(u, int(pos[u]))
                row.sync("chunked")
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
            if row.kv_both:  # both chips write every slot's K / V (TtMiMoModel.global_kv_both)
                cur_pos["pos_update"] = cur_pos["pos"]
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
                pool = decode_pool[r]
                pool[0].append(got[u])
                pool[1].append(rd)
                pool[2].append(got[u] - xd)
                pool[3].append(rd - xd)
                logger.info(f"decode step {step} row {r} slot {u} pos {int(pos[u])}: pcc {p:.5f} delta {pd:.5f}")
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
    assert worst_kv["k"] > 0.999 and worst_kv["v"] > 0.999, worst_kv
    assert worst_kv["v_pad"] == 0.0, "V padding columns must stay zero"


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("layer_idx, kv_both", [(1, False), (5, False), (5, True)], ids=["1", "5", "5-kvboth"])
def test_prefill_programs_do_not_key_on_start(mesh_device, layer_idx, kv_both):
    """Chunks of one padded length at new starts, in a new slot and on new blocks compile no new program: what lets
    vLLM serving prefill between decode trace replays (a program compiled after a capture keeps buffers the replays
    overwrite). The first user's chunks cover every start alignment of a 128-token chunk (window blocks and SDPA chunk
    size); the second user's chunks reach starts the first never did."""
    cfg, ckpt = load_config_and_ckpt()
    dims = cfg.attention_dims(layer_idx)
    weights = ref.load_layer_weights(ckpt, cfg, layer_idx)
    row = Row(dp_submeshes(mesh_device)[0], cfg, layer_idx, weights, dims, 0, CASES["short"], kv_both)
    device = row.plan.mesh_device

    def prefill_user(slot, length, chunk=128):
        x = (torch.randn(length, cfg.hidden_size) * HIDDEN_STD).bfloat16().float()
        for s in range(0, length, chunk):
            out, _ = row.prefill("chunked", x[s : s + chunk], slot, s)
            out.deallocate(True)
        ttnn.synchronize_device(device)

    prefill_user(0, 640)
    entries = device.num_program_cache_entries()
    prefill_user(1, 1536)
    new = device.num_program_cache_entries() - entries
    logger.info(f"layer {layer_idx}: {entries} programs after the first user, {new} new for the second")
    assert new == 0, f"{new} prefill programs keyed on the start, slot or blocks"
