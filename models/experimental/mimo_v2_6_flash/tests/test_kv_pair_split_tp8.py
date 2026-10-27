# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The global KV pair split (MIMO_KV_PAIR_SPLIT, vLLM mode; reference/kv_pair_split.py, tt/pair_split.py) on one
Galaxy row (row 0 of the 4x8 mesh).

* ``test_pair_positions_tp8``: the positions program (kernels/attn/pair_positions.cpp) against its host mirror on
  every chip, for positions around the 64 / 128 boundaries, idle slots and positions near 1M, with serial rows.
* ``test_pair_merge_tp8``: PairSplit.merge (the LSE all_gather, the partner slice, the fp32 weight and the scaled
  output) against reference.kv_pair_split.merge_weights on every chip, with users that see no key on one or both
  chips (LSE -1e30).
* ``test_pair_split_compact_verify_tp8``: the decode configurations serving runs, split vs unsplit: the compact SDPA
  over the first 4 table rows (MIMO_COMPACT_SDPA with 4 decode users per lane) and speculative verifies whose slots
  write K / V one after another (the serial rows) at positions straddling 63 / 64 and 127 / 128.
* ``test_pair_split_layer_tp8``: one global layer's attention through vLLM-style tables with the split on and off
  (64-position blocks, both copies on the pair): the same prompts prefilled in chunks (a first chunk, then later
  chunks through chunked MLA prefill), then decode steps whose positions straddle the 64 / 128 boundaries. Every
  prefill chunk's and decode step's output must agree (PCC > 0.999), and the split cache must hold, at each chip's
  local index, the K / V the unsplit cache holds at its global position. ``long`` does it at 32K context.

The first two need only ops that exist today; the layer tests need the MLA attention ops with ``return_lse`` (and
chunked MLA prefill's pair-split mode), and are skipped on a build without them.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.reference import kv_pair_split as kps
from models.experimental.mimo_v2_6_flash.reference import model as ref
from models.experimental.mimo_v2_6_flash.reference.spec_decode import serial_positions
from models.experimental.mimo_v2_6_flash.tests.common import (
    GALAXY_DEVICE_PARAMS,
    TP,
    cache_dir,
    dp_submeshes,
    load_config_and_ckpt,
    pcc,
)
from models.experimental.mimo_v2_6_flash.tt import kv as kv_lib
from models.experimental.mimo_v2_6_flash.tt.attention import PAIR_SPLIT_OPS, TtMiMoAttention, page_table_width
from models.experimental.mimo_v2_6_flash.tt.common import MeshPlan, pad_to
from models.experimental.mimo_v2_6_flash.tt.model import prefill_padded_len
from models.experimental.mimo_v2_6_flash.tt.pair_split import PairSplit
from models.experimental.mimo_v2_6_flash.tt.rope import MiMoRotarySetup

MAX_BATCH = 32
BLOCK = 64
HIDDEN_STD = 1.0
SERIAL = 4


def per_chip_torch(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def row_plan(mesh_device):
    return MeshPlan(dp_submeshes(mesh_device)[0])


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_pair_positions_tp8(mesh_device):
    plan = row_plan(mesh_device)
    pair = PairSplit(plan, MAX_BATCH, serial_slots=SERIAL)
    g = torch.Generator().manual_seed(7)
    edge = [0, 1, 62, 63, 64, 65, 126, 127, 128, 129, 191, 192, 255, 256, 1048575, 1048574, -1]
    cases = [
        torch.tensor(edge + [-1] * (MAX_BATCH - len(edge)), dtype=torch.int32),
        torch.randint(-1, 1 << 20, (MAX_BATCH,), generator=g, dtype=torch.int32),
        torch.full((MAX_BATCH,), -1, dtype=torch.int32),
    ]
    for positions in cases:
        cur = kv_lib.replicated(plan, positions, ttnn.int32)
        write, sdpa, serial = pair.positions(cur)
        got_w, got_s = per_chip_torch(write), per_chip_torch(sdpa)
        got_serial = [per_chip_torch(t) for t in serial]
        for chip in range(TP):
            w, s, ser = kps.positions_step(positions.tolist(), kps.parity(chip), SERIAL)
            assert got_w[chip].reshape(-1).tolist() == w, (chip, positions.tolist())
            assert got_s[chip].reshape(-1).tolist() == s, (chip, positions.tolist())
            for j in range(SERIAL):
                assert got_serial[j][chip].reshape(-1).tolist() == ser[j], (chip, j)
        cur.deallocate(True)


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "shape", [(1, 4, 16, 128), (1, 32, 16, 128), (1, 16, 256, 128)], ids=["decode4", "decode32", "prefill"]
)
def test_pair_merge_tp8(mesh_device, shape):
    plan = row_plan(mesh_device)
    pair = PairSplit(plan, MAX_BATCH)
    g = torch.Generator().manual_seed(11)
    o = torch.randn(TP, *shape[1:], generator=g).bfloat16()
    lse = torch.randn(TP, *shape[1:-1], 1, generator=g) * 8.0
    rows = shape[1] * shape[2]
    flat = lse.reshape(TP, rows)
    flat[0::2, 0] = kps.NO_KEY_LSE  # nothing visible on the even chip: the odd one carries it
    flat[1::2, 1] = kps.NO_KEY_LSE  # ... on the odd chip
    flat[:, 2] = kps.NO_KEY_LSE  # an idle user: nothing on either
    o.reshape(TP, rows, -1)[0::2, 0] = 0
    o.reshape(TP, rows, -1)[1::2, 1] = 0
    o.reshape(TP, rows, -1)[:, 2] = 0
    memory = ttnn.L1_MEMORY_CONFIG if shape[2] == 16 else ttnn.DRAM_MEMORY_CONFIG
    o_tt = kv_lib.per_chip(plan, o.reshape(TP, *shape[1:]).unsqueeze(1), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    lse_tt = kv_lib.per_chip(plan, lse.unsqueeze(1), ttnn.float32, ttnn.TILE_LAYOUT)
    out = per_chip_torch(pair.merge(o_tt, lse_tt, memory))
    worst = 1.0
    for chip in range(TP):
        partner = chip ^ 1
        alpha = kps.merge_weights(lse[chip].double(), lse[partner].double())
        want = (alpha * o[chip].double()).float()
        got = out[chip].float().reshape(want.shape)
        worst = min(worst, pcc(got, want))
        # Rows 0-2: the chip without keys contributes exactly 0; the other carries its row whole (alpha 1, up to the
        # bf16 rounding of the product).
        got_flat, want_flat = got.reshape(rows, -1)[:3], want.reshape(rows, -1)[:3]
        empty = o[chip].reshape(rows, -1)[:3].abs().sum(-1) == 0
        assert torch.equal(got_flat[empty], torch.zeros_like(got_flat[empty])), f"chip {chip}: no keys, nonzero"
        assert torch.allclose(got_flat[~empty], want_flat[~empty], rtol=1e-2, atol=1e-2), f"chip {chip}: carried row"
    logger.info(f"merge {shape}: worst PCC {worst:.6f}")
    assert worst > 0.9999


class Layout:
    """One global KV layout on the row, as vLLM lays it out: ``split`` 128-position blocks halved over the pair
    (setup_vllm_kv_cache with pair_split), else 64-position blocks with both copies on the pair. The caches start out
    random (a block nobody wrote shows up as an error); each user's blocks are disjoint and shuffled, block 0 is the
    null block."""

    def __init__(self, plan, attn, pair, split, max_seq, users, seed):
        self.plan, self.attn, self.pair, self.split = plan, attn, pair, split
        self.span = kps.PAIR if split else BLOCK
        per_user = -(-max_seq // self.span)
        width = page_table_width(per_user)
        # The paged cache ops want at least as many blocks as the table is wide (one user's padded row can exceed
        # a small pool); the extra blocks are never mapped.
        num_blocks = 1 + max(users * per_user, width)
        g = torch.Generator().manual_seed(seed)
        order = 1 + torch.randperm(num_blocks - 1, generator=g).to(torch.int32)
        self.table = torch.zeros(MAX_BATCH, width, dtype=torch.int32)
        for u in range(users):
            self.table[u, :per_user] = order[u * per_user : (u + 1) * per_user]
        self.table_tt = kv_lib.replicated(plan, self.table, ttnn.int32)
        self.kv = [
            ttnn.from_torch(
                torch.randn(num_blocks, 1, BLOCK, dim, generator=g) * 4,
                device=plan.mesh_device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=plan.replicate(),
            )
            for dim in (attn.head_dim, attn.v_head_dim)
        ]
        self.context_blocks = pad_to(-(-(max_seq + kv_lib.PREFILL_MAX_PAD) // self.span), 8)

    def _use(self):
        self.attn.pair = self.pair if self.split else None

    def prefill(self, x, slot, start):
        """One chunk (x [n, hidden], the attention's input) at ``start``, with the model's kv.PrefillTables."""
        plan, n = self.plan, x.shape[0]
        padded = prefill_padded_len(n, start, self.split)
        later = start > 0
        start_tt = kv_lib.replicated(plan, torch.tensor([start], dtype=torch.int32), ttnn.int32) if later else None
        tables = kv_lib.PrefillTables(
            plan,
            self.table.unsqueeze(0).expand(TP, -1, -1),
            slot,
            start,
            padded,
            BLOCK,
            0,
            context_blocks=self.context_blocks if later else None,
            start_tensor=start_tt,
            pair_split=self.split,
        )
        x_tt = ttnn.from_torch(
            torch.nn.functional.pad(x, (0, 0, 0, padded - n)).reshape(1, 1, padded, -1),
            device=plan.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=plan.replicate(),
        )
        self._use()
        self.attn.rope.begin_prefill_chunk()
        out = self.attn.forward_prefill(x_tt, slot, self.table_tt, self.kv, start_pos=start, real_len=n, tables=tables)
        self.attn.rope.end_prefill_chunk()
        x_tt.deallocate(True)
        tables.release()
        if start_tt is not None:
            start_tt.deallocate(True)
        got = plan.to_torch_replicated(out).float()[0, 0, :n]
        out.deallocate(True)
        return got

    def decode(self, x, positions, table=None, rows=None, serial=0):
        """One decode step (x [32, hidden], positions [32], -1 idle) through vLLM's tables, as the model drives it
        (TtMiMoModel.decode_forward_device). ``table``: the page table's rows instead of the users' (a verify gives
        every slot its request's row). ``rows`` (MIMO_COMPACT_SDPA): the SDPA reads only the first ``rows`` table rows
        and SDPA positions. ``serial`` (a verify): slots 0 .. serial - 1 write their K / V one after another."""
        plan = self.plan
        cur = kv_lib.replicated(plan, positions, ttnn.int32)
        rep = self.attn.rep
        made = [
            kv_lib.per_chip(plan, kv_lib.owned_positions(positions, TP, rep), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            kv_lib.per_chip(plan, kv_lib.owner_mask(positions, TP, rep), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        ]
        table_tt = self.table_tt if table is None else kv_lib.replicated(plan, table, ttnn.int32)
        dims = {"pos": cur, "pos_owned": made[0], "owner_mask": made[1]}
        sdpa_pos = made[0]
        self._use()
        if self.split:
            write, sdpa_pos, serial_rows = self.pair.positions(cur)
            dims.update(pair_write=write, pair_sdpa=sdpa_pos)
            if serial:
                dims["pair_serial"] = serial_rows[:serial]
        else:
            dims["pos_update"] = cur
        temps = [cur, *made]
        if serial:
            one = serial_positions(positions.tolist(), serial)
            dims["serial"] = [kv_lib.replicated(plan, torch.tensor(o, dtype=torch.int32), ttnn.int32) for o in one]
            temps += dims["serial"]
        if rows is not None:
            dims["pos_sdpa"] = ttnn.slice(sdpa_pos, (0,), (rows,))
            dims["page_table_sdpa"] = ttnn.slice(table_tt, (0, 0), (rows, table_tt.shape[-1]))
            temps += [dims["pos_sdpa"], dims["page_table_sdpa"]]
        rot = self.attn.rope.get_rot_mats(positions.clamp(min=0))
        x_tt = ttnn.from_torch(
            x.reshape(1, 1, MAX_BATCH, -1),
            device=plan.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=plan.replicate(),
        )
        out = self.attn.forward_decode(x_tt, rot, dims, table_tt, self.kv, compact=False)
        got = plan.to_torch_replicated(out).float()[0, 0]
        for t in (out, x_tt, *temps, *rot):
            t.deallocate(True)
        if table is not None:
            table_tt.deallocate(True)
        return got

    def caches(self):
        return [per_chip_torch(c) for c in self.kv]


def check_caches(on, off, lengths):
    """The split cache holds at chip c's local index l what the unsplit cache holds at global position G(l, c), for
    every position a user wrote: each chip's halves are the unsplit K / V. Equal up to the projection's rounding: a
    short later chunk is padded to 128 rows under the split (to 32 / 64 / 96 without), which can change the linear's
    program; a misplaced position reads the random initial cache (PCC ~0)."""
    on_kv, off_kv = on.caches(), off.caches()
    worst = 1.0
    for which in range(2):
        for chip in range(TP):
            c = kps.parity(chip)
            for u, length in enumerate(lengths):
                q = torch.tensor([p for p in range(length) if kps.own(p, c)], dtype=torch.long)
                if q.numel() < 2:
                    continue
                loc = torch.tensor([kps.local(int(p)) for p in q], dtype=torch.long)
                got = on_kv[which][chip][on.table[u, loc // BLOCK].long(), 0, loc % BLOCK]
                want = off_kv[which][chip][off.table[u, q // BLOCK].long(), 0, q % BLOCK]
                score = pcc(got, want)
                worst = min(worst, score)
                assert score > 0.9999, f"{'KV'[which]} chip {chip} user {u}: split cache PCC {score:.6f}"
    return worst


CASES = {
    # (prompt lengths, chunk size, decode steps, max_seq): the decode positions start at the prompt ends.
    "short": ((63, 64, 127, 128, 129, 300, 1000), 256, 3, 2048),
    "long": ((32768,), 4096, 3, 32768 + 1024),
}


@pytest.mark.skipif(not PAIR_SPLIT_OPS, reason="needs the MLA attention ops with return_lse (runs/kvsplit/SPEC.md)")
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("case", list(CASES))
def test_pair_split_layer_tp8(mesh_device, case):
    lengths, chunk, steps, max_seq = CASES[case]
    cfg, ckpt = load_config_and_ckpt()
    layer_idx = next(i for i in range(cfg.num_hidden_layers) if not cfg.is_swa(i))
    dims = cfg.attention_dims(layer_idx)
    plan = row_plan(mesh_device)
    rope = MiMoRotarySetup(plan.mesh_device, MAX_BATCH, dims.head_dim, dims.rope_dim, cfg.rope_theta)
    weights = ref.load_layer_weights(ckpt, cfg, layer_idx)["attn"]
    attn = TtMiMoAttention(plan, cfg, layer_idx, weights, rope, cache_dir=cache_dir("layer_bfp8"))
    assert attn.split_users
    pair = PairSplit(plan, MAX_BATCH)
    on = Layout(plan, attn, pair, True, max_seq, len(lengths), seed=1)
    off = Layout(plan, attn, pair, False, max_seq, len(lengths), seed=2)

    g = torch.Generator().manual_seed(1234)
    worst = {"prefill": 1.0, "decode": 1.0}
    for u, length in enumerate(lengths):
        x = (torch.randn(length, cfg.hidden_size, generator=g) * HIDDEN_STD).bfloat16().float()
        for s in range(0, length, chunk):
            a, b = on.prefill(x[s : s + chunk], u, s), off.prefill(x[s : s + chunk], u, s)
            score = pcc(a, b)
            worst["prefill"] = min(worst["prefill"], score)
            logger.info(f"{case} user {u} chunk at {s} ({min(chunk, length - s)} tokens): split vs not {score:.6f}")
    logger.info(f"{case}: cache after prefill, worst PCC {check_caches(on, off, lengths):.6f}")

    for step in range(steps):
        positions = torch.full((MAX_BATCH,), -1, dtype=torch.int32)
        positions[: len(lengths)] = torch.tensor(lengths, dtype=torch.int32) + step
        x = (torch.randn(MAX_BATCH, cfg.hidden_size, generator=g) * HIDDEN_STD).bfloat16().float()
        a, b = on.decode(x, positions), off.decode(x, positions)
        for u in range(len(lengths)):
            score = pcc(a[u], b[u])
            worst["decode"] = min(worst["decode"], score)
            logger.info(f"{case} decode step {step} user {u} at {int(positions[u])}: split vs not {score:.6f}")
    logger.info(f"{case}: cache after decode, worst PCC {check_caches(on, off, [n + steps for n in lengths]):.6f}")
    logger.info(f"{case}: worst {worst}")
    assert worst["prefill"] > 0.999 and worst["decode"] > 0.999, worst


@pytest.mark.skipif(not PAIR_SPLIT_OPS, reason="needs the MLA attention ops with return_lse (runs/kvsplit/SPEC.md)")
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_pair_split_compact_verify_tp8(mesh_device):
    """The decode configurations serving runs, split vs unsplit: MIMO_COMPACT_SDPA with 4 decode users (the SDPA over
    the first 4 table rows and SDPA positions, as decode_forward_device slices them: its program spreads each user over
    many cores with the tree reduction, the LSE comes from the reduction's root, and the merge runs on [1, 4, 16, 1]),
    and speculative verifies (4 slots of one request on its table row at consecutive positions straddling 63 / 64 and
    127 / 128, K / V written slot by slot through the serial rows, then the compact SDPA). Every slot's output must
    agree (PCC > 0.999) and the split cache must hold the unsplit one's K / V."""
    users, steps, max_seq = 4, 3, 2048
    lengths = (62, 126, 1000, 63)
    cfg, ckpt = load_config_and_ckpt()
    layer_idx = next(i for i in range(cfg.num_hidden_layers) if not cfg.is_swa(i))
    dims = cfg.attention_dims(layer_idx)
    plan = row_plan(mesh_device)
    rope = MiMoRotarySetup(plan.mesh_device, MAX_BATCH, dims.head_dim, dims.rope_dim, cfg.rope_theta)
    weights = ref.load_layer_weights(ckpt, cfg, layer_idx)["attn"]
    attn = TtMiMoAttention(plan, cfg, layer_idx, weights, rope, cache_dir=cache_dir("layer_bfp8"))
    pair = PairSplit(plan, MAX_BATCH, serial_slots=SERIAL)
    on = Layout(plan, attn, pair, True, max_seq, users, seed=3)
    off = Layout(plan, attn, pair, False, max_seq, users, seed=4)

    g = torch.Generator().manual_seed(4321)
    for u, length in enumerate(lengths):
        x = (torch.randn(length, cfg.hidden_size, generator=g) * HIDDEN_STD).bfloat16().float()
        for s in range(0, length, 256):
            on.prefill(x[s : s + 256], u, s)
            off.prefill(x[s : s + 256], u, s)

    worst = {"verify": 1.0, "decode": 1.0}
    # Verifies of users 0 and 1: every verify slot holds the request's table row.
    for u in (0, 1):
        positions = torch.full((MAX_BATCH,), -1, dtype=torch.int32)
        positions[:SERIAL] = lengths[u] + torch.arange(SERIAL, dtype=torch.int32)
        tables = []
        for layout in (on, off):
            t = layout.table.clone()
            t[:SERIAL] = layout.table[u]
            tables.append(t)
        x = (torch.randn(MAX_BATCH, cfg.hidden_size, generator=g) * HIDDEN_STD).bfloat16().float()
        a = on.decode(x, positions, table=tables[0], rows=SERIAL, serial=SERIAL)
        b = off.decode(x, positions, table=tables[1], rows=SERIAL, serial=SERIAL)
        for j in range(SERIAL):
            score = pcc(a[j], b[j])
            worst["verify"] = min(worst["verify"], score)
            logger.info(f"verify of user {u}, slot {j} at {int(positions[j])}: split vs not {score:.6f}")

    # Compact decode steps of the 4 users (users 0 and 1 past their verified positions).
    now = [lengths[0] + SERIAL, lengths[1] + SERIAL, lengths[2], lengths[3]]
    for step in range(steps):
        positions = torch.full((MAX_BATCH,), -1, dtype=torch.int32)
        positions[:users] = torch.tensor(now, dtype=torch.int32) + step
        x = (torch.randn(MAX_BATCH, cfg.hidden_size, generator=g) * HIDDEN_STD).bfloat16().float()
        a, b = on.decode(x, positions, rows=users), off.decode(x, positions, rows=users)
        for u in range(users):
            score = pcc(a[u], b[u])
            worst["decode"] = min(worst["decode"], score)
            logger.info(f"compact decode step {step} user {u} at {int(positions[u])}: split vs not {score:.6f}")
    logger.info(f"cache after the verifies and decode, worst PCC {check_caches(on, off, [n + steps for n in now]):.6f}")
    logger.info(f"worst {worst}")
    assert worst["verify"] > 0.999 and worst["decode"] > 0.999, worst
