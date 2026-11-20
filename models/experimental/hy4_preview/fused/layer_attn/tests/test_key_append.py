# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The index-key append of program P1 on a full-indexer layer (fused/layer_attn/program.py, PLAN.md section 20): the
new token's index key of the 8 token rows, finished on the key core of every chip (LayerNorm, rotation, Hadamard
basis, BFP8 pack) and patched into the 32-row block of the key store by the chip that owns the row's position.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file>
    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k "rung and solo and dry"
    ... -k "rung and solo and single", "rung and solo and block", then pair_row, parse galaxy, galaxy (each first board run alone)

A rung: P1 of a full layer with the latent append and the key append, checkpoint site and norm of layer 1 on the
golden's streams, random projection and indexer weights. On a part of the mesh the absent chips' Xa payloads (q_a
tiles, lane sums, pool tiles of the latent row AND of the key, from the host golden) are written into the landings
first. Checked on every chip that takes part (lines start with HY4_MK8A):
  words     nothing left in any status word or counter (X3's arrivals aside)
  key       the 4 key tiles as they landed, the keys before the Hadamard basis and the packed tiles: the same on
            every chip; the finished keys against float64 arithmetic on the landed tiles (LayerNorm with the layer's
            weight and bias, the interleaved rotation); the packed tiles' values against (those BF16 keys) @ H in
            float64, in steps of each face row's own exponent; and against ttnn.matmul of the same BF16 keys with
            the same matrix into a BFP8 tensor on the device (the prefill indexer's last step)
  blocks    for every block a chip's writer touched (it keeps the block as it READ it and as DRAM holds it after its
            writes drained): the address against fused/kv_store's frames.locate; the values of the bytes read
            against ttnn's read of the same rows of the store before the pass (the BYTE layout of a BFP8 tile, and
            the address); after the pass every patched row's bytes are the packed row's, and the bytes of the other
            rows of the block (31, or fewer when the pass has several rows in it) are bit-equal to the bytes read
            before; the values of the bytes after against ttnn's read of the store after the pass
  stores    all 32 chips' key stores through ttnn before and after: nothing changed outside the patched rows
  latent    the latent append's own checks (tests/test_append.py check_append), on the same step
"dry": the writer reads and keeps the blocks and writes nothing (AttnLayer(key_dry=True)); the store must be unchanged.
Positions "single": one row a user, every row in a block of its own. "block": token rows 0, 1, 2 carry ONE
conversation (the same frame table) at positions 5, 6 and 37: two rows of one block and one in the next; token rows
3 and 4 carry one conversation at the SAME position: the higher token row's key stays.
"""

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.indexer_scan import indexer_scan as IS
from models.experimental.hy4_preview.fused.kv_store import frames as kv_frames
from models.experimental.hy4_preview.fused.layer_attn import program as A
from models.experimental.hy4_preview.fused.layer_attn.tests import key_bytes as KB
from models.experimental.hy4_preview.fused.layer_attn.tests.test_append import IDLE, RUNGS, AppendCase, check_append
from models.experimental.hy4_preview.fused.layer_attn.tests.test_p1 import ALL, galaxy, gain, golden_layer
from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.tests.common import pcc
from models.experimental.hy4_preview.tt import frames

KEY_LAYERS, RANK = 3, 1  # layer slots a user in the key store, and this layer's
KEY_ROWS = 512  # stripe rows per slot: two frames; token row 6's table maps only the first
POSITIONS = {
    # owners chip 0 (twice: rows 5 and 131), chip 1, chip 4, chip 17, chip 31; one in the second frame of a user
    # whose table has only one (not written); one idle row
    "single": [5, 4096 + 3, 128 + 9, 4 * 128 + 127, 17 * 128 + 64, 31 * 128 + 1, 2 * 4096 + 11, IDLE],
    "block": [5, 6, 37, 4096 + 3, 4096 + 3, 31 * 128 + 1, 2 * 4096 + 11, IDLE],
    # on the prefill stack's own key tensor (test_prefill_block): two rows of chip 0 the stack has written
    "prefill": [5, 100, IDLE, IDLE, IDLE, IDLE, 2 * 4096 + 11, IDLE],
}
SHARED = {
    "single": {},
    "block": {1: 0, 2: 0, 4: 3},
    "prefill": {},
}  # token row -> the token row whose conversation it carries
UNMAPPED_USER = 6


class KeyCase(AppendCase):
    def __init__(self, mesh_device, positions="single", chips=None, dry=False, store=None, **kw):
        """store: (key tensor, layer slots a user, this layer's rank, stripe rows a slot, a reader of the tensor's
        values [chips, slots, rows, 128]) of a store that exists (the prefill stack's); default: a store of random
        keys in the prefill stack's memory layout."""
        self.kind = positions
        super().__init__(mesh_device, "truncate", chips=chips, full=True, key_dry=dry, **kw)
        a = self.a
        if store is None:
            g = torch.Generator().manual_seed(11)
            slots = self.cfg.users * KEY_LAYERS
            # the store before: keys of very different magnitudes from row to row (a row's exponents are its own)
            scale = 2.0 ** torch.randint(-6, 7, (slots, KEY_ROWS, 1), generator=g).float()
            self.keys_host = torch.randn(slots, KEY_ROWS, L.IDX_DIM, generator=g) * scale
            self.keys = IS.keys_to_device(mesh_device, self.keys_host)
            self.layers, self.rank, self.rows, self.read_store = KEY_LAYERS, RANK, KEY_ROWS, None
        else:
            self.keys, self.layers, self.rank, self.rows, self.read_store = store
        geometry = kv_frames.Geometry.contiguous(
            kv_frames.Kind("index", self.layers, L.IDX_DIM, A.KEY_BLOCK_BYTES, 0), self.rows // 256
        )
        self.tables = [geometry.identity_table(u) for u in range(self.cfg.users)]
        self.tables[UNMAPPED_USER] = geometry.identity_table(UNMAPPED_USER, frames=1)
        for u, v in SHARED[positions].items():
            self.tables[u] = self.tables[v]
        self.conversation = [SHARED[positions].get(u, u) for u in range(self.cfg.users)]
        a.set_keys(self.keys, self.rank, self.layers, tables=self.tables)
        self.geometry = a.key_geometry
        self.positions = list(POSITIONS[positions])
        real = torch.tensor([p if p != IDLE else 0 for p in self.positions])
        from models.experimental.hy4_preview.reference.attention import AttentionConfig, rope_cos_sin

        cos, sin = rope_cos_sin(real, AttentionConfig())
        self.cos, self.sin = frames.rope_tables(cos, sin)
        a.set_rope(self.cos, self.sin)
        a.set_positions(self.positions)
        ttnn.synchronize_device(mesh_device)

    def prefill_absent(self, x):
        """Xa of the chips that do not take part: q_a, lane sums, the latent row's pool tiles and the key's."""
        super().prefill_absent(x)
        updates = {}
        ones = torch.ones(L.ROWS)
        for coord in ALL:
            chip = self.a.topo.chip(coord)
            if coord in self.a.topo.chips or (L.pool_of(chip, True) or ("", 0))[0] != "key":
                continue
            g = L.pre_golden(x, ones, self.share(chip)[1])
            for key, items in self.a.key_prefill(coord, g["pool"].to(torch.bfloat16).float()).items():
                updates.setdefault(key, []).extend(items)
        if updates:
            self.a.write_arena(updates)

    def key_values(self):
        """[32 chips, slots, rows, 128]: every chip's key store as ttnn reads it (the stored values)."""
        if self.read_store is not None:
            return self.read_store().float().clone()
        t = ttnn.to_torch(self.keys, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0)).float()
        return t.reshape(len(ALL), -1, self.rows, L.IDX_DIM).clone()

    def slot(self, u):
        return self.conversation[u] * self.layers + self.rank

    def writes(self):
        """{chip: {(bank, offset): {tile row: token row}}} of the chips that take part: what each must patch (of two
        token rows at one stripe row the higher one's key), from the positions and fused/kv_store's arithmetic."""
        out = {}
        present = {self.a.topo.chip(c) for c in self.a.topo.chips}
        slot_bytes = self.geometry.slot_bytes(self.rank)
        for u, p in enumerate(self.positions):
            if p == IDLE:
                continue
            chip, row = kv_frames.chip_of(p), kv_frames.stripe_row(p)
            if chip not in present or self.tables[u][row >> 8] == kv_frames.UNMAPPED:
                continue
            where = kv_frames.locate(self.tables[u], slot_bytes, row, 0)
            out.setdefault(chip, {}).setdefault(where, {})[row % 32] = u
        return out

    def twice(self):
        """{chip: (u, v)}: the first pair of token rows u < v given the same stripe row of one block on a chip that
        takes part (what the key core must report)."""
        out = {}
        for chip, blocks in self.writes().items():
            pairs = []
            slot_bytes = self.geometry.slot_bytes(self.rank)
            mine = [
                u
                for u, p in enumerate(self.positions)
                if p != IDLE
                and kv_frames.chip_of(p) == chip
                and self.tables[u][kv_frames.stripe_row(p) >> 8] != kv_frames.UNMAPPED
            ]
            at = {
                u: (
                    kv_frames.locate(self.tables[u], slot_bytes, kv_frames.stripe_row(self.positions[u]), 0),
                    kv_frames.stripe_row(self.positions[u]) % 32,
                )
                for u in mine
            }
            pairs = [(u, v) for u in mine for v in mine if u < v and at[u] == at[v]]
            if pairs:
                out[chip] = min(pairs)
        return out

    def block_rows(self, where):
        """(slot, first stripe row) of the block at (bank, offset) of the slot-major key store."""
        bank, offset = where
        shard = offset // A.KEY_BLOCK_BYTES  # the bank's shard index: slot * (rows / 256) + frame of the slot
        slot, frame = divmod(shard, self.rows // 256)
        return slot, (frame * 8 + bank) * 32


def steps_off(values, exact, exps):
    """|values - exact| in steps of each face row's exponent ([8, 128] each; exps [8 rows, 4 tiles, 2 halves])."""
    step = np.exp2(exps.astype(np.float64) - 127 - 6).repeat(16, axis=-1).reshape(8, 128)
    return np.abs(values - exact) / step


def check_finish(case, label):
    """The status words and the key core's finished keys and packed tiles on every chip."""
    a = case.a
    topo = a.topo
    failed = []
    # the status: empty, except on the owner chip of a stripe row that two token rows of the pass were given (the
    # caller's error: the key core reports the first such pair in its status word and stores the higher row's key)
    status = a.status(p1_only=True)
    expected = {}
    for chip, by_block in case.twice().items():
        coord = next(c for c in topo.chips if topo.chip(c) == chip)
        u, v = by_block
        expected[(coord, A.KEY, "attn")] = {A.W_ASTATUS: hex(A.ST_KEY_SAME_ROW | u | v << 8)}
    if status != expected:
        failed.append(f"status words {status}, expected {expected}")
    images = {c: a.read_arena(c) for c in topo.chips}
    got = {c: a.read_key(c, images[c]) for c in topo.chips}
    first = got[topo.chips[0]]
    same = sum(
        torch.equal(got[c]["raw"], first["raw"])
        and torch.equal(got[c]["rot"], first["rot"])
        and np.array_equal(got[c]["packed"], first["packed"])
        for c in topo.chips
    )
    if same != len(topo.chips):
        failed.append(
            f"the key tiles, the finished keys or the packed tiles differ between chips ({same} of {len(topo.chips)} equal the first)"
        )
    # the chips' own key tiles against exact arithmetic on the stored weights for the device's x
    x = a.read_root()["x"]
    ones = torch.ones(L.ROWS)
    worst_pool = 1.0
    for c in topo.chips:
        chip = topo.chip(c)
        pool = L.pool_of(chip, True)
        if pool is None or pool[0] != "key":
            continue
        want = L.pre_golden(x, ones, case.share(chip)[1])["pool"]
        worst_pool = min(worst_pool, pcc(want, first["raw"][:, pool[1] * L.TILE : (pool[1] + 1) * L.TILE]))
    # the finished keys against float64 arithmetic on the landed tiles
    raw, rot = first["raw"].double(), first["rot"].double()
    w, b = case.lw.key_norm_w.double(), case.lw.key_norm_b.double()
    mean = raw.mean(dim=1, keepdim=True)
    var = (raw - mean).pow(2).mean(dim=1, keepdim=True)
    k = (raw - mean) * torch.rsqrt(var + a.key_eps) * w + b
    k = torch.cat((L.interleaved_rope(k[:, : L.ROPE], case.cos.double(), case.sin.double()), k[:, L.ROPE :]), dim=1)
    p_rot, g_rot = pcc(k, rot), gain(k, rot)
    med = float(((rot - k).abs() / k.abs().clamp(min=1e-30)).median())
    # the packed tiles against the BF16 keys in the Hadamard basis
    exact = (rot @ frames.hadamard(L.IDX_DIM).double()).numpy()
    packed = first["packed"]
    values = KB.values(packed)[: L.ROWS]
    exps = KB.split(packed)[0][:, : L.ROWS].transpose(1, 0, 2)  # [row, tile, half]
    off = steps_off(values, exact, exps)
    p_pack = pcc(torch.from_numpy(exact), torch.from_numpy(values))
    logger.info(
        f"HY4_MK8A {label}: status {status or 'empty'}; key tiles of the chips that take part vs exact arithmetic on the "
        f"stored weights pcc min {worst_pool:.6f}; finished keys (before the Hadamard basis) vs float64 on the landed "
        f"tiles: pcc {p_rot:.7f} gain {g_rot:.5f} (median relative error {med:.2e}); packed tiles vs (those BF16 keys) @ H "
        f"in float64: pcc {p_pack:.7f}, largest error {off.max():.3f} steps of the face row's exponent (mean "
        f"{off.mean():.3f}); the same on {same} of {len(topo.chips)} chips"
    )
    if worst_pool < 0.9995 or p_rot < 0.9999 or abs(g_rot - 1) > 5e-3 or off.max() > 1.0 + 1e-6:
        failed.append(
            f"key tiles pcc {worst_pool:.6f}, finished keys pcc {p_rot:.6f} gain {g_rot:.4f}, packed off by {off.max():.3f} steps"
        )
    reference = (first["rot"].clone(), values, exact, exps)
    return failed, got, packed, reference


def key_reference(case, reference, label):
    """The packed keys against ttnn's own matmul of the same BF16 keys with the same matrix into a BFP8 tensor on the
    device (the prefill indexer's last step: ttnn.matmul(k, H, dtype=bfloat8_b)). A stock op, run AFTER every status
    and read-back of the step has been taken; reported, not asserted."""
    rot, values, exact, exps = reference
    topo = case.a.topo
    padded = torch.zeros(1, 1, 32, L.IDX_DIM)
    padded[0, 0, : L.ROWS] = rot
    dev = lambda t, dtype: ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=case.mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(case.mesh)
    )
    config = ttnn.init_device_compute_kernel_config(
        case.mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    left, right = dev(padded, ttnn.bfloat16), dev(
        frames.hadamard(L.IDX_DIM).reshape(1, 1, L.IDX_DIM, L.IDX_DIM), ttnn.bfloat16
    )
    ref = ttnn.matmul(
        left, right, dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=config
    )
    ref_values = (
        ttnn.to_torch(ttnn.get_device_tensors(ref)[topo.chip(topo.chips[0])])
        .double()
        .reshape(32, L.IDX_DIM)[: L.ROWS]
        .numpy()
    )
    for t in (left, right, ref):
        ttnn.deallocate(t)
    differ = int((ref_values != values).sum())
    ref_off = steps_off(ref_values, exact, exps)
    logger.info(
        f"HY4_MK8A {label}: the packed keys against ttnn.matmul of the same BF16 keys into BFP8 on the device: {differ} of "
        f"{values.size} stored values differ (largest difference {float(np.abs(ref_values - values).max()):.3e}; ttnn's "
        f"own largest error against float64 {ref_off.max():.3f} steps of the face row's exponent, mean {ref_off.mean():.3f})"
    )


def check_key(case, before_values, label, previous=None):
    """The checks of the module's head; every read of the device is taken here. Returns (the blocks' bytes after the
    pass, what key_reference needs)."""
    a = case.a
    topo = a.topo
    after_values = case.key_values()
    failed, got, packed, reference = check_finish(case, label)
    wanted = case.writes()
    touched = torch.zeros(after_values.shape[:3], dtype=torch.bool)
    blocks = rows_patched = other_rows = 0
    for c in topo.chips:
        chip = topo.chip(c)
        want = wanted.get(chip, {})
        groups = {g["where"]: g for g in got[c]["groups"]}
        if set(groups) != set(want):
            failed.append(f"chip {chip}: blocks {sorted(groups)} read, {sorted(want)} expected")
            continue
        for where, rows in want.items():
            g = groups[where]
            slot, r0 = case.block_rows(where)
            blocks += 1
            if not np.array_equal(KB.values(g["before"]), before_values[chip, slot, r0 : r0 + 32].double().numpy()):
                failed.append(
                    f"chip {chip} block {where}: the bytes read are not the store's rows {r0}.. of slot {slot} (layout or address)"
                )
            if previous is not None and not np.array_equal(g["before"], previous[c][where]):
                failed.append(f"chip {chip} block {where}: the bytes read are not the bytes the step before left")
            if not np.array_equal(KB.values(g["after"]), after_values[chip, slot, r0 : r0 + 32].double().numpy()):
                failed.append(f"chip {chip} block {where}: the bytes read back are not what ttnn reads after the pass")
            keep = np.ones(KB.BLOCK_BYTES, dtype=bool)
            for r, u in ({} if a.key_dry else rows).items():
                keep &= KB.other_rows_mask(r)
                touched[chip, slot, r0 + r] = True
                rows_patched += 1
                if not np.array_equal(KB.row_bytes(g["after"], r), KB.row_bytes(packed, u)):
                    failed.append(f"chip {chip} block {where} row {r}: not token row {u}'s packed key")
            changed = int((g["after"][keep] != g["before"][keep]).sum())
            other_rows += 32 - (0 if a.key_dry else len(rows))
            if changed:
                failed.append(f"chip {chip} block {where}: {changed} bytes of the other rows changed")
    outside = int((after_values != before_values).any(dim=-1)[~touched].sum())
    logger.info(
        f"HY4_MK8A {label}{' [dry]' if a.key_dry else ''}: {blocks} blocks read on their owner chips "
        f"{ {chip: sorted(w) for chip, w in wanted.items()} }: addresses and bytes as the store's, {rows_patched} rows "
        f"patched with their packed keys, {other_rows} other rows of those blocks bit-equal before and after; rows changed "
        f"anywhere else in the 32 stores: {outside}"
    )
    if outside:
        failed.append(f"{outside} rows changed outside the patched ones")
    assert not failed, f"{label}: " + "; ".join(failed[:8])
    return {c: {g["where"]: g["after"] for g in got[c]["groups"]} for c in topo.chips}, reference


@galaxy
def test_mock(mesh_device):
    if not A.is_mock():
        pytest.skip("the mock's compile and size check")
    from models.experimental.hy4_preview.fused.layer_attn.tests.test_p1 import cb_report

    for dry in (True, False):
        case = KeyCase(mesh_device, dry=dry)
        a = case.a
        a.run_p1()
        a.run_p1(post=True)
        if not dry:
            names = (("root", (0, 0)), ("key tile", (4, 2)), ("head weights", (5, 2)), ("no pool tile", (7, 3)))
            cb_report(mesh_device, a, lambda c: a._p1_chip(c, True), names)
            for name, coord in names:
                for kernel, proc, noc, cores in A.noc_audit(a._p1_chip(coord, True)[0]):
                    if kernel.startswith(("attn_key", "attn_append")):
                        logger.info(
                            f"HY4_MK8A key mock NoC {name} chip: {kernel} processor {proc} NoC {noc} on {cores} core(s)"
                        )
            ops = {c: a.parsed[("p1", c)][4] for c in ((0, 0), (0, 1), (4, 2), (5, 2), (7, 3))}
            logger.info(f"HY4_MK8A key mock: sender ops per chip {ops}; persistent L1 {a.l1_bytes()} B per core")
        ttnn.deallocate(case.kv)
        ttnn.deallocate(case.keys)
        a.release()


@galaxy
@pytest.mark.parametrize("positions", ["dry", "single", "block"])
@pytest.mark.parametrize("rung", list(RUNGS))
def test_rung(mesh_device, rung, positions):
    hf = golden_layer(1)
    dry = positions == "dry"
    case = KeyCase(mesh_device, "single" if dry else positions, chips=RUNGS[rung], dry=dry)
    a = case.a
    streams = hf["streams_in"][slice(0, 8)].float()
    a.set_streams(streams)
    _, _, x = case.golden_x(streams)
    case.prefill_absent(x.to(torch.bfloat16).float())
    if A.is_mock():
        a.run_p1()
        return
    latent_before = case.store_bytes()
    before = case.key_values()
    previous = None
    for step in (1, 2):
        a.run_p1()
        ttnn.synchronize_device(mesh_device)
        label = f"key append {rung} {positions} step {step}"
        after = case.key_values()
        previous, reference = check_key(case, before, label, previous)
        if positions == "single":  # (the latent rows of token rows that share a position are not this test's)
            check_append(case, latent_before, f"{label}, the latent rows")
        key_reference(case, reference, label)  # (a stock ttnn op: after every status and read-back of the step)
        before = after
        a.reset_words()


@galaxy
@pytest.mark.parametrize("rung", ["pair_row", "galaxy"])
def test_parse(mesh_device, rung):
    """The senders' argument tables of a full layer's P1 with the pool ops of the latent row (chips 0..17, to the
    append core) and of the key (chips 18..21, to the key core), as the kernel reads them, with no fabric connection
    opened and nothing sent (the collective sender's parse-only build): every chip's page must equal the table the
    host built. The other kernels run into their bounded waits; the key writer is built dry and the latent rows the
    append core writes are rows of zeros at the owned positions: nothing of the stores is checked here."""
    hf = golden_layer(1)
    case = KeyCase(mesh_device, chips=RUNGS[rung], dry=True, wait_limit=2_000_000, parse_only=True)
    a = case.a
    a.set_streams(hf["streams_in"][slice(0, 8)].float())
    ttnn.synchronize_device(mesh_device)
    a.run_p1()
    if A.is_mock():
        return
    ttnn.synchronize_device(mesh_device)
    wrong = []
    for c in a.topo.chips:
        want = a.parsed[("p1", c)]
        page = a.read_parsed(c, len(want))
        logger.info(
            f"HY4_MK8A key parse {rung} chip {c}: {len(want)} words, {'equal to the host table' if page == want else 'DIFFERENT'}; "
            f"ops {want[4]}, connections {want[5]}, routes {want[6]}, fabric arguments from index {want[-2]}"
        )
        if page != want:
            wrong.append((c, [(i, hex(x), hex(y)) for i, (x, y) in enumerate(zip(page, want)) if x != y][:8]))
    a.reset()
    assert not wrong, f"{rung}: the sender parsed its table differently: {wrong}"


@galaxy
def test_latency(mesh_device):
    """What the key append costs P1 of a full layer: P1 on 32 chips as a trace (8 per trace and as a trace of its
    own) without any append and with both (the latent row's and the key's), in the same process; then the stamped
    build: on every chip, from the moment x arrived on it, when the key tiles had arrived on the key core, when the
    compute kernel had its tiles, when the blocks were written, beside the append core's "rows written", the chip's
    X3 hand-over and its last X3 landing (the end of P1 without the appends). The writer keeps nothing here
    (append_keep=False). P1 runs alone, so the words are zeroed between the measurements."""
    import os
    import statistics

    from models.experimental.hy4_preview.fused.common import tracing
    from models.experimental.hy4_preview.fused.layer_attn.tests.test_p1 import Case, us

    hf = golden_layer(1)
    streams = hf["streams_in"][slice(0, 8)].float()
    fid = os.environ.get("HY4_ATTN_FIDELITY", "recommended")
    out = {}
    for name, make in (
        (
            "without the appends",
            lambda timing: Case(mesh_device, full=True, timing=timing, fidelity=fid, watch_x3=True),
        ),
        (
            "with both appends",
            lambda timing: KeyCase(mesh_device, timing=timing, fidelity=fid, watch_x3=True, append_keep=False),
        ),
    ):
        for timing in (False, True) if name == "with both appends" else (False,):
            case = make(timing)
            a = case.a
            a.set_streams(streams)
            ttnn.synchronize_device(mesh_device)
            a.run_p1()
            if A.is_mock():
                a.release()
                continue
            ttnn.synchronize_device(mesh_device)
            for calls in (1, 8):
                a.reset_words()
                r = tracing.traced_ms(mesh_device, a.run_p1, calls=calls)
                out[(name, timing, calls)] = r["replay_ms"] * 1e3
            logger.info(
                f"HY4_MK8A key latency [{fid}] P1 of a full layer {name}, timing={timing}: {out[(name, timing, 8)]:.1f} us per "
                f"call with 8 per trace, {out[(name, timing, 1)]:.1f} as a trace of its own; transport words after: {a.t.status() or 'at rest'}"
            )
            if timing:
                topo = a.topo
                words = a.t.read_words()
                five = lambda c, core: [int(v) for v in a.t.core_words(words[c], core)[[8, 9, 10, 11, 15]]]
                rows = {
                    "key tiles arrived": [],
                    "key tiles handed to the compute kernel": [],
                    "key blocks written": [],
                    "latent rows written": [],
                    "X3 handed over": [],
                    "last X3 landing": [],
                    "key blocks written after the last X3 landing": [],
                }
                for c in topo.chips:
                    p, q, ap, ky = five(c, A.PRE_HUB), five(c, A.QUERY_HUB), five(c, A.APPEND), five(c, A.KEY)
                    t0 = p[1]
                    rows["key tiles arrived"].append(us(ky[2] - t0))
                    rows["key tiles handed to the compute kernel"].append(us(ky[3] - t0))
                    rows["key blocks written"].append(us(ky[4] - t0))
                    rows["latent rows written"].append(us(ap[4] - t0))
                    rows["X3 handed over"].append(us(q[4] - t0))
                    if c != topo.root:
                        last = max(us(five(c, core)[1] - t0) for core in A.MERGE)
                        rows["last X3 landing"].append(last)
                        rows["key blocks written after the last X3 landing"].append(us(ky[4] - t0) - last)
                for key, v in rows.items():
                    logger.info(
                        f"HY4_MK8A key stages [{fid}] per chip, us after x arrived on the chip: {key}: min {min(v):.1f} "
                        f"median {statistics.median(v):.1f} max {max(v):.1f}"
                    )
            a.reset()
            for name_ in ("kv", "keys"):
                if hasattr(case, name_):
                    ttnn.deallocate(getattr(case, name_))
            a.release()
    if out:
        logger.info(
            f"HY4_MK8A key latency [{fid}]: P1 of a full layer {out[('without the appends', False, 8)]:.1f} us without and "
            f"{out[('with both appends', False, 8)]:.1f} us with the latent and the key append (8 per trace); "
            f"{out[('without the appends', False, 1)]:.1f} and {out[('with both appends', False, 1)]:.1f} as traces of their own"
        )


@galaxy
@pytest.mark.parametrize("mode", ["dry", "patch"])
def test_prefill_block(mesh_device, mode):
    """The key append on blocks the PREFILL STACK wrote: the stack (layers 0..5: its index tensor has the 3 full
    layers 0, 1, 5 a user) prefills 300 tokens into users 0 and 1 and frees its weights; then P1 of a full layer on
    chip (0, 0) with host-written partners takes the stack's own index tensor as its key store (rank 1, the stack's
    slot arithmetic through the identity frame tables) at positions 5 (user 0) and 100 (user 1), rows the stack has
    written. "dry": the writer reads the two blocks and writes nothing: the bytes it read, decoded by the tile model
    (tests/key_bytes.py), must be the values the stack's own reader gives for those 32 rows (the byte layout of a
    tile the device's packer wrote, and the address in the stack's tensor). "patch": the two rows are patched; the
    other 31 rows of each block, written by the stack, must be bit-equal before and after, and nothing else in the
    stack's index tensor on 32 chips may change."""
    if A.is_mock():
        pytest.skip("the prefill stack is not built on the mock")
    from models.experimental.hy4_preview.tt import prefill_bridge as pb

    tokens = [int(t) for t in (torch.arange(300) * 7919 + 1000) % 100000]
    bridge = pb.PrefillBridge(mesh_device, num_layers=6, num_users=8, max_seq_len=8192)
    bridge.build()
    for user in (0, 1):
        bridge.prefill(tokens, user=user)
    bridge.release_model()
    full = pb.full_layers(bridge.indexer_types)
    assert full == [0, 1, 5], full
    index = bridge.kv_caches.index
    rows = 8192 // 32
    read = lambda: bridge.read_slabs(index)
    written = read().float()
    logger.info(
        f"HY4_MK8A key append on the prefill stack's tensor: index tensor per chip {tuple(written.shape[1:])}; rms of the "
        f"rows the stack wrote on chip 0, user 0, rank 1: rows 0..31 {float(written[0, 1, :32].pow(2).mean().sqrt()):.4f}, "
        f"rows 96..127 {float(written[0, 1, 96:128].pow(2).mean().sqrt()):.4f}; rows 128.. (never written) "
        f"{float(written[0, 1, 128:].pow(2).mean().sqrt()):.4f}"
    )
    # (every one of the 128 rows of chip 0 that the two users' prefills wrote holds a key)
    assert (
        float(written[0, 1, :128].abs().sum(dim=-1).min()) > 0
        and float(written[0, 4, :128].abs().sum(dim=-1).min()) > 0
    )
    hf = golden_layer(1)
    case = KeyCase(
        mesh_device, "prefill", chips=RUNGS["solo"], dry=mode == "dry", store=(index, len(full), 1, rows, read)
    )
    a = case.a
    streams = hf["streams_in"][slice(0, 8)].float()
    a.set_streams(streams)
    _, _, x = case.golden_x(streams)
    case.prefill_absent(x.to(torch.bfloat16).float())
    before = case.key_values()
    previous = None
    for step in (1, 2):
        a.run_p1()
        ttnn.synchronize_device(mesh_device)
        label = f"key append on the prefill stack's blocks [{mode}] step {step}"
        after = case.key_values()
        previous, reference = check_key(case, before, label, previous)
        before = after
        a.reset_words()
