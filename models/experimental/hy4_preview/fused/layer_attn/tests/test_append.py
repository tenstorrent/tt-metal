# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The latent append of program P1 (fused/layer_attn/program.py, PLAN.md section 17): the new token's cache row of
the 8 token rows, finished on the append core of every chip (kv norm, rotation, the FP8 pack) and written by the
chip that owns the row's position, at the address fused/kv_store's frame table gives.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file>
    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k "parse and pair_row"
    ... -k "rung and solo and truncate", "rung and solo and nearest", then pair_row, pair_col, galaxy (each first board run alone)

A rung: P1 with the append, checkpoint site and norm of layer 1 on the golden's streams, random projection weights.
On a part of the mesh the absent chips' Xa payloads (q_a tiles, lane sums, pool tiles, from the host golden) are
written into the landings first. Checked on every chip that takes part (lines start with HY4_MK8A):
  words     nothing left in any status word or counter (X3's arrivals aside, which P2 takes)
  pool      the 18 pool tiles as they landed: every chip holds the same; the tiles of the chips that take part
            against exact arithmetic on their stored weights for the device's x
  rows      the BF16 rows the append core finished (kept in its arena) against float64 arithmetic on the landed pool
            tiles: k_pass = rms_norm(raw, weight, eps), k_rope = the interleaved rotation; the same on every chip
  store     the 656-byte rows of the positions a chip owns, read back from the store, against the prefill stack's
            own pack (per_token_cast_to_fp8 with power-of-two scales + pack_scaled_fp8_kv_cache) of those BF16
            rows, run on the same chip: "truncate" bit for bit; "nearest" the same scales and RoPE bytes, every
            e4m3 value within one step of the prefill pack's and at most half a step from the BF16 value. Every
            other byte of the store on every chip unchanged (rows of other chips' positions, idle rows, unmapped
            frames).
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheGeometry, unpack_scaled_fp8_kv_cache
from models.experimental.hy4_preview.fused.layer_attn import program as A
from models.experimental.hy4_preview.fused.layer_attn.tests.test_p1 import ALL, Case, galaxy, gain, golden_layer
from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.fused.sparse_mla import layout as SL
from models.experimental.hy4_preview.fused.sparse_mla.tests import harness as hs
from models.experimental.hy4_preview.reference.attention import AttentionConfig, rms_norm, rope_cos_sin
from models.experimental.hy4_preview.tests.common import pcc
from models.experimental.hy4_preview.tt import frames

RUNGS = {"solo": [(0, 0)], "pair_row": [(0, 0), (0, 1)], "pair_col": [(0, 0), (1, 0)], "galaxy": None}
ROWS_PER_SLOT = 512  # two frames of 256 stripe rows per user and chip; the second user's table maps only the first
IDLE = 0xFFFFFFFF
# positions of the 8 token rows: owners chip 0 (twice: rows 5 and 131), chip 1, chip 4, chip 17, chip 31; one in
# the second frame of a user whose table has only one (not written); one idle row
POSITIONS = [5, 4096 + 3, 128 + 9, 4 * 128 + 127, 17 * 128 + 64, 31 * 128 + 1, 2 * 4096 + 11, IDLE]
UNMAPPED_USER = 6


class AppendCase(Case):
    def __init__(self, mesh_device, mode, chips=None, **kw):
        super().__init__(mesh_device, layer=1, chips=chips, append=mode, **kw)
        a = self.a
        self.mesh, self.mode = mesh_device, mode
        self.cfg = cfg = SL.SparseMlaConfig(rows_per_slot=ROWS_PER_SLOT, num_layers=1)
        latent, rope = hs.random_latents(cfg.users, cfg.rows_per_slot, cfg, seed=3)
        packed = hs.pack_rows(mesh_device, latent[:, None], rope[:, None])
        self.kv = ttnn.to_memory_config(packed, hs.kv_memory_config(mesh_device, cfg))
        ttnn.deallocate(packed)
        self.sel = hs.make_selection(mesh_device, cfg, [torch.zeros(0, dtype=torch.long)] * cfg.users)
        a.set_cache(cfg, self.kv, self.sel, 0)
        tables = [a.append_geometry().identity_table(u) for u in range(cfg.users)]
        tables[UNMAPPED_USER] = a.append_geometry().identity_table(UNMAPPED_USER, frames=1)
        a.set_frames(tables)
        self.positions = list(POSITIONS)
        real = torch.tensor([p if p != IDLE else 0 for p in self.positions])
        cos, sin = rope_cos_sin(real, AttentionConfig())
        self.cos, self.sin = frames.rope_tables(cos, sin)
        a.set_rope(self.cos, self.sin)
        a.set_positions(self.positions)
        ttnn.synchronize_device(mesh_device)

    def prefill_absent(self, x):
        """Xa of the chips that do not take part, pool tiles included."""
        super().prefill_absent(x)
        absent = [c for c in ALL if c not in self.a.topo.chips and L.pool_of(self.a.topo.chip(c), False) is not None]
        updates = {}
        ones = torch.ones(L.ROWS)
        for coord in absent:
            g = L.pre_golden(x, ones, self.share(self.a.topo.chip(coord))[1])
            for key, items in self.a.pool_prefill(coord, g["pool"].to(torch.bfloat16).float()).items():
                updates.setdefault(key, []).extend(items)
        if updates:
            self.a.write_arena(updates)

    def store_bytes(self):
        """[32 chips, 8 users, rows, 656] uint8: every chip's store as the host reads it."""
        physical = ttnn.to_torch(self.kv, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
        return (
            physical.view(torch.uint8)
            .reshape(len(ALL), self.cfg.users, self.cfg.rows_per_slot, self.cfg.row_bytes)
            .clone()
        )

    def owners(self):
        """{user: (chip, stripe row)} of the rows some chip that takes part must write."""
        out = {}
        present = {self.a.topo.chip(c) for c in self.a.topo.chips}
        for u, p in enumerate(self.positions):
            if p == IDLE or u == UNMAPPED_USER:
                continue
            chip, row = (p // 128) % 32, (p // 4096) * 128 + p % 128
            if chip in present:
                out[u] = (chip, row)
        return out


def e4m3_step(v):
    """The e4m3 step at |v| (v already divided by its scale)."""
    a = v.abs().double().clamp(min=2.0**-6)
    return 2.0 ** (torch.floor(torch.log2(a)) - 3)


def check_append(case, before, label):
    a, cfg, mode = case.a, case.cfg, case.mode
    topo = a.topo
    failed = []
    status = a.status(p1_only=True)
    if status:
        failed.append(f"words not at rest: {status}")
    images = {c: a.read_arena(c) for c in topo.chips}
    got = {c: a.read_append(c, images[c]) for c in topo.chips}
    first = got[topo.chips[0]]
    same = sum(
        torch.equal(got[c]["raw"], first["raw"]) and torch.equal(got[c]["rows"], first["rows"]) for c in topo.chips
    )
    if same != len(topo.chips):
        failed.append(
            f"the pool tiles or the finished rows differ between chips ({same} of {len(topo.chips)} equal the first)"
        )
    # the chips' own pool tiles against exact arithmetic on the stored weights for the device's x
    x = a.read_root()["x"]
    ones = torch.ones(L.ROWS)
    worst_pool = 1.0
    for c in topo.chips:
        chip = topo.chip(c)
        if L.pool_of(chip, False) is None:
            continue
        want = L.pre_golden(x, ones, case.share(chip)[1])["pool"]
        worst_pool = min(worst_pool, pcc(want, first["raw"][:, chip * L.TILE : (chip + 1) * L.TILE]))
    # the finished rows against float64 arithmetic on the landed tiles
    raw, rows = first["raw"].double(), first["rows"]
    k_pass = rms_norm(raw[:, : L.KV_LORA], case.lw.kv_norm.double(), a.kv_norm_eps)
    k_rope = L.interleaved_rope(raw[:, L.KV_LORA :], case.cos.double(), case.sin.double())
    p_pass, g_pass = pcc(k_pass, rows[:, : L.KV_LORA]), gain(k_pass, rows[:, : L.KV_LORA])
    p_rope, g_rope = pcc(k_rope, rows[:, L.KV_LORA :]), gain(k_rope, rows[:, L.KV_LORA :])
    step_err = float(((rows[:, : L.KV_LORA].double() - k_pass).abs() / k_pass.abs().clamp(min=1e-30)).median())
    logger.info(
        f"HY4_MK8A {label} [{mode}]: pool tiles of the chips that take part vs exact arithmetic on the stored weights "
        f"pcc min {worst_pool:.6f}; finished rows vs float64 on the landed tiles: k_pass pcc {p_pass:.7f} gain "
        f"{g_pass:.5f} (median relative error {step_err:.2e}), k_rope pcc {p_rope:.7f} gain {g_rope:.5f}; the same on "
        f"{same} of {len(topo.chips)} chips"
    )
    # (a pool tile is one 32-wide column block of the projections piece's pre units on 8 rows: 0.99988 to 0.99993
    # against exact arithmetic on the stored weights, chip by chip; the append's own arithmetic is the second check)
    if worst_pool < 0.9995 or p_pass < 0.9999 or p_rope < 0.9999 or abs(g_pass - 1) > 5e-3 or abs(g_rope - 1) > 5e-3:
        failed.append(
            f"pool pcc {worst_pool:.6f}, k_pass pcc {p_pass:.6f} gain {g_pass:.4f}, k_rope pcc {p_rope:.6f} gain {g_rope:.4f}"
        )
    # the store: the prefill stack's pack of the finished BF16 rows, on the device, as the reference bytes
    # (one slot of 32 rows, the token rows first: a shape the pack ops have run on)
    padded = torch.zeros(1, 1, 32, cfg.width)
    padded[0, 0, : cfg.users] = rows
    ref = hs.pack_rows(case.mesh, padded[..., : L.KV_LORA].contiguous(), padded[..., L.KV_LORA :].contiguous())
    ref_bytes = ttnn.to_torch(ref, mesh_composer=ttnn.ConcatMeshToTensor(case.mesh, dim=0))[:1]
    geometry = MlaKvCacheGeometry(latent_dim=cfg.latent, rope_dim=cfg.rope)
    ref_lat, ref_scale, ref_rope = (
        t.reshape(32, -1)[: cfg.users] for t in unpack_scaled_fp8_kv_cache(ref_bytes, geometry)
    )
    ref_bytes = ref_bytes.view(torch.uint8).reshape(32, cfg.row_bytes)[: cfg.users]
    ttnn.deallocate(ref)
    after = case.store_bytes()
    owners = case.owners()
    touched = torch.zeros(after.shape[:3], dtype=torch.bool)
    equal, within = 0, []
    for u, (chip, row) in owners.items():
        touched[chip, u, row] = True
        mine = after[chip, u, row]
        if mode == "truncate":
            equal += int(torch.equal(mine, ref_bytes[u]))
            if not torch.equal(mine, ref_bytes[u]):
                diff = (mine != ref_bytes[u]).nonzero().flatten()
                failed.append(
                    f"user {u} (chip {chip}, row {row}): {diff.numel()} bytes differ from the prefill pack, first at {int(diff[0])}"
                )
        else:
            lat, scale, rope_ = (
                t.reshape(-1)
                for t in unpack_scaled_fp8_kv_cache(mine.view(torch.float8_e4m3fn).reshape(1, 1, 1, -1), geometry)
            )
            if not (torch.equal(scale, ref_scale[u]) and torch.equal(rope_, ref_rope[u])):
                failed.append(f"user {u}: the scales or the RoPE bytes differ from the prefill pack's")
                continue
            s = scale.float().repeat_interleave(cfg.scale_block)
            value = rows[u, : L.KV_LORA].double() / s.double()  # what the cast divided
            code, trunc = lat.double(), ref_lat[u].double()
            step = e4m3_step(value)
            off = ((code - value).abs() > 0.5 * step * (1 + 1e-6)).sum()
            up = code - trunc
            if (
                int(off)
                or float(up.abs().max()) > float((2 * step).max())
                or float((up.sign() * value.sign()).min()) < 0
            ):
                failed.append(f"user {u}: {int(off)} e4m3 values are more than half a step from the BF16 value")
            within.append(float(((code.abs() - value.abs()) / value.abs().clamp(min=1e-30))[value.abs() > 0].mean()))
            equal += int(off == 0)
    untouched = int((after != before)[~touched].sum())
    logger.info(
        f"HY4_MK8A {label} [{mode}]: rows written by their owner chips {sorted(owners.items())}: "
        + (
            f"{equal} of {len(owners)} bit-equal to the prefill stack's pack of the same BF16 rows"
            if mode == "truncate"
            else f"{equal} of {len(owners)} with the prefill pack's scales and RoPE bytes and every e4m3 value within half a step of "
            f"the BF16 value; mean of (|stored| - |value|) / |value| {sum(within) / max(len(within), 1):+.4f}"
        )
        + f"; bytes changed anywhere else in the 32 stores: {untouched}"
    )
    if untouched:
        failed.append(f"{untouched} bytes changed outside the owned rows")
    if equal != len(owners):
        failed.append(f"{equal} of {len(owners)} rows as wanted")
    assert not failed, f"{label}: " + "; ".join(failed[:8])


@galaxy
@pytest.mark.parametrize("mode", ["truncate", "nearest"])
def test_mock(mesh_device, mode):
    if not A.is_mock():
        pytest.skip("the mock's compile and size check")
    from models.experimental.hy4_preview.fused.layer_attn.tests.test_p1 import cb_report

    case = AppendCase(mesh_device, mode)
    a = case.a
    a.run_p1()
    a.run_p1(post=True)
    cb_report(
        mesh_device, a, lambda c: a._p1_chip(c, True), (("root", (0, 0)), ("other", (0, 1)), ("no pool tile", (7, 3)))
    )
    logger.info(f"HY4_MK8A append mock [{mode}]: P1 with the append compiled; persistent L1 {a.l1_bytes()} B per core")


@galaxy
@pytest.mark.parametrize("mode", ["truncate", "nearest"])
@pytest.mark.parametrize("rung", list(RUNGS))
def test_rung(mesh_device, rung, mode):
    hf = golden_layer(1)
    case = AppendCase(mesh_device, mode, chips=RUNGS[rung])
    a = case.a
    streams = hf["streams_in"][slice(0, 8)].float()
    a.set_streams(streams)
    _, _, x = case.golden_x(streams)
    case.prefill_absent(x.to(torch.bfloat16).float())
    if A.is_mock():
        a.run_p1()
        return
    before = case.store_bytes()
    for step in (1, 2):
        a.run_p1()
        ttnn.synchronize_device(mesh_device)
        check_append(case, before, f"append {rung} step {step}")
        a.reset_words()
        # the second step writes the same rows again: restore nothing, the bytes must be the same


@galaxy
@pytest.mark.parametrize("rung", ["pair_row", "galaxy"])
def test_parse(mesh_device, rung):
    """The senders' argument tables of P1 with the pool op, as the kernel reads them on the chips of the rung, with
    no fabric connection opened and nothing sent (the collective sender's parse-only build): every chip's page must
    equal the table the host built. The other kernels run into their bounded waits (the append core then finishes
    and writes rows of zeros: nothing of the store is checked here); the words are zeroed afterwards."""
    hf = golden_layer(1)
    case = AppendCase(mesh_device, "truncate", chips=RUNGS[rung], wait_limit=2_000_000, parse_only=True)
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
            f"HY4_MK8A append parse {rung} chip {c}: {len(want)} words, {'equal to the host table' if page == want else 'DIFFERENT'}; "
            f"ops {want[4]}, connections {want[5]}, routes {want[6]}, fabric arguments from index {want[-2]}"
        )
        if page != want:
            wrong.append((c, [(i, hex(x), hex(y)) for i, (x, y) in enumerate(zip(page, want)) if x != y][:8]))
    a.reset()
    assert not wrong, f"{rung}: the sender parsed its table differently: {wrong}"


@galaxy
@pytest.mark.parametrize("mode", ["nearest"])
def test_latency(mesh_device, mode):
    """Does the append lengthen P1? P1 on 32 chips as a trace (8 per trace and as a trace of its own), without the
    append and with it, in the same process; then the stamped build with the append: on every chip, from the moment x
    arrived on it, when the pool tiles had arrived on the append core, when the compute kernel had its tiles, when
    the rows were written, against when the chip handed over X3 and when its last X3 landing arrived (the end of P1
    without the append). P1 runs alone here, so the words are zeroed between the measurements."""
    import statistics

    from models.experimental.hy4_preview.fused.common import tracing
    from models.experimental.hy4_preview.fused.layer_attn.tests.test_p1 import us

    hf = golden_layer(1)
    streams = hf["streams_in"][slice(0, 8)].float()
    fid = os.environ.get("HY4_ATTN_FIDELITY", "recommended")
    out = {}
    for name, make in (
        ("without the append", lambda timing: Case(mesh_device, timing=timing, fidelity=fid, watch_x3=True)),
        (
            "with the append",
            lambda timing: AppendCase(mesh_device, mode, timing=timing, fidelity=fid, watch_x3=True, append_keep=False),
        ),
    ):
        for timing in (False, True) if name == "with the append" else (False,):
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
                f"HY4_MK8A append latency [{fid}, {mode}] P1 {name}, timing={timing}: {out[(name, timing, 8)]:.1f} us per call "
                f"with 8 per trace, {out[(name, timing, 1)]:.1f} as a trace of its own; transport words after: {a.t.status() or 'at rest'}"
            )
            if timing:
                topo = a.topo
                words = a.t.read_words()
                five = lambda c, core: [int(v) for v in a.t.core_words(words[c], core)[[8, 9, 10, 11, 15]]]
                rows = {
                    "pool tiles arrived": [],
                    "tiles handed to the compute kernel": [],
                    "rows written": [],
                    "X3 handed over": [],
                    "last X3 landing": [],
                    "rows written after the last X3 landing": [],
                }
                for c in topo.chips:
                    p, q, ap = five(c, A.PRE_HUB), five(c, A.QUERY_HUB), five(c, A.APPEND)
                    t0 = p[1]
                    rows["pool tiles arrived"].append(us(ap[2] - t0))
                    rows["tiles handed to the compute kernel"].append(us(ap[3] - t0))
                    rows["rows written"].append(us(ap[4] - t0))
                    rows["X3 handed over"].append(us(q[4] - t0))
                    if c != topo.root:
                        last = max(us(five(c, core)[1] - t0) for core in A.MERGE)
                        rows["last X3 landing"].append(last)
                        rows["rows written after the last X3 landing"].append(us(ap[4] - t0) - last)
                for key, v in rows.items():
                    logger.info(
                        f"HY4_MK8A append stages [{fid}, {mode}] per chip, us after x arrived on the chip: {key}: min "
                        f"{min(v):.1f} median {statistics.median(v):.1f} max {max(v):.1f}"
                    )
            a.reset()
            ttnn.deallocate(case.kv) if hasattr(case, "kv") else None
            a.release()
    if out:
        logger.info(
            f"HY4_MK8A append latency [{fid}, {mode}]: P1 {out[('without the append', False, 8)]:.1f} us without and "
            f"{out[('with the append', False, 8)]:.1f} us with the append (8 per trace); "
            f"{out[('without the append', False, 1)]:.1f} and {out[('with the append', False, 1)]:.1f} as traces of their own"
        )
