# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The fused ends (model/ends_fused.py, PLAN.md section 46), piece by piece.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file>
    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k "test_in and solo"

test_in: kernels/ends_in.cpp on the root's site cores (data movement only, nothing over the fabric). A Stack with
random parameters; the host fills F, the streams and the coefficient page with other values, then:
  1. IN on 8 random ids: F equal to the table's rows bit for bit, the streams zero, the coefficients post = 1;
  2. the site's post program (what P1(post=True) of layer 0 does first): the 4 streams equal to the embedding row;
  3. IN again with two rows forced from the host and one id outside the table: the forced rows take the host's
     tokens, the others the fed-back ones, the id outside the table reads token 0;
  4. 20 launches back to back: the host's time of one (untraced).
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.ihc_site import layout as ihc_layout
from models.experimental.hy4_preview.fused.model import ends_fused as EF
from models.experimental.hy4_preview.fused.model.program import Stack, is_mock
from models.experimental.hy4_preview.fused.model.tests.test_mock import build_random, galaxy
from models.experimental.hy4_preview.reference.ihc import ihc_expand

C = 6144
RUNGS = {"solo": [(0, 0)], "all": None}
VOCAB = int(os.environ.get("HY4_FENDS_VOCAB", "4096"))


def random_table(vocab, seed=11):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(vocab, C, generator=g) * 0.05).to(torch.bfloat16)


def scribble(mesh, st, seed=5):
    """Other values in everything IN must write: F, the streams, the coefficient page."""
    g = torch.Generator().manual_seed(seed)
    st.a.set_f(torch.randn(8, C, generator=g))
    st.set_streams(torch.randn(8, 4, C, generator=g))
    page = torch.randn(1, 1, 32, 32 * st.site.K, generator=g)
    host = ttnn.from_torch(
        page, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )
    ttnn.copy_host_to_device_tensor(host, st.site.coef)
    ttnn.synchronize_device(mesh)


def root_coef(st, core=0):
    """(pre [8, 4], post [8, 4]) of the root's coefficient page on site core ``core``."""
    page = ttnn.to_torch(ttnn.get_device_tensors(st.site.coef)[st.topo.chip(st.topo.root)]).float()[0, 0]
    return ihc_layout.coef_from_page(page[:, 32 * core : 32 * core + 32], 8, st.site.lanes)


@galaxy
@pytest.mark.parametrize("rung", list(RUNGS))
def test_in(mesh_device, rung):
    mesh = mesh_device
    coords = RUNGS[rung]
    st = build_random(mesh, 2, **({} if coords is None else {"chips": coords}))
    table = random_table(VOCAB)
    t0 = time.perf_counter()
    ein = EF.EndsIn(st, table)
    ttnn.synchronize_device(mesh)
    logger.info(
        f"HY4_MK11 fused ends IN [{rung}]: table {VOCAB} x {C} BF16 to the device in {time.perf_counter() - t0:.1f} s; a core "
        f"moves {ein.slice} B of a token's row as {ein.slice // EF.PIECE} pieces to {ein.half} cores; L1 tensors {st.l1_bytes()} B per core"
    )
    g = torch.Generator().manual_seed(21)
    tokens = [int(v) for v in torch.randint(0, VOCAB, (8,), generator=g)]
    ids = EF.id_tensor(mesh, tokens)
    force = EF.id_tensor(mesh)
    post = st.layers[-1].mlp.post_program()
    scribble(mesh, st)
    ein.launch(ids)
    if is_mock():
        ein.launch(ids, force)
        Stack.launch([("post", *post)])
        logger.info(
            f"HY4_MK11 fused ends IN [{rung}]: compiled on the mock; program-cache entries {mesh.num_program_cache_entries()}"
        )
        return
    ttnn.synchronize_device(mesh)
    want = ein.host_f(table, tokens)
    f = st.a.read_root()["f"]
    streams = st.read_streams()
    pre, coef_post = root_coef(st)
    logger.info(
        f"HY4_MK11 fused ends IN [{rung}]: F equal to the table's rows {torch.equal(f, want)} (max abs {float((f - want).abs().max()):.3e}); "
        f"streams zero {not bool(streams.any())}; coefficients post = 1 {bool((coef_post == 1).all())}, pre = 0 {not bool(pre.any())}"
    )
    assert torch.equal(f, want), "F is not the embedding rows"
    assert not bool(streams.any()) and bool((coef_post == 1).all())
    assert all(bool((root_coef(st, k)[1] == 1).all()) for k in range(st.site.K)), "a site core's coefficient page"
    # the leading post-mix of layer 0's P1: streams += post x F
    Stack.launch([("post", *post)])
    ttnn.synchronize_device(mesh)
    streams = st.read_streams()
    expanded = ihc_expand(want.unsqueeze(0), 4)[0]
    logger.info(
        f"HY4_MK11 fused ends IN [{rung}]: after the site's post program the 4 streams equal the embedding row: {torch.equal(streams, expanded)}"
    )
    assert torch.equal(streams, expanded)
    # forced rows and an id outside the table
    fed = list(tokens)
    fed[6] = VOCAB + 5
    EF.write_ids(ids, fed)  # (the raw write: the asserting one, ein.write_ids, refuses this id)
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
        ein.write_ids(ids, fed)
    host = [None] * 8
    host[2], host[5] = (tokens[2] + 1) % VOCAB, (tokens[5] + 7) % VOCAB
    ein.write_ids(force, host, forced=True)
    scribble(mesh, st, seed=6)
    ein.launch(ids, force)
    ttnn.synchronize_device(mesh)
    now = list(tokens)
    now[2], now[5], now[6] = host[2], host[5], 0
    f = st.a.read_root()["f"]
    logger.info(
        f"HY4_MK11 fused ends IN [{rung}]: with rows 2 and 5 forced and row 6's id outside the table: F as expected {torch.equal(f, ein.host_f(table, now))}"
    )
    assert torch.equal(f, ein.host_f(table, now))
    status = st.status()
    assert not status, f"words not at rest: {status}"
    t0 = time.perf_counter()
    for _ in range(20):
        ein.launch(ids)
    ttnn.synchronize_device(mesh)
    logger.info(
        f"HY4_MK11 fused ends IN [{rung}]: 20 launches back to back, untraced: {(time.perf_counter() - t0) / 20 * 1e6:.0f} us each from the host"
    )
    f = st.a.read_root()["f"]
    fed[6] = 0
    assert torch.equal(f, ein.host_f(table, fed))


# ------------------------------------------------------------------------------------------------ HEAD
from models.experimental.hy4_preview.fused.layer_attn import program as A  # noqa: E402
from models.experimental.hy4_preview.reference.ihc import IHCConfig, IHCHead  # noqa: E402
from models.experimental.hy4_preview.tests.common import pcc  # noqa: E402

LAYERS = 2
K = int(os.environ.get("HY4_FENDS_K", "1"))


def random_head(seed=31):
    g = torch.Generator().manual_seed(seed)
    fn = torch.randn(4, 4 * C, generator=g) * 0.02
    base = torch.randn(4, generator=g)
    scale = torch.tensor([0.01])
    norm_w = 1.0 + 0.1 * torch.randn(C, generator=g)
    return fn, base, scale, norm_w


def random_shard(chip, seed=41):
    g = torch.Generator().manual_seed(seed + chip)
    return (torch.randn(EF.VOCAB_PER_CHIP, C, generator=g) * 0.05).to(torch.bfloat16)


def reference_x(streams, fn, base, scale, norm_w):
    """reference/ihc.py IHCHead on the streams [8, 4, C], then the final RMSNorm (eps 1e-5) with its weight: fp32."""
    cfg = IHCConfig(dim=C)
    head = IHCHead(cfg)
    head.fn.data, head.base.data, head.scale.data = fn.float(), base.float(), scale.float()
    with torch.no_grad():
        y = head(streams.float().unsqueeze(0))[0]
    return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + cfg.norm_eps) * norm_w.float()


def head_audit(head, coord):
    """One producer and one consumer processor per circular buffer of the HEAD's own kernels on a chip, and no
    buffer id declared twice on a core (the root also carries the site's buffers)."""
    desc, _ = head._chip_program(coord, EF.id_tensor(head.mesh), None)
    roles = {  # kernel file -> (pushes, pops or waits on)
        "stream_reader.cpp": ([EF.CB_W], []),
        "head_unit.cpp": ([EF.CB_X], [EF.CB_OUT]),
        "head_compute.cpp": ([EF.CB_OUT], [EF.CB_X, EF.CB_W]),
        "head_hub.cpp": ([], []),
    }
    table = {}
    for kernel in desc.kernels:
        name = str(kernel.kernel_source).split("/")[-1]
        if name not in roles:
            continue
        for kind, ids in zip(("push", "pop"), roles[name]):
            for core in ttnn.corerange_to_cores(kernel.core_ranges):
                for cb in ids:
                    table.setdefault(((core.x, core.y), cb), {"push": [], "pop": []})[kind].append(name)
    bad = {k: v for k, v in table.items() if len(v["push"]) != 1 or len(v["pop"]) != 1}
    assert not bad, f"{coord}: buffers without exactly one producer and one consumer: {list(bad.items())[:4]}"
    seen = set()
    for cb in desc.cbs:
        for fmt in cb.format_descriptors:
            for core in ttnn.corerange_to_cores(cb.core_ranges):
                key = (core.x, core.y, fmt.buffer_index)
                assert key not in seen, f"{coord}: circular buffer {fmt.buffer_index} twice on core {(core.x, core.y)}"
                seen.add(key)
    per_core = {}
    for cb in desc.cbs:
        backed = cb.has_buffer() if callable(cb.has_buffer) else cb.has_buffer
        if not backed:
            for c in ttnn.corerange_to_cores(cb.core_ranges):
                per_core[(c.x, c.y)] = per_core.get((c.x, c.y), 0) + cb.total_size
    worst = max(per_core, key=per_core.get)
    return len(table), len(desc.kernels), len(desc.cbs), per_core[worst], worst


def column_rounds(head, logits, want):
    """{"first column of a two-column unit" | "second column" | "the one-column units": (ids, pcc, max abs)} of the
    device's logits [8, 3776] against the host's: a fault of the compute kernel's second round shows as its own
    line."""
    groups = {"first": [], "second": [], "single": []}
    for col in range(EF.COLS):
        i = col % EF.COLS_PER_BANK
        name = "single" if i == EF.COLS_PER_BANK - 1 else ("first" if i % 2 == 0 else "second")
        groups[name] += [t for t in range(col * 32, col * 32 + 32) if t < logits.shape[-1]]
    out = {}
    for name, ids in groups.items():
        idx = torch.tensor(ids)
        a, b = logits[:, idx], want[:, idx]
        out[name] = (len(ids), round(pcc(b, a), 8), float(f"{float((a - b).abs().max()):.3e}"))
    return out


def build_head(mesh, coords, parse_only=False, virtual=None, shard=None):
    fn, base, scale, norm_w = random_head()
    kw = {} if coords is None else {"chips": coords}
    if os.environ.get("HY4_FENDS_TIMING") == "1":
        kw["timing"] = True
    if parse_only:
        kw.update(parse_only=True, attn_kwargs=dict(parse_only=True))
    st = build_random(mesh, LAYERS, extra_sites=[EF.head_site(fn, base, scale)], extra_gammas=[norm_w], **kw)
    t0 = time.perf_counter()
    stream = EF.head_stream(mesh, shard if shard is not None else random_shard(0))
    ttnn.synchronize_device(mesh)
    seconds = time.perf_counter() - t0
    head = EF.EndsHead(st, 2 * LAYERS, stream, K=K, virtual=virtual)
    return st, head, (fn, base, scale, norm_w), seconds


@galaxy
@pytest.mark.parametrize("rung", list(RUNGS))
def test_head_mock(mesh_device, rung):
    """The HEAD program assembled and compiled for the rung's chips: audits, L1, the units' placement."""
    if not is_mock():
        pytest.skip("the mock's compile and size check")
    mesh = mesh_device
    st, head, _, _ = build_head(mesh, RUNGS[rung])
    ids, kept = EF.id_tensor(mesh), EF.logits_tensor(mesh)
    before = mesh.num_program_cache_entries()
    head.launch(ids, kept)
    head.launch(ids)
    by_bank = {}
    for (x, y), bank, j in head.units:
        by_bank.setdefault(bank, []).append((x, y))
    logger.info(f"HY4_MK11 fused ends HEAD [{rung}] mock: units by DRAM bank {by_bank}")
    coords = (
        [st.topo.root]
        + [c for c in st.topo.chips if st.topo.is_head(c) and c != st.topo.root][:1]
        + [c for c in st.topo.chips if not st.topo.is_head(c)][:1]
    )
    for coord in coords:
        rows, kernels, cbs, worst, core = head_audit(head, coord)
        logger.info(
            f"HY4_MK11 fused ends HEAD [{rung}] mock chip {coord}: {kernels} kernels, {cbs} CBs; {rows} (core, buffer) rows each with one "
            f"producer and one consumer; largest program-buffer set {worst} B on core {core}; with the L1 tensors "
            f"{worst + st.l1_bytes()} of {st.l1_report()['core']} B"
        )
    feed = head.feed_audit()
    two = next(r for r in feed if r[2] == 0)
    second = next(r for r in feed if r[2] == 0 and r[3] == 1)
    one = next(r for r in feed if r[2] == EF.UNITS_PER_BANK - 1)
    logger.info(
        f"HY4_MK11 fused ends HEAD [{rung}] mock feed audit: {len(feed)} (unit, column) rows = {len({r[4] for r in feed})} tile columns on "
        f"{len({r[0] for r in feed})} units; x held (pushed 192 once, never popped); a two-column unit (bank {two[1]}, index 0): column 0 = tile "
        f"column {two[4]} {two[5]}, column 1 = tile column {second[4]} {second[5]}; the one-column unit of a bank (index 7): tile column {one[4]} {one[5]}; "
        f"units with one column {sum(1 for r in feed if r[2] == EF.UNITS_PER_BANK - 1)}"
    )
    assert len(feed) == EF.COLS and sorted(r[4] for r in feed) == list(range(EF.COLS))
    logger.info(
        f"HY4_MK11 fused ends HEAD [{rung}] mock: K {head.K}, list {head.list_bytes} B; hub arena ranges: logits {EF.A_LOGITS}, slots "
        f"{head.a_slots}, landings {head.a_land1} / {head.a_land2}, scratch {head.a_scratch}, end {head.a_end} of {A.ARENA_BYTES}; "
        f"L1 tensors {st.l1_bytes()} B per core; program-cache entries {before} -> {mesh.num_program_cache_entries()}"
    )


@galaxy
def test_head_solo(mesh_device):
    """The root alone, no fabric connection: the site's pass with the head's entry and the final norm, the sender's
    own copy of x, the hub, the 64 units (head_compute.cpp), the merge, the ids."""
    mesh = mesh_device
    shard = random_shard(0)
    st, head, (fn, base, scale, norm_w), seconds = build_head(mesh, [(0, 0)], shard=shard)
    logger.info(f"HY4_MK11 fused ends HEAD solo: the head stream (47.2 MB a chip) made and written in {seconds:.1f} s")
    g = torch.Generator().manual_seed(3)
    streams = torch.randn(8, 4, C, generator=g)
    st.set_streams(streams)
    st.a.set_f(torch.zeros(8, C))  # (the leading post-mix adds post x 0)
    ids, kept = EF.id_tensor(mesh, [7] * 8), EF.logits_tensor(mesh)
    head.launch(ids, kept)
    if is_mock():
        return
    ttnn.synchronize_device(mesh)
    status = st.status()
    logger.info(f"HY4_MK11 fused ends HEAD solo: status {status or 'empty'}")
    assert not status, f"words not at rest: {status}"
    root = (0, 0)
    x = st.a.read_root()["x"]
    want_x = reference_x(streams, fn, base, scale, norm_w)
    hub = head.read_hub(root)
    logger.info(
        f"HY4_MK11 fused ends HEAD solo: x (head readout + final norm, BF16) against the reference pcc {pcc(want_x, x):.7f}, max abs "
        f"{float((x - want_x).abs().max()):.3e} (BF16 step at the largest value {float(want_x.abs().max()) * 2.0**-8:.3e}); the hub's x equal to "
        f"the site's {torch.equal(hub['x'], x)}"
    )
    assert pcc(want_x, x) > 0.99999 and torch.equal(hub["x"], x)
    logits = EF.read_logits(kept, [0])[0]
    want = (x.double() @ shard.double().T).float()
    err = (logits - want).abs()
    logger.info(
        f"HY4_MK11 fused ends HEAD solo: logits [8, {logits.shape[1]}] against float64 on the device's x and the stored BF16 head: pcc "
        f"{pcc(want, logits):.8f}, max abs {float(err.max()):.3e}, rms {float(err.pow(2).mean().sqrt()):.3e} (logits rms {float(want.pow(2).mean().sqrt()):.3f}); "
        f"the hub's logits range equal to the kept tensor {torch.equal(hub['logits'], logits)}"
    )
    by_round = column_rounds(head, logits, want)
    logger.info(f"HY4_MK11 fused ends HEAD solo: logits by the unit's round: {by_round}")
    assert pcc(want, logits) > 0.99999 and torch.equal(hub["logits"], logits)
    assert all(v[1] > 0.99999 for v in by_round.values()), "a round of the units is off"
    keys, best = EF.host_best(logits, head.K)
    got = EF.read_ids(ids)
    host_ids = [int(v) for v in want.argmax(-1)]
    lists = hub["lists"]
    logger.info(
        f"HY4_MK11 fused ends HEAD solo: ids {got}; the first maximum of the device's own logits {[int(v) for v in best[:, 0]]}; of the "
        f"host's logits {host_ids}; the hub's final list equal to the K best of the device's logits: keys "
        f"{torch.equal(lists[2][:, :, 0], keys)}, ids {torch.equal(lists[2][:, :, 1], best)}"
    )
    assert got == [int(v) for v in best[:, 0]], "the ids are not the argmax of the device's logits"
    assert torch.equal(lists[2][:, :, 0], keys) and torch.equal(lists[2][:, :, 1], best)
    t0 = time.perf_counter()
    for _ in range(20):
        head.launch(ids, kept)
    ttnn.synchronize_device(mesh)
    logger.info(
        f"HY4_MK11 fused ends HEAD solo: 20 launches back to back, untraced: {(time.perf_counter() - t0) / 20 * 1e6:.0f} us each from the host"
    )
    assert not st.status() and EF.read_ids(ids) == got
    log_stamps(st, head, "solo")


def log_stamps(st, head, what):
    """The hubs' stamps of the last run (Stack(timing=True)): us from the hub's start."""
    if not st.timing:
        return
    stamps = head.mt.stamps(lambda coord: head.hub)
    words = head.mt.read_words()
    for coord in sorted(stamps):
        s = stamps[coord] + [int(head.mt.core_words(words[coord], head.hub)[15])]
        d = [((v - s[0]) & 0xFFFFFFFF) / 1000.0 for v in s]
        mb = EF.COLS * EF.ITEM / 1e6
        units = d[3] - d[2]
        logger.info(
            f"HY4_MK11 fused ends HEAD {what} stamps chip {coord}: x arrived {d[1]:.1f} us, units released {d[2]:.1f}, every list in "
            f"{d[3]:.1f}, handed on {d[4]:.1f}; release to lists {units:.1f} us = {mb / units * 1e3:.0f} GB/s for the {mb:.1f} MB stream"
        )


VIRTUAL = {
    "vhead": dict(mode=EF.MODE_HEAD, n1=4, idx1=1, n2=8, idx2=3),
    "vroot": dict(mode=EF.MODE_ROOT, n1=4, idx1=0, n2=8, idx2=0),
}


def read_list(st, head, coord, offset):
    """(keys [8, K], ids [8, K]) of one list in the hub's arena at ``offset``."""
    import numpy as np

    raw = st.a.arena_bytes(st.a.read_arena(coord), head.hub, offset, head.list_bytes)
    pairs = torch.from_numpy(np.frombuffer(raw, dtype=np.uint32).astype(np.int64)).reshape(8, head.K, 2)
    return pairs[:, :, 0], pairs[:, :, 1]


@galaxy
@pytest.mark.parametrize("form", list(VIRTUAL))
def test_head_virtual(mesh_device, form):
    """ONE chip as a head (vhead) and as the root (vroot) of the whole mesh, its partners' lists and arrival counts
    written by the host: every chip-local path of the two reduce stages without the fabric. Two launches: partners
    that win some rows by larger keys; then partners that tie with the chip's own best key (a lower id must win, a
    higher id must lose)."""
    mesh = mesh_device
    v = VIRTUAL[form]
    shard = random_shard(0)
    st, head, _, _ = build_head(mesh, [(0, 0)], shard=shard, virtual=v)
    root, Kc = (0, 0), head.K
    g = torch.Generator().manual_seed(9)
    st.set_streams(torch.randn(8, 4, C, generator=g))
    st.a.set_f(torch.zeros(8, C))
    ids, kept = EF.id_tensor(mesh, [7] * 8), EF.logits_tensor(mesh)
    key = lambda value: EF.sort_key(torch.full((8, Kc), float(value)))

    def partners(n, skip, winners, tie=None):
        """n slots but ``skip``: slot i wins rows winners[i] with a key above every logit and loses the others;
        tie: (own keys, own ids): slot `first` repeats the chip's key on row 5 with a LOWER id and on row 6 with a
        higher one."""
        out = {}
        first = min(i for i in range(n) if i != skip)
        for i in range(n):
            if i == skip:
                continue
            keys, tok = key(-1000.0 - i), torch.full((8, Kc), 100_000 + 1000 * i) + torch.arange(Kc)
            for r in winners.get(i, []):
                keys[r] = EF.sort_key(torch.full((Kc,), 500.0 + i)) - torch.arange(Kc)
            if tie is not None and i == first:
                keys[5, 0], tok[5, 0] = tie[0][5, 0], max(int(tie[1][5, 0]) - 1, 0)
                keys[6, 0], tok[6, 0] = tie[0][6, 0], int(tie[1][6, 0]) + 1
            out[i] = (keys, tok)
        return out

    def run(tie=None):
        land1 = partners(v["n1"], v["idx1"], {(v["idx1"] + 1) % v["n1"]: [0], (v["idx1"] + 2) % v["n1"]: [1]}, tie)
        land2 = (
            partners(v["n2"], v["idx2"], {(v["idx2"] + 1) % v["n2"]: [2], (v["idx2"] + 5) % v["n2"]: [1, 3]})
            if v["mode"] == EF.MODE_ROOT
            else {}
        )
        head.write_partners(root, land1, land2)
        head.launch(ids, kept)
        return land1, land2

    land1, land2 = run()
    if is_mock():
        return
    ttnn.synchronize_device(mesh)

    def check(land1, land2, what):
        status = st.status()
        hub = head.read_hub(root)
        own = (hub["lists"][0][:, :, 0], hub["lists"][0][:, :, 1])
        logits = EF.read_logits(kept, [0])[0]
        keys, best = EF.host_best(logits, Kc)
        assert torch.equal(own[0], keys) and torch.equal(own[1], best), "the chip's own list"
        row = EF.host_merge([own] + [land1[i] for i in sorted(land1)], Kc)
        got_row = (hub["lists"][1][:, :, 0], hub["lists"][1][:, :, 1])
        ok_row = torch.equal(got_row[0], row[0]) and torch.equal(got_row[1], row[1])
        if v["mode"] == EF.MODE_HEAD:
            sent = read_list(st, head, root, head.a_land2 + v["idx2"] * head.list_bytes)
            ok_sent = torch.equal(sent[0], row[0]) and torch.equal(sent[1], row[1])
            logger.info(
                f"HY4_MK11 fused ends HEAD {form} {what}: row list = the host's merge of the chip's and 3 partners' {ok_row}; rows won by a "
                f"partner {[r for r in range(8) if int(row[1][r, 0]) >= 100_000 or int(row[1][r, 0]) != int(own[1][r, 0])]}; the list in the "
                f"chip's own stage-2 slot (through the sender) equal {ok_sent}; status {status}"
            )
            assert ok_row and ok_sent
            # (the one count the chip's own op left in its own stage-2 word: nobody takes it on a head)
            assert status and all(
                val == {EF.tp.W_S2: "0x1"} and k[-2:] == (root, head.hub) for k, val in status.items()
            ), f"expected only the own stage-2 count: {status}"
            st.reset()
        else:
            final = EF.host_merge([row] + [land2[i] for i in sorted(land2)], Kc)
            got = (hub["lists"][2][:, :, 0], hub["lists"][2][:, :, 1])
            ok = torch.equal(got[0], final[0]) and torch.equal(got[1], final[1])
            tokens = EF.read_ids(ids)
            logger.info(
                f"HY4_MK11 fused ends HEAD {form} {what}: row list = the host's merge {ok_row}; final list = the host's merge with 7 heads "
                f"{ok}; ids {tokens} = the final list's first {tokens == [int(x) for x in final[1][:, 0]]}; status {status or 'empty'}"
            )
            assert ok_row and ok and tokens == [int(x) for x in final[1][:, 0]] and not status
        return own

    own = check(land1, land2, "partners winning rows")
    land1, land2 = run(tie=own)
    ttnn.synchronize_device(mesh)
    check(land1, land2, "ties on rows 5 (lower id) and 6 (higher id)")
    hub = head.read_hub(root)
    row = hub["lists"][1]
    logger.info(
        f"HY4_MK11 fused ends HEAD {form}: tie rows: row 5 own id {int(own[1][5, 0])} -> {int(row[5, 0, 1])} (the lower id), row 6 own id "
        f"{int(own[1][6, 0])} -> {int(row[6, 0, 1])} (kept)"
    )
    assert int(row[6, 0, 1]) == int(own[1][6, 0]) and (
        int(own[1][5, 0]) == 0 or int(row[5, 0, 1]) == int(own[1][5, 0]) - 1
    )


@galaxy
def test_head_parse(mesh_device):
    """32 chips, every sender parse-only (it writes its argument table as it read it to a page and opens no fabric
    connection): each chip's page against the table the host built. Nothing travels, so every hub runs into its
    bounded waits (short here) and leaves a status; the units still compute."""
    mesh = mesh_device
    fn, base, scale, norm_w = random_head()
    st = build_random(
        mesh,
        LAYERS,
        extra_sites=[EF.head_site(fn, base, scale)],
        extra_gammas=[norm_w],
        parse_only=True,
        attn_kwargs=dict(parse_only=True),
        wait_limit=2_000_000,
    )
    head = EF.EndsHead(st, 2 * LAYERS, EF.head_stream(mesh, random_shard(0)), K=K)
    ids = EF.id_tensor(mesh)
    head.launch(ids)
    if is_mock():
        return
    ttnn.synchronize_device(mesh)
    wrong = []
    for coord in st.topo.chips:
        page, want = head.read_parsed(coord)
        if page != want:
            wrong.append((coord, [(i, hex(a), hex(b)) for i, (a, b) in enumerate(zip(page, want)) if a != b][:6]))
    status = st.status()
    logger.info(
        f"HY4_MK11 fused ends HEAD parse-only on {len(st.topo.chips)} chips: sender pages equal to the host's tables on "
        f"{len(st.topo.chips) - len(wrong)} chips; words left by the bounded waits on {len(status)} cores (expected: nothing travelled)"
    )
    assert not wrong, f"sender pages differ: {wrong[:3]}"
    # what the bounded waits must have left, and nothing else: on every hub the code of its LAST wait that ran out
    # (no x: 0xC1; a head then its row's lists: 0xD2; the root then the heads' lists: 0xD3), each with 0 seen; on
    # every sender core the ready bit of the one op that was handed over to a sender that had already returned
    expected = {}
    for coord in st.topo.chips:
        code = 0xD3 if coord == st.topo.root else (0xD2 if st.topo.is_head(coord) else 0xC1)
        expected[(coord, head.hub)] = {EF.tp.W_STATUS: hex(code << 24)}
        expected[(coord, tuple(head.sender_core))] = {EF.tp.W_READY: hex(EF.Sender.bit(0))}
    got = {k[-2:]: val for k, val in status.items()}
    extra = {k: val for k, val in got.items() if expected.get(k) != val}
    missing = {k: val for k, val in expected.items() if k not in got}
    logger.info(
        f"HY4_MK11 fused ends HEAD parse-only: statuses as expected on {len(expected) - len(missing)} of {len(expected)} cores; others {extra}; missing {missing}"
    )
    assert not extra and not missing
    st.reset()
    head.reset()
    ttnn.synchronize_device(mesh)
    assert not st.status()


MESH_RUNGS = {
    "rowpair": [(0, 0), (0, 1)],
    "colpair": [(0, 0), (1, 0)],
    "square": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "all": None,
}


@galaxy
@pytest.mark.parametrize("rung", list(MESH_RUNGS))
def test_head_mesh(mesh_device, rung):
    """The HEAD over the fabric on a part of the mesh and on all of it: every chip its own shard of a random head;
    x on the chips, every chip's logits, and the ids against the first maximum over all chips' own logits."""
    mesh = mesh_device
    coords = MESH_RUNGS[rung]
    st, head, (fn, base, scale, norm_w), seconds = build_head(mesh, coords, shard=random_shard)
    chips = [st.topo.chip(c) for c in st.topo.chips]
    logger.info(
        f"HY4_MK11 fused ends HEAD [{rung}]: {len(chips)} chips; 32 head streams made and written in {seconds:.1f} s"
    )
    g = torch.Generator().manual_seed(3)
    streams = torch.randn(8, 4, C, generator=g)
    st.set_streams(streams)
    st.a.set_f(torch.zeros(8, C))
    ids, kept = EF.id_tensor(mesh, [7] * 8), EF.logits_tensor(mesh)
    head.launch(ids, kept)
    if is_mock():
        return
    ttnn.synchronize_device(mesh)
    status = st.status()
    logger.info(f"HY4_MK11 fused ends HEAD [{rung}]: status {status or 'empty'}")
    assert not status, f"words not at rest: {status}"

    def check(what):
        x = st.a.read_root()["x"]
        want_x = reference_x(streams, fn, base, scale, norm_w)
        probes = [st.topo.chips[0], st.topo.chips[-1]] + [c for c in st.topo.chips if st.topo.is_head(c)][1:2]
        same = [torch.equal(head.read_hub(c)["x"], x) for c in probes]
        logits = EF.read_logits(kept, chips)
        worst, rounds = (1.0, 0.0, None), {}
        for n, chip in enumerate(chips):
            want = (x.double() @ random_shard(chip).double().T).float()
            p, e = pcc(want, logits[n]), float((logits[n] - want).abs().max())
            if p < worst[0]:
                worst = (p, e, chip)
            for name, (count, pc, err) in column_rounds(head, logits[n], want).items():
                old = rounds.get(name, (1.0, 0.0))
                rounds[name] = (min(old[0], pc), max(old[1], err))
        whole = torch.cat([logits[n] for n in range(len(chips))], dim=-1)  # the chips' shards in the order of their ids
        offsets = torch.tensor([chip * EF.VOCAB_PER_CHIP for chip in chips]).repeat_interleave(EF.VOCAB_PER_CHIP)
        keys, best = EF.host_best(whole, head.K)
        want_ids = [int(offsets[int(b)] + int(b) % EF.VOCAB_PER_CHIP) for b in best[:, 0]]
        got = EF.read_ids(ids)
        logger.info(
            f"HY4_MK11 fused ends HEAD [{rung}] {what}: x against the reference pcc {pcc(want_x, x):.7f}; the hub's x equal to the root's on "
            f"chips {probes}: {same}; logits of {len(chips)} chips against float64 on x and their own shard: worst pcc {worst[0]:.8f} "
            f"(max abs {worst[1]:.3e}, chip {worst[2]}); by the unit's round (min pcc, max abs) {rounds}; ids {got} = the first maximum "
            f"over the chips' own logits {got == want_ids} (owner chips {[i // EF.VOCAB_PER_CHIP for i in got]})"
        )
        assert all(same) and worst[0] > 0.99999 and got == want_ids

    check("first launch")
    t0 = time.perf_counter()
    for _ in range(20):
        head.launch(ids, kept)
    ttnn.synchronize_device(mesh)
    logger.info(
        f"HY4_MK11 fused ends HEAD [{rung}]: 20 launches back to back, untraced: {(time.perf_counter() - t0) / 20 * 1e6:.0f} us each from the host"
    )
    assert not st.status()
    check("after 20 more launches")
    canary = head.canary_summary()
    per = {EF.MODE_ROOT: 5, EF.MODE_HEAD: 3, EF.MODE_CHIP: 1}
    expect = 21 * sum(per[head._roles(c)[0]] for c in st.topo.chips)
    logger.info(
        f"HY4_MK11 fused ends HEAD [{rung}] canary on the landings after 21 launches: {canary} (checks expected {expect})"
    )
    assert canary["failed"] == 0 and canary["checks"] == expect
    log_stamps(st, head, rung)


# ------------------------------------------------------------------------------------------------ the whole step
@galaxy
def test_step(mesh_device):
    """Two decode steps as ONE trace with the fused ends: ids (DRAM) -> IN -> P1(post) .. MLP of every layer ->
    HEAD -> ids -> (step 2) IN -> ... Checkpoint weights, the golden's cache rows (tests/test_chain.py Chain), each
    step's selection table and RoPE tensor written before the trace. Per step: every status, the canary, the tokens
    and logits against the golden's and against the reference tail on the step's OWN streams (kept by one copy op
    per step in the checking trace), per chip's shard of the vocabulary (ends.off_shards: the acceptance test of
    PLAN.md section 44; HY4_ENDS_REPEATS replays). Then the latency of a trace without the kept copies against the
    layers alone.
        HY4_ROPE_PAIRING=interleaved HY4_GOLDEN_DIR=generated/hy4_preview/golden_interleaved HY4_LATENT_SCALE=1.0430 \\
        TT_METAL_OPERATION_TIMEOUT_SECONDS=30 pytest <this file> -x -k test_step"""
    from models.experimental.hy4_preview.fused.common import tracing
    from models.experimental.hy4_preview.fused.model import ends
    from models.experimental.hy4_preview.fused.model.tests import test_chain as TC
    from models.experimental.hy4_preview.tests.common import hy4_tensors

    mesh = mesh_device
    chain = TC.Chain(mesh, wait_limit=30_000_000, head_site=True)
    st = chain.st
    try:
        t0 = time.perf_counter()
        table, head_w = hy4_tensors(["model.embed_tokens.weight", "lm_head.weight"])
        ein = EF.EndsIn(st, table)
        ttnn.synchronize_device(mesh)
        t1 = time.perf_counter()
        stream = EF.head_stream(mesh, lambda c: head_w[c * EF.VOCAB_PER_CHIP : (c + 1) * EF.VOCAB_PER_CHIP])
        head = EF.EndsHead(st, 2 * chain.n, stream, K=K)
        ttnn.synchronize_device(mesh)
        logger.info(
            f"HY4_MK11 fused ends step: the embedding table to the device in {t1 - t0:.0f} s, the head's 32 streams in "
            f"{time.perf_counter() - t1:.0f} s; L1 tensors {st.l1_bytes()} B per core; program-cache entries {mesh.num_program_cache_entries()}"
        )
        STEPS = int(
            os.environ.get("HY4_FENDS_STEPS", "2")
        )  # (1: a golden of 6 steps has no step behind its last row's)
        chain.set_batch(0, steps=STEPS)
        J = chain.J
        tokens = chain.tokens()
        T = [EF.id_tensor(mesh) for _ in range(STEPS + 1)]
        ein.write_ids(T[0], tokens)
        kept = [EF.logits_tensor(mesh) for _ in range(STEPS)]
        layer_programs = {}
        unions = []

        def launch_reading_unions(programs):
            """The layer programs one by one; after every MoE layer's program the rows' picked experts are read from
            the root's hub (the router is the same on every chip): the union of the 8 rows' 8 picks."""
            root_chip = st.topo.chip(st.topo.root)
            for n, state in enumerate(st.layers):
                Stack.launch(programs[3 * n : 3 * n + 3])
                if not state.mlp.dense:
                    ttnn.synchronize_device(mesh)
                    picks = (
                        ttnn.to_torch(ttnn.get_device_tensors(state.mlp.op.picks)[root_chip]).to(torch.int32) & 0xFFFF
                    )
                    unions.append(len(set(picks[:, :8].flatten().tolist())))

        def step(k, keep, read_unions=False):
            ein.launch(T[k])
            chain.use_step(k)
            layer_programs[k] = st.step_programs(first_post=True, last_post=False)
            if read_unions:
                launch_reading_unions(layer_programs[k])
            else:
                Stack.launch(layer_programs[k])
            head.launch(T[k + 1], kept[k] if keep else None)
            return ttnn.to_memory_config(st.a.streams, ttnn.DRAM_MEMORY_CONFIG) if keep else None

        label = f"fused ends in the trace, {chain.n} layers {chain.label} [{TC.MOE_FIDELITY}]"
        gold_unions = chain.gold_unions()
        logger.info(
            f"HY4_MK11 {label}: rows (prompt length, golden step, position, cached rows read on layer 0) "
            f"{[(g['meta']['prompt_len'], j, p, min(p + 1, 2048)) for (g, j, _), p in zip(chain.rows, chain.positions)]}; the union of the rows' "
            f"experts per MoE layer by the GOLDENS' own picks (the reference router, host): mean {sum(gold_unions) / max(len(gold_unions), 1):.2f}, "
            f"min {min(gold_unions, default=0)}, max {max(gold_unions, default=0)}"
        )
        t0 = time.perf_counter()
        results = [step(k, True, read_unions=k == 0 and not is_mock()) for k in range(STEPS)]
        if unions:
            logger.info(
                f"HY4_MK11 {label} EXPERT UNION of the 8 rows per MoE layer in step 1 (the device's own picks, untraced): mean "
                f"{sum(unions) / len(unions):.2f}, min {min(unions)}, max {max(unions)}; per layer {unions}"
            )
        if is_mock():
            logger.info(
                f"HY4_MK11 {label}: compiled on the mock; program-cache entries {mesh.num_program_cache_entries()}; L1 tensors {st.l1_bytes()} B per core"
            )
            return
        ttnn.synchronize_device(mesh)
        untraced_s = time.perf_counter() - t0
        status = TC.full_status(chain)
        assert not status, f"words not at rest after the untraced steps: {status}"
        ref_head, lm = TC.head_reference()
        root = st.topo.chip(st.topo.root)

        def read(k, copies):
            streams = ihc_layout.pages_to_streams(ttnn.to_torch(ttnn.get_device_tensors(copies[k])[root]).float()[0, 0])
            dev = EF.read_logits(kept[k]).permute(1, 0, 2).reshape(8, -1)  # the chips' shards in the order of their ids
            return streams, dev, torch.tensor(EF.read_ids(T[k + 1]))

        def check(copies, what):
            out = []
            for k in range(STEPS):
                streams, dev_logits, got = read(k, copies)
                gl, want = chain.gold_top("logits", k).float(), chain.gold_top("cached_logits_argmax", k)
                gold = chain.gold(chain.n - 1, "streams_out", k)
                _, ref_logits = TC.host_logits(ref_head, lm, streams)
                off = ends.off_shards(dev_logits, ref_logits)
                if off:
                    logger.warning(
                        f"HY4_MK11 {label} {what} step {k + 1}: ENDS OFF: chips' shards that leave the reference tail on the step's own streams {off}"
                    )
                fed = (
                    "HF's tokens"
                    if k == 0
                    else f"the device's tokens of step 1 ({int((out[0] == chain.gold_top('cached_logits_argmax')).sum())} of 8 are HF's)"
                )
                logger.info(
                    f"HY4_MK11 {label} {what} step {k + 1} (fed {fed}): streams after the last layer vs the golden pcc={pcc(gold, streams):.6f}; "
                    f"device logits vs the golden pcc={pcc(gl, dev_logits):.6f}, vs the reference tail on the same streams {pcc(ref_logits, dev_logits):.6f} "
                    f"(max abs {float((ref_logits - dev_logits).abs().max()):.3f}); greedy token = the golden's on {int((got == want).sum())} of 8 rows, = "
                    f"the reference tail's on {int((got == ref_logits.argmax(-1)).sum())} of 8, = the first maximum of the device's own logits on "
                    f"{int((got == dev_logits.argmax(-1)).sum())} of 8; shards off {off}"
                )
                fed_ok = [True] * 8 if k == 0 else [bool(v) for v in (out[0] == chain.gold_top("cached_logits_argmax"))]
                margin = gl.topk(2, dim=-1).values
                logger.info(
                    f"HY4_MK11 {label} {what} step {k + 1} per row: fed the golden's token {fed_ok}; streams pcc "
                    f"{[round(pcc(gold[u], streams[u]), 5) for u in range(8)]}; logits pcc vs the golden {[round(pcc(gl[u], dev_logits[u]), 5) for u in range(8)]}; "
                    f"token = the golden's {[bool(v) for v in (got == want)]}; the golden's own margin (logits) {[round(float(m[0] - m[1]), 3) for m in margin]}; "
                    f"the host tail on the device's streams gives the golden's token on {int((ref_logits.argmax(-1) == want).sum())} of 8"
                )
                assert bool(torch.isfinite(streams).all()) and not off
                assert (
                    int((got == dev_logits.argmax(-1)).sum()) == 8
                ), "the ids are not the argmax of the device's logits"
                out.append(got)
            return out

        check(results, "untraced")
        for t in results:
            ttnn.deallocate(t)
        trace_id, results = tracing.capture(mesh, lambda: [step(k, True) for k in range(STEPS)])
        ttnn.synchronize_device(mesh)
        try:
            view = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
            logger.info(
                f"HY4_MK11 {label}: trace buffer {view.total_bytes_allocated_per_bank} B per bank; untraced the two steps took {1e3 * untraced_s:.0f} ms"
            )
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh)
            status = TC.full_status(chain)
            assert not status, f"a status is left after the traced steps: {status}"
            check(results, "ONE trace")
            traced_rows = read(0, results)  # (streams, logits, ids of step 1 in the traced replay)
            canary = st.canary_summary()
            logger.info(
                f"HY4_MK11 {label} canary: {canary}; the HEAD's canary on its landings: {head.canary_summary()}"
            )
            assert head.canary_summary()["failed"] == 0
            for kind, v in canary.items():
                assert v["found"] == 0 and v["timeouts"] == 0 and v["checks"] == v["runs"], f"canary {kind}: {v}"
            repeats = int(os.environ.get("HY4_ENDS_REPEATS", "0"))
            if repeats:
                off, worst, wrong_ids, low = [], 1.0, 0, 1.0
                first, differ, distinct = {}, {k: [0, 0, 0] for k in range(STEPS)}, {k: set() for k in range(STEPS)}
                t0 = time.perf_counter()
                for i in range(repeats):
                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    for k in range(STEPS):
                        streams, dev, got = read(k, results)
                        # replay against replay: with AttnLayer(fixed_order=True) the same trace must give the same
                        # bits (streams after the last layer, logits, ids)
                        if k not in first:
                            first[k] = (streams, dev, got)
                        for n, (now, then) in enumerate(zip((streams, dev, got), first[k])):
                            differ[k][n] += int(not torch.equal(now, then))
                        distinct[k].add(hash(streams.numpy().tobytes()))
                        _, ref = TC.host_logits(ref_head, lm, streams)
                        shard = dev.shape[1] // 32
                        a, b = dev.reshape(8, 32, shard).double(), ref.reshape(8, 32, shard).double()
                        a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
                        per = (
                            ((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp(min=1e-30)).min(0).values
                        )  # per chip
                        worst = min(worst, float(per.min()))
                        low = min(low, pcc(ref, dev))
                        wrong_ids += int((got != dev.argmax(-1)).sum())
                        if float(per.min()) < 0.999 or bool((got != dev.argmax(-1)).any()):
                            off.append(
                                (i, k, [c for c in range(32) if float(per[c]) < 0.999], round(float(per.min()), 4))
                            )
                            if len(off) <= 8:  # the event's tensors, for the per-shard table
                                torch.save(
                                    {
                                        "replay": i,
                                        "step": k,
                                        "streams": streams,
                                        "device_logits": dev,
                                        "tail_logits": ref,
                                        "ids": got,
                                        "per_chip_min_pcc": per,
                                    },
                                    f"generated/hy4_preview/logs/mk11_fends_off_{i}_{k}.pt",
                                )
                status = TC.full_status(chain)
                logger.info(
                    f"HY4_MK11 {label} ENDS over {repeats} single replays of the {STEPS}-step trace ({STEPS * repeats} steps, {time.perf_counter() - t0:.0f} s): "
                    f"steps whose device logits leave the reference tail on their own streams in some chip's shard of the vocabulary (below 0.999): "
                    f"{len(off)}: (replay, step, chips, the shard's min pcc) {off[:20]}; the lowest shard pcc of all steps {worst:.6f}; the lowest "
                    f"logits pcc {low:.7f}; rows whose id is not the first maximum of the step's own device logits {wrong_ids}; status {status}"
                )
                logger.info(
                    f"HY4_MK11 {label} REPLAY AGAINST REPLAY over {repeats} replays (fixed_order {st.a.fixed_order}): replays whose (streams after the "
                    f"last layer, logits, ids) differ in any bit from the first replay's, per step {[differ[k] for k in range(STEPS)]}; distinct streams "
                    f"per step {[len(distinct[k]) for k in range(STEPS)]}"
                )
                hc = head.canary_summary()
                logger.info(
                    f"HY4_MK11 {label} the HEAD's canary on its landings after the replays: {hc}; the layers' canary {st.canary_summary()}"
                )
                assert not status and not off and wrong_ids == 0 and hc["failed"] == 0
        finally:
            ttnn.release_trace(mesh, trace_id)
        for t in results:
            ttnn.deallocate(t)
        if os.environ.get("HY4_CHAIN_TRACE_JOB", "A") == "A":
            logger.info(f"HY4_MK11 {label} job A done: two steps with the fused ends as one trace, every status empty")
            return
        # latency: the same two steps without the kept copies; then their layers alone
        trace_id, _ = tracing.capture(mesh, lambda: [step(k, False) for k in range(STEPS)])
        try:
            rounds = [tracing.replay_ms(mesh, trace_id, 20) / STEPS for _ in range(3)]
            ttnn.synchronize_device(mesh)
            status = TC.full_status(chain)
            assert not status, f"a status is left after the replays: {status}"
        finally:
            ttnn.release_trace(mesh, trace_id)

        # the same two steps without their HEADs: IN still writes each step's embedding (the ids of step 2 are the
        # ones the last full replay left in T[1]), so the layers run on the data of the full trace (a MoE layer's
        # time depends on the union of the rows' experts: the layers "alone" on whatever streams the last trace left
        # are another workload)
        def no_head(k):
            ein.launch(T[k])
            Stack.launch(layer_programs[k])

        r = tracing.traced_ms(mesh, lambda: [no_head(k) for k in range(STEPS)])
        no_head_ms = r["replay_ms"] / STEPS
        status = st.status()
        assert not status, f"words not at rest after the trace without HEAD: {status}"
        both = [p for k in range(STEPS) for p in layer_programs[k]]
        r = tracing.traced_ms(mesh, lambda: Stack.launch(both))
        layers_ms = r["replay_ms"] / STEPS
        best = min(rounds)
        logger.info(
            f"HY4_MK11 {label} LATENCY per step on the step's own data (golden {chain.label}, {STEPS} step(s) per trace, fixed_order {st.a.fixed_order}, "
            f"mean expert union of step 1 {sum(unions) / max(len(unions), 1):.2f}): IN + layers + HEAD {best:.3f} ms; IN + layers (no HEAD) {no_head_ms:.3f} ms: "
            f"HEAD costs {1e3 * (best - no_head_ms):.0f} us inside the step"
        )
        logger.info(
            f"HY4_MK11 {label} LATENCY per step, 8 rows, {STEPS} step(s) per trace, 20 replays back to back: WITH the fused ends {best:.3f} ms "
            f"(rounds {[round(v, 3) for v in rounds]}) = {8e3 / best:.1f} tokens/s; the layers alone on the streams the last trace left (another workload: no embedding between steps) "
            f"{layers_ms:.3f} ms = {8e3 / layers_ms:.1f} tokens/s; the ends {1e3 * (best - layers_ms):.0f} us per step"
        )
        status = st.status()
        assert not status, f"words not at rest after the layers-only trace: {status}"
        # the two programs as traces of their own (a trace of one program ends when it has ended on every chip, so
        # their sum over-states what they add inside the step, as PLAN.md section 16 says of every program)
        r_in = tracing.traced_ms(mesh, lambda: ein.launch(T[0]))
        r_head = tracing.traced_ms(mesh, lambda: head.launch(T[1], None))
        logger.info(
            f"HY4_MK11 {label} the ends' programs as traces of their own, 20 replays back to back: IN {1e3 * r_in['replay_ms']:.1f} us, "
            f"HEAD {1e3 * r_head['replay_ms']:.1f} us (its site pass includes the post-mix POST did: 13.7 us as a program of its own)"
        )
        status = st.status()
        assert not status, f"words not at rest after the ends' own traces: {status}"
        assert head.canary_summary()["failed"] == 0
        mix2 = os.environ.get("HY4_CHAIN_MIX2")
        if mix2:
            # DOES A ROW'S RESULT DEPEND ON THE OTHER ROWS OF ITS PASS? A second mixed step in the same process: the
            # rows named the same at the same index in both mixes have the same slot, cache rows, selection and token;
            # the other rows are other conversations (another union of experts in every MoE layer). Each kept row's
            # streams after the last layer, logits and id against the first mix's traced replay, bit for bit.
            assert chain.mix and STEPS == 1, "HY4_CHAIN_MIX2: a mixed step, one step per trace"
            first_mix, first_unions = list(chain.mix), list(unions)
            chain.mix = TC.parse_mix(mix2)
            assert {P for P, _ in chain.mix} <= set(chain.goldens), "the second mix: the first mix's goldens"
            chain.label = "MIX " + ",".join(f"{P}:{j}" for P, j in chain.mix)
            chain.set_batch(0, steps=1)
            ein.write_ids(T[0], chain.tokens())
            del unions[:]
            copy = step(0, True, read_unions=True)
            ttnn.synchronize_device(mesh)
            status = TC.full_status(chain)
            assert not status, f"words not at rest after the second mix's step: {status}"
            streams2, logits2, ids2 = read(0, [copy])
            ttnn.deallocate(copy)
            kept_rows = [u for u in range(8) if chain.mix[u] == first_mix[u]]
            bits = [
                (
                    u,
                    bool(torch.equal(streams2[u], traced_rows[0][u])),
                    bool(torch.equal(logits2[u], traced_rows[1][u])),
                    int(ids2[u]) == int(traced_rows[2][u]),
                )
                for u in kept_rows
            ]
            size = [
                (
                    u,
                    round(pcc(traced_rows[0][u], streams2[u]), 7),
                    round(pcc(traced_rows[1][u], logits2[u]), 7),
                    float((traced_rows[1][u] - logits2[u]).abs().max()),
                )
                for u in kept_rows
            ]
            want2 = chain.gold_top("cached_logits_argmax")
            logger.info(
                f"HY4_MK11 {label} OTHER ROWS CHANGED: second mix {chain.label} (mean expert union {sum(unions) / max(len(unions), 1):.2f}, the first mix "
                f"{sum(first_unions) / max(len(first_unions), 1):.2f}; per layer, layers whose union differs {sum(a != b for a, b in zip(unions, first_unions))} of {len(unions)}); "
                f"rows kept {kept_rows}: (row, streams after the last layer equal in every bit, logits equal, id equal) {bits}; (row, streams pcc, logits pcc, "
                f"logits max abs difference) {size}; the second mix's tokens = their goldens' on {int((ids2 == want2).sum())} of 8"
            )
    finally:
        chain.free()
