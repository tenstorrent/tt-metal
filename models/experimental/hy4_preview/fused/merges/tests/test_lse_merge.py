# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The cross-chip log-sum-exp merge (fused/merges/lse_merge.py) on the mesh the model opens.

Mock Galaxy (HY4_MOCK=1): every case compiles its programs and checks CB / L1 / kernel sizes; nothing runs.
Board, smallest first, each new kernel build alone with -x and a short operation timeout:

    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k "test_lse and solo"
    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k "test_lse and pairv"
    TT_METAL_OPERATION_TIMEOUT_SECONDS=25 pytest <this file> -x -k "test_lse and pairh"
    TT_METAL_OPERATION_TIMEOUT_SECONDS=25 pytest <this file> -x -k "test_lse and (quad or row0 or col0)"
    TT_METAL_OPERATION_TIMEOUT_SECONDS=25 pytest <this file> -x -k "test_lse and galaxy"
    pytest <this file> -x -k test_latency

A case gives every chip synthetic partials (O, max, sum) for 8 users and 64 heads: maxima spread over +-60 so
that the weights span many orders of magnitude, (chip, user) pairs without rows (the empty partial), one user with
no rows anywhere; the device's ctx of every chip's own heads is compared with the float64 merge of the same
partials and the sink (lse_merge.merge_golden = sparse_mla layout.merge = the reference's softmax). Result lines:
HY4_MK6.

Wait graph of the program, per chip: column sender S (BRISC), 8 merge cores M0..M7 (NCRISC + compute), row sender
T (the collective's sender, BRISC); the ctx core runs nothing. Fabric link 0 only. Waits on another CHIP are marked X.

    S    DRAM reads of the 56 partials that leave,     DRAM / M0's L1: passive memory, always answers. Completed
         16-byte read of M0's round word (barrier)     BEFORE the first fabric call.
    S    fabric open (start, then finish)              this chip's routers on link 0, the calls and their order
                                                       are collective_sender.cpp's. The fabric's own handshake:
                                                       not bounded by a kernel (as in the collective).
    S    per packet: writes flushed, a free slot in    the router's worker channel, which drains into the
         the router's worker channel                   neighbour's router and ends as an L1 write + increment on
                                                       the destination's merge core: a landing never pushes back
    S    fabric close, full barrier                    this chip's routers
    M_u  DRAM read of its local partial                DRAM
 X  M_u  column counter >= members - 1                 one fused write + increment from S of every OTHER chip of
                                                       the column, which S sends unconditionally (S waits for
                                                       nothing a merge core or another chip does). BOUNDED
                                                       (HY4_COLLECTIVE_WAIT_LIMIT); status bit 0.
    M_u  (compute) pages of the column landing         M_u's NCRISC, after the counter OR its bound
    M_u  cb_send from its compute kernel               the last column merge, which needs only that landing.
                                                       BOUNDED; status bit 2.
    T    ready bits of 8 ops                           M_u's hand-over after cb_send (or its bound). BOUNDED
                                                       (the collective's READY timeout, status in the site's ctrl).
 X  M_u  row counter >= row members                    T's own-slot count and one fused write + increment per
                                                       other chip of the row, sent once that chip's M_u handed
                                                       over: needs that chip's column stage (or its bounds),
                                                       which needs only column senders. BOUNDED; status bit 1.
    M_u  (compute) pages of the row landing, the sink  M_u's NCRISC, after the counter OR its bound
    M_u  cb_final from its compute kernel              the last merge. BOUNDED; status bit 3 (no ctx is written).
No cycle: column sends wait for nothing; row sends wait for column sends only; every wait on another chip is one of
the two counters, both bounded, both with a status bit in the merge core's state tile (word 0; 0x100 = the kernel
ended). Not bounded: the fabric's own handshakes on S and T, and the LLK calls inside the merge kernel (hence the
"solo" rung: the merge kernel and S's fabric-free build on one chip first).

STATE (2026-11-19): the first board run ever, "pairv" with the senders on fabric link 1, hit the operation timeout
(log mk6_lse_pairv) and the board had to be reset: link 1 had only been read from the mock. Since then: link 0 only,
S reads everything before its first fabric call and then follows collective_sender.cpp, the merge core's waits on
the merge kernel are bounded, and the order of first runs is "solo" (one chip, no fabric), then "pairv" alone with
-x and a 15 s operation timeout.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.experimental.hy4_preview.fused.collective import collective as cl
from models.experimental.hy4_preview.fused.common import tracing
from models.experimental.hy4_preview.fused.merges import lse_merge as lm
from models.experimental.hy4_preview.tests.common import pcc
from models.experimental.hy4_preview.tt import prefill_bridge as pb

GALAXY_MESH = (8, 4)
RESULTS = Path(os.environ.get("HY4_MK6_LSE_RESULTS", "generated/hy4_preview/logs/mk6_lse_latency.json"))
CLOCK_HZ = 1.343e9
MOCK = cl.is_mock()

galaxy = pytest.mark.parametrize(
    "device_params, mesh_device",
    [(dict(pb.device_params(), trace_region_size=32 << 20), GALAXY_MESH)],
    indirect=True,
    ids=["8x4-torus"],
)

# name -> (mesh rows, mesh columns) of the merged rectangle
RECTANGLES = {
    "solo": ([0], [0]),  # one chip: its own partial and the sink, no fabric (the merge kernels alone)
    "pairv": ([0, 1], [0]),
    "pairh": ([0], [0, 1]),
    "quad": ([0, 1], [0, 1]),
    "row0": ([0], [0, 1, 2, 3]),
    "col0": (list(range(8)), [0]),
    "galaxy": (None, None),
}


DRY_GROUPS = (1, 5)


def staged_checksum(inputs, users):
    """The sum of the uint32 words the column sender stages for DRY_GROUPS on the first chip: per group and user
    the max / sum tile, then the 16 O tiles (kernels/lse_sender.cpp)."""
    pages = lm.partial_pages(inputs[0][0], inputs[1][0], inputs[2][0])  # [users * 8, PAGE_VALUES]
    bits = pages.to(torch.bfloat16).view(torch.int16).to(torch.int64) & 0xFFFF
    total = 0
    for group in DRY_GROUPS:
        for u in range(users):
            page = bits[u * 8 + group]
            staged = torch.cat([page[16 * 256 : 17 * 256], page[: 16 * 256]])
            total += int((staged[0::2] | (staged[1::2] << 16)).sum())
    return total & 0xFFFFFFFF


def synthetic(chips, users, seed):
    """Partials as BF16 values: O [chips, users, 64, 512], max / sum [chips, users, 64], sink [64]."""
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(chips, users, 64, generator=g) * 30  # unscaled logits: weights exp((m - max) / 16)
    s = 1 + 60 * torch.rand(chips, users, 64, generator=g)
    o = torch.randn(chips, users, 64, 512, generator=g) * s[..., None] * torch.logspace(-2, 0.5, 512)
    empty = torch.rand(chips, users, generator=g) < 0.2
    empty[:, users - 1] = True  # the last user has no rows on any chip: ctx must be 0
    m = torch.where(empty[..., None], torch.tensor(lm.EMPTY_MAX), m)
    s = torch.where(empty[..., None], torch.tensor(0.0), s)
    o = torch.where(empty[..., None, None], torch.tensor(0.0), o)
    bf16 = lambda x: x.to(torch.bfloat16).float()
    return bf16(o), bf16(m), bf16(s), torch.randn(64, generator=g) * 2


def compare(name, merge, inputs, tag, floor):
    o, m, s, sink = inputs
    got = merge.read()
    gold = merge.golden(o, m, s, sink)
    assert bool(
        (got["status"] == 0x100).all()
    ), f"{name} {tag}: status {[hex(int(x)) for x in got['status'].flatten()]}"
    for site in merge.sites:
        site.check_idle()
    ctx = got["ctx"].double()
    worst_pcc, worst_abs, worst_rel = 1.0, 0.0, 0.0
    for k in range(len(merge.coords)):
        for u in range(merge.users):
            ref = gold[k, u]
            if float(ref.abs().max()) == 0.0:
                assert float(ctx[k, u].abs().max()) == 0.0, f"{name} {tag}: chip {k} user {u} has no rows, ctx is not 0"
                continue
            worst_pcc = min(worst_pcc, pcc(ctx[k, u], ref))
            err = (ctx[k, u] - ref).abs().max().item()
            worst_abs = max(worst_abs, err)
            worst_rel = max(worst_rel, err / ref.abs().max().item())
    logger.info(
        f"HY4_MK6 lse {name} {tag}: {len(merge.coords)} chips x {merge.users} users x {merge.heads_out} heads: worst "
        f"ctx pcc {worst_pcc:.7f}, max abs error {worst_abs:.3e} ({worst_rel:.2e} of the user's largest value)"
    )
    assert worst_pcc >= floor, f"{name} {tag}: ctx pcc {worst_pcc:.6f} < {floor}"
    return worst_pcc, worst_abs, worst_rel


def run_case(mesh_device, name, fidelity="hifi4", seeds=(0, 1, 2), floor=0.9999):
    rows, cols = RECTANGLES[name]
    # "solo" also runs the column sender without its fabric calls, as if it had head groups 1 and 5 to send
    merge = lm.LseMerge(
        mesh_device, rows, cols, fidelity=fidelity, dry_sender_groups=DRY_GROUPS if name == "solo" else ()
    )
    moved = merge.bytes_moved()
    logger.info(
        f"HY4_MK6 lse {name}: {merge.R} x {merge.C} chips, {merge.heads_out} heads out per chip; per chip: column "
        f"stage {moved['column_sent']} B out / in, row stage {moved['row_injected']} B injected, "
        f"{moved['row_received']} B in, {moved['packets']} packets; L1 per core {merge.l1_bytes()} B"
    )
    if MOCK:
        merge.compile()
        merge.compile(nop=True)
        logger.info(f"HY4_MK6 lse {name}: compiled on the mock mesh")
        return
    for seed in seeds:
        inputs = synthetic(len(merge.coords), merge.users, seed)
        merge.set_partials(*inputs[:3])
        merge.set_sink(inputs[3])
        merge()
        compare(name, merge, inputs, f"{fidelity} seed {seed}", floor)
        if name == "solo":
            report = merge.read()["dry"][0].tolist()
            want = staged_checksum(inputs, merge.users)
            logger.info(
                f"HY4_MK6 lse solo seed {seed}: dry column sender report {[hex(x) for x in report]}, checksum wanted {want:#x}"
            )
            assert (
                report[0] == 0x4C534452 and report[3] == want
            ), "the dry column sender did not stage the partials it would send"


@galaxy
@pytest.mark.parametrize("name", list(RECTANGLES))
def test_lse(mesh_device, name):
    run_case(mesh_device, name)


@galaxy
@pytest.mark.parametrize("fidelity", ["lofi", "hifi2"])
def test_lse_fidelity(mesh_device, fidelity):
    """The whole mesh at lower FPU fidelity of the O x weight multiplies: accuracy is logged, the floor is loose."""
    run_case(mesh_device, "galaxy", fidelity=fidelity, seeds=(0,), floor=0.99)


def phase_table(got):
    s = got["stamps"].reshape(-1, 8)
    names = [
        ("local partial read", 1, 0),
        ("column landing complete (from kernel start)", 2, 0),
        ("column merges", 3, 2),
        ("row hand-over", 4, 3),
        ("row stage on the fabric", 5, 4),
        ("row merges + sink + divide", 6, 5),
        ("ctx rows written", 7, 6),
        ("kernel start to end", 7, 0),
    ]
    out = {}
    for label, later, earlier in names:
        us = torch.sort(((s[:, later] - s[:, earlier]) & 0xFFFFFFFF).double() * 1e6 / CLOCK_HZ).values
        out[label] = (float(us[0]), float(us[len(us) // 2]), float(us[-1]))
    return out


@galaxy
@pytest.mark.parametrize("name", ["quad", "galaxy"])
def test_timing_build(mesh_device, name):
    """The kernels built with timing stamps: correct as the plain build, and where a merge core's time goes."""
    rows, cols = RECTANGLES[name]
    merge = lm.LseMerge(mesh_device, rows, cols, timing=True)
    if MOCK:
        merge.compile()
        return
    inputs = synthetic(len(merge.coords), merge.users, 5)
    merge.set_partials(*inputs[:3])
    merge.set_sink(inputs[3])
    for _ in range(2):
        merge()
    compare(name, merge, inputs, "timing build", 0.9999)
    table = phase_table(merge.read())
    logger.info(
        f"HY4_MK6 lse {name} phases (us at {CLOCK_HZ / 1e6:.0f} MHz, min / median / max over chips and users): "
        + "; ".join(f"{k}: {a:.1f} / {b:.1f} / {c:.1f}" for k, (a, b, c) in table.items())
    )


@galaxy
def test_latency(mesh_device):
    """Traced latency on 8x4, 8 users, 64 heads: the merge, and its launch floor (the same cores and buffers,
    kernels that return at once); 20 replays back to back, one synchronize, best of three, with 1 and with 16
    calls per trace. The result is checked again after the replays."""
    if MOCK:
        pytest.skip("latency needs the board")
    table = {"ttnn_collect_ms_per_user": 0.185, "us": {}}
    for fidelity in ("hifi4", "lofi"):
        merge = lm.LseMerge(mesh_device, fidelity=fidelity)
        table["bytes"] = merge.bytes_moved()
        inputs = synthetic(len(merge.coords), merge.users, 9)
        merge.set_partials(*inputs[:3])
        merge.set_sink(inputs[3])
        merge()
        floor = 0.9999 if fidelity == "hifi4" else 0.99
        compare("galaxy", merge, inputs, f"latency {fidelity} first run", floor)
        for form, fn in (("merge", merge), ("launch floor", lambda: merge(nop=True))):
            us = {calls: 1e3 * tracing.traced_ms(mesh_device, fn, 20, calls=calls)["replay_ms"] for calls in (1, 16)}
            table["us"][f"{form} {fidelity}"] = us
            logger.info(
                f"HY4_MK6 lse latency 8x4, 8 users, 64 heads, {fidelity}, {form}: {us[1]:.1f} us per call as a trace "
                f"of its own, {us[16]:.1f} us per call with 16 per trace (ttnn form: the collect it replaces, "
                f"185 us per user, PERF.md)"
            )
        merge()
        compare("galaxy", merge, inputs, f"latency {fidelity} after the replays", floor)
        merge.release()
    RESULTS.write_text(json.dumps(table, indent=1))
