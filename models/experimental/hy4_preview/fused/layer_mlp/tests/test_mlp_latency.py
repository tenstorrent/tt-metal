# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced latency of the MLP layer program on the 8x4 mesh (fused/layer_mlp/program.py) for 8 rows whose picks make a
union of 8, 16, 32 and 64 experts (plus the shared expert), broken down by stage.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file>
    pytest <this file> -x

Inputs. The router is real (scores, top-8, union and weights run on every chip) with a crafted weight: expert e
reads hidden channel e. The streams are four copies of a row pattern whose channels 0..255 are +4 for the row's 8
experts and -4 elsewhere, so the normalised x picks them whatever the pre-mix coefficients are (fused/moe/tests/
test_latency.py crafted). The expert weights are zero: the reads and the matmuls are the same, and F = 0 keeps the
streams (and so the picks) the same over the replays of the post + pre form, whose leading post adds F to them.

Measured, per union and form (pre: the root's leading iHC pass is the pre-mix; post_pre: post of the sublayer before
and pre, the form a layer runs): the program as a trace of 1 call and of 8 calls, 20 replays back to back and one
synchronize (fused/common/tracing.py). Then the same with kernels that leave wall-clock stamps (the transport's
words, the MoE piece's HY4_TIMING words), read after the last replay:

    root site     writer start -> x handed to the sender (iHC pass + norm + 16 staging writes)
    X1'           -> x counted on the root's hub (all packets handed to the routers and the own copy done)
    per chip      x arrived -> expert cores released (x multicast, router, union, meta page)
                  released -> partial complete (experts, shared expert)
                  partial complete -> handed on
    root reduce   start -> stage 1 landed -> stage 2 landed -> F written (the program's end on the root)
    X6            F written - (the root's x + the slowest chip's x-to-partial time): from the last partial being
                  ready to F on the site cores, with the broadcast's transit to that chip counted in (the chips'
                  clocks are not synchronised)

Next to it the ttnn form of the same stages (PERF.md 3.2, traced, 8 users): iHC pre-mix 0.302 + norm 0.096 + MoE input
all-gather 0.020 + router 0.170 + dispatch 0.051 + experts 0.848 + combine 0.083 + shared expert 0.097 + iHC post-mix
0.085 = 1.752 ms. Log lines start with HY4_MK8."""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.common import tracing
from models.experimental.hy4_preview.fused.layer_common import transport as tp
from models.experimental.hy4_preview.fused.layer_mlp.program import E, HIDDEN, MlpLayer, is_mock
from models.experimental.hy4_preview.fused.layer_mlp.tests.test_mlp_layer import FIRST_WAIT, galaxy, site_params
from models.experimental.hy4_preview.fused.moe import layout as L

UNIONS = [8, 16, 32, 48, 64]  # 43 to 57 is what 8 real token rows produce on checkpoint layers 1 to 5
# (U picked by the active rows, active rows): all 8 rows active, then 4 active rows and 4 idle ones
CASES = [(U, L.ROWS) for U in UNIONS] + [(16, 4), (32, 4)]
CYCLE_NS = 1e3 / 1350  # the cores' wall clock counts AICLK cycles (1350 MHz)
TTNN_US = 1752.0
ENTRIES = E + 2  # the routed experts and the shared expert's two BFP4 terms


def us(ticks):
    return (int(ticks) & 0xFFFFFFFF) * CYCLE_NS / 1e3


def crafted(U, active=L.ROWS, seed=0):
    """(streams [8, 4, 6144], the rows' picks). The first ``active`` rows pick 8 experts each, U distinct ones
    between them (U / 8 groups of 8, from expert 8 on). The other rows are idle: zero streams, so x = 0, every score
    is 0.5 and the row picks the 8 lowest ids (the router always picks 8; bias zero here): experts 0..7, which no
    active row uses. An idle row's output is zero."""
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(L.ROWS, HIDDEN, generator=g)
    s[:, :E] = -4.0
    picks = []
    for r in range(L.ROWS):
        if r >= active:
            s[r] = 0.0
            picks.append(list(range(8)))
            continue
        group = 1 + r * (U // 8) // active
        ids = list(range(8 * group, 8 * group + 8))
        s[r, ids] = 4.0
        picks.append(ids)
    return s[:, None, :].repeat(1, 4, 1).contiguous(), picks


def zero_streams(mesh_device):
    share = L.tp_share(32)
    rows = lambda stride: ENTRIES * stride // L.BFP4_TILE_BYTES * L.TILE
    return tuple(
        L.to_device_stream(mesh_device, torch.zeros(rows(stride), 256))
        for stride in (share.gate_up_stride, share.down_stride)
    )


def stages(mlp):
    """The stamps of the last run, as the module docstring's stages (us)."""
    topo, t = mlp.topo, mlp.t
    words = t.read_words()
    hub = (mlp.op.hub.x, mlp.op.hub.y)
    stamp = lambda coord, core: [int(v) for v in t.core_words(words[coord], core)[tp.W_STAMP : tp.W_STAMP + 4]] + [
        int(t.core_words(words[coord], core)[15])
    ]
    root = topo.root
    site0 = (mlp.site.cores[0].x, mlp.site.cores[0].y)
    s_site = stamp(root, site0)
    h = {c: stamp(c, hub) for c in topo.chips}
    red = stamp(root, t.reduce_cores(root)[0])
    router = {c: us(v[2] - v[1]) for c, v in h.items()}
    experts = {c: us(v[3] - v[2]) for c, v in h.items()}
    feed = {c: us(v[4] - v[3]) for c, v in h.items()}
    compute = {c: router[c] + experts[c] for c in h}
    slowest = max(compute, key=compute.get)
    others = [c for c in h if c != root]
    x_root = us(h[root][1] - h[root][0])
    f_done = us(red[3] - red[0])
    out = {
        "site": us(s_site[1] - s_site[0]),
        "x_root": x_root,
        "router_root": router[root],
        "router_max": max(router.values()),
        "experts_root": experts[root],
        "experts_others": sorted(experts[c] for c in others),
        "feed_max": max(feed.values()),
        "slowest": (slowest, compute[slowest]),
        "s1": us(red[1] - red[0]),
        "s2": us(red[2] - red[0]),
        "f": f_done,
        "x6": f_done - x_root - compute[slowest],
        "by_chip": {c: round(compute[c], 1) for c in h},
    }
    # the MoE piece's own words on the root: its hub's cycles to x sent, scores in, release sent
    status = mlp.read_chip(root)["status"]
    out["moe_hub_root"] = [us(v) for v in status[8:11]]
    return out


@galaxy
@pytest.mark.parametrize("timing", [False, True], ids=["plain", "stamps"])
@pytest.mark.parametrize("form", ["pre", "post_pre"])
def test_latency(mesh_device, form, timing):
    post = form == "post_pre"
    table = {}
    mlp = MlpLayer(mesh_device, timing=timing, wait_limit=FIRST_WAIT)
    streams = (
        tuple(L.to_device_stream(mesh_device, torch.zeros(64, 256)) for _ in range(2))
        if is_mock()
        else zero_streams(mesh_device)
    )
    mlp.set_layer(1, [site_params(1)], 0, torch.ones(HIDDEN), torch.eye(E, HIDDEN), torch.zeros(E), streams=streams)
    if is_mock():  # the build compiles; nothing is timed
        mlp.run(post=post)
        return
    for U, active in CASES:
        x, picks = crafted(U, active)
        slots = len({e for row in picks for e in row})
        name = f"U={U}" if active == L.ROWS else f"U={U} on {active} rows + {L.ROWS - active} idle rows (union {slots})"
        mlp.set_streams(x)
        mlp.set_f(torch.zeros(L.ROWS, HIDDEN))
        ttnn.synchronize_device(mesh_device)
        run = lambda: mlp.run(post=post)
        run()
        ttnn.synchronize_device(mesh_device)
        status = mlp.status()
        assert not status, f"{name}: words not at rest: {status}"
        ids = mlp.read_chip(mlp.topo.root)["ids"]
        assert all(sorted(ids[r].tolist()) == picks[r] for r in range(L.ROWS)), f"{name}: picks {ids.tolist()}"
        one = tracing.traced_ms(mesh_device, run, calls=1)["replay_ms"] * 1e3
        eight = tracing.traced_ms(mesh_device, run, calls=8)["replay_ms"] * 1e3
        status = mlp.status()
        assert not status, f"{name} after the traces: words not at rest: {status}"
        ids = mlp.read_chip(mlp.topo.root)["ids"]
        assert all(
            sorted(ids[r].tolist()) == picks[r] for r in range(L.ROWS)
        ), f"{name} after the traces: picks changed"
        table[name] = (one, eight)
        logger.info(
            f"HY4_MK8 mlp latency {form} {name} (+ shared) stamps={timing}: {eight:.1f} us per call with 8 calls per "
            f"trace, {one:.1f} us as a trace of its own; ttnn form {TTNN_US:.0f} us: {TTNN_US / eight:.1f}x"
        )
        if timing:
            s = stages(mlp)
            eo = s["experts_others"]
            logger.info(
                f"HY4_MK8 mlp stages {form} {name} (us): root site (iHC + norm + staging) {s['site']:.1f}; x on the "
                f"root's hub at {s['x_root']:.1f} (X1' send side {s['x_root'] - s['site']:.1f}); router (x multicast, "
                f"scores, union, release) root {s['router_root']:.1f} max {s['router_max']:.1f} [MoE hub words on the "
                f"root: x sent {s['moe_hub_root'][0]:.1f}, scores in {s['moe_hub_root'][1]:.1f}, released "
                f"{s['moe_hub_root'][2]:.1f}]; experts root {s['experts_root']:.1f}, other chips min {eo[0]:.1f} "
                f"median {eo[len(eo) // 2]:.1f} max {eo[-1]:.1f}; hand-on max {s['feed_max']:.1f}; slowest chip "
                f"{s['slowest'][0]} x-to-partial {s['slowest'][1]:.1f}; root reduce: stage 1 landed {s['s1']:.1f}, "
                f"stage 2 landed {s['s2']:.1f}, F written {s['f']:.1f}; X6 from the last partial to F (with the "
                f"broadcast's transit) {s['x6']:.1f}; traced {eight:.1f} per call"
            )
            grid = [[s["by_chip"][(r, c)] for c in range(4)] for r in range(8)]
            logger.info(f"HY4_MK8 mlp x-to-partial by chip {form} {name} (us, mesh rows x columns): {grid}")
    logger.info(
        f"HY4_MK8 mlp latency {form} stamps={timing} summary, us per call (8 calls per trace / own trace): "
        + ", ".join(f"{name}: {v[1]:.1f} / {v[0]:.1f}" for name, v in table.items())
        + f"; ttnn form {TTNN_US:.0f} us"
    )
