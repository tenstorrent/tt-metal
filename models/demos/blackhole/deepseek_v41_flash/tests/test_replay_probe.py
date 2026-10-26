# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bounded replay (DSV41_PF_REPLAY) through the SERVER path ``Model.prefill_interleaved`` on a layer subset: every scenario runs with the replay ON (R) and OFF (R = 0, same split traces, every chunk runs
every layer) in ONE process and the last-token logits of every user and two decode steps must be bit-identical.

    DSV41_PF_REPLAY=2560 DSV41_PF_PAGED=read DSV41_PF_UMASK=1 DSV41_PREFILL_UP=1 PROBE_LAYERS=0,1,2,3,14,20,21,24 pytest tests/test_replay_probe.py -s

Scenarios (PROBE_SCEN, comma list): equal (4 prompts of 9000 tokens), mixed (3000 / 5200 / 8192 / 9000), pieces (9000 tokens in 3 calls), stagger (user 0 one call ahead of the others), short (700 / 1500: below the
zone), apc (cold, then hit of the same prompts)."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.tt_transformers.tt.common import PagedAttentionConfig

LAYERS = [int(x) for x in os.environ.get("PROBE_LAYERS", "0,1,2,3,14,20,21,24").split(",")]
SCEN = os.environ.get("PROBE_SCEN", "equal,mixed,pieces,stagger,short,apc").split(",")
CHUNK = 512
B = 4
SPAD = 16384
R_ON = int(os.environ.get("DSV41_PF_REPLAY", "2560"))


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": int(os.environ.get("DSV41_TRACE_REGION", "650000000")),
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_replay_probe(mesh_device):
    md = mesh_device
    log = lambda s: print(s, flush=True)
    assert R_ON > 0, "set DSV41_PF_REPLAY (the split traces are captured at the first prefill)"
    max_ctx = 65536
    pp = default_page_params(max_ctx, 1)
    args, m, _, _ = create_tt_model(
        md,
        B,
        max_ctx,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=LAYERS,
        log=log,
    )
    assert m.Up == 1 and m.U == 1
    pm = m.prefill_model
    g = torch.Generator().manual_seed(77)
    toks = torch.randint(1000, 100000, (B, 12000), generator=g, dtype=torch.int64)

    def release():
        for b in range(B):
            m.release_user(b)

    def call(items, final, totals, want=True):
        kw = {}
        if final is not None:
            kw["final"] = final
        if totals is not None:
            kw["totals"] = totals
        return m.prefill_interleaved(items, CHUNK, s_pad_max=SPAD, want_logits=want, **kw)

    def decode(res, Es):
        tok = torch.tensor([int(res[b][0]) for b in range(B)], dtype=torch.long)
        out = []
        for k in range(2):
            o = m.decode_forward(
                tok, torch.tensor([e + k for e in Es], dtype=torch.long), enable_trace=False, reload_inputs=True
            )
            out.append(m.read_logits().float().clone())
            tok = o.clone().long()
        return out

    def scen(name, R):
        os.environ["DSV41_PF_REPLAY"] = str(R)
        release()
        if getattr(m, "apc", None) is not None:
            m.reset_prefix_cache()
        U = list(range(B))
        if name == "equal":
            Es = [9000] * B
            res = call([(b, toks[b], 0, 9000) for b in U], {b: True for b in U}, {b: 9000 for b in U})
        elif name == "mixed":
            Es = [3000, 5200, 8192, 9000]
            res = call([(b, toks[b], 0, Es[b]) for b in U], {b: True for b in U}, {b: Es[b] for b in U})
        elif name == "pieces":
            Es = [9000] * B
            for lo, hi in ((0, 2048), (2048, 5120)):
                call([(b, toks[b], lo, hi) for b in U], {b: False for b in U}, {b: 9000 for b in U}, want=False)
            res = call([(b, toks[b], 5120, 9000) for b in U], {b: True for b in U}, {b: 9000 for b in U})
        elif (
            name == "stagger"
        ):  # user 0 is one call (3 chunks) ahead of the others; every user's prompt ends at its own length
            Es = [9000, 9100, 8300, 7777]
            call([(0, toks[0], 0, 3072)], {0: False}, {0: Es[0]}, want=False)
            res = call(
                [(0, toks[0], 3072, Es[0])] + [(b, toks[b], 0, Es[b]) for b in (1, 2, 3)],
                {b: True for b in U},
                {b: Es[b] for b in U},
            )
        elif name == "short":
            Es = [700, 1500, 2400, 2700]
            res = call([(b, toks[b], 0, Es[b]) for b in U], {b: True for b in U}, {b: Es[b] for b in U})
        else:
            raise ValueError(name)
        lg = [res[b][1].float().clone() for b in U]
        return Es, [int(res[b][0]) for b in U], lg, decode(res, Es)

    def same(a, b):
        return (
            a[1] == b[1]
            and all(torch.equal(x, y) for x, y in zip(a[2], b[2]))
            and all(torch.equal(x, y) for x, y in zip(a[3], b[3]))
        )

    ok = True
    for name in [s for s in SCEN if s != "apc"]:
        on = scen(name, R_ON)
        n_a = pm.timing.get("replays_a_only", 0)
        off = scen(name, 0)
        eq = same(on, off)
        ok &= eq
        log(
            f"REPLAY_PROBE {name}: Es {on[0]} first on {on[1]} off {off[1]} bit-equal (first tokens, logits, 2 decode steps): {eq}; A-only replays in the ON call {n_a}"
        )
        if not eq:
            for b in range(B):
                log(f"   user {b}: logits max|d| {float((on[2][b] - off[2][b]).abs().max()):.5f}")
    if "apc" in SCEN:
        m.enable_apc(int(4 * 2**30))
        on_cold = scen("equal", R_ON)
        on_hit = None
        os.environ["DSV41_PF_REPLAY"] = str(R_ON)
        for b in range(B):
            m.release_user(b)
        res = call([(b, toks[b], 0, 9000) for b in range(B)], {b: True for b in range(B)}, {b: 9000 for b in range(B)})
        on_hit = (
            [9000] * B,
            [int(res[b][0]) for b in range(B)],
            [res[b][1].float().clone() for b in range(B)],
            decode(res, [9000] * B),
        )
        hits = m.apc.stats["restores"]
        off_cold = scen("equal", 0)
        eq1, eq2 = same(on_cold, on_hit), same(on_cold, off_cold)
        ok &= eq1 and eq2 and hits > 0
        log(f"REPLAY_PROBE apc: restores {hits}; cold(R) == hit(R): {eq1}; cold(R) == cold(0): {eq2}")
    log(f"REPLAY_PROBE ALL BIT-EQUAL: {ok}")
    assert ok
