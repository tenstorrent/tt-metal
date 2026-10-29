# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DSV41_PREFILL_UP_BATCH probe through the SERVER path ``Model.prefill_interleaved``, B=8 (2 prefill slots per mesh row, DSV41_PREFILL_UP=2), layer subset.

Scenarios (UPB_SCEN): ``aligned`` (8 prompts of equal length, one call), ``mixed`` (8 prompts of different lengths, one call), ``stagger`` (users 0-3 are one call ahead: the second call
continues them at start 2048 while users 4-7 start at 0 -> every replay of the call has slots at different starts).
Each scenario runs (a) as ONE call with all users (UP_BATCH: one replay per window round whatever the starts) and (b) as one call per distinct start group (the one-start-per-replay reference);
the last-token logits and two decode steps of every user must be equal. With UPB_SAVE=<dir> the logits of (a) are saved; with UPB_REF=<dir> they are compared with a file of an earlier run
(run once with DSV41_PREFILL_UP_BATCH=0 and UPB_SAVE to get the switch-off reference; the saved file of the off run holds its (a) = per-start-group calls).

    DSV41_PREFILL_UP=2 DSV41_PREFILL_UP_BATCH=1 DSV41_PF_PAGED=read DSV41_PF_UMASK=1 DSV41_PF_SINGLE=1 UPB_LAYERS=0,1,2,3,14,20,21,24 pytest tests/test_up_batch_probe.py -s
"""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.tt_transformers.tt.common import PagedAttentionConfig

LAYERS = [int(x) for x in os.environ.get("UPB_LAYERS", "0,1,2,3,14,20,21,24").split(",")]
SCEN = os.environ.get("UPB_SCEN", "aligned,mixed,stagger").split(",")
CHUNK = int(os.environ.get("UPB_CHUNK", "512"))
B = 8
SPAD = 16384


def pcc(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


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
def test_up_batch_probe(mesh_device):
    md = mesh_device
    log = lambda s: print(s, flush=True)
    max_ctx = 65536
    pp = default_page_params(max_ctx, B // 4)
    args, m, _, _ = create_tt_model(
        md,
        B,
        max_ctx,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=LAYERS,
        log=log,
    )
    assert m.U == 2 and m.Up == 2, (m.U, m.Up)
    pm = m.prefill_model
    log(
        f"UPB per_row={pm.dyn.per_row if pm.dyn is not None else None} switch={os.environ.get('DSV41_PREFILL_UP_BATCH')}"
    )
    g = torch.Generator().manual_seed(77)
    toks = torch.randint(1000, 100000, (B, 12000), generator=g, dtype=torch.int64)
    U = list(range(B))

    def release():
        for b in U:
            m.release_user(b)

    def call(items, final, totals, want=True):
        n0 = pm.timing.get("n_replays", 0)
        t0 = time.perf_counter()
        r = m.prefill_interleaved(items, CHUNK, s_pad_max=SPAD, want_logits=want, final=final, totals=totals)
        return r, time.perf_counter() - t0

    def decode(res, Es):
        tok = torch.tensor([int(res[b][0]) for b in U], dtype=torch.long)
        out = []
        for k in range(2):
            o = m.decode_forward(
                tok, torch.tensor([e + k for e in Es], dtype=torch.long), enable_trace=False, reload_inputs=True
            )
            out.append(m.read_logits().float().clone())
            tok = o.clone().long()
        return out

    def run(name, grouped):
        release()
        fin = {b: True for b in U}
        times = []
        if name == "aligned":
            Es = [6000] * B
            groups = [[(b, toks[b], 0, Es[b]) for b in U]]
            tot = Es
        elif name == "mixed":
            Es = [3000, 5200, 8192, 9000, 2500, 4096, 7000, 6100]
            groups = [[(b, toks[b], 0, Es[b]) for b in U]]
            tot = Es
        elif name == "stagger":
            Es = [9000, 9100, 8300, 7777, 7000, 8200, 6100, 9000]
            _, t = call(
                [(b, toks[b], 0, 2048) for b in range(4)],
                {b: False for b in range(4)},
                {b: Es[b] for b in range(4)},
                want=False,
            )
            times.append(t)
            groups = [[(b, toks[b], 2048, Es[b]) for b in range(4)] + [(b, toks[b], 0, Es[b]) for b in range(4, 8)]]
            tot = Es
        else:
            raise ValueError(name)
        res = {}
        if grouped:  # the one-start-per-replay reference: one call per distinct start
            items = groups[0]
            for st in sorted({it[2] for it in items}, reverse=True):
                grp = [it for it in items if it[2] == st]
                r, t = call(grp, {it[0]: True for it in grp}, {it[0]: tot[it[0]] for it in grp})
                times.append(t)
                res.update(r)
        else:
            r, t = call(groups[0], fin, {b: tot[b] for b in U})
            times.append(t)
            res.update(r)
        lg = [res[b][1].float().clone() for b in U]
        return Es, [int(res[b][0]) for b in U], lg, decode(res, Es), times

    def cmp(a, b):
        eq = (
            a[1] == b[1]
            and all(torch.equal(x, y) for x, y in zip(a[2], b[2]))
            and all(torch.equal(x, y) for x, y in zip(a[3], b[3]))
        )
        pc = min(pcc(x, y) for x, y in zip(a[2], b[2]))
        mx = max(float((x - y).abs().max()) for x, y in zip(a[2], b[2]))
        return eq, pc, mx

    ok = True
    for name in SCEN:
        one = run(name, grouped=False)
        grp = run(name, grouped=True)
        eq, pc, mx = cmp(one, grp)
        log(
            f"UPB {name}: Es {one[0]} first tokens one-call {one[1]} grouped {grp[1]}; logits+2 decode steps bit-equal {eq}, min pcc {pc:.6f}, max|d| {mx:.4f}; "
            f"wall s one-call {[round(t, 2) for t in one[4]]} grouped {[round(t, 2) for t in grp[4]]}"
        )
        ok &= pc > 0.999 and one[1] == grp[1]
        sd = os.environ.get("UPB_SAVE")
        if sd:
            os.makedirs(sd, exist_ok=True)
            torch.save({"first": one[1], "lg": one[2], "dec": one[3]}, f"{sd}/{name}.pt")
        rd = os.environ.get("UPB_REF")
        if rd and os.path.exists(f"{rd}/{name}.pt"):
            ref = torch.load(f"{rd}/{name}.pt")
            r = (one[0], ref["first"], ref["lg"], ref["dec"], [])
            eq2, pc2, mx2 = cmp(one, r)
            log(
                f"UPB {name} vs saved reference ({rd}): first tokens equal {one[1] == ref['first']}, bit-equal {eq2}, min pcc {pc2:.6f}, max|d| {mx2:.4f}"
            )
            ok &= pc2 > 0.999 and one[1] == ref["first"]
    log(f"UPB DONE ok={ok}")
    assert ok
