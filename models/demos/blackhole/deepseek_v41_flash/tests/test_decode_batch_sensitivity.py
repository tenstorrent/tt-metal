# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Does the decode logit row of a user depend on WHAT THE OTHER USERS OF THE BATCH ARE DOING? (APC divergence investigation, no prefix caching involved.)

The same prompt P is prefilled into user ``U0`` and decoded teacher-forced for STEPS steps with identical forced tokens; the other 15 users hold (A) other prompts, (B) other prompts of other lengths / content, (C) the
same prompt P itself at other users (same rows), and in run (D) P sits in ANOTHER user (another mesh row and in-row index). Per step the logits row of the observed user is compared bit-exactly with run A, with
the largest abs diff and the top-1 / top-2 margin of run A. If rows differ, decode numerics depend on the batch composition (a property of the baseline, not of prefix caching).

    DSV41_LAYERS=0-39 DSV41_POOL_DTYPE=fp8 pytest models/demos/blackhole/deepseek_v41_flash/tests/test_decode_batch_sensitivity.py -s
"""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.tt_transformers.tt.common import PagedAttentionConfig

U = 4
STEPS = int(os.environ.get("DSV41_BS_STEPS", "24"))
PLEN = int(os.environ.get("DSV41_BS_PLEN", "720"))
LAYERS = os.environ.get("DSV41_LAYERS", "0-3")
CHUNK = 512


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": int(os.environ.get("DSV41_IL_TRACE_REGION", "1600000000")),
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_decode_logits_vs_batch_composition(mesh_device):
    os.environ.update({"DSV41_PF_UMASK": "1", "DSV41_PREFILL_UP": "1", "DSV41_MO_OVERLAP": "0"})
    md = mesh_device
    log = lambda m: print(m, flush=True)
    a, _, b = LAYERS.partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    B = 4 * U
    S_pad = 2048
    max_ctx = S_pad + 256
    pp = default_page_params(max_ctx, U)
    args, m, _, _ = create_tt_model(
        md,
        B,
        max_ctx,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=layer_ids,
        log=log,
    )
    g = torch.Generator().manual_seed(77)
    rnd = lambda n: torch.randint(1000, 100000, (n,), generator=g, dtype=torch.int64)
    P = rnd(PLEN)
    forced = rnd(STEPS)
    others_a = [rnd(PLEN) for _ in range(B)]
    others_b = [rnd(PLEN - 37 * (u % 5) - 100) for u in range(B)]

    def run(prompts, observed, others_forced_seed):
        """prompts: {user: tokens}; returns the logits rows [STEPS, vocab] of ``observed`` and the argmax tokens"""
        for u in range(B):
            m.release_user(u)
        items = [(u, p, 0, int(p.numel())) for u, p in prompts.items()]
        if (
            os.environ.get("DSV41_BS_SEPARATE") == "1"
        ):  # one prefill call per user (no shared hand-off of the key slabs)
            for it in items:
                m.prefill_interleaved([it], CHUNK, s_pad_max=S_pad, want_logits=True)
        else:
            m.prefill_interleaved(items, CHUNK, s_pad_max=S_pad, want_logits=True)
        gg = torch.Generator().manual_seed(others_forced_seed)
        rows = []
        for k in range(STEPS):
            tok = torch.randint(1000, 100000, (B,), generator=gg, dtype=torch.int64)
            pos = torch.zeros(B, dtype=torch.long)
            for u, p in prompts.items():
                pos[u] = int(p.numel()) + k
            tok[observed] = forced[k]
            m.decode_forward(tok, pos, enable_trace=True, reload_inputs=True)
            rows.append(m.read_logits()[observed].clone().float())
        return torch.stack(rows)

    def full(user_p, observed, others):
        d = {u: others[u] for u in range(B)}
        d[observed] = P
        return d

    A = run(full(P, 1, others_a), 1, 1)  # reference
    if os.environ.get("DSV41_BS_QUICK") == "1":
        runs = {
            "one ROW-MATE (user 0) 100 shorter": run({**full(P, 1, others_a), 0: rnd(PLEN - 100)}, 1, 1),
            "one user of ANOTHER row (user 7) 100 shorter": run({**full(P, 1, others_a), 7: rnd(PLEN - 100)}, 1, 1),
            "one ROW-MATE (user 0) 1 shorter": run({**full(P, 1, others_a), 0: rnd(PLEN - 1)}, 1, 1),
            "one ROW-MATE (user 0) 1000 longer": run({**full(P, 1, others_a), 0: rnd(PLEN + 1000)}, 1, 1),
        }
        runs["moved to user 14"] = runs["one ROW-MATE (user 0) 1 shorter"]
    else:
        runs = {
            "repeat": run(full(P, 1, others_a), 1, 1),
            "all users = P (other prompts, same lengths)": run({u: P for u in range(B)}, 1, 3),
            "moved to user 14": run(full(P, 14, others_a), 14, 1),
            "other users: other lengths (PLEN-100-37*(u%5))": run(full(P, 1, others_b), 1, 2),
            "one other user much longer (1800 tokens)": run({**full(P, 1, others_a), 7: rnd(1800)}, 1, 1),
            "all other users 100 shorter": run({**{u: rnd(PLEN - 100) for u in range(B)}, 1: P}, 1, 1),
            "all other users 100 longer": run({**{u: rnd(PLEN + 100) for u in range(B)}, 1: P}, 1, 1),
        }
    D = runs["moved to user 14"]
    s = A.sort(dim=-1, descending=True).values
    margin = (s[:, 0] - s[:, 1]).tolist()
    out = {}
    for name, X in runs.items():
        eq = [bool(torch.equal(X[k], A[k])) for k in range(STEPS)]
        dif = [float((X[k] - A[k]).abs().max()) for k in range(STEPS)]
        am = [int(X[k].argmax()) == int(A[k].argmax()) for k in range(STEPS)]
        out[name] = (eq, dif, am)
        pc = [float(torch.corrcoef(torch.stack([X[k], A[k]]))[0, 1]) for k in range(STEPS)]
        log(
            f"   logit std {float(A.std()):.3g}, PCC vs reference min {min(pc):.4f} median {sorted(pc)[STEPS // 2]:.4f}"
        )
        log(
            f"{name}: bit-identical steps {sum(eq)}/{STEPS}, max abs logit diff {max(dif):.4g} (median {sorted(dif)[STEPS // 2]:.4g}), argmax equal {sum(am)}/{STEPS}"
        )
    log(
        f"top1-top2 margin of the reference: min {min(margin):.4g}, steps with margin < max diff: {sum(1 for k in range(STEPS) if margin[k] < max(max(v[1]) for v in out.values()))}/{STEPS}"
    )
    log("margins: " + ", ".join(f"{x:.2f}" for x in margin))
