# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full-model B=8 prefill of 8 parallel users through ``Model.prefill_interleaved`` (one call = what a server step with 8 prompts does): TTFT of the call, replays, replay time per call, and a needle check.

    DSV41_PREFILL_UP=2 DSV41_PREFILL_UP_BATCH=1 (or DSV41_PREFILL_UP=1 for the 2-waves-of-4 baseline) DSV41_PF_PAGED=read DSV41_PF_UMASK=1 DSV41_PF_SINGLE=1 UPF_LENS=32768,131072 pytest tests/test_up_batch_full.py -s

Env: UPF_SCEN (aligned,mixed,stagger,needle), UPF_LENS, UPF_MAXCTX (default 1048576), UPF_REPS (2), UPF_LAYERS (default all), UPF_CHUNK (512), UPF_NEEDLE (prompt json for the needle filler).
``aligned``: 8 prompts of the same length E in one call. ``mixed``: lengths E * (1 - i / 16), i = 0..7. ``stagger``: users 0-3 one call ahead (their first 4 chunks), then everything in one call.
``needle``: 8 prompts of different lengths each with its own 4-digit code at the start; the call is prefilled staggered, 24 greedy tokens are decoded and the code has to be in the text.
"""

import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.tt_transformers.tt.common import PagedAttentionConfig

SCEN = os.environ.get("UPF_SCEN", "aligned,mixed").split(",")
LENS = [int(x) for x in os.environ.get("UPF_LENS", "32768").split(",")]
MAXCTX = int(os.environ.get("UPF_MAXCTX", "1048576"))
REPS = int(os.environ.get("UPF_REPS", "2"))
CHUNK = int(os.environ.get("UPF_CHUNK", "512"))
LAYERS = os.environ.get("UPF_LAYERS")
B = 8


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": int(os.environ.get("DSV41_TRACE_REGION", "1600000000")),
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(21600)
@torch.no_grad()
def test_up_batch_full(mesh_device):
    md = mesh_device
    log = lambda s: print(s, flush=True)
    pp = default_page_params(MAXCTX, B // 4)
    args, m, _, _ = create_tt_model(
        md,
        B,
        MAXCTX,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=[int(x) for x in LAYERS.split(",")] if LAYERS else None,
        log=log,
    )
    pm = m.prefill_model
    log(
        f"UPF Up={m.Up} U={m.U} UP_BATCH={os.environ.get('DSV41_PREFILL_UP_BATCH', '0')} chunk {CHUNK} max_ctx {MAXCTX}"
    )
    rec = []
    orig_rw = pm.replay_window

    def rw(tokens, hashes, s0, C, active, vis=None, **kw):
        th, td = orig_rw(tokens, hashes, s0, C, active, vis=vis, **kw)
        rec.append((s0, th, td))
        return th, td

    pm.replay_window = rw
    if os.environ.get("UPF_WARM", "1") == "1":
        m.warm_serving(CHUNK, MAXCTX, [], [])
        log("UPF warm_serving done")
    U = list(range(B))
    g = torch.Generator().manual_seed(4321)
    toks = torch.randint(1000, 100000, (B, max(LENS + [1])), generator=g, dtype=torch.int64)

    def release():
        for b in U:
            m.release_user(b)

    def call(items, final):
        rec.clear()
        t0 = time.perf_counter()
        r = m.prefill_interleaved(
            items,
            CHUNK,
            s_pad_max=None,
            want_logits=False,
            final={b: f for b, f in final.items()},
            totals={b: e for b, _, _, e in items},
        )
        wall = time.perf_counter() - t0
        return r, wall, len(rec), dict(m.timing)

    bad = []
    for E in LENS:
        for name in [s for s in SCEN if s != "needle"]:
            if name == "aligned":
                Es = [E] * B
            else:
                Es = [(E * (16 - i) // 16) // CHUNK * CHUNK + 37 * (i % 3) for i in range(B)]
            for rep in range(REPS):
                release()
                pre = 0.0
                if name == "stagger":
                    _, w0, n0, _ = call([(b, toks[b], 0, 4 * CHUNK) for b in range(4)], {b: False for b in range(4)})
                    pre = w0
                    items = [(b, toks[b], 4 * CHUNK, Es[b]) for b in range(4)] + [
                        (b, toks[b], 0, Es[b]) for b in range(4, 8)
                    ]
                else:
                    items = [(b, toks[b], 0, Es[b]) for b in U]
                res, wall, n, tm = call(items, {b: True for b in U})
                td = [r[2] for r in rec]
                mean_td = sum(td) / max(1, len(td))
                nwin_max = max(-(-(it[3] - it[2]) // CHUNK) for it in items)
                wins_total = sum(-(-(it[3] - it[2]) // CHUNK) for it in items)
                log(
                    f"UPF {name} E={E} rep={rep} Es={Es}: wall {wall:.2f} s (TTFT of the call), replays {n} (longest user {nwin_max} windows, {wins_total} user-windows), "
                    f"wall/replay {wall / max(1, n) * 1e3:.0f} ms, wall per user-window {wall / wins_total * 1e3:.0f} ms, sync replay_mean {mean_td * 1e3:.0f} ms (0 = async), "
                    f"first tokens {[int(res[b][0]) for b in U]}"
                )
        if "needle" in SCEN:
            from transformers import AutoTokenizer

            tk = AutoTokenizer.from_pretrained(
                os.environ.get("DSV41_CKPT", "/mnt/tt-data/ssinghal/deepseek-v41-flash-1017")
            )
            body = json.load(
                open(os.environ.get("UPF_NEEDLE", "/mnt/tt-data/ssinghal/dsv4-logs/ttopt/needle_64k.json"))
            )[0]["prompt"]
            body = body.split("\n\n", 1)[1].rsplit("\n\n", 1)[0]
            fill = tk(body, add_special_tokens=False)["input_ids"]
            codes = [4817, 2953, 6071, 3306, 7742, 1589, 9024, 5368]
            tgt = [E * (16 - i) // 16 // CHUNK * CHUNK - 40 for i in range(B)]
            ids = []
            for b in U:
                h = tk(f"The secret code is {codes[b]}. Remember it.\n\n", add_special_tokens=False)["input_ids"]
                t = tk(
                    "\n\nWhat is the secret code mentioned at the very start? Answer with just the number.",
                    add_special_tokens=False,
                )["input_ids"]
                n = max(0, tgt[b] - len(h) - len(t) - 1)
                f = (fill * (n // len(fill) + 1))[:n]
                ids.append(torch.tensor([tk.bos_token_id] + h + f + t, dtype=torch.long))
            Es = [int(x.numel()) for x in ids]
            release()
            _, w0, _, _ = call([(b, ids[b], 0, 4 * CHUNK) for b in range(4)], {b: False for b in range(4)})
            items = [(b, ids[b], 4 * CHUNK, Es[b]) for b in range(4)] + [(b, ids[b], 0, Es[b]) for b in range(4, 8)]
            res, wall, n, _ = call(items, {b: True for b in U})
            log(f"UPF needle E={E}: lengths {Es}, call wall {wall:.2f} s, replays {n}")
            tok = torch.tensor([int(res[b][0]) for b in U], dtype=torch.long)
            outs = [[int(t_)] for t_ in tok]
            for k in range(int(os.environ.get("UPF_DEC", "64"))):
                o = m.decode_forward(
                    tok, torch.tensor([e + k for e in Es], dtype=torch.long), enable_trace=False, reload_inputs=True
                )
                tok = o.clone().long()
                for b in U:
                    outs[b].append(int(tok[b]))
            good = 0
            for b in U:
                txt = tk.decode(outs[b])
                hit = str(codes[b]) in txt
                good += hit
                log(f"UPF needle user {b} len {Es[b]} code {codes[b]} hit {hit}: {txt!r}")
            log(f"UPF needle E={E}: {good}/8 correct")
            bad.append(B - good)
    log(f"UPF DONE needle misses {bad}")
    assert not any(bad)
