# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-chunk replay time of the traced chunked prefill vs position, through the model's own ``prefill_interleaved`` (4 users = 4 mesh rows x 1 slot, as the B=4 demo), on a LAYER SUBSET.

    PROBE_LAYERS=0,1,2,3,14,20,21,24 PROBE_LENS=16384,32768,65536 PROBE_WMAX=1048576 PROBE_MODE=matched|single pytest tests/test_chunk_probe.py -s

PROBE_MODE=matched: every prompt length gets its own S_pad trace (``s_pad_max`` None: the demo's behaviour); single: ``s_pad_max`` = PROBE_WMAX (one trace for every length; with
DSV41_PF_SINGLE=1 the new single-trace chunk, else the old per-S_pad layout captured once at the max context).
Logs ``PROBE`` lines: per length the replay count, mean ms per replay and the ms of the replays at chosen positions, plus the number of captured chunk traces."""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.tt_transformers.tt.common import PagedAttentionConfig

LAYERS = [int(x) for x in os.environ.get("PROBE_LAYERS", "0,1,2,3,14,20,21,24").split(",")]
LENS = [int(x) for x in os.environ.get("PROBE_LENS", "16384,32768,65536").split(",")]
WMAX = int(os.environ.get("PROBE_WMAX", "1048576"))
MODE = os.environ.get("PROBE_MODE", "matched")
CHUNK = int(os.environ.get("PROBE_CHUNK", "512"))
REPS = int(os.environ.get("PROBE_REPS", "2"))
B = 4


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
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_chunk_probe(mesh_device):
    md = mesh_device
    log = lambda m: print(m, flush=True)
    max_ctx = WMAX if MODE == "single" else max(max(LENS) + 512, WMAX)
    pp = default_page_params(max_ctx, 1)
    args, m, _, _ = create_tt_model(
        md,
        B,
        max_ctx,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=LAYERS,
        log=log,
    )
    assert m.Up == 1 and m.U == 1, (m.Up, m.U)
    pm = m.prefill_model
    rec = []
    orig_rw = pm.replay_window

    def rw(tokens, hashes, s0, C, active, vis=None, **kw):  # (prepped / overlap / ops of the host-overlap path)
        th, td = orig_rw(tokens, hashes, s0, C, active, vis=vis, **kw)
        rec.append((s0, th, td))
        return th, td

    pm.replay_window = rw
    ncap = [0]
    orig_cap = pm.capture_dyn_trace

    def cap(C):
        ncap[0] += 1
        return orig_cap(C)

    pm.capture_dyn_trace = cap
    g = torch.Generator().manual_seed(1234)
    toks = torch.randint(1000, 100000, (B, max(LENS)), generator=g, dtype=torch.int64)
    for E in LENS:
        items = [(b, toks[b], 0, E) for b in range(B)]
        for rep in range(REPS):
            rec.clear()
            c0 = ncap[0]
            t0 = time.perf_counter()
            res = m.prefill_interleaved(
                items,
                CHUNK,
                s_pad_max=WMAX if MODE == "single" else None,
                want_logits=bool(os.environ.get("PROBE_SAVE")),
            )
            wall = time.perf_counter() - t0
            n = len(rec)
            td = [r[2] for r in rec]
            pos = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
            log(
                f"PROBE mode={MODE} E={E} rep={rep} S_pad={pm.dyn.S_pad} captures_this_call={ncap[0] - c0} captures_total={ncap[0]} replays={n} wall={wall:.2f}s "
                f"replay_mean={sum(td) / n * 1e3:.1f}ms host_mean={sum(r[1] for r in rec) / n * 1e3:.1f}ms "
                + " ".join(f"s0={rec[i][0]}:{td[i] * 1e3:.0f}ms" for i in pos)
                + f" first={[int(res[b][0]) for b in range(B)]}"
            )
        if os.environ.get("PROBE_SAVE") and rep == REPS - 1:
            torch.save(
                torch.stack([res[b][1] for b in range(B)]).float(),
                f"{os.environ['PROBE_SAVE']}.E{E}.{MODE}.single{os.environ.get('DSV41_PF_SINGLE', '0')}",
            )
    log("PROBE DONE")
