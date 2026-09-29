# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Flat routed expert in indexed mode (all-gather MoE architecture) vs the flat dispatch buffer, one chip, MiMo shapes.

Indexed mode: x = the all-gathered tokens of the dispatch group [chunk_size, emb_dim] and token_index [1, rows] names,
for every row of the flat (region) space, the gathered row it reads. The reference feeds the same rows pre-gathered
into a flat dispatch buffer (what dispatch builds today). y must be bit-identical; the profile gives both times.

Routing: one QuietBox chip (experts 0..63 of 256) for chunk_size = 2 x chunk_size_per_chip tokens, top-8 sampled per
token from the measured L1 / L5 router frequencies (routing_counts_1280tok.json) or uniform.

    scripts/run_safe_pytest.sh --profile models/demos/mimo_v2_d_p/tests/perf/test_flat_expert_indexed.py
    (tags: fl_{flat|indexed}_{route}_S{chunk_size_per_chip})
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import compute_constants
from models.demos.mimo_v2_d_p.tt.flat_expert import FlatRoutedExpert

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None

EMB, INTER, NUM_EXPERTS, TOPK, EPC, CHIPS_IN_GROUP, NUM_CHIPS, CAP = 4096, 2048, 256, 8, 64, 2, 4, 4
ROUTES = os.environ.get("MIMO_IDX_ROUTES", "L1,L5,uniform").split(",")
SEQS = [int(s) for s in os.environ.get("MIMO_IDX_SEQ", "640,2048").split(",")]
ITERS = int(os.environ.get("MIMO_IDX_ITERS", "3"))
MODES = os.environ.get("MIMO_IDX_MODES", "flat,indexed,indexed4,indexed8,indexedrm").split(",")
COUNTS = json.loads(Path(__file__).with_name("routing_counts_1280tok.json").read_text())


def route(name, chunk_size, gen):
    """[chunk_size, TOPK] global expert ids. bank / bankL1 (adversarial): token g only picks experts e with
    e % 8 == g % 8, so every expert's gathered rows g share one residue mod 8 = one DRAM bank of the interleaved x
    (page = row, bank = page % 8; chunk_size_per_chip is a multiple of 32, so g % 8 = token % 8)."""
    if name.startswith("bank"):
        p = torch.ones(NUM_EXPERTS) if name == "bank" else torch.tensor(COUNTS[name[4:]], dtype=torch.float) + 1e-3
        cls = torch.arange(NUM_EXPERTS) % 8
        pg = p.expand(chunk_size, -1) * (cls[None] == (torch.arange(chunk_size) % 8)[:, None])
        return torch.multinomial(pg, TOPK, replacement=False, generator=gen)
    if name == "uniform":
        p = torch.ones(NUM_EXPERTS)
    else:
        p = torch.tensor(COUNTS[name], dtype=torch.float) + 1e-3
    return torch.multinomial(p.expand(chunk_size, -1), TOPK, replacement=False, generator=gen)


def plan(idx, gids, rows):
    """counts / regions rows [1, NUM_EXPERTS], token_index [rows] (the route plan the all-gather path builds)."""
    counts = torch.zeros(1, NUM_EXPERTS, dtype=torch.int32)
    regions = torch.zeros(1, NUM_EXPERTS, dtype=torch.int32)
    tok = torch.zeros(rows, dtype=torch.int32)
    off = 0
    for g in gids:
        t = (idx == g).any(-1).nonzero().flatten()  # ascending gathered rows
        counts[0, g], regions[0, g] = len(t), off
        tok[off : off + len(t)] = t.int()
        off += -(-len(t) // 32) * 32
    assert off <= rows, (off, rows)
    return counts, regions, tok


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_flat_expert_indexed(device):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    gids = list(range(EPC))  # chip (0, 0) of the QuietBox EP table
    weights = [
        [
            (torch.randn(EMB, INTER) * 0.02, torch.randn(EMB, INTER) * 0.02, torch.randn(INTER, EMB) * 0.02)
            for _ in range(EPC)
        ]
    ]
    rm = lambda t, dt: ttnn.from_torch(
        t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for S in SEQS:
        chunk_size = CHIPS_IN_GROUP * S
        _, _, rows, _ = compute_constants(S, NUM_EXPERTS, TOPK, NUM_CHIPS, CHIPS_IN_GROUP, CAP)
        op = FlatRoutedExpert(device, weights, m=chunk_size, H=EMB, I=INTER, gids=[gids], n_global=NUM_EXPERTS, pin=1)
        for rname in ROUTES:
            idx = route(rname, chunk_size, gen)
            counts, regions, tok = plan(idx, gids, rows)
            x = torch.randn(chunk_size, EMB)
            flat = torch.zeros(rows, EMB)
            used = int(regions[0, gids[-1]] + counts[0, gids[-1]])
            flat[:used] = x[tok[:used].long()]
            for g in gids:  # padding rows of each region stay zero in the flat buffer
                o, c = int(regions[0, g]), int(counts[0, g])
                flat[o + c : o + -(-c // 32) * 32] = 0
            x_d, flat_d = rm(x, ttnn.bfloat16), rm(flat, ttnn.bfloat16)
            x8_d = rm(x.reshape(chunk_size * 8, EMB // 8), ttnn.bfloat16)  # 1 KB pages: each row over all 8 banks
            x4_d = rm(x.reshape(chunk_size * 4, EMB // 4), ttnn.bfloat16)  # 2 KB pages (one read each): 4 banks
            c_d, r_d, t_d = rm(counts, ttnn.uint32), rm(regions, ttnn.uint32), rm(tok[None], ttnn.uint32)
            act = [int(counts[0, g]) for g in gids if counts[0, g]]
            logger.info(f"S{S} {rname}: {len(act)} active experts, {sum(act)} rows, max {max(act)}, flat rows {rows}")
            ys = {}
            for mode, call in (
                ("flat", lambda: op(flat_d, c_d, r_d)),
                ("indexed", lambda: op(x_d, c_d, r_d, token_index=t_d)),
                ("indexed4", lambda: op(x4_d, c_d, r_d, token_index=t_d, x_pages_per_row=4)),
                ("indexed8", lambda: op(x8_d, c_d, r_d, token_index=t_d, x_pages_per_row=8)),
                ("indexedrm", lambda: op(x_d, c_d, r_d, token_index=t_d, y_row_major=True)),
            ):
                if mode not in MODES and mode != "flat":
                    continue
                y = call()
                ttnn.synchronize_device(device)
                ys[mode] = ttnn.to_torch(y).float()
                ttnn.deallocate(y)
                tag = f"fl_{mode}_{rname}_S{S}"
                for _ in range(ITERS):
                    signpost(f"{tag}_start")
                    y = call()
                    ttnn.synchronize_device(device)
                    signpost(f"{tag}_end")
                    ttnn.deallocate(y)
            for g in gids:
                o, c = int(regions[0, g]), int(counts[0, g])
                if "indexed" in ys:
                    assert torch.equal(ys["flat"][o : o + c], ys["indexed"][o : o + c]), f"S{S} {rname} expert {g}"
                for pm in [m_ for m_ in ("indexed4", "indexed8") if m_ in ys]:
                    assert torch.equal(ys["flat"][o : o + c], ys[pm][o : o + c]), f"S{S} {rname} expert {g} ({pm})"
                if "indexedrm" not in ys or c == 0:
                    continue
                d = (ys["indexedrm"][o : o + c] - ys["flat"][o : o + c]).abs().max().item()
                assert (
                    d <= 2**-6 * ys["flat"][o : o + c].abs().max().item() + 1e-6
                ), f"S{S} {rname} expert {g} (rm) {d}"
            logger.info(f"S{S} {rname}: indexed y bit-identical to the flat buffer's")
            for t in (x_d, x4_d, x8_d, flat_d, c_d, r_d, t_d):
                ttnn.deallocate(t)
