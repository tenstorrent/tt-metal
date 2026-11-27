# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""WA.1 (CPU only, no device): wo_a's DRAM floor and the bank-local layout's roofline from the model's real shapes,
the layout itself checked on the CPU, and the design file.

Shapes: the checkpoint's inference config and safetensors header (tests/woa/woa_roofline.load); today's measured us a
call: results/PB.MM256.json (256 rows) and P.81's op probe (160 rows, cited). Records woa_floor_us_r160 / _r256
(weights at 512 GB/s, a call), woa_today_us_r160 / _r256, woa_upper_bound_ms_c8192 / _c5120 (device ms a chunk,
layers 0-20: 20 calls x (today - max(all bytes at 512 GB/s, busiest core's compute at the HiFi2 peak))).

    scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/woa/test_woa_design.py
"""

import time
from pathlib import Path

import pytest
import torch

from models.demos.common.bringup.core.metrics import record
from models.demos.deepseek_v41_flash_d_p.tests.woa import woa_roofline as rl

DESIGN = Path(__file__).resolve().parents[2] / "bringup" / "WOA_BANK_DESIGN.md"
KILL_MS_8192 = 0.5  # the owner's kill line (WOA_PLAN.md, WA.1)


@pytest.fixture(scope="module")
def model():
    w, ctx = rl.load()
    return w, ctx


@pytest.fixture(scope="module")
def results(model):
    w, ctx = model
    meas = rl.measured()
    res = {c: rl.analyse(w, w.rows(c), meas[w.rows(c)], meas["calls"]) for c in (ctx["deploy_chunk"], rl.CHUNK_BATCH)}
    return meas, res


def test_shapes_are_the_checkpoints(model):
    w, ctx = model
    cfg = ctx["cfg"]
    assert ctx["deploy_chunk"] == rl.CHUNK_DEPLOY
    assert w.G == cfg.o_groups and w.N == cfg.o_lora_rank and w.K == cfg.n_heads // cfg.o_groups * cfg.head_dim
    assert "float8" in w.w_dtype  # fp8 checkpoint -> bfloat8_b (rule 7, DENSE_WEIGHTS)
    assert w.rows(rl.CHUNK_DEPLOY) == 160 and w.rows(rl.CHUNK_BATCH) == 256
    assert w.cap == 110  # MAX_WORKER_CORES on the Galaxy (rule 18, FIX.15)
    assert rl.bfp8_tile_bytes() == 1088


def test_bank_layout_on_cpu(model):
    """Every tile once, equal bank sizes (no padding, OV-3), <= 3 columns a core and <= the core cap, and the per-bank
    reads (one contiguous BW x PN read a K block) reproduce x @ w exactly, in today's K order, for every BW."""
    w, _ = model
    for cpb in (11, 12):
        plan = rl.bank_plan(w.G, w.NT, w.KT, rl.DRAM_BANKS, cpb)
        assert 88 <= len(plan) <= w.cap, len(plan)
        assert max(s.pn for s in plan) == 3
        cover = sorted((s.g, n) for s in plan for n in range(s.n0, s.n0 + s.pn))
        assert cover == [(g, n) for g in range(w.G) for n in range(w.NT)]
        if w.G == rl.DRAM_BANKS:
            assert all(s.g == s.bank for s in plan)  # one group a bank
    # exact on a reduced K (same G, N, bank plan rule): small integers, float64 sums are exact
    KT = 8
    g = torch.Generator().manual_seed(0)
    wt = torch.randint(-3, 4, (w.G, KT * 32, w.N), generator=g).double()
    x = torch.randint(-3, 4, (w.G, 64, KT * 32), generator=g).double()
    ref = torch.einsum("gmk,gkn->gmn", x, wt)
    for cpb in (11, 12):
        plan = rl.bank_plan(w.G, w.NT, KT, rl.DRAM_BANKS, cpb)
        regs = rl.bank_regions(wt, plan)
        assert len({r.shape[0] for r in regs}) == 1  # every bank the same size
        assert sum(r.shape[0] for r in regs) == w.G * KT * w.NT
        for bw in (1, 2, 4, 8):
            y = rl.grouped_mm_from_banks(x, regs, plan, KT, w.NT, bw)
            assert torch.equal(y, ref), (cpb, bw)


def test_host_packing_cost(model):
    """Constraint 4: the host-side region cut + the experts' bank_host_tensor for one layer of the real size."""
    from ttnn.bringup.moe_block_ttnn.flat_routed_expert.flat_routed_expert_weights import bank_host_tensor

    w, _ = model
    wt = torch.randn(w.G, w.K, w.N).bfloat16()
    plan = rl.bank_plan(w.G, w.NT, w.KT)
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        host = bank_host_tensor(rl.bank_regions(wt, plan), rl.DRAM_BANKS)
        dt = (time.perf_counter() - t0) * 1e3
        best = dt if best is None else min(best, dt)
    assert tuple(host.shape) == (w.G * w.KT * w.NT // rl.DRAM_BANKS * 32, rl.DRAM_BANKS * 32)
    print(f"host packing a layer: {best:.1f} ms")
    record("woa_pack_host_ms_layer", round(best, 1))


def test_roofline_and_record(results):
    meas, res = results
    r5, r8 = res[rl.CHUNK_DEPLOY], res[rl.CHUNK_BATCH]
    print()
    print(rl.table(res))
    print(meas["src160"])
    print(meas["src256"])
    assert meas["calls"] == 20
    for r in (r5, r8):
        assert r["cores"] == 96 and r["units_max"] == 3
        assert r["floor_w_us"] < r["best_us"] <= r["realistic_us"] < r["today_us"]
        assert r["upper_ms"] >= r["realistic_ms"] >= 0
    # the floor the owner's plan quotes: today 1.9-2.2x the weights at 512 GB/s
    assert 1.8 < r5["gap_w"] < 2.3 and 1.8 < r8["gap_w"] < 2.3
    record("woa_floor_us_r160", round(r5["floor_w_us"], 2))
    record("woa_floor_us_r256", round(r8["floor_w_us"], 2))
    record("woa_today_us_r160", meas[160])
    record("woa_today_us_r256", meas[256])
    record("woa_upper_bound_ms_c8192", round(r8["upper_ms"], 3))
    record("woa_upper_bound_ms_c5120", round(r5["upper_ms"], 3))
    record("woa_realistic_ms_c8192", round(r8["realistic_ms"], 3))
    record("woa_realistic_ms_c5120", round(r5["realistic_ms"], 3))
    record("woa_best_us_r160", round(r5["best_us"], 2))
    record("woa_best_us_r256", round(r8["best_us"], 2))
    record("woa_kill_8192", int(r8["upper_ms"] < KILL_MS_8192))


def test_design_file(results):
    meas, res = results
    assert DESIGN.exists(), DESIGN
    text = DESIGN.read_text()
    for s in (
        "## 1. Today's wo_a",
        "## 2. Where the gap goes",
        "## 3. The per-bank layout",
        "## 4. The switch, DRAM and load cost",
        "## 5. Roofline of the new layout",
        "## 6. Options of ttnn.bringup.grouped_matmul and the fork tests",
        "## 7. WA.2: the probe and its knob sweep",
        "KILL LINE",
        "WOA_BANK",
        "gmm_reader.cpp",
        "WO_A_GMM",
        "GMM256",
        "flat_routed_expert_weights",
        "plan.cpp",
    ):
        assert s in text, f"design file lacks {s!r}"
    head = text[: text.index("## 1.")]
    r5, r8 = res[rl.CHUNK_DEPLOY], res[rl.CHUNK_BATCH]
    # the headline numbers at the top are the ones computed here
    for v in (r8["upper_ms"], r5["upper_ms"], r8["realistic_ms"], r5["realistic_ms"]):
        assert f"{v:.2f} ms" in head, f"{v:.2f} ms not in the design file's head"
    verdict = "NOT killed" if r8["upper_ms"] >= KILL_MS_8192 else "KILLED"
    assert verdict in head, verdict
    for r in (r5, r8):
        for k in ("floor_w_us", "dram_all_us", "compute_us", "best_us", "realistic_us", "today_us"):
            assert f"{r[k]:.1f}" in text, (k, r[k])
