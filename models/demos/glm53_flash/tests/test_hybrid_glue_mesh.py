# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device A/B of the hybrid decode glue knobs (2026-10-26, _agents/hybrid_glue/STATUS.md). NOT RUN when written (the
chips belong to the validation queue): every case names its expected outcome. Importorskip-safe (no device at import).

(2,4) block (TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 TT_MESH_GRAPH_DESC_PATH=$W/tt_metal/fabric/mesh_graph_descriptors/
p150_x8_mesh_graph_descriptor.textproto MESH_DEVICE=P150x8 GLM53_MESH_SHAPE=2,4), the hybrid MoE layer 4 with real
weights (the stream expert op is the default since the 2026-10-26 rule; the first run converts the block's stream cache):

* ``test_row_glue_collectives_block``: the collectives alone on [1,1,T,4096] bf16 at T = 4 / 2 per row: the compact
  gather (untilize + row-major all_broadcast / concat + zero-padded tilize) is BIT-IDENTICAL to the padded
  ``all_gather_rows``; ``all_reduce_rows_rs_ag`` (reduce-scatter on the hidden dim + all-gather) vs the fp32 sum of
  the rows' partials: <= 8 bf16 ulps, PCC > 0.9999, every row the same tensor; traced us per call of each form.
* ``test_hybrid_gather_compact_block``: H1 -- ``GLM53_HYBRID_GATHER=compact`` vs ``padded`` through ``TtMoE``:
  BIT-IDENTICAL outputs on every chip (the gather is exact data movement, the expert op sees the same tiles),
  both > 0.995 vs HF, the L1 high-water mark unchanged, no L1-resident state.
* ``test_hybrid_combine_allreduce_block``: H2 -- ``GLM53_HYBRID_COMBINE=allreduce`` (rs + ag async pair + the E_r
  selector matmul) and ``allreduce_ccl`` (the ``ttnn.all_reduce`` composite) vs ``rs`` (the padded reduce-scatter
  composite): PCC >= 0.9999 and <= 8 bf16 ulps (the same 8 partials summed in another ring order: not bit
  identity), the row's columns byte-identical, > 0.995 vs HF.
* ``test_hybrid_glue_traced_block``: both knobs inside a captured trace, 3 replays on new inputs == the eager
  forward of the same inputs (bit-identical: the graph is static, the constants persistent).
* ``test_hybrid_glue_perf_block``: traced us per MoE forward for (padded, rs) / (compact, rs) / (padded, allreduce) /
  (compact, allreduce) at 4 users per row: the glue rows of the STATUS table (expected ~-60 us for H1 and ~-100 us
  for H2 per layer on the 8x4; the block's 2-row collectives are cheaper, the direction is what this case checks).
* ``test_dsa_dense_skip_block``: B8a through the block decoder test (``test_mesh_decoder`` layer 3, positions 0..S+N
  = every slot dense) with ``GLM53_POOL_GATHER=dense_skip``: the same per-user PCC bars as without the knob.

Chip 0 (``TT_VISIBLE_DEVICES=0``, mesh (1,1)):
* ``test_dsa_dense_skip_chip0``: B8a on layer 3 with the R2040 reference (prefill 2040 tokens, decode 2040..2050: every
  step dense): the dense-skip graph vs the gather graph on the same token -- expected BIT-IDENTICAL (same flash-MLA
  program over the same logical rows; PCC logged either way, bar > 0.9999), the trace captured with the knob and
  replayed through 2050 against the HF dumps (PCC > 0.995), then the regime guard: refreshing the positions to 2051
  under the captured dense-skip graph raises RuntimeError; after ``clear_graph_regime`` the eager sparse step runs.

The chain and the e2e (not here): the block decoder (``GLM53_MESH_B=4 GLM53_MESH_S=12 pytest tests/test_mesh_decoder.py``)
and the 22-layer block chain (``GLM53_MESH_SHAPE=2,4 GLM53_MODEL_LAYERS=0-21 GLM53_MODEL_B=4 pytest tests/test_mesh_model.py``)
with ``GLM53_HYBRID_GATHER=compact GLM53_HYBRID_COMBINE=allreduce GLM53_POOL_GATHER=dense_skip``, then the 8x4 e2e at
4 users/row (``GLM53_MESH_SHAPE=8,4 GLM53_MODEL_LAYERS=0-44 GLM53_MODEL_B=4``; the stream op is the default now, so no
``GLM53_STREAM_EXPERTS=1`` is needed) with the knobs one at a time (reference 69.1 ms / 135 ms TTFT), and a layer-4 tracy.

    TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 TT_MESH_GRAPH_DESC_PATH=$W/tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto \\
        MESH_DEVICE=P150x8 GLM53_MESH_SHAPE=2,4 pytest models/demos/glm53_flash/tests/test_hybrid_glue_mesh.py -k block -s
    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_hybrid_glue_mesh.py -k chip0 -s
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.tests.test_mesh_moe import l1_high_water, l1_resident  # noqa: E402
from models.demos.glm53_flash.tests.test_moe import LAYER, realistic_mlp_input, weights  # noqa: E402,F401 (fixture)
from models.demos.glm53_flash.tests.test_moe_block import timed_trace  # noqa: E402
from models.demos.glm53_flash.tests.utils import pcc  # noqa: E402
from models.demos.glm53_flash.tt import moe as M  # noqa: E402
from models.demos.glm53_flash.tt.config import GLM53FlashConfig  # noqa: E402
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params  # noqa: E402
from models.demos.glm53_flash.tt.moe import TtMoE  # noqa: E402

H = 4096
ROWS, COLS = (int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
STEPS = int(os.environ.get("GLM53_BLOCK_STEPS", "20"))
N_REP = 50
R2040 = Path("bringup/references/glm53_flash/R2040")
DSA_LAYER = 3


def _ulp(ref: torch.Tensor) -> float:
    m = float(ref.abs().max())
    return 2.0 ** (torch.tensor(m).log2().floor().item() - 7) if m > 0 else 2.0**-133


def _timed(mesh_device, fn, label) -> float:
    out = fn()
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(out)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(N_REP):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) / N_REP * 1e6
    logger.info(f"glue {label}: {us:.1f} us per call (traced replay)")
    ttnn.release_trace(mesh_device, tid)
    ttnn.deallocate(out)
    return us


def _moe_layer(mesh_device, weights):
    snapshot, text, raw, hf, sd = weights
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    mlp = {k[len("mlp.") :]: v for k, v in raw.items() if "experts." not in k or "shared_experts" in k}
    mlp["experts.gate_up_proj"] = sd["experts.gate_up_proj"]
    mlp["experts.down_proj"] = sd["experts.down_proj"]
    mesh = GLMMesh.from_device(mesh_device)
    t0 = time.perf_counter()
    tt = TtMoE(mesh_device, cfg, mlp, mesh=mesh)
    logger.info(
        f"hybrid MoE layer {LAYER} on {ROWS}x{COLS} (stream experts: {getattr(tt.experts, 'stream', None)}): built in "
        f"{time.perf_counter() - t0:.1f} s"
    )
    return tt, mesh, hf, snapshot, text, raw


def _inputs(snapshot, text, raw, rows_T: int, seed: int = 0):
    x = realistic_mlp_input(snapshot, text, raw, ROWS * 32).reshape(1, 1, ROWS * 32, -1)
    if seed:
        x = x.roll(seed, dims=2)
    xs = torch.cat([x[:, :, 32 * r : 32 * r + rows_T] for r in range(ROWS)], dim=2)  # [1,1,ROWS*T,4096]
    return xs


def _shards(mesh, y):
    return [ttnn.to_torch(d).float() for d in mesh.device_tensors(y)]


def _check_vs_hf(mesh, shards, y_hf, rows_T: int, label: str):
    worst = 1.0
    for r in range(ROWS):
        for c in range(COLS):
            worst = min(worst, pcc(y_hf[:, :, r * rows_T : (r + 1) * rows_T], shards[r * COLS + c][:, :, :rows_T]))
            if c:
                assert torch.equal(shards[r * COLS + c], shards[r * COLS]), (label, r, c, "columns differ")
    logger.info(f"{label}: worst pcc vs HF {worst:.6f}")
    assert worst > 0.995, label
    return worst


def _forward_with(tt, tt_x, monkeypatch, gather: str, combine: str):
    monkeypatch.setenv(M.HYBRID_GATHER_ENV, gather)
    monkeypatch.setenv(M.HYBRID_COMBINE_ENV, combine)
    return tt(tt_x)


# ------------------------------------------------------------------------------------------------ block: collectives
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [mesh_device_params(ROWS, COLS, trace_region_size=50_000_000)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("T", [4, 2])
def test_row_glue_collectives_block(mesh_device, T):
    mesh = GLMMesh.from_device(mesh_device)
    rows = mesh.rows
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 1, rows * T, H, generator=g).bfloat16()
    tt_x = mesh.upload(x, ttnn.bfloat16, rows=2)  # [1,1,T,H] per row

    class _Owner:  # the two helpers only need .mesh
        pass

    owner = _Owner()
    owner.mesh = mesh
    padded = mesh.all_gather_rows(tt_x, dim=2)
    compact = M.TtRoutedExperts._gather_rows_compact(owner, tt_x)
    for d_p, d_c in zip(mesh.device_tensors(padded), mesh.device_tensors(compact)):
        tp, tc = ttnn.to_torch(d_p), ttnn.to_torch(d_c)
        assert tuple(tc.shape) == (1, 1, rows * T, H) and torch.equal(tp, x) and torch.equal(tc, x)
    assert compact.layout == ttnn.TILE_LAYOUT and compact.dtype == ttnn.bfloat16
    logger.info(f"T={T}: compact gather bit-identical to the padded gather on all {rows * COLS} chips")
    ttnn.deallocate(padded)
    ttnn.deallocate(compact)
    # the row sum: rows*T partial rows per chip, summed over the rows
    parts = torch.randn(rows, 1, 1, rows * T, H, generator=g).bfloat16()
    tt_p = mesh.upload(parts.reshape(rows, 1, rows * T, H), ttnn.bfloat16, rows=0)
    ref = parts.float().sum(0)[0, 0]  # [rows*T, H]
    total = mesh.all_reduce_rows_rs_ag(tt_p)
    ulp = _ulp(ref)
    for d in mesh.device_tensors(total):
        t = ttnn.to_torch(d).float()[0, 0]
        assert float((t - ref).abs().max()) <= 8 * ulp and pcc(t, ref) > 0.9999
    ttnn.deallocate(total)
    # timings
    _timed(mesh_device, lambda: mesh.all_gather_rows(tt_x, dim=2), f"T={T} padded all_gather_rows (composite)")
    _timed(mesh_device, lambda: M.TtRoutedExperts._gather_rows_compact(owner, tt_x), f"T={T} compact gather")
    _timed(mesh_device, lambda: mesh.reduce_scatter_rows(tt_p, dim=2), f"T={T} padded reduce_scatter_rows (composite)")
    _timed(mesh_device, lambda: mesh.all_reduce_rows_rs_ag(tt_p), f"T={T} all_reduce_rows_rs_ag")
    _timed(mesh_device, lambda: mesh.all_reduce_rows(tt_p), f"T={T} all_reduce_rows (ttnn.all_reduce composite)")


# ------------------------------------------------------------------------------------------------ block: the MoE layer
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("device_params", [mesh_device_params(ROWS, COLS, trace_region_size=50_000_000)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("rows_T", [4, 2])
def test_hybrid_gather_compact_block(mesh_device, weights, rows_T, monkeypatch):
    tt, mesh, hf, snapshot, text, raw = _moe_layer(mesh_device, weights)
    xs = _inputs(snapshot, text, raw, rows_T)
    with torch.no_grad():
        y_hf = hf(xs.reshape(1, ROWS * rows_T, -1)).reshape(1, 1, ROWS * rows_T, -1)
    tt_x = mesh.upload(xs.bfloat16(), ttnn.bfloat16, rows=2)
    marks = [l1_high_water(mesh_device)]
    y0 = _forward_with(tt, tt_x, monkeypatch, "padded", "rs")
    ref = _shards(mesh, y0)
    _check_vs_hf(mesh, ref, y_hf, rows_T, f"T={rows_T} padded gather")
    ttnn.deallocate(y0)
    marks.append(l1_high_water(mesh_device))
    y1 = _forward_with(tt, tt_x, monkeypatch, "compact", "rs")
    got = _shards(mesh, y1)
    _check_vs_hf(mesh, got, y_hf, rows_T, f"T={rows_T} compact gather")
    ttnn.deallocate(y1)
    marks.append(l1_high_water(mesh_device))
    for i, (a, b) in enumerate(zip(ref, got)):
        p = pcc(a, b)
        logger.info(f"T={rows_T} chip {i}: compact vs padded pcc {p:.8f} bit-equal {torch.equal(a, b)}")
        assert torch.equal(a, b), f"chip {i}: the compact gather must feed the expert op the same tiles"
    assert marks[2] == marks[1], f"L1 leak {marks}"
    assert not l1_resident(tt, mesh)


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("device_params", [mesh_device_params(ROWS, COLS, trace_region_size=50_000_000)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("rows_T", [4, 2])
@pytest.mark.parametrize("combine", ["allreduce", "allreduce_ccl"])
def test_hybrid_combine_allreduce_block(mesh_device, weights, rows_T, combine, monkeypatch):
    tt, mesh, hf, snapshot, text, raw = _moe_layer(mesh_device, weights)
    xs = _inputs(snapshot, text, raw, rows_T)
    with torch.no_grad():
        y_hf = hf(xs.reshape(1, ROWS * rows_T, -1)).reshape(1, 1, ROWS * rows_T, -1)
    tt_x = mesh.upload(xs.bfloat16(), ttnn.bfloat16, rows=2)
    y0 = _forward_with(tt, tt_x, monkeypatch, "padded", "rs")
    ref = _shards(mesh, y0)
    _check_vs_hf(mesh, ref, y_hf, rows_T, f"T={rows_T} rs combine")
    ttnn.deallocate(y0)
    marks = [l1_high_water(mesh_device)]
    y1 = _forward_with(tt, tt_x, monkeypatch, "padded", combine)
    got = _shards(mesh, y1)
    _check_vs_hf(mesh, got, y_hf, rows_T, f"T={rows_T} {combine} combine")
    ttnn.deallocate(y1)
    marks.append(l1_high_water(mesh_device))
    y2 = _forward_with(tt, tt_x, monkeypatch, "padded", combine)
    ttnn.deallocate(y2)
    marks.append(l1_high_water(mesh_device))
    for i, (a, b) in enumerate(zip(ref, got)):
        a, b = a[:, :, :rows_T], b[:, :, :rows_T]
        p, ulps = pcc(a, b), float((a - b).abs().max()) / _ulp(a)
        logger.info(
            f"T={rows_T} chip {i}: {combine} vs rs pcc {p:.7f}, max {ulps:.1f} bf16 ulps, bit-equal {torch.equal(a, b)}"
        )
        assert p >= 0.9999 and ulps <= 8, (i, p, ulps)
    assert marks[2] == marks[1], f"L1 leak {marks}"
    assert not l1_resident(tt, mesh)


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("device_params", [mesh_device_params(ROWS, COLS, trace_region_size=80_000_000)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
def test_hybrid_glue_traced_block(mesh_device, weights, monkeypatch):
    rows_T = 4
    tt, mesh, hf, snapshot, text, raw = _moe_layer(mesh_device, weights)
    monkeypatch.setenv(M.HYBRID_GATHER_ENV, "compact")
    monkeypatch.setenv(M.HYBRID_COMBINE_ENV, "allreduce")
    xs = _inputs(snapshot, text, raw, rows_T)
    tt_x = mesh.upload(xs.bfloat16(), ttnn.bfloat16, rows=2)  # persistent input
    y = tt(tt_x)  # eager warm-up: compiles the programs, builds the selector constant
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(y)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    y_tr = tt(tt_x)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for rep in range(3):
        xs_new = _inputs(snapshot, text, raw, rows_T, seed=7 * (rep + 1))
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                xs_new.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh.shard_rows(2)
            ),
            tt_x,
        )
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        got = _shards(mesh, y_tr)
        y_e = tt(tt_x)
        ref = _shards(mesh, y_e)
        ttnn.deallocate(y_e)
        with torch.no_grad():
            y_hf = hf(xs_new.reshape(1, ROWS * rows_T, -1)).reshape(1, 1, ROWS * rows_T, -1)
        _check_vs_hf(mesh, got, y_hf, rows_T, f"replay {rep}")
        for i, (a, b) in enumerate(zip(ref, got)):
            assert torch.equal(a, b), f"replay {rep} chip {i}: the traced graph differs from the eager one"
    ttnn.release_trace(mesh_device, tid)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [mesh_device_params(ROWS, COLS, trace_region_size=80_000_000)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
def test_hybrid_glue_perf_block(mesh_device, weights, monkeypatch):
    rows_T = 4
    tt, mesh, hf, snapshot, text, raw = _moe_layer(mesh_device, weights)
    xs = _inputs(snapshot, text, raw, rows_T)
    tt_x = mesh.upload(xs.bfloat16(), ttnn.bfloat16, rows=2)
    table = {}
    for gather, combine in (("padded", "rs"), ("compact", "rs"), ("padded", "allreduce"), ("compact", "allreduce")):
        monkeypatch.setenv(M.HYBRID_GATHER_ENV, gather)
        monkeypatch.setenv(M.HYBRID_COMBINE_ENV, combine)
        _, dt = timed_trace(mesh_device, lambda: tt(tt_x), STEPS)
        table[(gather, combine)] = dt * 1e6
        logger.info(
            f"MoE layer {LAYER} on {ROWS}x{COLS}, {rows_T} users/row: gather={gather} combine={combine}: {dt * 1e6:.1f} us"
        )
    base = table[("padded", "rs")]
    logger.info(
        f"deltas vs (padded, rs) {base:.0f} us: compact {table[('compact', 'rs')] - base:+.0f}, "
        f"allreduce {table[('padded', 'allreduce')] - base:+.0f}, both {table[('compact', 'allreduce')] - base:+.0f}"
    )


# ------------------------------------------------------------------------------------------------ B8a dense skip
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [mesh_device_params(ROWS, COLS, trace_region_size=60_000_000)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
def test_dsa_dense_skip_block(mesh_device, monkeypatch):
    """The block decoder test on the DSA layer with the dense-skip graph (every slot at positions 0..S+N: dense)."""
    from models.demos.glm53_flash.tests import test_mesh_decoder as TMD

    monkeypatch.setenv("GLM53_POOL_GATHER", "dense_skip")
    TMD.test_mesh_decode_vs_hf(mesh_device, DSA_LAYER)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 64 * 1024 * 1024, "l1_small_size": 32768}], indirect=True
)
def test_dsa_dense_skip_chip0(mesh_device, monkeypatch):
    from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
    from models.demos.glm53_flash.tt.functional_decoder import (
        FunctionalDecoder,
        device_streams_to_host,
        host_streams_to_device,
    )

    def packed(streams: torch.Tensor, n_rows: int) -> torch.Tensor:
        _, S, n, C = streams.shape
        out = torch.zeros(1, 1, n_rows, n * C, dtype=torch.float32)
        out[0, 0, :S] = streams.reshape(S, n * C).float()
        return out

    monkeypatch.setenv("GLM53_SPARSE_DECODE", "traced")
    snapshot = default_snapshot_dir()
    raw = load_layer_raw(snapshot, DSA_LAYER, shard_headers(snapshot))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    pre = torch.load(R2040 / f"layer{DSA_LAYER}_prefill.pt")
    S0 = pre["x_in"].shape[1]
    n_dense = 11  # decode steps at positions 2040..2050: every one in the dense range
    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=DSA_LAYER, mesh_device=mesh_device, max_batch=1, max_seq_len=S0 + 64
    )
    attn = dec.layer.attn
    y = dec.prefill_forward(host_streams_to_device(pre["x_in"], mesh_device), user=0, seq_len=S0)
    assert pcc(pre["x_out"], device_streams_to_host(y, 4, S0)) > 0.995
    ttnn.deallocate(y)
    steps = [torch.load(R2040 / f"layer{DSA_LAYER}_decode{s}.pt") for s in range(n_dense + 1)]
    assert steps[0]["position"] == S0 and steps[n_dense]["position"] == 2051
    streams_dev = ttnn.from_torch(
        packed(steps[0]["x_in"], 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )
    dp = dec.make_decode_positions([steps[0]["position"]])
    assert dp.sparse.all_dense and dp.sparse.regime == "dense"

    def load_step(s):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(packed(steps[s]["x_in"], 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), streams_dev
        )
        dec.update_decode_positions(dp, [steps[s]["position"]])

    # step 0: the gather graph (reference) and the dense-skip graph on the same token (idempotent state writes)
    attn.pool_gather_mode = "gather"
    y_g = dec.decode_forward(streams_dev, positions=dp)
    out_g = device_streams_to_host(y_g, 4, 1)
    ttnn.deallocate(y_g)
    assert dp.sparse.graph_regime == "sparse"
    attn.pool_gather_mode = "dense_skip"
    y_d = dec.decode_forward(streams_dev, positions=dp)
    out_d = device_streams_to_host(y_d, 4, 1)
    ttnn.deallocate(y_d)
    assert dp.sparse.graph_regime == "dense"
    p = pcc(out_g, out_d)
    logger.info(
        f"step 0 pos {steps[0]['position']}: dense-skip vs gather pcc {p:.8f} bit-equal {torch.equal(out_g, out_d)}"
    )
    assert p > 0.9999
    assert pcc(steps[0]["x_out"], out_d) > 0.995
    # the trace with the knob: captured at step 1, replayed through position 2050
    load_step(1)
    trace_id, out_dev, warm = dec.capture_decode_trace(streams_dev, dp)
    assert pcc(steps[1]["x_out"], device_streams_to_host(warm, 4, 1)) > 0.995
    worst = 1.0
    for s in range(2, n_dense):
        load_step(s)
        dec.replay_decode(trace_id)
        p = pcc(steps[s]["x_out"], device_streams_to_host(out_dev, 4, 1))
        logger.info(f"  replay step {s} pos {steps[s]['position']}: pcc {p:.6f}")
        worst = min(worst, p)
    assert worst > 0.995
    # the guard: the captured dense-skip graph must not be replayed past 2050
    try:
        dec.update_decode_positions(dp, [steps[n_dense]["position"]])
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "leaving the dense regime under a captured dense-skip graph must be refused"
    ttnn.release_trace(mesh_device, trace_id)
    dp.sparse.clear_graph_regime()
    load_step(n_dense)  # 2051: the eager sparse step after the reset
    assert dp.sparse.regime == "sparse"
    y_s = dec.decode_forward(streams_dev, positions=dp)
    p = pcc(steps[n_dense]["x_out"], device_streams_to_host(y_s, 4, 1))
    logger.info(f"eager step {n_dense} pos 2051 (sparse regime): pcc {p:.6f}")
    assert p > 0.995 and dp.sparse.graph_regime == "sparse"
    ttnn.deallocate(y_s)
