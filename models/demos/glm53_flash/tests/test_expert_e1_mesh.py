# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device A/B of decode roofline action E1 (2026-10-26, _agents/expert_e1/STATUS.md). NOT RUN when written (the chips
belong to the validation queue): every case names its expected outcome. importorskip-safe (collects without a device).

Chip 0 (``TT_VISIBLE_DEVICES=<chip>``, mesh (1,1)), the 8x4 slice shape (72 experts x [4096, 512] gate|up, [256, 4096]
down, random bfp8 weights), the chip playing mesh column 1 (base 72) so the fold's window is a real one:
* ``test_e1_block_variants_chip0[T]`` (T = 4 and 32 real rows of the 32-row block): every knob combination of the
  block op -- today, scan_words, routing_fold, fold + words, fused, fused + words, fold + words + fused -- against
  today's program: BIT-IDENTICAL ``out`` (the fold reads the same bf16 bits the selector matmul produced, the word
  scan finds the same active set, the fused program runs the same compute kernels on the same tiles), PCC vs the
  torch golden > 0.9999, traced us per variant (the E1 ladder: expected ~351 -> ~338 (fold, the two host ops gone)
  -> ~329 (words) -> ~250 (fused) at 28 active; at T = 4 the fixed part dominates: ~157 -> ~93), and the device
  op count per call (today: selector matmul + x copy + 2 generic ops; fold: 2; fused: 1).
* ``test_e1_dispatch_32row_chip0``: the 256-token dispatch form (E1 (c)) with the measured plan (12 groups, 8 down
  readers), the roofline plan and the units plan: bit-identical outputs, traced us (the measured point: 779 us at 58
  active; the real-routing point ~31 active), vs the 8-block stream form.

(2,4) block (TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 TT_MESH_GRAPH_DESC_PATH=$W/tt_metal/fabric/
mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto MESH_DEVICE=P150x8 GLM53_MESH_SHAPE=2,4):
* ``test_e1_moe_layer_block[knobs-rows_T]``: ``tests/test_mesh_moe.py::test_mesh_moe_vs_hf`` (the whole hybrid MoE
  layer 4 vs HF on the block, 2 and 32 users per row) with GLM53_STREAM_FOLD=1 [+ GLM53_STREAM_SCAN=words]
  [+ GLM53_STREAM_FUSED=1] [+ GLM53_HYBRID_GATHER=compact]; at 32 users per row GLM53_EXPERTS_32ROW=dispatch /
  units (the 1024-wide block slices make the units split real). Expected: the same PCC vs HF as the default run
  (bit-identical expert outputs; the compact gather changes nothing numerically).
* ``test_e1_decoder_block``: ``tests/test_mesh_decoder.py::test_mesh_decode_vs_hf`` layer 4 with the three knobs.

8x4 (the model's env: GLM53_MESH_SHAPE=8,4 GLM53_MODEL_LAYERS=0-44 GLM53_MODEL_B=<users per row>):
* ``test_e1_e2e_8x4[config]``: ``tests/test_mesh_model.py::test_mesh_model_chain_and_perf`` at 4 users per row with
  the defaults (reference 69.1 ms per step, 2026-10-26), with fold + words (expected -1.3 ms), with fold + words +
  fused (expected -4.2 ms: ~65 ms), each + GLM53_HYBRID_GATHER=compact (the x copy folded); and at 32 users per
  row with GLM53_DENSE_EXPERTS=1 (the dense pass, reference 112-115 ms) vs GLM53_EXPERTS_32ROW=dispatch (expected
  ~-17 ms: the 1198 -> 779 us per layer of the chip measurement) and GLM53_EXPERTS_32ROW=dispatch
  GLM53_EXPERT_PLAN=roofline.

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_expert_e1_mesh.py -k chip0 -s
    TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 TT_MESH_GRAPH_DESC_PATH=... MESH_DEVICE=P150x8 GLM53_MESH_SHAPE=2,4 \\
        pytest models/demos/glm53_flash/tests/test_expert_e1_mesh.py -k block -s
    GLM53_MESH_SHAPE=8,4 GLM53_MODEL_LAYERS=0-44 pytest models/demos/glm53_flash/tests/test_expert_e1_mesh.py -k e2e -s
"""

from __future__ import annotations

import importlib
import os

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.custom_ops.expert_stream import plan as P  # noqa: E402
from models.demos.glm53_flash.custom_ops.expert_stream.op import (  # noqa: E402
    ExpertDispatch,
    ExpertStream,
    bfp8_round,
    down_stream_layout,
    expert_dispatch_golden,
    expert_stream_golden,
    fold_base_tensor,
    gate_up_stream_layout,
    upload_stream_weights,
)
from models.demos.glm53_flash.tests.test_moe import weights as layer_weights  # noqa: E402,F401 (fixture)
from models.demos.glm53_flash.tests.test_moe_block import count_ops, timed_trace  # noqa: E402
from models.demos.glm53_flash.tests.utils import pcc  # noqa: E402
from models.demos.glm53_flash.tt.mesh import mesh_device_params  # noqa: E402

E_TOTAL, E, K, I, H = 288, 72, 4096, 256, 4096
COL_BASE = 72  # the chip plays mesh column 1
STEPS = int(os.environ.get("GLM53_E1_STEPS", "20"))
BLOCK = (2, 4)
MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "1,4").split(","))  # the e2e's mesh (8,4)
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "256"))

VARIANTS = {
    "today": dict(),
    "words": dict(scan_words=True),
    "fold": dict(routing_fold=True),
    "fold+words": dict(routing_fold=True, scan_words=True),
    "fused": dict(fused=True),
    "fused+words": dict(fused=True, scan_words=True),
    "fold+words+fused": dict(routing_fold=True, scan_words=True, fused=True),
}


def full_routing(T: int, picks: int, seed: int) -> torch.Tensor:
    """[32, 288] bf16: rows < T pick `picks` global experts (the chip's 72 see ~picks / 4 of them), pad rows zero."""
    g = torch.Generator().manual_seed(seed)
    r = torch.zeros(32, E_TOTAL)
    for t in range(T):
        ids = torch.randperm(E_TOTAL, generator=g)[:picks]
        r[t, ids] = torch.rand(picks, generator=g) + 0.2
    return r.bfloat16()


@pytest.fixture(scope="module")
def weights():
    g = torch.Generator().manual_seed(0)
    w13 = (torch.randn(E, K, 2 * I, generator=g) * 0.02).bfloat16()
    w2 = (torch.randn(E, I, H, generator=g) * 0.02).bfloat16()
    return w13, w2


# ---------------------------------------------------------------------------------------------- chip 0: the block op
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("T", [4, 32])
def test_e1_block_variants_chip0(mesh_device, weights, T):
    w13_host, w2_host = weights
    w13 = upload_stream_weights(mesh_device, gate_up_stream_layout(w13_host.float()))
    w2 = upload_stream_weights(mesh_device, down_stream_layout(w2_host.float()))
    w13_q, w2_q = bfp8_round(w13_host.float()), bfp8_round(w2_host.float())
    x = torch.zeros(32, K).bfloat16()
    x[:T] = torch.randn(T, K).bfloat16()
    r_full = full_routing(T, 8 if T == 32 else 32, seed=T)  # T = 4: 32 picks per token -> ~8 local active
    r_sel = r_full[:, COL_BASE : COL_BASE + E].contiguous()  # today's selector matmul result (exact, host)
    active = [e for e in range(E) if int(r_sel[:, e].float().abs().sum() > 0)]
    _, y_ref = expert_stream_golden(x[:T], r_sel[:T], w13_q, w2_q)
    xd = ttnn.from_torch(x.reshape(1, 1, 32, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    xd = ttnn.to_memory_config(xd, ttnn.L1_MEMORY_CONFIG)  # the model's placement (GLM53_STREAM_X_L1=1)
    rd_sel = ttnn.from_torch(
        r_sel.reshape(1, 1, 32, E), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )
    rd_full = ttnn.from_torch(
        r_full.reshape(1, 1, 32, E_TOTAL), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )
    base = fold_base_tensor(COL_BASE, mesh_device)
    table = []
    y_today = None
    for name, flags in VARIANTS.items():
        op = ExpertStream(mesh_device, E, K, I, H, **flags)
        fold = flags.get("routing_fold", False)
        if flags.get("fused"):
            logger.info(
                f"{name}: fused placement gate|up {len(op.gate_up.assignments)} cores ({op.gate_up.groups}/bank), "
                f"down {len(op.down.assignments)} cores ({op.down.readers}/bank), {op.fused_op.n_sems} semaphores"
            )

        def run():
            act, out = op.alloc(mesh_device, act_l1=True)
            op.forward(xd, rd_full if fold else rd_sel, w13, w2, act, out, base=base if fold else None)
            ttnn.deallocate(act)
            return out

        top, dev = count_ops(mesh_device, run)
        y, dt = timed_trace(mesh_device, run, STEPS)
        y = y.reshape(32, H)[:T]
        p = pcc(y_ref, y)
        if y_today is None:
            y_today = y
        same = torch.equal(y, y_today)
        table.append((name, dt * 1e6, p, same, len(dev)))
        logger.info(
            f"T={T} ({len(active)} active): {name:18s} {dt * 1e6:8.1f} us  pcc {p:.6f}  bit-identical {same}  "
            f"device ops {len(dev)}"
        )
        assert p > 0.9999, (name, p)
        assert same, f"{name}: not bit-identical to today's program"
        if flags.get("fused"):
            assert len(dev) == 1, dev  # one generic op
        else:
            assert len(dev) == 2, dev
    logger.info("E1 ladder (us): " + ", ".join(f"{n} {t:.1f}" for n, t, *_ in table))
    today = table[0][1]
    fused = [t for n, t, *_ in table if n == "fold+words+fused"][0]
    logger.info(f"T={T}: fold+words+fused / today = {fused / today:.3f} (model: {P.e1_expected_us(len(active))})")


# ---------------------------------------------------------------------------------------------- chip 0: the 256-token form
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_e1_dispatch_32row_chip0(mesh_device, weights):
    """E1 (c): the dispatch plans of GLM53_EXPERTS_32ROW at T = 256 (the 8x4's 32 users per row): measured plan
    (12 groups / 8 readers), the roofline plan and units -- bit-identical, traced us; the 8-block stream form for
    reference (2410 us measured 2026-10-20 at 31 active; dispatch 779 at 58 active)."""
    w13_host, w2_host = weights
    w13 = upload_stream_weights(mesh_device, gate_up_stream_layout(w13_host.float()))
    w2 = upload_stream_weights(mesh_device, down_stream_layout(w2_host.float()))
    w13_q, w2_q = bfp8_round(w13_host.float()), bfp8_round(w2_host.float())
    T = 256
    g = torch.Generator().manual_seed(256)
    r = torch.zeros(T, E)
    for t in range(T):
        ids = torch.randperm(E, generator=g)[:2]  # ~2 of a token's 8 picks land on the chip: ~31 of 72 active
        r[t, ids] = torch.rand(2, generator=g) + 0.2
    r = r.bfloat16()
    x = torch.randn(T, K).bfloat16()
    _, y_ref, passes = expert_dispatch_golden(x, r, w13_q, w2_q)
    active = len({e for e, _ in passes})
    xd = ttnn.from_torch(x.reshape(1, 1, T, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    rd = ttnn.from_torch(r.reshape(1, 1, T, E), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    grid = mesh_device.compute_with_storage_grid_size()
    plans = {
        "measured": P.plan_for_32row((grid.x, grid.y), E, I, H, T=T),
        "roofline": P.plan_for_32row((grid.x, grid.y), E, I, H, T=T, roofline=True),
        "units": P.plan_for_32row((grid.x, grid.y), E, I, H, T=T, units=True),
    }
    y0 = None
    for name, plan in plans.items():
        kw = plan.op_kwargs()
        op = ExpertDispatch(mesh_device, E, K, I, H, T, **kw)

        def run():
            x_rm, counts = ExpertDispatch.prepare(xd, rd)
            act, table, out = op.alloc(mesh_device, act_l1=True)
            op.forward(x_rm, rd, counts, w13, w2, act, table, out)
            for t_ in (x_rm, counts, act, table):
                ttnn.deallocate(t_)
            return out

        y, dt = timed_trace(mesh_device, run, STEPS)
        y = y.reshape(T, H)
        p = pcc(y_ref, y)
        if y0 is None:
            y0 = y
        logger.info(
            f"T=256 ({active} active): {name:9s} {dt * 1e6:8.1f} us  pcc {p:.6f}  bit-identical {torch.equal(y, y0)}  {kw}"
        )
        assert p > 0.9999, (name, p)
        assert torch.equal(y, y0), name


# ---------------------------------------------------------------------------------------------- (2,4) block
BLOCK_KNOBS = {
    "fold": {"GLM53_STREAM_FOLD": "1"},
    "fold+words": {"GLM53_STREAM_FOLD": "1", "GLM53_STREAM_SCAN": "words"},
    "fold+words+fused": {"GLM53_STREAM_FOLD": "1", "GLM53_STREAM_SCAN": "words", "GLM53_STREAM_FUSED": "1"},
    "fold+words+fused+compact": {
        "GLM53_STREAM_FOLD": "1",
        "GLM53_STREAM_SCAN": "words",
        "GLM53_STREAM_FUSED": "1",
        "GLM53_HYBRID_GATHER": "compact",
    },
    "32row-dispatch": {"GLM53_EXPERTS_32ROW": "dispatch"},
    "32row-units": {"GLM53_EXPERTS_32ROW": "units"},
}


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [mesh_device_params(*BLOCK)], indirect=True)
@pytest.mark.parametrize("mesh_device", [BLOCK], indirect=True)
@pytest.mark.parametrize("knobs", list(BLOCK_KNOBS))
@pytest.mark.parametrize("rows_T", [2, 32])
def test_e1_moe_layer_block(mesh_device, layer_weights, knobs, rows_T, monkeypatch):
    """The hybrid MoE layer on the block with the knobs: tests/test_mesh_moe.py::test_mesh_moe_vs_hf (vs HF; the
    real layer-4 weights fixture of tests/test_moe.py, imported like test_mesh_moe does)."""
    if knobs.startswith("32row") and rows_T != 32:
        pytest.skip("the 32-row form is the 32 users/row case")
    for k, v in BLOCK_KNOBS[knobs].items():
        monkeypatch.setenv(k, v)
    tm = importlib.import_module("models.demos.glm53_flash.tests.test_mesh_moe")
    tm.test_mesh_moe_vs_hf(mesh_device, layer_weights, rows_T)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [mesh_device_params(*BLOCK, trace_region_size=60_000_000)], indirect=True)
@pytest.mark.parametrize("mesh_device", [BLOCK], indirect=True)
@pytest.mark.parametrize("knobs", ["fold+words", "fold+words+fused"])
def test_e1_decoder_block(mesh_device, knobs, monkeypatch):
    """Decoder layer 4 on the block vs HF with the knobs: tests/test_mesh_decoder.py::test_mesh_decode_vs_hf."""
    for k, v in BLOCK_KNOBS[knobs].items():
        monkeypatch.setenv(k, v)
    td = importlib.import_module("models.demos.glm53_flash.tests.test_mesh_decoder")
    td.test_mesh_decode_vs_hf(mesh_device, 4)


# ---------------------------------------------------------------------------------------------- 8x4 e2e
E2E = {
    "b4-defaults": ({"GLM53_MODEL_B": "4"}, 69.1),
    "b4-fold-words": ({"GLM53_MODEL_B": "4", "GLM53_STREAM_FOLD": "1", "GLM53_STREAM_SCAN": "words"}, 67.8),
    "b4-fold-words-fused": (
        {"GLM53_MODEL_B": "4", "GLM53_STREAM_FOLD": "1", "GLM53_STREAM_SCAN": "words", "GLM53_STREAM_FUSED": "1"},
        64.9,
    ),
    "b4-fold-words-fused-compact": (
        {
            "GLM53_MODEL_B": "4",
            "GLM53_STREAM_FOLD": "1",
            "GLM53_STREAM_SCAN": "words",
            "GLM53_STREAM_FUSED": "1",
            "GLM53_HYBRID_GATHER": "compact",
        },
        62.0,
    ),
    "b32-dense": ({"GLM53_MODEL_B": "32", "GLM53_DENSE_EXPERTS": "1"}, 112.3),
    "b32-dispatch": ({"GLM53_MODEL_B": "32", "GLM53_EXPERTS_32ROW": "dispatch"}, 95.0),
    "b32-dispatch-roofline": (
        {"GLM53_MODEL_B": "32", "GLM53_EXPERTS_32ROW": "dispatch", "GLM53_EXPERT_PLAN": "roofline"},
        95.0,
    ),
}


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1_000_000)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("config", list(E2E))
def test_e1_e2e_8x4(mesh_device, config, monkeypatch):
    """The 45-layer e2e chain + perf on the 8x4 (tests/test_mesh_model.py::test_mesh_model_chain_and_perf) with the
    E1 configurations; the expected decode step (ms) per row is the model's. The module reads its env at import, so
    it is (re)imported under the knobs (the layers are built inside the test, after the reload)."""
    env, expected_ms = E2E[config]
    if MESH != (8, 4):
        pytest.skip("the 8x4 e2e needs GLM53_MESH_SHAPE=8,4 (and GLM53_MODEL_LAYERS=0-44)")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    tm = importlib.import_module("models.demos.glm53_flash.tests.test_mesh_model")
    tm = importlib.reload(tm)
    logger.info(f"{config}: expected decode step ~{expected_ms} ms per step ({env})")
    tm.test_mesh_model_chain_and_perf(mesh_device)
