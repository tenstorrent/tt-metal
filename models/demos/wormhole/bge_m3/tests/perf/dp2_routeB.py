# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Route B: attack the ~0.8ms/program TRACE-REPLAY DISPATCH overhead (no precision risk).

Attribution (dp2_perf_latency.py) showed: device kernels ~880ms, traced wall
~1021ms; the ~140ms gap is on-device trace-replay overhead scaling with PROGRAM
COUNT (171 programs, ~0.8ms each). Host issue is 0.01ms; both chips balanced &
concurrent; not amortizable by pipelining. Route B tries to shrink that per-
program overhead WITHOUT changing the model math (PCC-safe by construction).

Part A - LAYER-COUNT SCALING (one device build):
  Slice model.layers to N in {6,12,18,24}, capture a trace each, measure wall.
  Fit wall(N) = intercept + slope*N. slope = per-layer (kernel + dispatch ovh);
  intercept = fixed one-time overhead. Compare slope to the ~36.5ms device
  kernel/layer to isolate dispatch-ovh/layer (=> per-program). This PROVES the
  hypothesis and quantifies the exact target, no dispatch knob involved.

Part B - DISPATCH-CONFIG SWEEP (parametrized device opens):
  Re-measure full 24-layer traced wall under dispatch knobs that can change the
  per-program launch cost on a (2,1) mesh:
    * WORKER dispatch, 1 CQ, FABRIC_1D   (control = current shipping config)
    * ETH-core dispatch                  (dispatch off worker cores)
    * num_command_queues = 2             (re-confirm at current checkpoint)
  Any config whose wall < ~1000ms with unchanged math is a free win.

Run:
  source /localdev/gtobar/bge_optimization/local_env.sh
  export TT_VISIBLE_DEVICES=0 BGE_M3_DATA_PARALLEL=1
  # Part A:
  pytest .../dp2_routeB.py::test_layer_scaling -s -q
  # Part B (each param = one device open + model build):
  pytest .../dp2_routeB.py::test_dispatch_variant -s -q
"""

import time

import pytest
from loguru import logger

import ttnn
from models.demos.wormhole.bge_m3.tests.perf.dp2_perf import _to_batchsharded_tensors, prepare_inputs
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

BATCH = 12
SEQ = 8192
DEVICE_KERNEL_MS_PER_LAYER = 36.5  # from the 876ms/24-layer device profile
PROGRAMS_PER_LAYER = 7


def _build(mesh):
    args, model, _ = create_tt_model(
        mesh_device=mesh,
        max_batch_size=BATCH,
        max_seq_len=SEQ,
        dtype=ttnn.bfloat8_b,
        data_parallel=True,
        use_experimental_encoder_sdpa=True,
        encoder_sdpa_q256_vbf4=True,
        use_qkv_scatter_matmul=True,
    )
    assert model._data_parallel, "DP mode not active"
    inp = prepare_inputs(args.tokenizer, BATCH, SEQ, args.pad_token_id)
    dev = _to_batchsharded_tensors(inp, mesh, device=True)
    return args, model, dev


def _capture_and_measure(model, mesh, dev, iters=20):
    out = model.forward(**dev)
    ttnn.synchronize_device(mesh)
    ttnn.deallocate(out)
    model.capture_trace(**dev, mesh_device=mesh, cq_id=0)
    tid, tdev = model._trace_id, model._trace_device
    for _ in range(4):
        ttnn.execute_trace(tdev, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(tdev)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ttnn.execute_trace(tdev, tid, cq_id=0, blocking=True)
        ttnn.synchronize_device(tdev)
        ts.append((time.perf_counter() - t0) * 1e3)
    model.release_trace()
    ts.sort()
    return ts[0], ts[len(ts) // 2]


# ── Part A: layer-count scaling ──────────────────────────────────────────────
@pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True, ids=["dp2_n300"])
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50_000_000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_layer_scaling(mesh_device):
    args, model, dev = _build(mesh_device)
    all_layers = list(model.layers)
    Ns = [6, 12, 18, 24]
    res = {}
    for N in Ns:
        model.layers = all_layers[:N]
        mn, md = _capture_and_measure(model, mesh_device, dev)
        res[N] = (mn, md)
        logger.info(f"[scaling] N={N:2d} layers: wall min={mn:8.2f}  med={md:8.2f} ms")
    model.layers = all_layers  # restore

    # linear fit on min-wall over the BF8-dominated range (6..24)
    xs = Ns
    ys = [res[n][0] for n in Ns]
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    disp_per_layer = slope - DEVICE_KERNEL_MS_PER_LAYER
    logger.info("=" * 74)
    logger.info("  PART A — LAYER SCALING FIT")
    logger.info("=" * 74)
    logger.info(f"  wall(N) = {intercept:.2f} + {slope:.2f} * N   (ms)")
    logger.info(f"  device kernel/layer     ~= {DEVICE_KERNEL_MS_PER_LAYER:.1f} ms")
    logger.info(f"  => DISPATCH ovh / layer  ~= {disp_per_layer:.2f} ms "
                f"(~{disp_per_layer / PROGRAMS_PER_LAYER:.2f} ms/program)")
    logger.info(f"  => fixed one-time intercept ~= {intercept:.2f} ms")
    logger.info(f"  24-layer wall = {res[24][0]:.1f} ms; to hit 1000ms need -{res[24][0]-1000:.1f} ms")
    if disp_per_layer > 0:
        logger.info(f"     = fuse ~{(res[24][0]-1000)/ (disp_per_layer/PROGRAMS_PER_LAYER):.1f} programs, "
                    f"OR cut per-program ovh by {(res[24][0]-1000)/171:.2f} ms/program across all 171")
    logger.info("=" * 74)


# ── Part B: dispatch-config sweep ────────────────────────────────────────────
_VARIANTS = [
    pytest.param(
        {"trace_region_size": 50_000_000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
        id="control_worker_1cq",
    ),
    pytest.param(
        {"trace_region_size": 50_000_000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D,
         "dispatch_core_type": ttnn.DispatchCoreType.ETH},
        id="eth_dispatch_1cq",
    ),
    pytest.param(
        {"trace_region_size": 50_000_000, "num_command_queues": 2, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
        id="worker_2cq",
    ),
]


@pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True, ids=["dp2_n300"])
@pytest.mark.parametrize("device_params", _VARIANTS, indirect=True)
def test_dispatch_variant(mesh_device, device_params):
    args, model, dev = _build(mesh_device)
    mn, md = _capture_and_measure(model, mesh_device, dev, iters=25)
    emb = BATCH / (mn / 1000.0)
    logger.info("=" * 74)
    logger.info(f"  PART B — dispatch variant: wall min={mn:.2f} med={md:.2f} ms  "
                f"({emb:.2f} emb/s, {1000.0/mn:.3f} req/s)  {'<<< SUB-1000!' if mn < 1000 else ''}")
    logger.info("=" * 74)
    logger.info(f"METRIC dispatch_wall_min_ms {mn:.3f}")
