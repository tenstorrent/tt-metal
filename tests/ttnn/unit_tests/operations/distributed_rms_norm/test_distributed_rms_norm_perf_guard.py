# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 config-spanning perf guard (measurement driver, not a gate): same-session A/B of the Phase 0 local
schedule knobs against the current defaults, one representative per regime (D1 / D3 / D1-HS / D3-WS) x γ (TILE /
ROW_MAJOR / none) x the Refinement 1 fallback rungs (forced, as in test_distributed_rms_norm_l1_fallbacks.py), plus the
decode and 2048x1280 perf cases. 1 x 4 ring, Ring, 2 links, bf16, HiFi4 + fp32 acc.

    scripts/run_safe_pytest.sh --no-precompile --run-all \
        tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm_perf_guard.py -s

Prints, per config and knob set: traced slowest / fastest chip (op) and traced slowest chip with the exchange stubbed.
"Phase 0" knobs: DRMS_COMBINE_LAG=1 DRMS_LOOKAHEAD=1 DRMS_NOC_SWAP=0 (the reader prologue position is not a knob).
Env: DRMS_GUARD_CASES (comma-separated case ids), DRMS_PT_UNTRACED / DRMS_PT_TRACED as the traced driver.
"""

import importlib.util as _ilu
import os
import pathlib as _pl
from dataclasses import replace

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest
import torch
import ttnn

import ttnn.operations.distributed_rms_norm.distributed_rms_norm_program_descriptor as pd
from eval.golden_tests.distributed_rms_norm.conftest import production_router_config
from eval.sharding import auto_shard_config
from ttnn.operations.distributed_rms_norm import distributed_rms_norm, plan_summary

_spec = _ilu.spec_from_file_location(
    "_drms_traced", _pl.Path(__file__).with_name("test_distributed_rms_norm_perf_traced.py")
)
_tr = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tr)
_spec = _ilu.spec_from_file_location("_drms_accept", _pl.Path(__file__).with_name("test_distributed_rms_norm.py"))
_acc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_acc)

ML = ttnn.TensorMemoryLayout
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
PHASE0 = {"DRMS_COMBINE_LAG": "1", "DRMS_LOOKAHEAD": "1", "DRMS_NOC_SWAP": "0"}
# DRMS_GUARD_BASE ("K=V,K=V") replaces the A side of the A/B (default: the Phase 0 knobs)
if os.environ.get("DRMS_GUARD_BASE"):
    PHASE0 = dict(kv.split("=") for kv in os.environ["DRMS_GUARD_BASE"].split(","))

# id, (M, W), memory layout, γ layout (None: no γ), forced ladder option (None: the plan's own)
CASES = [
    ("D3-focus", (640, 1792), ML.INTERLEAVED, TILE, None),
    ("D3-rmg", (640, 1792), ML.INTERLEAVED, RM, None),
    ("D3-nog", (640, 1792), ML.INTERLEAVED, None, None),
    ("D3-decode", (32, 1792), ML.INTERLEAVED, TILE, None),
    ("D3-2048", (2048, 1280), ML.INTERLEAVED, TILE, None),
    ("D1", (256, 96), ML.INTERLEAVED, TILE, None),
    ("D1-HS", (1024, 512), ML.HEIGHT_SHARDED, TILE, None),
    ("D3-WS", (64, 2048), ML.WIDTH_SHARDED, TILE, None),
    ("D3-WS-nog", (64, 2048), ML.WIDTH_SHARDED, None, None),
    ("rung-passes", (640, 1792), ML.INTERLEAVED, TILE, dict(PW=1, x_depth=2)),
    ("rung-chunk-stream", (640, 1792), ML.INTERLEAVED, TILE, dict(gamma_mode=pd.GAMMA_CHUNK, gamma_stream=True)),
    ("rung-reread", (640, 1792), ML.INTERLEAVED, TILE, dict(gamma_mode=pd.GAMMA_CHUNK, sq_rows=1, reread=True)),
]
_sel = [c for c in os.environ.get("DRMS_GUARD_CASES", "").split(",") if c]
if _sel:
    CASES = [c for c in CASES if c[0] in _sel]

pytestmark = pytest.mark.skipif(os.environ.get("UP_FRONT_COLLECT") == "1", reason="run with --no-precompile")


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "fabric_router_config": production_router_config(),
            "trace_region_size": 1 << 23,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["mesh1x4"], indirect=True)
def test_perf_guard(mesh_device, monkeypatch):
    mesh = mesh_device
    G = 4
    rows = []
    for cid, (M, W), ml, gl, opt in CASES:
        shape = (1, 1, M, W)
        x = _acc._stacked((1, G), shape, torch.bfloat16)
        mc = ttnn.DRAM_MEMORY_CONFIG
        if ml != ML.INTERLEAVED:
            mc = auto_shard_config(list(shape), ml, layout=TILE, dtype=ttnn.bfloat16, device=mesh)
        tx = _acc._to_mesh(x, mesh, dtype=ttnn.bfloat16, layout=TILE, memory_config=mc)
        gamma = tg = None
        if gl is not None:
            gamma = _acc._stacked((1, G), (1, 1, 1, W), torch.bfloat16)
            tg = _acc._gamma_to_mesh(gamma, mesh, dtype=ttnn.bfloat16, layout=gl)
        expected = _acc._reference(x, gamma, 1, 1e-5)
        out_mc = mc if ml != ML.INTERLEAVED else None
        run = lambda: distributed_rms_norm(
            tx,
            cluster_axis=1,
            gamma=tg,
            epsilon=1e-5,
            topology=ttnn.Topology.Ring,
            num_links=2,
            memory_config=out_mc,
        )
        with monkeypatch.context() as mp:
            if opt is not None:
                real = pd.first_fit
                mp.setattr(pd, "first_fit", lambda base, sz, budget: (0, replace(real(base, sz, budget)[1], **opt)))
            print(f"\nPLAN {cid} {plan_summary(tx, cluster_axis=1, gamma=tg, num_links=2)}")
            # correctness of the current defaults
            out = run()
            pcc = min(
                _acc._pcc(ttnn.to_torch(d).float(), expected[0, c]) for c, d in enumerate(ttnn.get_device_tensors(out))
            )
            assert pcc > 0.999, (cid, pcc)
            res = {}
            for knobs in ("phase0", "new"):
                env = PHASE0 if knobs == "phase0" else {}
                for k, v in env.items():
                    os.environ[k] = v
                try:
                    op = _tr._measure(mesh, run)
                    os.environ["DRMS_ABLATE"] = "exchange"
                    stub = _tr._measure(mesh, run)
                finally:
                    os.environ.pop("DRMS_ABLATE", None)
                    for k in env:
                        os.environ.pop(k, None)
                res[knobs] = (op[2], op[3], stub[2])
            o, n = res["phase0"], res["new"]
            line = (
                f"GUARD {cid:18s} op slowest {o[0]:6.2f} -> {n[0]:6.2f} | op fastest {o[1]:6.2f} -> {n[1]:6.2f} | "
                f"stub slowest {o[2]:6.2f} -> {n[2]:6.2f} us | pcc {pcc:.5f}"
            )
            print(line)
            rows.append(line)
    print("\n" + "\n".join(rows))
