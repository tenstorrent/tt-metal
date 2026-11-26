# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A/B of one sparse MLA layer: the early-Q reshard as the TTNN chain (all_to_all_async_generic -> head split ->
wkv_b1 linear) vs the fused all_to_all_matmul (local head split -> all_to_all_matmul on nope + all-to-all on rope).

Same built layer, same inputs, same process: the switch is mla.MLA_EARLY_Q_FUSED_A2AMM, read at call time. Each path
is timed as test_sparse_mla_perf times it (untraced: fastest of 10 after 10 warm-up; traced: one timed replay), in
two interleaved rounds (chain, fused, chain, fused); the fastest sample per (path, mode) is reported. The fused
layer output must match the chain's (per-device PCC).

Boxes: perf_cases.BOX (Galaxy 8x4; with MLA_PERF_RING4=1 and one 2x2 half visible, a LoudBox 1x4 ring).
Results: one JSON line per case under generated/early_q_fused_ab/.
"""

import json
import os
import time
from pathlib import Path

import pytest

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tests.sparse_mla.perf_cases import (
    BOX,
    GATE_CASES,
    SCENARIO_CACHE_TOKENS,
    build_sparse_perf_case,
    gate_params,
)
from models.demos.deepseek_v3_d_p.tests.sparse_mla.test_sparse_mla_perf import (
    _device_tensors,
    _time_traced,
    _time_untraced,
)
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import model_fabric_payload
from models.demos.deepseek_v3_d_p.tt.mla import mla as mla_module
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

PATHS = ("chain", "fused")
ROUNDS = int(os.environ.get("EARLY_Q_AB_ROUNDS", 2))
MODES = [m for m in os.environ.get("EARLY_Q_AB_MODES", "untraced,traced").split(",") if m]
OUT_PCC = 0.999
OUT_DIR = Path(os.environ.get("TT_METAL_HOME", ".")) / "generated" / "early_q_fused_ab"


def _set_path(path):
    mla_module.MLA_EARLY_Q_FUSED_A2AMM = path == "fused"


@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "mesh_device, device_params, variant, case_name",
    [p for p in gate_params() if not GATE_CASES.get(p.values[3], (None, True))[1]],
    indirect=["mesh_device", "device_params", "variant"],
)
@pytest.mark.parametrize("scenario", ["warm"])
def test_early_q_fused_ab(request, mesh_device, device_params, variant, case_name, config_only, scenario):
    chunk = BOX.chunk
    cache = BOX.cache_tokens(SCENARIO_CACHE_TOKENS[scenario])
    case = build_sparse_perf_case(
        mesh_device,
        variant,
        config_only,
        cache=cache,
        chunk=chunk,
        topology=per_axis_topology(device_params["fabric_config"]),
        sp_axis=BOX.sp_axis,
    )
    assert case.mla._early_q_reshard, f"{case_name}: this layer does not take the early-Q reshard path"
    start = cache
    samples = {(p, m): [] for p in PATHS for m in MODES}
    outputs = {}
    errors = {}
    try:
        for r in range(ROUNDS):
            for path in PATHS:
                _set_path(path)
                for mode in MODES:
                    if (path, mode) in errors:
                        continue
                    try:
                        timer = _time_untraced if mode == "untraced" else _time_traced
                        s, out = timer(mesh_device, case, start)
                        samples[(path, mode)] += list(s)
                        outputs.setdefault((path, mode), out)
                    except Exception as e:  # noqa: BLE001 - report which path/mode failed, keep the rest
                        errors[(path, mode)] = f"{type(e).__name__}: {e}"[:600]
                        print(f"EARLY_Q_AB {case_name} {path} {mode} round {r} FAILED: {errors[(path, mode)]}")
    finally:
        _set_path("chain")

    pcc = {}
    for mode in MODES:
        a, b = outputs.get(("chain", mode)), outputs.get(("fused", mode))
        if a is not None and b is not None:
            pcc[mode] = min(comp_pcc(x, y, OUT_PCC)[1] for x, y in zip(a, b))

    result = {
        "case": case_name,
        "box": BOX.name,
        "mesh": f"{BOX.mesh[0]}x{BOX.mesh[1]}",
        "scenario": scenario,
        "chunk_tokens": chunk,
        "cache_tokens": cache,
        "fabric": device_params["fabric_config"].name,
        "fabric_payload_bytes": model_fabric_payload(variant.name),
        "visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "rounds": ROUNDS,
        "best_ms": {f"{p}/{m}": min(v) for (p, m), v in samples.items() if v},
        "samples_ms": {f"{p}/{m}": v for (p, m), v in samples.items()},
        "output_pcc_fused_vs_chain": pcc,
        "errors": {f"{p}/{m}": e for (p, m), e in errors.items()},
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for mode in MODES:
        c, f = result["best_ms"].get(f"chain/{mode}"), result["best_ms"].get(f"fused/{mode}")
        if c and f:
            result[f"delta_ms_{mode}"] = f - c
            result[f"delta_pct_{mode}"] = 100.0 * (f - c) / c
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "results.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("EARLY_Q_AB_RESULT " + json.dumps({k: result[k] for k in result if k != "samples_ms"}))
    assert not errors, f"paths failed: {errors}"
    for mode, value in pcc.items():
        assert value >= OUT_PCC, f"{mode}: fused layer output differs from the chain's (min per-device PCC {value:.6f})"
