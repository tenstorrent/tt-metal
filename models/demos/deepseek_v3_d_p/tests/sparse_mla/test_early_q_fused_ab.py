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
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import model_fabric_payload
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
        "fused_impl": mla_module.MLA_A2AMM_IMPL,
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


# ============================================================================
# Block A/B: only the early-Q reshard (q_b_proj output -> all-to-all -> head split -> wkv_b1 -> RoPE + concat), for
# meshes where the whole layer cannot run with a TP ring, e.g. one 2x2 half of a LoudBox opened as a 1x4 ring (the
# sparse layer's KVPE read-back is a full-mesh snake that needs both mesh dims > 1). Same built layer and weights.
# ============================================================================
from models.demos.deepseek_v3_d_p.tests.measure.timing import time_untraced  # noqa: E402
from models.demos.deepseek_v3_d_p.tests.measure.trace import TracedStep, time_traced  # noqa: E402


def _block_input(case, mesh_device, chunk):
    """A q_b_proj-shaped input: [1, 1, chunk, H*qk] sharded sequence over SP and head columns over TP."""
    import torch

    mla = case.mla
    qk = mla.qk_nope_head_dim + mla.qk_rope_head_dim
    torch.manual_seed(0)
    q = torch.randn(1, 1, chunk, mla.num_heads * qk, dtype=torch.float32).to(torch.bfloat16)
    dims = (2, 3) if BOX.sp_axis == 0 else (3, 2)
    return ttnn.from_torch(
        q,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
    )


@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "mesh_device, device_params, variant, case_name",
    [p for p in gate_params() if not GATE_CASES.get(p.values[3], (None, True))[1]],
    indirect=["mesh_device", "device_params", "variant"],
)
@pytest.mark.parametrize("scenario", ["warm"])
def test_early_q_block_ab(request, mesh_device, device_params, variant, case_name, config_only, scenario):
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
    mla = case.mla
    assert mla._early_q_reshard, f"{case_name}: this layer does not take the early-Q reshard path"
    sp = tuple(mesh_device.shape)[BOX.sp_axis]
    seq_len_local = chunk // sp
    q_in = _block_input(case, mesh_device, chunk)
    start = cache

    def step():
        # The stem consumes its input: both paths pay the same clone.
        return mla._q_stem_seq_sharded(ttnn.clone(q_in), case.rope, start, seq_len_local, None)

    samples = {(p, m): [] for p in PATHS for m in MODES}
    outputs, errors = {}, {}
    try:
        for r in range(ROUNDS):
            for path in PATHS:
                _set_path(path)
                for mode in MODES:
                    if (path, mode) in errors:
                        continue
                    try:
                        if mode == "untraced":
                            s, out = time_untraced(step, lambda: ttnn.synchronize_device(mesh_device))
                            outputs.setdefault((path, mode), _device_tensors(out))
                            ttnn.deallocate(out)
                        else:
                            with TracedStep(mesh_device, step) as traced:
                                traced.capture()
                                s = time_traced(traced)
                                outputs.setdefault((path, mode), _device_tensors(traced.output))
                        samples[(path, mode)] += list(s)
                    except Exception as e:  # noqa: BLE001
                        errors[(path, mode)] = f"{type(e).__name__}: {e}"[:600]
                        print(f"EARLY_Q_BLOCK {case_name} {path} {mode} round {r} FAILED: {errors[(path, mode)]}")
    finally:
        _set_path("chain")

    pcc = {}
    for mode in MODES:
        a, b = outputs.get(("chain", mode)), outputs.get(("fused", mode))
        if a is not None and b is not None:
            pcc[mode] = min(comp_pcc(x, y, OUT_PCC)[1] for x, y in zip(a, b))
    result = {
        "kind": "block",
        "case": case_name,
        "box": BOX.name,
        "mesh": f"{BOX.mesh[0]}x{BOX.mesh[1]}",
        "scenario": scenario,
        "chunk_tokens": chunk,
        "fabric": device_params["fabric_config"].name,
        "fabric_payload_bytes": model_fabric_payload(variant.name),
        "visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "fused_impl": mla_module.MLA_A2AMM_IMPL,
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
            result[f"delta_us_{mode}"] = 1000.0 * (f - c)
            result[f"delta_pct_{mode}"] = 100.0 * (f - c) / c
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "results.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("EARLY_Q_BLOCK_RESULT " + json.dumps({k: result[k] for k in result if k != "samples_ms"}))
    assert not errors, f"paths failed: {errors}"
    for mode, value in pcc.items():
        assert value >= OUT_PCC, f"{mode}: fused block output differs from the chain's (min per-device PCC {value:.6f})"


# ============================================================================
# Host time per op (untraced): how long each call of the chain and of the fused path takes to return on the host,
# device synchronized before and after every call so no queue back-pressure is counted. Median and min over N calls
# after warm-up (program cache hits).
# ============================================================================
HOST_N = int(os.environ.get("EARLY_Q_HOST_N", 50))
HOST_WARM = 5


def _host_us(mesh_device, fn, setup=lambda: (), n=HOST_N):
    import statistics

    ts = []
    for i in range(HOST_WARM + n):
        args = setup()
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        out = fn(*args)
        t1 = time.perf_counter()
        ttnn.synchronize_device(mesh_device)
        if i >= HOST_WARM:
            ts.append((t1 - t0) * 1e6)
        for t in out if isinstance(out, (tuple, list)) else (out,):
            ttnn.deallocate(t)
        for t in args:
            ttnn.deallocate(t)
    return {"median_us": statistics.median(ts), "min_us": min(ts)}


@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "mesh_device, device_params, variant, case_name",
    [p for p in gate_params() if not GATE_CASES.get(p.values[3], (None, True))[1]],
    indirect=["mesh_device", "device_params", "variant"],
)
@pytest.mark.parametrize("scenario", ["warm"])
def test_early_q_host_ops(request, mesh_device, device_params, variant, case_name, config_only, scenario):
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
    mla = case.mla
    assert mla._early_q_reshard
    sp = tuple(mesh_device.shape)[BOX.sp_axis]
    seq_len_local = chunk // sp
    q_in = _block_input(case, mesh_device, chunk)
    a2a_kw = dict(num_links=mla.ccl_num_links, memory_config=ttnn.DRAM_MEMORY_CONFIG, cluster_axis=mla.tp_axis)
    lin_kw = mla._early_q_wkv_b1_kwargs(seq_len_local)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    a2amm_kw = dict(
        cluster_axis=mla.tp_axis,
        topology=ttnn.Topology.Ring,
        num_links=mla.ccl_num_links,
        memory_config=lin_kw["memory_config"],
        compute_kernel_config=cfg,
    )
    split_kw = dict(split_head_dim=mla.qk_nope_head_dim, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    clone_q = lambda: (ttnn.clone(q_in),)

    # Persistent intermediates for the per-op timings.
    q_seq = ttnn.experimental.all_to_all_async_generic(ttnn.clone(q_in), in_dim=3, out_dim=2, **a2a_kw)
    nope_seq, rope_seq_c = ttnn.experimental.nlp_create_q_heads_split(q_seq, num_heads=mla.num_heads, **split_kw)
    nope_l, rope_l = ttnn.experimental.nlp_create_q_heads_split(
        q_in, num_heads=mla.num_heads // mla.tp_factor, **split_kw
    )
    scratch = ttnn.all_to_all_matmul_scratch_tensor(nope_l, mla.wkv_b1_weight, **a2amm_kw)

    def stem(path):
        def fn(q):
            _set_path(path)
            try:
                return mla._q_stem_seq_sharded(q, case.rope, cache, seq_len_local, None)
            finally:
                _set_path("chain")

        return fn

    rows = {
        "chain/all_to_all_async_generic(q)": _host_us(
            mesh_device, lambda: ttnn.experimental.all_to_all_async_generic(q_in, in_dim=3, out_dim=2, **a2a_kw)
        ),
        "chain/nlp_create_q_heads_split(H)": _host_us(
            mesh_device, lambda: ttnn.experimental.nlp_create_q_heads_split(q_seq, num_heads=mla.num_heads, **split_kw)
        ),
        "chain/linear(wkv_b1)": _host_us(mesh_device, lambda: ttnn.linear(nope_seq, mla.wkv_b1_weight, **lin_kw)),
        "fused/nlp_create_q_heads_split(H/tp)": _host_us(
            mesh_device,
            lambda: ttnn.experimental.nlp_create_q_heads_split(
                q_in, num_heads=mla.num_heads // mla.tp_factor, **split_kw
            ),
        ),
        "fused/all_to_all_matmul": _host_us(
            mesh_device, lambda: ttnn.all_to_all_matmul(nope_l, mla.wkv_b1_weight, **a2amm_kw)
        ),
        "fused/all_to_all_matmul(scratch given)": _host_us(
            mesh_device,
            lambda: ttnn.all_to_all_matmul(nope_l, mla.wkv_b1_weight, scratch_tensor=scratch, **a2amm_kw),
        ),
        "fused/all_to_all_matmul_scratch_tensor only": _host_us(
            mesh_device, lambda: ttnn.all_to_all_matmul_scratch_tensor(nope_l, mla.wkv_b1_weight, **a2amm_kw)
        ),
        "fused/all_to_all_async_generic(rope)": _host_us(
            mesh_device, lambda: ttnn.experimental.all_to_all_async_generic(rope_l, in_dim=1, out_dim=2, **a2a_kw)
        ),
        "ref/clone(q)": _host_us(mesh_device, lambda: ttnn.clone(q_in)),
        "stem/chain": _host_us(mesh_device, stem("chain"), setup=clone_q),
        "stem/fused": _host_us(mesh_device, stem("fused"), setup=clone_q),
    }
    result = {
        "kind": "host_ops",
        "case": case_name,
        "box": BOX.name,
        "mesh": f"{BOX.mesh[0]}x{BOX.mesh[1]}",
        "fused_impl": "native",
        "n": HOST_N,
        "host_us": rows,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "results.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print(f"EARLY_Q_HOST {case_name} {BOX.name}")
    for k, v in rows.items():
        print(f"EARLY_Q_HOST   {k:48s} median {v['median_us']:9.1f} us   min {v['min_us']:9.1f} us")
