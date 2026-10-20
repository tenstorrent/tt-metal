# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pytest plugin: run test_ring_joint_sdpa.py (or any ring SDPA test) with a precision recipe injected.

    SDPA_RING_PRECISION=STANDARD pytest -p tests.nightly.blackhole.sdpa.ring_sdpa_recipe_plugin \
        tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py

With SDPA_RING_PRECISION set (STANDARD, BALANCED, ACCURATE or FAST), every
ttnn.transformer.ring_joint_scaled_dot_product_attention / ring_mla call gets precision=<recipe>; the caller's
compute_kernel_config, exp_approx_mode and matmul_math_fidelity are dropped (recipes fix them). FAST inputs go
through prepare_sdpa_input (BFP8 K/V for FAST, BFP8 inputs are first widened to BF16), and the gather buffers
are cast to the prepared K/V dtype. SDPA_RING_FAST_KV=bfloat4_b selects BFP4 instead.

Unset, the plugin only records (the legacy baseline). Either way it appends one JSON line per test to
SDPA_RING_RESULTS (default generated/sdpa_recipe_sweep/ring_joint_<recipe|legacy>.jsonl): outcome, duration,
the error's first line, every PCC / RMSE / relative L2 the test computed, and every utilization / device
duration the perf helpers computed.
"""
import json
import os
import time

import pytest
import torch
import ttnn

PRECISION_NAME = os.environ.get("SDPA_RING_PRECISION", "").upper()
PRECISION = getattr(ttnn.SDPAPrecision, PRECISION_NAME) if PRECISION_NAME else None
FAST_KV = getattr(ttnn, os.environ.get("SDPA_RING_FAST_KV", "bfloat8_b"))
OUT = os.environ.get(
    "SDPA_RING_RESULTS", f"generated/sdpa_recipe_sweep/ring_joint_{(PRECISION_NAME or 'legacy').lower()}.jsonl"
)

_records = {"metrics": [], "perf": []}


def _strip_config(kw):
    kw.pop("compute_kernel_config", None)
    pc = kw.get("program_config")
    if pc is not None and (pc.exp_approx_mode is not None or getattr(pc, "matmul_math_fidelity", None) is not None):
        fields = dict(
            compute_with_storage_grid_size=pc.compute_with_storage_grid_size,
            q_chunk_size=pc.q_chunk_size,
            k_chunk_size=pc.k_chunk_size,
        )
        for name in ("sub_core_grids", "max_cores_per_head_batch", "max_k_splits", "segmented_accumulation"):
            if hasattr(pc, name):
                fields[name] = getattr(pc, name)
        kw["program_config"] = ttnn.SDPAProgramConfig(**{k: v for k, v in fields.items() if v is not None})


_BUFFERS = (
    "persistent_output_buffer_k",
    "persistent_output_buffer_v",
    "persistent_output_buffer_kv",
    "persistent_output_buffer_joint_k",
    "persistent_output_buffer_joint_v",
)


def _fast_kv_dtype(kw):
    # K/V land in the caller's persistent gather buffers, which tests may inspect: keep their dtype.
    for name in _BUFFERS:
        if kw.get(name) is not None:
            return kw[name].dtype
    return FAST_KV


def _fast_input(t, is_query, kv_dtype=None):
    # Sharded tensors (ND-sharded caches) keep their layout: FAST then runs on the caller's unrounded values.
    if t is None or t.is_sharded():
        return t
    if t.dtype != ttnn.bfloat16:
        t = ttnn.typecast(t, ttnn.bfloat16)
    if is_query:
        return ttnn.transformer.prepare_sdpa_input(t, is_query=True)
    return ttnn.transformer.prepare_sdpa_input(t, is_query=False, dtype=kv_dtype)


def _wrap_ring_joint(orig):
    def ring_joint(q, k, v, jq=None, jk=None, jv=None, *args, **kw):
        _strip_config(kw)
        if PRECISION == ttnn.SDPAPrecision.FAST:
            kv_dtype = _fast_kv_dtype(kw)
            q, jq = _fast_input(q, True), _fast_input(jq, True)
            k, v, jk, jv = (_fast_input(t, False, kv_dtype) for t in (k, v, jk, jv))
        kw["precision"] = PRECISION
        return orig(q, k, v, jq, jk, jv, *args, **kw)

    return ring_joint


def _wrap_ring_mla(orig):
    def ring_mla(q, kv, *args, **kw):
        _strip_config(kw)
        if PRECISION == ttnn.SDPAPrecision.FAST:
            q, kv = _fast_input(q, True), _fast_input(kv, False, _fast_kv_dtype(kw))
        kw["precision"] = PRECISION
        return orig(q, kv, *args, **kw)

    return ring_mla


def pytest_configure(config):
    if PRECISION is not None:
        ttnn.transformer.ring_joint_scaled_dot_product_attention = _wrap_ring_joint(
            ttnn.transformer.ring_joint_scaled_dot_product_attention
        )
        ttnn.transformer.ring_mla = _wrap_ring_mla(ttnn.transformer.ring_mla)


def _patch_module(module):
    if getattr(module, "_ring_recipe_plugin_patched", False):
        return
    module._ring_recipe_plugin_patched = True

    def record_metrics(fn):
        def wrapped(expected, actual, *a, **k):
            result = fn(expected, actual, *a, **k)
            try:
                e, x = expected.double().flatten(), actual.double().flatten()
                rel = 100 * ((x - e).norm() / e.norm()).item()
                ec, xc = e - e.mean(), x - x.mean()
                pcc = ((ec @ xc) / (ec.norm() * xc.norm())).item()
                _records["metrics"].append({"pcc": pcc, "rel_l2_pct": rel})
            except Exception:
                pass
            return result

        wrapped.__name__ = fn.__name__
        return wrapped

    def record_perf(fn, name):
        def wrapped(*a, **k):
            result = fn(*a, **k)
            util = result[0] if isinstance(result, tuple) else result
            _records["perf"].append({"fn": name, "util_pct": float(util)})
            return result

        return wrapped

    for name in ("compute_pcc_rmse", "comp_pcc"):
        if hasattr(module, name):
            setattr(module, name, record_metrics(getattr(module, name)))
    for name in ("compute_ring_joint_utilization", "compute_chunked_prefill_perf_check_utilization"):
        if hasattr(module, name):
            setattr(module, name, record_perf(getattr(module, name), name))
    if hasattr(module, "profile_ring_joint_runtime_duration_ns") and hasattr(module, "profile_realtime_program"):

        def profile(mesh_device, run_fn):
            # The longest program, not the first: FAST's prepare_sdpa_input programs run before SDPA.
            _, records = module.profile_realtime_program(mesh_device, run_fn, collect_all=True)
            by_id = {}
            for record in records:
                by_id.setdefault(record["runtime_id"], []).append(record)
            longest = max(by_id.values(), key=lambda rs: max(r["duration_ns"] for r in rs))
            duration = int(max(r["duration_ns"] for r in longest))
            _records["perf"].append({"fn": "duration", "duration_ms": duration / 1e6})
            return duration, longest

        module.profile_ring_joint_runtime_duration_ns = profile


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _patch_module(item.module)
    _records["metrics"].clear()
    _records["perf"].clear()
    item._ring_recipe_t0 = time.time()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        error = ""
        if report.longrepr is not None and report.outcome != "passed":
            text = str(getattr(report.longrepr, "reprcrash", None) or report.longrepr)
            error = next((l.strip() for l in text.splitlines() if l.strip()), "")[:400]
        row = {
            "nodeid": item.nodeid,
            "precision": PRECISION_NAME or "LEGACY",
            "outcome": report.outcome,
            "seconds": round(time.time() - item._ring_recipe_t0, 2),
            "error": error,
            "metrics": list(_records["metrics"]),
            "perf": list(_records["perf"]),
        }
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "a") as f:
            f.write(json.dumps(row) + "\n")
