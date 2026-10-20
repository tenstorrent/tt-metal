# QuietBox (4-device ring) check of PR #59522 SDPA precision recipes on the ring SDPA perf configs.
#
# Reuses tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py (the PR's version) for inputs, sharding,
# references and realtime-profiler timing. Patches:
#   * fa_rand            -> rounded to BF16, so the reference sees the same inputs as the device
#   * compute_pcc_rmse   -> also records relative L2 (the harness reference is fp32 on those inputs)
#   * ring op entry      -> injects precision=<recipe>, drops compute_kernel_config / exp_approx_mode
#                           (recipes reject both), and runs prepare_sdpa_input for FAST
# Every case appends one row to ring_causal.csv; a rejected call records the exception's first line.
import csv
import os
import traceback
from dataclasses import replace
from unittest import mock

import pytest
import torch
import ttnn

from tests.nightly.blackhole.sdpa import test_ring_joint_sdpa as T

OUT = os.environ.get("SDPA_RECIPE_SWEEP_OUT", "generated/sdpa_recipe_sweep/ring_causal.csv")
VARIANTS = {
    "legacy": None,
    # No precision, fp32_dest_acc_en=True: the stack (#60197) routes these calls to the recipes by itself.
    "fp32dest": None,
    "standard": (ttnn.SDPAPrecision.STANDARD, ttnn.bfloat16),
    "balanced": (ttnn.SDPAPrecision.BALANCED, ttnn.bfloat16),
    "accurate": (ttnn.SDPAPrecision.ACCURATE, ttnn.bfloat16),
    "fast_bf16": (ttnn.SDPAPrecision.FAST, ttnn.bfloat16),
    "fast_bfp8": (ttnn.SDPAPrecision.FAST, ttnn.bfloat8_b),
    # Op-selected blocking (q/k chunk 0), for shapes where the harness's explicit chunks do not fit.
    "standard_auto": (ttnn.SDPAPrecision.STANDARD, ttnn.bfloat16),
    "fast_bfp8_auto": (ttnn.SDPAPrecision.FAST, ttnn.bfloat8_b),
}
RECORDS = []
_orig_ring = ttnn.transformer.ring_joint_scaled_dot_product_attention
_orig_ring_mla = ttnn.transformer.ring_mla
_orig_pcc = T.compute_pcc_rmse
_orig_fa_rand = T.fa_rand


def _recipe_kwargs(variant, q, k, v, kw):
    precision, kv_dtype = VARIANTS[variant]
    kw.pop("compute_kernel_config", None)
    pc = kw["program_config"]
    kw["program_config"] = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=pc.compute_with_storage_grid_size,
        q_chunk_size=0 if variant.endswith("_auto") else pc.q_chunk_size,
        k_chunk_size=0 if variant.endswith("_auto") else pc.k_chunk_size,
    )
    if precision == ttnn.SDPAPrecision.FAST:
        q = ttnn.transformer.prepare_sdpa_input(q, is_query=True)
        k = ttnn.transformer.prepare_sdpa_input(k, is_query=False, dtype=kv_dtype)
        if v is not None:
            v = ttnn.transformer.prepare_sdpa_input(v, is_query=False, dtype=kv_dtype)
        # The harness allocates the gather buffers in the model's (BF16, see fast_model) dtype; the
        # op needs them in the prepared K/V dtype.
        for name in ("persistent_output_buffer_k", "persistent_output_buffer_v", "persistent_output_buffer_kv"):
            buf = kw.get(name)
            if buf is not None and buf.dtype != kv_dtype:
                kw[name] = ttnn.typecast(buf, kv_dtype)
    kw["precision"] = precision
    return q, k, v, kw


def patched(variant):
    def ring(q, k, v, *args, **kw):
        if VARIANTS[variant] is not None:
            q, k, v, kw = _recipe_kwargs(variant, q, k, v, kw)
        return _orig_ring(q, k, v, *args, **kw)

    def ring_mla(q, kv, *args, **kw):
        if VARIANTS[variant] is not None:
            q, kv, _, kw = _recipe_kwargs(variant, q, kv, None, kw)
        return _orig_ring_mla(q, kv, *args, **kw)

    def pcc(expected, actual):
        p, r = _orig_pcc(expected, actual)
        e, a = expected.double(), actual.double()
        rel = 100 * ((a - e).norm() / e.norm()).item()
        RECORDS.append((p, r, rel, bool(torch.isfinite(actual).all())))
        return p, r

    def fa_rand(*shape):
        return _orig_fa_rand(*shape).bfloat16().float()

    return [
        mock.patch.object(ttnn.transformer, "ring_joint_scaled_dot_product_attention", ring),
        mock.patch.object(ttnn.transformer, "ring_mla", ring_mla),
        mock.patch.object(T, "compute_pcc_rmse", pcc),
        mock.patch.object(T, "fa_rand", fa_rand),
    ]


def write(row):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(
                [
                    "config",
                    "variant",
                    "phase",
                    "status",
                    "pcc",
                    "rmse",
                    "rel_l2_pct",
                    "duration_ms",
                    "util_pct",
                    "ref_util_pct",
                    "detail",
                ]
            )
        w.writerow(row)


def first_line(e):
    lines = [l for l in str(e).splitlines() if l.strip()]
    msg = next(
        (l for l in lines if "TT_FATAL" not in l and "TT_THROW" not in l and not l.startswith("info:")),
        lines[0] if lines else repr(e),
    )
    return f"{type(e).__name__}: {msg.strip()[:300]}"


def run(config, variant, phase, fn, ref_util=""):
    """fn returns (duration_ns or None, util or None). Accuracy rows come from RECORDS."""
    RECORDS.clear()
    patches = patched(variant)
    for p in patches:
        p.start()
    try:
        dur, util = fn()
        recs = RECORDS[:] if phase == "accuracy" else []
        if phase == "accuracy" and recs:
            # worst (largest rel L2) over the outputs the harness checked
            p, r, rel, finite = max(recs, key=lambda x: x[2])
            write(
                [
                    config,
                    variant,
                    phase,
                    "ok" if finite else "non-finite",
                    p,
                    r,
                    rel,
                    "",
                    "",
                    ref_util,
                    f"{len(recs)} outputs checked",
                ]
            )
        else:
            write(
                [
                    config,
                    variant,
                    phase,
                    "ok",
                    "",
                    "",
                    "",
                    dur / 1e6 if dur else "",
                    util if util is not None else "",
                    ref_util,
                    "",
                ]
            )
    except BaseException as e:  # pytest.fail / skip raise BaseException subclasses
        recs = RECORDS[:]
        detail = first_line(e)
        if recs:  # the op ran but the harness threshold assert fired: keep the numbers
            p, r, rel, finite = max(recs, key=lambda x: x[2])
            write([config, variant, phase, "ran, harness assert failed", p, r, rel, "", "", ref_util, detail])
        else:
            write(
                [
                    config,
                    variant,
                    phase,
                    "rejected" if "Skipped" not in type(e).__name__ else "skipped",
                    "",
                    "",
                    "",
                    "",
                    "",
                    ref_util,
                    detail,
                ]
            )
        traceback.print_exc()
    finally:
        for p in reversed(patches):
            p.stop()


def profile_sdpa_ns(mesh_device, fn):
    """Like T.profile_ring_joint_runtime_duration_ns, but picks the longest program instead of the first
    (FAST runs prepare_sdpa_input programs before SDPA). Duration = max over devices of that program."""
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active")
    _, records = T.profile_realtime_program(mesh_device, fn, collect_all=True)
    by_id = {}
    for r in records:
        by_id[r["runtime_id"]] = max(by_id.get(r["runtime_id"], 0), r["duration_ns"])
    return int(max(by_id.values()))


MC = T.MESH_CONFIG
# CHUNKED_FRESH_KV=1: profile chunked calls with a fresh right-sized K/V (as the ring_mla perf check does) instead
# of the reused max buffer + kv_actual_isl, which ring recipes reject (KV-pad rotation).
FRESH_KV = os.environ.get("CHUNKED_FRESH_KV") == "1"
CHUNK = T.CHUNKED_PREFILL_CHUNK_SIZE
FINAL_CHUNK = T.CHUNKED_PREFILL_TOTAL_SEQ // CHUNK - 1


# ---- non-chunked ring joint configs (QB perf check list + the minimax3 GQA causal perf shape) ----
def model_cfg(name):
    """'base+mod+mod': mods strip one feature at a time to expose each recipe blocker in turn."""
    base, *mods = name.split("+")
    m = replace(T.RING_JOINT_PERF_MODEL_CONFIGS[base], name=name)
    for mod in mods:
        if mod == "bf16kv":
            m = replace(m, kv_dtype=ttnn.bfloat16)
        elif mod == "nobal":
            m = replace(m, is_balanced=False)
        elif mod == "noncausal":
            m = replace(m, is_causal=False, is_balanced=False)
    return m


RING_CASES = [
    # (config, q_chunk, k_chunk, QB reference util or "")
    ("wan2_2_1xGLX", 288, 512, 68.5),
    ("mla_100k", 160, 320, 62.5),
    ("mla_100k+bf16kv", 160, 320, ""),
    ("mla_100k+bf16kv+nobal", 160, 320, ""),
    ("mla_100k+bf16kv+noncausal", 160, 320, ""),
    ("minimax3_gqa_causal_perf", 128, 512, ""),
    ("minimax3_gqa_causal_perf+bf16kv", 128, 512, ""),
    ("minimax3_gqa_causal_perf+bf16kv+nobal", 128, 512, ""),
    ("minimax3_gqa_causal_perf+bf16kv+noncausal", 128, 512, ""),
]


def fast_model(model, variant):
    # prepare_sdpa_input needs original BF16 K/V; it does the BFP8/BFP4 rounding itself.
    return replace(model, kv_dtype=ttnn.bfloat16) if variant.startswith("fast") else model


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("case", RING_CASES, ids=[c[0] for c in RING_CASES])
def test_ring_joint(case, variant):
    name, qc, kc, ref = case
    model = fast_model(model_cfg(name), variant)

    def acc():
        T.run_ring_joint_sdpa_model_configs(
            MC, model, [(qc, kc)], pcc_threshold=0.0, rmse_threshold=None, fp32_dest_acc_en=variant == "fp32dest"
        )
        return None, None

    run(name, variant, "accuracy", acc, ref)

    def perf():
        runtime = T.open_ring_joint_sdpa_runtime(MC, fp32_dest_acc_en=variant == "fp32dest")
        try:
            dur = profile_sdpa_ns(
                runtime.mesh_device,
                lambda: T.run_ring_joint_sdpa_model_configs(MC, model, [(qc, kc)], do_check=False, runtime=runtime),
            )
        finally:
            T.close_ring_joint_sdpa_runtime(runtime)
        util = T.compute_ring_joint_utilization(
            model.seq_len,
            model.seq_len * MC.sp_size,
            model.d_q,
            model.d_v,
            model.nhq,
            dur,
            MC.sdpa_cores,
            model.is_causal,
        )
        return dur, util

    run(name, variant, "perf", perf, ref)


# ---- chunked-prefill configs: kimi50k / kimi_k3 (ring_mla), minimax3_55k (ring joint GQA) ----
CHUNKED_CASES = [
    # (config, q_chunk, k_chunk, QB reference util, use_ring_mla)
    ("kimi50k", 32, 640, 69.54, True),
    ("kimi_k3", 32, 640, 71.0, True),
    ("minimax3_55k", 128, 512, 47.0, False),
    ("minimax3_55k+bf16kv", 128, 512, "", False),
]


def chunked_model(name, use_ring_mla):
    base, *mods = name.split("+")
    m = T.RING_MLA_CHUNKED_MODEL_CONFIGS[base] if use_ring_mla else T.MINIMAX3_GQA_CHUNKED_MODEL_CONFIGS[base]
    if "bf16kv" in mods:
        m = replace(m, name=name, kv_dtype=ttnn.bfloat16)
    return m


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("case", CHUNKED_CASES, ids=[c[0] for c in CHUNKED_CASES])
def test_chunked(case, variant):
    name, qc, kc, ref, use_ring_mla = case
    model = fast_model(chunked_model(name, use_ring_mla), variant)

    def acc():
        # Final chunk only: the largest K/V prefix, the same chunk the perf check profiles.
        with mock.patch.dict(os.environ, {T.CHUNKED_PREFILL_CHUNK_ID_ENV: str(FINAL_CHUNK)}):
            T.run_ring_joint_sdpa_chunked(
                MC,
                model,
                chunk_size=CHUNK,
                qk_configs=[(qc, kc)],
                persistent_buffer_mode="reuse_max",
                use_ring_mla=use_ring_mla,
                pcc_threshold=0.0,
                fp32_dest_acc_en=variant == "fp32dest",
            )
        return None, None

    run(name, variant, "accuracy", acc, ref)

    def perf():
        runtime = T.open_ring_joint_sdpa_runtime(MC, fp32_dest_acc_en=variant == "fp32dest")
        try:
            with mock.patch.dict(os.environ, {T.CHUNKED_PREFILL_CHUNK_ID_ENV: str(FINAL_CHUNK)}):
                dur = profile_sdpa_ns(
                    runtime.mesh_device,
                    lambda: T.run_ring_joint_sdpa_chunked(
                        MC,
                        model,
                        chunk_size=CHUNK,
                        qk_configs=[(qc, kc)],
                        persistent_buffer_mode="reuse_max",
                        use_ring_mla=use_ring_mla,
                        do_check=False,
                        reuse_kv_buffer=not use_ring_mla and not FRESH_KV,
                        runtime=runtime,
                    ),
                )
        finally:
            T.close_ring_joint_sdpa_runtime(runtime)
        util, _ = T.compute_chunked_prefill_perf_check_utilization(MC, model, CHUNK, FINAL_CHUNK, dur, MC.sdpa_cores)
        return dur, util

    run(name + ("+freshkv" if FRESH_KV else ""), variant, "perf", perf, ref)
