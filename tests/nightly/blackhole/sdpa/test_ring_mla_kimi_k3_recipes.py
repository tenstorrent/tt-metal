# Kimi K3 ring_mla on the QB 4-device ring: LEGACY vs ring_mla(precision=STANDARD) vs ring_mla(precision=FAST).
# Drives the PR's own run_ring_joint_sdpa_chunked (ring_mla, kimi_k3, q32/k640) on the final chunk, so shapes,
# sharding, chunked cache and realtime-profiler timing are the perf check's. Patches:
#   * fa_rand: Q and the latent rounded to BF16 (the caller's values);
#   * torch_sdpa_reference: FP64 causal attention on those values, rows of the measured chunk only. The device's
#     own input rounding counts (legacy/STANDARD: the cache's plain BFP8 cast; FAST: prepare_sdpa_input);
#   * compute_pcc_rmse: also records relative L2;
#   * ttnn.transformer.ring_mla: for a recipe, drops compute_kernel_config, passes precision=, and for FAST runs
#     prepare_sdpa_input on Q (7 bits) and the BF16 latent (5 bits, BFP8), and casts the gather buffer to BFP8.
import csv
import hashlib
import math
import os
import traceback
from dataclasses import replace
from unittest import mock

import pytest
import torch
import ttnn

from tests.nightly.blackhole.sdpa import test_ring_joint_sdpa as T

REF_CACHE = os.environ.get("KIMI_REF_CACHE", "generated/sdpa_recipe_sweep/ref_cache")
OUT = os.path.join("generated/sdpa_recipe_sweep", os.environ.get("KIMI_OUT", "kimi_k3_recipes.csv"))
MODEL = T.RING_MLA_CHUNKED_MODEL_CONFIGS["kimi_k3"]
MC = T.MESH_CONFIG
CHUNK = T.CHUNKED_PREFILL_CHUNK_SIZE  # 2560 on QB (640 rows per device)
QC, KC = int(os.environ.get("KIMI_QC", 32)), int(os.environ.get("KIMI_KC", 640))
REF_UTIL = 71.0  # QB perf-check reference (kimi_k3, q32/k640, 28160 tokens)

VARIANTS = {
    "legacy": None,
    "standard": ttnn.SDPAPrecision.STANDARD,
    "balanced": ttnn.SDPAPrecision.BALANCED,
    "accurate": ttnn.SDPAPrecision.ACCURATE,
    "fast": ttnn.SDPAPrecision.FAST,
}
# Exploration: the same recipe with the approximate exp (the harness passes exp_approx_mode=False).
# Exploration: K chunk / K slot overrides for the fp32-state recipes.
EXPLORE = {
    "standard_k512_slots2": (ttnn.SDPAPrecision.STANDARD, 512, "2"),
    "standard_k512_slots3": (ttnn.SDPAPrecision.STANDARD, 512, "3"),
    "legacy_k512_slots2": (None, 512, "2"),
    "legacy_k640_slots2": (None, 640, "2"),
    "standard_nomul": (ttnn.SDPAPrecision.STANDARD, 640, None),
    "standard_qk32": (ttnn.SDPAPrecision.STANDARD, 640, None),
    "standard_nocopy": (ttnn.SDPAPrecision.STANDARD, 640, None),
}
APPROX = {"standard_approx": ttnn.SDPAPrecision.STANDARD, "fast_approx": ttnn.SDPAPrecision.FAST, "legacy_approx": None}
SEL = os.environ.get("KIMI_VARIANTS", ",".join(VARIANTS)).split(",")
CONTEXTS = [int(x) for x in os.environ.get("KIMI_CHUNKS", "11,41").split(",")]  # 28160, 104960 tokens
OFFSETS = [float(x) for x in os.environ.get("KIMI_KV_MEAN_OFFSET", "0,1").split(",")]

RECORDS = []
_orig_pcc = T.compute_pcc_rmse
_orig_fa_rand = T.fa_rand
_orig_ring_mla = ttnn.transformer.ring_mla


def patches(final_rows, precision, approx=False):
    s, e = final_rows

    def fa_rand(*shape):
        return _orig_fa_rand(*shape).bfloat16().float()

    def reference(q, k, v, is_causal=False, attention_sink=None):
        assert is_causal and attention_sink is None
        out = torch.zeros(q.shape[0], q.shape[1], q.shape[2], v.shape[-1])
        # The FP64 reference of the measured rows is computed once per input set and kept on disk (it dominates the
        # run time at 500k+5k); the key covers the shapes, the rows and a strided sample of the inputs.
        digest = hashlib.sha256()
        for t in (q, k, v):
            digest.update(str(tuple(t.shape)).encode())
            digest.update(t.flatten()[:: max(1, t.numel() // 65536)].contiguous().numpy().tobytes())
        digest.update(f"{s}:{e}".encode())
        cache = os.path.join(REF_CACHE, digest.hexdigest()[:32] + ".pt")
        if os.path.exists(cache):
            out[0, :, s:e] = torch.load(cache)
            return out
        kk, vv = k[0, 0, :e].double(), v[0, 0, :e].double()
        keep = torch.arange(e)[None, :] <= torch.arange(s, e)[:, None]
        for h in range(q.shape[1]):
            sc = q[0, h, s:e].double() @ kk.T / math.sqrt(q.shape[-1])
            sc.masked_fill_(~keep, -math.inf)
            out[0, h, s:e] = (torch.softmax(sc, -1) @ vv).float()
        os.makedirs(REF_CACHE, exist_ok=True)
        torch.save(out[0, :, s:e].clone(), cache)
        return out

    def pcc(expected, actual):
        p, r = _orig_pcc(expected, actual)
        e_, a = expected.double(), actual.double()
        RECORDS.append((p, r, 100 * ((a - e_).norm() / e_.norm()).item(), bool(torch.isfinite(actual).all())))
        return p, r

    def ring_mla(q, kv, *args, **kw):
        if approx:
            pc = kw["program_config"]
            kw["program_config"] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=pc.compute_with_storage_grid_size,
                q_chunk_size=pc.q_chunk_size,
                k_chunk_size=pc.k_chunk_size,
                exp_approx_mode=True,
                max_k_splits=pc.max_k_splits,
                matmul_math_fidelity=pc.matmul_math_fidelity,
                segmented_accumulation=pc.segmented_accumulation,
            )
        if precision is not None:
            kw.pop("compute_kernel_config", None)
            kw["precision"] = precision
            if precision == ttnn.SDPAPrecision.FAST:
                q = ttnn.transformer.prepare_sdpa_input(q, is_query=True)
                kv = ttnn.transformer.prepare_sdpa_input(kv, is_query=False, dtype=ttnn.bfloat8_b)
                buf = kw["persistent_output_buffer_kv"]
                if buf.dtype != ttnn.bfloat8_b:
                    kw["persistent_output_buffer_kv"] = ttnn.typecast(buf, ttnn.bfloat8_b)
        return _orig_ring_mla(q, kv, *args, **kw)

    patches_ = []
    # KIMI_WORKER_L1=<bytes>: open the mesh with a smaller worker L1, which grows the kernel config buffer by the
    # difference (LLK_ASSERT_WORKER_L1_SIZE = 1444992 gives 16 KiB more program space).
    worker_l1 = os.environ.get("KIMI_WORKER_L1")
    if worker_l1:
        _orig_open = ttnn.open_mesh_device

        def open_mesh_device(*a, **kw):
            kw.setdefault("worker_l1_size", int(worker_l1))
            return _orig_open(*a, **kw)

        patches_.append(mock.patch.object(ttnn, "open_mesh_device", open_mesh_device))
    return patches_ + [
        mock.patch.object(T, "fa_rand", fa_rand),
        mock.patch.object(T, "torch_sdpa_reference", reference),
        mock.patch.object(T, "compute_pcc_rmse", pcc),
        mock.patch.object(ttnn.transformer, "ring_mla", ring_mla),
    ]


def profile_sdpa_ns(mesh_device, fn):
    """The perf check's timing, but the longest program rather than the first (FAST runs prepare_sdpa_input and
    a typecast before the SDPA). Duration = max over devices."""
    _, records = T.profile_realtime_program(mesh_device, fn, collect_all=True)
    by_id = {}
    for r in records:
        by_id[r["runtime_id"]] = max(by_id.get(r["runtime_id"], 0), r["duration_ns"])
    return int(max(by_id.values()))


def write(row):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(
                [
                    "variant",
                    "total_seq",
                    "kv_mean_offset",
                    "pcc",
                    "rmse",
                    "rel_l2_pct",
                    "duration_ms",
                    "util_pct",
                    "ref_util_pct",
                    "status",
                ]
            )
        w.writerow(row)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("variant", SEL)
@pytest.mark.parametrize("offset", OFFSETS)
@pytest.mark.parametrize("n_chunks", CONTEXTS)
def test_kimi_k3_recipe(n_chunks, offset, variant):
    approx = variant in APPROX
    precision = APPROX[variant] if approx else VARIANTS.get(variant)
    kc = KC
    explore_env = {}
    if variant in EXPLORE:
        precision, kc, slots = EXPLORE[variant]
        explore_env = {"TT_SDPA_KSLOTS_EXPERIMENT": slots} if slots else {}
        if variant == "standard_nomul":
            explore_env["SDPA_FP32_EXP_NOMUL"] = "1"
        if variant == "standard_qk32":
            explore_env["TT_SDPA_FP32_QK"] = "1"
        if variant == "standard_nocopy":
            explore_env["SDPA_FP32_EXP_NOCOPY"] = "1"
    # FAST rounds the BF16 latent itself (prepare_sdpa_input), so its harness cache stays BF16.
    model = replace(MODEL, kv_dtype=ttnn.bfloat16) if precision == ttnn.SDPAPrecision.FAST else MODEL
    total = CHUNK * n_chunks
    final = n_chunks - 1
    common = dict(
        chunk_size=CHUNK, total_seq=total, qk_configs=[(QC, kc)], persistent_buffer_mode="reuse_max", use_ring_mla=True
    )
    status = "ok"
    p = r = rel = dur = util = ""
    ps = patches((final * CHUNK, total), precision, approx) + [mock.patch.dict(os.environ, explore_env)]
    for x in ps:
        x.start()
    try:
        RECORDS.clear()
        # RING_SDPA_COMPUTE_ONLY outputs are garbage by design: skip the reference and only profile.
        if os.environ.get("RING_SDPA_COMPUTE_ONLY") != "1":
            with mock.patch.dict(os.environ, {T.CHUNKED_PREFILL_CHUNK_ID_ENV: str(final)}):
                T.run_ring_joint_sdpa_chunked(MC, model, pcc_threshold=0.0, kv_mean_offset=offset, **common)
            p, r, rel, finite = RECORDS[-1]
            status = "ok" if finite else "non-finite"
        if offset == OFFSETS[0]:  # perf does not depend on the values
            runtime = T.open_ring_joint_sdpa_runtime(MC)
            try:
                with mock.patch.dict(os.environ, {T.CHUNKED_PREFILL_CHUNK_ID_ENV: str(final)}):
                    dur = profile_sdpa_ns(
                        runtime.mesh_device,
                        lambda: T.run_ring_joint_sdpa_chunked(MC, model, do_check=False, runtime=runtime, **common),
                    )
            finally:
                T.close_ring_joint_sdpa_runtime(runtime)
            util, _ = T.compute_chunked_prefill_perf_check_utilization(MC, model, CHUNK, final, dur, MC.sdpa_cores)
            dur = dur / 1e6
    except BaseException as e:
        traceback.print_exc()
        status = "error: " + str(e).splitlines()[0][:200]
    finally:
        for x in reversed(ps):
            x.stop()
    write([variant, total, offset, p, r, rel, dur, util, REF_UTIL if n_chunks == 11 else "", status])
    print(variant, total, offset, rel, dur, util, status)
    assert status == "ok", status
