# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Performance comparison harness: TTTv1 ``TTSampling`` vs TTTv2 ``Sampling1D``.

Goal
----
Establish an apples-to-apples device-latency baseline for the two sampling
implementations so we can measure the effect of porting TTTv1-main performance
improvements into TTTv2 (``Sampling1D``).

- TTTv1 reference: ``models/common/sampling_main_ref/tt_sampling.py`` — a verbatim, isolated copy
  of **origin/main** ``TTSampling`` (the up-to-date version with the perf/logprobs features), NOT
  the current branch's older ``models/common/sampling`` copy. See that package's ``__init__`` for
  the source commit. This lets us benchmark against main without modifying branch TTTv1.
- TTTv2 target:    ``models/common/modules/sampling/sampling_1d.py``.

Both modules run the identical workload (same sharded logits, same k/p/temp, same
tt_ccl, same seeds) so the only difference measured is the module's op pipeline.

Two latencies are reported per (case, module):
- ``traced_us``: pure on-device time per call via trace capture + replay (no host
  dispatch). This is the production-relevant number (TracedLLMExecutor replays a
  trace per decode step). Headline metric.
- ``host_us``:   amortized host dispatch + device time per call, eager (no trace).
  Relevant to eager decode loops; noisier, includes Python overhead.

Results are appended to ``generated/sampling_perf/sampling_perf_compare.csv`` so a
"before" run (baseline) and an "after" run (post-port) can be diffed. Tag a run via
``SAMPLING_PERF_TAG=<label>`` (default ``"run"``).

Run
---
    SAMPLING_PERF_TAG=baseline \\
    pytest models/common/tests/modules/sampling/test_sampling_perf_compare.py -v -s

Mesh shapes are 1D only (Sampling1D constraint): (1,1), (1,2) on N150/N300; (1,8) on T3K.
This is a perf/benchmark harness, not a correctness gate — it asserts only that both
modules ran and produced valid token shapes, and reports token agreement informationally.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.sampling.sampling_1d import Sampling1D
from models.common.modules.tt_ccl import get_tt_ccl
from models.common.sampling_main_ref._utils import is_power_of_2, upper_power_of_2
from models.common.sampling_main_ref.tt_sampling import TTSampling  # verbatim origin/main copy

# ---------------------------------------------------------------------------
# Tunables (env-overridable)
# ---------------------------------------------------------------------------
N_WARMUP = int(os.environ.get("SAMPLING_PERF_WARMUP", "5"))
N_HOST_ITERS = int(os.environ.get("SAMPLING_PERF_HOST_ITERS", "50"))
N_TRACE_ITERS = int(os.environ.get("SAMPLING_PERF_TRACE_ITERS", "200"))
PERF_TAG = os.environ.get("SAMPLING_PERF_TAG", "run")
TRACE_REGION_SIZE = int(os.environ.get("SAMPLING_PERF_TRACE_REGION", str(23887872)))
RESULTS_CSV = Path("generated/sampling_perf/sampling_perf_compare.csv")

# Mesh-device params include trace_region_size so trace capture works (the plain
# tuple form used elsewhere opens with the default 0-byte trace region).
_MESH_PARAMS = [
    pytest.param({"mesh_shape": (1, 1), "trace_region_size": TRACE_REGION_SIZE}, id="1x1"),
    pytest.param({"mesh_shape": (1, 2), "trace_region_size": TRACE_REGION_SIZE}, id="1x2"),
    pytest.param({"mesh_shape": (1, 8), "trace_region_size": TRACE_REGION_SIZE}, id="1x8"),
]


# ---------------------------------------------------------------------------
# Workload cases
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    vocab_size: int
    k: int
    p: float
    temp: float
    force_argmax: bool
    label: str


_CASES = [
    # Representative decode configs. argmax fast-path + the common top-k paths.
    Case(128256, 1, 1.0, 1.0, True, "argmax-v128256"),
    Case(128256, 1, 0.0, 1.0, False, "topk1-v128256"),
    Case(128256, 10, 0.9, 1.0, False, "topk10-v128256"),
    Case(32768, 1, 0.0, 1.0, False, "topk1-v32768"),
    Case(32768, 10, 0.9, 1.0, False, "topk10-v32768"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _Args:
    """Minimal stand-in for the model-args object TTSampling reads in __init__."""

    def __init__(self, *, vocab_size, cluster_shape, force_argmax, pad=False):
        self.vocab_size = vocab_size
        self.padded_vocab_size = vocab_size  # cases are already tile/device aligned
        self.cluster_shape = cluster_shape
        self.max_batch_size = 32
        self.max_top_k = 32
        self.sampling_all_gather_axis = 0
        self.sub_core_grids = None
        self.sub_core_grid_topk = None
        self.start_core = ttnn.CoreCoord(0, 0)
        self.sampling_dp = 1
        # pad=True → TTSampling pads each per-device shard up to the next power of 2 before
        # ttnn.topk (the perf lever we want to demonstrate). Only affects the multi-device
        # branch; the 1×1 multi_step_reduction split path does NOT pad even with this flag.
        self.pad_logits_to_power_of_2 = pad
        # SAMPLING_AG_CONFIG only enables the argmax fast-path; omitted for top-k cases.
        if force_argmax:
            self.model_config = {
                "SAMPLING_AG_CONFIG": {
                    "allow_force_argmax": True,
                    "num_links": 1,
                    "topology": ttnn.Topology.Linear,
                }
            }
        else:
            self.model_config = {}


def _make_logits_tt(logits_host, mesh_device):
    """Vocab-sharded logits along the last dim for 1×N (replicated on 1×1)."""
    cluster_shape = tuple(mesh_device.shape)
    shard_dims = (None, None) if max(cluster_shape) == 1 else (None, -1)
    return ttnn.from_torch(
        logits_host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=cluster_shape),
    )


def _make_param_tensors(mesh_device, B, k, p, temp):
    cluster_shape = tuple(mesh_device.shape)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=cluster_shape)
    k_tt = ttnn.from_torch(
        torch.full((B,), k, dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mapper,
    )
    p_tt = ttnn.from_torch(
        torch.full((B,), p), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper
    )
    temp_tt = ttnn.from_torch(
        torch.full((B,), temp),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mapper,
    )
    return k_tt, p_tt, temp_tt


def _time_host_amortized(fn, mesh_device, n_warmup, n_iters):
    """Mean host dispatch + device time per call (single sync over the whole loop)."""
    for _ in range(n_warmup):
        fn()
    ttnn.synchronize_device(mesh_device)
    start = time.perf_counter()
    for _ in range(n_iters):
        fn()
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - start) / n_iters * 1e6  # microseconds


def _time_traced(fn, mesh_device, n_warmup, n_iters):
    """Pure on-device time per call via trace capture + replay. None if tracing fails."""
    try:
        for _ in range(n_warmup):
            fn()
        ttnn.synchronize_device(mesh_device)

        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        fn()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        start = time.perf_counter()
        for _ in range(n_iters):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        elapsed = (time.perf_counter() - start) / n_iters * 1e6  # microseconds

        ttnn.release_trace(mesh_device, tid)
        return elapsed
    except Exception as e:  # noqa: BLE001 — trace support varies by op; degrade gracefully
        logger.warning(f"[perf] trace capture failed, falling back to host timing only: {e}")
        return None


def _write_results(rows):
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(
                ["tag", "mesh", "vocab", "case", "k", "p", "temp", "module", "traced_us", "host_us", "token_agree"]
            )
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ttnn_mesh_device", _MESH_PARAMS, indirect=True)
@pytest.mark.parametrize("case", _CASES, ids=[c.label for c in _CASES])
def test_sampling_perf_compare(ttnn_mesh_device, case):
    mesh_device = ttnn_mesh_device
    cluster_shape = tuple(mesh_device.shape)
    mesh_id = "x".join(str(d) for d in cluster_shape)
    num_devices = mesh_device.get_num_devices()
    B = 32

    if case.vocab_size % max(cluster_shape) != 0:
        pytest.skip(f"vocab {case.vocab_size} not divisible by {max(cluster_shape)} devices")

    tt_ccl = get_tt_ccl(mesh_device) if num_devices > 1 else None

    torch.manual_seed(42)
    logits_host = torch.randn(1, 1, B, case.vocab_size, dtype=torch.bfloat16)

    # --- TTTv1: TTSampling (k/p/temp stored on the module) ---
    v1_args = _Args(vocab_size=case.vocab_size, cluster_shape=cluster_shape, force_argmax=case.force_argmax)
    v1 = TTSampling(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=v1_args,
        k=torch.full((B,), case.k, dtype=torch.int32),
        p=torch.full((B,), case.p),
        temp=torch.full((B,), case.temp),
    )

    # --- TTTv2: Sampling1D (k/p/temp per-call) ---
    v2 = Sampling1D(
        vocab_size=case.vocab_size,
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        allow_force_argmax=case.force_argmax,
    )
    v2.load_device_buffers()

    # Per-call params for v2. argmax fast-path → all None.
    if case.force_argmax:
        v2_k = v2_p = v2_temp = None
    else:
        v2_k, v2_p, v2_temp = _make_param_tensors(mesh_device, B, case.k, case.p, case.temp)

    logits_v1 = _make_logits_tt(logits_host, mesh_device)
    logits_v2 = _make_logits_tt(logits_host, mesh_device)

    def run_v1():
        return v1.forward(logits_v1)

    def run_v2():
        return v2.decode_forward(logits_v2, k=v2_k, p=v2_p, temp=v2_temp)

    # --- TTTv1 with power-of-2 padding (the perf lever we are demonstrating) ---
    # Padding only changes the multi-device topk path; it is meaningless on the argmax fast-path,
    # and (in TTTv1) it is NOT applied on the 1×1 multi_step_reduction split path. We still build
    # and time it on 1×1 to make that "no improvement on 1×1" fact explicit in the data.
    v1_pad = None
    run_v1_pad = None
    logits_v1_pad = None
    if not case.force_argmax:
        v1_pad_args = _Args(vocab_size=case.vocab_size, cluster_shape=cluster_shape, force_argmax=False, pad=True)
        v1_pad = TTSampling(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            args=v1_pad_args,
            k=torch.full((B,), case.k, dtype=torch.int32),
            p=torch.full((B,), case.p),
            temp=torch.full((B,), case.temp),
        )
        logits_v1_pad = _make_logits_tt(logits_host, mesh_device)

        def run_v1_pad():
            return v1_pad.forward(logits_v1_pad)

    # --- TTTv2 Sampling1D WITH the ported pad_to_power_of_2 (the change under test) ---
    # Same skip rule as v1_pad: padding only affects the multi-device topk path.
    v2_pad = None
    run_v2_pad = None
    if not case.force_argmax:
        v2_pad = Sampling1D(
            vocab_size=case.vocab_size,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            allow_force_argmax=False,
            pad_to_power_of_2=True,
        )
        v2_pad.load_device_buffers()
        logits_v2_pad = _make_logits_tt(logits_host, mesh_device)

        def run_v2_pad():
            return v2_pad.decode_forward(logits_v2_pad, k=v2_k, p=v2_p, temp=v2_temp)

    # Width that the per-device topk shard is padded to (informational). 1×1 uses the split path
    # (V/2 per half) which TTTv1 does not pad.
    per_dev = case.vocab_size // (2 if max(cluster_shape) == 1 else num_devices)
    padded_to = upper_power_of_2(per_dev) if not is_power_of_2(per_dev) else per_dev
    pad_note = f"per_dev={per_dev}→{padded_to}" + (
        " (1×1 split path: NOT padded by TTTv1)" if max(cluster_shape) == 1 else ""
    )

    # --- Correctness sanity + token agreement (informational) ---
    pad_tie_breaks = None
    tok_v1, _ = run_v1()
    tok_v2, _ = run_v2()
    ttnn.synchronize_device(mesh_device)
    t1 = to_torch_auto_compose(tok_v1).flatten()[:B].long()
    t2 = to_torch_auto_compose(tok_v2).flatten()[:B].long()
    assert t1.numel() == B and t2.numel() == B, "both modules must emit B tokens"
    agree = int((t1 == t2).sum().item())
    # Only v2 (the port we are tuning) is held to valid-range correctness. TTTv1 has a known
    # single-device (1×1) local_indices width bug — it emits out-of-range global indices there —
    # which TTTv2 fixes; we don't assert on v1's range, and 1×1 token disagreement is expected.
    assert t2.min() >= 0 and t2.max() < case.vocab_size, "Sampling1D emitted an out-of-range token"

    # Port correctness gate: padding adds only -inf entries, so the padded Sampling1D must select
    # tokens of the SAME logit value as the unpadded one. It need not pick the same INDEX: changing
    # the topk input width (e.g. 16032→16384) changes which index ttnn.topk returns among elements
    # that share the same bf16 value (a tie-break — documented throughout test_sampling_1d.py, and
    # TTTv1's padding has the identical effect). So we assert value-invariance, not index-equality.
    if run_v2_pad is not None:
        tok_v2_pad, _ = run_v2_pad()
        ttnn.synchronize_device(mesh_device)
        t2p = to_torch_auto_compose(tok_v2_pad).flatten()[:B].long()
        assert t2p.min() >= 0 and t2p.max() < case.vocab_size, "padded Sampling1D emitted out-of-range token"
        logits_2d = logits_host.squeeze().float()  # [B, V] (bf16 values widened to fp32)
        true_divergences = [
            b for b in range(B) if t2p[b] != t2[b] and logits_2d[b, t2p[b]].item() != logits_2d[b, t2[b]].item()
        ]
        pad_tie_breaks = int((t2p != t2).sum().item()) - len(true_divergences)
        assert not true_divergences, (
            f"pad_to_power_of_2 selected a different-VALUE token (not a tie-break) at batch "
            f"{true_divergences[:5]} (mesh={mesh_id} {case.label})"
        )

    # --- Timing ---
    modules = [("TTSampling-main", run_v1)]
    if run_v1_pad is not None:
        modules.append(("TTSampling-main-pad", run_v1_pad))
    modules.append(("Sampling1D", run_v2))
    if run_v2_pad is not None:
        modules.append(("Sampling1D-pad", run_v2_pad))

    timings = {}  # name -> (traced_us, host_us)
    for name, fn in modules:
        tr = _time_traced(fn, mesh_device, N_WARMUP, N_TRACE_ITERS)
        ho = _time_host_amortized(fn, mesh_device, N_WARMUP, N_HOST_ITERS)
        timings[name] = (tr, ho)

    def _fmt(x):
        return f"{x:9.2f}" if x is not None else "      n/a"

    def _ratio(num, den):
        if num is None or den is None or den == 0:
            return "n/a"
        return f"{num / den:.2f}x"

    table = "\n".join(f"        {name:<20} {_fmt(timings[name][0])} {_fmt(timings[name][1])}" for name, _ in modules)
    extra = ""
    if "Sampling1D-pad" in timings:
        v2_unpad_tr = timings["Sampling1D"][0]
        v2_pad_tr = timings["Sampling1D-pad"][0]
        v1_pad_tr = timings["TTSampling-main-pad"][0]
        extra = (
            f"\n        PORT: Sampling1D pad speedup (unpad/pad) traced={_ratio(v2_unpad_tr, v2_pad_tr)}"
            f"  | Sampling1D-pad vs TTSampling-main-pad traced={_ratio(v2_pad_tr, v1_pad_tr)} (parity → ~1.0x)"
        )

    tie_note = f"  port_pad_tie_breaks={pad_tie_breaks}/{B}" if pad_tie_breaks is not None else ""
    logger.info(
        f"\n[perf] mesh={mesh_id} {case.label} (k={case.k} p={case.p} t={case.temp}) "
        f"tokens_agree={agree}/{B}  pad:{pad_note}{tie_note}\n"
        f"        {'module':<20} {'traced_us':>9} {'host_us':>9}\n"
        f"{table}{extra}"
    )

    _write_results(
        [
            [
                PERF_TAG,
                mesh_id,
                case.vocab_size,
                case.label,
                case.k,
                case.p,
                case.temp,
                name,
                f"{timings[name][0]:.2f}" if timings[name][0] is not None else "",
                f"{timings[name][1]:.2f}",
                agree,
            ]
            for name, _ in modules
        ]
    )
