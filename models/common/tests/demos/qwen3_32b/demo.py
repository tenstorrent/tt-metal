# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 Qwen3-32B demo — accuracy and performance measurement on T3K and P150x4.

Uses ``EagerQwen3_32BExecutor`` / ``TracedQwen3_32BExecutor`` directly (no vLLM adapter).

**Mesh note.** Qwen3-32B has 64 attention heads and 8 KV heads. The TTTv2 composition supports
physical Wormhole T3K (TP8) and physical BlackHole P150x4 (TP4), matching TTTv1's BH model support.
The P150x4 path keeps batched prefill disabled until the plan's cross-cardinality experiment
passes and advertises the source Q128/Q1024 prefill-trace buckets. Consequently:
  - **T3K (8 devices): the established regression mesh.** Existing thresholds remain unchanged.
  - **P150x4 (4 devices): the BH qualification mesh.** It uses Ring fabric through the shared
    hardware-agnostic modules; full-model runs require a physical P150_X4 or P300_X2 product,
    not a device-count shortcut.
  - **ci-b1-DP-*: skipped** — every DP group is a single device, which cannot hold this 32B (same
    memory limit); you cannot have both 1-device-per-user and TP4/TP8. Genuine hardware-capacity
    guard (like the qwen25_7b N150 skip), matching TTTv1's supported tensor-parallel deployments.

CI cases (parity with TTTv1 ``simple_text_demo.py``):
    token-accuracy   - teacher-forcing top-1/top-5 vs the book ``.refpt``
    batch-1          - single-user latency
    batch-32         - short-context throughput (seq1024 / 200 decode)
    batch-32-ci      - CI-faithful batch-32 (seq2048 / 1024 decode; TTTv1 ci-32)
    eval-32          - 32-user cross-batch determinism (TTTv1 ci-eval-32)
    eval-32-perf-report - same three eval repeats; first repeat emits telemetry and enforces targets
    ci-b1-DP-{2..32} - single-user data-parallel scaling smoke (TTTv1 ci-b1-DP-*); all skip on T3K

Usage:
    # Token accuracy (gates against the committed book ``.refpt``)
    MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3-32B \\
      pytest models/common/tests/demos/qwen3_32b/demo.py -k "token-accuracy" -v

    # On-device sampling perf sweep (the T3K headline / TTTv1-comparable path)
    SAMPLING_MODE=on_device_topk MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3-32B \\
      pytest models/common/tests/demos/qwen3_32b/demo.py -k "batch-32-ci" -v

LazyWeight tensor cache: ``TT_CACHE_PATH/<device_name>`` when ``TT_CACHE_PATH`` is set, otherwise
``model_cache/<HF_MODEL>/<device_name>`` under the current working directory.
"""

import json
import math
import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig, AutoTokenizer

import ttnn
from models.common.device_utils import get_device_name
from models.common.models.qwen3_32b.executor import EagerQwen3_32BExecutor, TracedQwen3_32BExecutor
from models.common.models.qwen3_32b.model import QWEN3_32B_ACCURACY, QWEN3_32B_PERFORMANCE, Qwen3_32B
from models.common.sampling.sampling_params import SamplingParams
from models.common.tests.demos.cleanup_utils import cleanup_model_case
from models.common.tests.demos.run_helpers import (
    eval_decode_trace_mode,
    load_eval_repeat_prompts_batch32,
    require_canonical_eval_modes_in_ci,
    run_eval_repeat_batch32,
    run_perf_benchmark,
    run_teacher_forcing,
)
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.demos.utils.model_targets import resolve_accuracy_targets, resolve_metric_tolerance, resolve_perf_targets
from models.demos.utils.trace_region_sizes import resolve_trace_region_size
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import encode_prompt_hf

# =============================================================================
# Expected metrics — perf gates set from a same-box TTTv1-vs-TTTv2 sweep (on-device sampling),
# NOT PERF.md (PERF.md's 22.9/19.6 tok/s/u are unreachable on either stack).
#
# Rule (per cell): each ``tok_s_u`` target is the BETTER of TTTv1 vs TTTv2 for that sampling mode.
# TTTv1 has only an on-device sampling path, so:
#     on_device_topk : max(TTTv1_on_device, TTTv2_on_device_topk)
#     host           : TTTv2_host                      (TTTv1 has no host-sampling path)
# Decode throughput is prefill-independent, so batched prefill does NOT change ``tok_s_u``.
# ``ttft_ms`` targets are conservative upper bounds (batched prefill only LOWERS TTFT).
#
# Perf cases default to SAMPLING_MODE=on_device_topk, the path comparable to TTTv1's auto on-device
# sampling on T3K (vocab shards 8-way). The host path pays a full-vocab all-gather + PCIe readback per
# step (6-8x slower) and is NOT comparable to TTTv1 — measuring host vs TTTv1 fabricates a "gap". The
# host bucket below is left ungated ({}) unless separately measured; a case still RUNS + prints tok_s_u.
# =============================================================================

# top1/top5 teacher-forcing accuracy floors (book refpt), profile-split. Perf metrics live in the batch
# dicts below. Floors set conservatively below measured (5% PERF_TOLERANCE gives headroom).
EXPECTED_METRICS: dict = {
    "performance": {
        "T3K": {"top1": 89, "top5": 97},
    },
    "accuracy": {
        "T3K": {"top1": 95, "top5": 100},
    },
}

# batch-1 throughput, sampling-mode- and profile-aware. on_device_topk is the T3K headline; gate =
# better-of(TTTv1, TTTv2) per the parity rule. Prior-healthy same-box TTTv1 control (simple_text_demo
# -k batch-1, "Average speed"): perf 27.1 t/s/u (36.9ms/step, TTFT 118.8ms), acc 22.57 (44.3ms/step).
#
# DECODE GAP CLOSED (issue #49282, fixed by #49284). The base now carries the shared on-device decode
# loop + pipelined non-blocking readback (model-owned traced executor), and it IS wired into this
# model (TracedQwen3_32BExecutor(ondevice_decode_loop=...) on the perf path). That removes the per-step
# host round-trip (blocking readback + synchronize_device) that made TTTv2 ~35% slower at batch-1 on the
# old base (c93ed50, which had no on-device decode loop). On a healthy box TTTv2 on_device_topk reaches
# TTTv1 parity here (sibling qwen25_coder_32b, identical wiring/base: b1 97%). The gate stays at the
# prior-healthy TTTv1 best-of (27.1 / 22.6); ttft is a ceiling TTTv2 clears. NB: a run on a #893
# NUMA-degraded T3K depresses BOTH stacks ~1.8x (to ~14-15 t/s/u) — parity is then confirmed RELATIVE
# to same-box TTTv1 (measured b1 TTTv2 14.7 vs TTTv1 15.0 = 98%), never by lowering this gate.
EXPECTED_METRICS_BATCH1: dict = {
    "host": {
        "performance": {},
        "accuracy": {},
    },
    "on_device_topk": {
        "performance": {
            "T3K": {"tok_s_u": 27.5, "ttft_ms": 125}
        },  # best-of{TTTv2, TTTv1} — same-box decode is at ~parity (2026-07-25: TTTv2 27.5 vs TTTv1 27.9,
        # ~1.4% under; a diffuse shared-engine per-step delta, NOT lowered to a slow number — see PR.md).
        # b1 TTFT is noisy (both stacks span ~96-105ms); the 125 ceiling covers ON+OFF with headroom.
        "accuracy": {"T3K": {"tok_s_u": 23.1, "ttft_ms": 145}},  # best-of{TTTv2 22.5, TTTv1 23.16}; TTTv2
        # decode ~2.9% under TTTv1 (diffuse shared-engine per-step delta, HiFi4 path; not lowered to TTTv2 —
        # see PR.md). b1 TTFT noisy (~118-127ms both stacks); 145 ceiling covers ON+OFF.
    },
}

# Short-context batch-32 throughput (seq1024 / 200 decode), sampling-mode- and profile-aware. Runs BOTH
# batched-prefill ON (default) and DISABLE_BATCHED_PREFILL=1 (A/B). Decode tok_s_u is prefill-independent
# so the gate covers both knob states; ttft covers both (ON << OFF → gate above the sequential value).
# The short seq1024/200-decode leg has NO matching TTTv1 CI workload (TTTv1's CI batch-32 IS ci-32 =
# our batch-32-ci), so the gate = TTTv2-measured (a regression gate, conservative floor). Same-box
# on_device_topk: perf ~17.3-17.5 t/s/u (ON TTFT 50.8ms), acc ~15.4-20.3 (ON TTFT 59.7ms). The 200-step
# window carries more first-token/warmup overhead than the 1024-step batch-32-ci window, so these
# per-step averages run lower + noisier than batch-32-ci despite the smaller KV — a measurement-window
# effect, not a regression; the tok_s_u floors are set at the LOWEST observed across ON+OFF so they
# don't flap. ttft is keyed per profile to cover BOTH knob states: batched-ON prefill is ~50-60ms but
# the DISABLE_BATCHED_PREFILL=1 sequential 32-user prefill is ~103ms (perf) / ~113ms (acc, HiFi4), so
# the ceilings sit above the sequential value (batched prefill ~halves TTFT — a real win). Not a
# weakening: it's the real sequential-leg bound both ON and OFF clear (mirrors the llama1b pilot).
EXPECTED_METRICS_BATCH32: dict = {
    "host": {
        "performance": {},
        "accuracy": {},
    },
    "on_device_topk": {
        "performance": {"T3K": {"tok_s_u": 17.3, "ttft_ms": 110}},
        "accuracy": {"T3K": {"tok_s_u": 15.4, "ttft_ms": 120}},
    },
}

# CI-faithful batch-32 targets (the ``batch-32-ci`` leg), seq2048 + 1024-token decode budget = the
# DIRECT TTTv1 ci-32 analog. gate = better-of(TTTv1 ci-32, TTTv2). Prior-healthy same-box TTTv1 ci-32
# ("Average speed", seq2048/1024): perf 25.06 t/s/u (39.9ms/step, TTFT 41.1ms), acc 20.45 (48.9ms/step).
#
# DECODE GAP CLOSED (issue #49282, fixed by #49284). The on-device decode loop is wired into this model
# (removes the per-step host round-trip), so same-box decode step time is at TTTv1 parity within ~1-2%
# (a diffuse shared-engine per-step delta; see PR.md). The gate stays at the prior-healthy TTTv1 best-of;
# never lowered. NB: a #893 NUMA-degraded T3K depresses BOTH stacks ~1.8x — confirm parity RELATIVE to
# same-box TTTv1 there, never lower the gate to the degraded number.
#
# TTFT LEVER — minimal_matmul (model.py prefill_minimal_matmul, default ON; DISABLE_MINIMAL_MATMUL=1 to
# A/B off). The batch-32-ci prefill is matmul-compute-bound, so enabling minimal_matmul for the QKV + FF2
# prefill matmuls cuts ci-32 TTFT: same-box median-of-3 (2026-07-25) perf 47.3ms (OFF) -> 40.3ms (ON),
# acc ~56 -> 48.8ms — closing most of the old +28/36% gap vs TTTv1 (perf 37.4 / acc 41.5ms) down to
# ~+8% / +18%. Accuracy is unchanged with it ON (eval-32 64/64 host, batched ON+OFF; token-accuracy
# 90.6/98.6 perf, 96.7/100 acc). The ttft gate is a CEILING TTTv2 clears
# (batched-ON ~40/49 << the sequential-OFF ~103/113); the tolerance-free parity RED lives in PR.md,
# not a lowered gate.
EXPECTED_METRICS_BATCH32_CI: dict = {
    "host": {
        "performance": {},
        "accuracy": {},
    },
    "on_device_topk": {
        # best-of{TTTv2, same-box TTTv1 ci-32}. Decode: TTTv2 25.3/20.5 vs TTTv1 25.75/20.89 — ~1.7/1.9%
        # under (diffuse shared-engine per-step delta; NOT lowered to the TTTv2 number — see PR.md).
        # ttft is a CEILING TTTv2 clears (minimal_matmul-ON batched ~40/49 << the sequential-OFF ~103/113);
        # the tolerance-free TTFT parity RED is documented in PR.md + the shared-gap ticket.
        "performance": {"T3K": {"tok_s_u": 25.7, "ttft_ms": 110}},
        "accuracy": {"T3K": {"tok_s_u": 20.8, "ttft_ms": 120}},
    },
}

# Perf workload: natural-length prefill (these sample prompts are ~90-125 tokens -> 128 bucket,
# matching TTTv1), 200 decode steps. Accuracy uses the teacher-forcing refpt.
_PERF_NUM_DECODE_TOKENS = int(os.environ.get("PERF_NUM_DECODE_TOKENS", "200"))

PERF_TOLERANCE = 0.05

# Central target geometry for TTTv1 ``performance-ci-eval-32``. This is intentionally separate from
# batch-32-ci: the perf-report node runs the exact three rotated eval repeats and gates its first repeat.
_EVAL32_TARGET_SEQ_LEN = 686


def _resolve_eval32_perf_targets(hf_model: str, device_name: str, optimizations: str) -> dict | None:
    # The centralized p300x2 target is backed by a profile-matched performance run.  It is not an
    # accuracy-profile floor: the accuracy variant must still execute and emit telemetry, but its
    # measurements remain observational until an independent accuracy floor is frozen.
    if device_name == "P150x4" and optimizations != "performance":
        logger.warning(
            f"{optimizations}/eval-32-perf-report: no profile-matched P150x4 performance floor; "
            "running the full workload and reporting metrics observationally"
        )
        return None

    expected = resolve_perf_targets(
        hf_model,
        device_name,
        batch_size=32,
        seq_len=_EVAL32_TARGET_SEQ_LEN,
    )
    if not expected:
        if device_name == "P150x4":
            logger.warning(
                f"No centralized eval-32 performance floor for {hf_model} on {device_name} "
                f"(profile={optimizations}, batch_size=32, seq_len={_EVAL32_TARGET_SEQ_LEN}); "
                "running and reporting metrics observationally"
            )
            return None
        raise ValueError(
            f"No centralized eval-32 perf target for {hf_model} on {device_name} "
            f"(batch_size=32, seq_len={_EVAL32_TARGET_SEQ_LEN}); qualification gates fail closed."
        )
    required = ("decode_t/s/u", "prefill_time_to_first_token")
    missing = [metric for metric in required if metric not in expected]
    if missing:
        if device_name == "P150x4":
            logger.warning(
                f"Incomplete centralized eval-32 performance floor for {hf_model} on {device_name} "
                f"(profile={optimizations}): missing {missing}; running and reporting metrics observationally"
            )
            return None
        raise ValueError(
            f"Incomplete centralized eval-32 perf target for {hf_model} on {device_name}: missing {missing}"
        )
    return expected


def _assert_eval32_perf_target(result, expected: dict, *, case_name: str) -> None:
    decode_target = float(expected["decode_t/s/u"])
    ttft_target = float(expected["prefill_time_to_first_token"])
    decode_tolerance = resolve_metric_tolerance("decode_t/s/u", expected, PERF_TOLERANCE)
    ttft_tolerance = resolve_metric_tolerance("prefill_time_to_first_token", expected, PERF_TOLERANCE)
    failures = []
    if result.tok_s_u < decode_target * (1 - decode_tolerance):
        failures.append(f"tok/s/u {result.tok_s_u:.1f} < target {decode_target}")
    if result.ttft_ms > ttft_target * (1 + ttft_tolerance):
        failures.append(f"ttft_ms {result.ttft_ms:.1f} > target {ttft_target}")
    assert not failures, f"{case_name}: " + "; ".join(failures)


def _resolve_local_perf_floor(device_name: str, expected: dict, *, case_name: str) -> dict | None:
    if device_name != "P150x4":
        return expected
    missing = [metric for metric in ("tok_s_u", "ttft_ms") if metric not in expected]
    if missing:
        logger.warning(
            f"{case_name}: no complete profile-matched P150x4 performance floor (missing {missing}); "
            "running the full workload and reporting metrics observationally"
        )
        return None
    return expected


def _assert_local_perf_target(result, expected: dict, *, case_name: str) -> None:
    failures = []
    if result.tok_s_u < expected["tok_s_u"] * (1 - PERF_TOLERANCE):
        failures.append(f"tok/s/u {result.tok_s_u:.1f} < target {expected['tok_s_u']}")
    if result.ttft_ms > expected["ttft_ms"] * (1 + PERF_TOLERANCE):
        failures.append(f"ttft_ms {result.ttft_ms:.1f} > target {expected['ttft_ms']}")
    assert not failures, f"{case_name}: " + "; ".join(failures)


# batch-32-ci per-SKU max_seq_len (TTTv1 ci-32 parity is seq2048). Qwen3-32B is capped at 4096
# (TTTv1 reports a hang at 8192). P150x4 keeps the same CI geometry; physical memory feasibility is
# an explicit first hardware milestone and must pass before the remaining P150x4 perf floors are frozen.
_BATCH32_CI_MAX_SEQ_LEN: dict[str, int] = {
    "T3K": 2048,
    "P150x4": 2048,
}


def _sampling_bucket() -> str:
    """Map SAMPLING_MODE to a perf-gate bucket. Defaults to ``on_device_topk`` (the perf-case default
    for this T3K model), so the bucket always agrees with the runner. Non-topk on-device modes (e.g.
    force-argmax) also fall into ``on_device_topk`` so they stay gated, never silently un-gated."""
    return "host" if os.environ.get("SAMPLING_MODE", "on_device_topk").lower() == "host" else "on_device_topk"


# Qwen3-32B needs at least TP4: TTTv1 supports the model on physical P150x4 and TTTv2 composes the
# same BH geometry through explicit module wrappers. Single-device DP groups remain unsupported.
_MIN_TP_DEVICES = 4


def _skip_below_min_tp_devices(n_devices: int) -> None:
    """Skip when fewer than ``_MIN_TP_DEVICES`` devices are available for tensor parallelism."""
    if n_devices < _MIN_TP_DEVICES:
        pytest.skip(
            f"Qwen3-32B requires >={_MIN_TP_DEVICES}-device tensor parallelism: the 32B weights + KV "
            f"cache require T3K TP8 or P150x4 TP4. Have {n_devices} device(s) — use "
            "MESH_DEVICE=T3K or MESH_DEVICE=P150x4."
        )


# Mesh topology comes only from ``MESH_DEVICE`` (same naming as vLLM / other tt demos).
_MESH_DEVICE_TO_SHAPE: dict[str, tuple[int, int]] = {
    "T3K": (1, 8),
    "P150x4": (1, 4),
}


def _ttnn_mesh_device_param_from_env() -> dict:
    env = os.environ.get("MESH_DEVICE", "").strip()
    if not env:
        pytest.skip(
            "MESH_DEVICE must be set to T3K or P150x4. See module docstring.",
            allow_module_level=True,
        )
    shape = _MESH_DEVICE_TO_SHAPE.get(env)
    if shape is None:
        pytest.skip(
            f"Unsupported MESH_DEVICE={env!r} for Qwen3-32B; use T3K or P150x4.",
            allow_module_level=True,
        )
    param = {
        "mesh_shape": shape,
        "trace_region_size": resolve_trace_region_size("qwen3-32b", env),
        "num_command_queues": 1,
    }
    # TTTv2 multi-device executor dispatch requires explicit fabric. Both approved overlays use Ring
    # collectives, so the fixture fabric must match the model's construction-time topology choice.
    if shape != (1, 1):
        param["fabric_config"] = ttnn.FabricConfig.FABRIC_1D_RING
    return param


pytestmark = [
    pytest.mark.parametrize(
        "ttnn_mesh_device",
        [_ttnn_mesh_device_param_from_env()],
        indirect=True,
        ids=[os.environ.get("MESH_DEVICE", "mesh").strip() or "mesh"],
    ),
]


@pytest.fixture(scope="module")
def mesh_device(ttnn_mesh_device):
    """Real mesh for this file; shape is fixed by ``MESH_DEVICE`` (see ``pytestmark``)."""
    return ttnn_mesh_device


def _skip_unless_heads_divide_mesh(mesh_device: ttnn.MeshDevice, hf_model_id: str) -> None:
    """Attention1D TP requires n_heads and n_kv_heads divisible by device count."""
    n_dev = mesh_device.get_num_devices()
    if n_dev <= 1:
        return
    cfg = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True)
    n_h, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    if n_h % n_dev == 0 and n_kv % n_dev == 0:
        return
    pytest.skip(
        f"Incompatible mesh for {hf_model_id}: {n_dev} devices need "
        f"num_attention_heads ({n_h}) and num_key_value_heads ({n_kv}) each divisible by {n_dev}."
    )


def lazy_weight_cache_dir_for_demo(mesh_device: ttnn.MeshDevice, hf_model_id: str) -> Path:
    """Disk root for ``Qwen3_32B`` ``LazyWeight`` caches in this e2e demo.

    Matches ``models/tt_transformers/tt/model_config.py`` (HF checkpoint branch): if ``TT_CACHE_PATH``
    is set, use ``<TT_CACHE_PATH>/<device_name>``; otherwise ``model_cache/<HF_MODEL>/<device_name>``.
    Persistent cache materially reduces re-run cost for 64-layer 32B weight materialization.
    """
    device_name = get_device_name(mesh_device)
    hf = hf_model_id.strip("/")
    tt_cache = os.getenv("TT_CACHE_PATH")
    if tt_cache:
        root = Path(tt_cache) / device_name
    else:
        root = Path("model_cache") / hf / device_name
    root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Qwen3-32B demo LazyWeight cache directory: {root.resolve()}")
    return root


def _warmup_demo_executor(
    executor,
    *,
    kv_cache,
    page_table,
    prefill_compile_case=None,
    prefill_sampling_params=None,
    prefill_compile_execution=None,
):
    """Compile eager programs and representative requests before trace activation."""
    config = executor.config
    prefill_kwargs = {
        "kv_cache": kv_cache,
        "can_sample_on_device": config.device_sampling_enabled,
    }
    decode_kwargs = {
        "kv_cache": kv_cache,
        "max_batch_size": int(executor.model.config.max_batch_size),
        "num_blocks": int(page_table.shape[-1]),
        "can_sample_on_device": config.device_sampling_enabled,
    }
    executor.warmup_model_decode(enable_trace=False, **decode_kwargs)
    executor.warmup_model_prefill(enable_trace=False, **prefill_kwargs)
    if prefill_compile_case is not None:
        tokens, prompt_lens = prefill_compile_case
        executor.compile_prefill(
            tokens=tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=list(range(tokens.shape[0])),
            sampling_params=prefill_sampling_params,
            execution=prefill_compile_execution if prefill_compile_execution is not None else executor.eager_execution,
        )
    if config.trace.prefill_enabled:
        executor.warmup_model_prefill(enable_trace=True, **prefill_kwargs)
    if config.trace.decode_enabled:
        executor.warmup_model_decode(enable_trace=True, **decode_kwargs)


def ref_basename_for_hf(hf_model_id: str) -> str:
    """Match ``ModelArgs.model_name`` style used for ``.refpt`` filenames."""
    return hf_model_id.strip("/").split("/")[-1]


def _load_tokenizer(hf_model_id: str):
    """Load HF tokenizer with a writable-cache fallback.

    The default ``HF_HOME`` on shared dev hosts is often owned by another user, so
    ``AutoTokenizer.from_pretrained`` cannot create ``.locks/`` entries when tokenizer files are missing
    from the shared cache. On ``OSError`` / ``PermissionError`` from the default path, retry with
    ``cache_dir`` pointing at the user's home HF cache (tokenizer files are <10 MB so this is cheap).
    """
    try:
        return AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
    except (OSError, PermissionError) as e:
        msg = str(e)
        if "Permission" not in msg and "permission" not in msg:
            raise
        fallback = os.environ.get("TT_TOKENIZER_FALLBACK_CACHE", str(Path.home() / ".cache" / "huggingface"))
        logger.warning(
            f"Default HF cache not writable for tokenizer download ({e!s:.120}); " f"retrying with cache_dir={fallback}"
        )
        Path(fallback).mkdir(parents=True, exist_ok=True)
        return AutoTokenizer.from_pretrained(hf_model_id, cache_dir=fallback, trust_remote_code=True)


def load_reference_data(hf_model_id: str):
    """Load reference tensors and optional metadata from ``.refpt``."""
    name = ref_basename_for_hf(hf_model_id)
    ref_path = Path("models/tt_transformers/tests/reference_outputs") / f"{name}.refpt"
    if not ref_path.exists():
        pytest.skip(f"Reference file not found: {ref_path}")

    ref_data = torch.load(ref_path, map_location="cpu", weights_only=False)
    reference_tokens = ref_data["reference_tokens"]
    top5_tokens = ref_data["top5_tokens"]
    prompt_len = ref_data.get("prompt_len")
    metadata = ref_data.get("metadata")
    return reference_tokens, top5_tokens, prompt_len, metadata


def load_input_prompts(batch_size: int) -> list[str]:
    """Load input prompts for performance testing."""
    prompts_path = Path("models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json")
    if not prompts_path.exists():
        return ["What is the meaning of life?"] * batch_size

    with open(prompts_path) as f:
        data = json.load(f)

    prompts = (
        [entry["prompt"] for entry in data] if isinstance(data, list) else data.get("prompts", [data.get("prompt", "")])
    )
    while len(prompts) < batch_size:
        prompts = prompts * 2
    return prompts[:batch_size]


def tokenize_prompts(
    prompts: list[str],
    tokenizer,
    *,
    max_prefill_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize prompts to their natural length — TTTv1 ``preprocess_inputs_prefill`` semantics.

    Each prompt is encoded with the chat template at its real length. The returned ``[batch, max_len]``
    token tensor is right-padded to the batch-max for rectangularity, while the returned per-user
    lengths are the *real* token counts — the executor reads only ``tokens[user, :prompt_len]`` and then
    buckets each user to ``get_padded_prefill_len`` (128 / 1024 / next-pow2). This matches TTTv1 exactly
    (no fixed pad-to-N prefill budget) and is what lets equal-length users share a batched-prefill group.

    ``max_prefill_len`` is an optional clip *cap* (like TTTv1's ``max_prefill_len``): prompts longer
    than it are left-clipped to their most recent tokens. It is never a pad-up target.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    encoded: list[list[int]] = []
    for p in prompts:
        ids = list(encode_prompt_hf(tokenizer, p))
        if max_prefill_len is not None and len(ids) > max_prefill_len:
            ids = ids[-max_prefill_len:]
        encoded.append(ids)
    lens = [len(ids) for ids in encoded]
    max_len = max(lens)
    padded = [ids + [pad_id] * (max_len - len(ids)) for ids in encoded]
    t = torch.tensor(padded, dtype=torch.long)
    return t, torch.tensor(lens, dtype=torch.long)


def select_teacher_forcing_top5_slice(
    top5_tokens: torch.Tensor, reference_tokens: torch.Tensor, prompt_len: int, *, metadata_aligned: bool
) -> torch.Tensor:
    """Align ``top5_tokens`` with teacher-forcing targets across refpt conventions."""
    num_target = len(reference_tokens) - prompt_len
    target_tokens = reference_tokens[prompt_len : prompt_len + num_target]
    if num_target <= 0:
        raise ValueError("prompt_len must be smaller than reference length")

    if metadata_aligned and top5_tokens.shape[0] == num_target:
        logger.info(
            "Teacher-forcing top5 alignment: metadata-driven direct path "
            f"(top5_len={top5_tokens.shape[0]}, target_len={num_target})"
        )
        return top5_tokens

    candidates = []
    starts = (0, prompt_len - 1, prompt_len) if metadata_aligned else (prompt_len - 1, prompt_len)
    for start in starts:
        end = start + num_target
        if start < 0 or end > top5_tokens.shape[0]:
            continue
        aligned = top5_tokens[start:end]
        probe = min(16, num_target)
        score = sum(int(aligned[i, 0].item() == target_tokens[i].item()) for i in range(probe))
        candidates.append((score, start, aligned))

    if not candidates:
        raise ValueError(
            f"Cannot align top5 tokens: prompt_len={prompt_len}, num_target={num_target}, top5_len={top5_tokens.shape[0]}"
        )

    best_score, best_start, best = max(candidates, key=lambda x: x[0])
    logger.info(
        f"Teacher-forcing top5 alignment: start={best_start}, boundary score={best_score}/{min(16, num_target)}"
    )
    return best


def log_generated_text(prompts, generated_token_ids, tokenizer):
    """Print the final generated continuation for each user."""
    logger.info("Finished decoding, printing the final outputs...\n")
    for user, output_ids in enumerate(generated_token_ids):
        prompt_text = prompts[user] if user < len(prompts) else ""
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        short_prompt = (
            prompt_text[:100] + "\n<long prompt not printed in full>\n" + prompt_text[-100:]
            if len(prompt_text) > 200
            else prompt_text
        )
        logger.info(f"\n==USER {user} - PROMPT\n{short_prompt}\n==USER {user} - OUTPUT\n{generated_text}\n")


def log_teacher_forcing_text(prompt_tokens, predicted_tokens_per_user, reference_tokens, tokenizer):
    """Print prompt, predicted continuation, and reference continuation for every teacher-forced user."""
    reference_text = tokenizer.decode(reference_tokens.tolist(), skip_special_tokens=True).strip()
    for user, user_prompt_tokens in enumerate(prompt_tokens):
        prompt_text = tokenizer.decode(user_prompt_tokens.tolist(), skip_special_tokens=True)
        predicted_text = tokenizer.decode(predicted_tokens_per_user[user], skip_special_tokens=True).strip()
        short_prompt = (
            prompt_text[:100] + "\n<long prompt not printed in full>\n" + prompt_text[-100:]
            if len(prompt_text) > 200
            else prompt_text
        )
        logger.info(
            f"\n==USER {user} - PROMPT\n{short_prompt}\n==USER {user} - OUTPUT\n{predicted_text}\n"
            f"==USER {user} - REFERENCE\n{reference_text}\n"
        )


def create_model(
    mesh_device,
    optimizations: str,
    cache_dir: Path,
    *,
    max_batch_size: int = 32,
    max_seq_len: int | None = None,
    disable_batched_prefill: bool | None = None,
):
    """Build ``Qwen3_32B`` in executor (paged KV) mode on T3K or P150x4.

    Picks one of the two module-level precision recipes (``QWEN3_32B_ACCURACY`` /
    ``QWEN3_32B_PERFORMANCE``) — both defined in ``qwen3_32b/model.py`` and grounded in TTTv1's
    ``DecodersPrecision`` for Qwen3-32B. The dataclass owns the dtype + math-fidelity recipe; this demo
    just selects between the two and forwards it.

    ``max_batch_size`` must match the workload: decode DRAM matmul CB usage scales with tile-padded
    batch rows, so batch-1 perf tests should pass ``max_batch_size=1`` even when batch-32 / eval-32 /
    teacher-forcing cases need 32.

    ``max_seq_len`` overrides the default. Default (``None``): ``min(131072 // max_batch_size, 4096)``.
    Qwen3-32B is capped at 4096 (TTTv1 reports the model hangs at 8192). The ``batch-32-ci`` leg passes
    an explicit value (see ``_BATCH32_CI_MAX_SEQ_LEN``).
    """
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen3-32B")
    _skip_below_min_tp_devices(mesh_device.get_num_devices())
    _skip_unless_heads_divide_mesh(mesh_device, hf_model)

    precision = QWEN3_32B_PERFORMANCE if optimizations == "performance" else QWEN3_32B_ACCURACY

    if max_seq_len is None:
        # T3K: 64 layers × 8 KV heads / 8 dev × head_dim 128 → KV per device per layer is modest.
        # Capped at 4096 (TTTv1: "Qwen3-32B hangs at 8192, so we cap at 4096").
        max_seq_len = min(131072 // max_batch_size, 4096)

    try:
        model = Qwen3_32B.from_pretrained(
            mesh_device,
            hf_model,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            num_layers=None,
            cache_dir=cache_dir,
            precision=precision,
            executor_mode=True,
            disable_batched_prefill=disable_batched_prefill,
        )
    except Exception as e:
        # BH qualification nodes are required gates: construction failures must surface as failures,
        # not be converted into environmental skips. Preserve the established T3K skip behavior.
        if get_device_name(mesh_device) == "P150x4":
            raise
        pytest.skip(f"Could not build Qwen3-32B model (weights / memory / mesh): {e}")

    return model


# =============================================================================
# ci-b1-DP: single-user data-parallel scaling smoke (TTTv1 ci-b1-DP-* parity)
# =============================================================================
#
# One user per DP group, model replicated across ``data_parallel`` disjoint submeshes, instruct
# prompts, paged attention, trace on. The ONLY correctness check is the special-token garbage guard
# plus "runs to completion without hang/exception". This is a mesh / KV-cache / page-table scaling
# smoke, NOT an accuracy or perf gate.
#
# Per-case size table (TTTv1 simple_text_demo.py parity):
#   ci-b1-DP-2  : max_seq_len=1024, max_generated_tokens=200,  stop_at_eos=True
#   ci-b1-DP-4  : max_seq_len=4096, max_generated_tokens=2048, stop_at_eos=False
#   ci-b1-DP-8  : max_seq_len=4096, max_generated_tokens=2048, stop_at_eos=False
#   ci-b1-DP-16 : max_seq_len=1024, max_generated_tokens=200,  stop_at_eos=True
#   ci-b1-DP-32 : max_seq_len=1024, max_generated_tokens=200,  stop_at_eos=True
#
# Hardware feasibility: each DP group is one device (batch_size=1 per group), so
# ``data_parallel == n_devices``. Qwen3-32B needs 8-way TP (a single device cannot hold the 32B), so
# EVERY DP factor is inapplicable: you cannot have both 1-device-per-user AND 8-device TP. All factors
# cleanly ``pytest.skip`` (genuine hardware-capacity guard, matching TTTv1's T3K-only support). The case
# ids are present for parity with TTTv1 ``simple_text_demo.py``.
_DP_SIZE_TABLE: dict[int, dict] = {
    2: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
    4: {"max_seq_len": 4096, "max_generated_tokens": 2048, "stop_at_eos": False},
    8: {"max_seq_len": 4096, "max_generated_tokens": 2048, "stop_at_eos": False},
    16: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
    32: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
}


def create_dp_submeshes(mesh_device: ttnn.MeshDevice, data_parallel: int) -> list:
    """Partition the open parent mesh into ``data_parallel`` disjoint row-submeshes.

    Mirrors TTTv1 ``generator.create_submeshes`` minus the Galaxy reshape branch (no Galaxy reachable
    here). For the single-user DP cases ``n // data_parallel == 1``, so each submesh is a ``(1,1)``
    mesh. Fabric stays owned by the parent — do NOT set fabric per-submesh.
    """
    if data_parallel == 1:
        return [mesh_device]
    n = mesh_device.get_num_devices()
    assert n % data_parallel == 0, f"{n} devices not divisible by data_parallel={data_parallel}"
    return mesh_device.create_submeshes(ttnn.MeshShape(1, n // data_parallel))


def _dp_or_skip(mesh_device: ttnn.MeshDevice, data_parallel: int) -> None:
    """Skip unless the mesh has exactly ``data_parallel`` single-device DP groups."""
    n = mesh_device.get_num_devices()
    if n % data_parallel != 0 or (n // data_parallel) != 1:
        pytest.skip(f"DP-{data_parallel} needs {data_parallel} single-device groups; have {n} devices")


def assert_no_special_tokens(
    generated_token_ids, tokenizer, *, case_name: str = "", is_ci_env: bool | None = None
) -> None:
    """Garbage guard: no special token mid-stream. Mirrors TTTv1 ``simple_text_demo.py``.

    TTTv2's ``result.generated_token_ids[user]`` already starts at the first generated token, so
    unlike TTTv1 we do not slice off the prompt — these are output-only. Each user's output is
    truncated at the first turn boundary (EoS / ``<|im_end|>`` / ``<|im_start|>``) before scanning, then checked for any
    ``tokenizer.all_special_ids`` member. Following TTTv1, a survivor logs a warning always but
    hard-fails only under CI (``CI == "true"``), so local runs finish while CI stays strict.
    """
    if is_ci_env is None:
        is_ci_env = os.environ.get("CI") == "true"
    special = set(tokenizer.all_special_ids)
    stop = set()
    if tokenizer.eos_token_id is not None:
        stop.add(tokenizer.eos_token_id)
    # Qwen turn terminators. <|im_end|> (eos) ends the assistant turn; <|im_start|> OPENS a new turn —
    # i.e. the assistant's response is over and it has begun hallucinating the *next* turn, which is a
    # legitimate Qwen response terminator (serving stacks stop on it; HF generation_config omits it).
    # The perf benchmark runs a FIXED decode budget with stop_at_eos off, so an open-ended prompt is
    # force-decoded past its answer and greedily degenerates into "<|im_start|>user …" (verified
    # byte-identical on host and on_device_topk => inherent greedy divergence, not a sampling/decode-loop
    # artifact). Truncating the real response at either turn boundary before the garbage scan mirrors the
    # eval-32 stop-set augment and matches TTTv1, which STOPS generation at these tokens. This does not
    # hide garbage: any special id emitted mid-response (before the first turn boundary) is still flagged.
    for turn_tok in ("<|im_end|>", "<|im_start|>"):
        tid = tokenizer.convert_tokens_to_ids(turn_tok)
        if isinstance(tid, int) and tid >= 0:
            stop.add(tid)
    offenders = 0
    for out in generated_token_ids:
        seq = list(out)
        for i, t in enumerate(seq):
            if t in stop:
                seq = seq[:i]
                break
        if any(t in special for t in seq):
            offenders += 1
    if offenders:
        logger.warning(f"[{case_name}] model produced special tokens ({offenders}/{len(generated_token_ids)} users)")
        if is_ci_env:
            assert False, f"model produced special tokens ({offenders} users)"


def _run_dp_smoke(
    mesh_device: ttnn.MeshDevice,
    optimizations: str,
    cache_dir: Path,
    data_parallel: int,
    max_seq_len: int,
    max_gen_tokens: int,
    stop_at_eos: bool,
) -> None:
    """Single-user data-parallel scaling smoke across ``data_parallel`` submeshes.

    Builds one model + one traced executor + one KV cache + one page table per submesh (one user each),
    runs ``run_perf_benchmark`` per submesh sequentially, collects the per-submesh output, and asserts
    no special tokens. Every executor and model is cleaned up in ``finally``.
    """
    _dp_or_skip(mesh_device, data_parallel)
    # Each DP group is a single device (see _dp_or_skip: n // data_parallel == 1). Qwen3-32B cannot run
    # on a single device (needs 8-way TP — see _skip_below_min_tp_devices), so every DP factor is
    # inapplicable for this model: you cannot have both 1-device-per-user AND 8-device TP. Genuine
    # hardware-capacity guard (matches TTTv1's T3K-only support — TTTv1 can't DP a 32B on T3K either).
    _skip_below_min_tp_devices(mesh_device.get_num_devices() // data_parallel)

    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen3-32B")
    _skip_unless_heads_divide_mesh(mesh_device, hf_model)
    tokenizer = _load_tokenizer(hf_model)
    precision = QWEN3_32B_PERFORMANCE if optimizations == "performance" else QWEN3_32B_ACCURACY

    submeshes = create_dp_submeshes(mesh_device, data_parallel)

    # One prompt per DP group (load_input_prompts pads/truncates to the requested count).
    prompts = load_input_prompts(data_parallel)

    sampling_mode = os.environ.get("SAMPLING_MODE", "host").lower()
    _on_device_params = {
        "on_device": SamplingParams(temperature=0.0, top_k=1, top_p=0.0),
        "on_device_topk": SamplingParams(temperature=0.0, top_k=32, top_p=0.08),
    }

    models: list = []
    executors: list = []
    all_generated: list = []
    try:
        for i, sm in enumerate(submeshes):
            try:
                model = Qwen3_32B.from_pretrained(
                    sm,
                    hf_model,
                    max_batch_size=1,
                    max_seq_len=max_seq_len,
                    num_layers=None,
                    cache_dir=cache_dir,
                    precision=precision,
                    executor_mode=True,
                )
            except Exception as e:
                pytest.skip(f"Could not build Qwen3-32B model (weights / memory / mesh): {e}")
            models.append((model, sm))

            traced_executor = TracedQwen3_32BExecutor(model, sm)
            executors.append(traced_executor)

            ma = model.model_args
            assert ma is not None

            block_size = 32
            n_dev_sm = sm.get_num_devices()
            max_num_blocks_per_user = ma.max_seq_len // block_size
            max_num_blocks = max_num_blocks_per_user * ma.max_batch_size  # max_batch_size == 1

            kv_cache_shape = (max_num_blocks, ma.n_kv_heads // n_dev_sm, block_size, ma.head_dim)
            kv_cache = traced_executor.allocate_kv_cache(kv_cache_shape, torch.bfloat16, ma.n_layers)
            page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(
                ma.max_batch_size, max_num_blocks_per_user
            )

            input_tokens, prompt_lens = tokenize_prompts(prompts[i : i + 1], tokenizer)

            sampling_params = (
                _on_device_params[sampling_mode]
                if sampling_mode in _on_device_params and getattr(model, "supports_on_device_sampling", False)
                else None
            )
            logger.info(
                f"[ci-b1-DP-{data_parallel}] submesh {i} SAMPLING_MODE={sampling_mode} "
                f"-> sampling_params={sampling_params}, stop_at_eos={stop_at_eos}"
            )

            result = run_perf_benchmark(
                traced_executor,
                tokens=input_tokens,
                kv_cache=kv_cache,
                page_table=page_table,
                num_decode_tokens=max_gen_tokens,
                max_batch_size=1,
                prompt_lens=prompt_lens,
                sampling_params=sampling_params,
            )
            all_generated.append(result.generated_token_ids[0])
            log_generated_text(prompts[i : i + 1], result.generated_token_ids, tokenizer)

        assert_no_special_tokens(all_generated, tokenizer)
    finally:
        for ex in executors:
            ex.cleanup()
        for model, sm in models:
            cleanup_model_case(model, sm)
        # When data_parallel > 1 we carved child submeshes off the fixture-owned parent mesh. Those
        # submeshes share the parent's command queue, so the parent cannot be closed while they remain
        # in use. Drain the parent + submesh CQs before teardown.
        if data_parallel > 1:
            mesh_device.quiesce_devices()


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.parametrize(
    "test_config",
    [
        pytest.param("token-accuracy", id="token-accuracy"),
        pytest.param("batch-1", id="batch-1"),
        pytest.param("batch-32", id="batch-32"),
        pytest.param("batch-32-ci", id="batch-32-ci"),
        pytest.param("eval-32", id="eval-32"),
        pytest.param("eval-32-perf-report", id="eval-32-perf-report"),
        pytest.param("ci-b1-DP-2", id="ci-b1-DP-2"),
        pytest.param("ci-b1-DP-4", id="ci-b1-DP-4"),
        pytest.param("ci-b1-DP-8", id="ci-b1-DP-8"),
        pytest.param("ci-b1-DP-16", id="ci-b1-DP-16"),
        pytest.param("ci-b1-DP-32", id="ci-b1-DP-32"),
    ],
)
@pytest.mark.parametrize("optimizations", ["performance", "accuracy"])
def test_qwen3_32b(test_config, mesh_device, optimizations):
    """Main test entry for TTTv2 Qwen3-32B."""
    device_name = get_device_name(mesh_device)
    expected = EXPECTED_METRICS.get(optimizations, {}).get(device_name, {})
    model = None
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen3-32B")
    cache_dir = lazy_weight_cache_dir_for_demo(mesh_device, hf_model)

    try:
        # ci-b1-DP-*: single-user data-parallel smoke. Builds N models itself (one per submesh), so it
        # does NOT go through the shared create_model path below.
        if test_config.startswith("ci-b1-DP"):
            data_parallel = int(test_config.rsplit("-", 1)[1])
            sizes = _DP_SIZE_TABLE[data_parallel]
            _run_dp_smoke(
                mesh_device,
                optimizations,
                cache_dir,
                data_parallel=data_parallel,
                max_seq_len=sizes["max_seq_len"],
                max_gen_tokens=sizes["max_generated_tokens"],
                stop_at_eos=sizes["stop_at_eos"],
            )
            return

        if test_config in ("batch-32", "eval-32", "eval-32-perf-report"):
            # Short-context 32-user workload (seq1024). batch-32 is perf-gated; eval-32 is a determinism
            # check (not perf-gated).
            max_bs, max_seq_len = 32, 1024
            expected = EXPECTED_METRICS_BATCH32.get(_sampling_bucket(), {}).get(optimizations, {}).get(device_name, {})
        elif test_config == "batch-32-ci":
            # CI-faithful batch-32 leg (TTTv1 ci-32 parity): larger seq len + 1024 decode budget.
            max_bs = 32
            max_seq_len = _BATCH32_CI_MAX_SEQ_LEN.get(device_name, 2048)
            # Own perf gate measured at the seq2048/decode1024 workload (NOT the lighter batch-32
            # constant, which would be a config-artifact miss). Keyed by SAMPLING_MODE AND profile.
            # Non-topk on-device modes (force-argmax) fall into the on_device_topk bucket; cells not
            # measured fall back to the short-context batch-32 constant. If neither source provides a
            # complete profile-matched floor, the full run remains observational rather than blocked.
            _bucket = _sampling_bucket()
            expected = (
                EXPECTED_METRICS_BATCH32_CI.get(_bucket, {})
                .get(optimizations, {})
                .get(
                    device_name,
                    EXPECTED_METRICS_BATCH32.get(_bucket, {}).get(optimizations, {}).get(device_name, {}),
                )
            )
        else:
            # token-accuracy + batch-1: single-user, seq4096.
            max_bs, max_seq_len = 1, 4096
        model = create_model(
            mesh_device,
            optimizations,
            cache_dir,
            max_batch_size=max_bs,
            max_seq_len=max_seq_len,
        )

        if test_config == "token-accuracy":
            _run_token_accuracy(model, mesh_device, expected)
        elif test_config == "batch-1":
            perf_expected = (
                EXPECTED_METRICS_BATCH1.get(_sampling_bucket(), {}).get(optimizations, {}).get(device_name, {})
            )
            _run_perf_benchmark(model, mesh_device, perf_expected, batch_size=1, case_name=f"{optimizations}/batch-1")
        elif test_config == "batch-32":
            # Natural-length prefill: these sample prompts bucket to 128 (PERF.md Short-Context Batch-32
            # row), matching TTTv1's traced-prefill seq len without a forced pad.
            _run_perf_benchmark(model, mesh_device, expected, batch_size=32, case_name=f"{optimizations}/batch-32")
        elif test_config == "batch-32-ci":
            # CI-faithful leg: seq2048 + 1024 decode tokens (clamped in _run_perf_benchmark). Gated by
            # EXPECTED_METRICS_BATCH32_CI (measured at this workload, TTTv1-parity).
            _run_perf_benchmark(
                model,
                mesh_device,
                expected,
                batch_size=32,
                case_name=f"{optimizations}/batch-32-ci",
                num_decode_tokens=1024,
            )
        elif test_config in ("eval-32", "eval-32-perf-report"):
            # 32-user cross-batch determinism (self-consistency under prompt rotation).
            perf_report = test_config == "eval-32-perf-report"
            eval_expected = _resolve_eval32_perf_targets(hf_model, device_name, optimizations) if perf_report else None
            _run_eval_repeat_batch32(
                model,
                mesh_device,
                expected=eval_expected,
                case_name=f"{optimizations}/{test_config}",
                perf_report=perf_report,
            )
    finally:
        cleanup_model_case(model, mesh_device)


_CROSS_CARDINALITY_REQUEST_IDS = tuple(f"qwen3-32b-request-{index:02d}" for index in range(32))
_CROSS_CARDINALITY_SEEDS = tuple(2_026_081_701 + 104_729 * index for index in range(32))
# Keep the two longest corpus requests last. Prefixes 2 and 4 must contain multiple Q128 requests so
# those cardinalities exercise an actual batched prefill group rather than unrelated buckets.
_CROSS_CARDINALITY_PROMPT_ORDER = (*range(2, 32), 0, 1)
_CROSS_CARDINALITIES = (1, 2, 4, 32)
_CROSS_CARDINALITY_DECODE_TOKENS = 32


def _compare_cross_cardinality_token_ids(
    controls: dict[str, tuple[int, ...]],
    prefixes: dict[int, dict[str, tuple[int, ...]]],
) -> tuple[str, tuple[dict[str, object], ...]]:
    """Return an executed experiment verdict; token mismatch is a valid negative result."""

    expected_requests = set(_CROSS_CARDINALITY_REQUEST_IDS)
    if set(controls) != expected_requests:
        raise AssertionError("cross-cardinality controls must contain all 32 fixed request IDs")
    if tuple(prefixes) != _CROSS_CARDINALITIES:
        raise AssertionError(f"cross-cardinality prefixes must be {_CROSS_CARDINALITIES}")
    expected_token_count = _CROSS_CARDINALITY_DECODE_TOKENS + 1
    bad_controls = {
        request_id: len(token_ids)
        for request_id, token_ids in controls.items()
        if len(token_ids) != expected_token_count
    }
    if bad_controls:
        raise AssertionError(
            f"cross-cardinality controls must each return {expected_token_count} generated tokens: {bad_controls}"
        )

    mismatches = []
    for cardinality, outputs in prefixes.items():
        expected_ids = _CROSS_CARDINALITY_REQUEST_IDS[:cardinality]
        if tuple(outputs) != expected_ids:
            raise AssertionError(f"cardinality {cardinality} did not preserve fixed request order")
        bad_candidates = {
            request_id: len(outputs[request_id])
            for request_id in expected_ids
            if len(outputs[request_id]) != expected_token_count
        }
        if bad_candidates:
            raise AssertionError(
                f"cardinality {cardinality} candidates must each return {expected_token_count} generated tokens: "
                f"{bad_candidates}"
            )
        for request_id in expected_ids:
            expected = controls[request_id]
            actual = outputs[request_id]
            if actual != expected:
                first_difference = next(
                    (index for index, pair in enumerate(zip(expected, actual)) if pair[0] != pair[1]),
                    min(len(expected), len(actual)),
                )
                mismatches.append(
                    {
                        "cardinality": cardinality,
                        "request_id": request_id,
                        "first_token_difference": first_difference,
                        "control_token_count": len(expected),
                        "batched_token_count": len(actual),
                    }
                )
    verdict = "INVARIANT" if not mismatches else "BATCHED_PREFILL_REJECTED"
    return verdict, tuple(mismatches)


def _snapshot_cross_cardinality_prefill(executor, tokens, page_table, prompt_lens) -> tuple[dict[str, object], ...]:
    """Snapshot the same immutable prepared requests that execution will plan."""

    prepared = executor.prefill_runtime.prepare(
        tokens=tokens,
        page_table=page_table[: len(prompt_lens)],
        prompt_lens=prompt_lens,
        empty_slots=list(range(len(prompt_lens))),
        sampling_params=None,
    )
    return tuple(
        {
            "kind": item.request.kind,
            "source_rows": item.request.source_rows,
            "active_batch_size": len(item.request.source_rows),
            "padded_batch_size": item.request.padded_batch_size,
            "padded_sequence_length": item.request.padded_sequence_length,
            "operation_variants": tuple(signature.operation_variant for signature in item.program_signatures),
        }
        for item in prepared
    )


def _require_cross_cardinality_prefill_geometry(
    geometry: tuple[dict[str, object], ...], *, cardinality: int, batched_candidate: bool
) -> None:
    """Fail unless prepared requests prove the intended control/candidate geometry."""

    regular_single = {
        "kind": "single",
        "source_rows": (0,),
        "active_batch_size": 1,
        "padded_batch_size": 1,
        "padded_sequence_length": 128,
        "operation_variants": ("regular-single",),
    }
    if not batched_candidate:
        if (
            len(geometry) != 1
            or geometry[0]["kind"] != "single"
            or geometry[0]["source_rows"] != (0,)
            or geometry[0]["active_batch_size"] != 1
            or geometry[0]["padded_batch_size"] != 1
            or geometry[0]["padded_sequence_length"] not in (128, 1024)
            or geometry[0]["operation_variants"] != ("regular-single",)
        ):
            raise AssertionError(f"batch-1 control must prepare one regular-single request: {geometry}")
        return
    if cardinality == 1:
        if geometry != (regular_single,):
            raise AssertionError(f"cardinality {cardinality} must prepare one regular-single Q128 request: {geometry}")
        return

    if cardinality in (2, 4):
        expected = (
            {
                "kind": "batched",
                "source_rows": tuple(range(cardinality)),
                "active_batch_size": cardinality,
                "padded_batch_size": cardinality,
                "padded_sequence_length": 128,
                "operation_variants": ("regular-batched",),
            },
        )
    elif cardinality == 32:
        expected = (
            {
                "kind": "batched",
                "source_rows": tuple(range(30)),
                "active_batch_size": 30,
                "padded_batch_size": 32,
                "padded_sequence_length": 128,
                "operation_variants": ("regular-batched",),
            },
            {
                "kind": "batched",
                "source_rows": (30, 31),
                "active_batch_size": 2,
                "padded_batch_size": 2,
                "padded_sequence_length": 1024,
                "operation_variants": ("regular-batched",),
            },
        )
    else:
        raise AssertionError(f"unsupported cross-cardinality candidate {cardinality}")
    if geometry != expected:
        raise AssertionError(f"cardinality {cardinality} prepared-prefill geometry disagrees: {geometry}")


def _require_cross_cardinality_environment() -> None:
    conflicts = [name for name in ("DISABLE_BATCHED_PREFILL", "DISABLE_BATCHED_EXTRACT") if name in os.environ]
    if conflicts:
        raise RuntimeError(f"cross-cardinality qualification requires unset environment controls: {conflicts}")


def test_qwen3_32b_p150x4_seeded_cross_cardinality(mesh_device):
    """Compare true batch-1 controls with exact tokens from batched prefixes 1/2/4/32.

    A mismatch is a completed negative experiment, not a missing test: it emits the
    ``BATCHED_PREFILL_REJECTED`` verdict and retains P150x4's sequential-prefill policy. Only an
    invariant result emits ``INVARIANT``; neither verdict silently changes the checked-in policy.
    """
    if get_device_name(mesh_device) != "P150x4":
        pytest.skip("cross-cardinality qualification requires a physical P150x4")

    _require_cross_cardinality_environment()
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen3-32B")
    cache_dir = lazy_weight_cache_dir_for_demo(mesh_device, hf_model)
    model = None
    try:
        model = create_model(
            mesh_device,
            "accuracy",
            cache_dir,
            max_batch_size=32,
            max_seq_len=1024,
        )
        ma = model.model_args
        assert ma is not None
        assert ma.disable_batched_prefill is True, "P150x4 must enter qualification with sequential policy retained"
        assert ma.batched_prefill_batched_extract is True, "batched qualification requires batched last-token extract"

        tokenizer = _load_tokenizer(hf_model)
        corpus_prompts = load_eval_repeat_prompts_batch32()
        prompts = [corpus_prompts[index] for index in _CROSS_CARDINALITY_PROMPT_ORDER]
        assert len(prompts) == len(_CROSS_CARDINALITY_REQUEST_IDS) == 32
        block_size = 32
        blocks_per_user = ma.max_seq_len // block_size
        num_blocks = blocks_per_user * ma.max_batch_size
        page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(ma.max_batch_size, blocks_per_user)
        kv_cache_shape = (
            num_blocks,
            ma.n_kv_heads // mesh_device.get_num_devices(),
            block_size,
            ma.head_dim,
        )

        def make_executor(*, expected_disable_batched_prefill):
            executor = TracedQwen3_32BExecutor(
                model,
                mesh_device,
                ondevice_decode_loop=True,
                # Prefill stays eager, isolating cardinality, while decode trace is a silicon canary
                # for production's per-request seed refresh. Reuse limits the test to two captures.
                trace_mode=eval_decode_trace_mode("traced"),
            )
            assert (
                executor.prefill_runtime.config.disable_batched_prefill is expected_disable_batched_prefill
            ), "executor prefill policy snapshot disagrees with the requested experiment arm"
            kv_cache = executor.allocate_kv_cache(kv_cache_shape, torch.bfloat16, ma.n_layers)
            return executor, kv_cache

        def prepare_requests(executor, request_prompts, request_seeds, *, batched_candidate):
            input_tokens, prompt_lens = tokenize_prompts(request_prompts, tokenizer)
            if len(request_seeds) > 1:
                q128_group = prompt_lens[: min(4, len(request_seeds))]
                if not all(0 < int(length) <= 128 for length in q128_group):
                    raise RuntimeError(
                        "cross-cardinality prompt order must keep the first 2/4 requests in one Q128 batch"
                    )
            sampling_params = SamplingParams(
                temperature=[0.8] * len(request_seeds),
                top_k=[32] * len(request_seeds),
                top_p=[0.95] * len(request_seeds),
                seed=list(request_seeds),
            )
            geometry = _snapshot_cross_cardinality_prefill(executor, input_tokens, page_table, prompt_lens)
            _require_cross_cardinality_prefill_geometry(
                geometry,
                cardinality=len(request_seeds),
                batched_candidate=batched_candidate,
            )
            return input_tokens, prompt_lens, sampling_params, geometry

        def compile_prefill_case(executor, kv_cache, prepared_case):
            input_tokens, prompt_lens, _sampling_params, _geometry = prepared_case
            executor.compile_prefill(
                tokens=input_tokens,
                page_table=page_table[: len(prompt_lens)],
                kv_cache=kv_cache,
                prompt_lens=prompt_lens,
                empty_slots=list(range(len(prompt_lens))),
                sampling_params=None,
            )

        def activate_decode_trace(executor, kv_cache):
            assert executor.config.warmup.include_decode_top_k is True
            decode_kwargs = {
                "kv_cache": kv_cache,
                "max_batch_size": ma.max_batch_size,
                "num_blocks": page_table.shape[-1],
                "can_sample_on_device": True,
            }
            # Register eager decode programs (including the representative top-k alias), then
            # register and capture the same decode coverage exactly once. Prefill remains eager.
            executor.warmup_model_decode(enable_trace=False, **decode_kwargs)
            executor.warmup_model_decode(enable_trace=True, **decode_kwargs)
            compiler = executor.trace_compiler
            traced = executor.traced_executor
            assert compiler is not None and traced is not None
            coverage = compiler.registered_coverage("decode")
            assert executor.warmup.trace_activated is True
            assert compiler.trace_active is True
            assert compiler.trace_count == len(coverage) >= 1
            records = tuple(compiler.get(trace_key) for trace_key, _signature in coverage)
            assert all(record is not None and record.artifact is not None for record in records)
            topk_coverage = tuple(
                (trace_key, signature) for trace_key, signature in coverage if signature.sampling_path == "topk"
            )
            assert len(topk_coverage) == 1
            topk_trace_key, _topk_signature = topk_coverage[0]
            assert compiler.get(topk_trace_key).artifact is not None
            assert compiler.trace_association_count >= 1
            assert compiler.replay_count == 0
            assert traced.coverage_miss_count == 0
            return {
                "semantic_trace_count": compiler.trace_count,
                "trace_association_count": compiler.trace_association_count,
                "captured_decode_trace_count": len(coverage),
                "captured_topk_trace_count": len(topk_coverage),
                "topk_trace_key": topk_trace_key.digest,
                "trace_active": compiler.trace_active,
                "replay_count_before_requests": compiler.replay_count,
            }, topk_trace_key

        def run_requests(executor, kv_cache, prepared_case, *, expected_topk_trace_key, expected_semantic_trace_count):
            input_tokens, prompt_lens, sampling_params, geometry = prepared_case
            compiler = executor.trace_compiler
            traced = executor.traced_executor
            assert compiler is not None and traced is not None and compiler.trace_active
            prepared_decode = executor.decode_runtime.prepare(
                torch.zeros(ma.max_batch_size, dtype=torch.long),
                torch.zeros(ma.max_batch_size, dtype=torch.long),
                page_table,
                sampling_params=sampling_params,
                reset_batch=True,
            )
            assert prepared_decode.sampling_path == "topk"
            decode_program_key = executor.program_compiler.key_for(
                executor.decode_runtime.program_signature(prepared_decode)
            )
            assert compiler.trace_key_for_program(decode_program_key) == expected_topk_trace_key
            assert compiler.get(expected_topk_trace_key).artifact is not None
            replay_before = compiler.replay_count
            decode_replays_before = compiler.replay_counts["decode"]
            result = run_perf_benchmark(
                executor,
                tokens=input_tokens,
                kv_cache=kv_cache,
                page_table=page_table,
                num_decode_tokens=_CROSS_CARDINALITY_DECODE_TOKENS,
                max_batch_size=ma.max_batch_size,
                prompt_lens=prompt_lens,
                sampling_params=sampling_params,
                prefill_sampling_params=None,
            )
            generated = tuple(tuple(int(token) for token in output) for output in result.generated_token_ids)
            if len(generated) != len(prompt_lens):
                raise AssertionError(
                    f"cardinality {len(prompt_lens)} returned {len(generated)} outputs before token comparison"
                )
            replay_delta = compiler.replay_count - replay_before
            decode_replay_delta = compiler.replay_counts["decode"] - decode_replays_before
            if replay_delta != _CROSS_CARDINALITY_DECODE_TOKENS or decode_replay_delta != replay_delta:
                raise AssertionError(
                    f"cardinality {len(prompt_lens)} expected {_CROSS_CARDINALITY_DECODE_TOKENS} decode trace "
                    f"replays, observed total={replay_delta}, decode={decode_replay_delta}"
                )
            assert compiler.replay_counts["prefill"] == 0
            assert compiler.trace_count == expected_semantic_trace_count and compiler.trace_active
            assert compiler.get(expected_topk_trace_key).artifact is not None
            assert traced.coverage_miss_count == 0
            assert executor.program_compiler.post_activation_compile_rejections == 0
            return (
                generated,
                geometry,
                {
                    "cardinality": len(prompt_lens),
                    "decode_trace_replays": decode_replay_delta,
                    "trace_key": expected_topk_trace_key.digest,
                    "coverage_misses": traced.coverage_miss_count,
                    "post_activation_compile_rejections": executor.program_compiler.post_activation_compile_rejections,
                },
            )

        controls = {}
        control_geometry = []
        sequential_executor, sequential_kv_cache = make_executor(expected_disable_batched_prefill=True)
        try:
            control_cases = [
                prepare_requests(sequential_executor, [prompt], [seed], batched_candidate=False)
                for prompt, seed in zip(prompts, _CROSS_CARDINALITY_SEEDS, strict=True)
            ]
            # Decode trace activation seals the shared program compiler. Register every eager
            # prefill signature first so later controls cannot request unseen programs.
            for prepared_case in control_cases:
                compile_prefill_case(sequential_executor, sequential_kv_cache, prepared_case)
            control_trace_lifecycle, control_topk_trace_key = activate_decode_trace(
                sequential_executor, sequential_kv_cache
            )
            control_replay_evidence = []
            for request_id, prepared_case in zip(_CROSS_CARDINALITY_REQUEST_IDS, control_cases, strict=True):
                generated, geometry, replay_evidence = run_requests(
                    sequential_executor,
                    sequential_kv_cache,
                    prepared_case,
                    expected_topk_trace_key=control_topk_trace_key,
                    expected_semantic_trace_count=control_trace_lifecycle["semantic_trace_count"],
                )
                (controls[request_id],) = generated
                control_geometry.append(geometry)
                control_replay_evidence.append(replay_evidence)
            control_trace_lifecycle["replay_count_after_requests"] = sequential_executor.trace_compiler.replay_count
            assert control_trace_lifecycle["replay_count_after_requests"] == (
                len(_CROSS_CARDINALITY_REQUEST_IDS) * _CROSS_CARDINALITY_DECODE_TOKENS
            )
        finally:
            sequential_executor.cleanup()

        prefixes = {}
        candidate_geometry = {}
        ma.disable_batched_prefill = False
        try:
            candidate_executor, candidate_kv_cache = make_executor(expected_disable_batched_prefill=False)
            try:
                candidate_cases = {
                    cardinality: prepare_requests(
                        candidate_executor,
                        prompts[:cardinality],
                        _CROSS_CARDINALITY_SEEDS[:cardinality],
                        batched_candidate=True,
                    )
                    for cardinality in _CROSS_CARDINALITIES
                }
                for prepared_case in candidate_cases.values():
                    compile_prefill_case(candidate_executor, candidate_kv_cache, prepared_case)
                candidate_trace_lifecycle, candidate_topk_trace_key = activate_decode_trace(
                    candidate_executor, candidate_kv_cache
                )
                candidate_replay_evidence = []
                for cardinality, prepared_case in candidate_cases.items():
                    generated, geometry, replay_evidence = run_requests(
                        candidate_executor,
                        candidate_kv_cache,
                        prepared_case,
                        expected_topk_trace_key=candidate_topk_trace_key,
                        expected_semantic_trace_count=candidate_trace_lifecycle["semantic_trace_count"],
                    )
                    candidate_geometry[cardinality] = geometry
                    candidate_replay_evidence.append(replay_evidence)
                    prefixes[cardinality] = {
                        request_id: tokens
                        for request_id, tokens in zip(
                            _CROSS_CARDINALITY_REQUEST_IDS[:cardinality], generated, strict=True
                        )
                    }
                candidate_trace_lifecycle[
                    "replay_count_after_requests"
                ] = candidate_executor.trace_compiler.replay_count
                assert candidate_trace_lifecycle["replay_count_after_requests"] == (
                    len(_CROSS_CARDINALITIES) * _CROSS_CARDINALITY_DECODE_TOKENS
                )
            finally:
                candidate_executor.cleanup()
        finally:
            ma.disable_batched_prefill = True

        verdict, mismatches = _compare_cross_cardinality_token_ids(controls, prefixes)
        logger.info(
            "QWEN3_32B_CROSS_CARDINALITY_VERDICT="
            + json.dumps(
                {
                    "verdict": verdict,
                    "policy": "sequential",
                    "control_runs": len(controls),
                    "batched_cardinalities": list(_CROSS_CARDINALITIES),
                    "decode_tokens": _CROSS_CARDINALITY_DECODE_TOKENS,
                    "comparison": "exact_token_ids",
                    "execution": "eager_prefill_decode_traced",
                    "control_prefill_geometry": control_geometry,
                    "candidate_prefill_geometry": candidate_geometry,
                    "control_trace_lifecycle": control_trace_lifecycle,
                    "candidate_trace_lifecycle": candidate_trace_lifecycle,
                    "control_replay_evidence": control_replay_evidence,
                    "candidate_replay_evidence": candidate_replay_evidence,
                    "mismatch_count": len(mismatches),
                    "mismatches": list(mismatches),
                },
                sort_keys=True,
            )
        )
        assert ma.disable_batched_prefill is True, "qualification must retain sequential P150x4 policy"
    finally:
        cleanup_model_case(model, mesh_device)


def _run_token_accuracy(model, mesh_device, expected):
    """Teacher-forcing token accuracy vs ``.refpt`` (HF-generated)."""
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen3-32B")
    reference_tokens, top5_tokens, prompt_len, metadata = load_reference_data(hf_model)
    tokenizer = _load_tokenizer(hf_model)

    if reference_tokens.dim() > 1:
        reference_tokens = reference_tokens.squeeze()

    has_prompt_len_metadata = prompt_len is not None
    if has_prompt_len_metadata:
        prompt_len = int(prompt_len)
        logger.info(f"Using metadata-driven prompt_len={prompt_len} from reference artifact")
    else:
        prompt_len = len(reference_tokens) // 2
        logger.warning(f"Reference missing prompt_len metadata; falling back to legacy half split={prompt_len}")

    if metadata:
        meta_summary = {
            "hf_model_id": metadata.get("hf_model_id"),
            "revision": metadata.get("revision"),
            "generation_mode": metadata.get("generation_mode"),
            "created_at": metadata.get("created_at"),
        }
        logger.info(f"Reference metadata summary: {meta_summary}")

    prompt_tokens = reference_tokens[:prompt_len].unsqueeze(0)

    executor = EagerQwen3_32BExecutor(model, mesh_device)
    ma = model.model_args
    assert ma is not None

    max_batch_size = ma.max_batch_size
    prompt_tokens = prompt_tokens.repeat(max_batch_size, 1)
    max_seq_len = ma.max_seq_len
    block_size = 32
    max_num_blocks_per_user = max_seq_len // block_size
    max_num_blocks = max_num_blocks_per_user * max_batch_size

    kv_cache_shape = (max_num_blocks, ma.n_kv_heads // mesh_device.get_num_devices(), block_size, ma.head_dim)
    kv_cache = executor.allocate_kv_cache(kv_cache_shape, torch.bfloat16, ma.n_layers)
    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(max_batch_size, max_num_blocks_per_user)

    target_top5 = select_teacher_forcing_top5_slice(
        top5_tokens,
        reference_tokens,
        prompt_len,
        metadata_aligned=has_prompt_len_metadata,
    )
    is_ci_env = os.environ.get("CI") == "true"
    profiler = BenchmarkProfiler()
    profiler.start("run")
    # run_teacher_forcing times the prefill + per-step (teacher-forced) decode loop and, given the
    # profiler, brackets the "inference_prefill" / "inference_decode" steps itself — so the returned
    # result carries prefill/decode throughput alongside accuracy for CI benchmark-data emission.
    result = run_teacher_forcing(
        executor,
        prompt_tokens=prompt_tokens,
        reference_tokens=reference_tokens,
        top5_tokens=target_top5,
        kv_cache=kv_cache,
        page_table=page_table,
        max_batch_size=max_batch_size,
        profiler=profiler,
    )
    profiler.end("run")

    top1 = result.top1_accuracy() * 100
    top5 = result.top5_accuracy() * 100

    logger.info(
        f"Token accuracy — top1: {top1:.1f}%, top5: {top5:.1f}% | "
        f"TTFT: {result.ttft_ms:.1f}ms, decode: {result.decode_tok_s_u:.1f} tok/s/u"
    )
    log_teacher_forcing_text(prompt_tokens, result.predicted_tokens_per_user, reference_tokens[prompt_len:], tokenizer)

    # CI-dashboard telemetry: emit a ``demo_accuracy`` partial mirroring TTTv1 simple_text_demo.py
    # — the FULL perf measurement set (prefill_t/s, prefill_time_to_token, decode_t/s, decode_t/s/u)
    # PLUS top1/top5, all from this timed teacher-forcing run. create_benchmark_data /
    # save_partial_run_json are no-ops unless CI == "true" (they guard on it internally); the
    # is_ci_env guard here keeps the import/attr access off the local path too. Saved BEFORE the
    # accuracy asserts so telemetry is captured even when the gate later fails.
    if is_ci_env:
        num_target = len(reference_tokens) - prompt_len
        measurements = {
            "prefill_t/s": result.prefill_tok_s,
            "prefill_time_to_token": result.prefill_time_to_token_s,  # seconds (TTTv1 units)
            "decode_t/s": result.decode_tok_s,
            "decode_t/s/u": result.decode_tok_s_u,
        }
        benchmark_data = create_benchmark_data(
            profiler, measurements, {"inference_prefill": 0, "inference_decode": 1}, targets={}
        )
        benchmark_data.add_measurement(profiler, 0, "inference_decode", "top1_token_accuracy", top1, target=None)
        benchmark_data.add_measurement(profiler, 0, "inference_decode", "top5_token_accuracy", top5, target=None)
        benchmark_data.save_partial_run_json(
            profiler,
            run_type="demo_accuracy",
            ml_model_name=hf_model,
            ml_model_type="llm",
            device_name=get_device_name(mesh_device),
            num_layers=ma.n_layers,
            batch_size=1,
            input_sequence_length=prompt_len,
            output_sequence_length=num_target,
        )

    # Accuracy gate — threshold SOURCE is flag-controlled (currently ``is_ci_env``):
    #   use_centralized_targets = True  → mirror TTTv1: pull centralized targets via
    #       resolve_accuracy_targets and subtract an ABSOLUTE 0.5 pp (get_accuracy_thresholds,
    #       simple_text_demo.py). Missing entry is a hard error (never silently un-gate in CI).
    #   use_centralized_targets = False → use the demo's local EXPECTED_METRICS values DIRECTLY
    #       (no ratio tolerance — TTTv1 applies none to accuracy).
    # Measured accuracy is rounded up with math.ceil before the compare, matching TTTv1 exactly
    # (simple_text_demo.py:1657-1658, ``math.ceil(acc[...] * 100)``).
    device_name = get_device_name(mesh_device)
    # P150x4 is a qualification gate even outside CI; use the checked-in p300x2/bh_quietbox_2
    # targets rather than silently accepting the absent local metric bucket.
    use_centralized_targets = is_ci_env or device_name == "P150x4"
    if use_centralized_targets:
        central = resolve_accuracy_targets(hf_model, device_name, batch_size=1, seq_len=512)
        if not central or "top1" not in central or "top5" not in central:
            raise ValueError(
                f"No centralized accuracy target for {hf_model} on {device_name} "
                "(batch_size=1, seq_len=512); add an entry to models/model_targets.yaml."
            )
        min_top1 = float(central["top1"]) - 0.5
        min_top5 = float(central["top5"]) - 0.5
    else:
        min_top1 = float(expected.get("top1", 0))
        min_top5 = float(expected.get("top5", 0))

    # math.ceil matches TTTv1's integer-rounded accuracy check (simple_text_demo.py:1657-1658).
    meas_top1 = math.ceil(top1)
    meas_top5 = math.ceil(top5)
    assert meas_top1 >= min_top1, f"Top-1 accuracy {top1:.1f}% (ceil {meas_top1}) below threshold {min_top1:.1f}%"
    assert meas_top5 >= min_top5, f"Top-5 accuracy {top5:.1f}% (ceil {meas_top5}) below threshold {min_top5:.1f}%"


def _run_perf_benchmark(
    model,
    mesh_device,
    expected,
    batch_size,
    case_name,
    max_prefill_len: int | None = None,
    num_decode_tokens: int | None = None,
):
    """Timed prefill + decode (``TracedQwen3_32BExecutor``).

    Prefill uses each prompt's natural token length (TTTv1 ``preprocess_inputs_prefill`` semantics — the
    executor buckets to ``get_padded_prefill_len``); decode runs for ``num_decode_tokens`` steps
    (default ``_PERF_NUM_DECODE_TOKENS``). ``max_prefill_len`` is an optional clip cap for over-long
    prompts, never a pad-up target.

    The decode budget is clamped to what the paged KV cache can hold:
    ``effective = min(requested, max_seq_len - prompt_bucket - margin)`` so the high-water decode
    position never overruns the page table (the ``batch-32-ci`` leg requests 1024).
    """
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen3-32B")
    tokenizer = _load_tokenizer(hf_model)

    # On-device sampling toggle (see the rebase / sampling handoff docs):
    #   host            -> sampling_params=None (host-argmax; slow — full-vocab all-gather + PCIe
    #                      readback every step; NOT comparable to TTTv1)
    #   on_device       -> greedy temp=0,k=1,p=0 => trace-captured FORCE-ARGMAX full-vocab path
    #   on_device_topk  -> temp=0,k=32,p=0.08    => trace-captured TOP-K op path (gathers only the
    #                      [*,32] tuples; PERF.md-parity recipe, faster on >=8-dev meshes)
    # DEFAULT is on_device_topk: on T3K (8 devices) the vocab shards 8-ways and TTTv1 auto-uses
    # on-device sampling, so this is the apples-to-apples TTTv1-comparable path the gate measures.
    sampling_mode = os.environ.get("SAMPLING_MODE", "on_device_topk").lower()
    _on_device_params = {
        "on_device": SamplingParams(temperature=0.0, top_k=1, top_p=0.0),
        "on_device_topk": SamplingParams(temperature=0.0, top_k=32, top_p=0.08),
    }
    sampling_params = (
        _on_device_params[sampling_mode]
        if sampling_mode in _on_device_params and getattr(model, "supports_on_device_sampling", False)
        else None
    )
    logger.info(f"[{case_name}] SAMPLING_MODE={sampling_mode} -> sampling_params={sampling_params}")

    # Batched-prefill A/B knob (parity caveat #12): set DISABLE_BATCHED_PREFILL=1 to force the
    # sequential per-user prefill loop (the pre-feature baseline) for before/after TTFT comparison.
    if os.environ.get("DISABLE_BATCHED_PREFILL") and model.model_args is not None:
        model.model_args.disable_batched_prefill = True

    # Free-running perf run: enable the executor's on-device decode loop on the on-device sampling
    # path (inert on host/force-argmax; gated to the top-k path by _decode_loop_active). Mirrors
    # llama32_1b — removes the per-step host round-trip so decode stays on-device.
    traced_executor = TracedQwen3_32BExecutor(model, mesh_device, ondevice_decode_loop=sampling_params is not None)
    try:
        ma = model.model_args
        assert ma is not None

        block_size = 32
        max_seq_len = ma.max_seq_len
        max_batch_size = ma.max_batch_size
        max_num_blocks_per_user = max_seq_len // block_size
        max_num_blocks = max_num_blocks_per_user * max_batch_size

        kv_cache_shape = (max_num_blocks, ma.n_kv_heads // mesh_device.get_num_devices(), block_size, ma.head_dim)
        kv_cache = traced_executor.allocate_kv_cache(kv_cache_shape, torch.bfloat16, ma.n_layers)
        page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(max_batch_size, max_num_blocks_per_user)

        # Decode-token budget, clamped to the KV-cache headroom. Prompts bucket to ~128 and we keep a
        # 16-token margin, so the high-water decode position stays inside max_seq_len.
        _PROMPT_BUCKET = 128
        _DECODE_MARGIN = 16
        requested_decode = _PERF_NUM_DECODE_TOKENS if num_decode_tokens is None else num_decode_tokens
        effective_decode = min(requested_decode, max_seq_len - _PROMPT_BUCKET - _DECODE_MARGIN)
        logger.info(
            f"[{case_name}] num_decode_tokens: requested={requested_decode}, "
            f"effective={effective_decode} (max_seq_len={max_seq_len})"
        )

        prompts = load_input_prompts(batch_size)
        # Natural-length tokenization (matches TTTv1): the executor buckets each user's real length to
        # get_padded_prefill_len. These sample prompts are ~90-125 tokens -> 128 bucket.
        input_tokens, prompt_lens = tokenize_prompts(prompts, tokenizer, max_prefill_len=max_prefill_len)

        # Register the concrete prompt signature and capture configured traces before the shared
        # benchmark runner attempts its first traced replay.  In particular, a natural Q128 prompt
        # may end in any 32-token tile; compiling through the traced target associates that exact
        # tile program with the sampling-independent Q128 trace captured by this warmup barrier.
        _warmup_demo_executor(
            traced_executor,
            kv_cache=kv_cache,
            page_table=page_table,
            prefill_compile_case=(input_tokens, prompt_lens),
            prefill_sampling_params=sampling_params,
            prefill_compile_execution=traced_executor.traced_prefill_execution,
        )

        # BenchmarkProfiler brackets the timed prefill/decode regions inside run_perf_benchmark
        # (default-None ⇒ byte-inert for every other caller) so we can emit CI perf telemetry.
        is_ci_env = os.environ.get("CI") == "true"
        profiler = BenchmarkProfiler()
        profiler.start("run")
        result = run_perf_benchmark(
            traced_executor,
            tokens=input_tokens,
            kv_cache=kv_cache,
            page_table=page_table,
            num_decode_tokens=effective_decode,
            max_batch_size=max_batch_size,
            prompt_lens=prompt_lens,
            sampling_params=sampling_params,
            profiler=profiler,
        )
        profiler.end("run")

        logger.info(
            f"Performance [{case_name}] — TTFT: {result.ttft_ms:.1f}ms, "
            f"tok/s/u: {result.tok_s_u:.1f}, "
            f"tok/s: {result.tok_s:.1f}, "
            f"decode latency: {result.decode_latency_mean_ms:.2f}ms"
        )
        log_generated_text(prompts, result.generated_token_ids, tokenizer)

        # CI-dashboard telemetry: emit a ``demo_perf`` partial mirroring TTTv1 simple_text_demo.py.
        # Saved BEFORE the special-token guard and perf gate so telemetry is captured even when a
        # downstream assert fails. No-op unless CI == "true" (BenchmarkData guards on it).
        if is_ci_env:
            prefill_seq_len = int(prompt_lens.max())
            prefill_time_s = result.prefill_time_s
            measurements = {
                "prefill_t/s": (result.batch_size * prefill_seq_len) / prefill_time_s if prefill_time_s > 0 else 0.0,
                "prefill_time_to_token": prefill_time_s / result.batch_size,  # seconds (TTTv1 units)
                "decode_t/s": result.tok_s,
                "decode_t/s/u": result.tok_s_u,
            }
            benchmark_data = create_benchmark_data(
                profiler, measurements, {"inference_prefill": 0, "inference_decode": 1}, targets={}
            )
            benchmark_data.save_partial_run_json(
                profiler,
                run_type="demo_perf",
                ml_model_name=hf_model,
                ml_model_type="llm",
                device_name=get_device_name(mesh_device),
                num_layers=ma.n_layers,
                batch_size=result.batch_size,
                input_sequence_length=prefill_seq_len,
                output_sequence_length=effective_decode,
            )

        assert_no_special_tokens(result.generated_token_ids, tokenizer, case_name=case_name)

        # A complete, profile-matched floor is an acceptance gate.  A missing floor must not prevent
        # characterization: the workload above still executes and reports all metrics, but no partial
        # or self-derived threshold is applied.
        expected = _resolve_local_perf_floor(get_device_name(mesh_device), expected, case_name=case_name)

        if expected:
            _assert_local_perf_target(result, expected, case_name=case_name)
    finally:
        traced_executor.cleanup()


# ci-eval-32 determinism case: 3 rotated repeats of the batch-32 workload.
_EVAL_REPEAT_BATCHES = 3
_EVAL_NUM_DECODE_TOKENS = _PERF_NUM_DECODE_TOKENS
_EVAL_PERF_TRACE_PREFILL_BUCKETS = (128, 1024)


def _require_eval_perf_prefill_trace_parity(model_args) -> None:
    """Validate model-owned trace coverage and the BH eval-report batching policy.

    The determinism-only eval intentionally remains decode-only. The separately named
    performance-report leg compares against TTTv1 ``performance-ci-eval-32`` and replays captured
    prefill for both natural prompt buckets; its target-bearing BH path must also preserve the
    model-owned sequential policy. Fail closed rather than silently timing eager prefill when the
    model was constructed with insufficient context or incomplete model-owned trace coverage.
    """
    required_buckets = _EVAL_PERF_TRACE_PREFILL_BUCKETS
    coverage_ceiling = min(int(model_args.max_prefill_chunk_size), int(model_args.max_seq_len))
    if coverage_ceiling < max(required_buckets):
        raise ValueError(
            "eval-32-perf-report requires 128/1024 prefill trace coverage; "
            f"constructed context ceiling is {coverage_ceiling}"
        )

    # TTTv1's BH policy and the failed cross-cardinality qualification both require active-batch-1
    # prefill. Validate that construction supplied this model-owned policy; do not mutate the shared
    # model configuration or change the established T3K batching policy from the demo.
    num_devices = int(model_args.cluster_shape[0]) * int(model_args.cluster_shape[1])
    if num_devices == 4 and not model_args.disable_batched_prefill:
        raise RuntimeError("eval-32-perf-report requires model-owned sequential prefill on P150x4")

    advertised_buckets = tuple(getattr(model_args, "trace_prefill_supported_seq_lens", ()))
    if not set(required_buckets).issubset(advertised_buckets):
        raise ValueError(
            "eval-32-perf-report requires model-owned prefill trace buckets "
            f"{required_buckets}, got {advertised_buckets}"
        )
    if not all(model_args.can_enable_trace(bucket, num_cached_tokens=0) for bucket in required_buckets):
        raise RuntimeError("eval-32-perf-report model predicate rejects required prefill trace coverage")


def _run_eval_repeat_batch32(
    model,
    mesh_device,
    *,
    expected: dict | None = None,
    case_name: str = "eval-32",
    perf_report: bool = False,
):
    """32-user cross-batch determinism (self-consistency under prompt rotation).

    Runs the batch-32 prefill+decode loop ``_EVAL_REPEAT_BATCHES`` times, rotating the prompt->slot
    assignment by one each repeat (fresh traced executor + KV cache per repeat), then asserts that
    undoing the rotation lines up per-user outputs. No external golden. Honors the same ``SAMPLING_MODE``
    knob as ``_run_perf_benchmark`` (default host argmax — deterministic and mesh-agnostic, the
    recommended default for the determinism assert).

    Use the default (host argmax) for the determinism gate. Under ``SAMPLING_MODE=on_device_topk`` the
    accuracy profile's degenerate numeric-prompt continuations produce near-exact logit ties, and the
    on-device sampler's tie-break is slot-dependent (reduction order over the sharded vocab) → the
    cross-batch consistency assert can fail on those rotated slots. That is a property of on-device
    top-k sampling on tie-heavy degenerate output, NOT a determinism regression: host argmax passes
    both profiles with batched prefill ON and OFF, and the on-device failure is identical ON vs OFF
    (prefill-independent, so unrelated to batched prefill). See the port worklog + backlog.
    """
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen3-32B")
    tokenizer = _load_tokenizer(hf_model)
    require_canonical_eval_modes_in_ci(os.environ)

    # Qwen3 chat generation ends at <|im_end|>; the model opening a NEW turn (<|im_start|>) is a de-facto
    # response terminator as well (Qwen serving stacks list both as stops), but Qwen's HF
    # generation_config only carries <|im_end|>/<|endoftext|> as eos. Augment the tokenizer stop set (the
    # mechanism ``hf_stop_ids`` reads) with <|im_start|> so the determinism runner truncates a degenerate
    # turn-restart there — same pattern as the qwen25_7b / llama1b guards. Without this, a fixed-budget
    # greedy continuation of the numeric eval prompts can degenerate into "\n<|im_start|>user" (a
    # hallucinated new turn) deep in decode; which of the two equally-valid prefill numerics (batched vs
    # sequential) hits it is a near-tie, so the shared garbage guard would otherwise flag only one leg.
    # <|im_start|> is a legitimate response terminator, so truncating there is correct, not a loosening;
    # cross-batch consistency is still asserted on the truncated (real-response) tokens.
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    if isinstance(im_start_id, int) and im_start_id >= 0:
        existing = list(getattr(tokenizer, "stop_tokens", None) or [])
        tokenizer.stop_tokens = list({*existing, im_start_id})

    ma = model.model_args
    assert ma is not None

    if perf_report:
        _require_eval_perf_prefill_trace_parity(ma)

    # Batched-prefill A/B knob (parity caveat #12): DISABLE_BATCHED_PREFILL=1 forces the pure per-bucket
    # sequential prefill so eval-32 can be validated both ON and OFF.
    if os.environ.get("DISABLE_BATCHED_PREFILL"):
        ma.disable_batched_prefill = True

    block_size = 32
    max_seq_len = ma.max_seq_len
    max_batch_size = ma.max_batch_size
    max_num_blocks_per_user = max_seq_len // block_size
    max_num_blocks = max_num_blocks_per_user * max_batch_size

    kv_cache_shape = (max_num_blocks, ma.n_kv_heads // mesh_device.get_num_devices(), block_size, ma.head_dim)
    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(max_batch_size, max_num_blocks_per_user)

    # TTTv1 ci-eval-32 numeric prompts (parity).
    prompts = load_eval_repeat_prompts_batch32()

    def tokenize_fn(ps):
        return tokenize_prompts(ps, tokenizer)

    # Determinism-only eval defaults to host argmax. The perf-report parity leg defaults to TTTv1's
    # on-device top-k path so its checked-in bh_quietbox_2 targets compare the same sampling topology.
    default_sampling_mode = "on_device_topk" if perf_report else "host"
    sampling_mode = os.environ.get("SAMPLING_MODE", default_sampling_mode).lower()
    _on_device_params = {
        "on_device": SamplingParams(temperature=0.0, top_k=1, top_p=0.0),
        "on_device_topk": SamplingParams(temperature=0.0, top_k=32, top_p=0.08),
    }
    sampling_params = (
        _on_device_params[sampling_mode]
        if sampling_mode in _on_device_params and getattr(model, "supports_on_device_sampling", False)
        else None
    )
    representative_prefill = tokenize_fn(prompts)
    logger.info(f"[eval-32] SAMPLING_MODE={sampling_mode} -> sampling_params={sampling_params}")

    # Fresh traced executor + zeroed KV cache per repeat (driver owns the lifecycle), so the rotated
    # batches are fully independent — see run_eval_repeat_batch32 for why reuse corrupts the 3rd repeat.
    def make_executor():
        return TracedQwen3_32BExecutor(
            model,
            mesh_device,
            ondevice_decode_loop=sampling_params is not None,
            trace_mode=("all" if perf_report else eval_decode_trace_mode(os.environ.get("EVAL_DECODE_MODE", "traced"))),
        )

    def allocate_kv_cache(executor):
        kv_cache = executor.allocate_kv_cache(kv_cache_shape, torch.bfloat16, ma.n_layers)
        _warmup_demo_executor(
            executor,
            kv_cache=kv_cache,
            page_table=page_table,
            prefill_compile_case=representative_prefill,
            prefill_sampling_params=sampling_params,
            # Full-trace replay requires the exact concrete program alias to be registered before
            # `_warmup_demo_executor` crosses the capture barrier. Decode-only determinism keeps its
            # established eager compile path.
            prefill_compile_execution=executor.traced_prefill_execution if perf_report else None,
        )
        return kv_cache

    profiler = BenchmarkProfiler() if perf_report else None
    if profiler is not None:
        profiler.start("run")
    try:
        first_result = run_eval_repeat_batch32(
            make_executor=make_executor,
            allocate_kv_cache=allocate_kv_cache,
            page_table=page_table,
            prompts=prompts,
            tokenizer=tokenizer,
            tokenize_fn=tokenize_fn,
            num_decode_tokens=_EVAL_NUM_DECODE_TOKENS,
            max_batch_size=max_batch_size,
            sampling_params=sampling_params,
            repeat_batches=_EVAL_REPEAT_BATCHES,
            hf_model_id=hf_model,
            first_repeat_profiler=profiler,
            page_table_mode=os.environ.get("EVAL_PAGE_TABLE_MODE", "slot-stable"),
        )
    finally:
        if profiler is not None:
            profiler.end("run")

    if not perf_report:
        return first_result

    logger.info(
        f"Performance [{case_name}, first of {_EVAL_REPEAT_BATCHES} repeats] — "
        f"TTFT: {first_result.ttft_ms:.1f}ms, tok/s/u: {first_result.tok_s_u:.1f}, "
        f"tok/s: {first_result.tok_s:.1f}"
    )
    if os.environ.get("CI") == "true":
        prefill_seq_len = int(representative_prefill[1].max())
        measurements = {
            "prefill_t/s": (
                first_result.batch_size * prefill_seq_len / first_result.prefill_time_s
                if first_result.prefill_time_s > 0
                else 0.0
            ),
            "prefill_time_to_token": first_result.prefill_time_s / first_result.batch_size,
            "decode_t/s": first_result.tok_s,
            "decode_t/s/u": first_result.tok_s_u,
        }
        benchmark_data = create_benchmark_data(
            profiler,
            measurements,
            {"inference_prefill": 0, "inference_decode": 1},
            targets={},
        )
        benchmark_data.save_partial_run_json(
            profiler,
            run_type="demo_perf",
            ml_model_name=hf_model,
            ml_model_type="llm",
            device_name=get_device_name(mesh_device),
            num_layers=ma.n_layers,
            batch_size=first_result.batch_size,
            config_params={"optimization_profile": case_name.split("/", 1)[0]},
            input_sequence_length=prefill_seq_len,
            output_sequence_length=_EVAL_NUM_DECODE_TOKENS,
        )

    if expected is None:
        logger.warning(f"{case_name}: performance metrics are observational; no profile-matched floor was applied")
    else:
        _assert_eval32_perf_target(first_result, expected, case_name=case_name)
    return first_result
