# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 Phi-4 (microsoft/phi-4) demo — accuracy and performance measurement on N300.

Uses the model-owned ``Phi4Executor`` directly (no vLLM adapter).

**Mesh note — N300 only.** Phi-4 has 40 attention heads and 10 KV heads; both must divide the mesh
device count. On this stack only N300 (2 devices) is supported and gated:
  - **N150 (1 device): unsupported.** A single Wormhole device hits a hard L1 OOM at program-build
    time (distributed-layernorm reader CBs ~1.51 MB > ~1.50 MB L1), so the weights MUST be
    tensor-parallel-sharded over >=2 devices. Cleanly skipped via ``_skip_below_min_tp_devices``.
  - **N300 (2 devices): the validated mesh.** 40 attention heads and 10 KV heads both divide 2.
  - **T3K / TG ordinary TP8: incompatible** (8 ∤ 10 KV heads) — skipped via
    ``_skip_unless_heads_divide_mesh``. A physical T3K does run ``ci-b1-DP-4`` as four TP2 lanes.
  - **ci-b1-DP-***: only DP4×TP2 is feasible on an 8-device T3K; the retained DP2/8/16/32 IDs skip
    before model construction when their lane topology is incompatible.

CI cases (parity with TTTv1 ``simple_text_demo.py``):
    token-accuracy   - teacher-forcing top-1/top-5 vs the book ``.refpt``
    batch-1          - single-user latency
    batch-32         - short-context throughput (per-profile seq; 200 decode)
    batch-32-ci      - CI-faithful batch-32 (seq2048 perf / DRAM-clamped acc; 1024 decode; TTTv1 ci-32)
    eval-32          - 32-user cross-batch determinism (TTTv1 ci-eval-32)
    ci-b1-DP-{2..32} - single-user data-parallel scaling smoke (TTTv1 ci-b1-DP-*)

Usage::

    # Token accuracy test (accuracy mode)
    MESH_DEVICE=N300 HF_MODEL=microsoft/phi-4 \\
      pytest models/common/tests/demos/phi4/demo.py -k "not performance and token-accuracy" -v

    # On-device sampling perf sweep
    SAMPLING_MODE=on_device_topk MESH_DEVICE=N300 HF_MODEL=microsoft/phi-4 \\
      pytest models/common/tests/demos/phi4/demo.py -k "batch-32-ci" -v

LazyWeight tensor cache: ``TT_CACHE_PATH/<device_name>`` when ``TT_CACHE_PATH`` is set,
otherwise ``model_cache/<HF_MODEL>/<device_name>`` under the current working directory.

Reference artifact (``.refpt``): the token-accuracy test gates on the committed book reference
``models/tt_transformers/tests/reference_outputs/phi-4.refpt`` (real-corpus teacher-forced targets).
"""

import json
import math
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.llm_runtime.config import PagedKVCacheConfig, TraceConfig, WarmupConfig
from models.common.llm_runtime.lane_group import LaneGroupExecutor
from models.common.models.phi4.executor import Phi4Executor, Phi4ExecutorConfig
from models.common.models.phi4.hf_adaptor import DEFAULT_HF_REVISION, encode_prompt, from_pretrained
from models.common.models.phi4.model import PHI4_ACCURACY, PHI4_PERFORMANCE, Phi4Transformer
from models.common.sampling.sampling_params import SamplingParams
from models.common.tests.demos.cleanup_utils import cleanup_dp_model_case, cleanup_model_case
from models.common.tests.demos.run_helpers import assert_no_special_tokens as assert_no_special_tokens_shared
from models.common.tests.demos.run_helpers import (
    load_eval_repeat_prompts_batch32,
    make_contiguous_page_table,
    run_eval_repeat_batch32,
    run_perf_benchmark,
    run_teacher_forcing,
)
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.demos.utils.model_targets import resolve_accuracy_targets
from models.perf.benchmarking_utils import BenchmarkProfiler

# =============================================================================
# Expected metrics — perf gates set from FRESH same-box N300 measurement (consolidation round-1,
# 2026-07-25, base 32c1f0e882b, median of 3 interleaved same-session reps per gated cell), NOT PERF.md.
#
# TTTv1 DOES run Phi-4 (special-cased into the Llama-3/Mistral/Phi accuracy branch, model_config.py) and
# — unlike Qwen2-7B — its on-device sampling IS enabled on N300 (vocab 100352//2 = 50176 <= 64*1024), so
# TTTv1's default decode is on-device top-k (k=32), directly comparable to TTTv2 on_device_topk. Same-box
# TTTv1 ``simple_text_demo.py`` controls (performance profile) are the parity anchor. Best-of rule (per
# cell, per sampling mode): on_device_topk gate = better-of(TTTv2 odt, TTTv1 default); host gate =
# TTTv2_host (TTTv1 phi-4 default is on-device, so there is no TTTv1 host counterpart). TTTv1 accuracy
# OOMs on N300 (bank_manager; documented phi-4 limit) => accuracy gates anchor to the TTTv2 value.
#
# *** minimal_matmul (QKV+FF2) is ENABLED (model.py _Phi4WHTuning.prefill_minimal_matmul=True; A/B escape
# DISABLE_MINIMAL_MATMUL=1). On the 14B, batch-32-ci prefill is matmul-compute-bound (~80% FLOPs = the 3
# MLP matmuls); minimal_matmul (~2-2.5x faster than ttnn.linear on the large folded prefill matmuls, TTTv1
# parity) closes the batch-32-ci prefill-TTFT gap: A/B same-box median-of-3 odt = ON 49.1ms vs OFF 58.5ms,
# beating the TTTv1 ci-32 control (50.47ms). It also drops the host + acc b32-ci TTFT (~58->49 / ~68->58ms).
# Accuracy with it ON is TTTv1-parity (eval-32 64/64 ON+OFF+odt; token-accuracy 97.3/100 perf, 99.0/100
# acc). Decode is minimal_matmul-independent (b1 buckets to seq128 < the seq>128 gate). ***
#
# Fresh N300 medians (2026-07-25, minimal_matmul ON), t/s/u | TTFT-ms.  DECODE compared MEAN-to-MEAN over
# the full decode window (TTTv1's per-iter decays with seq position; its "Average speed" mean is the fair
# comparand, NOT the 1st-token peak). Decode values decode-latency-derived (higher precision than the
# 1-decimal print):
#   TTTv1 perf (on-device default, mean): b1 18.56|149.05  ci-32 16.20|50.47   (accuracy profile OOMs)
#   TTTv2 on_device_topk: perf b1 18.45|117.0  b32 17.7|49.1  ci-32 16.5|49.1 ; acc b1 16.3|136.7  b32 15.7|58.0  ci-32 14.9|58.1
#   TTTv2 host:           perf b1 25.2|125.0   b32 23.4|49.1  ci-32 21.6|49.1 ; acc b1 21.3|136.5  b32 20.1|58.0  ci-32 18.8|57.9
# Parity verdict (perf, TTTv2 odt vs TTTv1 default, tolerance-free mean-to-mean):
#   - batch-32-ci: DECODE 16.5 >= 16.20 (TTTv2 wins); TTFT 49.1 <= 50.47 (PARITY — closed by minimal_matmul).
#   - batch-1 TTFT faster (117.0 <= 149.05).
#   - batch-1 DECODE is the ONE residual RED: 18.45 vs TTTv1 18.56 (~0.6%; decode latency 54.19 vs 53.87
#     ms/step). minimal_matmul-independent; per-model CCL-tuning lever (24/4 -> house-default 10/2) A/B'd
#     and REFUTED (54.31ms == unchanged). It is a diffuse SHARED decode-critical-path residual (executor
#     decode loop / shared modules), escalated as a consolidation SHARED-GAP ticket — out of per-model scope.
# Decode tok_s_u is prefill-independent (batched prefill / minimal_matmul do not change it). tok_s_u gates
# sit at/just below the measured (best-of) value so the 5% PERF_TOLERANCE absorbs jitter yet catches
# regressions; never lowered below a prior gate. TTFT gates are conservative ceilings covering BOTH
# batched-prefill ON (default, ~49ms with minimal_matmul) and DISABLE_BATCHED_PREFILL=1 (~116ms) — the
# ceiling is NOT tightened below the sequential-fallback path. N300 is the only supported+gated SKU
# (N150 L1-OOM, T3K/TG 8 does not divide 10 KV heads).
# =============================================================================

# token-accuracy top1/top5 floors (phi-4.refpt), profile-split — the LOCAL gate for token-accuracy
# (sampling-independent; no PERF_TOLERANCE — TTTv1 applies none to accuracy). Below the measured same-box
# N300 top1/top5 (perf 97.5/100, acc 99.0/100). Under CI the gate instead uses the centralized target
# (resolve_accuracy_targets) minus an absolute 0.5 pp with math.ceil (see _run_token_accuracy).
EXPECTED_METRICS: dict = {
    "performance": {
        "N300": {"top1": 96, "top5": 99},
    },
    "accuracy": {
        "N300": {"top1": 98, "top5": 99},
    },
}

# batch-1 throughput, sampling-mode- and profile-aware. Fresh same-box N300 medians (2026-07-23). odt perf
# b1 18.6 >= TTTv1 18.58 (parity, best-of); host is the faster N300 path (TTTv1 phi-4 default is on-device,
# no host counterpart). batch-1 does not batch prefill, so its TTFT is the single-user prefill (~117-146ms).
EXPECTED_METRICS_BATCH1: dict = {
    "host": {
        "performance": {"N300": {"tok_s_u": 25.0, "ttft_ms": 135}},
        "accuracy": {"N300": {"tok_s_u": 21.0, "ttft_ms": 150}},
    },
    "on_device_topk": {
        "performance": {"N300": {"tok_s_u": 18.5, "ttft_ms": 135}},
        "accuracy": {"N300": {"tok_s_u": 16.2, "ttft_ms": 150}},
    },
}

# Short-context batch-32 throughput (FUNCTIONAL leg — NOT part of the TTTv1 perf comparison; its seq len
# differs from TTTv1's CI batch-32, which is ci-32 = our batch-32-ci). Runs BOTH batched-prefill ON
# (default) and DISABLE_BATCHED_PREFILL=1 (A/B). Gate = TTTv2 measured regression guard. ttft ceiling
# covers both knob states (ON ~58ms / OFF ~116ms). Fresh N300 (2026-07-23): host perf 23.6, acc 19.9;
# odt perf 17.8, acc 15.7.
EXPECTED_METRICS_BATCH32: dict = {
    "host": {
        "performance": {"N300": {"tok_s_u": 23.0, "ttft_ms": 125}},
        "accuracy": {"N300": {"tok_s_u": 19.5, "ttft_ms": 145}},
    },
    "on_device_topk": {
        "performance": {"N300": {"tok_s_u": 17.5, "ttft_ms": 125}},
        "accuracy": {"N300": {"tok_s_u": 15.5, "ttft_ms": 145}},
    },
}

# CI-faithful batch-32 (the ``batch-32-ci`` leg): TTTv1 ci-32 = seq2048 (perf) / seq1024 (acc, DRAM
# clamp) + 1024-token decode budget. Keyed by SAMPLING_MODE + profile. odt perf DECODE 16.5 >= TTTv1 ci-32
# mean 16.20 (best-of = TTTv2, mean-to-mean). With minimal_matmul ON the measured TTFT is now ~49ms ON
# (batched) / ~116ms OFF (sequential); the ttft ceiling (125) is a regression guard clearing both with
# margin. The prior batch-32-ci TTFT parity RED vs TTTv1 (~50ms) is now CLOSED — TTTv2 49.1 <= TTTv1 50.47
# same-box (see header). Cells absent fall back to EXPECTED_METRICS_BATCH32.
EXPECTED_METRICS_BATCH32_CI: dict = {
    "host": {
        "performance": {"N300": {"tok_s_u": 21.0, "ttft_ms": 125}},
        "accuracy": {"N300": {"tok_s_u": 18.3, "ttft_ms": 145}},
    },
    "on_device_topk": {
        "performance": {"N300": {"tok_s_u": 16.4, "ttft_ms": 125}},
        "accuracy": {"N300": {"tok_s_u": 14.8, "ttft_ms": 145}},
    },
}

# Perf workload: natural-length prefill (sample prompts ~90-125 tokens -> 128 bucket, matching TTTv1),
# 200 decode steps. Accuracy uses the teacher-forcing refpt. PERF_NUM_DECODE_TOKENS overrides the decode
# budget (mirrors the llama32_3b sibling) — used to shorten the window for tt-perf-report/Tracy profiling.
_PERF_NUM_DECODE_TOKENS = int(os.environ.get("PERF_NUM_DECODE_TOKENS", "200"))

PERF_TOLERANCE = 0.05

# 32-user max_seq_len is DRAM-bound on N300 (Phi-4 14B, ~12 GB/device). Accuracy weights (all-BFP8,
# ~8.5 GB/dev) leave less room for the 32-user BFP8 KV cache than performance (BFP4 FF1/3, ~6.6 GB/dev),
# so accuracy runs a shorter context. batch-32 short-context uses the existing validated values;
# batch-32-ci (TTTv1 ci-32 = seq2048) keeps seq2048 for performance and DRAM-clamps accuracy (a 32-user
# seq2048 BFP8 KV + accuracy weights exceed the N300 budget) — footnoted in perf_tables.
_BATCH32_MAX_SEQ_LEN: dict[str, int] = {"performance": 2048, "accuracy": 512}
_BATCH32_CI_MAX_SEQ_LEN: dict[str, int] = {"performance": 2048, "accuracy": 1024}

# eval-32 max_seq_len (both profiles). The ci-eval-32 numeric prompts bucket to a 1024-token prefill, so
# the page table needs >=1024 (32 blocks/user); 1024 also fits the 3-fresh-executor eval churn on N300
# for both profiles (seq2048 OOMs). Decode high-water (~201 prompt + 200 gen) < 1024.
_EVAL_MAX_SEQ_LEN = 1024


def _sampling_bucket() -> str:
    """Map SAMPLING_MODE to a perf-gate bucket. Non-topk on-device modes (e.g. force-argmax)
    fall into ``on_device_topk`` so they stay gated, never silently un-gated."""
    return "host" if os.environ.get("SAMPLING_MODE", "host").lower() == "host" else "on_device_topk"


# Phi-4 requires at least this many devices of tensor parallelism. The unsharded 14B overflows a single
# Wormhole device's ~1.5MB L1 at program-build (distributed-layernorm reader CBs), so the weights MUST be
# sharded across >=2 devices. N300 (2-dev TP) is the minimum viable and only validated mesh. Consequence:
# single-device configs cannot run this model, so N150 ordinary cases cleanly skip. DP cases run only
# when partitioning the physical mesh yields TP2 lanes (for example, DP4×TP2 on T3K).
_MIN_TP_DEVICES = 2
_PHI4_NUM_ATTENTION_HEADS = 40
_PHI4_NUM_KV_HEADS = 10


def _skip_below_min_tp_devices(n_devices: int) -> None:
    """Skip when fewer than ``_MIN_TP_DEVICES`` devices are available for tensor parallelism."""
    if n_devices < _MIN_TP_DEVICES:
        pytest.skip(
            f"Phi-4 requires >={_MIN_TP_DEVICES}-device tensor parallelism: the unsharded 14B overflows "
            f"a single device's L1 (distributed-layernorm reader CBs at program build). Have {n_devices} "
            f"device(s) — use MESH_DEVICE=N300."
        )


# T3K / TG are listed so the module imports on those hosts, but they cleanly skip at model build
# (8 ∤ 10 KV heads — ``_skip_unless_heads_divide_mesh``). N150x4 (1, 4) is omitted (4 ∤ 10 KV heads).
_MESH_DEVICE_TO_SHAPE: dict[str, tuple[int, int]] = {
    "N150": (1, 1),
    "N300": (1, 2),
    "T3K": (1, 8),
    "TG": (8, 4),
}


def _ttnn_mesh_device_param_from_env() -> dict:
    env = os.environ.get("MESH_DEVICE", "").strip()
    if not env:
        pytest.skip("MESH_DEVICE must be set (e.g. N300). See module docstring.", allow_module_level=True)
    shape = _MESH_DEVICE_TO_SHAPE.get(env)
    if shape is None:
        pytest.skip(
            f"Unsupported MESH_DEVICE={env!r}; use one of {sorted(_MESH_DEVICE_TO_SHAPE)}.", allow_module_level=True
        )
    # The model-owned runtime's representative batch-32 trace set measures 53,698,560 bytes.
    # Keep the region narrowly above that closed-world requirement.
    param = {"mesh_shape": shape, "trace_region_size": 60_000_000, "num_command_queues": 1}
    # TTTv2 multi-device executor dispatch (and the on-device sampling all-gather) stalls without
    # an explicit 1D fabric; the root conftest does not auto-enable it. FABRIC_1D on any >1-dev mesh.
    if shape != (1, 1):
        param["fabric_config"] = ttnn.FabricConfig.FABRIC_1D
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


def _skip_unless_heads_divide_mesh(mesh_device: ttnn.MeshDevice) -> None:
    """Attention1D TP requires n_heads and n_kv_heads divisible by device count."""
    n_dev = mesh_device.get_num_devices()
    if n_dev <= 1:
        return
    n_h, n_kv = _PHI4_NUM_ATTENTION_HEADS, _PHI4_NUM_KV_HEADS
    if n_h % n_dev == 0 and n_kv % n_dev == 0:
        return
    pytest.skip(
        f"Incompatible mesh for Phi-4: {n_dev} devices need "
        f"num_attention_heads ({n_h}) and num_key_value_heads ({n_kv}) each divisible by {n_dev}. "
        f"Try MESH_DEVICE=N300 (2)."
    )


def get_device_name(mesh_device: ttnn.MeshDevice) -> str:
    """Map mesh device count to a metrics bucket."""
    n = mesh_device.get_num_devices()
    if n == 1:
        return "N150"
    if n == 2:
        return "N300"
    if n == 8:
        return "T3K"
    return f"{n}dev"


def lazy_weight_cache_dir_for_demo(mesh_device: ttnn.MeshDevice, hf_model_id: str) -> Path:
    """Disk root for LazyWeight caches. Follows the same convention as other TTTv2 demos."""
    device_name = get_device_name(mesh_device)
    hf = hf_model_id.strip("/")
    tt_cache = os.getenv("TT_CACHE_PATH")
    root = Path(tt_cache) / device_name if tt_cache else Path("model_cache") / hf / device_name
    root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Phi-4 demo LazyWeight cache directory: {root.resolve()}")
    return root


def ref_basename_for_hf(hf_model_id: str) -> str:
    return hf_model_id.strip("/").split("/")[-1]


def load_reference_data(hf_model_id: str):
    """Load reference tensors and optional metadata from ``.refpt``."""
    name = ref_basename_for_hf(hf_model_id)
    ref_path = Path("models/tt_transformers/tests/reference_outputs") / f"{name}.refpt"
    if not ref_path.exists():
        pytest.skip(
            f"Reference file not found: {ref_path}. Expected the committed book reference "
            f"(generated via models/tt_transformers/tests/generate_reference_outputs.py)."
        )
    ref_data = torch.load(ref_path, map_location="cpu", weights_only=False)
    return (
        ref_data["reference_tokens"],
        ref_data["top5_tokens"],
        ref_data.get("prompt_len"),
        ref_data.get("metadata"),
    )


def load_input_prompts(batch_size: int) -> list[str]:
    """Load prompts for performance testing from shared sample file."""
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
    and lets equal-length users fuse into a batched prefill pass. ``max_prefill_len`` is an optional clip
    cap for over-long prompts, never a pad-up target.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    encoded: list[list[int]] = []
    for p in prompts:
        ids = list(encode_prompt(tokenizer, p))
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
        logger.info(f"Teacher-forcing top5 alignment: metadata-driven direct path (top5_len={top5_tokens.shape[0]})")
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
            f"Cannot align top5 tokens: prompt_len={prompt_len}, num_target={num_target}, "
            f"top5_len={top5_tokens.shape[0]}"
        )
    best_score, best_start, best = max(candidates, key=lambda x: x[0])
    logger.info(f"Teacher-forcing top5 alignment: start={best_start}, score={best_score}/{min(16, num_target)}")
    return best


def log_generated_text(prompts, generated_token_ids, tokenizer):
    logger.info("Finished decoding, printing final outputs...\n")
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
    mesh_device: ttnn.MeshDevice,
    optimizations: str,
    cache_dir: Path,
    *,
    max_batch_size: int = 32,
    max_seq_len: int | None = None,
) -> Phi4Transformer:
    """Build the provider-neutral Phi-4 graph through its HF adaptor."""
    hf_model = os.environ.get("HF_MODEL", "microsoft/phi-4")
    _skip_below_min_tp_devices(mesh_device.get_num_devices())
    _skip_unless_heads_divide_mesh(mesh_device)

    precision = PHI4_PERFORMANCE if optimizations == "performance" else PHI4_ACCURACY

    if max_seq_len is None:
        max_seq_len = _BATCH32_MAX_SEQ_LEN[optimizations] if max_batch_size == 32 else 4096

    llm = from_pretrained(
        mesh_device,
        hf_model=hf_model,
        hf_revision=DEFAULT_HF_REVISION,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        n_layers=None,
        cache_dir=cache_dir,
        optimizations=precision,
    )

    model = llm.model
    model.demo_tokenizer = llm.tokenizer
    return model


def create_executor(
    model: Phi4Transformer,
    *,
    traced: bool,
    device_sampling_enabled: bool,
    trace_mode=None,
) -> Phi4Executor:
    block_size = 32
    max_num_blocks = math.ceil(model.config.max_seq_len / block_size) * model.config.max_batch_size
    attention_config = model.config.block_configs[0].attention_config
    if trace_mode is None:
        trace_mode = "all" if traced else "none"
    return Phi4Executor(
        model,
        model.model_args,
        Phi4ExecutorConfig(
            trace=TraceConfig(mode=trace_mode),
            warmup=WarmupConfig(),
            paged_kv_cache=PagedKVCacheConfig(
                block_size=block_size,
                max_num_blocks=max_num_blocks,
                num_blocks=max_num_blocks,
                dtype=attention_config.kv_cache_dtype,
            ),
            device_sampling_enabled=device_sampling_enabled,
        ),
    )


def _warmup_demo_executor(
    executor,
    *,
    kv_cache,
    page_table,
    prefill_compile_case=None,
    prefill_sampling_params=None,
    prefill_compile_execution=None,
):
    """Compile eager programs before activating the selected trace families."""
    config = executor.config if hasattr(executor, "config") else executor.lanes[0].config
    can_sample_on_device = config.device_sampling_enabled
    prefill_kwargs = {"kv_cache": kv_cache, "can_sample_on_device": can_sample_on_device}
    decode_kwargs = {
        "kv_cache": kv_cache,
        "max_batch_size": int(
            executor.max_batch_size if hasattr(executor, "max_batch_size") else executor.model.config.max_batch_size
        ),
        "num_blocks": int(page_table.shape[-1]),
        "can_sample_on_device": can_sample_on_device,
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


# =============================================================================
# ci-b1-DP: single-user data-parallel scaling smoke (TTTv1 ci-b1-DP-* parity)
# =============================================================================
#
# One user per DP group, model replicated across ``data_parallel`` disjoint submeshes, instruct
# prompts, paged attention, trace on. The ONLY correctness check is the special-token garbage guard
# plus "runs to completion without hang/exception". This is a mesh / KV-cache / page-table scaling
# smoke test, NOT an accuracy or perf gate.
#
# Hardware feasibility: every lane serves one user and requires exactly TP2. A physical T3K therefore
# runs DP4 as four TP2 lanes; the other retained manifest factors are inapplicable and skip pre-build.
_DP_SIZE_TABLE: dict[int, dict] = {
    2: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
    4: {"max_seq_len": 4096, "max_generated_tokens": 2048, "stop_at_eos": False},
    8: {"max_seq_len": 4096, "max_generated_tokens": 2048, "stop_at_eos": False},
    16: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
    32: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
}


def _dp_lane_tp_or_skip(mesh_device: ttnn.MeshDevice, data_parallel: int) -> int:
    """Return devices per lane, accepting only Phi-4's validated TP2 topology."""
    n = mesh_device.get_num_devices()
    if n % data_parallel != 0:
        pytest.skip(f"DP-{data_parallel} cannot partition {n} devices into equal lanes")
    tensor_parallel = n // data_parallel
    if tensor_parallel != _MIN_TP_DEVICES:
        pytest.skip(
            f"DP-{data_parallel} on {n} devices creates TP{tensor_parallel} lanes; "
            f"Phi-4 requires TP{_MIN_TP_DEVICES} lanes"
        )
    return tensor_parallel


def _create_dp_submeshes(mesh_device: ttnn.MeshDevice, data_parallel: int, tensor_parallel: int) -> list:
    submeshes = list(mesh_device.create_submeshes(ttnn.MeshShape(1, tensor_parallel)))
    if len(submeshes) != data_parallel:
        raise ValueError(f"Expected {data_parallel} TP{tensor_parallel} submeshes, got {len(submeshes)}")
    return submeshes


def _dp_lane_cache_dir(cache_dir: Path, tensor_parallel: int) -> Path:
    device_name = {2: "N300"}.get(tensor_parallel, f"{tensor_parallel}dev")
    lane_cache_dir = cache_dir.parent / device_name
    lane_cache_dir.mkdir(parents=True, exist_ok=True)
    return lane_cache_dir


def _validate_dp_lane(model: Phi4Transformer, lane: Phi4Executor, tensor_parallel: int, max_seq_len: int) -> None:
    config = model.config
    attention = config.block_configs[0].attention_config
    if config.num_devices != tensor_parallel:
        raise ValueError(f"DP lane expected TP{tensor_parallel}, model uses TP{config.num_devices}")
    if attention.n_heads % tensor_parallel or attention.n_kv_heads % tensor_parallel:
        raise ValueError(
            f"DP lane TP{tensor_parallel} does not divide Phi-4 heads ({attention.n_heads}/{attention.n_kv_heads})"
        )
    if config.max_batch_size != 1:
        raise ValueError(f"DP lane must have capacity 1, got {config.max_batch_size}")
    expected_blocks = math.ceil(max_seq_len / 32)
    cache_config = lane.config.paged_kv_cache
    if cache_config.max_num_blocks != expected_blocks or cache_config.num_blocks != expected_blocks:
        raise ValueError(
            f"DP lane cache must contain {expected_blocks} blocks, got "
            f"max={cache_config.max_num_blocks}, resolved={cache_config.num_blocks}"
        )


def assert_no_special_tokens(
    generated_token_ids, tokenizer, *, case_name: str = "", is_ci_env: bool | None = None
) -> None:
    """Apply the shared strict guard after Phi-4 ChatML turn-boundary truncation."""
    stop = set()
    # Phi-4 ChatML turn terminators. <|im_end|> (eos) ends the assistant turn; <|im_start|> OPENS a new
    # turn — i.e. the assistant's response is over and it has begun hallucinating the *next* turn, which
    # is a legitimate response terminator (serving stacks stop on it; HF generation_config omits it). The
    # perf benchmark runs a FIXED decode budget with stop_at_eos off, so an open-ended prompt is
    # force-decoded past its answer and greedily degenerates into "<|im_start|>user …" (verified
    # byte-identical on host and on_device_topk => inherent greedy divergence, not a sampling/decode-loop
    # artifact). Truncating the real response at either turn boundary before the garbage scan mirrors the
    # eval-32 stop-set augment and matches TTTv1, which STOPS generation at these tokens. This does not
    # hide garbage: any special id emitted mid-response (before the first turn boundary) is still flagged.
    for turn_tok in ("<|im_end|>", "<|im_start|>"):
        tid = tokenizer.convert_tokens_to_ids(turn_tok)
        if isinstance(tid, int) and tid >= 0:
            stop.add(tid)
    truncated_outputs = []
    for out in generated_token_ids:
        seq = list(out)
        for i, t in enumerate(seq):
            if t in stop:
                seq = seq[:i]
                break
        truncated_outputs.append(seq)
    assert_no_special_tokens_shared(
        truncated_outputs,
        tokenizer,
        case_name=case_name,
        is_ci_env=is_ci_env,
    )


def _run_dp_smoke(
    mesh_device: ttnn.MeshDevice,
    optimizations: str,
    cache_dir: Path,
    data_parallel: int,
    max_seq_len: int,
    max_gen_tokens: int,
    stop_at_eos: bool,
) -> None:
    """Run one user per TP2 lane through the model-owned DP runtime."""
    tensor_parallel = _dp_lane_tp_or_skip(mesh_device, data_parallel)
    mesh_device.quiesce_devices()
    submeshes = _create_dp_submeshes(mesh_device, data_parallel, tensor_parallel)
    lane_cache_dir = _dp_lane_cache_dir(cache_dir, tensor_parallel)
    hf_model = os.environ.get("HF_MODEL", "microsoft/phi-4")
    precision = PHI4_PERFORMANCE if optimizations == "performance" else PHI4_ACCURACY
    prompts = load_input_prompts(data_parallel)
    sampling_mode = os.environ.get("SAMPLING_MODE", "host").lower()
    on_device_params = {
        "on_device": SamplingParams(temperature=0.0, top_k=1, top_p=0.0),
        "on_device_topk": SamplingParams(temperature=0.0, top_k=32, top_p=0.08),
    }

    models: list = []
    lanes: list = []
    group = None
    try:
        for submesh in submeshes:
            # A supported DP topology that fails to build is a real regression, not an inapplicable case.
            llm = from_pretrained(
                submesh,
                hf_model=hf_model,
                hf_revision=DEFAULT_HF_REVISION,
                max_batch_size=1,
                max_seq_len=max_seq_len,
                n_layers=None,
                cache_dir=lane_cache_dir,
                optimizations=precision,
            )
            model = llm.model
            model.demo_tokenizer = llm.tokenizer
            models.append((model, submesh))
            lane = create_executor(
                model,
                traced=True,
                device_sampling_enabled=sampling_mode in on_device_params,
            )
            lanes.append(lane)
            _validate_dp_lane(model, lane, tensor_parallel, max_seq_len)

        group = LaneGroupExecutor(lanes, mesh_device=mesh_device)
        tokenizer = models[0][0].demo_tokenizer
        kv_cache = group.allocate_kv_cache()
        # Each lane owns an independent pool, so every global row uses the same lane-local block IDs.
        page_table = make_contiguous_page_table(1, max_seq_len, 32).repeat(data_parallel, 1)
        input_tokens, prompt_lens = tokenize_prompts(prompts, tokenizer)
        sampling_params = (
            on_device_params[sampling_mode]
            if sampling_mode in on_device_params and getattr(models[0][0], "supports_on_device_sampling", False)
            else None
        )
        _warmup_demo_executor(
            group,
            kv_cache=kv_cache,
            page_table=page_table,
            prefill_compile_case=(input_tokens, prompt_lens),
            prefill_sampling_params=sampling_params,
            prefill_compile_execution=group.traced_prefill_execution,
        )
        logger.info(
            f"[ci-b1-DP-{data_parallel}] TP={tensor_parallel}, SAMPLING_MODE={sampling_mode} "
            f"-> sampling_params={sampling_params}, stop_at_eos={stop_at_eos}"
        )
        result = run_perf_benchmark(
            group,
            tokens=input_tokens,
            kv_cache=kv_cache,
            page_table=page_table,
            num_decode_tokens=max_gen_tokens,
            max_batch_size=data_parallel,
            prompt_lens=prompt_lens,
            sampling_params=sampling_params,
            prefill_sampling_params=None,
        )
        assert len(result.generated_token_ids) == data_parallel
        assert all(result.generated_token_ids), f"ci-b1-DP-{data_parallel}: every TP2 lane must return output"
        log_generated_text(prompts, result.generated_token_ids, tokenizer)
        assert_no_special_tokens(result.generated_token_ids, tokenizer, case_name=f"ci-b1-DP-{data_parallel}")
    finally:
        cleanup_dp_model_case(group, lanes, models, mesh_device, submeshes)


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
        pytest.param("ci-b1-DP-2", id="ci-b1-DP-2"),
        pytest.param("ci-b1-DP-4", id="ci-b1-DP-4"),
        pytest.param("ci-b1-DP-8", id="ci-b1-DP-8"),
        pytest.param("ci-b1-DP-16", id="ci-b1-DP-16"),
        pytest.param("ci-b1-DP-32", id="ci-b1-DP-32"),
    ],
)
@pytest.mark.parametrize("optimizations", ["performance", "accuracy"])
def test_phi4(test_config, mesh_device, optimizations):
    """Main test entry for TTTv2 Phi-4."""
    device_name = get_device_name(mesh_device)
    expected = EXPECTED_METRICS.get(optimizations, {}).get(device_name, {})
    model = None
    hf_model = os.environ.get("HF_MODEL", "microsoft/phi-4")
    cache_dir = lazy_weight_cache_dir_for_demo(mesh_device, hf_model)

    try:
        # ci-b1-DP-*: single-user data-parallel smoke. Builds N models itself (one per submesh),
        # so it does NOT go through the shared create_model path below.
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

        if test_config == "batch-32":
            max_bs, max_seq_len = 32, _BATCH32_MAX_SEQ_LEN[optimizations]
            expected = EXPECTED_METRICS_BATCH32.get(_sampling_bucket(), {}).get(optimizations, {}).get(device_name, {})
        elif test_config == "eval-32":
            # eval-32 runs 32 users × 3 rotated repeats, building a FRESH traced executor per repeat.
            # On a single device the 14B does not fit at all (L1 overflow); on N300 it runs. Skip on
            # 1-device SKUs (hardware-capability guard, matches TTTv1 N300-only support).
            _skip_below_min_tp_devices(mesh_device.get_num_devices())
            # Accuracy-profile eval-32 does NOT fit N300: the 14B all-BFP8 accuracy weights (~8.5 GB/dev)
            # leave no headroom for the 3 fresh-executor rotated repeats at the seq1024 the 201-token
            # ci-eval prompts require — repeat-1 KV allocation OOMs (bank_manager), reproduced in a fresh
            # process. This is a genuine DRAM-capacity limit, matching TTTv1's own phi-4-accuracy N300 OOM.
            # The performance profile (BFP4 MLP, ~6.6 GB/dev) fits and validates cross-batch determinism
            # ON and OFF on the HARDER low-precision path (higher-precision accuracy is strictly more
            # deterministic), so determinism coverage is intact. Hardware-capability guard, not a mask.
            if optimizations == "accuracy":
                pytest.skip(
                    "eval-32 accuracy: 14B all-BFP8 weights + seq1024 + 3-executor rotated-repeat churn "
                    "exceed N300 DRAM (repeat-1 KV OOM; TTTv1 phi-4-accuracy also OOMs N300). Performance "
                    "eval-32 validates determinism (ON+OFF) on the harder low-precision path."
                )
            # The ci-eval-32 numeric prompts are ~201 tokens → get_padded_prefill_len buckets them to a
            # 1024-token prefill (32 KV blocks/user), so max_seq_len MUST be >= 1024 or the batched-prefill
            # group page-table (num_blocks_in_seq(1024)=32) overruns a shorter page table (the "32 vs 16"
            # expand). 1024 also keeps the per-repeat KV + the 1024-bucket batched fold inside the N300
            # DRAM budget for both profiles (seq2048 OOMs the 3-executor eval churn). Same value as the
            # sibling Qwen ChatML eval-32. Decode high-water (~201 prompt + 200 gen) stays < 1024.
            max_bs, max_seq_len = 32, _EVAL_MAX_SEQ_LEN
        elif test_config == "batch-32-ci":
            # CI-faithful batch-32 leg (TTTv1 ci-32 parity): seq2048 (perf) / DRAM-clamped (acc) +
            # 1024 decode budget. Own perf gate measured at this workload (NOT the lighter batch-32
            # constant, which would be a config-artifact miss). Keyed by SAMPLING_MODE AND profile;
            # cells not measured fall back to the short-context batch-32 constant (stay gated).
            max_bs = 32
            max_seq_len = _BATCH32_CI_MAX_SEQ_LEN[optimizations]
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
            _run_perf_benchmark(model, mesh_device, expected, batch_size=32, case_name=f"{optimizations}/batch-32")
        elif test_config == "batch-32-ci":
            # CI-faithful leg: 1024 decode tokens (clamped to KV headroom in _run_perf_benchmark).
            _run_perf_benchmark(
                model,
                mesh_device,
                expected,
                batch_size=32,
                case_name=f"{optimizations}/batch-32-ci",
                num_decode_tokens=1024,
            )
        elif test_config == "eval-32":
            _run_eval_repeat_batch32(model, mesh_device)
    finally:
        if model is not None:
            cleanup_model_case(model, mesh_device)


def _run_token_accuracy(model: Phi4Transformer, mesh_device: ttnn.MeshDevice, expected: dict):
    """Teacher-forcing token accuracy vs ``.refpt`` (CPU-generated)."""
    hf_model = os.environ.get("HF_MODEL", "microsoft/phi-4")
    reference_tokens, top5_tokens, prompt_len, metadata = load_reference_data(hf_model)
    tokenizer = model.demo_tokenizer

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
        logger.info(
            f"Reference metadata: hf_model_id={metadata.get('hf_model_id')}, "
            f"revision={metadata.get('revision')}, created_at={metadata.get('created_at')}"
        )

    prompt_tokens = reference_tokens[:prompt_len].unsqueeze(0)

    executor = create_executor(model, traced=False, device_sampling_enabled=False)
    max_batch_size = model.config.max_batch_size
    prompt_tokens = prompt_tokens.repeat(max_batch_size, 1)
    max_seq_len = model.config.max_seq_len
    block_size = 32
    kv_cache = executor.allocate_kv_cache()
    page_table = make_contiguous_page_table(max_batch_size, max_seq_len, block_size)

    target_top5 = select_teacher_forcing_top5_slice(
        top5_tokens, reference_tokens, prompt_len, metadata_aligned=has_prompt_len_metadata
    )
    is_ci_env = os.environ.get("CI") == "true"
    profiler = BenchmarkProfiler()
    try:
        profiler.start("run")
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
    finally:
        executor.cleanup()

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
            num_layers=model.config.n_layers,
            batch_size=1,
            input_sequence_length=prompt_len,
            output_sequence_length=num_target,
        )

    # Accuracy gate — threshold SOURCE is flag-controlled (flag = is_ci_env). CI mirrors TTTv1:
    # centralized target via resolve_accuracy_targets minus an ABSOLUTE 0.5 pp (get_accuracy_thresholds,
    # simple_text_demo.py); a missing central entry is a hard error (never silently un-gate in CI). Local
    # runs use the demo's EXPECTED_METRICS DIRECTLY (no ratio tolerance — TTTv1 applies none to accuracy).
    # Measured accuracy is rounded up with math.ceil before the compare, matching TTTv1 exactly
    # (simple_text_demo.py:1657-1658).
    use_centralized_targets = is_ci_env
    device_name = get_device_name(mesh_device)
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
    model: Phi4Transformer,
    mesh_device: ttnn.MeshDevice,
    expected: dict,
    batch_size: int,
    case_name: str,
    max_prefill_len: int | None = None,
    num_decode_tokens: int | None = None,
):
    """Timed prefill + decode with the traced model-owned executor.

    Prefill uses each prompt's natural token length (TTTv1 ``preprocess_inputs_prefill`` semantics —
    the executor buckets to ``get_padded_prefill_len``); decode runs for ``num_decode_tokens`` steps
    (default ``_PERF_NUM_DECODE_TOKENS``), clamped to the paged-KV headroom so the high-water decode
    position never overruns the page table (the ``batch-32-ci`` leg requests 1024).
    """
    hf_model = os.environ.get("HF_MODEL", "microsoft/phi-4")
    tokenizer = model.demo_tokenizer

    # On-device sampling toggle (see sampling handoff docs):
    #   host            -> sampling_params=None (host-argmax, the default shipped path)
    #   on_device       -> greedy temp=0,k=1,p=0 => trace-captured FORCE-ARGMAX full-vocab path
    #   on_device_topk  -> temp=0,k=32,p=0.08    => trace-captured TOP-K op path (gathers only
    #                      the [*,32] tuples; faster than force-argmax)
    sampling_mode = os.environ.get("SAMPLING_MODE", "host").lower()
    _on_device_params = {
        "on_device": SamplingParams(temperature=0.0, top_k=1, top_p=0.0),
        "on_device_topk": SamplingParams(temperature=0.0, top_k=32, top_p=0.08),
    }
    sampling_params = (
        _on_device_params[sampling_mode]
        if sampling_mode in _on_device_params and getattr(model, "supports_on_device_sampling", False)
        else None
    )
    pipeline_readback = os.environ.get("PIPELINE_READBACK", "1").lower() not in ("0", "false", "no")
    logger.info(f"[{case_name}] SAMPLING_MODE={sampling_mode} -> sampling_params={sampling_params}")
    logger.info(f"[{case_name}] PIPELINE_READBACK={pipeline_readback}")

    # Free-running perf run: enable the executor's on-device decode loop on the on-device sampling path
    # (inert on host/force-argmax; gated to the top-k path by _decode_loop_active). This is the shared
    # #49284 decode-loop fix; it must be active on the perf path for on-device decode parity.
    traced_executor = create_executor(
        model,
        traced=True,
        device_sampling_enabled=sampling_params is not None,
    )
    try:
        block_size = 32
        max_seq_len = model.config.max_seq_len
        max_batch_size = model.config.max_batch_size
        kv_cache = traced_executor.allocate_kv_cache()
        page_table = make_contiguous_page_table(max_batch_size, max_seq_len, block_size)
        prompts = load_input_prompts(batch_size)
        input_tokens, prompt_lens = tokenize_prompts(prompts, tokenizer, max_prefill_len=max_prefill_len)
        prefill_sampling_params = None if mesh_device.get_num_devices() > 1 else sampling_params
        _warmup_demo_executor(
            traced_executor,
            kv_cache=kv_cache,
            page_table=page_table,
            prefill_compile_case=(input_tokens, prompt_lens),
            prefill_sampling_params=prefill_sampling_params,
            prefill_compile_execution=traced_executor.traced_prefill_execution,
        )

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
            prefill_sampling_params=prefill_sampling_params,
            pipeline_readback=pipeline_readback,
            profiler=profiler,
        )
        profiler.end("run")

        logger.info(
            f"Performance [{case_name}] — TTFT: {result.ttft_ms:.1f}ms, "
            f"tok/s/u: {result.tok_s_u:.1f}, tok/s: {result.tok_s:.1f}, "
            f"decode latency: {result.decode_latency_mean_ms:.2f}ms"
        )

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
                num_layers=model.config.n_layers,
                batch_size=result.batch_size,
                input_sequence_length=prefill_seq_len,
                output_sequence_length=effective_decode,
            )

        log_generated_text(prompts, result.generated_token_ids, tokenizer)
        assert_no_special_tokens(result.generated_token_ids, tokenizer, case_name=case_name)

        if expected:
            failures = []
            if "tok_s_u" in expected and result.tok_s_u < expected["tok_s_u"] * (1 - PERF_TOLERANCE):
                failures.append(f"tok/s/u {result.tok_s_u:.1f} below target {expected['tok_s_u']}")
            if "ttft_ms" in expected and result.ttft_ms > expected["ttft_ms"] * (1 + PERF_TOLERANCE):
                failures.append(f"ttft_ms {result.ttft_ms:.1f} above target {expected['ttft_ms']}")
            assert not failures, f"{case_name}: " + "; ".join(failures)
    finally:
        traced_executor.cleanup()


# ci-eval-32 determinism case: 3 rotated repeats of the batch-32 workload.
_EVAL_REPEAT_BATCHES = 3
_EVAL_NUM_DECODE_TOKENS = _PERF_NUM_DECODE_TOKENS


def _run_eval_repeat_batch32(model: Phi4Transformer, mesh_device: ttnn.MeshDevice):
    """32-user cross-batch determinism (self-consistency under prompt rotation).

    Runs the batch-32 prefill+decode loop ``_EVAL_REPEAT_BATCHES`` times, rotating the prompt->slot
    assignment by one each repeat (fresh traced executor + KV cache per repeat), then asserts that
    undoing the rotation lines up per-user outputs. Honors the same ``SAMPLING_MODE`` knob as
    ``_run_perf_benchmark`` (default host argmax — deterministic and mesh-agnostic).
    """
    hf_model = os.environ.get("HF_MODEL", "microsoft/phi-4")
    tokenizer = model.demo_tokenizer

    # Phi-4 uses the ChatML format (<|im_start|>role<|im_sep|>...<|im_end|>); a chat turn ends at
    # <|im_end|>, but the model opening a NEW turn (<|im_start|>) is a de-facto response terminator too.
    # Phi-4's HF generation_config only carries <|im_end|> as eos, so augment the tokenizer stop set (the
    # mechanism ``hf_stop_ids`` reads) with <|im_start|> so the determinism runner truncates a degenerate
    # turn-restart there — same reusable pattern as the Qwen ChatML models. <|im_start|> is a legitimate
    # response terminator, so truncating there is correct, not a loosening; cross-batch consistency is
    # still asserted on the truncated (real-response) tokens.
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    if isinstance(im_start_id, int) and im_start_id >= 0:
        existing = list(getattr(tokenizer, "stop_tokens", None) or [])
        tokenizer.stop_tokens = list({*existing, im_start_id})

    block_size = 32
    max_seq_len = model.config.max_seq_len
    max_batch_size = model.config.max_batch_size
    page_table = make_contiguous_page_table(max_batch_size, max_seq_len, block_size)

    # Fresh traced executor + zeroed KV cache per repeat (driver owns the lifecycle), so the rotated
    # batches are fully independent — see run_eval_repeat_batch32 for why reuse corrupts the 3rd repeat.
    def make_executor():
        return create_executor(
            model,
            traced=True,
            device_sampling_enabled=sampling_params is not None,
            trace_mode="decode_only",
        )

    def allocate_kv_cache(executor):
        kv_cache = executor.allocate_kv_cache()
        _warmup_demo_executor(
            executor,
            kv_cache=kv_cache,
            page_table=page_table,
            prefill_compile_case=representative_prefill,
            prefill_sampling_params=sampling_params,
        )
        return kv_cache

    # TTTv1 ci-eval-32 numeric prompts (parity).
    prompts = load_eval_repeat_prompts_batch32()

    def tokenize_fn(ps):
        return tokenize_prompts(ps, tokenizer)

    sampling_mode = os.environ.get("SAMPLING_MODE", "host").lower()
    _on_device_params = {
        "on_device": SamplingParams(temperature=0.0, top_k=1, top_p=0.0),
        "on_device_topk": SamplingParams(temperature=0.0, top_k=32, top_p=0.08),
    }
    sampling_params = (
        _on_device_params[sampling_mode]
        if sampling_mode in _on_device_params and getattr(model, "supports_on_device_sampling", False)
        else None
    )
    # Prompt rotation preserves this heterogeneous signature multiset. Register it before the
    # closed-world program gate is activated, while keeping prefill eager under decode-only tracing.
    representative_prefill = tokenize_fn(prompts)
    logger.info(f"[eval-32] SAMPLING_MODE={sampling_mode} -> sampling_params={sampling_params}")

    run_eval_repeat_batch32(
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
    )
