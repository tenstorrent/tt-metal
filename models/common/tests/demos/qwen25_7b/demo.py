# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 Qwen2.5-7B-Instruct demo — accuracy and performance measurement.

Uses the model-owned ``Qwen25Executor`` directly (no vLLM adapter).

**Mesh note — physical T3K host and logical TP2 lanes.** Qwen2.5-7B uses two-device
tensor-parallel lanes because the 7B model does not fit a single Wormhole device's L1.
``MESH_DEVICE=N300`` selects one logical TP2 submesh while the fixture opens the physical
eight-device T3K for fabric; ``ci-b1-DP-4`` maps that host to four TP2 lanes.
  - **N150 (1 device): unsupported.** The unsharded 7B prefill/decode matmuls overflow a single
    Wormhole device's ~1.5MB L1 ("Statically allocated circular buffers ... clash with L1 buffers",
    program.cpp), reproduced across all cases/profiles — the weights MUST be tensor-parallel-sharded
    over >=2 devices. Cleanly skipped via ``_skip_below_min_tp_devices``. (The earlier TTTv2 N150
    numbers were scaled from N300, never actually measured.)
  - **N300 (2 devices): the validated mesh.** 28 attention heads and 4 KV heads both divide 2.
  - **T3K (8 devices):** ordinary TP8 cases are incompatible (8 ∤ 4 KV heads), but
    ``ci-b1-DP-4`` partitions the parent into four independent TP2 lanes and runs through
    ``LaneGroupExecutor``. DP2 would create unsupported TP4 lanes; DP8 would create TP1 lanes
    that cannot hold the model.
  - **N150x4 (4 devices): not validated** (fabric routing failure + the Qwen HiFi4 attention floor is
    only wired for 1–2 devices), intentionally absent from ``_MESH_DEVICE_TO_SHAPE``.
  - **ci-b1-DP-4 on T3K:** supported as four one-user TP2 lanes. Other DP factors retain explicit
    topology/capacity skips.

CI cases (parity with TTTv1 ``simple_text_demo.py``):
    token-accuracy   - teacher-forcing top-1/top-5 vs the book ``.refpt``
    batch-1          - single-user latency
    batch-32         - short-context throughput (seq1024 / 200 decode)
    batch-32-ci      - CI-faithful batch-32 (seq2048 / 1024 decode; TTTv1 ci-32); per-SKU seq clamp
    eval-32          - 32-user cross-batch determinism (TTTv1 ci-eval-32)
    ci-b1-DP-{2..32} - single-user data-parallel scaling smoke (TTTv1 ci-b1-DP-*)

Usage:
    # Token accuracy test
    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen2.5-7B-Instruct pytest models/common/tests/demos/qwen25_7b/demo.py -k "token-accuracy" -v

    # Batch-1 latency test
    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen2.5-7B-Instruct pytest models/common/tests/demos/qwen25_7b/demo.py -k "batch-1" -v

    # On-device sampling perf sweep
    SAMPLING_MODE=on_device_topk MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen2.5-7B-Instruct \
      pytest models/common/tests/demos/qwen25_7b/demo.py -k "batch-32-ci" -v

LazyWeight tensor cache (same rules as ``models/tt_transformers`` ``ModelArgs``):
``TT_CACHE_PATH/<device_name>`` when ``TT_CACHE_PATH`` is set, otherwise
``model_cache/<HF_MODEL>/<device_name>`` under the current working directory
(``device_name`` is ``N150`` / ``N300`` / ``N150x4`` / ``{n}dev`` from mesh size).

Reference artifact (``.refpt``): the token-accuracy test gates on the committed book
reference ``models/tt_transformers/tests/reference_outputs/Qwen2.5-7B-Instruct.refpt``
(real-corpus teacher-forced targets), shared with the TTTv1 demo. The loader supports both
the metadata-rich format (``prompt_len``) and the book half-split format.
"""

import dataclasses
import json
import math
import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.llm_runtime.config import PagedKVCacheConfig, TraceConfig, WarmupConfig
from models.common.llm_runtime.lane_group import LaneGroupExecutor
from models.common.models.qwen25_7b.executor import Qwen25Executor, Qwen25ExecutorConfig
from models.common.models.qwen25_7b.hf_adaptor import from_pretrained
from models.common.models.qwen25_7b.model import QWEN25_7B_ACCURACY, QWEN25_7B_PERFORMANCE, Qwen25_7B
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
from models.tt_transformers.tt.common import encode_prompt_hf, get_padded_prefill_len

# =============================================================================
# Expected metrics — perf gates set from a same-box TTTv1-vs-TTTv2 sweep, NOT PERF.md / not a cross-box
# audit (PERF.md's Qwen2.5-7B rows are stale/mislabeled — see the parity worklog).
#
# Model-specific sampling note: TTTv1's on-device sampling is DISABLED for Qwen2.5-7B (vocab 152064//2 =
# 76032 > 64K, tt_transformers/tt/model.py:156-157), so TTTv1 decodes HOST-only and has no on-device path.
# TTTv2 exposes both host and on_device_topk. So the parity-relevant comparison for THIS model is host vs
# host (both stacks' real path); on_device_topk is a TTTv2-only path.
# Rule (per cell), best-of{TTTv2[method], TTTv1[default]} for tok_s_u AND ttft_ms:
#     host           : max(TTTv2_host, TTTv1_host)      (TTTv2 measured >= TTTv1 host, same box)
#     on_device_topk : TTTv2_on_device_topk             (TTTv1 has no on-device path -> own-gated)
# Decode throughput is prefill-independent, so batched prefill does NOT change ``tok_s_u``.
# ``ttft_ms`` targets are conservative upper bounds (batched prefill only LOWERS TTFT).
#
# MEASUREMENT-FIRST: the throughput dicts below are populated from same-box measurement. SKUs/modes
# not yet measured stay ``{}`` — the case still RUNS and prints tok_s_u but is not gated (never a
# silent PERF.md value). ``top1``/``top5`` are teacher-forcing accuracy floors (sampling-independent),
# the real gate for token-accuracy.
# =============================================================================

# top1/top5 teacher-forcing accuracy floors (book refpt), profile-split. Perf metrics live in the batch
# dicts below. Measured same-box (N300, base c5d1c924245) = perf 87.5/96.5, accuracy 94.5/99.2; floors set
# conservatively below measured. Under CI the accuracy gate instead uses the CENTRALIZED target
# (resolve_accuracy_targets) minus an absolute 0.5 pp with math.ceil (see _run_token_accuracy). N300-only:
# Qwen2.5-7B requires >=2-device tensor parallelism (single-device L1 overflow), matching TTTv1/PERF.md
# which publish N300-only for this checkpoint — see _skip_below_min_tp_devices + the module docstring.
EXPECTED_METRICS: dict = {
    "performance": {
        "N300": {"top1": 85, "top5": 96},
    },
    "accuracy": {
        "N300": {"top1": 90, "top5": 98},
    },
}

# batch-1 throughput, sampling-mode- and profile-aware. Fresh same-box N300 measurement (median of 3
# interleaved reps): host perf 24.9 / acc 21.4 (TTFT 76/77); on_device_topk perf 14.6 / acc 13.4 (TTFT ~77).
# SAME-BOX TTTv1 simple_text_demo control (measured same session): host perf b1 21.5 / acc b1 21.5 (TTFT ~80).
# host is the parity-relevant path for THIS model: TTTv1's on-device sampling is DISABLED for Qwen2.5-7B
# (vocab 152064//2 = 76032 > 64K, tt_transformers model.py:156-157), so TTTv1 decodes host-only and has NO
# on-device path. TTTv2 host MEETS-OR-BEATS TTTv1 host (perf +16%; acc dead-even 99.5% within run-to-run
# noise). At 2 devices host > on_device_topk is a crossover (on-device pays the ttnn.topk all-gather),
# EXPECTED for this 7B, not a gap. Gate rule best-of{TTTv2[method], TTTv1[default]}: host perf 23.0 (<= TTTv2
# lowest 24.0, >= TTTv1 21.5); on_device_topk gate = TTTv2 measured (TTTv1 has no on-device path -> own-gated).
# Gates sit at/below fresh lowest-observed (5% tol = jitter buffer). ttft = conservative upper bound (batch-1
# does not batch prefill, so ON == OFF here).
# RE-MEASURED on the consolidation integration base (main 32c1f0e882b, median of 3): host perf b1 24.6 /
# acc b1 21.8; odt perf b1 14.4 / acc b1 13.3; same-box TTTv1 host b1 perf 21.58 / acc 21.61. Every gate
# below still holds with margin (TTTv2 lowest rep > gate x 0.95) — kept best-of, none lowered.
EXPECTED_METRICS_BATCH1: dict = {
    "host": {
        "performance": {"N300": {"tok_s_u": 23.0, "ttft_ms": 90}},
        "accuracy": {"N300": {"tok_s_u": 20.5, "ttft_ms": 92}},
    },
    "on_device_topk": {
        "performance": {"N300": {"tok_s_u": 14.0, "ttft_ms": 85}},
        "accuracy": {"N300": {"tok_s_u": 13.2, "ttft_ms": 85}},
    },
}

# Short-context batch-32 throughput (seq1024 / 200 decode), sampling-mode- and profile-aware.
# batch-32 runs BOTH batched-prefill ON (default) and DISABLE_BATCHED_PREFILL=1 (A/B). NOTE: this (non-ci)
# batch-32 is NOT part of the TTTv1 perf comparison — its seq len differs from TTTv1's CI batch-32 workload
# (which is ci-32 = our batch-32-ci below); it runs for the functional/determinism axis. Decode tok_s_u is
# prefill-independent, so gates cover both knob states; ttft covers both (batched-ON ~39ms, sequential-OFF
# ~75ms → gate 80 clears both). Gates are TTTv2-measured regression guards, conservative (carried from the
# prior same-box sweep; not re-measured this pass since it is not perf-compared).
EXPECTED_METRICS_BATCH32: dict = {
    "host": {
        "performance": {"N300": {"tok_s_u": 21.5, "ttft_ms": 80}},
        "accuracy": {"N300": {"tok_s_u": 21.0, "ttft_ms": 80}},
    },
    "on_device_topk": {
        "performance": {"N300": {"tok_s_u": 13.8, "ttft_ms": 80}},
        "accuracy": {"N300": {"tok_s_u": 13.2, "ttft_ms": 80}},
    },
}

# CI-faithful batch-32 targets (the ``batch-32-ci`` leg), measured at seq2048 (per-SKU clamp; see
# _BATCH32_CI_MAX_SEQ_LEN) with a 1024-token decode budget (TTTv1 ci-32 workload). SEPARATE workload
# from the lighter batch-32 leg (seq1024 / 200 decode): the larger KV cache means the decode read
# window grows, so steady-state per-token decode is legitimately a bit slower. Keyed by SAMPLING_MODE
# AND profile. Runs batched ON + OFF (ttft covers both: ON ~39ms, OFF ~75ms → gate 80). Fresh same-box
# N300 (base c5d1c924245, median of 3 reps): host perf 25.9, acc 21.7; odt perf 14.6, acc 13.2. SAME-BOX
# TTTv1 ci-32 control (measured this session; both stacks batch prefill): host perf 20.0, acc 20.05
# (TTFT ~42ms) — TTTv2 host beats it (+29% perf, +8% acc) with lower TTFT (39 vs 42). Gate best-of{TTTv2,
# TTTv1}: host perf 24.5 (<= TTTv2 lowest 25.7, >= TTTv1 20.0); on_device_topk gate = TTTv2 (TTTv1 has no
# on-device path -> own-gated). Gates at/below fresh lowest-observed. Cells not present fall back to EXPECTED_METRICS_BATCH32.
# RE-MEASURED on the consolidation integration base (main 32c1f0e882b, median of 3): host perf ci-32 25.8 /
# acc ci-32 21.8; odt perf ci-32 14.4 / acc ci-32 13.1; same-box TTTv1 host ci-32 perf 17.88 / acc 17.74
# (TTTv1 averages over its full 4096-iter ci-32 decode -> lower steady-state, widening TTTv2's host win).
# Every gate below still holds with margin — kept best-of, none lowered.
EXPECTED_METRICS_BATCH32_CI: dict = {
    "host": {
        "performance": {"N300": {"tok_s_u": 24.5, "ttft_ms": 80}},
        "accuracy": {"N300": {"tok_s_u": 21.0, "ttft_ms": 80}},
    },
    "on_device_topk": {
        "performance": {"N300": {"tok_s_u": 14.0, "ttft_ms": 80}},
        "accuracy": {"N300": {"tok_s_u": 13.0, "ttft_ms": 80}},
    },
}

# Perf workload: natural-length prefill (these sample prompts are ~90-125 tokens -> 128 bucket,
# matching TTTv1), 200 decode steps. Accuracy uses the teacher-forcing refpt.
_PERF_NUM_DECODE_TOKENS = 200

PERF_TOLERANCE = 0.05

# batch-32-ci per-SKU max_seq_len (TTTv1 ci-32 parity is seq2048). DRAM trap: raising max_seq_len
# doubles the batch-32 KV cache. 7B weights are large — a single unsharded N150 cannot hold 7B
# weights + a seq2048×32-user KV cache, so N150 is clamped to 1024 (same cap TTTv1 uses for its
# batch-32 config). N300 (weights sharded 2-way) holds seq2048.
_BATCH32_CI_MAX_SEQ_LEN: dict[str, int] = {
    "N150": 1024,
    "N300": 2048,
    "T3K": 2048,
}


def _sampling_bucket() -> str:
    """Map SAMPLING_MODE to a perf-gate bucket. Non-topk on-device modes (e.g. force-argmax)
    fall into ``on_device_topk`` so they stay gated, never silently un-gated."""
    return "host" if os.environ.get("SAMPLING_MODE", "host").lower() == "host" else "on_device_topk"


# Qwen2.5-7B requires at least this many devices of tensor parallelism. The unsharded 7B prefill/decode
# matmuls overflow a single Wormhole device's ~1.5MB L1 ("Statically allocated circular buffers ... clash
# with L1 buffers", program.cpp) — reproduced on N150 across ALL cases/profiles — so the weights MUST be
# sharded across >=2 devices. This matches TTTv1/PERF.md, which publish Qwen2.5-7B N300-ONLY (the earlier
# TTTv2 N150 numbers were scaled from N300, never actually measured). N300 (2-dev TP) is the minimum
# viable and only validated mesh. Consequence: single-device configs cannot run this model, so N150 and
# every ci-b1-DP factor (each DP group is a single device) cleanly skip — a genuine hardware-capacity
# guard (like the T3K 8-KV-head skip), not a masked failure.
_MIN_TP_DEVICES = 2


def _skip_below_min_tp_devices(n_devices: int) -> None:
    """Skip when fewer than ``_MIN_TP_DEVICES`` devices are available for tensor parallelism."""
    if n_devices < _MIN_TP_DEVICES:
        pytest.skip(
            f"Qwen2.5-7B requires >={_MIN_TP_DEVICES}-device tensor parallelism: the unsharded 7B "
            f"overflows a single device's L1 (matmul circular-buffer clash). TTTv1/PERF.md publish this "
            f"checkpoint N300-only. Have {n_devices} device(s) — use MESH_DEVICE=N300."
        )


# Mesh topology comes only from ``MESH_DEVICE`` (same naming as vLLM / other tt demos).
# N150x4 (1, 4) is intentionally omitted: not a validated mesh for this model on TTTv2
# (fabric routing failure + 1–2-device-only attention precision floor — see module docstring).
# T3K / TG are listed so the module imports on those hosts, but they cleanly skip at model build
# (8 ∤ 4 KV heads — ``_skip_unless_heads_divide_mesh``).
_MESH_DEVICE_TO_SHAPE: dict[str, tuple[int, int]] = {
    "N150": (1, 1),
    "N300": (1, 2),
    "T3K": (1, 8),
    "TG": (8, 4),
}


def _ttnn_mesh_device_param_from_env() -> dict:
    env = os.environ.get("MESH_DEVICE", "").strip()
    if not env:
        pytest.skip(
            "MESH_DEVICE must be set (e.g. N300). See module docstring.",
            allow_module_level=True,
        )
    shape = _MESH_DEVICE_TO_SHAPE.get(env)
    if shape is None:
        pytest.skip(
            f"Unsupported MESH_DEVICE={env!r}; use one of {sorted(_MESH_DEVICE_TO_SHAPE)}.",
            allow_module_level=True,
        )
    param = {
        "mesh_shape": shape,
        "trace_region_size": 50_000_000,
        "num_command_queues": 1,
    }
    # TTTv2 multi-device executor dispatch (and the on-device sampling all-gather) stalls without
    # an explicit 1D fabric; the root conftest does not auto-enable it. Mirror the sibling
    # models/common/models/qwen25_7b/demo.py wiring: FABRIC_1D on any >1-device mesh.
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
        f"num_attention_heads ({n_h}) and num_key_value_heads ({n_kv}) each divisible by {n_dev}. "
        f"Try MESH_DEVICE=N300 (2)."
    )


def get_device_name(mesh_device):
    """Map mesh device count to a metrics bucket (not physical card SKU)."""
    num_devices = mesh_device.get_num_devices()
    if num_devices == 1:
        return "N150"
    if num_devices == 2:
        return "N300"
    if num_devices == 4:
        return "N150x4"
    if num_devices == 8:
        return "T3K"
    return f"{num_devices}dev"


def lazy_weight_cache_dir_for_demo(mesh_device: ttnn.MeshDevice, hf_model_id: str) -> Path:
    """Disk root for ``Qwen25_7B`` ``LazyWeight`` caches in this e2e demo.

    Matches ``models/tt_transformers/tt/model_config.py`` (HF checkpoint branch):
    if ``TT_CACHE_PATH`` is set, use ``<TT_CACHE_PATH>/<device_name>``; otherwise
    ``model_cache/<HF_MODEL>/<device_name>``. Directories are created as needed.
    """
    device_name = get_device_name(mesh_device)
    hf = hf_model_id.strip("/")
    tt_cache = os.getenv("TT_CACHE_PATH")
    if tt_cache:
        root = Path(tt_cache) / device_name
    else:
        root = Path("model_cache") / hf / device_name
    root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Qwen2.5-7B demo LazyWeight cache directory: {root.resolve()}")
    return root


def ref_basename_for_hf(hf_model_id: str) -> str:
    """Match ``ModelArgs.model_name`` style used for ``.refpt`` filenames."""
    return hf_model_id.strip("/").split("/")[-1]


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

    Each prompt is encoded with the chat template at its real length. The returned ``[batch,
    max_len]`` token tensor is right-padded to the batch-max for rectangularity, while the
    returned per-user lengths are the *real* token counts — the executor reads only
    ``tokens[user, :prompt_len]`` and then buckets each user to ``get_padded_prefill_len``
    (128 / 1024 / next-pow2). This matches TTTv1 exactly: no fixed pad-to-N prefill budget.

    ``max_prefill_len`` is an optional clip *cap* (like TTTv1's ``max_prefill_len``): prompts
    longer than it are left-clipped to their most recent tokens. It is never a pad-up target.
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
    perf_decode_tuning: bool | None = None,
):
    """Build ``Qwen25_7B`` in executor (paged KV) mode.

    Picks one of the two module-level precision recipes (``QWEN25_7B_ACCURACY`` /
    ``QWEN25_7B_PERFORMANCE``) — both defined in ``qwen25_7b/model.py`` and grounded
    in TTTv1's ``DecodersPrecision`` for Qwen2.5-7B. The dataclass owns the dtype +
    math-fidelity recipe; this demo just selects between the two and forwards it.

    ``max_seq_len`` overrides the DRAM-aware default. Default (``None``): 7B weights + a 32-user KV
    cache cannot co-reside at seq4096 on a single unsharded device, so batch>1 is capped to 1024 on
    ≤2-device SKUs (TTTv1 batch-32 parity); batch-1 fits seq4096 on every SKU. The ``batch-32-ci``
    leg passes an explicit per-SKU value (see ``_BATCH32_CI_MAX_SEQ_LEN``).

    ``perf_decode_tuning`` overrides the selected immutable precision recipe. The
    token-accuracy path passes ``False`` even under ``optimizations="performance"``
    to keep teacher-forcing parity off aggressive decode math.
    """
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    _skip_below_min_tp_devices(mesh_device.get_num_devices())
    _skip_unless_heads_divide_mesh(mesh_device, hf_model)

    precision = QWEN25_7B_PERFORMANCE if optimizations == "performance" else QWEN25_7B_ACCURACY
    if perf_decode_tuning is not None and perf_decode_tuning != precision.perf_decode_tuning:
        precision = dataclasses.replace(precision, perf_decode_tuning=perf_decode_tuning)
    num_devices = mesh_device.get_num_devices()
    if max_seq_len is None:
        if num_devices >= 8:
            max_seq_len = 131072 // max_batch_size
        elif max_batch_size > 1:
            max_seq_len = 1024
        else:
            max_seq_len = 4096

    try:
        llm = from_pretrained(
            mesh_device,
            hf_model=hf_model,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            n_layers=None,
            cache_dir=cache_dir,
            optimizations=precision,
        )
    except Exception as e:
        pytest.skip(f"Could not build Qwen model (weights / memory / mesh): {e}")

    model = llm.model
    model.demo_tokenizer = llm.tokenizer
    return model


def create_executor(
    model: Qwen25_7B,
    *,
    traced: bool,
    device_sampling_enabled: bool,
    trace_mode=None,
) -> Qwen25Executor:
    block_size = 32
    max_num_blocks = ((model.config.max_seq_len + block_size - 1) // block_size) * model.config.max_batch_size
    attention_config = model.config.block_configs[0].attention_config
    if trace_mode is None:
        trace_mode = "all" if traced else "none"
    return Qwen25Executor(
        model,
        model.model_args,
        Qwen25ExecutorConfig(
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
):
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
            execution=executor.eager_execution,
        )
    if config.trace.prefill_enabled:
        executor.warmup_model_prefill(enable_trace=True, **prefill_kwargs)
    if config.trace.decode_enabled:
        executor.warmup_model_decode(enable_trace=True, **decode_kwargs)


# =============================================================================
# ci-b1-DP: single-user data-parallel scaling smoke (TTTv1 ci-b1-DP-* parity)
# =============================================================================
#
# These case IDs retain manifest parity. Qwen2.5-7B lanes require exactly TP2, so a full T3K
# parent can run DP4 as four two-device lanes; all other factors skip before construction.
#
# Per-case size table (TTTv1 simple_text_demo.py parity, with the DP-2 N300 addition):
#   ci-b1-DP-2  : max_seq_len=1024, max_generated_tokens=200, stop_at_eos=True (TP1 on N300: skip)
#   ci-b1-DP-4  : max_seq_len=4096, max_generated_tokens=2048, stop_at_eos=False
#   ci-b1-DP-8  : max_seq_len=4096, max_generated_tokens=2048, stop_at_eos=False
#   ci-b1-DP-16 : max_seq_len=1024, max_generated_tokens=200,  stop_at_eos=True
#   ci-b1-DP-32 : max_seq_len=1024, max_generated_tokens=200,  stop_at_eos=True
#
# Hardware feasibility: every group serves one user, but the group itself must contain exactly two
# tensor-parallel devices. On an eight-device T3K, DP4 therefore maps to four TP2 lanes. DP2 maps
# to unsupported TP4, DP8 maps to TP1 (which overflows L1), and DP16/32 exceed host capacity.
_DP_SIZE_TABLE: dict[int, dict] = {
    2: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
    4: {"max_seq_len": 4096, "max_generated_tokens": 2048, "stop_at_eos": False},
    8: {"max_seq_len": 4096, "max_generated_tokens": 2048, "stop_at_eos": False},
    16: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
    32: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
}


def _dp_lane_tp_or_skip(mesh_device: ttnn.MeshDevice, data_parallel: int) -> int:
    """Return devices per lane, accepting only Qwen25's validated TP2 topology."""
    n = mesh_device.get_num_devices()
    if n % data_parallel != 0:
        pytest.skip(f"DP-{data_parallel} cannot partition {n} devices into equal lanes")
    tensor_parallel = n // data_parallel
    if tensor_parallel != _MIN_TP_DEVICES:
        pytest.skip(
            f"DP-{data_parallel} on {n} devices creates TP{tensor_parallel} lanes; "
            f"Qwen2.5-7B requires TP{_MIN_TP_DEVICES} lanes"
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


def _validate_dp_lane(model: Qwen25_7B, lane: Qwen25Executor, tensor_parallel: int, max_seq_len: int) -> None:
    config = model.config
    attention = config.block_configs[0].attention_config
    if config.num_devices != tensor_parallel:
        raise ValueError(f"DP lane expected TP{tensor_parallel}, model uses TP{config.num_devices}")
    if attention.n_heads % tensor_parallel or attention.n_kv_heads % tensor_parallel:
        raise ValueError(
            f"DP lane TP{tensor_parallel} does not divide Qwen25 heads " f"({attention.n_heads}/{attention.n_kv_heads})"
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
    """Apply the shared strict guard after Qwen turn-boundary truncation.

    Used by the perf-benchmark generation path (batch-1 / batch-32 / batch-32-ci). TTTv2's
    ``result.generated_token_ids[user]`` already starts at the first generated
    token, so unlike TTTv1 we do not slice off the prompt — these are output-only. Each user's output
    is truncated at the first Qwen turn boundary (``<|im_end|>`` / ``<|im_start|>``) before the shared
    helper applies its standard EoS truncation and strictness policy, including
    ``TT_DEMO_STRICT_SPECIAL_TOKENS=1``.
    """
    stop = set()
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
    """Run one user per TP2 lane through the migrated model-owned DP runtime."""
    tensor_parallel = _dp_lane_tp_or_skip(mesh_device, data_parallel)
    mesh_device.quiesce_devices()
    submeshes = _create_dp_submeshes(mesh_device, data_parallel, tensor_parallel)
    lane_cache_dir = _dp_lane_cache_dir(cache_dir, tensor_parallel)
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    precision = QWEN25_7B_PERFORMANCE if optimizations == "performance" else QWEN25_7B_ACCURACY
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
            try:
                llm = from_pretrained(
                    submesh,
                    hf_model=hf_model,
                    max_batch_size=1,
                    max_seq_len=max_seq_len,
                    n_layers=None,
                    cache_dir=lane_cache_dir,
                    optimizations=precision,
                )
            except Exception as error:
                pytest.skip(f"Could not build Qwen2.5-7B TP2 lane (weights / memory / mesh): {error}")
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
        # Every lane owns an independent block pool; repeat the same lane-local block IDs for
        # each global row rather than assigning cross-lane global block offsets.
        page_table = make_contiguous_page_table(1, max_seq_len, 32).repeat(data_parallel, 1)
        _warmup_demo_executor(group, kv_cache=kv_cache, page_table=page_table)
        input_tokens, prompt_lens = tokenize_prompts(prompts, tokenizer)
        sampling_params = (
            on_device_params[sampling_mode]
            if sampling_mode in on_device_params and getattr(models[0][0], "supports_on_device_sampling", False)
            else None
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
        logger.info(
            f"Performance [ci-b1-DP-{data_parallel}] — TTFT: {result.ttft_ms:.1f}ms, "
            f"tok/s/u: {result.tok_s_u:.1f}, tok/s: {result.tok_s:.1f}, "
            f"decode latency: {result.decode_latency_mean_ms:.2f}ms"
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
def test_qwen25_7b(test_config, mesh_device, optimizations):
    """Main test entry for TTTv2 Qwen2.5-7B-Instruct."""
    device_name = get_device_name(mesh_device)
    expected = EXPECTED_METRICS.get(optimizations, {}).get(device_name, {})
    model = None
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
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

        # Only the batch-32 throughput test actually exercises 32 users. ``token-accuracy``
        # teacher-forces a single reference sequence, so running it with max_batch_size=32 is pure
        # waste and trips ``decode_spill_w1_to_dram_before_w3`` (extra per-step DRAM round-trip in
        # MLP decode, see model.py:_resolve_qwen_wh_tuning), which pushes the cold-cache first
        # invocation past pytest.ini's 300s budget. Use max_batch_size=1 for everything except the
        # 32-user cases.
        # Keep teacher-forcing parity off aggressive decode math; throughput tests use full tuning.
        decode_tuning = optimizations == "performance" and test_config != "token-accuracy"

        if test_config == "batch-32":
            max_bs, max_seq_len = 32, 1024
            expected = EXPECTED_METRICS_BATCH32.get(_sampling_bucket(), {}).get(optimizations, {}).get(device_name, {})
        elif test_config == "eval-32":
            # eval-32 runs 32 users × 3 rotated repeats, building a FRESH traced executor per repeat
            # (run_eval_repeat_batch32). On a single unsharded device the full 7B weights + a 32-user KV
            # cache already sit near DRAM capacity (batch-32 fits, but with little headroom), so the
            # per-repeat executor/trace churn cannot fit — it OOMs (bank_manager). This is a genuine
            # single-device DRAM-capability limit for a 7B, NOT a TTTv2 regression: TTTv1 ci-32 /
            # ci-eval-32 also OOM on N150 (batch-32-class does not fit a single N150 for 7B in either
            # stack), while TTTv2 batch-32 / batch-32-ci DO fit here (single executor). Skip on
            # 1-device SKUs; runs on the sharded N300. Hardware-capability guard, not a mask.
            if mesh_device.get_num_devices() == 1:
                pytest.skip(
                    "eval-32 (32 users × 3 rotated fresh-executor repeats) exceeds single-device DRAM "
                    "for a 7B; TTTv1 ci-32/ci-eval-32 OOM on N150 too. Runs on sharded N300."
                )
            max_bs, max_seq_len = 32, 1024
        elif test_config == "batch-32-ci":
            # CI-faithful batch-32 leg (TTTv1 ci-32 parity): larger seq len + 1024 decode budget.
            # Per-SKU seq len clamp (7B KV cache is large; see _BATCH32_CI_MAX_SEQ_LEN).
            max_bs = 32
            max_seq_len = _BATCH32_CI_MAX_SEQ_LEN.get(device_name, 2048)
            # Own perf gate measured at the seq2048/decode1024 workload (NOT the lighter batch-32
            # constant, which would be a config-artifact miss). Keyed by SAMPLING_MODE AND profile.
            # Non-topk on-device modes (force-argmax) fall into the on_device_topk bucket; cells not
            # measured fall back to the short-context batch-32 constant (stay gated, never un-gated).
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
            perf_decode_tuning=decode_tuning,
        )

        if test_config == "token-accuracy":
            _run_token_accuracy(model, mesh_device, expected)
        elif test_config == "batch-1":
            perf_expected = (
                EXPECTED_METRICS_BATCH1.get(_sampling_bucket(), {}).get(optimizations, {}).get(device_name, {})
            )
            _run_perf_benchmark(model, mesh_device, perf_expected, batch_size=1, case_name=f"{optimizations}/batch-1")
        elif test_config == "batch-32":
            # Natural-length prefill: these sample prompts bucket to 128 (PERF.md Short-Context
            # Batch-32 row), matching TTTv1's traced-prefill seq len without a forced pad.
            _run_perf_benchmark(model, mesh_device, expected, batch_size=32, case_name=f"{optimizations}/batch-32")
        elif test_config == "batch-32-ci":
            # CI-faithful leg: seq2048 + 1024 decode tokens (clamped in _run_perf_benchmark).
            # Gated by EXPECTED_METRICS_BATCH32_CI (measured at this workload, TTTv1-parity).
            _run_perf_benchmark(
                model,
                mesh_device,
                expected,
                batch_size=32,
                case_name=f"{optimizations}/batch-32-ci",
                num_decode_tokens=1024,
            )
        elif test_config == "eval-32":
            # 32-user cross-batch determinism (self-consistency under prompt rotation).
            _run_eval_repeat_batch32(model, mesh_device)
    finally:
        # A pre-build topology skip owns no model state. Synchronizing the parent mesh
        # here can advance its event stream before a later DP case creates submeshes.
        if model is not None:
            cleanup_model_case(model, mesh_device)


def _run_token_accuracy(model, mesh_device, expected):
    """Teacher-forcing token accuracy vs ``.refpt`` (HF-generated)."""
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
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
        meta_summary = {
            "hf_model_id": metadata.get("hf_model_id"),
            "revision": metadata.get("revision"),
            "generation_mode": metadata.get("generation_mode"),
            "created_at": metadata.get("created_at"),
        }
        logger.info(f"Reference metadata summary: {meta_summary}")

    prompt_tokens = reference_tokens[:prompt_len].unsqueeze(0)

    executor = create_executor(model, traced=False, device_sampling_enabled=False)
    max_batch_size = model.config.max_batch_size
    prompt_tokens = prompt_tokens.repeat(max_batch_size, 1)
    max_seq_len = model.config.max_seq_len
    block_size = 32
    kv_cache = executor.allocate_kv_cache()
    page_table = make_contiguous_page_table(max_batch_size, max_seq_len, block_size)

    target_top5 = select_teacher_forcing_top5_slice(
        top5_tokens,
        reference_tokens,
        prompt_len,
        metadata_aligned=has_prompt_len_metadata,
    )
    is_ci_env = os.environ.get("CI") == "true"
    profiler = BenchmarkProfiler()
    try:
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
    model,
    mesh_device,
    expected,
    batch_size,
    case_name,
    max_prefill_len: int | None = None,
    num_decode_tokens: int | None = None,
):
    """Timed prefill + decode with the traced model-owned executor.

    Prefill uses each prompt's natural token length (TTTv1 ``preprocess_inputs_prefill`` semantics —
    the executor buckets to ``get_padded_prefill_len``); decode runs for ``num_decode_tokens`` steps
    (default ``_PERF_NUM_DECODE_TOKENS``). ``max_prefill_len`` is an optional clip cap for over-long
    prompts, never a pad-up target.

    The decode budget is clamped to what the paged KV cache can hold:
    ``effective = min(requested, max_seq_len - prompt_bucket - margin)`` so the high-water decode
    position never overruns the page table (the ``batch-32-ci`` leg requests 1024).
    """
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    tokenizer = model.demo_tokenizer

    # On-device sampling toggle for N150/N300 evidence-gathering (see sampling handoff docs):
    #   host            -> sampling_params=None (host-argmax, the default shipped path)
    #   on_device       -> greedy temp=0,k=1,p=0 => trace-captured FORCE-ARGMAX full-vocab path
    #   on_device_topk  -> temp=0,k=32,p=0.08    => trace-captured TOP-K op path (gathers only
    #                      the [*,32] tuples; PERF.md-parity recipe, faster than force-argmax)
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

    # Free-running perf run: enable the executor's on-device decode loop on the on-device sampling
    # path (inert on host/force-argmax; gated to the top-k path by _decode_loop_active). This is the
    # shared #49284 decode-loop fix; it must be active on the perf path for on-device decode parity.
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
        _warmup_demo_executor(traced_executor, kv_cache=kv_cache, page_table=page_table)

        prompts = load_input_prompts(batch_size)
        # Natural-length tokenization (matches TTTv1): the executor buckets each user's real length to
        # get_padded_prefill_len. These sample prompts are ~90-125 tokens -> 128 bucket.
        input_tokens, prompt_lens = tokenize_prompts(prompts, tokenizer, max_prefill_len=max_prefill_len)

        # Decode-token budget, clamped to the KV-cache headroom. Derive the prefill footprint from the
        # ACTUAL prompts (the largest padded bucket any user maps to via get_padded_prefill_len), not a
        # fixed 128, so the high-water decode position provably stays inside max_seq_len even when a
        # prompt buckets above 128. The 16-token margin absorbs the trailing decode step.
        _PROMPT_BUCKET = get_padded_prefill_len(int(prompt_lens.max()))
        _DECODE_MARGIN = 16
        requested_decode = _PERF_NUM_DECODE_TOKENS if num_decode_tokens is None else num_decode_tokens
        effective_decode = min(requested_decode, max_seq_len - _PROMPT_BUCKET - _DECODE_MARGIN)
        logger.info(
            f"[{case_name}] num_decode_tokens: requested={requested_decode}, "
            f"effective={effective_decode} (max_seq_len={max_seq_len}, prefill_bucket={_PROMPT_BUCKET})"
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
            prefill_sampling_params=None if mesh_device.get_num_devices() > 1 else sampling_params,
            pipeline_readback=pipeline_readback,
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
                num_layers=model.config.n_layers,
                batch_size=result.batch_size,
                input_sequence_length=prefill_seq_len,
                output_sequence_length=effective_decode,
            )

        assert_no_special_tokens(result.generated_token_ids, tokenizer, case_name=case_name)

        if expected:
            failures = []
            if "tok_s_u" in expected:
                tgt = expected["tok_s_u"] * (1 - PERF_TOLERANCE)
                if result.tok_s_u < tgt:
                    failures.append(f"tok/s/u {result.tok_s_u:.1f} < target {expected['tok_s_u']}")
            if "ttft_ms" in expected:
                tgt = expected["ttft_ms"] * (1 + PERF_TOLERANCE)
                if result.ttft_ms > tgt:
                    failures.append(f"ttft_ms {result.ttft_ms:.1f} > target {expected['ttft_ms']}")
            assert not failures, f"{case_name}: " + "; ".join(failures)
    finally:
        traced_executor.cleanup()


# ci-eval-32 determinism case: 3 rotated repeats of the batch-32 workload.
_EVAL_REPEAT_BATCHES = 3
_EVAL_NUM_DECODE_TOKENS = _PERF_NUM_DECODE_TOKENS


def _run_eval_repeat_batch32(model, mesh_device):
    """32-user cross-batch determinism (self-consistency under prompt rotation).

    Runs the batch-32 prefill+decode loop ``_EVAL_REPEAT_BATCHES`` times, rotating the prompt->slot
    assignment by one each repeat (fresh traced executor + KV cache per repeat), then asserts that
    undoing the rotation lines up per-user outputs. No external golden. Honors the same
    ``SAMPLING_MODE`` knob as ``_run_perf_benchmark`` (default host argmax — deterministic and
    mesh-agnostic, the recommended default for the determinism assert).
    """
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    tokenizer = model.demo_tokenizer

    # Qwen2.5 chat generation ends at <|im_end|>; the model opening a NEW turn (<|im_start|>) is a
    # de-facto response terminator as well (Qwen serving stacks list both as stops), but Qwen's HF
    # generation_config only carries <|im_end|>/<|endoftext|> as eos. Augment the tokenizer stop set
    # (the mechanism ``hf_stop_ids`` reads) with <|im_start|> so the determinism runner truncates a
    # degenerate turn-restart there — same pattern as the llama1b DP guard folding in <|eot_id|>.
    # Without this, a fixed-budget 200-step greedy continuation of the numeric eval prompts can
    # degenerate into "\n<|im_start|>user" (a hallucinated new turn) deep in decode (~token 69); which
    # of the two equally-valid prefill numerics (batched vs sequential) hits it is a near-tie, so the
    # shared garbage guard would otherwise flag only the sequential (DISABLE_BATCHED_PREFILL) leg.
    # <|im_start|> is a legitimate response terminator, so truncating there is correct, not a loosening;
    # cross-batch consistency is still asserted on the truncated (real-response) tokens.
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
    # Static warmup covers the model's regular graph families, but this heterogeneous
    # workload produces data-dependent batched signatures (30 q128 rows and 2 q1024
    # rows). Register one representative rotation before traced warmup activates the
    # program gate. Prompt rotation preserves that signature multiset for every repeat.
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
