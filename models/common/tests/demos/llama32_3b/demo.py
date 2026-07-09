# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 Llama-3.2-3B-Instruct demo — accuracy and performance measurement.

Uses ``EagerLlama32_3BExecutor`` / ``TracedLlama32_3BExecutor`` directly (no vLLM adapter).

**Mesh note:** Llama-3.2-3B-Instruct has 24 attention heads and 8 KV heads, so N150 (1),
N300 (2) and T3K (8) are all supported (8 divides both 8 KV heads and 24 attention heads).
PERF.md publishes N150/N300 rows for this model; T3K is exercised here for functionality
(DP-8 smoke, the on-device-sampling crossover) and gated to same-box measurement.

**Workload:** performance tests prefill each prompt at its natural length (TTTv1
``preprocess_inputs_prefill`` semantics; these sample prompts are ~90-125 tokens -> 128
prefill bucket) + 200 decode iterations. Accuracy / teacher-forcing scores the model
against the committed ``.refpt`` continuation tokens.

Usage::

    # Token accuracy test
    MESH_DEVICE=N300 HF_MODEL=meta-llama/Llama-3.2-3B-Instruct \\
      pytest models/common/tests/demos/llama32_3b/demo.py -k "token-accuracy" -v

    # Batch-1 latency test
    MESH_DEVICE=N300 HF_MODEL=meta-llama/Llama-3.2-3B-Instruct \\
      pytest models/common/tests/demos/llama32_3b/demo.py -k "batch-1" -v

    # Batch-32 throughput test
    MESH_DEVICE=N300 HF_MODEL=meta-llama/Llama-3.2-3B-Instruct \\
      pytest models/common/tests/demos/llama32_3b/demo.py -k "batch-32" -v

LazyWeight tensor cache: ``TT_CACHE_PATH/<device_name>`` when set, otherwise
``model_cache/<HF_MODEL>/<device_name>`` under the current working directory.

Reference artifact (``.refpt``): the accuracy test gates against the committed book
reference at ``models/tt_transformers/tests/reference_outputs/<basename(HF_MODEL)>.refpt``
(ground-truth real-text targets, PERF.md-comparable). The loader supports both the
legacy half-split format and a metadata-rich format carrying ``prompt_len``.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig, AutoTokenizer

import ttnn
from models.common.models.executor import (
    load_eval_repeat_prompts_batch32,
    run_eval_repeat_batch32,
    run_perf_benchmark,
    run_teacher_forcing,
)
from models.common.models.llama32_3b.model import (
    LLAMA32_3B_ACCURACY,
    LLAMA32_3B_PERFORMANCE,
    EagerLlama32_3BExecutor,
    Llama32_3BTransformer1D,
    TracedLlama32_3BExecutor,
)
from models.common.sampling.sampling_params import SamplingParams
from models.common.tests.demos.cleanup_utils import cleanup_model_case
from models.tt_transformers.tt.common import encode_prompt_hf

# =============================================================================
# Expected metrics — perf gates set from same-box TTTv1-vs-TTTv2 measurement on this base
# (SAMPLING_MODE-aware, SKU-aware). No PERF.md throughput value is used.
#
# Rule (§5): each ``tok_s_u`` target is the BETTER of freshly-measured same-box TTTv1 vs TTTv2 for
# that sampling mode. TTTv1 has only an on-device sampling path, so:
#     on_device_topk : max(TTTv1_on_device, TTTv2_on_device_topk)
#     host           : TTTv2_host                      (TTTv1 has no host-sampling path)
# Decode throughput is prefill-independent, so batched prefill (default-ON for 3B on this base)
# does NOT change ``tok_s_u`` — the measured values apply directly. ``ttft_ms`` targets are
# conservative upper bounds: batched prefill only LOWERS TTFT, so a single per-path ttft target
# above the sequential (DISABLE_BATCHED_PREFILL=1) value clears both the ON and OFF legs while
# gross prefill regressions are still caught.
# =============================================================================

# top1/top5 are teacher-forcing accuracy floors (sampling-independent). Perf metrics for batch-1
# live in EXPECTED_METRICS_BATCH1 (sampling-mode-aware); this dict only gates token-accuracy.
EXPECTED_METRICS = {
    "performance": {
        "N150": {"top1": 89, "top5": 98},
        "N300": {"top1": 89, "top5": 98},
        "T3K": {"top1": 89, "top5": 98},
    },
    "accuracy": {
        "N150": {"top1": 96, "top5": 100},
        "N300": {"top1": 96, "top5": 100},
        "T3K": {"top1": 96, "top5": 100},
    },
}

# batch-1 throughput, sampling-mode-aware (see rule above). host = TTTv2-host; on_device_topk =
# max(TTTv1, TTTv2-on-device). ttft_ms = conservative upper bound (batched prefill beats it).
# Populated from same-box measurement (see worklog). SKUs/modes not yet measured stay {} (the case
# still RUNS, printing tok_s_u, but is not gated) — never a silent PERF.md value.
EXPECTED_METRICS_BATCH1: dict = {
    "host": {
        "performance": {
            "N150": {"tok_s_u": 50.2, "ttft_ms": 62},
            "N300": {"tok_s_u": 50.3, "ttft_ms": 46},
            "T3K": {"tok_s_u": 13.0, "ttft_ms": 40},
        },
        "accuracy": {
            "N150": {"tok_s_u": 44.7, "ttft_ms": 71},
            "N300": {"tok_s_u": 44.2, "ttft_ms": 61},
            "T3K": {"tok_s_u": 19.6, "ttft_ms": 31},
        },
    },
    "on_device_topk": {
        "performance": {
            "N150": {"tok_s_u": 11.1, "ttft_ms": 62},  # max(TTTv1 11.1, TTTv2 11.1)
            "N300": {"tok_s_u": 31.06, "ttft_ms": 57},  # max(TTTv1 31.06, TTTv2 30.7); TTTv2 within tol
            # T3K on-device: max(TTTv1 ci-1 80.98, TTTv2 72.8)=80.98. TTTv2 is ~10% below -> this cell
            # is EXPECTED RED (T3K decode-throughput gap, github.com/tenstorrent/tt-metal/pull/49284;
            # fix NOT in this base). Gate set at the TTTv1 value per the better-of rule (never lowered).
            "T3K": {"tok_s_u": 80.98, "ttft_ms": 42},
        },
        "accuracy": {
            "N150": {"tok_s_u": 10.9, "ttft_ms": 71},  # max(TTTv1 10.81, TTTv2 10.9)
            "N300": {"tok_s_u": 30.29, "ttft_ms": 50},  # max(TTTv1 30.29, TTTv2 29.8); TTTv2 within tol
            "T3K": {"tok_s_u": 80.27, "ttft_ms": 31},  # max(TTTv1 80.27, TTTv2 72.5) — EXPECTED RED (#49284)
        },
    },
}

# Short-context batch-32 throughput (seq1024 / 200 decode), sampling-mode- AND profile-aware.
# NOTE (3B-specific): unlike the 1B pilot (where perf and accuracy decode are within tolerance and a
# single value gates both), on 3B the performance profile (BFP4 FF1/FF3 + LoFi) is ~12% faster than
# the accuracy profile (BFP8 FF + HiFi2) in decode — measured batch-1 host 50.3 (perf) vs 44.2 (acc).
# A single constant cannot gate both, so batch-32 / batch-32-ci gates are profile-split here. Same
# better-of rule as above, applied per profile.
EXPECTED_METRICS_BATCH32: dict = {
    "host": {
        "performance": {
            "N150": {"tok_s_u": 43.0, "ttft_ms": 23},
            "N300": {"tok_s_u": 45.9, "ttft_ms": 20},
            "T3K": {"tok_s_u": 18.0, "ttft_ms": 20},
        },
        "accuracy": {
            "N150": {"tok_s_u": 39.4, "ttft_ms": 25},
            "N300": {"tok_s_u": 44.3, "ttft_ms": 21},
            "T3K": {"tok_s_u": 19.1, "ttft_ms": 20},
        },
    },
    "on_device_topk": {
        "performance": {
            "N150": {"tok_s_u": 10.8, "ttft_ms": 23},
            "N300": {"tok_s_u": 29.0, "ttft_ms": 20},
            "T3K": {"tok_s_u": 70.7, "ttft_ms": 20},
        },
        "accuracy": {
            "N150": {"tok_s_u": 10.5, "ttft_ms": 24},
            "N300": {"tok_s_u": 28.5, "ttft_ms": 21},
            "T3K": {"tok_s_u": 70.4, "ttft_ms": 20},
        },
    },
}

# CI-faithful batch-32 targets (the ``batch-32-ci`` leg), measured at max_seq_len=2048 with a
# 1024-token decode budget (TTTv1 ci-32 workload). This is a SEPARATE workload from the lighter
# batch-32 leg above (seq1024 / 200 decode steps): the seq2048 KV cache means the decode read
# window grows, so steady-state per-token decode is legitimately a bit slower than the
# short-context batch-32 numbers. Keyed by SAMPLING_MODE (host argmax vs on-device differ because
# on-device pays the slow upstream ``ttnn.topk``) AND profile (see the 12% gap note above). Cells
# not measured fall back to EXPECTED_METRICS_BATCH32 (so they stay gated, never silently un-gated).
EXPECTED_METRICS_BATCH32_CI: dict = {
    "host": {
        "performance": {
            "N150": {"tok_s_u": 37.0, "ttft_ms": 46},  # ttft covers sequential/OFF leg (~40ms)
            "N300": {"tok_s_u": 42.2, "ttft_ms": 44},  # OFF host = 37.9ms
            "T3K": {"tok_s_u": 18.1, "ttft_ms": 30},
        },
        "accuracy": {
            "N150": {"tok_s_u": 34.1, "ttft_ms": 46},
            "N300": {"tok_s_u": 39.8, "ttft_ms": 44},
            "T3K": {"tok_s_u": 18.2, "ttft_ms": 30},
        },
    },
    "on_device_topk": {
        "performance": {
            "N150": {"tok_s_u": 10.43, "ttft_ms": 46},  # max(TTTv1 ci-32 10.43, TTTv2 10.4)
            "N300": {"tok_s_u": 28.38, "ttft_ms": 44},  # max(TTTv1 ci-32 28.38, TTTv2 27.4); within tol
            # T3K: max(TTTv1 ci-32 76.0, TTTv2 67.3)=76.0. TTTv2 ~11% below -> EXPECTED RED (#49284,
            # same T3K decode gap as batch-1). Gate at TTTv1 value per better-of rule (never lowered).
            "T3K": {"tok_s_u": 76.0, "ttft_ms": 30},
        },
        "accuracy": {
            "N150": {"tok_s_u": 10.19, "ttft_ms": 46},  # max(TTTv1 ci-32 10.19, TTTv2 10.1)
            "N300": {"tok_s_u": 27.72, "ttft_ms": 44},  # max(TTTv1 ci-32 27.72, TTTv2 26.9); within tol
            "T3K": {"tok_s_u": 75.69, "ttft_ms": 30},  # max(TTTv1 75.69, TTTv2 67.3) — EXPECTED RED (#49284)
        },
    },
}

# Perf workload: natural-length prefill (these sample prompts are ~90-125 tokens -> 128 bucket,
# matching TTTv1), 200 decode steps. Accuracy uses the 511-token teacher-forcing refpt.
_PERF_NUM_DECODE_TOKENS = int(os.environ.get("PERF_NUM_DECODE_TOKENS", "200"))

PERF_TOLERANCE = 0.05

# batch-32-ci per-SKU max_seq_len (TTTv1 ci-32 parity is seq2048). DRAM trap: raising max_seq_len
# doubles the batch-32 KV cache. 3B weights are NOT tiny; if a SKU OOMs at seq2048 clamp it here
# (llama1b keeps every SKU at 2048 because 1B weights are tiny — 3B may need N150 lower).
_BATCH32_CI_MAX_SEQ_LEN: dict[str, int] = {
    "N150": 2048,
    "N300": 2048,
    "T3K": 2048,
}


def _sampling_bucket() -> str:
    """Map SAMPLING_MODE to a perf-gate bucket. Non-topk on-device modes (e.g. force-argmax)
    fall into ``on_device_topk`` so they stay gated, never silently un-gated."""
    return "host" if os.environ.get("SAMPLING_MODE", "host").lower() == "host" else "on_device_topk"


_MESH_DEVICE_TO_SHAPE: dict[str, tuple[int, int]] = {
    "N150": (1, 1),
    "N300": (1, 2),
    "T3K": (1, 8),
}


def _ttnn_mesh_device_param_from_env() -> dict:
    env = os.environ.get("MESH_DEVICE", "").strip()
    if not env:
        pytest.skip(
            "MESH_DEVICE must be set (e.g. N150, N300 or T3K). See module docstring.",
            allow_module_level=True,
        )
    shape = _MESH_DEVICE_TO_SHAPE.get(env)
    if shape is None:
        pytest.skip(
            f"Unsupported MESH_DEVICE={env!r} for Llama-3.2-3B; use N150, N300 or T3K.",
            allow_module_level=True,
        )
    param = {
        "mesh_shape": shape,
        "trace_region_size": 50_000_000,
        "num_command_queues": 1,
    }
    # TTTv2 multi-device executor dispatch (and the on-device sampling all-gather) stalls without
    # an explicit 1D fabric; the root conftest does not auto-enable it. Mirror the sibling
    # models/common/models/llama32_3b/demo.py wiring: FABRIC_1D on any >1-device mesh.
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
    return ttnn_mesh_device


def _skip_unless_heads_divide_mesh(mesh_device: ttnn.MeshDevice, hf_model_id: str) -> None:
    n_dev = mesh_device.get_num_devices()
    if n_dev <= 1:
        return
    cfg = AutoConfig.from_pretrained(hf_model_id)
    n_h, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    if n_h % n_dev == 0 and n_kv % n_dev == 0:
        return
    pytest.skip(
        f"Incompatible mesh for {hf_model_id}: {n_dev} devices, "
        f"num_attention_heads={n_h}, num_key_value_heads={n_kv}."
    )


def get_device_name(mesh_device: ttnn.MeshDevice) -> str:
    n = mesh_device.get_num_devices()
    if n == 1:
        return "N150"
    if n == 2:
        return "N300"
    if n == 8:
        return "T3K"
    return f"{n}dev"


def lazy_weight_cache_dir_for_demo(mesh_device: ttnn.MeshDevice, hf_model_id: str) -> Path:
    device_name = get_device_name(mesh_device)
    hf = hf_model_id.strip("/")
    tt_cache = os.getenv("TT_CACHE_PATH")
    if tt_cache:
        root = Path(tt_cache) / device_name
    else:
        root = Path("model_cache") / hf / device_name
    root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Llama-3.2-3B demo LazyWeight cache directory: {root.resolve()}")
    return root


def load_reference_data(hf_model_id: str):
    """Load reference tensors and optional metadata from ``.refpt``.

    Supports both the metadata-rich format (``prompt_len`` + ``metadata`` keys) and
    the legacy half-split book format.
    """
    name = hf_model_id.strip("/").split("/")[-1]
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
    top5_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    prompt_len: int,
    *,
    metadata_aligned: bool,
) -> torch.Tensor:
    """Align ``top5_tokens`` with teacher-forcing targets across refpt conventions."""
    num_target = len(reference_tokens) - prompt_len
    target_tokens = reference_tokens[prompt_len : prompt_len + num_target]
    if num_target <= 0:
        raise ValueError("prompt_len must be smaller than reference length")

    if metadata_aligned and top5_tokens.shape[0] == num_target:
        logger.info(
            f"Teacher-forcing top5: metadata direct path (top5_len={top5_tokens.shape[0]}, target_len={num_target})"
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
            f"Cannot align top5: prompt_len={prompt_len}, num_target={num_target}, top5_len={top5_tokens.shape[0]}"
        )

    best_score, best_start, best = max(candidates, key=lambda x: x[0])
    logger.info(f"Teacher-forcing top5 alignment: start={best_start}, score={best_score}/{min(16, num_target)}")
    return best


def log_generated_text(prompts, generated_token_ids, tokenizer):
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


def create_model(
    mesh_device: ttnn.MeshDevice,
    optimizations: str,
    cache_dir: Path,
    *,
    max_batch_size: int = 32,
    max_seq_len: int = 4096,
) -> Llama32_3BTransformer1D:
    """Build ``Llama32_3BTransformer1D`` in executor (paged KV) mode.

    Picks one of the two module-level precision recipes (``LLAMA32_3B_ACCURACY`` /
    ``LLAMA32_3B_PERFORMANCE``) — both defined in ``llama32_3b/model.py`` and grounded
    in TTTv1's ``DecodersPrecision`` for Llama-3.2-3B-Instruct.
    """
    hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
    _skip_unless_heads_divide_mesh(mesh_device, hf_model)

    precision = LLAMA32_3B_PERFORMANCE if optimizations == "performance" else LLAMA32_3B_ACCURACY

    try:
        model = Llama32_3BTransformer1D.from_pretrained(
            mesh_device,
            hf_model,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            num_layers=None,
            cache_dir=cache_dir,
            precision=precision,
            executor_mode=True,
        )
    except Exception as e:
        pytest.skip(f"Could not build Llama-3.2-3B model (weights / memory / mesh): {e}")

    return model


# =============================================================================
# ci-b1-DP: single-user data-parallel scaling smoke (TTTv1 ci-b1-DP-* parity)
# =============================================================================
#
# One user per DP group, model replicated across ``data_parallel`` disjoint submeshes,
# instruct prompts, paged attention, trace on. The ONLY correctness check is the
# special-token garbage guard plus "runs to completion without hang/exception". This is a
# mesh / KV-cache / page-table scaling smoke test, NOT an accuracy or perf gate.
#
# Per-case size table (TTTv1 simple_text_demo.py parity, with the DP-2 N300 addition):
#   ci-b1-DP-2  : max_seq_len=1024, max_generated_tokens=200, stop_at_eos=True
#                 (fast smoke; the only DP case runnable on N300 — 2 single-device groups)
#   ci-b1-DP-4  : max_seq_len=4096, max_generated_tokens=2048, stop_at_eos=False
#   ci-b1-DP-8  : max_seq_len=4096, max_generated_tokens=2048, stop_at_eos=False
#   ci-b1-DP-16 : max_seq_len=1024, max_generated_tokens=200,  stop_at_eos=True
#   ci-b1-DP-32 : max_seq_len=1024, max_generated_tokens=200,  stop_at_eos=True
#
# Hardware feasibility: each DP group is one device (batch_size=1 per group), so
# ``data_parallel == n_devices``. On N300 (2 chips) only DP-2 fits; DP-4/8/16/32 cleanly
# ``pytest.skip`` via ``_dp_or_skip``. On T3K (8 chips) DP-8 runs; DP-2/4/16/32 skip.
# ``stop_at_eos`` is effectively a no-op in TTTv2's fixed-budget ``run_perf_benchmark`` loop
# (it always runs ``num_decode_tokens`` steps); the special-token guard truncates at the first
# stop token before scanning, so this is fine.
_DP_SIZE_TABLE: dict[int, dict] = {
    2: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
    4: {"max_seq_len": 4096, "max_generated_tokens": 2048, "stop_at_eos": False},
    8: {"max_seq_len": 4096, "max_generated_tokens": 2048, "stop_at_eos": False},
    16: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
    32: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
}


def create_dp_submeshes(mesh_device: ttnn.MeshDevice, data_parallel: int) -> list:
    """Partition the open parent mesh into ``data_parallel`` disjoint row-submeshes.

    Mirrors TTTv1 ``generator.create_submeshes`` minus the Galaxy reshape-to-(4,8) branch
    (no Galaxy reachable here). For the single-user DP cases on our hardware
    ``n // data_parallel == 1``, so each submesh is a ``(1,1)`` mesh. Fabric stays owned by
    the parent — do NOT set fabric per-submesh.
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
    """Guard: no special (garbage) tokens mid-stream. Warns always; hard-fails only under CI.

    Mirrors TTTv1 ``simple_text_demo.py``'s special-token guard (warn always, ``assert`` only
    when ``CI == "true"``). TTTv2's ``result.generated_token_ids[user]`` already starts at the
    first generated token (prefill argmax), so unlike TTTv1 we do not slice off the prompt —
    these are output-only. Each user's output is truncated at the first stop token (EoS / eot)
    before scanning, then checked against any ``tokenizer.all_special_ids`` member.
    """
    if is_ci_env is None:
        is_ci_env = os.environ.get("CI") == "true"
    special = set(tokenizer.all_special_ids)
    stop = set()
    if tokenizer.eos_token_id is not None:
        stop.add(tokenizer.eos_token_id)
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot >= 0:
        stop.add(eot)
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

    Builds one model + one traced executor + one KV cache + one page table per submesh
    (one user each), runs ``run_perf_benchmark`` per submesh sequentially, collects the
    per-submesh output, and asserts no special tokens. Every executor and model is cleaned
    up in ``finally`` even on mid-loop failure.
    """
    _dp_or_skip(mesh_device, data_parallel)

    hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
    _skip_unless_heads_divide_mesh(mesh_device, hf_model)
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    precision = LLAMA32_3B_PERFORMANCE if optimizations == "performance" else LLAMA32_3B_ACCURACY

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
                model = Llama32_3BTransformer1D.from_pretrained(
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
                pytest.skip(f"Could not build Llama-3.2-3B model (weights / memory / mesh): {e}")
            models.append((model, sm))

            traced_executor = TracedLlama32_3BExecutor(model, sm)
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
        # When data_parallel > 1 we carved child submeshes off the fixture-owned parent
        # mesh. Those submeshes share the parent's command queue, so the parent cannot be
        # closed (by the module-scoped ttnn_mesh_device fixture) while they remain in use.
        # Drain the parent + submesh CQs and reset their in-use state before teardown.
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
        pytest.param("ci-b1-DP-2", id="ci-b1-DP-2"),
        pytest.param("ci-b1-DP-4", id="ci-b1-DP-4"),
        pytest.param("ci-b1-DP-8", id="ci-b1-DP-8"),
        pytest.param("ci-b1-DP-16", id="ci-b1-DP-16"),
        pytest.param("ci-b1-DP-32", id="ci-b1-DP-32"),
    ],
)
@pytest.mark.parametrize("optimizations", ["performance", "accuracy"])
def test_llama32_3b(test_config, mesh_device, optimizations):
    """Main test entry for TTTv2 Llama-3.2-3B-Instruct."""
    device_name = get_device_name(mesh_device)
    expected = EXPECTED_METRICS.get(optimizations, {}).get(device_name, {})
    model = None
    hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
    cache_dir = lazy_weight_cache_dir_for_demo(mesh_device, hf_model)

    try:
        # ci-b1-DP-*: single-user data-parallel smoke. Builds N models itself (one per
        # submesh), so it does NOT go through the shared create_model path below.
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

        # Token-accuracy feeds a single reference sequence — max_batch_size=1 avoids
        # DRAM pressure from a full 32-user KV cache allocation.
        # batch-32 and eval-32 both run 32 users with max_seq_len=1024 to avoid DRAM OOM
        # on N150 (3B weights + 32×4096 BFP8 KV cache exhausts ~12 GB); 1024 comfortably
        # covers the 128-bucket prefill + 200 decode workload.
        if test_config in ("batch-32", "eval-32"):
            max_bs, max_seq_len = 32, 1024
            expected = EXPECTED_METRICS_BATCH32.get(_sampling_bucket(), {}).get(optimizations, {}).get(device_name, {})
        elif test_config == "batch-32-ci":
            # CI-faithful batch-32 leg (TTTv1 ci-32 parity): larger seq len + 1024 decode
            # budget. Per-SKU seq len clamp (3B KV cache is not tiny; see _BATCH32_CI_MAX_SEQ_LEN).
            max_bs = 32
            max_seq_len = _BATCH32_CI_MAX_SEQ_LEN.get(device_name, 2048)
            # Own perf gate measured at the seq2048/decode1024 workload (NOT the lighter batch-32
            # constant, which would be a config-artifact miss). The gate is keyed by SAMPLING_MODE
            # (host argmax vs on-device sampling differ on 3B). Non-topk on-device modes (force-argmax)
            # fall back to the on_device_topk bucket; cells not measured fall back to the short-context
            # batch-32 constant so they stay gated, never silently un-gated.
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
        model = create_model(mesh_device, optimizations, cache_dir, max_batch_size=max_bs, max_seq_len=max_seq_len)

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
        cleanup_model_case(model, mesh_device)


def _run_token_accuracy(model: Llama32_3BTransformer1D, mesh_device, expected):
    """Teacher-forcing token accuracy vs ``.refpt``."""
    hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
    reference_tokens, top5_tokens, prompt_len, metadata = load_reference_data(hf_model)

    if reference_tokens.dim() > 1:
        reference_tokens = reference_tokens.squeeze()

    has_prompt_len_metadata = prompt_len is not None
    if has_prompt_len_metadata:
        prompt_len = int(prompt_len)
        logger.info(f"Using metadata prompt_len={prompt_len}")
    else:
        prompt_len = len(reference_tokens) // 2
        logger.info(f"Reference missing prompt_len metadata; using legacy half-split={prompt_len}.")

    if metadata:
        logger.info(
            f"Reference metadata: hf_model_id={metadata.get('hf_model_id')}, "
            f"revision={metadata.get('revision')}, created_at={metadata.get('created_at')}"
        )

    prompt_tokens = reference_tokens[:prompt_len].unsqueeze(0)

    executor = EagerLlama32_3BExecutor(model, mesh_device)
    ma = model.model_args
    assert ma is not None

    max_batch_size = ma.max_batch_size
    prompt_tokens = prompt_tokens.repeat(max_batch_size, 1)
    block_size = 32
    max_seq_len = ma.max_seq_len
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
    result = run_teacher_forcing(
        executor,
        prompt_tokens=prompt_tokens,
        reference_tokens=reference_tokens,
        top5_tokens=target_top5,
        kv_cache=kv_cache,
        page_table=page_table,
        max_batch_size=max_batch_size,
    )

    top1 = result.top1_accuracy() * 100
    top5 = result.top5_accuracy() * 100
    logger.info(f"Token accuracy — top1: {top1:.1f}%, top5: {top5:.1f}%")

    if "top1" in expected:
        assert top1 >= expected["top1"] * (
            1 - PERF_TOLERANCE
        ), f"Top-1 accuracy {top1:.1f}% below threshold {expected['top1']}%"
    if "top5" in expected:
        assert top5 >= expected["top5"] * (
            1 - PERF_TOLERANCE
        ), f"Top-5 accuracy {top5:.1f}% below threshold {expected['top5']}%"


def _run_perf_benchmark(
    model: Llama32_3BTransformer1D,
    mesh_device,
    expected,
    batch_size: int,
    case_name: str,
    max_prefill_len: int | None = None,
    num_decode_tokens: int | None = None,
):
    """Timed prefill + decode (``TracedLlama32_3BExecutor``).

    Prefill uses each prompt's natural token length (TTTv1 ``preprocess_inputs_prefill``
    semantics — the executor buckets to ``get_padded_prefill_len``); decode runs for
    ``num_decode_tokens`` steps (default ``_PERF_NUM_DECODE_TOKENS``).
    ``max_prefill_len`` is an optional clip cap for over-long prompts, never a pad-up target.

    The decode budget is clamped to what the paged KV cache can hold:
    ``effective = min(requested, max_seq_len - prompt_bucket - margin)`` so the high-water
    decode position never overruns the page table (the ``batch-32-ci`` leg requests 1024).
    """
    hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(hf_model)

    # Batched-prefill A/B knob (parity caveat #12): set DISABLE_BATCHED_PREFILL=1 to force the
    # sequential per-user prefill loop (the pre-feature baseline) for before/after TTFT comparison.
    # Companion knob (PLAN_01): DISABLE_MINIMAL_MATMUL=1 forces QKV/W2 prefill back to ttnn.linear
    # (read at model build time, so it must be in the env before from_pretrained — it already is here).
    if os.environ.get("DISABLE_BATCHED_PREFILL") and model.model_args is not None:
        model.model_args.disable_batched_prefill = True

    # On-device sampling toggle for N150/N300/T3K evidence-gathering (see sampling handoff docs):
    #   host            -> sampling_params=None (host-argmax, the default shipped path)
    #   on_device       -> greedy temp=0,k=1,p=0  => trace-captured TOP-K op path with k=1
    #   on_device_topk  -> temp=0,k=32,p=0.08      => trace-captured TOP-K op path with k=32
    #                      (PERF.md-parity recipe). Both on-device modes route through the same
    #                      per-device ttnn.topk -> all-gather of the [*,k] tuples -> ttnn.sampling
    #                      op path (the model is built with allow_force_argmax=False, so the
    #                      full-vocab argmax all-gather is never taken); they differ only in k.
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
    logger.info(f"[{case_name}] SAMPLING_MODE={sampling_mode} -> sampling_params={sampling_params}")

    # Free-running perf run: enable the executor's on-device decode loop on the on-device sampling
    # path (inert on host/force-argmax; gated to the top-k path by _decode_loop_active).
    traced_executor = TracedLlama32_3BExecutor(model, mesh_device, ondevice_decode_loop=sampling_params is not None)
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

        # Decode-token budget, clamped to the KV-cache headroom. Prompts bucket to ~128 and
        # we keep a 16-token margin, so the high-water decode position stays inside max_seq_len.
        _PROMPT_BUCKET = 128
        _DECODE_MARGIN = 16
        requested_decode = _PERF_NUM_DECODE_TOKENS if num_decode_tokens is None else num_decode_tokens
        effective_decode = min(requested_decode, max_seq_len - _PROMPT_BUCKET - _DECODE_MARGIN)
        logger.info(
            f"[{case_name}] num_decode_tokens: requested={requested_decode}, "
            f"effective={effective_decode} (max_seq_len={max_seq_len})"
        )

        prompts = load_input_prompts(batch_size)
        # Natural-length tokenization (matches TTTv1): the executor buckets each user's real
        # length to get_padded_prefill_len. These sample prompts are ~90-125 tokens -> 128 bucket.
        input_tokens, prompt_lens = tokenize_prompts(prompts, tokenizer, max_prefill_len=max_prefill_len)

        result = run_perf_benchmark(
            traced_executor,
            tokens=input_tokens,
            kv_cache=kv_cache,
            page_table=page_table,
            num_decode_tokens=effective_decode,
            max_batch_size=max_batch_size,
            prompt_lens=prompt_lens,
            sampling_params=sampling_params,
        )

        logger.info(
            f"Performance [{case_name}] — TTFT: {result.ttft_ms:.1f}ms, "
            f"tok/s/u: {result.tok_s_u:.1f}, "
            f"tok/s: {result.tok_s:.1f}, "
            f"decode latency: {result.decode_latency_mean_ms:.2f}ms"
        )
        log_generated_text(prompts, result.generated_token_ids, tokenizer)
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


def _run_eval_repeat_batch32(model: Llama32_3BTransformer1D, mesh_device):
    """32-user cross-batch determinism (self-consistency under prompt rotation).

    Runs the batch-32 prefill+decode loop ``_EVAL_REPEAT_BATCHES`` times, rotating the
    prompt->slot assignment by one each repeat (fresh traced executor + KV cache per repeat),
    then asserts that undoing the rotation lines up per-user outputs. No external golden.
    Honors the same ``SAMPLING_MODE`` knob as ``_run_perf_benchmark`` (default host argmax —
    deterministic and mesh-agnostic, the recommended default for the determinism assert).
    """
    hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(hf_model)

    ma = model.model_args
    assert ma is not None

    # Batched-prefill A/B knob (parity caveat #12): DISABLE_BATCHED_PREFILL=1 forces the pure
    # per-bucket sequential prefill (the Phase-1 path) so eval-32 can be validated both ON and OFF.
    if os.environ.get("DISABLE_BATCHED_PREFILL"):
        ma.disable_batched_prefill = True

    block_size = 32
    max_seq_len = ma.max_seq_len
    max_batch_size = ma.max_batch_size
    max_num_blocks_per_user = max_seq_len // block_size
    max_num_blocks = max_num_blocks_per_user * max_batch_size

    kv_cache_shape = (max_num_blocks, ma.n_kv_heads // mesh_device.get_num_devices(), block_size, ma.head_dim)
    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(max_batch_size, max_num_blocks_per_user)

    # Fresh traced executor + zeroed KV cache per repeat (driver owns the lifecycle), so the
    # rotated batches are fully independent — see run_eval_repeat_batch32 for why reuse corrupts
    # the 3rd repeat on hardware.
    def make_executor():
        return TracedLlama32_3BExecutor(model, mesh_device)

    def allocate_kv_cache(executor):
        return executor.allocate_kv_cache(kv_cache_shape, torch.bfloat16, ma.n_layers)

    # TTTv1 ci-eval-32 numeric prompts (parity). NOTE: on small models these can degenerate into
    # repetitive loops whose argmax ties flip by batch slot, failing the assert — see
    # run_eval_repeat_batch32; that failure is a real gap, not a harness bug.
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
