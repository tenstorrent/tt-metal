# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 Mistral-7B-Instruct-v0.3 demo — accuracy and performance measurement.

Uses the model-owned ``Mistral7BExecutor`` directly (no vLLM adapter).

**Mesh note:** Mistral-7B-Instruct-v0.3 has 32 attention heads and 8 KV heads, so all of
N150 (1), N300 (2), T3K (8) are compatible (8 divides both). PERF.md publishes all three.

**Workload:** performance tests prefill each prompt at its natural length (TTTv1
``preprocess_inputs_prefill`` semantics; these sample prompts are ~90-125 tokens -> 128
prefill bucket) + 200 decode iterations. Accuracy / teacher-forcing scores the model
against the committed ``.refpt`` continuation tokens.

CI cases (parity with TTTv1 ``simple_text_demo.py``):
    token-accuracy   - teacher-forcing top-1/top-5 vs the book ``.refpt``
    batch-1          - single-user latency
    batch-32         - short-context throughput (seq1024 / 200 decode)
    batch-32-ci      - CI-faithful batch-32 (seq2048 / 1024 decode; TTTv1 ci-32); per-SKU seq clamp
    eval-32          - 32-user cross-batch determinism (TTTv1 ci-eval-32)
    ci-b1-DP-{2..32} - single-user data-parallel scaling smoke (TTTv1 ci-b1-DP-*)

Usage::

    # Token accuracy test
    MESH_DEVICE=N300 HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \\
      pytest models/common/tests/demos/mistral_7b/demo.py -k "token-accuracy" -v

    # Batch-1 latency test
    MESH_DEVICE=N300 HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \\
      pytest models/common/tests/demos/mistral_7b/demo.py -k "batch-1" -v

    # On-device sampling perf sweep
    SAMPLING_MODE=on_device_topk MESH_DEVICE=T3K HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \\
      pytest models/common/tests/demos/mistral_7b/demo.py -k "batch-32-ci" -v

LazyWeight tensor cache: ``TT_CACHE_PATH/<device_name>`` when set, otherwise
``model_cache/<HF_MODEL>/<device_name>`` under the current working directory.

Reference artifact (``.refpt``): the token-accuracy test gates on the committed book
reference ``models/tt_transformers/tests/reference_outputs/Mistral-7B-Instruct-v0.3.refpt``
(real-corpus teacher-forced targets), shared with the TTTv1 demo. The loader supports both
the metadata-rich format (``prompt_len``) and the book half-split format.
"""

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
from models.common.models.mistral_7b.executor import Mistral7BExecutor, Mistral7BExecutorConfig
from models.common.models.mistral_7b.hf_adaptor import from_pretrained
from models.common.models.mistral_7b.model import MISTRAL_ACCURACY, MISTRAL_PERFORMANCE, Mistral7B
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
from models.tt_transformers.tt.common import encode_prompt_hf

# =============================================================================
# Expected metrics — perf gates set from a same-box TTTv1-vs-TTTv2 sweep (on-device sampling),
# NOT PERF.md (PERF.md's Mistral N150/N300/T3K = 29.75/47.01/67.82 t/s/u were stale/aspirational;
# T3K 67.82 was met by neither stack).
#
# Rule (per cell): each ``tok_s_u`` target is the BETTER of TTTv1 vs TTTv2 for that sampling mode.
# TTTv1 has only an on-device sampling path, so:
#     on_device_topk : max(TTTv1_on_device, TTTv2_on_device_topk)
#     host           : TTTv2_host                      (TTTv1 has no host-sampling path)
# Decode throughput is prefill-independent, so batched prefill does NOT change ``tok_s_u``.
# ``ttft_ms`` targets are conservative upper bounds (batched prefill only LOWERS TTFT).
#
# MEASUREMENT-FIRST: the throughput dicts below are populated from same-box measurement. SKUs/modes
# not yet measured stay ``{}`` — the case still RUNS and prints tok_s_u but is not gated (never a
# silent PERF.md value). ``top1``/``top5`` are teacher-forcing accuracy floors (sampling-independent),
# the real gate for token-accuracy.
# =============================================================================

# top1/top5 teacher-forcing accuracy floors (book refpt). Perf metrics live in the batch dicts below.
EXPECTED_METRICS: dict = {
    "performance": {
        "N150": {"top1": 95, "top5": 99},
        "N300": {"top1": 95, "top5": 100},
        "T3K": {"top1": 95, "top5": 100},
    },
    "accuracy": {
        "N150": {"top1": 96, "top5": 100},
        "N300": {"top1": 97, "top5": 100},
        "T3K": {"top1": 98, "top5": 100},
    },
}

# batch-1 throughput, sampling-mode- and profile-aware. host = TTTv2-host; on_device_topk =
# max(TTTv1, TTTv2-on-device). Populated from same-box measurement; unmeasured cells stay {}.
# N300: TTTv2 odt (48.0/40.2) beats TTTv1 ci-1 (avg 41.73/38.25) on both profiles → gate = TTTv2.
# N150: host≈odt (32K vocab → cheap on-device sampling even on 1 dev). TTTv2 odt (30.5/26.4) ≥ TTTv1
# ci-1 (29.51/26.07) → gate = TTTv2.
# T3K: crossover SKU (odt >> host). TTTv2 odt (58.2/56.2) ≥ TTTv1 ci-1 (56.7/55.8) → gate = TTTv2.
# T3K host is dispatch-bound (host batch-1 acc 24.2 = cold-first-trace artifact) → gated to TTTv2-measured floor.
EXPECTED_METRICS_BATCH1: dict = {
    "host": {
        "performance": {
            "N150": {"tok_s_u": 30.4, "ttft_ms": 100},
            "N300": {"tok_s_u": 45.3, "ttft_ms": 70},
            "T3K": {"tok_s_u": 43.7, "ttft_ms": 42},
        },
        "accuracy": {
            "N150": {"tok_s_u": 26.3, "ttft_ms": 148},
            "N300": {"tok_s_u": 38.3, "ttft_ms": 92},
            "T3K": {"tok_s_u": 24.2, "ttft_ms": 50},
        },
    },
    "on_device_topk": {
        "performance": {
            "N150": {"tok_s_u": 30.5, "ttft_ms": 100},
            "N300": {"tok_s_u": 48.0, "ttft_ms": 70},
            "T3K": {"tok_s_u": 58.2, "ttft_ms": 42},
        },
        "accuracy": {
            "N150": {"tok_s_u": 26.4, "ttft_ms": 148},
            "N300": {"tok_s_u": 40.2, "ttft_ms": 92},
            "T3K": {"tok_s_u": 56.2, "ttft_ms": 50},
        },
    },
}

# Short-context batch-32 throughput (seq1024 / 200 decode), sampling-mode- and profile-aware.
# On a 7B the perf profile (BFP4 FF1/FF3 + LoFi) and accuracy profile (BFP8 FF + HiFi2) decode can
# differ >5%, so gates are profile-split (like the 3B pilot, unlike tiny 1B). Same better-of rule.
# batch-32 (short seq1024/200) has no matching TTTv1 CI workload (TTTv1's CI batch-32 IS ci-32 =
# our batch-32-ci) → gate = TTTv2-measured (regression gate), host and on_device_topk both.
EXPECTED_METRICS_BATCH32: dict = {
    "host": {
        "performance": {
            "N150": {"tok_s_u": 27.9, "ttft_ms": 36},
            "N300": {"tok_s_u": 41.3, "ttft_ms": 30},
            "T3K": {"tok_s_u": 40.0, "ttft_ms": 18},
        },
        "accuracy": {
            "N150": {"tok_s_u": 24.5, "ttft_ms": 44},
            "N300": {"tok_s_u": 35.0, "ttft_ms": 38},
            "T3K": {"tok_s_u": 41.0, "ttft_ms": 24},
        },
    },
    "on_device_topk": {
        "performance": {
            "N150": {"tok_s_u": 28.0, "ttft_ms": 36},
            "N300": {"tok_s_u": 44.3, "ttft_ms": 30},
            "T3K": {"tok_s_u": 57.0, "ttft_ms": 18},
        },
        "accuracy": {
            "N150": {"tok_s_u": 24.5, "ttft_ms": 44},
            "N300": {"tok_s_u": 37.8, "ttft_ms": 38},
            "T3K": {"tok_s_u": 55.1, "ttft_ms": 24},
        },
    },
}

# CI-faithful batch-32 targets (the ``batch-32-ci`` leg = TTTv1 ci-32 workload). Keyed by SAMPLING_MODE
# AND profile; cells not measured fall back to EXPECTED_METRICS_BATCH32 (stay gated, never un-gated).
# tok/s/u gates are the prior-healthy best-of {TTTv2 odt, TTTv1 ci-32} (never lowered).
# ttft_ms gates now reflect batched prefill (ON; single-pass 32-fold on >=2-dev, 8-fold on N150): the 32
# users fold into ONE traced prefill pass so TTFT matches TTTv1's batched prefill. Same-box 2026-07-17
# (tolerance-free): N300 v2 25.6 == v1 25.57 (PARITY), T3K v2 13.7 < v1 15.69 (BEATS). N150 TTTv1 ci-32
# OOMs on a single device (no TTFT anchor) → ttft gate = the TTTv2 8-fold measured value (TTTv2 runs
# batch-32 where TTTv1 cannot).
# DECODE parity is assessed SAME-BOX: TTTv2 odt >= TTTv1 ci-32 on every SKU (N300 35.8>33.19, T3K
# 45.2>34.52). The committed tok/s/u gates are prior-healthy floors; the reserved T3K box is #893
# NUMA-degraded on multi-chip D->H this session, depressing N300/T3K decode below the healthy gate (the
# same-box TTTv1 control is depressed MORE) — a box reason, not a code regression. The HEALTHY N150 SKU
# passes every committed gate, validating them; gates NOT lowered. T3K host gated to TTTv2 (no TTTv1 host).
EXPECTED_METRICS_BATCH32_CI: dict = {
    "host": {
        "performance": {
            "N150": {"tok_s_u": 25.1, "ttft_ms": 36},
            "N300": {"tok_s_u": 37.6, "ttft_ms": 30},
            "T3K": {"tok_s_u": 43.1, "ttft_ms": 18},
        },
        "accuracy": {
            "N150": {"tok_s_u": 22.3, "ttft_ms": 44},
            "N300": {"tok_s_u": 32.8, "ttft_ms": 38},
            "T3K": {"tok_s_u": 38.0, "ttft_ms": 24},
        },
    },
    "on_device_topk": {
        "performance": {
            "N150": {"tok_s_u": 25.2, "ttft_ms": 36},
            "N300": {"tok_s_u": 39.9, "ttft_ms": 30},
            "T3K": {"tok_s_u": 57.66, "ttft_ms": 18},
        },
        "accuracy": {
            "N150": {"tok_s_u": 22.4, "ttft_ms": 44},
            "N300": {"tok_s_u": 34.6, "ttft_ms": 38},
            "T3K": {"tok_s_u": 54.59, "ttft_ms": 24},
        },
    },
}

# Perf workload: natural-length prefill (these sample prompts are ~90-125 tokens -> 128 bucket,
# matching TTTv1), 200 decode steps. Accuracy uses the teacher-forcing refpt.
_PERF_NUM_DECODE_TOKENS = 200

PERF_TOLERANCE = 0.05

# batch-32-ci per-SKU max_seq_len (TTTv1 ci-32 parity is seq2048). DRAM trap: raising max_seq_len
# doubles the batch-32 KV cache. 7B weights are large — a single unsharded N150 cannot hold 7B
# weights + a seq2048×32-user KV cache, so N150 is clamped to 1024 (same cap TTTv1 uses for its
# batch-32 config). N300 (weights sharded 2-way) and T3K hold seq2048.
_BATCH32_CI_MAX_SEQ_LEN: dict[str, int] = {
    "N150": 1024,
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
            f"Unsupported MESH_DEVICE={env!r} for Mistral-7B; use N150, N300 or T3K.",
            allow_module_level=True,
        )
    param = {
        "mesh_shape": shape,
        "trace_region_size": 100_000_000 if env == "T3K" else 50_000_000,
        "num_command_queues": 1,
    }
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


def _skip_unless_heads_divide_mesh(mesh_device: ttnn.MeshDevice, hf_model_id: str) -> None:
    """Attention1D TP requires n_heads and n_kv_heads divisible by device count."""
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
    """Map mesh device count to a metrics bucket (matches PERF.md SKU keys)."""
    n = mesh_device.get_num_devices()
    if n == 1:
        return "N150"
    if n == 2:
        return "N300"
    if n == 8:
        return "T3K"
    return f"{n}dev"


def lazy_weight_cache_dir_for_demo(mesh_device: ttnn.MeshDevice, hf_model_id: str) -> Path:
    """Disk root for ``Mistral7B`` ``LazyWeight`` caches in this e2e demo."""
    device_name = get_device_name(mesh_device)
    hf = hf_model_id.strip("/")
    tt_cache = os.getenv("TT_CACHE_PATH")
    if tt_cache:
        root = Path(tt_cache) / device_name
    else:
        root = Path("model_cache") / hf / device_name
    root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Mistral-7B demo LazyWeight cache directory: {root.resolve()}")
    return root


def load_reference_data(hf_model_id: str):
    """Load reference tensors and optional metadata from ``.refpt``.

    Supports both the metadata-rich format (``prompt_len`` + ``metadata`` keys) and
    the book half-split format (the committed reference).
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
    max_seq_len: int | None = None,
) -> Mistral7B:
    """Build ``Mistral7B`` in executor (paged KV) mode.

    Picks one of the two module-level precision recipes (``MISTRAL_ACCURACY`` /
    ``MISTRAL_PERFORMANCE``) — both defined in ``mistral_7b/model.py`` and grounded in TTTv1's
    ``DecodersPrecision`` for Mistral-7B.

    ``max_seq_len`` overrides the DRAM-aware default. Default (``None``): 7B weights + a 32-user KV
    cache cannot co-reside at seq4096 on a single unsharded device, so batch>1 is capped to 1024 on
    ≤2-device SKUs (TTTv1 batch-32 parity); T3K spreads the KV across 8 devices and uses the full
    131072//batch budget; batch-1 fits seq4096 on every SKU. The ``batch-32-ci`` leg passes an
    explicit per-SKU value (see ``_BATCH32_CI_MAX_SEQ_LEN``).
    """
    hf_model = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    _skip_unless_heads_divide_mesh(mesh_device, hf_model)

    precision = MISTRAL_PERFORMANCE if optimizations == "performance" else MISTRAL_ACCURACY

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
        pytest.skip(f"Could not build Mistral model (weights / memory / mesh): {e}")

    model = llm.model
    model.demo_tokenizer = llm.tokenizer
    return model


def create_executor(
    model: Mistral7B,
    *,
    traced: bool,
    device_sampling_enabled: bool,
    trace_mode=None,
) -> Mistral7BExecutor:
    block_size = 32
    max_num_blocks = ((model.config.max_seq_len + block_size - 1) // block_size) * model.config.max_batch_size
    attention_config = model.config.block_configs[0].attention_config
    if trace_mode is None:
        trace_mode = "all" if traced else "none"
    return Mistral7BExecutor(
        model,
        model.model_args,
        Mistral7BExecutorConfig(
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
            execution=prefill_compile_execution or executor.eager_execution,
        )
    if config.trace.prefill_enabled:
        executor.warmup_model_prefill(enable_trace=True, **prefill_kwargs)
    if config.trace.decode_enabled:
        executor.warmup_model_decode(enable_trace=True, **decode_kwargs)


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
#   ci-b1-DP-2  : max_seq_len=1024, max_generated_tokens=200, stop_at_eos=True (only DP case on N300)
#   ci-b1-DP-4  : max_seq_len=4096, max_generated_tokens=2048, stop_at_eos=False
#   ci-b1-DP-8  : max_seq_len=4096, max_generated_tokens=2048, stop_at_eos=False (only DP case on T3K)
#   ci-b1-DP-16 : max_seq_len=1024, max_generated_tokens=200,  stop_at_eos=True
#   ci-b1-DP-32 : max_seq_len=1024, max_generated_tokens=200,  stop_at_eos=True
#
# Hardware feasibility: each DP group is one device (batch_size=1 per group), so
# ``data_parallel == n_devices``. On N300 (2 chips) only DP-2 fits; on T3K only DP-8; the rest cleanly
# ``pytest.skip`` via ``_dp_or_skip``. ``stop_at_eos`` is effectively a no-op in TTTv2's fixed-budget
# ``run_perf_benchmark`` loop; the special-token guard truncates at the first stop token before scanning.
_DP_SIZE_TABLE: dict[int, dict] = {
    2: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
    4: {"max_seq_len": 4096, "max_generated_tokens": 2048, "stop_at_eos": False},
    8: {"max_seq_len": 4096, "max_generated_tokens": 2048, "stop_at_eos": False},
    16: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
    32: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
}


def create_dp_submeshes(mesh_device: ttnn.MeshDevice, data_parallel: int) -> list:
    """Partition the open parent mesh into ``data_parallel`` disjoint row-submeshes.

    Mirrors TTTv1 ``generator.create_submeshes`` minus the Galaxy reshape branch (no Galaxy
    reachable here). For the single-user DP cases ``n // data_parallel == 1``, so each submesh is a
    ``(1,1)`` mesh. Fabric stays owned by the parent — do NOT set fabric per-submesh.
    """
    if data_parallel == 1:
        return [mesh_device]
    n = mesh_device.get_num_devices()
    assert n % data_parallel == 0, f"{n} devices not divisible by data_parallel={data_parallel}"
    return list(mesh_device.create_submeshes(ttnn.MeshShape(1, n // data_parallel)))


def _dp_or_skip(mesh_device: ttnn.MeshDevice, data_parallel: int) -> None:
    """Skip unless the mesh has exactly ``data_parallel`` single-device DP groups."""
    n = mesh_device.get_num_devices()
    if n % data_parallel != 0 or (n // data_parallel) != 1:
        pytest.skip(f"DP-{data_parallel} needs {data_parallel} single-device groups; have {n} devices")


def assert_no_special_tokens(
    generated_token_ids, tokenizer, *, case_name: str = "", is_ci_env: bool | None = None
) -> None:
    """No special (garbage) token mid-stream. Mirrors TTTv1 ``simple_text_demo.py``: warns always,
    hard-fails only under CI.

    TTTv2's ``result.generated_token_ids[user]`` already starts at the first generated token, so
    unlike TTTv1 we do not slice off the prompt — these are output-only. Each user's output is
    truncated at the first stop token (EoS; Mistral has no second stop token) before the special-id
    scan. Shared by the perf path and the DP smoke; CI-gating keeps local runs finishing (warn) while
    still failing CI.
    """
    stop = set()
    if tokenizer.eos_token_id is not None:
        stop.add(tokenizer.eos_token_id)
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
    """Run one user per single-device lane through the model-owned DP runtime."""
    _dp_or_skip(mesh_device, data_parallel)
    mesh_device.quiesce_devices()
    hf_model = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    _skip_unless_heads_divide_mesh(mesh_device, hf_model)
    precision = MISTRAL_PERFORMANCE if optimizations == "performance" else MISTRAL_ACCURACY
    submeshes = create_dp_submeshes(mesh_device, data_parallel)
    prompts = load_input_prompts(data_parallel)
    sampling_mode = os.environ.get("SAMPLING_MODE", "host").lower()
    _on_device_params = {
        "on_device": SamplingParams(temperature=0.0, top_k=1, top_p=0.0),
        "on_device_topk": SamplingParams(temperature=0.0, top_k=32, top_p=0.08),
    }

    models: list = []
    lanes: list = []
    group = None
    try:
        for sm in submeshes:
            llm = from_pretrained(
                sm,
                hf_model=hf_model,
                max_batch_size=1,
                max_seq_len=max_seq_len,
                n_layers=None,
                cache_dir=cache_dir,
                optimizations=precision,
            )
            model = llm.model
            model.demo_tokenizer = llm.tokenizer
            models.append((model, sm))
            lanes.append(
                create_executor(
                    model,
                    traced=True,
                    device_sampling_enabled=sampling_mode in _on_device_params,
                )
            )

        group = LaneGroupExecutor(lanes, mesh_device=mesh_device)
        tokenizer = models[0][0].demo_tokenizer
        kv_cache = group.allocate_kv_cache()
        page_table = make_contiguous_page_table(1, max_seq_len, 32).repeat(data_parallel, 1)
        input_tokens, prompt_lens = tokenize_prompts(prompts, tokenizer)
        sampling_params = (
            _on_device_params[sampling_mode]
            if sampling_mode in _on_device_params and getattr(models[0][0], "supports_on_device_sampling", False)
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
            f"[ci-b1-DP-{data_parallel}] SAMPLING_MODE={sampling_mode} "
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
        )
        assert len(result.generated_token_ids) == data_parallel
        assert all(result.generated_token_ids), f"ci-b1-DP-{data_parallel}: every lane must return output"
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
def test_mistral_7b(test_config, mesh_device, optimizations):
    """Main test entry for TTTv2 Mistral-7B-Instruct-v0.3."""
    device_name = get_device_name(mesh_device)
    expected = EXPECTED_METRICS.get(optimizations, {}).get(device_name, {})
    model = None
    hf_model = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
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

        # Token-accuracy feeds a single reference sequence — max_batch_size=1 avoids DRAM pressure
        # from a full 32-user KV cache. batch-32 / eval-32 run 32 users at seq1024 (short-context
        # workload); the 7B DRAM-aware create_model would also cap ≤2-dev SKUs there, but we pass
        # 1024 explicitly so T3K uses the same short-context seq len (not its 131072//32 default).
        if test_config == "batch-32":
            max_bs, max_seq_len = 32, 1024
            expected = EXPECTED_METRICS_BATCH32.get(_sampling_bucket(), {}).get(optimizations, {}).get(device_name, {})
        elif test_config == "eval-32":
            # eval-32 runs 32 users × 3 rotated repeats, building a FRESH traced executor per repeat
            # (run_eval_repeat_batch32). On a single unsharded device the full 7B weights + a 32-user KV
            # cache already sit at ~99% DRAM (batch-32 fits with only ~7MB free), so the per-repeat
            # executor/trace churn cannot fit — it OOMs (bank_manager). This is a genuine single-device
            # DRAM-capability limit for a 7B, NOT a TTTv2 regression: TTTv1 ci-32 / ci-eval-32 also OOM
            # on N150 (batch-32-class does not fit a single N150 for 7B in either stack), while TTTv2
            # batch-32 / batch-32-ci DO fit here (single executor). Skip on 1-device SKUs; runs on the
            # sharded N300 / T3K (64/64 cross-batch consistency). Hardware-capability guard, not a mask.
            if mesh_device.get_num_devices() == 1:
                pytest.skip(
                    "eval-32 (32 users × 3 rotated fresh-executor repeats) exceeds single-device DRAM "
                    "for a 7B; TTTv1 ci-32/ci-eval-32 OOM on N150 too. Runs on sharded N300/T3K."
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


def _run_token_accuracy(model: Mistral7B, mesh_device, expected):
    """Teacher-forcing token accuracy vs ``.refpt``."""
    hf_model = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    reference_tokens, top5_tokens, prompt_len, metadata = load_reference_data(hf_model)
    tokenizer = model.demo_tokenizer

    if reference_tokens.dim() > 1:
        reference_tokens = reference_tokens.squeeze()

    has_prompt_len_metadata = prompt_len is not None
    if has_prompt_len_metadata:
        prompt_len = int(prompt_len)
        logger.info(f"Using metadata prompt_len={prompt_len}")
    else:
        prompt_len = len(reference_tokens) // 2
        logger.info(f"Reference missing prompt_len metadata; using book half-split={prompt_len}.")

    if metadata:
        logger.info(
            f"Reference metadata: hf_model_id={metadata.get('hf_model_id')}, "
            f"revision={metadata.get('revision')}, created_at={metadata.get('created_at')}"
        )

    prompt_tokens = reference_tokens[:prompt_len].unsqueeze(0)

    executor = create_executor(model, traced=False, device_sampling_enabled=False)
    max_batch_size = model.config.max_batch_size
    prompt_tokens = prompt_tokens.repeat(max_batch_size, 1)
    block_size = 32
    max_seq_len = model.config.max_seq_len
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
        # run_teacher_forcing times prefill + per-step (teacher-forced) decode and, given the profiler,
        # brackets the "inference_prefill"/"inference_decode" steps itself, so the result carries prefill/
        # decode throughput alongside accuracy for CI benchmark-data emission.
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

    # CI-dashboard telemetry: emit a ``demo_accuracy`` partial mirroring TTTv1 simple_text_demo.py — the
    # FULL perf set (prefill_t/s, prefill_time_to_token, decode_t/s, decode_t/s/u) PLUS top1/top5, from
    # this timed teacher-forcing run. create_benchmark_data / save_partial_run_json are no-ops unless
    # CI == "true" (they guard internally); the is_ci_env guard keeps the import/attr access off the
    # local path too. Emitted BEFORE the asserts so telemetry survives a gate failure.
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

    # Accuracy gate — threshold SOURCE is flag-controlled (currently is_ci_env):
    #   CI (use_centralized_targets=True): mirror TTTv1 — centralized target − an ABSOLUTE 0.5 pp
    #       (get_accuracy_thresholds, simple_text_demo.py). Missing entry is a hard error (never silently
    #       un-gate in CI). NO PERF_TOLERANCE on accuracy.
    #   local (False): the demo's local EXPECTED_METRICS top1/top5 DIRECTLY (TTTv1 applies no ratio either).
    # Measured accuracy is rounded up with math.ceil first, matching TTTv1 (simple_text_demo.py:1657-1658).
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

    meas_top1 = math.ceil(top1)
    meas_top5 = math.ceil(top5)
    assert meas_top1 >= min_top1, f"Top-1 accuracy {top1:.1f}% (ceil {meas_top1}) below threshold {min_top1:.1f}%"
    assert meas_top5 >= min_top5, f"Top-5 accuracy {top5:.1f}% (ceil {meas_top5}) below threshold {min_top5:.1f}%"


def _run_perf_benchmark(
    model: Mistral7B,
    mesh_device,
    expected,
    batch_size: int,
    case_name: str,
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
    hf_model = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer = model.demo_tokenizer

    # The provider resolves DISABLE_BATCHED_PREFILL and DISABLE_MINIMAL_MATMUL while constructing
    # the immutable runtime/model configs, so both established A/B knobs remain build-time policy.

    # On-device sampling toggle (SAMPLING_MODE):
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
    # path (inert on host/force-argmax; gated to the top-k path by _decode_loop_active). Mirrors
    # llama32_1b's demo — advances position/rope on device and lets run_perf_benchmark pipeline the
    # per-step token readback (host one step behind the device), removing the per-step host overhead.
    # fast_prefill_last_token: slice the single consumed last-token row on device before readback so the
    # batch-1 host concat/readback moves one row instead of the full [1,1,32,vocab] tile — closes most of
    # the residual batch-1 PREFILL TTFT gap vs TTTv1 (which reads back only tokens). Inert for batch>1.
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

        # BenchmarkProfiler brackets the timed prefill/decode regions inside run_perf_benchmark
        # (default-None => byte-inert for every other caller) so we can emit CI perf telemetry.
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

        # CI-dashboard telemetry: emit a ``demo_perf`` partial mirroring TTTv1 simple_text_demo.py. Saved
        # BEFORE the special-token guard and perf gate so telemetry survives a downstream assert. No-op
        # unless CI == "true" (BenchmarkData guards on it).
        if is_ci_env:
            hf_model = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
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


def _run_eval_repeat_batch32(model: Mistral7B, mesh_device):
    """32-user cross-batch determinism (self-consistency under prompt rotation).

    Runs the batch-32 prefill+decode loop ``_EVAL_REPEAT_BATCHES`` times, rotating the prompt->slot
    assignment by one each repeat (fresh traced executor + KV cache per repeat), then asserts that
    undoing the rotation lines up per-user outputs. No external golden. Honors the same
    ``SAMPLING_MODE`` knob as ``_run_perf_benchmark`` (default host argmax — deterministic and
    mesh-agnostic, the recommended default for the determinism assert).
    """
    hf_model = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer = model.demo_tokenizer

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
