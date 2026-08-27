# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 Llama-3.3-70B-Instruct demo — accuracy and performance measurement.

Uses the model-owned ``Llama33_70BExecutor`` directly (no vLLM adapter).

**Mesh note:** Llama-3.3-70B-Instruct has 64 attention heads and 8 KV heads; both divide
T3K (8). The port raises on any other mesh. T3K is the only SKU PERF.md publishes per-user
TTFT for, so it is the primary (and only) bringup SKU here.

**Workload:** performance tests prefill each prompt at its natural length (TTTv1
``preprocess_inputs_prefill`` semantics; these sample prompts are ~90-125 tokens -> 128
prefill bucket, matching TTTv1's traced-prefill seq len for Llama-3.3-70B on T3K) + 200
decode iterations. Accuracy / teacher-forcing uses 511 continuation tokens.

Usage::

    # Token accuracy test
    MESH_DEVICE=T3K HF_MODEL=meta-llama/Llama-3.3-70B-Instruct \\
      pytest models/common/tests/demos/llama33_70b/demo.py -k "token-accuracy" -v

    # Batch-1 latency test
    MESH_DEVICE=T3K HF_MODEL=meta-llama/Llama-3.3-70B-Instruct \\
      pytest models/common/tests/demos/llama33_70b/demo.py -k "batch-1" -v

    # Batch-32 throughput test
    MESH_DEVICE=T3K HF_MODEL=meta-llama/Llama-3.3-70B-Instruct \\
      pytest models/common/tests/demos/llama33_70b/demo.py -k "batch-32" -v

LazyWeight tensor cache: ``TT_CACHE_PATH/<device_name>`` when set, otherwise
``model_cache/<HF_MODEL>/<device_name>`` under the current working directory.

Reference artifact (``.refpt``): the accuracy test gates on the committed book
reference at ``models/tt_transformers/tests/reference_outputs/<model>.refpt``
(ground-truth real-text targets, single teacher-forced pass), which is the
PERF.md-comparable methodology.
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
from models.common.models.llama33_70b.executor import Llama33_70BExecutor, Llama33_70BExecutorConfig
from models.common.models.llama33_70b.hf_adaptor import encode_prompt, from_pretrained
from models.common.models.llama33_70b.model import (
    LLAMA33_70B_ACCURACY,
    LLAMA33_70B_PERFORMANCE,
    Llama33_70BTransformer1D,
)
from models.common.sampling.sampling_params import SamplingParams
from models.common.tests.demos.cleanup_utils import cleanup_model_case
from models.common.tests.demos.run_helpers import (
    assert_no_special_tokens,
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
# Expected metrics — perf gates set from same-box TTTv1-vs-TTTv2 measurement on this base
# (SAMPLING_MODE-aware, profile-aware). No PERF.md throughput value is used (PERF.md is stale).
#
# Rule: each ``tok_s_u`` / ``ttft_ms`` target is the BETTER of freshly-measured
# same-box TTTv1 vs TTTv2 for that sampling mode. TTTv1 has only an on-device sampling path, so:
#     on_device_topk : max(TTTv1_on_device, TTTv2_on_device_topk)   [tok_s_u]; min(...) [ttft_ms]
#     host           : TTTv2_host                                   (TTTv1 has no host-sampling path)
# Decode throughput is prefill-independent, so batched prefill (default-ON here) does NOT change
# ``tok_s_u``. ``ttft_ms`` targets are conservative upper bounds (batched prefill only LOWERS TTFT).
# Llama-3.3-70B is T3K-only (64 attn / 8 KV heads ⇒ 8 devices); there are no N150/N300 rows.
# =============================================================================

# top1/top5 are teacher-forcing accuracy floors (sampling-independent); this dict gates only
# token-accuracy. Perf metrics live in the sampling-mode-aware dicts below.
EXPECTED_METRICS = {
    "performance": {
        "T3K": {"top1": 96, "top5": 100},
    },
    "accuracy": {
        "T3K": {"top1": 96, "top5": 100},
    },
}

# batch-1 throughput, sampling-mode- AND profile-aware. host = TTTv2-host; on_device_topk =
# max(TTTv1, TTTv2-on-device). Populated from same-box measurement this session.
# Cells not yet measured stay {} (the case still RUNS, printing tok_s_u, but is not gated) — never
# a silent PERF.md value.
EXPECTED_METRICS_BATCH1: dict = {
    "host": {
        "performance": {"T3K": {"tok_s_u": 10.5, "ttft_ms": 195}},  # TTTv2-host 2026-07-24 (10.52)
        "accuracy": {"T3K": {"tok_s_u": 9.4, "ttft_ms": 220}},  # TTTv2-host 2026-07-24 (9.41)
    },
    "on_device_topk": {
        # decode = best-of(TTTv1, TTTv2 odt); TTTv1 uses on-device on T3K. ttft = conservative upper
        # bound above the measured (single-user prefill TTFT is noisy; batch-1 has no batched prefill).
        "performance": {"T3K": {"tok_s_u": 17.40, "ttft_ms": 195}},  # best-of max(TTTv1 17.40, TTTv2 17.26) 2026-07-24
        "accuracy": {
            "T3K": {"tok_s_u": 14.86, "ttft_ms": 220}
        },  # best-of max(TTTv1 14.86, TTTv2 14.74); TTFT faster than TTTv1 (206<208)
    },
}

# Short-context batch-32 throughput (seq1024 / 200 decode), sampling-mode- AND profile-aware.
EXPECTED_METRICS_BATCH32: dict = {
    "host": {
        "performance": {"T3K": {"tok_s_u": 10.2, "ttft_ms": 90}},
        "accuracy": {"T3K": {"tok_s_u": 9.3, "ttft_ms": 100}},
    },
    "on_device_topk": {
        # decode: TTTv2 BEATS TTTv1 at batch-32 (better-of picks TTTv2). ttft = conservative upper
        # bound above measured TTTv2 (batched-prefill ON ~79/91 ms; +21% vs TTTv1 is the known
        # shared-engine batched-prefill CCL residual, documented as a cross-model item).
        "performance": {"T3K": {"tok_s_u": 16.7, "ttft_ms": 90}},  # max(TTTv1 16.06, TTTv2 16.7)
        "accuracy": {"T3K": {"tok_s_u": 14.4, "ttft_ms": 100}},  # max(TTTv1 13.85, TTTv2 14.4)
    },
}

# CI-faithful batch-32 targets (the ``batch-32-ci`` leg), measured at the batch-32-ci workload
# (seq clamp below + 1024-token decode budget; TTTv1 ci-32 workload). Separate from the lighter
# batch-32 leg: the longer decode budget grows the KV read window so steady-state per-token decode
# is a bit slower. Cells not measured fall back to EXPECTED_METRICS_BATCH32 (stay gated, never
# silently un-gated).
EXPECTED_METRICS_BATCH32_CI: dict = {
    "host": {
        "performance": {"T3K": {"tok_s_u": 9.6, "ttft_ms": 90}},  # TTTv2-host 2026-07-24 (9.68)
        "accuracy": {"T3K": {"tok_s_u": 8.9, "ttft_ms": 100}},  # TTTv2-host 2026-07-24 (8.87)
    },
    "on_device_topk": {
        # decode = best-of vs TTTv1 ci-32 (the matched CI leg). ttft = conservative upper bound
        # above measured TTTv2 (batched-prefill residual, as in batch-32).
        "performance": {
            "T3K": {"tok_s_u": 16.60, "ttft_ms": 90}
        },  # best-of max(TTTv2 16.56, TTTv1 ci-32 device-mean 16.60) 2026-07-24
        "accuracy": {
            "T3K": {"tok_s_u": 14.2, "ttft_ms": 100}
        },  # TTTv2 14.23 (TTTv1 ci-32-acc CI-perf-only -> own-gated); >= TTTv1 b32-acc 13.85
    },
}

# Perf workload: natural-length prefill (these sample prompts are ~90-125 tokens -> 128 bucket,
# matching TTTv1's traced-prefill seq len for Llama-3.3-70B on T3K), 200 decode steps.
# Accuracy uses the 511-token teacher-forcing refpt.
_PERF_NUM_DECODE_TOKENS = int(os.environ.get("PERF_NUM_DECODE_TOKENS", "200"))

PERF_TOLERANCE = 0.05

# batch-32-ci per-SKU max_seq_len (TTTv1 ci-32 parity is seq2048). DRAM trap: raising max_seq_len
# doubles the batch-32 KV cache, and 70B is the extreme case — BFP8 weights are ~9 GB/device on T3K,
# leaving only ~3 GB for KV + activations. batch-32 already runs at seq1024 (see the test body);
# seq2048 at batch-32 would roughly double that KV footprint and OOM the bank_manager. So batch-32-ci
# is CLAMPED to 1024 on T3K (still covers the 128-bucket prefill + a long ~880-token clamped decode
# budget). Mirrors the 3B ``_BATCH32_CI_MAX_SEQ_LEN`` clamp; 70B needs the lower value where 3B used 2048.
_BATCH32_CI_MAX_SEQ_LEN: dict[str, int] = {
    "T3K": 1024,
}


def _sampling_bucket() -> str:
    """Map SAMPLING_MODE to a perf-gate bucket. Non-topk on-device modes (e.g. force-argmax)
    fall into ``on_device_topk`` so they stay gated, never silently un-gated."""
    return "host" if os.environ.get("SAMPLING_MODE", "host").lower() == "host" else "on_device_topk"


_MESH_DEVICE_TO_SHAPE: dict[str, tuple[int, int]] = {
    "T3K": (1, 8),
}


def _ttnn_mesh_device_param_from_env() -> dict:
    env = os.environ.get("MESH_DEVICE", "").strip()
    if not env:
        pytest.skip(
            "MESH_DEVICE must be set to T3K. See module docstring.",
            allow_module_level=True,
        )
    shape = _MESH_DEVICE_TO_SHAPE.get(env)
    if shape is None:
        pytest.skip(
            f"Unsupported MESH_DEVICE={env!r} for Llama-3.3-70B; only T3K is supported "
            f"(64 attn heads / 8 KV heads ⇒ 8 devices).",
            allow_module_level=True,
        )
    param = {
        "mesh_shape": shape,
        "trace_region_size": 50_000_000,
        "num_command_queues": 1,
    }
    # TTTv2 multi-device executor dispatch (and the on-device sampling all-gather) stalls without
    # an explicit fabric; the root conftest does not auto-enable it. The Llama33 model resolves T3K
    # collectives to Ring topology, so the fabric config must match that topology.
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
    return ttnn_mesh_device


def _skip_unless_heads_divide_mesh(mesh_device: ttnn.MeshDevice) -> None:
    n_dev = mesh_device.get_num_devices()
    if 64 % n_dev == 0 and 8 % n_dev == 0:
        return
    pytest.skip(
        f"Incompatible mesh for Llama-3.3-70B-Instruct: {n_dev} devices, "
        "num_attention_heads=64, num_key_value_heads=8."
    )


def get_device_name(mesh_device: ttnn.MeshDevice) -> str:
    n = mesh_device.get_num_devices()
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
    logger.info(f"Llama-3.3-70B demo LazyWeight cache directory: {root.resolve()}")
    return root


def load_reference_data(hf_model_id: str):
    """Load reference tensors and optional metadata from ``.refpt``.

    Supports both the metadata-rich format (``prompt_len`` + ``metadata`` keys)
    and the book half-split format (``reference_tokens`` + ``top5_tokens`` only).
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
) -> Llama33_70BTransformer1D:
    """Build the provider-neutral graph through the Llama 3.3 HF adaptor."""
    hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
    _skip_unless_heads_divide_mesh(mesh_device)

    precision = LLAMA33_70B_PERFORMANCE if optimizations == "performance" else LLAMA33_70B_ACCURACY
    llm = from_pretrained(
        mesh_device,
        hf_model=hf_model,
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
    model: Llama33_70BTransformer1D,
    *,
    traced: bool,
    device_sampling_enabled: bool,
    trace_mode: str | None = None,
) -> Llama33_70BExecutor:
    block_size = 32
    max_num_blocks = math.ceil(model.config.max_seq_len / block_size) * model.config.max_batch_size
    attention_config = model.config.block_configs[0].attention_config
    if trace_mode is None:
        trace_mode = "all" if traced else "none"
    return Llama33_70BExecutor(
        model,
        model.model_args,
        Llama33_70BExecutorConfig(
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
            execution=(
                prefill_compile_execution if prefill_compile_execution is not None else executor.eager_execution
            ),
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
# instruct prompts, paged attention, trace on. The ONLY correctness check is the special-token
# garbage guard plus "runs to completion without hang/exception". This is a mesh / KV-cache /
# page-table scaling smoke test, NOT an accuracy or perf gate.
#
# Hardware feasibility on Llama-3.3-70B (T3K-only): one replica requires the full TP8 mesh,
# so an eight-device host has capacity for DP1 only. Every retained DP factor is rejected by
# ``_dp_or_skip`` before submesh creation or model construction. This also avoids the W0 DP-8
# cleanup bug, where an intended build-time skip was masked by a failing parent-mesh quiesce.
_DP_SIZE_TABLE: dict[int, dict] = {
    2: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
    4: {"max_seq_len": 4096, "max_generated_tokens": 2048, "stop_at_eos": False},
    8: {"max_seq_len": 4096, "max_generated_tokens": 2048, "stop_at_eos": False},
    16: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
    32: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
}


def _dp_or_skip(mesh_device: ttnn.MeshDevice, data_parallel: int) -> None:
    """Preserve DP case IDs while rejecting every topology before model construction.

    Llama 3.3 70B requires TP8, so an eight-device T3K has capacity for exactly one
    model replica. No collected DP factor can retain TP8 lanes.
    """
    n = mesh_device.get_num_devices()
    if n % data_parallel:
        pytest.skip(f"DP-{data_parallel} cannot partition {n} devices into equal lanes")
    pytest.skip(
        f"DP-{data_parallel} on {n} devices creates TP{n // data_parallel} lanes; "
        "Llama-3.3-70B requires one TP8 lane"
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
    """Apply the capacity guard for the retained TTTv1-parity DP node IDs."""
    del optimizations, cache_dir, max_seq_len, max_gen_tokens, stop_at_eos
    _dp_or_skip(mesh_device, data_parallel)


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
def test_llama33_70b(test_config, mesh_device, optimizations):
    """Main test entry for TTTv2 Llama-3.3-70B-Instruct."""
    device_name = get_device_name(mesh_device)
    expected = EXPECTED_METRICS.get(optimizations, {}).get(device_name, {})
    model = None
    hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
    cache_dir = lazy_weight_cache_dir_for_demo(mesh_device, hf_model)

    try:
        # ci-b1-DP-*: single-user data-parallel smoke. Builds N models itself (one per submesh),
        # so it does NOT go through the shared create_model path below. On 70B (T3K-only) every DP
        # leg self-skips as a hardware-capability guard (no 1-device group can hold 70B) — see
        # _run_dp_smoke.
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

        # Token-accuracy + batch-1 feed a single sequence — max_batch_size=1 avoids DRAM
        # pressure from a full 32-user KV cache allocation (70B BFP8 weights are ~9 GB/device
        # on T3K, leaving only ~3 GB for KV + activations).
        # batch-32 and eval-32 both run 32 users at max_seq_len=1024 to avoid DRAM OOM: 80 layers
        # × 1 KV head/dev × 128 head_dim × 32 batch at seq 4096 (≈2.7 GB/device) would overflow
        # alongside weights; 1024 (≈0.67 GB KV) still covers the natural-length prefill (~128 bucket)
        # + 200 decode workload.
        if test_config in ("batch-32", "eval-32"):
            max_bs, max_seq_len = 32, 1024
            expected = EXPECTED_METRICS_BATCH32.get(_sampling_bucket(), {}).get(optimizations, {}).get(device_name, {})
        elif test_config == "batch-32-ci":
            # CI-faithful batch-32 leg (TTTv1 ci-32 parity): a longer decode budget (1024 tokens,
            # clamped in _run_perf_benchmark) at the per-SKU seq len. 70B is DRAM-bound so the seq is
            # clamped to 1024 (see _BATCH32_CI_MAX_SEQ_LEN) rather than TTTv1's 2048. Gate keyed by
            # SAMPLING_MODE + profile; cells not measured fall back to the batch-32 constant (stay gated).
            max_bs = 32
            max_seq_len = _BATCH32_CI_MAX_SEQ_LEN.get(device_name, 1024)
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
            _run_perf_benchmark(model, mesh_device, expected, batch_size=32, case_name=f"{optimizations}/batch-32")
        elif test_config == "batch-32-ci":
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


def _run_token_accuracy(model: Llama33_70BTransformer1D, mesh_device, expected):
    """Teacher-forcing token accuracy vs ``.refpt``."""
    hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
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
        logger.info(f"Reference has no prompt_len metadata; using book half-split={prompt_len}.")

    if metadata:
        logger.info(
            f"Reference metadata: hf_model_id={metadata.get('hf_model_id')}, "
            f"revision={metadata.get('revision')}, created_at={metadata.get('created_at')}"
        )

    prompt_tokens = reference_tokens[:prompt_len].unsqueeze(0)

    executor = create_executor(model, traced=False, device_sampling_enabled=False)
    try:
        max_batch_size = model.config.max_batch_size
        prompt_tokens = prompt_tokens.repeat(max_batch_size, 1)
        kv_cache = executor.allocate_kv_cache()
        page_table = make_contiguous_page_table(max_batch_size, model.config.max_seq_len, 32)
        target_top5 = select_teacher_forcing_top5_slice(
            top5_tokens,
            reference_tokens,
            prompt_len,
            metadata_aligned=has_prompt_len_metadata,
        )
        is_ci_env = os.environ.get("CI") == "true"
        profiler = BenchmarkProfiler()
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

    # Accuracy gate — threshold SOURCE is flag-controlled (``is_ci_env``):
    #   use_centralized_targets = True  → mirror TTTv1: centralized targets via
    #       resolve_accuracy_targets minus an ABSOLUTE 0.5 pp (get_accuracy_thresholds,
    #       simple_text_demo.py). Missing entry is a hard error (never silently un-gate in CI).
    #   use_centralized_targets = False → the demo's local EXPECTED_METRICS values DIRECTLY
    #       (no ratio tolerance — TTTv1 applies none to accuracy).
    # Measured accuracy is rounded up with math.ceil first, matching TTTv1
    # (simple_text_demo.py:1657-1658, ``math.ceil(acc[...] * 100)``).
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
    model: Llama33_70BTransformer1D,
    mesh_device,
    expected,
    batch_size: int,
    case_name: str,
    max_prefill_len: int | None = None,
    num_decode_tokens: int | None = None,
):
    """Timed prefill + decode with the traced model-owned executor.

    Prefill uses each prompt's natural token length (TTTv1 ``preprocess_inputs_prefill``
    semantics — the executor buckets to ``get_padded_prefill_len``); decode runs for
    ``num_decode_tokens`` steps (default ``_PERF_NUM_DECODE_TOKENS``).
    ``max_prefill_len`` is an optional clip cap for over-long prompts, never a pad-up target.

    The decode budget is clamped to what the paged KV cache can hold:
    ``effective = min(requested, max_seq_len - prompt_bucket - margin)`` so the high-water
    decode position never overruns the page table (the ``batch-32-ci`` leg requests 1024).
    """
    hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
    tokenizer = model.demo_tokenizer

    # Batched-prefill A/B knob (parity caveat #12): set DISABLE_BATCHED_PREFILL=1 to force the
    # sequential per-user prefill loop (the pre-feature baseline) for before/after TTFT comparison.
    # Companion knob (PLAN_01): DISABLE_MINIMAL_MATMUL=1 forces QKV/W2 prefill back to ttnn.linear
    # (read at model build time, so it must be in the env before from_pretrained — it already is).
    if os.environ.get("DISABLE_BATCHED_PREFILL") and model.model_args is not None:
        model.model_args.disable_batched_prefill = True

    # On-device sampling toggle for SKU evidence-gathering:
    #   host            -> sampling_params=None (host-argmax, the default shipped path)
    #   on_device       -> greedy temp=0,k=1,p=0 => trace-captured top-k op path with k=1.
    #                      Sampling1D is built allow_force_argmax=False, so even greedy routes
    #                      through ttnn.topk (k=1 top-k == argmax-via-topk), NOT the force-argmax
    #                      full-vocab all-gather.
    #   on_device_topk  -> temp=0,k=32,p=0.08    => trace-captured top-k op path with k=32
    #                      (gathers only the [*,32] tuples). On T3K (8 dev) the vocab
    #                      shards 8-ways so on-device top-k is the faster path vs host readback.
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
    # path (inert on host/force-argmax; gated to the top-k path by _decode_loop_active). This is the
    # #49284 shared decode loop — the primary T3K decode-parity lever for this T3K-only 70B.
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
        prefill_sampling_params = None
        _warmup_demo_executor(
            traced_executor,
            kv_cache=kv_cache,
            page_table=page_table,
            prefill_compile_case=(input_tokens, prompt_lens),
            prefill_sampling_params=prefill_sampling_params,
            prefill_compile_execution=traced_executor.traced_prefill_execution,
        )

        # Decode-token budget, clamped to the KV-cache headroom. Prompts bucket to ~128 and we keep
        # a 16-token margin, so the high-water decode position stays inside max_seq_len.
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
            pipeline_readback=os.environ.get("PIPELINE_READBACK", "1").lower() not in ("0", "false", "no"),
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


# =============================================================================
# ci-eval-32 determinism case: 3 rotated repeats of the batch-32 workload.
# =============================================================================
_EVAL_REPEAT_BATCHES = 3
_EVAL_NUM_DECODE_TOKENS = _PERF_NUM_DECODE_TOKENS


def _run_eval_repeat_batch32(model: Llama33_70BTransformer1D, mesh_device):
    """32-user cross-batch determinism (self-consistency under prompt rotation).

    Runs the batch-32 prefill+decode loop ``_EVAL_REPEAT_BATCHES`` times, rotating the
    prompt->slot assignment by one each repeat (fresh traced executor + KV cache per repeat),
    then asserts that undoing the rotation lines up per-user outputs. No external golden.
    Honors the same ``SAMPLING_MODE`` knob as ``_run_perf_benchmark`` (default host argmax —
    deterministic and mesh-agnostic, the recommended default for the determinism assert).
    """
    hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
    tokenizer = model.demo_tokenizer

    # Batched-prefill A/B knob (parity caveat #12): DISABLE_BATCHED_PREFILL=1 forces the pure
    # per-bucket sequential prefill (the Phase-1 path) so eval-32 can be validated both ON and OFF.
    if os.environ.get("DISABLE_BATCHED_PREFILL") and model.model_args is not None:
        model.model_args.disable_batched_prefill = True

    block_size = 32
    max_seq_len = model.config.max_seq_len
    max_batch_size = model.config.max_batch_size
    page_table = make_contiguous_page_table(max_batch_size, max_seq_len, block_size)

    # Fresh traced executor + zeroed KV cache per repeat (driver owns the lifecycle), so the
    # rotated batches are fully independent — see run_eval_repeat_batch32 for why reuse corrupts
    # the 3rd repeat on hardware.
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
    # Prompt rotation preserves this heterogeneous signature multiset. Register it while
    # prefill remains eager under decode-only tracing and before the program set closes.
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
