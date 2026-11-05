# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 Phi-3.5-mini (microsoft/Phi-3.5-mini-instruct) demo — accuracy and performance measurement on one P150.

Uses the model-owned ``Phi35MiniExecutor`` directly (no vLLM adapter). Phi-3.5-mini (3.8B) fits one Blackhole
P150, so this demo runs single-device lanes only (``MESH_DEVICE=P150``).

Cases:
    token-accuracy   - teacher-forcing top-1/top-5 vs the book ``.refpt``
    batch-1          - single-user latency (natural-length ~128-token prompts, 200 decode steps)
    batch-32         - short-context throughput (200 decode steps)
    batch-32-ci      - CI-shaped batch-32 (seq2048, 1024 decode steps)
    eval-32          - 32-user cross-batch determinism

Usage::

    MESH_DEVICE=P150 HF_MODEL=microsoft/Phi-3.5-mini-instruct \\
      pytest models/common/tests/demos/phi35_mini/demo.py -k "performance and batch-1" -v

    SAMPLING_MODE=on_device_topk MESH_DEVICE=P150 HF_MODEL=microsoft/Phi-3.5-mini-instruct \\
      pytest models/common/tests/demos/phi35_mini/demo.py -k "performance and batch-32" -v

LazyWeight tensor cache: ``TT_CACHE_PATH/<device_name>`` when ``TT_CACHE_PATH`` is set,
otherwise ``model_cache/<HF_MODEL>/<device_name>`` under the current working directory.

Reference artifact (``.refpt``): the token-accuracy test gates on the committed book reference
``models/tt_transformers/tests/reference_outputs/Phi-3.5-mini-instruct.refpt``.
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
from models.common.models.phi35_mini.executor import Phi35MiniExecutor, Phi35MiniExecutorConfig
from models.common.models.phi35_mini.hf_adaptor import DEFAULT_HF_REVISION, encode_prompt, from_pretrained
from models.common.models.phi35_mini.model import PHI35_MINI_ACCURACY, PHI35_MINI_PERFORMANCE, Phi35MiniTransformer
from models.common.sampling.sampling_params import SamplingParams
from models.common.tests.demos.cleanup_utils import cleanup_model_case
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
# Expected metrics (P150).
# Token-accuracy floors follow models/model_targets.yaml (phi-3.5-mini, bh_p150: top1 89 / top5 99 in
# performance mode). Perf cells are empty until measured on this port: an empty cell is not gated.
# =============================================================================

EXPECTED_METRICS: dict = {
    "performance": {"P150": {"top1": 89, "top5": 99}},
    "accuracy": {"P150": {"top1": 89, "top5": 99}},
}

EXPECTED_METRICS_BATCH1: dict = {"host": {}, "on_device_topk": {}}
EXPECTED_METRICS_BATCH32: dict = {"host": {}, "on_device_topk": {}}
EXPECTED_METRICS_BATCH32_CI: dict = {"host": {}, "on_device_topk": {}}

# Perf workload: natural-length prefill (sample prompts ~90-125 tokens -> 128 bucket, matching TTTv1),
# 200 decode steps. Accuracy uses the teacher-forcing refpt. PERF_NUM_DECODE_TOKENS overrides the decode
# budget (mirrors the llama32_3b sibling) — used to shorten the window for tt-perf-report/Tracy profiling.
_PERF_NUM_DECODE_TOKENS = int(os.environ.get("PERF_NUM_DECODE_TOKENS", "200"))

PERF_TOLERANCE = 0.05

# 32-user max_seq_len: the BF16 KV cache of the accuracy profile is twice the BFP8 one, so accuracy runs a
# shorter context to stay inside the P150's 32 GB of DRAM.
_BATCH32_MAX_SEQ_LEN: dict[str, int] = {"performance": 2048, "accuracy": 1024}
_BATCH32_CI_MAX_SEQ_LEN: dict[str, int] = {"performance": 2048, "accuracy": 1024}

# eval-32 max_seq_len (both profiles). The ci-eval-32 numeric prompts bucket to a 1024-token prefill, so
# the page table needs >=1024 (32 blocks/user); 1024 bounds the 3-fresh-executor eval churn
# for both profiles. Decode high-water (~201 prompt + 200 gen) < 1024.
_EVAL_MAX_SEQ_LEN = 1024


def _sampling_bucket() -> str:
    """Map SAMPLING_MODE to a perf-gate bucket. Non-topk on-device modes (e.g. force-argmax)
    fall into ``on_device_topk`` so they stay gated, never silently un-gated."""
    return "host" if os.environ.get("SAMPLING_MODE", "host").lower() == "host" else "on_device_topk"


_MESH_DEVICE_TO_SHAPE: dict[str, tuple[int, int]] = {
    "P150": (1, 1),
}


def _ttnn_mesh_device_param_from_env() -> dict:
    env = os.environ.get("MESH_DEVICE", "").strip()
    if not env:
        pytest.skip("MESH_DEVICE must be set (P150). See module docstring.", allow_module_level=True)
    shape = _MESH_DEVICE_TO_SHAPE.get(env)
    if shape is None:
        pytest.skip(
            f"Unsupported MESH_DEVICE={env!r}; use one of {sorted(_MESH_DEVICE_TO_SHAPE)}.", allow_module_level=True
        )
    # The model-owned runtime's representative batch-32 trace set measures 53,698,560 bytes.
    # Keep the region narrowly above that closed-world requirement.
    param = {"mesh_shape": shape, "trace_region_size": 256_000_000, "num_command_queues": 1}
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


def get_device_name(mesh_device: ttnn.MeshDevice) -> str:
    """Map mesh device count to a metrics bucket."""
    n = mesh_device.get_num_devices()
    return "P150" if n == 1 else f"{n}dev"


def lazy_weight_cache_dir_for_demo(mesh_device: ttnn.MeshDevice, hf_model_id: str) -> Path:
    """Disk root for LazyWeight caches. Follows the same convention as other TTTv2 demos."""
    device_name = get_device_name(mesh_device)
    hf = hf_model_id.strip("/")
    tt_cache = os.getenv("TT_CACHE_PATH")
    root = Path(tt_cache) / device_name if tt_cache else Path("model_cache") / hf / device_name
    root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Phi-3.5-mini demo LazyWeight cache directory: {root.resolve()}")
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
    if os.environ.get("S11_PROMPTS"):  # S11 phi override: e.g. the Tier 2 fraud prompts
        with open(os.environ["S11_PROMPTS"]) as f:
            ps = [e["prompt"] for e in json.load(f)]
        return [ps[i % len(ps)] for i in range(batch_size)]
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
) -> Phi35MiniTransformer:
    """Build the provider-neutral Phi-3.5-mini graph through its HF adaptor."""
    hf_model = os.environ.get("HF_MODEL", "microsoft/Phi-3.5-mini-instruct")

    precision = PHI35_MINI_PERFORMANCE if optimizations == "performance" else PHI35_MINI_ACCURACY

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
    model: Phi35MiniTransformer,
    *,
    traced: bool,
    device_sampling_enabled: bool,
    trace_mode=None,
) -> Phi35MiniExecutor:
    block_size = 32
    max_num_blocks = math.ceil(model.config.max_seq_len / block_size) * model.config.max_batch_size
    attention_config = model.config.block_configs[0].attention_config
    if trace_mode is None:
        trace_mode = "all" if traced else "none"
    return Phi35MiniExecutor(
        model,
        model.model_args,
        Phi35MiniExecutorConfig(
            trace=TraceConfig(mode=trace_mode),
            warmup=WarmupConfig(),
            paged_kv_cache=PagedKVCacheConfig(
                block_size=block_size,
                max_num_blocks=max_num_blocks,
                num_blocks=max_num_blocks,
                dtype=attention_config.kv_cache_dtype,
                memory_config=model.kv_cache_memory_config(block_size),
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


def assert_no_special_tokens(
    generated_token_ids, tokenizer, *, case_name: str = "", is_ci_env: bool | None = None
) -> None:
    """Apply the shared strict guard after truncating each output at the first turn boundary.

    Phi-3.5-mini ends an assistant turn with ``<|end|>``; ``<|user|>`` / ``<|assistant|>`` open a new turn, and
    ``<|endoftext|>`` ends the document. The perf benchmark force-decodes a fixed budget, so a finished answer
    continues into a new turn; truncating there mirrors TTTv1, which stops at these tokens. A special token
    emitted before the first boundary is still flagged.
    """
    stop = set()
    for turn_tok in ("<|end|>", "<|user|>", "<|assistant|>", "<|endoftext|>"):
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
    ],
)
@pytest.mark.parametrize("optimizations", ["performance", "accuracy"])
def test_phi35_mini(test_config, mesh_device, optimizations):
    """Main test entry for TTTv2 Phi-3.5-mini."""
    device_name = get_device_name(mesh_device)
    expected = EXPECTED_METRICS.get(optimizations, {}).get(device_name, {})
    model = None
    hf_model = os.environ.get("HF_MODEL", "microsoft/Phi-3.5-mini-instruct")
    cache_dir = lazy_weight_cache_dir_for_demo(mesh_device, hf_model)

    try:
        if test_config == "batch-32":
            max_bs, max_seq_len = int(os.environ.get("S11_BATCH", "32")), _BATCH32_MAX_SEQ_LEN[optimizations]  # S11 phi override
            expected = EXPECTED_METRICS_BATCH32.get(_sampling_bucket(), {}).get(optimizations, {}).get(device_name, {})
        elif test_config == "eval-32":
            # 32 users x 3 rotated repeats, a fresh traced executor per repeat.
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
            _run_perf_benchmark(model, mesh_device, expected, batch_size=int(os.environ.get("S11_BATCH", "32")), case_name=f"{optimizations}/batch-32")
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


def _run_token_accuracy(model: Phi35MiniTransformer, mesh_device: ttnn.MeshDevice, expected: dict):
    """Teacher-forcing token accuracy vs ``.refpt`` (CPU-generated)."""
    hf_model = os.environ.get("HF_MODEL", "microsoft/Phi-3.5-mini-instruct")
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
    model: Phi35MiniTransformer,
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
    hf_model = os.environ.get("HF_MODEL", "microsoft/Phi-3.5-mini-instruct")
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
        _PROMPT_BUCKET = int(os.environ.get("S11_PROMPT_BUCKET", "128"))  # S11 phi override
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


def _run_eval_repeat_batch32(model: Phi35MiniTransformer, mesh_device: ttnn.MeshDevice):
    """32-user cross-batch determinism (self-consistency under prompt rotation).

    Runs the batch-32 prefill+decode loop ``_EVAL_REPEAT_BATCHES`` times, rotating the prompt->slot
    assignment by one each repeat (fresh traced executor + KV cache per repeat), then asserts that
    undoing the rotation lines up per-user outputs. Honors the same ``SAMPLING_MODE`` knob as
    ``_run_perf_benchmark`` (default host argmax — deterministic and mesh-agnostic).
    """
    hf_model = os.environ.get("HF_MODEL", "microsoft/Phi-3.5-mini-instruct")
    tokenizer = model.demo_tokenizer

    # A new user turn (<|user|>) after an answer is a response terminator too; add it to the stop set the
    # determinism runner reads so a degenerate turn restart is truncated there.
    user_turn_id = tokenizer.convert_tokens_to_ids("<|user|>")
    if isinstance(user_turn_id, int) and user_turn_id >= 0:
        existing = list(getattr(tokenizer, "stop_tokens", None) or [])
        tokenizer.stop_tokens = list({*existing, user_turn_id})

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
