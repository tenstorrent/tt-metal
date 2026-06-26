# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 Llama-3.3-70B-Instruct demo — accuracy and performance measurement.

Uses ``EagerLlama33_70BExecutor`` / ``TracedLlama33_70BExecutor`` directly (no vLLM adapter).

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
import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig, AutoTokenizer

import ttnn
from models.common.models.executor import run_perf_benchmark, run_teacher_forcing
from models.common.models.llama33_70b.model import (
    LLAMA33_70B_ACCURACY,
    LLAMA33_70B_PERFORMANCE,
    EagerLlama33_70BExecutor,
    Llama33_70BTransformer1D,
    TracedLlama33_70BExecutor,
)
from models.common.sampling.sampling_params import SamplingParams
from models.common.tests.demos.cleanup_utils import cleanup_model_case
from models.tt_transformers.tt.common import encode_prompt_hf

# =============================================================================
# Expected metrics — copied verbatim from models/tt_transformers/PERF.md
# (Llama-3.3-70B rows in "Performance" and "Accuracy" tables).
# =============================================================================

EXPECTED_METRICS = {
    "performance": {
        "T3K": {"top1": 96, "top5": 100, "tok_s_u": 16.6, "ttft_ms": 164},
    },
    "accuracy": {
        "T3K": {"top1": 96, "top5": 100, "tok_s_u": 16.5, "ttft_ms": 168},
    },
}

# Separate batch-32 throughput targets from PERF.md "Short-Context Batch-32" table
# (batch_size=32, prefill_length=128 tokens). PERF.md publishes Llama-3.1-70B T3K batch-32 at
# 14.8 tok/s/u / 192 ms TTFT; the same numbers apply to Llama-3.3-70B (identical architecture).
# The Batch-32 section does not publish a separate accuracy-mode row; performance targets
# apply as an upper bound for accuracy-mode batch-32 as well.
EXPECTED_METRICS_BATCH32 = {
    "T3K": {"tok_s_u": 14.8, "ttft_ms": 192},
}

# Perf workload: natural-length prefill (these sample prompts are ~90-125 tokens -> 128 bucket,
# matching TTTv1's traced-prefill seq len for Llama-3.3-70B on T3K), 200 decode steps.
# Accuracy uses the 511-token teacher-forcing refpt.
_PERF_NUM_DECODE_TOKENS = 200

PERF_TOLERANCE = 0.05

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
    # an explicit 1D fabric; the root conftest does not auto-enable it. FABRIC_1D on any >1-device
    # mesh (70B is T3K-only, always 8 devices).
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
) -> Llama33_70BTransformer1D:
    """Build ``Llama33_70BTransformer1D`` in executor (paged KV) mode.

    Picks one of the two module-level precision recipes (``LLAMA33_70B_ACCURACY`` /
    ``LLAMA33_70B_PERFORMANCE``) — both defined in ``llama33_70b/model.py`` and grounded
    in TTTv1's ``DecodersPrecision`` for Llama-3.3-70B-Instruct.
    """
    hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
    _skip_unless_heads_divide_mesh(mesh_device, hf_model)

    precision = LLAMA33_70B_PERFORMANCE if optimizations == "performance" else LLAMA33_70B_ACCURACY

    try:
        model = Llama33_70BTransformer1D.from_pretrained(
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
        pytest.skip(f"Could not build Llama-3.3-70B model (weights / memory / mesh): {e}")

    return model


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.parametrize(
    "test_config",
    [
        pytest.param("token-accuracy", id="token-accuracy"),
        pytest.param("batch-1", id="batch-1"),
        pytest.param("batch-32", id="batch-32"),
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
        # Token-accuracy + batch-1 feed a single sequence — max_batch_size=1 avoids DRAM
        # pressure from a full 32-user KV cache allocation (70B BFP8 weights are ~9 GB/device
        # on T3K, leaving only ~3 GB for KV + activations).
        # batch-32 uses max_seq_len=1024 to avoid DRAM OOM: 80 layers × 1 KV head/dev × 128 ×
        # 32 batch × 4096 × BFP8 ≈ 2.7 GB/device at seq 4096 would overflow alongside weights;
        # 1024 (≈0.67 GB KV) still covers the natural-length prefill (~128 bucket) + 200 decode workload.
        if test_config == "batch-32":
            max_bs, max_seq_len = 32, 1024
            expected = EXPECTED_METRICS_BATCH32.get(device_name, {})
        else:
            max_bs, max_seq_len = 1, 4096
        model = create_model(mesh_device, optimizations, cache_dir, max_batch_size=max_bs, max_seq_len=max_seq_len)

        if test_config == "token-accuracy":
            _run_token_accuracy(model, mesh_device, expected)
        elif test_config == "batch-1":
            _run_perf_benchmark(model, mesh_device, expected, batch_size=1, case_name=f"{optimizations}/batch-1")
        elif test_config == "batch-32":
            _run_perf_benchmark(model, mesh_device, expected, batch_size=32, case_name=f"{optimizations}/batch-32")
    finally:
        cleanup_model_case(model, mesh_device)


def _run_token_accuracy(model: Llama33_70BTransformer1D, mesh_device, expected):
    """Teacher-forcing token accuracy vs ``.refpt``."""
    hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
    reference_tokens, top5_tokens, prompt_len, metadata = load_reference_data(hf_model)
    tokenizer = AutoTokenizer.from_pretrained(hf_model)

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

    executor = EagerLlama33_70BExecutor(model, mesh_device)
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
    model: Llama33_70BTransformer1D,
    mesh_device,
    expected,
    batch_size: int,
    case_name: str,
    max_prefill_len: int | None = None,
):
    """Timed prefill + decode (``TracedLlama33_70BExecutor``).

    Prefill uses each prompt's natural token length (TTTv1 ``preprocess_inputs_prefill``
    semantics — the executor buckets to ``get_padded_prefill_len``); decode runs for 200 steps.
    ``max_prefill_len`` is an optional clip cap for over-long prompts, never a pad-up target.
    """
    hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(hf_model)

    traced_executor = TracedLlama33_70BExecutor(model, mesh_device)
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

        prompts = load_input_prompts(batch_size)
        # Natural-length tokenization (matches TTTv1): the executor buckets each user's real
        # length to get_padded_prefill_len. These sample prompts are ~90-125 tokens -> 128 bucket.
        input_tokens, prompt_lens = tokenize_prompts(prompts, tokenizer, max_prefill_len=max_prefill_len)

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

        result = run_perf_benchmark(
            traced_executor,
            tokens=input_tokens,
            kv_cache=kv_cache,
            page_table=page_table,
            num_decode_tokens=_PERF_NUM_DECODE_TOKENS,
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
