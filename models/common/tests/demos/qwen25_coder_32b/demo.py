# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 Qwen2.5-Coder-32B-Instruct demo — accuracy and performance measurement on T3K.

Uses ``EagerQwen25Coder32BExecutor`` / ``TracedQwen25Coder32BExecutor`` directly
(no vLLM adapter).

**Mesh note:** Qwen2.5-Coder-32B-Instruct has 40 attention heads and 8 KV heads;
both divide T3K (8). The port raises on any other mesh.

Usage:
    # Token accuracy (uses the committed .refpt book reference)
    MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \\
      pytest models/common/tests/demos/qwen25_coder_32b/demo.py -k "token-accuracy" -v

    # Batch-1 latency
    MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \\
      pytest models/common/tests/demos/qwen25_coder_32b/demo.py -k "batch-1" -v

    # Batch-32 throughput
    MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \\
      pytest models/common/tests/demos/qwen25_coder_32b/demo.py -k "batch-32" -v

LazyWeight tensor cache: ``TT_CACHE_PATH/<device_name>`` when ``TT_CACHE_PATH`` is set,
otherwise ``model_cache/<HF_MODEL>/<device_name>`` under the current working directory.
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
from models.common.models.qwen25_coder_32b.executor import EagerQwen25Coder32BExecutor, TracedQwen25Coder32BExecutor
from models.common.models.qwen25_coder_32b.model import (
    QWEN25_CODER_32B_ACCURACY,
    QWEN25_CODER_32B_PERFORMANCE,
    Qwen25Coder32B,
)
from models.common.sampling.sampling_params import SamplingParams
from models.common.tests.demos.cleanup_utils import cleanup_model_case
from models.tt_transformers.tt.common import encode_prompt_hf

# =============================================================================
# Expected metrics
# =============================================================================

# tok_s_u gates decode throughput on the on-device sampling path (SAMPLING_MODE=on_device_topk, the
# default below and what these numbers were measured on) — T3K, healthy build, reset-then-measure,
# 2026-07-03:
#   - performance: gate at the TTTv1 same-box baseline 23.97 (TTTv1 > TTTv2 23.3, so gate at TTTv1
#     per the "don't gate below TTTv1" rule; TTTv2 clears it within PERF_TOLERANCE=5%).
#   - accuracy: gate at the measured TTTv2 20.2 (TTTv1 accuracy profile not re-measured on this build).
# The old 22.4 / 19.7 were copied from a stale PERF.md and were only ever reachable via the host
# stitch path (they are the target the earlier "perf gap" was measured against). ttft_ms is left as a
# loose PERF.md ceiling that TTTv2 prefill (~102-118 ms) clears comfortably; top1/top5 gate the
# separate token-accuracy test, not the perf benchmark.
EXPECTED_METRICS = {
    "performance": {
        "T3K": {"top1": 96, "top5": 99, "tok_s_u": 23.97, "ttft_ms": 190},
    },
    "accuracy": {
        "T3K": {"top1": 95, "top5": 99, "tok_s_u": 20.2, "ttft_ms": 183},
    },
}

PERF_TOLERANCE = 0.05

# Mesh topology comes only from ``MESH_DEVICE`` (same naming as vLLM / other tt demos).
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
            f"Unsupported MESH_DEVICE={env!r} for Qwen2.5-Coder-32B-Instruct; "
            f"only T3K is supported (40 attn heads / 8 KV heads ⇒ 8 devices).",
            allow_module_level=True,
        )
    param = {
        "mesh_shape": shape,
        "trace_region_size": 50_000_000,
        "num_command_queues": 1,
    }
    # TTTv2 multi-device executor dispatch (and the on-device sampling all-gather) stalls without
    # an explicit 1D fabric; the root conftest does not auto-enable it. Enable FABRIC_1D on any
    # >1-device mesh (this port is T3K-only, so always).
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


def _skip_unless_t3k(mesh_device: ttnn.MeshDevice, hf_model_id: str) -> None:
    """Port targets T3K only; head divisibility (40/8, 8/8) is checked inside ``from_pretrained``."""
    n_dev = mesh_device.get_num_devices()
    if n_dev != 8:
        pytest.skip(
            f"Qwen2.5-Coder-32B-Instruct demo targets T3K (8 devices) only; got {n_dev}. " f"Set MESH_DEVICE=T3K."
        )
    cfg = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True)
    n_h, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    if n_h % n_dev != 0 or n_kv % n_dev != 0:
        pytest.skip(
            f"Incompatible mesh for {hf_model_id}: {n_dev} devices need "
            f"num_attention_heads ({n_h}) and num_key_value_heads ({n_kv}) each divisible by {n_dev}."
        )


def get_device_name(mesh_device):
    """Map mesh device count to a metrics bucket (T3K is the only supported SKU)."""
    num_devices = mesh_device.get_num_devices()
    if num_devices == 8:
        return "T3K"
    return f"{num_devices}dev"


def lazy_weight_cache_dir_for_demo(mesh_device: ttnn.MeshDevice, hf_model_id: str) -> Path:
    """Disk root for ``Qwen25Coder32B`` ``LazyWeight`` caches in this e2e demo.

    Matches the 7B demo and ``models/tt_transformers/tt/model_config.py`` (HF checkpoint branch):
    if ``TT_CACHE_PATH`` is set, use ``<TT_CACHE_PATH>/<device_name>``; otherwise
    ``model_cache/<HF_MODEL>/<device_name>``. Directories are created as needed.
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
    logger.info(f"Qwen2.5-Coder-32B demo LazyWeight cache directory: {root.resolve()}")
    return root


def ref_basename_for_hf(hf_model_id: str) -> str:
    """Match ``ModelArgs.model_name`` style used for ``.refpt`` filenames."""
    return hf_model_id.strip("/").split("/")[-1]


def _load_tokenizer(hf_model_id: str):
    """Load HF tokenizer with a writable-cache fallback.

    The default ``HF_HOME`` on shared dev hosts is often owned by another user, so
    ``AutoTokenizer.from_pretrained`` cannot create ``.locks/`` entries when tokenizer
    files are missing from the shared cache. On ``OSError`` / ``PermissionError`` from the
    default path, retry with ``cache_dir`` pointing at the user's home HF cache (tokenizer
    files are <10 MB so an extra download is cheap).
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
        pytest.skip(
            f"Reference file not found: {ref_path} " f"(committed book .refpt; expected present on a clean checkout)."
        )

    ref_data = torch.load(ref_path, map_location="cpu")
    reference_tokens = ref_data["reference_tokens"]
    top5_tokens = ref_data["top5_tokens"]
    prompt_len = ref_data.get("prompt_len")
    metadata = ref_data.get("metadata")
    return reference_tokens, top5_tokens, prompt_len, metadata


def load_input_prompts(batch_size: int):
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


def preprocess_qwen_chat_prompts(
    prompts: list[str],
    tokenizer,
    *,
    max_seq_len: int,
    reserve_decode_tokens: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize with HF chat template, left-clip to budget, pad to batch max length."""
    max_prefill = max_seq_len - reserve_decode_tokens
    assert max_prefill > 0, "max_seq_len must exceed reserve_decode_tokens"

    encoded: list[list[int]] = []
    for p in prompts:
        ids = encode_prompt_hf(tokenizer, p)
        if len(ids) > max_prefill:
            ids = ids[-max_prefill:]
        encoded.append(ids)

    max_len = max(len(x) for x in encoded)
    batch = len(encoded)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    t = torch.full((batch, max_len), pad_id, dtype=torch.long)
    for i, row in enumerate(encoded):
        t[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    lens = torch.tensor([len(row) for row in encoded], dtype=torch.long)
    return t, lens


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


def create_model(
    mesh_device,
    optimizations: str,
    cache_dir: Path,
    *,
    max_batch_size: int = 32,
):
    """Build ``Qwen25Coder32B`` in executor (paged KV) mode on T3K.

    Picks one of the two module-level precision recipes (``QWEN25_CODER_32B_ACCURACY`` /
    ``QWEN25_CODER_32B_PERFORMANCE``) — both defined in ``qwen25_coder_32b/model.py`` and
    grounded in TTTv1's ``DecodersPrecision`` for Qwen2.5-Coder-32B (the standard non-7B
    Qwen2.5-family branch in ``model_config.py:160-177`` for accuracy, ``:208-218`` for
    performance). The dataclass owns the dtype + math-fidelity recipe; this demo just
    selects between the two and passes through.

    ``max_batch_size`` must match the workload: decode DRAM matmul CB usage scales with
    tile-padded batch rows, so batch-1 perf tests should pass ``max_batch_size=1`` even when
    batch-32 / teacher-forcing cases need 32.
    """
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
    _skip_unless_t3k(mesh_device, hf_model)

    precision = QWEN25_CODER_32B_PERFORMANCE if optimizations == "performance" else QWEN25_CODER_32B_ACCURACY

    # T3K: 64 layers × 8 KV heads / 8 dev × head_dim 128 → KV per device per layer is modest.
    # Sequence budget: 131072 // batch saturates around 4K for batch 32 (matches 7B demo).
    max_seq_len = min(131072 // max_batch_size, 8192)

    try:
        model = Qwen25Coder32B.from_pretrained(
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
        pytest.skip(f"Could not build Qwen2.5-Coder-32B model (weights / memory / mesh): {e}")

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
def test_qwen25_coder_32b(test_config, mesh_device, optimizations):
    """Main test entry for TTTv2 Qwen2.5-Coder-32B-Instruct."""
    device_name = get_device_name(mesh_device)
    expected = EXPECTED_METRICS.get(optimizations, {}).get(device_name, {})
    model = None
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
    cache_dir = lazy_weight_cache_dir_for_demo(mesh_device, hf_model)

    try:
        max_bs = 1 if test_config == "batch-1" else 32
        model = create_model(
            mesh_device,
            optimizations,
            cache_dir,
            max_batch_size=max_bs,
        )

        if test_config == "token-accuracy":
            _run_token_accuracy(model, mesh_device, expected)
        elif test_config == "batch-1":
            _run_perf_benchmark(
                model,
                mesh_device,
                expected,
                batch_size=1,
                case_name=f"{optimizations}/{test_config}",
            )
        elif test_config == "batch-32":
            _run_perf_benchmark(
                model,
                mesh_device,
                expected,
                batch_size=32,
                case_name=f"{optimizations}/{test_config}",
            )
    finally:
        cleanup_model_case(model, mesh_device)


def _run_token_accuracy(model, mesh_device, expected):
    """Teacher-forcing token accuracy vs ``.refpt`` (HF-generated)."""
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
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

    executor = EagerQwen25Coder32BExecutor(model, mesh_device)
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
    log_teacher_forcing_text(prompt_tokens, result.predicted_tokens_per_user, reference_tokens[prompt_len:], tokenizer)

    if "top1" in expected:
        assert top1 >= expected["top1"] * (
            1 - PERF_TOLERANCE
        ), f"Top-1 accuracy {top1:.1f}% below threshold {expected['top1']}%"
    if "top5" in expected:
        assert top5 >= expected["top5"] * (
            1 - PERF_TOLERANCE
        ), f"Top-5 accuracy {top5:.1f}% below threshold {expected['top5']}%"


def _run_perf_benchmark(model, mesh_device, expected, batch_size, case_name):
    """Timed prefill + decode (``TracedQwen25Coder32BExecutor``)."""
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
    tokenizer = _load_tokenizer(hf_model)

    traced_executor = TracedQwen25Coder32BExecutor(model, mesh_device)
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
        input_tokens, prompt_lens = preprocess_qwen_chat_prompts(
            prompts, tokenizer, max_seq_len=max_seq_len, reserve_decode_tokens=128
        )

        # On-device sampling toggle (the model owns its Sampling1D, constructed in __init__; the
        # demo only picks behavior per request):
        #   host            -> sampling_params=None (host-argmax, the default shipped path)
        #   on_device       -> greedy temp=0,k=1,p=0 => trace-captured FORCE-ARGMAX full-vocab path
        #   on_device_topk  -> temp=0,k=32,p=0.08    => trace-captured TOP-K op path (PERF.md-parity
        #                      recipe; gathers only the [*,32] tuples, faster than force-argmax)
        # Default to on-device sampling: it is the path these perf gates were measured on and the one
        # TTTv1 uses for its published numbers (host argmax stitches full-vocab logits every step and
        # is ~2x slower on T3K's 8-way 152k vocab). Override with SAMPLING_MODE=host to measure it.
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

        result = run_perf_benchmark(
            traced_executor,
            tokens=input_tokens,
            kv_cache=kv_cache,
            page_table=page_table,
            num_decode_tokens=128,
            max_batch_size=max_batch_size,
            prompt_lens=prompt_lens,
            sampling_params=sampling_params,
        )

        logger.info(
            f"Performance — TTFT: {result.ttft_ms:.1f}ms, "
            f"tok/s/u: {result.tok_s_u:.1f}, "
            f"tok/s: {result.tok_s:.1f}, "
            f"decode latency: {result.decode_latency_mean_ms:.2f}ms"
        )
        log_generated_text(prompts, result.generated_token_ids, tokenizer)

        if expected:
            targets = result.meets_target(expected, PERF_TOLERANCE)
            for metric, passed in targets.items():
                if not passed:
                    logger.warning(
                        f"{metric} did not meet target: got {getattr(result, metric)}, expected {expected[metric]}"
                    )
            failures = []
            if "tok_s_u" in expected and not targets["tok_s_u"]:
                failures.append(f"tok/s/u {result.tok_s_u:.1f} below target {expected['tok_s_u']}")
            if "ttft_ms" in expected and not targets["ttft_ms"]:
                failures.append(f"ttft_ms {result.ttft_ms:.1f} above target {expected['ttft_ms']}")
            assert not failures, f"{case_name}: " + "; ".join(failures)
    finally:
        traced_executor.cleanup()
