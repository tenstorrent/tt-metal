# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 DeepSeek-R1-Distill-Qwen-14B demo — accuracy and performance measurement on N300.

Uses ``EagerDeepSeekR1Qwen14BExecutor`` / ``TracedDeepSeekR1Qwen14BExecutor`` directly
(no vLLM adapter).

**Mesh note:** DeepSeek-R1-Distill-Qwen-14B has 40 attention heads and 8 KV heads;
both must be divisible by the mesh device count. N300 (2 devices) is the primary target
and is the only SKU for which expected metrics are recorded here. N150 (1 device) is
supported for smoke but not claimed for throughput in this bringup.

Usage::

    # Generate reference first (required for token-accuracy):
    ./python_env/bin/python models/common/tests/demos/deepseek_r1_distill_qwen_14b/generate_book_refpt.py \\
        --hf-model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

    # Token accuracy test (accuracy mode)
    MESH_DEVICE=N300 HF_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
      pytest models/common/tests/demos/deepseek_r1_distill_qwen_14b/demo.py \\
        -k "not performance and token-accuracy" -v

    # Batch-1 latency test
    MESH_DEVICE=N300 HF_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
      pytest models/common/tests/demos/deepseek_r1_distill_qwen_14b/demo.py -k "batch-1" -v

    # Batch-32 throughput test
    MESH_DEVICE=N300 HF_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
      pytest models/common/tests/demos/deepseek_r1_distill_qwen_14b/demo.py -k "batch-32" -v

LazyWeight tensor cache: ``TT_CACHE_PATH/<device_name>`` when ``TT_CACHE_PATH`` is set,
otherwise ``model_cache/<HF_MODEL>/<device_name>`` under the current working directory.

Reference artifact (``.refpt``): generate with ``generate_book_refpt.py`` before
running token-accuracy tests. The file is written to
``models/tt_transformers/tests/reference_outputs/DeepSeek-R1-Distill-Qwen-14B.refpt``.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig, AutoTokenizer

import ttnn
from models.common.models.deepseek_r1_distill_qwen_14b.executor import (
    EagerDeepSeekR1Qwen14BExecutor,
    TracedDeepSeekR1Qwen14BExecutor,
)
from models.common.models.deepseek_r1_distill_qwen_14b.model import (
    DEEPSEEK_R1_14B_ACCURACY,
    DEEPSEEK_R1_14B_PERFORMANCE,
    DeepSeekR1Qwen14B,
)
from models.common.models.executor import run_perf_benchmark, run_teacher_forcing
from models.common.sampling.sampling_params import SamplingParams
from models.common.tests.demos.cleanup_utils import cleanup_model_case
from models.tt_transformers.tt.common import encode_prompt_hf

# =============================================================================
# Expected metrics
# =============================================================================

# DeepSeek-R1-Distill-Qwen-14B is **not** in models/tt_transformers/PERF.md.
# Throughput / TTFT targets are NOT asserted here (no TTTv1 baseline to compare against).
# Accuracy thresholds below are the measured TT-vs-HF top1/top5 token agreement against the committed
# book reference (generate_book_refpt.py over tale-of-two-cities; HF intrinsic ceiling 79.8/91.8 top1/top5).
# The 5% PERF_TOLERANCE absorbs run-to-run variation. Measured on N300 (2026-07-03):
#   performance 88.1/99.6, accuracy 96.5/100.0 (top1/top5). N150 does not fit (14B exceeds a single
#   Wormhole's L1) and is explicitly skipped in test_deepseek_r1_qwen_14b; its entries below are
#   placeholders and never asserted.
#
# When PERF.md is eventually updated for this model, copy verbatim and add tok_s_u / ttft_ms
# entries here (following the Mistral-7B / Qwen2.5-7B pattern).
EXPECTED_METRICS = {
    "performance": {
        "N150": {"top1": 75, "top5": 90},
        "N300": {"top1": 88, "top5": 99},
    },
    "accuracy": {
        "N150": {"top1": 80, "top5": 93},
        "N300": {"top1": 96, "top5": 99},
    },
}

PERF_TOLERANCE = 0.05

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
    cfg = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True)
    n_h, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    if n_h % n_dev == 0 and n_kv % n_dev == 0:
        return
    pytest.skip(
        f"Incompatible mesh for {hf_model_id}: {n_dev} devices need "
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
    return f"{n}dev"


def lazy_weight_cache_dir_for_demo(mesh_device: ttnn.MeshDevice, hf_model_id: str) -> Path:
    """Disk root for LazyWeight caches. Follows the same convention as other TTTv2 demos."""
    device_name = get_device_name(mesh_device)
    hf = hf_model_id.strip("/")
    tt_cache = os.getenv("TT_CACHE_PATH")
    if tt_cache:
        root = Path(tt_cache) / device_name
    else:
        root = Path("model_cache") / hf / device_name
    root.mkdir(parents=True, exist_ok=True)
    logger.info(f"DeepSeek-R1-Distill-Qwen-14B demo LazyWeight cache directory: {root.resolve()}")
    return root


def ref_basename_for_hf(hf_model_id: str) -> str:
    return hf_model_id.strip("/").split("/")[-1]


def _load_tokenizer(hf_model_id: str):
    """Load HF tokenizer with writable-cache fallback for permission-restricted shared hosts."""
    try:
        return AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
    except (OSError, PermissionError) as e:
        msg = str(e)
        if "Permission" not in msg and "permission" not in msg:
            raise
        fallback = os.environ.get("TT_TOKENIZER_FALLBACK_CACHE", str(Path.home() / ".cache" / "huggingface"))
        logger.warning(f"Default HF cache not writable ({e!s:.120}); retrying with cache_dir={fallback}")
        Path(fallback).mkdir(parents=True, exist_ok=True)
        return AutoTokenizer.from_pretrained(hf_model_id, cache_dir=fallback, trust_remote_code=True)


def load_reference_data(hf_model_id: str):
    """Load reference tensors and optional metadata from ``.refpt``."""
    name = ref_basename_for_hf(hf_model_id)
    ref_path = Path("models/tt_transformers/tests/reference_outputs") / f"{name}.refpt"
    if not ref_path.exists():
        pytest.skip(
            f"Reference file not found: {ref_path}. "
            f"Generate with: python models/common/tests/demos/deepseek_r1_distill_qwen_14b/generate_book_refpt.py "
            f"--hf-model {hf_model_id}"
        )
    ref_data = torch.load(ref_path, map_location="cpu")
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


def preprocess_chat_prompts(
    prompts: list[str],
    tokenizer,
    *,
    max_seq_len: int,
    reserve_decode_tokens: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize with HF chat template, left-clip to budget, pad to batch max length."""
    max_prefill = max_seq_len - reserve_decode_tokens
    assert max_prefill > 0
    encoded: list[list[int]] = []
    for p in prompts:
        ids = encode_prompt_hf(tokenizer, p)
        if len(ids) > max_prefill:
            ids = ids[-max_prefill:]
        encoded.append(ids)
    max_len = max(len(x) for x in encoded)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    t = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
    for i, row in enumerate(encoded):
        t[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    lens = torch.tensor([len(row) for row in encoded], dtype=torch.long)
    return t, lens


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
) -> DeepSeekR1Qwen14B:
    """Build ``DeepSeekR1Qwen14B`` in executor (paged KV) mode.

    Picks one of the two module-level precision recipes (``DEEPSEEK_R1_14B_ACCURACY`` /
    ``DEEPSEEK_R1_14B_PERFORMANCE``) — both defined in
    ``deepseek_r1_distill_qwen_14b/model.py`` and grounded in TTTv1's
    ``DecodersPrecision`` for the generic Qwen2 path.

    ``max_batch_size`` must match the workload: decode DRAM matmul CB usage scales with
    tile-padded batch rows, so batch-1 perf tests pass ``max_batch_size=1`` even when
    batch-32 / teacher-forcing cases need 32.
    """
    hf_model = os.environ.get("HF_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    _skip_unless_heads_divide_mesh(mesh_device, hf_model)

    precision = DEEPSEEK_R1_14B_PERFORMANCE if optimizations == "performance" else DEEPSEEK_R1_14B_ACCURACY

    # N300 DRAM budget for DS-R1-Distill-Qwen-14B (48 layers, ~12 GB per device):
    #   Performance mode weights ≈ 6.85 GB (BFP8 attn + BFP4 FF1/FF3 + full embedding)
    #     → KV budget ≈ 5 GB → max_seq_len ≤ 2048 at batch-32 (BFP8 KV ≈ 3.1 GB)
    #   Accuracy mode weights ≈ 9.7 GB (BF16 attn + BFP8 MLP + full embedding)
    #     → KV budget ≈ 2 GB → max_seq_len ≤ 512 at batch-32 (BF16 KV ≈ 1.57 GB)
    if max_batch_size == 32:
        max_seq_len = 512 if optimizations != "performance" else 2048
    else:
        max_seq_len = 4096

    try:
        model = DeepSeekR1Qwen14B.from_pretrained(
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
        pytest.skip(f"Could not build DeepSeek-R1-Distill-Qwen-14B model (weights / memory / mesh): {e}")

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
def test_deepseek_r1_qwen_14b(test_config, mesh_device, optimizations):
    """Main test entry for TTTv2 DeepSeek-R1-Distill-Qwen-14B."""
    device_name = get_device_name(mesh_device)
    if device_name == "N150":
        pytest.skip(
            "DeepSeek-R1-Distill-Qwen-14B (14B) does not fit a single Wormhole's L1: a forced run "
            "overflows a statically allocated circular buffer in the distributed LayerNorm device op "
            "(1512864 B vs 1499136 B max), surfacing at the first forward. N150 is excluded."
        )
    expected = EXPECTED_METRICS.get(optimizations, {}).get(device_name, {})
    model = None
    hf_model = os.environ.get("HF_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    cache_dir = lazy_weight_cache_dir_for_demo(mesh_device, hf_model)

    try:
        # batch-32 is the only case that actually needs 32 users. token-accuracy and
        # batch-1 use max_batch_size=1 to keep decode CB footprint minimal.
        max_bs = 32 if test_config == "batch-32" else 1
        model = create_model(mesh_device, optimizations, cache_dir, max_batch_size=max_bs)

        if test_config == "token-accuracy":
            _run_token_accuracy(model, mesh_device, expected, hf_model)
        elif test_config == "batch-1":
            _run_perf_benchmark(model, mesh_device, expected, batch_size=1, case_name=f"{optimizations}/{test_config}")
        elif test_config == "batch-32":
            _run_perf_benchmark(model, mesh_device, expected, batch_size=32, case_name=f"{optimizations}/{test_config}")
    finally:
        cleanup_model_case(model, mesh_device)


def _run_token_accuracy(model: DeepSeekR1Qwen14B, mesh_device: ttnn.MeshDevice, expected: dict, hf_model: str):
    """Teacher-forcing token accuracy vs ``.refpt`` (CPU-generated)."""
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
        logger.info(
            f"Reference metadata: hf_model_id={metadata.get('hf_model_id')}, "
            f"revision={metadata.get('revision')}, created_at={metadata.get('created_at')}"
        )

    prompt_tokens = reference_tokens[:prompt_len].unsqueeze(0)

    executor = EagerDeepSeekR1Qwen14BExecutor(model, mesh_device)
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


def _run_perf_benchmark(
    model: DeepSeekR1Qwen14B,
    mesh_device: ttnn.MeshDevice,
    expected: dict,
    batch_size: int,
    case_name: str,
):
    """Timed prefill + decode via ``TracedDeepSeekR1Qwen14BExecutor``."""
    hf_model = os.environ.get("HF_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    tokenizer = _load_tokenizer(hf_model)

    traced_executor = TracedDeepSeekR1Qwen14BExecutor(model, mesh_device)
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
        input_tokens, prompt_lens = preprocess_chat_prompts(
            prompts, tokenizer, max_seq_len=max_seq_len, reserve_decode_tokens=128
        )

        # On-device sampling toggle (SAMPLING_MODE env):
        #   host            -> sampling_params=None (host-argmax, the default shipped path)
        #   on_device       -> greedy temp=0,k=1,p=0  => trace-captured FORCE-ARGMAX full-vocab path
        #   on_device_topk  -> temp=0,k=32,p=0.08     => trace-captured TOP-K op path (gathers only
        #                      the [*,32] tuples; faster than force-argmax on >=8-device meshes)
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
            failures = []
            if "tok_s_u" in expected and result.tok_s_u < expected["tok_s_u"] * (1 - PERF_TOLERANCE):
                failures.append(f"tok/s/u {result.tok_s_u:.1f} below target {expected['tok_s_u']}")
            if "ttft_ms" in expected and result.ttft_ms > expected["ttft_ms"] * (1 + PERF_TOLERANCE):
                failures.append(f"ttft_ms {result.ttft_ms:.1f} above target {expected['ttft_ms']}")
            assert not failures, f"{case_name}: " + "; ".join(failures)
    finally:
        traced_executor.cleanup()
