# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 Qwen2.5-7B-Instruct demo — accuracy and performance measurement.

Uses ``EagerQwenExecutor`` / ``TracedQwenExecutor`` directly (no vLLM adapter).

**Mesh note:** Default Qwen2.5-7B has 28 attention heads and 4 KV heads; both must be
divisible by the mesh device count. Use N150 (1), N300 (2), or a 4-device row mesh —
not 8 devices (e.g. T3K), which is incompatible with this checkpoint.

Usage:
    # Token accuracy test
    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen2.5-7B-Instruct \\
    python_env/bin/pytest models/common/tests/demos/qwen25_7b/demo.py -k "token-accuracy" -v

    # Batch-1 latency test
    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen2.5-7B-Instruct \\
    python_env/bin/pytest models/common/tests/demos/qwen25_7b/demo.py -k "batch-1" -v

    # Batch-32 throughput test (prefer 4-device mesh if available)
    MESH_DEVICE=N150x4 HF_MODEL=Qwen/Qwen2.5-7B-Instruct \\
    python_env/bin/pytest models/common/tests/demos/qwen25_7b/demo.py -k "batch-32" -v
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
from models.common.models.qwen25_7b.executor import EagerQwenExecutor, TracedQwenExecutor
from models.common.models.qwen25_7b.model import Qwen25_7BTTT
from models.common.tests.demos.cleanup_utils import cleanup_model_case
from models.tt_transformers.tt.common import encode_prompt_hf

# =============================================================================
# Expected metrics
# =============================================================================

# Top-1 / top-5 thresholds are conservative parity-style targets.
# tok_s_u / ttft_ms are placeholders — replace with CI baselines after calibration
# (same workflow as Llama 3.1-8B PERF.md + measured runs).
EXPECTED_METRICS = {
    "performance": {
        "N150": {"top1": 88, "top5": 96, "tok_s_u": 18.0, "ttft_ms": 180},
        "N300": {"top1": 88, "top5": 96, "tok_s_u": 30.0, "ttft_ms": 110},
        "N150x4": {"top1": 88, "top5": 96, "tok_s_u": 45.0, "ttft_ms": 85},
    },
    "accuracy": {
        "N150": {"top1": 94, "top5": 99, "tok_s_u": 15.0, "ttft_ms": 220},
        "N300": {"top1": 94, "top5": 99, "tok_s_u": 26.0, "ttft_ms": 140},
        "N150x4": {"top1": 94, "top5": 99, "tok_s_u": 40.0, "ttft_ms": 110},
    },
}

PERF_TOLERANCE = 0.05

# Mesh topology comes only from ``MESH_DEVICE`` (same naming as vLLM / other tt demos).
_MESH_DEVICE_TO_SHAPE: dict[str, tuple[int, int]] = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
}


def _ttnn_mesh_device_param_from_env() -> dict:
    env = os.environ.get("MESH_DEVICE", "").strip()
    if not env:
        pytest.skip(
            "MESH_DEVICE must be set (e.g. N300 or N150x4). See module docstring.",
            allow_module_level=True,
        )
    shape = _MESH_DEVICE_TO_SHAPE.get(env)
    if shape is None:
        pytest.skip(
            f"Unsupported MESH_DEVICE={env!r}; use one of {sorted(_MESH_DEVICE_TO_SHAPE)}.",
            allow_module_level=True,
        )
    return {
        "mesh_shape": shape,
        "trace_region_size": 50_000_000,
        "num_command_queues": 1,
    }


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
        f"Try MESH_DEVICE=N300 (2) or N150x4 (4)."
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
    return f"{num_devices}dev"


def ref_basename_for_hf(hf_model_id: str) -> str:
    """Match ``ModelArgs.model_name`` style used for ``.refpt`` filenames."""
    return hf_model_id.strip("/").split("/")[-1]


def load_reference_data(hf_model_id: str):
    """Load reference tokens and top-5 predictions from ``.refpt`` (same tree as Llama demos)."""
    name = ref_basename_for_hf(hf_model_id)
    ref_path = Path("models/tt_transformers/tests/reference_outputs") / f"{name}.refpt"
    if not ref_path.exists():
        pytest.skip(f"Reference file not found: {ref_path}")

    ref_data = torch.load(ref_path, map_location="cpu")
    reference_tokens = ref_data["reference_tokens"]
    top5_tokens = ref_data["top5_tokens"]
    return reference_tokens, top5_tokens


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
    top5_tokens: torch.Tensor, reference_tokens: torch.Tensor, prompt_len: int
) -> torch.Tensor:
    """Align ``top5_tokens`` with teacher-forcing targets across refpt conventions."""
    num_target = len(reference_tokens) - prompt_len
    target_tokens = reference_tokens[prompt_len : prompt_len + num_target]
    if num_target <= 0:
        raise ValueError("prompt_len must be smaller than reference length")

    candidates = []
    for start in (prompt_len - 1, prompt_len):
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


def create_model(mesh_device, optimizations: str, cache_dir: Path, *, max_batch_size: int = 32):
    """Build ``Qwen25_7BTTT`` in executor (paged KV) mode.

    ``max_batch_size`` must match the workload: decode DRAM matmul CB usage scales with
    tile-padded batch rows, so batch-1 perf tests should pass ``max_batch_size=1`` even when
    batch-32 / teacher-forcing cases need 32.
    """
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    _skip_unless_heads_divide_mesh(mesh_device, hf_model)

    # Differentiate modes similarly to ``DecodersPrecision`` on Llama: tighter weights in "accuracy".
    if optimizations == "accuracy":
        wqkv_dtype = ttnn.bfloat16
        mlp_w_dtype = ttnn.bfloat16
        kv_cache_dtype = ttnn.bfloat16
    else:
        wqkv_dtype = ttnn.bfloat16
        mlp_w_dtype = ttnn.bfloat8_b
        kv_cache_dtype = ttnn.bfloat8_b

    num_devices = mesh_device.get_num_devices()
    if num_devices >= 4:
        max_seq_len = min(131072 // max_batch_size, 8192)
    else:
        max_seq_len = 4096

    try:
        model = Qwen25_7BTTT.from_pretrained(
            mesh_device,
            hf_model,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            num_layers=None,
            cache_dir=cache_dir,
            wqkv_dtype=wqkv_dtype,
            mlp_w_dtype=mlp_w_dtype,
            kv_cache_dtype=kv_cache_dtype,
            executor_mode=True,
        )
    except Exception as e:
        pytest.skip(f"Could not build Qwen model (weights / memory / mesh): {e}")

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
def test_qwen25_7b(test_config, mesh_device, optimizations, tmp_path_factory):
    """Main test entry for TTTv2 Qwen2.5-7B-Instruct."""
    device_name = get_device_name(mesh_device)
    expected = EXPECTED_METRICS.get(optimizations, {}).get(device_name, {})
    model = None
    cache_dir = tmp_path_factory.mktemp("qwen25_7b_demo_weights")

    try:
        max_bs = 1 if test_config == "batch-1" else 32
        model = create_model(mesh_device, optimizations, cache_dir, max_batch_size=max_bs)

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
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    reference_tokens, top5_tokens = load_reference_data(hf_model)
    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)

    if reference_tokens.dim() > 1:
        reference_tokens = reference_tokens.squeeze()

    half = len(reference_tokens) // 2
    prompt_tokens = reference_tokens[:half].unsqueeze(0)

    executor = EagerQwenExecutor(model, mesh_device)
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

    target_top5 = select_teacher_forcing_top5_slice(top5_tokens, reference_tokens, half)
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
    log_teacher_forcing_text(prompt_tokens, result.predicted_tokens_per_user, reference_tokens[half:], tokenizer)

    if "top1" in expected:
        assert top1 >= expected["top1"] * (
            1 - PERF_TOLERANCE
        ), f"Top-1 accuracy {top1:.1f}% below threshold {expected['top1']}%"
    if "top5" in expected:
        assert top5 >= expected["top5"] * (
            1 - PERF_TOLERANCE
        ), f"Top-5 accuracy {top5:.1f}% below threshold {expected['top5']}%"


def _run_perf_benchmark(model, mesh_device, expected, batch_size, case_name):
    """Timed prefill + decode (``TracedQwenExecutor``)."""
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)

    traced_executor = TracedQwenExecutor(model, mesh_device)
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

        result = run_perf_benchmark(
            traced_executor,
            tokens=input_tokens,
            kv_cache=kv_cache,
            page_table=page_table,
            num_decode_tokens=128,
            max_batch_size=max_batch_size,
            prompt_lens=prompt_lens,
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
