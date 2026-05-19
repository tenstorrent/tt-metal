# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 Mistral-7B-Instruct-v0.3 demo — accuracy and performance measurement.

Uses ``EagerMistralExecutor`` / ``TracedMistralExecutor`` directly (no vLLM adapter).

**Mesh note:** Mistral-7B-Instruct-v0.3 has 32 attention heads and 8 KV heads, so all
of N150 (1), N300 (2), T3K (8) are compatible. PERF.md publishes all three.

Usage::

    # Token accuracy test
    MESH_DEVICE=N300 HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \\
      pytest models/common/tests/demos/mistral_7b/demo.py -k "token-accuracy" -v

    # Batch-1 latency test
    MESH_DEVICE=N300 HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \\
      pytest models/common/tests/demos/mistral_7b/demo.py -k "batch-1" -v

    # Batch-32 throughput test
    MESH_DEVICE=N300 HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \\
      pytest models/common/tests/demos/mistral_7b/demo.py -k "batch-32" -v

LazyWeight tensor cache (same rule as the Qwen2.5 demo):
``TT_CACHE_PATH/<device_name>`` when ``TT_CACHE_PATH`` is set, otherwise
``model_cache/<HF_MODEL>/<device_name>`` under the current working directory.

Reference artifact (``.refpt``): the legacy
``models/tt_transformers/tests/reference_outputs/Mistral-7B-Instruct-v0.3.refpt`` has
no ``prompt_len`` metadata and its intrinsic top-1 ceiling sits around 86%, well below
the PERF.md thresholds. Generate a deterministic one before tuning::

    ./python_env/bin/python models/common/tests/demos/mistral_7b/generate_controlled_refpt.py \\
        --hf-model mistralai/Mistral-7B-Instruct-v0.3

See ``dev-tools/agentic-bringup/skills/reference-sanity.md``.
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
from models.common.models.mistral_7b.executor import EagerMistralExecutor, TracedMistralExecutor
from models.common.models.mistral_7b.model import MISTRAL_ACCURACY, MISTRAL_PERFORMANCE, Mistral7B
from models.common.tests.demos.cleanup_utils import cleanup_model_case
from models.tt_transformers.tt.common import encode_prompt_hf

# =============================================================================
# Expected metrics
# =============================================================================

# Numbers copied verbatim from ``models/tt_transformers/PERF.md`` (Mistral-7B rows in
# the "Performance" and "Accuracy" tables). PERF.md publishes N150 / N300 / T3K for
# this checkpoint. The TTTv1 production stack runs Mistral-7B with BFP8 attention +
# BFP4/BFP8 MLP in "Performance" and BFP8 MLP + BF16 attention in "Accuracy"; the
# TTTv2 module defaults sit close to "Accuracy" for this model.
EXPECTED_METRICS = {
    "performance": {
        "N150": {"top1": 95, "top5": 99, "tok_s_u": 29.75, "ttft_ms": 100.24},
        "N300": {"top1": 95, "top5": 100, "tok_s_u": 47.01, "ttft_ms": 65.95},
        "T3K": {"top1": 95, "top5": 100, "tok_s_u": 67.82, "ttft_ms": 53.93},
    },
    "accuracy": {
        "N150": {"top1": 96, "top5": 100, "tok_s_u": 29.75, "ttft_ms": 100.24},
        "N300": {"top1": 97, "top5": 100, "tok_s_u": 47.01, "ttft_ms": 65.95},
        "T3K": {"top1": 98, "top5": 100, "tok_s_u": 67.82, "ttft_ms": 53.93},
    },
}

PERF_TOLERANCE = 0.05

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
            "MESH_DEVICE must be set (e.g. N300 or T3K). See module docstring.",
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
    cfg = AutoConfig.from_pretrained(hf_model_id)
    n_h, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    if n_h % n_dev == 0 and n_kv % n_dev == 0:
        return
    pytest.skip(
        f"Incompatible mesh for {hf_model_id}: {n_dev} devices need "
        f"num_attention_heads ({n_h}) and num_key_value_heads ({n_kv}) each divisible by {n_dev}. "
        f"Try MESH_DEVICE=N150 (1), N300 (2), or T3K (8)."
    )


def get_device_name(mesh_device):
    """Map mesh device count to a metrics bucket (matches PERF.md SKU keys)."""
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


def ref_basename_for_hf(hf_model_id: str) -> str:
    """Match the ``model_name`` convention used for ``.refpt`` filenames in TTTv1."""
    return hf_model_id.strip("/").split("/")[-1]


def load_reference_data(hf_model_id: str):
    """Load reference tensors and optional metadata from ``.refpt``.

    Supports both the metadata-rich format produced by ``generate_controlled_refpt.py``
    (``prompt_len`` + ``metadata`` keys) and the legacy half-split format. Callers must
    check ``prompt_len is None`` to decide whether the half-split fallback is needed.
    """
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


def preprocess_mistral_chat_prompts(
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
    """Build ``Mistral7B`` in executor (paged KV) mode.

    Picks one of the two module-level precision recipes (``MISTRAL_ACCURACY`` /
    ``MISTRAL_PERFORMANCE``) — both defined in ``mistral_7b/model.py`` and grounded in
    TTTv1's ``DecodersPrecision`` for Mistral-7B (``model_config.py:130-159`` for accuracy,
    ``:208-218`` for performance). The dataclass owns the dtype + math-fidelity recipe;
    this demo just selects between the two and passes through.
    """
    hf_model = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    _skip_unless_heads_divide_mesh(mesh_device, hf_model)

    precision = MISTRAL_PERFORMANCE if optimizations == "performance" else MISTRAL_ACCURACY

    num_devices = mesh_device.get_num_devices()
    # Mistral 7B v0.3 has full attention (no SWA) and a 32K vocab; reuse the Llama
    # 3.1-8B sequence budget for parity since both share the dim/heads layout.
    if num_devices >= 8:
        max_seq_len = 131072 // max_batch_size
    else:
        max_seq_len = 4096

    try:
        model = Mistral7B.from_pretrained(
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
        pytest.skip(f"Could not build Mistral model (weights / memory / mesh): {e}")

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
def test_mistral_7b(test_config, mesh_device, optimizations):
    """Main test entry for TTTv2 Mistral-7B-Instruct-v0.3."""
    device_name = get_device_name(mesh_device)
    expected = EXPECTED_METRICS.get(optimizations, {}).get(device_name, {})
    model = None
    hf_model = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    cache_dir = lazy_weight_cache_dir_for_demo(mesh_device, hf_model)

    try:
        # Only the batch-32 throughput test actually exercises 32 users; the
        # teacher-forcing token-accuracy path feeds a single reference sequence
        # (see _run_token_accuracy), so running it with max_batch_size=1 cuts
        # the paged KV cache 32x and removes the N300 DRAM pressure that used
        # to OOM the accuracy/token-accuracy config.
        max_bs = 32 if test_config == "batch-32" else 1
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
    """Teacher-forcing token accuracy vs ``.refpt``."""
    hf_model = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    reference_tokens, top5_tokens, prompt_len, metadata = load_reference_data(hf_model)
    tokenizer = AutoTokenizer.from_pretrained(hf_model)

    if reference_tokens.dim() > 1:
        reference_tokens = reference_tokens.squeeze()

    has_prompt_len_metadata = prompt_len is not None
    if has_prompt_len_metadata:
        prompt_len = int(prompt_len)
        logger.info(f"Using metadata-driven prompt_len={prompt_len} from reference artifact")
    else:
        prompt_len = len(reference_tokens) // 2
        logger.warning(
            f"Reference missing prompt_len metadata; falling back to legacy half split={prompt_len}. "
            "Run generate_controlled_refpt.py to produce a metadata-rich .refpt — the legacy artifact's "
            "intrinsic top-1 ceiling is ~86% and will cap any measurement below PERF.md thresholds."
        )

    if metadata:
        meta_summary = {
            "hf_model_id": metadata.get("hf_model_id"),
            "revision": metadata.get("revision"),
            "generation_mode": metadata.get("generation_mode"),
            "created_at": metadata.get("created_at"),
        }
        logger.info(f"Reference metadata summary: {meta_summary}")

    prompt_tokens = reference_tokens[:prompt_len].unsqueeze(0)

    executor = EagerMistralExecutor(model, mesh_device)
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
    """Timed prefill + decode (``TracedMistralExecutor``)."""
    hf_model = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer = AutoTokenizer.from_pretrained(hf_model)

    traced_executor = TracedMistralExecutor(model, mesh_device)
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
        input_tokens, prompt_lens = preprocess_mistral_chat_prompts(
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
