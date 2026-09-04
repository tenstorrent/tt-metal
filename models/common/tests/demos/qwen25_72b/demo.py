# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 Qwen2.5-72B-Instruct demo — accuracy and performance measurement on T3K.

Uses the model-owned ``Qwen25_72BExecutor`` directly (no vLLM adapter).

**Mesh note — T3K only.** Qwen2.5-72B-Instruct has 64 attention heads and 8 KV heads; both
divide 8, and the 72B weights need 8-way tensor parallelism to fit (a single/2-device mesh cannot
hold the weights + KV cache). This matches TTTv1/PERF.md (T3K-only for this checkpoint).
Consequently:
  - **T3K (8 devices): the validated mesh.** ``from_pretrained`` rejects any non-8 mesh.
  - **ci-b1-DP-*: skipped** — every DP group is a single device, which cannot hold this 72B (same
    memory limit); you cannot have both 1-device-per-user and 8-device TP. Genuine hardware-capacity
    guard, matching TTTv1 which also can't DP a 72B on T3K.

CI cases (parity with TTTv1 ``simple_text_demo.py``):
    token-accuracy   - teacher-forcing top-1/top-5 vs the book ``.refpt``
    batch-1          - single-user latency
    batch-32         - short-context throughput (seq1024 / 200 decode)
    batch-32-ci      - CI-faithful batch-32 (seq2048 / 1024 decode; TTTv1 ci-32)
    eval-32          - 32-user cross-batch determinism (TTTv1 ci-eval-32)
    ci-b1-DP-{2..32} - single-user data-parallel scaling smoke (TTTv1 ci-b1-DP-*); all skip on T3K

Usage:
    # Token accuracy (gates against the committed book ``.refpt``)
    MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen2.5-72B-Instruct \\
      pytest models/common/tests/demos/qwen25_72b/demo.py -k "token-accuracy" -v

    # On-device sampling perf sweep (the T3K headline / TTTv1-comparable path)
    SAMPLING_MODE=on_device_topk MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen2.5-72B-Instruct \\
      pytest models/common/tests/demos/qwen25_72b/demo.py -k "batch-32-ci" -v

LazyWeight tensor cache: ``TT_CACHE_PATH/<device_name>`` when ``TT_CACHE_PATH`` is set, otherwise
``model_cache/<HF_MODEL>/<device_name>`` under the current working directory.
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
from models.common.models.qwen25_72b.executor import Qwen25_72BExecutor, Qwen25_72BExecutorConfig
from models.common.models.qwen25_72b.hf_adaptor import encode_prompt, from_pretrained, load_tokenizer
from models.common.models.qwen25_72b.model import QWEN25_72B_ACCURACY, QWEN25_72B_PERFORMANCE, Qwen25_72B
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
# Expected metrics — perf gates set from a same-box TTTv1-vs-TTTv2 sweep (on-device sampling),
# NOT PERF.md (PERF.md's 22.4/19.7 tok/s/u are stale, reachable only via the host stitch path).
#
# Rule (per cell): each ``tok_s_u`` target is the BETTER of TTTv1 vs TTTv2 for that sampling mode.
# TTTv1 has only an on-device sampling path, so:
#     on_device_topk : max(TTTv1_on_device, TTTv2_on_device_topk)
#     host           : TTTv2_host                      (TTTv1 has no host-sampling path)
# Decode throughput is prefill-independent, so batched prefill does NOT change ``tok_s_u``.
# ``ttft_ms`` targets are conservative upper bounds (batched prefill only LOWERS TTFT).
#
# Perf cases default to SAMPLING_MODE=on_device_topk, the path comparable to TTTv1's auto on-device
# sampling on T3K (vocab shards 8-way). The host path pays a full-vocab all-gather + PCIe readback per
# step (~2x slower on T3K) and is NOT comparable to TTTv1 — measuring host vs TTTv1 fabricates a "gap".
# The host bucket below is left ungated ({}) unless separately measured; a case still RUNS + prints
# tok_s_u. All on_device_topk values below are freshly measured, best-of vs same-box TTTv1.
# =============================================================================

# top1/top5 teacher-forcing accuracy floors (book refpt), profile-split. Perf metrics live in the batch
# dicts below. Floors set conservatively below measured (5% PERF_TOLERANCE gives headroom).
EXPECTED_METRICS: dict = {
    "performance": {
        "T3K": {"top1": 96, "top5": 99},
    },
    "accuracy": {
        "T3K": {"top1": 96, "top5": 99},
    },
}

# batch-1 throughput, sampling-mode- and profile-aware. on_device_topk is the T3K headline; gate =
# better-of(TTTv1, TTTv2) per the parity rule, finalized from a fresh same-box TTTv1-vs-TTTv2 matrix.
# host bucket left ungated ({}) — not the T3K-comparable path.
EXPECTED_METRICS_BATCH1: dict = {
    "host": {
        # host on T3K is the degenerate, non-shipped sampler (full-vocab all-gather + PCIe readback
        # every step → ~2x slower than on-device: measured 9.5 t/s/u). Ungated (runs + prints);
        # on-device is the CI-comparable path.
        "performance": {},
        "accuracy": {},
    },
    "on_device_topk": {
        # gate = better-of(TTTv1 default, TTTv2 odt). Same-box TTTv1 perf-ci-1 (base 32c1f0e882b) = 16.24
        # t/s/u (window-matched to TTTv2's 200-token decode window; on-device top-k, force_argmax=False) /
        # 181.69 ms TTFT; TTTv2 odt = 16.30 / 190.5 → decode PARITY (best-of 16.30; floor 16.1 conservative,
        # never lowered). TTFT b1 is a +4.9% RED residual (190.5 vs 181.69) — b1 buckets to seq128 where
        # minimal_matmul is inert (gated >128) and the device last-token slice is already used, so it is the
        # shared single-user prefill critical path (ticket b32ci-prefill-ttft-minimal-matmul, also_covers_b1).
        # TTTv1 ACCURACY b1 DRAM-OOMs (higher-precision recipe) → acc cells own-gate; TTTv2 acc == perf
        # (">70B" identical recipe). ttft gate 200 = best-of ceiling (b1 TTFT ~181-190, run-to-run noisy).
        "performance": {"T3K": {"tok_s_u": 16.1, "ttft_ms": 200}},
        "accuracy": {"T3K": {"tok_s_u": 16.1, "ttft_ms": 200}},
    },
}

# Short-context batch-32 throughput (seq1024 / 200 decode), sampling-mode- and profile-aware. Runs BOTH
# batched-prefill ON (default) and DISABLE_BATCHED_PREFILL=1 (A/B). Decode tok_s_u is prefill-independent
# so the gate covers both knob states; ttft covers both (ON 102 << OFF 179 → gate above the sequential).
EXPECTED_METRICS_BATCH32: dict = {
    "host": {
        # degenerate non-shipped T3K host path (measured 9.6 t/s/u). Ungated. See Table B.
        "performance": {},
        "accuracy": {},
    },
    "on_device_topk": {
        # gate = TTTv2 odt (same-box TTTv1 batch-32 trace-region OOMs on this base — >70 MB trace buffers
        # for the 32-user batched-prefill trace exceed TTTv1's hardcoded region; "use the side that works",
        # PARITY_RULES §2). TTTv2 b32 = 15.9 / 102 ms ON. Decode is batch-robust: 15.9 is only −1% vs the
        # b1 parity cell (16.1 ≈ TTTv1 16.06). ttft 185 = ceiling covering batched ON (102) AND the
        # DISABLE_BATCHED_PREFILL=1 sequential A/B baseline (179; batched prefill is a 1.75× TTFT win).
        "performance": {"T3K": {"tok_s_u": 15.9, "ttft_ms": 185}},
        "accuracy": {"T3K": {"tok_s_u": 15.9, "ttft_ms": 185}},
    },
}

# CI-faithful batch-32 targets (the ``batch-32-ci`` leg), seq1024 + 1024-token decode budget (clamped to
# ~880 by the KV headroom) = the DIRECT TTTv1 ci-32 analog (72B clamps seq to 1024, see
# _BATCH32_CI_MAX_SEQ_LEN). Runs batched ON + OFF; ttft is a ceiling covering both.
EXPECTED_METRICS_BATCH32_CI: dict = {
    "host": {
        # degenerate non-shipped T3K host path. Ungated. See Table B.
        "performance": {},
        "accuracy": {},
    },
    "on_device_topk": {
        # gate = best-of(TTTv1 default, TTTv2 odt). Decode: TTTv2 odt 15.56 (decode latency 64.25ms) vs
        # same-box TTTv1 ci-32 'Average speed' 15.54 = PARITY (both batch the prefill, grow KV over the
        # ~880-token window); gate floor 15.5 is conservative (best-of 15.56, never lowered). TTFT: with
        # minimal_matmul ENABLED (2026-07-25, model.py) TTTv2 b32-ci = 81.2 ms ON (A/B: minimal_matmul OFF
        # 97.2 ms → a −16.5% prefill win). Same-box TTTv1 ci-32 = 68.74 ms (batched) but ONLY runs after a
        # TEMPORARY, uncommitted trace-region bump (its committed 70 MB region trace-OOMs the 32-user
        # batched-prefill trace). ttft gate 185 is a best-of ceiling covering batched ON (81.2) AND
        # the DISABLE_BATCHED_PREFILL=1 sequential A/B baseline (~179); never lowered to a slow number.
        "performance": {"T3K": {"tok_s_u": 15.5, "ttft_ms": 185}},
        "accuracy": {"T3K": {"tok_s_u": 15.5, "ttft_ms": 185}},
    },
}

# Perf workload: natural-length prefill (these sample prompts are ~90-125 tokens -> 128 bucket,
# matching TTTv1), 200 decode steps. Accuracy uses the teacher-forcing refpt. PERF_NUM_DECODE_TOKENS
# overrides the decode-step count (e.g. a short window for tt-perf-report device profiling).
_PERF_NUM_DECODE_TOKENS = int(os.environ.get("PERF_NUM_DECODE_TOKENS", "200"))

PERF_TOLERANCE = 0.05

# batch-32-ci per-SKU max_seq_len. TTTv1 ci-32 parity is seq2048, but the 72B BFP4-MLP + BFP8-attn
# weights are ~9-10 GB/device on T3K and a 32-user KV cache at seq2048 DRAM-OOMs (bank_manager) — the
# same 80-layer / 1-KV-head-per-dev / head_dim-128 footprint as Llama-3.3-70B, which also clamps to
# 1024. 1024 still covers the 128-token prefill bucket + the ~880-token clamped decode budget (see the
# effective_decode clamp in _run_perf_benchmark). T3K-only.
_BATCH32_CI_MAX_SEQ_LEN: dict[str, int] = {
    "T3K": 1024,
}


def _sampling_bucket() -> str:
    """Map SAMPLING_MODE to a perf-gate bucket. Defaults to ``on_device_topk`` (the perf-case default
    for this T3K model), so the bucket always agrees with the runner. Non-topk on-device modes (e.g.
    force-argmax) also fall into ``on_device_topk`` so they stay gated, never silently un-gated."""
    return "host" if os.environ.get("SAMPLING_MODE", "on_device_topk").lower() == "host" else "on_device_topk"


# Qwen2.5-72B needs at least this many devices of tensor parallelism: the 72B weights + KV cache
# require 8-way sharding to fit (and 64/8 attn/KV heads divide 8). T3K (8 devices) is the minimum viable
# and only validated mesh, matching TTTv1/PERF.md which publish this checkpoint T3K-only. Consequence: no
# single-device config can run this model, so every ci-b1-DP factor (each DP group is a single device)
# cleanly skips — a genuine hardware-capacity guard, not a masked failure.
_MIN_TP_DEVICES = 8


def _skip_below_min_tp_devices(n_devices: int) -> None:
    """Skip when fewer than ``_MIN_TP_DEVICES`` devices are available for tensor parallelism."""
    if n_devices < _MIN_TP_DEVICES:
        pytest.skip(
            f"Qwen2.5-72B requires >={_MIN_TP_DEVICES}-device tensor parallelism: the 72B weights "
            f"+ KV cache need 8-way sharding to fit. TTTv1/PERF.md publish this checkpoint T3K-only. Have "
            f"{n_devices} device(s) — use MESH_DEVICE=T3K."
        )


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
            f"Unsupported MESH_DEVICE={env!r} for Qwen2.5-72B-Instruct; "
            f"only T3K is supported (64 attn heads / 8 KV heads ⇒ 8 devices).",
            allow_module_level=True,
        )
    param = {
        "mesh_shape": shape,
        # 80-layer 72B + 152k vocab + the seq=1024 batched-prefill trace (eval-32's numeric prompts
        # bucket to 1024) needs >50 MB: eval-32 ON/odt measured 53.2 MB of trace buffers. 70 MB gives
        # headroom for the on-device-sampling trace too; +20 MB/device DRAM is negligible vs the ~9-10 GB
        # of sharded 72B weights. (The 70B-Llama sibling fits in 50 MB — smaller vocab + unpadded FF.)
        "trace_region_size": 70_000_000,
        "num_command_queues": 1,
    }
    # The model resolves T3K collectives to Ring topology, so the fabric config must match that topology.
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
    """Real mesh for this file; shape is fixed by ``MESH_DEVICE`` (see ``pytestmark``)."""
    return ttnn_mesh_device


def _skip_unless_heads_divide_mesh(mesh_device: ttnn.MeshDevice, hf_model_id: str) -> None:
    """Attention1D TP requires n_heads and n_kv_heads divisible by device count."""
    n_dev = mesh_device.get_num_devices()
    if 64 % n_dev == 0 and 8 % n_dev == 0:
        return
    pytest.skip(
        f"Incompatible mesh for {hf_model_id}: {n_dev} devices need "
        f"num_attention_heads (64) and num_key_value_heads (8) each divisible by {n_dev}."
    )


def get_device_name(mesh_device):
    """Map mesh device count to a metrics bucket (T3K is the only supported SKU)."""
    num_devices = mesh_device.get_num_devices()
    if num_devices == 8:
        return "T3K"
    return f"{num_devices}dev"


def lazy_weight_cache_dir_for_demo(mesh_device: ttnn.MeshDevice, hf_model_id: str) -> Path:
    """Disk root for ``Qwen25_72B`` ``LazyWeight`` caches in this e2e demo.

    Matches ``models/tt_transformers/tt/model_config.py`` (HF checkpoint branch): if ``TT_CACHE_PATH``
    is set, use ``<TT_CACHE_PATH>/<device_name>``; otherwise ``model_cache/<HF_MODEL>/<device_name>``.
    Persistent cache materially reduces re-run cost for 80-layer 72B weight materialization.
    """
    device_name = get_device_name(mesh_device)
    hf = hf_model_id.strip("/")
    tt_cache = os.getenv("TT_CACHE_PATH")
    if tt_cache:
        root = Path(tt_cache) / device_name
    else:
        root = Path("model_cache") / hf / device_name
    root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Qwen2.5-72B demo LazyWeight cache directory: {root.resolve()}")
    return root


def ref_basename_for_hf(hf_model_id: str) -> str:
    """Match ``ModelArgs.model_name`` style used for ``.refpt`` filenames."""
    return hf_model_id.strip("/").split("/")[-1]


def _load_tokenizer(hf_model_id: str):
    return load_tokenizer(hf_model_id)


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

    Each prompt is encoded with the chat template at its real length. The returned ``[batch, max_len]``
    token tensor is right-padded to the batch-max for rectangularity, while the returned per-user
    lengths are the *real* token counts — the executor reads only ``tokens[user, :prompt_len]`` and then
    buckets each user to ``get_padded_prefill_len`` (128 / 1024 / next-pow2). This matches TTTv1 exactly
    (no fixed pad-to-N prefill budget) and is what lets equal-length users share a batched-prefill group.

    ``max_prefill_len`` is an optional clip *cap* (like TTTv1's ``max_prefill_len``): prompts longer
    than it are left-clipped to their most recent tokens. It is never a pad-up target.
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
):
    """Build ``Qwen25_72B`` in executor (paged KV) mode on T3K.

    Picks one of the two module-level precision recipes (``QWEN25_72B_ACCURACY`` /
    ``QWEN25_72B_PERFORMANCE``) — both defined in ``qwen25_72b/model.py`` and grounded in
    TTTv1's ``DecodersPrecision`` for Qwen2.5-72B. The dataclass owns the dtype + math-fidelity
    recipe; this demo just selects between the two and forwards it.

    ``max_batch_size`` must match the workload: decode DRAM matmul CB usage scales with tile-padded
    batch rows, so batch-1 perf tests should pass ``max_batch_size=1`` even when batch-32 / eval-32 /
    teacher-forcing cases need 32.

    ``max_seq_len`` overrides the default. Default (``None``): ``min(131072 // max_batch_size, 4096)``.
    The ``batch-32-ci`` leg passes an explicit value (see ``_BATCH32_CI_MAX_SEQ_LEN``).
    """
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    _skip_below_min_tp_devices(mesh_device.get_num_devices())
    _skip_unless_heads_divide_mesh(mesh_device, hf_model)

    precision = QWEN25_72B_PERFORMANCE if optimizations == "performance" else QWEN25_72B_ACCURACY

    if max_seq_len is None:
        # T3K: 80 layers × 8 KV heads / 8 dev × head_dim 128 → KV per device per layer is modest.
        # 4096 covers batch-1 (seq4096) and the teacher-forcing refpt; batch-32(-ci) pass explicit values.
        max_seq_len = min(131072 // max_batch_size, 4096)

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
    model: Qwen25_72B,
    *,
    traced: bool,
    device_sampling_enabled: bool,
    trace_mode=None,
) -> Qwen25_72BExecutor:
    block_size = 32
    max_num_blocks = ((model.config.max_seq_len + block_size - 1) // block_size) * model.config.max_batch_size
    attention_config = model.config.block_configs[0].attention_config
    if trace_mode is None:
        trace_mode = "all" if traced else "none"
    return Qwen25_72BExecutor(
        model,
        model.model_args,
        Qwen25_72BExecutorConfig(
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
            execution=prefill_compile_execution if prefill_compile_execution is not None else executor.eager_execution,
        )
    if config.trace.prefill_enabled:
        executor.warmup_model_prefill(enable_trace=True, **prefill_kwargs)
    if config.trace.decode_enabled:
        executor.warmup_model_decode(enable_trace=True, **decode_kwargs)


# =============================================================================
# ci-b1-DP: single-user data-parallel scaling smoke (TTTv1 ci-b1-DP-* parity)
# =============================================================================
#
# One user per DP group, model replicated across ``data_parallel`` disjoint submeshes, instruct
# prompts, paged attention, trace on. The ONLY correctness check is the special-token garbage guard
# plus "runs to completion without hang/exception". This is a mesh / KV-cache / page-table scaling
# smoke, NOT an accuracy or perf gate.
#
# Per-case size table (TTTv1 simple_text_demo.py parity):
#   ci-b1-DP-2  : max_seq_len=1024, max_generated_tokens=200,  stop_at_eos=True
#   ci-b1-DP-4  : max_seq_len=4096, max_generated_tokens=2048, stop_at_eos=False
#   ci-b1-DP-8  : max_seq_len=4096, max_generated_tokens=2048, stop_at_eos=False
#   ci-b1-DP-16 : max_seq_len=1024, max_generated_tokens=200,  stop_at_eos=True
#   ci-b1-DP-32 : max_seq_len=1024, max_generated_tokens=200,  stop_at_eos=True
#
# Hardware feasibility: each DP group is one device (batch_size=1 per group), so
# ``data_parallel == n_devices``. Qwen2.5-72B needs 8-way TP (a single device cannot hold the
# 72B), so EVERY DP factor is inapplicable: you cannot have both 1-device-per-user AND 8-device TP. All
# factors cleanly ``pytest.skip`` (genuine hardware-capacity guard, matching TTTv1's T3K-only support).
# The case ids are present for parity with TTTv1 ``simple_text_demo.py``.
_DP_SIZE_TABLE: dict[int, dict] = {
    2: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
    4: {"max_seq_len": 4096, "max_generated_tokens": 2048, "stop_at_eos": False},
    8: {"max_seq_len": 4096, "max_generated_tokens": 2048, "stop_at_eos": False},
    16: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
    32: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
}


def create_dp_submeshes(mesh_device: ttnn.MeshDevice, data_parallel: int) -> list:
    """Partition the open parent mesh into ``data_parallel`` disjoint row-submeshes.

    Mirrors TTTv1 ``generator.create_submeshes`` minus the Galaxy reshape branch (no Galaxy reachable
    here). For the single-user DP cases ``n // data_parallel == 1``, so each submesh is a ``(1,1)``
    mesh. Fabric stays owned by the parent — do NOT set fabric per-submesh.
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
    if n // data_parallel < _MIN_TP_DEVICES:
        pytest.skip(f"DP-{data_parallel} cannot provide the {_MIN_TP_DEVICES}-device TP group required by Qwen2.5-72B")


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

    Builds one model + one traced executor + one KV cache + one page table per submesh (one user each),
    runs ``run_perf_benchmark`` per submesh sequentially, collects the per-submesh output, and asserts
    no special tokens. Every executor and model is cleaned up in ``finally``.
    """
    _dp_or_skip(mesh_device, data_parallel)
    # Each DP group is a single device (see _dp_or_skip: n // data_parallel == 1). Qwen2.5-72B
    # cannot run on a single device (needs 8-way TP — see _skip_below_min_tp_devices), so every DP factor
    # is inapplicable for this model: you cannot have both 1-device-per-user AND 8-device TP. Genuine
    # hardware-capacity guard (matches TTTv1's T3K-only support — TTTv1 can't DP a 72B on T3K either).
    _skip_below_min_tp_devices(mesh_device.get_num_devices() // data_parallel)

    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    _skip_unless_heads_divide_mesh(mesh_device, hf_model)
    tokenizer = _load_tokenizer(hf_model)
    precision = QWEN25_72B_PERFORMANCE if optimizations == "performance" else QWEN25_72B_ACCURACY

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
                llm = from_pretrained(
                    sm,
                    hf_model=hf_model,
                    max_batch_size=1,
                    max_seq_len=max_seq_len,
                    n_layers=None,
                    cache_dir=cache_dir,
                    optimizations=precision,
                )
            except Exception as e:
                pytest.skip(f"Could not build Qwen2.5-72B model (weights / memory / mesh): {e}")
            model = llm.model
            models.append((model, sm))

            traced_executor = create_executor(
                model,
                traced=True,
                device_sampling_enabled=True,
            )
            executors.append(traced_executor)

            ma = model.model_args
            assert ma is not None

            kv_cache = traced_executor.allocate_kv_cache()
            page_table = make_contiguous_page_table(ma.max_batch_size, ma.max_seq_len, 32)

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

        assert_no_special_tokens(all_generated, tokenizer, case_name=f"ci-b1-DP-{data_parallel}")
    finally:
        for ex in executors:
            ex.cleanup()
        for model, sm in models:
            cleanup_model_case(model, sm)
        # When data_parallel > 1 we carved child submeshes off the fixture-owned parent mesh. Those
        # submeshes share the parent's command queue, so the parent cannot be closed while they remain
        # in use. Drain the parent + submesh CQs before teardown.
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
def test_qwen25_72b(test_config, mesh_device, optimizations):
    """Main test entry for TTTv2 Qwen2.5-72B-Instruct."""
    device_name = get_device_name(mesh_device)
    expected = EXPECTED_METRICS.get(optimizations, {}).get(device_name, {})
    model = None
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    cache_dir = lazy_weight_cache_dir_for_demo(mesh_device, hf_model)

    try:
        # ci-b1-DP-*: single-user data-parallel smoke. Builds N models itself (one per submesh), so it
        # does NOT go through the shared create_model path below.
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

        if test_config in ("batch-32", "eval-32"):
            # Short-context 32-user workload (seq1024). batch-32 is perf-gated; eval-32 is a determinism
            # check (not perf-gated).
            max_bs, max_seq_len = 32, 1024
            expected = EXPECTED_METRICS_BATCH32.get(_sampling_bucket(), {}).get(optimizations, {}).get(device_name, {})
        elif test_config == "batch-32-ci":
            # CI-faithful batch-32 leg (TTTv1 ci-32 parity): larger seq len + 1024 decode budget.
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
            # token-accuracy + batch-1: single-user, seq4096.
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
            # Natural-length prefill: these sample prompts bucket to 128 (PERF.md Short-Context Batch-32
            # row), matching TTTv1's traced-prefill seq len without a forced pad.
            _run_perf_benchmark(model, mesh_device, expected, batch_size=32, case_name=f"{optimizations}/batch-32")
        elif test_config == "batch-32-ci":
            # CI-faithful leg: seq2048 + 1024 decode tokens (clamped in _run_perf_benchmark). Gated by
            # EXPECTED_METRICS_BATCH32_CI (measured at this workload, TTTv1-parity).
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


def _run_token_accuracy(model, mesh_device, expected):
    """Teacher-forcing token accuracy vs ``.refpt`` (HF-generated)."""
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")
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

    executor = create_executor(model, traced=False, device_sampling_enabled=False)
    ma = model.model_args
    assert ma is not None

    max_batch_size = ma.max_batch_size
    prompt_tokens = prompt_tokens.repeat(max_batch_size, 1)
    max_seq_len = ma.max_seq_len
    kv_cache = executor.allocate_kv_cache()
    page_table = make_contiguous_page_table(max_batch_size, max_seq_len, 32)

    target_top5 = select_teacher_forcing_top5_slice(
        top5_tokens,
        reference_tokens,
        prompt_len,
        metadata_aligned=has_prompt_len_metadata,
    )
    is_ci_env = os.environ.get("CI") == "true"
    profiler = BenchmarkProfiler()
    profiler.start("run")
    # run_teacher_forcing times the prefill + per-step (teacher-forced) decode loop and, given the
    # profiler, brackets the "inference_prefill" / "inference_decode" steps itself — so the returned
    # result carries prefill/decode throughput alongside accuracy for CI benchmark-data emission.
    try:
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
            num_layers=ma.n_layers,
            batch_size=1,
            input_sequence_length=prompt_len,
            output_sequence_length=num_target,
        )

    # Accuracy gate — threshold SOURCE is flag-controlled (currently ``is_ci_env``):
    #   use_centralized_targets = True  → mirror TTTv1: centralized targets via resolve_accuracy_targets
    #       minus an ABSOLUTE 0.5 pp (get_accuracy_thresholds, simple_text_demo.py). A missing entry is
    #       a hard error (never silently un-gate in CI).
    #   use_centralized_targets = False → the demo's local EXPECTED_METRICS values DIRECTLY (no ratio
    #       tolerance — TTTv1 applies none to accuracy).
    # Measured accuracy is rounded up with math.ceil before the compare, matching TTTv1 exactly
    # (simple_text_demo.py, ``math.ceil(acc[...] * 100)``).
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
    model,
    mesh_device,
    expected,
    batch_size,
    case_name,
    max_prefill_len: int | None = None,
    num_decode_tokens: int | None = None,
):
    """Timed prefill + decode with the traced model-owned executor.

    Prefill uses each prompt's natural token length (TTTv1 ``preprocess_inputs_prefill`` semantics — the
    executor buckets to ``get_padded_prefill_len``); decode runs for ``num_decode_tokens`` steps
    (default ``_PERF_NUM_DECODE_TOKENS``). ``max_prefill_len`` is an optional clip cap for over-long
    prompts, never a pad-up target.

    The decode budget is clamped to what the paged KV cache can hold:
    ``effective = min(requested, max_seq_len - prompt_bucket - margin)`` so the high-water decode
    position never overruns the page table (the ``batch-32-ci`` leg requests 1024).
    """
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    tokenizer = _load_tokenizer(hf_model)

    # On-device sampling toggle (see the rebase / sampling handoff docs):
    #   host            -> sampling_params=None (host-argmax; slow — full-vocab all-gather + PCIe
    #                      readback every step; NOT comparable to TTTv1)
    #   on_device       -> greedy temp=0,k=1,p=0 => trace-captured FORCE-ARGMAX full-vocab path
    #   on_device_topk  -> temp=0,k=32,p=0.08    => trace-captured TOP-K op path (gathers only the
    #                      [*,32] tuples; PERF.md-parity recipe, faster on >=8-dev meshes)
    # DEFAULT is on_device_topk: on T3K (8 devices) the vocab shards 8-ways and TTTv1 auto-uses on-device
    # sampling, so this is the apples-to-apples TTTv1-comparable path the gate measures.
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

    # Batched-prefill A/B knob (parity caveat #12): set DISABLE_BATCHED_PREFILL=1 to force the
    # sequential per-user prefill loop (the pre-feature baseline) for before/after TTFT comparison.
    if os.environ.get("DISABLE_BATCHED_PREFILL") and model.model_args is not None:
        model.model_args.disable_batched_prefill = True

    # Free-running perf run: enable the executor's on-device decode loop on the on-device sampling path
    # (inert on host / force-argmax; gated to the top-k path by _decode_loop_active). This is the #49282
    # T3K decode-gap fix (shared engine #49284) — it must be active on the perf path for the T3K gate.
    traced_executor = create_executor(
        model,
        traced=True,
        device_sampling_enabled=sampling_params is not None,
    )
    try:
        ma = model.model_args
        assert ma is not None

        max_seq_len = ma.max_seq_len
        max_batch_size = ma.max_batch_size
        kv_cache = traced_executor.allocate_kv_cache()
        page_table = make_contiguous_page_table(max_batch_size, max_seq_len, 32)

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
        prefill_sampling_params = None
        _warmup_demo_executor(
            traced_executor,
            kv_cache=kv_cache,
            page_table=page_table,
            prefill_compile_case=(input_tokens, prompt_lens),
            prefill_sampling_params=prefill_sampling_params,
            prefill_compile_execution=traced_executor.traced_prefill_execution,
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
                num_layers=ma.n_layers,
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
    undoing the rotation lines up per-user outputs. No external golden. Honors the same ``SAMPLING_MODE``
    knob as ``_run_perf_benchmark`` (default host argmax — deterministic and mesh-agnostic, the
    recommended default for the determinism assert).

    Use the default (host argmax) for the determinism gate. Under ``SAMPLING_MODE=on_device_topk`` the
    accuracy profile's degenerate numeric-prompt continuations can produce near-exact logit ties, and
    the on-device sampler's tie-break is slot-dependent (reduction order over the sharded vocab) → the
    cross-batch consistency assert can flip on those rotated slots. That is a property of on-device
    top-k sampling on tie-heavy degenerate output, NOT a determinism regression: host argmax passes both
    profiles with batched prefill ON and OFF, and any on-device flip is identical ON vs OFF
    (prefill-independent, so unrelated to batched prefill).
    """
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    tokenizer = _load_tokenizer(hf_model)

    # Qwen2.5 chat generation ends at <|im_end|>; the model opening a NEW turn (<|im_start|>) is a
    # de-facto response terminator as well (Qwen serving stacks list both as stops), but Qwen's HF
    # generation_config only carries <|im_end|>/<|endoftext|> as eos. Augment the tokenizer stop set (the
    # mechanism ``hf_stop_ids`` reads) with <|im_start|> so the determinism runner truncates a degenerate
    # turn-restart there — same pattern as the qwen25_7b / qwen3_32b guards. Without this, a fixed-budget
    # greedy continuation of the numeric eval prompts can degenerate into "\n<|im_start|>user" (a
    # hallucinated new turn) deep in decode; which of the two equally-valid prefill numerics (batched vs
    # sequential) hits it is a near-tie, so the shared garbage guard would otherwise flag only one leg.
    # <|im_start|> is a legitimate response terminator, so truncating there is correct, not a loosening;
    # cross-batch consistency is still asserted on the truncated (real-response) tokens.
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    if isinstance(im_start_id, int) and im_start_id >= 0:
        existing = list(getattr(tokenizer, "stop_tokens", None) or [])
        tokenizer.stop_tokens = list({*existing, im_start_id})

    ma = model.model_args
    assert ma is not None

    # Batched-prefill A/B knob (parity caveat #12): DISABLE_BATCHED_PREFILL=1 forces the pure per-bucket
    # sequential prefill so eval-32 can be validated both ON and OFF.
    if os.environ.get("DISABLE_BATCHED_PREFILL"):
        ma.disable_batched_prefill = True

    max_seq_len = ma.max_seq_len
    max_batch_size = ma.max_batch_size
    page_table = make_contiguous_page_table(max_batch_size, max_seq_len, 32)

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
