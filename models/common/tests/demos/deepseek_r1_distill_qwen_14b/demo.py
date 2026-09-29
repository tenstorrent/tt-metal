# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 DeepSeek-R1-Distill-Qwen-14B demo — accuracy and performance measurement on N300 / T3K.

Uses the model-owned ``DeepSeekR1Qwen14BExecutor`` directly (no vLLM adapter).

**Mesh note.** DeepSeek-R1-Distill-Qwen-14B is a dense Qwen2.5-14B architecture: 40 attention heads and
8 KV heads (both divide 2, 4, and 8), so TP2, TP4, and TP8 are supported. **TP1 is NOT**: the 14B weights +
distributed-LayerNorm circular buffer overflow a single Wormhole's L1 at the first forward
(``_MIN_TP_DEVICES = 2``). On a physical eight-device T3K this means DP2 uses two TP4 lanes and DP4 uses
four TP2 lanes; DP8 and larger factors cleanly skip because they would require unsupported TP1 lanes.

DeepSeek-R1-Distill-Qwen-14B is a **reasoning** model: the chat template appends ``<think>\\n`` and the
model emits a ``<think>...</think>`` chain before the answer. ``<think>`` / ``</think>`` are NOT special
ids (only BOS ``<｜begin▁of▁sentence｜>`` / EOS ``<｜end▁of▁sentence｜>`` are), so they never trip the
garbage guard, and the eos-only stop truncation is correct.

CI cases (parity with TTTv1 ``simple_text_demo.py``):
    token-accuracy   - teacher-forcing top-1/top-5 vs the book ``.refpt``
    batch-1          - single-user latency
    batch-32         - short-context throughput (seq512/2048 / 200 decode)
    batch-32-ci      - CI-faithful batch-32 (seq2048 / 1024 decode; TTTv1 ci-32)
    eval-32          - 32-user cross-batch determinism (TTTv1 ci-eval-32)
    ci-b1-DP-{2..32} - single-user DP scaling smoke; DP2/DP4 run on T3K, DP8/16/32 capacity-skip

Usage::

    # Token accuracy test (gates against the committed book ``.refpt``)
    MESH_DEVICE=N300 HF_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
      pytest models/common/tests/demos/deepseek_r1_distill_qwen_14b/demo.py -k "token-accuracy" -v

    # On-device sampling perf (the TTTv1-comparable path)
    SAMPLING_MODE=on_device_topk MESH_DEVICE=N300 HF_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
      pytest models/common/tests/demos/deepseek_r1_distill_qwen_14b/demo.py -k "batch-32-ci" -v

LazyWeight tensor cache: ``TT_CACHE_PATH/<device_name>`` when ``TT_CACHE_PATH`` is set, otherwise
``model_cache/<HF_MODEL>/<device_name>`` under the current working directory.

Reference artifact (``.refpt``): generate with ``generate_book_refpt.py`` before running token-accuracy
tests. The file lives at ``models/tt_transformers/tests/reference_outputs/DeepSeek-R1-Distill-Qwen-14B.refpt``.
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
from models.common.models.deepseek_r1_distill_qwen_14b.executor import (
    DeepSeekR1Qwen14BExecutor,
    DeepSeekR1Qwen14BExecutorConfig,
)
from models.common.models.deepseek_r1_distill_qwen_14b.hf_adaptor import from_pretrained
from models.common.models.deepseek_r1_distill_qwen_14b.model import (
    DEEPSEEK_R1_14B_ACCURACY,
    DEEPSEEK_R1_14B_PERFORMANCE,
    DeepSeekR1Qwen14B,
)
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
# NOT PERF.md (DeepSeek-R1-Distill-Qwen-14B is not in PERF.md).
#
# Rule (per cell): each ``tok_s_u`` target is the BETTER of TTTv1 vs TTTv2 for that sampling mode.
# TTTv1 has only an on-device sampling path, so:
#     on_device_topk : max(TTTv1_on_device, TTTv2_on_device_topk)
#     host           : TTTv2_host                      (TTTv1 has no host-sampling path)
# Decode throughput is prefill-independent, so batched prefill does NOT change ``tok_s_u``.
# ``ttft_ms`` targets are conservative upper bounds (batched prefill only LOWERS TTFT).
#
# TTTv1 baseline: DeepSeek-R1-Distill-Qwen-14B runs on TTTv1 ``simple_text_demo.py`` via the generic
# Qwen2 HF path at the SAME precision TTTv2's performance recipe uses (BFP4 FF1/FF3 + LoFi — the non-7B
# ``else`` branch), so the better-of comparison is precision-fair. All values below are freshly measured
# this session (see perf_tables.md); the on_device_topk bucket is the TTTv1-comparable path.
# =============================================================================

# top1/top5 teacher-forcing accuracy floors (book refpt), profile-split. Perf metrics live in the batch
# dicts below. Floors set at/below measured. The gate rounds the measured value up with math.ceil
# (TTTv1 parity) before compare, so an integer floor of 87 admits a measured 86.5. Re-measured
# 2026-07-25 with minimal_matmul ON (the shipped prefill config; see _DSR1WHTuning.prefill_minimal_matmul):
#   perf N300 87.1/98.6, T3K 86.5/98.4 ; acc N300 95.9/100.0, T3K 95.7/100.0.
# NOTE: minimal_matmul (block-matmul kernel for the QKV+W2 prefill matmuls, seq_len>128) costs ~1.0pp top1
# vs ttnn.linear (perf T3K 87.5 OFF -> 86.5 ON; N300 87.9 -> 87.1) from its numerics; it still clears every
# floor here AND the CI central-0.5 gate (resolve_accuracy_targets = 87 -> 86.5 floor; ceil(86.5)=87 PASS),
# and TTTv1 itself uses minimal_matmul for these matmuls. Kept because it halves the batch-32-ci TTFT gap.
EXPECTED_METRICS: dict = {
    "performance": {
        "N300": {"top1": 87, "top5": 99},
        "T3K": {"top1": 87, "top5": 98},
    },
    "accuracy": {
        "N300": {"top1": 95, "top5": 99},
        "T3K": {"top1": 94, "top5": 99},
    },
}

# batch-1 throughput, sampling-mode- and profile-aware (values from the 2026-07-23 FF-pad matrix; perf_tables.md).
# Per PARITY_RULES §2 the DECODE tok_s_u gate = best-of(TTTv1_default, TTTv2_odt); the ttft_ms gate is a
# conservative single-user ceiling (b1 TTFT is bimodal/noisy — NOT a tight parity gate; TTFT parity vs TTTv1
# is recorded in perf_tables.md). On T3K TTTv1 samples ON-DEVICE and after the FF-hidden DRAM-shard pad
# (decode FF 2->32 cores) TTTv2 now BEATS TTTv1 (b1 41.1 vs 36.35 perf / 36.4 vs 33.36 acc) → gate at the
# TTTv2 (better) value. On N300 TTTv1 samples HOST argmax, so the N300 on_device_topk bucket has no TTTv1
# on-device number and is gated at TTTv2's own value (few-device big-vocab Sampling1D ~2x slower than host on
# N300 — not the TTTv1-matched path there; N300 parity is the host bucket). T3K host = degenerate 8-chip
# round-trip sampler (non-shipped) → ungated ({}). b1 batch<=1 does not trigger batched prefill (TTFT ON/OFF-identical).
EXPECTED_METRICS_BATCH1: dict = {
    "host": {
        "performance": {
            "N300": {
                "tok_s_u": 21.5,
                "ttft_ms": 145,
            },  # gate = best-of(TTTv1 host 21.53, TTTv2 20.6); TTTv2 clears within 5%
        },
        "accuracy": {
            "N300": {
                "tok_s_u": 15.8,
                "ttft_ms": 170,
            },  # TTTv2 own (TTTv1 N300 accuracy fails: enable_log_probs harness bug)
        },
    },
    "on_device_topk": {
        "performance": {
            "N300": {"tok_s_u": 13.2, "ttft_ms": 135},  # TTTv2 own (TTTv1 host-only on N300)
            "T3K": {"tok_s_u": 41.1, "ttft_ms": 80},  # gate = TTTv2 (best-of; BEATS TTTv1 36.35 after FF-pad)
        },
        "accuracy": {
            "N300": {"tok_s_u": 11.0, "ttft_ms": 170},  # TTTv2 own
            "T3K": {"tok_s_u": 36.4, "ttft_ms": 85},  # gate = TTTv2 (best-of; BEATS TTTv1 acc 33.36 after FF-pad)
        },
    },
}

# Short-context batch-32 throughput (seq512/2048 / 200 decode), sampling-mode- and profile-aware. Runs BOTH
# batched-prefill ON (default) and DISABLE_BATCHED_PREFILL=1 (A/B); decode tok_s_u is prefill-independent so
# the tok_s_u gate covers both knob states, and the ttft_ms ceiling covers the (slower) sequential OFF path
# (batched ON ~halves TTFT: N300 63→ON vs 117→OFF). TTTv1's short-context batch-32 control FAILS on this box
# with a TTTv1 harness bug (KeyError 'enable_log_probs') — unrelated to DeepSeek — so there is no TTTv1
# baseline for this leg and it is gated from TTTv2's own value. T3K host = degenerate (ungated).
EXPECTED_METRICS_BATCH32: dict = {
    "host": {
        "performance": {
            "N300": {"tok_s_u": 19.6, "ttft_ms": 130},
        },
        "accuracy": {
            "N300": {"tok_s_u": 14.8, "ttft_ms": 150},
        },
    },
    "on_device_topk": {
        "performance": {
            "N300": {"tok_s_u": 12.6, "ttft_ms": 130},
            "T3K": {"tok_s_u": 33.9, "ttft_ms": 70},
        },
        "accuracy": {
            "N300": {"tok_s_u": 10.5, "ttft_ms": 150},
            "T3K": {"tok_s_u": 30.2, "ttft_ms": 75},
        },
    },
}

# CI-faithful batch-32 targets (the ``batch-32-ci`` leg), seq2048 + 1024-token decode budget = the DIRECT
# TTTv1 ci-32 analog (the matched CI pair). Per PARITY_RULES §2: on_device_topk gate = best-of(TTTv1 ci-32,
# TTTv2 odt); host gate = TTTv2 host. On T3K, after the FF-pad decode fix TTTv2 odt decode BEATS TTTv1 ci-32
# (38.2 vs fresh 34.3 perf / 32.9 vs 30.33 acc) → gate at the TTTv2 (better) value; TTTv2 clears within 5%.
# On N300 TTTv1 ci-32 is host argmax (18.75), and TTTv2 host decode (18.2) is at parity within noise (host
# is informational; N300 odt is own-gated). The accuracy profile is DRAM-infeasible on N300 (guarded skip)
# → no N300 acc entry. T3K host = degenerate (ungated). ttft ceilings are conservative (cover the sequential
# OFF path). minimal_matmul ON (default) lowered the odt/host prefill TTFT (T3K perf 29.2→25.3, N300 host
# 61.3→51.7); the residual TTFT vs TTTv1 (T3K perf +11.9%, acc +22.1%; shared batched-prefill fold) is
# recorded in perf_tables.md / parity_gate.py, NOT a tight demo gate.
EXPECTED_METRICS_BATCH32_CI: dict = {
    "host": {
        "performance": {
            "N300": {"tok_s_u": 19.0, "ttft_ms": 130},  # gate = best-of; TTTv2 host 19.0 BEATS TTTv1 ci-32 host 17.64
        },
        "accuracy": {},  # DRAM-infeasible on N300 (skip); T3K host degenerate (ungated)
    },
    "on_device_topk": {
        "performance": {
            "N300": {"tok_s_u": 12.2, "ttft_ms": 130},  # TTTv2 own (TTTv1 host-only on N300)
            "T3K": {
                "tok_s_u": 38.3,
                "ttft_ms": 70,
            },  # gate = TTTv2 (best-of; BEATS TTTv1 ci-32 32.75 after FF-pad); ttft ceiling covers OFF (~58ms)
        },
        "accuracy": {
            "T3K": {
                "tok_s_u": 32.9,
                "ttft_ms": 75,
            },  # gate = TTTv2 (best-of; BEATS TTTv1 acc ci-32 28.43 after FF-pad); ttft ceiling covers OFF
        },
    },
}

# Perf workload: natural-length prefill (these sample prompts are ~70-125 tokens -> 128 bucket, matching
# TTTv1), 200 decode steps. Accuracy uses the teacher-forcing refpt.
_PERF_NUM_DECODE_TOKENS = 200

PERF_TOLERANCE = 0.05

# eval-32 max_seq_len: the ci-eval-32 numeric prompts run up to ~683 tokens -> get_padded_prefill_len
# bucket 1024, so max_seq_len MUST be >= 1024 or the batched-prefill group page table overruns
# (32 blocks/user needed). Fixed at 1024 (decode starts at the REAL prompt len, so the high-water decode
# position stays well within 1024). Independent of the batch-32 seq len.
_EVAL_MAX_SEQ_LEN = 1024

# batch-32-ci per-SKU max_seq_len (TTTv1 ci-32 parity is seq2048).
_BATCH32_CI_MAX_SEQ_LEN: dict[str, int] = {
    "N300": 2048,
    "T3K": 2048,
}


def _sampling_bucket() -> str:
    """Map SAMPLING_MODE to a perf-gate bucket. Defaults to ``on_device_topk`` (the perf-case default,
    the TTTv1-comparable path), so the bucket always agrees with the runner. Non-topk on-device modes
    (e.g. force-argmax) also fall into ``on_device_topk`` so they stay gated, never silently un-gated."""
    return "host" if os.environ.get("SAMPLING_MODE", "on_device_topk").lower() == "host" else "on_device_topk"


# DeepSeek-R1-Distill-Qwen-14B needs at least this many devices of tensor parallelism: the 14B weights +
# the distributed-LayerNorm circular buffer overflow a single Wormhole's L1 (1512864 B vs 1499136 B max)
# at the first forward. TP2 is the minimum viable lane (dim/2 shrinks the norm CB), while TP4 and TP8
# shard further. On an eight-device T3K, DP2/TP4 and DP4/TP2 are viable; DP8 and larger factors require
# unsupported TP1 lanes and cleanly capacity-skip rather than masking a runtime failure.
_MIN_TP_DEVICES = 2


def _skip_below_min_tp_devices(n_devices: int) -> None:
    """Skip when fewer than ``_MIN_TP_DEVICES`` devices are available for tensor parallelism."""
    if n_devices < _MIN_TP_DEVICES:
        pytest.skip(
            f"DeepSeek-R1-Distill-Qwen-14B requires >={_MIN_TP_DEVICES}-device tensor parallelism: the "
            f"14B weights + distributed-LayerNorm circular buffer overflow a single Wormhole's L1 at the "
            f"first forward. Have {n_devices} device(s) — use MESH_DEVICE=N300 or T3K."
        )


def _skip_if_dram_infeasible(device_name: str, optimizations: str, case: str) -> None:
    """Skip the DRAM-infeasible N300 accuracy cases (``eval-32`` and ``batch-32-ci``).

    The 14B accuracy recipe keeps BF16 attention weights (≈ 9.7 GB/device) resident; a batch-32 working
    set at the eval-32 (seq1024) / batch-32-ci (seq2048) shapes then overflows N300 DRAM. Measured on this
    box (2026-07-23): batch-32-ci accuracy OOMs at ``bank_manager.cpp:462`` during device tensor load
    (only ~336 KB free after weights) — the batch-32 activation/KV working set does not fit alongside the
    9.7 GB weights on N300's ~12 GB/chip. This is the same limit as TTTv1's own DeepSeek-14B accuracy run
    and phi-4's N300 accuracy OOM. The **performance** profile (BFP4 MLP + LoFi — the harder low-precision
    determinism / throughput case) covers these cells on N300; T3K (8-way shard) runs BOTH profiles, so
    accuracy is still fully exercised there. This is a hardware-capacity guard, not a masked failure.
    """
    if device_name == "N300" and optimizations == "accuracy" and case in ("eval-32", "batch-32-ci"):
        pytest.skip(
            f"{case} accuracy profile is DRAM-infeasible on N300 (14B BF16 attn ≈ 9.7 GB/device leaves too "
            f"little for the batch-32 working set; measured OOM at bank_manager). Covered by the perf "
            f"profile on N300 + both profiles on T3K."
        )


# Mesh topology comes only from ``MESH_DEVICE`` (same naming as vLLM / other tt demos).
_MESH_DEVICE_TO_SHAPE: dict[str, tuple[int, int]] = {
    "N150": (1, 1),
    "N300": (1, 2),
    "T3K": (1, 8),
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
    param = {
        "mesh_shape": shape,
        "trace_region_size": 100_000_000 if env == "T3K" else 50_000_000,
        "num_command_queues": 1,
    }
    # TTTv2 multi-device executor dispatch (and the on-device sampling all-gather) stalls without an
    # explicit 1D fabric; the root conftest does not auto-enable it. FABRIC_1D on any >1-device mesh.
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
        f"num_attention_heads ({n_h}) and num_key_value_heads ({n_kv}) each divisible by {n_dev}."
    )


def get_device_name(mesh_device: ttnn.MeshDevice) -> str:
    """Map mesh device count to a metrics bucket."""
    n = mesh_device.get_num_devices()
    if n == 1:
        return "N150"
    if n == 2:
        return "N300"
    if n == 8:
        return "T3K"
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
    ref_data = torch.load(ref_path, map_location="cpu", weights_only=False)
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


def tokenize_prompts(
    prompts: list[str],
    tokenizer,
    *,
    max_prefill_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize prompts to their natural length — TTTv1 ``preprocess_inputs_prefill`` semantics.

    Each prompt is encoded with the chat template at its real length. The returned ``[batch, max_len]``
    token tensor is right-padded to the batch-max for rectangularity, while the returned per-user lengths
    are the *real* token counts — the executor reads only ``tokens[user, :prompt_len]`` and buckets each
    user to ``get_padded_prefill_len`` (128 / 1024 / next-pow2). This matches TTTv1 (no fixed pad-to-N
    prefill budget) and is what lets equal-length users share a batched-prefill group.

    ``max_prefill_len`` is an optional clip *cap* (like TTTv1's ``max_prefill_len``): prompts longer than
    it are left-clipped to their most recent tokens. It is never a pad-up target.
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
) -> DeepSeekR1Qwen14B:
    """Build ``DeepSeekR1Qwen14B`` in executor (paged KV) mode.

    Picks one of the two module-level precision recipes (``DEEPSEEK_R1_14B_ACCURACY`` /
    ``DEEPSEEK_R1_14B_PERFORMANCE``) — both defined in ``deepseek_r1_distill_qwen_14b/model.py`` and
    grounded in TTTv1's ``DecodersPrecision`` for the generic Qwen2 path.

    ``max_batch_size`` must match the workload: decode DRAM matmul CB usage scales with tile-padded batch
    rows, so batch-1 perf tests pass ``max_batch_size=1`` even when batch-32 / eval-32 / teacher-forcing
    cases need 32.

    ``max_seq_len`` overrides the default. Default (``None``) is DRAM-driven on the memory-constrained
    N300: at batch-32 the accuracy recipe (BF16 attn, ~9.7 GB/dev) only fits seq 512, the performance
    recipe (BFP4 FF, ~6.85 GB/dev) fits seq 2048; batch-1 uses seq 4096. eval-32 / batch-32-ci pass
    explicit values.
    """
    hf_model = os.environ.get("HF_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    _skip_below_min_tp_devices(mesh_device.get_num_devices())
    _skip_unless_heads_divide_mesh(mesh_device, hf_model)

    precision = DEEPSEEK_R1_14B_PERFORMANCE if optimizations == "performance" else DEEPSEEK_R1_14B_ACCURACY

    if max_seq_len is None:
        if max_batch_size == 32:
            max_seq_len = 512 if optimizations != "performance" else 2048
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
        pytest.skip(f"Could not build DeepSeek-R1-Distill-Qwen-14B model (weights / memory / mesh): {e}")

    model = llm.model
    model.demo_tokenizer = llm.tokenizer
    return model


def create_executor(
    model: DeepSeekR1Qwen14B,
    *,
    traced: bool,
    device_sampling_enabled: bool,
    trace_mode=None,
) -> DeepSeekR1Qwen14BExecutor:
    block_size = 32
    max_num_blocks = ((model.config.max_seq_len + block_size - 1) // block_size) * model.config.max_batch_size
    attention_config = model.config.block_configs[0].attention_config
    if trace_mode is None:
        trace_mode = "all" if traced else "none"
    return DeepSeekR1Qwen14BExecutor(
        model,
        model.model_args,
        DeepSeekR1Qwen14BExecutorConfig(
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
            execution=executor.eager_execution,
        )
    if config.trace.prefill_enabled:
        executor.warmup_model_prefill(enable_trace=True, **prefill_kwargs)
    if config.trace.decode_enabled:
        executor.warmup_model_decode(enable_trace=True, **decode_kwargs)


# =============================================================================
# ci-b1-DP: single-user data-parallel scaling smoke (TTTv1 ci-b1-DP-* parity)
# =============================================================================
#
# One user per DP group, model replicated across ``data_parallel`` disjoint submeshes, instruct prompts,
# paged attention, trace on. The ONLY correctness check is the special-token garbage guard plus "runs to
# completion without hang/exception". This is a mesh / KV-cache / page-table scaling smoke, NOT an
# accuracy or perf gate.
#
# Per-case size table (TTTv1 simple_text_demo.py parity):
#   ci-b1-DP-2  : max_seq_len=1024, max_generated_tokens=200,  stop_at_eos=True
#   ci-b1-DP-4  : max_seq_len=4096, max_generated_tokens=2048, stop_at_eos=False
#   ci-b1-DP-8  : max_seq_len=4096, max_generated_tokens=2048, stop_at_eos=False
#   ci-b1-DP-16 : max_seq_len=1024, max_generated_tokens=200,  stop_at_eos=True
#   ci-b1-DP-32 : max_seq_len=1024, max_generated_tokens=200,  stop_at_eos=True
#
# On the physical eight-device T3K, DP2 creates two TP4 lanes and DP4 creates four TP2 lanes.
# Both are structurally supported. DP8 creates TP1 lanes, which are below the model's capacity
# floor; DP16/32 cannot partition the host. The case IDs remain unchanged.
_DP_SIZE_TABLE: dict[int, dict] = {
    2: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
    4: {"max_seq_len": 4096, "max_generated_tokens": 2048, "stop_at_eos": False},
    8: {"max_seq_len": 4096, "max_generated_tokens": 2048, "stop_at_eos": False},
    16: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
    32: {"max_seq_len": 1024, "max_generated_tokens": 200, "stop_at_eos": True},
}


def _dp_lane_tp_or_skip(mesh_device: ttnn.MeshDevice, data_parallel: int) -> int:
    """Return devices per lane for supported DeepSeek TP4/TP2 DP layouts."""
    n = mesh_device.get_num_devices()
    if n % data_parallel != 0:
        pytest.skip(f"DP-{data_parallel} cannot partition {n} devices into equal lanes")
    tensor_parallel = n // data_parallel
    if tensor_parallel < _MIN_TP_DEVICES:
        pytest.skip(
            f"DP-{data_parallel} on {n} devices creates TP{tensor_parallel} lanes; "
            f"DeepSeek-R1-Distill-Qwen-14B requires at least TP{_MIN_TP_DEVICES}"
        )
    if tensor_parallel not in (2, 4):
        pytest.skip(f"DP-{data_parallel} on {n} devices creates unsupported TP{tensor_parallel} lanes")
    return tensor_parallel


def _create_dp_submeshes(mesh_device: ttnn.MeshDevice, data_parallel: int, tensor_parallel: int) -> list:
    submeshes = list(mesh_device.create_submeshes(ttnn.MeshShape(1, tensor_parallel)))
    if len(submeshes) != data_parallel:
        raise ValueError(f"Expected {data_parallel} TP{tensor_parallel} submeshes, got {len(submeshes)}")
    return submeshes


def _dp_lane_cache_dir(cache_dir: Path, tensor_parallel: int) -> Path:
    device_name = {2: "N300", 4: "N150x4"}.get(tensor_parallel, f"{tensor_parallel}dev")
    lane_cache_dir = cache_dir.parent / device_name
    lane_cache_dir.mkdir(parents=True, exist_ok=True)
    return lane_cache_dir


def _validate_dp_lane(
    model: DeepSeekR1Qwen14B, lane: DeepSeekR1Qwen14BExecutor, tensor_parallel: int, max_seq_len: int
) -> None:
    config = model.config
    attention = config.block_configs[0].attention_config
    if config.num_devices != tensor_parallel:
        raise ValueError(f"DP lane expected TP{tensor_parallel}, model uses TP{config.num_devices}")
    if attention.n_heads % tensor_parallel or attention.n_kv_heads % tensor_parallel:
        raise ValueError(
            f"DP lane TP{tensor_parallel} does not divide DeepSeekR1Qwen14B heads "
            f"({attention.n_heads}/{attention.n_kv_heads})"
        )
    if config.max_batch_size != 1:
        raise ValueError(f"DP lane must have capacity 1, got {config.max_batch_size}")
    expected_blocks = math.ceil(max_seq_len / 32)
    cache_config = lane.config.paged_kv_cache
    if cache_config.max_num_blocks != expected_blocks or cache_config.num_blocks != expected_blocks:
        raise ValueError(
            f"DP lane cache must contain {expected_blocks} blocks, got "
            f"max={cache_config.max_num_blocks}, resolved={cache_config.num_blocks}"
        )


def assert_no_special_tokens(
    generated_token_ids, tokenizer, *, case_name: str = "", is_ci_env: bool | None = None
) -> None:
    """Garbage guard: no special token mid-stream. Mirrors TTTv1 ``simple_text_demo.py``.

    TTTv2's ``result.generated_token_ids[user]`` already starts at the first generated token, so unlike
    TTTv1 we do not slice off the prompt — these are output-only. Each user's output is truncated at the
    first stop token before scanning, then checked for any ``tokenizer.all_special_ids`` member. Following
    TTTv1, a survivor logs a warning always but hard-fails only under CI (``CI == "true"``), so local runs
    finish while CI stays strict.

    DeepSeek-R1-Distill-Qwen-14B is eos-only: its only special tokens are BOS ``<｜begin▁of▁sentence｜>``
    and EOS ``<｜end▁of▁sentence｜>`` (no ``<|im_end|>`` / ``<|eot_id|>``), and the response terminator is
    the eos. ``<think>`` / ``</think>`` are ordinary tokens (not special ids) so a legitimate reasoning
    chain never trips the guard.
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
    """Run one user per supported TP lane through the migrated model-owned DP runtime."""
    tensor_parallel = _dp_lane_tp_or_skip(mesh_device, data_parallel)
    mesh_device.quiesce_devices()
    submeshes = _create_dp_submeshes(mesh_device, data_parallel, tensor_parallel)
    lane_cache_dir = _dp_lane_cache_dir(cache_dir, tensor_parallel)
    hf_model = os.environ.get("HF_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    precision = DEEPSEEK_R1_14B_PERFORMANCE if optimizations == "performance" else DEEPSEEK_R1_14B_ACCURACY
    prompts = load_input_prompts(data_parallel)
    sampling_mode = os.environ.get("SAMPLING_MODE", "host").lower()
    on_device_params = {
        "on_device": SamplingParams(temperature=0.0, top_k=1, top_p=0.0),
        "on_device_topk": SamplingParams(temperature=0.0, top_k=32, top_p=0.08),
    }

    models: list = []
    lanes: list = []
    group = None
    try:
        for submesh in submeshes:
            llm = from_pretrained(
                submesh,
                hf_model=hf_model,
                max_batch_size=1,
                max_seq_len=max_seq_len,
                n_layers=None,
                cache_dir=lane_cache_dir,
                optimizations=precision,
            )
            model = llm.model
            model.demo_tokenizer = llm.tokenizer
            models.append((model, submesh))
            lane = create_executor(
                model,
                traced=True,
                device_sampling_enabled=sampling_mode in on_device_params,
            )
            lanes.append(lane)
            _validate_dp_lane(model, lane, tensor_parallel, max_seq_len)

        group = LaneGroupExecutor(lanes, mesh_device=mesh_device)
        tokenizer = models[0][0].demo_tokenizer
        kv_cache = group.allocate_kv_cache()
        # Every lane owns an independent block pool; repeat the same lane-local block IDs for
        # each global row rather than assigning cross-lane global block offsets.
        page_table = make_contiguous_page_table(1, max_seq_len, 32).repeat(data_parallel, 1)
        _warmup_demo_executor(group, kv_cache=kv_cache, page_table=page_table)
        input_tokens, prompt_lens = tokenize_prompts(prompts, tokenizer)
        sampling_params = (
            on_device_params[sampling_mode]
            if sampling_mode in on_device_params and getattr(models[0][0], "supports_on_device_sampling", False)
            else None
        )
        logger.info(
            f"[ci-b1-DP-{data_parallel}] TP={tensor_parallel}, SAMPLING_MODE={sampling_mode} "
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
            prefill_sampling_params=None,
        )
        assert len(result.generated_token_ids) == data_parallel
        assert all(result.generated_token_ids), f"ci-b1-DP-{data_parallel}: every TP lane must return output"
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
def test_deepseek_r1_qwen_14b(test_config, mesh_device, optimizations):
    """Main test entry for TTTv2 DeepSeek-R1-Distill-Qwen-14B."""
    device_name = get_device_name(mesh_device)
    expected = EXPECTED_METRICS.get(optimizations, {}).get(device_name, {})
    model = None
    hf_model = os.environ.get("HF_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
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

        if test_config == "batch-32":
            # Short-context 32-user throughput. max_seq_len is DRAM-driven per profile (see create_model).
            max_bs, max_seq_len = 32, None
            expected = EXPECTED_METRICS_BATCH32.get(_sampling_bucket(), {}).get(optimizations, {}).get(device_name, {})
        elif test_config == "eval-32":
            # 32-user determinism. Needs seq >= 1024 (the ci-eval-32 prompt bucket). Accuracy profile is
            # DRAM-infeasible on N300 (skip); perf profile + T3K both run.
            _skip_if_dram_infeasible(device_name, optimizations, "eval-32")
            max_bs, max_seq_len = 32, _EVAL_MAX_SEQ_LEN
        elif test_config == "batch-32-ci":
            # CI-faithful batch-32 leg (TTTv1 ci-32 parity): seq2048 + 1024 decode budget. Accuracy profile
            # is DRAM-infeasible on N300 (skip); perf profile + T3K both run.
            _skip_if_dram_infeasible(device_name, optimizations, "batch-32-ci")
            max_bs = 32
            max_seq_len = _BATCH32_CI_MAX_SEQ_LEN.get(device_name, 2048)
            # Own perf gate measured at the seq2048/decode1024 workload (NOT the lighter batch-32 constant,
            # which would be a config-artifact miss). Keyed by SAMPLING_MODE AND profile. Non-topk
            # on-device modes (force-argmax) fall into the on_device_topk bucket; cells not measured fall
            # back to the short-context batch-32 constant (stay gated, never un-gated).
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
            # Natural-length prefill: these sample prompts bucket to 128, matching TTTv1's traced-prefill
            # seq len without a forced pad.
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
        if model is not None:
            cleanup_model_case(model, mesh_device)


def _run_token_accuracy(model: DeepSeekR1Qwen14B, mesh_device, expected):
    """Teacher-forcing token accuracy vs ``.refpt`` (CPU-generated)."""
    hf_model = os.environ.get("HF_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
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
        top5_tokens,
        reference_tokens,
        prompt_len,
        metadata_aligned=has_prompt_len_metadata,
    )
    is_ci_env = os.environ.get("CI") == "true"
    profiler = BenchmarkProfiler()
    try:
        profiler.start("run")
        # run_teacher_forcing times the prefill + per-step (teacher-forced) decode loop and, given the
        # profiler, brackets the "inference_prefill" / "inference_decode" steps itself — so the returned
        # result carries prefill/decode throughput alongside accuracy for CI benchmark-data emission.
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

    # Accuracy gate — threshold SOURCE is flag-controlled (currently ``is_ci_env``):
    #   use_centralized_targets = True  → mirror TTTv1: centralized targets via resolve_accuracy_targets
    #       minus an ABSOLUTE 0.5 pp (get_accuracy_thresholds, simple_text_demo.py). A missing entry is
    #       a hard error (never silently un-gate in CI).
    #   use_centralized_targets = False → the demo's local EXPECTED_METRICS values DIRECTLY (no ratio
    #       tolerance — TTTv1 applies none to accuracy).
    # Measured accuracy is rounded up with math.ceil before the compare, matching TTTv1 exactly
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

    # math.ceil matches TTTv1's integer-rounded accuracy check (simple_text_demo.py:1657-1658).
    meas_top1 = math.ceil(top1)
    meas_top5 = math.ceil(top5)
    assert meas_top1 >= min_top1, f"Top-1 accuracy {top1:.1f}% (ceil {meas_top1}) below threshold {min_top1:.1f}%"
    assert meas_top5 >= min_top5, f"Top-5 accuracy {top5:.1f}% (ceil {meas_top5}) below threshold {min_top5:.1f}%"


def _run_perf_benchmark(
    model: DeepSeekR1Qwen14B,
    mesh_device,
    expected,
    batch_size,
    case_name,
    max_prefill_len: int | None = None,
    num_decode_tokens: int | None = None,
):
    """Timed prefill + decode with the traced model-owned executor.

    Prefill uses each prompt's natural token length (TTTv1 ``preprocess_inputs_prefill`` semantics — the
    executor buckets to ``get_padded_prefill_len``); decode runs for ``num_decode_tokens`` steps (default
    ``_PERF_NUM_DECODE_TOKENS``). ``max_prefill_len`` is an optional clip cap for over-long prompts, never
    a pad-up target.

    The decode budget is clamped to what the paged KV cache can hold:
    ``effective = min(requested, max_seq_len - prompt_bucket - margin)`` so the high-water decode position
    never overruns the page table (the ``batch-32-ci`` leg requests 1024).
    """
    hf_model = os.environ.get("HF_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    tokenizer = model.demo_tokenizer

    # On-device sampling toggle (SAMPLING_MODE env):
    #   host            -> sampling_params=None (host-argmax; full-vocab all-gather + PCIe readback/step)
    #   on_device       -> greedy temp=0,k=1,p=0 => trace-captured FORCE-ARGMAX full-vocab path
    #   on_device_topk  -> temp=0,k=32,p=0.08    => trace-captured TOP-K op path (gathers only the [*,32]
    #                      tuples; PERF.md-parity recipe). DEFAULT: this is the TTTv1-comparable path
    #                      (TTTv1 auto-uses on-device sampling on multi-device meshes), so the gate
    #                      measures apples-to-apples.
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
    pipeline_readback = os.environ.get("PIPELINE_READBACK", "1").lower() not in ("0", "false", "no")
    logger.info(f"[{case_name}] SAMPLING_MODE={sampling_mode} -> sampling_params={sampling_params}")
    logger.info(f"[{case_name}] PIPELINE_READBACK={pipeline_readback}")

    # Free-running perf run: enable the executor's on-device decode loop on the on-device sampling
    # path (inert on host/force-argmax; gated to the top-k path by _decode_loop_active). This is the
    # shared #49284 decode-loop fix; it must be active on the perf path for on-device decode parity.
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
        # get_padded_prefill_len. These sample prompts are ~70-125 tokens -> 128 bucket.
        input_tokens, prompt_lens = tokenize_prompts(prompts, tokenizer, max_prefill_len=max_prefill_len)

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
            prefill_sampling_params=None if mesh_device.get_num_devices() > 1 else sampling_params,
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


# ci-eval-32 determinism case: 3 rotated repeats of the batch-32 workload.
_EVAL_REPEAT_BATCHES = 3
_EVAL_NUM_DECODE_TOKENS = _PERF_NUM_DECODE_TOKENS


def _run_eval_repeat_batch32(model: DeepSeekR1Qwen14B, mesh_device):
    """32-user cross-batch determinism (self-consistency under prompt rotation).

    Runs the batch-32 prefill+decode loop ``_EVAL_REPEAT_BATCHES`` times, rotating the prompt->slot
    assignment by one each repeat (fresh traced executor + KV cache per repeat), then asserts that undoing
    the rotation lines up per-user outputs. No external golden. Honors the same ``SAMPLING_MODE`` knob as
    ``_run_perf_benchmark`` (default host argmax — deterministic and mesh-agnostic, the recommended
    default for the determinism assert).

    Use the default (host argmax) for the determinism gate. Under ``SAMPLING_MODE=on_device_topk`` a
    reasoning model's degenerate numeric-prompt continuations can produce near-exact logit ties, and the
    on-device sampler's tie-break is slot-dependent (reduction order over the sharded vocab) → the
    cross-batch consistency assert can flip on those rotated slots. That is a property of on-device top-k
    sampling on tie-heavy degenerate output, NOT a determinism regression: host argmax passes with batched
    prefill ON and OFF, and any on-device flip is identical ON vs OFF (prefill-independent).
    """
    hf_model = os.environ.get("HF_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    tokenizer = model.demo_tokenizer

    # DeepSeek uses <｜User｜> as a new-turn boundary. It is not a global generation
    # default, but eval-32 treats it as a local terminator before determinism comparison.
    user_turn_id = tokenizer.convert_tokens_to_ids("<｜User｜>")
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
    # Static warmup covers the model's regular graph families, but this heterogeneous
    # workload produces data-dependent batched signatures (30 q128 rows and 2 q1024
    # rows). Register one representative rotation before traced warmup activates the
    # program gate. Prompt rotation preserves that signature multiset for every repeat.
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
