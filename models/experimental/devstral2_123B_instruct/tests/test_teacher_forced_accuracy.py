# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Teacher-forced top-1 / top-5 token accuracy vs HuggingFace for Devstral-2-123B.

Sweeps prefill context lengths (powers of two from 32 through 256K by default). Tale of Two
Cities is tiled when a longer token stream is required. Each sweep point writes a per-length
``.refpt`` HF reference (generated on first run) and a JSON result under
``tests/teacher_forced_sweep_outputs/`` (``references/`` and ``results/`` subdirs).

Run sanity (CI gate, short prefill lengths)::

    pytest models/experimental/devstral2_123B_instruct/tests/test_teacher_forced_accuracy.py -k sanity -v

Run full sweep (all 14 prefill lengths, 32 … 256K)::

    pytest models/experimental/devstral2_123B_instruct/tests/test_teacher_forced_accuracy.py -k sweep -v

Environment overrides::

    DEVSTRAL2_MIN_TOP1_ACC         Minimum top-1 fraction (default 0.96)
    DEVSTRAL2_MIN_TOP5_ACC         Minimum top-5 fraction (default 0.99)
    DEVSTRAL2_NUM_LAYERS           Limit decoder layers (default: all 88)
    DEVSTRAL2_WEIGHT_CACHE_SEQ_LEN Tiled-weight cache key (default 262144)
"""

from __future__ import annotations

import bz2
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.experimental.devstral2_123B_instruct.demo.text_demo import (
    _current_pos_to_tt,
    _input_ids_host,
    _input_ids_to_tt,
    _logits_to_torch,
    _round_up,
)
from models.experimental.devstral2_123B_instruct.reference.hf_reference_loader import (
    DEVSTRAL2_MODEL_ID,
    load_devstral2_causal_lm,
)
from models.experimental.devstral2_123B_instruct.tests._devstral_weights import (
    devstral2_weight_cache_seq_len,
    model_prefill_weight_keys,
    require_hf_weights,
    require_text_config,
    resolve_devstral2_weight_cache_path,
)
from models.experimental.devstral2_123B_instruct.tt.model_args import (
    DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    Devstral2Args,
)
from models.experimental.devstral2_123B_instruct.tt.tt_ministral3_model import TtMinistral3ForCausalLM
from models.experimental.devstral2_123B_instruct.tt.weight_loading import DEVSTRAL2_LARGE_REPO_ID
from models.tt_transformers.tt.ccl import TT_CCL

_TESTS_DIR = Path(__file__).resolve().parent
_TALE_OF_TWO_CITIES = _TESTS_DIR.parents[2] / "tt_transformers" / "tests" / "tale-of-two-cities.txt.bz2"

# Full sweep (powers of two 32 … 256K). Pytest timeout always budgets for all 14 points.
FULL_SWEEP_PREFILL_LENGTHS = [
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
]

# Short prefill gate for CI / nightly sanity before the full 14-point sweep is enabled.
SANITY_SWEEP_PREFILL_LENGTHS = [32, 64, 128]

# Teacher-forced eval window after prefill. Matches tt-transformers CI token-accuracy run
# (max_generated_tokens=500); the .refpt there is 1024 tokens split 512 prefill / 512 eval but
# decode stops at 500 iterations.
TEACHER_EVAL_TOKENS = 500

# Stop on first failure by default; set True to run all points and fail at the end.
_SWEEP_RUN_ALL_BEFORE_FAIL = False

# Single pytest timeout for the full sweep (not per seq len). See ``_sweep_timeout_seconds()``.
_SWEEP_TIMEOUT_MARGIN = 1.25

_MIN_TOP1_ACC = float(os.environ.get("DEVSTRAL2_MIN_TOP1_ACC", "0.96"))
_MIN_TOP5_ACC = float(os.environ.get("DEVSTRAL2_MIN_TOP5_ACC", "0.99"))


def _sweep_output_dir() -> Path:
    return _TESTS_DIR / "teacher_forced_sweep_outputs"


def _sweep_timeout_seconds(prefill_lengths: list[int]) -> int:
    """Budget for TT model load, weight cache, cold HF ``.refpt`` generation, and sweep points.

    Calibrated from BH Loudbox run 2026-06-15T11:21Z (prefill 32/64/128, 500 eval, cold HF refs):
    **2270 s (~38 min)** for 3 points. Rates: HF hub load ~167 s/ref, HF forward ~0.07 s/token,
    TT prefill ~39 s/128-token chunk, TT decode ~0.42 s/step → **~71 h** for 14 points (25% margin).
    """
    tt_model_setup_sec = 600
    hf_hub_load_per_ref_sec = 167
    hf_forward_sec_per_token = 0.072
    tt_base_sec = 60
    tt_prefill_sec_per_chunk = 39
    tt_decode_sec_per_eval_token = 0.42
    kv_block_size = Devstral2Args.kv_block_size

    budget = tt_model_setup_sec
    for prefill_len in prefill_lengths:
        total_length = prefill_len + TEACHER_EVAL_TOKENS
        num_prefill_chunks = (prefill_len + kv_block_size - 1) // kv_block_size
        budget += hf_hub_load_per_ref_sec + int(total_length * hf_forward_sec_per_token)
        budget += tt_base_sec + num_prefill_chunks * tt_prefill_sec_per_chunk
        budget += int(TEACHER_EVAL_TOKENS * tt_decode_sec_per_eval_token)
    return int(budget * _SWEEP_TIMEOUT_MARGIN)


def _eval_tokens_for_prefill(prefill_len: int) -> int:
    """Fixed 500-token teacher-forced eval window (independent of prefill length)."""
    del prefill_len
    return TEACHER_EVAL_TOKENS


def _sweep_reference_path(prefill_len: int, total_length: int) -> Path:
    return _sweep_output_dir() / "references" / f"prefill_{prefill_len:06d}_total_{total_length:06d}.refpt"


def _sweep_result_path(prefill_len: int) -> Path:
    return _sweep_output_dir() / "results" / f"prefill_{prefill_len:06d}.json"


def _load_corpus_text() -> str:
    if not _TALE_OF_TWO_CITIES.is_file():
        raise FileNotFoundError(f"Corpus not found: {_TALE_OF_TWO_CITIES}")
    with bz2.open(_TALE_OF_TWO_CITIES, "rt", encoding="utf-8") as f:
        return f.read()


def _base_corpus_tokens(tokenizer) -> list[int]:
    return tokenizer.encode(_load_corpus_text(), add_special_tokens=True)


def _build_token_sequence(tokenizer, total_length: int) -> list[int]:
    """Return ``total_length`` tokens from Tale of Two Cities, tiling the stream when needed."""
    if total_length <= 0:
        raise ValueError(f"total_length must be positive, got {total_length}")

    base = _base_corpus_tokens(tokenizer)
    if not base:
        raise ValueError("Corpus tokenization produced an empty sequence")

    if total_length <= len(base):
        return base[:total_length]

    repeat_body = base[1:] if len(base) > 1 else base
    tokens = list(base)
    while len(tokens) < total_length:
        need = total_length - len(tokens)
        tokens.extend(repeat_body[:need])
    return tokens[:total_length]


class TokenAccuracy:
    """Record TT argmax predictions while feeding ground-truth reference tokens on the next step."""

    def __init__(
        self,
        reference_tokens: torch.Tensor,
        top5_tokens: torch.Tensor,
        *,
        max_eval_tokens: int,
        split_point: int,
    ) -> None:
        if reference_tokens.ndim == 2:
            reference_tokens = reference_tokens[0]
        self.input_prompt = reference_tokens[:split_point]
        self.reference_tokens = reference_tokens[split_point:]
        self.top5_tokens = top5_tokens[split_point - 1 :]
        self.max_eval_tokens = min(max_eval_tokens, len(self.reference_tokens))
        self.gt_pos = -1
        self.store_predicted_tokens: list[int] = []

    @property
    def prefill_len(self) -> int:
        return int(self.input_prompt.shape[-1])

    @property
    def eval_len(self) -> int:
        return self.max_eval_tokens

    def collect_predicted_tokens(self, token_id: int) -> int:
        self.store_predicted_tokens.append(int(token_id))
        self.gt_pos += 1
        idx = min(self.gt_pos, len(self.reference_tokens) - 1)
        return int(self.reference_tokens[idx].item())

    def compute_accuracy(self) -> tuple[float, float, int]:
        count_top1 = 0
        count_top5 = 0
        matching_sz = min(len(self.reference_tokens), len(self.store_predicted_tokens), self.max_eval_tokens)
        for i in range(matching_sz):
            pred = self.store_predicted_tokens[i]
            ref_top1 = int(self.top5_tokens[i, 0].item())
            if pred == ref_top1:
                count_top1 += 1
            if pred in self.top5_tokens[i, :].tolist():
                count_top5 += 1
        if matching_sz == 0:
            return 0.0, 0.0, 0
        return count_top1 / matching_sz, count_top5 / matching_sz, matching_sz

    def mismatch_indices(self) -> list[int]:
        indices: list[int] = []
        matching_sz = min(len(self.reference_tokens), len(self.store_predicted_tokens), self.max_eval_tokens)
        for i in range(matching_sz):
            pred = self.store_predicted_tokens[i]
            ref_top1 = int(self.top5_tokens[i, 0].item())
            if pred != ref_top1:
                indices.append(i)
        return indices


def _generate_reference_file(
    output_file: Path,
    *,
    total_length: int,
    prefill_len: int,
    chunk_size: int = 1024,
) -> None:
    """Tokenize (and tile) Tale of Two Cities, run HF forward, save ``reference_tokens`` + ``top5_tokens``."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Generating teacher-forcing reference at {output_file} "
        f"(total_length={total_length}, prefill_len={prefill_len})"
    )
    tokenizer = AutoTokenizer.from_pretrained(DEVSTRAL2_MODEL_ID, trust_remote_code=True)
    encoded_tokens = _build_token_sequence(tokenizer, total_length)

    model = load_devstral2_causal_lm()
    model.eval()
    device = next(model.parameters()).device

    encoded_tokens_tensor = torch.tensor(encoded_tokens, device=device).unsqueeze(0)
    all_top5_tokens: list[torch.Tensor] = []

    with torch.no_grad():
        for chunk_start in range(0, total_length - 1, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_length)
            chunk_tokens = encoded_tokens_tensor[:, chunk_start:chunk_end]
            chunk_next_tokens = encoded_tokens[chunk_start + 1 : chunk_end + 1]
            actual_chunk_size = min(len(chunk_tokens[0]), len(chunk_next_tokens))
            chunk_tokens = chunk_tokens[:, :actual_chunk_size]

            logits = model(chunk_tokens).logits
            probs = torch.softmax(logits, dim=-1)
            _, chunk_top5_tokens = torch.topk(probs, k=5, dim=-1)
            all_top5_tokens.append(chunk_top5_tokens.squeeze(0).cpu())

    data = {
        "reference_tokens": encoded_tokens_tensor[:, :total_length].clone().cpu(),
        "top5_tokens": torch.cat(all_top5_tokens, dim=0).cpu(),
        "model_id": DEVSTRAL2_MODEL_ID,
        "total_length": total_length,
        "prefill_len": prefill_len,
        "corpus_extended": total_length > len(_base_corpus_tokens(tokenizer)),
    }
    torch.save(data, output_file)
    logger.info(f"Saved reference outputs to {output_file}")


def _ensure_reference_file(
    reference_file: Path,
    *,
    total_length: int,
    prefill_len: int,
) -> Path:
    """Return path to ``.refpt``, generating it with HF when absent or stale."""
    if reference_file.is_file():
        data = torch.load(reference_file, weights_only=False)
        cached_len = int(data.get("total_length", 0))
        cached_prefill = int(data.get("prefill_len", 0))
        if cached_len == total_length and cached_prefill == prefill_len:
            logger.info(f"Using existing reference: {reference_file}")
            return reference_file
        logger.info(
            f"Stale reference at {reference_file} "
            f"(cached total={cached_len}, prefill={cached_prefill}; "
            f"want total={total_length}, prefill={prefill_len}); regenerating..."
        )
    else:
        logger.info(f"Reference not found at {reference_file}; generating from HF...")

    try:
        _generate_reference_file(reference_file, total_length=total_length, prefill_len=prefill_len)
    except Exception as exc:
        pytest.skip(
            "Could not generate teacher-forcing reference "
            "(set HF_TOKEN if gated, ensure offload disk space, or pre-generate the .refpt). "
            f"Error: {exc}"
        )
    return reference_file


def _load_reference_data(reference_file: Path) -> dict:
    logger.info(f"Loading teacher-forcing reference from {reference_file}")
    data = torch.load(reference_file, weights_only=False)
    if "reference_tokens" not in data or "top5_tokens" not in data:
        raise KeyError(f"{reference_file} must contain 'reference_tokens' and 'top5_tokens'")
    return data


def _mesh_device_param():
    return {
        "N150": (1, 1),
        "N300": (1, 2),
        "N150x4": (1, 4),
        "P150x4": (1, 4),
        "T3K": (1, 8),
        "TG": (8, 4),
    }.get(os.environ.get("MESH_DEVICE"), (1, 8))


def _test_seq_len(prefill_len: int, eval_tokens: int, kv_block_size: int) -> int:
    need = prefill_len + eval_tokens + 1
    return max(_round_up(need, kv_block_size), kv_block_size)


def _runtime_max_seq_len(test_seq_len: int, kv_block_size: int) -> int:
    cache_seq = devstral2_weight_cache_seq_len()
    return max(_round_up(test_seq_len, kv_block_size), cache_seq)


def _build_model(mesh_device, num_layers: int | None, max_seq_len: int):
    text_cfg = require_text_config()
    full_layers = int(text_cfg.num_hidden_layers)
    n_layers = num_layers or full_layers
    assert 0 < n_layers <= full_layers, f"requested {n_layers} layers, model has {full_layers}"

    weight_cache_path = resolve_devstral2_weight_cache_path(mesh_device, text_cfg, n_layers)

    args = Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=tuple(mesh_device.shape),
        max_seq_len=max_seq_len,
        max_batch_size=1,
    )

    base_keys = model_prefill_weight_keys(n_layers)
    want_lm_head = not args.tie_word_embeddings
    try:
        weight_keys = base_keys + (["lm_head.weight"] if want_lm_head else [])
        state_dict = require_hf_weights(weight_keys)
    except Exception:
        if want_lm_head:
            logger.warning("lm_head.weight unavailable on the Hub; falling back to tied embeddings.")
            state_dict = require_hf_weights(base_keys)
        else:
            raise

    tt_ccl = TT_CCL(mesh_device)
    model = TtMinistral3ForCausalLM(
        args,
        mesh_device,
        state_dict,
        tt_ccl,
        num_layers=n_layers,
        weight_cache_path=weight_cache_path,
    )
    return model, args, weight_cache_path


def _argmax_from_prefill_logits(
    logits: ttnn.Tensor,
    mesh_device,
    vocab_size: int,
    *,
    prefill_len: int,
    last_chunk_start: int,
) -> int:
    local_pos = (prefill_len - 1) - last_chunk_start
    logits_torch = _logits_to_torch(logits, mesh_device, vocab_size)
    return int(logits_torch[local_pos].argmax().item())


def _argmax_from_decode_logits(logits: ttnn.Tensor, mesh_device, vocab_size: int) -> int:
    logits_torch = _logits_to_torch(logits, mesh_device, vocab_size)
    return int(logits_torch[0].argmax().item())


@torch.no_grad()
def _run_teacher_forced_accuracy(
    model: TtMinistral3ForCausalLM,
    args: Devstral2Args,
    mesh_device,
    token_acc: TokenAccuracy,
) -> tuple[float, float, int]:
    prefill_len = token_acc.prefill_len
    eval_tokens = token_acc.eval_len
    kv_block_size = args.kv_block_size

    pad_id = args.pad_token_id if args.pad_token_id is not None else 0
    input_ids = token_acc.input_prompt.unsqueeze(0)
    num_prefill_chunks = max(1, (prefill_len + kv_block_size - 1) // kv_block_size)
    padded_prompt_len = num_prefill_chunks * kv_block_size

    input_ids_padded = torch.full((1, padded_prompt_len), int(pad_id), dtype=torch.long)
    input_ids_padded[:, :prefill_len] = input_ids

    prefill_chunk_dev = _input_ids_to_tt(
        torch.full((1, kv_block_size), int(pad_id), dtype=torch.long),
        mesh_device,
    )

    prefill_logits = None
    for chunk_idx in range(num_prefill_chunks):
        chunk_start = chunk_idx * kv_block_size
        chunk_tokens = input_ids_padded[:, chunk_start : chunk_start + kv_block_size]
        ttnn.copy_host_to_device_tensor(
            _input_ids_host(chunk_tokens, mesh_device),
            prefill_chunk_dev,
        )
        tt_out = model(prefill_chunk_dev, mode="prefill", start_pos=chunk_start)
        if chunk_idx == num_prefill_chunks - 1:
            prefill_logits = tt_out
        else:
            tt_out.deallocate(True)

    assert prefill_logits is not None
    ttnn.synchronize_device(mesh_device)

    last_chunk_start = (num_prefill_chunks - 1) * kv_block_size
    predicted = _argmax_from_prefill_logits(
        prefill_logits,
        mesh_device,
        args.vocab_size,
        prefill_len=prefill_len,
        last_chunk_start=last_chunk_start,
    )
    prefill_logits.deallocate(True)
    token_acc.collect_predicted_tokens(predicted)

    decode_tok_dev = _input_ids_to_tt(torch.zeros((1, 1), dtype=torch.long), mesh_device)
    current_pos = prefill_len

    for step in range(1, eval_tokens):
        ref_tok = int(token_acc.reference_tokens[step - 1].item())
        ttnn.copy_host_to_device_tensor(
            _input_ids_host(torch.tensor([[ref_tok]], dtype=torch.long), mesh_device),
            decode_tok_dev,
        )
        decode_pos_dev = _current_pos_to_tt(torch.tensor([current_pos], dtype=torch.long), mesh_device)
        decode_logits = model(decode_tok_dev, mode="decode", current_pos=decode_pos_dev)
        predicted = _argmax_from_decode_logits(decode_logits, mesh_device, args.vocab_size)
        decode_logits.deallocate(True)
        token_acc.collect_predicted_tokens(predicted)
        current_pos += 1

    ttnn.synchronize_device(mesh_device)
    return token_acc.compute_accuracy()


def _log_accuracy_table(token_acc: TokenAccuracy, tokenizer, *, max_rows: int = 20) -> None:
    rows = min(token_acc.eval_len, len(token_acc.store_predicted_tokens), max_rows)
    logger.info(f"{'Step':<8}{'Top1':<6}{'Top5':<6}{'Ref':<20}{'TT':<20}")
    logger.info("-" * 60)
    for i in range(rows):
        pred = token_acc.store_predicted_tokens[i]
        ref_top1 = int(token_acc.top5_tokens[i, 0].item())
        in_top5 = pred in token_acc.top5_tokens[i, :].tolist()
        ref_text = tokenizer.decode([ref_top1])[:18]
        tt_text = tokenizer.decode([pred])[:18]
        logger.info(
            f"{i:<8}{'x' if pred == ref_top1 else ' ':<6}{'x' if in_top5 else ' ':<6}"
            f"{ref_text!r:<20}{tt_text!r:<20}"
        )


def _dump_sweep_result(result: dict) -> Path:
    path = _sweep_result_path(result["prefill_len"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Wrote sweep result: {path}")
    return path


def _dump_sweep_summary(run_id: str, results: list[dict]) -> Path:
    summary_path = _sweep_output_dir() / "results" / f"sweep_summary_{run_id}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sweep_prefill_lengths": SWEEP_PREFILL_LENGTHS,
        "teacher_eval_tokens": TEACHER_EVAL_TOKENS,
        "sweep_timeout_seconds": _sweep_timeout_seconds(),
        "thresholds": {"top1_min": _MIN_TOP1_ACC, "top5_min": _MIN_TOP5_ACC},
        "num_points": len(results),
        "num_passed": sum(1 for r in results if r.get("passed")),
        "num_failed": sum(1 for r in results if not r.get("passed")),
        "results": results,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Wrote sweep summary: {summary_path}")
    return summary_path


def _run_sweep_point(
    mesh_device,
    model: TtMinistral3ForCausalLM,
    args: Devstral2Args,
    *,
    prefill_len: int,
    weight_cache_path: str,
    run_id: str,
    tokenizer,
) -> dict:
    eval_tokens = _eval_tokens_for_prefill(prefill_len)
    total_length = prefill_len + eval_tokens
    reference_file = _sweep_reference_path(prefill_len, total_length)

    kv_block_size = Devstral2Args.kv_block_size
    test_seq_len = _test_seq_len(prefill_len, eval_tokens, kv_block_size)
    max_seq_len = _runtime_max_seq_len(test_seq_len, kv_block_size)

    reference_file = _ensure_reference_file(
        reference_file,
        total_length=total_length,
        prefill_len=prefill_len,
    )
    reference_data = _load_reference_data(reference_file)
    token_acc = TokenAccuracy(
        reference_data["reference_tokens"],
        reference_data["top5_tokens"],
        max_eval_tokens=eval_tokens,
        split_point=prefill_len,
    )

    logger.info(
        f"Sweep prefill={prefill_len}: eval={eval_tokens}, total={total_length}, "
        f"test_seq_len={test_seq_len}, max_seq_len={max_seq_len}, reference={reference_file}"
    )

    top1, top5, n_tokens = _run_teacher_forced_accuracy(model, args, mesh_device, token_acc)
    _log_accuracy_table(token_acc, tokenizer)

    passed = top1 >= _MIN_TOP1_ACC and top5 >= _MIN_TOP5_ACC
    mismatches = token_acc.mismatch_indices()
    result = {
        "run_id": run_id,
        "prefill_len": prefill_len,
        "eval_tokens": eval_tokens,
        "total_length": total_length,
        "test_seq_len": test_seq_len,
        "max_seq_len": max_seq_len,
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "num_eval_tokens": n_tokens,
        "passed": passed,
        "thresholds": {"top1_min": _MIN_TOP1_ACC, "top5_min": _MIN_TOP5_ACC},
        "reference_file": str(reference_file),
        "weight_cache_path": weight_cache_path,
        "corpus_extended": bool(reference_data.get("corpus_extended", False)),
        "num_top1_mismatches": len(mismatches),
        "mismatch_steps": mismatches[:50],
        "predicted_tokens": token_acc.store_predicted_tokens,
        "reference_top1_tokens": [int(token_acc.top5_tokens[i, 0].item()) for i in range(n_tokens)],
    }

    logger.info(
        f"Sweep prefill={prefill_len}: top-1={top1:.4f} ({top1 * 100:.2f}%), "
        f"top-5={top5:.4f} ({top5 * 100:.2f}%), passed={passed}"
    )
    _dump_sweep_result(result)
    return result


_DEVICE_PARAMS = [
    {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "trace_region_size": 30_000_000,
        "num_command_queues": 1,
        "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    }
]


@pytest.mark.slow
@pytest.mark.models_performance_bare_metal
@pytest.mark.timeout(_sweep_timeout_seconds(FULL_SWEEP_PREFILL_LENGTHS))
@pytest.mark.parametrize("mesh_device", [_mesh_device_param()], indirect=True)
@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
def test_devstral2_teacher_forced_accuracy_sweep(mesh_device):
    """Teacher-forced accuracy sweep: prefill lengths 32, 64, …, 256K (powers of two)."""
    _run_teacher_forced_accuracy_sweep(mesh_device, FULL_SWEEP_PREFILL_LENGTHS)


@pytest.mark.models_performance_bare_metal
@pytest.mark.timeout(_sweep_timeout_seconds(SANITY_SWEEP_PREFILL_LENGTHS))
@pytest.mark.parametrize("mesh_device", [_mesh_device_param()], indirect=True)
@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
def test_devstral2_teacher_forced_accuracy_sanity(mesh_device):
    """Short prefill lengths (32/64/128) as a gate before the full sweep runs in CI."""
    _run_teacher_forced_accuracy_sweep(mesh_device, SANITY_SWEEP_PREFILL_LENGTHS)


def _run_teacher_forced_accuracy_sweep(mesh_device, prefill_lengths: list[int]) -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fail_fast = not _SWEEP_RUN_ALL_BEFORE_FAIL
    sweep_timeout_sec = _sweep_timeout_seconds(prefill_lengths)

    logger.info(
        f"Teacher-forced sweep run_id={run_id}, points={prefill_lengths}, "
        f"eval_tokens={TEACHER_EVAL_TOKENS}, pytest_timeout={sweep_timeout_sec}s "
        f"({sweep_timeout_sec / 3600:.1f}h)"
    )

    raw_layers = os.environ.get("DEVSTRAL2_NUM_LAYERS", "").strip()
    num_layers = int(raw_layers) if raw_layers else None

    kv_block_size = Devstral2Args.kv_block_size
    worst_prefill = max(prefill_lengths)
    worst_eval = _eval_tokens_for_prefill(worst_prefill)
    global_test_seq_len = _test_seq_len(worst_prefill, worst_eval, kv_block_size)
    global_max_seq_len = _runtime_max_seq_len(global_test_seq_len, kv_block_size)
    logger.info(
        f"Building model once for sweep: worst_prefill={worst_prefill}, "
        f"global_test_seq_len={global_test_seq_len}, global_max_seq_len={global_max_seq_len}"
    )

    model, args, weight_cache_path = _build_model(mesh_device, num_layers, global_max_seq_len)
    logger.info(f"TT weight cache (reuse): {weight_cache_path}")

    tokenizer = AutoTokenizer.from_pretrained(DEVSTRAL2_LARGE_REPO_ID, trust_remote_code=True)

    results: list[dict] = []
    failures: list[str] = []

    for prefill_len in prefill_lengths:
        result = _run_sweep_point(
            mesh_device,
            model,
            args,
            prefill_len=prefill_len,
            weight_cache_path=weight_cache_path,
            run_id=run_id,
            tokenizer=tokenizer,
        )
        results.append(result)
        if not result["passed"]:
            msg = (
                f"prefill={prefill_len}: top-1={result['top1_accuracy']:.4f} "
                f"(min {_MIN_TOP1_ACC:.4f}), top-5={result['top5_accuracy']:.4f} (min {_MIN_TOP5_ACC:.4f})"
            )
            failures.append(msg)
            if fail_fast:
                _dump_sweep_summary(run_id, results)
                pytest.fail(msg)

    summary_path = _dump_sweep_summary(run_id, results)
    logger.info(f"Sweep complete: {len(results)} points, {len(failures)} failed, summary={summary_path}")

    if failures:
        pytest.fail("Teacher-forced sweep failures:\n" + "\n".join(failures))
