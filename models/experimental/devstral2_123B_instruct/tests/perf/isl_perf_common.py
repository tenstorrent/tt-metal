# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TT-only ISL (input sequence length) performance sweep for Devstral-2-123B.

Mirrors ``demo/text_demo.py`` traced chunked prefill + decode trace (2CQ when enabled).
No HuggingFace model forward. KV/RoPE is sized once for the sweep using
``DEVSTRAL2_MAX_NEW_TOKENS`` (default 100, same as ``text_demo``) with trace-safe alignment.

Metrics (device time only; compile/capture excluded from timed windows):
  - ISL — requested input length (tokens)
  - TTFT (s) — prefill trace replay after capture
  - Prefill tok/s — ``padded_ISL / prefill_replay_s``
  - Decode tok/s/u — decode trace replays (``max_new_tokens``, wall-clock capped)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.devstral2_123B_instruct.demo.decode_trace_2cq import (
    DecodeTrace2CQ,
    decode_trace_2cq_enabled,
    signal_decode_step_done,
    stage_decode_inputs,
)
from models.experimental.devstral2_123B_instruct.demo.text_demo import (
    _current_pos_to_tt,
    _init_prefill_trace_buffers,
    _input_ids_to_tt,
    _prefill_flexible_kwargs,
    _time_tt_op,
    _update_prefill_trace_buffers,
)
from models.experimental.devstral2_123B_instruct.tests._devstral_weights import (
    devstral2_isl_perf_decode_replay_iters,
    devstral2_isl_perf_kv_max_seq_len,
    devstral2_max_new_tokens,
    devstral2_tt_weight_cache_dir,
    log_tt_weight_cache_status,
    model_prefill_weight_keys,
    require_hf_weights,
    require_text_config,
)
from models.experimental.devstral2_123B_instruct.tests.decoder_pcc_common import (
    PREFILL_SANITY_SEQ_LENGTHS,
    PREFILL_SWEEP_SEQ_LENGTHS,
)
from models.experimental.devstral2_123B_instruct.tt.model_args import Devstral2Args
from models.experimental.devstral2_123B_instruct.tt.tt_ministral3_model import TtMinistral3ForCausalLM
from models.tt_transformers.tt.ccl import TT_CCL

_SWEEP_TIMEOUT_MARGIN = 1.25
ISL_PERF_SANITY_TIMEOUT_SEC = int(os.environ.get("DEVSTRAL2_ISL_PERF_SANITY_TIMEOUT_SEC", str(3 * 3600)))
ISL_PERF_FULL_SWEEP_TIMEOUT_SEC = int(os.environ.get("DEVSTRAL2_ISL_PERF_SWEEP_TIMEOUT_SEC", str(6 * 3600)))

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "isl_sweep_perf_outputs"

_isl_perf_model_cache: dict[tuple[tuple[int, int], int], tuple[TtMinistral3ForCausalLM, Devstral2Args, str]] = {}


def _chunk_log_every(num_chunks: int) -> Optional[int]:
    """Return log interval in chunks, or None when chunk progress logging is disabled."""
    raw = os.environ.get("DEVSTRAL2_ISL_PERF_CHUNK_LOG_EVERY", "0").strip()
    every = int(raw)
    if every <= 0:
        return None
    return max(1, every)


def _log_prefill_chunk_progress(
    *,
    isl: int,
    phase: str,
    chunk_idx: int,
    num_chunks: int,
    phase_started_s: float,
) -> None:
    every = _chunk_log_every(num_chunks)
    if every is None:
        return
    if chunk_idx != 0 and chunk_idx != num_chunks - 1 and (chunk_idx + 1) % every != 0:
        return
    elapsed_s = time.perf_counter() - phase_started_s
    logger.info(
        f"ISL={isl} prefill {phase}: chunk {chunk_idx + 1}/{num_chunks} " f"({elapsed_s:.1f}s elapsed in {phase})"
    )


def _trace_prefill_enabled() -> bool:
    return os.environ.get("DEVSTRAL2_TRACE_PREFILL", "1").strip().lower() not in ("0", "false", "no")


def _require_traced_prefill(args: Devstral2Args) -> None:
    if not _trace_prefill_enabled():
        pytest.skip("ISL perf requires prefill trace (unset DEVSTRAL2_TRACE_PREFILL=0)")
    if not args.supports_flexible_chunked_sdpa:
        pytest.fail(
            "ISL perf requires flexible chunked SDPA for prefill trace; "
            f"got kv_page_table_blocks={args.kv_page_table_blocks_per_user}, "
            f"logical={args.kv_num_blocks_per_user}. Check KV alignment."
        )


def _estimate_isl_sweep_seconds(isl_lengths: list[int]) -> int:
    total = 0
    kv_block = Devstral2Args.kv_block_size
    for isl in isl_lengths:
        chunks = max(1, (isl + kv_block - 1) // kv_block)
        decode_iters = devstral2_isl_perf_decode_replay_iters()
        total += 120 + chunks * 45 + decode_iters * 2
    return total


def isl_sweep_timeout_seconds(isl_lengths: list[int]) -> int:
    if isl_lengths == PREFILL_SWEEP_SEQ_LENGTHS:
        return ISL_PERF_FULL_SWEEP_TIMEOUT_SEC
    if isl_lengths == PREFILL_SANITY_SEQ_LENGTHS:
        return ISL_PERF_SANITY_TIMEOUT_SEC
    return int(_estimate_isl_sweep_seconds(isl_lengths) * _SWEEP_TIMEOUT_MARGIN)


def build_isl_perf_model(
    mesh_device,
    isl_lengths: list[int],
) -> tuple[TtMinistral3ForCausalLM, Devstral2Args, str]:
    """Build full 88-layer TT model once; KV sized for the ISL sweep (no HF forward)."""
    text_cfg = require_text_config()
    num_layers = int(text_cfg.num_hidden_layers)
    model_max_seq_len = devstral2_isl_perf_kv_max_seq_len(isl_lengths)
    weight_cache_path = devstral2_tt_weight_cache_dir(mesh_device, text_cfg)
    log_tt_weight_cache_status(weight_cache_path, num_layers)

    max_new_tokens = devstral2_max_new_tokens()
    logger.info(f"ISL perf KV budget: max_seq_len={model_max_seq_len}, max_new_tokens={max_new_tokens}")

    args = Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=tuple(mesh_device.shape),
        max_seq_len=model_max_seq_len,
        max_batch_size=1,
    )
    _require_traced_prefill(args)

    base_keys = model_prefill_weight_keys(num_layers)
    want_lm_head = not args.tie_word_embeddings
    try:
        weight_keys = base_keys + (["lm_head.weight"] if want_lm_head else [])
        state_dict = require_hf_weights(weight_keys)
    except Exception:
        if want_lm_head:
            logger.warning("lm_head.weight unavailable; falling back to tied embeddings.")
            state_dict = require_hf_weights(base_keys)
        else:
            raise

    tt_ccl = TT_CCL(mesh_device)
    model = TtMinistral3ForCausalLM(
        args,
        mesh_device,
        state_dict,
        tt_ccl,
        num_layers=num_layers,
        weight_cache_path=weight_cache_path,
    )
    logger.info(
        f"ISL perf TT model ready: layers={num_layers}, max_seq_len={model_max_seq_len}, " f"cache={weight_cache_path}"
    )
    return model, args, weight_cache_path


def get_or_build_isl_perf_model(
    mesh_device,
    isl_lengths: list[int],
) -> tuple[TtMinistral3ForCausalLM, Devstral2Args, str]:
    model_max_seq_len = devstral2_isl_perf_kv_max_seq_len(isl_lengths)
    cache_key = (tuple(mesh_device.shape), model_max_seq_len)
    if cache_key not in _isl_perf_model_cache:
        logger.info(f"Building ISL perf TT model for mesh={cache_key[0]}, max_seq_len={model_max_seq_len}")
        _isl_perf_model_cache[cache_key] = build_isl_perf_model(mesh_device, isl_lengths)
    else:
        logger.info(f"Reusing ISL perf TT model for mesh={cache_key[0]}, max_seq_len={model_max_seq_len}")
    return _isl_perf_model_cache[cache_key]


def _padded_chunk_layout(isl: int) -> tuple[int, int, int]:
    kv_block_size = Devstral2Args.kv_block_size
    num_chunks = max(1, (isl + kv_block_size - 1) // kv_block_size)
    padded_len = num_chunks * kv_block_size
    return isl, padded_len, num_chunks


def _run_traced_chunked_prefill(
    mesh_device,
    model: TtMinistral3ForCausalLM,
    *,
    isl: int,
    input_ids_padded: torch.Tensor,
    num_chunks: int,
    kv_block_size: int,
    rotary_emb,
    prefill_chunk_dev: ttnn.Tensor,
) -> tuple[float, float]:
    """``text_demo`` traced prefill: compile + capture (untimed), timed trace replay."""
    chunk_start_dev, chunk_pt_dev, rope_dev_bufs = _init_prefill_trace_buffers(
        mesh_device,
        kv_block_size=kv_block_size,
        head_dim=model.args.head_dim,
    )

    t0 = time.perf_counter()
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * kv_block_size
        chunk_tokens = input_ids_padded[:, chunk_start : chunk_start + kv_block_size]
        chunk_page_table = _update_prefill_trace_buffers(
            rotary_emb=rotary_emb,
            mesh_device=mesh_device,
            prefill_chunk_dev=prefill_chunk_dev,
            chunk_tokens=chunk_tokens,
            chunk_start=chunk_start,
            kv_block_size=kv_block_size,
            chunk_start_dev=chunk_start_dev,
            chunk_pt_dev=chunk_pt_dev,
            rope_dev_bufs=rope_dev_bufs,
        )
        flex = _prefill_flexible_kwargs(
            chunk_start=chunk_start,
            chunk_start_dev=chunk_start_dev,
            chunk_page_table=chunk_page_table,
            rope_dev_bufs=rope_dev_bufs,
        )
        warm = model(prefill_chunk_dev, mode="prefill", **flex)
        warm.deallocate(True)
        _log_prefill_chunk_progress(
            isl=isl, phase="compile", chunk_idx=chunk_idx, num_chunks=num_chunks, phase_started_s=t0
        )
    ttnn.synchronize_device(mesh_device)
    prefill_compile_s = _time_tt_op(t0)
    logger.info(f"ISL={isl} prefill compile done: {prefill_compile_s:.1f}s ({num_chunks} chunk(s))")

    chunk_tokens = input_ids_padded[:, :kv_block_size]
    chunk_page_table = _update_prefill_trace_buffers(
        rotary_emb=rotary_emb,
        mesh_device=mesh_device,
        prefill_chunk_dev=prefill_chunk_dev,
        chunk_tokens=chunk_tokens,
        chunk_start=0,
        kv_block_size=kv_block_size,
        chunk_start_dev=chunk_start_dev,
        chunk_pt_dev=chunk_pt_dev,
        rope_dev_bufs=rope_dev_bufs,
    )
    flex0 = _prefill_flexible_kwargs(
        chunk_start=0,
        chunk_start_dev=chunk_start_dev,
        chunk_page_table=chunk_page_table,
        rope_dev_bufs=rope_dev_bufs,
    )

    prefill_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    prefill_logits = model(prefill_chunk_dev, mode="prefill", **flex0)
    ttnn.end_trace_capture(mesh_device, prefill_trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    t0 = time.perf_counter()
    ttnn.execute_trace(mesh_device, prefill_trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    _log_prefill_chunk_progress(isl=isl, phase="replay", chunk_idx=0, num_chunks=num_chunks, phase_started_s=t0)
    for chunk_idx in range(1, num_chunks):
        chunk_start = chunk_idx * kv_block_size
        chunk_tokens = input_ids_padded[:, chunk_start : chunk_start + kv_block_size]
        _update_prefill_trace_buffers(
            rotary_emb=rotary_emb,
            mesh_device=mesh_device,
            prefill_chunk_dev=prefill_chunk_dev,
            chunk_tokens=chunk_tokens,
            chunk_start=chunk_start,
            kv_block_size=kv_block_size,
            chunk_start_dev=chunk_start_dev,
            chunk_pt_dev=chunk_pt_dev,
            rope_dev_bufs=rope_dev_bufs,
        )
        ttnn.synchronize_device(mesh_device)
        ttnn.execute_trace(mesh_device, prefill_trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh_device)
        _log_prefill_chunk_progress(
            isl=isl, phase="replay", chunk_idx=chunk_idx, num_chunks=num_chunks, phase_started_s=t0
        )
    prefill_replay_s = _time_tt_op(t0)
    logger.info(f"ISL={isl} prefill replay done: {prefill_replay_s:.3f}s (TTFT window)")

    prefill_logits.deallocate(True)
    ttnn.release_trace(mesh_device, prefill_trace_id)
    return prefill_compile_s, prefill_replay_s


def measure_isl_perf_point(
    mesh_device,
    model: TtMinistral3ForCausalLM,
    args: Devstral2Args,
    *,
    isl: int,
) -> dict:
    """Measure one ISL point (traced chunked prefill + decode trace, like ``text_demo``)."""
    _require_traced_prefill(args)

    prompt_len, padded_len, num_chunks = _padded_chunk_layout(isl)
    max_new_tokens = devstral2_max_new_tokens()
    decode_replay_iters = devstral2_isl_perf_decode_replay_iters()

    if prompt_len + max_new_tokens > args.max_seq_len:
        pytest.fail(
            f"ISL={isl}: prompt({prompt_len}) + max_new_tokens({max_new_tokens}) "
            f"exceeds max_seq_len={args.max_seq_len}"
        )

    kv_block_size = Devstral2Args.kv_block_size
    pad_id = 0
    input_ids_padded = torch.full((1, padded_len), pad_id, dtype=torch.long)

    prefill_chunk_dev = _input_ids_to_tt(
        torch.full((1, kv_block_size), pad_id, dtype=torch.long),
        mesh_device,
    )
    decode_tok_dev = _input_ids_to_tt(torch.zeros((1, 1), dtype=torch.long), mesh_device)
    decode_pos_dev = _current_pos_to_tt(torch.tensor([prompt_len], dtype=torch.long), mesh_device)

    decode_2cq: Optional[DecodeTrace2CQ] = None
    if decode_replay_iters > 0 and decode_trace_2cq_enabled():
        decode_2cq = DecodeTrace2CQ.create(mesh_device, decode_tok_dev, decode_pos_dev)

    rotary_emb = model.model.rotary_emb
    decode_trace_id = None

    logger.info(
        f"ISL={isl} start: padded={padded_len}, chunks={num_chunks}, " f"decode_replay_iters={decode_replay_iters}"
    )

    try:
        prefill_compile_s, prefill_replay_s = _run_traced_chunked_prefill(
            mesh_device,
            model,
            isl=isl,
            input_ids_padded=input_ids_padded,
            num_chunks=num_chunks,
            kv_block_size=kv_block_size,
            rotary_emb=rotary_emb,
            prefill_chunk_dev=prefill_chunk_dev,
        )

        ttft_s = prefill_replay_s
        prefill_tok_per_s = padded_len / prefill_replay_s if prefill_replay_s > 0 else 0.0

        decode_compile_s = 0.0
        decode_capture_s = 0.0
        decode_replay_total_s = 0.0
        decode_tok_per_s_per_user = 0.0

        if decode_replay_iters > 0:
            decode_token = 0
            decode_pos = prompt_len
            logger.info(f"ISL={isl} decode: compile + capture + {decode_replay_iters} replay(s)")

            t0 = time.perf_counter()
            stage_decode_inputs(decode_2cq, mesh_device, decode_tok_dev, decode_pos_dev, decode_token, decode_pos)
            dec_warm = model(decode_tok_dev, mode="decode", current_pos=decode_pos_dev)
            ttnn.synchronize_device(mesh_device)
            signal_decode_step_done(decode_2cq)
            decode_compile_s = _time_tt_op(t0)
            dec_warm.deallocate(True)

            t0 = time.perf_counter()
            stage_decode_inputs(decode_2cq, mesh_device, decode_tok_dev, decode_pos_dev, decode_token, decode_pos)
            decode_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            decode_logits = model(decode_tok_dev, mode="decode", current_pos=decode_pos_dev)
            ttnn.end_trace_capture(mesh_device, decode_trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            signal_decode_step_done(decode_2cq)
            decode_capture_s = _time_tt_op(t0)

            # ``text_demo``: timed window is execute_trace + sync only (stage_decode_inputs outside).
            for replay_idx in range(decode_replay_iters):
                replay_pos = decode_pos + replay_idx
                stage_decode_inputs(decode_2cq, mesh_device, decode_tok_dev, decode_pos_dev, decode_token, replay_pos)
                t_step = time.perf_counter()
                ttnn.execute_trace(mesh_device, decode_trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                signal_decode_step_done(decode_2cq)
                decode_replay_total_s += _time_tt_op(t_step)
            decode_tok_per_s_per_user = (
                decode_replay_iters / decode_replay_total_s if decode_replay_total_s > 0 else 0.0
            )
            decode_logits.deallocate(True)

        logger.info(
            f"ISL={isl:>8}  TTFT={ttft_s:>10.3f}s  "
            f"prefill={prefill_tok_per_s:>14.1f} tok/s  "
            f"decode={decode_tok_per_s_per_user:>16.2f} tok/s/u"
            if decode_replay_iters > 0
            else (
                f"ISL={isl:>8}  TTFT={ttft_s:>10.3f}s  "
                f"prefill={prefill_tok_per_s:>14.1f} tok/s  "
                f"decode={'n/a':>16}"
            )
        )

        return {
            "isl": isl,
            "padded_isl": padded_len,
            "num_prefill_chunks": num_chunks,
            "ttft_s": ttft_s,
            "prefill_replay_s": prefill_replay_s,
            "prefill_tok_per_s": prefill_tok_per_s,
            "max_new_tokens": max_new_tokens,
            "decode_replay_iters": decode_replay_iters,
            "decode_replay_total_s": decode_replay_total_s,
            "decode_tok_per_s_per_user": decode_tok_per_s_per_user,
            "prefill_compile_s": prefill_compile_s,
            "decode_compile_s": decode_compile_s,
            "decode_capture_s": decode_capture_s,
            "decode_trace_2cq": float(decode_2cq is not None),
            "model_max_seq_len": args.max_seq_len,
        }
    finally:
        if decode_trace_id is not None:
            ttnn.release_trace(mesh_device, decode_trace_id)


def _format_results_table(rows: list[dict]) -> str:
    header = f"{'ISL':>8}  {'TTFT (s)':>10}  {'Prefill tok/s':>14}  {'Decode tok/s/u':>16}"
    lines = [header, "-" * len(header)]
    for row in rows:
        decode_str = f"{row['decode_tok_per_s_per_user']:>16.2f}" if row["decode_replay_iters"] > 0 else f"{'n/a':>16}"
        lines.append(
            f"{row['isl']:>8}  " f"{row['ttft_s']:>10.3f}  " f"{row['prefill_tok_per_s']:>14.1f}  " f"{decode_str}"
        )
    return "\n".join(lines)


def _sweep_payload(
    rows: list[dict],
    *,
    label: str,
    isl_lengths: list[int],
    complete: bool,
) -> dict:
    max_new_tokens = devstral2_max_new_tokens()
    completed_isl = [row["isl"] for row in rows]
    pending_isl = [isl for isl in isl_lengths if isl not in completed_isl]
    return {
        "label": label,
        "complete": complete,
        "model_max_seq_len": rows[0]["model_max_seq_len"] if rows else devstral2_isl_perf_kv_max_seq_len(isl_lengths),
        "max_new_tokens": max_new_tokens,
        "decode_replay_cap": int(os.getenv("DEVSTRAL2_ISL_PERF_DECODE_REPLAY_CAP", "32")),
        "isl_lengths": isl_lengths,
        "completed_isl": completed_isl,
        "pending_isl": pending_isl,
        "results": rows,
    }


def _write_isl_point_json(row: dict, *, label: str) -> Path:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUTPUT_DIR / f"isl_perf_{label}_isl_{row['isl']}.json"
    out_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    logger.info(f"Wrote ISL perf point JSON to {out_path}")
    return out_path


def _write_sweep_json(
    rows: list[dict],
    *,
    label: str,
    isl_lengths: list[int],
    complete: bool = True,
) -> Path:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUTPUT_DIR / f"isl_perf_{label}.json"
    payload = _sweep_payload(rows, label=label, isl_lengths=isl_lengths, complete=complete)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    status = "complete" if complete else f"{len(rows)}/{len(isl_lengths)} points"
    logger.info(f"Wrote ISL perf sweep JSON ({status}) to {out_path}")
    return out_path


def run_isl_perf_sweep(
    mesh_device,
    isl_lengths: list[int],
    *,
    label: str = "sweep",
) -> list[dict]:
    """Run TT-only ISL perf for each length; one aligned KV build, reused weight cache."""
    model, args, weight_cache_path = get_or_build_isl_perf_model(mesh_device, isl_lengths)
    logger.info(
        f"ISL perf sweep: points={isl_lengths}, max_seq_len={args.max_seq_len}, " f"weight_cache={weight_cache_path}"
    )

    rows: list[dict] = []
    logger.info("ISL perf sweep columns:\n" + _format_results_table([]).split("\n")[0])
    for point_idx, isl in enumerate(isl_lengths):
        logger.info(f"ISL perf point {point_idx + 1}/{len(isl_lengths)}: ISL={isl}")
        row = measure_isl_perf_point(mesh_device, model, args, isl=isl)
        rows.append(row)
        _write_isl_point_json(row, label=label)
        _write_sweep_json(rows, label=label, isl_lengths=isl_lengths, complete=False)
        logger.info(f"ISL perf progress ({label}):\n{_format_results_table(rows)}")

    table = _format_results_table(rows)
    logger.info(f"ISL perf sweep results ({label}):\n{table}")
    _write_sweep_json(rows, label=label, isl_lengths=isl_lengths, complete=True)
    return rows


__all__ = [
    "ISL_PERF_FULL_SWEEP_TIMEOUT_SEC",
    "ISL_PERF_SANITY_TIMEOUT_SEC",
    "PREFILL_SANITY_SEQ_LENGTHS",
    "PREFILL_SWEEP_SEQ_LENGTHS",
    "build_isl_perf_model",
    "get_or_build_isl_perf_model",
    "isl_sweep_timeout_seconds",
    "measure_isl_perf_point",
    "run_isl_perf_sweep",
]
