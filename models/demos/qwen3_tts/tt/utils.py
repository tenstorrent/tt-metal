# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the qwen3_tts inference pipeline.

The AR decode loop in particular is consumed by two call sites:

- ``generate_codes_ttnn`` (in ``tt/server.py``) — one-shot demo path that
  captures all traces inline and runs a single inference.
- ``run_inference`` (in ``tt/server.py``) — server path that reuses a
  pre-built ``TTSServerContext`` (traces + KV caches captured once at
  startup) across many requests.

Both flows produce the same per-step work; the only difference is whether
the trace state was captured per-request or pre-captured. To keep them in
sync, the AR loop body lives here as ``ar_decode_loop`` and consumes a
fully populated ``DecodeLoopState``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

import ttnn


@dataclass
class DecodeLoopState:
    """All state the AR decode loop reads or mutates.

    Built once per inference call. Field grouping mirrors the loop's order
    of use: input plumbing → CP traces → Talker traces → host scratch →
    per-request running state.
    """

    # ── Device / queueing ──────────────────────────────────────────────
    device: Any
    # ── CP traces + persistent state (per-bucket) ──────────────────────
    cp_kv_caches_persistent: List[Tuple[Any, Any]]
    cp_kv_zero_hosts: List[Tuple[Any, Any]]
    cp_prefill_trace_id: int
    cp_prefill_logits_tt: Any
    cp_decode_trace_ids: List[List[int]]  # [[cq0_ids], [cq1_ids]]
    cp_decode_logits_tts: List[List[Any]]
    cp_trace_prefill_embed_tt: Any
    cp_trace_prefill_mask_tt: Any
    cp_trace_prefill_cos_tt: Any
    cp_trace_prefill_sin_tt: Any
    cp_trace_prefill_mask_host: Any
    cp_trace_prefill_cos_host: Any
    cp_trace_prefill_sin_host: Any
    cp_trace_decode_embed_tts: List[Any]
    code_pred_embeds: List[Optional[torch.Tensor]]
    codec_embed_torch: torch.Tensor
    # ── Talker decode trace + buffers (per-bucket) ─────────────────────
    talker_decode_trace_id: int
    trace_embed_tt: Any
    trace_cos_tt: Any
    trace_sin_tt: Any
    trace_cur_pos_tt: Any
    trace_mask_tt: Any
    trace_hidden_out: Any  # baked Talker hidden output tensor
    trace_codec_logits_out: Any  # baked codec0 logits tensor
    # ── Talker per-frame H2D constants (precomputed cos/sin/mask/pos) ──
    talker_cos_h2d: List[Any]
    talker_sin_h2d: List[Any]
    talker_mask_h2d: List[Any]
    talker_cur_pos_h2d: List[Any]
    # ── Pre-allocated host scratch buffers ─────────────────────────────
    token_id_buf: torch.Tensor
    cp_prefill_embed_cpu: torch.Tensor
    cp_decode_embed_cpu: torch.Tensor
    talker_embed_cpu: torch.Tensor
    acc_code_embed: torch.Tensor
    # ── Per-request initial state (mutated by the loop) ────────────────
    talker_hidden_tt: Any
    talker_pos: int
    real_seq_len: int
    trailing_text_hidden: torch.Tensor
    tts_pad_embed: torch.Tensor
    token_0: int
    # ── Optional in-trace argmax token buffers (greedy fast path) ──────
    cp_prefill_token_tt: Optional[Any] = None
    cp_decode_token_tts: Optional[List[List[Any]]] = None
    talker_codec0_token_tt: Optional[Any] = None
    # ── Per-loop accumulators / outputs (filled by ar_decode_loop) ─────
    decode_step_times_ms: List[float] = field(default_factory=list)
    talker_times_ms: List[float] = field(default_factory=list)
    cp_times_ms: List[float] = field(default_factory=list)


def _read_device_token(token_tt: Any, index: int = 0) -> int:
    """Pull a single int token from a 1-element-or-shape ttnn tensor."""
    return int(ttnn.to_torch(token_tt).flatten()[index].item())


def ar_decode_loop(
    state: DecodeLoopState,
    config: Any,
    use_2cq: bool,
    *,
    streaming_decoder: Optional[Any] = None,
    sample_token_fn: Any,
    sample_from_tt_vocab_logits_fn: Any,
) -> Tuple[Optional[torch.Tensor], dict, float, float]:
    """Run the autoregressive decode loop.

    ``sample_token_fn`` and ``sample_from_tt_vocab_logits_fn`` are passed in
    rather than imported here to avoid a circular import with ``tt/server.py``
    (which defines them).

    Returns ``(codes, frame_breakdown_avg_ms, t_first_decode_end, t_last_step_end)``
    where ``codes`` is a ``[num_frames, num_code_groups]`` long tensor (or
    ``None`` if no frames generated).
    """
    device = state.device
    h2d_cq = 1 if use_2cq else 0
    trace_cq0_idle = ttnn.record_event(device, 0) if use_2cq else None
    cp_decode_input_ready = [trace_cq0_idle, trace_cq0_idle]

    # Env-gated on-device optimizations (same flags the demo loop honours).
    _device_cp_chain = bool(int(os.environ.get("TT_QWEN3_DEVICE_CP_CHAIN", "0")))
    _device_cp_sampling = False  # batch=1 regression, kept disabled.

    # Phase timers (TT_QWEN3_PHASE_TIMERS=1 → print rolling means every 20 steps).
    _phase_dbg = bool(int(os.environ.get("TT_QWEN3_PHASE_TIMERS", "0")))
    _phase_acc = {
        "past_h": 0.0,
        "cp_restore": 0.0,
        "cp_prefill": 0.0,
        "cp_decode": 0.0,
        "build_emb": 0.0,
        "talker_launch": 0.0,
        "codec0_d2h": 0.0,
        "codec0_cpu": 0.0,
    }
    _phase_n = 0

    frame_breakdown_sums = {
        "cp_input_prep_ms": 0.0,
        "cp_kv_restore_ms": 0.0,
        "cp_prefill_ms": 0.0,
        "cp_decode_ms": 0.0,
        "build_acc_embed_ms": 0.0,
        "talker_decode_ms": 0.0,
        "codec0_sample_device_logits_ms": 0.0,
        "codec0_sample_cpu_ms": 0.0,
        "cp_prefill_sample_device_logits_ms": 0.0,
        "cp_prefill_sample_cpu_ms": 0.0,
        "cp_decode_samples_device_logits_ms": 0.0,
        "cp_decode_samples_cpu_ms": 0.0,
    }
    frame_breakdown_frames = 0

    all_codes: List[List[int]] = []
    generated_code0_tokens: List[int] = []
    t_first_decode_end = 0.0
    t_last_step_end = 0.0

    # Local aliases (hot-loop hygiene; avoid attribute lookup per iteration).
    talker_hidden_tt = state.talker_hidden_tt
    talker_pos = state.talker_pos
    token_0 = state.token_0
    token_id_buf = state.token_id_buf
    codec_embed_torch = state.codec_embed_torch
    code_pred_embeds = state.code_pred_embeds
    cp_prefill_embed_cpu = state.cp_prefill_embed_cpu
    cp_decode_embed_cpu = state.cp_decode_embed_cpu
    talker_embed_cpu = state.talker_embed_cpu
    acc_code_embed = state.acc_code_embed
    real_seq_len = state.real_seq_len
    trailing_text_hidden = state.trailing_text_hidden
    tts_pad_embed = state.tts_pad_embed

    for step in range(config.max_new_tokens):
        if use_2cq:
            ttnn.wait_for_event(1, trace_cq0_idle)
        else:
            ttnn.synchronize_device(device)
        t_step_start = time.time()
        _step_pc = time.perf_counter()

        # === CodePredictor: generate codes 1-15 ===
        past_hidden_torch = ttnn.to_torch(talker_hidden_tt)[:, :, -1:, :].float()
        token_id_buf[0, 0] = token_0
        code0_embed = F.embedding(token_id_buf, codec_embed_torch).unsqueeze(1)
        cp_input = torch.cat([past_hidden_torch, code0_embed], dim=2)
        code_row: List[Any] = [token_0]
        _t_after_cp_input = time.perf_counter()

        # Restore CP constants corrupted by Talker's paged_update_cache.
        ttnn.copy_host_to_device_tensor(state.cp_trace_prefill_mask_host, state.cp_trace_prefill_mask_tt, cq_id=h2d_cq)
        ttnn.copy_host_to_device_tensor(state.cp_trace_prefill_cos_host, state.cp_trace_prefill_cos_tt, cq_id=h2d_cq)
        ttnn.copy_host_to_device_tensor(state.cp_trace_prefill_sin_host, state.cp_trace_prefill_sin_tt, cq_id=h2d_cq)
        for (k_zero, v_zero), (k_cache, v_cache) in zip(state.cp_kv_zero_hosts, state.cp_kv_caches_persistent):
            ttnn.copy_host_to_device_tensor(k_zero, k_cache, cq_id=h2d_cq)
            ttnn.copy_host_to_device_tensor(v_zero, v_cache, cq_id=h2d_cq)
        _t_after_kv = time.perf_counter()

        # CP prefill trace.
        cp_prefill_embed_cpu.copy_(cp_input.bfloat16())
        pfembed_host = ttnn.from_torch(cp_prefill_embed_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(pfembed_host, state.cp_trace_prefill_embed_tt, cq_id=h2d_cq)
        if use_2cq:
            write_ev = ttnn.record_event(device, 1)
            ttnn.wait_for_event(0, write_ev)
        ttnn.execute_trace(device, state.cp_prefill_trace_id, cq_id=0, blocking=False)
        if use_2cq:
            trace_cq0_idle = ttnn.record_event(device, 0)
        else:
            ttnn.synchronize_device(device)

        _prefill_sp: dict = {}
        if config.greedy and state.cp_prefill_token_tt is not None:
            _t_pf0 = time.perf_counter()
            token = _read_device_token(state.cp_prefill_token_tt, index=1)
            _prefill_sp["device_logits"] = time.perf_counter() - _t_pf0
        else:
            _pf_vocab = state.cp_prefill_logits_tt.shape[3]
            last_prefill_logits = ttnn.slice(state.cp_prefill_logits_tt, [0, 0, 1, 0], [1, 1, 2, _pf_vocab])
            token = sample_from_tt_vocab_logits_fn(
                last_prefill_logits,
                temperature=config.temperature,
                top_k=config.top_k,
                greedy=config.greedy,
                prof_acc=_prefill_sp,
            )
            ttnn.deallocate(last_prefill_logits)
        code_row.append(token)
        _t_after_cp_prefill = time.perf_counter()

        # CP decode traces (num_code_groups - 2 of them, double-buffered with 2cq).
        _decode_sp_agg = {"device_logits": 0.0, "cpu_sample": 0.0}
        for _trace_i, code_idx in enumerate(range(2, config.num_code_groups)):
            _buf_i = (_trace_i % 2) if use_2cq else 0

            # H2D embed for this iteration's input. Chain mode skips H2D for trace_i>=1
            # (the previous trace's in-trace ttnn.embedding wrote our buffer).
            _need_h2d = (not _device_cp_chain) or _trace_i == 0
            if _need_h2d:
                prev_embed_idx = code_idx - 2
                token_id_buf[0, 0] = token
                if prev_embed_idx < len(code_pred_embeds) and code_pred_embeds[prev_embed_idx] is not None:
                    next_embed = F.embedding(token_id_buf, code_pred_embeds[prev_embed_idx])
                else:
                    next_embed = F.embedding(token_id_buf, codec_embed_torch)
                next_embed = next_embed.unsqueeze(1).bfloat16()

                cp_decode_embed_cpu.copy_(next_embed)
                e_h = ttnn.from_torch(cp_decode_embed_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                if use_2cq:
                    ttnn.wait_for_event(1, cp_decode_input_ready[_buf_i])
                ttnn.copy_host_to_device_tensor(e_h, state.cp_trace_decode_embed_tts[_buf_i], cq_id=h2d_cq)
                if use_2cq:
                    write_ev = ttnn.record_event(device, 1)
                    ttnn.wait_for_event(0, write_ev)
            ttnn.execute_trace(device, state.cp_decode_trace_ids[_buf_i][_trace_i], cq_id=0, blocking=False)
            if use_2cq:
                cp_decode_input_ready[_buf_i] = ttnn.record_event(device, 0)
                trace_cq0_idle = cp_decode_input_ready[_buf_i]

            _dsp: dict = {}
            if _device_cp_chain:
                code_row.append(None)  # resolved after loop with aggregated D2H
                continue
            elif (config.greedy or _device_cp_sampling) and state.cp_decode_token_tts is not None:
                _t_dc0 = time.perf_counter()
                token = _read_device_token(state.cp_decode_token_tts[_buf_i][_trace_i], index=0)
                _dsp["device_logits"] = time.perf_counter() - _t_dc0
            else:
                token = sample_from_tt_vocab_logits_fn(
                    state.cp_decode_logits_tts[_buf_i][_trace_i],
                    temperature=config.temperature,
                    top_k=config.top_k,
                    greedy=config.greedy,
                    prof_acc=_dsp,
                )
            _decode_sp_agg["device_logits"] += _dsp.get("device_logits", 0.0)
            _decode_sp_agg["cpu_sample"] += _dsp.get("cpu_sample", 0.0)
            code_row.append(token)

        # Aggregated D2H for chain mode: dispatch all 14 traces, then read tokens.
        if _device_cp_chain and state.cp_decode_token_tts is not None:
            _t_dc0 = time.perf_counter()
            for _ti in range(config.num_code_groups - 2):
                _bi = (_ti % 2) if use_2cq else 0
                code_row[2 + _ti] = _read_device_token(state.cp_decode_token_tts[_bi][_ti], index=0)
            _decode_sp_agg["device_logits"] += time.perf_counter() - _t_dc0

        all_codes.append(code_row)
        if streaming_decoder is not None:
            streaming_decoder.add_tokens(torch.tensor(code_row, dtype=torch.long))
        if not use_2cq:
            ttnn.synchronize_device(device)
        t_cp_end = time.time()
        _t_after_cp_decode = time.perf_counter()

        # === Build next Talker input embedding (host F.embedding accumulator) ===
        _t_embed0 = time.perf_counter()
        acc_code_embed.zero_()
        for i, tok in enumerate(code_row):
            token_id_buf[0, 0] = tok
            if i == 0:
                acc_code_embed += F.embedding(token_id_buf, codec_embed_torch)
            else:
                if i - 1 < len(code_pred_embeds) and code_pred_embeds[i - 1] is not None:
                    acc_code_embed += F.embedding(token_id_buf, code_pred_embeds[i - 1])
                else:
                    acc_code_embed += F.embedding(token_id_buf, codec_embed_torch)
        next_embed = acc_code_embed
        trailing_len = trailing_text_hidden.shape[1]
        if step < trailing_len:
            next_embed = next_embed + trailing_text_hidden[:, step : step + 1, :]
        else:
            next_embed = next_embed + tts_pad_embed
        next_embed = next_embed.unsqueeze(1)
        _t_after_build_embed = time.perf_counter()

        # === Talker decode trace ===
        _talker_h2d_i = talker_pos - real_seq_len
        cos_host = state.talker_cos_h2d[_talker_h2d_i]
        sin_host = state.talker_sin_h2d[_talker_h2d_i]
        mask_host = state.talker_mask_h2d[_talker_h2d_i]
        cur_pos_host = state.talker_cur_pos_h2d[_talker_h2d_i]

        talker_embed_cpu.copy_(next_embed.bfloat16())
        embed_host = ttnn.from_torch(talker_embed_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if use_2cq:
            ttnn.wait_for_event(1, trace_cq0_idle)
        ttnn.copy_host_to_device_tensor(embed_host, state.trace_embed_tt, cq_id=h2d_cq)
        ttnn.copy_host_to_device_tensor(cos_host, state.trace_cos_tt, cq_id=h2d_cq)
        ttnn.copy_host_to_device_tensor(sin_host, state.trace_sin_tt, cq_id=h2d_cq)
        ttnn.copy_host_to_device_tensor(cur_pos_host, state.trace_cur_pos_tt, cq_id=h2d_cq)
        ttnn.copy_host_to_device_tensor(mask_host, state.trace_mask_tt, cq_id=h2d_cq)
        if use_2cq:
            write_ev = ttnn.record_event(device, 1)
            ttnn.wait_for_event(0, write_ev)
        ttnn.execute_trace(device, state.talker_decode_trace_id, cq_id=0, blocking=False)
        talker_hidden_tt = state.trace_hidden_out
        talker_pos += 1
        if use_2cq:
            trace_cq0_idle = ttnn.record_event(device, 0)
        else:
            ttnn.synchronize_device(device)
        t_talker_end = time.time()
        _t_talker_end_pc = time.perf_counter()
        state.talker_times_ms.append((t_talker_end - t_cp_end) * 1000)
        state.cp_times_ms.append((t_cp_end - t_step_start) * 1000)

        # Get next code 0 from Talker trace output.
        _c0_sp: dict = {}
        _t_before_codec0 = time.perf_counter()
        if config.greedy and config.repetition_penalty == 1.0 and state.talker_codec0_token_tt is not None:
            _t_c00 = time.perf_counter()
            token_0 = _read_device_token(state.talker_codec0_token_tt, index=0)
            _c0_sp["device_logits"] = time.perf_counter() - _t_c00
            _t_after_codec0_d2h = time.perf_counter()
            _t_after_codec0_cpu = _t_after_codec0_d2h
        else:
            _codec0_logits_torch = ttnn.to_torch(state.trace_codec_logits_out, dtype=torch.float32)
            _t_after_codec0_d2h = time.perf_counter()
            token_0 = sample_token_fn(
                _codec0_logits_torch.flatten(),
                config.temperature,
                config.top_k,
                config.greedy,
                config.repetition_penalty,
                generated_code0_tokens,
            )
            _t_after_codec0_cpu = time.perf_counter()
        generated_code0_tokens.append(token_0)

        # Phase timer (rolling means).
        if _phase_dbg and step >= 1:
            _phase_acc["past_h"] += (_t_after_cp_input - _step_pc) * 1000
            _phase_acc["cp_restore"] += (_t_after_kv - _t_after_cp_input) * 1000
            _phase_acc["cp_prefill"] += (_t_after_cp_prefill - _t_after_kv) * 1000
            _phase_acc["cp_decode"] += (_t_after_cp_decode - _t_after_cp_prefill) * 1000
            _phase_acc["build_emb"] += (_t_after_build_embed - _t_embed0) * 1000
            _phase_acc["talker_launch"] += (_t_talker_end_pc - _t_after_build_embed) * 1000
            _phase_acc["codec0_d2h"] += (_t_after_codec0_d2h - _t_before_codec0) * 1000
            _phase_acc["codec0_cpu"] += (_t_after_codec0_cpu - _t_after_codec0_d2h) * 1000
            _phase_n += 1

        # Frame breakdown (printed at end by caller).
        frame_breakdown_sums["cp_input_prep_ms"] += (_t_after_cp_input - _step_pc) * 1000
        frame_breakdown_sums["cp_kv_restore_ms"] += (_t_after_kv - _t_after_cp_input) * 1000
        frame_breakdown_sums["cp_prefill_ms"] += (_t_after_cp_prefill - _t_after_kv) * 1000
        frame_breakdown_sums["cp_decode_ms"] += (_t_after_cp_decode - _t_after_cp_prefill) * 1000
        frame_breakdown_sums["build_acc_embed_ms"] += (_t_after_build_embed - _t_embed0) * 1000
        frame_breakdown_sums["talker_decode_ms"] += (t_talker_end - t_cp_end) * 1000
        frame_breakdown_sums["codec0_sample_device_logits_ms"] += _c0_sp.get("device_logits", 0.0) * 1000
        frame_breakdown_sums["codec0_sample_cpu_ms"] += _c0_sp.get("cpu_sample", 0.0) * 1000
        frame_breakdown_sums["cp_prefill_sample_device_logits_ms"] += _prefill_sp.get("device_logits", 0.0) * 1000
        frame_breakdown_sums["cp_prefill_sample_cpu_ms"] += _prefill_sp.get("cpu_sample", 0.0) * 1000
        frame_breakdown_sums["cp_decode_samples_device_logits_ms"] += _decode_sp_agg.get("device_logits", 0.0) * 1000
        frame_breakdown_sums["cp_decode_samples_cpu_ms"] += _decode_sp_agg.get("cpu_sample", 0.0) * 1000
        frame_breakdown_frames += 1

        if token_0 == config.codec_eos_id:
            print(f"  EOS at step {step + 1}")
            break

        if not use_2cq:
            ttnn.synchronize_device(device)
        t_step_end = time.time()
        step_ms = (t_step_end - t_step_start) * 1000
        if step == 0:
            t_first_decode_end = t_step_end
        t_last_step_end = t_step_end
        state.decode_step_times_ms.append(step_ms)

        if (step + 1) % 20 == 0:
            print(f"  Generated {step + 1} frames...")
            if _phase_dbg and _phase_n > 0:
                parts = " ".join(f"{k}={_phase_acc[k] / _phase_n:.2f}" for k in _phase_acc)
                print(f"  [PHASE_MS, n={_phase_n}, mean per step] {parts}")

    # Write back mutated state for callers that want to inspect.
    state.talker_hidden_tt = talker_hidden_tt
    state.talker_pos = talker_pos
    state.token_0 = token_0

    frame_breakdown_avg_ms = (
        {k: v / frame_breakdown_frames for k, v in frame_breakdown_sums.items()} if frame_breakdown_frames > 0 else {}
    )

    if not all_codes:
        return None, frame_breakdown_avg_ms, t_first_decode_end, t_last_step_end

    codes = torch.tensor(all_codes, dtype=torch.long)
    return codes, frame_breakdown_avg_ms, t_first_decode_end, t_last_step_end
