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
from models.demos.qwen3_tts.tt.mesh_utils import to_torch as _mesh_to_torch


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
    # ── Fused single-trace CP frame (server.FusedCpState) ──────────────
    # When set, ar_decode_loop replaces the whole per-step CP block — input
    # prep, KV restore, prefill, every decode step, every sample, and the
    # Talker embedding build — with one execute_trace plus one 60-byte D2H.
    fused_cp: Optional[Any] = None
    # Per-step traces with in-trace device sampling (bisection path).
    cp_sampler: Optional[Any] = None
    # Canary allocated after trace capture (QWEN3_TTS_CANARY=1), to attribute
    # trace-replay memory corruption to a specific trace.
    canary_tt: Optional[Any] = None
    canary_ref: Optional[Any] = None
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
    return int(_mesh_to_torch(token_tt).flatten()[index].item())


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

    # (The old ``_device_cp_sampling`` hook is gone: device CP sampling is now a real
    # path — the fused frame by default, or per-step traces via QWEN3_TTS_CP_DEVSAMP=1.
    # Both go through state.cp_sampler / state.fused_cp.)
    # QWEN3_TTS_DEBUG_CODES=n prints the first n raw code rows (all 16 codebooks),
    # which is how the fused device path is compared against the host path.
    _debug_codes = int(os.environ.get("QWEN3_TTS_DEBUG_CODES", "0"))

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
        "cp_fused_h2d_ms": 0.0,
        "cp_fused_trace_ms": 0.0,
        "cp_fused_d2h_ms": 0.0,
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

    fused_cp = state.fused_cp
    _check_chips = False
    _check_corrupt = False
    _check_input = False
    _probe_steps = int(os.environ.get("QWEN3_TTS_CP_PROBE_STEPS", "0"))
    _inp_bad = [0, 0, 0, 0]
    _chip_mismatch = [0, 0, 0]
    _corrupt = [0, 0, 0, 0, 0]
    _mesh_mapper = None
    if fused_cp is not None:
        _dev_cls = device.__class__.__name__
        if _dev_cls == "MeshDevice" and device.get_num_devices() > 1:
            _mesh_mapper = ttnn.ReplicateTensorToMesh(device)
        _tok0_cpu = torch.zeros(1, 1, dtype=torch.int32)
        _n_cp_tokens = config.num_code_groups - 1
        # QWEN3_TTS_CP_CHECK_CHIPS=1: with TP>1 every chip runs the sampling kernel on
        # its own copy of the logits. If they ever disagree, each chip embeds a
        # DIFFERENT token on-device and the tensor-parallel halves silently diverge.
        _check_chips = os.environ.get("QWEN3_TTS_CP_CHECK_CHIPS", "0") == "1"
        _chip_mismatch = [0, 0, 0]
        _check_corrupt = os.environ.get("QWEN3_TTS_CP_CHECK_CORRUPT", "0") == "1"
        _check_input = os.environ.get("QWEN3_TTS_CP_CHECK_INPUT", "0") == "1"
        _inp_bad = [0, 0, 0, 0]
        talker_h_dbg = int(state.trace_embed_tt.shape[-1])
        _corrupt = [0, 0, 0, 0, 0]

    def _tbl_probe(tag, step):
        """Count clobbered rows in the device codec table. Bisection tool."""
        if fused_cp is None or fused_cp.codec_embed_tt is None:
            return
        t = fused_cp.codec_embed_tt
        if device.__class__.__name__ == "MeshDevice" and device.get_num_devices() > 1:
            d = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
            d = d[: codec_embed_torch.shape[0]].float()
        else:
            d = ttnn.to_torch(t).float()
        bad = (d != codec_embed_torch.bfloat16().float()).any(dim=-1)
        n = int(bad.sum())
        extra = ""
        if n:
            rows = bad.nonzero().flatten()
            r0 = int(rows[0])
            extra = (
                f"  rows {r0}..{int(rows[-1])}  first_row[:4]={[round(float(x),3) for x in d[r0, :4].tolist()]}"
                f"  sum={float(d[bad].sum()):.4f}"
            )
        print(f"  [probe] frame {step} {tag}: {n}/{d.shape[0]} wrong rows{extra}")

    for step in range(config.max_new_tokens):
        if use_2cq:
            ttnn.wait_for_event(1, trace_cq0_idle)
        else:
            ttnn.synchronize_device(device)
        t_step_start = time.time()
        _step_pc = time.perf_counter()

        # === CodePredictor: generate codes 1-15 ===
        if fused_cp is not None:
            # One trace does the entire CP frame. Host per frame: a fresh Gumbel
            # noise tile, the code-0 token id, this step's trailing-text row, one
            # execute_trace, and one D2H of the 15 sampled ids.
            fused_cp.sampler.refresh_noise()
            if not fused_cp.restores_in_trace:
                ttnn.assign(state.cp_trace_prefill_mask_host, state.cp_trace_prefill_mask_tt)
                ttnn.assign(state.cp_trace_prefill_cos_host, state.cp_trace_prefill_cos_tt)
                ttnn.assign(state.cp_trace_prefill_sin_host, state.cp_trace_prefill_sin_tt)
                for (k_zero, v_zero), (k_cache, v_cache) in zip(state.cp_kv_zero_hosts, state.cp_kv_caches_persistent):
                    ttnn.assign(k_zero, k_cache)
                    ttnn.assign(v_zero, v_cache)
            _tok0_cpu[0, 0] = token_0
            _tok0_host = ttnn.from_torch(
                _tok0_cpu, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=_mesh_mapper
            )
            ttnn.copy_host_to_device_tensor(_tok0_host, fused_cp.tok_bufs[0])
            ttnn.copy_host_to_device_tensor(fused_cp.trail_row_h2d[step], fused_cp.trail_row_tt)
            _t_fused_h2d = time.perf_counter()
            ttnn.execute_trace(device, fused_cp.trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            _t_fused_trace = time.perf_counter()
            if _check_input:
                # Compare what the fused trace actually built as the CP prefill input
                # against the host construction the per-step path uses.
                _dev_in = _mesh_to_torch(state.cp_trace_prefill_embed_tt).float().reshape(2, -1)
                _h = _mesh_to_torch(fused_cp.src_hidden_tt).float().reshape(-1)
                token_id_buf[0, 0] = token_0
                _e0 = F.embedding(token_id_buf, codec_embed_torch).reshape(-1)
                _exp = torch.stack([_h, _e0.bfloat16().float()])
                _d_h = float((_dev_in[0] - _exp[0]).abs().max())
                _d_e = float((_dev_in[1] - _exp[1]).abs().max())
                if _d_h > 0 or _d_e > 0:
                    _inp_bad[0] += 1
                    if _inp_bad[0] <= 3:
                        print(
                            f"  [input] frame {step}: cp_prefill_embed mismatch "
                            f"maxabs hidden={_d_h:.4g} code0_embed={_d_e:.4g}"
                        )
                        print(f"     dev row0[:4]={_dev_in[0][:4].tolist()} exp={_exp[0][:4].tolist()}")
                        print(f"     dev row1[:4]={_dev_in[1][:4].tolist()} exp={_exp[1][:4].tolist()}")
                _inp_bad[1] += 1

            if _check_chips:
                _all = ttnn.to_torch(fused_cp.tokens_out, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
                _rows = [_all[c].flatten()[:_n_cp_tokens].tolist() for c in range(_all.shape[0])]
                _dis = sum(1 for j in range(_n_cp_tokens) if len({r[j] for r in _rows}) > 1)
                _chip_mismatch[0] += _dis
                _chip_mismatch[1] += _n_cp_tokens
                if _dis and _chip_mismatch[2] < 5:
                    _chip_mismatch[2] += 1
                    print(f"  [chips] frame {step}: {_dis}/{_n_cp_tokens} tokens differ across chips")
                    for c, r in enumerate(_rows):
                        print(f"     chip{c}: {r}")
                _tok_row = _all[0].flatten()[:_n_cp_tokens]
            else:
                _tok_row = _mesh_to_torch(fused_cp.tokens_out).flatten()[:_n_cp_tokens]
            code_row = [token_0] + [int(t) for t in _tok_row.tolist()]
            if state.canary_tt is not None and step < 3:
                _cv = _mesh_to_torch(state.canary_tt).float()
                _n = int((_cv != state.canary_ref.float()).sum())
                print(
                    f"  [canary] frame {step} AFTER CP trace, BEFORE Talker trace: "
                    f"{'intact' if _n == 0 else str(_n) + ' elems clobbered'}"
                )
            if _check_input:
                # The decode-embed buffer still holds the LAST step's value, so this
                # checks the on-device ttnn.embedding at the end of the chain.
                _dev_e = _mesh_to_torch(state.cp_trace_decode_embed_tts[0]).float().reshape(-1)
                _last_tok = code_row[config.num_code_groups - 2]
                token_id_buf[0, 0] = _last_tok
                _tbl = code_pred_embeds[config.num_code_groups - 3]
                _exp_e = F.embedding(token_id_buf, _tbl).reshape(-1).bfloat16().float()
                _de = float((_dev_e - _exp_e).abs().max())
                if _de > 0:
                    _inp_bad[2] += 1
                    if _inp_bad[2] <= 3:
                        print(
                            f"  [input] frame {step}: LAST decode embed mismatch maxabs={_de:.4g} "
                            f"(token {_last_tok}); dev[:4]={_dev_e[:4].tolist()} exp={_exp_e[:4].tolist()}"
                        )
                if fused_cp.builds_talker_embed:
                    # The accumulated Talker input embedding the trace just wrote,
                    # against the host float32 sum the per-step path uses.
                    _dev_t = _mesh_to_torch(state.trace_embed_tt).float().reshape(-1)
                    _acc = torch.zeros(1, 1, talker_h_dbg, dtype=torch.float32)
                    for _i, _tk in enumerate(code_row):
                        token_id_buf[0, 0] = _tk
                        if _i == 0:
                            _acc += F.embedding(token_id_buf, codec_embed_torch)
                        elif _i - 1 < len(code_pred_embeds) and code_pred_embeds[_i - 1] is not None:
                            _acc += F.embedding(token_id_buf, code_pred_embeds[_i - 1])
                        else:
                            _acc += F.embedding(token_id_buf, codec_embed_torch)
                    _tl = trailing_text_hidden.shape[1]
                    _acc = _acc + (trailing_text_hidden[:, step : step + 1, :] if step < _tl else tts_pad_embed)
                    _exp_t = _acc.reshape(-1).bfloat16().float()
                    _dt = float((_dev_t - _exp_t).abs().max())
                    if _dt > 0:
                        _inp_bad[3] += 1
                        if _inp_bad[3] <= 3:
                            _nbad = int((_dev_t != _exp_t).sum())
                            print(
                                f"  [input] frame {step}: TALKER EMBED mismatch maxabs={_dt:.4g} "
                                f"({_nbad}/{_dev_t.numel()} elems)"
                            )
                            print(f"     dev[:4]={_dev_t[:4].tolist()}")
                            print(f"     exp[:4]={_exp_t[:4].tolist()}")
            _t_fused_d2h = time.perf_counter()
            frame_breakdown_sums["cp_fused_h2d_ms"] += (_t_fused_h2d - _step_pc) * 1000
            frame_breakdown_sums["cp_fused_trace_ms"] += (_t_fused_trace - _t_fused_h2d) * 1000
            frame_breakdown_sums["cp_fused_d2h_ms"] += (_t_fused_d2h - _t_fused_trace) * 1000
            _t_after_cp_input = _t_after_kv = _t_after_cp_prefill = _step_pc
            _t_after_cp_decode = _t_fused_d2h
            _prefill_sp = {}
            _decode_sp_agg = {"device_logits": 0.0, "cpu_sample": 0.0}
            t_cp_end = time.time()
        else:
            past_hidden_torch = _mesh_to_torch(talker_hidden_tt)[:, :, -1:, :].float()
            token_id_buf[0, 0] = token_0
            code0_embed = F.embedding(token_id_buf, codec_embed_torch).unsqueeze(1)
            cp_input = torch.cat([past_hidden_torch, code0_embed], dim=2)
            code_row: List[Any] = [token_0]
            _t_after_cp_input = time.perf_counter()

            if state.cp_sampler is not None:
                state.cp_sampler.refresh_noise()

            # Restore CP constants corrupted by Talker's paged_update_cache.
            # Source tensors are on device (not host) so we use ttnn.assign (D2D)
            # instead of copy_host_to_device_tensor (H2D) — same constant data,
            # no PCIe transfer needed, much faster.
            ttnn.assign(state.cp_trace_prefill_mask_host, state.cp_trace_prefill_mask_tt)
            ttnn.assign(state.cp_trace_prefill_cos_host, state.cp_trace_prefill_cos_tt)
            ttnn.assign(state.cp_trace_prefill_sin_host, state.cp_trace_prefill_sin_tt)
            for (k_zero, v_zero), (k_cache, v_cache) in zip(state.cp_kv_zero_hosts, state.cp_kv_caches_persistent):
                ttnn.assign(k_zero, k_cache)
                ttnn.assign(v_zero, v_cache)
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
            if (config.greedy or state.cp_sampler is not None) and state.cp_prefill_token_tt is not None:
                _t_pf0 = time.perf_counter()
                token = _read_device_token(state.cp_prefill_token_tt, index=0 if state.cp_sampler else 1)
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

                # H2D embed for this iteration's input.
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
                if (config.greedy or state.cp_sampler is not None) and state.cp_decode_token_tts is not None:
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

        if step < _debug_codes:
            print(f"  [codes] frame {step}: {code_row}")
        all_codes.append(code_row)
        if streaming_decoder is not None:
            streaming_decoder.add_tokens(torch.tensor(code_row, dtype=torch.long))
        if not use_2cq:
            ttnn.synchronize_device(device)
        t_cp_end = time.time()
        _t_after_cp_decode = time.perf_counter()

        # === Build next Talker input embedding ===
        # Fused path: the CP trace already accumulated it on device (in float32,
        # bit-exact with the host sum) straight into state.trace_embed_tt.
        _t_embed0 = time.perf_counter()
        if fused_cp is None or not fused_cp.builds_talker_embed:
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

        if fused_cp is None or not fused_cp.builds_talker_embed:
            talker_embed_cpu.copy_(next_embed.bfloat16())
            embed_host = ttnn.from_torch(talker_embed_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            if use_2cq:
                ttnn.wait_for_event(1, trace_cq0_idle)
            ttnn.copy_host_to_device_tensor(embed_host, state.trace_embed_tt, cq_id=h2d_cq)
        elif use_2cq:
            ttnn.wait_for_event(1, trace_cq0_idle)
        if _probe_steps and step < _probe_steps:
            _tbl_probe("before Talker H2Ds", step)
        ttnn.copy_host_to_device_tensor(cos_host, state.trace_cos_tt, cq_id=h2d_cq)
        ttnn.copy_host_to_device_tensor(sin_host, state.trace_sin_tt, cq_id=h2d_cq)
        ttnn.copy_host_to_device_tensor(cur_pos_host, state.trace_cur_pos_tt, cq_id=h2d_cq)
        ttnn.copy_host_to_device_tensor(mask_host, state.trace_mask_tt, cq_id=h2d_cq)
        if use_2cq:
            write_ev = ttnn.record_event(device, 1)
            ttnn.wait_for_event(0, write_ev)
        if _probe_steps and step < _probe_steps:
            ttnn.synchronize_device(device)
            _tbl_probe("after Talker H2Ds, before trace", step)
        ttnn.execute_trace(device, state.talker_decode_trace_id, cq_id=0, blocking=False)
        if _probe_steps and step < _probe_steps:
            ttnn.synchronize_device(device)
            _tbl_probe("after Talker trace", step)
            # Is the per-frame "restore CP constants" hack still needed? Compare each
            # constant against its pristine on-device source right after the Talker
            # trace, which is where the corruption was believed to happen.
            for _nm, _live, _src in (
                ("cp mask", state.cp_trace_prefill_mask_tt, state.cp_trace_prefill_mask_host),
                ("cp cos", state.cp_trace_prefill_cos_tt, state.cp_trace_prefill_cos_host),
                ("cp sin", state.cp_trace_prefill_sin_tt, state.cp_trace_prefill_sin_host),
            ):
                _a = _mesh_to_torch(_live).float()
                _b = _mesh_to_torch(_src).float()
                _nb = int((_a != _b).sum())
                print(
                    f"  [probe] frame {step} {_nm} after Talker trace: "
                    f"{'INTACT' if _nb == 0 else str(_nb) + '/' + str(_a.numel()) + ' elems differ'}"
                )
            for _li, ((_kz, _vz), (_kc, _vc)) in enumerate(zip(state.cp_kv_zero_hosts, state.cp_kv_caches_persistent)):
                if _li:
                    break
                _nk = int((_mesh_to_torch(_kc).float() != _mesh_to_torch(_kz).float()).sum())
                print(
                    f"  [probe] frame {step} cp k-cache[0] vs zero source: {_nk} elems differ "
                    f"(nonzero is EXPECTED — the CP forward writes it)"
                )
        talker_hidden_tt = state.trace_hidden_out
        if fused_cp is not None:
            # Snapshot the Talker hidden into the fused CP trace's own input buffer.
            # See FusedCpState.src_hidden_tt: the trace must not read another trace's
            # internal output tensor.
            ttnn.assign(state.trace_hidden_out, fused_cp.src_hidden_tt)
        talker_pos += 1
        if use_2cq:
            trace_cq0_idle = ttnn.record_event(device, 0)
        else:
            ttnn.synchronize_device(device)
        t_talker_end = time.time()
        state.talker_times_ms.append((t_talker_end - t_cp_end) * 1000)
        state.cp_times_ms.append((t_cp_end - t_step_start) * 1000)

        if _check_corrupt and fused_cp is not None:
            # The Talker's paged_fused_update_cache is already known to clobber the CP
            # prefill constants (hence the per-frame ttnn.assign restore). Check it is
            # not also clobbering the buffers the fused CP trace depends on.
            _sm = fused_cp.sampler
            _n_dev = _mesh_to_torch(_sm.noise_tt).float()
            _n_exp = _sm._noise_cpu.bfloat16().float()
            if step < 6 or step % 20 == 0:
                # NB: _mesh_to_torch assumes rank-4 (it slices dim 0), so read the
                # rank-2 table directly with the mesh composer instead.
                _t = fused_cp.codec_embed_tt
                if device.__class__.__name__ == "MeshDevice" and device.get_num_devices() > 1:
                    _tbl_dev = ttnn.to_torch(_t, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
                    _tbl_dev = _tbl_dev[: codec_embed_torch.shape[0]].float()
                else:
                    _tbl_dev = ttnn.to_torch(_t).float()
                _tbl_exp = codec_embed_torch.bfloat16().float()
                _rowbad = (_tbl_dev != _tbl_exp).any(dim=-1)
                _nb = int(_rowbad.sum())
                if _nb:
                    _corrupt[4] += 1
                    _rows = _rowbad.nonzero().flatten()
                    print(
                        f"  [corrupt] frame {step}: CODEC TABLE has {_nb}/{_tbl_dev.shape[0]} wrong rows, "
                        f"first={_rows[:6].tolist()} last={_rows[-3:].tolist()} "
                        f"maxabs={float((_tbl_dev - _tbl_exp).abs().max()):.4g}"
                    )
            _hs_dev = _mesh_to_torch(fused_cp.src_hidden_tt).float().reshape(-1)
            _hs_exp = _mesh_to_torch(state.trace_hidden_out).float().reshape(-1)
            if not torch.equal(_hs_dev, _hs_exp):
                _corrupt[3] += 1
                if _corrupt[3] <= 3:
                    print(
                        f"  [corrupt] frame {step}: SRC HIDDEN != trace_hidden_out "
                        f"maxabs={float((_hs_dev - _hs_exp).abs().max()):.4g}"
                    )
            if not torch.equal(_n_dev, _n_exp):
                _corrupt[0] += 1
                if _corrupt[0] <= 3:
                    _bad = (_n_dev != _n_exp).sum().item()
                    print(
                        f"  [corrupt] frame {step}: noise tile changed after Talker trace "
                        f"({_bad} of {_n_dev.numel()} elems); dev[0,0,0,:4]={_n_dev[0,0,0,:4].tolist()} "
                        f"exp={_n_exp[0,0,0,:4].tolist()}"
                    )
            _corrupt[1] += 1

        if state.canary_tt is not None and step < 3:
            _cv = _mesh_to_torch(state.canary_tt).float()
            _cr = state.canary_ref.float()
            _bad = _cv != _cr
            _n = int(_bad.sum())
            if _n:
                _rows = _bad.any(dim=-1).flatten().nonzero().flatten()
                print(
                    f"  [canary] frame {step} AFTER Talker trace: {_n}/{_cv.numel()} elems clobbered, "
                    f"rows {int(_rows[0])}..{int(_rows[-1])} of {_cv.shape[2]}"
                )
            else:
                print(f"  [canary] frame {step} AFTER Talker trace: intact")

        # Get next code 0 from Talker trace output.
        # Device-sampling fast path: when state.talker_codec0_token_tt is
        # populated, the Talker decode trace already produced the sampled
        # token (via the in-trace topk + ttnn.sampling pipeline). We just
        # need a small int D2H instead of a full vocab D2H blocking on the
        # async Talker exec.
        _c0_sp: dict = {}
        if state.talker_codec0_token_tt is not None:
            _t_c00 = time.perf_counter()
            token_0 = _read_device_token(state.talker_codec0_token_tt, index=0)
            _c0_sp["device_logits"] = time.perf_counter() - _t_c00
            _t_after_codec0_d2h = time.perf_counter()
            _t_after_codec0_cpu = _t_after_codec0_d2h
        else:
            _codec0_logits_torch = _mesh_to_torch(state.trace_codec_logits_out, dtype=torch.float32)
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

    # Write back mutated state for callers that want to inspect.
    state.talker_hidden_tt = talker_hidden_tt
    state.talker_pos = talker_pos
    state.token_0 = token_0

    if fused_cp is not None and _check_chips:
        print(f"  [chips] total cross-chip token disagreements: {_chip_mismatch[0]}/{_chip_mismatch[1]}")
    if fused_cp is not None and _check_input:
        print(
            f"  [input] frames with a wrong CP prefill input: {_inp_bad[0]}/{_inp_bad[1]}; "
            f"wrong last decode embed: {_inp_bad[2]}/{_inp_bad[1]}; "
            f"wrong Talker embed: {_inp_bad[3]}/{_inp_bad[1]}"
        )
    if fused_cp is not None and _check_corrupt:
        print(
            f"  [corrupt] noise tile: {_corrupt[0]}/{_corrupt[1]}  src hidden: {_corrupt[3]}/{_corrupt[1]}  "
            f"codec table clobbered on {_corrupt[4]} checked frames"
        )

    frame_breakdown_avg_ms = (
        {k: v / frame_breakdown_frames for k, v in frame_breakdown_sums.items()} if frame_breakdown_frames > 0 else {}
    )

    if not all_codes:
        return None, frame_breakdown_avg_ms, t_first_decode_end, t_last_step_end

    codes = torch.tensor(all_codes, dtype=torch.long)
    return codes, frame_breakdown_avg_ms, t_first_decode_end, t_last_step_end
