# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Metal-trace capture/replay for the EXTENDED Trace A: prosody→duration **plus the ASR TextEncoder**.

Trace A splits the Kokoro pipeline at the duration readback. The ASR ``TextEncoder`` (kmodel
``text_encoder``: embedding→CNN→BiLSTM) consumes only ``input_ids``/``input_lengths`` and emits
``t_en_bct`` = ``[B, C, T_tokens]`` — a *fixed shape at bucketed T_tokens*, independent of the
alignment/durations. So it belongs in Trace A's shape regime, not the decoder's (T_aligned). Folding
it in lets one capture swallow the encoder's ~480 host-driven BiLSTM dispatches.

This proves the extended region is trace-capturable and bit-identical on replay for BOTH outputs
(``dur_clipped`` and ``t_en_bct``), before wiring it into ``TTKModel``. Full-length (no padding) path
only — that path is write-free during capture (ids cached under trace-prep, no mask upload, LSTM
``anti``/zero-states cached/traced, no valid-mask).

Run::

    pytest models/experimental/kokoro/tests/test_tt_prosody_asr_trace.py -s
"""

from __future__ import annotations

import time

import pytest
import torch

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.model import KModel
from models.experimental.kokoro.tests.test_tt_kmodel_pcc import _find_checkpoint, _phonemize
from models.experimental.kokoro.tt.tt_conv import clear_trace_weight_prep_cache, set_trace_weight_prep
from models.experimental.kokoro.tt.tt_kmodel import (
    KokoroConfig,
    TTKModel,
    _attention_keep_mask_bt,
    _batch_is_full_length,
    preprocess_tt_kmodel,
)
from models.experimental.kokoro.tt.tt_lstm import tt_bilstm_nlc

_TRACE_REGION_SIZE = 200_000_000
_L1_SMALL_SIZE = 98304
_REPLAY_ITERS = 5
_EAGER_TIMING_ITERS = 3


@pytest.mark.parametrize(
    "device",
    [{"trace_region_size": _TRACE_REGION_SIZE, "l1_small_size": _L1_SMALL_SIZE}],
    indirect=True,
)
def test_tt_prosody_and_asr_metal_trace(device):
    """Capture prosody→duration + ASR TextEncoder in ONE trace; replay; check parity of both outputs."""
    ckpt = _find_checkpoint()
    if ckpt is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    phonemes, ref_s = _phonemize("Hello from Tenstorrent.")
    ref_s = ref_s.cpu()
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)

    ref = KModel(repo_id=KokoroConfig.repo_id, model=str(ckpt)).eval()
    params = preprocess_tt_kmodel(ref, device)
    model = TTKModel(device, ref, params)
    p = model.params
    ck = model._predictor.compute_kernel_config
    mc = ttnn.DRAM_MEMORY_CONFIG

    ids = list(filter(lambda i: i is not None, map(lambda ph: model.vocab.get(ph), phonemes)))
    input_ids = torch.LongTensor([[0, *ids, 0]])
    B, T = input_ids.shape
    input_lengths = torch.full((B,), T, dtype=torch.long)
    lengths_list = input_lengths.tolist()
    s_pred_cpu = ref_s[:, p.style_dim :]

    s_pred_tt = ttnn.from_torch(
        s_pred_cpu, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
    )
    keep_mask = ttnn.ones([B, T, 1], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)

    def _prosody_and_asr():
        """Extended Trace A region: returns (dur_clipped, t_en_bct). Full-length path only."""
        full_length = _batch_is_full_length(input_lengths, T)
        if full_length:
            bert_out = model._bert(input_ids, attention_mask=None)
        else:
            bert_out = model._bert(input_ids, attention_mask=_attention_keep_mask_bt(input_lengths, T).int())
        bfe = bert_out if bert_out.dtype == ttnn.float32 else ttnn.typecast(bert_out, ttnn.float32, memory_config=mc)
        d_en = ttnn.linear(
            bfe, p.bert_encoder_w, bias=p.bert_encoder_b, transpose_b=True, memory_config=mc, compute_kernel_config=ck
        )
        while len(d_en.shape) > 3:
            d_en = ttnn.squeeze(d_en, 0)
        d_en_bct = ttnn.permute(d_en, (0, 2, 1), memory_config=mc)
        if d_en_bct.dtype != ttnn.float32:
            d_en_bct = ttnn.typecast(d_en_bct, ttnn.float32, memory_config=mc)
        d_nlc = model._predictor._text_encoder.forward(
            d_en_bct=d_en_bct,
            style_bs=s_pred_tt,
            sequence_lengths=lengths_list,
            keep_mask_btl=keep_mask,
            compute_kernel_config=ck,
            memory_config=mc,
            wire_dtype=ttnn.float32,
        )
        x_lstm = tt_bilstm_nlc(
            x_nlc=d_nlc,
            fwd=p.predictor.lstm_fwd,
            rev=p.predictor.lstm_rev,
            compute_kernel_config=ck,
            memory_config=mc,
            sequence_lengths=lengths_list,
        )
        duration = model._predictor._duration_proj.forward(x_lstm, compute_kernel_config=ck, memory_config=mc)
        dur_sig = ttnn.sigmoid(duration, memory_config=mc)
        ttnn.deallocate(duration)
        dur_sum = ttnn.sum(dur_sig, dim=-1, memory_config=mc)
        ttnn.deallocate(dur_sig)
        dur_r = ttnn.round(dur_sum, memory_config=mc)
        ttnn.deallocate(dur_sum)
        dur_clipped = ttnn.clip(dur_r, min=1.0, memory_config=mc)
        ttnn.deallocate(dur_r)

        # --- ASR TextEncoder: same T_tokens regime, depends only on input_ids/lengths (NOT alignment).
        # Full-length path => text_mask=None => no mask upload during capture.
        t_en_bct = model._text_encoder(input_ids, input_lengths=input_lengths, text_mask=None)
        return dur_clipped, t_en_bct

    # Reference: prep OFF (untraced default path).
    y_off_dur_tt, y_off_asr_tt = _prosody_and_asr()
    ttnn.synchronize_device(device)
    y_off_dur = ttnn.to_torch(y_off_dur_tt).float()
    y_off_asr = ttnn.to_torch(y_off_asr_tt).float()
    ttnn.deallocate(y_off_dur_tt)
    ttnn.deallocate(y_off_asr_tt)

    tid = None
    set_trace_weight_prep(True)
    try:
        # Warmup (compile + populate every prep cache).
        y_e_dur_tt, y_e_asr_tt = _prosody_and_asr()
        ttnn.synchronize_device(device)
        y_e_dur = ttnn.to_torch(y_e_dur_tt).float()
        y_e_asr = ttnn.to_torch(y_e_asr_tt).float()
        ttnn.deallocate(y_e_dur_tt)
        ttnn.deallocate(y_e_asr_tt)

        eager_t0 = time.perf_counter()
        for _ in range(_EAGER_TIMING_ITERS):
            a, b = _prosody_and_asr()
            ttnn.synchronize_device(device)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
        eager_ms = (time.perf_counter() - eager_t0) / _EAGER_TIMING_ITERS * 1e3

        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        capturing = True
        try:
            y_t_dur_tt, y_t_asr_tt = _prosody_and_asr()
        finally:
            # Always end the capture, even on failure — leaving the device in capture mode makes the
            # subsequent close_device()/synchronize hang and hold the chip lock.
            ttnn.end_trace_capture(device, tid, cq_id=0)
            capturing = False

        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        trace_t0 = time.perf_counter()
        for _ in range(_REPLAY_ITERS):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        trace_ms = (time.perf_counter() - trace_t0) / _REPLAY_ITERS * 1e3

        y_t_dur = ttnn.to_torch(y_t_dur_tt).float()
        y_t_asr = ttnn.to_torch(y_t_asr_tt).float()
        ttnn.release_trace(device, tid)
        tid = None
        ttnn.deallocate(y_t_dur_tt)
        ttnn.deallocate(y_t_asr_tt)
    finally:
        if tid is not None:
            ttnn.release_trace(device, tid)
        set_trace_weight_prep(False)
        clear_trace_weight_prep_cache()
        ttnn.deallocate(s_pred_tt)
        ttnn.deallocate(keep_mask)

    assert torch.isfinite(y_t_dur).all() and torch.isfinite(y_t_asr).all(), "Traced output NaN/Inf"
    assert y_t_dur.shape == y_e_dur.shape == y_off_dur.shape
    assert y_t_asr.shape == y_e_asr.shape == y_off_asr.shape

    _, dur_pcc = comp_pcc(y_e_dur, y_t_dur, pcc=0.0)
    _, asr_pcc = comp_pcc(y_e_asr, y_t_asr, pcc=0.0)
    assert torch.equal(y_t_dur, y_e_dur), f"Duration replay diverged from eager (PCC={dur_pcc:.8f})."
    assert torch.equal(y_t_asr, y_e_asr), f"ASR replay diverged from eager (PCC={asr_pcc:.8f})."
    assert torch.equal(y_off_dur, y_e_dur), "Trace-prep changed duration vs default path."
    assert torch.equal(y_off_asr, y_e_asr), "Trace-prep changed ASR output vs default path."

    speedup = eager_ms / trace_ms if trace_ms > 0 else float("inf")
    print(
        f"\nTT prosody+ASR metal-trace (T_tokens={T}):\n"
        f"  eager warm forward : {eager_ms:8.2f} ms (avg of {_EAGER_TIMING_ITERS})\n"
        f"  trace replay       : {trace_ms:8.2f} ms (avg of {_REPLAY_ITERS})\n"
        f"  wall-clock speedup : {speedup:8.2f}x\n"
        f"  trace vs eager     : bit-identical (both outputs); prep-on == prep-off\n"
        f"  pred_dur sum       : {int(y_t_dur.sum().item())} frames; asr shape={tuple(y_t_asr.shape)}"
    )
