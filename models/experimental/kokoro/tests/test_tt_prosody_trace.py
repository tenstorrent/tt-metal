# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Metal-trace capture/replay for the Kokoro **prosody→duration** path ("Trace A").

Phase 2 of the fully-traced-demo plan (docs/generator_perf_optimizations.md). Trace A is everything
from ``input_ids`` up to (but not including) the duration readback that splits the pipeline:

    input_ids -> BERT -> bert_encoder linear -> prosody TextEncoder (DurationEncoder BiLSTMs)
              -> duration BiLSTM -> duration_proj -> sigmoid/sum/round/clip -> dur_clipped

It has a fixed shape once ``T_tokens`` (bucketed) is fixed, so it is trace-capturable. The blockers
were per-forward host->device writes — BERT embedding id/mask uploads, the BiLSTM zero states +
reversal matrix, and the duration-encoder style expand — all now cached under
``tt_trace_prep.set_trace_weight_prep`` (default OFF, byte-identical to the untraced path).

Asserts the trace replays the eager ``dur_clipped`` **bit-for-bit** and that prep-on equals prep-off
(the caching changes only *where* uploads happen, never the math). Reports the warm speedup.

Run::

    pytest models/experimental/kokoro/tests/test_tt_prosody_trace.py -s
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
def test_tt_prosody_to_duration_metal_trace(device):
    """Capture a metal trace of prosody→duration, replay it, and check parity."""
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

    # Persistent inputs (fixed identity across warmup/capture/replay) — the trace reads them by
    # address; the real integration would ``copy_host_to_device_tensor`` new values between chunks.
    s_pred_tt = ttnn.from_torch(
        s_pred_cpu, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
    )
    keep_mask = ttnn.ones([B, T, 1], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)

    def _prosody_to_duration():
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
        return dur_clipped

    # Reference: prep OFF (the untraced default path).
    y_off_tt = _prosody_to_duration()
    ttnn.synchronize_device(device)
    y_off = ttnn.to_torch(y_off_tt).float()
    ttnn.deallocate(y_off_tt)

    tid = None
    set_trace_weight_prep(True)
    try:
        # Warmup (compile + populate every prep cache).
        y_eager_tt = _prosody_to_duration()
        ttnn.synchronize_device(device)
        y_eager = ttnn.to_torch(y_eager_tt).float()
        ttnn.deallocate(y_eager_tt)

        eager_t0 = time.perf_counter()
        for _ in range(_EAGER_TIMING_ITERS):
            y_tmp = _prosody_to_duration()
            ttnn.synchronize_device(device)
            ttnn.deallocate(y_tmp)
        eager_ms = (time.perf_counter() - eager_t0) / _EAGER_TIMING_ITERS * 1e3

        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        y_traced_tt = _prosody_to_duration()
        ttnn.end_trace_capture(device, tid, cq_id=0)

        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        trace_t0 = time.perf_counter()
        for _ in range(_REPLAY_ITERS):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        trace_ms = (time.perf_counter() - trace_t0) / _REPLAY_ITERS * 1e3

        y_traced = ttnn.to_torch(y_traced_tt).float()
        ttnn.release_trace(device, tid)
        tid = None
        ttnn.deallocate(y_traced_tt)
    finally:
        if tid is not None:
            ttnn.release_trace(device, tid)
        set_trace_weight_prep(False)
        clear_trace_weight_prep_cache()
        ttnn.deallocate(s_pred_tt)
        ttnn.deallocate(keep_mask)

    assert torch.isfinite(y_traced).all(), "Traced prosody→duration produced NaN/Inf"
    assert y_traced.shape == y_eager.shape == y_off.shape

    _, parity_pcc = comp_pcc(y_eager, y_traced, pcc=0.0)
    assert torch.equal(
        y_traced, y_eager
    ), f"Trace replay diverged from eager (prep-on) output (parity PCC={parity_pcc:.8f})."
    # Prep-on must equal prep-off: caching changes only where uploads happen, never the math.
    assert torch.equal(y_off, y_eager), "Trace-prep caching changed the duration output vs the default path."

    speedup = eager_ms / trace_ms if trace_ms > 0 else float("inf")
    print(
        f"\nTT prosody→duration metal-trace (T_tokens={T}):\n"
        f"  eager warm forward : {eager_ms:8.2f} ms (avg of {_EAGER_TIMING_ITERS})\n"
        f"  trace replay       : {trace_ms:8.2f} ms (avg of {_REPLAY_ITERS})\n"
        f"  wall-clock speedup : {speedup:8.2f}x\n"
        f"  trace vs eager     : bit-identical; prep-on == prep-off: bit-identical\n"
        f"  pred_dur sum (frames): {int(y_traced.sum().item())}"
    )
