# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Metal-trace capture/replay for the Kokoro TT **decoder** (``asr/F0/N/s → audio``).

Phase 1 of the "fully-traced demo" plan (docs/generator_perf_optimizations.md): the decoder is the
whole post-alignment device graph — the AdaIN encode/decode resblocks, the F0/N stride-2 convs, the
ASR residual conv, and the generator (SineGen + STFT + upsample/resblocks + iSTFT). It has a fixed
shape once ``T_mel`` (= bucketed ``T_aligned``) is fixed, so it is trace-capturable. This extends the
generator-only trace (``test_tt_generator_trace.py``) to the full "Trace B" body.

Preconditions (same as the generator trace):
- Fallbacks OFF (device SineGen / on-device STFT) → static, host-round-trip-free graph.
- ``tt_conv.set_trace_weight_prep(True)`` so the decoder's many convs reuse cached prepared weights
  instead of re-uploading them each call (the host->device write trace capture forbids).

Asserts the trace replays the eager decoder output **bit-for-bit** and reports the warm speedup.

Run::

    KOKORO_GEN_L1=1 pytest models/experimental/kokoro/tests/test_tt_decoder_trace.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.m_source_rng import (
    deallocate_m_source_rng_tt,
    make_zero_m_source_rng,
    m_source_rng_shapes_from_f0,
    patched_m_source_torch_rng,
    upload_m_source_rng,
)
from models.experimental.kokoro.tests.test_tt_decoder_pcc import (
    _T_MEL,
    _build_decoder,
    _find_checkpoint,
    _load_trained_weights,
    _setup_inputs,
)
from models.experimental.kokoro.tt.tt_conv import clear_trace_weight_prep_cache, set_trace_weight_prep
from models.experimental.kokoro.tt.tt_decoder import TTDecoder, preprocess_tt_decoder

# The decoder graph is larger than the generator alone (adds encode/decode resblocks + F0/N/asr
# convs); a 200 MB DRAM trace region comfortably holds the captured command queue + static allocs.
_TRACE_REGION_SIZE = 200_000_000
_L1_SMALL_SIZE = 98304
_REPLAY_ITERS = 5
_EAGER_TIMING_ITERS = 3


@pytest.mark.parametrize(
    "device",
    [{"trace_region_size": _TRACE_REGION_SIZE, "l1_small_size": _L1_SMALL_SIZE}],
    indirect=True,
)
def test_tt_decoder_metal_trace_capture_replay(device):
    """Capture a metal trace of the whole decoder forward, replay it, and check parity.

    trace-vs-eager output must be bit-identical (faithful replay). PCC vs PyTorch is informational
    (bf16 device-SineGen/STFT ceiling), like the decoder smoke test.
    """
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    activations_in_l1 = os.environ.get("KOKORO_GEN_L1") == "1"

    ref = _build_decoder()
    _load_trained_weights(ref, ckpt_path)
    asr, F0_curve, N_curve, s = _setup_inputs(seed=3)

    params = preprocess_tt_decoder(ref, device, time_len_asr=_T_MEL)
    B_rng, T_har, dim = m_source_rng_shapes_from_f0(
        F0_curve,
        upsample_scale_full=int(params.generator.upsample_scale_full),
        dim=int(params.generator.m_source.sinegen.dim),
    )
    rng_cpu = make_zero_m_source_rng(B_rng, T_har, dim)

    with torch.no_grad(), patched_m_source_torch_rng(rng_cpu):
        y_ref = ref(asr, F0_curve, N_curve, s)

    tt_mod = TTDecoder(device, params)
    # Static-graph precondition: no CPU fallbacks in the decoder's generator.
    assert not tt_mod._generator._m_source._sinegen.use_torch_phase_fallback
    assert not tt_mod._generator._stft._use_torch_stft_fallback

    mc = ttnn.DRAM_MEMORY_CONFIG
    asr_nlc = asr.transpose(1, 2).contiguous()
    asr_tt = ttnn.from_torch(asr_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    F0_tt = ttnn.from_torch(F0_curve, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    N_tt = ttnn.from_torch(N_curve, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    rng_tt = upload_m_source_rng(rng_cpu, device, memory_config=mc)

    persistent = [asr_tt, F0_tt, N_tt, s_tt, rng_tt.rand_ini, rng_tt.sinegen_noise, rng_tt.source_noise]

    # Feed fresh clones so the captured graph never frees the persistent buffers we replay against
    # (the decoder/generator consume some inputs in-place). Clones read the persistent buffers by
    # address on every replay.
    def _forward():
        return tt_mod(
            ttnn.clone(asr_tt),
            ttnn.clone(F0_tt),
            ttnn.clone(N_tt),
            ttnn.clone(s_tt),
            memory_config=mc,
            sinegen_rand_ini=ttnn.clone(rng_tt.rand_ini),
            sinegen_noise_raw=ttnn.clone(rng_tt.sinegen_noise),
            source_noise_raw=ttnn.clone(rng_tt.source_noise),
        )

    tid = None
    set_trace_weight_prep(True)
    try:
        # 1. Warmup: compile programs, prime the program cache, prepare + cache conv weights once.
        y_eager_tt = _forward()
        ttnn.synchronize_device(device)
        y_eager = ttnn.to_torch(y_eager_tt).float()
        ttnn.deallocate(y_eager_tt)

        # 2. Time the eager (host-dispatch-bound) warm forward.
        eager_t0 = time.perf_counter()
        for _ in range(_EAGER_TIMING_ITERS):
            y_tmp = _forward()
            ttnn.synchronize_device(device)
            ttnn.deallocate(y_tmp)
        eager_ms = (time.perf_counter() - eager_t0) / _EAGER_TIMING_ITERS * 1e3

        # 3. Capture.
        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        y_traced_tt = _forward()
        ttnn.end_trace_capture(device, tid, cq_id=0)

        # 4. Replay.
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)  # warm the replay
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
        deallocate_m_source_rng_tt(rng_tt)
        for t in (asr_tt, F0_tt, N_tt, s_tt):
            ttnn.deallocate(t)

    while y_eager.dim() > y_ref.dim():
        y_eager.squeeze_(0)
    while y_traced.dim() > y_ref.dim():
        y_traced.squeeze_(0)

    assert torch.isfinite(y_traced).all(), "Traced decoder forward produced NaN/Inf"
    assert y_traced.shape == y_eager.shape, (y_traced.shape, y_eager.shape)

    _, parity_pcc = comp_pcc(y_eager, y_traced, pcc=0.0)
    assert torch.equal(y_traced, y_eager), (
        f"Trace replay diverged from eager output (parity PCC={parity_pcc:.8f}); "
        "trace should be a bit-identical replay of the captured graph."
    )

    _, ref_pcc = comp_pcc(y_ref, y_traced, pcc=0.0)
    speedup = eager_ms / trace_ms if trace_ms > 0 else float("inf")
    print(
        f"\nTTDecoder metal-trace (T_mel={_T_MEL}, l1_activations={activations_in_l1}):\n"
        f"  eager warm forward : {eager_ms:8.2f} ms (avg of {_EAGER_TIMING_ITERS})\n"
        f"  trace replay       : {trace_ms:8.2f} ms (avg of {_REPLAY_ITERS})\n"
        f"  wall-clock speedup : {speedup:8.2f}x\n"
        f"  trace-vs-eager PCC : {parity_pcc:.8f} (bit-identical)\n"
        f"  trace-vs-torch PCC : {ref_pcc:.6f} (informational; bf16 device-SineGen/STFT ceiling)"
    )
