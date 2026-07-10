# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Metal-trace capture/replay test for the Kokoro TT vocoder (``TTGenerator``).

Why the *generator* and not the full demo pipeline
---------------------------------------------------
``models/experimental/kokoro/demo/ttnn_kokoro_full_demo.py`` runs the whole
``TTKModel``. That pipeline **cannot** be captured as a single metal trace: it reads
predicted durations back to the host mid-forward, builds a dynamically-shaped alignment
matrix (``T_aligned = sum(pred_dur)``), and re-preprocesses the generator per chunk. Metal
trace requires a *static* device graph (fixed shapes, no host round-trips), so tracing is
only feasible on the post-duration, fixed-shape unit — the generator (see
``docs/generator_perf_optimizations.md`` "Metal-trace feasibility (scoped)").

This test therefore mirrors the demo's generator invocation (loaded weights, deterministic
inputs, fallbacks off / device SineGen so the graph is static) and proves:

1. A metal trace of the full ``TTGenerator.forward`` can be captured and replayed.
2. Replaying the trace reproduces the eager output **bit-for-bit** (faithful replay).
3. Replay collapses the ~180 ms host-dispatch-bound warm forward toward device-bound —
   the informational wall-clock speedup the doc's lever #10 predicts.

Trace capture forbids host->device writes, but ``ttnn.conv1d`` / ``conv_transpose2d`` re-upload
their prepared weights on every call. ``tt_conv.set_trace_weight_prep(True)`` makes the convs
prepare + cache their weights once (on the warmup forward) and reuse the on-device copies
thereafter, which is what lets the conv-heavy generator graph be captured at all.

Run (matching the demo's L1-activations opt-in path)::

    KOKORO_GEN_L1=1 pytest models/experimental/kokoro/tests/test_tt_generator_trace.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.tests.test_tt_generator_pcc import (
    _build_kokoro_generator,
    _find_checkpoint,
    _load_trained_weights,
    _setup_test_inputs,
)
from models.experimental.kokoro.tt.tt_conv import clear_trace_weight_prep_cache, set_trace_weight_prep
from models.experimental.kokoro.tt.tt_generator import TTGenerator, preprocess_tt_generator


# The generator is host-dispatch-bound (~1.4k ops); the captured command queue plus the
# static intermediate allocations comfortably fit a 100 MB DRAM trace region. ``l1_small_size``
# matches the demo default so the conv halo/interleave allocations behave identically.
_TRACE_REGION_SIZE = 100_000_000
_L1_SMALL_SIZE = 98304
_REPLAY_ITERS = 5
_EAGER_TIMING_ITERS = 3


def _build_generator_inputs(device, T_x: int):
    """Deterministic generator inputs + host device-tensors, mirroring the demo/smoke path.

    Returns ``(ref, params, y_ref, x_tt, s_tt, f0_tt, rand_ini_tt)``. The torch ``rand`` /
    ``randn_like`` monkeypatches make the reference SineGen/source noise deterministic so the
    reference audio (``y_ref``) is a stable PCC target; the TT side is fed the same ``rand_ini``.
    """
    ref = _build_kokoro_generator()
    ckpt_path = _find_checkpoint()
    _load_trained_weights(ref, ckpt_path)

    x, s, f0 = _setup_test_inputs(T_x)
    f0u_ref = ref.f0_upsamp(f0[:, None]).transpose(1, 2).contiguous()
    B, T_har, _ = f0u_ref.shape
    dim = ref.m_source.l_sin_gen.dim
    torch.manual_seed(123)
    rand_ini = torch.rand(B, dim)
    rand_ini[:, 0] = 0.0
    sinegen_noise_raw = torch.randn(B, T_har, dim)
    source_noise_raw = torch.randn(B, T_har, 1)

    def _fake_rand(*size, **kwargs):
        out = rand_ini.to(kwargs.get("device", rand_ini.device))
        return out.to(kwargs.get("dtype", out.dtype))

    def _fake_randn_like(t, **kwargs):
        if t.shape[-1] == 1:
            return source_noise_raw.to(device=t.device, dtype=t.dtype).expand_as(t).clone()
        return sinegen_noise_raw.to(device=t.device, dtype=t.dtype).expand_as(t).clone()

    real_rand, real_randn_like = torch.rand, torch.randn_like
    torch.rand, torch.randn_like = _fake_rand, _fake_randn_like
    try:
        with torch.no_grad():
            y_ref = ref(x, s, f0)
    finally:
        torch.rand, torch.randn_like = real_rand, real_randn_like

    params = preprocess_tt_generator(ref, device, time_len_x=T_x)

    x_nlc = x.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    f0_tt = ttnn.from_torch(f0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    rand_ini_tt = ttnn.from_torch(rand_ini.unsqueeze(1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    return ref, params, y_ref, x_tt, s_tt, f0_tt, rand_ini_tt


@pytest.mark.parametrize(
    "device",
    [{"trace_region_size": _TRACE_REGION_SIZE, "l1_small_size": _L1_SMALL_SIZE}],
    indirect=True,
)
def test_tt_generator_metal_trace_capture_replay(device):
    """Capture a metal trace of the whole generator forward, replay it, and check parity.

    Fallbacks are OFF (device SineGen / on-device STFT) so the graph is fully static and
    host-round-trip-free — the precondition for trace capture. PCC vs PyTorch reflects the
    known bf16 device-SineGen ceiling (~0.98, doc lever #10); the *trace-vs-eager* parity is
    the real assertion here and must be exact.
    """
    if _find_checkpoint() is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    T_x = 5
    activations_in_l1 = os.environ.get("KOKORO_GEN_L1") == "1"
    ref, params, y_ref, x_tt, s_tt, f0_tt, rand_ini_tt = _build_generator_inputs(device, T_x)

    tt_mod = TTGenerator(device, params, activations_in_l1=activations_in_l1)
    # Trace requires a static device graph: no CPU phase/STFT fallback round-trips.
    assert not tt_mod._m_source._sinegen.use_torch_phase_fallback
    assert not tt_mod._stft._use_torch_stft_fallback

    # ``TTGenerator.forward`` consumes (deallocates) its ``x_nlc`` input in-place when no dtype
    # cast is needed. For a persistent-input trace the captured graph must NOT free the buffers we
    # replay against, so feed the generator fresh clones and keep x/s/f0/rand_ini alive. The clones
    # become part of the traced graph and read the persistent buffers by address on every replay.
    def _forward():
        xc = ttnn.clone(x_tt)
        sc = ttnn.clone(s_tt)
        fc = ttnn.clone(f0_tt)
        rc = ttnn.clone(rand_ini_tt)
        return tt_mod(xc, sc, fc, sinegen_rand_ini=rc)

    # Enable prepared-weight caching so convs reuse on-device weights instead of re-uploading them
    # each forward (the sole host->device write that would otherwise abort trace capture — see the
    # ``tt_conv.set_trace_weight_prep`` note). The first warmup forward populates the cache; every
    # later forward + trace replay reuses it. Reset + free the cache in ``finally`` (global flag;
    # cached tensors live on the fixture device and must be dropped before it closes).
    tid = None
    set_trace_weight_prep(True)
    try:
        # --- 1. Eager warmup: compiles every program, primes the program cache, and prepares +
        #        caches every conv's weights once, so trace capture sees only cached dispatches. ---
        y_eager_tt = _forward()
        ttnn.synchronize_device(device)
        y_eager = ttnn.to_torch(y_eager_tt).float()
        ttnn.deallocate(y_eager_tt)

        # --- 2. Time the eager (host-dispatch-bound) warm forward. ---
        eager_t0 = time.perf_counter()
        for _ in range(_EAGER_TIMING_ITERS):
            y_tmp = _forward()
            ttnn.synchronize_device(device)
            ttnn.deallocate(y_tmp)
        eager_ms = (time.perf_counter() - eager_t0) / _EAGER_TIMING_ITERS * 1e3

        # --- 3. Capture the trace. The forward's inputs (x/s/f0/rand_ini) are persistent device
        #        tensors allocated above; the trace records the intermediate allocations + the
        #        output buffer, and references the input buffers by address on replay. ---
        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        y_traced_tt = _forward()
        ttnn.end_trace_capture(device, tid, cq_id=0)

        # --- 4. Replay the captured command queue (near-zero host overhead). ---
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)  # discard first (warm the replay)
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
        for t in (x_tt, s_tt, f0_tt, rand_ini_tt):
            ttnn.deallocate(t)

    # --- 5. Assertions. ---
    while y_eager.dim() > y_ref.dim():
        y_eager.squeeze_(0)
    while y_traced.dim() > y_ref.dim():
        y_traced.squeeze_(0)

    assert torch.isfinite(y_traced).all(), "Traced generator forward produced NaN/Inf"
    assert y_traced.shape == y_eager.shape, (y_traced.shape, y_eager.shape)

    # Faithful replay: the trace must reproduce the eager device output exactly.
    _, parity_pcc = comp_pcc(y_eager, y_traced, pcc=0.0)
    assert torch.equal(y_traced, y_eager), (
        f"Trace replay diverged from eager output (parity PCC={parity_pcc:.8f}); "
        "trace should be a bit-identical replay of the captured graph."
    )

    _, ref_pcc = comp_pcc(y_ref, y_traced, pcc=0.0)
    speedup = eager_ms / trace_ms if trace_ms > 0 else float("inf")
    print(
        f"\nTTGenerator metal-trace (T_x={T_x}, l1_activations={activations_in_l1}):\n"
        f"  eager warm forward : {eager_ms:8.2f} ms (avg of {_EAGER_TIMING_ITERS})\n"
        f"  trace replay       : {trace_ms:8.2f} ms (avg of {_REPLAY_ITERS})\n"
        f"  wall-clock speedup : {speedup:8.2f}x\n"
        f"  trace-vs-eager PCC : {parity_pcc:.8f} (bit-identical)\n"
        f"  trace-vs-torch PCC : {ref_pcc:.6f} (informational; bf16 device-SineGen ceiling ~0.98)"
    )
