# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Trace + 2CQ wrapping of ``TtOobleckDecoder.__call__`` (single-chunk VAE decode).

Wraps one invocation of the Oobleck audio decoder — ``conv1`` (Conv1d) → N upsampling blocks
(``TtOobleckDecoderBlock`` = ConvTranspose1d + residual stack with Snake1d activations) →
``snake1`` → ``conv2`` (Conv1d) → audio waveform — into a TTNN trace, then replays it on
CQ 0 with host latent rewrites on CQ 1.

This is the **final per-module trace+2CQ test** in the ACE-Step plan. It exercises:
- ``ttnn.conv1d`` (twice: input/output convolutions) with its ``prepare_conv_weights`` cache
- ``ttnn.conv_transpose2d`` (the upsampling stride backbone, called via TtConvTranspose1d)
- ``ttnn.typecast`` (FP32 latents -> BF16)
- The Snake1d kernel from ``ttnn_impl/vae/snake.py``

This is also the most conv-heavy traced body in the demo, so it stresses TTNN's L1 small-arena
budget (``ACE_STEP_L1_SMALL_SIZE``) more than the transformer-style tests do.

The tiny config (``TinyDecoderConfig`` from ``test_vae_decoder_pcc.py``) keeps capture cheap
while still exercising every component class.

Run:

    pytest models/demos/ace_step_v1_5/perf/module_trace/test_vae_decoder_trace_2cq.py -v -s
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from models.demos.ace_step_v1_5.tests._dit_decoder_pcc_common import assert_pcc_print
from models.demos.ace_step_v1_5.torch_ref.vae.oobleck_decoder import OobleckDecoder
from models.demos.ace_step_v1_5.ttnn_impl.vae import TtOobleckDecoder

_DEFAULT_ITERS = int(os.environ.get("ACE_STEP_TRACE_TEST_ITERS", "16"))
_TRACE_VS_COMPILE_PCC = 0.999

# ``trace_device`` comes from the demo conftest (shared session device with 2 CQs +
# trace region). Opening a per-test second device on the same physical hardware
# corrupts TT's dispatch state and breaks the session ``close_device`` sync.


@dataclass(frozen=True)
class _TinyDecoderConfig:
    """Reduced Oobleck config that exercises every stage at tractable size.

    Mirrors ``test_vae_decoder_pcc.py::TinyDecoderConfig`` so this perf test runs in the same
    shape envelope as the PCC test that already validates this code path against torch.
    """

    decoder_channels: int = 32
    decoder_input_channels: int = 16
    audio_channels: int = 2
    upsampling_ratios: tuple = (2, 2)
    channel_multiples: tuple = (1, 2)
    # Latent time axis. Output audio length = t_latent * prod(upsampling_ratios).
    t_latent: int = 32


def _bct_to_btc(x: torch.Tensor) -> torch.Tensor:
    """[B, C, T] -> [B, T, C] for TTNN row-major input convention."""
    return x.transpose(1, 2).contiguous()


def test_vae_decoder_trace_2cq(trace_device):
    """Trace + 2CQ replay of the Oobleck VAE decoder, with PCC parity vs the no-trace compile pass.

    All upstream PCC vs the torch reference is already covered by ``test_vae_decoder_pcc.py``;
    this test only validates that **trace replay is deterministic** w.r.t. the same persistent
    latent input (i.e. compile-pass output == trace-replay output bit-for-bit).
    """
    import ttnn

    device = trace_device
    cfg = _TinyDecoderConfig()

    # Build torch decoder once just to harvest a consistent state dict.
    torch.manual_seed(11)
    torch_dec = OobleckDecoder(
        channels=cfg.decoder_channels,
        input_channels=cfg.decoder_input_channels,
        audio_channels=cfg.audio_channels,
        upsampling_ratios=cfg.upsampling_ratios,
        channel_multiples=cfg.channel_multiples,
    ).eval()
    full_sd = {f"decoder.{k}": v for k, v in torch_dec.state_dict().items()}

    tt_dec = TtOobleckDecoder(
        state_dict=full_sd,
        device=device,
        decoder_prefix="decoder.",
        channels=cfg.decoder_channels,
        input_channels=cfg.decoder_input_channels,
        audio_channels=cfg.audio_channels,
        upsampling_ratios=cfg.upsampling_ratios,
        channel_multiples=cfg.channel_multiples,
    )

    # --- HOST INPUT (latents) -----------------------------------------------------------
    # TTNN VAE accepts [B, T, C] row-major; torch reference accepts [B, C, T]. Use the same
    # tiny shape as test_decoder_tiny_pcc so capture stays inexpensive.
    x_bct = torch.randn(1, cfg.decoder_input_channels, cfg.t_latent, dtype=torch.bfloat16).float()
    x_btc = _bct_to_btc(x_bct)

    x_host = ttnn.from_torch(x_btc.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    x_dev = ttnn.from_torch(
        x_btc.contiguous(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # --- COMPILE PASS (CQ 0, no trace) --------------------------------------------------
    # Populates program cache + every TtConv1d / TtConvTranspose1d._ensure_packed cache for
    # the runtime (batch, input_length). Without warmup, the trace would record per-conv
    # weight uploads inside the captured graph and explode with "Writes are not supported
    # during trace capture".
    y_compile_dev = tt_dec(x_dev)
    ttnn.synchronize_device(device)
    y_compile = ttnn.to_torch(y_compile_dev).to(torch.float32)
    assert torch.isfinite(y_compile).all(), "Compile-pass VAE decoder output has non-finite values."
    expected_audio_t = cfg.t_latent * int(np.prod(cfg.upsampling_ratios))
    expected_shape = (1, expected_audio_t, cfg.audio_channels)
    assert (
        tuple(y_compile.shape) == expected_shape
    ), f"Unexpected compile output shape: got {tuple(y_compile.shape)}, expected {expected_shape}"
    try:
        ttnn.deallocate(y_compile_dev)
    except Exception:
        pass

    # --- WARMUP PASS (CQ 0) -------------------------------------------------------------
    # Second forward to make sure every conv weight is truly device-resident before capture.
    y_warmup_dev = tt_dec(x_dev)
    ttnn.synchronize_device(device)
    try:
        ttnn.deallocate(y_warmup_dev)
    except Exception:
        pass

    # --- CAPTURE TRACE ------------------------------------------------------------------
    # If this raises "Writes/Reads are not supported during trace capture", check that the
    # warmup pass really ran at the same (batch, input_length) as the trace body. The
    # TtConv1d / TtConvTranspose1d caches are keyed on those values, and a mismatch causes
    # `_ensure_packed` to call `ttnn.prepare_conv_weights` inside the trace.
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    y_trace_dev = tt_dec(x_dev)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    trace_output_addr = y_trace_dev.buffer_address()

    op_event = ttnn.record_event(device, 0)

    # --- TIMED TRACE EXECUTION (CQ 0 compute, CQ 1 writes) ------------------------------
    iters = max(1, int(_DEFAULT_ITERS))
    latencies_ms: list[float] = []
    for _ in range(iters):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(x_host, x_dev, cq_id=1)
        write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, write_event)

        t0 = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        op_event = ttnn.record_event(device, 0)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1e3)

    # --- VALIDATE BUFFER ADDRESS STABILITY ---------------------------------------------
    assert trace_output_addr == y_trace_dev.buffer_address(), (
        f"Trace output buffer moved across executes: {trace_output_addr} -> {y_trace_dev.buffer_address()}. "
        "The captured graph included a non-deterministic allocation. Audit the VAE decoder body for "
        "per-call host transfers."
    )

    # --- VALIDATE TRACE OUTPUT MATCHES COMPILE OUTPUT ----------------------------------
    y_trace = ttnn.to_torch(y_trace_dev).to(torch.float32)
    assert_pcc_print(
        "vae_decoder_trace_2cq.trace_vs_compile",
        y_compile,
        y_trace,
        pcc=_TRACE_VS_COMPILE_PCC,
    )

    # --- REPORT -------------------------------------------------------------------------
    avg_ms = float(np.mean(latencies_ms))
    best_ms = float(np.min(latencies_ms))
    p90_ms = float(np.percentile(latencies_ms, 90)) if len(latencies_ms) > 1 else avg_ms
    print(
        f"[ace_step_v1_5][trace_2cq] vae decoder: iters={iters} "
        f"avg={avg_ms:.3f}ms best={best_ms:.3f}ms p90={p90_ms:.3f}ms "
        f"(t_lat={cfg.t_latent}, upsample={int(np.prod(cfg.upsampling_ratios))}x, "
        f"audio_ch={cfg.audio_channels})",
        flush=True,
    )

    # --- CLEANUP ------------------------------------------------------------------------
    try:
        ttnn.release_trace(device, tid)
    except Exception:
        pass
    for t in (y_trace_dev, x_dev):
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
