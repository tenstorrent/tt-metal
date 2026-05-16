# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Trace + 2CQ wrapping of the full per-step DiT body.

Builds the same composition as ``AceStepV15TTNNPipeline.forward``:

    concat(context_latents, xt) -> patch_embed -> DiT core -> output_head

but **without** running the per-step time-embed inside the trace (the timestep index
controls a ``ttnn.slice`` and so would bake the step number into the captured graph).
Instead, ``temb`` and ``timestep_proj`` are computed once on the host side via the
real ``TtTimestepEmbedding`` and become persistent inputs to the trace — exactly the
pattern the Stable Diffusion 1.4 perf test uses with its precomputed ``_tlist``.

This is the natural follow-on to ``test_dit_decoder_core_trace_2cq.py``:
it stresses the trace at a larger op count (patchify + core + de-patchify) and
exercises ``ttnn.conv1d`` (patch_embed) and the hand-rolled de-patchify ``ttnn.linear``
inside ``TtAceStepDiTOutputHead``, which were not in the previous body.

Run:

    pytest models/demos/ace_step_v1_5/perf/module_trace/test_pipe_body_trace_2cq.py -v -s

Useful env:

- ``ACE_STEP_TRACE_TEST_ITERS``: number of traced iterations to time (default 16).
- ``ACE_STEP_L1_SMALL_SIZE`` / ``TT_DEVICE_ID``: same semantics as the demo conftest.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from models.demos.ace_step_v1_5.tests._dit_decoder_pcc_common import assert_pcc_print
from models.demos.ace_step_v1_5.torch_ref.dit_decoder_core import make_tiny_state_dict
from models.demos.ace_step_v1_5.ttnn_impl.dit_decoder_core import (
    AceStepDecoderConfigTTNN,
    TtAceStepDiTCore,
    TtTimestepEmbedding,
)
from models.demos.ace_step_v1_5.ttnn_impl.output_head import TtAceStepDiTOutputHead
from models.demos.ace_step_v1_5.ttnn_impl.patchify import TtAceStepPatchEmbed1D

_DEFAULT_ITERS = int(os.environ.get("ACE_STEP_TRACE_TEST_ITERS", "16"))
# PCC of trace replay vs. compile-pass (no-trace) output. Trace records the same ops in the
# same order with the same persistent inputs, so the two should be bit-identical modulo BF16
# accumulation noise; 0.999 is a generous floor that catches any silent buffer-reuse bug.
_TRACE_VS_COMPILE_PCC = 0.999

# ``trace_device`` comes from the demo conftest (shared session device with 2 CQs +
# trace region). Opening a per-test second device on the same physical hardware
# corrupts TT's dispatch state and breaks the session ``close_device`` sync.


@dataclass(frozen=True)
class _PipeBodyConfig:
    """Tile-aligned tiny shapes that exercise patchify + core + output_head together."""

    batch: int = 1
    # Audio-frame axis: must be a multiple of patch_size after pad (we pick a multiple to avoid pad churn).
    frames: int = 32
    patch_size: int = 2
    audio_channels: int = 64  # xt last-dim
    ctx_channels: int = 128  # context-latent last-dim
    n_heads: int = 4
    head_dim: int = 32  # tile-aligned (same constraint as test_pcc_dit_decoder_core)
    cond_dim: int = 32
    intermediate: int = 256
    num_layers: int = 1
    enc_seq_len: int = 16

    @property
    def hidden_size(self) -> int:
        return int(self.n_heads * self.head_dim)

    @property
    def in_channels(self) -> int:
        return int(self.audio_channels + self.ctx_channels)


def _make_pipe_body_state_dict(cfg: _PipeBodyConfig) -> dict:
    """Synthetic state dict that satisfies patch_embed + core + output_head construction.

    Keys mirror the post-prefix-stripped state dict that ``AceStepV15TTNNPipeline`` builds from
    ``load_safetensors_state_dict(prefix='decoder.')``: ``proj_in.1.{weight,bias}``,
    ``proj_out.1.{weight,bias}``, ``norm_out.weight``, ``scale_shift_table``,
    ``condition_embedder.{weight,bias}``, and ``layers.{i}.*`` from ``make_tiny_state_dict``.
    """
    torch.manual_seed(0)
    sd: dict = {}

    # patch_embed (proj_in): conv1d weight [out, in, k] in PyTorch order.
    proj_in = nn.Conv1d(
        in_channels=cfg.in_channels,
        out_channels=cfg.hidden_size,
        kernel_size=cfg.patch_size,
        stride=cfg.patch_size,
        padding=0,
        bias=True,
    )
    sd["proj_in.1.weight"] = proj_in.weight.detach().clone().to(torch.float32).numpy()
    sd["proj_in.1.bias"] = proj_in.bias.detach().clone().to(torch.float32).numpy()

    # output_head (proj_out): convtranspose1d weight [in, out, k] in PyTorch order.
    proj_out = nn.ConvTranspose1d(
        in_channels=cfg.hidden_size,
        out_channels=cfg.audio_channels,
        kernel_size=cfg.patch_size,
        stride=cfg.patch_size,
        padding=0,
        bias=True,
    )
    sd["proj_out.1.weight"] = proj_out.weight.detach().clone().to(torch.float32).numpy()
    sd["proj_out.1.bias"] = proj_out.bias.detach().clone().to(torch.float32).numpy()

    # output_head modulation: norm_out + scale_shift_table.
    sd["norm_out.weight"] = torch.randn(cfg.hidden_size, dtype=torch.float32).numpy()
    sd["scale_shift_table"] = torch.randn(1, 2, cfg.hidden_size, dtype=torch.float32).numpy()

    # DiT core (condition_embedder + N layers). make_tiny_state_dict already produces fp32 NumPy.
    core_sd = make_tiny_state_dict(
        d_model=cfg.hidden_size,
        n_heads=cfg.n_heads,
        head_dim=cfg.head_dim,
        cond_dim=cfg.cond_dim,
        intermediate=cfg.intermediate,
        num_layers=cfg.num_layers,
    )
    sd.update(core_sd)

    # time_embed / time_embed_r (lookup-table MLP). Only needed because TtTimestepEmbedding
    # checks for these keys at init. They are run *outside* the trace.
    in_ch = 256
    for base in ("time_embed", "time_embed_r"):
        sd[f"{base}.linear_1.weight"] = torch.randn(cfg.hidden_size, in_ch, dtype=torch.float32).numpy()
        sd[f"{base}.linear_1.bias"] = torch.randn(cfg.hidden_size, dtype=torch.float32).numpy()
        sd[f"{base}.linear_2.weight"] = torch.randn(cfg.hidden_size, cfg.hidden_size, dtype=torch.float32).numpy()
        sd[f"{base}.linear_2.bias"] = torch.randn(cfg.hidden_size, dtype=torch.float32).numpy()
        sd[f"{base}.time_proj.weight"] = torch.randn(6 * cfg.hidden_size, cfg.hidden_size, dtype=torch.float32).numpy()
        sd[f"{base}.time_proj.bias"] = torch.randn(6 * cfg.hidden_size, dtype=torch.float32).numpy()

    return sd


def _build_components(device, cfg: _PipeBodyConfig):
    """Construct the four TTNN components used by ``pipe.forward``."""
    import ttnn

    sd = _make_pipe_body_state_dict(cfg)

    decoder_cfg = AceStepDecoderConfigTTNN(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_heads,
        head_dim=cfg.head_dim,
        rms_norm_eps=1e-6,
        sliding_window=None,
    )

    # patch_embed + output_head both want an object with .patch_size / .in_channels / etc.
    class _PatchEmbedCfg:
        patch_size = cfg.patch_size
        in_channels = cfg.in_channels
        hidden_size = cfg.hidden_size
        audio_acoustic_hidden_dim = cfg.audio_channels
        rms_norm_eps = 1e-6

    patch_embed = TtAceStepPatchEmbed1D(
        config=_PatchEmbedCfg,
        state_dict=sd,
        base_address="proj_in",
        device=device,
        expected_input_length=cfg.frames,
        activation_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
    )
    output_head = TtAceStepDiTOutputHead(
        config=_PatchEmbedCfg,
        state_dict=sd,
        base_address="",
        device=device,
        activation_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
    )
    core = TtAceStepDiTCore(cfg=decoder_cfg, state_dict=sd, mesh_device=device, dtype=ttnn.bfloat16)

    # Build a tiny TtTimestepEmbedding pair (host-side time embed; runs OUTSIDE the trace).
    # The timesteps table must be rank-1 with at least one valid index.
    timesteps_host = np.linspace(1.0, 0.0, num=9, dtype=np.float32)
    time_embed = TtTimestepEmbedding(
        cfg=decoder_cfg,
        state_dict=sd,
        base_address="time_embed",
        mesh_device=device,
        timesteps_host=timesteps_host,
        dtype=ttnn.bfloat16,
    )
    time_embed_r = TtTimestepEmbedding(
        cfg=decoder_cfg,
        state_dict=sd,
        base_address="time_embed_r",
        mesh_device=device,
        timesteps_host=timesteps_host,
        dtype=ttnn.bfloat16,
    )
    return patch_embed, core, output_head, time_embed, time_embed_r


def _pipe_body(
    *,
    ttnn,
    patch_embed,
    core,
    output_head,
    xt_dev,
    ctx_dev,
    enc_dev,
    temb_dev,
    timestep_proj_dev,
):
    """One Euler-step DiT body. All five inputs are persistent device tensors.

    Matches ``AceStepV15TTNNPipeline.forward`` minus the internal time-embed lookup; ``temb`` and
    ``timestep_proj`` are precomputed by the host outside the trace.
    """
    hidden = ttnn.concat([ctx_dev, xt_dev], dim=-1)
    patches, meta = patch_embed.forward(hidden)
    patches_out = core(patches, timestep_proj_dev, enc_dev)
    acoustic = output_head.forward(patches_out, temb_dev, meta)
    return acoustic


def _to_host_bf16(t: torch.Tensor) -> "ttnn.Tensor":
    """Build a TTNN host tensor matching the device buffers used inside the body."""
    import ttnn

    return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)


def test_pipe_body_trace_2cq(trace_device):
    """Validate trace + 2CQ replay of patch_embed + core + output_head as one body.

    Flow mirrors ``test_dit_decoder_core_trace_2cq`` but with five persistent inputs.

    Assertions:
        - compile pass produces finite output of the expected shape.
        - traced replay matches the compile-pass output at PCC >= ``_TRACE_VS_COMPILE_PCC``
          (trace must be deterministic w.r.t. the same persistent inputs).
        - trace output buffer address is stable across executes.
    """
    import ttnn

    device = trace_device
    cfg = _PipeBodyConfig()
    patch_embed, core, output_head, time_embed, time_embed_r = _build_components(device, cfg)

    # --- HOST TIME-EMBED (outside the trace) --------------------------------------------
    # Pick a fixed timestep_index for the trace. In the real denoise loop the host runs
    # `time_embed(idx)` per step and copies the result to `temb_dev` / `timestep_proj_dev`
    # via CQ 1 before each `execute_trace`.
    timestep_index = 0
    temb_t_dev, tp_t_dev = time_embed(int(timestep_index))
    temb_r_dev, tp_r_dev = time_embed_r.from_timestep_value(0.0)  # cached → no host upload after first call
    temb_combined_dev = ttnn.add(temb_t_dev, temb_r_dev)  # [1, hidden_size] TILE
    tp_combined_dev = ttnn.add(tp_t_dev, tp_r_dev)  # [1, 6, hidden_size] TILE

    # The output_head expects temb in ROW_MAJOR ([B, hidden_size]).
    # The core expects timestep_proj in ROW_MAJOR ([B, 6, hidden_size]).
    temb_for_head_dev = ttnn.to_layout(temb_combined_dev, ttnn.ROW_MAJOR_LAYOUT)
    tp_for_core_dev = ttnn.to_layout(tp_combined_dev, ttnn.ROW_MAJOR_LAYOUT)

    # --- HOST INPUTS --------------------------------------------------------------------
    torch.manual_seed(7)
    xt_torch = torch.randn(cfg.batch, cfg.frames, cfg.audio_channels, dtype=torch.bfloat16)
    ctx_torch = torch.randn(cfg.batch, cfg.frames, cfg.ctx_channels, dtype=torch.bfloat16)
    enc_torch = torch.randn(cfg.batch, cfg.enc_seq_len, cfg.cond_dim, dtype=torch.bfloat16)

    xt_host = _to_host_bf16(xt_torch)
    ctx_host = _to_host_bf16(ctx_torch)
    enc_host = _to_host_bf16(enc_torch)

    xt_dev = ttnn.from_torch(
        xt_torch.contiguous(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ctx_dev = ttnn.from_torch(
        ctx_torch.contiguous(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    enc_dev = ttnn.from_torch(
        enc_torch.contiguous(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    body_kwargs = dict(
        ttnn=ttnn,
        patch_embed=patch_embed,
        core=core,
        output_head=output_head,
        xt_dev=xt_dev,
        ctx_dev=ctx_dev,
        enc_dev=enc_dev,
        temb_dev=temb_for_head_dev,
        timestep_proj_dev=tp_for_core_dev,
    )

    # --- COMPILE PASS (CQ 0, no trace) --------------------------------------------------
    # Populates program cache, mask caches, conv-weight packing for (B, padded_T). We keep
    # the torch copy `y_compile` around as the ground-truth reference for trace replay.
    y_compile_dev = _pipe_body(**body_kwargs)
    ttnn.synchronize_device(device)
    y_compile = ttnn.to_torch(y_compile_dev).to(torch.float32)
    assert torch.isfinite(y_compile).all(), "Compile-pass output has non-finite values; check state-dict shapes."
    # Output shape: [B, frames, audio_channels] after de-patchify trims to original_seq_len.
    expected_shape = (cfg.batch, cfg.frames, cfg.audio_channels)
    assert (
        tuple(y_compile.shape) == expected_shape
    ), f"Unexpected compile output shape: got {tuple(y_compile.shape)}, expected {expected_shape}"
    try:
        ttnn.deallocate(y_compile_dev)
    except Exception:
        pass

    # --- WARMUP PASS (CQ 0) -------------------------------------------------------------
    # Second forward to make sure every cache hit is truly device-resident; without this the
    # captured trace would record additional first-time allocations that aren't part of the
    # steady-state Euler step.
    y_warmup_dev = _pipe_body(**body_kwargs)
    ttnn.synchronize_device(device)
    try:
        ttnn.deallocate(y_warmup_dev)
    except Exception:
        pass

    # --- CAPTURE TRACE ------------------------------------------------------------------
    # If this raises "Writes/Reads are not supported during trace capture", grep:
    #   rg "ttnn\.(ones_like|zeros_like|full|as_tensor|from_torch)" models/demos/ace_step_v1_5/ttnn_impl
    # and hoist the offending call to __init__ or replace with a scalar-op equivalent.
    #
    # `y_trace_dev` is the *handle* into which `execute_trace` will write the output; its
    # contents are undefined until at least one `execute_trace` has run. Do NOT call
    # `ttnn.to_torch(y_trace_dev)` between `end_trace_capture` and the first `execute_trace`.
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    y_trace_dev = _pipe_body(**body_kwargs)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    trace_output_addr = y_trace_dev.buffer_address()

    op_event = ttnn.record_event(device, 0)

    # --- TIMED TRACE EXECUTION (CQ 0 compute, CQ 1 writes) ------------------------------
    iters = max(1, int(_DEFAULT_ITERS))
    latencies_ms: list[float] = []
    for _ in range(iters):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(xt_host, xt_dev, cq_id=1)
        ttnn.copy_host_to_device_tensor(ctx_host, ctx_dev, cq_id=1)
        ttnn.copy_host_to_device_tensor(enc_host, enc_dev, cq_id=1)
        # temb / timestep_proj are not rewritten per iteration here (fixed timestep_index for
        # the perf test). In production, you'd copy_host_to_device_tensor those too with the
        # per-step values produced by `time_embed(idx)` on the host side.
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
        "The captured graph included an allocation that re-runs non-deterministically. Audit the body for "
        "per-call host transfers, then re-capture."
    )

    # --- VALIDATE TRACE OUTPUT MATCHES COMPILE OUTPUT ----------------------------------
    # `y_trace_dev` now holds the last `execute_trace` output. Trace replays must be
    # deterministic w.r.t. the same persistent inputs, so trace output should match the
    # compile-pass output bit-for-bit (modulo any non-deterministic kernel scheduling).
    y_trace = ttnn.to_torch(y_trace_dev).to(torch.float32)
    assert_pcc_print("pipe_body_trace_2cq.trace_vs_compile", y_compile, y_trace, pcc=_TRACE_VS_COMPILE_PCC)

    # --- REPORT -------------------------------------------------------------------------
    avg_ms = float(np.mean(latencies_ms))
    best_ms = float(np.min(latencies_ms))
    p90_ms = float(np.percentile(latencies_ms, 90)) if len(latencies_ms) > 1 else avg_ms
    print(
        f"[ace_step_v1_5][trace_2cq] pipe body: iters={iters} "
        f"avg={avg_ms:.3f}ms best={best_ms:.3f}ms p90={p90_ms:.3f}ms "
        f"(layers={cfg.num_layers}, frames={cfg.frames}, hidden={cfg.hidden_size})",
        flush=True,
    )

    # --- CLEANUP ------------------------------------------------------------------------
    try:
        ttnn.release_trace(device, tid)
    except Exception:
        pass
    for t in (
        y_trace_dev,
        xt_dev,
        ctx_dev,
        enc_dev,
        temb_for_head_dev,
        tp_for_core_dev,
        temb_t_dev,
        tp_t_dev,
        temb_r_dev,
        tp_r_dev,
        temb_combined_dev,
        tp_combined_dev,
    ):
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
