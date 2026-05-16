# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end ACE-Step v1.5 model: text encoder → DiT (TTNN) → VAE decoder.

This module wraps the full inference pipeline into a single class so that
callers only need to supply a text prompt and get a waveform tensor back.

Neural inference (Qwen3 embeddings, instrumental condition, DiT, Oobleck VAE) runs
on Tenstorrent via TTNN. Encoder SDPA masks are uploaded once per prompt via
:class:`AceStepV15TTNNPipeline.build_encoder_attention_mask_b1qk_optional` (no
per-step mask rebuild). Context latents from the silence template are cached on
device at init. DiT latents feed the VAE without a host round trip when using the
default path. Only Hugging Face **tokenization** (string → token ids) and
checkpoint **loading** (host files → device weights) use the CPU; optional
waveform peak-normalization can return a TTNN tensor via *return_waveform_ttnn*.

The standalone function :func:`run_ttnn_denoise_loop` is exported for reuse by
other scripts (e.g. the prompt-to-wav demo). For PyTorch VAE decode (e.g.
``--torch-vae`` in the demo), import :func:`decode_with_vae` from
``torch_ref.e2e_model``.

Performance profiling (Tracy / device profiler):

    TTNN_OP_PROFILER=1 is set when running ``python -m tracy -p -r -v -m pytest ...``
    (see ``tools/tracy``). Signposts label stages in the Tracy timeline and op CSV.

    For kernel-level CSV rows from the device, also set ``TT_METAL_DEVICE_PROFILER=1``.
    After the denoising loop and again after VAE decode, ``ttnn.synchronize_device`` and
    ``ttnn.ReadDeviceProfiler`` run when either env var is set so buffered device profiler data is
    flushed at stage boundaries (DiT vs VAE show up cleanly in reports).

Optional: set ``ACE_STEP_TRACY_EACH_DENOISE_STEP=1`` to emit one Tracy signpost per Euler step
(use shorter ``infer_steps`` if the report becomes too dense).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import torch

import ttnn

from .condition_encoder import TtAceStepInstrumentalConditionEncoder
from .dit_sampling_ttnn import (
    TtnnMomentumBufferApg,
    adg_guidance_velocity_ttnn,
    apg_guidance_velocity_ttnn,
    bf16_row_from_numpy_bc,
    concat_duplicate_batch,
    euler_subtract_v_dt,
    fp32_tile_to_row_bf16,
    slice_batch_btc,
    typecast_bf16_any_to_fp32_tile,
)
from .full_pipeline import AceStepV15TTNNPipeline
from .oobleck_vae_decoder import TtOobleckVaeDecoder
from .qwen3_embedding_encoder import TtQwen3EmbeddingEncoder


def _ace_step_prof_signpost(header: str, message: Optional[str] = None) -> None:
    """Emit a Tracy signpost when the repo ``tracy`` package is available (editable install)."""
    try:
        from tracy import signpost as _signpost  # type: ignore[import-untyped]
    except ImportError:
        return
    try:
        if message is None:
            _signpost(header)
        else:
            _signpost(header, message)
    except Exception:
        pass


def _ace_step_flush_device_profiler(device) -> None:
    """Sync and drain device profiler buffers when Tracy / device profiling is enabled."""
    if os.environ.get("TTNN_OP_PROFILER") != "1" and os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        return
    try:
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
    except Exception:
        pass


@dataclass
class E2EConfig:
    """Configuration for the end-to-end pipeline."""

    checkpoint_safetensors_path: str
    vae_dir: str
    text_model_dir: str
    silence_latent_path: str

    duration_sec: float = 10.0
    infer_steps: int = 50
    shift: float = 1.0
    guidance_scale: float = 7.0
    cfg_interval_start: float = 0.0
    cfg_interval_end: float = 1.0
    use_adg: bool = True
    seed: int = 0
    sample_rate: int = 48000
    qwen_safetensors_path: Optional[str] = None
    vae_chunk_latents: int = 32
    vae_overlap_latents: int = 4


def _build_t_schedule(
    *,
    shift: float,
    infer_steps: int,
) -> List[float]:
    """Build the diffusion timestep schedule."""
    t = [float(i) / float(infer_steps) for i in range(infer_steps, -1, -1)]
    if shift != 1.0:
        s = float(shift)
        t = [s * x / (1.0 + (s - 1.0) * x) for x in t]
    return t


# ---------------------------------------------------------------------------
# Standalone helpers – importable by demo scripts to avoid code duplication.
# ---------------------------------------------------------------------------


def to_numpy_f32(t: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a contiguous float32 numpy array on CPU."""
    return t.detach().to(dtype=torch.float32).cpu().contiguous().numpy()


def run_ttnn_denoise_loop(
    pipe: AceStepV15TTNNPipeline,
    device: ttnn.Device,
    act_dtype: ttnn.DataType,
    mem: ttnn.MemoryConfig,
    t_schedule: List[float],
    frames: int,
    enc_hs: Optional[torch.Tensor] = None,
    enc_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ctx_lat: Optional[torch.Tensor] = None,
    null_emb: Optional[torch.Tensor] = None,
    do_cfg: bool = False,
    seed: int = 0,
    *,
    use_adg: bool = False,
    guidance_scale: float = 7.0,
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    progress_fn: Optional[Callable[[int, int, float, float], None]] = None,
    enc_tt_pipe: Optional[ttnn.Tensor] = None,
    ctx_tt_pipe: Optional[ttnn.Tensor] = None,
    return_device_latents: bool = False,
    encoder_attention_mask_b1qk: Optional[ttnn.Tensor] = None,
    deallocate_ctx_latents: bool = True,
) -> Union[torch.Tensor, ttnn.Tensor]:
    """Run the TTNN DiT denoising loop (latents stay on device; Euler + APG/ADG).

    Matches the demo path from ``dit_sampling_ttnn``: ``ttnn.randn`` init, row-major BF16 DiT inputs,
    FLOAT32 TILE latent state, and ``attention_mask_1d_bt=None`` (optional NumPy 1D encoder
    mask or pre-uploaded *encoder_attention_mask_b1qk*).

    Args:
        pipe: TTNN pipeline instance.
        device: TTNN device handle.
        act_dtype: Reserved for API compatibility (DiT path uses BF16 row tensors from NumPy).
        mem: TTNN memory config (e.g. ``DRAM_MEMORY_CONFIG``).
        t_schedule: Descending timestep floats.
        frames: Temporal frame count.
        enc_hs: Encoder hidden states ``[B, S, D]`` (host path only).
        enc_mask: Encoder attention mask ``[B, S]`` (float/bool); host NumPy or Torch tensor.
            Omit when *encoder_attention_mask_b1qk* is set.
        ctx_lat: Context latents ``[B, T, 128]`` (host path only).
        null_emb: Null-condition embedding for CFG on the host path only
            (broadcastable to *enc_hs*).
        do_cfg: Whether classifier-free guidance is active (batch doubles).
        seed: RNG seed for ``ttnn.randn`` initial noise.
        use_adg: If True and *do_cfg*, apply TTNN ADG in the CFG interval; else APG.
        guidance_scale: CFG strength when *do_cfg*.
        cfg_interval_start: CFG applies for ``t`` in
            ``[cfg_interval_start, cfg_interval_end]`` (ACE-Step semantics).
        cfg_interval_end: See *cfg_interval_start*.
        progress_fn: ``(step_idx, num_steps, t_curr, dt)`` after each Euler step.
        enc_tt_pipe: Optional pre-built DiT encoder hidden states on device
            (e.g. CFG batch already concatenated). When set, *ctx_tt_pipe* must
            be set; *enc_hs*, *ctx_lat*, and *null_emb* are ignored.
        ctx_tt_pipe: Pre-built context latents on device (CFG batch when *do_cfg*).
        return_device_latents: When True, return on-device TILE latents without ``ttnn.to_torch``;
            caller deallocates. Encoder/context pipe tensors (and optional *encoder_attention_mask_b1qk*)
            are deallocated in this function.
        encoder_attention_mask_b1qk: Pre-built cross-attention SDPA mask (see
            :meth:`AceStepV15TTNNPipeline.build_encoder_attention_mask_b1qk_optional`).
            When set, ``pipe.forward`` does not take *encoder_attention_mask_1d_bk*.
        deallocate_ctx_latents: If False, skip ``ttnn.deallocate(ctx_tt_pipe)`` at the end (for a
            shared cached context tensor on non-CFG runs).

    Returns:
        Denoised latents ``[B, frames, 64]``: CPU float32 :class:`torch.Tensor`, or an on-device
        :class:`ttnn.Tensor` when *return_device_latents* is True.
    """
    _ = act_dtype  # DiT uses BF16 row-major staging from NumPy, not this dtype.
    num_steps = len(t_schedule)
    if num_steps < 1:
        raise ValueError("t_schedule must be non-empty")

    frames_i = int(frames)
    c_lat = 64
    cfg_lo = float(cfg_interval_start)
    cfg_hi = float(cfg_interval_end)
    gs = float(guidance_scale)

    # ``ttnn.rand`` is uniform in [from,to]; ACE-Step uses Gaussian latents like ``torch.randn``.
    if not hasattr(ttnn, "randn"):
        raise RuntimeError("This path needs ``ttnn.randn`` (Gaussian) for latent init; ``ttnn.rand`` is uniform-only.")
    xt_tt = ttnn.randn(
        (1, frames_i, c_lat),
        device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mem,
        seed=int(np.uint32(int(seed))),
    )

    if enc_mask is None and encoder_attention_mask_b1qk is None:
        raise ValueError("Provide enc_mask or encoder_attention_mask_b1qk.")

    encoder_attn_1d_bk_np: Optional[np.ndarray] = None
    if encoder_attention_mask_b1qk is None:
        assert enc_mask is not None
        if isinstance(enc_mask, np.ndarray):
            encoder_keep_np_single = np.asarray(enc_mask, dtype=np.float32)
        else:
            encoder_keep_np_single = np.asarray(enc_mask.detach().cpu().numpy(), dtype=np.float32)
        if encoder_keep_np_single.ndim != 2:
            raise ValueError(f"encoder_attention_mask must be rank-2 [B,S], got {encoder_keep_np_single.shape}")
        encoder_keep_np_single = (encoder_keep_np_single > np.float32(0.0)).astype(np.bool_)
        encoder_attn_1d_bk_np = (
            np.concatenate([encoder_keep_np_single, encoder_keep_np_single], axis=0)
            if do_cfg
            else encoder_keep_np_single
        )

    prebuilt = enc_tt_pipe is not None
    if prebuilt:
        if ctx_tt_pipe is None:
            raise ValueError("ctx_tt_pipe is required when enc_tt_pipe is set.")
    elif enc_hs is None or ctx_lat is None:
        raise ValueError("Host denoise path requires enc_hs and ctx_lat (or pass enc_tt_pipe and ctx_tt_pipe).")
    elif do_cfg and null_emb is None:
        raise ValueError("Host CFG path requires null_emb.")

    if not prebuilt:
        if do_cfg:
            assert enc_hs is not None and null_emb is not None and ctx_lat is not None
            enc_tt_pipe = bf16_row_from_numpy_bc(
                np.concatenate(
                    [to_numpy_f32(enc_hs), to_numpy_f32(null_emb.expand_as(enc_hs))],
                    axis=0,
                ),
                device=device,
                dram=mem,
            )
            ctx_row_one = bf16_row_from_numpy_bc(to_numpy_f32(ctx_lat), device=device, dram=mem)
            ctx_tt_pipe = concat_duplicate_batch(ctx_row_one)
            try:
                ttnn.deallocate(ctx_row_one)
            except Exception:
                pass
        else:
            assert enc_hs is not None and ctx_lat is not None
            enc_tt_pipe = bf16_row_from_numpy_bc(to_numpy_f32(enc_hs), device=device, dram=mem)
            ctx_tt_pipe = bf16_row_from_numpy_bc(to_numpy_f32(ctx_lat), device=device, dram=mem)

    momentum_ttnn = TtnnMomentumBufferApg() if do_cfg and not use_adg else None
    _trace_each_step = os.environ.get("ACE_STEP_TRACY_EACH_DENOISE_STEP", "").lower() in ("1", "true", "yes")
    # Flush the device-profiler marker buffer every N denoise steps to avoid the per-RISC 12000-entry
    # ring buffer filling up across the loop (which silently drops markers, then trips
    # ``Device data missing`` asserts in ``tools/tracy/process_ops_logs.py``). No-op when neither
    # ``TT_METAL_DEVICE_PROFILER`` nor ``TTNN_OP_PROFILER`` is set, so production runs are unaffected.
    try:
        _flush_every = int(os.environ.get("ACE_STEP_PROFILER_FLUSH_EVERY", "1"))
    except ValueError:
        _flush_every = 1
    if _flush_every < 0:
        _flush_every = 0

    # Precompute per-step (temb, timestep_proj) once at loop start.
    #
    # ``pipe.compute_temb_tp`` runs the time-embed MLP (~10 ops per step) on the host-control side
    # and returns persistent device tensors. The denoise body below calls
    # ``pipe.forward_with_temb_tp(... temb=temb_per_step[i], tp=tp_per_step[i] ...)`` so the
    # per-step time-embed compute is paid once at start of ``generate`` instead of every Euler step.
    # This is also the precondition for wrapping the DiT body in a TTNN trace: a single capture can
    # serve all N steps when the only per-step difference is the (temb, tp) device-tensor identities,
    # which the orchestrator can stream onto persistent buffers via CQ 1 before each ``execute_trace``.
    pipe_batch = 2 if do_cfg else 1
    temb_per_step: list[ttnn.Tensor] = []
    tp_per_step: list[ttnn.Tensor] = []
    for _idx in range(num_steps):
        _temb, _tp = pipe.compute_temb_tp(int(_idx), target_batch=pipe_batch)
        temb_per_step.append(_temb)
        tp_per_step.append(_tp)

    def _diffusion_iterate(*, step_idx: int, t_curr_f: float, euler_dt: float) -> None:
        nonlocal xt_tt
        xt_row = fp32_tile_to_row_bf16(xt_tt, dram=mem)
        if do_cfg:
            xt_pipe_in = concat_duplicate_batch(xt_row)
            try:
                ttnn.deallocate(xt_row)
            except Exception:
                pass
        else:
            xt_pipe_in = xt_row

        acoustic = pipe.forward_with_temb_tp(
            xt_bt64=xt_pipe_in,
            context_latents_bt128=ctx_tt_pipe,
            encoder_hidden_states_btd=enc_tt_pipe,
            temb_bd=temb_per_step[int(step_idx)],
            timestep_proj_b6d=tp_per_step[int(step_idx)],
            attention_mask_1d_bt=None,
            encoder_attention_mask_1d_bk=None if encoder_attention_mask_b1qk is not None else encoder_attn_1d_bk_np,
            encoder_attention_mask_b1qk=encoder_attention_mask_b1qk,
        )

        if do_cfg:
            apply_cfg_now = cfg_lo <= t_curr_f <= cfg_hi
            vpc_rm = slice_batch_btc(acoustic, 0, 1, frames_i, c_lat)
            vpu_rm = slice_batch_btc(acoustic, 1, 2, frames_i, c_lat)
            if apply_cfg_now:
                if use_adg:
                    vt_tt = adg_guidance_velocity_ttnn(
                        xt_tt,
                        vpc_rm,
                        vpu_rm,
                        float(t_curr_f),
                        gs,
                        device=device,
                        dram=mem,
                    )
                else:
                    vt_tt = apg_guidance_velocity_ttnn(
                        vpc_rm,
                        vpu_rm,
                        gs,
                        momentum_buffer=momentum_ttnn,
                        dims=[1],
                        dram=mem,
                    )
            else:
                try:
                    ttnn.deallocate(vpu_rm)
                except Exception:
                    pass
                vt_tt = typecast_bf16_any_to_fp32_tile(vpc_rm, dram=mem)
        else:
            vt_tt = typecast_bf16_any_to_fp32_tile(acoustic, dram=mem)

        try:
            ttnn.deallocate(xt_pipe_in)
        except Exception:
            pass
        try:
            ttnn.deallocate(acoustic)
        except Exception:
            pass

        xt_old = xt_tt
        xt_tt = euler_subtract_v_dt(xt=xt_tt, vt=vt_tt, dt=float(euler_dt), dram=mem)
        try:
            ttnn.deallocate(vt_tt)
        except Exception:
            pass
        try:
            ttnn.deallocate(xt_old)
        except Exception:
            pass

        if progress_fn is not None:
            progress_fn(step_idx, num_steps, t_curr_f, float(euler_dt))

    for step_idx in range(num_steps - 1):
        if _trace_each_step:
            _ace_step_prof_signpost("Denoise Step", f"step {step_idx}/{num_steps}")
        t_curr_f = float(t_schedule[step_idx])
        t_next_f = float(t_schedule[step_idx + 1])
        dt = t_curr_f - t_next_f
        _diffusion_iterate(step_idx=step_idx, t_curr_f=t_curr_f, euler_dt=dt)
        if _flush_every and ((step_idx + 1) % _flush_every) == 0:
            _ace_step_flush_device_profiler(device)

    if _trace_each_step:
        _ace_step_prof_signpost("Denoise Step", f"step {num_steps - 1}/{num_steps}")
    t_curr_final = float(t_schedule[-1])
    _diffusion_iterate(step_idx=num_steps - 1, t_curr_f=t_curr_final, euler_dt=t_curr_final)

    _ace_step_flush_device_profiler(device)

    try:
        ttnn.deallocate(enc_tt_pipe)
        if deallocate_ctx_latents:
            ttnn.deallocate(ctx_tt_pipe)
        if encoder_attention_mask_b1qk is not None:
            ttnn.deallocate(encoder_attention_mask_b1qk)
    except Exception:
        pass
    # Free the precomputed per-step time embeddings; otherwise they'd accumulate across generate()
    # calls (N tensors per step × 2 for temb+tp).
    for _t in temb_per_step:
        try:
            ttnn.deallocate(_t)
        except Exception:
            pass
    for _t in tp_per_step:
        try:
            ttnn.deallocate(_t)
        except Exception:
            pass
    if momentum_ttnn is not None:
        momentum_ttnn.reset()

    if return_device_latents:
        return xt_tt

    # Single device→Torch copy of latents for host VAE or other CPU consumers.
    pred_latents = ttnn.to_torch(xt_tt, dtype=torch.float32).contiguous()
    try:
        ttnn.deallocate(xt_tt)
    except Exception:
        pass
    return pred_latents


class AceStepE2EModel:
    """End-to-end ACE-Step v1.5: text → DiT (TTNN) → VAE → waveform.

    Stages:
        1. Text encoding (TTNN Qwen3-Embedding encoder; Hugging Face tokenizer → NumPy only)
        2. Instrumental condition (TTNN ``TtAceStepInstrumentalConditionEncoder``, same as
           ``run_prompt_to_wav.py`` with ``--ttnn-condition-embedding``)
        3. TTNN DiT denoising loop
        4. VAE decode (TTNN Oobleck)
    """

    def __init__(
        self,
        config: E2EConfig,
        device: ttnn.Device,
    ) -> None:
        self.config = config
        self.device = device
        if hasattr(device, "enable_program_cache"):
            device.enable_program_cache()

        self.act_dtype = getattr(ttnn, "bfloat16")
        self.mem = getattr(ttnn, "DRAM_MEMORY_CONFIG")

        self._tokenizer = None
        self._qwen: Optional[TtQwen3EmbeddingEncoder] = None
        self._condition_encoder: Optional[TtAceStepInstrumentalConditionEncoder] = None
        self._tt_vae: Optional[TtOobleckVaeDecoder] = None
        self._ctx_bt128_cached: Optional[ttnn.Tensor] = None

        self._load_silence_latent()

        self.t_schedule = _build_t_schedule(
            shift=config.shift,
            infer_steps=config.infer_steps,
        )
        self.timesteps_host = np.asarray(self.t_schedule + [0.0], dtype=np.float32)
        self.frames = int(round(config.duration_sec * 25.0))

        self.pipe = AceStepV15TTNNPipeline(
            device=device,
            checkpoint_safetensors_path=config.checkpoint_safetensors_path,
            timesteps_host=self.timesteps_host,
            expected_input_length=self.frames,
        )

        self._init_condition_encoder()
        self._init_qwen_encoder()
        self._init_ttnn_vae()
        self._ctx_bt128_cached = self._ctx_latents_ttnn()
        # Drain any device-profiler markers accumulated by weight uploads / conv-weight prep / JIT
        # during init. Without this, the 12000-marker per-RISC ring buffer is already full before the
        # first ``generate()`` call gets a chance to flush — markers from the first few DiT layers
        # then get silently dropped, and ``cpp_device_perf_report.csv`` ends up missing op rows that
        # ``tools/tracy/process_ops_logs.py`` later asserts on. No-op when device profiling is off.
        _ace_step_flush_device_profiler(self.device)

    def _init_qwen_encoder(self) -> None:
        qwen_st = self.config.qwen_safetensors_path
        if qwen_st is None:
            qwen_st = str(Path(self.config.text_model_dir) / "model.safetensors")
        if not Path(qwen_st).is_file():
            raise FileNotFoundError(f"Missing Qwen embedding weights at {qwen_st}")
        self._qwen = TtQwen3EmbeddingEncoder(
            device=self.device,
            hf_model_dir=str(self.config.text_model_dir),
            qwen_safetensors_path=qwen_st,
        )

    def _init_ttnn_vae(self) -> None:
        vdir = Path(self.config.vae_dir)
        if not (vdir / "config.json").is_file():
            raise FileNotFoundError(f"TTNN VAE expects a Hugging Face-style folder with config.json at {vdir}.")
        self._tt_vae = TtOobleckVaeDecoder.from_hf_vae_dir(
            str(vdir),
            device=self.device,
            latent_frames=int(self.frames),
            batch_size=1,
            activation_dtype=self.act_dtype,
            weights_dtype=self.act_dtype,
        )

    def _init_condition_encoder(self) -> None:
        if not Path(self.config.checkpoint_safetensors_path).is_file():
            raise FileNotFoundError(
                f"TTNN condition encoder needs DiT weights at {self.config.checkpoint_safetensors_path}."
            )
        self._condition_encoder = TtAceStepInstrumentalConditionEncoder(
            device=self.device,
            checkpoint_safetensors_path=self.config.checkpoint_safetensors_path,
            dtype=self.act_dtype,
        )

    def _load_silence_latent(self) -> None:
        silence = torch.load(self.config.silence_latent_path, map_location="cpu").to(torch.float32)
        if silence.ndim != 3:
            raise RuntimeError(f"Unexpected silence_latent rank: {tuple(silence.shape)}")
        if int(silence.shape[-1]) == 64:
            pass
        elif int(silence.shape[1]) == 64:
            silence = silence.transpose(1, 2).contiguous()
        else:
            raise RuntimeError(f"Unexpected silence_latent shape: {tuple(silence.shape)}")
        self._silence_np = silence.contiguous().numpy()

    def encode_text(self, prompt: str) -> tuple[ttnn.Tensor, np.ndarray]:
        """Encode a text prompt with TTNN Qwen (HF tokenizer → NumPy tokens only).

        Returns:
            (``text_hs_tt`` ``[1,1,S,D]`` on device, ``attention_mask`` ``[1,S]`` float in ``{0,1}``)
        """
        from transformers import AutoTokenizer

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.text_model_dir)
        dit_instruction = "Fill the audio semantic mask based on the given conditions:"
        metas = {"caption": prompt, "duration": self.config.duration_sec, "language": "en"}
        text_prompt = f"""# Instruction
{dit_instruction}

# Caption
{prompt}

# Metas
{metas}<|endoftext|>
"""
        tokens = self._tokenizer(text_prompt, padding="max_length", truncation=True, max_length=256)
        input_ids_np = np.asarray(tokens["input_ids"], dtype=np.uint32).reshape(1, -1)
        attn_mask_np = np.asarray(tokens["attention_mask"], dtype=np.float32).reshape(1, -1)
        if self._qwen is None:
            raise RuntimeError("Qwen TTNN encoder was not initialized.")
        text_hs_tt = self._qwen.forward(input_ids_np, attn_mask_np)
        return text_hs_tt, attn_mask_np

    def _ctx_latents_ttnn(self) -> ttnn.Tensor:
        """Silence-derived src latents + chunk mask on device (``[1,T,128]``)."""
        frames = int(self.frames)
        src = np.asarray(self._silence_np[:, :frames, :], dtype=np.float32)
        if src.shape[1] < frames:
            rep = (frames + src.shape[1] - 1) // int(src.shape[1])
            src = np.tile(src, (1, rep, 1))[:, :frames, :]
        chunk_np = np.ones((1, frames, 64), dtype=np.float32)
        src_latents_tt = ttnn.as_tensor(
            np.ascontiguousarray(src),
            device=self.device,
            dtype=self.act_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.mem,
        )
        chunk_masks_tt = ttnn.as_tensor(
            chunk_np,
            device=self.device,
            dtype=self.act_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.mem,
        )
        ctx_tt_one = ttnn.concat([src_latents_tt, chunk_masks_tt], dim=-1)
        try:
            ttnn.deallocate(src_latents_tt)
            ttnn.deallocate(chunk_masks_tt)
        except Exception:
            pass
        return ctx_tt_one

    def decode_vae(
        self,
        pred_latents: Union[torch.Tensor, np.ndarray, ttnn.Tensor],
        *,
        return_waveform_ttnn: bool = False,
    ) -> Union[torch.Tensor, ttnn.Tensor]:
        """Decode latents to waveform via the TTNN Oobleck VAE (tiled along time if needed).

        Args:
            pred_latents: ``[1, frames, 64]`` on host (NumPy or Torch) or already on device as TTNN
                (same as ``run_prompt_to_wav``: DiT TILE float32 latents need no re-upload).
            return_waveform_ttnn: If True, return ``wav_tt`` on device (no peak normalization).

        Returns:
            waveform [1, channels, samples] normalized to [-1, 1] (default), or raw ``ttnn`` audio
            when *return_waveform_ttnn* is True.
        """
        if self._tt_vae is None:
            raise RuntimeError("TTNN VAE was not initialized.")
        owns_lat_tt = True
        if isinstance(pred_latents, ttnn.Tensor):
            lat_tt = pred_latents
            owns_lat_tt = False
        elif hasattr(pred_latents, "detach"):
            lat_np = pred_latents.detach().float().cpu().contiguous().numpy()
            lat_tt = ttnn.as_tensor(
                np.asarray(lat_np, dtype=np.float32),
                device=self.device,
                dtype=self.act_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=self.mem,
            )
        else:
            lat_tt = ttnn.as_tensor(
                np.asarray(pred_latents, dtype=np.float32),
                device=self.device,
                dtype=self.act_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=self.mem,
            )
        chunk = int(os.environ.get("ACE_STEP_VAE_CHUNK_LATENTS", str(self.config.vae_chunk_latents)))
        overlap = int(os.environ.get("ACE_STEP_VAE_OVERLAP_LATENTS", str(self.config.vae_overlap_latents)))
        wav_tt = self._tt_vae.decode_tiled(lat_tt, chunk_size=chunk, overlap=overlap)
        try:
            if owns_lat_tt:
                ttnn.deallocate(lat_tt)
        except Exception:
            pass
        if return_waveform_ttnn:
            return wav_tt
        wav_bt_c = ttnn.to_torch(wav_tt, dtype=torch.float32).contiguous().numpy()
        wav_bct = np.ascontiguousarray(np.swapaxes(wav_bt_c, 1, 2))
        peak = np.maximum(np.amax(np.abs(wav_bct), axis=(1, 2), keepdims=True), 1e-8)
        wav_np = np.clip(wav_bct / peak, -1.0, 1.0)
        try:
            ttnn.deallocate(wav_tt)
        except Exception:
            pass
        return torch.from_numpy(wav_np)

    def generate(
        self,
        prompt: str,
        *,
        return_waveform_ttnn: bool = False,
    ) -> Union[torch.Tensor, ttnn.Tensor]:
        """Full end-to-end: text prompt → waveform tensor.

        Args:
            prompt: text description of the music.
            return_waveform_ttnn: If True, return raw TTNN waveform from the VAE (no host peak norm).

        Returns:
            waveform [1, channels, samples] normalized to [-1, 1] by default, or device tensor
            when *return_waveform_ttnn* is True.
        """
        if self._condition_encoder is None:
            raise RuntimeError("TTNN condition encoder was not initialized.")
        if self._ctx_bt128_cached is None:
            raise RuntimeError("Cached context latents were not initialized.")
        do_cfg = self.config.guidance_scale > 1.0 + 1e-6

        # Start each generate() with an empty device-profiler ring so the per-layer flush downstream
        # has full 12000-marker headroom; no-op when device profiling is off.
        _ace_step_flush_device_profiler(self.device)

        _ace_step_prof_signpost("ACE-Step E2E", "Start text encoding")
        text_hs_tt, attn_mask_np = self.encode_text(prompt)
        _ace_step_flush_device_profiler(self.device)

        _ace_step_prof_signpost("ACE-Step E2E", "Start condition encoding")
        enc_hs_tt_one, enc_mask_np, null_emb_tt = self._condition_encoder.forward(text_hs_tt, attn_mask_np)
        _ace_step_flush_device_profiler(self.device)
        try:
            ttnn.deallocate(text_hs_tt)
        except Exception:
            pass

        ctx_tt_one = self._ctx_bt128_cached
        enc_tt_pipe: ttnn.Tensor
        ctx_tt_pipe: ttnn.Tensor

        if do_cfg:
            d_enc = int(enc_hs_tt_one.shape[-1])
            s_enc = int(enc_hs_tt_one.shape[1])
            # ``null_emb_tt`` is the persistent ``self._condition_encoder.null_condition_emb``
            # initialized once at construction. ``ttnn.reshape`` returns a view in this build, so
            # deallocating ``null_4d`` previously freed the cached buffer and broke subsequent
            # ``generate`` calls with "Tensor is not allocated". Clone first so we own a temp buffer
            # and can free it safely.
            null_owned = (
                ttnn.clone(null_emb_tt) if hasattr(ttnn, "clone") else ttnn.to_memory_config(null_emb_tt, self.mem)
            )
            null_4d = ttnn.reshape(null_owned, (1, 1, 1, d_enc))
            null_rep_4d = ttnn.repeat(null_4d, (1, 1, s_enc, 1))
            null_rep = ttnn.reshape(null_rep_4d, (1, s_enc, d_enc))
            enc_tt_pipe = ttnn.concat([enc_hs_tt_one, null_rep], dim=0)
            ctx_tt_pipe = concat_duplicate_batch(ctx_tt_one)
            try:
                ttnn.deallocate(enc_hs_tt_one)
                # ``null_4d`` is a view of ``null_owned`` and ``null_rep`` is a view of ``null_rep_4d``;
                # deallocating the owners is sufficient (and avoids double-free on the views).
                ttnn.deallocate(null_rep_4d)
                ttnn.deallocate(null_owned)
            except Exception:
                pass
        else:
            enc_tt_pipe = enc_hs_tt_one
            ctx_tt_pipe = ctx_tt_one

        enc_row = np.asarray(enc_mask_np, dtype=np.float32).reshape(1, -1)
        enc_attn_1d_bk = np.concatenate([enc_row, enc_row], axis=0) if do_cfg else enc_row

        _ace_step_prof_signpost("ACE-Step E2E", "Start DiT preparation (SDPA mask)")
        b_mask = 2 if do_cfg else 1
        xt_dummy = ttnn.zeros(
            (b_mask, int(self.frames), 64),
            device=self.device,
            dtype=self.act_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.mem,
        )
        ctx_for_mask = concat_duplicate_batch(ctx_tt_one) if do_cfg else ctx_tt_one
        mask_tt = self.pipe.build_encoder_attention_mask_b1qk_optional(
            xt_bt64=xt_dummy,
            context_latents_bt128=ctx_for_mask,
            encoder_hidden_states_btd=enc_tt_pipe,
            encoder_attention_mask_1d_bk=enc_attn_1d_bk,
        )
        try:
            ttnn.deallocate(xt_dummy)
            if do_cfg:
                ttnn.deallocate(ctx_for_mask)
        except Exception:
            pass

        loop_kw = dict(
            pipe=self.pipe,
            device=self.device,
            act_dtype=self.act_dtype,
            mem=self.mem,
            t_schedule=self.t_schedule,
            frames=self.frames,
            enc_hs=None,
            ctx_lat=None,
            null_emb=None,
            do_cfg=do_cfg,
            seed=self.config.seed,
            use_adg=self.config.use_adg,
            guidance_scale=float(self.config.guidance_scale),
            cfg_interval_start=float(self.config.cfg_interval_start),
            cfg_interval_end=float(self.config.cfg_interval_end),
            enc_tt_pipe=enc_tt_pipe,
            ctx_tt_pipe=ctx_tt_pipe,
            return_device_latents=True,
            encoder_attention_mask_b1qk=mask_tt,
            deallocate_ctx_latents=do_cfg,
        )
        if mask_tt is None:
            loop_kw["enc_mask"] = enc_mask_np
        else:
            loop_kw["enc_mask"] = None

        _ace_step_prof_signpost("ACE-Step E2E", "Start denoising loop")
        pred_latents = run_ttnn_denoise_loop(**loop_kw)
        _ace_step_flush_device_profiler(self.device)
        try:
            _ace_step_prof_signpost("ACE-Step E2E", "Start VAE decode")
            wav = self.decode_vae(pred_latents, return_waveform_ttnn=return_waveform_ttnn)
            _ace_step_flush_device_profiler(self.device)
            _ace_step_prof_signpost("ACE-Step E2E", "Generation complete")
        finally:
            try:
                ttnn.deallocate(pred_latents)
            except Exception:
                pass
        return wav
