"""
ACE-Step v1.5 – persistent HTTP inference service.

Loads DiT, VAE, Qwen3-Embedding, and 5Hz-LM weights exactly once at
startup.  Every /generate call reuses the in-memory models; a structured
log line marks each component as LOAD (disk I/O) or REUSE (memory):

    ⏳ LOAD   TTNN-device
    ⏳ LOAD   DiT-handler          path=…/acestep-v15-base
    ⏳ LOAD   5Hz-LM               path=…/acestep-5Hz-lm-1.7B
    ⏳ LOAD   Qwen3-EmbeddingEnc   path=…/Qwen3-Embedding-0.6B/model.safetensors
    ⏳ LOAD   AudioCodeDetok       path=…/acestep-v15-base/model.safetensors
    ⏳ LOAD   ConditionEncoder     path=…/acestep-v15-base/model.safetensors
    ⏳ LOAD   DiT-pipeline         path=…/acestep-v15-base/model.safetensors
    ⏳ LOAD   VAE-decoder          path=…/vae
    ✅ Service ready – all weights in memory; disk I/O is done.

    ♻  REUSE  DiT-handler          [already in TTNN device memory]
    ♻  REUSE  5Hz-LM               [already in TTNN device memory]
    …

Start the service (weights load once):
    python serve_prompt_to_wav.py [--host 0.0.0.0] [--port 8765] \\
                                  [--ckpt_dir PATH] [--variant base] ...

Generate from a second terminal (no weight loading):
    curl -s -X POST http://localhost:8765/generate \\
         -H 'Content-Type: application/json' \\
         -d '{"prompt":"upbeat jazz","duration_sec":10.0,"out":"jazz.wav"}' | jq .

Health check:
    curl http://localhost:8765/health
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────
# Structured weight-reuse logger (shared with run_prompt_to_wav / weight_cache)
# ─────────────────────────────────────────────────────────────────
from loguru import logger as log

from models.demos.ace_step_v1_5.official_lm_preprocess import configure_acestep_logging
from models.demos.ace_step_v1_5.weight_cache import log_weight_load as _log_load
from models.demos.ace_step_v1_5.weight_cache import log_weight_reuse as _log_weight_reuse
from models.demos.ace_step_v1_5.weight_cache import log_weights_ready as _log_ready


def _log_reuse(component: str) -> None:
    _log_weight_reuse(component, source="device")


# ─────────────────────────────────────────────────────────────────
# Model registry – holds every loaded component across requests
# ─────────────────────────────────────────────────────────────────
class AceStepModelRegistry:
    """
    Holds every loaded model component.

    Call :meth:`load` once at startup.  After that every attribute is
    permanently live; :meth:`generate` runs inference against them.
    A threading.Lock ensures only one request runs on the TT device at
    a time (the device is single-occupancy).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False

        # ── TTNN runtime ──────────────────────────────────────────
        self.ttnn: Any = None
        self.dev: Any = None  # TT device (kept open forever)
        self.mem: Any = None  # DRAM_MEMORY_CONFIG

        # ── Host-side models (CPU/PyTorch) ────────────────────────
        self.dit_handler: Any = None  # AceStepHandler  (DiT + condition weights)
        self.llm_handler: Any = None  # LocalFiveHzLMHandler (5Hz LM)

        # ── TTNN encoder / condition models ───────────────────────
        self.qwen_enc: Any = None  # TtQwen3EmbeddingEncoder
        self.audio_detok: Any = None  # TtAceStepAudioCodeDetokenizer
        self.condition_enc: Any = None  # TtAceStepInstrumentalConditionEncoder

        # ── TTNN DiT + VAE ────────────────────────────────────────
        self.pipe: Any = None  # AceStepV15TTNNPipeline
        self.tt_vae: Any = None  # TtOobleckVaeDecoder

        # ── Metadata ──────────────────────────────────────────────
        self._safetensors_path: str = ""
        self._vae_dir: str = ""
        self._t_schedule_key: tuple = ()  # (infer_steps, shift, variant)
        self._t_schedule: list[float] = []
        self._ref_root: Path | None = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    # ── Startup: load everything ──────────────────────────────────
    def load(
        self,
        *,
        ckpt_dir: Path,
        variant: str,
        lm_variant: str,
        device_id: int,
        no_ttnn_strict: bool,
        use_trace: bool,
        t_schedule: list[float],
        safetensors_path: Path,
        vae_dir: Path,
        text_model_dir: Path,
        ref_root: Path,
        experimental_5hz_ttnn_lm: bool,
        use_torch_vae: bool,
    ) -> None:
        """Load all model weights from disk.  Subsequent calls are no-ops."""
        if self._loaded:
            _log_reuse("all-components (service already initialised)")
            return

        import torch  # noqa: F401 – ensure cuda context before TTNN

        # 1. Open the TT device ────────────────────────────────────
        _log_load("TTNN-device", f"device_id={device_id}")
        _configure_ttnn_runtime(no_ttnn_strict=no_ttnn_strict)
        import ttnn

        self.ttnn = ttnn
        if not no_ttnn_strict and hasattr(ttnn, "CONFIG") and hasattr(ttnn.CONFIG, "throw_exception_on_fallback"):
            ttnn.CONFIG.throw_exception_on_fallback = True
        num_cqs = 2 if use_trace else 1
        self.dev = _open_tt_device(ttnn, device_id=device_id, num_command_queues=num_cqs)
        if hasattr(self.dev, "enable_program_cache"):
            self.dev.enable_program_cache()
        self.mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        if self.mem is None:
            raise RuntimeError("TTNN build missing DRAM_MEMORY_CONFIG")

        # 2. AceStepHandler – DiT weights on CPU ───────────────────
        _log_load("DiT-handler", str(safetensors_path.parent))
        from acestep.handler import AceStepHandler

        from models.demos.ace_step_v1_5.ttnn_impl.five_hz_lm import LocalFiveHzLMHandler

        self.dit_handler = AceStepHandler()
        status, ok = self.dit_handler.initialize_service(
            project_root=str(ref_root),
            config_path=variant,
            device="cpu",
            use_flash_attention=False,
        )
        log.info("  DiT-handler status: %s", status)
        if not ok:
            raise RuntimeError("AceStepHandler.initialize_service failed")

        # 3. 5Hz LM weights ────────────────────────────────────────
        _log_load("5Hz-LM", str(ckpt_dir / lm_variant))
        self.llm_handler = LocalFiveHzLMHandler()
        lm_trace = bool(use_trace) and bool(experimental_5hz_ttnn_lm)
        status, ok = self.llm_handler.initialize(
            checkpoint_dir=str(ckpt_dir),
            lm_model_path=lm_variant,
            backend="pt",
            device="cpu",
            ttnn_causal_device=self.dev if experimental_5hz_ttnn_lm else None,
            experimental_ttnn_causal_lm=experimental_5hz_ttnn_lm,
            ttnn_lm_prefill_trace=lm_trace,
            ttnn_lm_decode_trace=lm_trace,
        )
        log.info("  5Hz-LM status: %s", status)
        if not ok:
            raise RuntimeError("5Hz LM initialize failed")
        self.llm_handler.set_ttnn_logits_device(self.dev)

        # 4. Qwen3 Embedding Encoder ───────────────────────────────
        qwen_safetensors = text_model_dir / "model.safetensors"
        _log_load("Qwen3-EmbeddingEnc", str(qwen_safetensors))
        from models.demos.ace_step_v1_5.ttnn_impl.qwen3_embedding_ace_step import (
            AceStepQwen3Encoder as TtQwen3EmbeddingEncoder,
        )

        self.qwen_enc = TtQwen3EmbeddingEncoder(
            device=self.dev,
            hf_model_dir=str(text_model_dir),
            qwen_safetensors_path=str(qwen_safetensors),
        )

        # 5. Audio-code detokenizer ────────────────────────────────
        _log_load("AudioCodeDetok", str(safetensors_path))
        from models.demos.ace_step_v1_5.ttnn_impl.audio_code_detokenizer import TtAceStepAudioCodeDetokenizer

        self.audio_detok = TtAceStepAudioCodeDetokenizer(
            device=self.dev,
            checkpoint_safetensors_path=str(safetensors_path),
            dtype=getattr(ttnn, "bfloat16", None),
        )

        # 6. Permanently patch dit_handler to use TTNN encoders ────
        #    We call attach but keep the patch live (don't call restore).
        log.info("  Attaching TTNN preprocessing patch to dit_handler (permanent)")
        from models.demos.ace_step_v1_5.official_lm_preprocess import attach_payload_preprocess_ttnn

        attach_payload_preprocess_ttnn(
            self.dit_handler,
            tt_qwen_encoder=self.qwen_enc,
            tt_audio_detokenizer=self.audio_detok,
            max_seq_len=256,
        )
        # Intentionally discard the returned restore-fn — the patch stays active.

        # 7. Condition encoder ─────────────────────────────────────
        _log_load("ConditionEncoder", str(safetensors_path))
        from models.demos.ace_step_v1_5.ttnn_impl.condition_encoder import TtAceStepInstrumentalConditionEncoder

        self.condition_enc = TtAceStepInstrumentalConditionEncoder(
            device=self.dev,
            checkpoint_safetensors_path=str(safetensors_path),
            dtype=getattr(ttnn, "bfloat16", None),
        )

        # 8. DiT pipeline (safetensors → TTNN device) ──────────────
        _log_load("DiT-pipeline", str(safetensors_path))
        timesteps_host = np.asarray(t_schedule + [0.0], dtype=np.float32)
        from models.demos.ace_step_v1_5.ttnn_impl.full_pipeline import AceStepV15TTNNPipeline

        self.pipe = AceStepV15TTNNPipeline(
            device=self.dev,
            checkpoint_safetensors_path=str(safetensors_path),
            timesteps_host=timesteps_host,
            expected_input_length=None,  # lazy-pack: supports variable duration
        )
        self._t_schedule = t_schedule
        self._t_schedule_key = tuple(t_schedule)

        # 9. VAE decoder (safetensors → TTNN device) ───────────────
        if not use_torch_vae:
            _log_load("VAE-decoder-TTNN", str(vae_dir))
            from models.demos.ace_step_v1_5.ttnn_impl.oobleck_vae_decoder import TtOobleckVaeDecoder

            self.tt_vae = TtOobleckVaeDecoder.from_hf_vae_dir(
                str(vae_dir),
                device=self.dev,
                latent_frames=None,
                batch_size=1,
                activation_dtype=getattr(ttnn, "bfloat16", None),
                weights_dtype=getattr(ttnn, "bfloat16", None),
            )
        else:
            log.info("  VAE-decoder: using PyTorch fallback (--torch-vae)")

        self._safetensors_path = str(safetensors_path)
        self._vae_dir = str(vae_dir)
        self._ref_root = ref_root
        self._loaded = True
        _log_ready()

    # ── Per-request: log REUSE, then run inference ────────────────
    def generate(
        self,
        *,
        prompt: str,
        duration_sec: float,
        seed: int,
        variant: str,
        lm_variant: str,
        t_schedule: list[float],
        guidance_scale: float,
        use_adg: bool,
        cfg_interval_start: float,
        cfg_interval_end: float,
        use_trace: bool,
        use_torch_vae: bool,
        vae_chunk_latents: int,
        vae_overlap_latents: int,
        out_path: Path,
        exp_ttnn_lm: bool,
        ckpt_dir: Path,
    ) -> Path:
        """Run inference using in-memory models.  Logs REUSE for every component."""
        if not self._loaded:
            raise RuntimeError("Registry not loaded — call load() first")

        with self._lock:
            return self._generate_locked(
                prompt=prompt,
                duration_sec=duration_sec,
                seed=seed,
                variant=variant,
                lm_variant=lm_variant,
                t_schedule=t_schedule,
                guidance_scale=guidance_scale,
                use_adg=use_adg,
                cfg_interval_start=cfg_interval_start,
                cfg_interval_end=cfg_interval_end,
                use_trace=use_trace,
                use_torch_vae=use_torch_vae,
                vae_chunk_latents=vae_chunk_latents,
                vae_overlap_latents=vae_overlap_latents,
                out_path=out_path,
                exp_ttnn_lm=exp_ttnn_lm,
                ckpt_dir=ckpt_dir,
            )

    def _generate_locked(self, **kw) -> Path:
        """Inner generate — called with self._lock held."""
        import torch

        from models.demos.ace_step_v1_5.acestep_preprocess_shim import GenerationConfig, GenerationParams
        from models.demos.ace_step_v1_5.official_lm_preprocess import (
            build_filtered_dit_kwargs_for_handler,
            handler_prepare_condition_payload,
        )
        from models.demos.ace_step_v1_5.ttnn_impl.dit_sampling_ttnn import concat_duplicate_batch
        from models.demos.ace_step_v1_5.ttnn_impl.e2e_model_tt import _E2EDenoiseTrace, run_ttnn_denoise_loop

        prompt = kw["prompt"]
        duration_sec = float(kw["duration_sec"])
        seed = int(kw["seed"])
        variant = kw["variant"]
        lm_variant = kw["lm_variant"]
        t_schedule = kw["t_schedule"]
        gs = float(kw["guidance_scale"])
        use_adg = bool(kw["use_adg"])
        cfg_lo = float(kw["cfg_interval_start"])
        cfg_hi = float(kw["cfg_interval_end"])
        use_trace = bool(kw["use_trace"])
        use_torch_vae = bool(kw["use_torch_vae"])
        vae_chunk_latents = int(kw["vae_chunk_latents"])
        vae_overlap_latents = int(kw["vae_overlap_latents"])
        out_path = Path(kw["out_path"])
        exp_ttnn_lm = bool(kw["exp_ttnn_lm"])
        ckpt_dir = Path(kw["ckpt_dir"])

        ttnn = self.ttnn
        dev = self.dev
        mem = self.mem

        torch.manual_seed(seed)
        np.random.seed(seed)

        log.info("─" * 60)
        log.info("REQUEST  prompt=%r  duration=%.1fs  seed=%d", prompt, duration_sec, seed)

        # ── Log REUSE for every component ─────────────────────────
        _log_reuse("DiT-handler")
        _log_reuse("5Hz-LM")
        _log_reuse("Qwen3-EmbeddingEnc")
        _log_reuse("AudioCodeDetok")
        _log_reuse("ConditionEncoder")

        if tuple(t_schedule) != self._t_schedule_key:
            log.warning(
                "Request t_schedule differs from loaded schedule (loaded %d steps, request %d steps). "
                "Using loaded schedule.  Restart service with matching --infer_steps/--shift to change.",
                len(self._t_schedule),
                len(t_schedule),
            )
            t_schedule = self._t_schedule
        _log_reuse("DiT-pipeline")
        if self.tt_vae is not None:
            _log_reuse("VAE-decoder-TTNN")

        # ── Run 5Hz LM + handler preprocessing ────────────────────
        params = GenerationParams(
            task_type="text2music",
            caption=prompt,
            lyrics="[Instrumental]",
            instrumental=True,
            reference_audio=None,
            duration=duration_sec,
            inference_steps=len(t_schedule),
            guidance_scale=gs,
            lm_cfg_scale=1.0 if exp_ttnn_lm else 2.0,
            use_adg=use_adg,
            cfg_interval_start=cfg_lo,
            cfg_interval_end=cfg_hi,
            shift=1.0,
            thinking=True,
            use_constrained_decoding=True,
            timesteps=None,
        )
        config = GenerationConfig(
            batch_size=1,
            use_random_seed=False,
            seeds=[seed],
            audio_format="wav",
            constrained_decoding_debug=True,
        )

        log.info("  Running 5Hz LM forward …")
        t0 = time.perf_counter()
        filtered = build_filtered_dit_kwargs_for_handler(
            self.dit_handler, self.llm_handler, params, config, progress=None
        )
        log.info("  5Hz LM done in %.2fs", time.perf_counter() - t0)

        # ── Condition encoding ─────────────────────────────────────
        log.info("  Running condition encoder …")
        t0 = time.perf_counter()
        payload, frames = handler_prepare_condition_payload(self.dit_handler, filtered)
        enc_hs_tt_one, enc_mask_np, ctx_tt_one, null_emb_tt = self.condition_enc.forward_payload(payload)
        enc_mask = torch.from_numpy(enc_mask_np).to(dtype=torch.float32)
        log.info("  Condition encoder done in %.2fs  frames=%d", time.perf_counter() - t0, frames)

        do_cfg = gs > 1.0 + 1e-6
        num_steps = len(t_schedule)

        # ── Build CFG-batched condition tensors ────────────────────
        if do_cfg:
            s_enc = int(enc_hs_tt_one.shape[1])
            d_enc = int(enc_hs_tt_one.shape[-1])
            null_4d = ttnn.reshape(null_emb_tt, (1, 1, 1, d_enc))
            null_rep_4d = ttnn.repeat(null_4d, (1, 1, s_enc, 1))
            null_rep = ttnn.reshape(null_rep_4d, (1, s_enc, d_enc))
            enc_tt_pipe = ttnn.concat([enc_hs_tt_one, null_rep], dim=0)
            ctx_tt_pipe = concat_duplicate_batch(ctx_tt_one)
            for _t in (enc_hs_tt_one, null_4d, null_rep_4d, null_rep, ctx_tt_one, null_emb_tt):
                try:
                    ttnn.deallocate(_t)
                except Exception:
                    pass
        else:
            enc_tt_pipe = enc_hs_tt_one
            ctx_tt_pipe = ctx_tt_one
            try:
                ttnn.deallocate(null_emb_tt)
            except Exception:
                pass

        # ── DiT denoising loop ─────────────────────────────────────
        log.info("  Running DiT denoise loop (%d steps, cfg=%s) …", num_steps, do_cfg)
        t0 = time.perf_counter()

        trace_state = _E2EDenoiseTrace()
        try:
            _trace_result = run_ttnn_denoise_loop(
                pipe=self.pipe,
                device=dev,
                act_dtype=getattr(ttnn, "bfloat16"),
                mem=mem,
                t_schedule=t_schedule,
                frames=int(frames),
                enc_mask=enc_mask,
                do_cfg=do_cfg,
                seed=seed,
                use_adg=use_adg,
                guidance_scale=gs,
                cfg_interval_start=cfg_lo,
                cfg_interval_end=cfg_hi,
                enc_tt_pipe=enc_tt_pipe,
                ctx_tt_pipe=ctx_tt_pipe,
                return_device_latents=(self.tt_vae is not None),
                progress_fn=lambda si, n, tc, dt: log.info(
                    "    step {}/{}  t={:.4f}  dt={:.4f}",
                    si + 1,
                    n - 1,
                    tc,
                    dt,
                ),
                trace_state=trace_state,
            )
        finally:
            trace_state.release(dev)

        log.info("  DiT loop done in %.2fs", time.perf_counter() - t0)

        # ── VAE decode ─────────────────────────────────────────────
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if self.tt_vae is not None:
            log.info("  VAE decode (TTNN) …")
            t0 = time.perf_counter()
            wav_tt = self.tt_vae.decode_tiled(_trace_result, chunk_size=vae_chunk_latents, overlap=vae_overlap_latents)
            wav_ntc = ttnn.to_torch(wav_tt, dtype=torch.float32).contiguous()
            try:
                ttnn.deallocate(wav_tt)
                ttnn.deallocate(_trace_result)
            except Exception:
                pass
            wav = wav_ntc.permute(0, 2, 1).detach().cpu().float()
            log.info("  VAE TTNN done in {%.2fs}", time.perf_counter() - t0)
        else:
            log.info("  VAE decode (PyTorch) …")
            import torch
            from diffusers.models import AutoencoderOobleck

            pred_latents = (
                _trace_result
                if isinstance(_trace_result, torch.Tensor)
                else ttnn.to_torch(_trace_result, dtype=torch.float32).contiguous()
            )
            torch_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            vae_dir = Path(self._vae_dir)
            vae = AutoencoderOobleck.from_pretrained(str(vae_dir)).eval().to(torch_dev)
            with torch.inference_mode():
                lat = pred_latents.transpose(1, 2).contiguous().to(device=torch_dev, dtype=next(vae.parameters()).dtype)
                wav = vae.decode(lat).sample.float().cpu()

        # Normalize and save
        peak = wav.abs().amax(dim=[1, 2], keepdim=True).clamp(min=1e-8)
        wav = (wav / peak).clamp(-1.0, 1.0)
        _save_wav(wav[0], out_path, sample_rate=48000)
        log.info("✅ Wrote: {%s}", out_path)
        return out_path


# ─────────────────────────────────────────────────────────────────
# Helpers copied / adapted from run_prompt_to_wav.py
# ─────────────────────────────────────────────────────────────────


def _prepend_ttnn_pkg_to_syspath() -> None:
    tt_metal_root = str(Path(__file__).resolve().parents[4])
    ttnn_pkg_root = str(Path(tt_metal_root) / "ttnn")
    for p in (tt_metal_root, ttnn_pkg_root):
        if p not in sys.path:
            sys.path.insert(0, p)


def _configure_ttnn_runtime(*, no_ttnn_strict: bool) -> None:
    _prepend_ttnn_pkg_to_syspath()
    if not no_ttnn_strict:
        os.environ["TTNN_CONFIG_OVERRIDES"] = '{"throw_exception_on_fallback": true}'


def _open_tt_device(ttnn: Any, *, device_id: int, num_command_queues: int = 1) -> Any:
    open_kwargs: dict = dict(
        device_id=int(device_id),
        l1_small_size=int(os.environ.get("ACE_STEP_L1_SMALL_SIZE", "98304")),
        trace_region_size=128 << 20,
    )
    if int(num_command_queues) > 1:
        open_kwargs["num_command_queues"] = int(num_command_queues)
    return ttnn.open_device(**open_kwargs)


def _save_wav(wav: Any, out_path: Path, sample_rate: int = 48000) -> None:
    import numpy as _np

    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio = wav.detach().float().cpu()
    if audio.ndim == 1:
        arr = audio.numpy()
    elif audio.ndim == 2:
        arr = (audio.transpose(0, 1).contiguous() if audio.shape[0] in (1, 2) else audio).numpy()
    else:
        raise ValueError(f"Expected rank 1 or 2 wav, got {tuple(audio.shape)}")
    try:
        import soundfile as sf

        sf.write(str(out_path), arr, samplerate=sample_rate)
        return
    except ModuleNotFoundError:
        pass
    from scipy.io import wavfile

    arr_i16 = (_np.clip(arr, -1.0, 1.0) * 32767.0).astype(_np.int16)
    wavfile.write(str(out_path), sample_rate, arr_i16)


_VALID_TIMESTEPS = [
    1.0,
    0.9545454545454546,
    0.9333333333333333,
    0.9,
    0.875,
    0.8571428571428571,
    0.8333333333333334,
    0.7692307692307693,
    0.75,
    0.6666666666666666,
    0.6428571428571429,
    0.625,
    0.5454545454545454,
    0.5,
    0.4,
    0.375,
    0.3,
    0.25,
    0.2222222222222222,
    0.125,
]
_SHIFT_TIMESTEPS: dict[float, list[float]] = {
    1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
    2.0: [
        1.0,
        0.9333333333333333,
        0.8571428571428571,
        0.7692307692307693,
        0.6666666666666666,
        0.5454545454545454,
        0.4,
        0.2222222222222222,
    ],
    3.0: [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3],
}

_DEFAULT_CKPT_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "ACE-Step-1.5-checkpoints"


def _build_t_schedule(*, shift: float, infer_steps: int, variant: str) -> list[float]:
    variant_l = (variant or "").lower()
    is_turbo = "turbo" in variant_l
    infer_steps = int(infer_steps)
    if infer_steps <= 1:
        raise ValueError("infer_steps must be >= 2")
    if is_turbo:
        s = min(_SHIFT_TIMESTEPS.keys(), key=lambda v: abs(v - float(shift)))
        if infer_steps == 8:
            return list(_SHIFT_TIMESTEPS[float(s)])
        lin = [1.0 - (i / float(infer_steps - 1)) for i in range(infer_steps)]
        mapped = [min(_VALID_TIMESTEPS, key=lambda v, t=t: abs(v - t)) for t in lin]
        out: list[float] = []
        for t in mapped:
            if not out or out[-1] != t:
                out.append(t)
        return sorted(out, reverse=True)
    t = [1.0 - (i / float(infer_steps)) for i in range(infer_steps)]
    if float(shift) != 1.0:
        sf_v = float(shift)
        t = [sf_v * x / (1.0 + (sf_v - 1.0) * x) for x in t]
    return t


_HF_REPO_MAP = {
    "acestep-v15-base": ("ACE-Step/acestep-v15-base", False),
    "acestep-v15-sft": ("ACE-Step/acestep-v15-sft", False),
    "acestep-v15-turbo": ("ACE-Step/Ace-Step1.5", True),
    "acestep-5Hz-lm-0.6B": ("ACE-Step/acestep-5Hz-lm-0.6B", False),
    "acestep-5Hz-lm-1.7B": ("ACE-Step/Ace-Step1.5", True),
    "acestep-5Hz-lm-4B": ("ACE-Step/acestep-5Hz-lm-4B", False),
    "vae": ("ACE-Step/Ace-Step1.5", True),
    "Qwen3-Embedding-0.6B": ("ACE-Step/Ace-Step1.5", True),
}


def _ensure_variant(name: str, ckpt_dir: Path) -> Path:
    local = ckpt_dir / name
    if any(local.glob("*.safetensors")) or any(local.glob("*.pt")):
        return local
    entry = _HF_REPO_MAP.get(name)
    if entry is None:
        raise FileNotFoundError(f"No HuggingFace repo mapping for variant '{name}'")
    repo_id, is_subfolder = entry
    from huggingface_hub import snapshot_download

    log.info("Downloading %s from %s …", name, repo_id)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if is_subfolder:
        snapshot_download(repo_id, allow_patterns=f"{name}/*", local_dir=str(ckpt_dir))
    else:
        snapshot_download(repo_id, local_dir=str(local))
    if not any(local.glob("*.safetensors")) and not any(local.glob("*.pt")):
        raise FileNotFoundError(f"Download succeeded but no weights found in {local}")
    return local


_WELL_KNOWN_REPO_ROOTS = [
    Path.home() / "proj_sdk" / "ACE-Step-1.5",
    Path.home() / "ACE-Step-1.5",
    Path("/opt") / "ACE-Step-1.5",
]
_VENDORED_ACESTEP_ROOT = Path(__file__).resolve().parent.parent / "torch_ref" / "_vendored_acestep"


def _resolve_ace_step_repo_root(*, ckpt_dir: str | None, ace_step_repo_root: str | None) -> Path | None:
    candidates: list[Path] = []
    if ace_step_repo_root:
        candidates.append(Path(ace_step_repo_root).expanduser().resolve())
    env = os.environ.get("ACE_STEP_REPO_ROOT")
    if env:
        candidates.append(Path(env).expanduser().resolve())
    candidates.append(_VENDORED_ACESTEP_ROOT)
    if ckpt_dir:
        cur = Path(ckpt_dir).expanduser().resolve()
        for _ in range(8):
            candidates.append(cur)
            if cur.parent == cur:
                break
            cur = cur.parent
    candidates.extend(_WELL_KNOWN_REPO_ROOTS)
    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if (c / "acestep" / "__init__.py").is_file():
            return c
    return None


# ─────────────────────────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────────────────────────
# Registry is module-level so the FastAPI lifespan can reference it.
_registry = AceStepModelRegistry()
_startup_args: dict = {}  # filled in main() before uvicorn.run()


def _make_app():
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ModuleNotFoundError as e:
        raise RuntimeError("fastapi + pydantic required: pip install 'fastapi[standard]'") from e

    app = FastAPI(title="ACE-Step v1.5 Persistent Service")

    @app.on_event("startup")
    def _startup() -> None:
        _registry.load(**_startup_args)

    @app.get("/health")
    def health():
        return {"status": "ready" if _registry.loaded else "loading", "weights_in_memory": _registry.loaded}

    class GenerateRequest(BaseModel):
        prompt: str
        duration_sec: float = 10.0
        seed: int = 0
        out: str = "service_out.wav"
        infer_steps: Optional[int] = None
        shift: float = 1.0
        guidance_scale: Optional[float] = None
        use_adg: Optional[bool] = None
        cfg_interval_start: float = 0.0
        cfg_interval_end: float = 1.0
        vae_chunk_latents: int = 32
        vae_overlap_latents: int = 4

    @app.post("/generate")
    def generate(req: GenerateRequest):
        if not _registry.loaded:
            from fastapi import HTTPException

            raise HTTPException(status_code=503, detail="Models still loading — retry shortly")

        sa = _startup_args  # server startup args (variant, lm_variant, etc.)
        variant = sa["variant"]
        lm_variant = sa["lm_variant"]

        infer_steps = req.infer_steps
        if infer_steps is None:
            infer_steps = 8 if "turbo" in variant.lower() else 50

        gs = req.guidance_scale
        if gs is None:
            gs = 1.0 if "turbo" in variant.lower() else 7.0
        gs = float(gs)

        use_adg = req.use_adg
        if use_adg is None:
            use_adg = "base" in variant.lower() and "turbo" not in variant.lower()

        t_schedule = _build_t_schedule(shift=req.shift, infer_steps=infer_steps, variant=variant)
        out_path = Path(req.out)

        t_wall = time.perf_counter()
        try:
            result = _registry.generate(
                prompt=req.prompt,
                duration_sec=req.duration_sec,
                seed=req.seed,
                variant=variant,
                lm_variant=lm_variant,
                t_schedule=t_schedule,
                guidance_scale=gs,
                use_adg=use_adg,
                cfg_interval_start=req.cfg_interval_start,
                cfg_interval_end=req.cfg_interval_end,
                use_trace=sa["use_trace"],
                use_torch_vae=sa["use_torch_vae"],
                vae_chunk_latents=req.vae_chunk_latents,
                vae_overlap_latents=req.vae_overlap_latents,
                out_path=out_path,
                exp_ttnn_lm=sa["experimental_5hz_ttnn_lm"],
                ckpt_dir=sa["ckpt_dir"],
            )
        except Exception as exc:
            log.exception("generate failed")
            raise HTTPException(status_code=500, detail=str(exc))

        elapsed = time.perf_counter() - t_wall
        return {
            "status": "ok",
            "out": str(result),
            "elapsed_s": round(elapsed, 2),
            "weights_reused": True,
        }

    return app


# ─────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="ACE-Step v1.5 persistent inference service (weights load once).",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--ckpt_dir", type=str, default=str(_DEFAULT_CKPT_DIR))
    ap.add_argument(
        "--variant",
        type=str,
        default="acestep-v15-base",
        choices=["acestep-v15-base", "acestep-v15-sft", "acestep-v15-turbo"],
    )
    ap.add_argument(
        "--lm_variant",
        type=str,
        default="acestep-5Hz-lm-1.7B",
        choices=["acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-4B"],
    )
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument(
        "--infer_steps",
        type=int,
        default=None,
        help="Steps for startup timestep schedule (default: 8 turbo / 50 base).",
    )
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--no-ttnn-strict", action="store_true")
    ap.add_argument(
        "--experimental-5hz-ttnn-causal-lm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Experimental TTNN 5 Hz LM (tt_transformers). Default: off (PyTorch HF).",
    )
    ap.add_argument("--torch-vae", action="store_true", help="Use PyTorch VAE instead of TTNN VAE.")
    ap.add_argument(
        "--use-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TTNN trace + 2CQ for DiT denoising.",
    )
    ap.add_argument("--ace-step-repo-root", type=str, default=None)
    args = ap.parse_args()

    configure_acestep_logging(level=os.environ.get("ACE_STEP_LOG_LEVEL", "INFO"))

    ckpt_dir = Path(args.ckpt_dir)
    variant = args.variant
    lm_variant = args.lm_variant
    infer_steps = args.infer_steps or (8 if "turbo" in variant.lower() else 50)

    # Ensure checkpoints exist
    for name in (variant, "vae", "Qwen3-Embedding-0.6B", lm_variant):
        _ensure_variant(name, ckpt_dir)

    model_dir = ckpt_dir / variant
    safetensors_path = model_dir / "model.safetensors"
    if not safetensors_path.is_file():
        shards = sorted(model_dir.glob("model-*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No weights at {safetensors_path}")
        safetensors_path = shards[0]

    vae_dir = ckpt_dir / "vae"
    text_model_dir = ckpt_dir / "Qwen3-Embedding-0.6B"

    ref_root = _resolve_ace_step_repo_root(ckpt_dir=str(ckpt_dir), ace_step_repo_root=args.ace_step_repo_root)
    if ref_root is None:
        raise RuntimeError(
            "Cannot find ACE-Step 'acestep' package.  Pass --ace-step-repo-root " "or set ACE_STEP_REPO_ROOT."
        )

    from models.demos.ace_step_v1_5.demo.ref_decoder_compare import ensure_acestep_repo_on_path

    ensure_acestep_repo_on_path(ref_root)

    t_schedule = _build_t_schedule(shift=args.shift, infer_steps=infer_steps, variant=variant)

    # Build the args dict that the FastAPI lifespan will pass to registry.load()
    global _startup_args
    _startup_args = dict(
        ckpt_dir=ckpt_dir,
        variant=variant,
        lm_variant=lm_variant,
        device_id=int(args.device_id),
        no_ttnn_strict=bool(args.no_ttnn_strict),
        use_trace=bool(args.use_trace),
        t_schedule=t_schedule,
        safetensors_path=safetensors_path,
        vae_dir=vae_dir,
        text_model_dir=text_model_dir,
        ref_root=ref_root,
        experimental_5hz_ttnn_lm=bool(getattr(args, "experimental_5hz_ttnn_causal_lm", False)),
        use_torch_vae=bool(args.torch_vae),
    )

    log.info("Starting ACE-Step v1.5 persistent service on %s:%d", args.host, args.port)
    log.info("Variant=%s  LM=%s  Steps=%d  Shift=%.1f", variant, lm_variant, infer_steps, args.shift)
    log.info("Weights will be loaded ONCE at startup — all /generate calls reuse them.")

    try:
        import uvicorn
    except ModuleNotFoundError:
        raise RuntimeError("uvicorn required: pip install 'uvicorn[standard]'")

    app = _make_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
