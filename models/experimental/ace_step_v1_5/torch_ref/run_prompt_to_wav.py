from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger

from models.experimental.ace_step_v1_5.torch_ref.e2e_model import decode_with_vae, run_torch_denoise_loop
from models.experimental.ace_step_v1_5.torch_ref.full_pipeline import AceStepV15TorchPipeline
from models.experimental.ace_step_v1_5.torch_ref.hf_generate import (
    build_t_schedule,
    default_guidance_scale,
    ensure_hf_modeling_ready,
    load_hf_ace_model,
    prepare_silence_and_masks,
    resolve_ace_step_repo_root,
    run_hf_generate_audio,
)

_build_t_schedule = build_t_schedule
_resolve_ace_step_repo_root = resolve_ace_step_repo_root


def _log_pipeline_banner(mode: str, *, details: str = "") -> None:
    msg = f"[ace_step_v1_5.torch_ref] pipeline={mode}"
    if details:
        msg = f"{msg} — {details}"
    logger.info(msg)


def _save_wav_fallback(wav: torch.Tensor, out_path: Path, sample_rate: int = 48000) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wav = wav.detach().float().cpu()
    if wav.ndim == 1:
        audio = wav.numpy()
    elif wav.ndim == 2:
        if wav.shape[0] in (1, 2):
            audio = wav.transpose(0, 1).contiguous().numpy()
        else:
            audio = wav.numpy()
    else:
        raise ValueError(f"Expected decoded wav rank 1 or 2, got shape {tuple(wav.shape)}")

    try:
        import soundfile as sf  # type: ignore

        sf.write(str(out_path), audio, samplerate=sample_rate)
        return
    except ModuleNotFoundError:
        pass

    from scipy.io import wavfile  # type: ignore

    audio_i16 = np.clip(audio, -1.0, 1.0)
    audio_i16 = (audio_i16 * 32767.0).astype(np.int16)
    wavfile.write(str(out_path), sample_rate, audio_i16)


def run_prompt_to_wav(
    *,
    prompt: str,
    ckpt_dir: str | None = None,
    variant: str = "acestep-v15-base",
    assets_repo_id: str = "ACE-Step/Ace-Step1.5",
    duration_sec: float = 10.0,
    shift: float = 1.0,
    infer_steps: int | None = None,
    timesteps: str | None = None,
    seed: int = 0,
    guidance_scale: Optional[float] = None,
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    guidance_interval_decay: float = 0.0,
    min_guidance_scale: float = 1.0,
    out_path: str | Path = "torch_out.wav",
    hf_prepare_condition: bool = True,
    use_dit_ref_sampler: bool = False,
    ace_step_repo_root: str | None = None,
    torch_device: str | torch.device | None = None,
    decoder_dtype: torch.dtype | None = None,
) -> Path:
    """
    End-to-end prompt → waveform using **PyTorch** (Qwen text encoder, ACE-Step DiT, Oobleck VAE).

    By default runs the full Hugging Face ``AutoModel.generate_audio()`` path (CFG, optional ADG on base,
    DCW, full ``decoder``). Set ``use_dit_ref_sampler=True`` for :class:`AceStepV15TorchPipeline` (tiny DiT
    core ref + Euler), e.g. TTNN module PCC parity.
    """

    def resolve_paths():
        if ckpt_dir:
            ckpt_dir_p = Path(ckpt_dir)
            model_dir = ckpt_dir_p / variant
            safetensors_path = model_dir / "model.safetensors"
            silence_latent_path = model_dir / "silence_latent.pt"
            vae_dir = ckpt_dir_p / "vae"
            text_model_dir = ckpt_dir_p / "Qwen3-Embedding-0.6B"
            return model_dir, safetensors_path, silence_latent_path, vae_dir, text_model_dir

        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("huggingface_hub is required when ckpt_dir is not provided") from e

        assets_snap = Path(snapshot_download(assets_repo_id, local_files_only=bool(os.environ.get("HF_HUB_OFFLINE"))))
        vae_dir = assets_snap / "vae"
        text_model_dir = assets_snap / "Qwen3-Embedding-0.6B"

        if variant == "acestep-v15-turbo":
            model_dir = assets_snap / "acestep-v15-turbo"
        else:
            model_dir = Path(
                snapshot_download(f"ACE-Step/{variant}", local_files_only=bool(os.environ.get("HF_HUB_OFFLINE")))
            )

        safetensors_path = model_dir / "model.safetensors"
        silence_latent_path = model_dir / "silence_latent.pt"
        return model_dir, safetensors_path, silence_latent_path, vae_dir, text_model_dir

    model_dir, safetensors_path, silence_latent_path, vae_dir, text_model_dir = resolve_paths()

    if not safetensors_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {safetensors_path}")
    if not silence_latent_path.is_file():
        raise FileNotFoundError(f"Missing silence_latent: {silence_latent_path}")
    if not vae_dir.is_dir():
        raise FileNotFoundError(f"Missing VAE directory: {vae_dir}")
    if not text_model_dir.is_dir():
        raise FileNotFoundError(f"Missing text encoder directory: {text_model_dir}")

    if torch_device is None:
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(torch_device) if isinstance(torch_device, str) else torch_device

    if decoder_dtype is None:
        decoder_dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    from safetensors.torch import load_file as torch_load_safetensors
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(text_model_dir))
    txt_model = AutoModel.from_pretrained(str(text_model_dir)).eval().to(dev)

    dit_instruction = "Fill the audio semantic mask based on the given conditions:"
    metas = {"caption": prompt, "duration": float(duration_sec), "language": "en"}
    text_prompt = f"""# Instruction
{dit_instruction}

# Caption
{prompt}

# Metas
{metas}<|endoftext|>
"""
    tokens = tok(text_prompt, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    input_ids = tokens["input_ids"].to(dev)
    attn_mask = tokens["attention_mask"].to(dev).to(torch.bool)

    with torch.inference_mode():
        out = txt_model(input_ids=input_ids, attention_mask=attn_mask)
        text_hidden_states = out.last_hidden_state

    sd_torch = torch_load_safetensors(str(safetensors_path), device="cpu")

    frames = int(round(float(duration_sec) * 25.0))
    if frames <= 0:
        raise ValueError("duration_sec must be > 0")

    silence, src_latents, chunk_masks = prepare_silence_and_masks(silence_latent_path, frames=frames)
    context_latents = torch.cat([src_latents, chunk_masks], dim=-1)

    if infer_steps is None:
        infer_steps = 8 if "turbo" in str(variant).lower() else 50

    gs = default_guidance_scale(variant=str(variant), guidance_scale=guidance_scale)

    pred_latents: torch.Tensor | None = None
    use_hf_generate = bool(hf_prepare_condition and not use_dit_ref_sampler)

    if use_hf_generate:
        _log_pipeline_banner(
            "caption-only",
            details="Qwen caption embed + HF generate_audio(); no 5Hz LM (use default official path for HF parity)",
        )
        try:
            ref_root = ensure_hf_modeling_ready(ckpt_dir=ckpt_dir, ace_step_repo_root=ace_step_repo_root)
            if ref_root is None:
                logger.warning(
                    "Set --ace-step-repo-root or ACE_STEP_REPO_ROOT to the ACE-Step-1.5 repo "
                    "(folder containing `acestep/`) so HF generate_audio() can import acestep."
                )

            logger.info("Loading HF AutoModel from {}", model_dir)
            ace = load_hf_ace_model(model_dir, device=dev, dtype=decoder_dtype)
            logger.info(
                "Running HF generate_audio (steps={}, guidance_scale={}, shift={})",
                infer_steps,
                gs,
                shift,
            )
            pred_latents = run_hf_generate_audio(
                ace,
                text_hidden_states=text_hidden_states,
                text_attention_mask=attn_mask,
                src_latents=src_latents,
                silence_latent=silence,
                chunk_masks=chunk_masks,
                device=dev,
                seed=int(seed),
                infer_steps=int(infer_steps),
                guidance_scale=gs,
                shift=float(shift),
                variant=str(variant),
                timesteps=timesteps,
                cfg_interval_start=float(cfg_interval_start),
                cfg_interval_end=float(cfg_interval_end),
            )
        except Exception as e:
            allow_fallback = os.environ.get("ACE_STEP_ALLOW_DIT_REF_FALLBACK", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            if not allow_fallback:
                raise RuntimeError(
                    f"HF generate_audio() failed ({type(e).__name__}: {e}). "
                    "Run without --caption-only for the full official pipeline, or set "
                    "ACE_STEP_ALLOW_DIT_REF_FALLBACK=1 to fall back to the DiT ref sampler."
                ) from e
            logger.warning(
                "HF generate_audio() unavailable ({}: {}). Falling back to DiT ref sampler.",
                type(e).__name__,
                e,
            )
            pred_latents = None

    encoder_hidden_states: torch.Tensor
    encoder_attention_mask: torch.Tensor
    context_latents_final: torch.Tensor

    if pred_latents is None:
        _log_pipeline_banner(
            "dit-ref-sampler",
            details="AceStepV15TorchPipeline Euler loop (TTNN PCC only; not HF parity)",
        )
        context_latents_final = context_latents

        if hf_prepare_condition:
            try:
                ref_root = ensure_hf_modeling_ready(ckpt_dir=ckpt_dir, ace_step_repo_root=ace_step_repo_root)
                if ref_root is None:
                    print(
                        "[ace_step_v1_5.torch_ref] Hint: set --ace-step-repo-root or ACE_STEP_REPO_ROOT to the "
                        "ACE-Step-1.5 repo (folder containing `acestep/`) so prepare_condition() can import acestep.",
                        flush=True,
                    )

                ace = load_hf_ace_model(model_dir, device=dev, dtype=decoder_dtype)
                hidden_size = int(getattr(getattr(ace, "config", None), "hidden_size", 0) or 0)
                if hidden_size <= 0:
                    hidden_size = int(sd_torch["decoder.condition_embedder.weight"].shape[0])

                B = int(src_latents.shape[0])
                lyric_dim = int(text_hidden_states.shape[-1])
                lyric_hidden_states = torch.zeros((B, 1, lyric_dim), dtype=torch.float32, device=dev)
                lyric_attention_mask = torch.ones((B, 1), dtype=torch.bool, device=dev)
                refer_audio_acoustic_hidden_states_packed = torch.zeros((B, 1, 64), dtype=torch.float32, device=dev)
                refer_audio_order_mask = torch.zeros((B,), dtype=torch.long, device=dev)
                latent_attention_mask = torch.ones((B, int(src_latents.shape[1])), dtype=torch.float32, device=dev)
                precomputed_lm_hints_25hz = src_latents.to(device=dev, dtype=torch.float32)

                with torch.inference_mode():
                    enc_hs, enc_mask, ctx = ace.prepare_condition(
                        text_hidden_states=text_hidden_states.to(device=dev, dtype=torch.float32),
                        text_attention_mask=attn_mask,
                        lyric_hidden_states=lyric_hidden_states,
                        lyric_attention_mask=lyric_attention_mask,
                        refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
                        refer_audio_order_mask=refer_audio_order_mask,
                        hidden_states=src_latents.to(device=dev, dtype=torch.float32),
                        attention_mask=latent_attention_mask,
                        silence_latent=silence.to(device=dev, dtype=torch.float32),
                        src_latents=src_latents.to(device=dev, dtype=torch.float32),
                        chunk_masks=chunk_masks.to(device=dev, dtype=torch.float32),
                        is_covers=torch.zeros((B,), dtype=torch.bool, device=dev),
                        precomputed_lm_hints_25Hz=precomputed_lm_hints_25hz,
                    )

                encoder_hidden_states = enc_hs.to(device="cpu", dtype=torch.float32)
                encoder_attention_mask = enc_mask.to(device="cpu", dtype=torch.float32)
                context_latents_final = ctx.to(device="cpu", dtype=torch.float32)
            except Exception as e:
                print(
                    f"[ace_step_v1_5.torch_ref] WARNING: HF prepare_condition() unavailable ({type(e).__name__}: {e}). "
                    "Falling back to lightweight conditioning.",
                    flush=True,
                )
                encoder_hidden_states = text_hidden_states.to(dtype=torch.float32).cpu()
                encoder_attention_mask = torch.ones(
                    (int(encoder_hidden_states.shape[0]), int(encoder_hidden_states.shape[1])), dtype=torch.float32
                )
                text_proj_key = "encoder.text_projector.weight"
                if text_proj_key in sd_torch:
                    W = sd_torch[text_proj_key].to(dtype=torch.float32)
                    encoder_hidden_states = torch.matmul(encoder_hidden_states, W.t())
        else:
            encoder_hidden_states = text_hidden_states.to(dtype=torch.float32).cpu()
            encoder_attention_mask = torch.ones(
                (int(encoder_hidden_states.shape[0]), int(encoder_hidden_states.shape[1])), dtype=torch.float32
            )
            cond_in = int(sd_torch["decoder.condition_embedder.weight"].shape[1])
            if int(encoder_hidden_states.shape[-1]) != cond_in:
                text_proj_key = "encoder.text_projector.weight"
                if text_proj_key not in sd_torch:
                    raise RuntimeError(
                        f"Text hidden dim {encoder_hidden_states.shape[-1]} != condition_embedder in {cond_in} "
                        f"and {text_proj_key!r} missing."
                    )
                W = sd_torch[text_proj_key].to(dtype=torch.float32)
                encoder_hidden_states = torch.matmul(encoder_hidden_states, W.t())

        t_schedule = _build_t_schedule(
            shift=float(shift),
            infer_steps=int(infer_steps),
            timesteps=timesteps,
            variant=str(variant),
        )
        timesteps_host = np.asarray(t_schedule + [0.0], dtype=np.float32)

        pipe = AceStepV15TorchPipeline(
            checkpoint_safetensors_path=str(safetensors_path),
            timesteps_host=timesteps_host,
            device=dev,
            dtype=decoder_dtype,
        )
        pipe.eval()

        ctx_t = context_latents_final.to(device=dev, dtype=decoder_dtype)
        enc_t = encoder_hidden_states.to(device=dev, dtype=decoder_dtype)

        do_cfg = float(gs) not in (0.0, 1.0)
        num_steps = len(t_schedule)
        cfg_start = max(0, min(int(num_steps * float(cfg_interval_start)), num_steps))
        cfg_end = max(cfg_start, min(int(num_steps * float(cfg_interval_end)), num_steps))

        def _current_guidance_scale(step_idx: int) -> float:
            base = float(gs)
            if not (cfg_start <= step_idx < cfg_end):
                return 1.0
            if float(guidance_interval_decay) > 0 and cfg_end - cfg_start > 1:
                progress = (step_idx - cfg_start) / float((cfg_end - cfg_start) - 1)
                min_s = float(min_guidance_scale)
                return base - (base - min_s) * progress * float(guidance_interval_decay)
            return base

        def _cfg_v(v_cond: torch.Tensor, v_uncond: torch.Tensor, scale: float) -> torch.Tensor:
            if float(scale) == 1.0:
                return v_cond
            return v_uncond + float(scale) * (v_cond - v_uncond)

        enc_uncond = torch.zeros_like(enc_t) if do_cfg else None

        def _apply_cfg(
            step_idx: int, t_curr: float, xt_h: torch.Tensor, vt_cond: torch.Tensor, vt_uncond: torch.Tensor
        ) -> torch.Tensor:
            return _cfg_v(vt_cond, vt_uncond, _current_guidance_scale(step_idx))

        def _progress(step_idx: int, total_steps: int, t_curr: float, dt: float) -> None:
            if step_idx == total_steps - 1:
                print(f"[torch_ref] final step {total_steps}/{total_steps} t={t_curr:.6g}", flush=True)
            else:
                print(f"[torch_ref] step {step_idx+1}/{total_steps} t={t_curr:.6g} dt={dt:.6g}", flush=True)

        pred_latents = run_torch_denoise_loop(
            pipe=pipe,
            t_schedule=t_schedule,
            frames=frames,
            enc_hs=enc_t,
            ctx_lat=ctx_t,
            null_emb=enc_uncond,
            do_cfg=do_cfg,
            seed=int(seed),
            cfg_fn=_apply_cfg if do_cfg else None,
            progress_fn=_progress,
        )

    assert pred_latents is not None

    from diffusers.models import AutoencoderOobleck

    vae = AutoencoderOobleck.from_pretrained(str(vae_dir)).eval().to(dev)
    wav = decode_with_vae(vae, pred_latents, dev)

    out_p = Path(out_path)
    _save_wav_fallback(wav[0], out_p, sample_rate=48000)
    print(f"Wrote: {out_p}")
    return out_p


def main() -> None:
    ap = argparse.ArgumentParser(description="ACE-Step v1.5: prompt → wav (PyTorch reference pipeline).")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Checkpoint root (default: ~/.cache/huggingface/hub/ACE-Step-1.5-checkpoints).",
    )
    ap.add_argument("--variant", type=str, default="acestep-v15-base")
    ap.add_argument(
        "--lm_variant",
        type=str,
        default="acestep-5Hz-lm-1.7B",
        choices=["acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-4B"],
        help="5 Hz LM variant for --use-official-acestep (default: acestep-5Hz-lm-1.7B).",
    )
    ap.add_argument(
        "--caption-only",
        action="store_true",
        help=(
            "Caption-only shortcut: Qwen text encoder + HF generate_audio() without 5Hz LM or handler "
            "preprocess. Lower quality than the default official pipeline; not HF profile parity."
        ),
    )
    ap.add_argument(
        "--use-official-acestep",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--assets-repo-id", type=str, default="ACE-Step/Ace-Step1.5")
    ap.add_argument("--duration_sec", type=float, default=10.0)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument(
        "--infer_steps",
        type=int,
        default=None,
        help="Diffusion steps (default: 8 for turbo, 50 for base/sft).",
    )
    ap.add_argument("--timesteps", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="CFG / diffusion_guidance_scale (default: 7.0 for base, 1.0 for turbo).",
    )
    ap.add_argument("--cfg_interval_start", type=float, default=0.0)
    ap.add_argument("--cfg_interval_end", type=float, default=1.0)
    ap.add_argument("--guidance_interval_decay", type=float, default=0.0)
    ap.add_argument("--min_guidance_scale", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="torch_out.wav")
    ap.add_argument(
        "--no_hf_prepare_condition",
        action="store_true",
        help="Skip ACE-Step HF prepare_condition(); use lightweight text conditioning only.",
    )
    ap.add_argument(
        "--use-dit-ref-sampler",
        action="store_true",
        help=(
            "Use the lightweight TorchAceStepDiTCoreRef Euler loop instead of HF generate_audio(). "
            "Useful for TTNN PCC parity; default path uses full PyTorch DiT (CFG, ADG on base, DCW)."
        ),
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for patch embed / output head / text encoder / VAE (default: cuda if available else cpu).",
    )
    ap.add_argument(
        "--ace-step-repo-root",
        type=str,
        default=None,
        help=(
            "Path to ACE-Step-1.5 repository root (must contain `acestep/`). "
            "If omitted, uses ACE_STEP_REPO_ROOT or walks parents of --ckpt_dir."
        ),
    )
    ap.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable 5Hz LM Chain-of-Thought (default: on; matches profile_inference --thinking).",
    )
    ap.add_argument(
        "--use-cot-metas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let 5Hz LM infer BPM/key/etc. via CoT (default: on).",
    )
    ap.add_argument(
        "--use-cot-caption",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Let 5Hz LM rewrite the caption via CoT (default: off; keep user prompt).",
    )
    ap.add_argument(
        "--use-cot-language",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let 5Hz LM detect vocal language via CoT (default: on).",
    )
    ap.add_argument(
        "--lyrics",
        type=str,
        default="[Instrumental]",
        help='Lyrics text (default: "[Instrumental]").',
    )
    ap.add_argument(
        "--instrumental",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate instrumental audio (default: on).",
    )
    args = ap.parse_args()
    if args.infer_steps is None:
        args.infer_steps = 8 if "turbo" in str(args.variant).lower() else 50

    from models.experimental.ace_step_v1_5.utils.official_lm_preprocess import configure_acestep_logging

    configure_acestep_logging(level=os.environ.get("ACE_STEP_LOG_LEVEL", "INFO"))

    use_official = not args.caption_only and not args.use_dit_ref_sampler
    if args.use_official_acestep:
        use_official = True

    if use_official:
        from models.experimental.ace_step_v1_5.demo.run_prompt_to_wav import _DEFAULT_CKPT_DIR, _ensure_variant
        from models.experimental.ace_step_v1_5.torch_ref.transformers_cache_compat import (
            apply_transformers_cache_compat,
        )

        apply_transformers_cache_compat()

        ckpt_dir = Path(args.ckpt_dir).expanduser() if args.ckpt_dir else _DEFAULT_CKPT_DIR
        os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(ckpt_dir)

        ref_root = _resolve_ace_step_repo_root(ckpt_dir=str(ckpt_dir), ace_step_repo_root=args.ace_step_repo_root)
        if ref_root is None:
            raise RuntimeError(
                "Could not find ACE-Step-1.5 repo root (folder containing `acestep/`). "
                "Pass --ace-step-repo-root or set ACE_STEP_REPO_ROOT."
            )
        from models.experimental.ace_step_v1_5.demo.ref_decoder_compare import ensure_acestep_repo_on_path

        ensure_acestep_repo_on_path(ref_root)
        from acestep.handler import AceStepHandler
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        # PyTorch-only LM (no TTNN device). ttnn_impl.five_hz_lm defaults to use_ttnn_causal_lm=True.
        from models.experimental.ace_step_v1_5.torch_ref.five_hz_lm import LocalFiveHzLMHandler

        _log_pipeline_banner(
            "official",
            details="AceStepHandler + 5Hz LM + generate_music (HF profile parity)",
        )

        dit_handler = AceStepHandler()
        llm_handler = LocalFiveHzLMHandler()

        device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Torch device: {}", device)
        for name in (args.variant, "vae", "Qwen3-Embedding-0.6B", args.lm_variant):
            _ensure_variant(name, ckpt_dir)

        status, ok = dit_handler.initialize_service(
            project_root=str(ref_root),
            config_path=args.variant,
            device=device,
            use_flash_attention=False,
        )
        logger.info(status)
        if not ok:
            raise RuntimeError("AceStepHandler.initialize_service failed")

        status, ok = llm_handler.initialize(
            checkpoint_dir=str(ckpt_dir),
            lm_model_path=args.lm_variant,
            backend="pt",
            device=device,
        )
        logger.info(status)
        if not ok:
            raise RuntimeError("5 Hz LM (local HF) initialize failed")

        gs = args.guidance_scale
        if gs is None:
            gs = 1.0 if "turbo" in str(args.variant).lower() else 7.0

        params = GenerationParams(
            task_type="text2music",
            caption=args.prompt,
            lyrics=args.lyrics,
            instrumental=bool(args.instrumental),
            reference_audio=None,
            duration=float(args.duration_sec),
            inference_steps=int(args.infer_steps),
            guidance_scale=float(gs),
            shift=float(args.shift),
            seed=int(args.seed),
            thinking=bool(args.thinking),
            use_cot_metas=bool(args.use_cot_metas),
            use_cot_caption=bool(args.use_cot_caption),
            use_cot_language=bool(args.use_cot_language),
            use_constrained_decoding=True,
            use_adg=("base" in str(args.variant).lower() and "turbo" not in str(args.variant).lower()),
            cfg_interval_start=float(args.cfg_interval_start),
            cfg_interval_end=float(args.cfg_interval_end),
        )
        logger.info(
            "GenerationParams: variant={} duration={}s steps={} guidance_scale={} thinking={} "
            "cot_metas={} cot_caption={} cot_language={}",
            args.variant,
            args.duration_sec,
            args.infer_steps,
            gs,
            args.thinking,
            args.use_cot_metas,
            args.use_cot_caption,
            args.use_cot_language,
        )
        config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[int(args.seed)], audio_format="wav")
        out_dir = Path(args.out).resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        result = generate_music(dit_handler, llm_handler, params, config, save_dir=str(out_dir))
        if not result.success:
            raise RuntimeError(result.error or "generate_music failed")
        first = Path(result.audios[0]["path"]).resolve()
        dst = Path(args.out).resolve()
        if first != dst:
            dst.write_bytes(first.read_bytes())
        logger.info("Wrote: {}", dst)
        return

    run_prompt_to_wav(
        prompt=args.prompt,
        ckpt_dir=args.ckpt_dir,
        variant=args.variant,
        assets_repo_id=args.assets_repo_id,
        duration_sec=args.duration_sec,
        shift=args.shift,
        infer_steps=args.infer_steps,
        timesteps=args.timesteps,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        cfg_interval_start=args.cfg_interval_start,
        cfg_interval_end=args.cfg_interval_end,
        guidance_interval_decay=args.guidance_interval_decay,
        min_guidance_scale=args.min_guidance_scale,
        out_path=args.out,
        hf_prepare_condition=not args.no_hf_prepare_condition,
        use_dit_ref_sampler=bool(args.use_dit_ref_sampler),
        ace_step_repo_root=args.ace_step_repo_root,
        torch_device=args.device,
    )


if __name__ == "__main__":
    main()
