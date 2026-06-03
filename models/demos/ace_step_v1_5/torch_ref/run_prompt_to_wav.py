from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from models.demos.ace_step_v1_5.torch_ref.e2e_model import decode_with_vae, run_torch_denoise_loop
from models.demos.ace_step_v1_5.torch_ref.full_pipeline import AceStepV15TorchPipeline

# Turbo discrete timesteps (aligned with acestep turbo modeling).
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


def _build_t_schedule(*, shift: float, infer_steps: int, timesteps: str | None, variant: str) -> list[float]:
    """Inference timestep schedule (t_curr per step; terminal 0 is appended separately for the ref pipeline)."""
    variant_l = (variant or "").lower()
    is_turbo = "turbo" in variant_l

    if timesteps:
        raw = [float(x.strip()) for x in timesteps.split(",") if x.strip()]
        while raw and raw[-1] == 0.0:
            raw.pop()
        if not raw:
            raise ValueError("--timesteps provided but empty after removing zeros")
        if is_turbo:
            mapped = [min(_VALID_TIMESTEPS, key=lambda v, t=t: abs(v - t)) for t in raw]
            out: list[float] = []
            for t in mapped:
                if not out or out[-1] != t:
                    out.append(t)
            return out
        return raw

    infer_steps = int(infer_steps)
    if infer_steps <= 1:
        raise ValueError("--infer_steps must be >= 2")

    if is_turbo:
        s = min(_SHIFT_TIMESTEPS.keys(), key=lambda v: abs(v - float(shift)))
        if infer_steps == 8:
            return list(_SHIFT_TIMESTEPS[float(s)])
        lin = [1.0 - (i / float(infer_steps - 1)) for i in range(infer_steps)]
        mapped = [min(_VALID_TIMESTEPS, key=lambda v, t=t: abs(v - t)) for t in lin]
        out = []
        for t in mapped:
            if not out or out[-1] != t:
                out.append(t)
        return sorted(out, reverse=True)

    t = [1.0 - (i / float(infer_steps)) for i in range(infer_steps)]
    if float(shift) != 1.0:
        s = float(shift)
        t = [s * x / (1.0 + (s - 1.0) * x) for x in t]
    return t


_VENDORED_ACESTEP_ROOT = Path(__file__).resolve().parent / "_vendored_acestep"


def _resolve_ace_step_repo_root(*, ckpt_dir: str | None, ace_step_repo_root: str | None) -> Path | None:
    """
    Directory that contains the ``acestep/`` package (clone of ACE-Step-1.5), for ``trust_remote_code``.

    Order: explicit ``ace_step_repo_root`` → env ``ACE_STEP_REPO_ROOT`` → vendored copy under
    ``torch_ref/_vendored_acestep/`` → walk parents of ``ckpt_dir``.
    """

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

    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if (c / "acestep" / "__init__.py").is_file():
            return c
    return None


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

    silence = torch.load(str(silence_latent_path), map_location="cpu").to(torch.float32)
    if silence.ndim != 3:
        raise RuntimeError(f"Unexpected silence_latent rank: {tuple(silence.shape)}")
    if int(silence.shape[-1]) == 64:
        pass
    elif int(silence.shape[1]) == 64:
        silence = silence.transpose(1, 2).contiguous()
    else:
        raise RuntimeError(f"Unexpected silence_latent shape: {tuple(silence.shape)}")

    src_latents = silence[:, :frames, :].contiguous()
    if src_latents.shape[1] < frames:
        rep = (frames + src_latents.shape[1] - 1) // src_latents.shape[1]
        src_latents = src_latents.repeat(1, rep, 1)[:, :frames, :].contiguous()

    chunk_masks = torch.ones((1, frames, 64), dtype=torch.float32)
    context_latents = torch.cat([src_latents, chunk_masks], dim=-1)

    if infer_steps is None:
        infer_steps = 8 if "turbo" in str(variant).lower() else 50

    gs = float(guidance_scale) if guidance_scale is not None else (1.0 if "turbo" in str(variant).lower() else 7.0)

    pred_latents: torch.Tensor | None = None
    use_hf_generate = bool(hf_prepare_condition and not use_dit_ref_sampler)

    if use_hf_generate:
        try:
            ref_root = _resolve_ace_step_repo_root(ckpt_dir=ckpt_dir, ace_step_repo_root=ace_step_repo_root)
            if ref_root is not None:
                from models.demos.ace_step_v1_5.demo.ref_decoder_compare import ensure_acestep_repo_on_path

                ensure_acestep_repo_on_path(ref_root)
            else:
                print(
                    "[ace_step_v1_5.torch_ref] Hint: set --ace-step-repo-root or ACE_STEP_REPO_ROOT to the "
                    "ACE-Step-1.5 repo (folder containing `acestep/`) so HF generate_audio() can import acestep.",
                    flush=True,
                )

            from transformers import AutoModel as _AutoModel

            ace = _AutoModel.from_pretrained(str(model_dir), trust_remote_code=True).eval().to(dev)

            B = int(src_latents.shape[0])
            lyric_dim = int(text_hidden_states.shape[-1])
            lyric_hidden_states = torch.zeros((B, 1, lyric_dim), dtype=torch.float32, device=dev)
            lyric_attention_mask = torch.ones((B, 1), dtype=torch.bool, device=dev)
            refer_audio_acoustic_hidden_states_packed = torch.zeros((B, 1, 64), dtype=torch.float32, device=dev)
            refer_audio_order_mask = torch.zeros((B,), dtype=torch.long, device=dev)
            latent_attention_mask = torch.ones((B, int(src_latents.shape[1])), dtype=torch.float32, device=dev)

            timesteps_tensor = None
            if timesteps:
                t_sched = _build_t_schedule(
                    shift=float(shift),
                    infer_steps=int(infer_steps),
                    timesteps=timesteps,
                    variant=str(variant),
                )
                timesteps_tensor = torch.tensor(t_sched + [0.0], device=dev, dtype=torch.float32)

            use_adg = "base" in str(variant).lower() and "turbo" not in str(variant).lower()

            with torch.inference_mode():
                gen_out = ace.generate_audio(
                    text_hidden_states=text_hidden_states.to(device=dev, dtype=torch.float32),
                    text_attention_mask=attn_mask,
                    lyric_hidden_states=lyric_hidden_states,
                    lyric_attention_mask=lyric_attention_mask,
                    refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
                    refer_audio_order_mask=refer_audio_order_mask,
                    src_latents=src_latents.to(device=dev, dtype=torch.float32),
                    chunk_masks=chunk_masks.to(device=dev, dtype=torch.float32),
                    is_covers=torch.zeros((B,), dtype=torch.bool, device=dev),
                    silence_latent=silence.to(device=dev, dtype=torch.float32),
                    attention_mask=latent_attention_mask,
                    seed=int(seed),
                    infer_steps=int(infer_steps),
                    diffusion_guidance_scale=gs,
                    use_adg=use_adg,
                    shift=float(shift),
                    timesteps=timesteps_tensor,
                    use_progress_bar=False,
                    infer_method="ode",
                    sampler_mode="euler",
                    cfg_interval_start=float(cfg_interval_start),
                    cfg_interval_end=float(cfg_interval_end),
                    precomputed_lm_hints_25Hz=None,
                )
            pred_latents = gen_out["target_latents"].float().cpu()
        except Exception as e:
            print(
                f"[ace_step_v1_5.torch_ref] WARNING: HF generate_audio() unavailable ({type(e).__name__}: {e}). "
                "Falling back to DiT ref sampler.",
                flush=True,
            )
            pred_latents = None

    encoder_hidden_states: torch.Tensor
    encoder_attention_mask: torch.Tensor
    context_latents_final: torch.Tensor

    if pred_latents is None:
        context_latents_final = context_latents

        if hf_prepare_condition:
            try:
                ref_root = _resolve_ace_step_repo_root(ckpt_dir=ckpt_dir, ace_step_repo_root=ace_step_repo_root)
                if ref_root is not None:
                    from models.demos.ace_step_v1_5.demo.ref_decoder_compare import ensure_acestep_repo_on_path

                    ensure_acestep_repo_on_path(ref_root)
                else:
                    print(
                        "[ace_step_v1_5.torch_ref] Hint: set --ace-step-repo-root or ACE_STEP_REPO_ROOT to the "
                        "ACE-Step-1.5 repo (folder containing `acestep/`) so prepare_condition() can import acestep.",
                        flush=True,
                    )

                from transformers import AutoModel as _AutoModel

                ace = _AutoModel.from_pretrained(str(model_dir), trust_remote_code=True).eval().to(dev)
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
        "--use-official-acestep",
        action="store_true",
        help=(
            "Use ACE-Step's official Torch pipeline (LLMHandler + AceStepHandler + generate_music). "
            "Runs entirely on PyTorch (CPU/CUDA); does not open a TTNN device."
        ),
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
    args = ap.parse_args()
    if args.infer_steps is None:
        args.infer_steps = 8 if "turbo" in str(args.variant).lower() else 50

    if args.use_official_acestep:
        from models.demos.ace_step_v1_5.demo.run_prompt_to_wav import _DEFAULT_CKPT_DIR, _ensure_variant
        from models.demos.ace_step_v1_5.torch_ref.transformers_cache_compat import apply_transformers_cache_compat

        apply_transformers_cache_compat()

        ckpt_dir = Path(args.ckpt_dir).expanduser() if args.ckpt_dir else _DEFAULT_CKPT_DIR
        os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(ckpt_dir)

        ref_root = _resolve_ace_step_repo_root(ckpt_dir=str(ckpt_dir), ace_step_repo_root=args.ace_step_repo_root)
        if ref_root is None:
            raise RuntimeError(
                "Could not find ACE-Step-1.5 repo root (folder containing `acestep/`). "
                "Pass --ace-step-repo-root or set ACE_STEP_REPO_ROOT."
            )
        from models.demos.ace_step_v1_5.demo.ref_decoder_compare import ensure_acestep_repo_on_path

        ensure_acestep_repo_on_path(ref_root)
        from acestep.handler import AceStepHandler
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        # PyTorch-only LM (no TTNN device). ttnn_impl.five_hz_lm defaults to use_ttnn_causal_lm=True.
        from models.demos.ace_step_v1_5.torch_ref.five_hz_lm import LocalFiveHzLMHandler

        dit_handler = AceStepHandler()
        llm_handler = LocalFiveHzLMHandler()

        device = args.device if args.device else "cpu"
        for name in (args.variant, "vae", "Qwen3-Embedding-0.6B", args.lm_variant):
            _ensure_variant(name, ckpt_dir)

        status, ok = dit_handler.initialize_service(
            project_root=str(ref_root),
            config_path=args.variant,
            device=device,
            use_flash_attention=False,
        )
        print(status, flush=True)
        if not ok:
            raise RuntimeError("AceStepHandler.initialize_service failed")

        status, ok = llm_handler.initialize(
            checkpoint_dir=str(ckpt_dir),
            lm_model_path=args.lm_variant,
            backend="pt",
            device=device,
        )
        print(status, flush=True)
        if not ok:
            raise RuntimeError("5 Hz LM (local HF) initialize failed")

        params = GenerationParams(
            task_type="text2music",
            caption=args.prompt,
            lyrics="[Instrumental]",
            instrumental=True,
            reference_audio=None,
            duration=float(args.duration_sec),
            inference_steps=int(args.infer_steps),
            thinking=True,
            use_constrained_decoding=True,
            use_adg=("base" in str(args.variant).lower() and "turbo" not in str(args.variant).lower()),
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
        print(f"Wrote: {dst}", flush=True)
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
