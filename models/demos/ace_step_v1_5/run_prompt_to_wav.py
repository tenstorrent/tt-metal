from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

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

_SHIFT_TIMESTEPS = {
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


def _build_t_schedule(shift: float = 1.0, infer_steps: int = 8, timesteps: str | None = None):
    """
    Build a timestep schedule similar to ACE-Step's turbo mapping.

    - If `timesteps` is provided (comma-separated floats), map each value to the nearest valid timestep.
    - Else if `infer_steps` == 8, use the known SHIFT schedule.
    - Else, build a descending linear schedule and map each entry to nearest valid timestep.
    """
    if timesteps:
        raw = [float(x.strip()) for x in timesteps.split(",") if x.strip()]
        # drop explicit trailing zeros
        while raw and raw[-1] == 0.0:
            raw.pop()
        if not raw:
            raise ValueError("--timesteps provided but empty after removing zeros")
        mapped = [min(_VALID_TIMESTEPS, key=lambda v, t=t: abs(v - t)) for t in raw]
        # de-dup while preserving order
        out = []
        for t in mapped:
            if not out or out[-1] != t:
                out.append(t)
        return out

    infer_steps = int(infer_steps)
    if infer_steps <= 1:
        raise ValueError("--infer_steps must be >= 2")

    s = min(_SHIFT_TIMESTEPS.keys(), key=lambda v: abs(v - float(shift)))
    if infer_steps == 8:
        return list(_SHIFT_TIMESTEPS[float(s)])

    lin = [1.0 - (i / float(infer_steps - 1)) for i in range(infer_steps)]
    mapped = [min(_VALID_TIMESTEPS, key=lambda v, t=t: abs(v - t)) for t in lin]
    out = []
    for t in mapped:
        if not out or out[-1] != t:
            out.append(t)
    # ensure descending
    out = sorted(out, reverse=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help=(
            "Optional local checkpoints root containing subfolders like `acestep-v15-base/`, `vae/`, "
            "and `Qwen3-Embedding-0.6B/`. If omitted, weights are downloaded from Hugging Face."
        ),
    )
    ap.add_argument(
        "--variant",
        type=str,
        default="acestep-v15-base",
        help=(
            "Decoder variant. For HF download: `acestep-v15-base`, `acestep-v15-sft` (standalone repos), "
            "or `acestep-v15-turbo` (subfolder inside `ACE-Step/Ace-Step1.5`)."
        ),
    )
    ap.add_argument(
        "--assets-repo-id",
        type=str,
        default="ACE-Step/Ace-Step1.5",
        help="HF repo that provides shared assets (VAE + text encoder).",
    )
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--duration_sec", type=float, default=10.0)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--infer_steps", type=int, default=8, help="Number of diffusion steps (8 is turbo-style).")
    ap.add_argument(
        "--timesteps",
        type=str,
        default=None,
        help="Optional custom timesteps as comma-separated floats (will be mapped to nearest valid ACE-Step values).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guidance_scale", type=float, default=1.0, help="Classifier-free guidance scale (1.0 disables).")
    ap.add_argument(
        "--cfg_interval_start",
        type=float,
        default=0.0,
        help="Start of CFG interval as fraction of steps [0,1].",
    )
    ap.add_argument(
        "--cfg_interval_end",
        type=float,
        default=1.0,
        help="End of CFG interval as fraction of steps [0,1].",
    )
    ap.add_argument(
        "--guidance_interval_decay",
        type=float,
        default=0.0,
        help="If >0, linearly decay guidance scale within CFG interval down to --min_guidance_scale.",
    )
    ap.add_argument("--min_guidance_scale", type=float, default=1.0, help="Minimum guidance scale when decay enabled.")
    ap.add_argument("--out", type=str, default="ttnn_out.wav")
    ap.add_argument(
        "--hf_prepare_condition",
        action="store_true",
        help=(
            "Use the official HF ACE-Step model `prepare_condition()` (trust_remote_code) "
            "to build encoder_hidden_states + context_latents for better quality."
        ),
    )
    args = ap.parse_args()

    def resolve_paths():
        # Prefer a user-provided local checkpoint layout.
        if args.ckpt_dir:
            ckpt_dir = Path(args.ckpt_dir)
            model_dir = ckpt_dir / args.variant
            safetensors_path = model_dir / "model.safetensors"
            silence_latent_path = model_dir / "silence_latent.pt"
            vae_dir = ckpt_dir / "vae"
            text_model_dir = ckpt_dir / "Qwen3-Embedding-0.6B"
            return model_dir, safetensors_path, silence_latent_path, vae_dir, text_model_dir

        # Otherwise, fetch from Hugging Face.
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("huggingface_hub is required when --ckpt_dir is not provided") from e

        # Shared assets repo: contains `vae/` and `Qwen3-Embedding-0.6B/` subfolders.
        assets_snap = Path(
            snapshot_download(args.assets_repo_id, local_files_only=bool(os.environ.get("HF_HUB_OFFLINE")))
        )
        vae_dir = assets_snap / "vae"
        text_model_dir = assets_snap / "Qwen3-Embedding-0.6B"

        # Decoder weights.
        if args.variant == "acestep-v15-turbo":
            # Turbo decoder is nested in the umbrella assets repo.
            model_dir = assets_snap / "acestep-v15-turbo"
        else:
            # Base/SFT are standalone repos: ACE-Step/<variant>
            model_dir = Path(
                snapshot_download(f"ACE-Step/{args.variant}", local_files_only=bool(os.environ.get("HF_HUB_OFFLINE")))
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

    # --------------------------
    # Host path: prompt -> encoder_hidden_states (Torch/Transformers)
    # --------------------------
    import torch
    from safetensors.torch import load_file as torch_load_safetensors
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    # Text encoder (host; allowed)
    tok = AutoTokenizer.from_pretrained(str(text_model_dir))
    txt_model = AutoModel.from_pretrained(str(text_model_dir)).eval()

    # Keep this on GPU if available; otherwise CPU.
    torch_dev = "cuda" if torch.cuda.is_available() else "cpu"
    txt_model = txt_model.to(device=torch_dev)

    dit_instruction = "Fill the audio semantic mask based on the given conditions:"
    metas = {"caption": args.prompt, "duration": float(args.duration_sec), "language": "en"}
    text_prompt = f"""# Instruction
{dit_instruction}

# Caption
{args.prompt}

# Metas
{metas}<|endoftext|>
"""
    tokens = tok(text_prompt, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device=torch_dev)
    attn_mask = tokens["attention_mask"].to(device=torch_dev).to(torch.bool)

    with torch.inference_mode():
        out = txt_model(input_ids=input_ids, attention_mask=attn_mask)
        # [B, S, text_hidden_dim]
        text_hidden_states = out.last_hidden_state

    sd_torch = torch_load_safetensors(str(safetensors_path), device="cpu")

    # --------------------------
    # Host path: init latents + context (silence + chunk masks)
    # --------------------------
    frames = int(round(float(args.duration_sec) * 25.0))
    if frames <= 0:
        raise ValueError("duration_sec must be > 0")

    silence = torch.load(str(silence_latent_path), map_location="cpu").to(torch.float32)
    # Accept either layout:
    # - [1, T_ref, 64] (time-major)
    # - [1, 64, T_ref] (channels-first)
    if silence.ndim != 3:
        raise RuntimeError(f"Unexpected silence_latent rank: {tuple(silence.shape)}")
    if int(silence.shape[-1]) == 64:
        silence = silence  # [1, T, 64]
    elif int(silence.shape[1]) == 64:
        silence = silence.transpose(1, 2).contiguous()  # [1, T, 64]
    else:
        raise RuntimeError(f"Unexpected silence_latent shape: {tuple(silence.shape)} (expected [1,T,64] or [1,64,T])")
    src_latents = silence[:, :frames, :].contiguous()  # [1, T, 64]
    if src_latents.shape[1] < frames:
        # tile if needed
        rep = (frames + src_latents.shape[1] - 1) // src_latents.shape[1]
        src_latents = src_latents.repeat(1, rep, 1)[:, :frames, :].contiguous()

    # Full text-to-music generation marks the entire latent span as generated.
    # HF `prepare_condition` concatenates this mask with `src_latents` as context.
    chunk_masks = torch.ones((1, frames, 64), dtype=torch.float32)
    context_latents = torch.cat([src_latents, chunk_masks], dim=-1)  # [1, T, 128]

    # Initial noise xt in latent space [1,T,64]
    noise = torch.randn((1, frames, 64), dtype=torch.float32)
    xt = noise

    # --------------------------
    # Optional: HF `prepare_condition()` for better parity/quality (host)
    # --------------------------
    if args.hf_prepare_condition:
        # Best-effort parity path: try to use official HF ACE-Step `prepare_condition()`.
        # Some checkpoints require extra python deps (e.g. `vector_quantize_pytorch`) that may
        # not be present in minimal bring-up envs; in that case, fall back to a lightweight
        # conditioning builder that matches the same tensor contracts used by the decoder.
        try:
            from transformers import AutoModel as _AutoModel

            ace = _AutoModel.from_pretrained(str(model_dir), trust_remote_code=True).eval()
            ace = ace.to(device=torch_dev)

            with torch.inference_mode():
                hidden_size = int(getattr(getattr(ace, "config", None), "hidden_size", 0) or 0)
                if hidden_size <= 0:
                    hidden_size = int(sd_torch["decoder.condition_embedder.weight"].shape[0])

                B = int(src_latents.shape[0])
                lyric_hidden_states = torch.zeros((B, 1, hidden_size), dtype=torch.float32, device=torch_dev)
                lyric_attention_mask = torch.ones((B, 1), dtype=torch.bool, device=torch_dev)

                refer_audio_acoustic_hidden_states_packed = torch.zeros(
                    (B, 1, 64), dtype=torch.float32, device=torch_dev
                )
                refer_audio_order_mask = torch.zeros((B,), dtype=torch.long, device=torch_dev)

                latent_attention_mask = torch.ones(
                    (B, int(src_latents.shape[1])), dtype=torch.float32, device=torch_dev
                )
                precomputed_lm_hints_25hz = src_latents.to(device=torch_dev, dtype=torch.float32)

                enc_hs, _enc_mask, ctx = ace.prepare_condition(
                    text_hidden_states=text_hidden_states.to(device=torch_dev, dtype=torch.float32),
                    text_attention_mask=attn_mask.to(device=torch_dev),
                    lyric_hidden_states=lyric_hidden_states,
                    lyric_attention_mask=lyric_attention_mask,
                    refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
                    refer_audio_order_mask=refer_audio_order_mask,
                    hidden_states=src_latents.to(device=torch_dev, dtype=torch.float32),
                    attention_mask=latent_attention_mask,
                    silence_latent=silence.to(device=torch_dev, dtype=torch.float32),
                    src_latents=src_latents.to(device=torch_dev, dtype=torch.float32),
                    chunk_masks=chunk_masks.to(device=torch_dev, dtype=torch.float32),
                    is_covers=torch.zeros((B,), dtype=torch.bool, device=torch_dev),
                    precomputed_lm_hints_25Hz=precomputed_lm_hints_25hz,
                )

                encoder_hidden_states = enc_hs.to(device="cpu", dtype=torch.float32)
                context_latents = ctx.to(device="cpu", dtype=torch.float32)
        except Exception as e:
            print(
                f"[ace_step_v1_5] WARNING: HF prepare_condition() unavailable ({type(e).__name__}: {e}). "
                "Falling back to lightweight conditioning (text_projector + context_latents).",
                flush=True,
            )

            # Lightweight fallback:
            # - Apply the condition encoder's `text_projector` if present.
            # - Do not attempt lyric/timbre packing or LM hints here.
            encoder_hidden_states = text_hidden_states.to(dtype=torch.float32).to(device="cpu")
            text_proj_key = "encoder.text_projector.weight"
            if text_proj_key in sd_torch:
                W = sd_torch[text_proj_key].to(dtype=torch.float32)  # [hidden_size, text_hidden_dim]
                encoder_hidden_states = torch.matmul(encoder_hidden_states, W.t())
    else:
        # Fallback: approximate conditioning using text encoder states.
        # Keep raw text encoder states here. The TTNN decoder core applies the checkpoint's
        # `condition_embedder` projection on-device, matching the decoder-side contract.
        encoder_hidden_states = text_hidden_states.to(dtype=torch.float32).to(device="cpu")
        cond_in = int(sd_torch["decoder.condition_embedder.weight"].shape[1])
        if int(encoder_hidden_states.shape[-1]) != cond_in:
            text_proj_key = "encoder.text_projector.weight"
            if text_proj_key not in sd_torch:
                raise RuntimeError(
                    f"Text encoder hidden dim {encoder_hidden_states.shape[-1]} does not match "
                    f"decoder.condition_embedder input dim {cond_in}, and {text_proj_key!r} is missing."
                )
            # Some checkpoint variants expect the text-projector output as the condition input.
            W = sd_torch[text_proj_key].to(dtype=torch.float32)
            encoder_hidden_states = torch.matmul(encoder_hidden_states, W.t())

    # --------------------------
    # TTNN path: diffusion sampler (device-pure)
    # --------------------------
    import sys

    # Ensure tt-metal + ttnn python package are importable before importing `ttnn`
    tt_metal_root = str(Path(__file__).resolve().parents[3])  # .../tt-metal
    ttnn_pkg_root = str(Path(tt_metal_root) / "ttnn")
    # NOTE: do NOT add `tt-metal/tools` here; it contains a `tracy` package that
    # depends on optional plotting deps (e.g. seaborn). We ship a local no-op
    # `tracy.py` stub in this demo folder to satisfy TTNN init.
    for p in (tt_metal_root, ttnn_pkg_root):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Force strict no-fallback mode *before* importing ttnn, because ttnn prints CONFIG at import-time.
    # This makes any implicit fallback (including device->host repacking) throw immediately.
    os.environ["TTNN_CONFIG_OVERRIDES"] = '{"throw_exception_on_fallback": true}'

    import ttnn
    from models.demos.ace_step_v1_5.full_pipeline import AceStepV15TTNNPipeline

    # Strict mode: any fallback should hard-fail (proves device-pure execution).
    if hasattr(ttnn, "CONFIG") and hasattr(ttnn.CONFIG, "throw_exception_on_fallback"):
        ttnn.CONFIG.throw_exception_on_fallback = True
        print(f"TTNN strict: throw_exception_on_fallback={ttnn.CONFIG.throw_exception_on_fallback}")

    dev = ttnn.open_device(device_id=int(args.device_id))
    if hasattr(dev, "enable_program_cache"):
        dev.enable_program_cache()

    try:
        t_schedule = _build_t_schedule(float(args.shift), infer_steps=int(args.infer_steps), timesteps=args.timesteps)
        timesteps_host = np.asarray(t_schedule + [0.0], dtype=np.float32)

        expected_input_length = int(context_latents.shape[1])
        pipe = AceStepV15TTNNPipeline(
            device=dev,
            checkpoint_safetensors_path=str(safetensors_path),
            timesteps_host=timesteps_host,
            expected_input_length=expected_input_length,
        )

        # One host->device staging phase (all inputs/weights)
        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        if mem is None:
            raise RuntimeError("TTNN build missing DRAM_MEMORY_CONFIG; cannot stage host tensors to device safely.")
        act_dtype = getattr(ttnn, "bfloat16", None)
        if act_dtype is None:
            raise RuntimeError("TTNN build missing bfloat16 dtype; SDPA requires BF16/BF8/BF4.")

        xt_tt = ttnn.as_tensor(xt.numpy(), device=dev, dtype=act_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem)
        ctx_tt = ttnn.as_tensor(
            context_latents.numpy(), device=dev, dtype=act_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem
        )
        enc_tt = ttnn.as_tensor(
            encoder_hidden_states.numpy(), device=dev, dtype=act_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem
        )

        def _cfg_v(v_cond, v_uncond, scale: float):
            if float(scale) == 1.0:
                return v_cond
            # v = v_uncond + s * (v_cond - v_uncond)
            delta = ttnn.subtract(v_cond, v_uncond)
            delta = ttnn.multiply(delta, float(scale))
            return ttnn.add(v_uncond, delta)

        do_cfg = float(args.guidance_scale) not in (0.0, 1.0)
        num_steps = len(t_schedule)
        cfg_start = int(num_steps * float(args.cfg_interval_start))
        cfg_end = int(num_steps * float(args.cfg_interval_end))
        cfg_start = max(0, min(cfg_start, num_steps))
        cfg_end = max(cfg_start, min(cfg_end, num_steps))

        def _current_guidance_scale(step_idx: int) -> float:
            base = float(args.guidance_scale)
            if not (cfg_start <= step_idx < cfg_end):
                return 1.0
            if float(args.guidance_interval_decay) > 0 and cfg_end - cfg_start > 1:
                progress = (step_idx - cfg_start) / float((cfg_end - cfg_start) - 1)
                min_s = float(args.min_guidance_scale)
                return base - (base - min_s) * progress * float(args.guidance_interval_decay)
            return base

        if do_cfg:
            # Build unconditional conditioning tensors: zeros encoder, same context.
            # This matches the common "uncond = 0" CFG pattern.
            import torch

            enc_uncond = torch.zeros_like(encoder_hidden_states)
            enc_uncond_tt = ttnn.as_tensor(
                enc_uncond.numpy(), device=dev, dtype=act_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem
            )

        for step_idx, t_curr in enumerate(t_schedule):
            t_curr_f = float(t_curr)
            if step_idx == len(t_schedule) - 1:
                break
            t_next_f = float(t_schedule[step_idx + 1])
            dt = t_curr_f - t_next_f

            # Model input: concat(context_latents, xt) -> [B,T,192]
            model_in = (
                ttnn.concat([ctx_tt, xt_tt], dim=-1)
                if hasattr(ttnn, "concat")
                else ttnn.concatenate([ctx_tt, xt_tt], dim=-1)
            )

            s = _current_guidance_scale(step_idx)
            if float(s) not in (0.0, 1.0):
                # Duplicate batch: [2,T,192] and run one TTNN forward.
                model_in_2 = (
                    ttnn.concat([model_in, model_in], dim=0)
                    if hasattr(ttnn, "concat")
                    else ttnn.concatenate([model_in, model_in], dim=0)
                )
                enc_2 = (
                    ttnn.concat([enc_tt, enc_uncond_tt], dim=0)
                    if hasattr(ttnn, "concat")
                    else ttnn.concatenate([enc_tt, enc_uncond_tt], dim=0)
                )
                vt2 = pipe.forward(
                    hidden_states_btC=model_in_2, timestep_index=step_idx, encoder_hidden_states_btd=enc_2
                )
                vt_cond = ttnn.slice(vt2, (0, 0, 0), (1, int(vt2.shape[1]), int(vt2.shape[2])))
                vt_uncond = ttnn.slice(vt2, (1, 0, 0), (2, int(vt2.shape[1]), int(vt2.shape[2])))
                vt = _cfg_v(vt_cond, vt_uncond, float(s))
            else:
                vt = pipe.forward(hidden_states_btC=model_in, timestep_index=step_idx, encoder_hidden_states_btd=enc_tt)

            vt_dt = ttnn.multiply(vt, float(dt))
            xt_tt = ttnn.subtract(xt_tt, vt_dt)

        # Final step: x0 = x - v * t
        model_in = (
            ttnn.concat([ctx_tt, xt_tt], dim=-1)
            if hasattr(ttnn, "concat")
            else ttnn.concatenate([ctx_tt, xt_tt], dim=-1)
        )
        s_last = _current_guidance_scale(len(t_schedule) - 1)
        if float(s_last) not in (0.0, 1.0):
            model_in_2 = (
                ttnn.concat([model_in, model_in], dim=0)
                if hasattr(ttnn, "concat")
                else ttnn.concatenate([model_in, model_in], dim=0)
            )
            enc_2 = (
                ttnn.concat([enc_tt, enc_uncond_tt], dim=0)
                if hasattr(ttnn, "concat")
                else ttnn.concatenate([enc_tt, enc_uncond_tt], dim=0)
            )
            vt2 = pipe.forward(
                hidden_states_btC=model_in_2, timestep_index=len(t_schedule) - 1, encoder_hidden_states_btd=enc_2
            )
            vt_cond = ttnn.slice(vt2, (0, 0, 0), (1, int(vt2.shape[1]), int(vt2.shape[2])))
            vt_uncond = ttnn.slice(vt2, (1, 0, 0), (2, int(vt2.shape[1]), int(vt2.shape[2])))
            vt = _cfg_v(vt_cond, vt_uncond, float(s_last))
        else:
            vt = pipe.forward(
                hidden_states_btC=model_in, timestep_index=len(t_schedule) - 1, encoder_hidden_states_btd=enc_tt
            )
        t_last = float(t_schedule[-1])
        xt_tt = ttnn.subtract(xt_tt, ttnn.multiply(vt, float(t_last)))

        # Single device->host at end
        pred_latents = ttnn.to_torch(xt_tt).to(torch.float32)  # [1,T,64]
    finally:
        ttnn.close_device(dev)

    # --------------------------
    # Host path: VAE decode + save wav (host; after TTNN completes)
    # --------------------------
    from diffusers.models import AutoencoderOobleck

    def _save_wav_fallback(wav: torch.Tensor, out_path: Path, sample_rate: int = 48000) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        wav = wav.detach().float().cpu()
        if wav.ndim == 1:
            audio = wav.numpy()
        elif wav.ndim == 2:
            # VAE returns [channels, samples]; soundfile/scipy expect [samples, channels].
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

        import numpy as np
        from scipy.io import wavfile  # type: ignore

        audio_i16 = np.clip(audio, -1.0, 1.0)
        audio_i16 = (audio_i16 * 32767.0).astype(np.int16)
        wavfile.write(str(out_path), sample_rate, audio_i16)

    vae = AutoencoderOobleck.from_pretrained(str(vae_dir)).eval()
    vae = vae.to(device=torch_dev)

    with torch.inference_mode():
        lat = pred_latents.transpose(1, 2).contiguous().to(device=torch_dev, dtype=next(vae.parameters()).dtype)
        wav = vae.decode(lat).sample  # [B, channels, samples]
        wav = wav.float().cpu()
        # Match ACE-Step's anti-clipping behavior: only scale down if the waveform clips.
        peak = wav.abs().amax(dim=[1, 2], keepdim=True)
        wav = torch.where(peak > 1.0, wav / peak.clamp(min=1.0), wav)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # ACE-Step's `save_audio` pulls in `torchaudio` at import-time, which may not
    # be installed in bring-up envs. Save wav directly instead.
    _save_wav_fallback(wav[0], out_path, sample_rate=48000)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
