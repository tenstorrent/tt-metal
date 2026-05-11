from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from models.demos.ace_step_v1_5.checkpoint_paths import (
    ACE_STEP_CHECKPOINT_DIR_ENV,
    resolve_ace_step_checkpoints_root,
    resolve_ace_step_source_root,
)


def _build_t_schedule(shift: float = 1.0):
    # Matches `SHIFT_TIMESTEPS` for fix_nfe=8 in ACE-Step turbo (excluding final 0).
    if float(shift) == 1.0:
        return [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
    if float(shift) == 2.0:
        return [
            1.0,
            0.9333333333333333,
            0.8571428571428571,
            0.7692307692307693,
            0.6666666666666666,
            0.5454545454545454,
            0.4,
            0.2222222222222222,
        ]
    if float(shift) == 3.0:
        return [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3]
    raise ValueError("shift must be 1.0, 2.0, or 3.0 for bring-up script")


def main():
    _default_ckpt = resolve_ace_step_checkpoints_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument(
        "--ckpt_dir",
        type=str,
        default=str(_default_ckpt) if _default_ckpt is not None else "",
        help=(
            "ACE-Step checkpoints root (contains Qwen3-Embedding-0.6B/, vae/, variant dirs). "
            f"Default: discover next to tt-metal or ${ACE_STEP_CHECKPOINT_DIR_ENV}."
        ),
    )
    ap.add_argument("--variant", type=str, default="acestep-v15-turbo")
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--duration_sec", type=float, default=10.0)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="ttnn_out.wav")
    ap.add_argument(
        "--cq_trace",
        action="store_true",
        help="Use TTNN CQ trace capture/replay per decoder timestep (needs trace_region_size).",
    )
    ap.add_argument(
        "--trace_region_size",
        type=int,
        default=256 << 20,
        help="Trace buffer bytes for open_device when --cq_trace (default 256 MiB).",
    )
    ap.add_argument(
        "--cq_trace_no_prep",
        action="store_true",
        help="Disable Tracer prep-run before capture (faster setup, higher first-capture risk).",
    )
    ap.add_argument(
        "--use_torch_text_encoder",
        action="store_true",
        help="Run Qwen3 with PyTorch+Transformers instead of TTNN (debug / fallback).",
    )
    ap.add_argument(
        "--torch_vae",
        action="store_true",
        help="Decode latents with PyTorch+Diffusers AutoencoderOobleck (default: TTNN decoder on device).",
    )
    args = ap.parse_args()

    import sys

    ace_step_root = resolve_ace_step_source_root()
    if ace_step_root is not None and str(ace_step_root) not in sys.path:
        sys.path.insert(0, str(ace_step_root))

    ckpt_dir = Path(args.ckpt_dir).expanduser() if str(args.ckpt_dir).strip() else None
    if ckpt_dir is None or not ckpt_dir.is_dir():
        ckpt_dir = resolve_ace_step_checkpoints_root()
    if ckpt_dir is None:
        raise FileNotFoundError(
            "Could not find ACE-Step checkpoints. Pass --ckpt_dir (directory containing "
            "Qwen3-Embedding-0.6B/) or set "
            f"{ACE_STEP_CHECKPOINT_DIR_ENV}, "
            "or place ACE-Step-1.5 next to tt-metal (…/ACE-Step-1.5/checkpoints/)."
        )
    ckpt_dir = ckpt_dir.resolve()
    model_dir = ckpt_dir / args.variant
    safetensors_path = model_dir / "model.safetensors"
    silence_latent_path = model_dir / "silence_latent.pt"
    vae_dir = ckpt_dir / "vae"
    text_model_dir = ckpt_dir / "Qwen3-Embedding-0.6B"

    if not safetensors_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {safetensors_path}")
    if not silence_latent_path.is_file():
        raise FileNotFoundError(f"Missing silence_latent: {silence_latent_path}")

    # --------------------------
    # Host path: tokenization (always). Qwen3 forward: TTNN (default) or PyTorch (--use_torch_text_encoder).
    # --------------------------
    import torch
    from transformers import AutoTokenizer

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    tok = AutoTokenizer.from_pretrained(str(text_model_dir))
    tokens = tok(args.prompt, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    input_ids_np = tokens["input_ids"].numpy().astype(np.uint32)
    attn_mask_np = tokens["attention_mask"].numpy().astype(np.float32)

    torch_dev = "cuda" if torch.cuda.is_available() else "cpu"
    text_hidden_f32_np = None
    if args.use_torch_text_encoder:
        from transformers import AutoModel

        txt_model = AutoModel.from_pretrained(str(text_model_dir)).eval().to(device=torch_dev)
        input_ids = tokens["input_ids"].to(device=torch_dev)
        attn_mask = tokens["attention_mask"].to(device=torch_dev).to(torch.bool)
        with torch.inference_mode():
            out = txt_model(input_ids=input_ids, attention_mask=attn_mask)
            text_hidden_f32_np = out.last_hidden_state.detach().float().cpu().numpy()

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

    chunk_masks = torch.zeros((1, frames, 64), dtype=torch.float32)
    context_latents = torch.cat([src_latents, chunk_masks], dim=-1)  # [1, T, 128]

    # Initial noise xt in latent space [1,T,64]
    noise = torch.randn((1, frames, 64), dtype=torch.float32)
    xt = noise

    # --------------------------
    # TTNN path: diffusion sampler (device-pure)
    # --------------------------
    import os
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
    from models.demos.ace_step_v1_5.cq_trace import AceStepCQTracers
    from models.demos.ace_step_v1_5.full_pipeline import AceStepV15TTNNPipeline
    from models.demos.ace_step_v1_5.ttnn_impl.oobleck_vae_decoder import TtOobleckVaeDecoder
    from models.demos.ace_step_v1_5.ttnn_impl.text_projector import (
        TtAceStepTextProjector,
        load_text_projector_weight_numpy,
    )

    proj_weight_np = load_text_projector_weight_numpy(str(safetensors_path))

    # Strict mode: any fallback should hard-fail (proves device-pure execution).
    if hasattr(ttnn, "CONFIG") and hasattr(ttnn.CONFIG, "throw_exception_on_fallback"):
        ttnn.CONFIG.throw_exception_on_fallback = True
        print(f"TTNN strict: throw_exception_on_fallback={ttnn.CONFIG.throw_exception_on_fallback}")

    trace_kw = {}
    if args.cq_trace:
        trace_kw["trace_region_size"] = int(args.trace_region_size)
        print(
            f"CQ trace enabled: trace_region_size={trace_kw['trace_region_size']} "
            f"prep_run={not args.cq_trace_no_prep}"
        )

    dev = ttnn.open_device(device_id=int(args.device_id), **trace_kw)
    if hasattr(dev, "enable_program_cache"):
        dev.enable_program_cache()

    cq_runner = None
    torch_latents_after_tt: torch.Tensor | None = None
    tt_audio_nlc_fp32_cpu: torch.Tensor | None = None
    try:
        t_schedule = _build_t_schedule(float(args.shift))
        timesteps_host = np.asarray(t_schedule + [0.0], dtype=np.float32)

        expected_input_length = int(context_latents.shape[1])
        pipe = AceStepV15TTNNPipeline(
            device=dev,
            checkpoint_safetensors_path=str(safetensors_path),
            timesteps_host=timesteps_host,
            expected_input_length=expected_input_length,
        )

        tt_vae = None
        if not args.torch_vae:
            if not Path(vae_dir, "config.json").is_file():
                raise FileNotFoundError(
                    f"TTNN VAE decode needs a Hugging Face-style VAE folder at {vae_dir} (config.json). "
                    "Use --torch_vae if your layout differs."
                )
            act_dtype_vae = getattr(ttnn, "bfloat16", None)
            tt_vae = TtOobleckVaeDecoder.from_hf_vae_dir(
                str(vae_dir),
                device=dev,
                latent_frames=int(frames),
                batch_size=1,
                activation_dtype=act_dtype_vae,
                weights_dtype=act_dtype_vae,
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

        text_proj = TtAceStepTextProjector(
            device=dev,
            weight_f32_numpy=proj_weight_np,
            weights_dtype=act_dtype,
            weight_memory_config=mem,
        )
        if args.use_torch_text_encoder:
            if text_hidden_f32_np is None:
                raise RuntimeError("Torch text path requires text_hidden_f32_np")
            enc_tt = text_proj.forward(text_hidden_f32_np, activation_dtype=act_dtype)
        else:
            from models.demos.ace_step_v1_5.ttnn_impl.qwen3_embedding_encoder import TtQwen3EmbeddingEncoder

            qwen_enc = TtQwen3EmbeddingEncoder(
                device=dev,
                hf_model_dir=str(text_model_dir),
                qwen_safetensors_path=str(text_model_dir / "model.safetensors"),
            )
            qh = qwen_enc.forward(input_ids_np, attn_mask_np)
            enc_tt = text_proj.forward_from_hidden(qh, activation_dtype=act_dtype)

        if args.cq_trace:
            cq_runner = AceStepCQTracers(
                dev,
                pipe,
                enc_tt,
                prep_run=not args.cq_trace_no_prep,
                clone_prep_inputs=True,
            )

        for step_idx, t_curr in enumerate(t_schedule):
            t_curr_f = float(t_curr)
            if step_idx == len(t_schedule) - 1:
                break
            t_next_f = float(t_schedule[step_idx + 1])
            dt = t_curr_f - t_next_f

            # Model input: concat(context_latents, xt) -> [1,T,192]
            if hasattr(ttnn, "concat"):
                model_in = ttnn.concat([ctx_tt, xt_tt], dim=-1)
            else:
                model_in = ttnn.concatenate([ctx_tt, xt_tt], dim=-1)

            if cq_runner is not None:
                vt = cq_runner.forward(model_in, step_idx)
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
        final_idx = len(t_schedule) - 1
        if cq_runner is not None:
            vt = cq_runner.forward(model_in, final_idx)
        else:
            vt = pipe.forward(hidden_states_btC=model_in, timestep_index=final_idx, encoder_hidden_states_btd=enc_tt)
        t_last = float(t_schedule[-1])
        xt_tt = ttnn.subtract(xt_tt, ttnn.multiply(vt, float(t_last)))

        if tt_vae is not None:
            tt_audio_nlc_fp32_cpu = ttnn.to_torch(tt_vae(xt_tt)).float()
        else:
            torch_latents_after_tt = ttnn.to_torch(xt_tt).to(torch.float32)  # [1,T,64]
    finally:
        if cq_runner is not None:
            cq_runner.release()
        ttnn.close_device(dev)

    # --------------------------
    # Host path: optional PyTorch VAE decode (if --torch_vae), then save wav (host)
    # --------------------------
    def _save_wav_fallback(wav_1d: torch.Tensor, out_path: Path, sample_rate: int = 44100) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        wav_1d = wav_1d.detach().float().cpu()
        if wav_1d.ndim != 1:
            wav_1d = wav_1d.flatten()
        audio = wav_1d.numpy()

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

    if tt_audio_nlc_fp32_cpu is not None:
        wav = tt_audio_nlc_fp32_cpu.permute(0, 2, 1)  # [B, audio_channels, samples]
    else:
        if torch_latents_after_tt is None:
            raise RuntimeError("Internal error: diffusion latents missing for PyTorch VAE decode.")
        from diffusers.models import AutoencoderOobleck

        vae = AutoencoderOobleck.from_pretrained(str(vae_dir)).eval()
        vae = vae.to(device=torch_dev)
        with torch.inference_mode():
            lat = (
                torch_latents_after_tt.transpose(1, 2)
                .contiguous()
                .to(device=torch_dev, dtype=next(vae.parameters()).dtype)
            )
            wav = vae.decode(lat).sample  # [B, channels, samples]
            wav = wav.float().cpu()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # ACE-Step's `save_audio` pulls in `torchaudio` at import-time, which may not
    # be installed in bring-up envs. Save wav directly instead.
    _save_wav_fallback(wav[0], out_path, sample_rate=44100)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
