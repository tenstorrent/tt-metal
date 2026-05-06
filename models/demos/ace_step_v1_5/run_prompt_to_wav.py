from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, default="/home/ubuntu/ign-sakthi/ACE-Step-1.5/checkpoints")
    ap.add_argument("--variant", type=str, default="acestep-v15-turbo")
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--duration_sec", type=float, default=10.0)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="ttnn_out.wav")
    args = ap.parse_args()

    # Make ACE-Step package importable (read-only reference; we do not modify it).
    import sys

    ace_step_root = Path("/home/ubuntu/ign-sakthi/ACE-Step-1.5")
    if str(ace_step_root) not in sys.path:
        sys.path.insert(0, str(ace_step_root))

    ckpt_dir = Path(args.ckpt_dir)
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

    tokens = tok(args.prompt, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device=torch_dev)
    attn_mask = tokens["attention_mask"].to(device=torch_dev).to(torch.bool)

    with torch.inference_mode():
        out = txt_model(input_ids=input_ids, attention_mask=attn_mask)
        # [B, S, text_hidden_dim]
        text_hidden_states = out.last_hidden_state

    # Project to decoder hidden size using ACE-Step condition encoder text_projector weights.
    # Weight key in the main checkpoint is `encoder.text_projector.weight` (no bias).
    sd_torch = torch_load_safetensors(str(safetensors_path), device="cpu")
    W = sd_torch["encoder.text_projector.weight"].to(dtype=torch.float32)  # [hidden, text_hidden_dim]
    text_h = text_hidden_states.to(dtype=torch.float32).to(device="cpu")  # host staging
    encoder_hidden_states = torch.matmul(text_h, W.t())  # [B, S, hidden]
    encoder_attn_mask = attn_mask.to(device="cpu")

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

    import ttnn
    from models.demos.ace_step_v1_5.full_pipeline import AceStepV15TTNNPipeline

    dev = ttnn.open_device(device_id=int(args.device_id))
    if hasattr(dev, "enable_program_cache"):
        dev.enable_program_cache()

    try:
        t_schedule = _build_t_schedule(float(args.shift))
        timesteps_host = np.asarray(t_schedule + [0.0], dtype=np.float32)

        pipe = AceStepV15TTNNPipeline(
            device=dev,
            checkpoint_safetensors_path=str(safetensors_path),
            timesteps_host=timesteps_host,
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

            vt = pipe.forward(hidden_states_btC=model_in, timestep_index=step_idx, encoder_hidden_states_btd=enc_tt)

            vt_dt = ttnn.multiply(vt, float(dt))
            xt_tt = ttnn.subtract(xt_tt, vt_dt)

        # Final step: x0 = x - v * t
        model_in = (
            ttnn.concat([ctx_tt, xt_tt], dim=-1)
            if hasattr(ttnn, "concat")
            else ttnn.concatenate([ctx_tt, xt_tt], dim=-1)
        )
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

    vae = AutoencoderOobleck.from_pretrained(str(vae_dir)).eval()
    vae = vae.to(device=torch_dev)

    with torch.inference_mode():
        lat = pred_latents.transpose(1, 2).contiguous().to(device=torch_dev, dtype=next(vae.parameters()).dtype)
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
