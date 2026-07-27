# SPDX-License-Identifier: Apache-2.0
"""Phase A (coqui venv, CPU): conditioning + tokenize + generate codes, capture the GPT
inputs_embeds for the return_latent forward, and produce a full-CPU baseline wav."""
import argparse, os

os.environ["COQUI_TOS_AGREED"] = "1"

# Shim: coqui's tortoise code imports isin_mps_friendly which newer transformers dropped.
import torch
import transformers.pytorch_utils as _ptu

if not hasattr(_ptu, "isin_mps_friendly"):
    _ptu.isin_mps_friendly = lambda elements, test_elements: torch.isin(elements, test_elements)

import soundfile as sf
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

CKPT = os.environ.get("XTTS_CKPT_DIR", "/localdev/acicovic/xtts_ref")  # coqui checkpoint dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--ref", required=True, help="path to reference audio .pt (waveform)")
    ap.add_argument("--sr", type=int, default=22050, help="sample rate of the .pt waveform")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--work", required=True)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)

    print("[A] loading XTTS (CPU)...")
    config = XttsConfig()
    config.load_json(os.path.join(CKPT, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=CKPT, eval=True, use_deepspeed=False)
    model.eval()

    # reference .pt waveform -> tensor-level conditioning (bypasses torchcodec file IO).
    # Handles: raw tensor/ndarray, (tensor, sr) tuple, or a HuggingFace audio-sample dict
    # {"audio": {"array": ndarray, "sampling_rate": int}}.
    import numpy as np

    raw = torch.load(args.ref, map_location="cpu", weights_only=False)
    sr_eff = args.sr
    if isinstance(raw, dict) and isinstance(raw.get("audio"), dict) and "array" in raw["audio"]:
        sr_eff = int(raw["audio"].get("sampling_rate", args.sr))
        raw = raw["audio"]["array"]
    elif isinstance(raw, dict):
        for k in ("array", "audio", "waveform", "wav"):
            if k in raw:
                raw = raw[k]
                break
    elif isinstance(raw, (tuple, list)):
        raw = raw[0]
    wav = raw if torch.is_tensor(raw) else torch.as_tensor(np.asarray(raw))
    wav = wav.squeeze()
    if wav.dim() > 1:
        wav = wav[0]
    audio = wav.float().reshape(1, -1)  # [1, N]
    print(f"[A] ref waveform {tuple(audio.shape)} @ {sr_eff}Hz")

    # capture the conditioning-encoder input mel (Block 1 input) so TT can recompute
    # gpt_cond_latent on device in phase B.
    _mel = {}
    _h = model.gpt.conditioning_encoder.register_forward_hook(lambda m, a, o: _mel.update(x=a[0].detach().clone()))
    gpt_cond_latent = model.get_gpt_cond_latents(audio, sr_eff)
    _h.remove()
    # capture the speaker-encoder logmel (Block 2 input; the mel/STFT front-end stays CPU)
    _spk = {}
    _hs = model.hifigan_decoder.speaker_encoder.instancenorm.register_forward_pre_hook(
        lambda m, a: _spk.update(logmel=a[0].detach().clone())
    )
    speaker_embedding = model.get_speaker_embedding(audio, sr_eff)
    _hs.remove()
    torch.save(_mel["x"].float(), os.path.join(args.work, "cond_mel_in.pt"))  # [1,80,T]
    torch.save(_spk["logmel"].float(), os.path.join(args.work, "speaker_logmel.pt"))  # [1,64,T]
    print(
        f"[A] gpt_cond_latent {tuple(gpt_cond_latent.shape)}  speaker_embedding {tuple(speaker_embedding.shape)}  "
        f"cond_mel_in {tuple(_mel['x'].shape)}"
    )

    # Hooks: capture the (longest) inputs_embeds fed to the GPT2 transformer (= the
    # return_latent full-sequence forward), and the CPU mel-latent output for reference.
    # Bind all captures to the SAME return_latent forward (generate also drives the
    # transformer, so a max-seq heuristic grabs the wrong call).
    cap = {"emb": None, "mel_lat": None, "lh": None, "attn_mask": None, "want": False}

    def gpt2_pre(mod, a, kw):
        if cap["want"]:
            cap["emb"] = kw.get("inputs_embeds").detach().clone()
            am = kw.get("attention_mask")
            cap["attn_mask"] = None if am is None else am.detach().clone()

    def gpt2_out(mod, a, out):
        if cap["want"]:
            cap["lh"] = out.last_hidden_state.detach().clone()

    h1 = model.gpt.gpt.register_forward_pre_hook(gpt2_pre, with_kwargs=True)
    h1b = model.gpt.gpt.register_forward_hook(gpt2_out)

    _orig_gpt_forward = model.gpt.forward

    def _wrapped_gpt_forward(*a, **k):
        want = bool(k.get("return_latent"))
        cap["want"] = want
        r = _orig_gpt_forward(*a, **k)
        cap["want"] = False
        if want:
            if len(a) > 2 and hasattr(a[2], "shape"):
                cap["audio_codes"] = a[2].detach().clone()  # the generated gpt_codes
            if hasattr(r, "dim") and r.dim() == 3 and r.shape[-1] == 1024:
                cap["mel_lat"] = r.detach().clone()
        return r

    model.gpt.forward = _wrapped_gpt_forward

    # Capture the generation prefix embedding [cond, start_text, text, stop_text] that
    # coqui feeds gpt_inference (start_audio_token is appended after it during decode).
    _orig_store = model.gpt.gpt_inference.store_prefix_emb

    def _store(pemb):
        cap["prefix_emb"] = pemb.detach().clone()
        return _orig_store(pemb)

    model.gpt.gpt_inference.store_prefix_emb = _store

    print(f"[A] inference (sampled generate + return_latent + vocode) ...")
    # coqui default decode = stochastic sampling (do_sample=True) — the baseline now reflects
    # XTTS's real inference mode (temperature 0.75 / top_k 50 / top_p 0.85 / rep-penalty 10 defaults).
    out = model.inference(args.text, args.lang, gpt_cond_latent, speaker_embedding, do_sample=True)
    h1.remove()
    h1b.remove()
    model.gpt.forward = _orig_gpt_forward

    am = cap.get("attn_mask")
    print(f"[A] attn_mask: {None if am is None else (tuple(am.shape), 'all_ones=' + str(bool((am==1).all())))}")
    if cap.get("lh") is not None:
        torch.save(cap["lh"].float(), os.path.join(args.work, "last_hidden_cpu.pt"))
        print(f"[A] saved last_hidden_cpu {tuple(cap['lh'].shape)}")

    base_wav = out["wav"]
    if hasattr(base_wav, "detach"):
        base_wav = base_wav.detach().cpu().numpy()
    sf.write(os.path.join(args.work, "baseline_cpu.wav"), base_wav.squeeze(), 24000)

    emb = cap["emb"]
    mel_lat = cap["mel_lat"]
    assert emb is not None, "failed to capture inputs_embeds"
    assert mel_lat is not None, "failed to capture mel latents"
    mel_len = mel_lat.shape[1]
    print(f"[A] captured emb {tuple(emb.shape)}  mel_lat {tuple(mel_lat.shape)}  mel_len={mel_len}")

    if cap.get("audio_codes") is not None:
        torch.save(cap["audio_codes"].cpu(), os.path.join(args.work, "audio_codes.pt"))
        print(f"[A] audio_codes {tuple(cap['audio_codes'].shape)} first8={cap['audio_codes'].flatten()[:8].tolist()}")
    if cap.get("prefix_emb") is not None:
        torch.save(cap["prefix_emb"].float(), os.path.join(args.work, "prefix_emb.pt"))
        print(f"[A] prefix_emb {tuple(cap['prefix_emb'].shape)}")
    torch.save(
        {
            "start_audio": int(model.gpt.start_audio_token),
            "stop_audio": int(model.gpt.stop_audio_token),
            "repetition_penalty": 10.0,
        },
        os.path.join(args.work, "gen_meta.pt"),
    )
    torch.save(emb.float(), os.path.join(args.work, "emb.pt"))
    torch.save(speaker_embedding.float(), os.path.join(args.work, "speaker_embedding.pt"))
    torch.save(mel_lat.float(), os.path.join(args.work, "gpt_latents_cpu.pt"))
    torch.save({"mel_len": int(mel_len)}, os.path.join(args.work, "meta.pt"))
    print(f"[A] wrote handoffs + baseline_cpu.wav to {args.work}")


if __name__ == "__main__":
    main()
