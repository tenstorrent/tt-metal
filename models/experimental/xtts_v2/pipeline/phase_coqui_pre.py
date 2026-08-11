# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Phase A (coqui venv, CPU): conditioning + tokenize + generate codes, capture the GPT
inputs_embeds for the return_latent forward, and produce a full-CPU baseline wav.

One request (the original single-card flow):

    phase_coqui_pre.py --text "..." --ref ref.pt --sr 24000 --work out/work

Many requests for data-parallel serving (`phase_tt_mesh.py`), one work dir per request:

    phase_coqui_pre.py --texts-file texts.txt --ref ref.pt --sr 24000 --work-root out/reqs

The model is loaded once and the reference clip's conditioning (gpt_cond_latent /
speaker_embedding, which depend on the *reference audio only*, not the text) is computed once
and copied into every work dir — only the text-dependent captures are redone per request.
"""
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

CKPT = os.environ.get("XTTS_CKPT_DIR")  # coqui checkpoint dir (config.json, model.pth, vocab.json)


def load_model():
    print("[A] loading XTTS (CPU)...")
    config = XttsConfig()
    config.load_json(os.path.join(CKPT, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=CKPT, eval=True, use_deepspeed=False)
    model.eval()
    return model


def load_ref_audio(path, sr):
    """reference .pt waveform -> tensor-level conditioning (bypasses torchcodec file IO).
    Handles: raw tensor/ndarray, (tensor, sr) tuple, or a HuggingFace audio-sample dict
    {"audio": {"array": ndarray, "sampling_rate": int}}."""
    import numpy as np

    raw = torch.load(path, map_location="cpu", weights_only=False)
    sr_eff = sr
    if isinstance(raw, dict) and isinstance(raw.get("audio"), dict) and "array" in raw["audio"]:
        sr_eff = int(raw["audio"].get("sampling_rate", sr))
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
    return audio, sr_eff


def capture_conditioning(model, audio, sr_eff):
    """Both conditioning branches + the Block-1/Block-2 *inputs* (which the TT path recomputes
    on device). Depends only on the reference clip, so it is shared by every request."""
    # capture the conditioning-encoder input mel (Block 1 input)
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
    print(
        f"[A] gpt_cond_latent {tuple(gpt_cond_latent.shape)}  speaker_embedding {tuple(speaker_embedding.shape)}  "
        f"cond_mel_in {tuple(_mel['x'].shape)}"
    )
    return {
        "gpt_cond_latent": gpt_cond_latent,
        "speaker_embedding": speaker_embedding,
        "cond_mel_in": _mel["x"].float(),
        "speaker_logmel": _spk["logmel"].float(),
    }


def capture_request(model, text, lang, cond, work):
    """Run one full-CPU inference and write this request's handoffs into `work`."""
    os.makedirs(work, exist_ok=True)
    gpt_cond_latent = cond["gpt_cond_latent"]
    speaker_embedding = cond["speaker_embedding"]
    torch.save(cond["cond_mel_in"], os.path.join(work, "cond_mel_in.pt"))  # [1,80,T]
    torch.save(cond["speaker_logmel"], os.path.join(work, "speaker_logmel.pt"))  # [1,64,T]

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
    out = model.inference(text, lang, gpt_cond_latent, speaker_embedding, do_sample=True)
    h1.remove()
    h1b.remove()
    model.gpt.forward = _orig_gpt_forward
    model.gpt.gpt_inference.store_prefix_emb = _orig_store

    am = cap.get("attn_mask")
    print(f"[A] attn_mask: {None if am is None else (tuple(am.shape), 'all_ones=' + str(bool((am==1).all())))}")
    if cap.get("lh") is not None:
        torch.save(cap["lh"].float(), os.path.join(work, "last_hidden_cpu.pt"))
        print(f"[A] saved last_hidden_cpu {tuple(cap['lh'].shape)}")

    base_wav = out["wav"]
    if hasattr(base_wav, "detach"):
        base_wav = base_wav.detach().cpu().numpy()
    sf.write(os.path.join(work, "baseline_cpu.wav"), base_wav.squeeze(), 24000)

    emb = cap["emb"]
    mel_lat = cap["mel_lat"]
    assert emb is not None, "failed to capture inputs_embeds"
    assert mel_lat is not None, "failed to capture mel latents"
    mel_len = mel_lat.shape[1]
    print(f"[A] captured emb {tuple(emb.shape)}  mel_lat {tuple(mel_lat.shape)}  mel_len={mel_len}")

    if cap.get("audio_codes") is not None:
        torch.save(cap["audio_codes"].cpu(), os.path.join(work, "audio_codes.pt"))
        print(f"[A] audio_codes {tuple(cap['audio_codes'].shape)} first8={cap['audio_codes'].flatten()[:8].tolist()}")
    if cap.get("prefix_emb") is not None:
        torch.save(cap["prefix_emb"].float(), os.path.join(work, "prefix_emb.pt"))
        print(f"[A] prefix_emb {tuple(cap['prefix_emb'].shape)}")
    torch.save(
        {
            "start_audio": int(model.gpt.start_audio_token),
            "stop_audio": int(model.gpt.stop_audio_token),
            "repetition_penalty": 10.0,
        },
        os.path.join(work, "gen_meta.pt"),
    )
    torch.save(emb.float(), os.path.join(work, "emb.pt"))
    torch.save(speaker_embedding.float(), os.path.join(work, "speaker_embedding.pt"))
    torch.save(mel_lat.float(), os.path.join(work, "gpt_latents_cpu.pt"))
    torch.save({"mel_len": int(mel_len)}, os.path.join(work, "meta.pt"))
    with open(os.path.join(work, "text.txt"), "w") as f:
        f.write(text + "\n")
    print(f"[A] wrote handoffs + baseline_cpu.wav to {work}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="single request's text")
    ap.add_argument("--texts-file", help="one text per line; N requests -> N work dirs under --work-root")
    ap.add_argument("--ref", required=True, help="path to reference audio .pt (waveform)")
    ap.add_argument("--sr", type=int, default=22050, help="sample rate of the .pt waveform")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--work", help="work dir (single request)")
    ap.add_argument("--work-root", help="parent dir for per-request work dirs (with --texts-file)")
    ap.add_argument("--limit", type=int, default=0, help="with --texts-file: use only the first N texts")
    args = ap.parse_args()
    if not CKPT:  # checked after argparse, not at import time, so --help works without the env
        raise SystemExit("set XTTS_CKPT_DIR to the coqui XTTS-v2 checkpoint dir")

    if args.texts_file:
        assert args.work_root, "--texts-file needs --work-root"
        with open(args.texts_file) as f:
            texts = [ln.strip() for ln in f if ln.strip()]
        if args.limit:
            texts = texts[: args.limit]
    else:
        assert args.text and args.work, "need --text and --work (or --texts-file and --work-root)"
        texts = [args.text]

    model = load_model()
    audio, sr_eff = load_ref_audio(args.ref, args.sr)
    cond = capture_conditioning(model, audio, sr_eff)

    if args.texts_file:
        os.makedirs(args.work_root, exist_ok=True)
        for i, text in enumerate(texts):
            work = os.path.join(args.work_root, f"r{i}")
            print(f"\n[A] === request {i + 1}/{len(texts)} -> {work} ===")
            capture_request(model, text, args.lang, cond, work)
        print(f"\n[A] prepared {len(texts)} request work dirs under {args.work_root}")
    else:
        capture_request(model, texts[0], args.lang, cond, args.work)


if __name__ == "__main__":
    main()
