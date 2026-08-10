# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Re-capture the golden/ fixtures for Blocks 1, 2 and 4 from a real coqui XTTS-v2 run.

Runs in the **coqui venv** (needs the `TTS` package), NOT the tt-metal python_env:

    XTTS_CKPT_DIR=/home/acicovic/xtts_ref \
      /home/acicovic/xtts_cpu_venv/bin/python capture_goldens.py \
        --ref /home/acicovic/xtts_ref/ref.pt --sr 24000 --text "..."

Why this exists: the `golden/**/*.pt` fixtures the PCC tests load are gitignored and were
never committed, so a fresh checkout has none. Block 3's goldens are synthetic and can be
rebuilt from the checkpoint alone (`reference/xtts_gpt_ref.py --ckpt <model.pth>`), but
Blocks 1/2/4 are validated against *real activations* of the coqui model, which only exist
if you run coqui. This script hooks the coqui modules at exactly the block boundaries the
tests assume and writes:

  golden/cond/mel_in.pt              [1,80,T]      conditioning-encoder input mel
  golden/cond/enc_out.pt            [1,1024,T]    conditioning-encoder output
  golden/cond/perc_out.pt           [1,32,1024]   Perceiver output (pre-transpose)
  golden/cond/gpt_cond_latent.pt    [1,32,1024]   Block-1 contract output
  golden/speaker/logmel.pt          [1,64,T]      speaker-encoder logmel (instancenorm input)
  golden/speaker/audio_16k.pt       [1,N]         16 kHz reference waveform
  golden/speaker/speaker_embedding.pt [1,512,1]   d-vector
  golden/hifigan/z.pt               [1,1024,L]    vocoder generator input
  golden/hifigan/g.pt               [1,512,1]     d-vector as fed to the generator
  golden/hifigan/wav.pt             [1,1,L*256]   waveform
  golden/hifigan/dbg_conv_pre.pt    [1,512,L]     per-stage oracle (conv_pre, pre-cond)
  golden/hifigan/dbg_ups{0..3}.pt   [1,C,L']      per-stage oracles (transpose conv, pre-cond/MRF)
"""

import argparse
import os

os.environ["COQUI_TOS_AGREED"] = "1"

import numpy as np
import torch

# Shim: coqui's tortoise code imports isin_mps_friendly, which transformers >= 5 dropped.
import transformers.pytorch_utils as _ptu

if not hasattr(_ptu, "isin_mps_friendly"):
    _ptu.isin_mps_friendly = lambda elements, test_elements: torch.isin(elements, test_elements)

import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

CKPT = os.environ.get("XTTS_CKPT_DIR", "/home/acicovic/xtts_ref")
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden")


def load_ref_waveform(path, sr):
    """Accept a raw tensor/ndarray, a (tensor, sr) tuple, or a HF audio dict."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and isinstance(raw.get("audio"), dict) and "array" in raw["audio"]:
        sr = int(raw["audio"].get("sampling_rate", sr))
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
    return wav.float().reshape(1, -1), sr


def as_1x32x1024(t):
    """Normalize a cond latent to the contract's [1,32,1024] (coqui carries it [1,1024,32])."""
    assert t.dim() == 3 and 32 in t.shape[1:] and 1024 in t.shape[1:], f"unexpected cond latent {tuple(t.shape)}"
    return t if t.shape[1] == 32 else t.transpose(1, 2).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference-voice waveform .pt")
    ap.add_argument("--sr", type=int, default=22050, help="sample rate of the .pt waveform")
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--out", default=GOLDEN)
    args = ap.parse_args()

    for sub in ("cond", "speaker", "hifigan"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    print(f"[cap] loading XTTS from {CKPT} (CPU)")
    config = XttsConfig()
    config.load_json(os.path.join(CKPT, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=CKPT, eval=True, use_deepspeed=False)
    model.eval()

    audio, sr_eff = load_ref_waveform(args.ref, args.sr)
    print(f"[cap] ref waveform {tuple(audio.shape)} @ {sr_eff} Hz ({audio.shape[1] / sr_eff:.2f}s)")

    # ---------------- Block 1: conditioning encoder + Perceiver ----------------
    cap = {}

    def enc_hook(mod, inp, out):
        cap.setdefault("mel_in", []).append(inp[0].detach().clone())
        cap.setdefault("enc_out", []).append(out.detach().clone())

    def perc_hook(mod, inp, out):
        cap.setdefault("perc_out", []).append(out.detach().clone())

    h1 = model.gpt.conditioning_encoder.register_forward_hook(enc_hook)
    h2 = model.gpt.conditioning_perceiver.register_forward_hook(perc_hook)
    gpt_cond_latent = model.get_gpt_cond_latents(audio, sr_eff)
    h1.remove()
    h2.remove()

    n_chunks = len(cap["mel_in"])
    # get_gpt_cond_latents chunks the reference at 6s and MEANS the per-chunk style embeddings.
    # A golden pair (mel_in -> enc_out/perc_out) is only well defined for a single chunk, and
    # gpt_cond_latent == perc_out only then. Keep the reference clip <= 6s.
    assert n_chunks == 1, (
        f"reference clip produced {n_chunks} conditioning chunks; the golden pair requires exactly 1 "
        f"(use a reference clip <= 6s, or the mean over chunks won't match mel_in)"
    )
    mel_in = cap["mel_in"][0].float()
    enc_out = cap["enc_out"][0].float()
    perc_out = cap["perc_out"][0].float()
    gpt_cond_latent = as_1x32x1024(gpt_cond_latent.float())

    torch.save(mel_in, os.path.join(args.out, "cond", "mel_in.pt"))
    torch.save(enc_out, os.path.join(args.out, "cond", "enc_out.pt"))
    torch.save(perc_out, os.path.join(args.out, "cond", "perc_out.pt"))
    torch.save(gpt_cond_latent, os.path.join(args.out, "cond", "gpt_cond_latent.pt"))
    print(
        f"[cap] cond: mel_in {tuple(mel_in.shape)}  enc_out {tuple(enc_out.shape)}  "
        f"perc_out {tuple(perc_out.shape)}  gpt_cond_latent {tuple(gpt_cond_latent.shape)}"
    )

    # ---------------- Block 2: ResNet speaker encoder ----------------
    spk = {}
    h3 = model.hifigan_decoder.speaker_encoder.instancenorm.register_forward_pre_hook(
        lambda m, a: spk.update(logmel=a[0].detach().clone())
    )
    speaker_embedding = model.get_speaker_embedding(audio, sr_eff)
    h3.remove()
    audio_16k = torchaudio.functional.resample(audio, sr_eff, 16000)

    torch.save(spk["logmel"].float(), os.path.join(args.out, "speaker", "logmel.pt"))
    torch.save(audio_16k.float(), os.path.join(args.out, "speaker", "audio_16k.pt"))
    torch.save(speaker_embedding.float(), os.path.join(args.out, "speaker", "speaker_embedding.pt"))
    print(
        f"[cap] speaker: logmel {tuple(spk['logmel'].shape)}  audio_16k {tuple(audio_16k.shape)}  "
        f"d-vector {tuple(speaker_embedding.shape)}"
    )

    # ---------------- Block 4: HiFi-GAN vocoder ----------------
    # The generator (waveform_decoder) is called once per inference with z and g; its own
    # conv_pre / ups[i] outputs are the per-stage oracles the TT test prints.
    voc = {}
    gen = model.hifigan_decoder.waveform_decoder

    def gen_pre(mod, a, kw):
        voc["z"] = a[0].detach().clone()
        g = kw.get("g") if "g" in kw else (a[1] if len(a) > 1 else None)
        voc["g"] = None if g is None else g.detach().clone()

    handles = [gen.register_forward_pre_hook(gen_pre, with_kwargs=True)]
    handles.append(gen.register_forward_hook(lambda m, a, o: voc.update(wav=o.detach().clone())))
    handles.append(gen.conv_pre.register_forward_hook(lambda m, a, o: voc.update(conv_pre=o.detach().clone())))
    for i in range(len(gen.ups)):
        handles.append(
            gen.ups[i].register_forward_hook(lambda m, a, o, i=i: voc.update(**{f"ups{i}": o.detach().clone()}))
        )

    print(f"[cap] running coqui inference (sampled) to drive the vocoder ...")
    torch.manual_seed(0)
    model.inference(args.text, args.lang, gpt_cond_latent, speaker_embedding, do_sample=True)
    for h in handles:
        h.remove()

    z = voc["z"].float()
    if z.dim() == 2:
        z = z.unsqueeze(0)
    wav = voc["wav"].float().reshape(1, 1, -1)
    torch.save(z, os.path.join(args.out, "hifigan", "z.pt"))
    torch.save(voc["g"].float().reshape(1, 512, 1), os.path.join(args.out, "hifigan", "g.pt"))
    torch.save(wav, os.path.join(args.out, "hifigan", "wav.pt"))
    torch.save(voc["conv_pre"].float(), os.path.join(args.out, "hifigan", "dbg_conv_pre.pt"))
    for i in range(len(gen.ups)):
        torch.save(voc[f"ups{i}"].float(), os.path.join(args.out, "hifigan", f"dbg_ups{i}.pt"))
    print(
        f"[cap] hifigan: z {tuple(z.shape)}  g {tuple(voc['g'].shape)}  wav {tuple(wav.shape)}  "
        f"conv_pre {tuple(voc['conv_pre'].shape)}  ups3 {tuple(voc['ups3'].shape)}"
    )
    print(f"[cap] wrote goldens under {os.path.normpath(args.out)}")


if __name__ == "__main__":
    main()
