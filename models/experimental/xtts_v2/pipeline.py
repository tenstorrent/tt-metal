# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""XTTS-v2 end-to-end inference pipeline — TTNN Block 3 (GPT) + coqui/reference glue.

Produces a 24 kHz WAV so end-to-end audio quality can be judged by ear. The pipeline is a chain of
swappable seams; today Block 3 (GPT autoregressive decode) is our TTNN implementation and the rest
is coqui's (== our validated references, PCC=1.0). To swap in a future TTNN block, replace the
corresponding step:

    text ──[tokenizer]──► text_tokens ─────────────────────────────┐
    speaker ──[built-in latents / Blocks 1+2]──► gpt_cond_latent ───┤
                                                 speaker_embedding ─┼─► [Block 4 vocoder] ─► wav
                              [GPT prefix]──► prefix_emb ─► [Block 3: TTNN GPT] ─► latents ─┘

Assets (fetched to reference/weights/, gitignored): model.pth, config.json, vocab.json,
speakers_xtts.pth. Run:
    TT_METAL_HOME=<repo> PYTHONPATH=<repo> python models/experimental/xtts_v2/pipeline.py
"""

import os
import re
import sys
import types
import unicodedata

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(_HERE, "reference")
WEIGHTS = os.path.join(REF, "weights")
CKPT = os.path.join(WEIGHTS, "model.pth")
VOCAB = os.path.join(WEIGHTS, "vocab.json")
SPEAKERS = os.path.join(WEIGHTS, "speakers_xtts.pth")
OUT_WAV = os.path.join(_HERE, "generated", "xtts_out.wav")
OUTPUT_SR = 24000


def _install_shims():
    """Make the vendored coqui code importable under our lean env."""
    ko = types.ModuleType("ko_speech_tools")
    ko.hangul_romanize = lambda *a, **k: ""  # Korean-only; unused for English
    sys.modules.setdefault("ko_speech_tools", ko)
    import transformers.pytorch_utils as ptu  # transformers 5.x removed isin_mps_friendly

    if not hasattr(ptu, "isin_mps_friendly"):
        ptu.isin_mps_friendly = lambda elements, test_elements: torch.isin(elements, test_elements)
    if REF not in sys.path:
        sys.path.insert(0, REF)


# --------------------------------------------------------------------------------------
# Tokenizer — exact XTTS multilingual_cleaners (en) + [lang]/[SPACE] wrapping + raw BPE.
# (Reimplemented so we don't need coqui's tokenizer.py dep tree; see reference/PROVENANCE.md.)
# --------------------------------------------------------------------------------------
_WHITESPACE = re.compile(r"\s+")


def _multilingual_cleaners_en(text):
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    # replace_symbols(lang="en")
    text = text.replace(";", ",").replace("-", " ").replace(":", ",").replace("&", " and ")
    text = re.sub(r'[\<\>\(\)\[\]\"]+', "", text)  # remove_aux_symbols
    return _WHITESPACE.sub(" ", text).strip()  # collapse_whitespace


class XttsTokenizer:
    def __init__(self, vocab_file=VOCAB):
        from tokenizers import Tokenizer

        self.t = Tokenizer.from_file(vocab_file)

    def token_to_id(self, s):
        return self.t.token_to_id(s)

    def encode(self, text, lang="en"):
        if lang != "en":
            raise NotImplementedError("pipeline is wired for English (en) only for now")
        txt = _multilingual_cleaners_en(text)
        txt = f"[{lang}]{txt}".replace(" ", "[SPACE]")
        return self.t.encode(txt).ids


# --------------------------------------------------------------------------------------
# Conditioning — built-in speaker latents (bypasses Blocks 1+2). To exercise Blocks 1/2,
# swap this for reference-audio -> conditioning encoder + speaker encoder.
# --------------------------------------------------------------------------------------
def load_speaker(name=None):
    sp = torch.load(SPEAKERS, map_location="cpu", weights_only=False)
    if name is None:
        name = list(sp.keys())[0]
    e = sp[name]
    return e["gpt_cond_latent"].float(), e["speaker_embedding"].float(), name  # [1,32,1024], [1,512,1]


# --------------------------------------------------------------------------------------
# GPT prefix — coqui text embeddings + positions, prepended with the conditioning latents
# (mirrors GPT.compute_embeddings). This is the prefix_emb our TTNN decoder prefills.
# --------------------------------------------------------------------------------------
def build_gpt_embedder(tok):
    from models.experimental.xtts_v2.tests._coqui_groundtruth import build_coqui_gpt, load_gpt_weights

    gpt = build_coqui_gpt(load_gpt_weights(CKPT))
    gpt.start_text_token = tok.token_to_id("[START]")
    gpt.stop_text_token = tok.token_to_id("[STOP]")
    return gpt


@torch.no_grad()
def build_prefix(gpt, cond_latents, text_tokens):
    import torch.nn.functional as F

    ti = torch.tensor(text_tokens, dtype=torch.long).unsqueeze(0)  # [1, T]
    ti = F.pad(ti, (0, 1), value=gpt.stop_text_token)
    ti = F.pad(ti, (1, 0), value=gpt.start_text_token)
    emb = gpt.text_embedding(ti) + gpt.text_pos_embedding(ti)  # [1, T+2, 1024]
    return torch.cat([cond_latents, emb], dim=1)  # [1, P, 1024]


# --------------------------------------------------------------------------------------
# Block 4 — HiFi-GAN vocoder (coqui HifiDecoder: latents -> interpolate -> waveform).
# --------------------------------------------------------------------------------------
def build_vocoder():
    # HifiDecoder imports `load_fsspec` from trainer.io and pulls ResNetSpeakerEncoder (losses/coqpit);
    # stub those training-only deps with placeholder modules (same trick as the reference tests).
    from models.experimental.xtts_v2.tests._coqui_groundtruth import _force_stub

    _force_stub(["trainer", "trainer.io", "trainer.generic_utils", "TTS.encoder.losses", "coqpit"])
    from TTS.tts.layers.xtts.hifigan_decoder import HifiDecoder

    dec = HifiDecoder()  # constructor defaults match config.json
    obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    sub = {k[len("hifigan_decoder."):]: v for k, v in state.items() if k.startswith("hifigan_decoder.")}
    dec.load_state_dict(sub, strict=False)
    dec.eval()
    return dec


class XttsPipeline:
    """End-to-end XTTS-v2. Host-side glue on CPU; only Block 3 runs on the TT device."""

    def __init__(self, device):
        _install_shims()
        from models.experimental.xtts_v2.reference.xtts_gpt_ref import load_gen_head

        from models.experimental.xtts_v2.tt.ttnn_xtts_hifigan import TtHifiganGenerator

        self.device = device
        self.tok = XttsTokenizer()
        self.gpt = build_gpt_embedder(self.tok)  # host: text embeddings + prefix
        self.heads = load_gen_head(CKPT)  # host: mel_emb / mel_pos / mel_head
        self.vocoder = build_vocoder()  # Block 4 wrapper (does the latents->z interpolation on host)
        self.vocoder.waveform_decoder = TtHifiganGenerator(device)  # Block 4 generator on TTNN

    @torch.no_grad()
    def generate(self, text, speaker=None, language="en", max_new=400, seed=0):
        import models.experimental.xtts_v2.tt.ttnn_xtts_gpt as port

        cond, spk, name = load_speaker(speaker)
        tokens = self.tok.encode(text, language)
        prefix = build_prefix(self.gpt, cond, tokens)  # [1, P, 1024]
        P = prefix.shape[1]
        max_new = min(max_new, self.heads["mel_pos"].shape[0] - 1, 605)
        out = port.generate_traced(  # Block 3 (TTNN); XTTS needs sampling (greedy collapses)
            self.device, prefix, self.heads, max_new=max_new, max_seq=P + max_new + 8,
            use_trace=True, stop_token=port.STOP_AUDIO_TOKEN,
            do_sample=True, temperature=0.75, top_k=50, top_p=0.85, repetition_penalty=10.0, seed=seed,
        )
        wav = self.vocoder(out["latents"], g=spk)  # Block 4 -> [1, 1, L] @ 24 kHz
        return wav.squeeze().cpu(), {"speaker": name, "text_tokens": len(tokens), "audio_codes": out["codes"].numel()}


def save_wav(wav, path=OUT_WAV, sr=OUTPUT_SR):
    import wave

    os.makedirs(os.path.dirname(path), exist_ok=True)
    a = (wav.detach().clamp(-1, 1).numpy() * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(a.tobytes())
    return path


def main():
    import ttnn

    text = "Hello! This is a test of the text to speech system, running on Tenstorrent hardware."
    device = ttnn.open_device(device_id=0, trace_region_size=80_000_000, l1_small_size=131072)
    try:
        pipe = XttsPipeline(device)
        wav, info = pipe.generate(text, max_new=300)
    finally:
        ttnn.close_device(device)
    path = save_wav(wav)
    dur = wav.shape[0] / OUTPUT_SR
    print(f"[pipeline] {info} | wav {tuple(wav.shape)} ({dur:.2f}s @ {OUTPUT_SR} Hz) -> {path}")


if __name__ == "__main__":
    main()
