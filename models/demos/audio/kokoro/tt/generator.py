# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""First-class, fully-on-device generator for hexgrad/Kokoro-82M.

``KokoroGenerator`` wraps :class:`KokoroDevicePipeline` (whose ``synthesize_device``
runs the entire pipeline on device — TT plbert, prosody predictor, text encoder, and
the ISTFTNet vocoder) and adds the text frontend (grapheme-to-phoneme, plbert-context
chunking, chunk joining) so callers work in text/voice terms, like ``WhisperGenerator``.

Two entrypoints:
- ``generate(text, voice)`` -> one 24 kHz waveform (chunks synthesized + joined).
- ``stream(text, voice)`` -> yields each chunk's waveform as it is produced.

Reuse/streaming across chunks is safe: ttnn caches prepared conv weights by input
length, so distinct chunk lengths would otherwise grow L1_SMALL until OOM; the
generator clears the program cache after each chunk to keep that bounded. (A fixed
chunk-length/padding scheme would avoid the per-chunk recompile — future perf work.)
"""

import sys
import types

import numpy as np
import torch

# Stub spaCy before importing kokoro (misaki.en imports spacy, whose numpy-2 ABI
# clashes with the numpy 1.26 TTNN is built against). Only the espeak G2P path is used.
if "spacy" not in sys.modules:
    sys.modules["spacy"] = types.ModuleType("spacy")

from models.demos.audio.kokoro.tt.device_pipeline import KokoroDevicePipeline

MODEL_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"
MAX_PHONEME_TOKENS = 510  # plbert context is 512; KModel reserves 2 (bos/eos)
# Hard device limit is MAX_PHONEME_TOKENS, but the ISTFTNet vocoder's L1_SMALL
# scratch grows with a chunk's frame count; long single chunks overflow L1 (and
# recompile a distinct kernel shape each time). Grouping sentences up to this many
# phoneme tokens keeps every chunk within the validated envelope and improves
# streaming latency (audio starts sooner). Long single sentences still fall back to
# clause/word splitting below.
CHUNK_PHONEME_TOKENS = 160
INTER_CHUNK_PAUSE_S = 0.30
BOUNDARY_FADE_MS = 10


class _Utt:
    def __init__(self, text):
        self.text = text


def _edge_fade(audio: np.ndarray, sample_rate: int, ms: int) -> np.ndarray:
    """Raised-cosine fade-in/out on the first/last ``ms`` to suppress chunk-edge clicks."""
    n = min(int(sample_rate * ms / 1000), audio.size // 2)
    if n <= 0:
        return audio
    ramp = (0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))).astype(np.float32)
    audio = audio.copy()
    audio[:n] *= ramp
    audio[-n:] *= ramp[::-1]
    return audio


class KokoroGenerator:
    """Fully-on-device Kokoro-82M text-to-speech generator."""

    def __init__(self, kmodel, mesh_device, default_voice: str = DEFAULT_VOICE):
        self.km = kmodel
        self.mesh = mesh_device
        self.pipe = KokoroDevicePipeline(kmodel, mesh_device)
        self.default_voice = default_voice
        self._g2p = None  # lazy: espeak backend
        self._voices = {}  # name -> style pack tensor [510, 1, 256]

    # ---------------------------------------------------------------- frontend
    def _g2p_fn(self):
        if self._g2p is None:
            from misaki.espeak import EspeakFallback

            self._g2p = EspeakFallback(british=False)
        return self._g2p

    def _load_voice(self, name: str):
        if name not in self._voices:
            from huggingface_hub import hf_hub_download

            self._voices[name] = torch.load(hf_hub_download(MODEL_ID, f"voices/{name}.pt"), weights_only=True)
        return self._voices[name]

    def _n_tokens(self, phonemes: str) -> int:
        return sum(1 for p in phonemes if self.km.vocab.get(p) is not None)

    def _chunk_text(self, text: str):
        """Sentence-aligned chunks whose phonemization stays within the plbert context."""
        import re

        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s] or [text.strip()]
        chunks, current = [], ""
        g2p = self._g2p_fn()
        for sentence in sentences:
            candidate = (current + " " + sentence).strip() if current else sentence
            ps, _ = g2p(_Utt(candidate))
            if ps is not None and self._n_tokens(ps) <= CHUNK_PHONEME_TOKENS:
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = ""
            ps, _ = g2p(_Utt(sentence))
            if ps is not None and self._n_tokens(ps) <= CHUNK_PHONEME_TOKENS:
                current = sentence
                continue
            buf = ""
            for piece in re.split(r"(?<=[,;:])\s+", sentence):
                cand = (buf + " " + piece).strip() if buf else piece
                ps, _ = g2p(_Utt(cand))
                if ps is not None and self._n_tokens(ps) <= CHUNK_PHONEME_TOKENS:
                    buf = cand
                else:
                    if buf:
                        chunks.append(buf)
                    buf = piece
            if buf:
                current = buf
        if current:
            chunks.append(current)
        return chunks or [text.strip()]

    # ---------------------------------------------------------------- device synth
    def _synth_chunk(self, chunk: str, voice: str, speed: float):
        """One chunk -> 24 kHz waveform, entirely on device. Returns None if empty."""
        g2p = self._g2p_fn()
        ps, _ = g2p(_Utt(chunk))
        ids = [i for i in (self.km.vocab.get(p) for p in ps) if i is not None][:MAX_PHONEME_TOKENS]
        if not ids:
            return None
        input_ids = torch.LongTensor([[0, *ids, 0]])
        pack = self._load_voice(voice)
        ref_s = pack[min(len(ids) - 1, pack.shape[0] - 1)]
        with torch.no_grad():
            audio = self.pipe.synthesize_device(input_ids, ref_s, speed).detach().cpu().numpy()
        # ttnn caches prepared conv weights by input length; clear so distinct chunk
        # lengths don't accumulate in L1_SMALL across a stream.
        self.mesh.clear_program_cache()
        return audio.astype(np.float32)

    # ---------------------------------------------------------------- API
    def stream(self, text: str, voice: str = None, speed: float = 1.0):
        """Yield each chunk's waveform (np.float32, 24 kHz) as it is synthesized on device."""
        voice = voice or self.default_voice
        for chunk in self._chunk_text(text):
            audio = self._synth_chunk(chunk, voice, speed)
            if audio is not None and audio.size:
                yield _edge_fade(audio, SAMPLE_RATE, BOUNDARY_FADE_MS)

    def generate(self, text: str, voice: str = None, speed: float = 1.0) -> np.ndarray:
        """Full text -> single 24 kHz waveform (chunks synthesized on device + joined)."""
        gap = np.zeros(int(SAMPLE_RATE * INTER_CHUNK_PAUSE_S), dtype=np.float32)
        parts = []
        for audio in self.stream(text, voice, speed):
            if parts:
                parts.append(gap)
            parts.append(audio)
        if not parts:
            raise ValueError("No pronounceable content produced from input text")
        return np.concatenate(parts)

    def teardown(self):
        try:
            if self.pipe is not None and getattr(self.pipe, "_plbert_dec", None) is not None:
                self.pipe._plbert_dec.release_traces()
        except Exception:
            pass
        self.pipe = None


def build_generator(mesh_device, model_id: str = MODEL_ID, default_voice: str = DEFAULT_VOICE):
    """Load the host KModel (weights) and build a fully-on-device KokoroGenerator."""
    from kokoro.model import KModel

    km = KModel(repo_id=model_id).eval()
    return KokoroGenerator(km, mesh_device, default_voice=default_voice)
