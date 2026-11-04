# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tekken tokenizer + Voxtral-TTS prompt assembly, reimplemented from `tekken.json`.

Replaces `mistral_common`; validated by exact token-id match against its `encode_speech_request`
(tests/test_tokenizer_ref.py). Needs `regex`: tekken's split pattern uses Unicode property classes
that stdlib `re` cannot parse.

Run to check against the shipped ground-truth prompts:
    PYTHONPATH=<repo> python models/experimental/voxtral_tts/reference/voxtral_tokenizer_ref.py
"""

import base64
import json
import os

from models.experimental.voxtral_tts.reference.voxtral_paths import DOWNLOAD_HINT, MODEL_DIR

DEFAULT_TEKKEN = os.path.join(MODEL_DIR, "tekken.json")

# Special ids, resolved by NAME from tekken.json rather than hard-coded (asserted in __init__).
BOS = "<s>"
BEGIN_AUDIO = "[BEGIN_AUDIO]"
AUDIO = "[AUDIO]"
NEXT_AUDIO_TEXT = "[NEXT_AUDIO_TEXT]"
REPEAT_AUDIO_TEXT = "[REPEAT_AUDIO_TEXT]"


def _bpe(ranks, piece):
    """Classic tiktoken byte-pair merge: repeatedly merge the adjacent pair with the LOWEST rank."""
    if piece in ranks:
        return [ranks[piece]]
    parts = [bytes([b]) for b in piece]
    while len(parts) > 1:
        best, best_i = None, None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and (best is None or r < best):
                best, best_i = r, i
        if best_i is None:
            break
        parts[best_i : best_i + 2] = [parts[best_i] + parts[best_i + 1]]
    out = []
    for p in parts:
        if p in ranks:
            out.append(ranks[p])
        else:  # unreachable for well-formed vocabularies (all 256 single bytes are present)
            out.extend(ranks[bytes([b])] for b in p)
    return out


class TekkenTokenizer:
    """Byte-level BPE over tekken.json. `encode`/`decode` handle text; `build_prompt` assembles
    the full TTS prompt including the audio placeholders."""

    def __init__(self, path=DEFAULT_TEKKEN):
        import regex  # not stdlib re: the split pattern uses Unicode property classes

        if not os.path.exists(path):
            raise FileNotFoundError(f"tekken.json not found: {path}\n{DOWNLOAD_HINT}")
        with open(path) as f:
            d = json.load(f)
        cfg = d["config"]
        self.n_special = cfg["default_num_special_tokens"]
        self.vocab_size = cfg["default_vocab_size"]
        self.pattern = regex.compile(cfg["pattern"])

        # Only the first (vocab_size - n_special) entries are in the released vocabulary.
        n_regular = self.vocab_size - self.n_special
        self.ranks = {}
        self.by_rank = {}
        for v in d["vocab"][:n_regular]:
            b = base64.b64decode(v["token_bytes"])
            self.ranks[b] = v["rank"]
            self.by_rank[v["rank"]] = b

        self.special = {s["token_str"]: s["rank"] for s in d["special_tokens"]}
        for name in (BOS, BEGIN_AUDIO, AUDIO, NEXT_AUDIO_TEXT, REPEAT_AUDIO_TEXT):
            assert name in self.special, f"tekken.json is missing special token {name!r}"
        self.voice_frames = dict(d["audio"]["voice_num_audio_tokens"])

    # -- ids <-> bytes -----------------------------------------------------------------
    def encode(self, text):
        """Raw text -> token ids, with no normalization: tekken is byte-level and case/space sensitive."""
        out = []
        for m in self.pattern.findall(text):
            out.extend(r + self.n_special for r in _bpe(self.ranks, m.encode("utf-8")))
        return out

    def decode(self, ids):
        """Regular ids -> text. Special ids (< n_special) are skipped."""
        buf = b"".join(self.by_rank[i - self.n_special] for i in ids if i >= self.n_special)
        return buf.decode("utf-8", errors="replace")

    # -- prompt ------------------------------------------------------------------------
    def n_audio_tokens(self, voice):
        if voice not in self.voice_frames:
            raise KeyError(f"unknown voice {voice!r}; available: {sorted(self.voice_frames)}")
        return self.voice_frames[voice]

    def build_prompt(self, text, voice):
        """text + voice name -> prompt ids, matching mistral_common's encode_speech_request."""
        sp = self.special
        n = self.n_audio_tokens(voice)
        return (
            [sp[BOS], sp[BEGIN_AUDIO]]
            + [sp[AUDIO]] * n
            + [sp[NEXT_AUDIO_TEXT]]
            + self.encode(text)
            + [sp[REPEAT_AUDIO_TEXT], sp[BEGIN_AUDIO]]
        )

    @property
    def audio_token_id(self):
        return self.special[AUDIO]

    @property
    def voices(self):
        return sorted(self.voice_frames)


def main():
    tok = TekkenTokenizer()
    print(f"[tok] vocab {tok.vocab_size} ({tok.n_special} special) | {len(tok.voices)} voices")
    print(
        f"[tok] audio_token_id={tok.audio_token_id} "
        f"| frames: min {min(tok.voice_frames.values())} max {max(tok.voice_frames.values())}"
    )

    text = "It took me quite a long time to develop a voice, and now that I have it I am not going to be silent."
    for voice in ("neutral_male", "cheerful_female"):
        ids = tok.build_prompt(text, voice)
        n_aud = sum(1 for t in ids if t == tok.audio_token_id)
        i36, i35 = ids.index(tok.special[NEXT_AUDIO_TEXT]), ids.index(tok.special[REPEAT_AUDIO_TEXT])
        rt = tok.decode(ids[i36 + 1 : i35])
        print(f"[tok] {voice:16s} {len(ids):4d} ids | {n_aud} placeholders | round-trip exact: {rt == text}")

    # Byte-level edge cases: non-ASCII, emoji, digits, repeated whitespace.
    for probe in ("Café déjà vu", "1234 numbers", "emoji 🎤 test", "  leading and   inner spaces"):
        ids = tok.encode(probe)
        ok = tok.decode(ids) == probe
        print(f"[tok] round-trip {'OK ' if ok else 'FAIL'} {len(ids):3d} ids  {probe!r}")


if __name__ == "__main__":
    main()
