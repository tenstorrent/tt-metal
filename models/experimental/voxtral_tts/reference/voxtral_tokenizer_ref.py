# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tekken tokenizer + Voxtral-TTS prompt assembly, reimplemented from `tekken.json`.

Replaces `mistral_common` (which needs the whole mistral-common dep tree) with the same trick
the XTTS-v2 reference used for coqui's tokenizer: read the vocab file directly and reimplement
the algorithm. Validated by EXACT TOKEN-ID MATCH against `mistral_common`'s
`encode_speech_request` output — see tests/test_tokenizer_ref.py.

DEPENDENCY NOTE: this is the only file in reference/ that imports anything beyond torch, and it
needs exactly one thing — `regex` (not stdlib `re`). tekken's split pattern uses Unicode
property classes (\\p{L}, \\p{Lu}, \\p{N}, \\p{M} ...) which stdlib `re` cannot parse at all.
Approximating them with ASCII classes would tokenize this English test sentence identically and
then silently diverge on anything accented, which is worse than a small dependency.

FORMAT (tekken.json v7):
  config.pattern              tiktoken-style split regex
  config.default_vocab_size   131072 total ids
  config.default_num_special_tokens  1000 -> ids 0..999 are special, regular ids are rank + 1000
  vocab[]                     150000 entries {rank, token_bytes(base64)}; only the first
                              (131072 - 1000) = 130072 are in the released vocabulary
  special_tokens[]            1000 entries {rank, token_str}
  audio.voice_num_audio_tokens per-voice reference length in FRAMES

PROMPT LAYOUT (reverse-engineered from mistral_common output, then confirmed by round-trip):

    <s>              1
    [BEGIN_AUDIO]    25
    [AUDIO] x N      24     N = voice_num_audio_tokens[voice] -- the voice's reference length;
                            the pipeline substitutes the preset's N rows over these
    [NEXT_AUDIO_TEXT] 36
    <text ids>              tekken BPE of the raw text, no normalization
    [REPEAT_AUDIO_TEXT] 35
    [BEGIN_AUDIO]    25     generation starts after this

Reads naturally: "here is a reference voice; NEXT is the text; REPEAT it; now begin audio."

Run to check against the shipped ground-truth prompts:
    PYTHONPATH=<repo> python models/experimental/voxtral_tts/reference/voxtral_tokenizer_ref.py
"""

import base64
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEKKEN = os.environ.get("VOXTRAL_TEKKEN", os.path.join(_HERE, "weights", "tekken.json"))

# Special ids, resolved by NAME from tekken.json rather than hard-coded (asserted in __init__).
BOS = "<s>"
BEGIN_AUDIO = "[BEGIN_AUDIO]"
AUDIO = "[AUDIO]"
NEXT_AUDIO_TEXT = "[NEXT_AUDIO_TEXT]"
REPEAT_AUDIO_TEXT = "[REPEAT_AUDIO_TEXT]"


def _bpe(ranks, piece):
    """Classic tiktoken byte-pair merge: repeatedly merge the adjacent pair with the LOWEST rank.

    Pieces are short (the split regex keeps them to roughly a word), so the naive O(n^2) scan is
    fine and matches the reference implementation's result exactly."""
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
        import regex  # only dependency beyond stdlib; see module docstring

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"tekken.json not found: {path}\n"
                "hf download mistralai/Voxtral-4B-TTS-2603 tekken.json --local-dir "
                "models/experimental/voxtral_tts/reference/weights"
            )
        with open(path) as f:
            d = json.load(f)
        cfg = d["config"]
        self.n_special = cfg["default_num_special_tokens"]
        self.vocab_size = cfg["default_vocab_size"]
        self.pattern = regex.compile(cfg["pattern"])

        # Only the first (vocab_size - n_special) vocab entries are in the released vocabulary;
        # tekken.json ships 150000 but the model's embedding table is 131072 wide.
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
        """Raw text -> token ids. NO normalization: tekken is byte-level and case/space
        sensitive, and mistral_common does not pre-clean TTS input either."""
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
    print(f"[tok] audio_token_id={tok.audio_token_id} "
          f"| frames: min {min(tok.voice_frames.values())} max {max(tok.voice_frames.values())}")

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
