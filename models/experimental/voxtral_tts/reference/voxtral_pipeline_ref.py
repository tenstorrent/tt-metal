# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end CPU reference pipeline for Voxtral TTS: prompt ids + voice preset -> 24 kHz WAV.

Chains the three reference blocks and is the golden the TTNN pipeline must reproduce. All three
blocks are ours (torch-only); nothing here imports vllm / mistral_common / einops.

    prompt_ids ──┬──[tok_embeddings]────────────┐
    voice preset ┴──[substitute at id==24]──────┴─► inputs_embeds [1,P,3072]
                                                      │
                    ┌── [Block 1: AR backbone] ◄───────┘   prefill -> h, then one step per frame
                    │            │
                    │            ▼ h [1,3072]
                    │   [Block 2: flow matching, 7 Euler steps + CFG] ─► 37 codes
                    │            │
                    └── embed_frame(37 codes) ◄┘        (feedback; stop on [END_AUDIO])
                                 │
                        codes [1,37,T] ─► [Block 3: codec decoder] ─► wav [1,1,T*1920] @ 24 kHz

TOKENIZER IS OUT OF SCOPE (host-side, exactly as in the XTTS-v2 reference). The prompt layout
produced by mistral_common's `encode_speech_request` is:

    [1]  BOS
    [25] begin_audio
    [24] x 169          <- audio placeholders, REPLACED IN ORDER by the voice preset's 169 rows
    [35] <text ids> [35]
    [25] begin_audio    <- generation starts after this

so `--prompt-ids` takes a JSON dump of `tokenized.tokens` (see `scripts/dump_prompt_ids.py`).
The only thing this pipeline needs from the tokenizer is that rule: every `audio_token_id`
position consumes one row of the preset, everything else is a `tok_embeddings` lookup.

THE PLACEHOLDER COUNT IS VOICE-SPECIFIC. The tokenizer emits one `audio_token_id` per frame of
*that voice's* reference clip, and the presets differ a lot — ar_male 67 frames (5.4 s) up to
neutral_female 218 (17.4 s). So a prompt dumped for one voice cannot be reused with another;
re-run the dump script per voice. `--voice` therefore only overrides which preset FILE is
loaded, and `build_inputs_embeds` asserts the counts agree rather than silently misaligning the
conditioning.

VOICE CLONING FROM YOUR OWN AUDIO IS NOT POSSIBLE with the public checkpoint (the codec encoder
is not shipped) — only the 20 presets. See PROVENANCE.md finding 1.

Run:
    PYTHONPATH=<repo> python models/experimental/voxtral_tts/reference/voxtral_pipeline_ref.py \
        --prompt-ids prompt_ids.json --voice neutral_male --max-frames 150
"""

import argparse
import json
import os
import time
import wave

import torch

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as backbone
from models.experimental.voxtral_tts.reference import voxtral_codec_ref as codec
from models.experimental.voxtral_tts.reference import voxtral_flow_ref as flow
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    CFG_ALPHA,
    DEFAULT_CKPT,
    DIM,
    END_AUDIO_ID,
    FRAME_RATE,
    NUM_CODEBOOKS,
    N_DECODING_STEPS,
    SAMPLING_RATE,
    WEIGHTS_DIR,
)

VOICE_DIR = os.path.join(WEIGHTS_DIR, "voice_embedding")
OUT_WAV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "generated", "voxtral_ref.wav")
AUDIO_TOKEN_ID = 24  # from tekken.json special_ids.audio; asserted against the prompt dump
SAMPLES_PER_FRAME = 1920  # 240-sample patch x 8 upsample


def load_prompt(path):
    """JSON dump of mistral_common's tokenized.tokens (+ the special ids it used).

    Kept as an alternative to --text so a prompt produced by the real mistral_common can be
    replayed byte-for-byte; --text uses our own tekken reimplementation, which is validated to
    produce identical ids (tests/test_tokenizer_ref.py)."""
    with open(path) as f:
        d = json.load(f)
    aid = d.get("audio_token_id", AUDIO_TOKEN_ID)
    assert aid == AUDIO_TOKEN_ID, f"prompt dump says audio_token_id={aid}, pipeline assumes {AUDIO_TOKEN_ID}"
    return torch.tensor(d["ids"], dtype=torch.long), d.get("text", ""), d.get("voice")


def build_prompt_from_text(text, voice):
    """text + voice -> prompt ids, using our own tekken reimplementation (no mistral_common)."""
    from models.experimental.voxtral_tts.reference.voxtral_tokenizer_ref import TekkenTokenizer

    tok = TekkenTokenizer()
    assert tok.audio_token_id == AUDIO_TOKEN_ID
    return torch.tensor(tok.build_prompt(text, voice), dtype=torch.long)


def load_voice(name, voice_dir=VOICE_DIR):
    """A preset is [T_ref, 3072] bf16 — reference speech ALREADY embedded into the backbone's
    space (it bypasses both the absent codec encoder and the 37-codebook embedding)."""
    p = os.path.join(voice_dir, f"{name}.pt")
    if not os.path.exists(p):
        avail = sorted(f[:-3] for f in os.listdir(voice_dir)) if os.path.isdir(voice_dir) else []
        raise FileNotFoundError(f"no voice preset {name!r}; available: {avail}")
    return torch.load(p, map_location="cpu", weights_only=False).float()


def build_inputs_embeds(ids, voice, w):
    """Text ids -> tok_embeddings; every `audio_token_id` position consumes the next preset row."""
    mask = ids == AUDIO_TOKEN_ID
    n = int(mask.sum())
    assert n == voice.shape[0], (
        f"prompt has {n} audio placeholders but the preset has {voice.shape[0]} rows. The count is "
        f"voice-specific — re-dump the prompt for THIS voice:\n"
        f"    <venv>/bin/python models/experimental/voxtral_tts/scripts/dump_prompt_ids.py "
        f"--text '...' --voice <name>"
    )
    embeds = w["tok_embeddings"][ids.clamp(min=0)].clone()  # [P, 3072]
    embeds[mask] = voice.to(embeds.dtype)
    return embeds.unsqueeze(0)  # [1, P, 3072]


@torch.no_grad()
def generate(ids, voice, wb, wf, max_frames=150, cfg_alpha=CFG_ALPHA, seed=0, verbose=True):
    """Blocks 1+2 autoregressive loop -> frames [T, 37] (offset applied, EOA excluded)."""
    if seed is not None:
        torch.manual_seed(seed)
    embeds = build_inputs_embeds(ids, voice, wb)
    dec = backbone.IncrementalBackbone(wb)

    t0 = time.perf_counter()
    h = dec.prefill(embeds)  # [1, 1, 3072]
    t_prefill = time.perf_counter() - t0
    if verbose:
        print(f"[pipeline] prefill P={embeds.shape[1]} in {t_prefill:.1f}s")

    frames, t0 = [], time.perf_counter()
    for i in range(max_frames):
        codes = flow.reference_frame(h[:, 0], wf, cfg_alpha=cfg_alpha)  # [1, 37]
        if int(codes[0, 0]) == END_AUDIO_ID:
            if verbose:
                print(f"[pipeline] [END_AUDIO] at frame {i} — natural stop")
            break
        frames.append(codes)
        h = dec.step(backbone.embed_frame(wb, codes[0]))
        if verbose and (i + 1) % 10 == 0:
            el = time.perf_counter() - t0
            print(f"[pipeline]   {i + 1} frames ({(i + 1) / FRAME_RATE:.1f}s audio) "
                  f"| {el / (i + 1):.2f}s/frame")
    else:
        if verbose:
            print(f"[pipeline] hit max_frames={max_frames} without [END_AUDIO]")
    if not frames:
        raise RuntimeError("model emitted [END_AUDIO] on the first frame — nothing to decode")
    return torch.cat(frames, dim=0), t_prefill, time.perf_counter() - t0


def save_wav(wav, path=OUT_WAV, sr=SAMPLING_RATE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a = (wav.detach().reshape(-1).clamp(-1, 1).numpy() * 32767).astype("<i2")
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(a.tobytes())
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--text", default=None, help="raw text (tokenized by our tekken reimplementation)")
    ap.add_argument("--prompt-ids", default=None, help="alternative: JSON dump of mistral_common ids")
    ap.add_argument("--voice", default="neutral_male", help="one of the 20 presets")
    ap.add_argument("--max-frames", type=int, default=750, help="12.5 frames = 1s; model's native cap is ~1500")
    ap.add_argument("--cfg-alpha", type=float, default=CFG_ALPHA)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT_WAV)
    ap.add_argument("--threads", type=int, default=0, help="0 = leave torch default")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    if (args.text is None) == (args.prompt_ids is None):
        ap.error("give exactly one of --text or --prompt-ids")
    if args.text is not None:
        text, voice_name = args.text, args.voice
        ids = build_prompt_from_text(text, voice_name)
    else:
        ids, text, dump_voice = load_prompt(args.prompt_ids)
        voice_name = dump_voice or args.voice
    print(f"[pipeline] text: {text!r}")
    print(f"[pipeline] prompt {len(ids)} ids, {int((ids == AUDIO_TOKEN_ID).sum())} audio placeholders "
          f"| voice {voice_name!r} | {N_DECODING_STEPS} Euler steps, cfg {args.cfg_alpha} "
          f"| threads {torch.get_num_threads()}")

    voice = load_voice(voice_name)
    t0 = time.perf_counter()
    wb = backbone.load_backbone_state(args.ckpt)
    wf = flow.load_flow_state(args.ckpt)
    wc = codec.load_codec_state(args.ckpt)
    print(f"[pipeline] loaded 3 blocks in {time.perf_counter() - t0:.1f}s "
          f"(voice {tuple(voice.shape)})")

    frames, t_prefill, t_gen = generate(ids, voice, wb, wf, args.max_frames, args.cfg_alpha, args.seed)
    del wb, wf  # ~15 GB; free before the codec runs

    t0 = time.perf_counter()
    codes = codec.strip_offset_and_trim(frames)  # [1, 37, T]
    wav = codec.reference_decode(codes, wc)
    t_codec = time.perf_counter() - t0

    secs = wav.shape[-1] / SAMPLING_RATE
    path = save_wav(wav, args.out)
    assert wav.shape[-1] == frames.shape[0] * SAMPLES_PER_FRAME
    print(f"\n[pipeline] {frames.shape[0]} frames -> {tuple(wav.shape)} = {secs:.2f}s @ {SAMPLING_RATE} Hz")
    print(f"[pipeline] peak |x| {wav.abs().max():.3f} | prefill {t_prefill:.1f}s | "
          f"generate {t_gen:.1f}s ({t_gen / frames.shape[0]:.2f}s/frame) | codec {t_codec:.1f}s")
    print(f"[pipeline] -> {path}")


if __name__ == "__main__":
    main()
