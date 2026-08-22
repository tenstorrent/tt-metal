# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Run the PyTorch CosyVoice-300M reference across all 4 modes and 5 languages.

This is the baseline the TTNN port is measured against: it produces the audio
that `eval_wer_sim.py` scores, and the tok/s + RTF numbers that make the Stage 1
perf gates (R8: >= 30 tok/s, RTF < 0.5) meaningful rather than absolute.

RUN THIS IN THE CosyVoice VENV:

    PYTHONPATH=$COSYVOICE_REPO:$COSYVOICE_REPO/third_party/Matcha-TTS \
    $COSYVOICE_ENV/bin/python run_reference.py --out <dir>

Mode / checkpoint mapping (CosyVoice-300M alone cannot do all four):

    sft            CosyVoice-300M-SFT       needs spk2info.pt for named speakers
    zero_shot      CosyVoice-300M           reference wav + its transcript
    cross_lingual  CosyVoice-300M           reference wav, <|lang|>-tagged text
    instruct       CosyVoice-300M-Instruct  named speaker + instruction string
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

DEFAULT_COSYVOICE = os.environ.get("COSYVOICE_REPO", "/mnt/CosyVoice")
SEED = 1986

# One sentence per language. Kept short so a CPU run finishes in minutes;
# `--long` swaps in the fuller sentences used by the upstream demo.
TEXTS = {
    "zh": "收到好友从远方寄来的生日礼物，那份意外的惊喜让我心中充满了甜蜜的快乐。",
    "en": "The quick brown fox jumps over the lazy dog while the morning sun rises slowly.",
    "ja": "遠くの友人から届いた誕生日プレゼントに、思いがけない驚きと喜びを感じました。",
    "yue": "收到朋友由遠方寄嚟嘅生日禮物，嗰份意外嘅驚喜令我心入面充滿咗快樂。",
    "ko": "멀리 있는 친구가 보내준 생일 선물은 뜻밖의 놀라움과 기쁨을 안겨 주었습니다.",
}

# Cross-lingual mode requires the language tag as a literal prefix; the tokenizer
# maps <|zh|> <|en|> <|ja|> <|yue|> <|ko|> to special ids.
XLING_TAG = {"zh": "<|zh|>", "en": "<|en|>", "ja": "<|ja|>", "yue": "<|yue|>", "ko": "<|ko|>"}

# Preferred SFT/instruct speaker per language, resolved against the checkpoint's
# actual spk2info at runtime (falls back to the first available speaker).
SPK_PREF = {
    "zh": ["中文女", "中文男"],
    "en": ["英文女", "英文男"],
    "ja": ["日语男"],
    "yue": ["粤语女"],
    "ko": ["韩语女"],
}

# CosyVoice-1 instruct takes a CHARACTER / STYLE DESCRIPTION, not a directive.
#
# frontend_instruct() puts this string into the LLM's `prompt_text` slot, and
# TransformerLM.inference does `text = concat([prompt_text, text])` -- so the
# instruct text is literally prepended to what the model reads, with
# <|endofprompt|> marking the boundary. A description-shaped prefix conditions the
# voice; a DIRECTIVE-shaped one gets read aloud.
#
# That is not hypothetical. An earlier version of this file used CosyVoice-2
# instruct2 phrasing ("用四川话说这句话，语气轻快活泼。"), and the model dutifully
# spoke the instruction before the sentence:
#     "用四川话说这句话,语气轻快活泼。收到好友从远方寄来的生日礼物,..."
# -- 42% CER against the intended text, and entirely self-inflicted. Directive
# phrasing belongs to inference_instruct2 (CosyVoice-2), not inference_instruct.
#
# English descriptions are used for every language because that is the form the
# 300M-Instruct checkpoint was trained on; the spoken text stays in-language.
INSTRUCTS = {
    "zh": "A gentle female speaker with normal pitch, slow speaking rate, " "and a warm, happy emotion.<|endofprompt|>",
    "en": "Theo 'Crimson', is a fiery, passionate rebel leader. Fights with fervor "
    "for justice, but struggles with impulsiveness.<|endofprompt|>",
    "ja": "A calm male speaker with low pitch, steady speaking rate, "
    "and a neutral, reassuring emotion.<|endofprompt|>",
    "yue": "A cheerful female speaker with bright pitch, brisk speaking rate, " "and a lively emotion.<|endofprompt|>",
    "ko": "A soft female speaker with normal pitch, gentle speaking rate, "
    "and a kind, comforting emotion.<|endofprompt|>",
}

ZERO_SHOT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"


class TokenCounter:
    """Counts LLM speech tokens and times the first one.

    The reference pops its per-utterance token dict at the end of tts(), so the
    count has to be intercepted while the run is live.
    """

    def __init__(self, model):
        self.model = model
        self.reset()
        self._orig = model.token2wav

        def wrapper(token, *a, **kw):
            self.tokens = max(self.tokens, int(token.shape[1]))
            return self._orig(token, *a, **kw)

        model.token2wav = wrapper

    def reset(self):
        self.tokens = 0

    def close(self):
        self.model.token2wav = self._orig


def resolve_spk(available: list[str], lang: str) -> str | None:
    for want in SPK_PREF.get(lang, []):
        if want in available:
            return want
    return available[0] if available else None


def run_one(cv, mode, lang, root, out_dir, seed, stream=False):
    import torchaudio
    from cosyvoice.utils.common import set_all_random_seed

    text = TEXTS[lang]
    counter = TokenCounter(cv.model)
    set_all_random_seed(seed)

    spk = None
    if mode in ("sft", "instruct"):
        spk = resolve_spk(cv.list_available_spks(), lang)
        if spk is None:
            return {"mode": mode, "lang": lang, "skipped": "no speakers in spk2info"}

    t0 = time.time()
    if mode == "sft":
        gen = cv.inference_sft(text, spk, stream=stream)
    elif mode == "zero_shot":
        gen = cv.inference_zero_shot(
            text, ZERO_SHOT_PROMPT_TEXT, os.path.join(root, "asset", "zero_shot_prompt.wav"), stream=stream
        )
    elif mode == "cross_lingual":
        gen = cv.inference_cross_lingual(
            XLING_TAG[lang] + text, os.path.join(root, "asset", "cross_lingual_prompt.wav"), stream=stream
        )
    elif mode == "instruct":
        gen = cv.inference_instruct(text, spk, INSTRUCTS[lang], stream=stream)
    else:
        raise ValueError(mode)

    chunks, first_chunk_at = [], None
    for out in gen:
        if first_chunk_at is None:
            first_chunk_at = time.time() - t0
        chunks.append(out["tts_speech"])
    wall = time.time() - t0
    counter.close()

    wav = torch.concat(chunks, dim=1)
    dur = wav.shape[1] / cv.sample_rate
    tag = f"{mode}_{lang}" + ("_stream" if stream else "")
    path = os.path.join(out_dir, f"{tag}.wav")
    torchaudio.save(path, wav, cv.sample_rate)

    return {
        "mode": mode,
        "lang": lang,
        "stream": stream,
        "speaker": spk,
        "text": text,
        "wav": os.path.basename(path),
        "audio_seconds": round(dur, 3),
        "wall_seconds": round(wall, 2),
        "rtf": round(wall / dur, 3),
        "speech_tokens": counter.tokens,
        "tokens_per_second": round(counter.tokens / wall, 2) if wall else None,
        "first_chunk_seconds": round(first_chunk_at, 2) if first_chunk_at else None,
        "chunks": len(chunks),
        "peak": round(wav.abs().max().item(), 4),
        "rms": round(wav.pow(2).mean().sqrt().item(), 5),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosyvoice-root", default=DEFAULT_COSYVOICE)
    # Repo-relative, matching gen_golden.py and export_weights.py. This defaulted to an
    # absolute path in the author's own working tree, which `os.makedirs` would have
    # created on anyone else's machine.
    ap.add_argument("--out", default=None)
    ap.add_argument("--modes", default="sft,zero_shot,cross_lingual,instruct")
    ap.add_argument("--langs", default="zh,en,ja,yue,ko")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--stream", action="store_true", help="also run each mode in streaming mode (R3 baseline)")
    args = ap.parse_args()

    root = args.cosyvoice_root
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "third_party", "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import CosyVoice

    torch.set_num_threads(args.threads)
    args.out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "reference")
    os.makedirs(args.out, exist_ok=True)

    ckpt_for = {
        "sft": "CosyVoice-300M-SFT",
        "zero_shot": "CosyVoice-300M",
        "cross_lingual": "CosyVoice-300M",
        "instruct": "CosyVoice-300M-Instruct",
    }
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]

    results, loaded, cv = [], None, None
    # Group by checkpoint so each 1.7 GB load happens once.
    for ckpt in ["CosyVoice-300M", "CosyVoice-300M-SFT", "CosyVoice-300M-Instruct"]:
        todo = [m for m in modes if ckpt_for[m] == ckpt]
        if not todo:
            continue
        model_dir = os.path.join(root, "pretrained_models", ckpt)
        if not os.path.isdir(model_dir):
            print(f"!! {model_dir} missing, skipping {todo}")
            continue
        print(f"\n=== loading {ckpt}")
        t0 = time.time()
        cv = CosyVoice(model_dir, load_jit=False, load_trt=False, fp16=False)
        loaded = ckpt
        print(f"    loaded in {time.time()-t0:.1f}s; speakers: {cv.list_available_spks()}")
        for mode in todo:
            for lang in langs:
                for stream in [False, True] if args.stream else [False]:
                    label = f"{mode}/{lang}" + ("/stream" if stream else "")
                    print(f"  -> {label} ...", end="", flush=True)
                    try:
                        r = run_one(cv, mode, lang, root, args.out, args.seed, stream)
                        r["checkpoint"] = loaded
                        results.append(r)
                        if "skipped" in r:
                            print(f" SKIPPED ({r['skipped']})")
                        else:
                            print(
                                f" {r['audio_seconds']}s audio, {r['wall_seconds']}s wall, "
                                f"RTF {r['rtf']}, {r['speech_tokens']} tok "
                                f"({r['tokens_per_second']} tok/s)"
                            )
                    except Exception as e:  # keep going; one bad mode must not lose the run
                        print(f" FAILED: {type(e).__name__}: {e}")
                        results.append(
                            {
                                "mode": mode,
                                "lang": lang,
                                "stream": stream,
                                "checkpoint": loaded,
                                "error": f"{type(e).__name__}: {e}",
                            }
                        )
        del cv

    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": "cpu",
        "threads": args.threads,
        "torch": torch.__version__,
        "seed": args.seed,
        "results": results,
    }
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    ok = [r for r in results if "error" not in r and "skipped" not in r]
    print(f"\n{len(ok)}/{len(results)} runs produced audio -> {args.out}")
    if ok:
        print(
            f"  median RTF {sorted(r['rtf'] for r in ok)[len(ok)//2]:.2f}"
            f"   median tok/s {sorted(r['tokens_per_second'] for r in ok)[len(ok)//2]:.1f}"
        )
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
