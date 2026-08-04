# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Score synthesised speech for intelligibility (WER/CER) and speaker similarity.

This is the measurement path for R9 of the bring-up plan -- "WER < 3.0, speaker
similarity > 60". It is deliberately framework-agnostic: it scores a directory of
wavs plus the results.json that produced them, so the identical command scores the
PyTorch reference and the TTNN port, and the two are directly comparable.

RUN THIS IN THE CosyVoice VENV:

    /root/tt/cosyvoice_env/bin/python eval_wer_sim.py --run-dir <dir>
    /root/tt/cosyvoice_env/bin/python eval_wer_sim.py --run-dir <ttnn> --baseline <ref>

Protocol, and where it departs from Seed-TTS Eval
-------------------------------------------------
Seed-TTS Eval scores English WER with Whisper-large-v3 and Chinese CER with
Paraformer-zh, and computes SIM-o from a WavLM-large model fine-tuned for speaker
verification (distributed via a Google Drive link).

Two deliberate substitutions, both documented rather than silent:

  1. Whisper is used for every language, so Chinese/Cantonese/Japanese CER comes
     from Whisper rather than Paraformer. Whisper is weaker on zh than Paraformer,
     so the CER reported here is pessimistic -- an upper bound on the true error.
  2. SIM uses microsoft/wavlm-base-plus-sv (WavLMForXVector, on the HF hub) rather
     than the Drive-hosted wavlm-large SV checkpoint. Same family, same cosine-of-
     x-vectors formulation, fetched reproducibly.

Both are held fixed across reference and TTNN runs, which is what makes the
comparison sound even where the absolute number is not the published one. The
campplus cosine is reported alongside as a diagnostic ONLY: CosyVoice conditions
on campplus embeddings, so scoring with campplus is self-referential and inflated.
It is never the reported SIM.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata

import numpy as np

# Languages scored by character error rate rather than word error rate: they are
# not whitespace-delimited, so word-level scoring is meaningless.
CER_LANGS = {"zh", "yue", "ja", "ko"}
WHISPER_LANG = {"zh": "zh", "en": "en", "ja": "ja", "yue": "zh", "ko": "ko"}

SIM_MODEL = "microsoft/wavlm-base-plus-sv"


# --------------------------------------------------------------------------
# text normalisation + edit distance
# --------------------------------------------------------------------------
_PUNCT = re.compile(r"[\s\.,!?;:\"'`~@#$%^&*()\[\]{}<>/\\|+=_-–—…、。，！？；：" "''（）《》【】〈〉「」『』·]+")


def normalize(text: str, lang: str) -> list[str]:
    """Lowercase, strip punctuation, then split into the scoring unit."""
    text = unicodedata.normalize("NFKC", text).lower()
    if lang in CER_LANGS:
        # characters, with all separators and punctuation removed
        return [c for c in _PUNCT.sub("", text) if c.strip()]
    return [w for w in _PUNCT.sub(" ", text).split() if w]


def edit_distance(ref: list, hyp: list) -> tuple[int, int, int, int]:
    """Levenshtein with operation counts. Returns (distance, sub, ins, del)."""
    n, m = len(ref), len(hyp)
    # dp[j] holds (cost, sub, ins, del) for the current row
    prev = [(j, 0, j, 0) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, 0, i)] + [None] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur[j] = prev[j - 1]
            else:
                sub = (prev[j - 1][0] + 1, prev[j - 1][1] + 1, prev[j - 1][2], prev[j - 1][3])
                dele = (prev[j][0] + 1, prev[j][1], prev[j][2], prev[j][3] + 1)
                ins = (cur[j - 1][0] + 1, cur[j - 1][1], cur[j - 1][2] + 1, cur[j - 1][3])
                cur[j] = min(sub, dele, ins, key=lambda t: t[0])
        prev = cur
    return prev[m]


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
# Peak RSS observed loading each on CPU fp32, in GB. large-v3 is 1.55 B params and
# genuinely needs ~9 GB resident -- it was OOM-killed on an 11 GB host merely for
# being loaded while a synthesis run was live. Scoring must never share a box with
# generation.
ASR_RSS_GB = {"large-v3": 9.0, "large-v2": 9.0, "large": 9.0, "medium": 4.5, "small": 2.0, "base": 1.0, "tiny": 0.7}


def _check_ram(model_name: str) -> None:
    need = ASR_RSS_GB.get(model_name, 4.0)
    try:
        with open("/proc/meminfo") as fh:
            avail = next(int(l.split()[1]) for l in fh if l.startswith("MemAvailable"))
        avail_gb = avail / 1e6
    except Exception:
        return
    if avail_gb < need:
        raise SystemExit(
            f"whisper '{model_name}' needs ~{need:.1f} GB resident; only "
            f"{avail_gb:.1f} GB available.\n"
            f"Either free memory (synthesis must not be running concurrently) or "
            f"pass --asr-model medium."
        )


class ASR:
    def __init__(self, name: str = "large-v3"):
        import whisper

        _check_ram(name)
        print(f"[asr ] loading whisper {name} (cpu) ...", flush=True)
        self.model = whisper.load_model(name, device="cpu")
        self.name = name

    @staticmethod
    def _load_16k_mono(wav_path: str):
        """Decode to the float32 16 kHz mono array whisper wants.

        whisper.load_audio() shells out to `ffmpeg`, which is a system binary this
        harness has no business requiring -- it is not installed on this host and
        would be one more thing to get right in every container. torchaudio reads
        wav directly, so we hand whisper the array and skip the subprocess.
        """
        import torchaudio

        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(0)  # to mono
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav.contiguous().float().numpy()

    def transcribe(self, wav_path: str, lang: str) -> str:
        out = self.model.transcribe(
            self._load_16k_mono(wav_path),
            language=WHISPER_LANG.get(lang, lang),
            fp16=False,
            temperature=0.0,
            beam_size=None,
        )
        return out["text"].strip()


class SpeakerSim:
    """Cosine similarity of WavLM x-vectors, resampled to the model's 16 kHz."""

    def __init__(self, name: str = SIM_MODEL):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        print(f"[sim ] loading {name} (cpu) ...", flush=True)
        self.torch = torch
        self.fe = AutoFeatureExtractor.from_pretrained(name)
        self.model = WavLMForXVector.from_pretrained(name).eval()
        self.sr = self.fe.sampling_rate
        self.name = name
        self._cache: dict[str, "np.ndarray"] = {}

    def embed(self, wav_path: str):
        if wav_path in self._cache:
            return self._cache[wav_path]
        import torchaudio

        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(0, keepdim=True)  # to mono
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        inputs = self.fe(wav.squeeze(0).numpy(), sampling_rate=self.sr, return_tensors="pt", padding=True)
        with self.torch.no_grad():
            emb = self.model(**inputs).embeddings
        emb = self.torch.nn.functional.normalize(emb, dim=-1).squeeze(0).numpy()
        self._cache[wav_path] = emb
        return emb

    def score(self, a: str, b: str) -> float:
        return float(np.dot(self.embed(a), self.embed(b)))


class CampplusSim:
    """Diagnostic only -- the model conditions on these embeddings."""

    def __init__(self, onnx_path: str):
        import onnxruntime

        opt = onnxruntime.SessionOptions()
        opt.log_severity_level = 3
        self.sess = onnxruntime.InferenceSession(onnx_path, sess_options=opt, providers=["CPUExecutionProvider"])
        self._cache: dict[str, "np.ndarray"] = {}

    def embed(self, wav_path: str):
        if wav_path in self._cache:
            return self._cache[wav_path]
        import torchaudio
        import torchaudio.compliance.kaldi as kaldi

        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        feat = kaldi.fbank(wav, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        emb = self.sess.run(None, {self.sess.get_inputs()[0].name: feat.unsqueeze(0).numpy()})[0].flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        self._cache[wav_path] = emb
        return emb

    def score(self, a: str, b: str) -> float:
        return float(np.dot(self.embed(a), self.embed(b)))


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def score_run(run_dir, asr, sim, campplus, prompt_for):
    with open(os.path.join(run_dir, "results.json")) as fh:
        run = json.load(fh)

    scored = []
    for r in run["results"]:
        if "wav" not in r:
            scored.append(r)
            continue
        wav = os.path.join(run_dir, r["wav"])
        lang = r["lang"]
        unit = "cer" if lang in CER_LANGS else "wer"

        hyp = asr.transcribe(wav, lang)
        ref_u, hyp_u = normalize(r["text"], lang), normalize(hyp, lang)
        dist, s, i, d = edit_distance(ref_u, hyp_u)
        rate = 100.0 * dist / max(1, len(ref_u))

        entry = dict(r)
        entry.update(
            {
                "asr_hypothesis": hyp,
                "unit": unit,
                f"{unit}_percent": round(rate, 2),
                "ref_units": len(ref_u),
                "errors": {"sub": s, "ins": i, "del": d},
            }
        )

        prompt = prompt_for(r)
        if prompt and os.path.exists(prompt):
            entry["sim"] = round(100.0 * sim.score(wav, prompt), 2)
            entry["sim_model"] = sim.name
            if campplus:
                entry["sim_campplus_diagnostic"] = round(100.0 * campplus.score(wav, prompt), 2)
        scored.append(entry)

        print(
            f"  {r['mode']:<14}{lang:<4} {unit.upper()} {rate:6.2f}%"
            f"  SIM {entry.get('sim', float('nan')):6.2f}   {hyp[:52]}",
            flush=True,
        )

    run["scored"] = scored
    run["asr_model"] = asr.name
    run["sim_model"] = sim.name
    return run


def aggregate(run: dict) -> dict:
    ok = [r for r in run["scored"] if "unit" in r]
    if not ok:
        return {}

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(float(np.mean(xs)), 2) if xs else None

    # Corpus-level error rate: total errors / total units, not a mean of ratios.
    tot_err = sum(sum(r["errors"].values()) for r in ok)
    tot_ref = sum(r["ref_units"] for r in ok)
    agg = {
        "n": len(ok),
        "corpus_error_rate_percent": round(100.0 * tot_err / max(1, tot_ref), 2),
        "wer_percent_en": mean([r.get("wer_percent") for r in ok if r["unit"] == "wer"]),
        "cer_percent_cjk": mean([r.get("cer_percent") for r in ok if r["unit"] == "cer"]),
        "sim_mean": mean([r.get("sim") for r in ok]),
        "sim_campplus_diagnostic_mean": mean([r.get("sim_campplus_diagnostic") for r in ok]),
        "rtf_mean": mean([r.get("rtf") for r in ok]),
        "tokens_per_second_mean": mean([r.get("tokens_per_second") for r in ok]),
    }
    # R9 / R8 gates, evaluated but never enforced here -- the pytest perf suite owns
    # enforcement. This is the number, stated plainly.
    agg["gates"] = {
        "R9_wer_lt_3.0": (agg["wer_percent_en"] is not None and agg["wer_percent_en"] < 3.0),
        "R9_sim_gt_60": (agg["sim_mean"] is not None and agg["sim_mean"] > 60.0),
        "R8_rtf_lt_0.5": (agg["rtf_mean"] is not None and agg["rtf_mean"] < 0.5),
        "R8_tok_per_s_ge_30": (agg["tokens_per_second_mean"] is not None and agg["tokens_per_second_mean"] >= 30.0),
    }
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="directory holding results.json + wavs from run_reference.py")
    ap.add_argument("--baseline", default=None, help="a second scored run to diff against (e.g. the PyTorch reference)")
    ap.add_argument(
        "--asr-model",
        default="large-v3",
        help="whisper size; Seed-TTS Eval uses large-v3. 'medium'/'small' iterate faster.",
    )
    ap.add_argument("--cosyvoice-root", default=os.environ.get("COSYVOICE_ROOT", "/root/tt/CosyVoice"))
    ap.add_argument("--no-campplus", action="store_true")
    ap.add_argument("--out", default=None, help="default <run-dir>/scores.json")
    args = ap.parse_args()

    root = args.cosyvoice_root
    # Which reference wav each mode should be compared against for speaker similarity.
    zero_shot_prompt = os.path.join(root, "asset", "zero_shot_prompt.wav")
    xling_prompt = os.path.join(root, "asset", "cross_lingual_prompt.wav")

    def prompt_for(r):
        if r["mode"] == "zero_shot":
            return zero_shot_prompt
        if r["mode"] == "cross_lingual":
            return xling_prompt
        # SFT/instruct speak as a checkpoint-internal speaker with no reference wav.
        # Self-consistency is scored instead: same speaker across languages should
        # cluster, so the zh utterance of that speaker acts as the anchor.
        anchor = os.path.join(args.run_dir, f"{r['mode']}_zh.wav")
        return anchor if os.path.exists(anchor) and r["lang"] != "zh" else None

    asr = ASR(args.asr_model)
    sim = SpeakerSim()
    campplus = None
    if not args.no_campplus:
        p = os.path.join(root, "pretrained_models", "CosyVoice-300M", "campplus.onnx")
        campplus = CampplusSim(p) if os.path.exists(p) else None

    print(f"\nscoring {args.run_dir}")
    run = score_run(args.run_dir, asr, sim, campplus, prompt_for)
    run["aggregate"] = aggregate(run)

    out = args.out or os.path.join(args.run_dir, "scores.json")
    with open(out, "w") as fh:
        json.dump(run, fh, indent=2, ensure_ascii=False)

    a = run["aggregate"]
    print("\n=== aggregate ===")
    for k, v in a.items():
        if k != "gates":
            print(f"  {k:<34} {v}")
    print("  gates:")
    for k, v in a["gates"].items():
        print(f"    {k:<28} {'PASS' if v else 'FAIL'}")

    if args.baseline:
        with open(os.path.join(args.baseline, "scores.json")) as fh:
            base = json.load(fh)["aggregate"]
        print("\n=== vs baseline ===")
        for k in ("corpus_error_rate_percent", "wer_percent_en", "cer_percent_cjk", "sim_mean"):
            if base.get(k) is not None and a.get(k) is not None:
                print(f"  {k:<34} {base[k]:8.2f} -> {a[k]:8.2f}  ({a[k]-base[k]:+.2f})")

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
