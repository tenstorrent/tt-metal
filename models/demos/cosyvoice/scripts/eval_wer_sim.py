# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Score synthesised speech for intelligibility (WER/CER) and speaker similarity.

This is the measurement path for R9 of the bring-up plan -- "WER < 3.0, speaker
similarity > 60". It is deliberately framework-agnostic: it scores a directory of
wavs plus the results.json that produced them, so the identical command scores the
PyTorch reference and the TTNN port, and the two are directly comparable.

RUN THIS IN THE CosyVoice VENV:

    $COSYVOICE_ENV/bin/python eval_wer_sim.py --run-dir <dir>
    $COSYVOICE_ENV/bin/python eval_wer_sim.py --run-dir <ttnn> --baseline <ref>

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

# Whisper's tokenizer knows 100 languages INCLUDING Cantonese ("yue"), but only
# large-v3 was trained with the 100-token set -- medium and large-v2 have 99 and
# would reject it. Falling back to "zh" scores Cantonese as Mandarin, which does
# not work: it produced 52-87% CER on audio that is perfectly intelligible.
# That is a measurement artifact, so the fallback is recorded per-utterance
# (`asr_lang_fallback`) rather than quietly folded into the aggregate.
WHISPER_LANG = {"zh": "zh", "en": "en", "ja": "ja", "yue": "yue", "ko": "ko"}
WHISPER_LANG_FALLBACK = {"yue": "zh"}

SIM_MODEL = "microsoft/wavlm-base-plus-sv"


# --------------------------------------------------------------------------
# text normalisation + edit distance
# --------------------------------------------------------------------------
_PUNCT = re.compile(r"[\s\.,!?;:\"'`~@#$%^&*()\[\]{}<>/\\|+=_-–—…、。，！？；：" "''（）《》【】〈〉「」『』·]+")


def _to_simplified(text: str) -> str:
    """Fold Traditional Chinese to Simplified before scoring.

    Whisper chooses its script freely and is not consistent about it -- the same
    Chinese audio transcribes as 以後 on one run and 以后 on another. Against a
    Simplified reference every such pair is counted as a substitution, so the CER
    ends up reporting *which script the ASR happened to pick* rather than what was
    said.

    Measured: the PyTorch reference's own audio for the golden utterance scored
    **35.71 % CER** because Whisper emitted Traditional, while TTNN audio for the
    same sentence scored 7.14 % purely because it drew Simplified. Two implementations
    that sound the same, a 5x apparent difference, and the model had nothing to do
    with it. Folding first drops the reference to 14.29 % and TTNN to 7.14 %, both
    now genuine homophone confusions (得/的, 哟/呦).

    Degrades to a no-op rather than failing if `zhconv` is absent, since it is a
    scoring refinement and not a dependency of the port.
    """
    try:
        import zhconv
    except ImportError:
        return text
    return zhconv.convert(text, "zh-cn")


def normalize(text: str, lang: str) -> list[str]:
    """Lowercase, strip punctuation, then split into the scoring unit."""
    text = unicodedata.normalize("NFKC", text).lower()
    if lang in CER_LANGS:
        # Script folding must happen before the split, and applies to zh and yue
        # both -- Cantonese transcripts come back in either script too.
        text = _to_simplified(text)
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
        # 100 => trained with the Cantonese token; 99 => it must be faked as zh.
        self.num_languages = getattr(self.model, "num_languages", 99)
        self.fallbacks: dict[str, str] = {}

    def language_for(self, lang: str) -> tuple[str, bool]:
        """Returns (whisper language code, whether we had to substitute)."""
        want = WHISPER_LANG.get(lang, lang)
        if self.num_languages < 100 and want in WHISPER_LANG_FALLBACK:
            sub = WHISPER_LANG_FALLBACK[want]
            if want not in self.fallbacks:
                print(
                    f"[asr ] !! whisper '{self.name}' has {self.num_languages} language "
                    f"tokens and cannot score '{want}'; falling back to '{sub}'. "
                    f"Those numbers are NOT comparable -- use --asr-model large-v3.",
                    flush=True,
                )
                self.fallbacks[want] = sub
            return sub, True
        return want, False

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

    def transcribe(self, wav_path: str, lang: str) -> tuple[str, bool]:
        code, fell_back = self.language_for(lang)
        out = self.model.transcribe(
            self._load_16k_mono(wav_path),
            language=code,
            fp16=False,
            temperature=0.0,
            beam_size=None,
        )
        return out["text"].strip(), fell_back


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

        hyp, fell_back = asr.transcribe(wav, lang)
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
                "asr_lang_fallback": fell_back,
            }
        )

        prompt = prompt_for(r)
        if prompt and os.path.exists(prompt):
            entry["sim"] = round(100.0 * sim.score(wav, prompt), 2)
            entry["sim_model"] = sim.name
            entry["sim_reference"] = os.path.basename(prompt)
            if campplus:
                entry["sim_campplus_diagnostic"] = round(100.0 * campplus.score(wav, prompt), 2)
        scored.append(entry)

        simtxt = f"{entry['sim']:6.2f}" if "sim" in entry else "   n/a"
        flag = "  [asr-lang substituted]" if fell_back else ""
        print(f"  {r['mode']:<14}{lang:<4} {unit.upper()} {rate:6.2f}%  SIM {simtxt}   {hyp[:52]}{flag}", flush=True)

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
    def gate(value, ok):
        """A metric that was not measured is 'n/a', NOT a failure.

        Scoring an instruct-only run has no speaker similarity to report -- those
        modes have no reference wav -- and reporting that as FAIL would look like a
        quality regression where there is simply no measurement.
        """
        return "n/a" if value is None else bool(ok(value))

    agg["gates"] = {
        "R9_wer_lt_3.0": gate(agg["wer_percent_en"], lambda v: v < 3.0),
        "R9_sim_gt_60": gate(agg["sim_mean"], lambda v: v > 60.0),
        "R8_rtf_lt_0.5": gate(agg["rtf_mean"], lambda v: v < 0.5),
        "R8_tok_per_s_ge_30": gate(agg["tokens_per_second_mean"], lambda v: v >= 30.0),
    }
    # Any utterance whose ASR language had to be substituted is not comparable;
    # surface the count rather than letting it disappear into the corpus average.
    subs = [r for r in ok if r.get("asr_lang_fallback")]
    if subs:
        agg["asr_lang_substituted"] = sorted({r["lang"] for r in subs})
        agg["asr_lang_substituted_n"] = len(subs)
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
    ap.add_argument("--cosyvoice-root", default=os.environ.get("COSYVOICE_REPO", "/mnt/CosyVoice"))
    ap.add_argument("--no-campplus", action="store_true")
    ap.add_argument("--out", default=None, help="default <run-dir>/scores.json")
    args = ap.parse_args()

    root = args.cosyvoice_root
    # Which reference wav each mode should be compared against for speaker similarity.
    zero_shot_prompt = os.path.join(root, "asset", "zero_shot_prompt.wav")
    xling_prompt = os.path.join(root, "asset", "cross_lingual_prompt.wav")

    def prompt_for(r):
        """The reference wav a synthesised utterance should be compared against.

        SIM is a *voice-cloning* metric: it asks whether the output sounds like a
        given reference speaker. That question only has an answer for the modes that
        take a reference wav.

        SFT and instruct speak as a checkpoint-internal speaker with no reference
        recording at all, so they get None. An earlier version anchored them to the
        same mode's Chinese utterance -- which is wrong, because the language sweep
        deliberately uses a DIFFERENT speaker per language (英文女 for en, 中文女 for
        zh, ...). That compared two different voices and duly reported ~39-49
        similarity, a number that looked like a quality problem and was actually a
        harness bug. Scoring SFT/instruct properly needs two utterances from the
        same speaker; until the sweep generates those, reporting nothing is the
        honest answer.
        """
        if r["mode"] == "zero_shot":
            return zero_shot_prompt
        if r["mode"] == "cross_lingual":
            return xling_prompt
        return None

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
        print(f"    {k:<28} {'n/a  (not measured)' if v == 'n/a' else ('PASS' if v else 'FAIL')}")

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
