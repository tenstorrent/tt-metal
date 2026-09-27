# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Score a run directory for intelligibility (WER) and speaker similarity, and compare two runs.

RUN IN THE REFERENCE VENV (see requirements-reference*.txt and scripts/reference_env.py):

    LIBRISPEECH_ROOT=<dir containing LibriSpeech/> COSYVOICE2_REPO=<upstream checkout> \\
        $COSYVOICE2_REF_ENV/bin/python eval_wer_sim.py --run-dir <ttnn run> [--baseline <reference run>]

A run directory is what `scripts/run_reference.py` or `demo/demo.py` wrote: wavs plus a `results.json` carrying
each case's corpus entry. The same command scores the PyTorch reference and the TTNN port, so the two are
comparable by construction. Scores go to `<run-dir>/scores.json`. With `--baseline` (a scored run), a per-utterance
and corpus-level comparison is printed as a Markdown table.

**Protocol** (this is CosyVoice1's scorer, models/experimental/cosyvoice/scripts/eval_wer_sim.py, adapted):
- **WER:** Whisper large-v3 on CPU, greedy (temperature 0), language given.
  - The reference text is the case's text (for LibriSpeech, its real transcript).
  - Normalization: NFKC, lowercase, punctuation stripped, then word-level edit distance. Chinese, Japanese and
    Korean are scored per character (CER).
  - Corpus-level WER is total errors over total reference words, not a mean of per-utterance rates.
- **Speaker similarity:** `microsoft/wavlm-base-plus-sv` (`WavLMForXVector`, revision pinned) x-vectors,
  cosine x 100, between the generated wav and the case's prompt wav, both at 16 kHz.
  - This is not the paper's SIM model (a WavLM-large SV checkpoint), so compare TT with the PyTorch reference
    scored here, never with the paper's figure.
  - The CAM++ cosine is reported as a diagnostic only. The model conditions on CAM++ embeddings, so that score
    is self-referential.
- **Audio** is read with soundfile, like the reference's own `load_wav` shim: torchaudio >= 2.9 needs TorchCodec
  to load audio. It is resampled with torchaudio.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference_env  # noqa: E402

ASR_MODEL = "large-v3"
SIM_MODEL = "microsoft/wavlm-base-plus-sv"
# pinned, so scores are reproducible and the loaded config is the one checked
SIM_REVISION = "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"
CER_LANGS = {"zh", "yue", "ja", "ko"}

# CosyVoice1's punctuation class, hyphen escaped (unescaped, `_-–` is a range that swallows a-z)
_PUNCT = re.compile(r"[\s\.,!?;:\"'`~@#$%^&*()\[\]{}<>/\\|+=_\-–—…、。，！？；：" "''（）《》【】〈〉「」『』·]+")


def normalize(text: str, lang: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    if lang in CER_LANGS:
        return [c for c in _PUNCT.sub("", text) if c.strip()]
    return [w for w in _PUNCT.sub(" ", text).split() if w]


def edit_distance(ref: list, hyp: list) -> tuple[int, int, int, int]:
    """Levenshtein distance with operation counts: (distance, substitutions, insertions, deletions)."""
    prev = [(j, 0, j, 0) for j in range(len(hyp) + 1)]
    for i in range(1, len(ref) + 1):
        cur = [(i, 0, 0, i)] + [None] * len(hyp)
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur[j] = prev[j - 1]
            else:
                sub = (prev[j - 1][0] + 1, prev[j - 1][1] + 1, prev[j - 1][2], prev[j - 1][3])
                dele = (prev[j][0] + 1, prev[j][1], prev[j][2], prev[j][3] + 1)
                ins = (cur[j - 1][0] + 1, cur[j - 1][1], cur[j - 1][2] + 1, cur[j - 1][3])
                cur[j] = min(sub, dele, ins, key=lambda t: t[0])
        prev = cur
    return prev[-1]


def load_16k_mono(path: str):
    """float32 torch tensor [T] at 16 kHz."""
    import soundfile
    import torch
    import torchaudio

    data, sr = soundfile.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T.copy()).mean(dim=0)
    return torchaudio.functional.resample(wav, sr, 16000) if sr != 16000 else wav


class ASR:
    def __init__(self, name: str = ASR_MODEL):
        import whisper

        print(f"[asr] loading whisper {name} (cpu)", flush=True)
        self.model = whisper.load_model(name, device="cpu")
        self.name = name

    def transcribe(self, wav_path: str, lang: str) -> str:
        audio = load_16k_mono(wav_path).numpy()
        return self.model.transcribe(audio, language=lang, fp16=False, temperature=0.0, beam_size=None)["text"].strip()


class SpeakerSim:
    def __init__(self, name: str = SIM_MODEL):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        print(f"[sim] loading {name} (cpu)", flush=True)
        self.torch = torch
        self.fe = AutoFeatureExtractor.from_pretrained(name, revision=SIM_REVISION)
        self.model = WavLMForXVector.from_pretrained(name, revision=SIM_REVISION).eval()
        self.name = name
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, path: str) -> np.ndarray:
        if path not in self._cache:
            inputs = self.fe(load_16k_mono(path).numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
            with self.torch.no_grad():
                emb = self.model(**inputs).embeddings
            self._cache[path] = self.torch.nn.functional.normalize(emb, dim=-1).squeeze(0).numpy()
        return self._cache[path]

    def score(self, a: str, b: str) -> float:
        return 100.0 * float(np.dot(self.embed(a), self.embed(b)))


class CampplusSim:
    """Diagnostic only: CosyVoice2 conditions on these embeddings."""

    def __init__(self, onnx_path: str):
        import onnxruntime

        opt = onnxruntime.SessionOptions()
        opt.log_severity_level = 3
        self.sess = onnxruntime.InferenceSession(onnx_path, sess_options=opt, providers=["CPUExecutionProvider"])
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, path: str) -> np.ndarray:
        if path not in self._cache:
            import torchaudio.compliance.kaldi as kaldi

            feat = kaldi.fbank(load_16k_mono(path).unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            emb = self.sess.run(None, {self.sess.get_inputs()[0].name: feat.unsqueeze(0).numpy()})[0].flatten()
            self._cache[path] = emb / (np.linalg.norm(emb) + 1e-9)
        return self._cache[path]

    def score(self, a: str, b: str) -> float:
        return 100.0 * float(np.dot(self.embed(a), self.embed(b)))


def prompt_wav(result: dict) -> str | None:
    """The case's prompt wav: the path the run recorded, else resolved like scripts/prepare_inputs.py does."""
    if result.get("prompt_wav_abs") and os.path.exists(result["prompt_wav_abs"]):
        return result["prompt_wav_abs"]
    env = "COSYVOICE2_REPO" if result.get("set") == "cosyvoice1_parity" else "LIBRISPEECH_ROOT"
    base = os.environ.get(env)
    return os.path.join(base, result["prompt_wav"]) if base and result.get("prompt_wav") else None


def score_run(run_dir: str, asr: ASR, sim: SpeakerSim, campplus: CampplusSim | None) -> dict:
    with open(os.path.join(run_dir, "results.json")) as fh:
        run = json.load(fh)
    scored = []
    for r in run["results"]:
        lang = r["lang"]
        unit = "cer" if lang in CER_LANGS else "wer"
        wav = os.path.join(run_dir, r["wav"])
        hyp = asr.transcribe(wav, lang)
        ref_u, hyp_u = normalize(r["text"], lang), normalize(hyp, lang)
        dist, s, i, d = edit_distance(ref_u, hyp_u)
        entry = {
            **r,
            "asr_hypothesis": hyp,
            "unit": unit,
            "error_rate_percent": round(100.0 * dist / max(1, len(ref_u)), 2),
            "ref_units": len(ref_u),
            "errors": {"sub": s, "ins": i, "del": d},
        }
        prompt = prompt_wav(r)
        if prompt and os.path.exists(prompt):
            entry["sim"] = round(sim.score(wav, prompt), 2)
            if campplus is not None:
                entry["sim_campplus_diagnostic"] = round(campplus.score(wav, prompt), 2)
        scored.append(entry)
        print(
            f"  {r['case_id']:<34} {unit.upper()} {entry['error_rate_percent']:6.2f}%  "
            f"SIM {entry.get('sim', float('nan')):6.2f}   {hyp[:60]}",
            flush=True,
        )
    run["scored"] = scored
    run["scoring"] = {"asr_model": f"whisper {asr.name}", "sim_model": sim.name, "sim_scale": "cosine x 100"}
    run["aggregate"] = aggregate(scored)
    return run


def aggregate(scored: list[dict]) -> dict:
    words = [r for r in scored if r["unit"] == "wer"]
    sims = [r["sim"] for r in scored if "sim" in r]
    camp = [r["sim_campplus_diagnostic"] for r in scored if "sim_campplus_diagnostic" in r]
    errs = sum(sum(r["errors"].values()) for r in words)
    refs = sum(r["ref_units"] for r in words)
    return {
        "n": len(scored),
        "corpus_wer_percent": round(100.0 * errs / max(1, refs), 2) if words else None,
        "corpus_word_errors": errs,
        "corpus_ref_words": refs,
        "sim_mean": round(float(np.mean(sims)), 2) if sims else None,
        "sim_campplus_diagnostic_mean": round(float(np.mean(camp)), 2) if camp else None,
    }


def comparison(run: dict, base: dict) -> str:
    """Per-utterance and corpus-level Markdown table over the cases both runs scored."""
    mine = {r["case_id"]: r for r in run["scored"]}
    theirs = {r["case_id"]: r for r in base["scored"]}
    common = [c for c in theirs if c in mine]
    lines = [
        f"| case | words | WER % {base['backend']} | WER % {run['backend']} | SIM {base['backend']} | "
        f"SIM {run['backend']} | audio s {base['backend']} / {run['backend']} |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in common:
        a, b = theirs[c], mine[c]
        lines.append(
            f"| {c} | {a['ref_units']} | {a['error_rate_percent']:.2f} | {b['error_rate_percent']:.2f} | "
            f"{a.get('sim', float('nan')):.2f} | {b.get('sim', float('nan')):.2f} | "
            f"{a.get('audio_s', float('nan')):.2f} / {b.get('audio_s', float('nan')):.2f} |"
        )
    agg_a, agg_b = aggregate([theirs[c] for c in common]), aggregate([mine[c] for c in common])
    lines.append(
        f"| **corpus ({len(common)} utterances)** | {agg_a['corpus_ref_words']} | "
        f"**{agg_a['corpus_wer_percent']:.2f}** ({agg_a['corpus_word_errors']} errors) | "
        f"**{agg_b['corpus_wer_percent']:.2f}** ({agg_b['corpus_word_errors']} errors) | "
        f"**{agg_a['sim_mean']:.2f}** | **{agg_b['sim_mean']:.2f}** | |"
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--baseline", default=None, help="a run directory already scored (its scores.json)")
    ap.add_argument("--asr-model", default=ASR_MODEL)
    ap.add_argument("--no-campplus", action="store_true")
    args = ap.parse_args()

    campplus = None
    if not args.no_campplus:
        path = os.path.join(reference_env.model_dir(), "campplus.onnx")
        campplus = CampplusSim(path) if os.path.exists(path) else None
    run = score_run(args.run_dir, ASR(args.asr_model), SpeakerSim(), campplus)
    out = os.path.join(args.run_dir, "scores.json")
    with open(out, "w") as fh:
        json.dump(run, fh, indent=2, ensure_ascii=False)
    print(f"\n{args.run_dir}: {json.dumps(run['aggregate'])}\nwrote {out}")
    if args.baseline:
        with open(os.path.join(args.baseline, "scores.json")) as fh:
            base = json.load(fh)
        print("\n" + comparison(run, base))
    return 0


if __name__ == "__main__":
    sys.exit(main())
