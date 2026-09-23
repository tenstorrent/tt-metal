"""Re-score this round's four sentences' saved WAVs with a contraction-expansion fix
applied to the WER normalization pipeline (jiwer.ExpandCommonEnglishContractions,
inserted BEFORE lowercase/punctuation removal, since it needs the apostrophe still
present to recognize "we're" as a contraction). Host-only: no device needed, reuses the
WAVs already on disk from the earlier warm-regression run.

Run: /opt/venv/bin/python rescore_wer_fix.py
"""
import os

import jiwer
import numpy as np
import torchaudio
import whisper
from scipy.io import wavfile

OUT_DIR = os.environ.get("OUT_DIR", "/tmp/cosyvoice2_perf_2026_09_22")
WAVDIR = f"{OUT_DIR}/regression_wavs"

TEXTS = {
    0: "Please close the door when you leave.",
    1: "We are going to the park this weekend, and the kids want to bring their bikes and a big picnic lunch.",
    2: "The weather was nice yesterday, so we sat outside for a while and talked about our plans for the summer holidays.",
    3: "My sister called me last night to tell me about her new job. She likes her team, the office is close to her house, and she can finally take the train instead of driving every day.",
}

OLD_NORM = jiwer.Compose(
    [jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()]
)
NEW_NORM = jiwer.Compose(
    [
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

asr = whisper.load_model("base.en")

print(f"{'utt':4s} {'rep':5s} {'old WER':>8s} {'new WER':>8s}  hyp")
rows = []
for ui, target_text in TEXTS.items():
    for rep in ("new", "r1", "r2", "r3"):
        path = f"{WAVDIR}/utt{ui}_{rep}.wav"
        if not os.path.exists(path):
            continue
        sr, wav24_i16 = wavfile.read(path)
        assert sr == 24000
        import torch

        wav24 = torch.from_numpy(wav24_i16.astype(np.float32) / 32767.0).unsqueeze(0)
        w16 = torchaudio.functional.resample(wav24, 24000, 16000).squeeze(0).numpy()
        hyp = asr.transcribe(w16, language="en", fp16=False)["text"].strip()
        wer_old = jiwer.wer(target_text, hyp, truth_transform=OLD_NORM, hypothesis_transform=OLD_NORM) * 100
        wer_new = jiwer.wer(target_text, hyp, truth_transform=NEW_NORM, hypothesis_transform=NEW_NORM) * 100
        rows.append((ui, rep, wer_old, wer_new, hyp))
        flag = "  <-- changed" if abs(wer_old - wer_new) > 1e-6 else ""
        print(f"{ui:<4d} {rep:5s} {wer_old:7.2f}% {wer_new:7.2f}%  {hyp!r}{flag}")

print("\n=== per-utterance summary (warm reps only, mean) ===")
for ui in TEXTS:
    w = [r for r in rows if r[0] == ui and r[1] != "new"]
    if not w:
        continue
    old_m = sum(r[2] for r in w) / len(w)
    new_m = sum(r[3] for r in w) / len(w)
    print(f"utt {ui}: old {old_m:.2f}%  new {new_m:.2f}%  {'CHANGED' if abs(old_m-new_m) > 1e-6 else 'unchanged'}")
