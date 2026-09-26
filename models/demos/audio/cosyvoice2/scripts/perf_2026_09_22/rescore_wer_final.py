"""Final WER re-score with a PRECISE contraction fix -- jiwer.ExpandCommonEnglishContractions
also expands bare "'s" to " is" unconditionally, which incorrectly hits possessives too
(measured: "Layton's" -> "Layton is", moving the original single-utterance WER even though
no contraction is involved there). This uses jiwer.SubstituteRegexes with the SAME rule
list minus that one ambiguous rule, so only genuine contractions ("we're", "can't", "let's",
"n't", "'re", "'d", "'ll", "'t", "'ve", "'m") are expanded; possessive "'s" is left alone.

Host-only: reuses the WAVs already on disk (both the four-sentence regression WAVs and the
original single-utterance eval's WAV).

Run: /opt/venv/bin/python rescore_wer_final.py
"""
import os

import jiwer
import numpy as np
import torch
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

# Same rule list as jiwer.ExpandCommonEnglishContractions, minus the ambiguous bare
# 's -> is rule (which also catches possessives, e.g. "Layton's" -> "Layton is").
PRECISE_CONTRACTIONS = jiwer.SubstituteRegexes(
    {
        r"won't": "will not",
        r"can't": "can not",
        r"let's": "let us",
        r"n't": " not",
        r"'re": " are",
        r"'d": " would",
        r"'ll": " will",
        r"'t": " not",
        r"'ve": " have",
        r"'m": " am",
    }
)
NEW_NORM = jiwer.Compose(
    [
        PRECISE_CONTRACTIONS,
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def score(target, hyp):
    old = jiwer.wer(target, hyp, truth_transform=OLD_NORM, hypothesis_transform=OLD_NORM) * 100
    new = jiwer.wer(target, hyp, truth_transform=NEW_NORM, hypothesis_transform=NEW_NORM) * 100
    return old, new


asr = whisper.load_model("base.en")

print("=== this round's four sentences (warm reps, using saved WAVs) ===")
print(f"{'utt':4s} {'rep':5s} {'old WER':>8s} {'new WER':>8s}  hyp")
rows = []
for ui, target_text in TEXTS.items():
    for rep in ("new", "r1", "r2", "r3"):
        path = f"{WAVDIR}/utt{ui}_{rep}.wav"
        if not os.path.exists(path):
            continue
        sr, wav24_i16 = wavfile.read(path)
        assert sr == 24000
        wav24 = torch.from_numpy(wav24_i16.astype(np.float32) / 32767.0).unsqueeze(0)
        w16 = torchaudio.functional.resample(wav24, 24000, 16000).squeeze(0).numpy()
        hyp = asr.transcribe(w16, language="en", fp16=False)["text"].strip()
        wer_old, wer_new = score(target_text, hyp)
        rows.append((ui, rep, wer_old, wer_new, hyp))
        flag = "  <-- changed" if abs(wer_old - wer_new) > 1e-6 else ""
        print(f"{ui:<4d} {rep:5s} {wer_old:7.2f}% {wer_new:7.2f}%  {hyp!r}{flag}")

print("\nper-utterance summary (warm reps only, mean):")
for ui in TEXTS:
    w = [r for r in rows if r[0] == ui and r[1] != "new"]
    if not w:
        continue
    old_m = sum(r[2] for r in w) / len(w)
    new_m = sum(r[3] for r in w) / len(w)
    print(f"  utt {ui}: old {old_m:.2f}%  new {new_m:.2f}%  {'CHANGED' if abs(old_m-new_m) > 1e-6 else 'unchanged'}")

print("\n=== original Stage 1 eval (REF_IDX=0, TGT_IDX=3, fp32/eager -- matches the doc's 4.17% methodology) ===")
orig_target = "He has grave doubts whether sir frederick leighton's work is really greek after all and can discover in it but little of rocky ithaca."
orig_path = f"{OUT_DIR}/original_stage1_synth.wav"
sr, wav24_i16 = wavfile.read(orig_path)
assert sr == 24000
wav24 = torch.from_numpy(wav24_i16.astype(np.float32) / 32767.0).unsqueeze(0)
w16 = torchaudio.functional.resample(wav24, 24000, 16000).squeeze(0).numpy()
orig_hyp = asr.transcribe(w16, language="en", fp16=False)["text"].strip()
old_o, new_o = score(orig_target, orig_hyp)
print(f"target: {orig_target!r}")
print(f"hyp:    {orig_hyp!r}")
print(f"WER old (doc's reported methodology): {old_o:.2f}%  (doc reported 4.17%)")
print(f"WER new (precise contraction fix, possessive 's untouched): {new_o:.2f}%")
print(f"moved: {'YES' if abs(old_o - new_o) > 1e-6 else 'NO'}")
