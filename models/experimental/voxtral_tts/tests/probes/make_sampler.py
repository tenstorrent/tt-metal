"""Build a listening sampler from a generated run's wavs, plus an index of what is what.

    python tests/probes/make_sampler.py [TAG]        # default: shipcheck
    python tests/probes/make_sampler.py FP32REF      # the fp32 CPU reference twin

TAG is `generate_quality_set.py --tag`, and this reads `results{TAG}s{SEED}.json` plus the wavs
beside it, so any tagged run can be turned into a sampler. `_prg_` reproduces the §6.58 one.

`FP32REF` builds the same clips in the same order from `make_ref_ab.py`'s renders, so the two
files are comparable CLIP FOR CLIP by ear. They will not be the same length: the reference is a
different implementation and free-running generation is chaotic, so it picks its own frame counts
(§6.7). A duration difference is not a defect — the reference generates the 125-word paragraph in
469 frames against the device's 451, and STATUS §3 records that as agreement within 2%.

WHY IT IS TAGGED. It used to hardcode the `_prg` arm of the §6.52 15x3 A/B, so it could only ever
rebuild that sampler -- which is how `SAMPLER_p150_HEAD.wav` came to sit next to a HEAD that had
moved twice underneath it (§6.65 traced the loop, §6.67 resharded the norm, §6.72 changed the head
split; all three moved codes and frame counts). A sampler is only evidence about the build that
produced it.

Order puts both long-form English cases first (they carry the WER number), then prosody, then the
languages. Case 6 appears twice on purpose -- it is the only utterance in 45 with any click, and
the fp32 reference clicks MORE on the same seed (69 vs 51), so it is a model property; the point
is to hear how bad it is.

THREE FIXTURE PROMPTS ARE DELIBERATELY ADVERSARIAL and are labelled as such below. They are not a
port defect and should not be listened to as one: STATUS §3.2 records the fp32 CPU reference
collapsing into a repetition loop on the emoji text (6257% WER) and producing comparable nonsense
on the symbol run. This index used to name them by VOICE alone ("casual male", "neutral female",
"Dutch"), which invites exactly the wrong conclusion from whoever is listening.
"""
import json, os, sys, wave
import numpy as np

HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
GEN = os.path.join(HERE, "generated")
SR, GAP = 24000, 0.6

# (case, seed, note). A negative case id used to mean "same case, seed 1"; the seed is explicit now.
ORDER = [(2, 0, "long-form English #1, 125 words - the WER case"),
         (3, 0, "long-form English #2"),
         (1, 0, "prosody / cheerful"),
         (0, 0, "neutral male"),
         (11, 0, "ADVERSARIAL: emoji + CAPS + MiXeD cAsE -- the fp32 reference collapses here too"),
         (10, 0, "ADVERSARIAL: digit run + symbol run -- it vocalises the symbols; no reference"),
         (5, 0, "French"),
         (6, 0, "German seed 0 - clean, 1 click"),
         (6, 1, "German seed 1 - 51 CLICKS on device, the worst of 45; fp32 ref clicks 69"),
         (7, 0, "Spanish"),
         (8, 0, "Italian - 6 words, 37-44% silence, the lowest MOS of the set (2.58-2.70)"),
         (9, 0, "Portuguese"),
         (12, 0, "Hindi"),
         (13, 0, "Arabic"),
         (14, 0, "ADVERSARIAL: literal Tab\\t and \\n newline handling")]


def read(p):
    with wave.open(p, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "shipcheck"
    ref = tag.upper() == "FP32REF"
    fx = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"]

    if ref:
        rec = os.path.join(GEN, "results_FP32REF.json")
        rows = json.load(open(rec)) if os.path.exists(rec) else []
        res = {}
        for r in rows:
            res.setdefault(r["seed"], {})[r["case"]] = r
        wav_name = lambda ci, sd: f"case{ci}_{fx[ci]['voice']}_FP32REF_s{sd}.wav"
        dst = os.path.join(GEN, "SAMPLER_FP32REF.wav")
    else:
        # keyed by case, per seed -- the seed-1 lookup used to index the raw LIST by case number,
        # which is only correct while every case is present and in order.
        res = {s: {r["case"]: r for r in json.load(open(os.path.join(GEN, f"results{tag}s{s}.json")))}
               for s in sorted({sd for _, sd, _ in ORDER})}
        wav_name = lambda ci, sd: f"case{ci}_{fx[ci]['voice']}{tag}s{sd}.wav"
        dst = os.path.join(GEN, f"SAMPLER_{tag}.wav")

    gap = np.zeros(int(SR * GAP), dtype=np.int16)
    out, idx, t, missing = [], [], 0.0, 0
    for ci, seed, note in ORDER:
        p = os.path.join(GEN, wav_name(ci, seed))
        if not os.path.exists(p):
            print(f"  MISSING {os.path.basename(p)}")
            missing += 1
            continue
        a = read(p)
        out.append(a); out.append(gap)
        dur = len(a) / SR
        # a reference render may predate the JSON record; report rather than crash
        cl = res.get(seed, {}).get(ci, {}).get("click_count", "?")
        idx.append((t, t + dur, ci, seed, fx[ci]["voice"], note, cl, fx[ci]["text"][:60]))
        t += dur + GAP
    if not out:
        print("  nothing to build")
        return
    y = np.concatenate(out)
    with wave.open(dst, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(y.tobytes())
    print(f"  wrote {dst}  ({len(y)/SR:.1f}s, {len(idx)} of {len(ORDER)} clips"
          f"{f', {missing} MISSING' if missing else ''})\n")
    print(f"  {'start':>7} {'end':>7} {'case':>5} {'seed':>4} {'voice':<16} {'clicks':>6}  what")
    for a, b, ci, sd, v, note, cl, txt in idx:
        print(f"  {a:>6.1f}s {b:>6.1f}s {ci:>5} {sd:>4} {v:<16} {str(cl):>6}  {note}")
        print(f"  {'':>21} {txt!r}")


if __name__ == "__main__":
    main()
