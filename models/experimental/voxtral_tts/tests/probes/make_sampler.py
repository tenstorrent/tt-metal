"""Build a listening sampler from a generated run's wavs, plus an index of what is what.

    python tests/probes/make_sampler.py [TAG]        # default: shipcheck

TAG is `generate_quality_set.py --tag`, and this reads `results{TAG}s{SEED}.json` plus the wavs
beside it, so any tagged run can be turned into a sampler. `_prg_` reproduces the §6.58 one.

WHY IT IS TAGGED NOW. It used to hardcode the `_prg` arm of the §6.52 15x3 A/B, so it could only
ever rebuild that sampler -- which is how `SAMPLER_p150_HEAD.wav` came to sit next to a HEAD that
had moved twice underneath it (§6.65 traced the loop, §6.67 resharded the norm, both moving codes
and frame counts). A sampler is only evidence about the build that produced it.

Order puts both long-form English cases first (they carry the WER number), then prosody, then the
languages. Case 6 appears twice on purpose -- it is the only utterance in 45 with any click, and
the fp32 reference clicks MORE on the same seed (69 vs 60), so it is a model property; the point
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

def read(p):
    with wave.open(p, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "shipcheck"
    fx = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"]
    # keyed by case, per seed -- the seed-1 lookup used to index the raw LIST by case number,
    # which is only correct while every case is present and in order.
    res = {s: {r["case"]: r for r in json.load(open(os.path.join(GEN, f"results{tag}s{s}.json")))}
           for s in (0, 1)}
    order = [(2, "long-form English #1, 125 words - the WER case"),
             (3, "long-form English #2"),
             (1, "prosody / cheerful"),
             (0, "neutral male"),
             (11, "ADVERSARIAL: emoji + CAPS + MiXeD cAsE -- the fp32 reference collapses here too"),
             (10, "ADVERSARIAL: digit run + symbol run -- it vocalises the symbols; no defined reference"),
             (5, "French"), (6, "German seed 0 - clean, 1 click"),
             (-6, "German seed 1 - 51 CLICKS, the worst of all 45; fp32 ref clicks 69 here"),
             (7, "Spanish"),
             (8, "Italian - 6 words, 37-44% silence, the lowest MOS of the set (2.58-2.70)"),
             (9, "Portuguese"), (12, "Hindi"), (13, "Arabic"),
             (14, "ADVERSARIAL: literal Tab\\t and \\n newline handling")]
    gap = np.zeros(int(SR * GAP), dtype=np.int16)
    out, idx, t = [], [], 0.0
    for ci, note in order:
        seed = 1 if ci < 0 else 0            # negative case id = same case, seed 1
        ci = abs(ci)
        p = os.path.join(GEN, f"case{ci}_{fx[ci]['voice']}{tag}s{seed}.wav")
        if not os.path.exists(p):
            print(f"  MISSING {p}")
            continue
        a = read(p)
        out.append(a); out.append(gap)
        dur = len(a) / SR
        idx.append((t, t + dur, ci, seed, fx[ci]["voice"], note,
                    res[seed][ci]["click_count"], fx[ci]["text"][:60]))
        t += dur + GAP
    y = np.concatenate(out)
    dst = os.path.join(GEN, f"SAMPLER_{tag}.wav")
    with wave.open(dst, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(y.tobytes())
    print(f"  wrote {dst}  ({len(y)/SR:.1f}s, {len(idx)} clips)\n")
    print(f"  {'start':>7} {'end':>7} {'case':>5} {'seed':>4} {'voice':<16} {'clicks':>6}  what")
    for a, b, ci, sd, v, note, cl, txt in idx:
        print(f"  {a:>6.1f}s {b:>6.1f}s {ci:>5} {sd:>4} {v:<16} {cl:>6}  {note}")
        print(f"  {'':>21} {txt!r}")

if __name__ == "__main__":
    main()
