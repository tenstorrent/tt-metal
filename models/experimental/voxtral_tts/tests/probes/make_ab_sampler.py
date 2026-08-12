"""Pair device audio against the fp32 CPU reference in ONE file, for an A/B listening pass.

    python tests/probes/make_ab_sampler.py [DEVICE_TAG]      # default: hs_hand

STATUS 3 asks for exactly this and has never had it: "one developer saying 'ok' is not a
substitute for a MOS-style eval ... with a side-by-side against the fp32 reference." make_ref_ab.py
renders the reference clips; this stitches each one against its device twin, reference SECOND so
the ear hears the port first and the ground truth as the answer.

THE REFERENCE CLIPS DO NOT GO STALE, which is why this is cheap. voxtral_pipeline_ref is pure CPU
fp32 and deterministic per (text, voice, seed), so a reference wav rendered at any commit is still
byte-identical to what today's reference would produce. Only the DEVICE side moves. That is also
what makes the comparison meaningful: the reference is a fixed target, not a second implementation
drifting alongside ours.

LEVELS ARE MATCHED, and 3.2 is why: matched pairs there differed by 11-16% on some cases, "enough
to bias a casual comparison" -- a louder clip simply sounds better. Both sides are peak-normalised
to a common target and the ORIGINAL peaks are printed, so a level difference is reported rather
than heard. Nothing else is altered.
"""
import json, os, sys, wave
import numpy as np

HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
GEN = os.path.join(HERE, "generated")
SR, GAP, TARGET = 24000, 0.7, 0.5

def read(p):
    with wave.open(p, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768

def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "hs_hand"
    fx = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"]
    # ONE order list, shared with make_sampler.py, so the three samplers cannot drift apart
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_sampler import ORDER
    pairs = ORDER
    gap = np.zeros(int(SR * GAP), dtype=np.float32)
    out, idx, t = [], [], 0.0
    for ci, seed, note in pairs:
        voice = fx[ci]["voice"]
        dev = os.path.join(GEN, f"case{ci}_{voice}{tag}s{seed}.wav")
        ref = os.path.join(GEN, f"case{ci}_{voice}_FP32REF_s{seed}.wav")
        if not (os.path.exists(dev) and os.path.exists(ref)):
            print(f"  SKIP case {ci} seed {seed}: "
                  f"{'device' if not os.path.exists(dev) else 'reference'} wav missing")
            continue
        for which, p in (("DEVICE", dev), ("fp32 REF", ref)):
            a = read(p)
            peak = float(np.abs(a).max())
            out.append(a / max(peak, 1e-9) * TARGET); out.append(gap)
            dur = len(a) / SR
            idx.append((t, t + dur, ci, seed, which, voice, peak, dur, note))
            t += dur + GAP
    if not out:
        print("  nothing to build -- run tests/probes/make_ref_ab.py first")
        return
    y = np.concatenate(out)
    dst = os.path.join(GEN, f"SAMPLER_AB_{tag}_vs_fp32.wav")
    with wave.open(dst, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    print(f"  wrote {dst}  ({len(y)/SR:.1f}s, {len(idx)} clips, {len(idx)//2} pairs)")
    print(f"  levels matched to peak {TARGET}; original peaks printed below\n")
    print(f"  {'start':>7} {'end':>7} {'case':>5} {'seed':>4} {'which':<9} {'peak':>6} {'dur':>6}  what")
    for a, b, ci, sd, wch, v, pk, dur, note in idx:
        print(f"  {a:>6.1f}s {b:>6.1f}s {ci:>5} {sd:>4} {wch:<9} {pk:>6.3f} {dur:>5.1f}s  "
              f"{note if wch == 'DEVICE' else ''}")

if __name__ == "__main__":
    main()
