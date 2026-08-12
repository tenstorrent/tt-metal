"""fp32 CPU reference audio for a side-by-side against the device, on the current build's prompts.

    python tests/probes/make_ref_ab.py [CASE:SEED ...]      # default: 1:0 6:1

STATUS 3: "one developer saying 'ok' is not a substitute for a MOS-style eval ... with a
side-by-side against the fp32 reference". Renders the reference half; `make_ab_sampler.py` stitches
each clip against its device twin. Defaults:
  case 1  cheerful_female, ~95 frames -- prosody, the thing WER cannot measure
  case 6  de_male seed 1 -- the only utterance of 45 with audible clicks
The device clicks LESS than the reference there (51 vs 69), so the pair answers "is our port worse"
by ear, not just by counter. `2:0` adds the 125-word long-form case, which is where the WER claim
lives and the only clip long enough to judge prosody over -- but it is ~450 frames of CPU fp32 at
~0.9 s/frame, i.e. minutes, where the short ones are seconds.

THESE RENDERS DO NOT GO STALE. The reference is pure CPU fp32 and deterministic per
(text, voice, seed), so a wav rendered at any commit still matches what today's reference produces;
only the device side moves. Re-render only if `reference/` itself changes.
"""
import json, os, sys, torch, wave
import numpy as np
from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref
from models.experimental.voxtral_tts.reference import voxtral_codec_ref as cref
from models.experimental.voxtral_tts.reference import voxtral_flow_ref as fref
from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
from models.experimental.voxtral_tts.scripts.generate_quality_set import artifacts, frame_budget
HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
GEN = os.path.join(HERE, "generated")
fx = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"]
wb, wf, wc = bref.load_backbone_state(), fref.load_flow_state(), cref.load_codec_state()
WANT = [tuple(int(v) for v in a.split(":")) for a in sys.argv[1:]] or [(1, 0), (6, 1)]
for ci, seed in WANT:
    c = fx[ci]
    fr = pref.generate(torch.tensor(c["ids"], dtype=torch.long), pref.load_voice(c["voice"]),
                       wb, wf, max_frames=frame_budget(c["text"]), seed=seed, verbose=False)
    if isinstance(fr, tuple): fr = fr[0]
    wav = cref.reference_decode(cref.strip_offset_and_trim(fr), wc)
    x = np.clip(wav.reshape(-1).numpy(), -1, 1)
    dst = os.path.join(GEN, f"case{ci}_{c['voice']}_FP32REF_s{seed}.wav")
    with wave.open(dst, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes((x * 32767).astype(np.int16).tobytes())
    a = artifacts(wav)
    print(f"  case {ci} seed {seed} fp32 REFERENCE: {fr.shape[0]} frames, clicks "
          f"{a['click_count']}, peak {a['peak']:.3f} -> {os.path.basename(dst)}", flush=True)
    # Record it. These renders cost minutes of CPU and used to leave nothing behind but a wav, so
    # their frame counts and click counts -- the numbers the device is compared AGAINST -- lived
    # only in a terminal. Accumulated, not overwritten: one render per invocation is normal.
    rec = os.path.join(GEN, "results_FP32REF.json")
    rows = json.load(open(rec)) if os.path.exists(rec) else []
    rows = [r for r in rows if (r["case"], r["seed"]) != (ci, seed)]
    rows.append({"case": ci, "seed": seed, "voice": c["voice"], "frames": int(fr.shape[0]),
                 "click_count": a["click_count"], "peak": round(float(a["peak"]), 4),
                 "wav": os.path.basename(dst)})
    json.dump(sorted(rows, key=lambda r: (r["case"], r["seed"])), open(rec, "w"), indent=1)
