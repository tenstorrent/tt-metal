"""fp32 CPU reference audio for a side-by-side against the device, on the current build's prompts.

STATUS 3: "one developer saying 'ok' is not a substitute for a MOS-style eval ... with a
side-by-side against the fp32 reference". This produces that side-by-side for two cases:
  case 1  cheerful_female, 101 frames -- prosody, the thing WER cannot measure
  case 6  de_male seed 1 -- the only utterance of 90 with audible clicks
The device already clicks LESS than the reference here (60 vs 69), so the pair answers "is our
port worse" by ear, not just by counter.
"""
import json, os, torch, wave
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
for ci, seed in ((1, 0), (6, 1)):
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
          f"{a['click_count']}, peak {a['peak']:.3f} -> {os.path.basename(dst)}")
