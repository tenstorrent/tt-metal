"""Case 6 clicks: ours, or the model's? The only audio defect in 90 generated utterances.

`artifacts()` counts discontinuities (|step| > 0.5 at 24 kHz) and across all 90 utterances of the
15x3 A/B ONLY case 6 ("Gruesse aus Muenchen - die Strasse ist schoen.", de_male, 8 words, 2.6 s)
ever registers one -- 48 on base seed 1, 60 on the program-config arm seed 1, 0-1 on seeds 0 and 2.
It is in BOTH arms equally, so it is not a regression from 6.52. But "not a regression" is not
"not a bug", and this is the one thing in the whole quality set that would be audible.

The question a shipping decision needs: does the FP32 CPU REFERENCE click on the same prompt and
seed? If it does, this is the model and no amount of device work removes it. If only the device
does, we have a real defect and it is localised to one short German utterance.

Runs the whole pipeline on host, no device at all -- Block 1 + Block 2 + codec in fp32.
"""
import json, os, sys, torch
from models.experimental.voxtral_tts.reference import voxtral_flow_ref as fref
from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref
from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
from models.experimental.voxtral_tts.scripts.generate_quality_set import artifacts
HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
case = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"][6]
print(f"  case 6: {case['voice']}  {case['text']!r}")
wb, wf = bref.load_backbone_state(), fref.load_flow_state()
from models.experimental.voxtral_tts.reference import voxtral_codec_ref as _c
_wc = _c.load_codec_state()
for seed in (1, 0):
    frames = pref.generate(torch.tensor(case["ids"], dtype=torch.long),
                           pref.load_voice(case["voice"]), wb, wf,
                           max_frames=48, seed=seed, verbose=False)
    if isinstance(frames, tuple):
        frames = frames[0]
    from models.experimental.voxtral_tts.reference import voxtral_codec_ref as cref
    wav = cref.reference_decode(cref.strip_offset_and_trim(frames), _wc)
    a = artifacts(wav)
    print(f"  fp32 REFERENCE seed {seed}: frames {frames.shape[0]:>3}  clicks {a['click_count']:>3}"
          f"  peak {a['peak']:.3f}  silent {a['silent_%']:.1f}%")
print("  device, same prompt: seed 1 -> 48 clicks (base) / 60 (prg); seed 0 -> 0 / 1")
