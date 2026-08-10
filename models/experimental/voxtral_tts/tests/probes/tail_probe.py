"""Does residual-as-bias make rare bad utterances MORE LIKELY? The tail, not the mean.

8 seeds gave BEFORE 0/8 below MOS 3.0 and AFTER 1/8, with the AFTER median HIGHER (4.427 vs
4.336). That is exactly the regime n=8 cannot resolve: one rare event either way is unremarkable.
The mean is the wrong statistic here -- a single catastrophic utterance is user-visible where a
0.2 shift in a mean is not.

So: many seeds, on the three prompts that actually produce low scores (11 emoji, 8 Italian with
heavy ellipsis, 4 one-word), counting FAILURES rather than averaging.

One process per arm, model loaded once -- 6.21's warning about process history still applies, but
it applies IDENTICALLY to both arms, so the paired comparison holds. Wavs go to a temp tag so the
quality set is not polluted.
"""
import json, os, sys, wave
import numpy as np, torch, ttnn
from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
from models.experimental.voxtral_tts.scripts.generate_quality_set import artifacts, frame_budget
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline, open_device

HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
OUT = os.path.join(HERE, "generated", "tailprobe")
CASES, NSEED = (11, 8, 4), 24
arm = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
dev = open_device()
try:
    pipe = TtVoxtralPipeline(dev, max_seq_len=2048)
    fx = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"]
    rows = []
    for ci in CASES:
        c = fx[ci]
        e = pref.build_inputs_embeds(torch.tensor(c["ids"], dtype=torch.long),
                                     pref.load_voice(c["voice"]), pipe.wb)
        for s in range(NSEED):
            f, _, _ = pipe.generate(e, max_frames=frame_budget(c["text"]), seed=s, verbose=False)
            wav = pipe.decode(f)
            p = os.path.join(OUT, f"{arm}_c{ci}_s{s}.wav")
            x = np.clip(wav.reshape(-1).numpy(), -1, 1)
            with wave.open(p, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
                w.writeframes((x * 32767).astype(np.int16).tobytes())
            a = artifacts(wav)
            rows.append({"case": ci, "seed": s, "wav": p, "frames": int(f.shape[0]),
                         "clicks": a["click_count"], "voice": c["voice"]})
            print(f"  {arm} case {ci} seed {s}: {f.shape[0]} frames", flush=True)
    json.dump(rows, open(os.path.join(OUT, f"{arm}.json"), "w"), indent=2)
finally:
    ttnn.close_device(dev)
