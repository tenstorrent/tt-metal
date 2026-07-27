# SPDX-License-Identifier: Apache-2.0
"""CPU per-component perf for the XTTS-v2 pipeline, timed on the SAME sub-networks that TT
accelerates (so the numbers line up with bench_warm.py's TT blocks):
  Block1 = gpt.conditioning_encoder + gpt.conditioning_perceiver
  Block2 = hifigan_decoder.speaker_encoder
  Block3 = gpt.gpt_inference transformer forwards (per-token decode) + one-shot prefix
  Block4 = hifigan_decoder.waveform_decoder
Front-end mel/STFT is excluded (it stays CPU in both the CPU and TT pipelines)."""
import argparse, os, time

os.environ["COQUI_TOS_AGREED"] = "1"
import torch
import transformers.pytorch_utils as _ptu

if not hasattr(_ptu, "isin_mps_friendly"):
    _ptu.isin_mps_friendly = lambda elements, test_elements: torch.isin(elements, test_elements)

import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

CKPT = os.environ.get("XTTS_CKPT_DIR", "/localdev/acicovic/xtts_ref")  # coqui checkpoint dir


class Acc:
    """Accumulating wall-time hook: sums forward wall time and counts calls for a module."""

    def __init__(self, mod):
        self.t = 0.0
        self.n = 0
        self._t0 = None
        mod.register_forward_pre_hook(self._pre)
        mod.register_forward_hook(self._post)

    def _pre(self, m, a):
        self._t0 = time.perf_counter()

    def _post(self, m, a, o):
        self.t += time.perf_counter() - self._t0
        self.n += 1

    def reset(self):
        self.t = 0.0
        self.n = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count())

    cfg = XttsConfig()
    cfg.load_json(os.path.join(CKPT, "config.json"))
    model = Xtts.init_from_config(cfg)
    model.load_checkpoint(cfg, checkpoint_dir=CKPT, eval=True, use_deepspeed=False)
    model.eval()

    raw = torch.load(args.ref, map_location="cpu", weights_only=False)
    wav = raw if torch.is_tensor(raw) else torch.as_tensor(np.asarray(raw))
    wav = wav.squeeze()
    audio = wav.float().reshape(1, -1)

    enc = Acc(model.gpt.conditioning_encoder)
    perc = Acc(model.gpt.conditioning_perceiver)
    spk = Acc(model.hifigan_decoder.speaker_encoder)
    gpt = Acc(model.gpt.gpt_inference)
    voc = Acc(model.hifigan_decoder.waveform_decoder)

    res = {}

    # Block1: conditioning encoder + perceiver (warm-averaged)
    N1 = 10
    with torch.no_grad():
        model.get_gpt_cond_latents(audio, args.sr)  # warm
        enc.reset()
        perc.reset()
        for _ in range(N1):
            model.get_gpt_cond_latents(audio, args.sr)
    res["Block1 cond + perceiver"] = (enc.t + perc.t) / N1 * 1000.0

    # Block2: ResNet speaker encoder (warm-averaged)
    N2 = 10
    with torch.no_grad():
        model.get_speaker_embedding(audio, args.sr)  # warm
        spk.reset()
        for _ in range(N2):
            model.get_speaker_embedding(audio, args.sr)
    res["Block2 speaker encoder"] = spk.t / N2 * 1000.0

    # Block3 + Block4: one real inference. gpt_inference forwards = decode steps; waveform_decoder = vocode.
    gpt.reset()
    voc.reset()
    gpt_cond = model.get_gpt_cond_latents(audio, args.sr)
    spk_emb = model.get_speaker_embedding(audio, args.sr)
    t_infer0 = time.perf_counter()
    with torch.no_grad():
        model.inference(args.text, args.lang, gpt_cond, spk_emb, do_sample=True)
    t_infer = (time.perf_counter() - t_infer0) * 1000.0

    res["Block3 GPT decode (per token)"] = gpt.t / max(gpt.n, 1) * 1000.0
    res["Block3 GPT total (%d tok)" % gpt.n] = gpt.t * 1000.0
    res["Block4 HiFi-GAN vocoder"] = voc.t / max(voc.n, 1) * 1000.0

    print("\n==================== CPU per-component perf (%d threads) ====================" % torch.get_num_threads())
    for k, v in res.items():
        print(f"  {k:34s} {v:10.2f} ms")
    print(f"  {'(full model.inference wall)':34s} {t_infer:10.2f} ms   gpt_fwd_calls={gpt.n} voc_calls={voc.n}")


if __name__ == "__main__":
    main()
