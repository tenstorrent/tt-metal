# SPDX-License-Identifier: Apache-2.0
"""Phase C (CPU): emit the final wav. Uses OUR TT HiFi-GAN output (Block 4) if phase B
produced it (vocoder_wav_tt.pt) — then this phase only writes the file. Otherwise falls
back to coqui's HiFi-GAN on the TT gpt_latents + d-vector."""
import argparse, os

os.environ["COQUI_TOS_AGREED"] = "1"

import torch
import transformers.pytorch_utils as _ptu

if not hasattr(_ptu, "isin_mps_friendly"):
    _ptu.isin_mps_friendly = lambda elements, test_elements: torch.isin(elements, test_elements)

import soundfile as sf

CKPT = os.environ.get("XTTS_CKPT_DIR", "/localdev/acicovic/xtts_ref")  # coqui checkpoint dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Prefer OUR TT HiFi-GAN waveform (Block 4 on device, produced in phase B).
    tt_wav = os.path.join(args.work, "vocoder_wav_tt.pt")
    if os.path.exists(tt_wav):
        wav = torch.load(tt_wav).cpu().squeeze().numpy()
        sf.write(args.out, wav, 24000)
        print(f"[C] vocoder = TT HiFi-GAN (Block 4); wrote wav -> {args.out}")
        return

    # Fallback: coqui HiFi-GAN on the TT gpt_latents + d-vector.
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.load_json(os.path.join(CKPT, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=CKPT, eval=True, use_deepspeed=False)
    model.eval()

    gpt_latents_tt = torch.load(os.path.join(args.work, "gpt_latents_tt.pt"))
    spk_tt = os.path.join(args.work, "speaker_embedding_tt.pt")
    speaker_embedding = torch.load(
        spk_tt if os.path.exists(spk_tt) else os.path.join(args.work, "speaker_embedding.pt")
    )
    print(
        f"[C] vocoder = coqui HiFi-GAN (Block 4 not on TT yet); d-vector from "
        f"{'TT (Block 2)' if os.path.exists(spk_tt) else 'coqui CPU'}"
    )
    with torch.inference_mode():
        wav = model.hifigan_decoder(gpt_latents_tt, g=speaker_embedding).cpu().squeeze().numpy()
    sf.write(args.out, wav, 24000)
    print(f"[C] wrote wav -> {args.out}")


if __name__ == "__main__":
    main()
