# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generate reference + TT speech and write both to WAV for A/B listening.

Usage (from tt-metal root):
    export VIBEVOICE_MODEL_PATH=$PWD/models/experimental/vibevoice/weights/VibeVoice-1.5B
    python models/experimental/vibevoice/dump_audio.py --out_dir /tmp/vv_audio --max_new_tokens 128

Writes <out_dir>/ref.wav and <out_dir>/tt.wav (24 kHz) and prints a mel-distance metric.
"""
import argparse
import sys
from pathlib import Path

import torch
import ttnn

from models.experimental.vibevoice.common.config import MODEL_PATH, DEFAULT_TXT_PATH, VOICES_DIR
from models.experimental.vibevoice.tt.ttnn_vibevoice_model import TTVibeVoiceModel

_VV_ROOT = Path(__file__).resolve().parent
for _p in (_VV_ROOT / "reference", _VV_ROOT.parent.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SR = 24000


def _voice():
    p = VOICES_DIR / "en-Alice_woman.wav"
    return str(p) if p.is_file() else str(next(VOICES_DIR.glob("*.wav")))


def _write_wav(path, audio_1d):
    import soundfile as sf

    sf.write(path, audio_1d.detach().to(torch.float32).numpy(), SR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/tmp/vv_audio")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--cfg_scale", type=float, default=1.3)
    ap.add_argument("--num_steps", type=int, default=10)
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    with open(DEFAULT_TXT_PATH, encoding="utf-8") as f:
        script = f.read().strip().replace("’", "'")
    processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)
    ref = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, device_map="cpu", attn_implementation="sdpa"
    )
    ref.eval()
    ref.set_ddpm_inference_steps(num_steps=args.num_steps)
    inputs = processor(
        text=[script], voice_samples=[[_voice()]], padding=True, return_tensors="pt", return_attention_mask=True
    )

    with torch.no_grad():
        _, prefill_embeds = ref._process_speech_inputs(inputs["speech_tensors"].to(ref.dtype), inputs["speech_masks"])

    print("[dump_audio] reference generate...")
    torch.manual_seed(0)
    ref_out = ref.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        cfg_scale=args.cfg_scale,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=False,
        is_prefill=True,
        show_progress_bar=False,
    )
    ref_speech = ref_out.speech_outputs[0].to(torch.float32).reshape(-1)
    _write_wav(str(Path(args.out_dir) / "ref.wav"), ref_speech)
    print(f"[dump_audio] wrote ref.wav  ({ref_speech.numel()/SR:.2f}s)")

    mesh = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        tt_model = TTVibeVoiceModel.from_checkpoint(
            mesh, MODEL_PATH, cfg_scale=args.cfg_scale, num_diffusion_steps=args.num_steps
        )
        tt_model.set_speech_scale_bias(ref.model.speech_scaling_factor.item(), ref.model.speech_bias_factor.item())
        torch.manual_seed(0)
        with torch.no_grad():
            ref._process_speech_inputs(inputs["speech_tensors"].to(ref.dtype), inputs["speech_masks"])
        print("[dump_audio] TT generate...")
        tt_out = tt_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            speech_input_mask=inputs["speech_input_mask"],
            prefill_speech_embeds=prefill_embeds,
            tokenizer=processor.tokenizer,
            cfg_scale=args.cfg_scale,
            num_diffusion_steps=args.num_steps,
            max_new_tokens=args.max_new_tokens,
        )
        tt_speech = tt_out.speech_outputs[0].to(torch.float32).reshape(-1)
    finally:
        ttnn.close_device(mesh)
    _write_wav(str(Path(args.out_dir) / "tt.wav"), tt_speech)
    print(f"[dump_audio] wrote tt.wav   ({tt_speech.numel()/SR:.2f}s)")

    # Objective perceptual-ish metric: log-mel L1 distance (alignment-tolerant-ish)
    try:
        import torchaudio

        mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=1024, hop_length=256, n_mels=80)
        n = min(ref_speech.numel(), tt_speech.numel())
        lm_ref = torch.log(mel(ref_speech[:n]) + 1e-5)
        lm_tt = torch.log(mel(tt_speech[:n]) + 1e-5)
        print(f"[dump_audio] log-mel L1 distance = {(lm_ref - lm_tt).abs().mean().item():.4f}  (lower=closer)")
    except Exception as e:
        print(f"[dump_audio] (torchaudio mel metric skipped: {e})")
    print(f"[dump_audio] DONE. Listen to {args.out_dir}/ref.wav and {args.out_dir}/tt.wav")


if __name__ == "__main__":
    main()
