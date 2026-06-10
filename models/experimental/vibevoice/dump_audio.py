# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generate reference + TT speech and write both to WAV for A/B listening.

Usage (from tt-metal root):
    python models/experimental/vibevoice/dump_audio.py --out_dir /tmp/vv_audio

Writes <out_dir>/ref.wav and <out_dir>/tt.wav (24 kHz) and prints a mel-distance metric.
By default generation runs until EOS (no AR step cap). Use --max_new_tokens to limit length.
"""
import argparse
import sys
from pathlib import Path

import torch
import ttnn

from models.experimental.vibevoice.common.config import VOICES_DIR
from models.experimental.vibevoice.common.model_utils import ensure_model_weights
from models.experimental.vibevoice.common.resource_utils import ensure_demo_resources, load_script
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
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Cap autoregressive steps (default: None = until EOS). "
        "1p_vibevoice.txt needs ~300 steps (~39s audio); 128 stops at ~17s mid-script.",
    )
    ap.add_argument("--cfg_scale", type=float, default=1.3)
    ap.add_argument("--num_steps", type=int, default=10)
    ap.add_argument(
        "--match-ref-tokens",
        action="store_true",
        help="Replay HuggingFace AR tokens on TT (same duration/frame count as ref.wav). "
        "Without this, TT uses on-device greedy LM and may run longer before EOS.",
    )
    ap.add_argument(
        "--ref-lm",
        action="store_true",
        help="Drive TT AR loop with CPU fp32 reference LM (closer token match; combine with --match-ref-tokens for exact length).",
    )
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    try:
        ensure_demo_resources()
        model_path = str(ensure_model_weights())
    except Exception as exc:
        print(f"[dump_audio] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    script = load_script()
    processor = VibeVoiceProcessor.from_pretrained(model_path)
    ref = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="cpu", attn_implementation="sdpa"
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
    prefill_len = inputs["input_ids"].shape[1]
    ref_gen = ref_out.sequences[0, prefill_len:]
    diffusion_id = processor.tokenizer.speech_diffusion_id
    ref_diffusion = int((ref_gen == diffusion_id).sum().item())
    print(
        f"[dump_audio] ref: {ref_gen.numel()} AR tokens ({ref_diffusion} diffusion frames), "
        f"{ref_speech.numel()/SR:.2f}s audio"
        + (
            " (may be truncated — no EOS before cap)"
            if args.max_new_tokens and ref_gen.numel() >= args.max_new_tokens
            else ""
        )
    )
    _write_wav(str(Path(args.out_dir) / "ref.wav"), ref_speech)
    print(f"[dump_audio] wrote ref.wav  ({ref_speech.numel()/SR:.2f}s)")

    mesh = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        tt_model = TTVibeVoiceModel.from_checkpoint(
            mesh, model_path, cfg_scale=args.cfg_scale, num_diffusion_steps=args.num_steps
        )
        tt_model.set_speech_scale_bias(ref.model.speech_scaling_factor.item(), ref.model.speech_bias_factor.item())
        torch.manual_seed(0)
        with torch.no_grad():
            ref._process_speech_inputs(inputs["speech_tensors"].to(ref.dtype), inputs["speech_masks"])
        mode = []
        if args.match_ref_tokens:
            mode.append("match-ref-tokens")
        if args.ref_lm:
            mode.append("ref-lm")
        print(f"[dump_audio] TT generate... ({', '.join(mode) or 'free-running TT LM'})")
        tt_out = tt_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            speech_input_mask=inputs["speech_input_mask"],
            prefill_speech_embeds=prefill_embeds,
            tokenizer=processor.tokenizer,
            cfg_scale=args.cfg_scale,
            num_diffusion_steps=args.num_steps,
            max_new_tokens=args.max_new_tokens,
            forced_token_ids=ref_gen if args.match_ref_tokens else None,
            ref_inference=ref if args.ref_lm else None,
        )
        tt_speech = tt_out.speech_outputs[0].to(torch.float32).reshape(-1)
        tt_gen = tt_out.sequences[0, prefill_len:]
        tt_diffusion = int((tt_gen == diffusion_id).sum().item())
        token_match = (tt_gen == ref_gen).sum().item() if tt_gen.numel() == ref_gen.numel() else 0
        print(
            f"[dump_audio] tt:  {tt_gen.numel()} AR tokens ({tt_diffusion} diffusion frames), "
            f"{tt_speech.numel()/SR:.2f}s audio"
            + (f", token match {token_match}/{ref_gen.numel()}" if ref_gen.numel() else "")
            + (
                " (may be truncated — no EOS before cap)"
                if args.max_new_tokens and tt_gen.numel() >= args.max_new_tokens
                else ""
            )
        )
        if not args.match_ref_tokens and tt_gen.numel() != ref_gen.numel():
            print(
                "[dump_audio] hint: duration differs because TT LM chose a different EOS step "
                f"({tt_gen.numel()} vs {ref_gen.numel()} AR tokens). "
                "Re-run with --match-ref-tokens for the same length as ref.wav."
            )
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
