# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
VibeVoice TTNN demo — run inference and compare against website golden audio.

Defaults to the shortest golden clip with a local text script (1p_CH2EN /
resources/text/1p_Ch2EN.txt vs resources/golden/1p_CH2EN.wav from
https://microsoft.github.io/VibeVoice/).

Usage (from tt-metal root):
    python models/experimental/vibevoice/demo_ttnn.py
    python models/experimental/vibevoice/demo_ttnn.py --demo 2p_goat
    python models/experimental/vibevoice/demo_ttnn.py --max_new_tokens 128
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
import ttnn

from models.experimental.vibevoice.common.golden_audio_utils import (
    MINIMAL_GOLDEN_DEMO_ID,
    download_golden_demo,
    get_golden_demo,
    minimal_golden_demo,
    text_path_for_demo,
)
from models.experimental.vibevoice.common.model_utils import ensure_model_weights
from models.experimental.vibevoice.common.resource_utils import ensure_demo_resources, load_script
from models.experimental.vibevoice.tt.ttnn_vibevoice_model import TTVibeVoiceModel

_VV_ROOT = Path(__file__).resolve().parent
for _p in (_VV_ROOT / "reference", _VV_ROOT.parent.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SR = 24000


def _write_wav(path: Path, audio_1d: torch.Tensor) -> None:
    import soundfile as sf

    sf.write(str(path), audio_1d.detach().to(torch.float32).numpy(), SR)


def _load_wav(path: Path) -> torch.Tensor:
    import soundfile as sf

    data, file_sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.tensor(data, dtype=torch.float32)
    if file_sr != SR:
        import torchaudio

        wav = torchaudio.functional.resample(wav.unsqueeze(0), file_sr, SR).squeeze(0)
    return wav


def _voice_path() -> str:
    from models.experimental.vibevoice.common.config import VOICES_DIR

    p = VOICES_DIR / "en-Alice_woman.wav"
    return str(p) if p.is_file() else str(next(VOICES_DIR.glob("*.wav")))


def _compare_audio(golden: torch.Tensor, tt: torch.Tensor) -> dict:
    dur_g = golden.numel() / SR
    dur_t = tt.numel() / SR
    n = min(golden.numel(), tt.numel())
    prefix_rms = (golden[:n] - tt[:n]).pow(2).mean().sqrt().item()
    log_mel_l1 = float("nan")
    try:
        import torchaudio

        mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=1024, hop_length=256, n_mels=80)
        lm_g = torch.log(mel(golden[:n]) + 1e-5)
        lm_t = torch.log(mel(tt[:n]) + 1e-5)
        log_mel_l1 = (lm_g - lm_t).abs().mean().item()
    except (ImportError, OSError):
        pass
    return {
        "golden_sec": dur_g,
        "tt_sec": dur_t,
        "duration_ratio": dur_t / dur_g if dur_g > 0 else float("nan"),
        "prefix_rms": prefix_rms,
        "log_mel_l1": log_mel_l1,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="VibeVoice TTNN demo vs website golden audio")
    ap.add_argument(
        "--demo",
        default=None,
        help=f"Golden demo id (default: shortest with text, usually {MINIMAL_GOLDEN_DEMO_ID})",
    )
    ap.add_argument("--output_dir", default="/tmp/vv_ttnn_out")
    ap.add_argument("--model_path", default=None, help="VibeVoice checkpoint (auto-download if omitted)")
    ap.add_argument("--voice", default=None, help="Voice cloning WAV (default: en-Alice_woman)")
    ap.add_argument("--cfg_scale", type=float, default=1.3)
    ap.add_argument("--num_steps", type=int, default=10)
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Optional AR cap (default: until EOS)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    demo_entry = get_golden_demo(args.demo) if args.demo else minimal_golden_demo()
    demo_id = demo_entry.id
    golden_wav = download_golden_demo(demo_id)
    text_path = text_path_for_demo(demo_id)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ensure_demo_resources()
        model_path = str(ensure_model_weights(args.model_path))
    except Exception as exc:
        print(f"[demo_ttnn] ERROR: {exc}", file=sys.stderr)
        return 1

    script = load_script(text_path)
    voice = args.voice or _voice_path()

    print(f"[demo_ttnn] demo={demo_id}  text={text_path.name}  golden={golden_wav.name}")
    print(f"[demo_ttnn] {demo_entry.website_title} ({demo_entry.website_section})")

    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    processor = VibeVoiceProcessor.from_pretrained(model_path)
    inputs = processor(
        text=[script],
        voice_samples=[[voice]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    prefill_len = inputs["input_ids"].shape[1]

    # Voice prefill embeds on CPU (avoids long-waveform acoustic encode on device L1).
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference

    ref_prefill = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="sdpa",
    )
    ref_prefill.eval()
    with torch.no_grad():
        _, prefill_embeds = ref_prefill._process_speech_inputs(
            inputs["speech_tensors"].to(ref_prefill.dtype),
            inputs["speech_masks"],
        )
    speech_scale = ref_prefill.model.speech_scaling_factor.item()
    speech_bias = ref_prefill.model.speech_bias_factor.item()
    del ref_prefill

    mesh = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        print("[demo_ttnn] Loading TTVibeVoiceModel...")
        tt_model = TTVibeVoiceModel.from_checkpoint(
            mesh,
            model_path,
            cfg_scale=args.cfg_scale,
            num_diffusion_steps=args.num_steps,
        )
        tt_model.set_speech_scale_bias(speech_scale, speech_bias)

        torch.manual_seed(args.seed)
        print("[demo_ttnn] TT generate...")
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
        tt_gen = tt_out.sequences[0, prefill_len:]
    finally:
        ttnn.close_device(mesh)

    tt_path = out_dir / "tt.wav"
    golden_out = out_dir / "golden.wav"
    _write_wav(tt_path, tt_speech)
    shutil.copy2(golden_wav, golden_out)

    golden = _load_wav(golden_wav)
    metrics = _compare_audio(golden, tt_speech)

    mel_msg = (
        f"log-mel L1={metrics['log_mel_l1']:.4f}"
        if metrics["log_mel_l1"] == metrics["log_mel_l1"]
        else f"prefix RMS={metrics['prefix_rms']:.4f}"
    )
    print(
        f"[demo_ttnn] TT:  {tt_gen.numel()} AR tokens, {metrics['tt_sec']:.2f}s → {tt_path}\n"
        f"[demo_ttnn] Golden: {metrics['golden_sec']:.2f}s → {golden_out}\n"
        f"[demo_ttnn] Compare (prefix {min(golden.numel(), tt_speech.numel())/SR:.2f}s): "
        f"{mel_msg}  duration ratio={metrics['duration_ratio']:.3f}"
    )
    if abs(metrics["duration_ratio"] - 1.0) > 0.05:
        print(
            "[demo_ttnn] note: duration differs from golden — website audio used a fixed Microsoft "
            "demo run; TT free-running LM may stop at a different EOS step."
        )

    print(f"[demo_ttnn] DONE → {out_dir}/tt.wav  vs  {out_dir}/golden.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
