"""
Kokoro TTNN Demo

Run full TTS inference on a Tenstorrent device and save the result as a .wav file.
Also prints timing breakdowns and PCC vs PyTorch reference.

CLI:
    python models/experimental/kokoro/demo/run_kokoro_ttnn.py \
        --text "Hello from Tenstorrent" \
        --voice af_heart \
        --output output.wav

Optional flags:
    --repo_id hexgrad/Kokoro-82M
    --model_path /path/to/kokoro-v1_0.pth
    --config_path /path/to/config.json
    --device_id 0
    --disable_complex       use CustomSTFT (more portable)
    --no_ref_compare        skip PyTorch reference run + PCC
"""

import argparse
import os
import sys
import time

import torch

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ttnn

from models.experimental.kokoro.reference.model import KModel
from models.experimental.kokoro.reference.pipeline import KPipeline
from models.experimental.kokoro.tt.kokoro_ttnn_model import KokoroTTNNModel


# ── PCC helper ───────────────────────────────────────────────────────────────


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return ((a * b).sum() / (torch.sqrt((a**2).sum()) * torch.sqrt((b**2).sum()) + 1e-8)).item()


# ── Save WAV ─────────────────────────────────────────────────────────────────


def save_wav(audio: torch.Tensor, path: str, sample_rate: int = 24000):
    try:
        import soundfile as sf

        sf.write(path, audio.numpy(), sample_rate)
    except ImportError:
        try:
            import scipy.io.wavfile as wav

            wav.write(path, sample_rate, audio.numpy().astype("float32"))
        except ImportError:
            # Minimal PCM writer
            import wave

            with wave.open(path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                pcm = (audio.clamp(-1, 1).numpy() * 32767).astype("int16")
                wf.writeframes(pcm.tobytes())


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Kokoro TTNN Demo")
    parser.add_argument("--text", type=str, default="Hello from Tenstorrent")
    parser.add_argument("--voice", type=str, default="af_heart")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--repo_id", type=str, default="hexgrad/Kokoro-82M")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--disable_complex", action="store_true")
    parser.add_argument("--no_ref_compare", action="store_true")
    args = parser.parse_args()

    total_start = time.perf_counter()

    # ── Open TT device ───────────────────────────────────────────────────────
    print(f"Opening TT device {args.device_id}...")
    device = ttnn.open_device(device_id=args.device_id)

    # ── Load reference KModel ────────────────────────────────────────────────
    print("Loading Kokoro 82M weights...")
    load_start = time.perf_counter()
    kmodel_kwargs = {}
    if args.model_path:
        kmodel_kwargs["model"] = args.model_path
    if args.config_path:
        kmodel_kwargs["config"] = args.config_path

    ref_kmodel = KModel(
        repo_id=args.repo_id,
        disable_complex=args.disable_complex,
        **kmodel_kwargs,
    ).eval()
    load_end = time.perf_counter()
    print(f"  Model loaded in {load_end - load_start:.2f}s")

    # ── Build TTNN model ─────────────────────────────────────────────────────
    print("Building TTNN model...")
    tt_start = time.perf_counter()
    ttnn_model = KokoroTTNNModel.from_kmodel(ref_kmodel, device)
    tt_end = time.perf_counter()
    print(f"  TTNN model built in {tt_end - tt_start:.2f}s")

    # ── Reference pipeline (for G2P and optionally PCC) ──────────────────────
    ref_pipeline = KPipeline(
        lang_code="a",
        repo_id=args.repo_id,
        model=ref_kmodel if not args.no_ref_compare else False,
    )

    # ── Tokenisation ─────────────────────────────────────────────────────────
    print(f'\nText: "{args.text}"')
    tok_start = time.perf_counter()
    _, tokens = ref_pipeline.g2p(args.text)
    # Collect phonemes from first chunk
    phonemes_list = []
    for gs, ps, tks in ref_pipeline.en_tokenize(tokens):
        phonemes_list.append((gs, ps))
    tok_end = time.perf_counter()
    print(f"  Tokenisation: {tok_end - tok_start:.3f}s")
    print(f"  Phonemes: {' | '.join(ps for _, ps in phonemes_list)}")

    # ── Load voice ───────────────────────────────────────────────────────────
    voice_pack = ref_pipeline.load_voice(args.voice)

    # ── TTNN inference ───────────────────────────────────────────────────────
    all_audio_tt = []
    all_audio_ref = []
    infer_start = time.perf_counter()

    for gs, ps in phonemes_list:
        if not ps:
            continue
        ref_s = voice_pack[len(ps) - 1].unsqueeze(0)  # (1, 256)

        # TTNN run
        tt_out = ttnn_model(ps, ref_s, speed=1.0, return_output=True)
        all_audio_tt.append(tt_out.audio)

        # Reference run (if comparing)
        if not args.no_ref_compare:
            with torch.no_grad():
                ref_out = ref_kmodel(ps, ref_s, return_output=True)
            all_audio_ref.append(ref_out.audio)

    infer_end = time.perf_counter()
    infer_time = infer_end - infer_start
    print(f"  TTNN inference: {infer_time:.3f}s")

    # ── Concatenate audio ────────────────────────────────────────────────────
    audio_tt = torch.cat(all_audio_tt) if all_audio_tt else torch.zeros(0)

    # ── Vocoder (already inside model, no extra step needed) ─────────────────
    # The waveform is the final output of decoder.generator.stft.inverse(...)
    voc_time = 0.0  # embedded in inference

    # ── Save output ──────────────────────────────────────────────────────────
    wav_start = time.perf_counter()
    save_wav(audio_tt.cpu(), args.output)
    wav_end = time.perf_counter()
    print(f"  WAV write: {wav_end - wav_start:.3f}s")
    print(f"  Output saved to: {args.output}")
    print(f"  Audio duration: {len(audio_tt)/24000:.2f}s  ({len(audio_tt)} samples @ 24 kHz)")

    # ── PCC vs reference ─────────────────────────────────────────────────────
    if not args.no_ref_compare and all_audio_ref:
        audio_ref = torch.cat(all_audio_ref)
        min_len = min(len(audio_tt), len(audio_ref))
        pcc = compute_pcc(audio_tt[:min_len], audio_ref[:min_len])
        status = "PASS" if pcc >= 0.97 else "FAIL"
        print(f"\n[PCC] Audio vs PyTorch reference: {pcc:.6f} ({status})")

    # ── Timing summary ───────────────────────────────────────────────────────
    total_end = time.perf_counter()
    print(f"\n─── Timing summary ───────────────────────────────")
    print(f"  Tokenisation  : {tok_end - tok_start:.3f}s")
    print(f"  Model load    : {load_end - load_start:.2f}s")
    print(f"  TTNN build    : {tt_end - tt_start:.2f}s")
    print(f"  TTNN inference: {infer_time:.3f}s")
    print(f"  Vocoder       : (embedded in inference)")
    print(f"  Total         : {total_end - total_start:.2f}s")
    print(f"──────────────────────────────────────────────────")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
