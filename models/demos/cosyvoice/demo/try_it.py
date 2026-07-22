"""Interactive CosyVoice2 demo — test zero_shot, cross_lingual, instruct2 with your own inputs.

Usage:
    source /root/tt-metal/python_env/bin/activate
    cd /root/tt-metal
    python models/demos/cosyvoice/demo/try_it.py
"""

import sys
from pathlib import Path

import soundfile

DEMO_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = DEMO_ROOT / "model_data" / "cosyvoice2-0.5B"
CV_SRC = DEMO_ROOT / "model_data" / "CosyVoice_src"
ASSET_DIR = CV_SRC / "asset"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

sys.path.insert(0, str(DEMO_ROOT))


def save_wav(waveform, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio = waveform.detach().cpu().numpy()
    if audio.ndim == 2:
        audio = audio[0]
    path = OUTPUT_DIR / f"{name}.wav"
    soundfile.write(str(path), audio, 24000)
    print(f"  -> saved: {path} ({len(audio)/24000:.1f}s)")


def main():
    import ttnn
    from models.demos.cosyvoice.tt.pipeline import TtnnCosyVoice

    print("Opening N300 device...")
    device = ttnn.open_device(device_id=0, l1_small_size=64 * 1024)

    print("Loading pipeline (LLM + flow + vocoder)...")
    pipe = TtnnCosyVoice(device, model_dir=str(CKPT_DIR))

    default_prompt = str(ASSET_DIR / "zero_shot_prompt.wav")
    default_prompt_text = "希望你以后能够做的比我还好呦。"
    cross_lingual_prompt = str(ASSET_DIR / "cross_lingual_prompt.wav")

    print("\n=== CosyVoice2 Interactive Demo ===")
    print("Modes: zero_shot, cross_lingual, instruct2")
    print("Type 'quit' to exit.\n")

    while True:
        mode = input("Mode [zero_shot/cross_lingual/instruct2]: ").strip()
        if mode.lower() in ("quit", "q", "exit"):
            break

        if mode == "zero_shot":
            text = input("TTS text: ").strip()
            prompt_text = input(f"Prompt text [{default_prompt_text}]: ").strip() or default_prompt_text
            prompt_wav = input(f"Prompt wav [{default_prompt}]: ").strip() or default_prompt
            print("  Generating...")
            wav = pipe.inference_zero_shot(text, prompt_text, prompt_wav)
            save_wav(wav, "try_zero_shot")

        elif mode == "cross_lingual":
            text = input("TTS text (any language, use <|zh|><|en|><|ja|><|yue|><|ko|> tags): ").strip()
            prompt_wav = input(f"Prompt wav [{cross_lingual_prompt}]: ").strip() or cross_lingual_prompt
            print("  Generating...")
            wav = pipe.inference_cross_lingual(text, prompt_wav)
            save_wav(wav, "try_cross_lingual")

        elif mode == "instruct2":
            text = input("TTS text: ").strip()
            instruct = input("Instruct text (e.g. 用四川话说这句话<|endofprompt|>): ").strip()
            if not instruct.endswith("<|endofprompt|>"):
                instruct += "<|endofprompt|>"
            prompt_wav = input(f"Prompt wav [{default_prompt}]: ").strip() or default_prompt
            print("  Generating...")
            wav = pipe.inference_instruct2(text, instruct, prompt_wav)
            save_wav(wav, "try_instruct2")

        else:
            print(f"  Unknown mode: {mode}")

    ttnn.close_device(device)
    print("Done.")


if __name__ == "__main__":
    main()
