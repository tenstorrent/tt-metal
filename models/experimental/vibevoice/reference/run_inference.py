# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Reference end-to-end VibeVoice-1.5B TTS inference (PyTorch gold for PCC).

Adapted from VibeVoice/demo/inference_from_file.py; uses vendored vibevoice under reference/.
"""

import argparse
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import List, Tuple

import torch
from transformers.utils import logging

_REFERENCE_DIR = Path(__file__).resolve().parent
_VIBEVOICE_ROOT = _REFERENCE_DIR.parent
_TT_METAL_ROOT = _VIBEVOICE_ROOT.parent.parent.parent

for path in (_REFERENCE_DIR, _TT_METAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.experimental.vibevoice.common.config import (  # noqa: E402
    DEFAULT_TXT_PATH,
    MODEL_PATH,
    VOICES_DIR,
)
from vibevoice.modular.modeling_vibevoice_inference import (  # noqa: E402
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor  # noqa: E402

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class VoiceMapper:
    """Maps speaker names to voice WAV paths under resources/voices/."""

    def __init__(self, voices_dir: Path):
        self.voices_dir = voices_dir
        self.setup_voice_presets()

        new_dict = {}
        for name, path in self.voice_presets.items():
            if "_" in name:
                name = name.split("_")[0]
            if "-" in name:
                name = name.split("-")[-1]
            new_dict[name] = path
        self.voice_presets.update(new_dict)

    def setup_voice_presets(self):
        if not self.voices_dir.exists():
            print(f"Warning: Voices directory not found at {self.voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return

        self.voice_presets = {}
        wav_files = [
            f for f in os.listdir(self.voices_dir) if f.lower().endswith(".wav") and (self.voices_dir / f).is_file()
        ]

        for wav_file in wav_files:
            name = os.path.splitext(wav_file)[0]
            self.voice_presets[name] = str(self.voices_dir / wav_file)

        self.voice_presets = dict(sorted(self.voice_presets.items()))
        self.available_voices = {name: path for name, path in self.voice_presets.items() if os.path.exists(path)}

        print(f"Found {len(self.available_voices)} voice files in {self.voices_dir}")
        print(f"Available voices: {', '.join(self.available_voices.keys())}")

    def get_voice_path(self, speaker_name: str) -> str:
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]

        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_lower or speaker_lower in preset_name.lower():
                return path

        default_voice = list(self.voice_presets.values())[0]
        print(f"Warning: No voice preset found for '{speaker_name}', using default: {default_voice}")
        return default_voice


def parse_txt_script(txt_content: str) -> Tuple[List[str], List[str]]:
    lines = txt_content.strip().split("\n")
    scripts = []
    speaker_numbers = []
    speaker_pattern = r"^Speaker\s+(\d+):\s*(.*)$"

    current_speaker = None
    current_text = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(speaker_pattern, line, re.IGNORECASE)
        if match:
            if current_speaker and current_text:
                scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
                speaker_numbers.append(current_speaker)
            current_speaker = match.group(1).strip()
            current_text = match.group(2).strip()
        else:
            current_text = f"{current_text} {line}".strip() if current_text else line

    if current_speaker and current_text:
        scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
        speaker_numbers.append(current_speaker)

    return scripts, speaker_numbers


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice-1.5B reference TTS from text file")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Local model weights directory")
    parser.add_argument("--txt_path", type=str, default=str(DEFAULT_TXT_PATH), help="Script text file")
    parser.add_argument(
        "--speaker_names",
        type=str,
        nargs="+",
        default=["Alice"],
        help="Speaker names in order (e.g. Alice)",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory for output WAV")
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")),
        help="Device: cuda | mps | cpu",
    )
    parser.add_argument(
        "--disable_prefill",
        action="store_true",
        help="Disable voice-cloning prefill (is_prefill=False)",
    )
    parser.add_argument("--cfg_scale", type=float, default=1.3, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device.lower() == "mpx":
        print("Note: device 'mpx' detected, treating it as 'mps'.")
        args.device = "mps"

    if args.device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available. Falling back to CPU.")
        args.device = "cpu"

    print(f"Using device: {args.device}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    voice_mapper = VoiceMapper(VOICES_DIR)

    if not os.path.exists(args.txt_path):
        print(f"Error: txt file not found: {args.txt_path}")
        return

    print(f"Reading script from: {args.txt_path}")
    with open(args.txt_path, encoding="utf-8") as f:
        txt_content = f.read()

    scripts, speaker_numbers = parse_txt_script(txt_content)
    if not scripts:
        print("Error: No valid speaker scripts found in the txt file")
        return

    speaker_name_mapping = {str(i): name for i, name in enumerate(args.speaker_names, 1)}

    voice_samples = []
    unique_speaker_numbers = []
    seen = set()
    for speaker_num in speaker_numbers:
        if speaker_num not in seen:
            unique_speaker_numbers.append(speaker_num)
            seen.add(speaker_num)

    for speaker_num in unique_speaker_numbers:
        speaker_name = speaker_name_mapping.get(speaker_num, f"Speaker {speaker_num}")
        voice_path = voice_mapper.get_voice_path(speaker_name)
        voice_samples.append(voice_path)
        print(f"Speaker {speaker_num} ('{speaker_name}') -> Voice: {os.path.basename(voice_path)}")

    full_script = "\n".join(scripts).replace("\u2019", "'")

    print(f"Loading processor & model from {args.model_path}")
    processor = VibeVoiceProcessor.from_pretrained(args.model_path)

    if args.device == "mps":
        load_dtype = torch.float32
        attn_impl_primary = "sdpa"
    elif args.device == "cuda":
        load_dtype = torch.bfloat16
        attn_impl_primary = "flash_attention_2"
    else:
        load_dtype = torch.float32
        attn_impl_primary = "sdpa"

    print(f"torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")

    try:
        if args.device == "mps":
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                args.model_path,
                torch_dtype=load_dtype,
                attn_implementation=attn_impl_primary,
                device_map=None,
            )
            model.to("mps")
        elif args.device == "cuda":
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                args.model_path,
                torch_dtype=load_dtype,
                device_map="cuda",
                attn_implementation=attn_impl_primary,
            )
        else:
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                args.model_path,
                torch_dtype=load_dtype,
                device_map="cpu",
                attn_implementation=attn_impl_primary,
            )
    except Exception as e:
        if attn_impl_primary == "flash_attention_2":
            print(f"[ERROR] {type(e).__name__}: {e}")
            print(traceback.format_exc())
            print("Falling back to SDPA.")
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                args.model_path,
                torch_dtype=load_dtype,
                device_map=(args.device if args.device in ("cuda", "cpu") else None),
                attn_implementation="sdpa",
            )
            if args.device == "mps":
                model.to("mps")
        else:
            raise

    print(f"Voice cloning: is_prefill={not args.disable_prefill}")
    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)

    inputs = processor(
        text=[full_script],
        voice_samples=[voice_samples],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    target_device = args.device
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(target_device)

    print(f"Starting generation with cfg_scale: {args.cfg_scale}")
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=args.cfg_scale,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=True,
        is_prefill=not args.disable_prefill,
    )
    generation_time = time.time() - start_time
    print(f"Generation time: {generation_time:.2f} seconds")

    sample_rate = 24000
    audio_duration = 0.0
    rtf = float("inf")
    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        audio = outputs.speech_outputs[0]
        audio_samples = audio.shape[-1] if len(audio.shape) > 0 else len(audio)
        audio_duration = audio_samples / sample_rate
        rtf = generation_time / audio_duration if audio_duration > 0 else float("inf")
        print(f"Generated audio duration: {audio_duration:.2f} seconds")
        print(f"RTF: {rtf:.2f}x")
    else:
        print("No audio output generated")

    txt_filename = os.path.splitext(os.path.basename(args.txt_path))[0]
    output_path = os.path.join(args.output_dir, f"{txt_filename}_generated.wav")
    os.makedirs(args.output_dir, exist_ok=True)
    processor.save_audio(outputs.speech_outputs[0], output_path=output_path)
    print(f"Saved output to {output_path}")


if __name__ == "__main__":
    main()
