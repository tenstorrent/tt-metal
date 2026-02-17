# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Llasa-3B PyTorch Reference Implementation

Based on the official examples from https://huggingface.co/HKUSTAudio/Llasa-3B
Supports two modes:
  1. Zero-shot TTS: text only → speech tokens → audio
  2. Prompted TTS (voice cloning): audio prompt + text → speech tokens → audio

Usage:
    # Zero-shot TTS
    python llasa_reference.py --text "Hello, this is a test."

    # Prompted TTS (voice cloning)
    python llasa_reference.py --text "Target text to speak." \\
        --prompt_text "Text spoken in the prompt audio." \\
        --prompt_wav prompt.wav
"""

import argparse
import os
import time

import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from xcodec2.modeling_xcodec2 import XCodec2Model


def ids_to_speech_tokens(speech_ids):
    """Convert integer speech IDs to string tokens like <|s_123|>."""
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str


def extract_speech_ids(speech_tokens_str):
    """Extract integer speech IDs from string tokens like <|s_123|>."""
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith("<|s_") and token_str.endswith("|>"):
            num_str = token_str[4:-2]
            try:
                num = int(num_str)
                speech_ids.append(num)
            except ValueError:
                pass
        else:
            # Non-speech tokens are skipped
            pass
    return speech_ids


def run_reference_zero_shot(text, output_dir, device="cpu"):
    """
    Mode 1: Zero-shot TTS — generate speech from text only.
    Based on the official HuggingFace example.
    """
    print(f"=== Zero-shot TTS on {device} ===")

    llasa_3b = os.environ.get("HF_MODEL", "HKUSTAudio/Llasa-3B")

    print(f"Loading Llasa-3B from {llasa_3b}...")
    tokenizer = AutoTokenizer.from_pretrained(llasa_3b)
    model = AutoModelForCausalLM.from_pretrained(llasa_3b)
    model.eval()
    model.to(device)

    print("Loading XCodec2...")
    Codec_model = XCodec2Model.from_pretrained("HKUSTAudio/xcodec2")
    Codec_model.eval().to(device)

    # Format input using the Llasa chat template
    formatted_text = f"<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"
    chat = [
        {"role": "user", "content": "Convert the text to speech:" + formatted_text},
        {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"},
    ]

    input_ids = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors="pt", continue_final_message=True)
    input_ids = input_ids.to(device)
    speech_end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

    print(f"Input tokens: {input_ids.shape[1]}")
    print("Generating speech tokens...")

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=2048,  # Model was trained with max_length=2048
            eos_token_id=speech_end_id,
            do_sample=True,
            top_p=1,
            temperature=0.8,
        )
    gen_time = time.time() - t0

    # Extract the speech tokens (exclude input tokens and EOS)
    generated_ids = outputs[0][input_ids.shape[1] : -1]
    speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    speech_ids = extract_speech_ids(speech_tokens)

    print(f"Generated {len(speech_ids)} speech tokens in {gen_time:.2f}s " f"({len(speech_ids)/gen_time:.1f} tok/s)")

    # Save tokens for comparison with TTNN
    os.makedirs(output_dir, exist_ok=True)
    torch.save(generated_ids.cpu(), os.path.join(output_dir, "generated_ids.pt"))
    torch.save(torch.tensor(speech_ids), os.path.join(output_dir, "speech_ids.pt"))

    if not speech_ids:
        print("No speech tokens generated.")
        return

    # Decode speech tokens to waveform using XCodec2
    print("Decoding to waveform...")
    speech_tokens_tensor = torch.tensor(speech_ids).to(device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        gen_wav = Codec_model.decode_code(speech_tokens_tensor)

    output_wav_path = os.path.join(output_dir, "reference_output.wav")
    sf.write(output_wav_path, gen_wav[0, 0, :].cpu().numpy(), 16000)
    audio_duration = gen_wav.shape[-1] / 16000
    print(f"Saved audio to {output_wav_path} ({audio_duration:.2f}s @ 16kHz)")


def run_reference_prompted(target_text, prompt_text, prompt_wav_path, output_dir, device="cpu"):
    """
    Mode 2: Prompted TTS (voice cloning) — generate speech in the style of a prompt.
    Based on the official HuggingFace example.
    """
    print(f"=== Prompted TTS (Voice Cloning) on {device} ===")

    llasa_3b = os.environ.get("HF_MODEL", "HKUSTAudio/Llasa-3B")

    print(f"Loading Llasa-3B from {llasa_3b}...")
    tokenizer = AutoTokenizer.from_pretrained(llasa_3b)
    model = AutoModelForCausalLM.from_pretrained(llasa_3b)
    model.eval()
    model.to(device)

    print("Loading XCodec2...")
    Codec_model = XCodec2Model.from_pretrained("HKUSTAudio/xcodec2")
    Codec_model.eval().to(device)

    # Encode the prompt audio to speech tokens using XCodec2
    print(f"Encoding prompt audio from {prompt_wav_path}...")
    prompt_wav, sr = sf.read(prompt_wav_path)  # Only 16kHz speech supported
    prompt_wav_tensor = torch.from_numpy(prompt_wav).float().unsqueeze(0)

    with torch.no_grad():
        vq_code_prompt = Codec_model.encode_code(input_waveform=prompt_wav_tensor)
    print(f"Prompt VQ code shape: {vq_code_prompt.shape}")

    vq_code_prompt = vq_code_prompt[0, 0, :]

    # Truncate prompt audio if too long to fit within max_length=2048
    # Budget: ~60 tokens for chat template/text, rest split between prompt speech + generated speech
    MAX_PROMPT_SPEECH_TOKENS = 150  # ~3 seconds of audio at 50 tokens/sec
    if len(vq_code_prompt) > MAX_PROMPT_SPEECH_TOKENS:
        print(
            f"Warning: Prompt audio is {len(vq_code_prompt)} tokens ({len(vq_code_prompt)/50:.1f}s). "
            f"Truncating to {MAX_PROMPT_SPEECH_TOKENS} tokens ({MAX_PROMPT_SPEECH_TOKENS/50:.1f}s) "
            f"to fit within max_length=2048."
        )
        vq_code_prompt = vq_code_prompt[:MAX_PROMPT_SPEECH_TOKENS]

    speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)

    # Combine prompt_text + target_text as the full input
    input_text = prompt_text + target_text
    formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

    # The assistant message includes the encoded prompt speech tokens
    chat = [
        {"role": "user", "content": "Convert the text to speech:" + formatted_text},
        {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + "".join(speech_ids_prefix)},
    ]

    input_ids = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors="pt", continue_final_message=True)
    input_ids = input_ids.to(device)
    speech_end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

    print(f"Input tokens: {input_ids.shape[1]} (includes {len(speech_ids_prefix)} prompt speech tokens)")

    if input_ids.shape[1] >= 2048:
        print(
            f"Error: Input ({input_ids.shape[1]} tokens) exceeds max_length=2048. "
            f"Use a shorter prompt audio or text."
        )
        return

    print(f"Token budget: {2048 - input_ids.shape[1]} tokens available for generation")
    print("Generating speech tokens...")

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=2048,  # Model was trained with max_length=2048
            eos_token_id=speech_end_id,
            do_sample=True,
            top_p=1,
            temperature=0.8,
        )
    gen_time = time.time() - t0

    # Extract speech tokens — include the prompt prefix tokens
    generated_ids = outputs[0][input_ids.shape[1] - len(speech_ids_prefix) : -1]
    speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    speech_ids = extract_speech_ids(speech_tokens)

    print(f"Generated {len(speech_ids)} speech tokens in {gen_time:.2f}s " f"({len(speech_ids)/gen_time:.1f} tok/s)")

    # Save tokens
    os.makedirs(output_dir, exist_ok=True)
    torch.save(generated_ids.cpu(), os.path.join(output_dir, "generated_ids_prompted.pt"))
    torch.save(torch.tensor(speech_ids), os.path.join(output_dir, "speech_ids_prompted.pt"))

    if not speech_ids:
        print("No speech tokens generated.")
        return

    # Decode speech tokens to waveform
    print("Decoding to waveform...")
    speech_tokens_tensor = torch.tensor(speech_ids).to(device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        gen_wav = Codec_model.decode_code(speech_tokens_tensor)

    output_wav_path = os.path.join(output_dir, "reference_output_prompted.wav")
    sf.write(output_wav_path, gen_wav[0, 0, :].cpu().numpy(), 16000)
    audio_duration = gen_wav.shape[-1] / 16000
    print(f"Saved audio to {output_wav_path} ({audio_duration:.2f}s @ 16kHz)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llasa-3B PyTorch Reference")
    parser.add_argument(
        "--text",
        type=str,
        default="Dealing with family secrets is never easy. " "Yet, sometimes, omission is a form of protection.",
    )
    parser.add_argument(
        "--prompt_text", type=str, default=None, help="Text spoken in the prompt audio (for voice cloning mode)"
    )
    parser.add_argument(
        "--prompt_wav", type=str, default=None, help="Path to 16kHz prompt WAV file (for voice cloning mode)"
    )
    parser.add_argument("--output_dir", type=str, default="reference_output")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.prompt_text and args.prompt_wav:
        run_reference_prompted(
            target_text=args.text,
            prompt_text=args.prompt_text,
            prompt_wav_path=args.prompt_wav,
            output_dir=args.output_dir,
            device=args.device,
        )
    else:
        run_reference_zero_shot(
            text=args.text,
            output_dir=args.output_dir,
            device=args.device,
        )
