"""End-to-end Inworld TTS demo: text -> LLM -> speech tokens -> TTNN codec decoder -> audio.

Usage:
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    python models/demos/inworld_tts/demo_tts.py --text "Hello, this is a test."

Pipeline:
    1. LLM (merged_model, HuggingFace on CPU) generates <|s_N|> speech tokens
    2. Speech token IDs are extracted as integer VQ codes
    3. FSQ dequantizer (CPU) converts codes to 2048-dim embeddings
    4. TTNN codec decoder (VocosBackbone on device) converts embeddings to audio
    5. ISTFTHead (CPU) produces final waveform at 16kHz
"""

import argparse
import os
import re
import sys
import time

import soundfile as sf
import torch

# Add training venv for vector_quantize_pytorch ONLY (append, don't prepend, to avoid conflicts)
TRAIN_VENV = os.path.join(os.path.dirname(__file__), "train_venv", "lib")
for p in sorted(
    [os.path.join(TRAIN_VENV, d, "site-packages") for d in os.listdir(TRAIN_VENV) if d.startswith("python")],
    reverse=True,
):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)  # append to avoid overriding main env packages

# ---- Constants ----
MERGED_MODEL_DIR = os.path.join(os.path.dirname(__file__), "training", "merged_model")
CODEC_CKPT = os.path.join(
    os.path.dirname(__file__),
    "training/vectorized_data_full/.cache/models--HKUSTAudio--xcodec2/"
    "snapshots/06071873ab345f44488d235dae3cb10b5901fd90/ckpt/epoch=4-step=1400000.ckpt",
)
SAMPLE_RATE = 16000
TOKEN_RATE = 50  # speech tokens per second


def load_llm(model_dir, device="cpu"):
    """Load the merged LLaMA SpeechLM model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading LLM from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    print(f"  Vocab size: {model.config.vocab_size}, Layers: {model.config.num_hidden_layers}")
    return model, tokenizer


def load_codec_decoder(ckpt_path):
    """Load the xcodec2 codec decoder (quantizer + backbone weights)."""
    from vector_quantize_pytorch import ResidualFSQ

    print(f"Loading codec decoder from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    # Build the FSQ quantizer
    quantizer = ResidualFSQ(
        dim=2048,
        levels=[4, 4, 4, 4, 4, 4, 4, 4],
        num_quantizers=1,
    )
    # Load quantizer weights
    q_sd = {}
    for k, v in sd.items():
        if k.startswith("generator.quantizer."):
            q_sd[k.replace("generator.quantizer.", "")] = v
    quantizer.load_state_dict(q_sd, strict=False)
    quantizer.eval()

    # Extract decoder weights (backbone + head + fc_post_a)
    decoder_sd = {}
    for k, v in sd.items():
        if k.startswith("generator.backbone."):
            decoder_sd[k.replace("generator.", "")] = v
        elif k.startswith("generator.head."):
            decoder_sd[k.replace("generator.", "")] = v
        elif k.startswith("fc_post_a."):
            decoder_sd[k] = v

    print(f"  Loaded {len(q_sd)} quantizer params, {len(decoder_sd)} decoder params")
    return quantizer, decoder_sd


def build_prompt(text, tokenizer):
    """Build the TTS prompt for the LLM using the chat template.

    Uses LLaMA 3 chat format with the text as user message and
    <|speech_start|> as the beginning of the assistant response.
    The model continues generating <|s_N|> tokens until <|speech_end|>.
    """
    messages = [
        {"role": "user", "content": text},
    ]
    # Apply chat template with generation prompt (adds assistant header)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Append speech_start token as the beginning of assistant's speech response
    prompt += "<|speech_start|>"
    return prompt


def extract_speech_ids(token_strings):
    """Parse <|s_N|> tokens into integer VQ codes."""
    speech_ids = []
    for s in token_strings:
        m = re.match(r"<\|s_(\d+)\|>", s)
        if m:
            speech_ids.append(int(m.group(1)))
    return speech_ids


def generate_speech_tokens(model, tokenizer, text, max_tokens=1792, temperature=0.8):
    """Generate speech tokens from text using the LLM."""
    prompt = build_prompt(text, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].to(model.device)

    speech_end_id = tokenizer.convert_tokens_to_ids("<|speech_end|>")
    # Also include eot_id as stop token to prevent runaway generation
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    stop_ids = [speech_end_id, eot_id]

    print(f"Generating speech tokens (max={max_tokens}, temp={temperature})...")
    print(f"  speech_end_id={speech_end_id}, eot_id={eot_id}")
    t0 = time.time()

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            repetition_penalty=1.2,
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id or eot_id,
        )

    generated_ids = output[0][input_ids.shape[1] :]  # strip input prefix
    # Remove trailing stop tokens
    while len(generated_ids) > 0 and generated_ids[-1].item() in stop_ids:
        generated_ids = generated_ids[:-1]

    token_strings = tokenizer.convert_ids_to_tokens(generated_ids.tolist())
    speech_ids = extract_speech_ids(token_strings)

    elapsed = time.time() - t0
    print(f"  Generated {len(speech_ids)} speech tokens in {elapsed:.1f}s ({len(speech_ids)/elapsed:.1f} tok/s)")
    print(f"  Duration: ~{len(speech_ids)/TOKEN_RATE:.1f}s of audio")

    return speech_ids


def decode_to_audio_ttnn(speech_ids, quantizer, decoder_sd, device):
    """Decode speech IDs to audio using TTNN codec decoder."""
    from models.demos.inworld_tts.tt.codec_decoder import TtCodecDecoder

    print("Decoding speech tokens to audio via TTNN...")
    t0 = time.time()

    # Build TTNN codec decoder
    decoder = TtCodecDecoder(
        device=device,
        state_dict=decoder_sd,
        quantizer=quantizer,
        backbone_prefix="backbone.",
        head_prefix="head.",
    )

    # Convert speech IDs to tensor
    vq_codes = torch.tensor(speech_ids, dtype=torch.long).unsqueeze(0)  # [1, T]

    # Run decoder
    with torch.no_grad():
        audio = decoder(vq_codes)  # [1, 1, num_samples]

    elapsed = time.time() - t0
    audio_np = audio.squeeze().numpy()
    print(f"  Decoded to {len(audio_np)} samples ({len(audio_np)/SAMPLE_RATE:.2f}s) in {elapsed:.1f}s")

    return audio_np


def decode_to_audio_reference(speech_ids, quantizer, decoder_sd):
    """Decode speech IDs to audio using PyTorch reference (for comparison)."""
    from models.demos.inworld_tts.reference.functional import (
        codec_decoder_forward,
        extract_backbone_weights,
        extract_istft_weights,
    )

    print("Decoding speech tokens to audio via PyTorch reference...")
    t0 = time.time()

    vq_codes = torch.tensor(speech_ids, dtype=torch.long).unsqueeze(0)  # [1, T]

    backbone_weights = extract_backbone_weights(decoder_sd)
    istft_weights = extract_istft_weights(decoder_sd)

    with torch.no_grad():
        audio = codec_decoder_forward(
            vq_codes,
            quantizer,
            decoder_sd["fc_post_a.weight"],
            decoder_sd["fc_post_a.bias"],
            backbone_weights,
            istft_weights,
        )

    elapsed = time.time() - t0
    audio_np = audio.squeeze().numpy()
    print(f"  Decoded to {len(audio_np)} samples ({len(audio_np)/SAMPLE_RATE:.2f}s) in {elapsed:.1f}s")

    return audio_np


def main():
    parser = argparse.ArgumentParser(description="Inworld TTS demo")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, default="/tmp/tts_output.wav", help="Output WAV path")
    parser.add_argument("--reference-output", type=str, default=None, help="Also save reference audio for comparison")
    parser.add_argument("--max-tokens", type=int, default=1792, help="Max speech tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--skip-llm", action="store_true", help="Use random speech IDs (skip LLM, for testing)")
    parser.add_argument("--num-tokens", type=int, default=100, help="Number of random tokens (with --skip-llm)")
    args = parser.parse_args()

    # Step 1: Generate speech tokens
    if args.skip_llm:
        print(f"Skipping LLM, using {args.num_tokens} random speech IDs...")
        speech_ids = torch.randint(0, 65536, (args.num_tokens,)).tolist()
    else:
        model, tokenizer = load_llm(MERGED_MODEL_DIR)
        speech_ids = generate_speech_tokens(model, tokenizer, args.text, args.max_tokens, args.temperature)
        del model  # free GPU memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if len(speech_ids) == 0:
        print("ERROR: No speech tokens generated!")
        return

    # Step 2: Decode to audio via TTNN
    import ttnn

    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        quantizer, decoder_sd = load_codec_decoder(CODEC_CKPT)
        audio_ttnn = decode_to_audio_ttnn(speech_ids, quantizer, decoder_sd, device)
        sf.write(args.output, audio_ttnn, SAMPLE_RATE)
        print(f"TTNN audio saved to {args.output}")

        # Optional: also run reference for comparison
        if args.reference_output:
            audio_ref = decode_to_audio_reference(speech_ids, quantizer, decoder_sd)
            sf.write(args.reference_output, audio_ref, SAMPLE_RATE)
            print(f"Reference audio saved to {args.reference_output}")

            # Compare PCC
            pcc = torch.corrcoef(torch.stack([torch.tensor(audio_ttnn).flatten(), torch.tensor(audio_ref).flatten()]))[
                0, 1
            ].item()
            print(f"TTNN vs Reference PCC: {pcc:.6f}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
