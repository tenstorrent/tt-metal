"""End-to-end Inworld TTS demo: text -> LLM -> speech tokens -> TTNN codec decoder -> audio.

Usage:
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate

    # Zero-shot (no voice prompt):
    python models/demos/inworld_tts/demo_tts.py --text "Hello, this is a test."

    # Voice cloning (with audio prompt):
    python models/demos/inworld_tts/demo_tts.py --text "Hello world." --prompt-audio prompt.wav

    # Skip LLM, test codec only:
    python models/demos/inworld_tts/demo_tts.py --text "test" --skip-llm --num-tokens 100

Pipeline:
    1. [Optional] Encode prompt audio -> VQ codes via TTNN codec encoder
    2. LLM generates <|s_N|> speech tokens (prompt speech IDs + new tokens)
    3. Speech token IDs are extracted as integer VQ codes
    4. TTNN codec decoder converts VQ codes -> audio waveform
    5. [If voice cloning] Trim prompt audio portion from output
"""

import argparse
import os
import re
import sys
import time

import soundfile as sf
import torch

# Add training venv for vector_quantize_pytorch ONLY (append to avoid conflicts)
TRAIN_VENV = os.path.join(os.path.dirname(__file__), "train_venv", "lib")
for p in sorted(
    [os.path.join(TRAIN_VENV, d, "site-packages") for d in os.listdir(TRAIN_VENV) if d.startswith("python")],
    reverse=True,
):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)

# ---- Constants ----
MERGED_MODEL_DIR = "/home/ttuser/models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
CODEC_CKPT = os.path.join(
    os.path.dirname(__file__),
    "/home/ttuser/models/models--HKUSTAudio--xcodec2/"
    "snapshots/06071873ab345f44488d235dae3cb10b5901fd90/ckpt/epoch=4-step=1400000.ckpt",
)
SAMPLE_RATE = 16000
TOKEN_RATE = 50  # speech tokens per second

# Tokenizer constants (from Inworld TTS)
SPEECH_START_TOKEN = "<|speech_start|>"
SPEECH_END_TOKEN = "<|speech_end|>"
TEXT_PROMPT_START_TOKEN = "<|text_prompt_start|>"
TEXT_PROMPT_END_TOKEN = "<|text_prompt_end|>"
VOICE_DESCRIPTION_START_TOKEN = "<|voice_description_start|>"
VOICE_DESCRIPTION_END_TOKEN = "<|voice_description_end|>"
SOUND_EFFECT_START_TOKEN = "<|sound_effect_start|>"
SOUND_EFFECT_END_TOKEN = "<|sound_effect_end|>"
SPEECH_TOKEN_PATTERN = "<|s_{0}|>"
_EXPECTED_VOCAB_SIZE = 193856
CODEBOOK_SIZE = 65536  # 4^8 for FSQ levels=[4,4,4,4,4,4,4,4]


# ---------------------------------------------------------------------------
# Codec loading
# ---------------------------------------------------------------------------
def load_codec(ckpt_path):
    """Load the full xcodec2 codec (quantizer + encoder weights + decoder weights)."""
    from vector_quantize_pytorch import ResidualFSQ

    print(f"Loading codec from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    # Build FSQ quantizer
    quantizer = ResidualFSQ(dim=2048, levels=[4, 4, 4, 4, 4, 4, 4, 4], num_quantizers=1)
    q_sd = {k.replace("generator.quantizer.", ""): v for k, v in sd.items() if k.startswith("generator.quantizer.")}
    quantizer.load_state_dict(q_sd, strict=False)
    quantizer.eval()

    # Extract decoder weights
    decoder_sd = {}
    for k, v in sd.items():
        if k.startswith("generator.backbone."):
            decoder_sd[k.replace("generator.", "")] = v
        elif k.startswith("generator.head."):
            decoder_sd[k.replace("generator.", "")] = v
        elif k.startswith("fc_post_a."):
            decoder_sd[k] = v

    print(f"  Loaded {len(q_sd)} quantizer, {len(decoder_sd)} decoder params")
    return quantizer, decoder_sd, sd  # return full sd for encoder


# ---------------------------------------------------------------------------
# Audio encoding (voice cloning prompt)
# ---------------------------------------------------------------------------
def encode_audio_prompt(audio_path, codec_sd, quantizer, device):
    """Encode a prompt audio file to VQ codes using the TTNN codec encoder.

    Args:
        audio_path: path to WAV file (will be resampled to 16kHz if needed)
        codec_sd: full codec state dict (for encoder weights)
        quantizer: FSQ quantizer
        device: TTNN device
    Returns:
        speech_ids: list of integer VQ codes
        duration_s: duration of the prompt audio in seconds
    """
    import torchaudio

    from models.demos.inworld_tts.tt.codec_encoder import TtCodecEncoder

    print(f"Encoding prompt audio: {audio_path}")
    t0 = time.time()

    # Load and resample audio
    waveform, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # mono
    waveform = waveform.unsqueeze(0)  # [1, 1, samples]

    duration_s = waveform.shape[-1] / SAMPLE_RATE

    # Build encoder
    encoder = TtCodecEncoder(device=device, state_dict=codec_sd, quantizer=quantizer)

    # Encode
    with torch.no_grad():
        vq_codes = encoder(waveform)  # [1, 1, T]

    speech_ids = vq_codes.squeeze().tolist()
    elapsed = time.time() - t0

    print(f"  Audio: {duration_s:.1f}s -> {len(speech_ids)} tokens in {elapsed:.1f}s")
    return speech_ids, duration_s


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
def build_tokenizer(model_dir, max_seq_len=8192, codebook_size=CODEBOOK_SIZE):
    """Build tokenizer with speech tokens added.
    
    Based on Inworld TTS tokenization.py:
    https://github.com/inworld-ai/tts/blob/a8556e420a2e6e0b18f506f369aae9efa65d2c78/tts/core/tokenization.py#L11
    
    Args:
        model_dir: HuggingFace model directory
        max_seq_len: Maximum sequence length
        codebook_size: Number of speech tokens to add (default: 65536 for 4^8 FSQ)
    
    Returns:
        tokenizer: Tokenizer with speech tokens, vocab_size = 193856
    """
    from transformers import AutoTokenizer

    print(f"Building tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        model_max_length=max_seq_len,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token
    original_vocab_size = len(tokenizer)
    print(f"  Original vocab size: {original_vocab_size}")

    # Check if already expanded
    if original_vocab_size == _EXPECTED_VOCAB_SIZE:
        print("  Tokenizer already has the correct size.")
        return tokenizer

    # Add special tokens
    new_tokens = [
        SPEECH_START_TOKEN,
        SPEECH_END_TOKEN,
        TEXT_PROMPT_START_TOKEN,
        TEXT_PROMPT_END_TOKEN,
        VOICE_DESCRIPTION_START_TOKEN,
        VOICE_DESCRIPTION_END_TOKEN,
        SOUND_EFFECT_START_TOKEN,
        SOUND_EFFECT_END_TOKEN,
    ]

    # Add speech tokens <|s_0|> through <|s_N|>
    new_tokens.extend([SPEECH_TOKEN_PATTERN.format(i) for i in range(codebook_size)])

    num_added_tokens = tokenizer.add_tokens(sorted(new_tokens))
    new_vocab_size = len(tokenizer)

    # Pad with extra tokens if needed to reach expected size
    if new_vocab_size < _EXPECTED_VOCAB_SIZE:
        num_extra_tokens = _EXPECTED_VOCAB_SIZE - new_vocab_size
        extra_tokens = [f"<extra_token_{i}>" for i in range(num_extra_tokens)]
        tokenizer.add_tokens(extra_tokens)
        new_vocab_size = len(tokenizer)
        print(f"  Added {num_extra_tokens} padding tokens to reach {new_vocab_size}.")

    if new_vocab_size != _EXPECTED_VOCAB_SIZE:
        raise ValueError(
            f"Expected tokenizer size to be {_EXPECTED_VOCAB_SIZE}, "
            f"but got {new_vocab_size}!"
        )

    print(f"  Added {num_added_tokens} speech tokens. Final vocab size: {new_vocab_size}")
    return tokenizer


def load_llm(model_dir, device="cpu"):
    """Load the merged LLaMA SpeechLM model and tokenizer."""
    from transformers import AutoModelForCausalLM

    print(f"Loading LLM from {model_dir}...")
    tokenizer = build_tokenizer(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    print(f"  Model vocab size: {model.config.vocab_size}, Layers: {model.config.num_hidden_layers}")
    return model, tokenizer


def build_prompt(text, tokenizer, prompt_speech_ids=None):
    """Build the TTS prompt for the LLM.

    Follows the official Inworld prompt format:
    - User message with text
    - Assistant begins with <|speech_start|> followed by prompt speech tokens (if any)
    - Model continues generating speech tokens until <|speech_end|>
    """
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt += SPEECH_START_TOKEN

    # Add prompt speech tokens for voice cloning
    if prompt_speech_ids:
        for sid in prompt_speech_ids:
            prompt += SPEECH_TOKEN_PATTERN.format(sid)

    return prompt


def extract_speech_ids(token_strings):
    """Parse <|s_N|> tokens into integer VQ codes."""
    speech_ids = []
    for s in token_strings:
        m = re.match(r"<\|s_(\d+)\|>", s)
        if m:
            speech_ids.append(int(m.group(1)))
    return speech_ids


def generate_speech_tokens(model, tokenizer, text, prompt_speech_ids=None, max_tokens=1792, temperature=0.8):
    """Generate speech tokens from text using the LLM."""
    prompt = build_prompt(text, tokenizer, prompt_speech_ids)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(model.device)

    speech_end_id = tokenizer.convert_tokens_to_ids(SPEECH_END_TOKEN)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    stop_ids = [speech_end_id, eot_id]

    num_prompt_tokens = len(prompt_speech_ids) if prompt_speech_ids else 0
    print(f"Generating speech tokens (max={max_tokens}, temp={temperature}, prompt_tokens={num_prompt_tokens})...")
    t0 = time.time()

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            top_k=50,
            repetition_penalty=1.1,
            # frequency_penalty=0.3,
            eos_token_id=stop_ids,
            pad_token_id=eot_id,
        )

    generated_ids = output[0][input_ids.shape[1] :]
    # Remove trailing stop tokens
    while len(generated_ids) > 0 and generated_ids[-1].item() in stop_ids:
        generated_ids = generated_ids[:-1]

    token_strings = tokenizer.convert_ids_to_tokens(generated_ids.tolist())
    speech_ids = extract_speech_ids(token_strings)

    # For voice cloning, the prompt speech IDs are part of the generation
    # Prepend them to match the official behavior
    if prompt_speech_ids:
        speech_ids = prompt_speech_ids + speech_ids

    elapsed = time.time() - t0
    new_tokens = len(speech_ids) - num_prompt_tokens
    print(
        f"  Generated {new_tokens} new + {num_prompt_tokens} prompt = {len(speech_ids)} total tokens in {elapsed:.1f}s"
    )
    print(f"  Audio duration: ~{len(speech_ids) / TOKEN_RATE:.1f}s")

    return speech_ids, num_prompt_tokens


# ---------------------------------------------------------------------------
# Audio decoding
# ---------------------------------------------------------------------------
def decode_to_audio_ttnn(speech_ids, quantizer, decoder_sd, device):
    """Decode speech IDs to audio using TTNN codec decoder."""
    from models.demos.inworld_tts.tt.codec_decoder import TtCodecDecoder

    print("Decoding speech tokens to audio via TTNN...")
    t0 = time.time()

    decoder = TtCodecDecoder(
        device=device,
        state_dict=decoder_sd,
        quantizer=quantizer,
        backbone_prefix="backbone.",
        head_prefix="head.",
    )

    vq_codes = torch.tensor(speech_ids, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        audio = decoder(vq_codes)

    elapsed = time.time() - t0
    audio_np = audio.squeeze().numpy()
    print(f"  Decoded to {len(audio_np)} samples ({len(audio_np) / SAMPLE_RATE:.2f}s) in {elapsed:.1f}s")
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

    vq_codes = torch.tensor(speech_ids, dtype=torch.long).unsqueeze(0)
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
    print(f"  Decoded to {len(audio_np)} samples ({len(audio_np) / SAMPLE_RATE:.2f}s) in {elapsed:.1f}s")
    return audio_np


def trim_prompt_audio(audio_np, num_prompt_tokens):
    """Trim the prompt audio portion from the generated output.

    The first num_prompt_tokens tokens correspond to the prompt audio
    that was used for voice cloning. Remove them to get only the new speech.
    """
    if num_prompt_tokens <= 0:
        return audio_np
    samples_to_trim = int(num_prompt_tokens / TOKEN_RATE * SAMPLE_RATE)
    if samples_to_trim >= len(audio_np):
        return audio_np
    return audio_np[samples_to_trim:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Inworld TTS demo")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, default="/tmp/tts_output.wav", help="Output WAV path")
    parser.add_argument("--reference-output", type=str, default=None, help="Also save reference audio for comparison")
    parser.add_argument("--prompt-audio", type=str, default=None, help="Prompt audio for voice cloning (WAV file)")
    parser.add_argument("--max-tokens", type=int, default=1792, help="Max speech tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--skip-llm", action="store_true", help="Use random speech IDs (skip LLM, for testing)")
    parser.add_argument("--num-tokens", type=int, default=100, help="Number of random tokens (with --skip-llm)")
    parser.add_argument("--keep-prompt", action="store_true", help="Keep prompt audio in output (don't trim)")
    args = parser.parse_args()

    import ttnn

    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        # Load codec (decoder + encoder weights)
        quantizer, decoder_sd, codec_sd = load_codec(CODEC_CKPT)

        # Step 1: Encode prompt audio (voice cloning)
        prompt_speech_ids = None
        num_prompt_tokens = 0
        if args.prompt_audio:
            prompt_speech_ids, prompt_duration = encode_audio_prompt(args.prompt_audio, codec_sd, quantizer, device)
            num_prompt_tokens = len(prompt_speech_ids)
            print(f"  Encoded {prompt_duration:.1f}s prompt -> {num_prompt_tokens} tokens")

        # Step 2: Generate speech tokens
        if args.skip_llm:
            print(f"Skipping LLM, using {args.num_tokens} random speech IDs...")
            speech_ids = torch.randint(0, 65536, (args.num_tokens,)).tolist()
            num_prompt_tokens = 0
        else:
            model, tokenizer = load_llm(MERGED_MODEL_DIR)
            speech_ids, num_prompt_tokens = generate_speech_tokens(
                model, tokenizer, args.text, prompt_speech_ids, args.max_tokens, args.temperature
            )
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if len(speech_ids) == 0:
            print("ERROR: No speech tokens generated!")
            return

        # Step 3: Decode to audio via TTNN
        audio_ttnn = decode_to_audio_ttnn(speech_ids, quantizer, decoder_sd, device)

        # Step 4: Trim prompt audio if voice cloning
        if num_prompt_tokens > 0 and not args.keep_prompt:
            audio_ttnn = trim_prompt_audio(audio_ttnn, num_prompt_tokens)
            print(f"  Trimmed {num_prompt_tokens} prompt tokens, output: {len(audio_ttnn) / SAMPLE_RATE:.2f}s")

        sf.write(args.output, audio_ttnn, SAMPLE_RATE)
        print(f"TTNN audio saved to {args.output}")

        # Optional: reference comparison
        if args.reference_output:
            audio_ref = decode_to_audio_reference(speech_ids, quantizer, decoder_sd)
            if num_prompt_tokens > 0 and not args.keep_prompt:
                audio_ref = trim_prompt_audio(audio_ref, num_prompt_tokens)
            sf.write(args.reference_output, audio_ref, SAMPLE_RATE)
            print(f"Reference audio saved to {args.reference_output}")

            min_len = min(len(audio_ttnn), len(audio_ref))
            pcc = torch.corrcoef(
                torch.stack([torch.tensor(audio_ttnn[:min_len]).flatten(), torch.tensor(audio_ref[:min_len]).flatten()])
            )[0, 1].item()
            print(f"TTNN vs Reference PCC: {pcc:.6f}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
