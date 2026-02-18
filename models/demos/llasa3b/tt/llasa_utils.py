# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Llasa-3B Speech Utilities

Provides speech-token manipulation and XCodec2 audio decode functions
used by the Llasa-3B TTS demo pipeline.

Llasa-3B extends the LLaMA-3.2-3B vocabulary with 65,536 XCodec2 speech tokens
of the form <|s_0|> through <|s_65535|>. These functions handle:
  - Converting between integer speech IDs and their string token representations
  - Formatting text into the Llasa chat template for TTS inference
  - Decoding generated speech tokens into audio waveforms via XCodec2
"""

import os

import torch
from loguru import logger

# ============================================================================
# Speech Token Manipulation
# ============================================================================


def ids_to_speech_tokens(speech_ids):
    """Convert integer speech IDs to string tokens like <|s_123|>.

    Args:
        speech_ids: List or iterable of integer speech token IDs (0–65535).

    Returns:
        List of string tokens, e.g. ["<|s_0|>", "<|s_1|>", ...].
    """
    return [f"<|s_{sid}|>" for sid in speech_ids]


def extract_speech_ids(speech_tokens_str):
    """Extract integer speech IDs from string tokens like <|s_123|>.

    Silently skips any tokens that don't match the <|s_N|> pattern.

    Args:
        speech_tokens_str: List of token strings from tokenizer.batch_decode().

    Returns:
        List of integer speech IDs.
    """
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith("<|s_") and token_str.endswith("|>"):
            try:
                num = int(token_str[4:-2])
                speech_ids.append(num)
            except ValueError:
                pass
    return speech_ids


# ============================================================================
# Chat Template Formatting
# ============================================================================


def format_llasa_chat(text, tokenizer, prompt_text=None, prompt_speech_tokens=None):
    """Format input text into the Llasa chat template.

    Supports two modes:
      - Zero-shot TTS: text only (default)
      - Prompted TTS (voice cloning): text + prompt audio speech tokens

    Args:
        text: The text to convert to speech.
        tokenizer: HuggingFace tokenizer for Llasa-3B.
        prompt_text: Optional text corresponding to prompt audio (for voice cloning).
        prompt_speech_tokens: Optional list of speech token strings (for voice cloning).

    Returns:
        input_ids: Tensor of token IDs [1, seq_len].
    """
    if prompt_text and prompt_speech_tokens:
        # Prompted TTS (voice cloning): combine prompt text + target text
        # Ensure there is a space between prompt and target text
        if prompt_text and not prompt_text.endswith(" ") and not text.startswith(" "):
            combined_text = prompt_text + " " + text
        else:
            combined_text = prompt_text + text

        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{combined_text}<|TEXT_UNDERSTANDING_END|>"
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + "".join(prompt_speech_tokens)},
        ]
    else:
        # Zero-shot TTS: text only
        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"},
        ]

    input_ids = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors="pt", continue_final_message=True)
    return input_ids


# ============================================================================
# XCodec2 Audio Decode
# ============================================================================

# XCodec2 output rate: 50 speech tokens per second of audio at 16kHz
XCODEC2_TOKENS_PER_SECOND = 50
XCODEC2_SAMPLE_RATE = 16000

# Max prompt speech tokens for voice cloning (~20 seconds of audio)
# This leaves room for generation within the 2048 token training limit
MAX_PROMPT_SPEECH_TOKENS = 1000


def encode_prompt_audio(wav_path, max_tokens=MAX_PROMPT_SPEECH_TOKENS, device="cpu"):
    """Encode a prompt WAV file into XCodec2 speech token strings for voice cloning.

    Loads the WAV file, encodes it with XCodec2's encoder to produce VQ codes,
    and converts them to speech token strings. Automatically truncates to
    max_tokens to fit within the model's 2048 token budget.

    Args:
        wav_path: Path to a 16kHz WAV file.
        max_tokens: Maximum number of speech tokens to keep (default: 150 = ~3s).
        device: Device for XCodec2 inference ("cpu" or "cuda").

    Returns:
        List of speech token strings (e.g. ["<|s_123|>", "<|s_456|>", ...]),
        or None on failure.
    """
    try:
        import soundfile as sf
        from xcodec2.modeling_xcodec2 import XCodec2Model

        logger.info(f"Loading XCodec2 encoder for prompt audio...")
        codec_model = XCodec2Model.from_pretrained("HKUSTAudio/xcodec2")
        codec_model.eval().to(device)

        logger.info(f"Encoding prompt audio from {wav_path}...")
        prompt_wav, sr = sf.read(wav_path)
        if sr != XCODEC2_SAMPLE_RATE:
            logger.warning(
                f"Prompt audio sample rate is {sr}Hz, expected {XCODEC2_SAMPLE_RATE}Hz. Results may be poor."
            )

        prompt_wav_tensor = torch.from_numpy(prompt_wav).float().unsqueeze(0).to(device)

        with torch.no_grad():
            vq_code_prompt = codec_model.encode_code(input_waveform=prompt_wav_tensor)

        vq_code_prompt = vq_code_prompt[0, 0, :]
        num_tokens = len(vq_code_prompt)
        audio_duration = num_tokens / XCODEC2_TOKENS_PER_SECOND
        logger.info(f"Prompt encoded: {num_tokens} tokens ({audio_duration:.1f}s)")

        # Truncate if needed to fit within token budget
        if num_tokens > max_tokens:
            logger.info(
                f"Truncating prompt from {num_tokens} tokens ({audio_duration:.1f}s) "
                f"to {max_tokens} tokens ({max_tokens / XCODEC2_TOKENS_PER_SECOND:.1f}s)"
            )
            vq_code_prompt = vq_code_prompt[:max_tokens]

        speech_token_strs = ids_to_speech_tokens(vq_code_prompt.cpu().numpy().tolist())
        return speech_token_strs

    except ImportError as e:
        logger.warning(f"Failed to load XCodec2 or dependencies: {e}")
        return None
    except OSError as e:
        logger.warning(f"Failed to load shared library or audio file: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error during prompt audio encoding: {e}")
        return None


def decode_speech_to_audio(speech_ids, output_path="output.wav", device="cpu"):
    """Decode speech token IDs to audio waveform using XCodec2.

    Loads XCodec2 from HuggingFace, decodes the speech token IDs to a waveform,
    and writes the result as a 16kHz WAV file. Gracefully handles missing
    dependencies (e.g. torchaudio/CUDA issues in CPU-only environments).

    Args:
        speech_ids: List of integer speech token IDs from model output.
        output_path: Path to save the output WAV file.
        device: Device for XCodec2 inference ("cpu" or "cuda").

    Returns:
        output_path on success, None on failure.
    """
    try:
        import soundfile as sf
        from xcodec2.modeling_xcodec2 import XCodec2Model

        logger.info("Loading XCodec2 model...")
        codec_model = XCodec2Model.from_pretrained("HKUSTAudio/xcodec2")
        codec_model.eval().to(device)

        speech_tokens_tensor = torch.tensor(speech_ids).unsqueeze(0).unsqueeze(0).to(device)

        logger.info(f"Decoding {len(speech_ids)} speech tokens to waveform...")
        with torch.no_grad():
            gen_wav = codec_model.decode_code(speech_tokens_tensor)

        wav_data = gen_wav[0, 0, :].cpu().numpy()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        sf.write(output_path, wav_data, XCODEC2_SAMPLE_RATE)

        audio_duration = len(wav_data) / XCODEC2_SAMPLE_RATE
        logger.info(f"Saved audio to {output_path} ({audio_duration:.2f}s @ {XCODEC2_SAMPLE_RATE}Hz)")
        return output_path

    except ImportError as e:
        logger.warning(f"Failed to load XCodec2 or dependencies: {e}")
        logger.warning("Skipping audio generation. The TTNN model output is still valid.")
        return None
    except OSError as e:
        logger.warning(f"Failed to load shared library: {e}")
        logger.warning("Skipping audio generation. The TTNN model output is still valid.")
        return None
    except Exception as e:
        logger.warning(f"Error during audio decoding: {e}")
        return None
