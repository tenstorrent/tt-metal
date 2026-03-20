# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Pure Reference TTS Demo with ICL (In-Context Learning) Mode

Generates speech from text using ONLY our reference implementations.
No qwen_tts package dependency. Requires reference audio for voice cloning.

Pipeline:
1. Load reference audio and encode to codes (Speech Tokenizer Encoder)
2. Extract speaker embedding (Speaker Encoder)
3. Create ICL input embeddings (reference codes + text)
4. Run Talker autoregressively to generate first codebook
5. Run Code Predictor to generate remaining 15 codebooks
6. Decode codes to audio (Speech Tokenizer Decoder)

Usage:
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    python models/demos/qwen3_tts/demo/demo_pure_reference_tts.py \
        --text "Hello world" \
        --ref-audio /tmp/clone_ref.wav \
        --ref-text "Reference text spoken in the audio"
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


@dataclass
class TTSConfig:
    """Configuration for TTS generation - from actual model config."""

    # Model dimensions
    hidden_size: int = 2048
    text_hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 6144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0

    # Vocab sizes
    vocab_size: int = 3072  # codec vocab size
    text_vocab_size: int = 151936

    # Codec special tokens (from config.json)
    codec_bos_id: int = 2149
    codec_eos_id: int = 2150
    codec_pad_id: int = 2148
    codec_think_id: int = 2154
    codec_nothink_id: int = 2155
    codec_think_bos_id: int = 2156
    codec_think_eos_id: int = 2157

    # Language IDs
    codec_language_ids: dict = None

    # TTS special tokens (from config.json)
    tts_bos_token_id: int = 151672  # <tts_text_bos>
    tts_eos_token_id: int = 151673  # <tts_text_eod>
    tts_pad_token_id: int = 151671  # <tts_pad>

    # Generation
    max_new_tokens: int = 256
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    greedy: bool = False  # Use greedy decoding (causes repetitive output - not recommended)
    repetition_penalty: float = 1.0  # >1.0 discourages repetition (e.g., 1.1-1.3)

    # Code predictor
    num_code_groups: int = 16
    code_predictor_layers: int = 5
    code_predictor_hidden: int = 1024

    def __post_init__(self):
        if self.codec_language_ids is None:
            self.codec_language_ids = {
                "english": 2050,
                "chinese": 2055,
                "german": 2053,
                "italian": 2070,
                "portuguese": 2071,
                "spanish": 2054,
                "japanese": 2058,
                "korean": 2064,
                "french": 2061,
                "russian": 2069,
            }


def load_weights():
    """Load all model weights."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print("Loading model weights...")
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    model_path = Path(model_path)

    # Main model weights
    main_dict = load_file(model_path / "model.safetensors")
    main_dict = {k: v.float() for k, v in main_dict.items()}
    print(f"  Loaded {len(main_dict)} main weights")

    # Speech tokenizer weights (decoder)
    speech_path = model_path / "speech_tokenizer" / "model.safetensors"
    speech_dict = load_file(speech_path)
    decoder_weights = {k[8:]: v.float() for k, v in speech_dict.items() if k.startswith("decoder.")}
    print(f"  Loaded {len(decoder_weights)} decoder weights")

    return main_dict, decoder_weights, model_path


def load_tokenizer():
    """Load the text tokenizer."""
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        trust_remote_code=True,
    )
    return tokenizer


def encode_reference_audio(audio_path: str, weights: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode reference audio to codec codes (Mimi). Speaker embedding is computed
    separately after optional trimming so it matches the waveform used for ICL.

    Returns:
        ref_codes: [seq_len, 16] - RVQ codes for all 16 codebooks
        audio_data: [num_samples] mono float32 @ 24 kHz
    """
    from scipy import signal

    from models.demos.qwen3_tts.reference.functional import speech_tokenizer_encoder_forward_mimi

    print("\nEncoding reference audio...")

    # Load audio
    audio_data, sr = sf.read(audio_path)
    audio_data = torch.from_numpy(audio_data.astype(np.float32))
    if audio_data.dim() == 2:
        audio_data = audio_data.mean(dim=1)
    if sr != 24000:
        num_samples = int(len(audio_data) * 24000 / sr)
        audio_data = torch.from_numpy(signal.resample(audio_data.numpy(), num_samples).astype(np.float32))

    print(f"  Audio duration: {len(audio_data)/24000:.2f}s")

    # Encode to codes
    ref_codes = speech_tokenizer_encoder_forward_mimi(audio_data.unsqueeze(0))  # [1, 16, seq_len]
    ref_codes = ref_codes.squeeze(0).T  # [seq_len, 16]
    print(f"  Reference codes: {ref_codes.shape}")

    return ref_codes, audio_data


def extract_speaker_embedding_reference(audio_data: torch.Tensor, weights: dict) -> torch.Tensor:
    """ECAPA speaker embedding from 24 kHz mono waveform (matches TTNN pipeline)."""
    from models.demos.qwen3_tts.reference.functional import (
        SpeakerEncoderConfig,
        compute_mel_spectrogram_qwen,
        extract_speaker_encoder_weights,
        speaker_encoder_forward,
    )

    mel = compute_mel_spectrogram_qwen(audio_data)
    speaker_weights = extract_speaker_encoder_weights(weights)
    speaker_config = SpeakerEncoderConfig()
    speaker_embedding = speaker_encoder_forward(mel, speaker_weights, speaker_config)
    print(f"  Speaker embedding: {speaker_embedding.shape}")
    return speaker_embedding


def create_icl_embedding(
    target_text: str,
    ref_text: str,
    ref_codes: torch.Tensor,
    tokenizer,
    weights: dict,
    config: TTSConfig,
    speaker_embedding: torch.Tensor,
    language: str = "english",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create ICL (In-Context Learning) input embeddings for voice cloning.

    ICL Structure:
    1. Role tokens: <|im_start|>assistant\n
    2. Codec prefix: [think, think_bos, lang_id, think_eos] + tts_pad
    3. Speaker embedding (optional)
    4. Codec suffix: [pad] + tts_bos
    5. ICL prompt: (ref_text + target_text + tts_eos) + codec_pad
    6. Reference codes: codec_bos + sum(codec_embeds) + tts_pad
    7. Final: tts_pad + codec_bos

    Returns:
        inputs_embeds: [batch, seq_len, hidden_size]
        trailing_text_hidden: [batch, remaining_text_len, hidden_size] - text embeddings to add during generation
        tts_pad_embed: for generation continuation after trailing_text_hidden is exhausted
    """
    print("\nCreating ICL embeddings...")

    # Get embedding weights
    text_embed_weight = weights["talker.model.text_embedding.weight"]
    codec_embed_weight = weights["talker.model.codec_embedding.weight"]
    text_proj_fc1_weight = weights["talker.text_projection.linear_fc1.weight"]
    text_proj_fc1_bias = weights["talker.text_projection.linear_fc1.bias"]
    text_proj_fc2_weight = weights["talker.text_projection.linear_fc2.weight"]
    text_proj_fc2_bias = weights["talker.text_projection.linear_fc2.bias"]

    def project_text(text_embeds):
        h = F.linear(text_embeds, text_proj_fc1_weight, text_proj_fc1_bias)
        h = F.silu(h)
        return F.linear(h, text_proj_fc2_weight, text_proj_fc2_bias)

    # Get code predictor embeddings for codebooks 1-15
    code_predictor_embeds = []
    for i in range(config.num_code_groups - 1):
        key = f"talker.code_predictor.model.codec_embedding.{i}.weight"
        if key in weights:
            code_predictor_embeds.append(weights[key])

    # Get TTS special embeddings
    tts_tokens = torch.tensor([[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]])
    tts_embeds = F.embedding(tts_tokens, text_embed_weight)
    tts_embeds_proj = project_text(tts_embeds)
    tts_bos_embed = tts_embeds_proj[:, 0:1, :]
    tts_eos_embed = tts_embeds_proj[:, 1:2, :]
    tts_pad_embed = tts_embeds_proj[:, 2:3, :]

    # Tokenize reference and target text - just the text, no role tokens
    # Official code uses just the text content without separators
    ref_text_ids = tokenizer.encode(ref_text, add_special_tokens=False, return_tensors="pt")
    target_text_ids = tokenizer.encode(target_text, add_special_tokens=False, return_tensors="pt")

    # Role tokens from full format
    role_formatted = "<|im_start|>assistant\n"
    role_ids = tokenizer.encode(role_formatted, add_special_tokens=False, return_tensors="pt")

    print(f"  Role tokens: {role_ids.shape[1]}")
    print(f"  Reference text tokens: {ref_text_ids.shape[1]}")
    print(f"  Target text tokens: {target_text_ids.shape[1]}")
    print(f"  Reference codes: {ref_codes.shape[0]}")

    # Language ID
    lang_id = config.codec_language_ids.get(language.lower(), config.codec_language_ids["english"])

    # === Build the input embedding ===

    # 1. Role tokens
    role_embeds = F.embedding(role_ids, text_embed_weight)
    role_embeds_proj = project_text(role_embeds)

    # 2. Codec prefix: [think, think_bos, lang_id, think_eos, pad, bos]
    codec_prefix_ids = torch.tensor(
        [
            [
                config.codec_think_id,
                config.codec_think_bos_id,
                lang_id,
                config.codec_think_eos_id,
            ]
        ]
    )
    codec_suffix_ids = torch.tensor(
        [
            [
                config.codec_pad_id,
                config.codec_bos_id,
            ]
        ]
    )
    codec_prefix_embeds = F.embedding(codec_prefix_ids, codec_embed_weight)
    codec_suffix_embeds = F.embedding(codec_suffix_ids, codec_embed_weight)

    # Insert speaker embedding between prefix and suffix
    if speaker_embedding is not None:
        codec_input_embedding = torch.cat(
            [
                codec_prefix_embeds,
                speaker_embedding.view(1, 1, -1),
                codec_suffix_embeds,
            ],
            dim=1,
        )
    else:
        codec_input_embedding = torch.cat([codec_prefix_embeds, codec_suffix_embeds], dim=1)

    # Prefix combined: tts_pad * (len-2) + tts_bos + codec[:-1]
    # Then add first text token to complete the prefix
    codec_len = codec_input_embedding.shape[1]
    prefix_text = torch.cat(
        [
            tts_pad_embed.expand(-1, codec_len - 2, -1),
            tts_bos_embed,
        ],
        dim=1,
    )
    prefix_combined = prefix_text + codec_input_embedding[:, :-1, :]

    # 3. Build text_embed: ref_text + target_text + eos (projected)
    combined_text_ids = torch.cat([ref_text_ids, target_text_ids], dim=1)
    combined_text_embeds = F.embedding(combined_text_ids, text_embed_weight)
    combined_text_proj = project_text(combined_text_embeds)
    text_embed = torch.cat([combined_text_proj, tts_eos_embed], dim=1)
    text_lens = text_embed.shape[1]

    # 4. Build codec_embed: codec_bos + sum(ref_codes across 16 codebooks)
    ref_len = ref_codes.shape[0]
    codec_embeds_list = []

    for i in range(config.num_code_groups):
        code_ids = ref_codes[:, i : i + 1]  # [ref_len, 1]
        if i == 0:
            # First codebook uses talker codec_embedding
            cb_embed = F.embedding(code_ids, codec_embed_weight)  # [ref_len, 1, 2048]
        else:
            # Other codebooks use code_predictor embeddings
            cb_embed = F.embedding(code_ids, code_predictor_embeds[i - 1])  # [ref_len, 1, 2048]
        codec_embeds_list.append(cb_embed)

    # Stack and sum: [ref_len, 16, 2048] -> [ref_len, 2048] -> [1, ref_len, 2048]
    stacked_embeds = torch.cat(codec_embeds_list, dim=1)  # [ref_len, 16, 2048]
    summed_embeds = stacked_embeds.sum(dim=1)  # [ref_len, 2048]
    summed_embeds = summed_embeds.unsqueeze(0)  # [1, ref_len, 2048]

    # Prepend codec_bos
    codec_bos_embed = F.embedding(torch.tensor([[config.codec_bos_id]]), codec_embed_weight)
    codec_embed = torch.cat([codec_bos_embed, summed_embeds], dim=1)  # [1, ref_len+1, 2048]
    codec_lens = codec_embed.shape[1]

    # 5. Build ICL input embed (official streaming mode)
    # CRITICAL: Handle trailing_text_hidden correctly!
    # - If text_lens > codec_lens: remaining text tokens are added step-by-step during generation
    # - If text_lens <= codec_lens: tts_pad_embed is used for all generation steps
    if text_lens > codec_lens:
        # Truncate text to codec_lens and add, save remaining for generation
        icl_input_embed = text_embed[:, :codec_lens, :] + codec_embed
        trailing_text_hidden = text_embed[:, codec_lens:, :]  # Remaining text embeddings!
        print(f"  Text > Codec: {text_lens} > {codec_lens}")
        print(f"  trailing_text_hidden: {trailing_text_hidden.shape[1]} tokens")
    else:
        # Pad text with tts_pad to codec_lens, then add
        padding_len = codec_lens - text_lens
        text_padded = torch.cat([text_embed, tts_pad_embed.expand(-1, padding_len, -1)], dim=1)
        icl_input_embed = text_padded + codec_embed
        trailing_text_hidden = tts_pad_embed  # Just tts_pad for all steps
        print(f"  Text <= Codec: {text_lens} <= {codec_lens}")

    print(f"  Text tokens: {text_lens}, Codec positions: {codec_lens}")
    print(f"  ICL input shape: {icl_input_embed.shape}")

    # === Concatenate all parts ===
    # Official structure: role + prefix + icl_input_embed
    inputs_embeds = torch.cat(
        [
            role_embeds_proj,  # [1, 3, 2048]
            prefix_combined,  # [1, codec_len-1, 2048]
            icl_input_embed,  # [1, codec_lens, 2048]
        ],
        dim=1,
    )

    print(f"  Total input length: {inputs_embeds.shape[1]}")
    print(f"    Role: 3, Prefix: {codec_len-1}, ICL: {codec_lens}")

    return inputs_embeds, trailing_text_hidden, tts_pad_embed


def sample_token(
    logits: torch.Tensor,
    temperature: float = 0.9,
    top_k: int = 50,
    greedy: bool = False,
    repetition_penalty: float = 1.0,
    generated_tokens: list = None,
) -> int:
    """Sample next token from logits with optional repetition penalty.

    Args:
        logits: Logits tensor [vocab_size]
        temperature: Sampling temperature (ignored if greedy=True)
        top_k: Top-k filtering (ignored if greedy=True)
        greedy: Use argmax instead of sampling
        repetition_penalty: Penalty for previously generated tokens (>1.0 discourages repetition)
        generated_tokens: List of previously generated token IDs for repetition penalty
    """
    if greedy:
        return logits.argmax().item()

    # Apply repetition penalty to previously generated tokens
    if repetition_penalty != 1.0 and generated_tokens:
        for token_id in set(generated_tokens):
            if 0 <= token_id < logits.size(-1):
                if logits[token_id] > 0:
                    logits[token_id] = logits[token_id] / repetition_penalty
                else:
                    logits[token_id] = logits[token_id] * repetition_penalty

    logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()


def generate_codes(
    inputs_embeds: torch.Tensor,
    trailing_text_hidden: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    weights: dict,
    config: TTSConfig,
) -> torch.Tensor:
    """
    Generate codec tokens autoregressively using reference Talker and Code Predictor.

    CORRECT architecture (matching official):
    - For each step, generate codebook 0 with talker, then codebooks 1-15 with code predictor
    - Use SUM of ALL 16 codebook embeddings + trailing_text_hidden[step] for next talker input
    - After exhausting trailing_text_hidden, use tts_pad_embed
    - This matches the official generate loop exactly

    Args:
        inputs_embeds: Initial embedding sequence [1, seq_len, hidden_size]
        trailing_text_hidden: Remaining text embeddings [1, text_len, hidden_size] or [1, 1, hidden_size]
        tts_pad_embed: Padding embedding [1, 1, hidden_size]

    Returns:
        codes: [seq_len, 16] - all 16 codebooks
    """
    from models.demos.qwen3_tts.reference.functional import (
        Qwen3TTSCodePredictorConfig,
        Qwen3TTSConfig,
        code_predictor_forward,
        compute_mrope_frequencies,
        decoder_layer,
        extract_code_predictor_weights,
        extract_talker_weights,
        rms_norm,
    )

    print("\nGenerating codes with reference Talker + Code Predictor...")

    # Extract talker weights
    talker_weights = extract_talker_weights(weights)
    codec_head = weights["talker.codec_head.weight"]
    codec_embed_weight = weights["talker.model.codec_embedding.weight"]

    # Extract code predictor weights
    code_predictor_weights = extract_code_predictor_weights(weights)
    code_predictor_weights = {k.replace("model.", ""): v for k, v in code_predictor_weights.items()}
    code_predictor_config = Qwen3TTSCodePredictorConfig()

    # Get projection weights (2048 -> 1024)
    mtp_proj_weight = code_predictor_weights.get("small_to_mtp_projection.weight")
    mtp_proj_bias = code_predictor_weights.get("small_to_mtp_projection.bias")

    def project_to_code_predictor(x):
        if mtp_proj_weight is not None:
            return F.linear(x, mtp_proj_weight, mtp_proj_bias)
        return x

    # Get code predictor embeddings (15 codebook embeddings, 2048 dim each)
    code_pred_embeds = []
    for i in range(config.num_code_groups - 1):
        key = f"codec_embedding.{i}.weight"
        if key in code_predictor_weights:
            code_pred_embeds.append(code_predictor_weights[key])

    # Get LM heads (15 heads)
    lm_heads = []
    for i in range(config.num_code_groups - 1):
        key = f"lm_head.{i}.weight"
        if key in code_predictor_weights:
            lm_heads.append(code_predictor_weights[key])

    # Talker config
    talker_config = Qwen3TTSConfig()
    talker_config.num_hidden_layers = config.num_hidden_layers

    # Current sequence for talker
    hidden_states = inputs_embeds.clone()
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype

    # Generated codes (all 16 codebooks)
    all_codes = []

    # Track generated code 0 tokens for repetition penalty
    generated_code0_tokens = []

    # Autoregressive generation
    for step in range(config.max_new_tokens):
        current_seq_len = hidden_states.shape[1]

        # Compute RoPE
        cos, sin = compute_mrope_frequencies(config.head_dim, current_seq_len, config.rope_theta, device)
        cos = cos.to(dtype)
        sin = sin.to(dtype)

        # Causal mask
        attention_mask = (
            torch.triu(
                torch.full((current_seq_len, current_seq_len), float("-inf"), device=device, dtype=dtype),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Forward through all talker layers
        x = hidden_states
        for layer_idx in range(config.num_hidden_layers):
            layer_prefix = f"layers.{layer_idx}."
            layer_weights = {
                k.replace(layer_prefix, ""): v for k, v in talker_weights.items() if k.startswith(layer_prefix)
            }
            x = decoder_layer(x, layer_weights, cos, sin, talker_config, attention_mask=attention_mask, use_mrope=True)

        # Final norm
        x = rms_norm(x, talker_weights["norm.weight"], config.rms_norm_eps)

        # Get last hidden state
        last_hidden = x[:, -1:, :]  # [1, 1, 2048]

        # Project to vocab and sample codebook 0
        logits = F.linear(last_hidden.squeeze(1), codec_head)
        token_0 = sample_token(
            logits[0],
            config.temperature,
            config.top_k,
            config.greedy,
            config.repetition_penalty,
            generated_code0_tokens,
        )
        generated_code0_tokens.append(token_0)

        # Check for EOS
        if token_0 == config.codec_eos_id:
            print(f"  EOS at step {step}")
            break

        # Generate codebooks 1-15 using code predictor
        code_row = [token_0]

        # Build code predictor input: [hidden_state, codebook_0_embed] both projected to 1024
        hidden_proj = project_to_code_predictor(last_hidden)  # [1, 1, 1024]
        token_0_embed = F.embedding(torch.tensor([[token_0]]), codec_embed_weight)  # [1, 1, 2048]
        token_0_embed_proj = project_to_code_predictor(token_0_embed)  # [1, 1, 1024]
        cp_input = torch.cat([hidden_proj, token_0_embed_proj], dim=1)  # [1, 2, 1024]

        # Collect all 16 codebook embeddings for next step input
        all_cb_embeds = [token_0_embed]  # Start with codebook 0 in 2048 dim

        # Generate remaining 15 codebooks
        for cb_idx in range(config.num_code_groups - 1):
            cp_output = code_predictor_forward(cp_input, code_predictor_weights, code_predictor_config)
            cb_hidden = cp_output[:, -1, :]  # [1, 1024]
            cb_logits = F.linear(cb_hidden, lm_heads[cb_idx])
            cb_token = sample_token(cb_logits[0], config.temperature, config.top_k, greedy=config.greedy)
            code_row.append(cb_token)

            # Get embedding for this codebook (2048 dim, from code predictor embeddings)
            cb_embed = F.embedding(torch.tensor([[cb_token]]), code_pred_embeds[cb_idx])  # [1, 1, 2048]
            all_cb_embeds.append(cb_embed)

            # Continue code predictor with projected embedding (1024 dim)
            if cb_idx < len(code_pred_embeds) - 1:
                cb_embed_proj = project_to_code_predictor(cb_embed)  # [1, 1, 1024]
                cp_input = torch.cat([cp_input, cb_embed_proj], dim=1)

        all_codes.append(code_row)

        # Build next talker input: SUM of all 16 codebook embeddings + text embedding
        # CRITICAL: Use trailing_text_hidden[step] if available, else tts_pad_embed
        all_cb_embeds_stacked = torch.cat(all_cb_embeds, dim=1)  # [1, 16, 2048]
        next_embed = all_cb_embeds_stacked.sum(dim=1, keepdim=True)  # [1, 1, 2048]

        # Add text embedding for this step (matching official generate loop)
        trailing_len = trailing_text_hidden.shape[1]
        if step < trailing_len:
            # Use remaining text tokens from trailing_text_hidden
            next_embed = next_embed + trailing_text_hidden[:, step : step + 1, :]
        else:
            # After exhausting trailing text, use tts_pad
            next_embed = next_embed + tts_pad_embed

        # Append to talker sequence
        hidden_states = torch.cat([hidden_states, next_embed], dim=1)

        if (step + 1) % 50 == 0:
            print(f"  Generated {step + 1} tokens...")

    print(f"  Generated {len(all_codes)} code rows (each with 16 codebooks)")

    if len(all_codes) == 0:
        print("  WARNING: No tokens generated!")
        return None

    codes = torch.tensor(all_codes, dtype=torch.long)
    print(f"  Full codes shape: {codes.shape}")
    print(f"  Code 0: {codes[:, 0].tolist()}")
    torch.save(codes, "/tmp/ref_last_codes.pt")

    return codes


def decode_audio(codes: torch.Tensor, decoder_weights: dict) -> torch.Tensor:
    """Decode codes to audio using reference Speech Tokenizer Decoder."""
    from models.demos.qwen3_tts.reference.functional import (
        SpeechTokenizerDecoderConfig,
        speech_tokenizer_decoder_forward,
    )

    print("\nDecoding to audio with reference decoder...")

    # Filter out special tokens (>= 2048) - decoder expects [0, 2048)
    # Talker vocab is 3072, with special tokens (EOS=2150, BOS=2149, etc.) >= 2048
    codes_filtered = codes.clone()
    special_mask = codes_filtered >= 2048
    if special_mask.any():
        print(f"  Filtering {special_mask.sum().item()} special tokens (>= 2048)")
        codes_filtered = codes_filtered.clamp(max=2047)  # Clamp to valid range

    # Check for any invalid tokens
    print(f"  Codes range: [{codes_filtered.min().item()}, {codes_filtered.max().item()}]")

    # Reshape codes: [seq_len, 16] -> [1, 16, seq_len]
    codes_input = codes_filtered.T.unsqueeze(0)

    config = SpeechTokenizerDecoderConfig()
    audio = speech_tokenizer_decoder_forward(codes_input, decoder_weights, config)

    print(f"  Audio shape: {audio.shape}")
    print(f"  Duration: {audio.shape[-1] / 24000:.2f}s")

    return audio


def main():
    parser = argparse.ArgumentParser(description="Pure Reference TTS Demo with ICL")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--ref-audio", type=str, required=True, help="Reference audio for voice cloning")
    parser.add_argument("--ref-text", type=str, required=True, help="Text spoken in the reference audio")
    parser.add_argument("--output", type=str, default="/tmp/pure_reference_tts.wav", help="Output audio path")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--language", type=str, default="english", help="Language (english, chinese, etc.)")
    parser.add_argument(
        "--greedy", action="store_true", help="Use greedy decoding (causes repetitive output - not recommended)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty >1.0 discourages repetition (e.g., 1.1-1.3, default: 1.0)",
    )
    parser.add_argument(
        "--save-inputs",
        type=str,
        default=None,
        help="Save ICL embeddings (inputs_embeds, trailing_text_hidden, tts_pad_embed) to .pt file",
    )
    parser.add_argument(
        "--auto-trim-bleed",
        action="store_true",
        help="Automatically detect and trim reference audio bleed using Whisper",
    )
    parser.add_argument(
        "--target-word",
        type=str,
        default=None,
        help="Expected first word of target text for bleed detection (auto-extracted if not set)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Pure Reference TTS Demo (ICL Mode)")
    print("=" * 80)
    print(f"Target text: {args.text}")
    print(f"Reference audio: {args.ref_audio}")
    print(f"Reference text: {args.ref_text}")
    print()

    # Verify reference audio exists
    if not Path(args.ref_audio).exists():
        print(f"ERROR: Reference audio not found: {args.ref_audio}")
        return

    # Load weights
    main_weights, decoder_weights, model_path = load_weights()

    # Load tokenizer
    tokenizer = load_tokenizer()

    # Config
    config = TTSConfig()
    config.max_new_tokens = args.max_tokens
    config.greedy = args.greedy
    config.repetition_penalty = args.repetition_penalty

    from models.demos.qwen3_tts.demo.reference_icl_utils import trim_reference_for_icl_conditioning

    # Encode reference audio, then shorten if needed so text_len > codec_len
    ref_codes, audio_data = encode_reference_audio(args.ref_audio, main_weights)
    ref_codes, audio_data = trim_reference_for_icl_conditioning(
        ref_codes, audio_data, tokenizer, args.ref_text, args.text
    )
    speaker_embedding = extract_speaker_embedding_reference(audio_data, main_weights)

    # Create ICL embeddings
    inputs_embeds, trailing_text_hidden, tts_pad_embed = create_icl_embedding(
        target_text=args.text,
        ref_text=args.ref_text,
        ref_codes=ref_codes,
        tokenizer=tokenizer,
        weights=main_weights,
        config=config,
        speaker_embedding=speaker_embedding,
        language=args.language,
    )

    if args.save_inputs:
        torch.save(
            {
                "inputs_embeds": inputs_embeds.detach().cpu(),
                "trailing_text_hidden": trailing_text_hidden.detach().cpu(),
                "tts_pad_embed": tts_pad_embed.detach().cpu(),
            },
            args.save_inputs,
        )
        print(f"  Saved ICL inputs to: {args.save_inputs}")

    # Generate codes
    start_time = time.time()
    codes = generate_codes(inputs_embeds, trailing_text_hidden, tts_pad_embed, main_weights, config)
    gen_time = time.time() - start_time

    if codes is None:
        print("ERROR: Failed to generate codes")
        return

    # Decode to audio
    audio = decode_audio(codes, decoder_weights)

    # Save
    audio_np = audio.squeeze().detach().cpu().float().numpy()
    sf.write(args.output, audio_np, 24000)
    print(f"\nSaved to: {args.output}")

    # Auto-trim bleed if requested
    if args.auto_trim_bleed:
        from models.demos.qwen3_tts.demo.bleed_detector import detect_bleed, print_bleed_report, trim_audio

        # Extract first word from target text if not provided
        target_word = args.target_word
        if target_word is None:
            first_word = args.text.split()[0].rstrip(",.!?;:") if args.text.split() else "Hello"
            target_word = first_word

        print(f"\n  Running bleed detection (target word: '{target_word}')...")
        bleed_results = detect_bleed(args.output, target_word)
        print_bleed_report(bleed_results)

        if bleed_results["bleed_duration"] > 0.1:
            trim_time = max(0, bleed_results["bleed_duration"] - 0.1)
            trimmed_path = args.output.replace(".wav", "_trimmed.wav")
            trim_audio(args.output, trimmed_path, trim_time)
            print(f"  Bleed-trimmed audio saved to: {trimmed_path}")

            # Update main output with trimmed version
            trim_audio(args.output, args.output, trim_time)
            audio_np = audio_np[int(trim_time * 24000) :]
            print(f"  Main output updated with trimmed audio")

    # Summary
    print("\n" + "=" * 80)
    print("Summary - All Reference Components Used")
    print("=" * 80)
    print(f"Target text: {args.text}")
    print(f"Reference audio: {args.ref_audio}")
    print(f"Generated tokens: {len(codes)}")
    print(f"Audio duration: {len(audio_np) / 24000:.2f}s")
    print(f"Generation time: {gen_time:.2f}s")
    print(f"Output: {args.output}")
    print()
    print("Components used:")
    print("  ✓ Speech Tokenizer Encoder (MimiModel)")
    print("  ✓ Speaker Encoder (ECAPA-TDNN)")
    print("  ✓ Talker (28-layer transformer)")
    print("  ✓ Code Predictor (5-layer transformer)")
    print("  ✓ Speech Tokenizer Decoder (ConvNext)")
    print("=" * 80)


if __name__ == "__main__":
    main()
