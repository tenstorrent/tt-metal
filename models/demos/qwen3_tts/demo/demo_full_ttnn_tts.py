# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Full TTNN TTS Demo.

Uses TTNN for ALL model components:
- Speaker Encoder (TTNN) - extract speaker embedding
- Text Projection (TTNN) - project text embeddings
- Talker (TTNN) - 28-layer transformer + codec_head
- CodePredictor (TTNN) - 5-layer transformer + 15 LM heads

Only the Speech Tokenizer (encoder/decoder) uses reference PyTorch implementation
since it requires 1D convolutions with reflect padding not available in TTNN.

KV Cache Optimization:
- Talker uses KV cache: prefill ICL sequence once, then decode 1 token at a time
- CodePredictor uses KV cache: prefill [past_hidden, code0] once, then decode codes 1-14
- This reduces complexity from O(n^2) to O(n) for generation

Usage:
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    python models/demos/qwen3_tts/demo/demo_full_ttnn_tts.py \
        --text "Hello, how are you today?" \
        --ref-audio /path/to/reference.wav \
        --ref-text "Reference audio transcript" \
        --output /tmp/ttnn_tts_output.wav
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

import ttnn


def allocate_kv_cache(
    device,
    num_layers: int,
    batch_size: int,
    num_kv_heads: int,
    max_seq_len: int,
    head_dim: int,
) -> List[Tuple[ttnn.Tensor, ttnn.Tensor]]:
    """
    Allocate KV cache tensors for all layers.

    Args:
        device: TTNN device
        num_layers: Number of transformer layers
        batch_size: Batch size (typically 1)
        num_kv_heads: Number of KV heads
        max_seq_len: Maximum sequence length to cache
        head_dim: Dimension per head

    Returns:
        List of (k_cache, v_cache) tuples, one per layer
    """
    kv_caches = []
    for _ in range(num_layers):
        # Shape: [batch, num_kv_heads, max_seq_len, head_dim]
        k_cache = ttnn.zeros(
            [batch_size, num_kv_heads, max_seq_len, head_dim],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v_cache = ttnn.zeros(
            [batch_size, num_kv_heads, max_seq_len, head_dim],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        kv_caches.append((k_cache, v_cache))
    return kv_caches


def deallocate_kv_cache(kv_caches: List[Tuple[ttnn.Tensor, ttnn.Tensor]]):
    """Deallocate KV cache tensors."""
    for k_cache, v_cache in kv_caches:
        ttnn.deallocate(k_cache)
        ttnn.deallocate(v_cache)


@dataclass
class TTSConfig:
    """Configuration for TTS generation."""

    # Codec special tokens
    codec_bos_id: int = 2149
    codec_eos_id: int = 2150
    codec_pad_id: int = 2148
    codec_think_id: int = 2154
    codec_think_bos_id: int = 2156
    codec_think_eos_id: int = 2157

    # TTS special tokens
    tts_bos_token_id: int = 151672
    tts_eos_token_id: int = 151673
    tts_pad_token_id: int = 151671

    # Language IDs
    codec_language_ids: dict = None

    # Generation
    max_new_tokens: int = 256
    temperature: float = 0.9
    top_k: int = 50
    greedy: bool = False

    # Model dims
    hidden_size: int = 2048
    num_code_groups: int = 16

    def __post_init__(self):
        if self.codec_language_ids is None:
            self.codec_language_ids = {
                "english": 2050,
                "chinese": 2055,
                "japanese": 2058,
            }


def load_weights():
    """Load model weights."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print("Loading model weights...")
    model_path = Path(snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", allow_patterns=["*.safetensors"]))

    # Main model weights
    main_dict = {}
    for f in model_path.glob("*.safetensors"):
        if "speech_tokenizer" not in str(f):
            main_dict.update(load_file(f))
    print(f"  Loaded {len(main_dict)} main weights")

    # Speech tokenizer decoder weights (for audio synthesis)
    speech_path = model_path / "speech_tokenizer" / "model.safetensors"
    speech_dict = load_file(speech_path)
    decoder_weights = {k[8:]: v.float() for k, v in speech_dict.items() if k.startswith("decoder.")}
    print(f"  Loaded {len(decoder_weights)} decoder weights")

    return main_dict, decoder_weights


def encode_reference_audio(audio_path: str, main_weights: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode reference audio to codes and extract speaker embedding using TTNN speaker encoder.

    Returns:
        ref_codes: [seq_len, 16] - RVQ codes
        speaker_embedding: [1, 2048] - Speaker embedding (from TTNN)
    """
    from scipy import signal

    from models.demos.qwen3_tts.reference.functional import speech_tokenizer_encoder_forward_mimi

    print("\nEncoding reference audio...")

    # Load and resample audio
    audio_data, sr = sf.read(audio_path)
    audio_data = torch.from_numpy(audio_data.astype(np.float32))
    if audio_data.dim() == 2:
        audio_data = audio_data.mean(dim=1)
    if sr != 24000:
        num_samples = int(len(audio_data) * 24000 / sr)
        audio_data = torch.from_numpy(signal.resample(audio_data.numpy(), num_samples).astype(np.float32))

    print(f"  Audio duration: {len(audio_data)/24000:.2f}s")

    # Encode to codes using reference implementation (speech tokenizer encoder)
    ref_codes = speech_tokenizer_encoder_forward_mimi(audio_data.unsqueeze(0))  # [1, 16, seq_len]
    ref_codes = ref_codes.squeeze(0).T  # [seq_len, 16]
    print(f"  Reference codes: {ref_codes.shape}")

    return ref_codes, audio_data


def create_icl_embedding_ttnn(
    target_text: str,
    ref_text: str,
    ref_codes: torch.Tensor,
    speaker_embedding: torch.Tensor,
    tokenizer,
    model,
    device,
    config: TTSConfig,
    main_weights: dict,
    language: str = "english",
) -> Tuple[ttnn.Tensor, torch.Tensor, torch.Tensor, list]:
    """
    Create ICL (In-Context Learning) input embeddings using TTNN operations.

    Uses TTNN for:
    - Text embedding lookup
    - Text projection
    - Codec embedding lookup

    Returns:
        inputs_embeds: TTNN tensor [batch, 1, seq_len, hidden_size]
        trailing_text_hidden: PyTorch tensor for remaining text
        tts_pad_embed: PyTorch tensor for padding
    """
    print("\nCreating ICL embeddings (TTNN)...")

    # Get embedding weights from model for PyTorch operations
    # (needed for combining embeddings before TTNN forward)
    text_embed_weight = model.talker.text_embedding
    codec_embed_weight = model.talker.codec_embedding

    # Tokenize texts
    ref_text_ids = tokenizer.encode(ref_text, add_special_tokens=False, return_tensors="pt")
    target_text_ids = tokenizer.encode(target_text, add_special_tokens=False, return_tensors="pt")
    role_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False, return_tensors="pt")

    print(f"  Role tokens: {role_ids.shape[1]}")
    print(f"  Reference text tokens: {ref_text_ids.shape[1]}")
    print(f"  Target text tokens: {target_text_ids.shape[1]}")
    print(f"  Reference codes: {ref_codes.shape[0]}")

    # === Get embeddings using TTNN ===

    # TTS special token embeddings (text embedding + projection)
    tts_tokens = torch.tensor([[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]])
    tts_tokens_tt = ttnn.from_torch(tts_tokens, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tts_embeds_tt = model.get_text_embedding(tts_tokens_tt)
    tts_embeds_proj_tt = model.project_text(tts_embeds_tt)
    tts_embeds_proj = ttnn.to_torch(tts_embeds_proj_tt).squeeze(1).float()  # [1, 3, 2048]

    tts_bos_embed = tts_embeds_proj[:, 0:1, :]
    tts_eos_embed = tts_embeds_proj[:, 1:2, :]
    tts_pad_embed = tts_embeds_proj[:, 2:3, :]

    # Role token embeddings
    role_ids_tt = ttnn.from_torch(role_ids, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    role_embeds_tt = model.get_text_embedding(role_ids_tt)
    role_embeds_proj_tt = model.project_text(role_embeds_tt)
    role_embeds_proj = ttnn.to_torch(role_embeds_proj_tt).squeeze(1).float()

    # Combined text embeddings (ref_text + target_text)
    combined_text_ids = torch.cat([ref_text_ids, target_text_ids], dim=1)
    combined_text_ids_tt = ttnn.from_torch(
        combined_text_ids, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    combined_text_embeds_tt = model.get_text_embedding(combined_text_ids_tt)
    combined_text_proj_tt = model.project_text(combined_text_embeds_tt)
    combined_text_proj = ttnn.to_torch(combined_text_proj_tt).squeeze(1).float()

    # Full text embedding with EOS
    text_embed = torch.cat([combined_text_proj, tts_eos_embed], dim=1)
    text_lens = text_embed.shape[1]

    # === Codec embeddings ===
    lang_id = config.codec_language_ids.get(language.lower(), config.codec_language_ids["english"])

    # Codec prefix tokens
    codec_prefix_ids = torch.tensor(
        [[config.codec_think_id, config.codec_think_bos_id, lang_id, config.codec_think_eos_id]]
    )
    codec_suffix_ids = torch.tensor([[config.codec_pad_id, config.codec_bos_id]])

    codec_prefix_ids_tt = ttnn.from_torch(
        codec_prefix_ids, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    codec_suffix_ids_tt = ttnn.from_torch(
        codec_suffix_ids, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    codec_prefix_embeds = ttnn.to_torch(model.get_codec_embedding(codec_prefix_ids_tt)).float()
    codec_suffix_embeds = ttnn.to_torch(model.get_codec_embedding(codec_suffix_ids_tt)).float()

    # Ensure 3D shape [batch, seq, hidden]
    if codec_prefix_embeds.dim() == 2:
        codec_prefix_embeds = codec_prefix_embeds.unsqueeze(0)
    if codec_suffix_embeds.dim() == 2:
        codec_suffix_embeds = codec_suffix_embeds.unsqueeze(0)

    # Insert speaker embedding
    codec_input_embedding = torch.cat(
        [
            codec_prefix_embeds,
            speaker_embedding.view(1, 1, -1),
            codec_suffix_embeds,
        ],
        dim=1,
    )

    # Prefix combined
    codec_len = codec_input_embedding.shape[1]
    prefix_text = torch.cat(
        [
            tts_pad_embed.expand(-1, codec_len - 2, -1),
            tts_bos_embed,
        ],
        dim=1,
    )
    prefix_combined = prefix_text + codec_input_embedding[:, :-1, :]

    # === Reference codes embedding (sum of all 16 codebooks) ===
    # Get CodePredictor embeddings for codebooks 1-15 from main_weights
    code_pred_embeds = []
    for i in range(config.num_code_groups - 1):
        key = f"talker.code_predictor.model.codec_embedding.{i}.weight"
        if key in main_weights:
            code_pred_embeds.append(main_weights[key].float())
        else:
            print(f"  WARNING: Missing CodePredictor embedding {key}")
    print(f"  Loaded {len(code_pred_embeds)} CodePredictor embeddings")

    # Get main codec embedding from TTNN tensor
    codec_embed_torch = ttnn.to_torch(codec_embed_weight).squeeze(0).squeeze(0).float()

    # Build reference code embeddings using proper codebook embeddings
    ref_len = ref_codes.shape[0]
    codec_embeds_list = []

    for i in range(config.num_code_groups):
        code_ids = ref_codes[:, i : i + 1]  # [ref_len, 1]
        if i == 0:
            # First codebook uses main talker codec_embedding
            cb_embed = F.embedding(code_ids, codec_embed_torch)
        else:
            # Codebooks 1-15 use CodePredictor embeddings
            if i - 1 < len(code_pred_embeds):
                cb_embed = F.embedding(code_ids, code_pred_embeds[i - 1])
            else:
                cb_embed = F.embedding(code_ids, codec_embed_torch)  # Fallback
        codec_embeds_list.append(cb_embed)

    stacked_embeds = torch.cat(codec_embeds_list, dim=1)  # [ref_len, 16, 2048]
    summed_embeds = stacked_embeds.sum(dim=1).unsqueeze(0)  # [1, ref_len, 2048]

    # Prepend codec_bos
    codec_bos_ids = torch.tensor([[config.codec_bos_id]])
    codec_bos_ids_tt = ttnn.from_torch(codec_bos_ids, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    codec_bos_embed = ttnn.to_torch(model.get_codec_embedding(codec_bos_ids_tt)).float()  # [1, 1, 2048]
    if codec_bos_embed.dim() == 2:
        codec_bos_embed = codec_bos_embed.unsqueeze(1)  # [1, 1, 2048]

    codec_embed = torch.cat([codec_bos_embed, summed_embeds], dim=1)
    codec_lens = codec_embed.shape[1]

    # === Build ICL input ===
    if text_lens > codec_lens:
        icl_input_embed = text_embed[:, :codec_lens, :] + codec_embed
        trailing_text_hidden = text_embed[:, codec_lens:, :]
        print(f"  Text > Codec: {text_lens} > {codec_lens}")
        print(f"  trailing_text_hidden: {trailing_text_hidden.shape[1]} tokens")
    else:
        padding_len = codec_lens - text_lens
        text_padded = torch.cat([text_embed, tts_pad_embed.expand(-1, padding_len, -1)], dim=1)
        icl_input_embed = text_padded + codec_embed
        trailing_text_hidden = tts_pad_embed
        print(f"  Text <= Codec: {text_lens} <= {codec_lens}")

    # Concatenate all parts
    inputs_embeds = torch.cat(
        [
            role_embeds_proj,
            prefix_combined,
            icl_input_embed,
        ],
        dim=1,
    )

    print(f"  Total input length: {inputs_embeds.shape[1]}")

    # Convert to TTNN
    inputs_embeds_tt = ttnn.from_torch(
        inputs_embeds.unsqueeze(1),  # [1, 1, seq_len, 2048]
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return inputs_embeds_tt, trailing_text_hidden, tts_pad_embed, code_pred_embeds


def sample_token(logits: torch.Tensor, temperature: float = 0.9, top_k: int = 50, greedy: bool = False) -> int:
    """Sample next token from logits."""
    if greedy:
        return logits.argmax().item()
    logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()


def generate_codes_ttnn(
    model,
    device,
    inputs_embeds_tt: ttnn.Tensor,
    trailing_text_hidden: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    code_pred_embeds: list,
    config: TTSConfig,
    use_kv_cache: bool = True,
) -> torch.Tensor:
    """
    Generate codec tokens autoregressively using TTNN Talker and CodePredictor.

    IMPORTANT: CodePredictor generates codes 1-15 AUTOREGRESSIVELY:
    - Each code is conditioned on all previous codes
    - Each code group uses its own embedding table and LM head
    - This matches the official qwen_tts implementation

    KV Cache Optimization:
    - Talker: Prefill ICL sequence once, then decode 1 token at a time
    - CodePredictor: Prefill [past_hidden, code0] once per frame, decode codes 1-14
    - Reduces complexity from O(n^2) to O(n)

    Args:
        model: Qwen3TTS model (TTNN)
        device: TTNN device
        inputs_embeds_tt: Initial embeddings [1, 1, seq_len, 2048]
        trailing_text_hidden: Remaining text embeddings
        tts_pad_embed: Padding embedding
        code_pred_embeds: List of CodePredictor embedding weights (for codes 1-15)
        config: TTS configuration
        use_kv_cache: Whether to use KV cache optimization (default True)

    Returns:
        codes: [seq_len, 16] - all 16 codebooks
    """
    from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat

    mode_str = "with KV cache" if use_kv_cache else "without KV cache"
    print(f"\nGenerating codes with TTNN ({mode_str})...")

    # Get transformation matrices
    talker_trans_mat = get_transformation_mat(model.talker_config.head_dim, device)
    cp_trans_mat = get_transformation_mat(model.code_predictor_config.head_dim, device)

    # Get codec embedding for building next input (Talker embedding for code 0)
    codec_embed_torch = ttnn.to_torch(model.talker.codec_embedding).squeeze(0).squeeze(0).float()

    all_codes = []
    prefill_len = inputs_embeds_tt.shape[2]
    max_talker_seq_len = prefill_len + config.max_new_tokens + 16  # Extra buffer

    # === Allocate Talker KV cache ===
    talker_kv_caches = None
    if use_kv_cache:
        print(
            f"  Allocating Talker KV cache ({model.talker_config.num_hidden_layers} layers, max_seq={max_talker_seq_len})"
        )
        talker_kv_caches = allocate_kv_cache(
            device=device,
            num_layers=model.talker_config.num_hidden_layers,
            batch_size=1,
            num_kv_heads=model.talker_config.num_key_value_heads,
            max_seq_len=max_talker_seq_len,
            head_dim=model.talker_config.head_dim,
        )

    try:
        # === PREFILL: Run Talker on full ICL sequence ===
        prefill_seq_len = inputs_embeds_tt.shape[2]
        position_ids = torch.arange(prefill_seq_len)
        talker_cos, talker_sin = get_rope_tensors(
            device, model.talker_config.head_dim, prefill_seq_len, position_ids, model.talker_config.rope_theta
        )

        talker_hidden_tt, talker_kv_caches = model.talker.forward_from_hidden(
            inputs_embeds_tt,
            talker_cos,
            talker_sin,
            talker_trans_mat,
            attention_mask=None,
            kv_caches=talker_kv_caches,
            start_pos=0,
            mode="prefill",
        )

        # Get codec logits for code 0 (from prefill)
        codec_logits = model.talker.get_codec_logits(talker_hidden_tt)
        codec_logits_torch = ttnn.to_torch(codec_logits)[:, :, -1, :].squeeze().float()
        token_0 = sample_token(codec_logits_torch, config.temperature, config.top_k, config.greedy)

        if token_0 == config.codec_eos_id:
            print("  EOS at prefill")
            return None

        # Track current position in Talker sequence
        talker_pos = prefill_seq_len

        for step in range(config.max_new_tokens):
            # === CodePredictor: Generate codes 1-15 for this frame ===
            past_hidden_torch = ttnn.to_torch(talker_hidden_tt)[:, :, -1:, :].float()  # [1, 1, 1, 2048]

            # Get code 0 embedding
            code0_embed = F.embedding(torch.tensor([[token_0]]), codec_embed_torch)
            code0_embed = code0_embed.unsqueeze(1)  # [1, 1, 1, 2048]

            # Initial CodePredictor input: [past_hidden, code0_embed]
            cp_input = torch.cat([past_hidden_torch, code0_embed], dim=2)  # [1, 1, 2, 2048]

            code_row = [token_0]

            # Allocate CodePredictor KV cache for this frame
            max_cp_seq_len = 32  # past_hidden + 16 codes
            cp_kv_caches = None
            if use_kv_cache:
                cp_kv_caches = allocate_kv_cache(
                    device=device,
                    num_layers=model.code_predictor_config.num_hidden_layers,
                    batch_size=1,
                    num_kv_heads=model.code_predictor_config.num_key_value_heads,
                    max_seq_len=max_cp_seq_len,
                    head_dim=model.code_predictor_config.head_dim,
                )

            try:
                # PREFILL CodePredictor with [past_hidden, code0_embed]
                cp_prefill_len = cp_input.shape[2]
                cp_position_ids = torch.arange(cp_prefill_len)
                cp_cos, cp_sin = get_rope_tensors(
                    device,
                    model.code_predictor_config.head_dim,
                    cp_prefill_len,
                    cp_position_ids,
                    model.code_predictor_config.rope_theta,
                )

                cp_input_tt = ttnn.from_torch(
                    cp_input,
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                # Run CodePredictor prefill (predicts code 1)
                logits_tt, cp_kv_caches = model.code_predictor.forward_single_step(
                    cp_input_tt,
                    cp_cos,
                    cp_sin,
                    cp_trans_mat,
                    generation_step=1,  # Predicting code 1
                    kv_caches=cp_kv_caches,
                    start_pos=0,
                    mode="prefill",
                )

                logits_torch = ttnn.to_torch(logits_tt)[:, :, -1, :].squeeze().float()
                token = sample_token(logits_torch, config.temperature, config.top_k, config.greedy)
                code_row.append(token)

                cp_pos = cp_prefill_len  # Current position in CodePredictor

                # DECODE codes 2-15
                for code_idx in range(2, config.num_code_groups):
                    # Get embedding for previous code
                    prev_code_idx = code_idx - 1  # Index of embedding table for previous code
                    if prev_code_idx - 1 < len(code_pred_embeds) and code_pred_embeds[prev_code_idx - 1] is not None:
                        next_code_embed = F.embedding(torch.tensor([[token]]), code_pred_embeds[prev_code_idx - 1])
                    else:
                        next_code_embed = F.embedding(torch.tensor([[token]]), codec_embed_torch)

                    next_code_embed = next_code_embed.unsqueeze(1)  # [1, 1, 1, 2048]

                    if use_kv_cache:
                        # DECODE mode: single token with KV cache
                        cp_position_ids = torch.tensor([cp_pos])
                        cp_cos, cp_sin = get_rope_tensors(
                            device,
                            model.code_predictor_config.head_dim,
                            1,
                            cp_position_ids,
                            model.code_predictor_config.rope_theta,
                        )

                        cp_input_tt = ttnn.from_torch(
                            next_code_embed,
                            device=device,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )

                        logits_tt, cp_kv_caches = model.code_predictor.forward_single_step(
                            cp_input_tt,
                            cp_cos,
                            cp_sin,
                            cp_trans_mat,
                            generation_step=code_idx,
                            kv_caches=cp_kv_caches,
                            start_pos=cp_pos,
                            mode="decode",
                        )
                        cp_pos += 1
                    else:
                        # No KV cache: append and recompute full sequence
                        cp_input = torch.cat([cp_input, next_code_embed], dim=2)
                        cp_seq_len = cp_input.shape[2]

                        cp_position_ids = torch.arange(cp_seq_len)
                        cp_cos, cp_sin = get_rope_tensors(
                            device,
                            model.code_predictor_config.head_dim,
                            cp_seq_len,
                            cp_position_ids,
                            model.code_predictor_config.rope_theta,
                        )

                        cp_input_tt = ttnn.from_torch(
                            cp_input,
                            device=device,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )

                        logits_tt, _ = model.code_predictor.forward_single_step(
                            cp_input_tt,
                            cp_cos,
                            cp_sin,
                            cp_trans_mat,
                            generation_step=code_idx,
                        )

                    logits_torch = ttnn.to_torch(logits_tt)[:, :, -1, :].squeeze().float()
                    token = sample_token(logits_torch, config.temperature, config.top_k, config.greedy)
                    code_row.append(token)

            finally:
                # Deallocate CodePredictor KV cache for this frame
                if cp_kv_caches is not None:
                    deallocate_kv_cache(cp_kv_caches)

            all_codes.append(code_row)

            # === Build next Talker input embedding ===
            all_cb_embeds = []
            for i, tok in enumerate(code_row):
                if i == 0:
                    cb_embed = F.embedding(torch.tensor([[tok]]), codec_embed_torch)
                else:
                    if i - 1 < len(code_pred_embeds) and code_pred_embeds[i - 1] is not None:
                        cb_embed = F.embedding(torch.tensor([[tok]]), code_pred_embeds[i - 1])
                    else:
                        cb_embed = F.embedding(torch.tensor([[tok]]), codec_embed_torch)
                all_cb_embeds.append(cb_embed)

            stacked = torch.cat(all_cb_embeds, dim=1)
            next_embed = stacked.sum(dim=1, keepdim=True)  # [1, 1, 2048]

            # Add text embedding
            trailing_len = trailing_text_hidden.shape[1]
            if step < trailing_len:
                next_embed = next_embed + trailing_text_hidden[:, step : step + 1, :]
            else:
                next_embed = next_embed + tts_pad_embed

            next_embed = next_embed.unsqueeze(1)  # [1, 1, 1, 2048]

            # === DECODE: Run Talker on single token ===
            if use_kv_cache:
                position_ids = torch.tensor([talker_pos])
                talker_cos, talker_sin = get_rope_tensors(
                    device, model.talker_config.head_dim, 1, position_ids, model.talker_config.rope_theta
                )

                next_embed_tt = ttnn.from_torch(
                    next_embed,
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                talker_hidden_tt, talker_kv_caches = model.talker.forward_from_hidden(
                    next_embed_tt,
                    talker_cos,
                    talker_sin,
                    talker_trans_mat,
                    attention_mask=None,
                    kv_caches=talker_kv_caches,
                    start_pos=talker_pos,
                    mode="decode",
                )
                talker_pos += 1
            else:
                # No KV cache: concatenate and recompute full sequence
                current_torch = ttnn.to_torch(inputs_embeds_tt).float()
                combined = torch.cat([current_torch, next_embed], dim=2)

                inputs_embeds_tt = ttnn.from_torch(
                    combined,
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                seq_len = inputs_embeds_tt.shape[2]
                position_ids = torch.arange(seq_len)
                talker_cos, talker_sin = get_rope_tensors(
                    device, model.talker_config.head_dim, seq_len, position_ids, model.talker_config.rope_theta
                )

                talker_hidden_tt, _ = model.talker.forward_from_hidden(
                    inputs_embeds_tt,
                    talker_cos,
                    talker_sin,
                    talker_trans_mat,
                    attention_mask=None,
                )

            # Get next code 0
            codec_logits = model.talker.get_codec_logits(talker_hidden_tt)
            codec_logits_torch = ttnn.to_torch(codec_logits)[:, :, -1, :].squeeze().float()
            token_0 = sample_token(codec_logits_torch, config.temperature, config.top_k, config.greedy)

            if token_0 == config.codec_eos_id:
                print(f"  EOS at step {step + 1}")
                break

            if (step + 1) % 20 == 0:
                print(f"  Generated {step + 1} frames...")

    finally:
        # Deallocate Talker KV cache
        if talker_kv_caches is not None:
            deallocate_kv_cache(talker_kv_caches)

    print(f"  Generated {len(all_codes)} code frames")

    if len(all_codes) == 0:
        return None

    return torch.tensor(all_codes, dtype=torch.long)


def decode_audio(codes: torch.Tensor, decoder_weights: dict) -> torch.Tensor:
    """Decode codes to audio using reference Speech Tokenizer Decoder."""
    from models.demos.qwen3_tts.reference.functional import (
        SpeechTokenizerDecoderConfig,
        speech_tokenizer_decoder_forward,
    )

    print("\nDecoding to audio...")

    # Filter special tokens
    codes_filtered = codes.clone().clamp(max=2047)

    # Reshape: [seq_len, 16] -> [1, 16, seq_len]
    codes_input = codes_filtered.T.unsqueeze(0)

    config = SpeechTokenizerDecoderConfig()
    audio = speech_tokenizer_decoder_forward(codes_input, decoder_weights, config)

    print(f"  Audio duration: {audio.shape[-1] / 24000:.2f}s")

    return audio


def run_full_ttnn_tts(
    text: str,
    ref_audio: str,
    ref_text: str,
    output_path: str = "/tmp/ttnn_tts_output.wav",
    max_new_tokens: int = 256,
    device_id: int = 0,
    language: str = "english",
    greedy: bool = False,
    use_kv_cache: bool = True,
):
    """Run full TTNN TTS pipeline."""
    print("=" * 80)
    print("Full TTNN TTS Demo")
    print("=" * 80)
    print(f"Text: {text}")
    print(f"Reference: {ref_audio}")
    print(f"Max tokens: {max_new_tokens}")
    print(f"KV cache: {'enabled' if use_kv_cache else 'disabled'}")
    print()

    timings = {}

    # Load weights
    load_start = time.time()
    main_weights, decoder_weights = load_weights()
    timings["load_weights"] = time.time() - load_start

    # Load tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", trust_remote_code=True)

    # Open device
    print(f"\nOpening TT device {device_id}...")
    device = ttnn.open_device(device_id=device_id)

    try:
        # Initialize TTNN model
        print("\nInitializing TTNN model...")
        init_start = time.time()

        from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS

        model = Qwen3TTS(device=device, state_dict=main_weights)

        timings["model_init"] = time.time() - init_start
        print(f"  Model initialized in {timings['model_init']:.2f}s")

        # Config
        config = TTSConfig()
        config.max_new_tokens = max_new_tokens
        config.greedy = greedy

        # Encode reference audio (speech tokenizer encoder - reference impl)
        encode_start = time.time()
        ref_codes, audio_data = encode_reference_audio(ref_audio, main_weights)
        timings["encode_ref"] = time.time() - encode_start

        # Extract speaker embedding (TTNN)
        spk_start = time.time()
        speaker_embedding = model.extract_speaker_embedding(audio_data)
        timings["speaker_embed"] = time.time() - spk_start
        print(f"  Speaker embedding: {speaker_embedding.shape} (extracted with TTNN)")

        # Create ICL embeddings (TTNN)
        icl_start = time.time()
        inputs_embeds_tt, trailing_text_hidden, tts_pad_embed, code_pred_embeds = create_icl_embedding_ttnn(
            target_text=text,
            ref_text=ref_text,
            ref_codes=ref_codes,
            speaker_embedding=speaker_embedding,
            tokenizer=tokenizer,
            model=model,
            device=device,
            config=config,
            main_weights=main_weights,
            language=language,
        )
        timings["icl_embed"] = time.time() - icl_start

        # Generate codes (TTNN)
        gen_start = time.time()
        codes = generate_codes_ttnn(
            model=model,
            device=device,
            inputs_embeds_tt=inputs_embeds_tt,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            code_pred_embeds=code_pred_embeds,
            config=config,
            use_kv_cache=use_kv_cache,
        )
        timings["generation"] = time.time() - gen_start

        if codes is None:
            print("ERROR: Failed to generate codes")
            return

        # Decode to audio (reference impl)
        decode_start = time.time()
        audio = decode_audio(codes, decoder_weights)
        timings["decode"] = time.time() - decode_start

        # Save audio
        audio_np = audio.squeeze().detach().cpu().float().numpy()
        sf.write(output_path, audio_np, 24000)

        # Summary
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"\n{'Phase':<30} {'Time (ms)':<15} {'Component'}")
        print("-" * 70)
        print(f"{'Load weights':<30} {timings['load_weights']*1000:>10.1f}   PyTorch")
        print(f"{'Model init':<30} {timings['model_init']*1000:>10.1f}   TTNN")
        print(f"{'Encode ref audio':<30} {timings['encode_ref']*1000:>10.1f}   Reference (Speech Tok Enc)")
        print(f"{'Speaker embedding':<30} {timings['speaker_embed']*1000:>10.1f}   TTNN")
        print(f"{'ICL embedding':<30} {timings['icl_embed']*1000:>10.1f}   TTNN")
        num_frames = len(codes) if codes is not None else 0
        print(f"{'Generation (' + str(num_frames) + ' frames)':<30} {timings['generation']*1000:>10.1f}   TTNN")
        print(f"{'Decode audio':<30} {timings['decode']*1000:>10.1f}   Reference (Speech Tok Dec)")
        print("-" * 70)

        total_gen_time = timings["generation"]
        tokens_per_sec = len(codes) / total_gen_time if total_gen_time > 0 else 0
        print(f"{'Throughput':<30} {tokens_per_sec:>10.2f}   frames/sec")
        print(f"{'Avg per frame':<30} {total_gen_time*1000/len(codes):>10.2f}   ms/frame")

        print(f"\nOutput saved to: {output_path}")
        print(f"Audio duration: {len(audio_np) / 24000:.2f}s")
        print("=" * 80)

    finally:
        ttnn.close_device(device)
        print("\nDevice closed")


def main():
    parser = argparse.ArgumentParser(description="Full TTNN TTS Demo")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--ref-audio", type=str, required=True, help="Reference audio path")
    parser.add_argument("--ref-text", type=str, required=True, help="Reference audio transcript")
    parser.add_argument("--output", type=str, default="/tmp/ttnn_tts_output.wav", help="Output path")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--device-id", type=int, default=0, help="TT device ID")
    parser.add_argument("--language", type=str, default="english", help="Language")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    parser.add_argument("--no-kv-cache", action="store_true", help="Disable KV cache (slower)")
    args = parser.parse_args()

    run_full_ttnn_tts(
        text=args.text,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        output_path=args.output,
        max_new_tokens=args.max_tokens,
        device_id=args.device_id,
        language=args.language,
        greedy=args.greedy,
        use_kv_cache=not args.no_kv_cache,
    )


if __name__ == "__main__":
    main()
