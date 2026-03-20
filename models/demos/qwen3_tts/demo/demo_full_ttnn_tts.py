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
from models.demos.qwen3_tts.demo.reference_icl_utils import trim_reference_for_icl_conditioning


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
    repetition_penalty: float = 1.0  # >1.0 discourages repetition (e.g., 1.1-1.3)

    # Post-processing: trim reference echo from start of generated audio.
    # ICL TTS models briefly reproduce the reference speaker's last word
    # before transitioning to the target text. Trim those leading codec frames.
    # Default 4 frames = 0.32s at 12.5fps. Set to 0 to disable.
    trim_codec_frames: int = 4

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


def encode_reference_audio(
    audio_path: str, main_weights: dict, cache_path: str = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode reference audio to codes and extract speaker embedding using TTNN speaker encoder.

    Caches results to disk so the slow CPU MimiModel only runs once.
    Cache path defaults to <audio_path>.refcache.pt

    Returns:
        ref_codes: [seq_len, 16] - RVQ codes
        audio_data: [num_samples] - raw waveform (for TTNN speaker encoder)
    """
    from pathlib import Path

    if cache_path is None:
        cache_path = str(Path(audio_path).with_suffix("")) + ".refcache.pt"

    # Load from cache if available
    if Path(cache_path).exists():
        print(f"\nLoading cached reference encoding from {cache_path}")
        cached = torch.load(cache_path, weights_only=True)
        ref_codes = cached["ref_codes"]
        audio_data = cached["audio_data"]
        print(f"  Reference codes: {ref_codes.shape}  (loaded from cache)")
        print(f"  Audio duration: {len(audio_data)/24000:.2f}s")
        return ref_codes, audio_data

    from scipy import signal

    from models.demos.qwen3_tts.reference.functional import speech_tokenizer_encoder_forward_mimi

    print("\nEncoding reference audio (first run - will cache result)...")

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

    # Save to cache
    torch.save({"ref_codes": ref_codes, "audio_data": audio_data}, cache_path)
    print(f"  Cached to {cache_path}")

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


def sample_token(
    logits: torch.Tensor,
    temperature: float = 0.9,
    top_k: int = 50,
    greedy: bool = False,
    repetition_penalty: float = 1.0,
    generated_tokens: List[int] = None,
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


SUPPORTED_PREFILL_LENS = [32, 64, 96, 128, 192, 256, 384, 512]


def get_padded_prefill_len(seq_len: int) -> int:
    """Get the smallest supported prefill length >= seq_len."""
    for bucket in SUPPORTED_PREFILL_LENS:
        if seq_len <= bucket:
            return bucket
    return SUPPORTED_PREFILL_LENS[-1]


def build_prefill_attn_mask(
    real_seq_len: int,
    padded_seq_len: int,
    max_seq_len: int,
    num_heads: int,
) -> torch.Tensor:
    """Build a causal + padding mask for traced Talker prefill.

    Returns float32 tensor [1, num_heads, padded_seq_len, max_seq_len]:
      - Row i can attend to columns 0..min(i, real_seq_len-1)
      - All other positions (padding columns, future columns, empty cache) are -inf
    """
    mask = torch.full((1, num_heads, padded_seq_len, max_seq_len), float("-inf"))
    for i in range(padded_seq_len):
        end_j = min(i + 1, real_seq_len)
        mask[0, :, i, :end_j] = 0.0
    return mask


def generate_codes_ttnn(
    model,
    device,
    inputs_embeds_tt: ttnn.Tensor,
    trailing_text_hidden: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    code_pred_embeds: list,
    config: TTSConfig,
    use_kv_cache: bool = True,
    use_trace: bool = True,
) -> torch.Tensor:
    """
    Generate codec tokens autoregressively using TTNN Talker and CodePredictor.

    Flow (SpeechT5-style):
    Following the tt-metal reference pattern (generator.py / simple_text_demo.py):
    1. Pad input to bucket size + build causal+padding mask
    2. Warmup: compile ALL TTNN kernels with exact shapes (dummy data)
    3. Allocate persistent KV caches
    4. Run Talker prefill NON-TRACED (fills KV cache, gets first token)
    5. Capture decode traces: Talker decode, CP prefill, CP decode x13
    6. Generation loop: execute traces only (measured inference)

    Args:
        model: Qwen3TTS model (TTNN)
        device: TTNN device
        inputs_embeds_tt: Initial embeddings [1, 1, seq_len, 2048]
        trailing_text_hidden: Remaining text embeddings
        tts_pad_embed: Padding embedding
        code_pred_embeds: List of CodePredictor embedding weights (for codes 1-15)
        config: TTS configuration
        use_kv_cache: Whether to use KV cache optimization (default True)
        use_trace: Whether to use trace (default True)

    Returns:
        codes: [seq_len, 16] - all 16 codebooks
    """
    from models.demos.qwen3_tts.tt.rope import compute_rope_frequencies, get_rope_tensors, get_transformation_mat

    mode_str = "with KV cache" if use_kv_cache else "without KV cache"
    print(f"\nGenerating codes with TTNN ({mode_str})...")

    # Get transformation matrices
    talker_trans_mat = get_transformation_mat(model.talker_config.head_dim, device)
    cp_trans_mat = get_transformation_mat(model.code_predictor_config.head_dim, device)

    # Get codec embedding for building next input (Talker embedding for code 0)
    codec_embed_torch = ttnn.to_torch(model.talker.codec_embedding).squeeze(0).squeeze(0).float()

    all_codes = []
    real_seq_len = inputs_embeds_tt.shape[2]
    talker_h = model.talker_config.hidden_size
    head_dim = model.talker_config.head_dim
    _talker_num_heads = model.talker_config.num_attention_heads
    cp_head_dim = model.code_predictor_config.head_dim
    cp_rope_theta = model.code_predictor_config.rope_theta
    _cp_num_heads = model.code_predictor_config.num_attention_heads
    max_cp_seq_len = 32

    # === STEP 1: Pad input to bucket size ===
    padded_seq_len = get_padded_prefill_len(real_seq_len)
    print(f"  Input padding: {real_seq_len} -> {padded_seq_len} (bucket)")

    if padded_seq_len > real_seq_len:
        pad_len = padded_seq_len - real_seq_len
        pad_zeros = ttnn.from_torch(
            torch.zeros(1, 1, pad_len, talker_h, dtype=torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        inputs_embeds_tt = ttnn.concat([inputs_embeds_tt, pad_zeros], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(pad_zeros)

    _TILE = 32
    max_talker_seq_len = (((padded_seq_len + config.max_new_tokens + 16) + _TILE - 1) // _TILE) * _TILE

    # Pre-compute RoPE tables
    _max_rope_pos = max_talker_seq_len + config.max_new_tokens + 50
    talker_cos_table, talker_sin_table = compute_rope_frequencies(
        head_dim, _max_rope_pos, model.talker_config.rope_theta
    )
    cp_cos_table, cp_sin_table = compute_rope_frequencies(cp_head_dim, max_cp_seq_len + 5, cp_rope_theta)

    # === STEP 2: Warmup — compile ALL kernels with exact shapes ===
    print(f"  Warmup: compiling kernels for padded_prefill={padded_seq_len}, decode=1 ...")

    # --- Talker prefill warmup: standard prefill (no mask, attention over seq_len only) ---
    wu_pf = ttnn.from_torch(
        torch.zeros(1, 1, padded_seq_len, talker_h),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    wu_pf_pos = torch.arange(padded_seq_len)
    wu_pf_cos, wu_pf_sin = get_rope_tensors(device, head_dim, padded_seq_len, wu_pf_pos, model.talker_config.rope_theta)
    wu_talker_kv = allocate_kv_cache(
        device=device,
        num_layers=model.talker_config.num_hidden_layers,
        batch_size=1,
        num_kv_heads=model.talker_config.num_key_value_heads,
        max_seq_len=max_talker_seq_len,
        head_dim=head_dim,
    )
    wu_pf_hidden, wu_talker_kv = model.talker.forward_from_hidden(
        wu_pf,
        wu_pf_cos,
        wu_pf_sin,
        talker_trans_mat,
        kv_caches=wu_talker_kv,
        start_pos=0,
        mode="prefill",
    )
    _ = model.talker.get_codec_logits(wu_pf_hidden)

    # --- Talker decode warmup: paged_update_cache + full-cache attention ---
    wu_dc = ttnn.from_torch(
        torch.zeros(1, 1, 1, talker_h),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    wu_dc_pos = torch.tensor([padded_seq_len])
    wu_dc_cos, wu_dc_sin = get_rope_tensors(device, head_dim, 1, wu_dc_pos, model.talker_config.rope_theta)
    wu_cur_pos = ttnn.from_torch(
        torch.tensor([padded_seq_len], dtype=torch.int32),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    wu_decode_mask = ttnn.from_torch(
        torch.full((1, _talker_num_heads, 1, max_talker_seq_len), float("-inf")),
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    wu_talker_hidden = model.talker.forward_from_hidden(
        wu_dc,
        wu_dc_cos,
        wu_dc_sin,
        talker_trans_mat,
        kv_caches=wu_talker_kv,
        start_pos=padded_seq_len,
        mode="decode",
        cur_pos_tensor=wu_cur_pos,
        decode_attn_mask=wu_decode_mask,
    )[0]
    _ = model.talker.get_codec_logits(wu_talker_hidden)
    deallocate_kv_cache(wu_talker_kv)
    ttnn.deallocate(wu_cur_pos)
    ttnn.deallocate(wu_decode_mask)

    # --- CP warmups (same as before) ---
    wu_cp2 = ttnn.from_torch(
        torch.zeros(1, 1, 2, talker_h),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    wu_cp2_pos = torch.arange(2)
    wu_cp2_cos, wu_cp2_sin = get_rope_tensors(device, cp_head_dim, 2, wu_cp2_pos, cp_rope_theta)
    wu_cp_kv = allocate_kv_cache(
        device=device,
        num_layers=model.code_predictor_config.num_hidden_layers,
        batch_size=1,
        num_kv_heads=model.code_predictor_config.num_key_value_heads,
        max_seq_len=max_cp_seq_len,
        head_dim=cp_head_dim,
    )
    wu_cp_prefill_mask_host = torch.full((1, _cp_num_heads, 2, max_cp_seq_len), float("-inf"))
    wu_cp_prefill_mask_host[0, :, 0, 0] = 0.0
    wu_cp_prefill_mask_host[0, :, 1, 0:2] = 0.0
    wu_cp_prefill_mask_tt = ttnn.from_torch(
        wu_cp_prefill_mask_host,
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    model.code_predictor.forward_single_step(
        wu_cp2,
        wu_cp2_cos,
        wu_cp2_sin,
        cp_trans_mat,
        generation_step=1,
        kv_caches=wu_cp_kv,
        start_pos=0,
        mode="prefill",
        cp_prefill_mask=wu_cp_prefill_mask_tt,
        return_hidden_state=False,
    )
    wu_cp1 = ttnn.from_torch(
        torch.zeros(1, 1, 1, talker_h),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    wu_cp1_pos = torch.tensor([2])
    wu_cp1_cos, wu_cp1_sin = get_rope_tensors(device, cp_head_dim, 1, wu_cp1_pos, cp_rope_theta)
    wu_cp_decode_mask = ttnn.from_torch(
        torch.full((1, _cp_num_heads, 1, max_cp_seq_len), float("-inf")),
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    model.code_predictor.forward_single_step(
        wu_cp1,
        wu_cp1_cos,
        wu_cp1_sin,
        cp_trans_mat,
        generation_step=2,
        kv_caches=wu_cp_kv,
        start_pos=2,
        mode="decode",
        cur_pos_tensor=None,
        decode_attn_mask=wu_cp_decode_mask,
        return_hidden_state=False,
    )
    ttnn.deallocate(wu_cp_decode_mask)
    ttnn.deallocate(wu_cp_prefill_mask_tt)
    deallocate_kv_cache(wu_cp_kv)
    ttnn.synchronize_device(device)
    for t in [wu_pf, wu_dc, wu_cp2, wu_cp1]:
        ttnn.deallocate(t)
    print("  Warmup complete.")

    # === STEP 3: Allocate persistent KV caches + pre-allocated trace tensors ===
    talker_kv_caches = allocate_kv_cache(
        device=device,
        num_layers=model.talker_config.num_hidden_layers,
        batch_size=1,
        num_kv_heads=model.talker_config.num_key_value_heads,
        max_seq_len=max_talker_seq_len,
        head_dim=head_dim,
    )
    print(f"  Allocated Talker KV cache ({model.talker_config.num_hidden_layers} layers, max_seq={max_talker_seq_len})")

    cp_kv_caches_persistent = allocate_kv_cache(
        device=device,
        num_layers=model.code_predictor_config.num_hidden_layers,
        batch_size=1,
        num_kv_heads=model.code_predictor_config.num_key_value_heads,
        max_seq_len=max_cp_seq_len,
        head_dim=cp_head_dim,
    )
    print(f"  Allocated CP KV cache ({model.code_predictor_config.num_hidden_layers} layers, max_seq={max_cp_seq_len})")

    # Pre-allocate zero host tensors for CP KV cache reset between frames
    cp_kv_zero_hosts = []
    for layer_kv in cp_kv_caches_persistent:
        k_cache, v_cache = layer_kv
        k_zero = ttnn.from_torch(
            torch.zeros(k_cache.shape[0], k_cache.shape[1], k_cache.shape[2], k_cache.shape[3], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        v_zero = ttnn.from_torch(
            torch.zeros(v_cache.shape[0], v_cache.shape[1], v_cache.shape[2], v_cache.shape[3], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        cp_kv_zero_hosts.append((k_zero, v_zero))

    # === STEP 4: Run Talker prefill (non-traced, standard path) ===
    # Standard prefill: attention over seq_len (not full cache), much faster.
    # The standard path in attention.py fills the KV cache via to_torch/from_torch,
    # which may reallocate cache tensors. We capture the returned updated caches
    # and use those for trace capture (same as the reference in generator.py).
    print("  STEP 4: Building prefill RoPE...")
    prefill_pos = torch.arange(padded_seq_len)
    prefill_cos_tt, prefill_sin_tt = get_rope_tensors(
        device, head_dim, padded_seq_len, prefill_pos, model.talker_config.rope_theta
    )

    ttnn.synchronize_device(device)
    t_prefill_start = time.time()

    print("  STEP 4: Running Talker prefill forward (28 layers, non-traced)...")
    prefill_hidden_out, talker_kv_caches = model.talker.forward_from_hidden(
        inputs_embeds_tt,
        prefill_cos_tt,
        prefill_sin_tt,
        talker_trans_mat,
        kv_caches=talker_kv_caches,
        start_pos=0,
        mode="prefill",
    )
    prefill_logits_out = model.talker.get_codec_logits(prefill_hidden_out)
    ttnn.synchronize_device(device)

    # Track generated code 0 tokens for repetition penalty
    generated_code0_tokens = []

    codec_logits_full = ttnn.to_torch(prefill_logits_out).squeeze(1).float()
    codec_logits_torch = codec_logits_full[0, real_seq_len - 1, :]
    token_0 = sample_token(
        codec_logits_torch,
        config.temperature,
        config.top_k,
        config.greedy,
        config.repetition_penalty,
        generated_code0_tokens,
    )
    generated_code0_tokens.append(token_0)
    ttnn.synchronize_device(device)
    t_prefill_end = time.time()

    print(f"  Talker prefill done (non-traced): {(t_prefill_end - t_prefill_start)*1000:.1f} ms, token_0={token_0}")

    if token_0 == config.codec_eos_id:
        print("  EOS at prefill")
        deallocate_kv_cache(talker_kv_caches)
        deallocate_kv_cache(cp_kv_caches_persistent)
        return None

    talker_pos = real_seq_len
    talker_hidden_tt = prefill_hidden_out

    talker_decode_mask_host = torch.full((1, _talker_num_heads, 1, max_talker_seq_len), float("-inf"))
    talker_decode_mask_host[0, :, 0, :real_seq_len] = 0.0

    ttnn.deallocate(prefill_cos_tt)
    ttnn.deallocate(prefill_sin_tt)

    # === STEP 5: Pre-allocate ALL trace input tensors, then capture traces ===
    # Pre-allocate all trace input tensors BEFORE any trace capture.
    trace_embed_tt = ttnn.from_torch(
        torch.zeros(1, 1, 1, talker_h, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    trace_cos_tt = ttnn.from_torch(
        torch.ones(1, 1, 1, head_dim, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    trace_sin_tt = ttnn.from_torch(
        torch.zeros(1, 1, 1, head_dim, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    trace_cur_pos_tt = ttnn.from_torch(
        torch.tensor([padded_seq_len], dtype=torch.int32),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    trace_mask_tt = ttnn.from_torch(
        torch.full((1, _talker_num_heads, 1, max_talker_seq_len), float("-inf")),
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    cp_prefill_pos = torch.arange(2)
    cp_trace_prefill_cos_tt, cp_trace_prefill_sin_tt = get_rope_tensors(
        device, cp_head_dim, 2, cp_prefill_pos, cp_rope_theta
    )
    cp_trace_prefill_cos_host = ttnn.from_torch(
        ttnn.to_torch(cp_trace_prefill_cos_tt).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    cp_trace_prefill_sin_host = ttnn.from_torch(
        ttnn.to_torch(cp_trace_prefill_sin_tt).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    cp_prefill_mask_host_torch = torch.full((1, _cp_num_heads, 2, max_cp_seq_len), float("-inf"))
    cp_prefill_mask_host_torch[0, :, 0, 0] = 0.0
    cp_prefill_mask_host_torch[0, :, 1, 0:2] = 0.0
    cp_trace_prefill_mask_tt = ttnn.from_torch(
        cp_prefill_mask_host_torch,
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cp_trace_prefill_mask_host = ttnn.from_torch(
        cp_prefill_mask_host_torch.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
    )
    cp_trace_prefill_embed_tt = ttnn.from_torch(
        torch.zeros(1, 1, 2, talker_h, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    cp_trace_decode_embed_tt = ttnn.from_torch(
        torch.zeros(1, 1, 1, talker_h, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cp_trace_decode_cos_tt = ttnn.from_torch(
        torch.ones(1, 1, 1, cp_head_dim, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cp_trace_decode_sin_tt = ttnn.from_torch(
        torch.zeros(1, 1, 1, cp_head_dim, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cp_trace_decode_mask_tt = ttnn.from_torch(
        torch.full((1, _cp_num_heads, 1, max_cp_seq_len), float("-inf")),
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # === STEP 5: Capture ALL traces (after prefill, KV cache is populated) ===
    # --- 5a: Talker decode trace ---
    print("  Capturing Talker decode trace (includes codec_head)...")
    talker_decode_trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    trace_hidden_out, _ = model.talker.forward_from_hidden(
        trace_embed_tt,
        trace_cos_tt,
        trace_sin_tt,
        talker_trans_mat,
        kv_caches=talker_kv_caches,
        cur_pos_tensor=trace_cur_pos_tt,
        decode_attn_mask=trace_mask_tt,
        mode="decode",
    )
    trace_codec_logits_out = model.talker.get_codec_logits(trace_hidden_out)
    ttnn.end_trace_capture(device, talker_decode_trace_id, cq_id=0)
    ttnn.synchronize_device(device)
    print("  Talker decode trace captured.")

    # --- 5b: CP prefill trace ---
    print("  Capturing CP prefill trace (seq_len=2, includes lm_heads[0])...")
    cp_prefill_trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    cp_prefill_logits_tt, _ = model.code_predictor.forward_single_step(
        cp_trace_prefill_embed_tt,
        cp_trace_prefill_cos_tt,
        cp_trace_prefill_sin_tt,
        cp_trans_mat,
        generation_step=1,
        kv_caches=cp_kv_caches_persistent,
        start_pos=0,
        mode="prefill",
        cp_prefill_mask=cp_trace_prefill_mask_tt,
        return_hidden_state=False,
    )
    ttnn.end_trace_capture(device, cp_prefill_trace_id, cq_id=0)
    ttnn.synchronize_device(device)
    print("  CP prefill trace captured.")

    # --- 5c: CP decode traces x13 ---
    cp_decode_trace_ids = []
    cp_decode_logits_tts = []
    print(f"  Capturing {config.num_code_groups - 2} CP decode traces (one per lm_head)...")
    for _step_code_idx in range(2, config.num_code_groups):
        _trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        _logits_tt, _ = model.code_predictor.forward_single_step(
            cp_trace_decode_embed_tt,
            cp_trace_decode_cos_tt,
            cp_trace_decode_sin_tt,
            cp_trans_mat,
            generation_step=_step_code_idx,
            kv_caches=cp_kv_caches_persistent,
            start_pos=_step_code_idx,
            mode="decode",
            cur_pos_tensor=None,
            decode_attn_mask=cp_trace_decode_mask_tt,
            return_hidden_state=False,
        )
        ttnn.end_trace_capture(device, _trace_id, cq_id=0)
        ttnn.synchronize_device(device)
        cp_decode_trace_ids.append(_trace_id)
        cp_decode_logits_tts.append(_logits_tt)
    print(f"  Captured {len(cp_decode_trace_ids)} CP decode traces.")
    print("  All traces captured. Starting measured inference...")

    # === STEP 6: Measured inference (generation loop only; prefill already ran in STEP 4) ===
    decode_step_times = []
    talker_times_ms = []
    cp_times_ms = []
    t_first_decode_end = 0.0

    try:
        # --- Generation loop ---
        for step in range(config.max_new_tokens):
            ttnn.synchronize_device(device)
            t_step_start = time.time()

            # === CodePredictor: Generate codes 1-15 ===
            past_hidden_torch = ttnn.to_torch(talker_hidden_tt)[:, :, -1:, :].float()
            code0_embed = F.embedding(torch.tensor([[token_0]]), codec_embed_torch).unsqueeze(1)
            cp_input = torch.cat([past_hidden_torch, code0_embed], dim=2)

            code_row = [token_0]

            # Restore CP constants corrupted by Talker's paged_update_cache
            ttnn.copy_host_to_device_tensor(cp_trace_prefill_mask_host, cp_trace_prefill_mask_tt)
            ttnn.copy_host_to_device_tensor(cp_trace_prefill_cos_host, cp_trace_prefill_cos_tt)
            ttnn.copy_host_to_device_tensor(cp_trace_prefill_sin_host, cp_trace_prefill_sin_tt)
            for (k_zero, v_zero), (k_cache, v_cache) in zip(cp_kv_zero_hosts, cp_kv_caches_persistent):
                ttnn.copy_host_to_device_tensor(k_zero, k_cache)
                ttnn.copy_host_to_device_tensor(v_zero, v_cache)

            # CP prefill trace
            pfembed_host = ttnn.from_torch(cp_input.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(pfembed_host, cp_trace_prefill_embed_tt)
            ttnn.execute_trace(device, cp_prefill_trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)

            _pf_vocab = cp_prefill_logits_tt.shape[3]
            last_prefill_logits = ttnn.slice(cp_prefill_logits_tt, [0, 0, 1, 0], [1, 1, 2, _pf_vocab])
            logits_torch = ttnn.to_torch(last_prefill_logits).squeeze().float()
            ttnn.deallocate(last_prefill_logits)
            token = sample_token(logits_torch, config.temperature, config.top_k, config.greedy)
            code_row.append(token)

            cp_pos = 2
            cp_decode_mask_host = torch.full((1, _cp_num_heads, 1, max_cp_seq_len), float("-inf"))
            cp_decode_mask_host[0, :, 0, 0:2] = 0.0

            # CP decode traces x13
            for _trace_i, code_idx in enumerate(range(2, config.num_code_groups)):
                prev_embed_idx = code_idx - 2
                if prev_embed_idx < len(code_pred_embeds) and code_pred_embeds[prev_embed_idx] is not None:
                    next_embed = F.embedding(torch.tensor([[token]]), code_pred_embeds[prev_embed_idx])
                else:
                    next_embed = F.embedding(torch.tensor([[token]]), codec_embed_torch)
                next_embed = next_embed.unsqueeze(1).bfloat16()

                cp_cos_val = cp_cos_table[cp_pos : cp_pos + 1].unsqueeze(0).unsqueeze(0).bfloat16()
                cp_sin_val = cp_sin_table[cp_pos : cp_pos + 1].unsqueeze(0).unsqueeze(0).bfloat16()
                cp_decode_mask_host[0, :, 0, cp_pos] = 0.0

                e_h = ttnn.from_torch(next_embed, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                c_h = ttnn.from_torch(cp_cos_val, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                s_h = ttnn.from_torch(cp_sin_val, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                m_h = ttnn.from_torch(cp_decode_mask_host.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
                ttnn.copy_host_to_device_tensor(e_h, cp_trace_decode_embed_tt)
                ttnn.copy_host_to_device_tensor(c_h, cp_trace_decode_cos_tt)
                ttnn.copy_host_to_device_tensor(s_h, cp_trace_decode_sin_tt)
                ttnn.copy_host_to_device_tensor(m_h, cp_trace_decode_mask_tt)

                ttnn.execute_trace(device, cp_decode_trace_ids[_trace_i], cq_id=0, blocking=False)
                ttnn.synchronize_device(device)

                logits_torch = ttnn.to_torch(cp_decode_logits_tts[_trace_i]).squeeze().float()
                token = sample_token(logits_torch, config.temperature, config.top_k, config.greedy)
                code_row.append(token)
                cp_pos += 1

            all_codes.append(code_row)
            ttnn.synchronize_device(device)
            t_cp_end = time.time()

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
            next_embed = stacked.sum(dim=1, keepdim=True)

            trailing_len = trailing_text_hidden.shape[1]
            if step < trailing_len:
                next_embed = next_embed + trailing_text_hidden[:, step : step + 1, :]
            else:
                next_embed = next_embed + tts_pad_embed

            next_embed = next_embed.unsqueeze(1)

            # === Talker decode trace ===
            talker_decode_mask_host[0, :, 0, talker_pos] = 0.0

            cos_val = talker_cos_table[talker_pos : talker_pos + 1].unsqueeze(0).unsqueeze(0).bfloat16()
            sin_val = talker_sin_table[talker_pos : talker_pos + 1].unsqueeze(0).unsqueeze(0).bfloat16()

            embed_host = ttnn.from_torch(next_embed.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            cos_host = ttnn.from_torch(cos_val, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            sin_host = ttnn.from_torch(sin_val, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            cur_pos_host = ttnn.from_torch(
                torch.tensor([talker_pos], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            mask_host = ttnn.from_torch(talker_decode_mask_host.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(embed_host, trace_embed_tt)
            ttnn.copy_host_to_device_tensor(cos_host, trace_cos_tt)
            ttnn.copy_host_to_device_tensor(sin_host, trace_sin_tt)
            ttnn.copy_host_to_device_tensor(cur_pos_host, trace_cur_pos_tt)
            ttnn.copy_host_to_device_tensor(mask_host, trace_mask_tt)

            ttnn.execute_trace(device, talker_decode_trace_id, cq_id=0, blocking=False)
            talker_hidden_tt = trace_hidden_out
            talker_pos += 1
            ttnn.synchronize_device(device)
            t_talker_end = time.time()
            talker_times_ms.append((t_talker_end - t_cp_end) * 1000)
            cp_times_ms.append((t_cp_end - t_step_start) * 1000)

            # Get next code 0 from trace output
            codec_logits_torch = ttnn.to_torch(trace_codec_logits_out)[:, :, -1, :].squeeze().float()
            token_0 = sample_token(
                codec_logits_torch,
                config.temperature,
                config.top_k,
                config.greedy,
                config.repetition_penalty,
                generated_code0_tokens,
            )
            generated_code0_tokens.append(token_0)

            if token_0 == config.codec_eos_id:
                print(f"  EOS at step {step + 1}")
                break

            ttnn.synchronize_device(device)
            t_step_end = time.time()
            step_ms = (t_step_end - t_step_start) * 1000
            if step == 0:
                t_first_decode_end = t_step_end
            decode_step_times.append(step_ms)

            if (step + 1) % 20 == 0:
                print(f"  Generated {step + 1} frames...")

    finally:
        ttnn.release_trace(device, talker_decode_trace_id)
        ttnn.release_trace(device, cp_prefill_trace_id)
        for _tid in cp_decode_trace_ids:
            ttnn.release_trace(device, _tid)
        for t in [
            trace_embed_tt,
            trace_cos_tt,
            trace_sin_tt,
            trace_cur_pos_tt,
            trace_mask_tt,
            cp_trace_prefill_embed_tt,
            cp_trace_prefill_cos_tt,
            cp_trace_prefill_sin_tt,
            cp_trace_prefill_mask_tt,
            cp_trace_decode_embed_tt,
            cp_trace_decode_cos_tt,
            cp_trace_decode_sin_tt,
            cp_trace_decode_mask_tt,
        ]:
            if t is not None:
                ttnn.deallocate(t)
        deallocate_kv_cache(talker_kv_caches)
        deallocate_kv_cache(cp_kv_caches_persistent)

    print(f"  Generated {len(all_codes)} code frames")

    if len(all_codes) == 0:
        return None

    codes = torch.tensor(all_codes, dtype=torch.long)
    print(f"  Code 0 sample (first 5 frames): {codes[:5, 0].tolist()}")
    print(f"  Code 0 in valid range [0,2047]: {((codes[:,0]>=0) & (codes[:,0]<=2047)).sum()}/{len(codes)}")
    torch.save(codes, "/tmp/last_generated_codes.pt")

    # === Performance metrics ===
    prefill_ms = (t_prefill_end - t_prefill_start) * 1000
    if decode_step_times:
        ttft_ms = prefill_ms + decode_step_times[0]
        if len(decode_step_times) > 1:
            steady_ms = decode_step_times[1:]
            tokens_per_sec = 1000.0 / (sum(steady_ms) / len(steady_ms))
        else:
            tokens_per_sec = 1000.0 / decode_step_times[0]
    else:
        ttft_ms = prefill_ms
        tokens_per_sec = 0.0

    print(f"\n  --- Performance (Qwen3-TTS on N150, all traced) ---")
    print(f"  Prefill  ({real_seq_len} real / {padded_seq_len} padded tokens): {prefill_ms:.1f} ms")
    print(f"  TTFT     (prefill + 1 decode):  {ttft_ms:.1f} ms")
    print(f"  Decode throughput:              {tokens_per_sec:.2f} frames/sec")
    if decode_step_times:
        avg_decode_ms = sum(decode_step_times) / len(decode_step_times)
        print(f"  Avg decode step:                {avg_decode_ms:.1f} ms/frame")
    if talker_times_ms:
        print(f"  Avg Talker decode:              {sum(talker_times_ms)/len(talker_times_ms):.1f} ms/frame")
    if cp_times_ms:
        print(f"  Avg CodePredictor:              {sum(cp_times_ms)/len(cp_times_ms):.1f} ms/frame")
    print(f"  Traced: Talker decode, CP prefill, CP decode x{len(cp_decode_trace_ids)} (Talker prefill: non-traced)")
    print(f"  ----------------------------------------")

    return codes


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
    repetition_penalty: float = 1.0,
    use_kv_cache: bool = True,
    use_trace: bool = True,
    ref_cache: str = None,
    trim_frames: int = 4,
    load_cpu_inputs: str = None,
    auto_trim_bleed: bool = False,
    target_word: str = None,
):
    """Run full TTNN TTS pipeline."""
    print("=" * 80)
    print("Full TTNN TTS Demo")
    print("=" * 80)
    print(f"Text: {text}")
    print(f"Reference: {ref_audio}")
    print(f"Max tokens: {max_new_tokens}")
    print(f"Decoding: {'greedy' if greedy else f'sampling (temp=0.9, top_k=50, rep_penalty={repetition_penalty})'}")
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

    # Open device with explicit trace region.
    print(f"\nOpening TT device {device_id}...")
    # NOTE: Very large trace regions can fail device open due to TLB window allocation limits.
    # Start with a moderate size and increase only if trace capture requires it.
    device = ttnn.open_device(device_id=device_id, l1_small_size=32768, trace_region_size=100000000)
    device.enable_program_cache()

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
        config.repetition_penalty = repetition_penalty
        config.trim_codec_frames = trim_frames

        if load_cpu_inputs:
            print(f"\n  Loading CPU-computed ICL inputs from: {load_cpu_inputs}")
            cpu_data = torch.load(load_cpu_inputs, map_location="cpu", weights_only=True)
            inputs_embeds_cpu = cpu_data["inputs_embeds"].float()
            trailing_text_hidden = cpu_data["trailing_text_hidden"].float()
            tts_pad_embed = cpu_data["tts_pad_embed"].float()
            inputs_embeds_tt = ttnn.from_torch(
                inputs_embeds_cpu.unsqueeze(1),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            code_pred_embeds = []
            for i in range(config.num_code_groups - 1):
                key = f"talker.code_predictor.model.codec_embedding.{i}.weight"
                if key in main_weights:
                    code_pred_embeds.append(main_weights[key].float())
            print(f"  inputs_embeds: {inputs_embeds_cpu.shape}")
            print(f"  trailing_text_hidden: {trailing_text_hidden.shape}")
            print(f"  code_pred_embeds: {len(code_pred_embeds)}")
            timings["encode_ref"] = 0.0
            timings["speaker_embed"] = 0.0
            timings["icl_embed"] = 0.0
        else:
            # Encode reference audio (speech tokenizer encoder - reference impl, cached)
            encode_start = time.time()
            ref_codes, audio_data = encode_reference_audio(ref_audio, main_weights, cache_path=ref_cache)
            ref_codes, audio_data = trim_reference_for_icl_conditioning(
                ref_codes, audio_data, tokenizer, ref_text, text
            )
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
            use_trace=use_trace,
        )
        timings["generation"] = time.time() - gen_start

        if codes is None:
            print("ERROR: Failed to generate codes")
            return

        # Trim reference echo: ICL TTS models briefly "echo" the reference
        # speaker's last word before the target text. Remove leading frames.
        if config.trim_codec_frames > 0 and len(codes) > config.trim_codec_frames:
            print(f"  Trimming {config.trim_codec_frames} leading codec frames (reference echo removal)")
            codes = codes[config.trim_codec_frames :]

        # Decode to audio (reference impl)
        decode_start = time.time()
        audio = decode_audio(codes, decoder_weights)
        timings["decode"] = time.time() - decode_start

        # Save audio
        audio_np = audio.squeeze().detach().cpu().float().numpy()
        sf.write(output_path, audio_np, 24000)

        # Auto-trim bleed if requested
        if auto_trim_bleed:
            from models.demos.qwen3_tts.demo.bleed_detector import detect_bleed, print_bleed_report, trim_audio

            # Extract first word from target text if not provided
            if target_word is None:
                # Get first word, stripping punctuation
                first_word = text.split()[0].rstrip(",.!?;:") if text.split() else "Hello"
                target_word = first_word

            print(f"\n  Running bleed detection (target word: '{target_word}')...")
            bleed_results = detect_bleed(output_path, target_word)
            print_bleed_report(bleed_results)

            if bleed_results["bleed_duration"] > 0.1:
                # Trim the audio
                trim_time = max(0, bleed_results["bleed_duration"] - 0.1)  # Small margin
                trimmed_path = output_path.replace(".wav", "_trimmed.wav")
                trim_audio(output_path, trimmed_path, trim_time)
                print(f"  Bleed-trimmed audio saved to: {trimmed_path}")

                # Also update the main output with trimmed version
                trim_audio(output_path, output_path, trim_time)
                audio_np = audio_np[int(trim_time * 24000) :]
                print(f"  Main output updated with trimmed audio")

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

        print(f"  (TTFT and decode throughput breakdown printed above during generation)")

        print(f"\nOutput saved to: {output_path}")
        print(f"Audio duration: {len(audio_np) / 24000:.2f}s")
        print("=" * 80)

    finally:
        ttnn.close_device(device)
        print("\nDevice closed")


def get_default_reference_path():
    """Get path to included Jim reference audio."""
    import os

    return os.path.join(os.path.dirname(__file__), "jim_reference.wav")


def main():
    parser = argparse.ArgumentParser(description="Full TTNN TTS Demo")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Reference audio path (default: included jim_reference.wav)",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default="Let me also go over the review slides.",
        help="Reference audio transcript (default: transcript for jim_reference.wav)",
    )
    parser.add_argument("--output", type=str, default="/tmp/ttnn_tts_output.wav", help="Output path")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--device-id", type=int, default=0, help="TT device ID")
    parser.add_argument("--language", type=str, default="english", help="Language")
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
        "--trim-frames",
        type=int,
        default=4,
        help="Codec frames to trim from start (removes reference echo, default: 4)",
    )
    parser.add_argument("--no-kv-cache", action="store_true", help="Disable KV cache (slower)")
    parser.add_argument("--no-trace", action="store_true", help="Disable trace (use non-traced KV cache decode)")
    parser.add_argument(
        "--ref-cache",
        type=str,
        default=None,
        help="Path to cached reference encoding (.pt). Auto-derived from --ref-audio if not set.",
    )
    parser.add_argument(
        "--load-cpu-inputs",
        type=str,
        default=None,
        help="Load CPU-computed ICL embeddings from .pt file (skips speaker encoder & ICL construction)",
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

    # Use default Jim reference if not specified
    ref_audio = args.ref_audio if args.ref_audio else get_default_reference_path()

    run_full_ttnn_tts(
        text=args.text,
        ref_audio=ref_audio,
        ref_text=args.ref_text,
        output_path=args.output,
        max_new_tokens=args.max_tokens,
        device_id=args.device_id,
        language=args.language,
        greedy=args.greedy,
        repetition_penalty=args.repetition_penalty,
        use_kv_cache=not args.no_kv_cache,
        use_trace=not args.no_trace,
        ref_cache=args.ref_cache,
        trim_frames=args.trim_frames,
        load_cpu_inputs=args.load_cpu_inputs,
        auto_trim_bleed=args.auto_trim_bleed,
        target_word=args.target_word,
    )


if __name__ == "__main__":
    main()
