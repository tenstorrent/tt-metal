# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS server-side TTNN implementation.

Reusable server-context, AR-loop, and ICL-embedding code for the Qwen3-TTS
pipeline. Demos (CLI / web) and inference servers all import from here.

Public surface:
- ``TTSConfig``, ``TTSServerContext``        – per-deployment config + warmed state
- ``init_server_context(device, model, config, main_weights)``
                                              – pre-compiles all kernels and
                                                captures all production traces
- ``run_inference(ctx, model, device, ...)`` – per-request AR loop (zero
                                                trace capture / kernel compile)
- ``encode_reference_audio(audio_path, ...)`` – Mimi encode → ref_codes
                                                (cached in .refcache.pt)
- ``create_icl_embedding_ttnn(...)``         – build ICL embedding for prefill
- ``decode_audio(codes, decoder_weights)``   – Mimi decode → 24 kHz waveform
- ``load_weights()``                         – HF download (main + speech_tokenizer)
- ``allocate_kv_cache(...)``, ``deallocate_kv_cache(...)``

Architecture:
- Speaker Encoder (TTNN) – ECAPA-TDNN
- Text Projection (TTNN) – project text embeddings
- Talker (TTNN) – 28-layer transformer + codec_head
- CodePredictor (TTNN) – 5-layer transformer + 15 LM heads
- Speech Tokenizer (Mimi) is reference PyTorch (uses 1D conv + reflect pad
  not available in TTNN today).

KV Cache:
- Talker uses KV cache: prefill ICL sequence once, then decode 1 token at a time.
- CodePredictor uses KV cache: prefill [past_hidden, code0], then decode codes 1-14.
- Drops O(n²) → O(n) for generation.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

import ttnn


def _user_path_no_dotdot(path: str) -> Path:
    p = Path(path).expanduser()
    if ".." in p.parts:
        raise ValueError("paths must not contain '..' path components")
    return p.resolve()


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
            # Codec language IDs as used by demo_pure_reference_tts.py / HF Qwen3-TTS.
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
    """Load model weights."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print("Loading model weights...")
    model_path = Path(snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", allow_patterns=["*.safetensors"])).resolve()

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
    audio_path: str, main_weights: dict = None, cache_path: str = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode reference audio to codes and extract speaker embedding using TTNN speaker encoder.

    Caches results to disk so the slow CPU MimiModel only runs once.
    Cache path defaults to <audio_path>.refcache.pt

    ``main_weights`` is unused (kept for call-site compatibility); pass ``None``.

    Returns:
        ref_codes: [seq_len, 16] - RVQ codes
        audio_data: [num_samples] - raw waveform (for TTNN speaker encoder)
    """
    audio_p = _user_path_no_dotdot(audio_path)
    if not audio_p.is_file():
        raise FileNotFoundError(audio_path)

    if cache_path is None:
        cache_path = str(audio_p.with_suffix("")) + ".refcache.pt"
    else:
        cache_path = str(_user_path_no_dotdot(cache_path))

    # Load from cache if available
    if Path(cache_path).exists():
        print(f"\nLoading cached reference encoding from {cache_path}")
        cached = torch.load(cache_path, weights_only=True)
        ref_codes = cached["ref_codes"]
        audio_data = cached["audio_data"]
        print(f"  Reference codes: {ref_codes.shape}  (loaded from cache)")
        print(f"  Audio duration: {len(audio_data)/24000:.2f}s")
        return ref_codes, audio_data

    from models.demos.qwen3_tts.reference.functional import speech_tokenizer_encoder_forward_mimi

    print("\nEncoding reference audio (first run - will cache result)...")

    # Load audio — convert to WAV via ffmpeg first so soundfile can read any format
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_p), "-ac", "1", "-ar", "24000", "-f", "wav", tmp_wav],
        check=True,
        capture_output=True,
    )
    audio_data, sr = sf.read(tmp_wav)
    import os

    os.unlink(tmp_wav)
    audio_data = torch.from_numpy(audio_data.astype(np.float32))
    if audio_data.dim() == 2:
        audio_data = audio_data.mean(dim=1)
    # sr is already 24000 (ffmpeg resampled above), so no further resampling needed

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

    # float32: faster topk/softmax/multinomial on CPU than bfloat16 for this vocab size
    if logits.dtype != torch.float32:
        logits = logits.float()

    # Apply repetition penalty to previously generated tokens (vectorized; hot loop)
    if repetition_penalty != 1.0 and generated_tokens:
        idx = torch.tensor(list(set(generated_tokens)), dtype=torch.long, device=logits.device)
        vocab = logits.numel()
        idx = idx[(idx >= 0) & (idx < vocab)]
        if idx.numel() > 0:
            vals = logits[idx]
            logits[idx] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)

    logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def build_cp_decode_trace_h2d_constants(
    cp_cos_table: torch.Tensor,
    cp_sin_table: torch.Tensor,
    num_cp_heads: int,
    max_cp_seq_len: int,
    num_decode_traces: int,
) -> Tuple[list, list, list]:
    """Pre-build TILE-layout host tensors for CP decode trace H2D (cos, sin, mask per trace index).

    Masks match the cumulative attention pattern used in the generation loop for each
    decode position (cp_pos = 2 .. num_decode_traces+1).
    """
    cos_hosts: list = []
    sin_hosts: list = []
    mask_hosts: list = []
    for trace_i in range(num_decode_traces):
        cp_pos = 2 + trace_i
        mh = torch.full((1, num_cp_heads, 1, max_cp_seq_len), float("-inf"), dtype=torch.float32)
        mh[0, :, 0, : cp_pos + 1] = 0.0
        cs = cp_cos_table[cp_pos : cp_pos + 1].unsqueeze(0).unsqueeze(0).bfloat16()
        sn = cp_sin_table[cp_pos : cp_pos + 1].unsqueeze(0).unsqueeze(0).bfloat16()
        cos_hosts.append(ttnn.from_torch(cs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))
        sin_hosts.append(ttnn.from_torch(sn, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))
        mask_hosts.append(ttnn.from_torch(mh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT))
    return cos_hosts, sin_hosts, mask_hosts


def build_talker_decode_trace_h2d_constants(
    talker_cos_table: torch.Tensor,
    talker_sin_table: torch.Tensor,
    num_talker_heads: int,
    max_talker_seq_len: int,
    real_seq_len: int,
) -> Tuple[list, list, list, list]:
    """Pre-build TILE/RM host tensors for Talker decode H2D: cos, sin, mask, cur_pos per absolute T.

    For each decode position ``T`` in ``[real_seq_len, max_talker_seq_len)``, the mask matches
    the incremental pattern: prefill columns ``0:real_seq_len`` plus decode columns
    ``real_seq_len : T+1`` set to 0; remainder ``-inf``.
    """
    cos_h: list = []
    sin_h: list = []
    mask_h: list = []
    pos_h: list = []
    for T in range(real_seq_len, max_talker_seq_len):
        mh = torch.full((1, num_talker_heads, 1, max_talker_seq_len), float("-inf"), dtype=torch.float32)
        mh[0, :, 0, :real_seq_len] = 0.0
        mh[0, :, 0, real_seq_len : T + 1] = 0.0
        cs = talker_cos_table[T : T + 1].unsqueeze(0).unsqueeze(0).bfloat16()
        sn = talker_sin_table[T : T + 1].unsqueeze(0).unsqueeze(0).bfloat16()
        cos_h.append(ttnn.from_torch(cs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))
        sin_h.append(ttnn.from_torch(sn, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))
        mask_h.append(ttnn.from_torch(mh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT))
        pos_h.append(
            ttnn.from_torch(torch.tensor([T], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        )
    return cos_h, sin_h, mask_h, pos_h


def _alloc_token_buf(device, shape=(1, 1, 1)) -> ttnn.Tensor:
    """Allocate a uint32 ROW_MAJOR DRAM token output buffer for ttnn.argmax."""
    return ttnn.from_torch(
        torch.zeros(shape, dtype=torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _argmax_into(logits_tt: ttnn.Tensor, out_tok_tt: ttnn.Tensor) -> None:
    """Untilize + multicore argmax into a pre-allocated uint32 RM token buffer.

    Multicore ``ttnn.argmax`` only supports ROW_MAJOR inputs. Plain ``ttnn.untilize``
    on a tile-padded Y (e.g. ``[1,1,1,vocab]`` padded to ``[1,1,32,vocab]``)
    silently mixes padded rows into row 0. Use ``untilize_with_unpadding`` with
    the LOGICAL last index to drop tile padding cleanly.
    """
    # End index per dim is logical_size - 1.
    end = [int(s) - 1 for s in logits_tt.shape]
    logits_rm = ttnn.untilize_with_unpadding(logits_tt, output_tensor_end=end, use_multicore=True)
    ttnn.argmax(logits_rm, dim=-1, keepdim=False, use_multicore=True, output_tensor=out_tok_tt)
    ttnn.deallocate(logits_rm)


def _read_device_token(tok_tt: ttnn.Tensor, index: int = 0) -> int:
    """D2H read of a single token id from on-device argmax/sampling output (uint32 RM).

    For ttnn.argmax, the output is a small [...,1] uint32 tensor and we read element
    [index]. For ttnn.sampling (which requires 32 users per kernel invocation, so
    we replicate batch=1 across 32 users), the output is shape [1,1,1,32] and we
    read user[0] (index=0).
    """
    return int(ttnn.to_torch(tok_tt).flatten()[index].item())


# ttnn.sampling kernel hardcodes 32 users. We replicate our batch=1 logits across
# 32 users, sample, then take user[0]'s token. The other 31 users are wasted compute
# but cheap (~µs) compared to the 14× CPU sample/embed/H2D round-trips this lets us
# eliminate per frame.
_SAMPLING_USERS = 32
# topk inner-dim must be a multiple of 32. Default sampling config is top_k=50 →
# round up to 64 for the topk kernel; the per-user k-tensor enforces top_k_actual.
_SAMPLING_MAX_TOP_K = 64


class _DeviceSampler:
    """In-trace ``ttnn.topk + ttnn.sampling`` for the CP decode autoregressive loop.

    Every per-frame CP decode trace currently round-trips through the CPU between
    steps to:
      (1) D2H full vocab logits, (2) ``torch.multinomial`` sample, (3) lookup
      embedding via ``F.embedding``, (4) ``ttnn.from_torch`` + H2D the embed to
      the next trace's input buffer.
    This class lets the trace emit the sampled token id as a device tensor so the
    follow-on ``ttnn.embedding`` (also captured in trace) can write directly into
    the next trace's input buffer — closing the loop on-device.

    Per-trace usage:
        sampler = _DeviceSampler(device, top_k=50, top_p=1.0, temperature=0.9)
        # Before trace_capture:
        tok_tt = sampler.alloc_token_buf()
        # Inside trace_capture:
        sampler.append_sampling(logits_tt, tok_tt)
        # After replay, optional small D2H for code_row tracking / EOS:
        token = _read_device_token(tok_tt, index=0)
        # OR feed tok_tt directly into ttnn.embedding (also captured in trace).

    Notes:
      * Output buffer is shape ``[1,1,1,32]`` UINT32 ROW_MAJOR. user[0] holds the
        true sample; users 1..31 are duplicate sampling work.
      * Param tensors (k/p/temp) are pre-allocated once and baked into the trace.
        To change params at runtime, ``copy_host_to_device_tensor`` over them
        before replay.
    """

    def __init__(self, device, top_k: int, top_p: float, temperature: float, seed: int = 0):
        self.device = device
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.seed = seed
        if top_k > _SAMPLING_MAX_TOP_K:
            raise ValueError(f"top_k={top_k} exceeds compiled max ({_SAMPLING_MAX_TOP_K})")
        # ttnn.topk and ttnn.sampling default to 1 core when sub_core_grids is unset,
        # leaving 32-user-replicated rows serialized. Pass an explicit 32-core grid
        # (one core per user) so per-row top-k runs in parallel — same pattern as
        # tt_transformers' SamplingGenerator.
        _grid_size = device.compute_with_storage_grid_size()
        self._sub_core_grids = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_grid_size.x - 1, _grid_size.y - 1))]
        )
        # Per-user param tensors (32 users replicated).
        self.k_tensor = ttnn.from_torch(
            torch.full((_SAMPLING_USERS,), top_k, dtype=torch.int32),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.p_tensor = ttnn.from_torch(
            torch.full((_SAMPLING_USERS,), top_p, dtype=torch.float32),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.temp_tensor = ttnn.from_torch(
            torch.full((_SAMPLING_USERS,), temperature, dtype=torch.float32),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def alloc_token_buf(self) -> ttnn.Tensor:
        """Output token buffer for one ``ttnn.sampling`` call (uint32 RM, [1,1,1,32])."""
        return ttnn.from_torch(
            torch.zeros(1, 1, 1, _SAMPLING_USERS, dtype=torch.int32),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def append_sampling(self, logits_tt: ttnn.Tensor, out_tok_tt: ttnn.Tensor) -> None:
        """Append topk → sampling ops; result written into ``out_tok_tt``.

        Safe to call inside a trace_capture block.
        ``logits_tt`` shape ``[1,1,1,vocab]`` BFLOAT16 TILE; we replicate to
        ``[1,1,32,vocab]`` for the kernel's 32-user requirement.
        """
        logits_32 = ttnn.repeat(logits_tt, ttnn.Shape([1, 1, _SAMPLING_USERS, 1]))
        # ttnn.topk's multicore path requires width >= 8192 (multi_core_min_width).
        # Our CP vocab (2048) and codec vocab (3072) sit below that threshold, forcing
        # single-core topk at ~570 µs/call. Pad with -inf so padded positions never
        # appear in top-K → multicore activates → ~65-core run at ~240 µs/call.
        _vocab = int(logits_32.shape[-1])
        if _vocab < 8192:
            logits_padded = ttnn.pad(
                logits_32,
                [(0, 0), (0, 0), (0, 0), (0, 8192 - _vocab)],
                value=-1e30,
            )
            ttnn.deallocate(logits_32)
        else:
            logits_padded = logits_32
        topk_values_tt, topk_indices_tt = ttnn.topk(
            logits_padded,
            k=_SAMPLING_MAX_TOP_K,
            dim=-1,
            largest=True,
            sorted=True,
            sub_core_grids=self._sub_core_grids,
        )
        ttnn.deallocate(logits_padded)
        indices_rm = ttnn.to_layout(topk_indices_tt, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(topk_indices_tt)
        indices_int32 = ttnn.typecast(indices_rm, ttnn.int32)
        ttnn.deallocate(indices_rm)
        ttnn.sampling(
            topk_values_tt,
            indices_int32,
            k=self.k_tensor,
            p=self.p_tensor,
            temp=self.temp_tensor,
            seed=self.seed,
            sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                ttnn.CoreCoord(0, 0),
                _SAMPLING_USERS,
                self._sub_core_grids,
                row_wise=True,
            ),
            output_tensor=out_tok_tt,
        )
        ttnn.deallocate(topk_values_tt)
        ttnn.deallocate(indices_int32)


def _slice_user0_token(tok_32_tt: ttnn.Tensor) -> ttnn.Tensor:
    """Slice user[0]'s token from a ``[1,1,1,32]`` sampling output → ``[1,1,1,1]``.

    The result is a uint32 RM tensor suitable for ``ttnn.embedding`` indices.
    """
    return ttnn.slice(tok_32_tt, [0, 0, 0, 0], [1, 1, 1, 1])


def sample_from_tt_vocab_logits(
    logits_tt: ttnn.Tensor,
    *,
    temperature: float,
    top_k: int,
    greedy: bool,
    repetition_penalty: float = 1.0,
    generated_tokens: Optional[List[int]] = None,
    prof_acc: Optional[Dict[str, float]] = None,
) -> int:
    """Argmax or sample from on-device logits [..., seq, vocab].

    **Greedy** uses full-vocab ``to_torch`` then CPU ``argmax`` (same sequence-axis
    handling as sampling). The previous ``untilize`` + on-device ``argmax`` path
    could return invalid token IDs for some sliced / trace-backed logits tensors,
    which then broke ``F.embedding`` in the CP decode loop.

    **Sampling** (temperature / top_k / multinomial) uses full-vocab ``to_torch`` on
    device logits, then :func:`sample_token` on the host. On-device top-k paths were
    measured as slower than bf16 logits D2H for this demo's vocab width and trace count.

    If ``prof_acc`` is set, adds seconds to keys ``device_logits`` (full logits
    ``to_torch``) and ``cpu_sample`` (:func:`sample_token` only).
    """
    _pc = time.perf_counter
    t0 = _pc() if prof_acc is not None else 0.0
    th = ttnn.to_torch(logits_tt, dtype=torch.bfloat16)
    if th.ndim >= 3 and th.shape[-2] > 1:
        th = th[:, :, -1, :]
    th1d = th.reshape(-1).contiguous()
    t1 = _pc() if prof_acc is not None else 0.0
    if greedy:
        out = int(th1d.float().argmax().item())
        if prof_acc is not None:
            prof_acc["device_logits"] = prof_acc.get("device_logits", 0.0) + (t1 - t0)
        return out
    with torch.inference_mode():
        out = sample_token(
            th1d,
            temperature,
            top_k,
            greedy,
            repetition_penalty,
            generated_tokens,
        )
    if prof_acc is not None:
        prof_acc["device_logits"] = prof_acc.get("device_logits", 0.0) + (t1 - t0)
        prof_acc["cpu_sample"] = prof_acc.get("cpu_sample", 0.0) + (_pc() - t1)
    return out


SUPPORTED_PREFILL_LENS = [32, 64, 96, 128, 192, 256, 384, 512, 1024]


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
    use_2cq: bool = False,
    streaming_decoder=None,
) -> Union[Tuple[torch.Tensor, dict], Tuple[None, dict]]:
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
        use_2cq: If True, issue H2D copies on CQ1 and overlap with trace on CQ0 (requires
            device opened with num_command_queues=2; see tech_reports/AdvancedPerformanceOptimizationsForModels).

    Returns:
        (codes, compile_timings): codes are [seq_len, 16] or None. compile_timings holds
        warmup, trace_capture, avg_decode_ms, steady_avg_decode_ms, steady_frames_per_sec,
        num_generated_frames, and use_2cq.
    """
    from models.demos.qwen3_tts.tt.rope import compute_rope_frequencies, get_rope_tensors, get_transformation_mat

    mode_str = "with KV cache" if use_kv_cache else "without KV cache"
    print(f"\nGenerating codes with TTNN ({mode_str})...")
    if use_2cq and not use_trace:
        print("  Note: 2 CQ mode requires trace; ignoring --use-2cq")
        use_2cq = False
    if use_2cq:
        print("2 CQ: H2D on CQ1, traces on CQ0 (AdvancedPerformanceOptimizationsForModels §2.3.2)")

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
    # Multiple traced-prefill buckets — short prompts (e.g. 61 tokens) take
    # bucket 64 to avoid numerical drift from padding deeper into a 128 trace.
    _TRACED_PREFILL_BUCKETS = (32, 64, 128)
    if real_seq_len <= _TRACED_PREFILL_BUCKETS[-1]:
        padded_seq_len = next(b for b in _TRACED_PREFILL_BUCKETS if b >= real_seq_len)
    else:
        padded_seq_len = get_padded_prefill_len(real_seq_len)
    print(f"  Input padding: {real_seq_len} -> {padded_seq_len} (bucket)")

    if padded_seq_len > real_seq_len:
        pad_len = padded_seq_len - real_seq_len
        pad_zeros = ttnn.from_torch(
            torch.zeros(1, 1, pad_len, talker_h, dtype=torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        inputs_embeds_tt = ttnn.concat([inputs_embeds_tt, pad_zeros], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
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
    t_warmup_start = time.time()
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
    t_warmup_end = time.time()

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

    # === STEP 3.5: Capture Talker prefill traces for buckets [32, 64, 128] ===
    # Standard path inside the trace — no prefill_attn_mask, so attention uses
    # k=freshly projected (sliced, width=bucket) + fused SDPA (is_causal=True).
    # Numerics identical to non-traced run. Cache write goes through fill_cache
    # (constant batch_idx=0 → trace-safe).
    TRACE_PREFILL_BUCKETS = [32, 64, 128]
    talker_prefill_traces = {}
    print(f"  Capturing Talker prefill traces for buckets {TRACE_PREFILL_BUCKETS}...")
    for _bucket in TRACE_PREFILL_BUCKETS:
        # Persistent input embed buffer (zero-padded; per-call copy_h2d overwrites).
        _pf_embed_tt = ttnn.from_torch(
            torch.zeros(1, 1, _bucket, talker_h, dtype=torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Constant RoPE for positions [0..bucket); baked into trace.
        _pf_cos_tt, _pf_sin_tt = get_rope_tensors(
            device, head_dim, _bucket, torch.arange(_bucket), model.talker_config.rope_theta
        )
        # Untraced warmup (compiles kernels).
        _wu_h, _ = model.talker.forward_from_hidden(
            _pf_embed_tt,
            _pf_cos_tt,
            _pf_sin_tt,
            talker_trans_mat,
            kv_caches=talker_kv_caches,
            start_pos=0,
            mode="prefill",
        )
        _ = model.talker.get_codec_logits(_wu_h)
        ttnn.synchronize_device(device)

        _trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        try:
            _trace_h, _ = model.talker.forward_from_hidden(
                _pf_embed_tt,
                _pf_cos_tt,
                _pf_sin_tt,
                talker_trans_mat,
                kv_caches=talker_kv_caches,
                start_pos=0,
                mode="prefill",
            )
            _trace_logits = model.talker.get_codec_logits(_trace_h)
        finally:
            ttnn.end_trace_capture(device, _trace_id, cq_id=0)
        ttnn.synchronize_device(device)
        talker_prefill_traces[_bucket] = {
            "trace_id": _trace_id,
            "embed_tt": _pf_embed_tt,
            "hidden_out": _trace_h,
            "logits_out": _trace_logits,
        }
        print(f"    bucket={_bucket}: trace captured")

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

    generated_code0_tokens = []

    # PCC check: when TT_QWEN3_PCC_TRACE_PREFILL=1, run BOTH non-traced and
    # traced prefill on the same input and report PCC of the codec logits
    # at the sample position. Cache state must be reset between the two runs.
    import os as _os_pcc

    _do_pcc = _os_pcc.environ.get("TT_QWEN3_PCC_TRACE_PREFILL", "0") == "1"
    _ref_logits_torch = None
    if _do_pcc and padded_seq_len in talker_prefill_traces:
        # Non-traced reference run (writes cache via fill_cache).
        _ref_h, _ = model.talker.forward_from_hidden(
            inputs_embeds_tt,
            prefill_cos_tt,
            prefill_sin_tt,
            talker_trans_mat,
            kv_caches=talker_kv_caches,
            start_pos=0,
            mode="prefill",
        )
        _ref_logits = model.talker.get_codec_logits(_ref_h)
        ttnn.synchronize_device(device)
        _ref_logits_torch = ttnn.to_torch(_ref_logits).squeeze(1).float()
        # Reset cache so the upcoming traced run starts identically.
        for (k_zero, v_zero), (k_cache, v_cache) in zip(cp_kv_zero_hosts[:0] + [], talker_kv_caches[:0] + []):
            pass  # placeholder; we'll reset via fresh zero-host below
        for layer_kv in talker_kv_caches:
            k_cache, v_cache = layer_kv
            _kc_shape = tuple(int(d) for d in k_cache.shape)
            _vc_shape = tuple(int(d) for d in v_cache.shape)
            _kz = ttnn.from_torch(
                torch.zeros(_kc_shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            _vz = ttnn.from_torch(
                torch.zeros(_vc_shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(_kz, k_cache)
            ttnn.copy_host_to_device_tensor(_vz, v_cache)

    if padded_seq_len in talker_prefill_traces:
        _pf = talker_prefill_traces[padded_seq_len]
        _embed_host_torch = ttnn.to_torch(inputs_embeds_tt)
        _embed_host = ttnn.from_torch(
            _embed_host_torch.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(_embed_host, _pf["embed_tt"])
        ttnn.execute_trace(device, _pf["trace_id"], cq_id=0, blocking=True)
        prefill_logits_out = _pf["logits_out"]
        prefill_hidden_out = _pf["hidden_out"]
        codec_logits_full = ttnn.to_torch(prefill_logits_out).squeeze(1).float()
        codec_logits_torch = codec_logits_full[0, real_seq_len - 1, :]
        if _do_pcc and _ref_logits_torch is not None:
            _a = _ref_logits_torch[0, real_seq_len - 1, :]
            _b = codec_logits_torch
            _pcc = torch.corrcoef(torch.stack([_a.flatten(), _b.flatten()]))[0, 1].item()
            _max_abs = (_a - _b).abs().max().item()
            print(f"  [PCC] traced vs non-traced @ pos={real_seq_len-1}: PCC={_pcc:.6f}  max|Δ|={_max_abs:.4f}")
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
        print(
            f"  Talker prefill done (TRACED bucket={padded_seq_len}): {(t_prefill_end - t_prefill_start)*1000:.1f} ms, token_0={token_0}"
        )
    else:
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
        return None, {
            "warmup": t_warmup_end - t_warmup_start,
            "trace_capture": 0.0,
            "avg_decode_ms": 0.0,
            "steady_avg_decode_ms": 0.0,
            "steady_frames_per_sec": 0.0,
            "num_generated_frames": 0,
            "use_2cq": use_2cq,
        }

    talker_pos = real_seq_len
    talker_hidden_tt = prefill_hidden_out

    ttnn.deallocate(prefill_cos_tt)
    ttnn.deallocate(prefill_sin_tt)

    # === STEP 5: Pre-allocate ALL trace input tensors, then capture traces ===
    # Pre-allocate all trace input tensors BEFORE any trace capture.
    # These are persistent buffers H2D'd per-frame and READ inside the trace; keep
    # them in L1 so the consuming kernel skips a DRAM→L1 staging on every call.
    trace_embed_tt = ttnn.from_torch(
        torch.zeros(1, 1, 1, talker_h, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    trace_cos_tt = ttnn.from_torch(
        torch.ones(1, 1, 1, head_dim, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    trace_sin_tt = ttnn.from_torch(
        torch.zeros(1, 1, 1, head_dim, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    trace_cur_pos_tt = ttnn.from_torch(
        torch.tensor([padded_seq_len], dtype=torch.int32),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,  # paged_fused_update_cache requires DRAM
    )
    trace_mask_tt = ttnn.from_torch(
        torch.full((1, _talker_num_heads, 1, max_talker_seq_len), float("-inf")),
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
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
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    cp_trace_prefill_mask_host = ttnn.from_torch(
        cp_prefill_mask_host_torch.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
    )
    cp_trace_prefill_embed_tt = ttnn.from_torch(
        torch.zeros(1, 1, 2, talker_h, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    _n_cp_decode = config.num_code_groups - 2
    cp_decode_cos_h2d, cp_decode_sin_h2d, cp_decode_mask_h2d = build_cp_decode_trace_h2d_constants(
        cp_cos_table, cp_sin_table, _cp_num_heads, max_cp_seq_len, _n_cp_decode
    )

    cp_trace_decode_embed_tts = [
        ttnn.from_torch(
            torch.zeros(1, 1, 1, talker_h, dtype=torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        for _ in range(2)
    ]
    cp_trace_decode_cos_tts = [
        ttnn.to_device(h, device, memory_config=ttnn.L1_MEMORY_CONFIG) for h in cp_decode_cos_h2d
    ]
    cp_trace_decode_sin_tts = [
        ttnn.to_device(h, device, memory_config=ttnn.L1_MEMORY_CONFIG) for h in cp_decode_sin_h2d
    ]
    cp_trace_decode_mask_tts = [
        ttnn.to_device(h, device, memory_config=ttnn.L1_MEMORY_CONFIG) for h in cp_decode_mask_h2d
    ]

    # === STEP 5: Capture ALL traces (after prefill, KV cache is populated) ===
    # Program cache must be warm before trace capture; otherwise compile uploads binaries
    # during capture and TT_FATAL: "Writes are not supported during trace capture".
    t_trace_start = time.time()

    # --- 5a: Talker decode trace ---
    print("  Untraced warmup: Talker decode + codec_head (same tensors as trace)...")
    _wu_th, _ = model.talker.forward_from_hidden(
        trace_embed_tt,
        trace_cos_tt,
        trace_sin_tt,
        talker_trans_mat,
        kv_caches=talker_kv_caches,
        cur_pos_tensor=trace_cur_pos_tt,
        decode_attn_mask=trace_mask_tt,
        mode="decode",
    )
    _wu_codec_logits = model.talker.get_codec_logits(_wu_th)

    # On-device codec0 sampling/argmax — emit token id from the trace so the
    # hot loop only D2H's an int (4B) instead of the full vocab logits.
    #
    #   greedy=True               → _argmax_into (existing fast path)
    #   greedy=False & DS env on  → _DeviceSampler (topk + ttnn.sampling)
    #   greedy=False & DS env off → no in-trace sampling; loop falls back to
    #                                full-logits D2H + host sample.
    _device_sampling = bool(int(os.environ.get("TT_QWEN3_DEVICE_SAMPLING", "0")))
    talker_codec0_sampler: Optional[_DeviceSampler] = None
    if _device_sampling and not config.greedy:
        talker_codec0_sampler = _DeviceSampler(device, top_k=config.top_k, top_p=1.0, temperature=config.temperature)
        talker_codec0_token_tt = talker_codec0_sampler.alloc_token_buf()
    else:
        talker_codec0_token_tt = _alloc_token_buf(device, shape=(1, 1, 1)) if config.greedy else None
    cp_prefill_token_tt = _alloc_token_buf(device, shape=(1, 1, 2))
    if config.greedy:
        _argmax_into(_wu_codec_logits, talker_codec0_token_tt)
    elif talker_codec0_sampler is not None:
        talker_codec0_sampler.append_sampling(_wu_codec_logits, talker_codec0_token_tt)
    ttnn.synchronize_device(device)

    print("  Capturing Talker decode trace (includes codec_head)...")
    talker_decode_trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    try:
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
        if config.greedy:
            _argmax_into(trace_codec_logits_out, talker_codec0_token_tt)
        elif talker_codec0_sampler is not None:
            talker_codec0_sampler.append_sampling(trace_codec_logits_out, talker_codec0_token_tt)
    finally:
        ttnn.end_trace_capture(device, talker_decode_trace_id, cq_id=0)
    ttnn.synchronize_device(device)
    print("  Talker decode trace captured.")

    # --- 5b: CP prefill trace ---
    print("  Untraced warmup: CP prefill (same tensors as trace)...")
    for (k_zero, v_zero), (k_cache, v_cache) in zip(cp_kv_zero_hosts, cp_kv_caches_persistent):
        ttnn.copy_host_to_device_tensor(k_zero, k_cache)
        ttnn.copy_host_to_device_tensor(v_zero, v_cache)
    _wu_cp_pf_logits, cp_kv_caches_persistent = model.code_predictor.forward_single_step(
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
    if config.greedy:
        _argmax_into(_wu_cp_pf_logits, cp_prefill_token_tt)
    ttnn.synchronize_device(device)

    print("  Capturing CP prefill trace (seq_len=2, includes lm_heads[0])...")
    cp_prefill_trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    try:
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
        if config.greedy:
            # cp_prefill_logits_tt is [1,1,2,vocab]; argmax over -1 yields [1,1,2]; we
            # consume only index 1 on host (the post-code0 logits position).
            _argmax_into(cp_prefill_logits_tt, cp_prefill_token_tt)
    finally:
        ttnn.end_trace_capture(device, cp_prefill_trace_id, cq_id=0)
    ttnn.synchronize_device(device)
    print("  CP prefill trace captured.")

    # --- 5c: CP decode traces x13 ---
    # Step 2 on-device chain (when not greedy): each captured trace runs
    #   forward → topk → sampling → ttnn.embedding(token, codec_pred_table[code_idx-1])
    # with the embedding output written into ``cp_trace_decode_embed_tts[buf_out]``,
    # i.e. the OTHER buffer that the NEXT trace will read from. This eliminates the
    # CPU sample + F.embedding + H2D between consecutive CP decode steps.
    # In greedy mode we keep the simpler argmax path (no sampling kernel needed).
    cp_decode_trace_ids = [[], []]
    cp_decode_logits_tts = [[], []]
    cp_decode_token_tts = [[], []]
    # On-device chain (opt-in via TT_QWEN3_DEVICE_CP_CHAIN=1): in-trace topk +
    # sampling + embedding + copy → next trace's input buffer. Currently regresses
    # ~4.6 ms/frame because the 32-user ttnn.sampling kernel is heavy at batch=1;
    # the design pays off after Step 5 (mega-trace fusion) cuts inter-trace syncs.
    import os as _os

    _device_cp_chain = bool(int(_os.environ.get("TT_QWEN3_DEVICE_CP_CHAIN", "0")))
    _device_cp_sampling = False  # batch=1 regression — see comments above the chain path.
    _greedy_chain = _device_cp_chain and config.greedy
    cp_sampler = None
    if _device_cp_chain and not config.greedy:
        cp_sampler = _DeviceSampler(device, top_k=config.top_k, top_p=1.0, temperature=config.temperature)
        print(
            f"  CP decode on-device chain (sampling): topk(k={_SAMPLING_MAX_TOP_K}) + sampling + embed "
            f"(top_k={cp_sampler.top_k}, temp={cp_sampler.temperature}) — TT_QWEN3_DEVICE_CP_CHAIN=1"
        )
    elif _greedy_chain:
        print("  CP decode on-device chain (greedy): argmax + embed — TT_QWEN3_DEVICE_CP_CHAIN=1")
    print(f"  Capturing {config.num_code_groups - 2} CP decode traces (one per lm_head)...")
    for _buf_i in range(2):
        for _trace_i, _step_code_idx in enumerate(range(2, config.num_code_groups)):
            _cp_cos_tt = cp_trace_decode_cos_tts[_trace_i]
            _cp_sin_tt = cp_trace_decode_sin_tts[_trace_i]
            _cp_mask_tt = cp_trace_decode_mask_tts[_trace_i]
            _buf_out = (_buf_i + 1) % 2  # this trace writes its output embed for the NEXT trace
            # The codec_pred table for this step's sampled token: code_idx-1 maps to
            # CodePredictor.codec_embeddings_tt[code_idx-1] (table for the token id we
            # JUST sampled; e.g., when code_idx=2 we sampled code 2, look up its embed
            # in code_pred table 1 → output is the input embedding for trace[code_idx=3]).
            _embed_table_tt = model.code_predictor.codec_embeddings_tt[_step_code_idx - 1]
            print(f"    Untraced warmup: CP decode (buf={_buf_i}, generation_step={_step_code_idx})...")
            _wu_cp_dc_logits, cp_kv_caches_persistent = model.code_predictor.forward_single_step(
                cp_trace_decode_embed_tts[_buf_i],
                _cp_cos_tt,
                _cp_sin_tt,
                cp_trans_mat,
                generation_step=_step_code_idx,
                kv_caches=cp_kv_caches_persistent,
                start_pos=_step_code_idx,
                mode="decode",
                cur_pos_tensor=None,
                decode_attn_mask=_cp_mask_tt,
                return_hidden_state=False,
            )
            if config.greedy:
                _tok_buf = _alloc_token_buf(device, shape=(1, 1, 1))
                _argmax_into(_wu_cp_dc_logits, _tok_buf)
                if _greedy_chain:
                    # Warmup the chain ops (load kernels — same TT_FATAL constraint as
                    # the sampling chain warmup below).
                    _wu_tok_4d = ttnn.reshape(_tok_buf, [1, 1, 1, 1])
                    _wu_embed_out = ttnn.embedding(
                        _wu_tok_4d,
                        _embed_table_tt,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )
                    _wu_embed_out_4d = ttnn.reshape(_wu_embed_out, [1, 1, 1, _wu_embed_out.shape[-1]])
                    ttnn.copy(_wu_embed_out_4d, cp_trace_decode_embed_tts[_buf_out])
                    ttnn.deallocate(_wu_embed_out)  # frees the underlying buffer (4d is an alias view)
            elif _device_cp_chain:
                _tok_buf = cp_sampler.alloc_token_buf()  # [1,1,1,32] uint32 RM
                cp_sampler.append_sampling(_wu_cp_dc_logits, _tok_buf)
                # Warmup the in-trace chain: slice → embedding → reshape → copy → target.
                # First invocation of each kernel loads its binary; that's a device write,
                # which TT_FATAL'd if it happened during trace_capture. Run it untraced now.
                _wu_tok_slice = _slice_user0_token(_tok_buf)
                _wu_embed_out = ttnn.embedding(
                    _wu_tok_slice,
                    _embed_table_tt,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                _wu_embed_out_4d = ttnn.reshape(_wu_embed_out, [1, 1, 1, _wu_embed_out.shape[-1]])
                ttnn.copy(_wu_embed_out_4d, cp_trace_decode_embed_tts[_buf_out])
                ttnn.deallocate(_wu_tok_slice)
                ttnn.deallocate(_wu_embed_out)  # frees the underlying buffer (4d is an alias view)
            elif _device_cp_sampling:
                _tok_buf = cp_sampler.alloc_token_buf()  # [1,1,1,32] uint32 RM
                cp_sampler.append_sampling(_wu_cp_dc_logits, _tok_buf)
            else:
                _tok_buf = None  # CPU sample path: no device token buffer needed.
            cp_decode_token_tts[_buf_i].append(_tok_buf)
            ttnn.synchronize_device(device)

            _trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            try:
                _logits_tt, _ = model.code_predictor.forward_single_step(
                    cp_trace_decode_embed_tts[_buf_i],
                    _cp_cos_tt,
                    _cp_sin_tt,
                    cp_trans_mat,
                    generation_step=_step_code_idx,
                    kv_caches=cp_kv_caches_persistent,
                    start_pos=_step_code_idx,
                    mode="decode",
                    cur_pos_tensor=None,
                    decode_attn_mask=_cp_mask_tt,
                    return_hidden_state=False,
                )
                if config.greedy:
                    _argmax_into(_logits_tt, _tok_buf)
                    if _greedy_chain:
                        # On-device chain (greedy): take argmax token, look up its embed,
                        # copy into the next CP decode trace's input buffer.
                        _tok_4d = ttnn.reshape(_tok_buf, [1, 1, 1, 1])
                        _embed_out = ttnn.embedding(
                            _tok_4d,
                            _embed_table_tt,
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=ttnn.L1_MEMORY_CONFIG,
                        )
                        _embed_out_4d = ttnn.reshape(_embed_out, [1, 1, 1, _embed_out.shape[-1]])
                        ttnn.copy(_embed_out_4d, cp_trace_decode_embed_tts[_buf_out])
                        ttnn.deallocate(_embed_out_4d)
                elif _device_cp_chain:
                    cp_sampler.append_sampling(_logits_tt, _tok_buf)
                    # On-device chain: take user[0]'s sampled token, look up its embed,
                    # copy into the next CP decode trace's input buffer. Closes the loop.
                    _tok_slice = _slice_user0_token(_tok_buf)
                    _embed_out = ttnn.embedding(
                        _tok_slice,
                        _embed_table_tt,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    _embed_out_4d = ttnn.reshape(_embed_out, [1, 1, 1, _embed_out.shape[-1]])
                    ttnn.copy(_embed_out_4d, cp_trace_decode_embed_tts[_buf_out])
                    ttnn.deallocate(_tok_slice)
                    ttnn.deallocate(_embed_out_4d)
                elif _device_cp_sampling:
                    cp_sampler.append_sampling(_logits_tt, _tok_buf)
            finally:
                ttnn.end_trace_capture(device, _trace_id, cq_id=0)
            ttnn.synchronize_device(device)
            cp_decode_trace_ids[_buf_i].append(_trace_id)
            cp_decode_logits_tts[_buf_i].append(_logits_tt)
    print(f"  Captured {len(cp_decode_trace_ids[0])} CP decode traces x2 buffers.")
    t_trace_end = time.time()
    print("  All traces captured. Starting measured inference...")
    print(f"  Device CQ mode: {'2 (H2D on CQ1, traces on CQ0)' if use_2cq else '1 (H2D and traces share CQ0)'}")

    # === STEP 6: Measured inference (generation loop only; prefill already ran in STEP 4) ===
    decode_step_times = []
    talker_times_ms = []
    cp_times_ms = []
    t_first_decode_end = 0.0
    trace_cq0_idle = ttnn.record_event(device, 0) if use_2cq else None
    h2d_cq = 1 if use_2cq else 0
    cp_decode_input_ready = [trace_cq0_idle, trace_cq0_idle]

    # Preallocated host buffers (generation hot loop): avoids per-step torch/tensor churn.
    token_id_buf = torch.zeros((1, 1), dtype=torch.long)
    cp_prefill_embed_cpu = torch.empty(1, 1, 2, talker_h, dtype=torch.bfloat16)
    talker_embed_cpu = torch.empty(1, 1, 1, talker_h, dtype=torch.bfloat16)
    cp_decode_embed_cpu = torch.empty(1, 1, 1, talker_h, dtype=torch.bfloat16)
    acc_code_embed = torch.zeros(1, 1, talker_h, dtype=torch.float32)
    talker_cos_h2d, talker_sin_h2d, talker_mask_h2d, talker_cur_pos_h2d = build_talker_decode_trace_h2d_constants(
        talker_cos_table, talker_sin_table, _talker_num_heads, max_talker_seq_len, real_seq_len
    )

    # NB: Do not reuse a single ttnn.from_torch(..., layout=TILE_LAYOUT) across loop iterations:
    # tilization copies into an internal host buffer; mutating the torch tensor + copy_ does not refresh it.

    frame_breakdown_sums = {
        "cp_input_prep_ms": 0.0,
        "cp_kv_restore_ms": 0.0,
        "cp_prefill_ms": 0.0,
        "cp_decode_ms": 0.0,
        "build_acc_embed_ms": 0.0,
        "talker_decode_ms": 0.0,
        "codec0_sample_device_logits_ms": 0.0,
        "codec0_sample_cpu_ms": 0.0,
        "cp_prefill_sample_device_logits_ms": 0.0,
        "cp_prefill_sample_cpu_ms": 0.0,
        "cp_decode_samples_device_logits_ms": 0.0,
        "cp_decode_samples_cpu_ms": 0.0,
    }
    frame_breakdown_frames = 0

    try:
        # === STEP 6: AR generation loop — delegates to the shared helper
        # in tt/utils.py so the demo path here and the server path in
        # run_inference run the EXACT same loop body.
        from models.demos.qwen3_tts.tt.utils import DecodeLoopState, ar_decode_loop

        loop_state = DecodeLoopState(
            device=device,
            cp_kv_caches_persistent=cp_kv_caches_persistent,
            cp_kv_zero_hosts=cp_kv_zero_hosts,
            cp_prefill_trace_id=cp_prefill_trace_id,
            cp_prefill_logits_tt=cp_prefill_logits_tt,
            cp_decode_trace_ids=cp_decode_trace_ids,
            cp_decode_logits_tts=cp_decode_logits_tts,
            cp_trace_prefill_embed_tt=cp_trace_prefill_embed_tt,
            cp_trace_prefill_mask_tt=cp_trace_prefill_mask_tt,
            cp_trace_prefill_cos_tt=cp_trace_prefill_cos_tt,
            cp_trace_prefill_sin_tt=cp_trace_prefill_sin_tt,
            cp_trace_prefill_mask_host=cp_trace_prefill_mask_host,
            cp_trace_prefill_cos_host=cp_trace_prefill_cos_host,
            cp_trace_prefill_sin_host=cp_trace_prefill_sin_host,
            cp_trace_decode_embed_tts=cp_trace_decode_embed_tts,
            code_pred_embeds=code_pred_embeds,
            codec_embed_torch=codec_embed_torch,
            talker_decode_trace_id=talker_decode_trace_id,
            trace_embed_tt=trace_embed_tt,
            trace_cos_tt=trace_cos_tt,
            trace_sin_tt=trace_sin_tt,
            trace_cur_pos_tt=trace_cur_pos_tt,
            trace_mask_tt=trace_mask_tt,
            trace_hidden_out=trace_hidden_out,
            trace_codec_logits_out=trace_codec_logits_out,
            talker_codec0_token_tt=talker_codec0_token_tt,
            talker_cos_h2d=talker_cos_h2d,
            talker_sin_h2d=talker_sin_h2d,
            talker_mask_h2d=talker_mask_h2d,
            talker_cur_pos_h2d=talker_cur_pos_h2d,
            token_id_buf=token_id_buf,
            cp_prefill_embed_cpu=cp_prefill_embed_cpu,
            cp_decode_embed_cpu=cp_decode_embed_cpu,
            talker_embed_cpu=talker_embed_cpu,
            acc_code_embed=acc_code_embed,
            talker_hidden_tt=talker_hidden_tt,
            talker_pos=talker_pos,
            real_seq_len=real_seq_len,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            token_0=token_0,
        )
        codes_tensor, frame_breakdown_avg_ms_helper, t_first_decode_end, _t_last = ar_decode_loop(
            loop_state,
            config,
            use_2cq,
            streaming_decoder=streaming_decoder,
            sample_token_fn=sample_token,
            sample_from_tt_vocab_logits_fn=sample_from_tt_vocab_logits,
        )
        decode_step_times = loop_state.decode_step_times_ms
        talker_times_ms = loop_state.talker_times_ms
        cp_times_ms = loop_state.cp_times_ms
        all_codes = [] if codes_tensor is None else codes_tensor.tolist()

    finally:
        ttnn.synchronize_device(device)
        ttnn.release_trace(device, talker_decode_trace_id)
        ttnn.release_trace(device, cp_prefill_trace_id)
        for _tid_list in cp_decode_trace_ids:
            for _tid in _tid_list:
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
        ]:
            if t is not None:
                ttnn.deallocate(t)
        for t in cp_trace_decode_embed_tts:
            ttnn.deallocate(t)
        for t in cp_trace_decode_cos_tts + cp_trace_decode_sin_tts + cp_trace_decode_mask_tts:
            ttnn.deallocate(t)
        deallocate_kv_cache(talker_kv_caches)
        deallocate_kv_cache(cp_kv_caches_persistent)

    print(f"  Generated {len(all_codes)} code frames")

    if len(all_codes) == 0:
        return None, {
            "warmup": t_warmup_end - t_warmup_start,
            "trace_capture": t_trace_end - t_trace_start,
            "avg_decode_ms": 0.0,
            "steady_avg_decode_ms": 0.0,
            "steady_frames_per_sec": 0.0,
            "num_generated_frames": 0,
            "use_2cq": use_2cq,
            "frame_breakdown_avg_ms": {},
        }

    codes = torch.tensor(all_codes, dtype=torch.long)
    print(f"  Code 0 sample (first 5 frames): {codes[:5, 0].tolist()}")
    print(f"  Code 0 in valid range [0,2047]: {((codes[:,0]>=0) & (codes[:,0]<=2047)).sum()}/{len(codes)}")
    torch.save(codes, "/tmp/last_generated_codes.pt")

    # === Performance metrics ===
    prefill_ms = (t_prefill_end - t_prefill_start) * 1000
    steady_avg_decode_ms = 0.0
    if decode_step_times:
        ttft_ms = prefill_ms + decode_step_times[0]
        if len(decode_step_times) > 1:
            steady_ms = decode_step_times[1:]
            steady_avg_decode_ms = sum(steady_ms) / len(steady_ms)
            tokens_per_sec = 1000.0 / steady_avg_decode_ms
        else:
            steady_avg_decode_ms = decode_step_times[0]
            tokens_per_sec = 1000.0 / decode_step_times[0]
    else:
        ttft_ms = prefill_ms
        tokens_per_sec = 0.0

    avg_decode_ms = sum(decode_step_times) / len(decode_step_times) if decode_step_times else 0.0

    print(f"\n  --- Performance (Qwen3-TTS on N150, all traced) ---")
    print(f"  Prefill  ({real_seq_len} real / {padded_seq_len} padded tokens): {prefill_ms:.1f} ms")
    print(f"  TTFT     (prefill + 1 decode):  {ttft_ms:.1f} ms")
    print(f"  Decode throughput:              {tokens_per_sec:.2f} frames/sec")
    if decode_step_times:
        print(f"  Avg decode step:                {avg_decode_ms:.1f} ms/frame")
    if talker_times_ms:
        print(f"  Avg Talker decode:              {sum(talker_times_ms)/len(talker_times_ms):.1f} ms/frame")
    if cp_times_ms:
        print(f"  Avg CodePredictor:              {sum(cp_times_ms)/len(cp_times_ms):.1f} ms/frame")
    frame_breakdown_avg_ms = (
        {k: v / frame_breakdown_frames for k, v in frame_breakdown_sums.items()} if frame_breakdown_frames > 0 else {}
    )
    if frame_breakdown_avg_ms:
        print(f"  --- Frame breakdown (avg ms/frame, {frame_breakdown_frames} frames) ---")
        print(f"    CP input prep (D2H talker hidden + embed): {frame_breakdown_avg_ms['cp_input_prep_ms']:.2f}")
        print(f"    CP KV + mask restore H2D:                  {frame_breakdown_avg_ms['cp_kv_restore_ms']:.2f}")
        print(f"    CP prefill trace + 1st sample:             {frame_breakdown_avg_ms['cp_prefill_ms']:.2f}")
        print(f"    CP decode traces + samples:                {frame_breakdown_avg_ms['cp_decode_ms']:.2f}")
        print(f"    Build accumulated codec embed (CPU):       {frame_breakdown_avg_ms['build_acc_embed_ms']:.2f}")
        print(f"    Talker decode trace (wall sub-interval):   {frame_breakdown_avg_ms['talker_decode_ms']:.2f}")
        print(
            f"    Codec0 sample D2H logits / CPU:           {frame_breakdown_avg_ms['codec0_sample_device_logits_ms']:.2f} / {frame_breakdown_avg_ms['codec0_sample_cpu_ms']:.2f}"
        )
        print(
            f"    CP prefill sample D2H / CPU:               {frame_breakdown_avg_ms['cp_prefill_sample_device_logits_ms']:.2f} / {frame_breakdown_avg_ms['cp_prefill_sample_cpu_ms']:.2f}"
        )
        print(
            f"    CP decode samples D2H / CPU (sum 14):       {frame_breakdown_avg_ms['cp_decode_samples_device_logits_ms']:.2f} / {frame_breakdown_avg_ms['cp_decode_samples_cpu_ms']:.2f}"
        )
    print(
        f"  Traced: Talker decode, CP prefill, CP decode x{len(cp_decode_trace_ids[0])} (double-buffered) "
        f"(Talker prefill: non-traced)"
    )
    print(f"  ----------------------------------------")

    compile_timings = {
        "warmup": t_warmup_end - t_warmup_start,
        "trace_capture": t_trace_end - t_trace_start,
        "avg_decode_ms": avg_decode_ms,
        "steady_avg_decode_ms": steady_avg_decode_ms,
        "steady_frames_per_sec": tokens_per_sec,
        "num_generated_frames": len(all_codes),
        "use_2cq": use_2cq,
        "frame_breakdown_avg_ms": frame_breakdown_avg_ms,
        "frame_breakdown_frames": frame_breakdown_frames,
    }
    return codes, compile_timings


# ===========================================================================
# Server-mode infrastructure: warmup_bucket, TTSServerContext, init_server_context, run_inference
# ===========================================================================


def warmup_bucket(device, model, config, padded_seq_len: int):
    """Pre-compile all TTNN kernels for a given prefill bucket size (no output kept)."""
    from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat

    _TILE = 32
    max_talker_seq_len = (((padded_seq_len + config.max_new_tokens + 16) + _TILE - 1) // _TILE) * _TILE

    talker_h = model.talker_config.hidden_size
    head_dim = model.talker_config.head_dim
    _talker_num_heads = model.talker_config.num_attention_heads
    cp_head_dim = model.code_predictor_config.head_dim
    cp_rope_theta = model.code_predictor_config.rope_theta
    _cp_num_heads = model.code_predictor_config.num_attention_heads
    max_cp_seq_len = 32

    talker_trans_mat = get_transformation_mat(head_dim, device)
    cp_trans_mat = get_transformation_mat(cp_head_dim, device)

    print(f"  Warmup bucket={padded_seq_len} (max_talker_seq={max_talker_seq_len})...")

    # --- Talker prefill warmup ---
    wu_pf = ttnn.from_torch(
        torch.zeros(1, 1, padded_seq_len, talker_h),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    wu_pf_cos, wu_pf_sin = get_rope_tensors(
        device, head_dim, padded_seq_len, torch.arange(padded_seq_len), model.talker_config.rope_theta
    )
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

    # --- Talker decode warmup ---
    wu_dc = ttnn.from_torch(
        torch.zeros(1, 1, 1, talker_h),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    wu_dc_cos, wu_dc_sin = get_rope_tensors(
        device, head_dim, 1, torch.tensor([padded_seq_len]), model.talker_config.rope_theta
    )
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

    # --- CP prefill warmup ---
    wu_cp2 = ttnn.from_torch(
        torch.zeros(1, 1, 2, talker_h),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    wu_cp2_cos, wu_cp2_sin = get_rope_tensors(device, cp_head_dim, 2, torch.arange(2), cp_rope_theta)
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

    # --- CP decode warmup ---
    wu_cp1 = ttnn.from_torch(
        torch.zeros(1, 1, 1, talker_h),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    wu_cp1_cos, wu_cp1_sin = get_rope_tensors(device, cp_head_dim, 1, torch.tensor([2]), cp_rope_theta)
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
    print(f"  Warmup complete for bucket={padded_seq_len}.")


def warmup_all_buckets(device, model, config):
    """Pre-compile kernels for all supported prefill bucket sizes."""
    print("Warming up all prefill buckets...")
    for bucket in SUPPORTED_PREFILL_LENS:
        warmup_bucket(device, model, config, bucket)
    print("All buckets warmed up.")


@dataclass
class TTSServerContext:
    """Persistent per-request-reusable state for the TTS web server.

    Holds pre-captured CP + Talker traces and shared model helpers. ALL traces
    captured once at init time; per-request ``run_inference`` only executes
    cached traces — no compile, no new trace capture, no fresh KV allocation.

    Call ``init_server_context()`` to populate. Call ``run_inference()`` to use.
    """

    # Shared CP state (bucket-independent, pre-captured at startup)
    cp_kv_caches_persistent: list  # [(k_cache, v_cache), ...]
    cp_kv_zero_hosts: list  # [(k_zero, v_zero), ...]
    cp_prefill_trace_id: int
    cp_decode_trace_ids: list
    cp_prefill_logits_tt: object  # output buffer baked into CP prefill trace
    cp_decode_logits_tts: list  # output buffers baked into CP decode traces
    cp_trace_prefill_embed_tt: object
    cp_trace_prefill_cos_tt: object
    cp_trace_prefill_sin_tt: object
    cp_trace_prefill_mask_tt: object
    cp_trace_prefill_cos_host: object
    cp_trace_prefill_sin_host: object
    cp_trace_prefill_mask_host: object
    cp_trace_decode_embed_tts: list
    cp_trace_decode_cos_tts: list
    cp_trace_decode_sin_tts: list
    cp_trace_decode_mask_tts: list

    # Persistent Talker decode state (bucket-keyed; one trace per prefill bucket).
    # Eliminates per-request begin_trace_capture and KV allocation that
    # previously stalled the worker on the second request.
    talker_kv_caches_by_bucket: Dict[int, list]
    talker_kv_zero_hosts_by_bucket: Dict[int, list]
    talker_decode_trace_id_by_bucket: Dict[int, int]
    trace_decode_embed_tt: object  # persistent input buffer (1, 1, 1, talker_h)
    trace_decode_cos_tt: object
    trace_decode_sin_tt: object
    trace_decode_cur_pos_tt: object
    trace_decode_mask_tt_by_bucket: Dict[int, object]
    trace_decode_codec_logits_out_by_bucket: Dict[int, object]  # baked output
    trace_decode_hidden_out_by_bucket: Dict[int, object]  # baked talker hidden output

    # Shared model helpers
    talker_trans_mat: object
    cp_trans_mat: object
    talker_cos_table: torch.Tensor
    talker_sin_table: torch.Tensor
    cp_cos_table: torch.Tensor
    cp_sin_table: torch.Tensor
    code_pred_embeds: list
    codec_embed_torch: torch.Tensor
    max_cp_seq_len: int
    _talker_num_heads: int
    _cp_num_heads: int

    # Optional device-sampling output buffers (per-bucket) — populated only when
    # TT_QWEN3_DEVICE_SAMPLING=1; greedy/host-sampling paths leave these empty.
    trace_decode_codec0_token_tt_by_bucket: Dict[int, object] = field(default_factory=dict)


def init_server_context(device, model, config, main_weights: dict) -> "TTSServerContext":
    """
    Pre-compile all kernels and pre-capture CP traces for web server use.

    At startup:
    - Warms all prefill buckets (compiles all TTNN kernels once)
    - Allocates persistent CP KV caches
    - Captures all CP traces (1 prefill + N-2 decode)

    Talker KV caches and Talker decode trace are allocated per-request in run_inference()
    because the standard prefill path reallocates KV buffers (avoids L1 overflow on large buckets).

    Returns a TTSServerContext with all state needed for run_inference().
    """
    from models.demos.qwen3_tts.tt.rope import compute_rope_frequencies, get_rope_tensors, get_transformation_mat

    print("Initializing TTS server context...")

    # --- Warm up all buckets first (compiles all kernels) ---
    warmup_all_buckets(device, model, config)

    # Capture ECAPA SE/FC traces FIRST, BEFORE the heavier CP/Talker traces.
    # Order matters: traces captured later may end up in trace_region positions
    # adjacent to/overlapping with executed traces, causing corruption.
    print("  Capturing SE-block ECAPA traces (early) ...")
    model.speaker_encoder.capture_se_block_traces()
    print(f"    captured {len(getattr(model.speaker_encoder, '_se_traces', {}))} SE-block traces")
    print("  Capturing FC linear ECAPA trace (early) ...")
    model.speaker_encoder.capture_fc_trace()
    print(f"    fc_trace captured: {getattr(model.speaker_encoder, '_fc_trace', None) is not None}")

    _TILE = 32
    talker_h = model.talker_config.hidden_size
    head_dim = model.talker_config.head_dim
    _talker_num_heads = model.talker_config.num_attention_heads
    cp_head_dim = model.code_predictor_config.head_dim
    cp_rope_theta = model.code_predictor_config.rope_theta
    _cp_num_heads = model.code_predictor_config.num_attention_heads
    max_cp_seq_len = 32

    talker_trans_mat = get_transformation_mat(head_dim, device)
    cp_trans_mat = get_transformation_mat(cp_head_dim, device)

    # Pre-compute RoPE tables (sized for largest bucket + max_new_tokens)
    largest_bucket = SUPPORTED_PREFILL_LENS[-1]
    largest_max_talker_seq = (((largest_bucket + config.max_new_tokens + 16) + _TILE - 1) // _TILE) * _TILE
    _max_rope_pos = largest_max_talker_seq + config.max_new_tokens + 50
    talker_cos_table, talker_sin_table = compute_rope_frequencies(
        head_dim, _max_rope_pos, model.talker_config.rope_theta
    )
    cp_cos_table, cp_sin_table = compute_rope_frequencies(cp_head_dim, max_cp_seq_len + 5, cp_rope_theta)

    # CodePredictor embedding weights (for building next-token embeds)
    codec_embed_torch = ttnn.to_torch(model.talker.codec_embedding).squeeze(0).squeeze(0).float()
    code_pred_embeds = []
    for i in range(config.num_code_groups - 1):
        key = f"talker.code_predictor.model.codec_embedding.{i}.weight"
        if key in main_weights:
            code_pred_embeds.append(main_weights[key].float())
        else:
            code_pred_embeds.append(None)

    # === Shared CP persistent state (bucket-independent) ===
    print("  Setting up shared CP state...")
    cp_kv_caches_persistent = allocate_kv_cache(
        device=device,
        num_layers=model.code_predictor_config.num_hidden_layers,
        batch_size=1,
        num_kv_heads=model.code_predictor_config.num_key_value_heads,
        max_seq_len=max_cp_seq_len,
        head_dim=cp_head_dim,
    )
    cp_kv_zero_hosts = []
    for k_cache, v_cache in cp_kv_caches_persistent:
        k_zero = ttnn.from_torch(
            torch.zeros(*k_cache.shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        v_zero = ttnn.from_torch(
            torch.zeros(*v_cache.shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        cp_kv_zero_hosts.append((k_zero, v_zero))

    # Pre-allocate CP trace input tensors. seq=2 has padded_shape=[1,1,32,talker_h]
    # (one tile). DRAM placement avoids an L1-internal shard remap that some kernel
    # variants need to issue from host inside trace capture. Talker prefill embed
    # already uses DRAM here for the same reason.
    cp_trace_prefill_embed_tt = ttnn.from_torch(
        torch.zeros(1, 1, 2, talker_h, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
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
    cp_prefill_mask_torch = torch.full((1, _cp_num_heads, 2, max_cp_seq_len), float("-inf"))
    cp_prefill_mask_torch[0, :, 0, 0] = 0.0
    cp_prefill_mask_torch[0, :, 1, 0:2] = 0.0
    cp_trace_prefill_mask_tt = ttnn.from_torch(
        cp_prefill_mask_torch,
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cp_trace_prefill_mask_host = ttnn.from_torch(
        cp_prefill_mask_torch.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
    )

    cp_trace_decode_embed_tts = [
        ttnn.from_torch(
            torch.zeros(1, 1, 1, talker_h, dtype=torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for _ in range(2)
    ]
    _n_cp_decode = config.num_code_groups - 2
    cp_decode_cos_h2d, cp_decode_sin_h2d, cp_decode_mask_h2d = build_cp_decode_trace_h2d_constants(
        cp_cos_table, cp_sin_table, _cp_num_heads, max_cp_seq_len, _n_cp_decode
    )
    cp_trace_decode_cos_tts = [
        ttnn.to_device(h, device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for h in cp_decode_cos_h2d
    ]
    cp_trace_decode_sin_tts = [
        ttnn.to_device(h, device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for h in cp_decode_sin_h2d
    ]
    cp_trace_decode_mask_tts = [
        ttnn.to_device(h, device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for h in cp_decode_mask_h2d
    ]

    # Run dummy CP prefill to populate KV cache before trace capture
    dummy_cp_input = ttnn.from_torch(
        torch.zeros(1, 1, 2, talker_h, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    model.code_predictor.forward_single_step(
        dummy_cp_input,
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
    ttnn.deallocate(dummy_cp_input)
    ttnn.synchronize_device(device)

    # Capture CP prefill trace
    print("  Capturing CP prefill trace...")
    cp_prefill_trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    try:
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
    finally:
        ttnn.end_trace_capture(device, cp_prefill_trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    # Run dummy CP decode steps to populate KV positions 2-15 before decode trace capture
    dummy_cp_dec = ttnn.from_torch(
        torch.zeros(1, 1, 1, talker_h, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for _step_code_idx in range(2, config.num_code_groups):
        _cos, _sin = get_rope_tensors(device, cp_head_dim, 1, torch.tensor([_step_code_idx]), cp_rope_theta)
        _mask = ttnn.from_torch(
            torch.full((1, _cp_num_heads, 1, max_cp_seq_len), float("-inf")),
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        model.code_predictor.forward_single_step(
            dummy_cp_dec,
            _cos,
            _sin,
            cp_trans_mat,
            generation_step=_step_code_idx,
            kv_caches=cp_kv_caches_persistent,
            start_pos=_step_code_idx,
            mode="decode",
            cur_pos_tensor=None,
            decode_attn_mask=_mask,
            return_hidden_state=False,
        )
        ttnn.deallocate(_mask)
    ttnn.deallocate(dummy_cp_dec)
    ttnn.synchronize_device(device)

    # Capture CP decode traces (x14, one per code index 2..15)
    print(f"  Capturing {config.num_code_groups - 2} CP decode traces...")
    cp_decode_trace_ids = [[], []]
    cp_decode_logits_tts = [[], []]
    for _buf_i in range(2):
        for _trace_i, _step_code_idx in enumerate(range(2, config.num_code_groups)):
            _cp_cos_tt = cp_trace_decode_cos_tts[_trace_i]
            _cp_sin_tt = cp_trace_decode_sin_tts[_trace_i]
            _cp_mask_tt = cp_trace_decode_mask_tts[_trace_i]
            _, cp_kv_caches_persistent = model.code_predictor.forward_single_step(
                cp_trace_decode_embed_tts[_buf_i],
                _cp_cos_tt,
                _cp_sin_tt,
                cp_trans_mat,
                generation_step=_step_code_idx,
                kv_caches=cp_kv_caches_persistent,
                start_pos=_step_code_idx,
                mode="decode",
                cur_pos_tensor=None,
                decode_attn_mask=_cp_mask_tt,
                return_hidden_state=False,
            )
            ttnn.synchronize_device(device)

            _trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            try:
                _logits_tt, _ = model.code_predictor.forward_single_step(
                    cp_trace_decode_embed_tts[_buf_i],
                    _cp_cos_tt,
                    _cp_sin_tt,
                    cp_trans_mat,
                    generation_step=_step_code_idx,
                    kv_caches=cp_kv_caches_persistent,
                    start_pos=_step_code_idx,
                    mode="decode",
                    cur_pos_tensor=None,
                    decode_attn_mask=_cp_mask_tt,
                    return_hidden_state=False,
                )
            finally:
                ttnn.end_trace_capture(device, _trace_id, cq_id=0)
            ttnn.synchronize_device(device)
            cp_decode_trace_ids[_buf_i].append(_trace_id)
            cp_decode_logits_tts[_buf_i].append(_logits_tt)
    print(f"  Captured {len(cp_decode_trace_ids[0])} CP decode traces x2 buffers.")

    # Zero-reset CP KV caches for first real request
    for (k_zero, v_zero), (k_cache, v_cache) in zip(cp_kv_zero_hosts, cp_kv_caches_persistent):
        ttnn.copy_host_to_device_tensor(k_zero, k_cache)
        ttnn.copy_host_to_device_tensor(v_zero, v_cache)

    # ─── Persistent Talker decode state (one trace per prefill bucket) ──────
    # Hoists what used to be per-request alloc + capture inside run_inference.
    # Per-request handling now reduces to: zero-reset KV cache + execute trace.
    # Buckets match the Talker prefill traces above.
    TRACE_DECODE_BUCKETS = (32, 64, 96, 128, 192, 256, 384, 512, 1024)
    print(f"  Allocating persistent Talker KV caches + decode traces for buckets {TRACE_DECODE_BUCKETS}...")
    trace_decode_embed_tt = ttnn.from_torch(
        torch.zeros(1, 1, 1, talker_h, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    trace_decode_cos_tt = ttnn.from_torch(
        torch.ones(1, 1, 1, head_dim, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    trace_decode_sin_tt = ttnn.from_torch(
        torch.zeros(1, 1, 1, head_dim, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    trace_decode_cur_pos_tt = ttnn.from_torch(
        torch.tensor([1], dtype=torch.int32),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    talker_kv_caches_by_bucket = {}
    talker_kv_zero_hosts_by_bucket = {}
    talker_decode_trace_id_by_bucket = {}
    trace_decode_mask_tt_by_bucket = {}
    trace_decode_codec_logits_out_by_bucket = {}
    trace_decode_hidden_out_by_bucket = {}
    trace_decode_codec0_token_tt_by_bucket: Dict[int, object] = {}

    # Optional on-device codec0 sampling (TT_QWEN3_DEVICE_SAMPLING=1): bake
    # topk + ttnn.sampling into each bucket's Talker decode trace so the
    # codec0 token comes back as a small int D2H instead of a full vocab D2H
    # blocking on Talker exec. Eliminates the per-frame codec0_d2h wait.
    _device_sampling = bool(int(os.environ.get("TT_QWEN3_DEVICE_SAMPLING", "0")))
    talker_sampler = (
        _DeviceSampler(device, top_k=config.top_k, top_p=1.0, temperature=config.temperature)
        if _device_sampling and not config.greedy
        else None
    )

    for _bucket in TRACE_DECODE_BUCKETS:
        _max_talker_seq_len = (((_bucket + config.max_new_tokens + 16) + _TILE - 1) // _TILE) * _TILE
        # Allocate KV cache for this bucket
        _kv = allocate_kv_cache(
            device=device,
            num_layers=model.talker_config.num_hidden_layers,
            batch_size=1,
            num_kv_heads=model.talker_config.num_key_value_heads,
            max_seq_len=_max_talker_seq_len,
            head_dim=head_dim,
        )
        # Zero-fill host buffers for cheap per-request reset (mirrors cp_kv_zero_hosts pattern)
        _kv_zero_hosts = []
        for _kc, _vc in _kv:
            _zk = ttnn.from_torch(
                torch.zeros(*_kc.shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            _zv = ttnn.from_torch(
                torch.zeros(*_vc.shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            _kv_zero_hosts.append((_zk, _zv))
        # Per-bucket decode mask (sized to the bucket's max_talker_seq_len)
        _mask = ttnn.from_torch(
            torch.full((1, _talker_num_heads, 1, _max_talker_seq_len), float("-inf")),
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Per-bucket codec0 token output buffer for on-device sampling
        # (only allocated when the optional device-sampling path is on).
        _codec0_token_tt = talker_sampler.alloc_token_buf() if talker_sampler is not None else None

        # Untraced warmup so program cache is hot before begin_trace_capture
        # (writes are not allowed during trace capture).
        _wu_h, _ = model.talker.forward_from_hidden(
            trace_decode_embed_tt,
            trace_decode_cos_tt,
            trace_decode_sin_tt,
            talker_trans_mat,
            kv_caches=_kv,
            cur_pos_tensor=trace_decode_cur_pos_tt,
            decode_attn_mask=_mask,
            mode="decode",
        )
        _wu_logits = model.talker.get_codec_logits(_wu_h)
        if talker_sampler is not None:
            talker_sampler.append_sampling(_wu_logits, _codec0_token_tt)
        ttnn.synchronize_device(device)

        # Capture trace.
        _trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        try:
            _hidden, _ = model.talker.forward_from_hidden(
                trace_decode_embed_tt,
                trace_decode_cos_tt,
                trace_decode_sin_tt,
                talker_trans_mat,
                kv_caches=_kv,
                cur_pos_tensor=trace_decode_cur_pos_tt,
                decode_attn_mask=_mask,
                mode="decode",
            )
            _logits_out = model.talker.get_codec_logits(_hidden)
            if talker_sampler is not None:
                talker_sampler.append_sampling(_logits_out, _codec0_token_tt)
        finally:
            ttnn.end_trace_capture(device, _trace_id, cq_id=0)
        ttnn.synchronize_device(device)

        # Zero-reset KV cache for first real request that hits this bucket
        for (_zk, _zv), (_kc, _vc) in zip(_kv_zero_hosts, _kv):
            ttnn.copy_host_to_device_tensor(_zk, _kc)
            ttnn.copy_host_to_device_tensor(_zv, _vc)

        talker_kv_caches_by_bucket[_bucket] = _kv
        talker_kv_zero_hosts_by_bucket[_bucket] = _kv_zero_hosts
        talker_decode_trace_id_by_bucket[_bucket] = _trace_id
        trace_decode_mask_tt_by_bucket[_bucket] = _mask
        trace_decode_codec_logits_out_by_bucket[_bucket] = _logits_out
        trace_decode_hidden_out_by_bucket[_bucket] = _hidden
        if _codec0_token_tt is not None:
            trace_decode_codec0_token_tt_by_bucket[_bucket] = _codec0_token_tt
        print(
            f"    bucket={_bucket}: max_talker_seq={_max_talker_seq_len}, trace captured"
            f"{' (with on-device codec0 sampling)' if _codec0_token_tt is not None else ''}"
        )

    # Capture SE-block traces for ECAPA. Without these, on-device
    # ``extract_speaker_embedding`` calls after run_inference's trace exec
    # produce inf values (or hang). With these, the SE forward replays a
    # pre-captured trace at request time and is safe.
    print("Server context initialized. ALL traces pre-captured (ECAPA SE + CP + Talker decode).")

    return TTSServerContext(
        cp_kv_caches_persistent=cp_kv_caches_persistent,
        cp_kv_zero_hosts=cp_kv_zero_hosts,
        cp_prefill_trace_id=cp_prefill_trace_id,
        cp_decode_trace_ids=cp_decode_trace_ids,
        cp_prefill_logits_tt=cp_prefill_logits_tt,
        cp_decode_logits_tts=cp_decode_logits_tts,
        cp_trace_prefill_embed_tt=cp_trace_prefill_embed_tt,
        cp_trace_prefill_cos_tt=cp_trace_prefill_cos_tt,
        cp_trace_prefill_sin_tt=cp_trace_prefill_sin_tt,
        cp_trace_prefill_mask_tt=cp_trace_prefill_mask_tt,
        cp_trace_prefill_cos_host=cp_trace_prefill_cos_host,
        cp_trace_prefill_sin_host=cp_trace_prefill_sin_host,
        cp_trace_prefill_mask_host=cp_trace_prefill_mask_host,
        cp_trace_decode_embed_tts=cp_trace_decode_embed_tts,
        cp_trace_decode_cos_tts=cp_trace_decode_cos_tts,
        cp_trace_decode_sin_tts=cp_trace_decode_sin_tts,
        cp_trace_decode_mask_tts=cp_trace_decode_mask_tts,
        talker_kv_caches_by_bucket=talker_kv_caches_by_bucket,
        talker_kv_zero_hosts_by_bucket=talker_kv_zero_hosts_by_bucket,
        talker_decode_trace_id_by_bucket=talker_decode_trace_id_by_bucket,
        trace_decode_embed_tt=trace_decode_embed_tt,
        trace_decode_cos_tt=trace_decode_cos_tt,
        trace_decode_sin_tt=trace_decode_sin_tt,
        trace_decode_cur_pos_tt=trace_decode_cur_pos_tt,
        trace_decode_mask_tt_by_bucket=trace_decode_mask_tt_by_bucket,
        trace_decode_codec_logits_out_by_bucket=trace_decode_codec_logits_out_by_bucket,
        trace_decode_hidden_out_by_bucket=trace_decode_hidden_out_by_bucket,
        trace_decode_codec0_token_tt_by_bucket=trace_decode_codec0_token_tt_by_bucket,
        talker_trans_mat=talker_trans_mat,
        cp_trans_mat=cp_trans_mat,
        talker_cos_table=talker_cos_table,
        talker_sin_table=talker_sin_table,
        cp_cos_table=cp_cos_table,
        cp_sin_table=cp_sin_table,
        code_pred_embeds=code_pred_embeds,
        codec_embed_torch=codec_embed_torch,
        max_cp_seq_len=max_cp_seq_len,
        _talker_num_heads=_talker_num_heads,
        _cp_num_heads=_cp_num_heads,
    )


def run_inference(
    ctx: TTSServerContext,
    model,
    device,
    inputs_embeds_tt: "ttnn.Tensor",
    trailing_text_hidden: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    config: TTSConfig,
    use_2cq: bool = False,
) -> tuple:
    """
    Run TTS inference using server context.

    Per-request cost:
    - Zero kernel compile (pre-done at startup)
    - Zero CP trace capture (pre-captured in ctx)
    - One Talker KV alloc + standard prefill + Talker trace capture per request
    - Fast decode loop (all traces executed)

    use_2cq:
        If True, device must be opened with num_command_queues=2; overlaps H2D (CQ1) with trace exec (CQ0).

    Returns:
        (codes, timings, perf_text)
        codes: torch.Tensor [num_frames, 16]
        timings: dict with timing breakdowns
        perf_text: formatted string for display
    """
    from models.demos.qwen3_tts.tt.rope import get_rope_tensors

    _TILE = 32
    real_seq_len = inputs_embeds_tt.shape[2]
    padded_seq_len = get_padded_prefill_len(real_seq_len)
    max_talker_seq_len = (((padded_seq_len + config.max_new_tokens + 16) + _TILE - 1) // _TILE) * _TILE

    talker_h = model.talker_config.hidden_size
    head_dim = model.talker_config.head_dim

    # Pad input to bucket size
    if padded_seq_len > real_seq_len:
        pad_len = padded_seq_len - real_seq_len
        pad_zeros = ttnn.from_torch(
            torch.zeros(1, 1, pad_len, talker_h, dtype=torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        inputs_embeds_tt = ttnn.concat([inputs_embeds_tt, pad_zeros], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(pad_zeros)

    # Look up persistent Talker KV cache + decode trace for this bucket
    # (allocated once at init_server_context time; zero-reset per request below).
    if padded_seq_len not in ctx.talker_kv_caches_by_bucket:
        raise RuntimeError(
            f"No pre-allocated Talker state for bucket={padded_seq_len}. "
            f"Available: {sorted(ctx.talker_kv_caches_by_bucket.keys())}"
        )
    talker_kv_caches = ctx.talker_kv_caches_by_bucket[padded_seq_len]
    talker_kv_zero_hosts = ctx.talker_kv_zero_hosts_by_bucket[padded_seq_len]
    talker_decode_trace_id = ctx.talker_decode_trace_id_by_bucket[padded_seq_len]
    trace_embed_tt = ctx.trace_decode_embed_tt
    trace_cos_tt = ctx.trace_decode_cos_tt
    trace_sin_tt = ctx.trace_decode_sin_tt
    trace_cur_pos_tt = ctx.trace_decode_cur_pos_tt
    trace_mask_tt = ctx.trace_decode_mask_tt_by_bucket[padded_seq_len]
    trace_codec_logits_out = ctx.trace_decode_codec_logits_out_by_bucket[padded_seq_len]
    trace_hidden_out = ctx.trace_decode_hidden_out_by_bucket[padded_seq_len]

    # Zero-reset Talker KV cache so prior request's prefill positions don't leak.
    for (_zk, _zv), (_kc, _vc) in zip(talker_kv_zero_hosts, talker_kv_caches):
        ttnn.copy_host_to_device_tensor(_zk, _kc)
        ttnn.copy_host_to_device_tensor(_zv, _vc)

    all_codes = []
    timings = {}

    try:
        # === Talker prefill (standard path — avoids L1 overflow on large buckets) ===
        prefill_pos = torch.arange(padded_seq_len)
        prefill_cos_tt, prefill_sin_tt = get_rope_tensors(
            device, head_dim, padded_seq_len, prefill_pos, model.talker_config.rope_theta
        )
        ttnn.synchronize_device(device)
        t_prefill_start = time.time()

        prefill_hidden_out, talker_kv_caches = model.talker.forward_from_hidden(
            inputs_embeds_tt,
            prefill_cos_tt,
            prefill_sin_tt,
            ctx.talker_trans_mat,
            kv_caches=talker_kv_caches,
            start_pos=0,
            mode="prefill",
        )
        prefill_logits_out = model.talker.get_codec_logits(prefill_hidden_out)
        ttnn.synchronize_device(device)
        t_prefill_end = time.time()
        timings["prefill"] = t_prefill_end - t_prefill_start

        ttnn.deallocate(prefill_cos_tt)
        ttnn.deallocate(prefill_sin_tt)

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

        if token_0 == config.codec_eos_id:
            return None, timings, "EOS at prefill"

        talker_pos = real_seq_len
        talker_hidden_tt = prefill_hidden_out

        talker_cos_h2d, talker_sin_h2d, talker_mask_h2d, talker_cur_pos_h2d = build_talker_decode_trace_h2d_constants(
            ctx.talker_cos_table, ctx.talker_sin_table, ctx._talker_num_heads, max_talker_seq_len, real_seq_len
        )

        # === Generation loop ===
        decode_step_times = []
        talker_times_ms = []
        cp_times_ms = []
        trace_cq0_idle = ttnn.record_event(device, 0) if use_2cq else None
        h2d_cq = 1 if use_2cq else 0
        cp_decode_input_ready = [trace_cq0_idle, trace_cq0_idle]

        _th = model.talker_config.hidden_size
        token_id_buf = torch.zeros((1, 1), dtype=torch.long)
        cp_prefill_embed_cpu = torch.empty(1, 1, 2, _th, dtype=torch.bfloat16)
        talker_embed_cpu = torch.empty(1, 1, 1, _th, dtype=torch.bfloat16)
        cp_decode_embed_cpu = torch.empty(1, 1, 1, _th, dtype=torch.bfloat16)
        acc_code_embed = torch.zeros(1, 1, _th, dtype=torch.float32)

        t_decode_start = time.time()
        # Run the AR generation loop via the shared helper in tt/utils.py
        # so this server path runs the EXACT same loop body as the demo's
        # generate_codes_ttnn (no per-call drift between paths).
        from models.demos.qwen3_tts.tt.utils import DecodeLoopState, ar_decode_loop

        loop_state = DecodeLoopState(
            device=device,
            cp_kv_caches_persistent=ctx.cp_kv_caches_persistent,
            cp_kv_zero_hosts=ctx.cp_kv_zero_hosts,
            cp_prefill_trace_id=ctx.cp_prefill_trace_id,
            cp_prefill_logits_tt=ctx.cp_prefill_logits_tt,
            cp_decode_trace_ids=ctx.cp_decode_trace_ids,
            cp_decode_logits_tts=ctx.cp_decode_logits_tts,
            cp_trace_prefill_embed_tt=ctx.cp_trace_prefill_embed_tt,
            cp_trace_prefill_mask_tt=ctx.cp_trace_prefill_mask_tt,
            cp_trace_prefill_cos_tt=ctx.cp_trace_prefill_cos_tt,
            cp_trace_prefill_sin_tt=ctx.cp_trace_prefill_sin_tt,
            cp_trace_prefill_mask_host=ctx.cp_trace_prefill_mask_host,
            cp_trace_prefill_cos_host=ctx.cp_trace_prefill_cos_host,
            cp_trace_prefill_sin_host=ctx.cp_trace_prefill_sin_host,
            cp_trace_decode_embed_tts=ctx.cp_trace_decode_embed_tts,
            code_pred_embeds=ctx.code_pred_embeds,
            codec_embed_torch=ctx.codec_embed_torch,
            talker_decode_trace_id=talker_decode_trace_id,
            trace_embed_tt=trace_embed_tt,
            trace_cos_tt=trace_cos_tt,
            trace_sin_tt=trace_sin_tt,
            trace_cur_pos_tt=trace_cur_pos_tt,
            trace_mask_tt=trace_mask_tt,
            trace_hidden_out=trace_hidden_out,
            trace_codec_logits_out=trace_codec_logits_out,
            talker_codec0_token_tt=ctx.trace_decode_codec0_token_tt_by_bucket.get(padded_seq_len),
            talker_cos_h2d=talker_cos_h2d,
            talker_sin_h2d=talker_sin_h2d,
            talker_mask_h2d=talker_mask_h2d,
            talker_cur_pos_h2d=talker_cur_pos_h2d,
            token_id_buf=token_id_buf,
            cp_prefill_embed_cpu=cp_prefill_embed_cpu,
            cp_decode_embed_cpu=cp_decode_embed_cpu,
            talker_embed_cpu=talker_embed_cpu,
            acc_code_embed=acc_code_embed,
            talker_hidden_tt=talker_hidden_tt,
            talker_pos=talker_pos,
            real_seq_len=real_seq_len,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            token_0=token_0,
        )
        codes_tensor, _frame_breakdown, _t_first, _t_last = ar_decode_loop(
            loop_state,
            config,
            use_2cq,
            sample_token_fn=sample_token,
            sample_from_tt_vocab_logits_fn=sample_from_tt_vocab_logits,
        )
        decode_step_times = loop_state.decode_step_times_ms
        talker_times_ms = loop_state.talker_times_ms
        cp_times_ms = loop_state.cp_times_ms
        all_codes = [] if codes_tensor is None else codes_tensor.tolist()
        t_decode_end = time.time()
        timings["decode_loop"] = t_decode_end - t_decode_start

    finally:
        ttnn.synchronize_device(device)
        # Persistent Talker state (KV caches, decode trace, mask, embed/cos/sin/cur_pos
        # buffers) is owned by ctx and is NOT released here. It survives across requests;
        # KV caches are zero-reset at the start of run_inference.

    if len(all_codes) == 0:
        return None, timings, "No frames generated"

    codes = torch.tensor(all_codes, dtype=torch.long)
    num_frames = len(codes)

    # Build performance text
    prefill_ms = timings["prefill"] * 1000
    decode_ms = timings.get("decode_loop", 0) * 1000
    avg_step = sum(decode_step_times) / len(decode_step_times) if decode_step_times else 0
    avg_talker = sum(talker_times_ms) / len(talker_times_ms) if talker_times_ms else 0
    avg_cp = sum(cp_times_ms) / len(cp_times_ms) if cp_times_ms else 0
    inference_ms = prefill_ms + decode_ms
    audio_duration = num_frames / 12.0

    lines = [
        f"{'Phase':<35} {'Time (ms)':>10}",
        "-" * 48,
        f"{'Prefill (' + str(real_seq_len) + ' / ' + str(padded_seq_len) + ' tokens)':<35} {prefill_ms:>10.1f}",
        f"{'Decode loop (' + str(num_frames) + ' frames)':<35} {decode_ms:>10.1f}",
        f"{'  Avg step / Talker / CP (ms)':<35} {'%.1f / %.1f / %.1f' % (avg_step, avg_talker, avg_cp):>10}",
        "-" * 48,
        f"{'Inference time':<35} {inference_ms:>10.1f}",
        f"{'Audio duration':<35} {audio_duration:>10.2f}s",
    ]
    perf_text = "\n".join(lines)
    print("\n" + perf_text)

    timings["inference"] = timings["prefill"] + timings.get("decode_loop", 0)
    timings["num_frames"] = num_frames
    timings["audio_duration"] = audio_duration

    return codes, timings, perf_text


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
