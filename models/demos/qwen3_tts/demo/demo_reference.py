# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS PyTorch Reference Demo Script

This script generates audio using pure PyTorch reference implementations.
Use this to verify the reference implementation produces good quality audio
before debugging TTNN.

Usage:
    python models/demos/qwen3_tts/demo/demo_reference.py \
        --text "Hello, this is a test." \
        --audio-output output_reference.wav
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from models.demos.qwen3_tts.reference.functional import (
    Qwen3TTSCodePredictorConfig,
    Qwen3TTSConfig,
    SpeechTokenizerDecoderConfig,
    attention,
    codebook_lookup_rvq,
    compute_mrope_frequencies,
    compute_rope_frequencies,
    conv_decoder_block,
    convnext_block,
    extract_code_predictor_weights,
    extract_speech_tokenizer_decoder_weights,
    extract_talker_weights,
    pre_transformer_forward,
    rms_norm,
    snake_activation,
    swiglu_mlp,
)


def speech_tokenizer_decoder_forward_debug(
    token_ids: torch.Tensor,
    weights: dict,
    config: SpeechTokenizerDecoderConfig,
) -> torch.Tensor:
    """Debug version of speech tokenizer decoder with verbose output."""
    batch_size, num_quantizers, seq_len = token_ids.shape
    device = token_ids.device

    print(f"  [DEBUG] Input token_ids: {token_ids.shape}")

    # 1. Codebook lookup with proper RVQ processing
    rvq_first_codebook = weights.get("quantizer.rvq_first.vq.layers.0._codebook.embedding_sum")
    rvq_first_output_proj = weights.get("quantizer.rvq_first.output_proj.weight")
    rvq_rest_codebooks = []
    for i in range(num_quantizers - 1):
        key = f"quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"
        if key in weights:
            rvq_rest_codebooks.append(weights[key])
    rvq_rest_output_proj = weights.get("quantizer.rvq_rest.output_proj.weight")

    print(f"  [DEBUG] rvq_first_codebook: {rvq_first_codebook.shape if rvq_first_codebook is not None else None}")
    print(f"  [DEBUG] rvq_rest_codebooks: {len(rvq_rest_codebooks)}")
    print(
        f"  [DEBUG] rvq_first_output_proj: {rvq_first_output_proj.shape if rvq_first_output_proj is not None else None}"
    )
    print(f"  [DEBUG] rvq_rest_output_proj: {rvq_rest_output_proj.shape if rvq_rest_output_proj is not None else None}")

    if rvq_first_output_proj is not None and rvq_rest_output_proj is not None:
        embeddings = codebook_lookup_rvq(
            token_ids,
            rvq_first_codebook,
            rvq_rest_codebooks,
            rvq_first_output_proj,
            rvq_rest_output_proj,
        )
    else:
        codebooks = []
        if rvq_first_codebook is not None:
            codebooks.append(rvq_first_codebook)
        codebooks.extend(rvq_rest_codebooks)
        embeddings = None
        for i, codebook in enumerate(codebooks):
            if i >= num_quantizers:
                break
            ids = token_ids[:, i, :]
            emb = F.embedding(ids, codebook)
            embeddings = emb if embeddings is None else embeddings + emb

    print(f"  [DEBUG] After codebook lookup: {embeddings.shape}")

    # 2. Pre-transformer
    pre_transformer_weights = {
        k.replace("pre_transformer.", ""): v for k, v in weights.items() if k.startswith("pre_transformer.")
    }
    print(f"  [DEBUG] pre_transformer_weights: {len(pre_transformer_weights)} keys")

    if pre_transformer_weights:
        hidden_states = pre_transformer_forward(embeddings, pre_transformer_weights, config, skip_output_proj=True)
    else:
        hidden_states = embeddings

    print(f"  [DEBUG] After pre_transformer: {hidden_states.shape}")

    # 3. Pre-conv
    if "pre_conv.conv.weight" in weights:
        print(f"  [DEBUG] pre_conv weight: {weights['pre_conv.conv.weight'].shape}")
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.conv1d(
            hidden_states,
            weights["pre_conv.conv.weight"],
            weights.get("pre_conv.conv.bias"),
            padding=weights["pre_conv.conv.weight"].shape[-1] // 2,
        )
    else:
        hidden_states = hidden_states.transpose(1, 2)

    print(f"  [DEBUG] After pre_conv: {hidden_states.shape}")

    # 4. Upsampler (ConvNeXt blocks)
    for i, ratio in enumerate(config.upsampling_ratios):
        upsample_prefix = f"upsample.{i}."
        conv_weight = weights.get(f"{upsample_prefix}0.conv.weight")

        if conv_weight is not None:
            print(f"  [DEBUG] upsample.{i} conv weight: {conv_weight.shape}, ratio: {ratio}")
            conv_bias = weights.get(f"{upsample_prefix}0.conv.bias")
            hidden_states = F.conv_transpose1d(hidden_states, conv_weight, conv_bias, stride=ratio)
            print(f"  [DEBUG] After upsample.{i} conv_transpose: {hidden_states.shape}")

            # ConvNeXt block
            convnext_weights = {
                k.replace(f"{upsample_prefix}1.", ""): v
                for k, v in weights.items()
                if k.startswith(f"{upsample_prefix}1.")
            }
            if convnext_weights and convnext_weights.get("dwconv.conv.weight") is not None:
                hidden_states = convnext_block(
                    hidden_states,
                    dwconv_weight=convnext_weights.get("dwconv.conv.weight"),
                    dwconv_bias=convnext_weights.get("dwconv.conv.bias"),
                    pwconv1_weight=convnext_weights.get("pwconv1.weight"),
                    pwconv1_bias=convnext_weights.get("pwconv1.bias"),
                    pwconv2_weight=convnext_weights.get("pwconv2.weight"),
                    pwconv2_bias=convnext_weights.get("pwconv2.bias"),
                    norm_weight=convnext_weights.get("norm.weight"),
                    norm_bias=convnext_weights.get("norm.bias"),
                    gamma=convnext_weights.get("gamma"),
                )
                print(f"  [DEBUG] After upsample.{i} convnext: {hidden_states.shape}")

    print(f"  [DEBUG] After all upsampling: {hidden_states.shape}")

    # 5. Conv decoder
    if "decoder.0.conv.weight" in weights:
        print(f"  [DEBUG] decoder.0 conv weight: {weights['decoder.0.conv.weight'].shape}")
        hidden_states = F.conv1d(
            hidden_states,
            weights["decoder.0.conv.weight"],
            weights.get("decoder.0.conv.bias"),
            padding=weights["decoder.0.conv.weight"].shape[-1] // 2,
        )
        print(f"  [DEBUG] After decoder.0: {hidden_states.shape}")

    # Decoder blocks
    for i, rate in enumerate(config.upsample_rates):
        block_prefix = f"decoder.{i + 1}."
        block_weights = {k.replace(block_prefix, ""): v for k, v in weights.items() if k.startswith(block_prefix)}
        if block_weights:
            print(f"  [DEBUG] decoder.{i + 1} rate: {rate}, block_weights: {len(block_weights)}")
            hidden_states = conv_decoder_block(hidden_states, block_weights, rate)
            print(f"  [DEBUG] After decoder.{i + 1}: {hidden_states.shape}")

    # Final activation + conv
    if "decoder.5.alpha" in weights:
        hidden_states = snake_activation(hidden_states, weights["decoder.5.alpha"], weights["decoder.5.beta"])
        print(f"  [DEBUG] After decoder.5 snake: {hidden_states.shape}")

    if "decoder.6.conv.weight" in weights:
        print(f"  [DEBUG] decoder.6 conv weight: {weights['decoder.6.conv.weight'].shape}")
        hidden_states = F.conv1d(
            hidden_states,
            weights["decoder.6.conv.weight"],
            weights.get("decoder.6.conv.bias"),
            padding=weights["decoder.6.conv.weight"].shape[-1] // 2,
        )
        print(f"  [DEBUG] After decoder.6: {hidden_states.shape}")

    audio = torch.tanh(hidden_states)
    return audio


def load_hf_weights(model_id: str, cache_dir: Optional[str] = None) -> dict:
    """Load weights from HuggingFace model."""
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("Please install safetensors: pip install safetensors")

    from huggingface_hub import snapshot_download

    print(f"Loading model from HuggingFace: {model_id}")
    start_time = time.time()

    model_path = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json"],
    )
    model_path = Path(model_path)

    state_dict = {}
    for safetensor_file in model_path.glob("*.safetensors"):
        print(f"  Loading {safetensor_file.name}...")
        state_dict.update(load_file(safetensor_file))

    load_time = time.time() - start_time
    print(f"Loaded {len(state_dict)} weight tensors in {load_time:.2f}s")

    return state_dict


def load_speech_tokenizer_weights(model_id: str, cache_dir: Optional[str] = None) -> dict:
    """Load Speech Tokenizer Decoder weights from HuggingFace model."""
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("Please install safetensors: pip install safetensors")

    from huggingface_hub import snapshot_download

    print(f"Loading speech tokenizer weights from: {model_id}")

    model_path = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        allow_patterns=["speech_tokenizer/*.safetensors"],
    )
    model_path = Path(model_path)

    speech_tokenizer_path = model_path / "speech_tokenizer" / "model.safetensors"
    if not speech_tokenizer_path.exists():
        print(f"  Warning: Speech tokenizer weights not found at {speech_tokenizer_path}")
        return {}

    print(f"  Loading {speech_tokenizer_path.name}...")
    state_dict = load_file(speech_tokenizer_path)

    # Extract decoder weights (remove "decoder." prefix)
    decoder_weights = extract_speech_tokenizer_decoder_weights(state_dict)
    print(f"Loaded {len(decoder_weights)} speech tokenizer decoder weight tensors")

    return decoder_weights


def save_audio(audio: torch.Tensor, output_path: str, sample_rate: int = 24000):
    """Save audio tensor to WAV file."""
    try:
        import scipy.io.wavfile as wavfile
    except ImportError:
        raise ImportError("Please install scipy: pip install scipy")

    audio = audio.squeeze()
    if audio.dim() > 1:
        audio = audio[0]

    audio = audio.detach().cpu().float().numpy()
    audio = (audio * 32767).astype("int16")

    wavfile.write(output_path, sample_rate, audio)
    print(f"Audio saved to: {output_path}")


def talker_forward_reference(
    input_ids: torch.Tensor,
    weights: dict,
    config: Qwen3TTSConfig,
    use_text_embedding: bool = False,
) -> torch.Tensor:
    """
    Forward pass through the Qwen3-TTS Talker model using reference implementation.

    Supports both text and codec embedding modes.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Token embeddings
    if use_text_embedding and "text_embedding.weight" in weights:
        hidden_states = F.embedding(input_ids, weights["text_embedding.weight"])
        print(f"  Using text embedding, hidden shape: {hidden_states.shape}")
    else:
        hidden_states = F.embedding(input_ids, weights["codec_embedding.weight"])
        print(f"  Using codec embedding, hidden shape: {hidden_states.shape}")

    # Compute RoPE frequencies for MROPE
    cos, sin = compute_mrope_frequencies(config.head_dim, seq_len, config.rope_theta, device)
    cos = cos.to(hidden_states.dtype)
    sin = sin.to(hidden_states.dtype)

    # Create causal attention mask
    attention_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=hidden_states.dtype),
        diagonal=1,
    )
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

    # Process through decoder layers
    for layer_idx in range(config.num_hidden_layers):
        layer_prefix = f"layers.{layer_idx}."
        layer_weights = {k.replace(layer_prefix, ""): v for k, v in weights.items() if k.startswith(layer_prefix)}

        # Decoder layer forward
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, layer_weights["input_layernorm.weight"], config.rms_norm_eps)

        hidden_states = attention(
            hidden_states,
            q_proj_weight=layer_weights["self_attn.q_proj.weight"],
            k_proj_weight=layer_weights["self_attn.k_proj.weight"],
            v_proj_weight=layer_weights["self_attn.v_proj.weight"],
            o_proj_weight=layer_weights["self_attn.o_proj.weight"],
            q_norm_weight=layer_weights["self_attn.q_norm.weight"],
            k_norm_weight=layer_weights["self_attn.k_norm.weight"],
            cos=cos,
            sin=sin,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_mask=attention_mask,
            use_mrope=True,
            mrope_section=config.mrope_section,
            mrope_interleaved=config.mrope_interleaved,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = rms_norm(hidden_states, layer_weights["post_attention_layernorm.weight"], config.rms_norm_eps)

        hidden_states = swiglu_mlp(
            hidden_states,
            gate_proj_weight=layer_weights["mlp.gate_proj.weight"],
            up_proj_weight=layer_weights["mlp.up_proj.weight"],
            down_proj_weight=layer_weights["mlp.down_proj.weight"],
        )

        hidden_states = residual + hidden_states

    # Final norm
    hidden_states = rms_norm(hidden_states, weights["norm.weight"], config.rms_norm_eps)

    return hidden_states


def code_predictor_forward_reference(
    inputs_embeds: torch.Tensor,
    weights: dict,
    config: Qwen3TTSCodePredictorConfig,
) -> tuple:
    """
    Forward pass through Code Predictor using reference implementation.

    Returns:
        tuple: (hidden_states, logits_list)
    """
    batch_size, seq_len, _ = inputs_embeds.shape
    device = inputs_embeds.device

    hidden_states = inputs_embeds

    # Compute standard RoPE frequencies
    cos, sin = compute_rope_frequencies(config.head_dim, seq_len, config.rope_theta, device)
    cos = cos.to(hidden_states.dtype)
    sin = sin.to(hidden_states.dtype)

    # Create causal attention mask
    attention_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=hidden_states.dtype),
        diagonal=1,
    )
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

    # Process through decoder layers
    for layer_idx in range(config.num_hidden_layers):
        layer_prefix = f"model.layers.{layer_idx}."
        layer_weights = {k.replace(layer_prefix, ""): v for k, v in weights.items() if k.startswith(layer_prefix)}

        residual = hidden_states
        hidden_states = rms_norm(hidden_states, layer_weights["input_layernorm.weight"], config.rms_norm_eps)

        hidden_states = attention(
            hidden_states,
            q_proj_weight=layer_weights["self_attn.q_proj.weight"],
            k_proj_weight=layer_weights["self_attn.k_proj.weight"],
            v_proj_weight=layer_weights["self_attn.v_proj.weight"],
            o_proj_weight=layer_weights["self_attn.o_proj.weight"],
            q_norm_weight=layer_weights["self_attn.q_norm.weight"],
            k_norm_weight=layer_weights["self_attn.k_norm.weight"],
            cos=cos,
            sin=sin,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_mask=attention_mask,
            use_mrope=False,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = rms_norm(hidden_states, layer_weights["post_attention_layernorm.weight"], config.rms_norm_eps)

        hidden_states = swiglu_mlp(
            hidden_states,
            gate_proj_weight=layer_weights["mlp.gate_proj.weight"],
            up_proj_weight=layer_weights["mlp.up_proj.weight"],
            down_proj_weight=layer_weights["mlp.down_proj.weight"],
        )

        hidden_states = residual + hidden_states

    # Final norm
    hidden_states = rms_norm(hidden_states, weights["model.norm.weight"], config.rms_norm_eps)

    # Apply LM heads to get logits for each code group
    # Code predictor has 15 LM heads (for code groups 1-15, group 0 uses codec_embedding)
    num_lm_heads = config.num_code_groups - 1  # 15
    logits_list = []

    for g in range(num_lm_heads):
        lm_head_key = f"lm_head.{g}.weight"
        if lm_head_key in weights:
            logits = F.linear(hidden_states, weights[lm_head_key])
            logits_list.append(logits)

    return hidden_states, logits_list


def run_reference_demo(
    model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    cache_dir: Optional[str] = None,
    text: Optional[str] = None,
    audio_output: str = "output_reference.wav",
    seq_len: int = 128,
):
    """Run Qwen3-TTS reference demo with pure PyTorch."""
    print("=" * 80)
    print("Qwen3-TTS PyTorch Reference Demo")
    print("=" * 80)

    # Load weights
    state_dict = load_hf_weights(model_id, cache_dir)
    speech_tokenizer_weights = load_speech_tokenizer_weights(model_id, cache_dir)

    # Initialize configs
    talker_config = Qwen3TTSConfig()
    cp_config = Qwen3TTSCodePredictorConfig()
    speech_config = SpeechTokenizerDecoderConfig()

    # Extract model weights
    print("\nExtracting model weights...")
    talker_weights = extract_talker_weights(state_dict)
    cp_weights = extract_code_predictor_weights(state_dict)
    print(f"  Talker weights: {len(talker_weights)}")
    print(f"  Code Predictor weights: {len(cp_weights)}")

    # Create input based on mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_text_embedding = False

    if text is not None:
        # Real TTS mode - tokenize text input
        print(f"\n*** REAL TTS MODE (Reference) ***")
        print(f"Text: {text}")
        from transformers import AutoProcessor

        print("Loading tokenizer...")
        processor = AutoProcessor.from_pretrained(model_id)

        # Format text with special tokens (Qwen3-TTS format)
        formatted_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        inputs = processor(text=formatted_text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        seq_len = input_ids.shape[1]
        use_text_embedding = True
        print(f"Tokenized to {seq_len} tokens")
    else:
        # Benchmark mode - use random codec tokens
        print(f"\nBenchmark mode - using random codec tokens (seq_len={seq_len})")
        torch.manual_seed(42)
        input_ids = torch.randint(0, talker_config.audio_vocab_size, (1, seq_len), device=device)

    # Move weights to device
    print(f"\nMoving weights to device: {device}")
    talker_weights = {k: v.to(device) for k, v in talker_weights.items()}
    cp_weights = {k: v.to(device) for k, v in cp_weights.items()}

    # Run Talker forward pass
    print("\n" + "=" * 80)
    print("TALKER FORWARD PASS (Reference)")
    print("=" * 80)

    start_time = time.time()
    talker_hidden_states = talker_forward_reference(
        input_ids, talker_weights, talker_config, use_text_embedding=use_text_embedding
    )
    talker_time = time.time() - start_time
    print(f"  Talker output shape: {talker_hidden_states.shape}")
    print(f"  Talker time: {talker_time*1000:.2f} ms")

    # Project talker output to code predictor input dimension
    # Talker output: 2048, Code Predictor input: 1024
    print("\n  Projecting talker output to code predictor input...")

    # Use the connector weights if available
    connector_key = "talker.code_predictor.connector.weight"
    if connector_key in state_dict:
        connector_weight = state_dict[connector_key].to(device)
        cp_input = F.linear(talker_hidden_states, connector_weight)
        print(f"  Using connector projection: {talker_hidden_states.shape[-1]} -> {cp_input.shape[-1]}")
    else:
        # Fallback: simple linear projection
        cp_input = talker_hidden_states[:, :, : cp_config.hidden_size]
        print(f"  Using slice projection: {talker_hidden_states.shape[-1]} -> {cp_input.shape[-1]}")

    # Run Code Predictor forward pass
    print("\n" + "=" * 80)
    print("CODE PREDICTOR FORWARD PASS (Reference)")
    print("=" * 80)

    start_time = time.time()
    cp_hidden_states, logits_list = code_predictor_forward_reference(cp_input, cp_weights, cp_config)
    cp_time = time.time() - start_time
    print(f"  Code Predictor output shape: {cp_hidden_states.shape}")
    print(f"  Number of LM head outputs: {len(logits_list)}")
    print(f"  Code Predictor time: {cp_time*1000:.2f} ms")

    # Convert logits to token IDs
    print("\n" + "=" * 80)
    print("GENERATING CODEC TOKENS")
    print("=" * 80)

    token_ids_list = []

    # First code group uses codec_head from talker
    # talker.codec_head.weight: [3072, 2048] maps talker hidden states to codec vocab
    codec_head_key = "talker.codec_head.weight"
    if codec_head_key in state_dict:
        codec_head_weight = state_dict[codec_head_key].to(device)
        first_logits = F.linear(talker_hidden_states, codec_head_weight)
        first_token_ids = torch.argmax(first_logits, dim=-1)
        print(f"  Codec head logits shape: {first_logits.shape}")
        print(f"  First code group tokens shape: {first_token_ids.shape}")
    else:
        print("  WARNING: codec_head not found, using zeros for first code group")
        first_token_ids = torch.zeros((1, seq_len), dtype=torch.long, device=device)
    token_ids_list.append(first_token_ids)

    for i, logits in enumerate(logits_list):
        print(f"  LM head {i} logits shape: {logits.shape}")
        token_ids = torch.argmax(logits, dim=-1)
        token_ids_list.append(token_ids)

    # Stack tokens: [batch, num_quantizers, seq_len]
    num_quantizers = speech_config.num_quantizers
    while len(token_ids_list) < num_quantizers:
        token_ids_list.append(torch.zeros_like(token_ids_list[0]))

    token_ids = torch.stack(token_ids_list[:num_quantizers], dim=1)
    print(f"  Final token IDs shape: {token_ids.shape}")

    # Clamp token IDs to valid range for speech tokenizer codebooks
    # RVQ codebook size is 2048, but model generates tokens with vocab 3072 (codec_head) or 2048 (lm_heads)
    rvq_codebook_size = 2048
    token_ids = torch.clamp(token_ids, 0, rvq_codebook_size - 1)
    print(f"  Token IDs clamped to range [0, {rvq_codebook_size - 1}]")

    # Generate audio
    print("\n" + "=" * 80)
    print("SPEECH TOKENIZER DECODER (Reference)")
    print("=" * 80)

    # Move speech tokenizer weights to device and convert to correct dtype
    speech_weights = {k: v.to(device).float() for k, v in speech_tokenizer_weights.items()}

    # Debug: print speech tokenizer decoder weights keys
    print(f"  Total speech weights: {len(speech_weights)}")

    # Add "decoder." prefix to numeric keys (like TTNN does)
    fixed_weights = {}
    for k, v in speech_weights.items():
        if k[0].isdigit():
            fixed_weights[f"decoder.{k}"] = v
        else:
            fixed_weights[k] = v
    print(f"  Fixed weights (with decoder. prefix): {len(fixed_weights)}")

    # Add verbose debug output for decoder flow
    start_time = time.time()
    audio = speech_tokenizer_decoder_forward_debug(token_ids, fixed_weights, speech_config)
    audio_time = time.time() - start_time

    print(f"  Audio shape: {audio.shape}")
    print(f"  Audio generation time: {audio_time*1000:.2f} ms")

    # Save audio
    sample_rate = speech_config.output_sample_rate
    duration_sec = audio.shape[-1] / sample_rate
    print(f"  Audio duration: {duration_sec:.2f} seconds ({sample_rate} Hz)")

    save_audio(audio, audio_output, sample_rate)

    print("\n" + "=" * 80)
    print("Reference demo completed!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS PyTorch Reference Demo")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to synthesize (enables real TTS mode)",
    )
    parser.add_argument(
        "--audio-output",
        type=str,
        default="output_reference.wav",
        help="Output path for generated audio",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length for benchmark mode",
    )

    args = parser.parse_args()

    run_reference_demo(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        text=args.text,
        audio_output=args.audio_output,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()
