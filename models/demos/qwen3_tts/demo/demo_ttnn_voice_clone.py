# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
TTNN Voice Clone Demo with Autoregressive Generation

This demo uses:
1. Official qwen_tts for input preparation (voice clone prompt creation)
2. TTNN model for autoregressive inference
3. Official qwen_tts for audio decoding

Usage:
    # Run in tt-metal environment
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    python models/demos/qwen3_tts/demo/demo_ttnn_voice_clone.py \
        --ref-audio /tmp/clone_ref.wav \
        --text "Hello, this is a test."
"""

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

import ttnn


def load_hf_weights(model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base") -> dict:
    """Load weights from HuggingFace."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print(f"Loading model weights from: {model_id}")
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    model_path = Path(model_path)

    state_dict = {}
    for f in model_path.glob("*.safetensors"):
        state_dict.update(load_file(f))

    print(f"  Loaded {len(state_dict)} weight tensors")
    return state_dict


def create_voice_clone_input(
    ref_code: torch.Tensor,
    text_tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Create input sequence for voice cloning.

    The input combines:
    - Reference audio's first RVQ codebook (codec tokens)
    - Text tokens

    Args:
        ref_code: RVQ codes from reference audio [seq_len, 16]
        text_tokens: Tokenized text [batch, text_len]

    Returns:
        Combined input tokens [batch, total_len]
    """
    # Take first codebook from ref_code (semantic level)
    # These are RVQ tokens (0-2047), used directly as codec tokens
    codec_tokens = ref_code[:, 0].unsqueeze(0)  # [1, ref_len]

    # Combine: codec_tokens + text_tokens
    # Note: In actual model, there may be special separator tokens
    input_ids = torch.cat([codec_tokens, text_tokens], dim=1)

    return input_ids


def compute_mixed_embeddings(
    codec_tokens: torch.Tensor,
    text_tokens: torch.Tensor,
    codec_embedding_weight: torch.Tensor,
    text_embedding_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Compute mixed embeddings for codec and text tokens.

    Args:
        codec_tokens: Codec token IDs [batch, codec_len]
        text_tokens: Text token IDs [batch, text_len]
        codec_embedding_weight: Codec embedding [3072, 2048]
        text_embedding_weight: Text embedding [151936, 2048]

    Returns:
        Combined embeddings [batch, codec_len + text_len, 2048]
    """
    # Embed codec tokens
    codec_embeds = F.embedding(codec_tokens, codec_embedding_weight)

    # Embed text tokens
    text_embeds = F.embedding(text_tokens, text_embedding_weight)

    # Concatenate
    combined = torch.cat([codec_embeds, text_embeds], dim=1)

    return combined


def run_ttnn_generation(
    device,
    model,
    codec_tokens: torch.Tensor,
    text_tokens: torch.Tensor,
    codec_embedding_weight: torch.Tensor,
    text_embedding_weight: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
) -> Tuple[List[int], List[List[int]]]:
    """
    Run autoregressive generation with TTNN model.

    Args:
        device: TTNN device
        model: TTNN Qwen3TTS model
        codec_tokens: Reference codec tokens [batch, codec_len]
        text_tokens: Text tokens [batch, text_len]
        codec_embedding_weight: Codec embedding weights
        text_embedding_weight: Text embedding weights
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter

    Returns:
        Tuple of (codec_tokens, code_predictor_tokens)
        - codec_tokens: List of generated first codebook tokens
        - code_predictor_tokens: List of [15] tokens for each position
    """
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSCodePredictorConfig, Qwen3TTSTalkerConfig
    from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat

    talker_config = Qwen3TTSTalkerConfig()
    cp_config = Qwen3TTSCodePredictorConfig()

    # Pre-compute transformation matrices
    talker_trans_mat = get_transformation_mat(talker_config.head_dim, device)
    cp_trans_mat = get_transformation_mat(cp_config.head_dim, device)

    batch_size = codec_tokens.shape[0]
    initial_codec_len = codec_tokens.shape[1]
    text_len = text_tokens.shape[1]
    total_len = initial_codec_len + text_len

    generated_codec_tokens = []
    generated_cp_tokens = []  # List of [15] tokens per position

    print(f"\nStarting generation (max {max_new_tokens} tokens)...")
    print(f"  Initial codec tokens: {initial_codec_len}")
    print(f"  Text tokens: {text_len}")
    print(f"  Total initial length: {total_len}")

    # Current state
    current_codec_tokens = codec_tokens.clone()

    for step in range(max_new_tokens):
        current_codec_len = current_codec_tokens.shape[1]
        current_total_len = current_codec_len + text_len

        # Compute mixed embeddings on CPU
        hidden_states_cpu = compute_mixed_embeddings(
            current_codec_tokens,
            text_tokens,
            codec_embedding_weight,
            text_embedding_weight,
        )  # [batch, total_len, 2048]

        # Add extra dimension for TTNN: [batch, 1, seq_len, hidden]
        hidden_states_cpu = hidden_states_cpu.unsqueeze(1)

        # Convert to TTNN
        hidden_states = ttnn.from_torch(
            hidden_states_cpu,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Get RoPE tensors for current sequence length
        position_ids = torch.arange(current_total_len)
        talker_cos, talker_sin = get_rope_tensors(
            device, talker_config.head_dim, current_total_len, position_ids, talker_config.rope_theta
        )
        cp_cos, cp_sin = get_rope_tensors(
            device, cp_config.head_dim, current_total_len, position_ids, cp_config.rope_theta
        )

        # Run Talker forward from hidden states (skip embedding)
        hidden_states = model.talker.forward_from_hidden(
            hidden_states,
            talker_cos,
            talker_sin,
            talker_trans_mat,
            attention_mask=None,
        )

        # Get codec logits (first RVQ codebook prediction)
        codec_logits = model.talker.get_codec_logits(hidden_states)
        codec_logits_torch = ttnn.to_torch(codec_logits)  # [batch, 1, seq_len, 3072]

        # Take logits for last position
        last_codec_logits = codec_logits_torch[:, 0, -1, :]  # [batch, 3072]

        # Sample next codec token
        if temperature > 0:
            probs = F.softmax(last_codec_logits / temperature, dim=-1)
            # Top-k sampling
            if top_k > 0:
                top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
                sampled_idx = torch.multinomial(top_k_probs, num_samples=1)
                next_codec_token = top_k_indices.gather(-1, sampled_idx).squeeze(-1)
            else:
                next_codec_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_codec_token = torch.argmax(last_codec_logits, dim=-1)

        # Check for EOS (tokens >= 2048 might be special tokens)
        if next_codec_token.item() >= 2048:
            print(f"  Step {step}: EOS token {next_codec_token.item()}")
            break

        generated_codec_tokens.append(next_codec_token.item())

        # Run CodePredictor to get remaining codebooks
        cp_logits_list = model.code_predictor(
            hidden_states,
            cp_cos,
            cp_sin,
            cp_trans_mat,
            attention_mask=None,
        )

        # Sample from each LM head (15 codebooks)
        cp_tokens = []
        for logits in cp_logits_list:
            logits_torch = ttnn.to_torch(logits)[:, 0, -1, :]  # [batch, 2048]
            if temperature > 0:
                probs = F.softmax(logits_torch / temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                token = torch.argmax(logits_torch, dim=-1)
            cp_tokens.append(token.item())

        generated_cp_tokens.append(cp_tokens)

        # Append next codec token
        # next_codec_token is [batch] -> need [batch, 1]
        next_token_expanded = next_codec_token.view(batch_size, 1)
        current_codec_tokens = torch.cat([current_codec_tokens, next_token_expanded], dim=1)

        if step % 10 == 0:
            print(f"  Step {step}: token={next_codec_token.item()}, seq_len={current_input.shape[1]}")

        # Clean up TTNN tensors
        ttnn.deallocate(input_ttnn)
        ttnn.deallocate(hidden_states)
        ttnn.deallocate(codec_logits)
        for logits in cp_logits_list:
            ttnn.deallocate(logits)

    print(f"  Generated {len(generated_codec_tokens)} tokens")
    return generated_codec_tokens, generated_cp_tokens


def tokens_to_rvq_codes(
    codec_tokens: List[int],
    cp_tokens: List[List[int]],
) -> torch.Tensor:
    """
    Convert generated tokens to RVQ code format.

    Args:
        codec_tokens: First codebook tokens (from codec_head)
        cp_tokens: Remaining 15 codebook tokens (from code_predictor)

    Returns:
        RVQ codes [seq_len, 16]
    """
    seq_len = len(codec_tokens)
    rvq_codes = torch.zeros(seq_len, 16, dtype=torch.int64)

    for i in range(seq_len):
        # First codebook from codec_head (clamp to 2047 since vocab is 3072 but RVQ is 2048)
        rvq_codes[i, 0] = min(codec_tokens[i], 2047)

        # Remaining 15 codebooks from code_predictor
        for j, token in enumerate(cp_tokens[i]):
            rvq_codes[i, j + 1] = token

    return rvq_codes


def run_demo(
    ref_audio: str = "/tmp/clone_ref.wav",
    ref_text: str = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
    text: str = "Hello, this is a test of the text to speech system.",
    audio_output: str = "/tmp/ttnn_voice_clone.wav",
    device_id: int = 0,
    max_new_tokens: int = 50,
    use_official_decode: bool = True,
):
    """Run the TTNN voice clone demo."""
    print("=" * 80)
    print("TTNN Voice Clone Demo with Autoregressive Generation")
    print("=" * 80)

    # Check if we can load qwen_tts for input/output processing
    try:
        from qwen_tts import Qwen3TTSModel

        has_qwen_tts = True
        print("Using official qwen_tts for input preparation and audio decoding")
    except ImportError:
        has_qwen_tts = False
        print("WARNING: qwen_tts not available. Using pre-extracted tensors.")

    # Load HF weights
    state_dict = load_hf_weights()

    # Open TTNN device
    print(f"\nOpening TT device {device_id}...")
    device = ttnn.open_device(device_id=device_id)

    try:
        # Initialize TTNN model
        print("\nInitializing TTNN model...")
        from models.demos.qwen3_tts.tt.model_config import Qwen3TTSCodePredictorConfig, Qwen3TTSTalkerConfig
        from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS

        model_init_start = time.time()
        model = Qwen3TTS(
            device=device,
            state_dict=state_dict,
            talker_config=Qwen3TTSTalkerConfig(),
            code_predictor_config=Qwen3TTSCodePredictorConfig(),
        )
        model_init_time = time.time() - model_init_start
        print(f"Model initialized in {model_init_time:.2f}s")

        # Check codec_head is loaded
        if model.talker.codec_head is None:
            raise ValueError("codec_head not loaded!")
        print(f"  codec_head vocab size: {model.talker.codec_head_vocab_size}")

        # Prepare input
        if has_qwen_tts:
            print("\n" + "=" * 80)
            print("Creating Voice Clone Prompt (using qwen_tts)")
            print("=" * 80)

            official_model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                device_map="cpu",
                dtype=torch.float32,
            )

            prompt_items = official_model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
            prompt_item = prompt_items[0] if isinstance(prompt_items, list) else prompt_items

            ref_code = prompt_item.ref_code  # [ref_len, 16]
            print(f"  ref_code shape: {ref_code.shape}")

            # Tokenize text
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
            text_tokens = tokenizer(text, return_tensors="pt")["input_ids"]
            print(f"  text_tokens shape: {text_tokens.shape}")

            # Separate codec and text tokens
            codec_tokens = ref_code[:, 0].unsqueeze(0)  # First codebook only [1, ref_len]
            print(f"  codec_tokens shape: {codec_tokens.shape}")
            print(f"  text_tokens shape: {text_tokens.shape}")
        else:
            # Use pre-extracted tensors
            print("\n" + "=" * 80)
            print("Loading Pre-extracted Tensors")
            print("=" * 80)

            tensor_path = Path("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
            if not tensor_path.exists():
                raise FileNotFoundError(f"Pre-extracted tensors not found at {tensor_path}")

            data = torch.load(tensor_path)
            ref_code = data["ref_code"]

            # Codec tokens: first codebook of ref_code
            codec_tokens = ref_code[:, 0].unsqueeze(0)  # [1, ref_len]

            # Simple text tokenization (placeholder)
            # In practice, need proper tokenizer
            text_tokens = torch.randint(0, 151936, (1, 10))

            print(f"  codec_tokens shape: {codec_tokens.shape}")
            print(f"  text_tokens shape: {text_tokens.shape}")

        # Get embedding weights
        codec_embedding_weight = state_dict["talker.model.codec_embedding.weight"]
        text_embedding_weight = state_dict["talker.model.text_embedding.weight"]
        print(f"\nEmbedding weights loaded:")
        print(f"  codec_embedding: {codec_embedding_weight.shape}")
        print(f"  text_embedding: {text_embedding_weight.shape}")

        # Run TTNN generation
        print("\n" + "=" * 80)
        print("Running TTNN Autoregressive Generation")
        print("=" * 80)

        gen_start = time.time()
        generated_codec, cp_tokens = run_ttnn_generation(
            device=device,
            model=model,
            codec_tokens=codec_tokens,
            text_tokens=text_tokens,
            codec_embedding_weight=codec_embedding_weight,
            text_embedding_weight=text_embedding_weight,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=50,
        )
        # Rename for clarity
        codec_tokens_generated = generated_codec
        gen_time = time.time() - gen_start

        print(f"\nGeneration completed in {gen_time:.2f}s")
        print(f"  Generated {len(codec_tokens_generated)} tokens")
        print(f"  Tokens per second: {len(codec_tokens_generated) / gen_time:.2f}")

        # Convert to RVQ codes
        if len(codec_tokens_generated) > 0:
            rvq_codes = tokens_to_rvq_codes(codec_tokens_generated, cp_tokens)
            print(f"  RVQ codes shape: {rvq_codes.shape}")

            # Decode audio
            if use_official_decode and has_qwen_tts:
                print("\n" + "=" * 80)
                print("Decoding Audio (using qwen_tts speech tokenizer)")
                print("=" * 80)

                # Use official speech tokenizer decoder
                tts_model = official_model.model
                speech_tokenizer = tts_model.speech_tokenizer

                # Decode RVQ codes to audio
                audio = speech_tokenizer.decode(rvq_codes.unsqueeze(0))  # Add batch dim
                if isinstance(audio, torch.Tensor):
                    audio = audio.squeeze().numpy()

                print(f"  Audio shape: {audio.shape}")
                print(f"  Duration: {len(audio) / 24000:.2f} seconds")

                # Save audio
                import soundfile as sf

                sf.write(audio_output, audio, 24000)
                print(f"\nAudio saved to: {audio_output}")
            else:
                print("\nSkipping audio decode (qwen_tts not available)")
                # Save tokens for later decoding
                torch.save(
                    {
                        "codec_tokens": codec_tokens,
                        "cp_tokens": cp_tokens,
                        "rvq_codes": rvq_codes,
                    },
                    "/tmp/ttnn_generated_tokens.pt",
                )
                print("Saved tokens to /tmp/ttnn_generated_tokens.pt")
        else:
            print("No tokens generated!")

    finally:
        ttnn.close_device(device)

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="TTNN Voice Clone Demo")
    parser.add_argument(
        "--ref-audio", type=str, default="/tmp/clone_ref.wav", help="Reference audio file for voice cloning"
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default="Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
        help="Transcript of reference audio",
    )
    parser.add_argument(
        "--text", type=str, default="Hello, this is a test of the text to speech system.", help="Text to synthesize"
    )
    parser.add_argument("--audio-output", type=str, default="/tmp/ttnn_voice_clone.wav", help="Output audio file path")
    parser.add_argument("--device-id", type=int, default=0, help="TTNN device ID")
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum number of tokens to generate")

    args = parser.parse_args()

    run_demo(
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        text=args.text,
        audio_output=args.audio_output,
        device_id=args.device_id,
        max_new_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
