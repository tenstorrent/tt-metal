# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Hybrid TTS Demo - Reference Talker + Official Code Predictor

Tests whether the issue is in the talker or code predictor by using
reference talker with official code predictor.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


def load_weights():
    """Load all model weights."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print("Loading model weights...")
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    model_path = Path(model_path)

    main_dict = load_file(model_path / "model.safetensors")
    main_dict = {k: v.float() for k, v in main_dict.items()}

    speech_path = model_path / "speech_tokenizer" / "model.safetensors"
    speech_dict = load_file(speech_path)
    decoder_weights = {k[8:]: v.float() for k, v in speech_dict.items() if k.startswith("decoder.")}

    return main_dict, decoder_weights


def main():
    parser = argparse.ArgumentParser(description="Hybrid TTS Demo")
    parser.add_argument("--text", type=str, default="Hello, how are you today?")
    parser.add_argument("--ref-audio", type=str, default="/tmp/clone_ref.wav")
    parser.add_argument(
        "--ref-text",
        type=str,
        default="Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
    )
    parser.add_argument("--output", type=str, default="/tmp/hybrid_tts.wav")
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    print("=" * 80)
    print("Hybrid TTS Demo - Reference Talker + Official Code Predictor")
    print("=" * 80)

    # Load official model for code predictor
    print("\n[1] Loading official model for code predictor...")
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cpu",
        dtype=torch.float32,
    )
    tts_model = model.model
    talker = tts_model.talker
    code_predictor = talker.code_predictor

    # Load reference weights
    print("\n[2] Loading reference weights...")
    main_dict, decoder_weights = load_weights()

    from transformers import AutoTokenizer

    from models.demos.qwen3_tts.reference.functional import (
        Qwen3TTSConfig,
        SpeakerEncoderConfig,
        SpeechTokenizerDecoderConfig,
        compute_mel_spectrogram_qwen,
        compute_mrope_frequencies,
        decoder_layer,
        extract_speaker_encoder_weights,
        extract_talker_weights,
        rms_norm,
        speaker_encoder_forward,
        speech_tokenizer_decoder_forward,
        speech_tokenizer_encoder_forward_mimi,
    )

    talker_weights = extract_talker_weights(main_dict)
    codec_head = main_dict["talker.codec_head.weight"]
    codec_embed_weight = main_dict["talker.model.codec_embedding.weight"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", trust_remote_code=True)

    # Get TTS special embeddings
    text_embed_weight = main_dict["talker.model.text_embedding.weight"]
    text_proj_fc1_weight = main_dict["talker.text_projection.linear_fc1.weight"]
    text_proj_fc1_bias = main_dict["talker.text_projection.linear_fc1.bias"]
    text_proj_fc2_weight = main_dict["talker.text_projection.linear_fc2.weight"]
    text_proj_fc2_bias = main_dict["talker.text_projection.linear_fc2.bias"]

    def project_text(text_embeds):
        h = F.linear(text_embeds, text_proj_fc1_weight, text_proj_fc1_bias)
        h = F.silu(h)
        return F.linear(h, text_proj_fc2_weight, text_proj_fc2_bias)

    # Special tokens
    tts_bos_token_id = 151672
    tts_eos_token_id = 151673
    tts_pad_token_id = 151671
    codec_bos_id = 2149
    codec_eos_id = 2150
    codec_pad_id = 2148
    codec_think_id = 2154
    codec_think_bos_id = 2156
    codec_think_eos_id = 2157
    lang_id = 2050  # english

    # Get code predictor embeddings
    code_pred_embeds = []
    for i in range(15):
        code_pred_embeds.append(code_predictor.get_input_embeddings()[i].weight.detach())

    # =========================================================================
    # Step 1: Encode reference audio
    # =========================================================================
    print("\n[3] Encoding reference audio...")
    from scipy import signal

    audio_data, sr = sf.read(args.ref_audio)
    audio_data = torch.from_numpy(audio_data.astype(np.float32))
    if audio_data.dim() == 2:
        audio_data = audio_data.mean(dim=1)
    if sr != 24000:
        num_samples = int(len(audio_data) * 24000 / sr)
        audio_data = torch.from_numpy(signal.resample(audio_data.numpy(), num_samples).astype(np.float32))

    mel = compute_mel_spectrogram_qwen(audio_data)
    speaker_weights = extract_speaker_encoder_weights(main_dict)
    speaker_config = SpeakerEncoderConfig()
    speaker_embedding = speaker_encoder_forward(mel, speaker_weights, speaker_config)

    ref_codes = speech_tokenizer_encoder_forward_mimi(audio_data.unsqueeze(0)).squeeze(0).T
    print(f"  Reference codes: {ref_codes.shape}")

    # =========================================================================
    # Step 2: Create ICL embeddings
    # =========================================================================
    print("\n[4] Creating ICL embeddings...")

    # Get TTS special embeddings
    tts_tokens = torch.tensor([[tts_bos_token_id, tts_eos_token_id, tts_pad_token_id]])
    tts_embeds = F.embedding(tts_tokens, text_embed_weight)
    tts_embeds_proj = project_text(tts_embeds)
    tts_bos_embed = tts_embeds_proj[:, 0:1, :]
    tts_eos_embed = tts_embeds_proj[:, 1:2, :]
    tts_pad_embed = tts_embeds_proj[:, 2:3, :]

    # Tokenize texts
    ref_text_ids = tokenizer.encode(args.ref_text, add_special_tokens=False, return_tensors="pt")
    target_text_ids = tokenizer.encode(args.text, add_special_tokens=False, return_tensors="pt")
    role_formatted = "<|im_start|>assistant\n"
    role_ids = tokenizer.encode(role_formatted, add_special_tokens=False, return_tensors="pt")

    # Build embeddings
    role_embeds_proj = project_text(F.embedding(role_ids, text_embed_weight))

    codec_prefix_ids = torch.tensor([[codec_think_id, codec_think_bos_id, lang_id, codec_think_eos_id]])
    codec_suffix_ids = torch.tensor([[codec_pad_id, codec_bos_id]])
    codec_prefix_embeds = F.embedding(codec_prefix_ids, codec_embed_weight)
    codec_suffix_embeds = F.embedding(codec_suffix_ids, codec_embed_weight)
    codec_input_embedding = torch.cat(
        [codec_prefix_embeds, speaker_embedding.view(1, 1, -1), codec_suffix_embeds], dim=1
    )

    codec_len = codec_input_embedding.shape[1]
    prefix_text = torch.cat([tts_pad_embed.expand(-1, codec_len - 2, -1), tts_bos_embed], dim=1)
    prefix_combined = prefix_text + codec_input_embedding[:, :-1, :]

    combined_text_ids = torch.cat([ref_text_ids, target_text_ids], dim=1)
    combined_text_proj = project_text(F.embedding(combined_text_ids, text_embed_weight))
    text_embed = torch.cat([combined_text_proj, tts_eos_embed], dim=1)
    text_lens = text_embed.shape[1]

    ref_len = ref_codes.shape[0]
    codec_embeds_list = []
    for i in range(16):
        code_ids = ref_codes[:, i : i + 1]
        if i == 0:
            cb_embed = F.embedding(code_ids, codec_embed_weight)
        else:
            cb_embed = F.embedding(code_ids, code_pred_embeds[i - 1])
        codec_embeds_list.append(cb_embed)

    stacked_embeds = torch.cat(codec_embeds_list, dim=1)
    summed_embeds = stacked_embeds.sum(dim=1).unsqueeze(0)
    codec_bos_embed = F.embedding(torch.tensor([[codec_bos_id]]), codec_embed_weight)
    codec_embed = torch.cat([codec_bos_embed, summed_embeds], dim=1)
    codec_lens = codec_embed.shape[1]

    if text_lens >= codec_lens:
        icl_input_embed = text_embed[:, :codec_lens, :] + codec_embed
    else:
        padding_len = codec_lens - text_lens
        text_padded = torch.cat([text_embed, tts_pad_embed.expand(-1, padding_len, -1)], dim=1)
        icl_input_embed = text_padded + codec_embed

    inputs_embeds = torch.cat([role_embeds_proj, prefix_combined, icl_input_embed], dim=1)
    print(f"  Input embeddings: {inputs_embeds.shape}")

    # =========================================================================
    # Step 3: Generate with Reference Talker + Official Code Predictor
    # =========================================================================
    print("\n[5] Generating with Reference Talker + Official Code Predictor...")

    talker_config = Qwen3TTSConfig()
    talker_config.num_hidden_layers = 28
    hidden_states = inputs_embeds.clone()

    all_codes = []
    num_code_groups = 16
    head_dim = 128
    rope_theta = 1000000.0
    rms_norm_eps = 1e-6

    for step in range(args.max_tokens):
        current_seq_len = hidden_states.shape[1]

        # Compute RoPE
        cos, sin = compute_mrope_frequencies(head_dim, current_seq_len, rope_theta, hidden_states.device)
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)

        # Causal mask
        attention_mask = (
            torch.triu(
                torch.full(
                    (current_seq_len, current_seq_len),
                    float("-inf"),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                ),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Reference Talker forward
        x = hidden_states
        for layer_idx in range(28):
            layer_prefix = f"layers.{layer_idx}."
            layer_weights = {
                k.replace(layer_prefix, ""): v for k, v in talker_weights.items() if k.startswith(layer_prefix)
            }
            x = decoder_layer(x, layer_weights, cos, sin, talker_config, attention_mask=attention_mask, use_mrope=True)

        x = rms_norm(x, talker_weights["norm.weight"], rms_norm_eps)
        last_hidden = x[:, -1:, :]

        # Generate codebook 0
        logits = F.linear(last_hidden.squeeze(1), codec_head)
        token_0 = logits.argmax(dim=-1).item()

        if token_0 == codec_eos_id:
            print(f"  EOS at step {step}")
            break

        # Use OFFICIAL code predictor for codebooks 1-15
        token_0_embed = F.embedding(torch.tensor([[token_0]]), codec_embed_weight)

        with torch.no_grad():
            predictor_result = code_predictor.generate(
                inputs_embeds=torch.cat((last_hidden, token_0_embed), dim=1),
                max_new_tokens=15,
                do_sample=False,  # Greedy
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        codec_ids = torch.cat((torch.tensor([[token_0]]), predictor_result.sequences), dim=-1)
        all_codes.append(codec_ids.squeeze().tolist())

        # Build next input: sum of all 16 codebook embeddings + tts_pad
        all_cb_embeds = [token_0_embed]
        for i in range(15):
            cb_embed = code_predictor.get_input_embeddings()[i](predictor_result.sequences[..., i : i + 1])
            all_cb_embeds.append(cb_embed)

        all_cb_stacked = torch.cat(all_cb_embeds, dim=1)
        next_embed = all_cb_stacked.sum(dim=1, keepdim=True) + tts_pad_embed

        hidden_states = torch.cat([hidden_states, next_embed], dim=1)

        if (step + 1) % 20 == 0:
            print(f"  Generated {step + 1} tokens...")

    print(f"  Generated {len(all_codes)} tokens")

    if len(all_codes) == 0:
        print("ERROR: No tokens generated!")
        return

    # =========================================================================
    # Step 4: Decode to audio
    # =========================================================================
    print("\n[6] Decoding to audio...")

    codes = torch.tensor(all_codes, dtype=torch.long)
    codes_filtered = codes.clamp(max=2047)
    codes_input = codes_filtered.T.unsqueeze(0)

    decoder_config = SpeechTokenizerDecoderConfig()
    audio = speech_tokenizer_decoder_forward(codes_input, decoder_weights, decoder_config)

    audio_np = audio.squeeze().detach().cpu().float().numpy()
    sf.write(args.output, audio_np, 24000)

    print(f"\n  Output: {args.output}")
    print(f"  Duration: {len(audio_np)/24000:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
