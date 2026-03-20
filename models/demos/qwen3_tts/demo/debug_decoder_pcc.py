# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Debug decoder PCC - compare reference vs official at each stage.
"""

from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def main():
    print("=" * 80)
    print("Debug Decoder PCC - Reference vs Official")
    print("=" * 80)

    # Load official model
    print("\n[1] Loading official model...")
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cpu",
        dtype=torch.float32,
    )
    speech_tokenizer = model.model.speech_tokenizer.model
    official_decoder = speech_tokenizer.decoder

    # Load reference weights
    print("\n[2] Loading reference weights...")
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    model_path = Path(model_path)

    speech_path = model_path / "speech_tokenizer" / "model.safetensors"
    speech_dict = load_file(speech_path)
    decoder_weights = {k[8:]: v.float() for k, v in speech_dict.items() if k.startswith("decoder.")}
    print(f"  Loaded {len(decoder_weights)} decoder weights")

    # Generate some codes using official model
    print("\n[3] Generating codes with official model...")
    ref_audio_path = "/tmp/clone_ref.wav"
    ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    target_text = "Hello, how are you?"

    # Capture codes from official generation
    captured_codes = []

    def capture_decoder_input(module, args, kwargs):
        if len(args) > 0:
            captured_codes.append(args[0].clone().detach())
        return args, kwargs

    hook = official_decoder.register_forward_pre_hook(capture_decoder_input, with_kwargs=True)

    wavs, sr = model.generate_voice_clone(
        text=target_text,
        language="English",
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        max_new_tokens=50,
    )

    hook.remove()

    if not captured_codes:
        print("ERROR: No codes captured!")
        return

    codes = captured_codes[0]
    print(f"  Captured codes shape: {codes.shape}")  # [batch, 16, seq_len]
    print(f"  Codes range: [{codes.min().item()}, {codes.max().item()}]")

    # Save official audio
    sf.write("/tmp/debug_official.wav", wavs[0], sr)
    print(f"  Official audio saved: {len(wavs[0])/sr:.2f}s")

    # Now decode with reference and compare at each stage
    print("\n[4] Comparing decoder stages...")

    from models.demos.qwen3_tts.reference.functional import SpeechTokenizerDecoderConfig

    config = SpeechTokenizerDecoderConfig()

    # === Stage 1: Codebook lookup ===
    print("\n  [Stage 1] Codebook Lookup...")

    # Official codebook lookup
    with torch.no_grad():
        # Official uses quantizer.decode()
        official_emb = official_decoder.quantizer.decode(codes)
        print(f"    Official embedding: {official_emb.shape}")

    # Reference codebook lookup
    # RVQ first (semantic)
    rvq_first_emb = decoder_weights["quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"]
    rvq_first_usage = decoder_weights["quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"]
    rvq_first_codebook = rvq_first_emb / rvq_first_usage.clamp(min=1e-6).unsqueeze(-1)

    # RVQ rest (acoustic) - 15 codebooks
    rvq_rest_codebooks = []
    for i in range(15):
        emb_key = f"quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"
        usage_key = f"quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage"
        emb = decoder_weights[emb_key]
        usage = decoder_weights[usage_key]
        codebook = emb / usage.clamp(min=1e-6).unsqueeze(-1)
        rvq_rest_codebooks.append(codebook)

    # Lookup
    codes_np = codes[0].numpy()  # [16, seq_len]
    seq_len = codes_np.shape[1]

    # First codebook
    first_codes = codes_np[0]  # [seq_len]
    ref_first_emb = rvq_first_codebook[first_codes]  # [seq_len, 256]

    # Rest codebooks - sum them
    rest_emb_sum = torch.zeros(seq_len, 256)
    for i in range(15):
        cb_codes = codes_np[i + 1]
        cb_emb = rvq_rest_codebooks[i][cb_codes]
        rest_emb_sum += cb_emb

    # Output projections
    rvq_first_proj = decoder_weights["quantizer.rvq_first.output_proj.weight"]
    rvq_rest_proj = decoder_weights["quantizer.rvq_rest.output_proj.weight"]

    ref_first_proj = F.linear(ref_first_emb, rvq_first_proj)  # [seq_len, 512]
    ref_rest_proj = F.linear(rest_emb_sum, rvq_rest_proj)  # [seq_len, 512]

    # Concatenate
    ref_emb = torch.cat([ref_first_proj, ref_rest_proj], dim=-1)  # [seq_len, 1024]
    ref_emb = ref_emb.unsqueeze(0)  # [1, seq_len, 1024]

    print(f"    Reference embedding: {ref_emb.shape}")

    pcc_emb = compute_pcc(ref_emb, official_emb)
    print(f"    Codebook PCC: {pcc_emb:.6f}")

    if pcc_emb < 0.99:
        print("    *** MISMATCH in codebook lookup ***")
        print(f"    Ref: mean={ref_emb.mean():.4f}, std={ref_emb.std():.4f}")
        print(f"    Off: mean={official_emb.mean():.4f}, std={official_emb.std():.4f}")

    # === Stage 2: Pre-conv ===
    print("\n  [Stage 2] Pre-conv...")

    with torch.no_grad():
        # Official pre_conv
        official_preconv = official_decoder.pre_conv(official_emb.transpose(1, 2))
        print(f"    Official pre_conv: {official_preconv.shape}")

    # Reference pre_conv
    pre_conv_weight = decoder_weights["pre_conv.conv.weight"]
    pre_conv_bias = decoder_weights["pre_conv.conv.bias"]

    ref_preconv_input = ref_emb.transpose(1, 2)  # [1, 1024, seq_len]
    # Causal padding
    kernel_size = pre_conv_weight.shape[2]
    pad_left = kernel_size - 1
    ref_preconv_padded = F.pad(ref_preconv_input, (pad_left, 0), mode="constant", value=0)
    ref_preconv = F.conv1d(ref_preconv_padded, pre_conv_weight, pre_conv_bias)

    print(f"    Reference pre_conv: {ref_preconv.shape}")

    pcc_preconv = compute_pcc(ref_preconv, official_preconv)
    print(f"    Pre-conv PCC: {pcc_preconv:.6f}")

    if pcc_preconv < 0.99:
        print("    *** MISMATCH in pre_conv ***")

    # === Stage 3: Pre-transformer ===
    print("\n  [Stage 3] Pre-transformer...")

    with torch.no_grad():
        # Need to run through official pre_transformer
        official_pretrans_input = official_preconv.transpose(1, 2)  # [1, seq_len, 512]
        official_pretrans = official_decoder.pre_transformer(official_pretrans_input)
        if hasattr(official_pretrans, "last_hidden_state"):
            official_pretrans = official_pretrans.last_hidden_state
        print(f"    Official pre_transformer: {official_pretrans.shape}")

    # Reference pre_transformer - this is complex, let's just run it
    from models.demos.qwen3_tts.reference.functional import pre_transformer_forward

    ref_pretrans_input = ref_preconv.transpose(1, 2)  # [1, seq_len, 512]
    ref_pretrans = pre_transformer_forward(ref_pretrans_input, decoder_weights, config)

    print(f"    Reference pre_transformer: {ref_pretrans.shape}")

    pcc_pretrans = compute_pcc(ref_pretrans, official_pretrans)
    print(f"    Pre-transformer PCC: {pcc_pretrans:.6f}")

    if pcc_pretrans < 0.99:
        print("    *** MISMATCH in pre_transformer ***")
        print(f"    Ref: mean={ref_pretrans.mean():.4f}, std={ref_pretrans.std():.4f}")
        print(f"    Off: mean={official_pretrans.mean():.4f}, std={official_pretrans.std():.4f}")

    # === Stage 4: Upsampler ===
    print("\n  [Stage 4] Upsampler...")

    with torch.no_grad():
        official_upsample = official_decoder.upsample(official_pretrans.transpose(1, 2))
        print(f"    Official upsample: {official_upsample.shape}")

    # Reference upsampler
    from models.demos.qwen3_tts.reference.functional import upsample_block

    ref_upsample_input = ref_pretrans.transpose(1, 2)  # [1, 512, seq_len]

    # Two upsample blocks
    for i in range(2):
        prefix = f"upsample.{i}."
        block_weights = {k.replace(prefix, ""): v for k, v in decoder_weights.items() if k.startswith(prefix)}
        ref_upsample_input = upsample_block(ref_upsample_input, block_weights, upsample_rate=2)

    ref_upsample = ref_upsample_input
    print(f"    Reference upsample: {ref_upsample.shape}")

    pcc_upsample = compute_pcc(ref_upsample, official_upsample)
    print(f"    Upsample PCC: {pcc_upsample:.6f}")

    if pcc_upsample < 0.99:
        print("    *** MISMATCH in upsampler ***")

    # === Stage 5: Conv Decoder ===
    print("\n  [Stage 5] Conv Decoder...")

    with torch.no_grad():
        official_audio = official_decoder.decoder(official_upsample)
        print(f"    Official audio: {official_audio.shape}")

    # Reference conv decoder
    from models.demos.qwen3_tts.reference.functional import conv_decoder_block, snake_activation

    ref_decoder_input = ref_upsample

    # Conv decoder has multiple blocks with different upsample rates
    upsample_rates = [8, 5, 4, 3]
    channels = [512, 256, 128, 64, 32]

    for i, (rate, in_ch, out_ch) in enumerate(zip(upsample_rates, channels[:-1], channels[1:])):
        prefix = f"decoder.{i}."
        block_weights = {k.replace(prefix, ""): v for k, v in decoder_weights.items() if k.startswith(prefix)}
        ref_decoder_input = conv_decoder_block(ref_decoder_input, block_weights, upsample_rate=rate)

    # Final conv
    final_conv_weight = decoder_weights["decoder.4.conv.weight"]
    final_conv_bias = decoder_weights["decoder.4.conv.bias"]

    # Apply snake activation first
    snake_alpha = decoder_weights["decoder.4.snake.alpha"]
    ref_decoder_input = snake_activation(ref_decoder_input, snake_alpha)

    # Final conv (no padding needed for kernel_size=7)
    kernel_size = final_conv_weight.shape[2]
    pad_left = kernel_size - 1
    ref_decoder_padded = F.pad(ref_decoder_input, (pad_left, 0), mode="constant", value=0)
    ref_audio = F.conv1d(ref_decoder_padded, final_conv_weight, final_conv_bias)

    # Tanh
    ref_audio = torch.tanh(ref_audio)

    print(f"    Reference audio: {ref_audio.shape}")

    pcc_audio = compute_pcc(ref_audio, official_audio)
    print(f"    Final Audio PCC: {pcc_audio:.6f}")

    if pcc_audio < 0.99:
        print("    *** MISMATCH in conv decoder ***")
        print(
            f"    Ref: mean={ref_audio.mean():.4f}, std={ref_audio.std():.4f}, range=[{ref_audio.min():.4f}, {ref_audio.max():.4f}]"
        )
        print(
            f"    Off: mean={official_audio.mean():.4f}, std={official_audio.std():.4f}, range=[{official_audio.min():.4f}, {official_audio.max():.4f}]"
        )

    # Save reference audio
    ref_audio_np = ref_audio.squeeze().detach().cpu().float().numpy()
    sf.write("/tmp/debug_reference.wav", ref_audio_np, 24000)
    print(f"\n  Reference audio saved: {len(ref_audio_np)/24000:.2f}s")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  Codebook lookup PCC:  {pcc_emb:.6f}")
    print(f"  Pre-conv PCC:         {pcc_preconv:.6f}")
    print(f"  Pre-transformer PCC:  {pcc_pretrans:.6f}")
    print(f"  Upsampler PCC:        {pcc_upsample:.6f}")
    print(f"  Final Audio PCC:      {pcc_audio:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
