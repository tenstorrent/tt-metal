# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Trace through decoder to find where values become small.
"""

from pathlib import Path

import torch
import torch.nn.functional as F


def trace_decoder():
    """Trace through decoder step by step."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    from models.demos.qwen3_tts.reference.functional import (
        SpeechTokenizerDecoderConfig,
        codebook_lookup_rvq,
        convnext_block,
        pre_transformer_forward,
        snake_activation,
    )

    # Load weights
    model_path = snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", allow_patterns=["*.safetensors"])
    model_path = Path(model_path)
    speech_tokenizer_path = model_path / "speech_tokenizer" / "model.safetensors"
    raw_dict = load_file(speech_tokenizer_path)
    state_dict = {k[8:]: v for k, v in raw_dict.items() if k.startswith("decoder.")}

    # Load test codes
    data = torch.load("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
    codes = data["ref_code"].T.unsqueeze(0)  # [1, 16, 101]

    config = SpeechTokenizerDecoderConfig()

    def print_stats(name, x):
        print(f"{name}: shape={x.shape}, range=[{x.min():.4f}, {x.max():.4f}], std={x.std():.4f}")

    # Step 1: Codebook lookup
    print("=" * 60)
    print("Step 1: Codebook lookup")
    rvq_first_codebook = state_dict.get("quantizer.rvq_first.vq.layers.0._codebook.embedding_sum")
    rvq_first_cluster_usage = state_dict.get("quantizer.rvq_first.vq.layers.0._codebook.cluster_usage")
    rvq_first_output_proj = state_dict.get("quantizer.rvq_first.output_proj.weight")

    rvq_rest_codebooks = []
    rvq_rest_cluster_usages = []
    for i in range(15):
        key = f"quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"
        usage_key = f"quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage"
        if key in state_dict:
            rvq_rest_codebooks.append(state_dict[key])
            rvq_rest_cluster_usages.append(state_dict.get(usage_key))
    rvq_rest_output_proj = state_dict.get("quantizer.rvq_rest.output_proj.weight")

    hidden = codebook_lookup_rvq(
        codes,
        rvq_first_codebook,
        rvq_rest_codebooks,
        rvq_first_output_proj,
        rvq_rest_output_proj,
        rvq_first_cluster_usage,
        rvq_rest_cluster_usages,
    )
    print_stats("  After codebook lookup", hidden)

    # Step 2: Pre-conv
    print("\n" + "=" * 60)
    print("Step 2: Pre-conv")
    pre_conv_weight = state_dict.get("pre_conv.conv.weight")
    kernel_size = pre_conv_weight.shape[-1]
    padding = kernel_size - 1  # Causal padding
    hidden = F.pad(hidden, (padding, 0), mode="constant", value=0)
    hidden = F.conv1d(hidden, pre_conv_weight)
    print_stats("  After pre_conv", hidden)

    # Step 3: Pre-transformer
    print("\n" + "=" * 60)
    print("Step 3: Pre-transformer")
    pre_transformer_weights = {
        k.replace("pre_transformer.", ""): v for k, v in state_dict.items() if k.startswith("pre_transformer.")
    }
    hidden_t = hidden.transpose(1, 2)  # [B, T, C]
    hidden_t = pre_transformer_forward(hidden_t, pre_transformer_weights, config)
    hidden = hidden_t.transpose(1, 2)  # [B, C, T]
    print_stats("  After pre_transformer", hidden)

    # Step 4: Upsample blocks
    print("\n" + "=" * 60)
    print("Step 4: Upsample blocks")
    for i, ratio in enumerate(config.upsampling_ratios):
        print(f"\n  Upsample block {i} (ratio={ratio}):")
        # ConvTranspose1d upsample
        conv_weight = state_dict.get(f"upsample.{i}.0.conv.weight")
        if conv_weight is not None:
            hidden = F.conv_transpose1d(hidden, conv_weight, stride=ratio)
            # Trim padding
            pad = conv_weight.shape[-1] - ratio
            if pad > 0:
                hidden = hidden[..., :-pad]
            print_stats(f"    After conv_transpose", hidden)

            # ConvNeXt block
            convnext_weights = {
                k.replace(f"upsample.{i}.1.", ""): v for k, v in state_dict.items() if k.startswith(f"upsample.{i}.1.")
            }
            if convnext_weights and "dwconv.conv.weight" in convnext_weights:
                hidden = convnext_block(
                    hidden,
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
                print_stats(f"    After ConvNeXt", hidden)

    # Step 5: Decoder initial conv
    print("\n" + "=" * 60)
    print("Step 5: Decoder initial conv")
    if "decoder.0.conv.weight" in state_dict:
        conv_weight = state_dict["decoder.0.conv.weight"]
        kernel_size = conv_weight.shape[-1]
        hidden = F.pad(hidden, (kernel_size - 1, 0), mode="constant", value=0)
        hidden = F.conv1d(hidden, conv_weight)
        print_stats("  After decoder.0", hidden)

    # Step 6: Decoder blocks
    print("\n" + "=" * 60)
    print("Step 6: Decoder blocks")
    for i, rate in enumerate(config.upsample_rates):
        print(f"\n  Decoder block {i + 1} (rate={rate}):")
        block_prefix = f"decoder.{i + 1}."

        # Snake activation before upsample (block.0)
        alpha_key = f"{block_prefix}block.0.alpha"
        beta_key = f"{block_prefix}block.0.beta"
        if alpha_key in state_dict and beta_key in state_dict:
            hidden = snake_activation(hidden, state_dict[alpha_key], state_dict[beta_key])
            print_stats(f"    After snake (block.0)", hidden)

        # ConvTranspose1d upsample (block.1)
        upsample_weight = state_dict.get(f"{block_prefix}block.1.conv.weight")
        if upsample_weight is not None:
            kernel_size = upsample_weight.shape[-1]
            padding = (kernel_size - rate) // 2
            hidden = F.conv_transpose1d(hidden, upsample_weight, stride=rate, padding=padding)
            print_stats(f"    After upsample (block.1)", hidden)

        # Residual layers (block.2, block.3, block.4 with dilations 1, 3, 9)
        for j, dilation in enumerate([1, 3, 9]):
            res_prefix = f"{block_prefix}block.{j + 2}."
            residual = hidden

            # act1 + conv1
            act1_alpha = state_dict.get(f"{res_prefix}act1.alpha")
            act1_beta = state_dict.get(f"{res_prefix}act1.beta")
            if act1_alpha is not None:
                hidden = snake_activation(hidden, act1_alpha, act1_beta)

            conv1_weight = state_dict.get(f"{res_prefix}conv1.conv.weight")
            if conv1_weight is not None:
                kernel_size = conv1_weight.shape[-1]
                effective_kernel = (kernel_size - 1) * dilation + 1
                hidden = F.pad(hidden, (effective_kernel - 1, 0), mode="constant", value=0)
                hidden = F.conv1d(hidden, conv1_weight, dilation=dilation)

            # act2 + conv2
            act2_alpha = state_dict.get(f"{res_prefix}act2.alpha")
            act2_beta = state_dict.get(f"{res_prefix}act2.beta")
            if act2_alpha is not None:
                hidden = snake_activation(hidden, act2_alpha, act2_beta)

            conv2_weight = state_dict.get(f"{res_prefix}conv2.conv.weight")
            if conv2_weight is not None:
                hidden = F.conv1d(hidden, conv2_weight)

            hidden = residual + hidden

        print_stats(f"    After residuals", hidden)

    # Step 7: Final activation and conv
    print("\n" + "=" * 60)
    print("Step 7: Final conv")
    if "decoder.5.alpha" in state_dict:
        hidden = snake_activation(hidden, state_dict["decoder.5.alpha"], state_dict["decoder.5.beta"])
        print_stats("  After final snake", hidden)

    if "decoder.6.conv.weight" in state_dict:
        conv_weight = state_dict["decoder.6.conv.weight"]
        kernel_size = conv_weight.shape[-1]
        hidden = F.pad(hidden, (kernel_size - 1, 0), mode="constant", value=0)
        hidden = F.conv1d(hidden, conv_weight)
        print_stats("  After decoder.6", hidden)

    # Final clamp
    audio = hidden.clamp(min=-1, max=1)
    print_stats("  Final audio", audio)


if __name__ == "__main__":
    trace_decoder()
