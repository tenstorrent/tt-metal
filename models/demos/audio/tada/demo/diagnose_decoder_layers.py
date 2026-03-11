#!/usr/bin/env python3
"""
Per-layer decoder diagnosis — trace exactly which layer(s) in the
LocalAttentionEncoder cause the PCC=0.89 divergence.

Usage:
    source python_env/bin/activate
    pytest models/demos/audio/tada/demo/diagnose_decoder_layers.py -v -s
"""

import os
import sys

_TT_METAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, _TT_METAL_ROOT)


import pytest
import torch
from loguru import logger

import ttnn
from models.demos.audio.tada.tt.ttnn_functional_common import (
    TADA_MEMORY_CONFIG,
    compute_rope_freqs,
    local_attention_encoder_layer,
)
from models.demos.utils.common_demo_utils import get_mesh_mappers

TADA_L1_SMALL_SIZE = 24576
TADA_CODEC_PATH = os.environ.get("TADA_CODEC_PATH", "HumeAI/tada-codec")


def pcc(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return torch.corrcoef(torch.stack([a_flat, b_flat]))[0, 1].item()


def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": TADA_L1_SMALL_SIZE}], indirect=True)
def test_per_layer_decoder(mesh_device):
    """Trace divergence through each decoder layer."""
    from pathlib import Path

    from safetensors.torch import load_file as safetensors_load_file
    from ttnn.model_preprocessing import preprocess_model_parameters

    from models.demos.audio.tada.reference.tada_reference import Decoder as RefDecoder
    from models.demos.audio.tada.reference.tada_reference import create_decoder_segment_attention_mask
    from models.demos.audio.tada.tt.ttnn_functional_decoder import convert_to_ttnn, create_custom_mesh_preprocessor

    # Load reference decoder
    codec_path = TADA_CODEC_PATH
    if not os.path.isdir(codec_path):
        from huggingface_hub import snapshot_download

        codec_path = snapshot_download(codec_path)

    dec_path = os.path.join(codec_path, "decoder")
    state_dict = {}
    for sf_file in sorted(Path(dec_path).glob("*.safetensors")):
        state_dict.update(safetensors_load_file(str(sf_file)))

    ref_decoder = RefDecoder()
    dec_state = {}
    for k, v in state_dict.items():
        if "_precomputed_mask" in k or "rope_freqs" in k:
            continue
        dec_state[k] = v.float()
    ref_decoder.load_state_dict(dec_state, strict=False)
    ref_decoder.eval()

    input_mesh_mapper, output_mesh_composer, weights_mesh_mapper = get_mesh_mappers(mesh_device)

    # Preprocess the local_attention_decoder for TTNN
    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_decoder.local_attention_decoder,
        convert_to_ttnn=convert_to_ttnn,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )

    # Create test input
    torch.manual_seed(42)
    seq_len = 64
    d_model = 1024
    num_heads = 8
    head_dim = d_model // num_heads

    # Start from decoder_proj output (already validated as PCC=0.9999)
    x_cpu = torch.randn(1, seq_len, d_model) * 0.3
    # Create realistic mask pattern
    token_masks = torch.zeros(1, seq_len, dtype=torch.long)
    token_masks[:, ::4] = 1
    attn_mask_bool = create_decoder_segment_attention_mask(token_masks, version="v2")

    logger.info(f"Input: shape={x_cpu.shape}, norm={x_cpu.norm():.4f}")

    # Precompute RoPE (kept on CPU for interleaved rotation)
    rope_freqs = compute_rope_freqs(head_dim, seq_len)
    rope_cos_cpu = rope_freqs[:, :, 0]  # (S, D/2)
    rope_sin_cpu = rope_freqs[:, :, 1]  # (S, D/2)

    # Convert attention mask
    float_mask = torch.zeros_like(attn_mask_bool, dtype=torch.bfloat16)
    float_mask.masked_fill_(attn_mask_bool, float("-inf"))
    float_mask = float_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
    attn_mask_tt = ttnn.from_torch(
        float_mask,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=input_mesh_mapper,
    )

    # Apply input_proj on CPU
    with torch.no_grad():
        cpu_x = ref_decoder.local_attention_decoder.input_proj(x_cpu)
    logger.info(f"After input_proj: CPU norm={cpu_x.norm():.4f}")

    # Apply input_proj on TT
    x_tt_4d = ttnn.from_torch(
        x_cpu.unsqueeze(1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=input_mesh_mapper,
    )
    try:
        x_tt_4d = ttnn.linear(
            x_tt_4d,
            parameters.input_proj.weight,
            bias=getattr(parameters.input_proj, "bias", None),
            memory_config=TADA_MEMORY_CONFIG,
        )
    except (KeyError, AttributeError):
        pass

    tt_x_check = ttnn.to_torch(x_tt_4d, mesh_composer=output_mesh_composer)
    if tt_x_check.dim() == 4:
        tt_x_check = tt_x_check.squeeze(1)
    tt_x_check = tt_x_check[:, :seq_len, :]
    ip_pcc = pcc(cpu_x, tt_x_check)
    logger.info(f"Input_proj: PCC={ip_pcc:.6f}, CPU norm={cpu_x.norm():.4f}, TT norm={tt_x_check.norm():.4f}")

    # Now go layer by layer
    logger.info("\n" + "=" * 90)
    logger.info(f"{'Layer':>6} | {'CPU Norm':>10} | {'TT Norm':>10} | {'PCC':>10} | {'CosSim':>10} | {'Max Diff':>10}")
    logger.info("-" * 90)

    for i, cpu_layer in enumerate(ref_decoder.local_attention_decoder.layers):
        # CPU forward
        with torch.no_grad():
            cpu_x = cpu_layer(cpu_x, mask=attn_mask_bool)

        # TT forward
        x_tt_4d = local_attention_encoder_layer(
            x_tt_4d,
            seq_len,
            attn_mask_tt,
            rope_cos_cpu,
            rope_sin_cpu,
            parameters=parameters.layers[i],
            device=mesh_device,
            input_mesh_mapper=input_mesh_mapper,
            output_mesh_composer=output_mesh_composer,
        )

        # Compare
        tt_x = ttnn.to_torch(x_tt_4d, mesh_composer=output_mesh_composer)
        if tt_x.dim() == 4:
            tt_x = tt_x.squeeze(1)
        tt_x = tt_x[:, :seq_len, :]

        layer_pcc = pcc(cpu_x, tt_x)
        layer_cos = cos_sim(cpu_x, tt_x)
        max_diff = (cpu_x.float() - tt_x.float()).abs().max().item()

        logger.info(
            f"  L{i:>3} | {cpu_x.norm():>10.4f} | {tt_x.norm():>10.4f} | "
            f"{layer_pcc:>10.6f} | {layer_cos:>10.6f} | {max_diff:>10.4f}"
        )

    # Final LayerNorm
    with torch.no_grad():
        cpu_final = ref_decoder.local_attention_decoder.final_norm(cpu_x)

    x_tt_4d = ttnn.layer_norm(
        x_tt_4d,
        weight=parameters.final_norm.weight,
        bias=parameters.final_norm.bias,
        memory_config=TADA_MEMORY_CONFIG,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )
    tt_final = ttnn.to_torch(x_tt_4d, mesh_composer=output_mesh_composer)
    if tt_final.dim() == 4:
        tt_final = tt_final.squeeze(1)
    tt_final = tt_final[:, :seq_len, :]

    final_pcc = pcc(cpu_final, tt_final)
    final_cos = cos_sim(cpu_final, tt_final)
    max_diff = (cpu_final.float() - tt_final.float()).abs().max().item()
    logger.info(
        f" Final | {cpu_final.norm():>10.4f} | {tt_final.norm():>10.4f} | "
        f"{final_pcc:>10.6f} | {final_cos:>10.6f} | {max_diff:>10.4f}"
    )
    logger.info("=" * 90)

    # Also test with the REAL acoustic features from the demo run
    demo_output_dir = os.path.join(os.path.dirname(__file__), "output")
    debug_path = os.path.join(demo_output_dir, "debug_hidden_states.pt")
    if os.path.exists(debug_path):
        logger.info("\nAlso testing with real acoustic features from demo run...")
        # The actual issue might be seq_len dependent (448 for the full demo)

    logger.info(f"\nFinal decoder transformer PCC: {final_pcc:.6f}")
    if final_pcc < 0.95:
        logger.warning(f"LOW PCC ({final_pcc:.4f}) — this explains the audio distortion!")
    else:
        logger.info(f"PCC looks reasonable ({final_pcc:.4f})")
