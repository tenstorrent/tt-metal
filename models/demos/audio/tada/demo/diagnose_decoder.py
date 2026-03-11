#!/usr/bin/env python3
"""
Diagnose decoder quality issue by comparing intermediate outputs
between TTNN and CPU reference decoder paths.

Loads the acoustic features from the last demo run and compares:
1. decoder_proj output (linear projection)
2. local_attention_encoder output (6-layer transformer)
3. DACDecoder CNN output (waveform)

Usage:
    source python_env/bin/activate
    pytest models/demos/audio/tada/demo/diagnose_decoder.py -v -s
"""

import os
import sys

_TT_METAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, _TT_METAL_ROOT)

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.audio.tada.tt.ttnn_functional_common import TADA_MEMORY_CONFIG, local_attention_encoder
from models.demos.audio.tada.tt.ttnn_functional_decoder import decoder_forward
from models.demos.utils.common_demo_utils import get_mesh_mappers

TADA_L1_SMALL_SIZE = 24576
TADA_CODEC_PATH = os.environ.get("TADA_CODEC_PATH", "HumeAI/tada-codec")


def pcc(a, b):
    """Pearson correlation coefficient."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return torch.corrcoef(torch.stack([a_flat, b_flat]))[0, 1].item()


def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": TADA_L1_SMALL_SIZE}], indirect=True)
def test_diagnose_decoder(mesh_device):
    """Compare decoder intermediate outputs between TTNN and CPU."""
    from pathlib import Path

    from safetensors.torch import load_file as safetensors_load_file

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

    # Setup mesh mappers
    input_mesh_mapper, output_mesh_composer, weights_mesh_mapper = get_mesh_mappers(mesh_device)

    # Preprocess for TTNN
    from ttnn.model_preprocessing import preprocess_model_parameters

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_decoder,
        convert_to_ttnn=convert_to_ttnn,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )

    # Create test input — use a simple pattern that's easy to validate
    torch.manual_seed(42)
    seq_len = 64  # Short sequence for debugging
    batch_size = 1
    embed_dim = 512

    encoded_expanded = torch.randn(batch_size, seq_len, embed_dim) * 0.5
    # Make some positions non-zero (simulating real acoustic features)
    token_masks = torch.zeros(batch_size, seq_len, dtype=torch.long)
    token_masks[:, ::4] = 1  # Every 4th position has content
    # Zero out masked positions
    for b in range(batch_size):
        for t in range(seq_len):
            if token_masks[b, t] == 0:
                encoded_expanded[b, t] = 0.0

    logger.info(f"Input: shape={encoded_expanded.shape}, norm={encoded_expanded.norm():.4f}")
    logger.info(f"Token masks: sum={token_masks.sum()}, pattern={token_masks[0, :20].tolist()}")

    # ========== CPU Reference Path ==========
    with torch.no_grad():
        # Step 1: decoder_proj
        cpu_proj = ref_decoder.decoder_proj(encoded_expanded.float())
        logger.info(f"CPU decoder_proj: shape={cpu_proj.shape}, norm={cpu_proj.norm():.4f}")

        # Step 2: attention mask
        attn_mask = create_decoder_segment_attention_mask(token_masks, version="v2")
        logger.info(f"Attention mask: shape={attn_mask.shape}, True%={attn_mask.float().mean():.3f}")

        # Step 3: local attention encoder
        cpu_decoded = ref_decoder.local_attention_decoder(cpu_proj, mask=attn_mask)
        logger.info(f"CPU local_attn: shape={cpu_decoded.shape}, norm={cpu_decoded.norm():.4f}")

        # Step 4: DAC decoder
        cpu_wav = ref_decoder.wav_decoder(cpu_decoded.transpose(1, 2).float())
        logger.info(f"CPU waveform: shape={cpu_wav.shape}, norm={cpu_wav.norm():.4f}")

    # ========== TTNN Path ==========
    # Step 1: Transfer and project
    x_tt = ttnn.from_torch(
        encoded_expanded.unsqueeze(1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=input_mesh_mapper,
    )
    x_tt = ttnn.linear(
        x_tt,
        parameters.decoder_proj.weight,
        bias=parameters.decoder_proj.bias,
        memory_config=TADA_MEMORY_CONFIG,
    )

    # Check decoder_proj output
    tt_proj = ttnn.to_torch(x_tt, mesh_composer=output_mesh_composer)
    if tt_proj.dim() == 4:
        tt_proj = tt_proj.squeeze(1)
    tt_proj = tt_proj[:, :seq_len, :]
    proj_pcc = pcc(cpu_proj, tt_proj)
    proj_cos = cos_sim(cpu_proj, tt_proj)
    logger.info(f"TTNN decoder_proj: norm={tt_proj.norm():.4f}, PCC={proj_pcc:.6f}, cos={proj_cos:.6f}")

    # Step 2: local attention encoder
    x_tt_attn = ttnn.from_torch(
        tt_proj.unsqueeze(1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=input_mesh_mapper,
    )
    x_tt_attn = local_attention_encoder(
        x_tt_attn,
        seq_len,
        attention_mask_torch=attn_mask,
        parameters=parameters.local_attention_decoder,
        device=mesh_device,
        input_mesh_mapper=input_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )

    tt_decoded = ttnn.to_torch(x_tt_attn, mesh_composer=output_mesh_composer)
    if tt_decoded.dim() == 4:
        tt_decoded = tt_decoded.squeeze(1)
    tt_decoded = tt_decoded[:, :seq_len, :]
    attn_pcc = pcc(cpu_decoded, tt_decoded)
    attn_cos = cos_sim(cpu_decoded, tt_decoded)
    logger.info(f"TTNN local_attn: norm={tt_decoded.norm():.4f}, PCC={attn_pcc:.6f}, cos={attn_cos:.6f}")
    logger.info(f"  CPU norm={cpu_decoded.norm():.4f}, norm ratio={tt_decoded.norm() / cpu_decoded.norm():.4f}")

    # Step 3: DAC decoder with both outputs
    with torch.no_grad():
        tt_wav = ref_decoder.wav_decoder(tt_decoded.transpose(1, 2).float())
    wav_pcc = pcc(cpu_wav, tt_wav)
    wav_cos = cos_sim(cpu_wav, tt_wav)
    logger.info(f"TTNN->DAC waveform: norm={tt_wav.norm():.4f}, PCC={wav_pcc:.6f}, cos={wav_cos:.6f}")
    logger.info(f"  CPU wav norm={cpu_wav.norm():.4f}, norm ratio={tt_wav.norm() / cpu_wav.norm():.4f}")

    # ========== Full TTNN decoder_forward path ==========
    wav_full = decoder_forward(
        encoded_expanded,
        token_masks,
        parameters.decoder_proj.weight,
        parameters.decoder_proj.bias,
        parameters.local_attention_decoder,
        ref_decoder.wav_decoder,
        device=mesh_device,
        input_mesh_mapper=input_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )
    full_pcc = pcc(cpu_wav, wav_full)
    full_cos = cos_sim(cpu_wav, wav_full)
    logger.info(f"\nFull decoder_forward: norm={wav_full.norm():.4f}, PCC={full_pcc:.6f}, cos={full_cos:.6f}")
    logger.info(f"  CPU wav norm={cpu_wav.norm():.4f}, norm ratio={wav_full.norm() / cpu_wav.norm():.4f}")

    # ========== Per-layer analysis ==========
    logger.info("\n" + "=" * 80)
    logger.info("PER-LAYER ANALYSIS")
    logger.info("=" * 80)

    # Run through layers one at a time on CPU and TT, comparing after each
    with torch.no_grad():
        cpu_x = cpu_proj.clone()
        tt_x = tt_proj.clone()

        # Check input_proj (identity for decoder where d_input == d_model)
        try:
            cpu_x_proj = ref_decoder.local_attention_decoder.input_proj(cpu_x)
            logger.info(f"Input proj exists, CPU norm: {cpu_x_proj.norm():.4f}")
            cpu_x = cpu_x_proj
        except Exception:
            logger.info("No input_proj (identity)")

        for i, layer in enumerate(ref_decoder.local_attention_decoder.layers):
            # CPU layer
            cpu_x_prev = cpu_x.clone()
            cpu_x = layer(cpu_x, mask=attn_mask)

            # TT layer - we need to trace through the TTNN layers
            logger.info(f"  Layer {i}: CPU norm={cpu_x.norm():.4f}, " f"delta_norm={(cpu_x - cpu_x_prev).norm():.4f}")

        # CPU final norm
        cpu_final = ref_decoder.local_attention_decoder.final_norm(cpu_x)
        logger.info(f"  Final norm: CPU={cpu_final.norm():.4f}")
        final_pcc = pcc(cpu_final, cpu_decoded)
        logger.info(f"  Verify CPU final == cpu_decoded: PCC={final_pcc:.6f}")

    # ========== Summary ==========
    logger.info("\n" + "=" * 80)
    logger.info("DECODER DIAGNOSIS SUMMARY")
    logger.info("=" * 80)
    logger.info(
        f"  decoder_proj:    PCC={proj_pcc:.6f}  cos={proj_cos:.6f}  norm_ratio={tt_proj.norm()/cpu_proj.norm():.4f}"
    )
    logger.info(
        f"  local_attn:      PCC={attn_pcc:.6f}  cos={attn_cos:.6f}  norm_ratio={tt_decoded.norm()/cpu_decoded.norm():.4f}"
    )
    logger.info(
        f"  DAC waveform:    PCC={wav_pcc:.6f}  cos={wav_cos:.6f}  norm_ratio={tt_wav.norm()/cpu_wav.norm():.4f}"
    )
    logger.info(
        f"  full pipeline:   PCC={full_pcc:.6f}  cos={full_cos:.6f}  norm_ratio={wav_full.norm()/cpu_wav.norm():.4f}"
    )
    logger.info("=" * 80)

    # Quality thresholds
    assert proj_pcc > 0.99, f"decoder_proj PCC too low: {proj_pcc}"
