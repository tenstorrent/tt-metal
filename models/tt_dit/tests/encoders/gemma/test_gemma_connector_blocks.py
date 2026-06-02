# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Isolated parity test for the video connector transformer blocks.

Feeds identical random features through the device video connector (register
replacement -> on-device RoPE -> blocks -> final norm) and the reference
Embeddings1DConnector, and asserts PCC. Isolating the connector from the Gemma
encoder keeps a regression here distinguishable from upstream bf16 drift.
"""

import glob
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[6]))

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.tt_dit.encoders.gemma.encoder_pair import _replace_padded_with_registers
from models.tt_dit.models.transformers.ltx.rope_ltx import reshape_interleaved_to_bhnd
from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline
from models.tt_dit.utils.check import assert_quality

VIDEO_PREFIX = "model.diffusion_model.video_embeddings_connector."
AGG_PREFIXES = ("text_embedding_projection.video_aggregate_embed.", "text_embedding_projection.audio_aggregate_embed.")


def _gemma_path():
    c = glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/*/")
    )
    return c[0].rstrip("/") if c else None


def _ltx_ckpt():
    c = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/*/ltx-2.3-22b-dev.safetensors"
        )
    )
    return c[0] if c else None


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_connector_blocks_isolated(*, mesh_device):
    gemma, ckpt = _gemma_path(), _ltx_ckpt()
    if not gemma or not ckpt:
        pytest.skip("assets missing")

    sys.path.insert(0, "LTX-2/packages/ltx-core/src")
    from ltx_core.model.transformer.rope import LTXRopeType, generate_freq_grid_pytorch, precompute_freqs_cis
    from ltx_core.text_encoders.gemma.embeddings_connector import Embeddings1DConnector

    # --- device connector (load video connector weights) ---
    pipe = LTXPipeline.create_pipeline(mesh_device, checkpoint_name=None, gemma_path=gemma, mode="av")
    conn_state = {}
    ref_sd = {}
    with safe_open(ckpt, "pt") as f:
        for k in f.keys():
            if k.startswith(VIDEO_PREFIX):
                conn_state[k] = f.get_tensor(k)
                ref_sd[k[len(VIDEO_PREFIX) :]] = f.get_tensor(k)  # keep bf16 (matches reference)
            elif k.startswith(AGG_PREFIXES):
                conn_state[k] = f.get_tensor(k)
    pipe.gemma_encoder_pair.load_embeddings_connectors(conn_state, audio_num_blocks=8)

    # --- reference connector (CPU) ---
    ref = (
        Embeddings1DConnector(
            attention_head_dim=128,
            num_attention_heads=32,
            num_layers=8,
            positional_embedding_theta=10000.0,
            positional_embedding_max_pos=[1],
            # Checkpoint is SPLIT-rotation; the device path mirrors it via load-time _permute_qk
            # + interleaved rotary_embedding_llama (equivalent). Reference must be SPLIT to match.
            rope_type=LTXRopeType.SPLIT,
            double_precision_rope=False,
            apply_gated_attention=True,
            num_learnable_registers=128,
        )
        .float()
        .eval()
    )
    missing, unexpected = ref.load_state_dict(ref_sd, strict=False)
    logger.info(f"ref connector load: missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")

    # --- identical random input WITH PADDING (left-pad) → register replacement active ---
    # This mirrors the real encode path: only n_real real tokens, the rest become registers.
    seq = 256
    dim = 4096
    n_real = 17
    torch.manual_seed(0)
    x = (torch.randn(1, seq, dim) * 0.5).bfloat16()
    binary_mask = torch.zeros(1, seq, dtype=torch.long)
    binary_mask[:, seq - n_real :] = 1  # left-pad: real tokens at the end
    additive_mask = torch.where(binary_mask[:, None, None, :].bool(), 0.0, -1e9)  # (1,1,1,seq)

    with torch.no_grad():
        ref_out = ref(x.float().clone(), additive_mask)[0].float()

    # device: replicate _run_connector — register replacement -> rope -> blocks -> final norm
    registers = ttnn.to_torch(
        ttnn.get_device_tensors(pipe.gemma_encoder_pair.video_connector.learnable_registers.data)[0]
    ).float()
    x_replaced = _replace_padded_with_registers(x.float(), binary_mask, registers, 128).bfloat16()

    # Device blocks now take head-split (BHND) interleaved cos/sin + a trans_mat and run the
    # on-device rotary_embedding_llama kernel (Q/K permuted at load). Mirror _run_connector.
    rope_cos, rope_sin = precompute_freqs_cis(
        torch.arange(seq, dtype=torch.float32)[None, None, :],
        dim=dim,
        out_dtype=torch.float32,
        theta=10000.0,
        max_pos=[1],
        num_attention_heads=32,
        rope_type=LTXRopeType.INTERLEAVED,
        freq_grid_generator=generate_freq_grid_pytorch,
    )  # (1, seq, dim)
    rope_cos = ttnn.from_torch(
        reshape_interleaved_to_bhnd(rope_cos, 32).bfloat16(),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    rope_sin = ttnn.from_torch(
        reshape_interleaved_to_bhnd(rope_sin, 32).bfloat16(),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    trans_mat = pipe._prepare_trans_mat()
    tt_x = ttnn.from_torch(x_replaced, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    for block in pipe.gemma_encoder_pair.video_connector.transformer_1d_blocks:
        tt_x = block(tt_x, rope_cos=rope_cos, rope_sin=rope_sin, trans_mat=trans_mat)
    tt_x = ttnn.experimental.dit_rms_norm_unary_fused(
        tt_x, weight=None, epsilon=1e-6, compute_kernel_config=pipe.gemma_encoder_pair.video_connector.rmsnorm_cc
    )
    dev_out = ttnn.to_torch(ttnn.get_device_tensors(tt_x)[0]).float()

    logger.info(f"CONNECTOR (with registers)  ref={tuple(ref_out.shape)} dev={tuple(dev_out.shape)}")
    assert_quality(ref_out, dev_out, pcc=0.99)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])
