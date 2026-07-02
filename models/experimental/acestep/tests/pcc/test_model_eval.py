# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model-level evaluation: AceStepModelConfig + create_tt_model over an eval dataset.

Unlike the per-module PCC tests, this exercises the public factory contract:

    args  = AceStepModelConfig.from_hf(...)
    model = create_tt_model(args, device)
    out   = model.forward(...)

against the genuine HF AceStepDiTModel, over a small evaluation dataset of denoise-step inputs
(varied timesteps, sequence lengths, and random seeds — the distribution the flow-matching loop
actually feeds the DiT). Reports mean/min PCC across the dataset. Requires model.safetensors.
"""

import pytest
import torch

import ttnn
from loguru import logger
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.reference.weight_utils import checkpoint_path, load_module_weights
from models.experimental.acestep.tt.model_config import AceStepModelConfig, create_tt_model
from models.common.utility_functions import comp_pcc
from models.experimental.acestep.tests.test_utils import (
    HEAD_DIM,
    require_single_device,
    to_torch,
    to_ttnn_tensor,
)

IN_CHANNELS = 192
OUT_CHANNELS = 64
PATCH = 2
HIDDEN_CH = 64
NUM_DIT_LAYERS = 24  # full model

# Evaluation dataset: (seq_len, enc_len, seed) — varied denoise-step conditions.
EVAL_DATASET = [
    (128, 96, 0),
    (256, 128, 1),
    (256, 64, 2),
    (512, 160, 3),
]
# Per-sample timesteps sampled deterministically from the seed (flow-matching t in (0,1)).


def _have_checkpoint():
    try:
        checkpoint_path()
        return True
    except AssertionError:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _have_checkpoint(), reason="model.safetensors not downloaded")
def test_model_eval_dataset(device):
    require_single_device(device)

    # Public factory contract.
    args = AceStepModelConfig.from_hf(num_hidden_layers=NUM_DIT_LAYERS)
    model = create_tt_model(args, device)

    # Reference DiT with the same real weights.
    m = load_modeling_module()
    hf = load_config()
    hf._attn_implementation = "eager"
    hf.num_hidden_layers = NUM_DIT_LAYERS
    ref = m.AceStepDiTModel(hf).eval()
    load_module_weights(ref, "decoder.", allow_extra=True)

    rope = Qwen3RotaryEmbedding(hf)

    pccs = []
    for seq_len, enc_len, seed in EVAL_DATASET:
        torch.manual_seed(seed)
        hidden = torch.randn(1, seq_len, HIDDEN_CH)
        context = torch.randn(1, seq_len, IN_CHANNELS - HIDDEN_CH)
        encoder = torch.randn(1, enc_len, args.hidden_size)
        t = torch.rand(1)
        t_r = torch.rand(1)

        with torch.no_grad():
            (ref_out, *_) = ref(
                hidden_states=hidden,
                timestep=t,
                timestep_r=t_r,
                attention_mask=None,
                encoder_hidden_states=encoder,
                encoder_attention_mask=None,
                context_latents=context,
            )

        tprime = seq_len // PATCH
        position_ids = torch.arange(tprime).unsqueeze(0)
        cos, sin = rope(torch.zeros(1, tprime, HEAD_DIM), position_ids)
        sliding = None
        if tprime > args.sliding_window:
            mk = m.create_4d_mask(
                seq_len=tprime,
                dtype=torch.float32,
                device=hidden.device,
                attention_mask=None,
                sliding_window=args.sliding_window,
                is_sliding_window=True,
                is_causal=False,
            )
            sliding = ttnn.from_torch(mk, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        hidden_tt = to_ttnn_tensor(hidden.reshape(1, 1, seq_len, HIDDEN_CH), device)
        context_tt = to_ttnn_tensor(context.reshape(1, 1, seq_len, IN_CHANNELS - HIDDEN_CH), device)
        encoder_tt = to_ttnn_tensor(encoder.reshape(1, 1, enc_len, args.hidden_size), device)
        cos_tt = ttnn.from_torch(cos.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        sin_tt = ttnn.from_torch(sin.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        out_tt = model.forward(hidden_tt, context_tt, t, t_r, cos_tt, sin_tt, encoder_tt, sliding_mask=sliding)
        out = to_torch(out_tt, expected_shape=(1, 1, seq_len, OUT_CHANNELS)).reshape(1, seq_len, OUT_CHANNELS)

        # comp_pcc returns (passing, message) where message is the PCC float (or a text reason).
        _, pcc_msg = comp_pcc(ref_out, out, 0.95)
        try:
            pcc_val = float(pcc_msg)
        except (TypeError, ValueError):
            pcc_val = float(str(pcc_msg).split()[-1])
        pccs.append(pcc_val)
        logger.info(f"eval sample seq={seq_len} enc={enc_len} seed={seed}: PCC={pcc_val:.4f}")

    mean_pcc = sum(pccs) / len(pccs)
    min_pcc = min(pccs)
    logger.info(f"EVAL DATASET: n={len(pccs)} mean_pcc={mean_pcc:.4f} min_pcc={min_pcc:.4f}")

    # Every sample must clear the e2e requirement.
    assert min_pcc >= 0.95, f"eval dataset min PCC {min_pcc:.4f} < 0.95 (per-sample: {[f'{p:.4f}' for p in pccs]})"
