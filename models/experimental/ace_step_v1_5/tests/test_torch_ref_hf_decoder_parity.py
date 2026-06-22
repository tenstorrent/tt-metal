# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Compare torch_ref DiT core vs official HF decoder forward (when checkpoint available)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from models.experimental.ace_step_v1_5.demo.ref_decoder_compare import (
    hf_decoder_velocity,
    load_hf_decoder_from_checkpoint_dir,
    pearson_pcc,
)
from models.experimental.ace_step_v1_5.torch_ref.full_pipeline import AceStepV15TorchPipeline
from models.experimental.ace_step_v1_5.torch_ref.hf_generate import ensure_hf_modeling_ready

_TURBO_CKPT = Path(os.environ.get("ACESTEP_CHECKPOINTS_DIR", "/home/iguser/tt-ign/ACE-Step-1.5/checkpoints"))
_TURBO_DIR = _TURBO_CKPT / "acestep-v15-turbo"
_HF_REPO = Path("/home/iguser/tt-ign/ACE-Step-1.5")

PCC_THRESHOLD = 0.98


def _has_turbo_checkpoint() -> bool:
    return (_TURBO_DIR / "model.safetensors").is_file() and _HF_REPO.is_dir()


@pytest.mark.skipif(not _has_turbo_checkpoint(), reason="ACE-Step turbo checkpoint or HF repo not present")
def test_hf_decoder_matches_torch_ref_single_forward():
    """HF ``AceStepDiTModel`` vs ``AceStepV15TorchPipeline`` on one denoise forward."""
    ref_root = ensure_hf_modeling_ready(ckpt_dir=str(_TURBO_CKPT), ace_step_repo_root=str(_HF_REPO))
    assert ref_root is not None

    safetensors_path = _TURBO_DIR / "model.safetensors"
    timesteps_host = np.array([1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0], dtype=np.float32)

    torch.manual_seed(0)
    frames = 128
    B = 1
    xt = torch.randn(B, frames, 64, dtype=torch.float32)
    ctx = torch.randn(B, frames, 128, dtype=torch.float32)

    hf_dec = load_hf_decoder_from_checkpoint_dir(_TURBO_DIR, ref_repo_root=ref_root, torch_dtype=torch.float32)
    cond_dim = int(hf_dec.condition_embedder.weight.shape[1])
    enc = torch.randn(B, 64, cond_dim, dtype=torch.float32)

    vt_hf = hf_decoder_velocity(
        hf_dec,
        xt=xt,
        context_latents=ctx,
        encoder_hidden_states=enc,
        t_curr=1.0,
        device="cpu",
        dtype=torch.float32,
    )

    pipe = AceStepV15TorchPipeline(
        checkpoint_safetensors_path=str(safetensors_path),
        timesteps_host=timesteps_host,
        device="cpu",
        dtype=torch.float32,
    )
    with torch.inference_mode():
        vt_ref = (
            pipe.forward(
                xt_bt64=xt,
                context_latents_bt128=ctx,
                timestep_index=0,
                encoder_hidden_states_btd=enc,
            )
            .float()
            .cpu()
        )

    pcc = pearson_pcc(vt_hf.reshape(-1), vt_ref.reshape(-1))
    assert pcc >= PCC_THRESHOLD, f"HF vs torch_ref decoder PCC {pcc:.4f} < {PCC_THRESHOLD}"
