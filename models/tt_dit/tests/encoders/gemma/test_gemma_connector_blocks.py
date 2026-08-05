# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Isolated parity: device video connector vs diffusers ``LTX2ConnectorTransformer1d``
on identical random features, asserting PCC. Isolating from the Gemma encoder keeps a
regression here distinguishable from upstream bf16 drift.
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
from models.tt_dit.encoders.gemma.embeddings_connector import EmbeddingsConnector
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.mochi import get_rot_transformation_mat
from models.tt_dit.utils.tensor import bf16_tensor

VIDEO_PREFIX = "model.diffusion_model.video_embeddings_connector."


def _raw_video_connector_to_diffusers(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Raw Lightricks video-connector weights → diffusers LTX2ConnectorTransformer1d keys
    (transformer_1d_blocks → transformer_blocks, attn1.q_norm/k_norm → attn1.norm_q/norm_k)."""
    out: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        sub = k[len(VIDEO_PREFIX) :]
        sub = sub.replace("transformer_1d_blocks.", "transformer_blocks.")
        sub = sub.replace(".attn1.q_norm.", ".attn1.norm_q.").replace(".attn1.k_norm.", ".attn1.norm_k.")
        out[sub] = v
    return out


def _ltx_ckpt() -> str | None:
    explicit = os.environ.get("LTX_CHECKPOINT")
    if explicit and os.path.exists(explicit):
        return explicit
    cands = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/*/ltx-2.3-22b-dev.safetensors"
        )
    )
    return cands[0] if cands else None


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_connector_blocks_isolated(*, mesh_device):
    ckpt = _ltx_ckpt()
    if not ckpt:
        pytest.skip("LTX checkpoint not found")

    pytest.importorskip("diffusers")
    from diffusers.pipelines.ltx2.connectors import LTX2ConnectorTransformer1d

    # --- device connector built directly (no pipeline): TP=1 on 1x1, weights from raw ckpt ---
    enc_parallel = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=1))
    enc_ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    connector = EmbeddingsConnector(
        output_dim=4096,
        num_blocks=8,
        num_heads=32,
        mesh_device=mesh_device,
        ccl_manager=enc_ccl,
        parallel_config=enc_parallel,
    )
    raw_video = {}
    with safe_open(ckpt, "pt") as f:
        for k in f.keys():
            if k.startswith(VIDEO_PREFIX):
                raw_video[k] = f.get_tensor(k)
    # The TT loader strips the prefix, renames blocks, and permutes Q/K split→interleaved.
    connector.load_torch_state_dict({k[len(VIDEO_PREFIX) :]: v for k, v in raw_video.items()}, strict=False)

    # --- reference: diffusers video connector (SPLIT rope = checkpoint's rotation, CPU fp32) ---
    ref = (
        LTX2ConnectorTransformer1d(
            num_attention_heads=32,
            attention_head_dim=128,
            num_layers=8,
            num_learnable_registers=128,
            rope_base_seq_len=4096,
            rope_theta=10000.0,
            rope_double_precision=False,
            rope_type="split",
            gated_attention=True,
        )
        .float()
        .eval()
    )
    inc = ref.load_state_dict(_raw_video_connector_to_diffusers(raw_video), strict=False)
    logger.info(f"ref connector load: missing={len(inc.missing_keys)} unexpected={len(inc.unexpected_keys)}")

    # --- identical random input WITH PADDING (left-pad) → register replacement active ---
    # Mirrors the real encode path: only n_real real tokens, the rest become registers.
    seq, dim, n_real = 256, 4096, 17
    torch.manual_seed(0)
    x = (torch.randn(1, seq, dim) * 0.5).bfloat16()
    binary_mask = torch.zeros(1, seq, dtype=torch.long)
    binary_mask[:, seq - n_real :] = 1  # left-pad: real tokens at the end

    # reference: diffusers binarizes the additive mask at >= -9000.
    add_mask = torch.where(binary_mask[:, None, None, :].bool(), 0.0, -1e9)  # (1,1,1,seq)
    with torch.no_grad():
        ref_out = ref(x.float().clone(), add_mask)[0].float()

    # device: connector forward (register replacement → on-device RoPE → blocks → norm).
    tt_feat = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)
    dev_out = connector(tt_feat, binary_mask, trans_mat=trans_mat)

    logger.info(f"CONNECTOR (with registers)  ref={tuple(ref_out.shape)} dev={tuple(dev_out.shape)}")
    assert_quality(ref_out, dev_out, pcc=0.998)
