# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device parity for the DiffVAE deterministic stages, against upstream on real weights.

Ground truth comes from ``capture_stages.py``, which drives upstream's own decoder one stage
at a time with the shipped checkpoint and injected noise, so these compare against real
activations rather than a synthetic-weight mirror. Per-block targets matter here: a 24-block
network compared only at its output tells you something is wrong but never where.

Generate the capture first (host only, no device):

  PYTHONPATH=/tmp/LTX-2/packages/ltx-core/src:. python capture_stages.py \
      latents/latent_0_1x128x4x34x60.pt --crop 10 --out stages/crop10.safetensors
"""

import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

import ttnn
from models.tt_dit.layers.na3d import build_device_plan, plan_na3d
from models.tt_dit.models.vae import diffvae_ltx
from models.tt_dit.models.vae.diffvae_ltx import (
    DeterministicStages,
    NABlock,
    decoder_config,
    default_rope_dim_split,
    rope_tables,
)
from models.tt_dit.utils.check import assert_quality

CAPTURE = Path(
    os.environ.get(
        "DIFFVAE_CAPTURE",
        os.path.expanduser("~/ltx25_diffvae/stages/crop10.safetensors"),
    )
)
CHECKPOINT = Path(
    os.environ.get(
        "DIFFVAE_CHECKPOINT",
        os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors"),
    )
)

# Stage 0 of the shipped config: 2048 channels, kernel (3,7,7), 4 blocks.
STAGE_DIM = 2048
STAGE_KERNEL = (3, 7, 7)


def _block_weights(stage: int, block: int) -> dict[str, torch.Tensor]:
    prefix = f"decoder.det_stages.{stage}.{block}."
    with safe_open(str(CHECKPOINT), "pt") as handle:
        return {key[len(prefix) :]: handle.get_tensor(key).float() for key in handle.keys() if key.startswith(prefix)}


def _captured(names: tuple[str, ...]) -> tuple[torch.Tensor, ...]:
    with safe_open(str(CAPTURE), "pt") as handle:
        return tuple(handle.get_tensor(name).float() for name in names)


@pytest.mark.parametrize("block_index", [0, 1])
@pytest.mark.diffvae_gate
def test_na_block_matches_upstream(*, device, block_index):
    """One deterministic NA block, real weights, real activations."""
    if not CAPTURE.exists():
        pytest.skip(f"missing {CAPTURE}; run capture_stages.py first")
    if not CHECKPOINT.exists():
        pytest.skip(f"missing {CHECKPOINT}")

    source = "stage0.conv_in" if block_index == 0 else f"det0.block{block_index - 1}"
    hidden, expected = _captured((source, f"det0.block{block_index}"))
    _, t, h, w, dim = hidden.shape
    assert dim == STAGE_DIM, f"capture has dim {dim}, expected {STAGE_DIM}"

    block = NABlock(STAGE_DIM, STAGE_KERNEL, head_dim=64, mesh_device=device)
    block.load_state_dict(_block_weights(0, block_index))

    tokens = t * h * w
    tt_hidden = ttnn.from_torch(
        hidden.reshape(tokens, dim), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    cos, sin = rope_tables((t, h, w), default_rope_dim_split(64), mesh_device=device)
    plan = build_device_plan(plan_na3d((t, h, w), STAGE_KERNEL), mesh_device=device)

    actual = block(tt_hidden, dims=(t, h, w), cos=cos, sin=sin, device_plan=plan)

    actual = ttnn.to_torch(actual).reshape(1, t, h, w, dim)

    assert_quality(expected, actual, pcc=0.99)


@pytest.mark.diffvae_gate
def test_row_chunking_is_exact(*, device, monkeypatch):
    """Chunking the pointwise parts changes nothing about the result.

    A block's peak is set by its SwiGLU, whose hidden width is 4x the activation, so at 6s
    1920x1088 its three intermediates come to 30 GiB. Running those in row chunks is only sound
    because they are pointwise in the site axis, and that is what this pins: same weights, same
    input, chunked against whole, bit for bit. Sizes here are far below :data:`CHUNK_BYTES`, so
    the budget is shrunk to force the loop, with a row count that leaves a short final chunk.
    """
    torch.manual_seed(0)
    dim, head_dim, kernel = 128, 64, (3, 3, 3)
    hidden = (int(dim * 4.0) + 15) // 16 * 16
    dims = (5, 8, 7)
    tokens = dims[0] * dims[1] * dims[2]  # 280, not a multiple of TILE
    assert tokens % diffvae_ltx.TILE != 0, "want a ragged final chunk"

    weights = {
        "norm1.weight": (dim,),
        "norm2.weight": (dim,),
        "attn.qkv.weight": (3 * dim, dim),
        "attn.qkv.bias": (3 * dim,),
        "attn.proj.weight": (dim, dim),
        "attn.proj.bias": (dim,),
        "attn.q_norm.weight": (head_dim,),
        "attn.k_norm.weight": (head_dim,),
        "mlp.w_gate.weight": (hidden, dim),
        "mlp.w_up.weight": (hidden, dim),
        "mlp.w_down.weight": (dim, hidden),
    }
    state = {name: torch.randn(shape) * 0.1 for name, shape in weights.items()}
    hidden_states = torch.randn(tokens, dim)

    cos, sin = rope_tables(dims, default_rope_dim_split(head_dim), mesh_device=device)
    plan = build_device_plan(plan_na3d(dims, kernel), mesh_device=device)

    def run() -> torch.Tensor:
        block = NABlock(dim, kernel, head_dim=head_dim, mesh_device=device)
        block.load_state_dict({key: value.clone() for key, value in state.items()})
        tt_hidden = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return ttnn.to_torch(block(tt_hidden, dims=dims, cos=cos, sin=sin, device_plan=plan))

    whole = run()
    monkeypatch.setattr(diffvae_ltx, "CHUNK_BYTES", 2 * diffvae_ltx.TILE * hidden * 2)
    chunked = run()

    assert torch.equal(whole, chunked), (whole - chunked).abs().max()


@pytest.mark.diffvae_gate
def test_det_stages_match_upstream(*, device):
    """Stages 1-3 end to end: 14 blocks, three kernels, three upsample strides.

    Starts from the normalized latent rather than ``conv_in``'s output, so the per-channel
    statistics folded into ``conv_in`` are exercised too. Stopping at ``det2.upsampled`` is
    what the capture offers — stage 4's boundary has upstream's trailing-ghost crop applied,
    which is a tiling concern handled at the pipeline level, not here.
    """
    if not CAPTURE.exists():
        pytest.skip(f"missing {CAPTURE}; run capture_stages.py first")
    if not CHECKPOINT.exists():
        pytest.skip(f"missing {CHECKPOINT}")

    latent, expected = _captured(("input.latent_padded", "det2.upsampled"))
    config = decoder_config(CHECKPOINT)

    stages = DeterministicStages(
        in_channels=config["in_channels"],
        stage_channels=config["stage_channels"],
        stage_depths=config["stage_depths"],
        stage_kernels=config["stage_kernels"],
        upsamples=config["upsamples"],
        head_dim=config["head_dim"],
        mesh_device=device,
    )
    stages.load_checkpoint(CHECKPOINT)

    # Channels-last, flattened to tokens: the decoder's own first move on the latent.
    _, channels, t, h, w = latent.shape
    tokens = latent.permute(0, 2, 3, 4, 1).reshape(t * h * w, channels)
    tt_latent = ttnn.from_torch(tokens, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    actual, dims = stages(tt_latent, dims=(t, h, w), stages=3)
    assert dims == tuple(expected.shape[1:4]), f"dims {dims} != capture {tuple(expected.shape[1:4])}"
    actual = ttnn.to_torch(actual).reshape(1, *dims, expected.shape[-1])

    assert_quality(expected, actual, pcc=0.99)
