# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# UNVALIDATED — this test was authored without Tenstorrent hardware and has NOT
# been run. It mirrors tests/unit/test_feedforward.py: it instantiates the
# OFFICIAL reference Ideogram4TransformerBlock with random init, loads its
# state_dict into the tt block, runs both on real Ideogram dims, and asserts
# PCC >= 0.99. Run it on a TT box via the verify.py command in the bringup report.
#
# pcc-debugger watch-list (highest-risk parts of this single-stream block):
#   1. rotate-half MRoPE (cos/sin = cat(freqs, freqs)). Precomputed on host here
#      from the reference Ideogram4MRoPE so the convention is identical; the risk
#      is the tt-side _apply_rope half-split matching exactly.
#   2. block-diagonal segment-id mask -> additive bias (0 / -inf) fed to SDPA.
#   3. tanh-gated 4-branch AdaLN (no shift terms).
#   4. double-RMSNorm sandwich residual (norm2 on the attn/ff *output*).
#   5. head_dim=256 / 18 heads — single-device (TP=1) is the validated config.
# =============================================================================

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.transformer_ideogram4 import Ideogram4TransformerBlock
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....reference.ideogram4 import modeling_ideogram4
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor

# Real Ideogram 4.0 denoiser dims (confirmed against reference Ideogram4Config).
EMB_DIM = 4608
NUM_HEADS = 18
HEAD_DIM = EMB_DIM // NUM_HEADS  # 256
INTERMEDIATE_SIZE = 12288
ADALN_DIM = 512
ROPE_THETA = 5_000_000
MROPE_SECTION = (24, 20, 20)
NORM_EPS = 1e-5

OUTPUT_IMAGE_INDICATOR = modeling_ideogram4.OUTPUT_IMAGE_INDICATOR
LLM_TOKEN_INDICATOR = modeling_ideogram4.LLM_TOKEN_INDICATOR
IMAGE_POSITION_OFFSET = 65536


def _build_inputs(batch_size: int, text_len: int, image_len: int, torch_dtype):
    """Construct a realistic single-stream batch: [text tokens | image tokens].

    Returns torch tensors mirroring the reference block's forward signature plus
    the derived cos/sin (B, L, head_dim) and a (B, 1, L, L) additive attn mask.
    """
    seq_len = text_len + image_len

    # hidden states (already-fused h = x + llm_features + indicator embed upstream).
    x = torch.randn(batch_size, seq_len, EMB_DIM, dtype=torch_dtype)

    # adaln_input: reference passes F.silu(adaln_proj(t_cond)); per-sample => (B,1,adaln_dim).
    adaln_input = torch.randn(batch_size, 1, ADALN_DIM, dtype=torch_dtype)

    # segment ids: one packed sample => all-same segment (full attention). Use a
    # second segment on part of the batch>1 case to exercise the block-diagonal mask.
    segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

    # position_ids (B, L, 3) = (t, h, w). Text tokens: 1D positions on all 3 axes.
    # Image tokens: a (h, w) grid offset so they never collide with text positions.
    position_ids = torch.zeros(batch_size, seq_len, 3, dtype=torch.long)
    # text: linear positions 0..text_len-1 on every axis.
    text_pos = torch.arange(text_len)
    position_ids[:, :text_len, 0] = text_pos
    position_ids[:, :text_len, 1] = text_pos
    position_ids[:, :text_len, 2] = text_pos
    # image: square-ish grid.
    grid_h = int(round(image_len**0.5))
    while image_len % grid_h != 0:
        grid_h -= 1
    grid_w = image_len // grid_h
    hh = torch.arange(grid_h).repeat_interleave(grid_w)
    ww = torch.arange(grid_w).repeat(grid_h)
    position_ids[:, text_len:, 0] = IMAGE_POSITION_OFFSET
    position_ids[:, text_len:, 1] = IMAGE_POSITION_OFFSET + hh
    position_ids[:, text_len:, 2] = IMAGE_POSITION_OFFSET + ww

    # cos/sin from the reference MRoPE (ground-truth convention).
    rope = modeling_ideogram4.Ideogram4MRoPE(
        head_dim=HEAD_DIM, base=ROPE_THETA, mrope_section=MROPE_SECTION
    )
    cos, sin = rope(position_ids)  # (B, L, head_dim), float32
    cos = cos.to(torch_dtype)
    sin = sin.to(torch_dtype)

    # additive attention bias from segment ids: (B, 1, L, L), 0 attend / -inf block.
    same_seg = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)  # (B, L, L)
    attn_bias = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.float32)
    attn_bias.masked_fill_(~same_seg.unsqueeze(1), float("-inf"))

    return x, adaln_input, segment_ids, cos, sin, attn_bias


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("batch_size", "text_len", "image_len"),
    [
        # (text tokens, image tokens). Small text + a moderate image grid keeps a
        # single device within memory while exercising the full block numerics
        # (block math is sequence-length independent).
        pytest.param(1, 64, 1024, id="text64_img1024"),
        pytest.param(1, 128, 4096, id="text128_img4096_1024px"),
    ],
)
def test_transformer_block(
    *,
    mesh_device: ttnn.MeshDevice,
    batch_size: int,
    text_len: int,
    image_len: int,
) -> None:
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16

    seq_len = text_len + image_len

    # ---- reference (ground truth), random init ----
    torch_block = modeling_ideogram4.Ideogram4TransformerBlock(
        hidden_size=EMB_DIM,
        intermediate_size=INTERMEDIATE_SIZE,
        num_heads=NUM_HEADS,
        norm_eps=NORM_EPS,
        adanln_dim=ADALN_DIM,
    ).to(dtype=torch_dtype)
    torch_block.eval()

    x, adaln_input, segment_ids, cos, sin, attn_bias = _build_inputs(
        batch_size, text_len, image_len, torch_dtype
    )

    with torch.no_grad():
        torch_out = torch_block(
            x,
            segment_ids=segment_ids,
            cos=cos,
            sin=sin,
            adaln_input=adaln_input,
        )

    # ---- tt block ----
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=1, mesh_axis=1),
        sequence_parallel=ParallelFactor(factor=1, mesh_axis=0),
    )
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)

    tt_block = Ideogram4TransformerBlock(
        hidden_size=EMB_DIM,
        intermediate_size=INTERMEDIATE_SIZE,
        num_heads=NUM_HEADS,
        norm_eps=NORM_EPS,
        adaln_dim=ADALN_DIM,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=None,
    )
    tt_block.load_torch_state_dict(torch_block.state_dict())

    # tt tensors. cos/sin -> (B, 1, L, head_dim) to broadcast over heads in _apply_rope.
    tt_x = bf16_tensor(x, device=mesh_device)
    tt_adaln = bf16_tensor(adaln_input, device=mesh_device)
    tt_cos = bf16_tensor(cos.unsqueeze(1), device=mesh_device)
    tt_sin = bf16_tensor(sin.unsqueeze(1), device=mesh_device)

    # Additive attn mask. For a single packed segment the mask is all-zero (full
    # attention) so we pass None to let SDPA skip the mask entirely. When segments
    # differ, build the bias tensor on device.
    if torch.isneginf(attn_bias).any():
        tt_attn_mask = bf16_tensor(attn_bias, device=mesh_device)
    else:
        tt_attn_mask = None

    tt_out = tt_block(
        tt_x,
        cos=tt_cos,
        sin=tt_sin,
        adaln_input=tt_adaln,
        attn_mask=tt_attn_mask,
    )

    tt_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])

    logger.info(f"ideogram4 block: B={batch_size} text={text_len} image={image_len} seq={seq_len}")
    assert_quality(torch_out, tt_out_torch, pcc=0.99)
