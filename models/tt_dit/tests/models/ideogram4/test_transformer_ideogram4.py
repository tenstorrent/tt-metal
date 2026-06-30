# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# VALIDATED on Blackhole. Instantiates the OFFICIAL reference
# Ideogram4TransformerBlock with random init, loads its state_dict into the tt
# block, runs both on real Ideogram dims, and asserts PCC >= 0.99.
#
# Coverage (2x4 BH mesh, FABRIC_1D, Linear): {B=1,2} x {1,2 segments} x
# {1088, 4224 seq} x {tp1sp1, tp2, tp4(pad 20), sp2, sp2tp2, sp2tp4}.
#
#   1. rotate-half MRoPE (cos/sin = cat(freqs, freqs)). Precomputed on host here
#      from the reference Ideogram4MRoPE so the convention is identical.
#   2. block-diagonal segment-id mask -> additive bias (0 / -inf) fed to SDPA;
#      with SP it is sharded on the query-row dim (K/V are all-gathered).
#   3. tanh-gated 4-branch AdaLN (no shift terms).
#   4. double-RMSNorm sandwich residual (norm2 on the attn/ff *output*).
#   5. head_dim=256 / 18 heads — TP pads heads up to a multiple of tp_factor
#      (18 -> 20 for tp=4); SP sharding uses ring attention (or K/V gather + mask).
# =============================================================================

import math

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.transformer_ideogram4 import Ideogram4TransformerBlock
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....reference.ideogram4 import modeling_ideogram4
from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.padding import PaddingConfig
from ....utils.tensor import bf16_tensor

# Ring SDPA (sequence parallel) requires the padded sequence to be divisible by
# k_chunk_size * sp_factor. Must match Ideogram4TransformerBlock.sdpa_k_chunk_size.
SDPA_K_CHUNK = 256


def _sp_padded_len(seq_len: int, sp_factor: int) -> int:
    if sp_factor == 1:
        return seq_len
    divisor = SDPA_K_CHUNK * sp_factor
    return math.ceil(seq_len / divisor) * divisor


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


def _build_inputs(batch_size: int, text_len: int, image_len: int, torch_dtype, num_segments: int = 1):
    """Construct a realistic single-stream batch: [text tokens | image tokens].

    Returns torch tensors mirroring the reference block's forward signature plus
    the derived cos/sin (B, L, head_dim) and a (B, 1, L, L) additive attn mask.

    num_segments > 1 packs that many independent samples into the sequence (contiguous,
    roughly equal spans) so the block-diagonal segment mask blocks cross-segment
    attention — exercising the masked SDPA path.
    """
    seq_len = text_len + image_len

    # hidden states (already-fused h = x + llm_features + indicator embed upstream).
    x = torch.randn(batch_size, seq_len, EMB_DIM, dtype=torch_dtype)

    # adaln_input: reference passes F.silu(adaln_proj(t_cond)); per-sample => (B,1,adaln_dim).
    adaln_input = torch.randn(batch_size, 1, ADALN_DIM, dtype=torch_dtype)

    # segment ids: contiguous spans => block-diagonal attention. num_segments==1 is the
    # all-same-segment (full attention) case.
    segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    if num_segments > 1:
        bounds = torch.linspace(0, seq_len, num_segments + 1).round().long()
        for s in range(num_segments):
            segment_ids[:, bounds[s] : bounds[s + 1]] = s

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
    rope = modeling_ideogram4.Ideogram4MRoPE(head_dim=HEAD_DIM, base=ROPE_THETA, mrope_section=MROPE_SECTION)
    cos, sin = rope(position_ids)  # (B, L, head_dim), float32
    cos = cos.to(torch_dtype)
    sin = sin.to(torch_dtype)

    # additive attention bias from segment ids: (B, 1, L, L), 0 attend / -inf block.
    same_seg = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)  # (B, L, L)
    attn_bias = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.float32)
    attn_bias.masked_fill_(~same_seg.unsqueeze(1), float("-inf"))

    return x, adaln_input, segment_ids, cos, sin, attn_bias


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links"),
    [
        # factor == submesh axis size. TP shards heads (padded to a multiple of tp);
        # SP shards the sequence (ring attention). Full mesh is the 2x4 BH loudbox.
        # factor == submesh axis size; sp on axis 0 (size up to 2), tp on axis 1 (size up to 4).
        pytest.param((2, 4), (1, 1), 0, 1, 1, id="tp1sp1"),  # regression: single device
        pytest.param((2, 4), (1, 2), 0, 1, 1, id="tp2"),  # 18 heads % 2 == 0, no padding
        pytest.param((2, 4), (1, 4), 0, 1, 1, id="tp4_pad20"),  # 18 -> 20 heads
        pytest.param((2, 4), (2, 1), 0, 1, 1, id="sp2"),  # sequence parallel only (TP=1)
        pytest.param((2, 4), (2, 2), 0, 1, 1, id="sp2tp2"),
        pytest.param((2, 4), (2, 4), 0, 1, 1, id="sp2tp4_pad20"),  # SP=2, TP=4 (pad 20)
        pytest.param((4, 2), (4, 2), 0, 1, 1, id="sp4tp2"),  # SP=4, TP=2 (full mesh, 4x2 arrangement)
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("batch_size", "text_len", "image_len"),
    [
        pytest.param(1, 64, 1024, id="b1_text64_img1024"),
        pytest.param(2, 64, 1024, id="b2_text64_img1024"),  # batch > 1
        pytest.param(1, 128, 4096, id="b1_text128_img4096"),  # larger sequence (4224)
    ],
)
@pytest.mark.parametrize("num_segments", [1, 2], ids=["1seg", "2seg"])
def test_transformer_block(
    *,
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    batch_size: int,
    text_len: int,
    image_len: int,
    num_segments: int,
) -> None:
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16

    seq_len = text_len + image_len

    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    sp_factor = tuple(submesh_device.shape)[sp_axis]
    tp_factor = tuple(submesh_device.shape)[tp_axis]

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
        batch_size, text_len, image_len, torch_dtype, num_segments
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
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )
    ccl_manager = CCLManager(submesh_device, num_links=num_links, topology=ttnn.Topology.Linear)

    # Head padding for TP divisibility (18 heads -> next multiple of tp_factor).
    padding_config = (
        PaddingConfig.from_tensor_parallel_factor(NUM_HEADS, HEAD_DIM, tp_factor)
        if NUM_HEADS % tp_factor != 0
        else None
    )

    tt_block = Ideogram4TransformerBlock(
        hidden_size=EMB_DIM,
        intermediate_size=INTERMEDIATE_SIZE,
        num_heads=NUM_HEADS,
        norm_eps=NORM_EPS,
        adaln_dim=ADALN_DIM,
        mesh_device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )
    tt_block.load_torch_state_dict(torch_block.state_dict())

    # Sequence-parallel sharding: pad the sequence to k_chunk*sp, shard x and the
    # cos/sin rope tables on the sequence dim across the SP axis. The hidden dim stays
    # replicated (TP only shards inside the matmuls), so a single-axis shard suffices.
    padded_len = _sp_padded_len(seq_len, sp_factor)
    cos4 = cos.unsqueeze(1)  # (B, 1, L, head_dim) to broadcast over heads in _apply_rope
    sin4 = sin.unsqueeze(1)
    if sp_factor > 1:
        x = torch.nn.functional.pad(x, (0, 0, 0, padded_len - seq_len))
        cos4 = torch.nn.functional.pad(cos4, (0, 0, 0, padded_len - seq_len))
        sin4 = torch.nn.functional.pad(sin4, (0, 0, 0, padded_len - seq_len))
        tt_x = bf16_tensor(x, device=submesh_device, mesh_axis=sp_axis, shard_dim=1)
        tt_cos = bf16_tensor(cos4, device=submesh_device, mesh_axis=sp_axis, shard_dim=2)
        tt_sin = bf16_tensor(sin4, device=submesh_device, mesh_axis=sp_axis, shard_dim=2)
    else:
        tt_x = bf16_tensor(x, device=submesh_device)
        tt_cos = bf16_tensor(cos4, device=submesh_device)
        tt_sin = bf16_tensor(sin4, device=submesh_device)
    tt_adaln = bf16_tensor(adaln_input, device=submesh_device)

    # Block-diagonal segment mask (additive 0 / -inf). num_segments==1 => full attention.
    if num_segments == 1:
        tt_attn_mask = None
    elif sp_factor == 1:
        # Replicated [B, 1, L, L] mask fed to plain SDPA.
        assert torch.isneginf(attn_bias).any(), "multi-segment input should produce a non-trivial mask"
        tt_attn_mask = bf16_tensor(attn_bias, device=submesh_device)
    else:
        # SP + mask: the block all-gathers K/V to the full (padded) sequence and runs
        # masked SDPA with Q kept sequence-sharded, so the mask is [B, 1, padded_L, padded_L]
        # sharded on the query-row dim. Pad segment ids with a sentinel id so real queries
        # never attend to padded keys (and vice-versa).
        seg_padded = torch.full((batch_size, padded_len), num_segments, dtype=torch.long)
        seg_padded[:, :seq_len] = segment_ids
        same_seg = seg_padded.unsqueeze(2) == seg_padded.unsqueeze(1)  # (B, padded_L, padded_L)
        bias = torch.zeros(batch_size, 1, padded_len, padded_len, dtype=torch.float32)
        bias.masked_fill_(~same_seg.unsqueeze(1), float("-inf"))
        tt_attn_mask = bf16_tensor(bias, device=submesh_device, mesh_axis=sp_axis, shard_dim=2)

    tt_out = tt_block(
        tt_x,
        cos=tt_cos,
        sin=tt_sin,
        adaln_input=tt_adaln,
        attn_mask=tt_attn_mask,
        spatial_sequence_length=seq_len,
    )

    # Output is replicated on TP, sharded on SP -> concat the SP shards, take a TP replica.
    tt_out_torch = tensor.to_torch(tt_out, mesh_axes=[None, sp_axis, None])
    tt_out_torch = tt_out_torch[:, :seq_len, :]

    logger.info(
        f"ideogram4 block: B={batch_size} seq={seq_len} tp={tp_factor} sp={sp_factor} "
        f"padded_heads={tt_block.padded_heads} padded_len={padded_len}"
    )
    assert_quality(torch_out, tt_out_torch, pcc=0.99)
