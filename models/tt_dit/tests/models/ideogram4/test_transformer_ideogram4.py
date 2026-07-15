# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Ideogram 4.0 single-stream transformer PCC vs the OFFICIAL reference. Two tests:
#   * test_transformer_block: one Ideogram4TransformerBlock (random init) on real
#     Ideogram dims across the TP/SP/segment sweep.
#   * test_transformer_model: the full transformer (top + blocks + bottom) under
#     parallelism, random or real weights.
# Both load the reference state_dict into the tt module and assert PCC >= 0.99.
#
# Block coverage (2x4 BH mesh, FABRIC_1D, Linear): {B=1,2} x {1,2 segments} x
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
import os

import pytest
import torch
from loguru import logger
from safetensors.torch import load_file

import ttnn

from ....models.transformers.transformer_ideogram4 import (
    Ideogram4Transformer,
    Ideogram4TransformerBlock,
    rope_halfsplit_to_interleaved,
)
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....reference.ideogram4 import modeling_ideogram4
from ....reference.ideogram4.constants import IMAGE_POSITION_OFFSET, LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR
from ....reference.ideogram4.dequant import dequant_fp8_state_dict
from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.padding import PaddingConfig
from ....utils.tensor import bf16_tensor
from ....utils.test import line_params, ring_params

# Real shipped fp8 checkpoint (ideogram-ai/ideogram-4-fp8), dequantized to bf16. Override via
# env for CI. Used by the weights="real" parametrizations (weights="random" uses fresh init).
FP8 = os.environ.get("IDEOGRAM4_WEIGHTS")
# Real-weight cases need the gated fp8 checkpoint; skip (don't error) when it isn't configured.
_NEEDS_WEIGHTS = pytest.mark.skipif(not FP8, reason="IDEOGRAM4_WEIGHTS not set (gated fp8 checkpoint)")


def _real_transformer_sd(num_layers: int) -> dict:
    """Dequantized real transformer weights, filtered to the first num_layers layers so the
    (strict) tt loader accepts a truncated-depth model."""
    sd = dequant_fp8_state_dict(load_file(f"{FP8}/transformer/diffusion_pytorch_model.safetensors"))
    return {k: v for k, v in sd.items() if not (k.startswith("layers.") and int(k.split(".")[1]) >= num_layers)}


# Ring SDPA (sequence parallel) only needs each per-device shard tile-aligned, i.e. the
# padded sequence divisible by tile_size * sp_factor. Must match pipeline._sp_padded_len.
_SP_TILE = 32


def _sp_padded_len(seq_len: int, sp_factor: int) -> int:
    if sp_factor == 1:
        return seq_len
    divisor = _SP_TILE * sp_factor
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
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology"),
    [
        # factor == submesh axis size. TP shards heads (padded to a multiple of tp);
        # SP shards the sequence (ring attention). Full mesh is the 2x4 BH loudbox.
        # factor == submesh axis size; sp on axis 0 (size up to 2), tp on axis 1 (size up to 4).
        pytest.param(
            (2, 4), (1, 1), 0, 1, 1, line_params, ttnn.Topology.Linear, id="tp1sp1"
        ),  # regression: single device
        pytest.param(
            (2, 4), (1, 2), 0, 1, 1, line_params, ttnn.Topology.Linear, id="tp2"
        ),  # 18 heads % 2 == 0, no padding
        pytest.param((2, 4), (1, 4), 0, 1, 1, line_params, ttnn.Topology.Linear, id="tp4_pad20"),  # 18 -> 20 heads
        pytest.param(
            (2, 4), (2, 1), 0, 1, 1, line_params, ttnn.Topology.Linear, id="sp2"
        ),  # sequence parallel only (TP=1)
        pytest.param((2, 4), (2, 2), 0, 1, 1, line_params, ttnn.Topology.Linear, id="sp2tp2"),
        pytest.param(
            (2, 4), (2, 4), 0, 1, 1, line_params, ttnn.Topology.Linear, id="sp2tp4_pad20"
        ),  # SP=2, TP=4 (pad 20)
        pytest.param(
            (4, 2), (4, 2), 0, 1, 2, line_params, ttnn.Topology.Linear, id="sp4tp2"
        ),  # SP=4, TP=2 (full 4x2 loudbox); num_links=2
        # BH Galaxy 4x8, 2D torus Ring: SP=8 (axis 1), TP=4 (axis 0), 2 links/neighbor. 18 heads -> pad 20.
        pytest.param((4, 8), (4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, id="bh_galaxy_sp8tp4"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("batch_size", "text_len", "image_len"),
    [
        pytest.param(1, 64, 1024, id="b1_text64_img1024"),
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
    topology: ttnn.Topology,
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
    ccl_manager = CCLManager(submesh_device, num_links=num_links, topology=topology)

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
    cos4 = cos.unsqueeze(1)  # (B, 1, L, head_dim) to broadcast over heads
    sin4 = sin.unsqueeze(1)
    # tt block uses interleaved RoPE (rotary_embedding_llama); permute cos/sin to match.
    # The torch reference above keeps the original half-split tables (unchanged).
    cos4, sin4 = rope_halfsplit_to_interleaved(cos4, sin4, HEAD_DIM)
    # The residual stream is FRACTURED on hidden (TP axis) and sharded on sequence (SP).
    # Shard x on both axes when tp>1: SP on the sequence dim, TP on the hidden dim.
    if sp_factor > 1:
        x = torch.nn.functional.pad(x, (0, 0, 0, padded_len - seq_len))
        cos4 = torch.nn.functional.pad(cos4, (0, 0, 0, padded_len - seq_len))
        sin4 = torch.nn.functional.pad(sin4, (0, 0, 0, padded_len - seq_len))
        if tp_factor > 1:
            from ....utils.tensor import bf16_tensor_2dshard

            tt_x = bf16_tensor_2dshard(x, device=submesh_device, shard_mapping={sp_axis: 1, tp_axis: 2})
        else:
            tt_x = bf16_tensor(x, device=submesh_device, mesh_axis=sp_axis, shard_dim=1)
        tt_cos = bf16_tensor(cos4, device=submesh_device, mesh_axis=sp_axis, shard_dim=2)
        tt_sin = bf16_tensor(sin4, device=submesh_device, mesh_axis=sp_axis, shard_dim=2)
    else:
        if tp_factor > 1:
            tt_x = bf16_tensor(x, device=submesh_device, mesh_axis=tp_axis, shard_dim=2)
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

    # Output is FRACTURED on hidden (TP), sharded on SP. mesh_axes maps tensor dim ->
    # mesh axis: sequence (dim 1) on sp_axis, hidden (dim 2) on tp_axis.
    out_mesh_axes = [None, sp_axis, tp_axis if tp_factor > 1 else None]
    tt_out_torch = tensor.to_torch(tt_out, mesh_axes=out_mesh_axes)
    tt_out_torch = tt_out_torch[:, :seq_len, :]

    logger.info(
        f"ideogram4 block: B={batch_size} seq={seq_len} tp={tp_factor} sp={sp_factor} "
        f"padded_heads={tt_block.padded_heads} padded_len={padded_len}"
    )
    assert_quality(torch_out, tt_out_torch, pcc=0.99, relative_rmse=0.05)


# =============================================================================
# Full transformer (embeddings + input_proj + block stack + adaLN + final layer,
# i.e. the top and bottom outside the decoder blocks) vs the OFFICIAL reference
# Ideogram4Transformer, under TP / SP. 2 layers is enough to prove the wiring;
# weights="random" is the parallelism sweep, weights="real" the shipped-weight
# fidelity check. Asserts PCC >= 0.99 on the image-position velocity prediction.
# =============================================================================


def _build_model_inputs(config, batch_size, llm_len, image_len):
    """[llm tokens | image tokens] packed single-stream batch for the full model."""
    seq_len = llm_len + image_len
    llm_features = torch.randn(batch_size, seq_len, config.llm_features_dim)
    x = torch.randn(batch_size, seq_len, config.in_channels)
    t = torch.rand(batch_size)  # flow-matching time in [0, 1]
    segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)  # single segment

    indicator = torch.full((batch_size, seq_len), OUTPUT_IMAGE_INDICATOR, dtype=torch.long)
    indicator[:, :llm_len] = LLM_TOKEN_INDICATOR

    # position_ids (B, L, 3) = (t, h, w): llm tokens linear on all axes, image on an offset grid.
    position_ids = torch.zeros(batch_size, seq_len, 3, dtype=torch.long)
    lpos = torch.arange(llm_len)
    position_ids[:, :llm_len, 0] = lpos
    position_ids[:, :llm_len, 1] = lpos
    position_ids[:, :llm_len, 2] = lpos
    grid_h = int(round(image_len**0.5))
    while image_len % grid_h != 0:
        grid_h -= 1
    grid_w = image_len // grid_h
    hh = torch.arange(grid_h).repeat_interleave(grid_w)
    ww = torch.arange(grid_w).repeat(grid_h)
    position_ids[:, llm_len:, 0] = IMAGE_POSITION_OFFSET
    position_ids[:, llm_len:, 1] = IMAGE_POSITION_OFFSET + hh
    position_ids[:, llm_len:, 2] = IMAGE_POSITION_OFFSET + ww

    return llm_features, x, t, position_ids, segment_ids, indicator


def _seq_shard(t: torch.Tensor, seq_dim: int, padded_len: int, sp_factor: int, sp_axis: int, device, *, dtype):
    """Pad a tensor on its sequence dim and shard it across the SP axis (or replicate)."""
    if sp_factor > 1:
        pad = padded_len - t.shape[seq_dim]
        if pad:
            spec = [0, 0] * (t.ndim - seq_dim - 1) + [0, pad]
            t = torch.nn.functional.pad(t, spec)
        mesh_axes = [None] * t.ndim
        mesh_axes[seq_dim] = sp_axis
        layout = ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.uint32 else ttnn.TILE_LAYOUT
        return tensor.from_torch(t, device=device, dtype=dtype, layout=layout, mesh_axes=mesh_axes)
    if dtype == ttnn.uint32:
        return tensor.from_torch(t, device=device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    return bf16_tensor(t, device=device)


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "weights"),
    [
        # random-weight parallelism sweep (proves the full-model wiring under TP / SP).
        pytest.param((1, 1), (1, 1), 0, 1, 1, {}, ttnn.Topology.Linear, "random", id="tp1sp1"),
        pytest.param((2, 4), (1, 2), 0, 1, 1, line_params, ttnn.Topology.Linear, "random", id="tp2"),
        pytest.param((2, 4), (1, 4), 0, 1, 1, line_params, ttnn.Topology.Linear, "random", id="tp4_pad20"),
        pytest.param((2, 4), (2, 1), 0, 1, 1, line_params, ttnn.Topology.Linear, "random", id="sp2"),
        pytest.param((2, 4), (2, 2), 0, 1, 1, line_params, ttnn.Topology.Linear, "random", id="sp2tp2"),
        pytest.param((2, 4), (2, 4), 0, 1, 1, line_params, ttnn.Topology.Linear, "random", id="sp2tp4_pad20"),
        # BH Galaxy 4x8, 2D torus Ring: SP=8 (axis 1), TP=4 (axis 0), 2 links/neighbor.
        pytest.param((4, 8), (4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, "random", id="bh_galaxy_sp8tp4"),
        # real-weight fidelity: shipped checkpoint on single device and under production TP.
        pytest.param((1, 1), (1, 1), 0, 1, 1, {}, ttnn.Topology.Linear, "real", id="tp1sp1_real", marks=_NEEDS_WEIGHTS),
        pytest.param(
            (2, 4), (1, 4), 0, 1, 1, line_params, ttnn.Topology.Linear, "real", id="tp4_real", marks=_NEEDS_WEIGHTS
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("num_layers", [2], ids=["layers2"])
@pytest.mark.parametrize(
    ("batch_size", "llm_len", "image_len"),
    [pytest.param(1, 64, 256, id="llm64_img256")],
)
def test_transformer_model(
    *,
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    weights: str,
    num_layers: int,
    batch_size: int,
    llm_len: int,
    image_len: int,
) -> None:
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16
    seq_len = llm_len + image_len

    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    sp_factor = tuple(submesh_device.shape)[sp_axis]
    tp_factor = tuple(submesh_device.shape)[tp_axis]

    config = modeling_ideogram4.Ideogram4Config(num_layers=num_layers)
    torch_model = modeling_ideogram4.Ideogram4Transformer(config).to(dtype=torch_dtype)
    if weights == "real":
        state_dict = _real_transformer_sd(num_layers)
        _load = torch_model.load_state_dict(state_dict, strict=False)
        # strict=False tolerates missing non-persistent buffers (e.g. rotary_emb), but UNEXPECTED
        # keys mean a broken checkpoint key map — fail loudly rather than silently dropping weights.
        assert not _load.unexpected_keys, f"unexpected checkpoint keys: {_load.unexpected_keys[:5]}"
    else:
        state_dict = torch_model.state_dict()  # random init
    torch_model.eval()

    llm_features, x, t, position_ids, segment_ids, indicator = _build_model_inputs(
        config, batch_size, llm_len, image_len
    )
    llm_features = llm_features.to(torch_dtype)
    x = x.to(torch_dtype)

    with torch.no_grad():
        torch_out = torch_model(
            llm_features=llm_features,
            x=x,
            t=t,
            position_ids=position_ids,
            segment_ids=segment_ids,
            indicator=indicator,
        )

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )
    ccl_manager = CCLManager(submesh_device, num_links=num_links, topology=topology)
    padding_config = (
        PaddingConfig.from_tensor_parallel_factor(config.num_heads, config.emb_dim // config.num_heads, tp_factor)
        if config.num_heads % tp_factor != 0
        else None
    )

    tt_model = Ideogram4Transformer(
        emb_dim=config.emb_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        intermediate_size=config.intermediate_size,
        adaln_dim=config.adanln_dim,
        in_channels=config.in_channels,
        llm_features_dim=config.llm_features_dim,
        norm_eps=config.norm_eps,
        mesh_device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )
    tt_model.load_torch_state_dict(state_dict)

    cos, sin = torch_model.rotary_emb(position_ids)
    t_sin = Ideogram4Transformer.sinusoidal_embedding(t, config.emb_dim)
    llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(torch.float32).unsqueeze(-1)
    output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(torch.float32).unsqueeze(-1)
    image_idx = (indicator == OUTPUT_IMAGE_INDICATOR).to(torch.int32)
    padded_len = _sp_padded_len(seq_len, sp_factor)

    dev = submesh_device
    _cos_il, _sin_il = rope_halfsplit_to_interleaved(
        cos.unsqueeze(1), sin.unsqueeze(1), config.emb_dim // config.num_heads
    )
    tt_out = tt_model(
        x=_seq_shard(x, 1, padded_len, sp_factor, sp_axis, dev, dtype=ttnn.bfloat16),
        llm_features=_seq_shard(llm_features, 1, padded_len, sp_factor, sp_axis, dev, dtype=ttnn.bfloat16),
        t_sin=bf16_tensor(t_sin.unsqueeze(1), device=dev),
        cos=_seq_shard(_cos_il, 2, padded_len, sp_factor, sp_axis, dev, dtype=ttnn.bfloat16),
        sin=_seq_shard(_sin_il, 2, padded_len, sp_factor, sp_axis, dev, dtype=ttnn.bfloat16),
        image_indicator_index=_seq_shard(image_idx, 1, padded_len, sp_factor, sp_axis, dev, dtype=ttnn.uint32),
        llm_token_mask=_seq_shard(llm_token_mask, 1, padded_len, sp_factor, sp_axis, dev, dtype=ttnn.bfloat16),
        output_image_mask=_seq_shard(output_image_mask, 1, padded_len, sp_factor, sp_axis, dev, dtype=ttnn.bfloat16),
        attn_mask=None,
        spatial_sequence_length=seq_len,
    )
    tt_out_torch = tensor.to_torch(tt_out, mesh_axes=[None, sp_axis, None])[:, :seq_len]

    logger.info(
        f"ideogram4 model [{weights}]: layers={num_layers} tp={tp_factor} sp={sp_factor} "
        f"padded_heads={tt_model.layers[0].padded_heads} seq={seq_len} padded_len={padded_len}"
    )
    # Only OUTPUT_IMAGE positions carry a meaningful velocity (the reference zeros the
    # image-latent input at llm positions; their output is unused conditioning noise).
    image_mask = (indicator == OUTPUT_IMAGE_INDICATOR)[0]  # (L,), same across batch
    assert_quality(torch_out[:, image_mask], tt_out_torch[:, image_mask], pcc=0.99, relative_rmse=0.05)
