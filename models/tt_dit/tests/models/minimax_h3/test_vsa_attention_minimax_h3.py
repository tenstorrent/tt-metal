# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""VSA as the fourth MiniMaxH3Attention path, on the 4x8 galaxy (R2 + R5 at attention level).

The torch reference mirrors the diffusers attention processor exactly (same projections, QK-norm,
rotary) but swaps dense SDPA for the validated `vsa_oracle` in tiled order, gate included -- so
this pins the full wiring: QKV + fused norm/RoPE, the SP K/V all-gather (R2), the coarse stage,
vsa_sdpa, the gate projection, and to_out.

Gates covered here at attention level: sparsity 0 vs the dense ring path (R6a), sparsity > 0 with
zero and random gate vs the torch oracle (R6b/c), striped == identity after unpacking (R6d).
"""

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3Attention as TorchMiniMaxH3Attention,
    MiniMaxH3RotaryPosEmbed,
    _apply_rotary_emb,
)
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import MiniMaxH3Attention, prepare_rope_tables
from ....models.transformers.minimax_h3.vsa_stages_minimax_h3 import MiniMaxH3VSAConfig, MiniMaxH3VSACoarseStage
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3.packing import build_packed_sequence
from ....pipelines.minimax_h3.vsa_geometry import build_vsa_geometry
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor_2dshard
from ....utils.test import skip_if_unsupported_num_links
from .common import GALAXY_RING, randomize_norm_weights, upload_rope
from .vsa_oracle import vsa_attention

HIDDEN_SIZE = 5376
NUM_HEADS = 56
HEAD_DIM = 128
QK_NORM_EPS = 1e-5
ROPE_FREQ_DIM = 16
ROPE_THETA = 10000.0
PATCH = (1, 2, 2)


def _make_layout(num_text: int, num_audio_latents: int, grid: tuple[int, int, int]):
    t, h, w = grid
    layout = build_packed_sequence(
        text_token_tags=torch.ones(num_text, dtype=torch.long),
        num_latent_frames=t,
        latent_height=h * PATCH[1],
        latent_width=w * PATCH[2],
        num_audio_latents=num_audio_latents,
        patch_size=PATCH,
    )
    prefix_segments = (num_text, 0, 2 * num_audio_latents)
    return layout, prefix_segments


def _torch_vsa_reference(torch_model, x, rope_cos, rope_sin, geometry, sparsity, gate_weight):
    """The diffusers processor with dense SDPA swapped for the tiled VSA oracle."""
    with torch.no_grad():
        q = torch_model.to_q(x).unflatten(-1, (NUM_HEADS, HEAD_DIM))
        k = torch_model.to_k(x).unflatten(-1, (NUM_HEADS, HEAD_DIM))
        v = torch_model.to_v(x).unflatten(-1, (NUM_HEADS, HEAD_DIM))
        q = torch_model.norm_q(q)
        k = torch_model.norm_k(k)
        q = _apply_rotary_emb(q, rope_cos, rope_sin)
        k = _apply_rotary_emb(k, rope_cos, rope_sin)

        # [1, S, H, D] -> tiled [1, H, S_pad, D]
        tile = lambda tensor: geometry.pack_rows(tensor[0], dim=0).permute(1, 0, 2).unsqueeze(0)
        gate = None
        if gate_weight is not None:
            gate = tile((x @ gate_weight.T).unflatten(-1, (NUM_HEADS, HEAD_DIM))[0].unsqueeze(0))
        out_tiled = vsa_attention(tile(q), tile(k), tile(v), geometry, sparsity, gate_tiled=gate)
        out = geometry.unpack_rows(out_tiled.squeeze(0).permute(1, 0, 2))  # [S, H, D]
        return torch_model.to_out[0](out.reshape(1, -1, NUM_HEADS * HEAD_DIM))


def _setup_tt_model(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, geometry, vsa_config, state_dict):
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )
    tt_model = MiniMaxH3Attention(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=2 * 3 * ROPE_FREQ_DIM,
        qk_norm_eps=QK_NORM_EPS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        vsa_config=vsa_config,
    )
    tt_model.load_torch_state_dict(state_dict)
    if vsa_config is not None:
        stage = MiniMaxH3VSACoarseStage(
            geometry,
            sparsity=vsa_config.sparsity,
            head_dim=HEAD_DIM,
            mesh_device=mesh_device,
            sp_axis=sp_axis,
            ccl_manager=ccl_manager,
        )
        tt_model.set_vsa_stage(stage)
    return tt_model, ccl_manager


def _run_tt(tt_model, mesh_device, sp_axis, tp_axis, geometry, x, rope_cos, rope_sin):
    """Upload the tiled-order input + permuted rope tables, run, return output in tiled order."""
    x_tiled = geometry.pack_rows(x[0], dim=0).unsqueeze(0).unsqueeze(0)  # [1, 1, S_pad, hidden]
    cos_tiled = geometry.permute_metadata(rope_cos, dim=0)
    sin_tiled = geometry.permute_metadata(rope_sin, dim=0)
    cos_t, sin_t = prepare_rope_tables(cos_tiled, sin_tiled, HEAD_DIM)
    tt_cos, tt_sin = upload_rope(cos_t, sin_t, mesh_device=mesh_device, sp_axis=sp_axis)
    tt_x = bf16_tensor_2dshard(x_tiled, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    tt_out = tt_model(tt_x, N=geometry.padded_len, rope_cos=tt_cos, rope_sin=tt_sin)
    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 3
    return ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
    )


def _torch_attention_model():
    model = TorchMiniMaxH3Attention(
        hidden_size=HIDDEN_SIZE, heads=NUM_HEADS, dim_head=HEAD_DIM, qk_norm_eps=QK_NORM_EPS
    ).to(torch.float32)
    randomize_norm_weights(model)
    model.eval()
    return model


@GALAXY_RING
def test_vsa_sparsity0_matches_ring_path(
    mesh_device, sp_axis, tp_axis, num_links, is_fsdp, topology, reset_seeds
) -> None:
    """R6a at attention level: sparsity 0, zero gate, padless full-tile geometry vs the ring path."""
    MIN_PCC = 0.9995

    skip_if_unsupported_num_links(mesh_device, num_links)
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    if sp_factor != 8:
        pytest.skip("VSA v0 targets 4x8")

    # padless: prefix segments are multiples of 64 and the video grid has no ragged tails
    layout, prefix_segments = _make_layout(num_text=192, num_audio_latents=160, grid=(8, 8, 8))
    geometry = build_vsa_geometry(prefix_segments, (8, 8, 8), sp_factor=sp_factor)
    assert geometry.n_pad_tiles == 0 and int(geometry.valid_counts.min()) == 64

    torch_model = _torch_attention_model()
    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    rope_cos, rope_sin = rope(layout.position_ids)
    x = torch.randn(1, geometry.seq_len, HIDDEN_SIZE)

    vsa_model, _ = _setup_tt_model(
        mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, geometry,
        MiniMaxH3VSAConfig(sparsity=0.0), torch_model.state_dict(),
    )  # fmt: skip
    assert vsa_model.gate_compress_is_zero
    vsa_out = _run_tt(vsa_model, mesh_device, sp_axis, tp_axis, geometry, x, rope_cos, rope_sin)

    ring_model, _ = _setup_tt_model(
        mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, geometry, None, torch_model.state_dict()
    )
    ring_out = _run_tt(ring_model, mesh_device, sp_axis, tp_axis, geometry, x, rope_cos, rope_sin)

    assert_quality(ring_out, vsa_out, pcc=MIN_PCC)


@GALAXY_RING
@pytest.mark.parametrize("gate", ["zero", "random"])
@pytest.mark.parametrize("placement", ["identity", "striped"])
def test_vsa_attention_vs_torch_oracle(
    mesh_device, sp_axis, tp_axis, num_links, is_fsdp, topology, gate, placement, reset_seeds
) -> None:
    """R6b/c/d at attention level: sparsity 0.75, ragged tiles + pad tiles, vs the torch VSA oracle."""
    MIN_PCC = 0.99
    SPARSITY = 0.75

    skip_if_unsupported_num_links(mesh_device, num_links)
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    if sp_factor != 8:
        pytest.skip("VSA v0 targets 4x8")

    # ragged everywhere: partial prefix tails, ragged video tails in all three dims, pad tiles
    layout, prefix_segments = _make_layout(num_text=70, num_audio_latents=65, grid=(9, 10, 13))
    geometry = build_vsa_geometry(prefix_segments, (9, 10, 13), sp_factor=sp_factor, placement=placement)
    assert geometry.n_pad_tiles > 0

    torch_model = _torch_attention_model()
    state = dict(torch_model.state_dict())
    gate_weight = None
    if gate == "random":
        torch.manual_seed(3)
        gate_weight = 0.02 * torch.randn(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE)
        state["to_gate_compress.weight"] = gate_weight

    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    rope_cos, rope_sin = rope(layout.position_ids)
    x = torch.randn(1, geometry.seq_len, HIDDEN_SIZE)

    ref = _torch_vsa_reference(torch_model, x, rope_cos, rope_sin, geometry, SPARSITY, gate_weight)

    vsa_model, _ = _setup_tt_model(
        mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, geometry,
        MiniMaxH3VSAConfig(sparsity=SPARSITY, placement=placement, k_chunk_blocks=2), state,
    )  # fmt: skip
    assert vsa_model.gate_compress_is_zero == (gate == "zero")
    tt_out = _run_tt(vsa_model, mesh_device, sp_axis, tp_axis, geometry, x, rope_cos, rope_sin)

    # unpack to the original packed order and compare on real rows only
    tt_rows = geometry.unpack_rows(tt_out.reshape(geometry.padded_len, HIDDEN_SIZE), dim=0)
    logger.info(f"comparing {tt_rows.shape} rows (placement={placement}, gate={gate})")
    assert_quality(ref[0], tt_rows, pcc=MIN_PCC)
