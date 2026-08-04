# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention as TorchMiniMaxH3Attention
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import MiniMaxH3Attention
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor_2dshard, from_torch
from ....utils.test import ring_params_req_exact_devices, skip_if_unsupported_num_links
from .common import randomize_norm_weights

# MiniMax-H3 transformer config, shared by the `transformer/` (t2va) and `transformer_ref/` partitions.
HIDDEN_SIZE = 5376
NUM_HEADS = 56
HEAD_DIM = 128
QK_NORM_EPS = 1e-5
ROPE_FREQ_DIM = 16
ROPE_THETA = 10000.0
PATCH_SIZE = (1, 2, 2)

# NOTE: MiniMax-H3's attention inner dim (56 * 128 = 7168) is *larger* than the residual stream
# (5376), unlike Wan. to_q/k/v are 5376 -> 7168 and to_out is 7168 -> 5376, all bias-free.


def _packed_position_ids(T: int, H: int, W: int) -> torch.Tensor:
    """`(seq_len, 3)` (t, h, w) rotary coordinates of a video patch grid, row-major over (t, h, w).

    The attention module does not distinguish modalities -- it is full self-attention over one packed
    sequence -- so a pure video grid is a sufficient stand-in for the real packed layout here.
    """
    p_t, p_h, p_w = PATCH_SIZE
    grid_t, grid_h, grid_w = T // p_t, H // p_h, W // p_w
    coords = torch.meshgrid(torch.arange(grid_t), torch.arange(grid_h), torch.arange(grid_w), indexing="ij")
    return torch.stack([c.reshape(-1) for c in coords], dim=-1)


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param(
            (4, 8), 1, 0, 2, ring_params_req_exact_devices, ttnn.Topology.Ring, False, id="4x8sp1tp0nl2_ring_is_fsdp0"
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("T", "H", "W"),
    [
        # Grids chosen so seq_len is divisible by sp_factor * TILE (8 * 32 = 256). A padless packed
        # sequence needs no attention mask, which is what the reference's fast path assumes.
        pytest.param(4, 32, 32, id="small_s1024"),
        pytest.param(21, 64, 64, id="s21504"),
    ],
)
def test_minimax_h3_attention(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    T: int,
    H: int,
    W: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
    reset_seeds,
) -> None:
    # Measured 0.9997 on both params at bringup; 0.995 leaves margin without being a rubber stamp.
    MIN_PCC = 0.995

    skip_if_unsupported_num_links(mesh_device, num_links)

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    assert NUM_HEADS % tp_factor == 0, f"{NUM_HEADS} heads must divide across TP={tp_factor}"

    B = 1
    position_ids = _packed_position_ids(T, H, W)
    seq_len = position_ids.shape[0]
    assert seq_len % (sp_factor * ttnn.TILE_SIZE) == 0, (
        f"seq_len={seq_len} must be divisible by sp_factor * TILE ({sp_factor * ttnn.TILE_SIZE}) "
        "to keep the packed sequence padless"
    )
    logger.info(f"seq_len={seq_len} ({seq_len // sp_factor} per SP device), tp_factor={tp_factor}")

    # Reference attention with random weights -- the 66GB checkpoint is not needed to validate the op.
    torch_model = TorchMiniMaxH3Attention(
        hidden_size=HIDDEN_SIZE,
        heads=NUM_HEADS,
        dim_head=HEAD_DIM,
        qk_norm_eps=QK_NORM_EPS,
    ).to(torch.float32)
    # Without this the per-head norm_q/norm_k weights are all ones, so QK-norm weight loading is
    # untested; see `randomize_norm_weights`.
    randomize_norm_weights(torch_model)
    torch_model.eval()

    # Real RoPE rather than random cos/sin: MiniMax-H3 rotates only the leading 2 * 3 * rope_freq_dim
    # = 96 of the 128 head channels and passes the remaining 32 through unchanged. Note the rotate-half
    # split is over those 96 channels (pairing i with i+48), *not* over the full head_dim, so cos/sin
    # cannot simply be padded out to 128 and fed to a standard head_dim-wide rotary kernel.
    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    rope_cos, rope_sin = rope(position_ids)  # each (seq_len, 96)
    rotary_dim = rope_cos.shape[-1]
    logger.info(f"rotary_dim={rotary_dim} of head_dim={HEAD_DIM} ({HEAD_DIM - rotary_dim} pass-through)")

    spatial_input = torch.randn((B, seq_len, HIDDEN_SIZE), dtype=torch.float32)

    logger.info("Running torch model")
    with torch.no_grad():
        torch_out = torch_model(
            hidden_states=spatial_input,
            rotary_emb=(rope_cos, rope_sin),
            attention_mask=None,
        )

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
        qk_norm_eps=QK_NORM_EPS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    # spatial: seq fractured on SP, hidden fractured on TP
    tt_spatial = bf16_tensor_2dshard(
        spatial_input.unsqueeze(0), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3}
    )
    # cos/sin are shared by every head, so they are fractured on SP and replicated on TP
    tt_rope_cos = from_torch(
        rope_cos.reshape(1, 1, seq_len, rotary_dim),
        device=mesh_device,
        dtype=ttnn.float32,
        mesh_axes=[..., sp_axis, None],
    )
    tt_rope_sin = from_torch(
        rope_sin.reshape(1, 1, seq_len, rotary_dim),
        device=mesh_device,
        dtype=ttnn.float32,
        mesh_axes=[..., sp_axis, None],
    )
    logger.info(f"tt_spatial {tt_spatial.shape}, tt_rope_cos {tt_rope_cos.shape}")

    logger.info("Running TT model")
    tt_out = tt_model(
        tt_spatial,
        N=seq_len,
        rope_cos=tt_rope_cos,
        rope_sin=tt_rope_sin,
    )

    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 3
    tt_out = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
    )
    tt_out = tt_out[:, :, :seq_len, :]

    assert_quality(torch_out, tt_out, pcc=MIN_PCC)
