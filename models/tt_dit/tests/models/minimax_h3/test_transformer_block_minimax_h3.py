# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM, MiniMaxH3RotaryPosEmbed
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock as TorchMiniMaxH3Block
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.transformer_block_minimax_h3 import MiniMaxH3TransformerBlock
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
FFN_DIM = 14336
TIME_EMBED_DIM = 2688
NORM_EPS = 1e-5
QK_NORM_EPS = 1e-5
ROPE_FREQ_DIM = 16
ROPE_THETA = 10000.0

# Token tags, per the reference: 0 video, 1 text, 2 audio (-1 padding, unused here).
TAG_VIDEO, TAG_TEXT, TAG_AUDIO = 0, 1, 2


def _packed_layout(num_text: int, num_audio: int, num_video: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one packed-sequence layout: `(position_ids, token_tags, timestep_indices)`.

    The block is agnostic to how the pipeline orders rows -- it only reads per-row modality tags and
    timestep indices through `adaln_indices` -- so this is a representative layout rather than the
    real t2va one: text rows, then audio rows, then video rows.

    Two distinct timesteps are used so the AdaLN table is addressed at more than one noise level, as
    the real model does when it serves conditioning rows and target rows in a single forward: text and
    the first video frame are clean (timestep 0), the remaining video and all audio are noisy
    (timestep 1). That covers four distinct `(timestep, modality)` table rows including row 0, so an
    off-by-one-modality error in the per-row gather cannot pass unnoticed.

    Video rows get a (t, h, w) patch grid; text and audio rows advance the shared `t` clock with
    h = w = 0, which is enough to exercise the 3-axis rope on every modality.
    """
    grid_h = grid_w = 8
    frame = grid_h * grid_w
    assert num_video % frame == 0, "num_video must fill whole (h, w) frames"
    grid_t = num_video // frame
    assert grid_t >= 2, "need at least one conditioning frame and one target frame"

    tags = torch.cat(
        [
            torch.full((num_text,), TAG_TEXT, dtype=torch.long),
            torch.full((num_audio,), TAG_AUDIO, dtype=torch.long),
            torch.full((num_video,), TAG_VIDEO, dtype=torch.long),
        ]
    )
    # Text rows clean; audio noisy; first video frame clean (conditioning), rest noisy (target).
    timestep_indices = torch.cat(
        [
            torch.zeros(num_text, dtype=torch.long),
            torch.ones(num_audio, dtype=torch.long),
            torch.zeros(frame, dtype=torch.long),
            torch.ones(num_video - frame, dtype=torch.long),
        ]
    )

    vt, vh, vw = torch.meshgrid(torch.arange(grid_t), torch.arange(grid_h), torch.arange(grid_w), indexing="ij")
    video_pos = torch.stack([vt.reshape(-1), vh.reshape(-1), vw.reshape(-1)], dim=-1)

    def clock_pos(n: int) -> torch.Tensor:
        return torch.stack([torch.arange(n), torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)], -1)

    position_ids = torch.cat([clock_pos(num_text), clock_pos(num_audio), video_pos], dim=0)
    return position_ids, tags, timestep_indices


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
    ("num_text", "num_audio", "num_video"),
    [
        # seq_len must be divisible by sp_factor * TILE (8 * 32 = 256) so the packed sequence is
        # padless and needs no attention mask.
        pytest.param(512, 256, 1280, id="small_s2048"),
        pytest.param(512, 256, 20736, id="s21504"),
    ],
)
def test_minimax_h3_transformer_block(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    num_text: int,
    num_audio: int,
    num_video: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
    reset_seeds,
) -> None:
    # This threshold has to be tight, and not for numerical-precision reasons: the block output is
    # `residual + gate * branch`, so the residual stream dominates it and swamps errors in the
    # modulation path. Measured at real dims with the torch reference, every plausible per-row gather
    # bug still lands at PCC >= 0.9959 against the correct output (measured with the norm weights
    # randomized, see `randomize_norm_weights`):
    #   all rows -> table row 0    0.9959      row order reversed         0.9969
    #   off-by-one modality        0.9962      timestep ignored           0.9973
    #   tags/timesteps swapped     0.9967      modality ignored           0.9984
    # A loose threshold (0.99, say) would therefore pass a completely broken AdaLN gather. The real
    # implementation measures 0.999995 on both params, so 0.9995 sits clear of both bounds.
    MIN_PCC = 0.9995

    skip_if_unsupported_num_links(mesh_device, num_links)

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    B = 1
    position_ids, token_tags, timestep_indices = _packed_layout(num_text, num_audio, num_video)
    seq_len = position_ids.shape[0]
    assert seq_len == num_text + num_audio + num_video
    assert seq_len % (sp_factor * ttnn.TILE_SIZE) == 0, (
        f"seq_len={seq_len} must be divisible by sp_factor * TILE ({sp_factor * ttnn.TILE_SIZE}) "
        "to keep the packed sequence padless"
    )

    num_timesteps = int(timestep_indices.max().item()) + 1
    # Row -> AdaLN table row, exactly as the reference computes it.
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0)
    assert int(adaln_indices.max().item()) < num_timesteps * MINIMAX_H3_MODALITY_NUM
    logger.info(
        f"seq_len={seq_len} ({seq_len // sp_factor} per SP device), num_timesteps={num_timesteps}, "
        f"adaln table rows={num_timesteps * MINIMAX_H3_MODALITY_NUM}, "
        f"tags present={sorted(set(token_tags.tolist()))}"
    )

    # Reference block with random weights -- the 66GB checkpoint is not needed to validate the block.
    torch_model = TorchMiniMaxH3Block(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        attention_head_dim=HEAD_DIM,
        ffn_dim=FFN_DIM,
        time_embed_dim=TIME_EMBED_DIM,
        norm_eps=NORM_EPS,
        qk_norm_eps=QK_NORM_EPS,
    ).to(torch.float32)
    # Without this every RMSNorm weight is ones and norm weight loading is untested; see
    # `randomize_norm_weights`.
    randomize_norm_weights(torch_model)
    torch_model.eval()

    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    rope_cos, rope_sin = rope(position_ids)  # each (seq_len, 96)
    rotary_dim = rope_cos.shape[-1]

    spatial_input = torch.randn((B, seq_len, HIDDEN_SIZE), dtype=torch.float32)
    # `temb` is the shared timestep embedding: one row per *distinct* timestep, not per batch item.
    temb_input = torch.randn((num_timesteps, TIME_EMBED_DIM), dtype=torch.float32)

    logger.info("Running torch model")
    with torch.no_grad():
        torch_out = torch_model(
            hidden_states=spatial_input,
            temb=temb_input,
            adaln_indices=adaln_indices,
            rotary_emb=(rope_cos, rope_sin),
            attention_mask=None,
        )

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )

    tt_model = MiniMaxH3TransformerBlock(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        ffn_dim=FFN_DIM,
        time_embed_dim=TIME_EMBED_DIM,
        norm_eps=NORM_EPS,
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
    # temb is a handful of rows shared by every device: replicated. Kept float32 to match the
    # reference, where the SiLU runs at temb's own precision before the bfloat16 AdaLN projection.
    tt_temb = from_torch(
        temb_input.reshape(1, 1, num_timesteps, TIME_EMBED_DIM),
        device=mesh_device,
        dtype=ttnn.float32,
    )
    # One AdaLN table row index per local row of the packed sequence, so fractured on SP.
    tt_adaln_indices = from_torch(
        adaln_indices.to(torch.int32).reshape(1, 1, 1, seq_len),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.Layout.ROW_MAJOR,
        mesh_axes=[..., None, sp_axis],
    )
    # cos/sin are shared by every head: fractured on SP, replicated on TP.
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
    logger.info(f"tt_spatial {tt_spatial.shape}, tt_temb {tt_temb.shape}, tt_adaln_indices {tt_adaln_indices.shape}")

    logger.info("Running TT model")
    tt_out = tt_model(
        tt_spatial,
        N=seq_len,
        temb=tt_temb,
        adaln_indices=tt_adaln_indices,
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
