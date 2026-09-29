# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""With SP = 1 the DiT's attention must not let real rows attend to the padded tail.

`MiniMaxH3Attention` picks ring attention only when the sequence-parallel factor is above 1. Ring
attention takes `logical_n` and masks everything at or past it internally; plain
`scaled_dot_product_attention` has no such argument, so on a 1x1 or 1x4 mesh the pad rows are
ordinary keys unless a block-diagonal window fences them off. That is a silent accuracy loss, not a
crash, and it scales with the padding fraction -- which the bucket ladder makes large.

The check is TT-against-TT: the same block over the same real rows, once with no padding at all and
once padded to twice the length. Only the window can make those agree.
"""

import torch
from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM, MiniMaxH3RotaryPosEmbed
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock as TorchMiniMaxH3Block
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import prepare_rope_tables
from ....models.transformers.minimax_h3.transformer_block_minimax_h3 import MiniMaxH3TransformerBlock
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.tensor import bf16_tensor_2dshard, from_torch
from .common import SMALL_LINE_PARALLEL, packed_layout, randomize_norm_weights, upload_rope

# A small block rather than the 5376-wide production one: the window is a property of the attention
# mask, not of the widths, and a 385 M-parameter block would make this a minutes-long test. Head dim
# stays 128 so the rope tables keep the real model's shape, and `num_heads` and `hidden_size` stay
# divisible by TP=4 and by 32 * 4.
SMALL_BLOCK = dict(hidden_size=1024, num_heads=8, head_dim=128, ffn_dim=2048, time_embed_dim=512)
NORM_EPS = 1e-5
ROPE_FREQ_DIM = 16
ROPE_THETA = 10000.0

# 512 real rows padded to 1024. Half the sequence being pad is what the bucket ladder's coarse rungs
# actually produce, and it makes the no-window case unmistakable rather than marginal.
NUM_TEXT, NUM_AUDIO, NUM_VIDEO = 64, 64, 384
PADDED_LEN = 1024

# Windowed and unpadded must agree to the bf16 noise floor: same arithmetic over the same keys.
MAX_REL_RMSE_WINDOWED = 0.03  # measured 0.0168 on 1x4: the bf16 floor for this block
# Without the window the pad keys enter every real row's softmax. This is a lower bound on the
# damage, not a target -- it fails the test if dropping the window stops mattering, which would mean
# the windowed leg above had stopped proving anything.
MIN_REL_RMSE_UNWINDOWED = 0.30  # measured 0.673 on 1x4


def _rel_rmse(reference: torch.Tensor, test: torch.Tensor) -> float:
    """`||test - reference|| / ||reference||`, in float64.

    Not PCC: letting the pad rows into the softmax rescales every real row's attention output by
    roughly the fraction of real keys, and PCC cannot see a scale factor at all -- the first version
    of this test scored 0.9997 on output that had lost half its magnitude. Relative RMSE sees it.
    """
    reference = reference.detach().flatten().to(torch.float64)
    test = test.detach().flatten().to(torch.float64)
    return float(torch.linalg.vector_norm(test - reference) / torch.linalg.vector_norm(reference))


def _logical_n(mesh_device: ttnn.MeshDevice, value: int) -> ttnn.Tensor:
    return from_torch(
        torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.Layout.ROW_MAJOR,
        mesh_axes=[..., None, None],
    )


def _windows(mesh_device: ttnn.MeshDevice, boundaries: list[int]) -> ttnn.Tensor:
    return from_torch(
        torch.tensor(boundaries, dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.Layout.ROW_MAJOR,
        mesh_axes=[None],
    )


@SMALL_LINE_PARALLEL
def test_minimax_h3_sp1_attention_excludes_pad_rows(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
    reset_seeds,
) -> None:
    mesh_shape = tuple(mesh_device.shape)
    sp_factor, tp_factor = mesh_shape[sp_axis], mesh_shape[tp_axis]
    assert sp_factor == 1, f"this test is about the SP=1 path; mesh {mesh_shape} has sp_factor {sp_factor}"

    hidden_size, head_dim = SMALL_BLOCK["hidden_size"], SMALL_BLOCK["head_dim"]
    time_embed_dim = SMALL_BLOCK["time_embed_dim"]

    torch_block = TorchMiniMaxH3Block(
        hidden_size=hidden_size,
        num_attention_heads=SMALL_BLOCK["num_heads"],
        attention_head_dim=head_dim,
        ffn_dim=SMALL_BLOCK["ffn_dim"],
        time_embed_dim=time_embed_dim,
        norm_eps=NORM_EPS,
        qk_norm_eps=NORM_EPS,
    ).to(torch.float32)
    randomize_norm_weights(torch_block)
    state_dict = torch_block.state_dict()

    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )

    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 3
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=mesh_shape)

    def upload(spatial: torch.Tensor, adaln: torch.Tensor, positions: torch.Tensor):
        rows = spatial.shape[0]
        cos, sin = prepare_rope_tables(*rope(positions), head_dim)
        tt_cos, tt_sin = upload_rope(cos, sin, mesh_device=mesh_device, sp_axis=sp_axis)
        return dict(
            spatial=bf16_tensor_2dshard(
                spatial.reshape(1, 1, rows, hidden_size), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3}
            ),
            adaln_indices=from_torch(
                adaln.to(torch.int32).reshape(1, 1, 1, rows),
                device=mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.Layout.ROW_MAJOR,
                mesh_axes=[..., None, sp_axis],
            ),
            rope_cos=tt_cos,
            rope_sin=tt_sin,
        )

    def read(out: ttnn.Tensor, rows: int) -> torch.Tensor:
        return ttnn.to_torch(out, mesh_composer=composer)[:, :, :rows, :]

    positions, tags, timestep_indices = packed_layout(NUM_TEXT, NUM_AUDIO, NUM_VIDEO)
    seq_len = positions.shape[0]
    assert seq_len % ttnn.TILE_SIZE == 0, f"seq_len {seq_len} must be tile-aligned to run unpadded"
    assert PADDED_LEN > seq_len
    num_timesteps = int(timestep_indices.max().item()) + 1
    adaln = timestep_indices * MINIMAX_H3_MODALITY_NUM + tags.clamp(min=0)

    real = torch.randn((seq_len, hidden_size), dtype=torch.float32)
    # Pad rows carry content drawn from the same distribution as the real rows. Zero rows would
    # still perturb a maskless softmax, but by a smaller and layout-dependent amount; matching the
    # distribution makes the unwindowed leg's damage a property of the missing mask alone.
    pad = torch.randn((PADDED_LEN - seq_len, hidden_size), dtype=torch.float32)
    padded = torch.cat([real, pad])
    padded_positions, padded_tags, padded_ts = packed_layout(NUM_TEXT, NUM_AUDIO, NUM_VIDEO, padded_len=PADDED_LEN)
    padded_adaln = padded_ts * MINIMAX_H3_MODALITY_NUM + padded_tags.clamp(min=0)

    tt_temb = from_torch(
        torch.randn((num_timesteps, time_embed_dim), dtype=torch.float32).reshape(1, 1, num_timesteps, time_embed_dim),
        device=mesh_device,
        dtype=ttnn.float32,
    )

    block_kwargs = dict(
        **SMALL_BLOCK,
        norm_eps=NORM_EPS,
        qk_norm_eps=NORM_EPS,
        rotary_dim=rope(positions)[0].shape[-1],
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )
    block = MiniMaxH3TransformerBlock(**block_kwargs)
    block.load_torch_state_dict(state_dict)

    real_in = upload(real, adaln, positions)
    padded_in = upload(padded, padded_adaln, padded_positions)
    window = _windows(mesh_device, [0, seq_len, PADDED_LEN])

    # ---- the numerics, on the attention alone -------------------------------------------------
    # `block.attn` without `addcmul_residual` returns the attention branch on its own. The block's
    # gated residual would otherwise dominate: the same comparison on the block output moves by only
    # a couple of percent even when the attention branch has lost half its magnitude.
    def attn_only(inputs, *, rows, windows):
        return read(
            block.attn(
                inputs["spatial"],
                logical_n=_logical_n(mesh_device, seq_len),
                rope_cos=inputs["rope_cos"],
                rope_sin=inputs["rope_sin"],
                cu_window_seqlens=windows,
            ),
            rows,
        )

    reference = attn_only(real_in, rows=seq_len, windows=None)
    windowed = attn_only(padded_in, rows=seq_len, windows=window)
    unwindowed = attn_only(padded_in, rows=seq_len, windows=None)

    rmse_windowed = _rel_rmse(reference, windowed)
    rmse_unwindowed = _rel_rmse(reference, unwindowed)
    logger.info(
        f"mesh={mesh_shape} attention over {seq_len} real rows padded to {PADDED_LEN}: "
        f"windowed rel-RMSE {rmse_windowed:.5f}, unwindowed rel-RMSE {rmse_unwindowed:.5f}"
    )

    assert rmse_windowed <= MAX_REL_RMSE_WINDOWED, (
        f"padded-with-window rel-RMSE {rmse_windowed:.5f} > {MAX_REL_RMSE_WINDOWED}: the pad rows are "
        "still reaching the real rows' attention"
    )
    assert rmse_unwindowed >= MIN_REL_RMSE_UNWINDOWED, (
        f"padded-without-window rel-RMSE {rmse_unwindowed:.5f} < {MIN_REL_RMSE_UNWINDOWED}: dropping the "
        "window no longer changes the result, so the windowed leg above proves nothing"
    )

    # ---- the plumbing, on the block -----------------------------------------------------------
    # The numerics above hold for an attention called directly with a window, which upstream already
    # supported. What was missing is the block passing one down at all, so assert that the block's
    # `cu_window_seqlens` reaches the attention: same inputs, two windows, different answers.
    block_windowed = read(
        block(
            padded_in["spatial"],
            _logical_n(mesh_device, seq_len),
            temb=tt_temb,
            adaln_indices=padded_in["adaln_indices"],
            rope_cos=padded_in["rope_cos"],
            rope_sin=padded_in["rope_sin"],
            cu_window_seqlens=window,
        ),
        seq_len,
    )
    block_unwindowed = read(
        block(
            padded_in["spatial"],
            _logical_n(mesh_device, seq_len),
            temb=tt_temb,
            adaln_indices=padded_in["adaln_indices"],
            rope_cos=padded_in["rope_cos"],
            rope_sin=padded_in["rope_sin"],
            cu_window_seqlens=None,
        ),
        seq_len,
    )
    block_delta = _rel_rmse(block_windowed, block_unwindowed)
    logger.info(f"mesh={mesh_shape} block output moves by rel-RMSE {block_delta:.5f} when the window is dropped")
    assert block_delta > 0.0, (
        "MiniMaxH3TransformerBlock produced the same output with and without cu_window_seqlens: it is "
        "not forwarding the argument to its attention"
    )
