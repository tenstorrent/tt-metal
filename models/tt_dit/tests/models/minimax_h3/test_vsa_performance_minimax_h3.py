# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device perf of one MiniMax-H3 transformer block with the VSA attention path at 768P.

The dense baseline is `test_performance_minimax_h3.py::test_minimax_h3_transformer_block_perf`
(same fixtures, same signposts). Run under `scripts/run_safe_pytest.sh --profile`, then
`tt-perf-report <csv> --start-signpost start --end-signpost stop`. Profile one duration at a time
with `-k`: a multi-parameter profiled run yields a CSV containing only the first parameter's ops.
"""

from __future__ import annotations

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM, MiniMaxH3RotaryPosEmbed
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock as TorchMiniMaxH3Block
from loguru import logger
from tracy import signpost

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import prepare_rope_tables
from ....models.transformers.minimax_h3.transformer_block_minimax_h3 import MiniMaxH3TransformerBlock
from ....models.transformers.minimax_h3.vsa_stages_minimax_h3 import MiniMaxH3VSAConfig, MiniMaxH3VSACoarseStage
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3.vsa_geometry import build_vsa_geometry
from ....utils.tensor import bf16_tensor_2dshard, from_torch, local_device_to_torch
from ....utils.test import skip_if_unsupported_num_links
from .common import GALAXY_RING, REAL_BLOCK_CONFIG, ROPE_FREQ_DIM, ROPE_THETA, TT_BLOCK_CONFIG, upload_rope
from .test_performance_minimax_h3 import NUM_TEXT_TOKENS, _packed_sizes

HIDDEN_SIZE = REAL_BLOCK_CONFIG["hidden_size"]
HEAD_DIM = REAL_BLOCK_CONFIG["attention_head_dim"]
TIME_EMBED_DIM = REAL_BLOCK_CONFIG["time_embed_dim"]

SPARSITY = 0.9  # the production working point
K_CHUNK_BLOCKS = 2


@GALAXY_RING
@pytest.mark.parametrize(
    "duration_s",
    [
        pytest.param(5.0, id="5s_768p"),
        pytest.param(10.0, id="10s_768p"),
        pytest.param(15.0, id="15s_768p"),
    ],
)
@pytest.mark.timeout(2700)
def test_minimax_h3_vsa_block_perf(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    duration_s: float,
    is_fsdp: bool,
    topology: ttnn.Topology,
    reset_seeds,
) -> None:
    skip_if_unsupported_num_links(mesh_device, num_links)
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    if (sp_factor, tp_factor) != (8, 4):
        pytest.skip("VSA v0 targets 4x8 (TP=4/SP=8)")

    sizes = _packed_sizes(duration_s)
    grid = (sizes["latent_frames"], sizes["grid_h"], sizes["grid_w"])
    # audio rows kept even (stereo channel-major); _packed_sizes reports latents, rows are 2x... it
    # reports num_audio directly -- match the dense perf test's row count exactly.
    prefix_segments = (sizes["num_text"], 0, sizes["num_audio"])
    geometry = build_vsa_geometry(prefix_segments, grid, sp_factor=sp_factor)
    logger.info(
        f"{duration_s:g}s @ 768P: seq_len={geometry.seq_len} -> {geometry.padded_len} tiled "
        f"({geometry.n_tiles} tiles, {geometry.n_pad_tiles} pad, {geometry.padded_len // sp_factor} rows/device)"
    )

    num_timesteps = 2
    # per-row metadata in packed order, then permuted to tile order
    tags = torch.cat(
        [
            torch.ones(sizes["num_text"], dtype=torch.long),
            2 * torch.ones(sizes["num_audio"], dtype=torch.long),
            torch.zeros(sizes["num_video"], dtype=torch.long),
        ]
    )
    ts_idx = torch.cat(
        [
            torch.zeros(sizes["num_text"], dtype=torch.long),
            torch.ones(sizes["num_audio"], dtype=torch.long),
            torch.ones(sizes["num_video"], dtype=torch.long),
        ]
    )
    adaln = ts_idx * MINIMAX_H3_MODALITY_NUM + tags
    position_ids = torch.zeros(geometry.seq_len, 3, dtype=torch.float64)
    video = torch.meshgrid(*(torch.arange(d) for d in grid), indexing="ij")
    position_ids[sizes["num_text"] + sizes["num_audio"] :] = torch.stack(
        [c.reshape(-1) for c in video], dim=-1
    ).to(torch.float64)

    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    with torch.no_grad():
        rope_cos, rope_sin = rope(geometry.permute_metadata(position_ids, dim=0))
    rope_cos, rope_sin = prepare_rope_tables(rope_cos, rope_sin, HEAD_DIM)

    torch_block = TorchMiniMaxH3Block(**REAL_BLOCK_CONFIG).to(torch.float32)
    state = dict(torch_block.state_dict())
    torch.manual_seed(4)
    # nonzero gate: the perf number must include the full coarse-output branch
    state["attn.to_gate_compress.weight"] = 0.02 * torch.randn(56 * HEAD_DIM, HIDDEN_SIZE)
    del torch_block

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )
    vsa_config = MiniMaxH3VSAConfig(sparsity=SPARSITY, k_chunk_blocks=K_CHUNK_BLOCKS)
    tt_block = MiniMaxH3TransformerBlock(
        **TT_BLOCK_CONFIG,
        rotary_dim=2 * 3 * ROPE_FREQ_DIM,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        vsa_config=vsa_config,
    )
    tt_block.load_torch_state_dict(state)
    stage = MiniMaxH3VSACoarseStage(
        geometry,
        sparsity=SPARSITY,
        head_dim=HEAD_DIM,
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        ccl_manager=ccl_manager,
    )
    tt_block.attn.set_vsa_stage(stage)
    logger.info(f"selection: k={stage.k} of {stage.n_candidates} candidates, W={stage.index_width}")

    tt_spatial = bf16_tensor_2dshard(
        torch.randn(1, 1, geometry.padded_len, HIDDEN_SIZE),
        device=mesh_device,
        shard_mapping={sp_axis: 2, tp_axis: 3},
    )
    tt_temb = from_torch(torch.randn(1, 1, num_timesteps, TIME_EMBED_DIM), device=mesh_device, dtype=ttnn.float32)
    tt_adaln = from_torch(
        geometry.permute_metadata(adaln, dim=0).to(torch.int32).reshape(1, 1, 1, geometry.padded_len),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.Layout.ROW_MAJOR,
        mesh_axes=[..., None, sp_axis],
    )
    tt_rope_cos, tt_rope_sin = upload_rope(rope_cos, rope_sin, mesh_device=mesh_device, sp_axis=sp_axis)

    def run_block() -> ttnn.Tensor:
        out = tt_block(
            tt_spatial,
            N=geometry.padded_len,
            temb=tt_temb,
            adaln_indices=tt_adaln,
            rope_cos=tt_rope_cos,
            rope_sin=tt_rope_sin,
        )
        ttnn.synchronize_device(mesh_device)
        return out

    logger.info("iteration 1: compiling kernels and populating the program cache")
    run_block()

    logger.info("iteration 2: warm run (the profiled region)")
    signpost("start")
    tt_out = run_block()
    signpost("stop")

    local = local_device_to_torch(tt_out).float()
    assert torch.isfinite(local).all(), "block output contains NaN or Inf"
    logger.info(f"output {tuple(tt_out.shape)}, local shard std={local.std().item():.4f}")
