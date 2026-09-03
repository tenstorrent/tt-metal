# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""R5: VSA transformer block at production shape, traced and untraced, on the 4x8 galaxy.

The scope check is completion, not parity (parity is pinned at attention level and by R6): the
traced forward must capture and replay at 15 s / 768p with the full VSA graph -- coarse stage,
top-k selection, index assembly, vsa_sdpa, and the gate branch -- and produce finite output.
"""

import os

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM, MiniMaxH3RotaryPosEmbed
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock as TorchMiniMaxH3Block
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import prepare_rope_tables
from ....models.transformers.minimax_h3.transformer_block_minimax_h3 import MiniMaxH3TransformerBlock
from ....models.transformers.minimax_h3.vsa_stages_minimax_h3 import MiniMaxH3VSACoarseStage, MiniMaxH3VSAConfig
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3.vsa_geometry import build_vsa_geometry
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor_2dshard, from_torch
from ....utils.test import ring_params_8k_req_exact_devices, skip_if_unsupported_num_links
from .common import (
    REAL_BLOCK_CONFIG,
    ROPE_FREQ_DIM,
    ROPE_THETA,
    TT_BLOCK_CONFIG,
    packed_layout,
    randomize_norm_weights,
    upload_rope,
)

HEAD_DIM = REAL_BLOCK_CONFIG["attention_head_dim"]
HIDDEN_SIZE = REAL_BLOCK_CONFIG["hidden_size"]
TIME_EMBED_DIM = REAL_BLOCK_CONFIG["time_embed_dim"]

# 15 s / 768p production sizes: 362 frames -> 107 latent frames, 768x1344 canvas -> (24, 42)
# patch grid; text 512; audio at 40 latents/s -> 604 rows (kept even for the stereo layout).
_15S = dict(num_text=512, num_audio=604, grid=(107, 24, 42))

_ring_8k_trace = {**ring_params_8k_req_exact_devices, "trace_region_size": 150_000_000, "l1_small_size": 65536}

GALAXY_4X8_TRACE = pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param((4, 8), 1, 0, 2, _ring_8k_trace, ttnn.Topology.Ring, False, id="4x8sp1tp0nl2_ring_trace"),
    ],
    indirect=["mesh_device", "device_params"],
)


@GALAXY_4X8_TRACE
@pytest.mark.parametrize("traced", [False, True], ids=["untraced", "traced"])
@pytest.mark.timeout(2700)
def test_vsa_block_15s_768p(mesh_device, sp_axis, tp_axis, num_links, is_fsdp, topology, traced, reset_seeds) -> None:
    skip_if_unsupported_num_links(mesh_device, num_links)
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    if (sp_factor, tp_factor) != (8, 4):
        pytest.skip("VSA v0 targets 4x8 (TP=4/SP=8)")

    num_text, num_audio, grid = _15S["num_text"], _15S["num_audio"], _15S["grid"]
    t, gh, gw = grid
    num_video = t * gh * gw
    seq_len = num_text + num_audio + num_video

    placement = os.environ.get("VSA_PLACEMENT", "identity")  # see test_vsa_performance_minimax_h3
    geometry = build_vsa_geometry((num_text, 0, num_audio), grid, sp_factor=sp_factor, placement=placement)
    logger.info(
        f"15s/768p: seq_len={seq_len}, tiles={geometry.n_tiles} ({geometry.n_pad_tiles} pad), "
        f"padded_len={geometry.padded_len} ({geometry.padded_len // sp_factor} rows/device)"
    )

    num_timesteps = 2
    position_ids, tags, timestep_indices = packed_layout(num_text, num_audio, num_video, (gh, gw), padded_len=None)
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + tags.clamp(min=0)

    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    with torch.no_grad():
        rope_cos, rope_sin = rope(position_ids)
    # tiled order: permute per-row metadata (pad slots replicate a valid row of their tile)
    rope_cos = geometry.permute_metadata(rope_cos, dim=0)
    rope_sin = geometry.permute_metadata(rope_sin, dim=0)
    adaln_tiled = geometry.permute_metadata(adaln_indices, dim=0)
    rope_cos, rope_sin = prepare_rope_tables(rope_cos, rope_sin, HEAD_DIM)

    torch_block = TorchMiniMaxH3Block(**REAL_BLOCK_CONFIG).to(torch.float32)
    randomize_norm_weights(torch_block)
    state = dict(torch_block.state_dict())
    torch.manual_seed(4)
    # random nonzero gate so the trace covers the full coarse-output branch
    state["attn.to_gate_compress.weight"] = 0.02 * torch.randn(56 * HEAD_DIM, HIDDEN_SIZE)
    del torch_block

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )
    # VSA_KERNEL=v1 selects the per-row gather kernel (bisecting trace issues against the stream path)
    vsa_config = MiniMaxH3VSAConfig(
        sparsity=0.9,
        k_chunk_blocks=2,
        streaming=os.environ.get("VSA_KERNEL", "stream") != "v1",
        placement=placement,
    )
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
    assert not tt_block.attn.gate_compress_is_zero
    stage = MiniMaxH3VSACoarseStage(
        geometry,
        sparsity=vsa_config.sparsity,
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
        adaln_tiled.to(torch.int32).reshape(1, 1, 1, geometry.padded_len),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.Layout.ROW_MAJOR,
        mesh_axes=[..., None, sp_axis],
    )
    tt_rope_cos, tt_rope_sin = upload_rope(rope_cos, rope_sin, mesh_device=mesh_device, sp_axis=sp_axis)

    def run_block() -> ttnn.Tensor:
        return tt_block(
            tt_spatial,
            N=geometry.padded_len,
            temb=tt_temb,
            adaln_indices=tt_adaln,
            rope_cos=tt_rope_cos,
            rope_sin=tt_rope_sin,
        )

    logger.info("compile run")
    tt_out = run_block()
    ttnn.synchronize_device(mesh_device)
    untraced_local = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
    # VSA_REPEAT=n: extra untraced runs exercise the program-cache-hit path (fresh tensor
    # addresses, override_runtime_arguments) without trace in the picture.
    for i in range(int(os.environ.get("VSA_REPEAT", "1")) - 1):
        logger.info(f"untraced repeat {i + 1}")
        tt_out = run_block()
        ttnn.synchronize_device(mesh_device)
        rep_local = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
        assert_quality(untraced_local, rep_local, pcc=0.998)

    if traced:
        logger.info("capturing trace")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_out = run_block()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        logger.info("executing trace")
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        ttnn.release_trace(mesh_device, trace_id)

    local = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
    assert torch.isfinite(local).all(), "block output contains NaN or Inf"
    logger.info(f"output {tuple(tt_out.shape)}, local shard std={local.std().item():.4f}")
    if traced:
        # the replay must reproduce the untraced block (bf16 rounding-order noise only)
        assert_quality(untraced_local, local, pcc=0.998)
