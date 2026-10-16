# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""One MiniMax-H3 transformer block under Tracy, dense or Sol-Attn, for the per-op breakdown.

Same recipe as `test_minimax_h3_transformer_block_perf` (compile run, then one warm run between the
`start`/`stop` signposts), at the packed 10 s / 768p geometry with the 39-token calibrated prompt, random
block weights/inputs. Sol config from env: SOL_BP_MODE (dense | gather | chunk), SOL_BP_TAU (1.0),
SOL_BP_DENSE_PREFIX (0/1, default 0), SOL_BP_DURATION (10).

    python -m tracy -r -p --no-web-server -m pytest \
        models/tt_dit/tests/models/minimax_h3/test_sol_block_perf_minimax_h3.py -s -k 4x8
"""

from __future__ import annotations

import os

import torch
from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM, MiniMaxH3RotaryPosEmbed
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock as TorchMiniMaxH3Block
from loguru import logger
from tracy import signpost

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import prepare_rope_tables
from ....models.transformers.minimax_h3.transformer_block_minimax_h3 import MiniMaxH3TransformerBlock
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3.packing import padded_sequence_length
from ....utils.tensor import bf16_tensor_2dshard, from_torch, local_device_to_torch
from ....utils.test import skip_if_unsupported_num_links
from .common import (
    GALAXY_RING,
    REAL_BLOCK_CONFIG,
    ROPE_FREQ_DIM,
    ROPE_THETA,
    TT_BLOCK_CONFIG,
    packed_layout,
    upload_rope,
)
from .common_av import CALIBRATED_FOX_PROMPT_NUM_TOKENS
from .test_transformer_minimax_h3 import (
    ATTENTION_HEAD_DIM,
    HIDDEN_SIZE,
    TIME_EMBED_DIM,
    _packed_sizes,
    logical_length_tensor,
)


@GALAXY_RING
def test_sol_block_perf(mesh_device, sp_axis, tp_axis, num_links, is_fsdp, topology, reset_seeds):
    skip_if_unsupported_num_links(mesh_device, num_links)
    mode = os.environ.get("SOL_BP_MODE", "dense")
    tau = float(os.environ.get("SOL_BP_TAU", "1.0"))
    dense_prefix = os.environ.get("SOL_BP_DENSE_PREFIX", "0") == "1"
    duration_s = float(os.environ.get("SOL_BP_DURATION", "10"))

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    sizes = _packed_sizes(duration_s, CALIBRATED_FOX_PROMPT_NUM_TOKENS)
    seq_len = sizes["seq_len"]
    padded_len = padded_sequence_length(seq_len, sp_factor)
    sink_tokens = sizes["num_text"] + sizes["num_audio"]  # t2va prefix: text + audio precede the video rows
    logger.info(
        f"{duration_s:g}s: seq_len {seq_len} padded {padded_len} ({padded_len // sp_factor} rows/device), "
        f"sink {sink_tokens}, mode {mode}, tau {tau}, dense_prefix {dense_prefix}"
    )

    num_timesteps = 2
    position_ids, tags, timestep_indices = packed_layout(
        sizes["num_text"], sizes["num_audio"], sizes["num_video"], (sizes["grid_h"], sizes["grid_w"]), padded_len
    )
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + tags.clamp(min=0)
    torch_block = TorchMiniMaxH3Block(**REAL_BLOCK_CONFIG).to(torch.float32)
    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    with torch.no_grad():
        rope_cos, rope_sin = rope(position_ids)
    rope_cos, rope_sin = prepare_rope_tables(rope_cos, rope_sin, ATTENTION_HEAD_DIM)

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )
    tt_block = MiniMaxH3TransformerBlock(
        **TT_BLOCK_CONFIG,
        rotary_dim=2 * 3 * ROPE_FREQ_DIM,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )
    tt_block.load_torch_state_dict(torch_block.state_dict())
    del torch_block
    attn = tt_block.attn
    if mode != "dense":
        attn.sol_attn = True
        attn.sol_active = True
        attn.sol_tau = tau
        attn.sol_thresh_type = "diag"
        attn.sol_band = 1
        attn.sol_dense_prefix = dense_prefix
        attn.sol_mode = mode
        attn.sol_logical_n = seq_len
        attn.sol_sink_tokens = sink_tokens
        attn._sol_prepass = None

    tt_spatial = bf16_tensor_2dshard(
        torch.randn(1, 1, padded_len, HIDDEN_SIZE), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3}
    )
    tt_temb = from_torch(torch.randn(1, 1, num_timesteps, TIME_EMBED_DIM), device=mesh_device, dtype=ttnn.float32)
    tt_adaln = from_torch(
        adaln_indices.to(torch.int32).reshape(1, 1, 1, padded_len),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.Layout.ROW_MAJOR,
        mesh_axes=[..., None, sp_axis],
    )
    tt_rope_cos, tt_rope_sin = upload_rope(rope_cos, rope_sin, mesh_device=mesh_device, sp_axis=sp_axis)

    def run_block():
        out = tt_block(
            tt_spatial,
            logical_length_tensor(mesh_device, seq_len),
            temb=tt_temb,
            adaln_indices=tt_adaln,
            rope_cos=tt_rope_cos,
            rope_sin=tt_rope_sin,
        )
        ttnn.synchronize_device(mesh_device)
        return out

    logger.info("iteration 1: compiling kernels and populating the program cache")
    if mode != "dense":
        attn.sol_report_density = True
    run_block()
    if mode != "dense":
        logger.info(f"kept density ({mode}, tau {tau}): {attn.sol_last_density:.4f}")
        attn.sol_report_density = False
    logger.info("iteration 2: warm run (the profiled region)")
    signpost("start")
    tt_out = run_block()
    signpost("stop")
    local = local_device_to_torch(tt_out).float()
    assert torch.isfinite(local).all(), "block output contains NaN or Inf"
    logger.info(f"output {tuple(tt_out.shape)}, local shard std={local.std().item():.4f}")

    # Host-vs-device check (SOL_BP_HOST_REPS=N): enqueue N blocks back to back without syncing and time the host
    # enqueue alone, then sync and time the whole thing. If enqueue/block ~= wall/block the loop is host-bound.
    host_reps = int(os.environ.get("SOL_BP_HOST_REPS", "0"))
    if host_reps > 0:
        import time

        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        x = tt_spatial
        for _ in range(host_reps):
            # The fused FF2 addcmul returns the residual buffer in place, so chain it instead of freeing it.
            x = tt_block(
                x,
                logical_length_tensor(mesh_device, seq_len),
                temb=tt_temb,
                adaln_indices=tt_adaln,
                rope_cos=tt_rope_cos,
                rope_sin=tt_rope_sin,
            )
        t_enq = time.perf_counter() - t0
        ttnn.synchronize_device(mesh_device)
        t_all = time.perf_counter() - t0
        logger.info(
            f"HOSTCHECK mode={mode} tau={tau}: host enqueue {1e3 * t_enq / host_reps:.1f} ms/block, "
            f"wall incl. device drain {1e3 * t_all / host_reps:.1f} ms/block over {host_reps} blocks"
        )
