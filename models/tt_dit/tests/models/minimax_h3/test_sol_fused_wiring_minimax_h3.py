# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""End-to-end wiring check of the fused scoring step: one MiniMaxH3Attention module (random weights) run NUM_LAYERS
times with sol_fused=False (ttnn pre-pass) and sol_fused=True (fused programs); the sparse op's block mask of every
forward is captured and compared on host, as are the module outputs. Synthetic inputs, 5 s geometry by default.

  pytest models/tt_dit/tests/models/minimax_h3/test_sol_fused_wiring_minimax_h3.py -k 4x8sp1tp0nl2_ring_is_fsdp0 -s
Env: SOL_WIRING_LAYERS (default 10), SOL_WIRING_SEQ (rows per chip, default 4736), SOL_SINK_TOKENS (default 453).
"""

from __future__ import annotations

import os
import time

import torch
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention as TorchMiniMaxH3Attention
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import MiniMaxH3Attention, prepare_rope_tables
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.tensor import bf16_tensor_2dshard
from .common import GALAXY_RING, REAL_BLOCK_CONFIG, ROPE_FREQ_DIM, ROPE_THETA, randomize_norm_weights, upload_rope
from .test_sol_bench_minimax_h3 import _position_ids
from .test_transformer_minimax_h3 import logical_length_tensor

NUM_ATTENTION_HEADS = REAL_BLOCK_CONFIG["num_attention_heads"]
ATTENTION_HEAD_DIM = REAL_BLOCK_CONFIG["attention_head_dim"]
HIDDEN_SIZE = REAL_BLOCK_CONFIG["hidden_size"]
QK_NORM_EPS = REAL_BLOCK_CONFIG["qk_norm_eps"]


@GALAXY_RING
def test_sol_fused_wiring(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds):
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    seq_local = int(os.environ.get("SOL_WIRING_SEQ", "4736"))
    seq_len = seq_local * sp_factor
    n_layers = int(os.environ.get("SOL_WIRING_LAYERS", "10"))
    sink = int(os.environ.get("SOL_SINK_TOKENS", "453"))
    torch.manual_seed(0)

    torch_model = TorchMiniMaxH3Attention(
        hidden_size=HIDDEN_SIZE, heads=NUM_ATTENTION_HEADS, dim_head=ATTENTION_HEAD_DIM, qk_norm_eps=QK_NORM_EPS
    ).to(torch.float32)
    randomize_norm_weights(torch_model)
    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    rope_cos, rope_sin = rope(_position_ids(seq_len))
    rotary_dim = rope_cos.shape[-1]
    tt_rope_cos_t, tt_rope_sin_t = prepare_rope_tables(rope_cos, rope_sin, ATTENTION_HEAD_DIM)
    spatial_input = torch.randn((1, seq_len, HIDDEN_SIZE), dtype=torch.float32)
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )
    tt_spatial = bf16_tensor_2dshard(
        spatial_input.unsqueeze(0), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3}
    )
    tt_rope_cos, tt_rope_sin = upload_rope(tt_rope_cos_t, tt_rope_sin_t, mesh_device=mesh_device, sp_axis=sp_axis)
    logical_n = logical_length_tensor(mesh_device, seq_len)
    dims = [None, None]
    dims[sp_axis] = 0
    dims[tp_axis] = 1
    to_host = lambda t: ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))
    )

    runs = {}
    orig_ring = ttnn.transformer.sparse_ring_attention
    for fused in (False, True):
        tt_model = MiniMaxH3Attention(
            hidden_size=HIDDEN_SIZE,
            num_heads=NUM_ATTENTION_HEADS,
            head_dim=ATTENTION_HEAD_DIM,
            rotary_dim=rotary_dim,
            qk_norm_eps=QK_NORM_EPS,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            sol_attn=True,
            sol_tau=1.0,
            sol_fused=fused,
        )
        tt_model.load_torch_state_dict(torch_model.state_dict())
        tt_model.sol_active = True
        tt_model.sol_logical_n = seq_len
        tt_model.sol_sink_tokens = sink
        masks, outs, times = [], [], []

        def capture_ring(*args, **kwargs):
            masks.append(to_host(args[3]).float())  # block_mask
            return orig_ring(*args, **kwargs)

        ttnn.transformer.sparse_ring_attention = capture_ring
        try:
            for layer in range(
                n_layers
            ):  # the same module stands in for n_layers layers (fresh pre-pass outputs each call)
                ttnn.synchronize_device(mesh_device)
                t0 = time.perf_counter()
                out = tt_model(tt_spatial, logical_n=logical_n, rope_cos=tt_rope_cos, rope_sin=tt_rope_sin)
                ttnn.synchronize_device(mesh_device)
                times.append((time.perf_counter() - t0) * 1e3)
                outs.append(to_host(out).float())
                ttnn.deallocate(out)
        finally:
            ttnn.transformer.sparse_ring_attention = orig_ring
        runs[fused] = (masks, outs, times)
        logger.info(
            f"sol_fused={fused}: {n_layers} forwards, module wall ms (first / median): {times[0]:.1f} / {sorted(times)[len(times) // 2]:.1f}; mask shape {tuple(masks[0].shape)}"
        )

    m_ttnn, o_ttnn, _ = runs[False]
    m_fused, o_fused, _ = runs[True]
    assert len(m_fused) == n_layers == len(m_ttnn), (len(m_fused), len(m_ttnn))
    for i in range(n_layers):
        agree = (m_fused[i] == m_ttnn[i]).float().mean().item()
        same_as_first = (m_fused[i] == m_fused[0]).float().mean().item()
        a, b = o_fused[i].flatten().double(), o_ttnn[i].flatten().double()
        pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
        logger.info(
            f"layer {i}: mask fused-vs-ttnn agreement {agree:.4f} (fused density {m_fused[i].mean():.4f}, ttnn {m_ttnn[i].mean():.4f}), fused mask identical to layer 0: {same_as_first:.4f}, output PCC {pcc:.5f}"
        )
        assert agree > 0.98, (i, agree)
        assert same_as_first == 1.0, "the fused pre-pass must be deterministic across repeated calls"
        assert pcc > 0.995, (i, pcc)
