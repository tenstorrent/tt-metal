# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Correctness of the separate prefix-row recompute (`MiniMaxH3Attention._sol_prefix_dense_rows`).

One attention layer, 4x8 Galaxy, TP=4 / SP=8, random weights and input. Runs the module dense and in Sol-Attn
mode with the three prefix modes and checks, on the host:
  - rows outside the prefix are bit-identical across the three prefix modes (same sparse op, same mask there);
  - the prefix rows from "separate" match the dense output as well as "fused" does (both are exact, bf16 noise
    apart), and both beat "off";
  - module wall time per mode, so the recompute's cost is visible next to the fused rule's straggler.

    SOL_PREFIX_SEQ=4736|9184|13664 (default 4736), SOL_TAU (default 2.0), SOL_SINK_TOKENS (default 1245).
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
from ....models.transformers.minimax_h3.sol_prepass import BLOCK
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


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@GALAXY_RING
def test_sol_prefix_rows(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds):
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    seq_local = int(os.environ.get("SOL_PREFIX_SEQ", "4736"))
    seq_len = seq_local * sp_factor
    sink = int(os.environ.get("SOL_SINK_TOKENS", "1245"))
    tau = float(os.environ.get("SOL_TAU", "2.0"))
    n_pre = -(-sink // BLOCK) * BLOCK
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
    )  # [sp, tp, N_local, hidden / tp]

    outs, times = {}, {}
    for prefix_mode in ("dense", "off", "fused", "separate"):
        sol = prefix_mode != "dense"
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
            **({"sol_attn": True, "sol_tau": tau, "sol_prefix_mode": prefix_mode} if sol else {}),
        )
        tt_model.load_torch_state_dict(torch_model.state_dict())
        if sol:
            tt_model.sol_active = True
            tt_model.sol_logical_n = seq_len
            tt_model.sol_sink_tokens = sink
        ms = []
        for _ in range(3):  # compile run + 2 timed
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            out = tt_model(tt_spatial, logical_n=logical_n, rope_cos=tt_rope_cos, rope_sin=tt_rope_sin)
            ttnn.synchronize_device(mesh_device)
            ms.append((time.perf_counter() - t0) * 1e3)
            host = to_host(out).float()
            ttnn.deallocate(out)
        outs[prefix_mode] = host
        times[prefix_mode] = ms
        logger.info(f"[prefix] mode={prefix_mode}: module wall ms {' / '.join(f'{m:.2f}' for m in ms)}")
        del tt_model

    pre = lambda name: outs[name][0, :, :n_pre]  # shard 0's prefix rows, all TP slices
    rest = lambda name: torch.cat([outs[name][0, :, n_pre:].flatten(), outs[name][1:].flatten()])

    # Outside the prefix the three sparse variants run the same sparse op on the same mask.
    for a, b in (("off", "separate"), ("off", "fused")):
        diff = (rest(a) - rest(b)).abs().max().item()
        logger.info(f"[prefix] non-prefix rows {a} vs {b}: max abs diff {diff:.3g}")
        assert diff < 1e-2, (a, b, diff)

    p_sep, p_fused, p_off = (
        _pcc(pre("separate"), pre("dense")),
        _pcc(pre("fused"), pre("dense")),
        _pcc(pre("off"), pre("dense")),
    )
    p_sf = _pcc(pre("separate"), pre("fused"))
    logger.info(
        f"[prefix] prefix rows vs dense: separate {p_sep:.5f}, fused {p_fused:.5f}, off {p_off:.5f}; separate vs fused {p_sf:.5f}"
    )
    assert p_sep > 0.995, p_sep
    assert p_fused > 0.995, p_fused
    assert p_sf > 0.995, p_sf
    assert p_sep >= p_off - 1e-3, "the separate recompute should not be worse than no recompute"
    logger.info(
        "[prefix] module wall (best timed run) ms: " + ", ".join(f"{k} {min(v[1:]):.2f}" for k, v in times.items())
    )
