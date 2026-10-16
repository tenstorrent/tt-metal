# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Attention-op benchmark for the Sol-Attn port, reproducing `~/h3_attn_timing` exactly.

Method (from h3_attention_timing_main_4x8.md): one `MiniMaxH3Attention` layer with random norm
weights, 4x8 Blackhole Galaxy, TP=4 / SP=8, `CCLManager(num_links=2, topology=Ring)`, `seq_local` in
{4736, 9184, 13664}; wall clock with `ttnn.synchronize_device` before/after; 1 compile run then
REPS timed runs; the module time covers QKV projection + fused norm/RoPE + ring-joint SDPA (incl. the
K/V all-gather) + output projection; the op-only time wraps the monkey-patched
`ttnn.transformer.ring_joint_scaled_dot_product_attention` call in two host syncs.

Modes:
  dense           -- the production path (baseline).
  sol             -- Sol-Attn path (`sol_attn=True`), tau from SOL_TAU (comma list), using the pre-pass
                     and sparse kernel; reports achieved chunk density next to every number.
  agather         -- the ring K/V all-gather alone (comm floor), via ring_attention_all_gather_async.

SOL_BENCH_PROFILE=1 runs the timed loop under the device profiler (TT_METAL_DEVICE_PROFILER=1 must be
set in the environment before the process starts) and flushes it with ReadDeviceProfiler.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention as TorchMiniMaxH3Attention
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import MiniMaxH3Attention, prepare_rope_tables
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.tensor import bf16_tensor, bf16_tensor_2dshard
from .common import GALAXY_RING, REAL_BLOCK_CONFIG, randomize_norm_weights, upload_rope
from .test_transformer_minimax_h3 import logical_length_tensor

NUM_ATTENTION_HEADS = REAL_BLOCK_CONFIG["num_attention_heads"]
ATTENTION_HEAD_DIM = REAL_BLOCK_CONFIG["attention_head_dim"]
HIDDEN_SIZE = REAL_BLOCK_CONFIG["hidden_size"]
QK_NORM_EPS = REAL_BLOCK_CONFIG["qk_norm_eps"]
from .common import ROPE_FREQ_DIM, ROPE_THETA  # noqa: E402

REPS = int(os.environ.get("SOL_BENCH_REPS", "3"))
WORKING_POINTS = {4736: "768p_5s", 9184: "768p_10s", 13664: "768p_15s"}


def _position_ids(seq_len: int) -> torch.Tensor:
    """A (t, h, w) grid covering `seq_len` rows: 768p-like 48x84 frames, t-major, truncated."""
    grid_h, grid_w = 48, 84
    grid_t = -(-seq_len // (grid_h * grid_w))
    coords = torch.meshgrid(torch.arange(grid_t), torch.arange(grid_h), torch.arange(grid_w), indexing="ij")
    return torch.stack([c.reshape(-1) for c in coords], dim=-1)[:seq_len]


def _timed(fn, mesh_device, reps):
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    out = fn()
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) * 1e3, out


@GALAXY_RING
@pytest.mark.parametrize("seq_local", [4736, 9184, 13664], ids=lambda s: WORKING_POINTS[s])
@pytest.mark.parametrize("mode", ["dense", "sol", "agather"])
def test_sol_bench_attention(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, seq_local, mode, reset_seeds):
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    seq_len = seq_local * sp_factor
    torch.manual_seed(0)

    torch_model = TorchMiniMaxH3Attention(
        hidden_size=HIDDEN_SIZE, heads=NUM_ATTENTION_HEADS, dim_head=ATTENTION_HEAD_DIM, qk_norm_eps=QK_NORM_EPS
    ).to(torch.float32)
    randomize_norm_weights(torch_model)
    position_ids = _position_ids(seq_len)
    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    rope_cos, rope_sin = rope(position_ids)
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

    # SOL_BENCH_CHUNKS="256x256" overrides the measured production chunk sizes (dense chunk-size sweep).
    chunks_env = os.environ.get("SOL_BENCH_CHUNKS")
    if chunks_env:
        qc, kc = (int(x) for x in chunks_env.split("x"))
        MiniMaxH3Attention.measured_sdpa_chunk_sizes = {
            k: (qc, kc) for k in MiniMaxH3Attention.measured_sdpa_chunk_sizes
        }
    taus = [float(t) for t in os.environ.get("SOL_TAU", "1.0").split(",")] if mode == "sol" else [None]
    results = []
    for tau in taus:
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
            **({"sol_attn": True, "sol_tau": tau} if mode == "sol" else {}),
        )
        tt_model.load_torch_state_dict(torch_model.state_dict())
        if mode == "sol":
            tt_model.sol_active = True
            tt_model.sol_logical_n = seq_len
            tt_model.sol_sink_tokens = int(os.environ.get("SOL_SINK_TOKENS", "1245"))
            tt_model.sol_report_density = True

        op_times = []
        orig_ring = ttnn.transformer.ring_joint_scaled_dot_product_attention

        def timed_ring(*args, **kwargs):
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            out = orig_ring(*args, **kwargs)
            ttnn.synchronize_device(mesh_device)
            op_times.append((time.perf_counter() - t0) * 1e3)
            return out

        def module_call():
            return tt_model(tt_spatial, logical_n=logical_n, rope_cos=tt_rope_cos, rope_sin=tt_rope_sin)

        if mode == "agather":
            # comm floor: the fused ring all-gather alone, on K/V-shaped tensors
            k_shape = (1, NUM_ATTENTION_HEADS // tp_factor, seq_local, ATTENTION_HEAD_DIM)
            k_dev = bf16_tensor(torch.randn(k_shape), device=mesh_device)
            v_dev = bf16_tensor(torch.randn(k_shape), device=mesh_device)

            def module_call():  # noqa: F811
                return ttnn.experimental.ring_attention_all_gather_async(
                    [k_dev, v_dev],
                    persistent_output_buffer=[
                        ccl_manager.get_ag_ping_pong_buffer(k_shape, 2, sp_axis, dtype=ttnn.bfloat16),
                        ccl_manager.get_ag_ping_pong_buffer(k_shape, 2, sp_axis, dtype=ttnn.bfloat16),
                    ],
                    dim=2,
                    multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(sp_axis),
                    num_links=num_links,
                    cluster_axis=sp_axis,
                    mesh_device=mesh_device,
                    topology=topology,
                    subdevice_id=ccl_manager.ccl_sub_device_id,
                )

        ttnn.transformer.ring_joint_scaled_dot_product_attention = timed_ring
        try:
            compile_ms, _ = _timed(module_call, mesh_device, 1)
            op_times.clear()
            # density comes from the compile run; the host readback must not be inside the timed reps
            density = getattr(tt_model, "sol_last_density", None)
            tt_model.sol_report_density = False
            profile = os.environ.get("SOL_BENCH_PROFILE") == "1"
            if profile:
                ttnn.ReadDeviceProfiler(mesh_device)
            module_ms = []
            for _ in range(REPS):
                ms, out = _timed(module_call, mesh_device, 1)
                module_ms.append(ms)
            if profile:
                ttnn.ReadDeviceProfiler(mesh_device)
        finally:
            ttnn.transformer.ring_joint_scaled_dot_product_attention = orig_ring

        rec = {
            "mode": mode,
            "chunks": chunks_env or "measured",
            "tau": tau,
            "seq_local": seq_local,
            "global_seq": seq_len,
            "compile_ms": compile_ms,
            "module_ms": module_ms,
            "op_ms": op_times,
            "chunk_density": density,
        }
        results.append(rec)
        logger.info(
            f"[{WORKING_POINTS[seq_local]}] {mode} chunks={chunks_env or 'measured'} tau={tau}: compile {compile_ms:.0f} ms; "
            f"module {' / '.join(f'{m:.2f}' for m in module_ms)} ms; op {' / '.join(f'{m:.2f}' for m in op_times)} ms; "
            f"density {density}"
        )
        del tt_model

    out_dir = Path(os.environ.get("SOL_BENCH_OUT", str(Path.home() / "sol_attn_work" / "bench")))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.jsonl", "a") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")
