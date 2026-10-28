# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Attention-op benchmark for the Sol-Attn port.

Method: one `MiniMaxH3Attention` layer with random norm
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
from .common import _BH_ONLY, GALAXY_RING, REAL_BLOCK_CONFIG, _ring_8k_trace, randomize_norm_weights, upload_rope
from .test_transformer_minimax_h3 import logical_length_tensor

NUM_ATTENTION_HEADS = REAL_BLOCK_CONFIG["num_attention_heads"]
ATTENTION_HEAD_DIM = REAL_BLOCK_CONFIG["attention_head_dim"]
HIDDEN_SIZE = REAL_BLOCK_CONFIG["hidden_size"]
QK_NORM_EPS = REAL_BLOCK_CONFIG["qk_norm_eps"]
from .common import ROPE_FREQ_DIM, ROPE_THETA  # noqa: E402

REPS = int(os.environ.get("SOL_BENCH_REPS", "3"))
WORKING_POINTS = {4736: "768p_5s", 9184: "768p_10s", 13664: "768p_15s"}

# SOL_BENCH_TRACE=1: open the 4x8 mesh with a trace region and time the attention op as a captured trace replayed
# REPS times, so the measurement is the device time plus one synchronize, with no per-call host dispatch (the
# production model runs under trace; the untraced "op" time includes the host dispatch of the fused op). The module
# time is then one captured run and meaningless; the per-rep density readback is from that run. Not combinable with
# SOL_BENCH_PROFILE.
TRACE = os.environ.get("SOL_BENCH_TRACE") == "1"
_BENCH_RING = (
    pytest.mark.parametrize(
        ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
        [
            pytest.param(
                (4, 8),
                1,
                0,
                2,
                _ring_8k_trace,
                ttnn.Topology.Ring,
                False,
                id="4x8sp1tp0nl2_ring_is_fsdp0",
                marks=_BH_ONLY,
            )
        ],
        indirect=["mesh_device", "device_params"],
    )
    if TRACE
    else GALAXY_RING
)


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


@_BENCH_RING
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
        MiniMaxH3Attention.measured_resident_k_chunks = {}  # the override sets resident's k too
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
            # SOL_BENCH_MODE: gather | chunk | compact | pack (pack = compact + page packing; pair it with
            # SOL_BENCH_CHUNKS=64xK: packing needs the 64-row Q chunk, the only size whose L1 fits the staging).
            tt_model.sol_mode = os.environ.get("SOL_BENCH_MODE", "compact")
            # SOL_BENCH_PREFIX: how the prefix (text + condition) query rows get their exact attention.
            #   separate: the sparse op treats them like any row; a separate dense op recomputes them
            #             (MiniMaxH3Attention._sol_prefix_dense_rows). Its cost is in the module time and in the
            #             "op" time below (which wraps the ring op + that recompute).
            #   fused:    the pre-pass marks them dense inside the sparse op. The fully dense query chunks land on
            #             a few cores of each shard-0 chip, and the chain lockstep drags those chips to dense pace.
            #   off:      no exact recompute (the floor: the sparse op alone). The default, so the number isolates
            #             the sparse kernel; pass SOL_BENCH_PREFIX=separate to include the prefix recompute.
            # SOL_BENCH_DENSE_PREFIX=0 is the old spelling of "off".
            prefix = os.environ.get("SOL_BENCH_PREFIX", "off")
            if os.environ.get("SOL_BENCH_DENSE_PREFIX", "1") == "0":
                prefix = "off"
            tt_model.sol_prefix_mode = prefix

        op_times = []
        orig_ring = ttnn.transformer.ring_joint_scaled_dot_product_attention

        # The "op" time wraps the ring op and, in prefix mode "separate", the prefix recompute that follows it
        # (both are the attention proper; everything else in the module is projections and norms).
        orig_prefix = MiniMaxH3Attention._sol_prefix_dense_rows
        prefix_times = []

        trace_armed = False  # set after the compile run: a trace capture needs the program in the cache

        def timed_ring(*args, **kwargs):
            if not (TRACE and trace_armed):
                ttnn.synchronize_device(mesh_device)
                t0 = time.perf_counter()
                out = orig_ring(*args, **kwargs)
                ttnn.synchronize_device(mesh_device)
                op_times.append((time.perf_counter() - t0) * 1e3)
                return out
            # Traced: capture this call, then replay it REPS times while its inputs are still live (the replays
            # read the same input buffers and write the same output buffer), timing each replay.
            ttnn.synchronize_device(mesh_device)
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            out = orig_ring(*args, **kwargs)
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            for _ in range(REPS):
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                op_times.append((time.perf_counter() - t0) * 1e3)
            ttnn.release_trace(mesh_device, trace_id)
            return out

        def timed_prefix(self, *args, **kwargs):
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            out = orig_prefix(self, *args, **kwargs)
            ttnn.synchronize_device(mesh_device)
            ms = (time.perf_counter() - t0) * 1e3
            prefix_times.append(ms)
            op_times[-1] += ms
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
        MiniMaxH3Attention._sol_prefix_dense_rows = timed_prefix
        try:
            compile_ms, _ = _timed(module_call, mesh_device, 1)
            op_times.clear()
            prefix_times.clear()
            trace_armed = True
            # density comes from the compile run; the host readback must not be inside the timed reps
            density = getattr(tt_model, "sol_last_density", None)
            # 64x64-block density of the same mask (resident / chunk modes skip at the coarser K-chunk level, so
            # the two differ there; identical in the block-form modes). This is the sparsity the model requires.
            block_density = getattr(tt_model, "sol_last_block_density", None)
            # The per-call density readback stays on during the timed reps: the resident op's time tracks the mask
            # density call by call, and a mask that changes between calls on identical inputs points at a corrupted
            # pre-pass gather. The readback sits in the module time, never in the op time.
            densities = []
            block_densities = []
            profile = os.environ.get("SOL_BENCH_PROFILE") == "1" and not TRACE
            if profile:
                ttnn.ReadDeviceProfiler(mesh_device)
            module_ms = []
            if TRACE and mode == "agather":
                # The gather alone under trace: capture the call, replay it REPS times.
                ttnn.synchronize_device(mesh_device)
                trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                module_call()
                ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
                ttnn.synchronize_device(mesh_device)
                for _ in range(REPS):
                    t0 = time.perf_counter()
                    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh_device)
                    op_times.append((time.perf_counter() - t0) * 1e3)
                ttnn.release_trace(mesh_device, trace_id)
                module_ms = list(op_times)
            else:
                for _ in range(1 if TRACE else REPS):
                    ms, out = _timed(module_call, mesh_device, 1)
                    module_ms.append(ms)
                    densities.append(getattr(tt_model, "sol_last_density", None))
                    block_densities.append(getattr(tt_model, "sol_last_block_density", None))
            if profile:
                ttnn.ReadDeviceProfiler(mesh_device)
        finally:
            ttnn.transformer.ring_joint_scaled_dot_product_attention = orig_ring
            MiniMaxH3Attention._sol_prefix_dense_rows = orig_prefix

        rec = {
            "mode": mode,
            "chunks": chunks_env or "measured",
            "tau": tau,
            "seq_local": seq_local,
            "global_seq": seq_len,
            "compile_ms": compile_ms,
            "module_ms": module_ms,
            "op_ms": op_times,
            "prefix_ms": prefix_times,
            "prefix_mode": tt_model.sol_prefix_mode if mode == "sol" else None,
            "chunk_density": density,
            "chunk_density_per_rep": densities,
            "block_density": block_density,
            "block_density_per_rep": block_densities,
        }
        results.append(rec)
        prefix_note = (
            f" (incl. prefix recompute {' / '.join(f'{m:.2f}' for m in prefix_times)} ms)" if prefix_times else ""
        )
        logger.info(
            f"[{WORKING_POINTS[seq_local]}]{' [trace]' if TRACE else ''} {mode}{'/' + tt_model.sol_mode + '/prefix=' + tt_model.sol_prefix_mode if mode == 'sol' else ''} "
            f"chunks={chunks_env or 'measured'} tau={tau}: compile {compile_ms:.0f} ms; "
            f"module {' / '.join(f'{m:.2f}' for m in module_ms)} ms; op {' / '.join(f'{m:.2f}' for m in op_times)} ms"
            f"{prefix_note}; density {density}"
            + (
                f"; density per rep {' / '.join('-' if d is None else f'{d:.3f}' for d in densities)}"
                if any(d is not None for d in densities)
                else ""
            )
            + (f"; block density (64x64) {block_density:.4f}" if block_density is not None else "")
            + (
                f"; block density per rep {' / '.join('-' if d is None else f'{d:.3f}' for d in block_densities)}"
                if any(d is not None for d in block_densities)
                else ""
            )
        )
        del tt_model

    out_dir = Path(os.environ.get("SOL_BENCH_OUT", str(Path.home() / "sol_attn_work" / "bench")))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.jsonl", "a") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")
