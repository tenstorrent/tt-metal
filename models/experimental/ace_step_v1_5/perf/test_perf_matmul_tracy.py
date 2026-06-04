# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tracy perf harness for a **single DiT linear** (LoFi + BFP8 weights, L1 in0/out).

Isolates ``MatmulDeviceOperation`` without full attention or denoise loop. Use to A/B
``ACE_STEP_DIT_FORCE_L1_MATMUL``, ``ACE_STEP_DIT_MAX_FUSED_M``, and ``packer_l1_acc``.

Run from repository root::

    TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r -v -m pytest \\
        models/experimental/ace_step_v1_5/perf/test_perf_matmul_tracy.py::test_perf_ace_step_matmul_tracy_profile \\
        -v -s

Then::

    tt-perf-report generated/profiler/reports/<timestamp>/cpp_device_perf_report.csv

Environment:

- ``ACE_STEP_MATMUL_PERF_PROJ``: ``q_proj`` (default), ``o_proj``, ``gate_proj``, ``down_proj``
- ``ACE_STEP_MATMUL_PERF_BATCH`` (default ``2``): CFG-style batch
- ``ACE_STEP_MATMUL_PERF_SEQ`` (default ``64``)
- ``ACE_STEP_MATMUL_PERF_ITERS`` (default ``64``)
- ``ACE_STEP_PERF_WARMUP`` (default ``4``)
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import Profiler
from models.experimental.ace_step_v1_5.tests._dit_decoder_pcc_common import tiny_dit_decoder_fixture
from models.experimental.ace_step_v1_5.tt_device import ace_step_dit_weight_mesh_mapper
from models.experimental.ace_step_v1_5.ttnn_impl.dit_decoder_core import _maybe_get
from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import (
    ace_step_dit_attn_linear_program_config,
    ace_step_dit_mlp_down_proj_linear_program_config,
    ace_step_dit_mlp_gate_up_linear_program_config,
    ace_step_dit_weight_dtype,
    ace_step_dit_weight_layout,
    ace_step_dit_weight_memory_config,
    ace_step_enable_tracy_profiler_env,
    ace_step_flush_device_profiler,
    ace_step_init_dit_linear_compute_kernel_config,
    ace_step_linear_kwargs_memory_config,
    ace_step_linear_l1_memory_config,
    ace_step_matmul_activation,
)


def _tracy_signpost(label: str) -> None:
    if os.environ.get("CI", "").lower() in ("true", "1", "yes"):
        return
    try:
        from tracy import signpost  # type: ignore[import-untyped]
    except ImportError:
        return
    try:
        signpost(label)
    except Exception:
        pass


def _resolve_proj_weights(sd: dict, proj: str):
    for prefix in ("layers.0.self_attn", "layers.0.mlp"):
        key = f"{prefix}.{proj}.weight"
        w = _maybe_get(sd, key)
        if w is not None:
            return w, int(w.shape[1]), int(w.shape[0])
    raise KeyError(proj)


def _program_config(device, proj: str, *, seq_len: int, in_dim: int, out_dim: int, batch: int):
    if proj in ("gate_proj", "up_proj"):
        return ace_step_dit_mlp_gate_up_linear_program_config(
            device, seq_len=seq_len, hidden_size=in_dim, intermediate_size=out_dim, batch_size=batch
        )
    if proj == "down_proj":
        return ace_step_dit_mlp_down_proj_linear_program_config(
            device,
            seq_len=seq_len,
            intermediate_size=in_dim,
            hidden_size=out_dim,
            batch_size=batch,
        )
    return ace_step_dit_attn_linear_program_config(
        device, seq_len=seq_len, in_dim=in_dim, out_dim=out_dim, batch_size=batch
    )


def _run_matmul_tracy_harness(device: ttnn.Device) -> None:
    proj = os.environ.get("ACE_STEP_MATMUL_PERF_PROJ", "q_proj").strip()
    batch = max(1, int(os.environ.get("ACE_STEP_MATMUL_PERF_BATCH", "2")))
    seq_len = max(32, int(os.environ.get("ACE_STEP_MATMUL_PERF_SEQ", "64")))
    iters = max(1, int(os.environ.get("ACE_STEP_MATMUL_PERF_ITERS", "64")))
    warmup = max(0, int(os.environ.get("ACE_STEP_PERF_WARMUP", "4")))

    _, sd, _, _, _ = tiny_dit_decoder_fixture(seq_len=seq_len, intermediate=256)
    w_host, in_dim, out_dim = _resolve_proj_weights(sd, proj)

    dram = ace_step_dit_weight_memory_config(ttnn)
    l1 = ace_step_linear_l1_memory_config(ttnn)
    w_dtype = ace_step_dit_weight_dtype(ttnn, ttnn.bfloat16)
    mapper = ace_step_dit_weight_mesh_mapper(device)

    w_tt = ttnn.as_tensor(
        w_host,
        device=device,
        dtype=w_dtype,
        layout=ace_step_dit_weight_layout(ttnn, w_dtype, default_layout=ttnn.TILE_LAYOUT),
        memory_config=dram,
        mesh_mapper=mapper,
    )

    x_host = torch.randn(batch, 1, seq_len, in_dim, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(
        x_host,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=l1,
    )

    pc = _program_config(device, proj, seq_len=seq_len, in_dim=in_dim, out_dim=out_dim, batch=batch)
    ck = ace_step_init_dit_linear_compute_kernel_config(device)
    lin_kw: dict = {"transpose_b": True}
    if ck is not None:
        lin_kw["compute_kernel_config"] = ck
    if pc is not None:
        lin_kw["program_config"] = pc
    lin_kw["memory_config"] = ace_step_linear_kwargs_memory_config(pc, linear_out_l1=l1, dram=dram)

    prof = Profiler()
    _tracy_signpost("ace_step_matmul_warmup_start")
    for _ in range(warmup):
        x_in = ace_step_matmul_activation(
            ttnn, x_tt, lin_kw, l1_fn=lambda t: ttnn.to_memory_config(t, l1), dram_mc=dram
        )
        y = ttnn.linear(x_in, w_tt, **lin_kw)
        ttnn.deallocate(y)
    ace_step_flush_device_profiler(ttnn, device)

    _tracy_signpost("ace_step_matmul_timed_start")
    for _ in range(iters):
        prof.start("matmul_linear")
        x_in = ace_step_matmul_activation(
            ttnn, x_tt, lin_kw, l1_fn=lambda t: ttnn.to_memory_config(t, l1), dram_mc=dram
        )
        y = ttnn.linear(x_in, w_tt, **lin_kw)
        prof.end("matmul_linear")
        ttnn.deallocate(y)
    ace_step_flush_device_profiler(ttnn, device)
    _tracy_signpost("ace_step_matmul_timed_end")

    logger_msg = (
        f"matmul_tracy proj={proj} batch={batch} seq={seq_len} "
        f"shape={batch}x{seq_len}x{in_dim}x{out_dim} pc={'yes' if pc else 'no'}"
    )
    print(logger_msg, flush=True)


@pytest.mark.timeout(600)
def test_perf_ace_step_matmul_tracy_profile(device):
    ace_step_enable_tracy_profiler_env()
    _run_matmul_tracy_harness(device)
