# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy capture of ONE eager SP prefill, to attribute the per-layer span-independent cost.

The SP wavefront decomposes (measured, by varying layer count at fixed span) as

    wavefront = n_layers * per_layer(span) + (SP-1) * ~1.6 ms
    per_layer(span) ~= 0.82 ms  +  1.67 us/token

At SP=8 / span 512 the 0.82 ms/layer term is ~19.6 ms of die-0's 40.1 ms -- half the
critical path, and completely insensitive to span, so no amount of extra SP touches it.
This capture exists to settle WHAT it is: weight streaming (bytes) or op launch (count),
which imply opposite fixes.

Traced replay is not profilable, so this signposts the EAGER prefill; the op mix is the
same set the trace replays.

Usage:
    export HF_MODEL=Qwen/Qwen3.5-2B TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0
    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \
      python -m tracy -p -r -v -m pytest \
      models/demos/blackhole/qwen36/tests/perf/perf_sp_prefill_tracy.py -sv
"""

import os

import torch
from loguru import logger

from models.demos.blackhole.qwen36.tests.test_sp_prefill import SP_DIES, _close_sp_mesh, _open_sp_mesh, model_path
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE
from models.demos.blackhole.qwen36.tt.sp_prefill import SPPrefill

try:
    from tracy import signpost
except ImportError:

    def signpost(*_a, **_k):
        pass


def test_perf_sp_prefill_tracy():
    T = int(os.environ.get("SP_ISL", "4096"))
    layers = os.environ.get("QWEN36_E2E_LAYERS")
    layer_indices = [int(x) for x in layers.split(",")] if layers else None

    mesh_owner, mesh = _open_sp_mesh(trace_region_size=64 * 1024 * 1024, l1_small_size=GDN_CONV1D_L1_SMALL_SIZE)
    try:
        sp = SPPrefill(
            mesh,
            n_spans=SP_DIES,
            span_len=T // SP_DIES,
            max_seq_len=T,
            hf_model=model_path(),
            layer_indices=layer_indices,
        )
        try:
            torch.manual_seed(0)
            tokens = torch.randint(1000, 100000, (1, T), dtype=torch.long)
            # SPPrefill.prefill() already synchronizes every die before returning (it logs its
            # own wall time). An extra ttnn.synchronize_device on the PARENT submesh deadlocks
            # against the per-die socket state: the measured pass completes, then the sync never
            # returns and the run wedges. Rely on prefill()'s own sync.
            sp.prefill(tokens)  # warmup / compile, OUTSIDE the signposts

            logger.info(f"measured eager SP prefill: SP={SP_DIES} ISL={T} span={T // SP_DIES}")
            signpost("start")
            sp.prefill(tokens)
            signpost("stop")
        finally:
            sp.close()
    finally:
        _close_sp_mesh(mesh_owner)
