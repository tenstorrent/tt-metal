# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TRACED variant of test_decode_profile.py: per-op device times and op-to-op gaps inside the decode trace.

The eager profile gives kernel times but not the gaps a trace replay leaves between ops; the full
ternary stack runs 44.25 ms/token traced against 37.03 ms of eager device time. This primes the
decode trace, flushes the profiler, replays PROFILE_STEPS traced steps and flushes after each.
Original docstring follows.

Per-op device profile of whole-model decode steps on one chip (all layers and the LM head).

Builds the model as qwen36 text_demo does, runs WARMUP decode steps, flushes the device profiler, then runs
PROFILE_STEPS eager decode steps between tracy signposts, flushing after each, so the profile holds decode
only (a profiled demo run keeps the prefill and drops the decode steps). Device kernel times are the same
eager or traced; the gaps between ops are not, so take wall time per token from the traced demo.
    TT_METAL_DEVICE_PROFILER=1 python -m tracy -r -v --no-web-server --op-support-count 12000 \
      -m pytest models/demos/blackhole/bonsai2/tp1/test_decode_profile.py -s
Set QWEN36_TP1_GDN=1 to profile the GDN layers as TPGatedDeltaNet at TP=1.
"""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.tt_transformers.tt.generator import Generator

WARMUP = 2
PROFILE_STEPS = 2
BLOCK_SIZE = 64
NUM_BLOCKS = 64


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "num_command_queues": 2}], indirect=True)
def test_decode_trace_profile(mesh_device):
    if os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        pytest.skip("needs the device profiler (TT_METAL_DEVICE_PROFILER=1 under python -m tracy)")
    from tracy import signpost

    mesh_device.enable_program_cache()
    model = Qwen36Model.from_pretrained(
        mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * BLOCK_SIZE, enable_mtp=False
    )
    args = model.args
    model.allocate_kv_caches([NUM_BLOCKS, args.n_kv_heads, BLOCK_SIZE, args.head_dim], ttnn.bfloat16, batch_size=1)
    model.reset_state()
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    gen = Generator([model], [args], mesh_device)

    def step(pos, first):
        gen.decode_forward(
            torch.tensor([[1000 + pos]], dtype=torch.long),
            torch.tensor([pos]),
            page_table=page_table,
            kv_cache=None,
            enable_trace=True,
            read_from_device=True,
            reload_inputs=True,
            reload_page_table=first,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )

    for pos in range(WARMUP):
        step(pos, first=pos == 0)
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)
    for i in range(PROFILE_STEPS):
        signpost(f"decode_{i}")
        step(WARMUP + i, first=False)
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
    signpost("end")
