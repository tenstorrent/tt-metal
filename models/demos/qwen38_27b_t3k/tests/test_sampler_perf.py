# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The device sampler's top-k needs a power-of-two width to reach more than one core.

`248320 / 8` is 31040, and `topk_multicore_structurally_eligible` requires a power-of-two
reduced width for the multi-core bitonic network; the large-indices route that would otherwise
carry this width is Blackhole-only. Unpadded, the top-k runs on a single Tensix core across the
whole row and the sampling step costs about 8.8 ms against about 4.0 ms padded. These tests hold
that configuration in place and check that padding has not changed which token is sampled.

One layer is enough: the sampler's cost is fixed in batch, since tile padding makes a 1-row and a
32-row logit tensor the same 970 tiles. It has to be a `full_attention` layer, every fourth index,
because a `linear_attention` layer reaches the KDA performance model and that asserts Blackhole,
which aborts the run under the device profiler.
"""

import os
import time

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_t3k.tt.decoder_tp import native_mesh_shape
from models.demos.qwen38_27b_t3k.tt.generator import build_generator, configure_fabric

PROMPT = 64
FULL_ATTENTION_LAYER = 3
# Tracy drops after 1000 ops per device and a sampling step is about 26 programs, so a profiled
# run has to ask for far fewer repetitions than a timing run.
WARMUP_STEPS = int(os.getenv("QWEN_SAMPLER_WARMUP", "8"))
TIMED_STEPS = int(os.getenv("QWEN_SAMPLER_STEPS", "40"))
# Padded measures ~4.0 ms with a p90 near 5.1; unpadded measures ~8.8 ms with a very tight
# spread. This sits between the two so a revert to single-core top-k fails rather than slows.
SINGLE_CORE_REGRESSION_MS = 6.5


@pytest.fixture(scope="module")
def sampler():
    from pathlib import Path

    root = Path(__file__).parents[1]
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*native_mesh_shape()), trace_region_size=1073741824)
    try:
        gen = build_generator(
            root, mesh, precision_config=root / "config/precision.json", layer_indices=[FULL_ATTENTION_LAYER]
        )
        model = gen.model
        cache = model.allocate_cache(batch_size=1, capacity=1024)
        table = torch.arange(cache.num_pages, dtype=torch.int32).reshape(1, -1)
        gen.bind_cache(cache, table)
        prompt = torch.randint(1000, 5000, (1, PROMPT), dtype=torch.int32)
        gen.prefill_forward(prompt, page_table=table, kv_cache=cache, prompt_lens=[PROMPT], start_pos=[0], slots=[0])
        batch = cache.batch_size
        ids = ttnn.reshape(gen.tokens, [1, 32])[:, :batch]
        hidden = model.embed(ids, batch=batch, length=1)
        logits = model.logits(hidden, decode=True)
        # k=1 makes sampling a selection of the maximum; temperature must stay positive.
        gen.set_batch_sampling_params(top_k=[1] * 32, top_p=[0.0] * 32, temperature=[1.0] * 32)
        yield gen, mesh, logits, batch
        gen.close()
    finally:
        ttnn.close_mesh_device(mesh)


def test_greedy_sampling_selects_the_same_token_as_a_host_argmax(sampler):
    """The pad fills with -float_max and the index offset still spans the unpadded shard, so the
    padded columns cannot win and global token ids are unchanged."""
    gen, mesh, logits, batch = sampler
    host = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
    expected = int(host[0, 0, 0].argmax())
    gen._sampling_step(logits)
    ttnn.synchronize_device(mesh)
    sampled = ttnn.to_torch(ttnn.get_device_tensors(gen.tokens)[0]).reshape(-1)[:batch].tolist()
    assert sampled[0] == expected, f"sampled {sampled[0]} but the host argmax over {host.shape[-1]} is {expected}"


def test_sampling_step_does_not_regress_to_a_single_core_topk(sampler):
    gen, mesh, logits, _ = sampler
    for _ in range(WARMUP_STEPS):
        gen._sampling_step(logits)
    ttnn.synchronize_device(mesh)
    times = []
    for _ in range(TIMED_STEPS):
        start = time.perf_counter()
        gen._sampling_step(logits)
        ttnn.synchronize_device(mesh)
        times.append((time.perf_counter() - start) * 1e3)
    times.sort()
    p50 = times[len(times) // 2]
    assert p50 < SINGLE_CORE_REGRESSION_MS, (
        f"sampling step p50 {p50:.3f} ms exceeds {SINGLE_CORE_REGRESSION_MS} ms; "
        "pad_logits_to_power_of_2 may have been turned off"
    )
