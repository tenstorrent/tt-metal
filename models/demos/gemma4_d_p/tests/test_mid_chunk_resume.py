# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A request that resumes inside a chunk (a multi-turn prefix hit) fills its KV slot like the same tokens prefilled
from 0 in whole chunks.

Slot 0 prefills [0, N) in aligned chunks. Slot 1 prefills a first turn [0, T), then resumes at P, T rounded down to
the runtime's resume granularity (a per-rank slab), rewriting [P, T). Each chunk is rotated as the producer rotates it.
The negative control resumes at a tile inside a slab, so one rank's rows wrap two slabs and need a second sliding halo
slot the gather buffer doesn't have; it must fail.
"""

import time

import pytest
import torch
from loguru import logger

from models.demos.common.prefill.adapter import PrefillRunParams
from models.demos.common.prefill.chunk_layout import rotate_chunk_tokens
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.demo.text_demo_prefill import TRACE_REGION_SIZE
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.attention import ring_prefill
from models.demos.gemma4_d_p.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4_d_p.tt.runners.adapters.gemma4 import Gemma4PrefillAdapter, Gemma4ServiceConfig
from models.demos.gemma4_d_p.tt.runners.kv_caches import allocate_ring_kv_caches
from models.demos.gemma4_d_p.tt.runners.kv_validation import cache_metrics, read_cache_tensor
from models.demos.gemma4_d_p.tt.runners.runtime import Gemma4PrefillRuntime

N_CHUNKS = 4
EARLY_LAYERS = range(16)  # deeper layers drift run to run even without a resume
# Same model and tokens; only the chunk boundaries differ, which reorders accumulation.
MIN_PCC = 0.99


def _prefill(runtime, caches, prompt, slot, start, end):
    """Chunks covering [start, end) from start, padded and rotated as the producer sends them; per-chunk ms."""
    chunk, cp = runtime.config.chunk_size, runtime.mesh_config.cp_degree
    times = []
    for s in range(start, end, chunk):
        e = min(end, s + chunk)
        tokens = rotate_chunk_tokens(prompt[s:e].tolist() + [0] * (chunk - (e - s)), s, cp)
        tensor = runtime.make_chunk_input(tokens)
        t0 = time.perf_counter()
        runtime.prefill_chunk(tensor, caches, slot_id=slot, actual_start=s, actual_end=e)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def _min_head_pcc(caches, n):
    """Worst per-head PCC of slot 1 against slot 0 over the early layers, and how many heads matched exactly."""
    worst, identical, heads, staging = 1.0, 0, 0, {}
    for layer in EARLY_LAYERS:
        cache = caches.layers[layer]
        for tensor in (cache.kv,) if hasattr(cache, "kv") else (cache.k, cache.v):
            expected_heads = read_cache_tensor(tensor, 0, n, staging)
            for expected, actual in zip(expected_heads, read_cache_tensor(tensor, 1, n, staging)):
                metrics = cache_metrics(expected, actual)
                worst, identical, heads = min(worst, metrics.pcc), identical + metrics.identical, heads + 1
    return worst, identical, heads


@torch.no_grad()
@pytest.mark.timeout(1800)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
@pytest.mark.parametrize(
    "chunk, inside_slab",
    [(2048, False), (4096, False), (8192, False), (2048, True)],
    ids=["2048", "4096", "8192", "2048-inside-slab"],
)
def test_mid_chunk_resume(mesh_device, chunk, inside_slab, monkeypatch):
    adapter = Gemma4PrefillAdapter()
    hf_config = adapter.load_hf_config()
    n = N_CHUNKS * chunk
    slab = chunk // Gemma4ServiceConfig.MESH_SHAPE[0]
    step = ring_prefill.TILE_HEIGHT if inside_slab else slab
    first_turn = 2 * chunk + 5 * slab + 49
    resume = first_turn // step * step
    if inside_slab:
        monkeypatch.setattr(Gemma4PrefillRuntime, "resume_granularity", step)  # skip the start check
    monkeypatch.setattr(Gemma4ServiceConfig, "CHUNK_SIZE", chunk)  # read_cache_tensor's chunk
    params = PrefillRunParams(
        mesh_shape=Gemma4ServiceConfig.MESH_SHAPE,
        num_layers=60,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=n + chunk,  # a resumed request's last padded chunk ends past n
        chunk_size=chunk,
        num_users=2,
        capacity_factor=8,
        num_links=2,
        gate_mode_name="DEVICE_FP32",
        kv_only_last_layer=True,
        weight_cache_path=None,
        use_trace=True,
    )
    caches = allocate_ring_kv_caches(
        MeshConfig(mesh_device),
        Gemma4ModelArgs.from_hf_config(hf_config),
        num_users=2,
        max_seq_len=params.max_seq_len,
        prefill_chunk_size=chunk,
    )
    runtime = Gemma4PrefillRuntime(
        mesh_device=mesh_device, hf_model_id=adapter.hf_model_id, tt_cache_path=adapter.tt_cache_path, config=params
    )
    runtime.compile(caches)
    runtime.capture_trace(caches)

    prompt = torch.randint(0, hf_config.vocab_size, (n,))
    aligned = _prefill(runtime, caches, prompt, 0, 0, n)
    _prefill(runtime, caches, prompt, 1, 0, first_turn)
    resumed = _prefill(runtime, caches, prompt, 1, resume, n)
    worst, identical, heads = _min_head_pcc(caches, n)
    runtime.release_trace()

    logger.info(
        f"[resume] chunk={chunk} inside_slab={inside_slab} resume={resume}: min head PCC {worst:.6f} over "
        f"layers 0-15, {identical}/{heads} heads identical; ms per chunk aligned {[round(t, 1) for t in aligned]}, "
        f"resumed {[round(t, 1) for t in resumed]}"
    )
    if inside_slab:
        assert worst < MIN_PCC, f"a start inside a slab should corrupt the wrapping rank's KV, got min head PCC {worst}"
    else:
        assert worst >= MIN_PCC, f"resumed KV min head PCC {worst} < {MIN_PCC}"
