# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A request that resumes inside a chunk (a multi-turn prefix hit) fills its KV slot like the same tokens prefilled
from 0 in whole chunks.

Slot 0 prefills [0, N) in aligned chunks. Every other slot prefills a first turn [0, T), then resumes at P, T rounded
down to the runtime's resume granularity (a per-rank slab), rewriting [P, T). Batched, two slots resume at different P
and their resumed chunks fill packed forwards in order, so a forward holds one slot's consecutive entries and another
slot's. Each forward is packed and rotated as the producer sends it. The negative control resumes at a tile inside a
slab, so one rank's rows wrap two slabs and need a second sliding halo slot the gather buffer doesn't have; it must
fail.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.adapter import PrefillRunParams
from models.demos.common.prefill.chunk_layout import pack_forward_tokens
from models.demos.common.prefill.forward_entries import PrefillEntry, pad_entries
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.demo.text_demo_prefill import TRACE_REGION_SIZE
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.attention import ring_prefill
from models.demos.gemma4_d_p.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4_d_p.tt.runners.adapters.gemma4 import Gemma4PrefillAdapter, Gemma4ServiceConfig
from models.demos.gemma4_d_p.tt.runners.batched_runtime import Gemma4BatchedPrefillRuntime
from models.demos.gemma4_d_p.tt.runners.kv_caches import allocate_ring_kv_caches
from models.demos.gemma4_d_p.tt.runners.kv_validation import EARLY_LAYERS, cache_metrics, read_cache_tensor
from models.demos.gemma4_d_p.tt.runners.runtime import Gemma4PrefillRuntime

N_CHUNKS = 4
# Same model and tokens; only the chunk boundaries differ, which reorders accumulation.
MIN_PCC = 0.99


def _chunks(slot, start, end, chunk):
    return [PrefillEntry(slot, s, min(end, s + chunk)) for s in range(start, end, chunk)]


def _run(runtime, caches, prompt, entries):
    """One forward of ``entries``, packed and rotated as the producer sends it; its ms."""
    chunk, cp = runtime.config.chunk_size, runtime.mesh_config.cp_degree
    entries = pad_entries(entries, runtime.config.forward_chunks)
    chunks = [prompt[e.start : e.end].tolist() + [0] * (chunk - (e.end - e.start)) for e in entries]
    tensor = ttnn.from_torch(
        torch.tensor(pack_forward_tokens(chunks, [e.start for e in entries], cp)).reshape(1, -1),
        device=runtime.mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(runtime.mesh_device, runtime.config.mesh_shape, dims=(1, None)),
    )
    t0 = time.perf_counter()
    if runtime.config.forward_chunks == 1:
        e = entries[0]
        runtime.prefill_chunk(tensor, caches, slot_id=e.slot, actual_start=e.start, actual_end=e.end)
    else:
        runtime.prefill_forward(tensor, caches, entries)
    return (time.perf_counter() - t0) * 1000


def _min_head_pcc(caches, slots, n, chunk):
    """Per slot: the worst per-head PCC against slot 0 over the early layers, and how many heads matched exactly."""
    results = {slot: (1.0, 0, 0) for slot in slots}
    staging = {}
    for layer in range(EARLY_LAYERS):
        cache = caches.layers[layer]
        for tensor in (cache.kv,) if hasattr(cache, "kv") else (cache.k, cache.v):
            reference = read_cache_tensor(tensor, 0, n, staging, chunk_size=chunk)
            for slot in slots:
                for expected, actual in zip(reference, read_cache_tensor(tensor, slot, n, staging, chunk_size=chunk)):
                    metrics = cache_metrics(expected, actual)
                    worst, identical, heads = results[slot]
                    results[slot] = (min(worst, metrics.pcc), identical + metrics.identical, heads + 1)
    return [results[slot] for slot in slots]


@torch.no_grad()
@pytest.mark.timeout(1800)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
@pytest.mark.parametrize(
    "chunk, forward_chunks, inside_slab",
    [(2048, 1, False), (4096, 1, False), (8192, 1, False), (2048, 1, True), (2048, 3, False)],
    ids=["2048", "4096", "8192", "2048-inside-slab", "2048-batched"],
)
def test_mid_chunk_resume(mesh_device, chunk, forward_chunks, inside_slab, monkeypatch):
    adapter = Gemma4PrefillAdapter()
    hf_config = adapter.load_hf_config()
    n = N_CHUNKS * chunk
    slab = chunk // Gemma4ServiceConfig.MESH_SHAPE[0]
    step = ring_prefill.TILE_HEIGHT if inside_slab else slab
    # First turns end mid-slab; batched, a second slot ends elsewhere.
    first_turns = [2 * chunk + 5 * slab + 49]
    if forward_chunks > 1:
        first_turns.append(2 * chunk + 4 * slab + 113)
    resumes = [t // step * step for t in first_turns]
    if inside_slab:
        monkeypatch.setattr(Gemma4PrefillRuntime, "resume_granularity", step)  # skip the start check
    num_users = 1 + len(resumes)
    params = PrefillRunParams(
        mesh_shape=Gemma4ServiceConfig.MESH_SHAPE,
        num_layers=60,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=n + chunk,  # a resumed request's last padded chunk ends past n
        chunk_size=chunk,
        num_users=num_users,
        capacity_factor=8,
        num_links=2,
        gate_mode_name="DEVICE_FP32",
        kv_only_last_layer=True,
        weight_cache_path=None,
        use_trace=True,
        forward_chunks=forward_chunks,
    )
    caches = allocate_ring_kv_caches(
        MeshConfig(mesh_device),
        Gemma4ModelArgs.from_hf_config(hf_config),
        num_users=num_users,
        max_seq_len=params.max_seq_len,
        prefill_chunk_size=chunk,
    )
    runtime_class = Gemma4BatchedPrefillRuntime if forward_chunks > 1 else Gemma4PrefillRuntime
    runtime = runtime_class(
        mesh_device=mesh_device, hf_model_id=adapter.hf_model_id, tt_cache_path=adapter.tt_cache_path, config=params
    )
    runtime.compile(caches)
    runtime.capture_trace(caches)

    prompt = torch.randint(0, hf_config.vocab_size, (n,))
    aligned = [_run(runtime, caches, prompt, [e]) for e in _chunks(0, 0, n, chunk)]
    resumed_entries = []
    for slot, (first_turn, resume) in enumerate(zip(first_turns, resumes), start=1):
        for e in _chunks(slot, 0, first_turn, chunk):
            _run(runtime, caches, prompt, [e])
        resumed_entries += _chunks(slot, resume, n, chunk)
    forwards = [resumed_entries[i : i + forward_chunks] for i in range(0, len(resumed_entries), forward_chunks)]
    resumed = [_run(runtime, caches, prompt, forward) for forward in forwards]
    results = _min_head_pcc(caches, range(1, num_users), n, chunk)
    runtime.release_trace()

    for resume, (worst, identical, heads) in zip(resumes, results):
        logger.info(
            f"[resume] chunk={chunk} forward_chunks={forward_chunks} inside_slab={inside_slab} resume={resume}: "
            f"min head PCC {worst:.6f} over layers 0-{EARLY_LAYERS - 1}, {identical}/{heads} heads identical"
        )
    logger.info(
        f"[resume] ms per forward: aligned {[round(t, 1) for t in aligned]}, resumed {[round(t, 1) for t in resumed]}"
    )
    worst = min(r[0] for r in results)
    if inside_slab:
        assert worst < MIN_PCC, f"a start inside a slab should corrupt the wrapping rank's KV, got min head PCC {worst}"
    else:
        assert worst >= MIN_PCC, f"resumed KV min head PCC {worst} < {MIN_PCC}"
