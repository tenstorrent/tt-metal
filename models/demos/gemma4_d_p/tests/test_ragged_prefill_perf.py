# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Editable packed-prefill workload on Blackhole CP8/TP4.

Run with the model/cache environment from docs/PREFILL_SERVICE.md:
    pytest models/demos/gemma4_d_p/tests/test_ragged_prefill_perf.py -sv --timeout=3600

Uses all 60 layers by default; GEMMA4_RAGGED_TEST_LAYERS=6 selects a shorter run.
"""

import os
import time
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.common.weight_cache import build_cached_state_dict
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.common import create_tt_model, weight_cache_identity
from models.demos.gemma4_d_p.tt.model_config import Gemma4ModelArgs, resolve_cache_dir_from_tt_cache_path
from models.demos.gemma4_d_p.tt.precision import Gemma4Precision
from models.demos.gemma4_d_p.tt.ragged_prefill import PrefillRequest, RaggedPrefillPlan
from models.demos.gemma4_d_p.tt.runners.kv_caches import allocate_ring_kv_caches
from models.demos.gemma4_d_p.tt.runners.runtime import Gemma4PrefillRuntime


@pytest.fixture
def run_batch(mesh_device):
    model_id = os.environ.get("HF_MODEL", "google/gemma-4-31B-it")
    root = os.environ["TT_CACHE_PATH"]
    mesh = MeshConfig(mesh_device)
    args = Gemma4ModelArgs.from_hf_config(Gemma4ModelArgs.load_hf_config(model_id))
    num_layers = int(os.environ.get("GEMMA4_RAGGED_TEST_LAYERS", "60"))
    if not 1 <= num_layers <= args.num_hidden_layers:
        raise ValueError(f"GEMMA4_RAGGED_TEST_LAYERS must be between 1 and {args.num_hidden_layers}")
    config = SimpleNamespace(num_users=4, chunk_size=8192, max_seq_len=32768, mesh_shape=(8, 4), num_layers=num_layers)
    cache_dir = resolve_cache_dir_from_tt_cache_path(root, dtype=ttnn.bfloat16, mesh_shape=config.mesh_shape)
    identity = weight_cache_identity(
        model_id, args.num_hidden_layers, config.mesh_shape, Gemma4Precision.load(model_id)
    )
    state = build_cached_state_dict(cache_dir, args=args, build_variant=identity["build_variant"])
    caches = allocate_ring_kv_caches(
        mesh, args, num_users=config.num_users, max_seq_len=config.max_seq_len, num_layers=num_layers
    )
    _, model, _, _ = create_tt_model(
        mesh,
        config.chunk_size,
        max_batch_size=config.num_users,
        max_seq_len=config.max_seq_len,
        num_layers=num_layers,
        hf_model_id=model_id,
        state_dict=state,
        ring_kv_caches=caches,
        tt_cache_path=root,
    )
    runtime = Gemma4PrefillRuntime(mesh_device=mesh_device, hf_model_id=model_id, tt_cache_path=root, config=config)
    runtime.model = model
    generator = torch.Generator().manual_seed(0)
    batch_number = 0
    next_request_id = 0
    print(
        f"\nRAGGED PERF: {num_layers} layers, CP8/TP4, chunk={config.chunk_size}, slots={config.num_users}\n"
        "Wall time includes packing, staging, execution and synchronization.\n"
        "capture+replay also includes warmup/capture; replay uses the resident trace.\n"
        "Excludes model loading, random token generation, output downloads and external KV transfer.\n"
        "Request completion_ms is measured from this batch's start; all requests finish together.\n",
        flush=True,
    )

    def run_batch(lengths, *, starts=None):
        """One batch call. List index is the KV slot; end = start + length (exclusive).

        Start zero begins a new request in that slot. Nonzero starts continue
        its preceding full chunk, retaining the same request ID and cached KV.
        Repeat the same tile-rounded lengths to measure a trace replay.
        """
        nonlocal batch_number, next_request_id
        starts = [0] * len(lengths) if starts is None else starts
        if len(starts) != len(lengths) or not 1 <= len(lengths) <= config.num_users:
            raise ValueError("Provide 1 to 4 lengths and an equally sized list of starts")
        if any(length < 1 or length > config.chunk_size for length in lengths):
            raise ValueError(f"Each request chunk must contain 1 to {config.chunk_size} tokens")
        requests = []
        for slot, (length, start) in enumerate(zip(lengths, starts)):
            if start == 0:
                request_id = next_request_id
                next_request_id += 1
            else:
                request_id = runtime.slot_requests[slot]
                if request_id is None:
                    raise ValueError(f"Slot {slot} has no prefix; run its full initial chunk first")
            tokens = tuple(torch.randint(1, min(10000, model.vocab_size), (length,), generator=generator).tolist())
            requests.append(PrefillRequest(request_id, slot, start, tokens))

        plan = RaggedPrefillPlan.for_requests(requests, chunk_size=config.chunk_size)
        phase = "replay" if plan in runtime.ragged_variants else "capture+replay"
        ttnn.synchronize_device(mesh_device)
        start_time = time.perf_counter()
        result = runtime.prefill_batch(requests, caches)
        elapsed = time.perf_counter() - start_time  # prefill_batch synchronizes before returning.
        result.deallocate()
        batch_number += 1
        print(
            f"Batch {batch_number:02d} | {phase:14s} | {sum(lengths):,} useful / {plan.packed_size:,} packed tokens"
            f" | {elapsed * 1000:,.2f} ms | {sum(lengths) / elapsed:,.0f} useful tok/s\n"
            "  request  slot      start        end   tokens  completion_ms",
            flush=True,
        )
        for request in requests:
            print(
                f"  {request.request_id:7d}  {request.slot_id:4d}  {request.actual_start:9d}  {request.actual_end:9d}"
                f"  {len(request.token_ids):7d}  {elapsed * 1000:13.2f}",
                flush=True,
            )
        print(flush=True)

    try:
        yield run_batch
    finally:
        runtime.release_trace()


@pytest.mark.timeout(3600)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": 256_000_000})
def test_ragged_prefill_perf(run_batch):
    # List index = slot. Token ranges are [start, start + length).
    # New shapes capture; consecutive matching shapes replay the resident trace.
    run_batch([543, 2012, 998, 123], starts=[0, 0, 0, 0])
    run_batch([543, 2012, 998, 123], starts=[0, 0, 0, 0])
    run_batch([7012, 643, 22], starts=[0, 0, 0])
    run_batch([7012, 643, 22], starts=[0, 0, 0])
    run_batch([8192, 8192], starts=[0, 0])
    run_batch([8192, 8192], starts=[8192, 8192])
    run_batch([1025, 33], starts=[16384, 0])
    run_batch([1055, 63], starts=[0, 0])
