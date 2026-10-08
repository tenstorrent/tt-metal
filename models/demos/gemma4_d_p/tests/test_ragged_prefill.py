# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CP8/TP4 numerical, trace, cache and wall-time comparisons against batch one.

Uses six real decoder layers (one full sliding/global cycle) by default. Set
GEMMA4_RAGGED_TEST_LAYERS=60 for the complete model. No GPU reference is needed:
each request has an independent batch-one slot in the same model.
"""

import json
import os
import time
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.common.weight_cache import build_cached_state_dict
from models.demos.common.prefill.runners.migration import _build_device_map
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.demo.text_demo_prefill import _cp_gather_torch
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.common import create_tt_model, weight_cache_identity
from models.demos.gemma4_d_p.tt.model_config import Gemma4ModelArgs, resolve_cache_dir_from_tt_cache_path
from models.demos.gemma4_d_p.tt.precision import Gemma4Precision
from models.demos.gemma4_d_p.tt.ragged_prefill import PrefillRequest, round_up
from models.demos.gemma4_d_p.tt.runners.kv_caches import Gemma4KvCaches, allocate_ring_kv_caches
from models.demos.gemma4_d_p.tt.runners.kv_chunk_table import build_kv_chunk_address_table
from models.demos.gemma4_d_p.tt.runners.kv_validation import check_table_samples, read_cache_tensor
from models.demos.gemma4_d_p.tt.runners.runtime import Gemma4PrefillRuntime


def _tokens(seed, length):
    return tuple(torch.randint(1, 10000, (length,), generator=torch.Generator().manual_seed(seed)).tolist())


def _independent(runtime, caches, request, slot):
    inp = runtime.make_chunk_input((*request.token_ids, *([0] * (8192 - len(request.token_ids)))))
    runtime.prefill_chunk(inp, caches, slot_id=slot, actual_start=request.actual_start, actual_end=request.actual_end)
    return _cp_gather_torch(runtime.output, runtime.mesh_config)[..., : len(request.token_ids), :]


def _assert_similar(actual, expected, name):
    if torch.equal(actual, expected):
        print(f"{name}: PCC=1.000000 RRMSE=0.000000 (bitwise equal)", flush=True)
        return
    actual, expected = actual.float().flatten(), expected.float().flatten()
    pcc = torch.corrcoef(torch.stack((actual, expected)))[0, 1].item()
    relative_rmse = ((actual - expected).square().mean() / expected.square().mean()).sqrt().item()
    print(f"{name}: PCC={pcc:.6f} RRMSE={relative_rmse:.6f}", flush=True)
    assert pcc > 0.995, name
    assert relative_rmse < 0.08, name


@pytest.mark.timeout(3600)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": 256_000_000})
def test_packed_requests_outputs_cache_replay_and_timing(mesh_device, tmp_path):
    model_id = os.environ.get("HF_MODEL", "google/gemma-4-31B-it")
    mesh = MeshConfig(mesh_device)
    num_layers = int(os.environ.get("GEMMA4_RAGGED_TEST_LAYERS", "6"))
    args = Gemma4ModelArgs.from_hf_config(Gemma4ModelArgs.load_hf_config(model_id))
    root = os.environ["TT_CACHE_PATH"]
    cache_dir = resolve_cache_dir_from_tt_cache_path(root, dtype=ttnn.bfloat16, mesh_shape=(8, 4))
    identity = weight_cache_identity(model_id, args.num_hidden_layers, (8, 4), Gemma4Precision.load(model_id))
    state = build_cached_state_dict(cache_dir, args=args, build_variant=identity["build_variant"])
    caches = allocate_ring_kv_caches(mesh, args, num_users=4, max_seq_len=32768, num_layers=num_layers)
    _, model, _, _ = create_tt_model(
        mesh,
        8192,
        max_batch_size=4,
        max_seq_len=32768,
        num_layers=num_layers,
        hf_model_id=model_id,
        state_dict=state,
        ring_kv_caches=caches,
        tt_cache_path=root,
    )
    config = SimpleNamespace(num_users=4, chunk_size=8192, max_seq_len=32768, mesh_shape=(8, 4), num_layers=num_layers)
    runtime = Gemma4PrefillRuntime(mesh_device=mesh_device, hf_model_id=model_id, tt_cache_path=root, config=config)
    runtime.model = model
    # Compare both paths using the production reduce-scatter arithmetic.
    model.stable_prefill_reductions = False
    runtime.input_tokens = runtime.make_chunk_input([0] * 8192)
    runtime.positions = runtime.make_chunk_input(range(8192))
    runtime.metadata = None
    model.set_prefill_rope_positions(runtime.positions)
    model._prefill_metadata_external = True
    runtime._stage_positions(0, 0)
    # The first sliding/global cycle is enough to exercise every migration
    # config/head and CP rank, even when the numerical run uses all 60 layers.
    migration_caches = Gemma4KvCaches(
        layers=caches.layers[:6],
        layer_types=caches.layer_types[:6],
        num_users=4,
        max_seq_len=32768,
        cp=8,
        tp=4,
    )
    table = build_kv_chunk_address_table(mesh_device=mesh_device, kv_caches=migration_caches, chunk_size=8192)
    device_map = {(mesh_id, chip_id): uid for mesh_id, chip_id, uid in _build_device_map(mesh_device, (8, 4))}
    warmup = runtime._forward()
    ttnn.synchronize_device(mesh_device)
    warmup.deallocate(True)
    # Validation also launches slice/untilize kernels. Compile every readback
    # shape before capture so their persistent program buffers cannot land in
    # the trace's transient allocation range between replays.
    for tensor in (caches.layers[0].k, caches.layers[5].kv):
        for slot in range(4):
            for extent in (8192, 16384):
                read_cache_tensor(tensor, slot, extent)
    runtime.capture_trace(caches)

    cases = (
        (PrefillRequest(10, 2, 0, _tokens(1, 8192)), PrefillRequest(11, 3, 0, _tokens(2, 33))),
        (PrefillRequest(10, 2, 8192, _tokens(3, 1025)), PrefillRequest(12, 3, 0, _tokens(4, 1023))),
        # Same trace shape, changed slots, prefixes, identities and valid lengths.
        (PrefillRequest(13, 3, 0, _tokens(5, 1055)), PrefillRequest(14, 2, 0, _tokens(6, 1024))),
        # Change only the second request; the first output and KV must be bitwise unchanged.
        (PrefillRequest(13, 3, 0, _tokens(5, 1055)), PrefillRequest(17, 2, 0, _tokens(9, 1024))),
        (PrefillRequest(15, 2, 0, _tokens(7, 31)),),
        (PrefillRequest(16, 2, 0, _tokens(8, 32)),),
        (PrefillRequest(18, 2, 0, _tokens(10, 8192)), PrefillRequest(19, 3, 0, _tokens(11, 8192))),
    )
    try:
        # Finish independent references first, so packed replays can change
        # metadata without switching back to (and recapturing) the legacy graph.
        for boundary, requests in enumerate(cases):
            print(f"Independent boundary={boundary}", flush=True)
            expected = [_independent(runtime, caches, req, slot) for slot, req in enumerate(requests)]
            expected_kv = {}
            for slot, req in enumerate(requests):
                for layer_idx, cache in enumerate(caches.layers):
                    for kind, tensor in enumerate((cache.kv,) if hasattr(cache, "kv") else (cache.k, cache.v)):
                        expected_kv[slot, layer_idx, kind] = read_cache_tensor(
                            tensor, slot, round_up(req.actual_end, 8192)
                        )[..., : req.actual_end, :].clone()
            torch.save((expected, expected_kv), tmp_path / f"reference_{boundary}.pt")
        last_variant = None
        previous_outputs = {}
        previous_kv = {}
        for boundary, requests in enumerate(cases):
            print(f"Packed boundary={boundary}", flush=True)
            expected, expected_kv = torch.load(tmp_path / f"reference_{boundary}.pt", weights_only=True)
            acknowledgements = []
            runtime.set_layer_completion_sink(lambda layer, identity: acknowledgements.append((layer, identity)))
            tails = {}
            for req in requests:
                for layer_idx, cache in enumerate(caches.layers):
                    for kind, tensor in enumerate((cache.kv,) if hasattr(cache, "kv") else (cache.k, cache.v)):
                        tails[req.request_id, layer_idx, kind] = read_cache_tensor(
                            tensor, req.slot_id, round_up(req.actual_end, 8192)
                        )[..., round_up(req.actual_end, 32) :, :].clone()
            result = runtime.prefill_batch(requests, caches)
            if boundary in (2, 3, 5):
                assert next(iter(runtime.ragged_variants.values())) is last_variant
            last_variant = next(iter(runtime.ragged_variants.values()))
            outputs = result.to_torch()
            if boundary == 3:
                torch.testing.assert_close(outputs[13], previous_outputs[13], rtol=0, atol=0)
            current_kv = {}
            for slot, (req, reference) in enumerate(zip(requests, expected)):
                for layer_idx, cache in enumerate(caches.layers):
                    for kind, tensor in enumerate((cache.kv,) if hasattr(cache, "kv") else (cache.k, cache.v)):
                        populated = read_cache_tensor(tensor, req.slot_id, round_up(req.actual_end, 8192))
                        if layer_idx in (0, 5):
                            config_start = 0 if hasattr(cache, "kv") else (4 if kind == 0 else 20)
                            for head, rows in enumerate(populated):
                                check_table_samples(
                                    table, device_map, layer_idx, req.slot_id, config_start + head, rows
                                )
                        torch.testing.assert_close(
                            populated[..., round_up(req.actual_end, 32) :, :],
                            tails[req.request_id, layer_idx, kind],
                            rtol=0,
                            atol=0,
                        )
                        actual = populated[..., : req.actual_end, :]
                        if boundary == 3 and req.request_id == 13:
                            torch.testing.assert_close(actual, previous_kv[13, layer_idx, kind], rtol=0, atol=0)
                        current_kv[req.request_id, layer_idx, kind] = actual.clone()
                        _assert_similar(
                            actual, expected_kv[slot, layer_idx, kind], f"KV boundary={boundary} layer={layer_idx}"
                        )
                _assert_similar(outputs[req.request_id], reference, f"boundary={boundary} request={req.request_id}")
            assert acknowledgements == [
                (layer, req.completion_id) for req in result.requests for layer in range(num_layers)
            ]
            result.deallocate()
            runtime.set_layer_completion_sink(None)
            previous_outputs, previous_kv = outputs, current_kv

        measurements = []
        for lengths in ((33, 1025), (8192, 8192)):
            requests = tuple(PrefillRequest(20 + i, 2 + i, 0, _tokens(20 + i, n)) for i, n in enumerate(lengths))

            def run_independent():
                latencies = []
                start = time.perf_counter()
                for slot, req in enumerate(requests):
                    inp = runtime.make_chunk_input((*req.token_ids, *([0] * (8192 - len(req.token_ids)))))
                    runtime.prefill_chunk(inp, caches, slot_id=slot, actual_start=0, actual_end=len(req.token_ids))
                    latencies.append(time.perf_counter() - start)
                return latencies

            run_independent()  # Switch and capture outside the replay measurement.
            independent = [run_independent() for _ in range(3)]
            start = time.perf_counter()
            runtime.prefill_batch(requests, caches).deallocate()
            cold_seconds = time.perf_counter() - start
            for repeat in range(3):
                start = time.perf_counter()
                result = runtime.prefill_batch(requests, caches)
                result.deallocate()
                packed_seconds = time.perf_counter() - start
                independent_seconds = independent[repeat][-1]
                measurements.append(
                    dict(
                        lengths=lengths,
                        repeat=repeat,
                        layers=num_layers,
                        cold_seconds=cold_seconds,
                        independent_seconds=independent_seconds,
                        packed_seconds=packed_seconds,
                        independent_request_seconds=independent[repeat],
                        packed_request_seconds=[packed_seconds] * len(requests),
                        useful_tokens=sum(lengths),
                        packed_rows=result.plan.packed_size,
                        independent_tokens_per_second=sum(lengths) / independent_seconds,
                        packed_tokens_per_second=sum(lengths) / packed_seconds,
                    )
                )
        print("RAGGED_MEASUREMENTS=" + json.dumps(measurements), flush=True)
        (tmp_path / "ragged_measurements.json").write_text(json.dumps(measurements, indent=2) + "\n")
    finally:
        runtime.release_trace()
