# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-batch output/KV, isolation, migration and trace comparisons with independent calls."""

import os

import pytest
import torch

import ttnn
from models.common.weight_cache import build_cached_state_dict
from models.demos.common.prefill.runners.migration import _build_device_map
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.demo.text_demo_prefill import _cp_gather_torch
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.chunked_batch import ChunkedBatchPlan, ChunkedRequest
from models.demos.gemma4_d_p.tt.common import create_tt_model, weight_cache_identity
from models.demos.gemma4_d_p.tt.model_config import Gemma4ModelArgs, resolve_cache_dir_from_tt_cache_path
from models.demos.gemma4_d_p.tt.precision import Gemma4Precision
from models.demos.gemma4_d_p.tt.runners.chunked_batch_runtime import ChunkedBatchRuntime
from models.demos.gemma4_d_p.tt.runners.kv_caches import Gemma4KvCaches, allocate_ring_kv_caches
from models.demos.gemma4_d_p.tt.runners.kv_chunk_table import build_kv_chunk_address_table
from models.demos.gemma4_d_p.tt.runners.kv_validation import check_table_samples, read_cache_tensor


def build_model(mesh_device, *, chunk_size, num_slots, context_len, num_layers):
    model_id = os.environ.get("HF_MODEL", "google/gemma-4-31B-it")
    mesh = MeshConfig(mesh_device)
    args = Gemma4ModelArgs.from_hf_config(Gemma4ModelArgs.load_hf_config(model_id))
    root = os.environ["TT_CACHE_PATH"]
    cache_dir = resolve_cache_dir_from_tt_cache_path(root, dtype=ttnn.bfloat16, mesh_shape=(8, 4))
    identity = weight_cache_identity(model_id, args.num_hidden_layers, (8, 4), Gemma4Precision.load(model_id))
    state = build_cached_state_dict(cache_dir, args=args, build_variant=identity["build_variant"])
    caches = allocate_ring_kv_caches(
        mesh, args, num_users=num_slots, max_seq_len=context_len, num_layers=num_layers, prefill_chunk_size=chunk_size
    )
    _, model, _, _ = create_tt_model(
        mesh,
        chunk_size,
        max_batch_size=num_slots,
        max_seq_len=context_len,
        num_layers=num_layers,
        hf_model_id=model_id,
        state_dict=state,
        ring_kv_caches=caches,
        tt_cache_path=root,
    )
    return model, caches


def tokens(seed, length=4096):
    return tuple(torch.randint(1, 10000, (length,), generator=torch.Generator().manual_seed(seed)).tolist())


def cache_tensors(cache):
    return (cache.kv,) if hasattr(cache, "kv") else (cache.k, cache.v)


def assert_similar(actual, expected, name):
    actual, expected = actual.float().flatten(), expected.float().flatten()
    pcc = torch.corrcoef(torch.stack((actual, expected)))[0, 1].item()
    rrmse = ((actual - expected).square().mean() / expected.square().mean()).sqrt().item()
    print(f"{name}: PCC={pcc:.6f} RRMSE={rrmse:.6f}", flush=True)
    assert pcc > 0.995, name
    assert rrmse < 0.08, name


@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": 256_000_000})
def test_chunked_batch_outputs_cache_and_replay(mesh_device, tmp_path):
    layers = int(os.environ.get("GEMMA4_BATCH_TEST_LAYERS", "6"))
    lanes, chunk = (2, 4096) if os.environ.get("GEMMA4_BATCH_TEST_SHAPE", "2x4k") == "2x4k" else (4, 1024)
    plan = ChunkedBatchPlan(batch_size=lanes, chunk_size=chunk)
    num_slots = lanes * 2
    model, caches = build_model(
        mesh_device, chunk_size=chunk, num_slots=num_slots, context_len=chunk * 8, num_layers=layers
    )
    mesh = model.mesh_config
    model._prefill_metadata_external = True

    def host(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32).reshape(1, chunk),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (8, 4), dims=(1, None)),
        )

    inp = ttnn.to_device(host([0] * chunk), mesh_device)
    pos = ttnn.to_device(host(range(chunk)), mesh_device)
    model.set_prefill_rope_positions(pos)
    first = tuple(ChunkedRequest(i, i, 0, tokens(i + 1, chunk)) for i in range(lanes))
    mixed = tuple(
        reversed(
            tuple(
                ChunkedRequest(
                    i if i % 2 == 0 else 100 + i,
                    i,
                    chunk if i % 2 == 0 else 0,
                    tokens(20 + i, chunk if i % 2 == 0 else (33, chunk - 1)[i // 2]),
                )
                for i in range(lanes)
            )
        )
    )
    isolation = (*first[:-1], ChunkedRequest(lanes - 1, lanes - 1, 0, tokens(99, chunk)))
    cases = (first, mixed, isolation)
    # All independent references complete before capturing the shared batch.
    for boundary, batch in enumerate(cases):
        outputs, kv = {}, {}
        for req in batch:
            slot = req.slot_id + lanes
            ttnn.copy_host_to_device_tensor(host((*req.token_ids, *([0] * (chunk - len(req.token_ids))))), inp)
            ttnn.copy_host_to_device_tensor(host(range(req.actual_start, req.actual_start + chunk)), pos)
            model.prefill_metadata.update(slot_idx=slot, kv_actual_global=req.actual_start)
            output = model(model.transform_and_embed_prefill_inputs_device(inp))
            ttnn.synchronize_device(mesh_device)
            outputs[req.request_id] = _cp_gather_torch(output, mesh)[0, 0, : len(req.token_ids)].clone()
            output.deallocate(True)
            for layer, cache in enumerate(caches.layers):
                for kind, tensor in enumerate(cache_tensors(cache)):
                    kv[req.request_id, layer, kind] = read_cache_tensor(
                        tensor, slot, req.actual_start + chunk, chunk_size=chunk
                    )[:, : req.actual_end].clone()
        torch.save((outputs, kv), tmp_path / f"reference_{boundary}.pt")
        print(f"Independent boundary {boundary} complete", flush=True)
    inp.deallocate(True)
    pos.deallocate(True)

    # Compile all cache readback geometries before trace capture, including slots.
    for tensor in (caches.layers[0].k, caches.layers[5].kv):
        for slot in range(lanes):
            for extent in (chunk, 2 * chunk):
                read_cache_tensor(tensor, slot, extent, chunk_size=chunk)
    migration_caches = Gemma4KvCaches(
        layers=caches.layers[:6],
        layer_types=caches.layer_types[:6],
        num_users=num_slots,
        max_seq_len=chunk * 8,
        cp=8,
        tp=4,
    )
    table = build_kv_chunk_address_table(mesh_device=mesh_device, kv_caches=migration_caches, chunk_size=chunk)
    device_map = {(mesh_id, chip_id): uid for mesh_id, chip_id, uid in _build_device_map(mesh_device, (8, 4))}
    runtime = ChunkedBatchRuntime(model, num_slots=num_slots, plan=plan)
    runtime.capture()
    first_outputs = None
    try:
        for boundary, batch in enumerate(cases):
            expected, expected_kv = torch.load(tmp_path / f"reference_{boundary}.pt", weights_only=True)
            tails = {}
            for req in batch:
                for layer, cache in enumerate(caches.layers):
                    for kind, tensor in enumerate(cache_tensors(cache)):
                        tails[req.request_id, layer, kind] = read_cache_tensor(
                            tensor, req.slot_id, req.actual_start + chunk, chunk_size=chunk
                        )[:, ((req.actual_end + 31) // 32) * 32 :].clone()
            acknowledgements = []
            runtime.layer_completion_sink = lambda layer, req_id: acknowledgements.append((layer, req_id))
            runtime.prefill_batch(batch)
            outputs = runtime.to_torch()
            assert acknowledgements == [(layer, req.request_id) for req in batch for layer in range(layers)]
            for req in batch:
                assert_similar(
                    outputs[req.request_id],
                    expected[req.request_id],
                    f"output boundary={boundary} request={req.request_id}",
                )
                for layer, cache in enumerate(caches.layers):
                    for kind, tensor in enumerate(cache_tensors(cache)):
                        actual = read_cache_tensor(tensor, req.slot_id, req.actual_start + chunk, chunk_size=chunk)
                        assert_similar(
                            actual[:, : req.actual_end],
                            expected_kv[req.request_id, layer, kind],
                            f"KV boundary={boundary} request={req.request_id} layer={layer} kind={kind}",
                        )
                        torch.testing.assert_close(
                            actual[:, ((req.actual_end + 31) // 32) * 32 :],
                            tails[req.request_id, layer, kind],
                            rtol=0,
                            atol=0,
                        )
                        if layer in (0, 5) and len(req.token_ids) == chunk:
                            config = 0 if hasattr(cache, "kv") else (4 if kind == 0 else 20)
                            for head, rows in enumerate(actual):
                                check_table_samples(
                                    table, device_map, layer, req.slot_id, config + head, rows, chunk_size=chunk
                                )
            if boundary == 0:
                first_outputs = outputs
            if boundary == 2:
                for req_id in range(lanes - 1):
                    torch.testing.assert_close(outputs[req_id], first_outputs[req_id], rtol=0, atol=0)
            runtime.layer_completion_sink = None
            runtime.execute()
            for req_id, output in runtime.to_torch().items():
                torch.testing.assert_close(output, outputs[req_id], rtol=0, atol=0)
    finally:
        runtime.close()
