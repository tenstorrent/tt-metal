# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compare rotated service requests with aligned prefill on the same model."""

import os
from functools import partial

import torch

import ttnn
from models.common.weight_cache import build_cached_state_dict, weight_cache_is_complete
from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.common.prefill.chunk_layout import chunk_positions
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.common import weight_cache_identity
from models.demos.gemma4_d_p.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4_d_p.tt.precision import Gemma4Precision
from tests.ttnn.utils_for_testing import comp_pcc


@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": 32 * 1024 * 1024})
def test_rotated_prefill_matches_aligned(mesh_device, monkeypatch):
    from models.demos.gemma4_d_p.tt import tt_prefill_runtime as runtime_module

    layers = int(os.environ.get("GEMMA4_ROTATED_TEST_LAYERS", "6"))
    adapter = get_adapter("gemma4_31b")
    hf_config = adapter.load_hf_config()
    args = Gemma4ModelArgs.from_hf_config(hf_config)
    cache_path = adapter.weight_cache_path((8, 4))
    model_path = os.environ.get("HF_MODEL", adapter.hf_model_default)
    identity = weight_cache_identity(model_path, 60, (8, 4), Gemma4Precision.load(model_path, (8, 4)))
    assert weight_cache_is_complete(cache_path, **identity), "Populate the full dedicated Gemma4 cache first"
    state = build_cached_state_dict(cache_path, args=args, build_variant=identity["build_variant"])
    # A small-layer test can reuse the verified full-model tensor cache without loading HF weights.
    monkeypatch.setattr(runtime_module, "create_tt_model", partial(runtime_module.create_tt_model, state_dict=state))
    params = PrefillRunParams(
        mesh_shape=(8, 4),
        num_layers=layers,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=32768,
        chunk_size=8192,
        num_users=2,
        capacity_factor=8,
        num_links=2,
        gate_mode_name="DEVICE_FP32",
        kv_only_last_layer=True,
        weight_cache_path=cache_path,
        use_trace=True,
    )
    caches = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
    runtime = adapter.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
    runtime.compile(caches)
    runtime.capture_trace(caches)
    metadata_addresses = [
        t.buffer_address()
        for t in (*runtime.model.ccl_manager.get_ring_metadata(), runtime.model.ccl_manager.get_ring_valid_end())
    ]
    tokens = torch.randint(1, 10000, (32768,), generator=torch.Generator().manual_seed(43)).tolist()
    selected_layers = sorted({0, min(5, layers - 1), layers - 1})

    def run(slot, start, end):
        runtime.prefill_chunk(
            runtime.make_chunk_input(tokens[start:end], actual_start=start, actual_end=end),
            caches,
            slot_id=slot,
            actual_start=start,
            actual_end=end,
        )
        return [ttnn.to_torch(ttnn.get_device_tensors(runtime._trace_output)[r * 4]).float()[0, 0] for r in range(8)]

    # Keep one output row per tile; sample cache rows independently below.
    baseline_output = {}
    for start in range(0, 32768, 8192):
        shards = run(0, start, start + 8192)
        for rank, shard in enumerate(shards):
            for row in range(0, 1024, 32):
                baseline_output[start + rank * 1024 + row] = shard[row].clone()

    def cache_samples(slot, end):
        result = {}
        for layer in selected_layers:
            cache = caches[layer]
            tensors = (cache.kv,) if hasattr(cache, "kv") else cache
            for kind, tensor in enumerate(tensors):
                for rank in range(8):
                    shard = ttnn.to_torch(ttnn.get_device_tensors(tensor)[rank * 4]).float()[slot]
                    local = torch.arange(shard.shape[-2])
                    positions = (local // 1024) * 8192 + rank * 1024 + local % 1024
                    valid = positions < end
                    # Include entire valid cache, so partial tiles and every head are checked.
                    result[layer, kind, rank] = shard[:, valid].clone()
                    pad = (positions >= end) & (positions < ((end + 127) // 128) * 128)
                    assert torch.count_nonzero(shard[:, pad]) == 0, (layer, kind, rank, end)
        return result

    baseline = cache_samples(0, 32768)
    previous = None
    try:
        for start, end in [(0, 7000), (6976, 9000), (8992, 17123), (17120, 25312), (25312, 32768)]:
            print(f"Checking {layers}-layer rotated request [{start}, {end})")
            output = run(1, start, end)
            mismatches = []

            def compare(expected, actual, label):
                passed, pcc = comp_pcc(expected, actual, 0.999)
                if not passed:
                    mismatches.append((label, pcc))

            positions = torch.tensor(chunk_positions(start, 8192, 8)).reshape(8, 1024)
            for rank, shard in enumerate(output):
                rows = [r for r in range(0, 1024, 32) if positions[rank, r] < end]
                if rows:
                    expected = torch.stack([baseline_output[int(positions[rank, r])] for r in rows])
                    actual = shard[rows]
                    assert torch.isfinite(actual).all()
                    compare(expected, actual, ("output", rank))
            current = cache_samples(1, end)
            for key, actual in current.items():
                expected = baseline[key][:, : actual.shape[1]]
                if actual.numel():
                    assert torch.isfinite(actual).all()
                    compare(expected, actual, ("cache", key))
                if previous is not None:
                    # Earlier cache rows must survive each continuation; exclude the replayed tile.
                    rank = key[2]
                    local = torch.arange(previous[key].shape[1])
                    absolute = (local // 1024) * 8192 + rank * 1024 + local % 1024
                    untouched = absolute < start
                    torch.testing.assert_close(
                        actual[:, : len(local)][:, untouched], previous[key][:, untouched], rtol=0, atol=0
                    )
            assert not mismatches, (start, end, mismatches)
            previous = current
            # Slot 0 is the aligned reference and must never be overwritten by slot 1.
            for key, actual in cache_samples(0, 32768).items():
                torch.testing.assert_close(actual, baseline[key], rtol=0, atol=0)
            assert metadata_addresses == [
                t.buffer_address()
                for t in (
                    *runtime.model.ccl_manager.get_ring_metadata(),
                    runtime.model.ccl_manager.get_ring_valid_end(),
                )
            ]
    finally:
        runtime.release_trace()
