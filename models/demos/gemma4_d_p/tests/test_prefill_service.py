# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import ttnn
from models.demos.gemma4_d_p.tt.runners.runtime import Gemma4PrefillRuntime


def read_slot_samples(runtime, kv_cache, slot):
    samples = {}
    num_banks = runtime.mesh_device.dram_grid_size().x
    for row, column in ((0, 0), (0, 3), (7, 0), (7, 3)):
        fabric_node = runtime.mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(row, column))
        device_id = ttnn.cluster.get_chip_unique_id_from_fabric_node_id(
            int(fabric_node.mesh_id), int(fabric_node.chip_id)
        )
        for layer_idx, cache in enumerate(kv_cache.layers):
            tensors = (cache.kv,) if hasattr(cache, "kv") else (cache.k, cache.v)
            for tensor_idx, tensor in enumerate(tensors):
                _, heads, local_seq_len, head_dim = tensor.shape
                blocks = local_seq_len // 32
                chunk_bytes = head_dim // 32 * 1088
                for block in (0, blocks - 1):
                    shard = slot * heads * blocks + block
                    bank = shard % num_banks
                    offset = shard // num_banks * chunk_bytes
                    address = bank << 32 | int(tensor.buffer_address()) + offset
                    raw = bytes(ttnn.experimental.disaggregation.read_dram_umd(device_id, address, chunk_bytes))
                    mantissas = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 1088)[:, 64:]
                    assert np.any(mantissas & 0x7F), (slot, layer_idx, row, column, block)
                    samples[layer_idx, tensor_idx, row, column, block] = raw
    return samples


@pytest.mark.timeout(7200)
def test_prefill_service_six_slots(monkeypatch, tmp_path):
    monkeypatch.setenv("PREFILL_MANIFEST", str(Path(__file__).parents[1] / "tt/runners/manifest.json"))
    monkeypatch.setenv("PREFILL_H2D_SERVICE_ID", f"gemma4_test_{os.getpid()}")
    monkeypatch.setenv("PREFILL_NUM_USERS", "6")
    from models.demos.common.prefill.runners import prefill_runner

    snapshots = {}
    original_prefill = Gemma4PrefillRuntime.prefill_chunk
    original_loop = prefill_runner.run_request_loop

    def checked_prefill(runtime, input_tensor, kv_cache, **request):
        original_prefill(runtime, input_tensor, kv_cache, **request)
        if request["actual_end"] == 262144:
            hidden = ttnn.to_torch(ttnn.get_device_tensors(runtime.output)[0]).float()
            assert torch.isfinite(hidden).all()
            assert hidden.std() > 0.001
            snapshots[request["slot_id"]] = read_slot_samples(runtime, kv_cache, request["slot_id"])

    def checked_loop(runtime, kv_cache, *args, **kwargs):
        original_loop(runtime, kv_cache, *args, **kwargs)
        assert runtime.slot_ends == [262144] * 6
        assert set(snapshots) == set(range(6))
        for slot, expected in snapshots.items():
            assert read_slot_samples(runtime, kv_cache, slot) == expected, f"Slot {slot} was overwritten"
        for slot in range(1, 6):
            assert snapshots[slot] != snapshots[0], f"Slot {slot} contains slot 0's prompt"

    monkeypatch.setattr(Gemma4PrefillRuntime, "prefill_chunk", checked_prefill)
    monkeypatch.setattr(prefill_runner, "run_request_loop", checked_loop)
    producer_log = tmp_path / "producer.log"
    with producer_log.open("w") as log:
        producer = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "models.demos.gemma4_d_p.tt.runners.prefill_producer",
                "--results",
                str(tmp_path / "results.json"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            prefill_runner.main()
            assert producer.wait(timeout=60) == 0, producer_log.read_text()
        finally:
            if producer.poll() is None:
                producer.terminate()
                producer.wait(timeout=30)
