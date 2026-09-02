# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

import torch

import ttnn
from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric


@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": 256_000_000})
def test_common_prefill_runtime_traced(mesh_device, tmp_path):
    adapter = get_adapter("gemma4_31b")
    hf_config = adapter.load_hf_config()
    max_seq_len = int(os.environ.get("GEMMA4_COMMON_TEST_MAX_SEQ_LEN", "16384"))
    num_users = int(os.environ.get("GEMMA4_COMMON_TEST_NUM_USERS", "2"))
    serialize_migration_table = os.environ.get("GEMMA4_COMMON_TEST_SERIALIZE_TABLE", "1") == "1"
    params = PrefillRunParams(
        mesh_shape=(8, 4),
        num_layers=60,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=max_seq_len,
        chunk_size=8192,
        num_users=num_users,
        capacity_factor=8,
        num_links=2,
        gate_mode_name="DEVICE_FP32",
        kv_only_last_layer=True,
        weight_cache_path=adapter.weight_cache_path((8, 4)),
        use_trace=True,
    )
    runtime = adapter.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
    caches = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
    runtime.compile(caches)
    runtime.capture_trace(caches)

    mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()]),
    )

    def socket_shaped_tokens(value):
        return ttnn.from_torch(
            torch.full((8, 1, 1024), value, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    runtime.prefill_chunk(socket_shaped_tokens(0), caches, slot_id=0, actual_start=0, actual_end=8192)
    runtime.prefill_chunk(socket_shaped_tokens(1), caches, slot_id=1, actual_start=0, actual_end=8192)
    runtime.prefill_chunk(socket_shaped_tokens(2), caches, slot_id=0, actual_start=8192, actual_end=16384)

    if serialize_migration_table:
        table_path = runtime.build_kv_chunk_table(caches, path=str(tmp_path / "gemma4_kv.pb"))
        assert os.path.getsize(table_path) > 0
    runtime.release_trace()

    class AckCounter:
        count = 0

        def inject(self, delta):
            self.count += delta

    acknowledgements = AckCounter()
    runtime.set_layer_ack_channel(acknowledgements)
    runtime.capture_trace(caches)
    runtime.prefill_chunk(socket_shaped_tokens(3), caches, slot_id=1, actual_start=8192, actual_end=16384)
    assert acknowledgements.count == 60
    runtime.release_trace()
