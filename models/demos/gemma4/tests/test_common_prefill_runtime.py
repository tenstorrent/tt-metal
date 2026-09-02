# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import struct
import time

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
    baseline_input = socket_shaped_tokens(2)
    baseline_start = time.perf_counter()
    runtime.prefill_chunk(baseline_input, caches, slot_id=0, actual_start=8192, actual_end=16384)
    baseline_ms = (time.perf_counter() - baseline_start) * 1000
    print(f"Gemma 4 traced baseline replay: {baseline_ms:.1f} ms")

    if serialize_migration_table:
        table_path = runtime.build_kv_chunk_table(caches, path=str(tmp_path / "gemma4_kv.pb"))
        assert os.path.getsize(table_path) > 0
    runtime.release_trace()

    d2h_service = ttnn.D2HStreamService(
        mesh_device,
        global_spec=None,
        fifo_size_bytes=4 * 1024,
        worker_cores=ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
        metadata_size_bytes=12,
    )
    runtime.set_d2h_ack_service(d2h_service)
    runtime.capture_trace(caches)
    assert runtime._trace_controller.num_segments == 1

    # Capture executes one warm pass. Drain its records before checking the real request.
    for _ in range(runtime.warmup_ack_count()):
        assert struct.unpack("<III", d2h_service.read_metadata()) == (0, 0, 8192)

    metadata_msg = runtime._make_metadata_msg((1, 8192, 16384))
    d2h_input = socket_shaped_tokens(3)
    from ttnn._experimental.layer_completion import LayerCompletionRouter

    ring_name = f"/tt_gemma4_d2h_test_ring_{os.getpid()}"
    channel_name = f"/tt_gemma4_d2h_test_channel_{os.getpid()}"
    router = LayerCompletionRouter(
        rank=0,
        world_size=1,
        master_rank=0,
        ring_shm_name=ring_name,
        scheduler_channel_shm_name=channel_name,
    )
    ack_service = ttnn.LayerAckService(
        d2h_service,
        ring_name,
        source_rank=0,
        num_layers=60,
        first_layer_idx=0,
        local_layers=60,
    )
    ack_service.start()
    replay_start = time.perf_counter()
    runtime.prefill_chunk(
        d2h_input,
        caches,
        slot_id=1,
        actual_start=8192,
        actual_end=16384,
        d2h_service=d2h_service,
        metadata_msg=metadata_msg,
    )
    replay_ms = (time.perf_counter() - replay_start) * 1000
    deadline = time.monotonic() + 10
    while router.processed < 60 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert router.processed == 60
    print(f"Gemma 4 traced D2H replay: {replay_ms:.1f} ms")
    persisted = ttnn.to_torch(ttnn.get_device_tensors(runtime.trace_metadata_msg)[0]).flatten().tolist()
    assert persisted == [1, 8192, 16384]
    ack_service.stop()
    router.stop()
    del ack_service
    del router
    metadata_msg.deallocate(True)
    runtime.release_trace()
    del d2h_service
