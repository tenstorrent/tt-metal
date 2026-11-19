# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Small global-pool gather/remap/reset device probe; author has NOT run it.

Only the device owner may invoke, serialized under tt-device-usage:

timeout 180 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  PYTHONPATH=/home/ttuser/qwen38-flash/tt-metal python_env/bin/python \
  -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_shared_kv_pool \
  --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/shared_kv_pool_device.json

No model weights. Exercises the actual LocalSelectedPagedKV custom kernel with
16 global blocks and only4 logical pages, the actual SlotStateMover, and actual
generator request reset. This does not prove paged_update_cache writes or full
model generation; those require a reduced generator shared-pool regression.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1":
        parser.error("Allocation tracking must be enabled")
    if os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0":
        parser.error("Program-cache tracking must remain enabled")
    import torch

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.cache_pool import SharedKVPool
    from models.autoports.qwen_qwen3_8_flash_next.tt.generator import QwenGenerator
    from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import LocalSelectedPagedKV
    from models.autoports.qwen_qwen3_8_flash_next.tt.slot_state import SlotStateMover

    torch.set_num_threads(4)
    report = dict(passed=False, checks=[], trace_tracking=True, paged_update_tested=False)
    mesh, trace = None, None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=4 * 1024 * 1024)

        def upload(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                value, device=mesh, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        def refresh(destination, value):
            host = ttnn.from_torch(value.contiguous(), dtype=destination.dtype, layout=destination.layout)
            ttnn.copy_host_to_device_tensor(host, destination)

        def state(marker):
            attention = SimpleNamespace(
                capacity=128,
                page_size=32,
                position=upload(torch.tensor([marker], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                page_table=upload(torch.arange(4, dtype=torch.int32).reshape(1, 4), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                k_cache=upload(torch.zeros(4, 1, 32, 256)),
                v_cache=upload(torch.zeros(4, 1, 32, 256)),
                raw_history=upload(torch.full((1, 1, 3, 128), float(marker))),
                compressed_keys=upload(torch.full((1, 1, 32, 128), float(marker))),
            )
            layer = SimpleNamespace(conv=None, recurrent=None, ple_conv=None, token_history=None, attention=attention)
            return SimpleNamespace(capacity=128, position=0, layers=[layer], shared_kv=False)

        caches, scratch = [state(1), state(2)], state(3)
        pool = SharedKVPool([((16, 1, 32, 256), torch.bfloat16, 0)], mesh)
        displaced = pool.bind([*caches, scratch])
        for tensor in displaced:
            ttnn.deallocate(tensor)
        del displaced
        k_pool, v_pool = pool.pair_for_attention_layer(0)
        pattern = torch.arange(16).reshape(16, 1, 1, 1).expand(16, 1, 32, 256).contiguous().bfloat16()
        refresh(k_pool, pattern)
        refresh(v_pool, pattern + 32)
        tables = pool.normalize_tables(batch_size=2, capacity=128, page_table=[[15], [14]], lengths=[32, 32])
        for slot, cache in enumerate(caches):
            refresh(cache.layers[0].attention.page_table, tables[0][slot : slot + 1])
        mover = SlotStateMover(caches, scratch)
        mover.warmup()
        owner = QwenGenerator.__new__(QwenGenerator)
        owner.mesh, owner.max_batch_size, owner.capacity = mesh, 2, 128
        owner.shared_pool, owner.cache_setup_error, owner.owns_cache_storage = pool, None, False
        owner.caches, owner.counters, owner.page_tables = caches, {"cache_resets": 0}, {}
        owner.slot_mover = mover
        owner.prepared = True
        owner.traces, owner.prefill_traces, owner.sampling_traces = {}, {}, {}
        # Compile the exact reset programs before capture. Scratch has the same
        # request-owned geometry and aliases the pool, so this warmup cannot
        # overwrite K/V or alter either live slot's table/state.
        owner._reset_cache(scratch)
        owner.counters["cache_resets"] = 0
        pool.freeze()
        selected = torch.full((1, 2080), 2**32 - 1, dtype=torch.int64)
        selected[0, :32] = torch.arange(32)
        selected_tt = upload(selected, ttnn.uint32)
        keys, values = [upload(torch.zeros(1, 1, 2080, 256)) for _ in range(2)]
        op = LocalSelectedPagedKV(
            k_cache=k_pool,
            v_cache=v_pool,
            page_table=caches[0].layers[0].attention.page_table,
            selected_ids=selected_tt,
            keys=keys,
            values=values,
        )
        op.prepare()
        op.forward()
        ttnn.synchronize_device(mesh)
        addresses = [tensor.buffer_address() for tensor in op.tensors]
        trace = owner._record_trace(op.forward)

        def check(block, label):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            for output, expected in ((keys, block), (values, block + 32)):
                for rank in ttnn.get_device_tensors(output):
                    value = ttnn.to_torch(rank)
                    assert torch.all(value[0, 0, :32] == expected), label
                    assert torch.all(value[0, 0, 32:] == 0), label
            assert addresses == [tensor.buffer_address() for tensor in op.tensors]
            report["checks"].append(label)

        check(15, "global-ID15-with-logical-page-count4")
        check(15, "unchanged-table-replay")
        mover.apply([1, 0])
        check(14, "slot-remap-moves-table-without-KV-copy")
        assert caches[0].layers[0].attention.k_cache is caches[1].layers[0].attention.k_cache
        owner._reset_cache(caches[1])
        check(14, "other-slot-reset-preserves-shared-KV")
        owner._reset_cache(caches[0])
        check(14, "own-reset-preserves-shared-KV-and-table")
        refresh(caches[0].layers[0].attention.page_table, torch.tensor([[13, 0, 0, 0]], dtype=torch.int32))
        check(13, "changed-global-page-table-replay")
        assert owner.counters["cache_resets"] == 2, "Expected exactly the two explicit request resets"
        report["passed"] = True
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
