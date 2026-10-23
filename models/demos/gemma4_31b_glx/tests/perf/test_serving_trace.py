# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The serving runtime's traced path (PREFILL_USE_TRACE=1) writes the same KV bytes and sends the same acks as the
op-by-op path (manual perf pass, not a frozen test).

One request of SERVE_CHUNKS chunks (the last one ends 32 tokens early, so it is padded) runs op by op into slot 0, then
traced into slot 1 twice: the first traced pass runs each chunk variant op by op and captures it, the second replays
every chunk. Slot 1 must then equal slot 0 bit for bit in every checked layer, and every pass must ack each layer once
per chunk in order.
"""

import os
from types import SimpleNamespace

import pytest
import torch

import ttnn

PAD = 0xFFFFFFFF


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200 * 1024 * 1024}],
    indirect=True,
    ids=["ring"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["8x4"])
def test_serving_trace(mesh_device):
    from models.demos.gemma4_31b_glx.tt.serving import Gemma431bGlxAdapter

    mesh = mesh_device
    n_layers = int(os.environ.get("SERVE_LAYERS", "60"))
    n_chunks = int(os.environ.get("SERVE_CHUNKS", "3"))
    chunk, sp = 5120, 8
    seq = 56320
    check_layers = sorted({0, 5, n_layers - 1} & set(range(n_layers)))

    def params(use_trace):
        return SimpleNamespace(
            mesh_shape=(8, 4),
            sp_axis=0,
            tp_axis=1,
            num_layers=n_layers,
            first_layer_idx=0,
            is_first_rank=True,
            is_last_rank=True,
            max_seq_len=seq,
            chunk_size=chunk,
            num_users=2,
            use_trace=use_trace,
        )

    adapter = Gemma431bGlxAdapter()
    kv = adapter.allocate_kv_cache(mesh_device=mesh, hf_config=None, params=params(False))
    rt = adapter.build_runtime(mesh_device=mesh, hf_config=None, params=params(False))
    rt.compile(kv)
    rt_t = adapter.build_runtime(mesh_device=mesh, hf_config=None, params=params(True))
    rt_t.model, rt_t.kv = rt.model, kv  # one model (weights) for both runtimes

    gen = torch.Generator().manual_seed(1)
    tokens = torch.randint(0, 262144, (n_chunks * chunk,), generator=gen)
    actual = n_chunks * chunk - 32

    def engine_input(start, end):
        ids = torch.full((chunk,), PAD, dtype=torch.int64)
        ids[: end - start] = tokens[start:end]
        from models.demos.gemma4_31b_glx.tt.model import chunk_positions

        dev = ids[chunk_positions(start, chunk, sp) - start]
        return ttnn.from_torch(
            dev.reshape(sp, 1, chunk // sp),
            device=mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, (8, 4), dims=(0, None)),
        )

    def serve(runtime, slot):
        acks = []
        runtime.set_layer_completion_sink(lambda layer, rid: acks.append((layer, rid)))
        for c in range(n_chunks):
            a, b = c * chunk, min((c + 1) * chunk, actual)
            runtime.prefill_chunk(engine_input(a, b), kv, slot_id=slot, actual_start=a, actual_end=b, request_id=c)
        ttnn.synchronize_device(mesh)
        want = [(layer, c) for c in range(n_chunks) for layer in range(n_layers)]
        assert acks == want, (acks[:5], len(acks), len(want))

    def slot_rows(slot):
        out = {}
        for layer in check_layers:
            c = kv.layers[layer]
            for name, t in (("kv", c),) if not isinstance(c, tuple) else (("k", c[0]), ("v", c[1])):
                part = ttnn.slice(t, (slot, 0, 0, 0), (slot + 1, *tuple(t.shape)[1:]), memory_config=ttnn.DRAM_MEMORY_CONFIG)
                out[(layer, name)] = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(part)]
                ttnn.deallocate(part)
        return out

    serve(rt, 0)
    ref = slot_rows(0)
    serve(rt_t, 1)  # captures each variant after running it op by op
    n_traces = len(rt.model.traces)
    serve(rt_t, 1)  # replays only
    got = slot_rows(1)
    bad = []
    for k in ref:
        for d, (x, y) in enumerate(zip(ref[k], got[k])):
            if not torch.equal(x, y):
                bad.append((k, d, float((x.float() - y.float()).abs().max())))
    print(f"SERVE traces {n_traces}; layers checked {check_layers}; mismatches {len(bad)} {bad[:4]}", flush=True)
    rt_t.release_trace()
    assert n_traces >= 1 and not bad
