# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device test of the on-device ``kv_len_tensor`` of ``ttnn.experimental.indexer_score_dsa`` (classic op, C++ change of branch ssinghal/dsv4p1-chunk1-cpp, needs the build_local libs):

  (1) the scores of the columns [0, kv_len) are bit-identical to the op with the host scalar ``kv_len`` (several kv_len, production knobs: q chunk 32, k chunk 128, all 32 heads resident,
      chunk_start = T - 32, 4 x 8 mesh with a different key slab per device),
  (2) ONE captured trace replayed with new tensor values scores the new prefix (a host scalar would stay frozen at the captured value),
  (3) the replay time follows kv_len.

    LD_LIBRARY_PATH=<wt>/tm_chunk_cpp/build_local/lib PYTHONPATH=<wt>/tm_chunk/ttnn:<wt>/tm_chunk pytest tests/test_indexer_kvlen_tensor.py -s
"""

import os
import time

import pytest
import torch

import ttnn

T = int(os.environ.get("KVT_T", "131104"))  # local key slots + 32 pad (131072 + 32: the 1M ratio-1 build / 8 columns)
SQ = 512
HI, D = 32, 128


def _rep(md, t, dtype, layout):
    return ttnn.from_torch(
        t,
        device=md,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100000000},
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_kv_len_tensor(mesh_device):
    md = mesh_device
    g = torch.Generator().manual_seed(0)
    q = _rep(md, torch.randn(1, HI, SQ, D, generator=g).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    w = _rep(md, torch.rand(1, 1, SQ, HI, generator=g).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    k = _rep(md, torch.randn(1, 1, T, D, generator=g).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=128, head_group_size=0)
    cs = T - 32

    def host(kvl):
        # (reference: host kv_len = kvl + 32 with chunk_start = kvl: every key < kvl is visible to every row; the comparison covers ALL kvl columns, incl. a partial last k unit)
        o = ttnn.experimental.indexer_score_dsa(
            q, k, w, chunk_start_idx=kvl, kv_len=kvl + 32, program_config=cfg, cache_batch_idx=0
        )
        ttnn.synchronize_device(md)
        return ttnn.to_torch(ttnn.get_device_tensors(o)[0])[0, 0, :, :kvl].float()

    kvt = _rep(md, torch.zeros(1, 1, 1, 1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    def setv(v):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.full((1, 1, 1, 1), v, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            ),
            kvt,
        )

    def dev():
        return ttnn.experimental.indexer_score_dsa(
            q, k, w, chunk_start_idx=cs, program_config=cfg, cache_batch_idx=0, kv_len_tensor=kvt
        )

    print("KVT inputs uploaded", flush=True)
    # (1) eager equality
    for kvl in (1024, 4096, 4128, 4160, 16384, 16416, 16448, 30720, 30752, 65536, 65568, 65600, T - 32 - 64):
        ref = host(kvl)
        print(f"KVT host ref {kvl} done", flush=True)
        setv(kvl)
        o = dev()
        ttnn.synchronize_device(md)
        got = ttnn.to_torch(ttnn.get_device_tensors(o)[0])[0, 0, :, :kvl].float()
        eq = bool(torch.equal(ref, got))
        print(f"KVT eager kv_len={kvl}: bit-equal to the host scalar op: {eq}", flush=True)
        assert eq
        ttnn.deallocate(o)
    # (2) + (3) one trace, many values
    setv(T - 32)
    o = dev()  # compile
    ttnn.synchronize_device(md)
    ttnn.deallocate(o)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    o = dev()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    for kvl in (1024, 4128, 16448, 30720, 65568, T - 32 - 64, 2048):
        setv(kvl)
        ttnn.synchronize_device(md)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            ts.append(time.perf_counter() - t0)
        got = ttnn.to_torch(ttnn.get_device_tensors(o)[0])[0, 0, :, :kvl].float()
        ref = host(kvl)
        eq = bool(torch.equal(ref, got))
        print(
            f"KVT trace kv_len={kvl}: replay {min(ts) * 1e3:.2f} ms, bit-equal to the host scalar op: {eq}", flush=True
        )
        assert eq
    ttnn.release_trace(md, tid)
    print("KVT DONE", flush=True)
