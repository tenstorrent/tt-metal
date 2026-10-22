# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device probes of the ttnn ops the single-trace indexer (tt/prefill_single.py) relies on, on the 4x8 mesh (replicated data):

  (a) topk_large_indices(valid_length_tensor=...) on a row-major bf16 score tensor whose width is far above the valid length, incl. valid lengths BELOW k = 512 and rows that are -inf at the end,
  (b) ttnn.gather on a ROW-MAJOR bf16 score tensor with row-major uint32 / tile uint32 indices (the scores of the local top-512),
  (c) narrow (32 / 64 column) row-major slice + tile add + slice_write back,
  (d) key_append (tt/paged_kernels/key_append.cpp): in-place tile writes at tensor-driven row blocks, users with offset 0.
    pytest tests/test_single_ops_probe.py -s
"""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import prefill_single as PS


def _rep(md, t, dtype, layout):
    return ttnn.from_torch(
        t,
        device=md,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


def _dev0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


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
def test_single_ops(mesh_device):
    md = mesh_device
    g = torch.Generator().manual_seed(1)
    C, W = 512, 8256  # width: valid prefix + stale tail
    sc = torch.randn(1, 1, C, W, generator=g).to(torch.bfloat16)
    sc_d = _rep(md, sc, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    nv = _rep(md, torch.zeros(1, 1, 1, 1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    def setv(v):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.full((1, 1, 1, 1), v, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            ),
            nv,
        )

    # (a) top-k on the valid prefix
    for v in (64, 256, 512, 1024, 8192):
        setv(v)
        ids = ttnn.experimental.topk_large_indices(sc_d, k=512, valid_length_tensor=nv)
        ttnn.synchronize_device(md)
        got = _dev0(ids).reshape(C, 512).to(torch.int64)
        want = torch.topk(sc[0, 0, :, :v].float(), min(512, v), dim=1).indices
        nvalid = min(512, v)
        ok_sets = all(set(got[r, :nvalid].tolist()) == set(want[r].tolist()) for r in range(0, C, 37))
        tail = got[:, nvalid:]
        tail_sentinel = bool((tail == 0xFFFFFFFF).all()) if tail.numel() else True
        print(
            f"PROBE topk valid={v}: top-{nvalid} sets equal {ok_sets}; entries after them are 0xFFFFFFFF: {tail_sentinel}",
            flush=True,
        )
        ttnn.deallocate(ids)
    # rows that are -inf at the end (the staircase): fewer visible than k
    sc2 = sc.clone()
    sc2[:, :, :, 100:] = float("-inf")
    sc2_d = _rep(md, sc2, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    setv(1024)
    ids = ttnn.experimental.topk_large_indices(sc2_d, k=512, valid_length_tensor=nv)
    got = _dev0(ids).reshape(C, 512).to(torch.int64)
    print(
        f"PROBE topk with -inf beyond column 100: first 100 valid {bool((got[:, :100] < 100).all())}, rest sentinel {bool((got[:, 100:] == 0xFFFFFFFF).all())}",
        flush=True,
    )

    # (b) gather on RM scores
    setv(1024)
    ids = ttnn.experimental.topk_large_indices(sc_d, k=512, valid_length_tensor=nv)
    idt = ttnn.to_layout(ids, ttnn.ROW_MAJOR_LAYOUT)
    for name, ix in (("rm", idt), ("tile", ttnn.to_layout(ttnn.typecast(ids, ttnn.uint32), ttnn.TILE_LAYOUT))):
        try:
            vals = ttnn.gather(sc_d, 3, index=ix)
            ttnn.synchronize_device(md)
            v = _dev0(vals).reshape(C, 512)
            want = torch.gather(sc[0, 0], 1, _dev0(ids).reshape(C, 512).to(torch.int64))
            print(
                f"PROBE gather(RM scores, {name} index): equal to torch.gather: {bool(torch.equal(v, want))}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"PROBE gather(RM scores, {name} index) FAILED: {str(e)[:300]}", flush=True)

    # (c) narrow slice -> tile add -> slice_write
    for cw in (32, 64):
        blk = ttnn.slice(sc_d, [0, 0, 0, 0], [1, 1, C, cw])
        t = ttnn.to_layout(blk, ttnn.TILE_LAYOUT)
        add = _rep(md, torch.full((1, 1, C, cw), -1.0), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        out = ttnn.to_layout(ttnn.add(t, add), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.experimental.slice_write(out, sc_d, [0, 0, 0, 0], [1, 1, C, cw], [1, 1, 1, 1])
        ttnn.synchronize_device(md)
        got = _dev0(sc_d)
        want = sc.clone()
        want[:, :, :, :cw] = (sc[:, :, :, :cw].float() - 1.0).to(torch.bfloat16)
        print(f"PROBE narrow slice/add/slice_write width {cw}: equal {bool(torch.equal(got, want))}", flush=True)
        sc = want

    # (d) key_append
    U, Cc8, T = 2, 64, 1024
    src = torch.randn(U, 1, Cc8, 128, generator=g).to(torch.bfloat16)
    dst0 = torch.randn(U, 1, T, 128, generator=g).to(torch.bfloat16)
    src_d = _rep(md, src, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    dst_d = _rep(md, dst0, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    rows = md.shape[0]
    kofs = torch.zeros(rows * U, 32, dtype=torch.int32)
    kofs[:, 0] = torch.tensor([5, 0, 9, 7, 0, 3, 11, 0])  # (mesh row, user)
    kofs_d = ttnn.from_torch(
        kofs,
        device=md,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape)),
    )
    PS.key_append(src_d, kofs_d, dst_d, U, Cc8 // 32, T)
    ttnn.synchronize_device(md)
    devs = ttnn.get_device_tensors(dst_d)
    ok = True
    for r in range(rows):
        got = ttnn.to_torch(devs[r * md.shape[1]])  # [U,1,T,128]
        want = dst0.clone()
        for u in range(U):
            o = int(kofs[r * U + u, 0]) * 32
            want[u, :, o : o + Cc8] = src[u]
        ok &= bool(torch.equal(got, want))
    print(f"PROBE key_append (U={U}, per-(row,user) offsets, offset 0 = scratch): all rows equal {ok}", flush=True)
    # (e) gather of uint32 data above 65535 (global slots of a 1M context), row-major, and the same through the tile path
    data = torch.randint(0, 1 << 21, (1, 1, C, 4128), generator=g, dtype=torch.int64).to(torch.int32)
    data[:, :, :, 4096:] = -1
    idx = torch.randint(0, 4128, (1, 1, C, 512), generator=g, dtype=torch.int64).to(torch.int32)
    want = torch.gather(data[0, 0].to(torch.int64), 1, idx[0, 0].to(torch.int64)).reshape(1, 1, C, 512)
    for name, lay in (("rm", ttnn.ROW_MAJOR_LAYOUT), ("tile", ttnn.TILE_LAYOUT)):
        d = _rep(md, data, ttnn.uint32, lay)
        i = _rep(md, idx, ttnn.uint32, lay)
        o = ttnn.gather(d, 3, index=i)
        ttnn.synchronize_device(md)
        got = ttnn.to_torch(ttnn.get_device_tensors(ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT))[0]).to(torch.int64)
        w = want.to(torch.int64) & 0xFFFFFFFF
        print(
            f"PROBE gather uint32 values < 2^21 ({name}): equal {bool(torch.equal(got & 0xFFFFFFFF, w))}; mismatching entries {int((got & 0xFFFFFFFF != w).sum())}",
            flush=True,
        )
    print("PROBE DONE", flush=True)
