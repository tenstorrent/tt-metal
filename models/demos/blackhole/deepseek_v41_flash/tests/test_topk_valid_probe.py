# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""topk_large_indices(valid_length_tensor=...) vs the scalar valid_length vs torch on wide bf16 score rows (the local top-512 of the single-trace indexer), valid lengths up to 32k."""

import pytest
import torch

import ttnn


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
def test_topk_valid(mesh_device):
    md = mesh_device
    g = torch.Generator().manual_seed(3)
    C, W = 512, 33024
    sc = (torch.randn(1, 1, C, W, generator=g) * 4).to(torch.bfloat16)
    sc_d = _rep(md, sc, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    nv = _rep(md, torch.zeros(1, 1, 1, 1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    for v in (1024, 4096, 8192, 8224, 12288, 16384, 16448, 24576, 32832):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.full((1, 1, 1, 1), v, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            ),
            nv,
        )
        out = {}
        for name, kw in (("tensor", dict(valid_length_tensor=nv)), ("scalar", dict(valid_length=v))):
            ids = ttnn.experimental.topk_large_indices(sc_d, k=512, **kw)
            ttnn.synchronize_device(md)
            out[name] = ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).reshape(C, 512).to(torch.int64)
        want = torch.topk(sc[0, 0, :, :v].float(), 512, dim=1).values.sum(1)
        for name, got in out.items():
            ok = (got < v).all(dim=1)
            vals = torch.gather(sc[0, 0].float(), 1, got.clamp(max=W - 1)).sum(1)
            good = ok & ((vals - want).abs() < 1e-2 * want.abs().clamp(min=1))
            print(
                f"TOPK valid={v} {name}: rows with all ids < valid and value-sum equal to torch: {int(good.sum())}/{C}",
                flush=True,
            )
    # gather of bf16 scores by local top-512 ids on a wide row-major tensor
    for hi in (4096, 8192, 8224, 16384, 16448, 32832):
        idx = torch.randint(0, hi, (1, 1, C, 512), generator=g, dtype=torch.int64).to(torch.int32)
        want = torch.gather(sc[0, 0], 1, idx[0, 0].to(torch.int64)).reshape(1, 1, C, 512)
        i_d = _rep(md, idx, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        o = ttnn.gather(sc_d, 3, index=i_d)
        ttnn.synchronize_device(md)
        got = ttnn.to_torch(ttnn.get_device_tensors(ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT))[0])
        print(
            f"GATHER RM bf16 width {W}, ids < {hi}: mismatching entries {int((got != want).sum())} of {got.numel()}",
            flush=True,
        )
    print("TOPK DONE", flush=True)
